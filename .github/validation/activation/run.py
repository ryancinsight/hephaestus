#!/usr/bin/env python3
"""Bounded fixed-revision activation validation; local mode never acquires a device."""
import argparse
import datetime
import hashlib
import json
import os
from pathlib import Path
import platform
import re
import shutil
import signal
import subprocess
import sys
import time
import tomllib

sys.dont_write_bytecode = True
from verify import InvalidEvidence, diagnostic_identity, exports, manifest, measured_run, require, sha256, timing_summary

HERE = Path(__file__).resolve().parent


class Stopped(InvalidEvidence):
    def __init__(self, status, message):
        super().__init__(message)
        self.status = status


def atomic_json(path, data):
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(data, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def installed_context(environ=None, system=None):
    env = os.environ if environ is None else environ
    require(env.get('GITHUB_ACTIONS') == 'true', 'hosted execution refuses a local process')
    require(env.get('GITHUB_EVENT_NAME') == 'pull_request', 'only automatic PR validation is supported')
    require((platform.system() if system is None else system) == 'Darwin', 'Metal validation requires the intended macOS cohort')
    require(env.get('GITHUB_REPOSITORY') == 'ryancinsight/hephaestus', 'unexpected hosted repository')
    require(env.get('HEPHAESTUS_HOSTED_ACTIVATION_INTENT') == 'reviewed-metal-validation', 'missing reviewed workflow intention')
    require(re.fullmatch(r'\d+', env.get('GITHUB_RUN_ID', '')) is not None, 'missing hosted run identity')


def budget_record(spec, origin, now):
    limits = spec['budgets']
    require(isinstance(origin, int) and 0 < origin <= now, 'invalid/future first-user-step origin')
    elapsed = now - origin
    remaining = limits['job_seconds'] - elapsed
    needed = limits['phase_seconds'] + limits['artifact_reserve_seconds'] + limits['setup_margin_seconds']
    return {'origin_scope': 'first user step; runner initialization predates this clock',
            'origin_epoch': origin, 'observed_epoch': now, 'elapsed_seconds': elapsed,
            'raw_remaining_seconds': remaining, 'required_remaining_seconds': needed,
            'setup_margin_seconds': limits['setup_margin_seconds'],
            'job_budget_seconds': limits['job_seconds'], 'phase_seconds': limits['phase_seconds'],
            'artifact_reserve_seconds': limits['artifact_reserve_seconds'], 'sufficient': remaining >= needed}


def external_graph(path):
    packages = tomllib.loads(path.read_text())['package']
    graph = sorted((p['name'], p['version'], p['source'], p.get('checksum')) for p in packages if 'source' in p)
    return hashlib.sha256(json.dumps(graph, separators=(',', ':')).encode()).hexdigest(), len(graph)


def ensure_comparison_object(checkout, revision, hosted, execute):
    probe = execute(['git', '-C', str(checkout), 'cat-file', '-e', revision + '^{commit}'], 'object', check=False)[0]
    if probe['exit_code']:
        require(hosted, 'exact comparison object missing; local mode never fetches')
        execute(['git', '-C', str(checkout), 'fetch', '--no-tags', '--depth=1',
                 'https://github.com/ryancinsight/hephaestus.git', revision], 'fetch-fixed-object', cap=60)


class Driver:
    def __init__(self, args):
        self.args = args
        self.spec = json.loads((HERE / 'comparison_spec.json').read_text())
        self.hosted = args.hosted
        if self.hosted:
            installed_context()
        self.source = args.source.resolve()
        self.artifacts = args.artifacts.resolve()
        self.work = args.work_root.resolve()
        require(self.source != self.work and self.source not in self.work.parents, 'scratch directory must be outside source checkout')
        require(self.source != self.artifacts and self.source not in self.artifacts.parents, 'artifacts must be outside source checkout')
        self.artifacts.mkdir(parents=True, exist_ok=True)
        self.packet = self.artifacts / 'activation'
        self.packet.mkdir(exist_ok=False)  # A prior attempt is never replayed or overwritten.
        self.work.mkdir(parents=True, exist_ok=False)
        self.record = {'schema': 1, 'mode': 'hosted' if self.hosted else 'prepare-only',
                       'status': 'STARTED', 'created_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
                       'builds': {}, 'commands': [], 'campaign': [], 'device_attempts': 0,
                       'performance_accepted': False, 'native_shader_compiler_version': 'unknown unless independently exposed',
                       'memory_contract': self.spec['memory_contract']}
        self.deadline = time.monotonic() + self.spec['budgets']['phase_seconds']
        for name in ['comparison_spec.json', 'activation_paths.rs', 'export_expressions.rs', 'verify.py', 'run.py', 'baseline_failures.json', 'predict_baseline.py', 'README.md']:
            file = HERE / name
            if file.exists():
                shutil.copyfile(file, self.packet / name)
        self.save()

    def check_budget(self):
        if self.hosted:
            origin_text = os.environ.get('HEPHAESTUS_METAL_JOB_STARTED_EPOCH', '')
            require(origin_text.isdigit(), 'missing first-user-step origin')
            clock = budget_record(self.spec, int(origin_text), int(time.time()))
            self.record['clock'] = clock
            if not clock['sufficient']:
                self.record['status'] = 'INSUFFICIENT_REMAINING_JOB_BUDGET_UNMEASURED'
                self.save()
                raise Stopped('INSUFFICIENT_REMAINING_JOB_BUDGET_UNMEASURED', 'remaining job budget is below phase+artifact reserve+setup margin')
        self.save()

    def relative(self, path):
        return str(Path(path).relative_to(self.artifacts))

    def save(self):
        atomic_json(self.artifacts / 'activation-result.json', self.record)

    def command(self, args, name, cap=30, env=None, cwd=None, check=True):
        remaining = self.deadline - time.monotonic()
        require(remaining > 0, 'validation phase deadline exhausted')
        number = len(self.record['commands'])
        directory = self.packet / 'commands'
        directory.mkdir(exist_ok=True)
        stem = directory / f'{number:02d}-{name}'
        output, error, usage = stem.with_suffix('.stdout'), stem.with_suffix('.stderr'), stem.with_suffix('.resources')
        system = platform.system()
        started = time.monotonic()
        with output.open('wb') as stdout, error.open('wb') as stderr:
            child = subprocess.Popen(args, cwd=cwd, env=env, stdout=stdout, stderr=stderr, start_new_session=True)
            timed_out = False
            stop = started + min(cap, remaining)
            while True:
                waited, wait_status, resources = os.wait4(child.pid, os.WNOHANG)
                if waited:
                    break
                if time.monotonic() >= stop:
                    os.killpg(child.pid, signal.SIGKILL)
                    _, wait_status, resources = os.wait4(child.pid, 0)
                    timed_out = True
                    break
                time.sleep(0.05)
            status = os.waitstatus_to_exitcode(wait_status)
            child.returncode = status
        rss = int(resources.ru_maxrss) * (1 if system == 'Darwin' else 1024)
        resource_record = {'peak_rss_bytes': rss, 'native_unit': 'bytes (Darwin wait4)' if system == 'Darwin' else 'KiB (Linux wait4), converted to bytes',
                           'user_seconds': resources.ru_utime, 'system_seconds': resources.ru_stime,
                           'scope': 'wait4 child usage, including OS-accounted waited descendants; no hard memory enforcement',
                           'hard_memory_limit_enforced': False}
        atomic_json(usage, resource_record)
        item = {'name': name, 'args': args, 'exit_code': status, 'timed_out': timed_out,
                'elapsed_seconds': time.monotonic() - started, 'stdout': self.relative(output),
                'stderr': self.relative(error), 'resources': resource_record}
        self.record['commands'].append(item)
        self.save()
        if timed_out:
            raise Stopped('CHILD_TIMEOUT_UNMEASURED', name + ': timeout; process group killed, execution incomplete, no retry')
        if check:
            require(status == 0, name + ': nonzero exit ' + str(status))
        return item, output, error

    def capture(self, args, name, cap=30):
        _, out, _ = self.command(args, name, cap=cap)
        require(out.stat().st_size < 1 << 20, 'oversized identity capture')
        return out.read_text().rstrip('\n')

    def source_identity(self):
        head = self.capture(['git', '-C', str(self.source), 'rev-parse', 'HEAD'], 'validation-head')
        tree = self.capture(['git', '-C', str(self.source), 'rev-parse', 'HEAD^{tree}'], 'validation-tree')
        fixed = self.spec['cohorts']['head']['revision']
        # The CI checkout may be the PR merge tree. It is provenance, never the
        # comparison product head. No production change may enter implicitly.
        provenance_checkout = self.work / 'provenance'
        self.command(['git', 'clone', '--no-hardlinks', '--dissociate', '--no-checkout', str(self.source), str(provenance_checkout)], 'provenance-clone')
        self.ensure_object(provenance_checkout, fixed, 'provenance')
        require(self.capture(['git', '-C', str(provenance_checkout), 'rev-parse', fixed + '^{tree}'], 'provenance-comparison-tree') == self.spec['cohorts']['head']['tree'], 'provenance comparison tree mismatch')
        changed = self.capture(['git', '-C', str(provenance_checkout), 'diff', '--name-only', fixed, head, '--', 'crates'], 'validation-production-diff')
        production = [p for p in changed.splitlines() if '/src/' in p and not p.startswith('crates/hephaestus-conformance/')]
        require(not production, 'validation checkout changes production source: ' + repr(production))
        if self.hosted:
            status = self.capture(['git', '-C', str(self.source), 'status', '--porcelain=v1'], 'validation-status')
            require(not status, 'hosted validation checkout is dirty')
        self.record['validation_source'] = {'head': head, 'tree': tree,
            'comparison_head': fixed, 'production_diff_from_comparison': production,
            'github_sha': os.environ.get('GITHUB_SHA'), 'github_run_id': os.environ.get('GITHUB_RUN_ID'),
            'github_run_attempt': os.environ.get('GITHUB_RUN_ATTEMPT')}
        toolchains = self.capture(['rustup', 'toolchain', 'list'], 'installed-toolchains')
        require(any(line.split()[0].startswith(self.spec['toolchain'] + '-') for line in toolchains.splitlines()), 'required compiler is not installed; no installation attempted')
        compiler = self.capture(['rustc', '+' + self.spec['toolchain'], '-vV'], 'compiler')
        self.record['compiler'] = compiler
        self.record['platform'] = {'system': platform.platform(), 'machine': platform.machine(),
                                   'native_shader_compiler_identity': 'not inferred from Rust/Naga/Xcode'}
        self.record['platform']['physical_memory_bytes'] = None
        self.record['platform']['physical_memory_status'] = 'not queried; previous EPERM read remains unavailable, no retry/recovery'
        if self.hosted:
            provenance = json.loads((self.artifacts / 'cohort-provenance.json').read_text())
            require(provenance['rustc']['stdout'].strip() == compiler, 'focused diagnostic compiler differs')
            require(provenance['head_and_tree']['stdout'].splitlines() == [head, tree], 'focused diagnostic source differs')
            self.record['diagnostic_identity'] = diagnostic_identity(self.artifacts / 'witness-nextest.log')
            self.record['native_shader_compiler_version'] = provenance.get('native_shader_compiler_version', 'unknown')
        self.save()

    def ensure_object(self, checkout, revision, name):
        def execute(args, suffix, **kwargs):
            return self.command(args, name + '-' + suffix, **kwargs)
        ensure_comparison_object(checkout, revision, self.hosted, execute)

    def build(self, cohort):
        identity = self.spec['cohorts'][cohort]
        checkout = self.work / cohort
        self.command(['git', 'clone', '--no-hardlinks', '--dissociate', '--no-checkout', str(self.source), str(checkout)], cohort + '-clone')
        self.ensure_object(checkout, identity['revision'], cohort)
        self.command(['git', '-C', str(checkout), 'checkout', '--detach', identity['revision']], cohort + '-checkout')
        require(self.capture(['git', '-C', str(checkout), 'rev-parse', 'HEAD^{tree}'], cohort + '-tree') == identity['tree'], 'comparison tree mismatch')
        require(not (checkout / '.git/objects/info/alternates').exists(), 'comparison clone borrows object storage')
        graph, count = external_graph(checkout / 'Cargo.lock')
        require(graph == identity['external_package_graph_sha256'] and count == identity['external_packages'], 'locked external graph mismatch')
        shutil.copyfile(checkout / 'Cargo.lock', self.packet / (cohort + '-Cargo.lock'))
        bench = checkout / 'crates/hephaestus-wgpu/benches/activation_paths.rs'
        require(not bench.exists(), 'comparison contains an unexpected benchmark overlay')
        shutil.copyfile(HERE / 'activation_paths.rs', bench)
        cargo = checkout / 'crates/hephaestus-wgpu/Cargo.toml'
        with cargo.open('a') as target:
            target.write('\n[[bench]]\nname = "activation_paths"\nharness = false\n')
        status = self.capture(['git', '-C', str(checkout), 'status', '--porcelain=v1', '--untracked-files=all'], cohort + '-overlay-status')
        require(sorted(line[3:] for line in status.splitlines()) == ['crates/hephaestus-wgpu/Cargo.toml', 'crates/hephaestus-wgpu/benches/activation_paths.rs'], 'unexpected overlay source delta')
        env = os.environ.copy()
        for key in ['RUSTFLAGS', 'RUSTDOCFLAGS', 'CARGO_ENCODED_RUSTFLAGS']:
            env.pop(key, None)
        require(self.hosted or os.environ.get('CARGO_HOME'), 'local prepare requires an explicitly private CARGO_HOME')
        env.update(CARGO_TARGET_DIR=str(self.work / 'target'), CARGO_BUILD_JOBS='2',
                   ACTIVATION_SOURCE_REVISION=identity['revision'], ACTIVATION_EXPRESSION_SHA256=identity['expression_sha256'],
                   ACTIVATION_HARNESS_SHA256=sha256(HERE / 'activation_paths.rs'),
                   ACTIVATION_VALIDATION_HEAD=self.record['validation_source']['head'], PYTHONDONTWRITEBYTECODE='1')
        args = ['cargo', '+' + self.spec['toolchain'], 'bench', '--manifest-path', str(checkout / 'Cargo.toml'),
                '-p', 'hephaestus-wgpu', '--bench', 'activation_paths', '--no-run', '--locked', '--offline', '--message-format=json']
        _, build_json, _ = self.command(args, cohort + '-build', cap=self.spec['budgets']['build_seconds'], env=env)
        binary, core = None, None
        with build_json.open() as stream:
            for line in stream:
                require(len(line) < 1 << 19, 'oversized Cargo JSON record')
                artifact = json.loads(line)
                if artifact.get('reason') != 'compiler-artifact':
                    continue
                target = artifact['target']['name']
                if target == 'activation_paths' and artifact.get('executable'):
                    binary = Path(artifact['executable'])
                if target == 'hephaestus_core':
                    core = next((Path(p) for p in artifact['filenames'] if p.endswith('.rlib')), core)
        require(binary is not None and core is not None, 'missing compiled benchmark/core artifact')
        preserved = self.packet / (cohort + '-activation_paths')
        shutil.copy2(binary, preserved)
        exporter = self.work / (cohort + '-expression-export')
        self.command(['rustc', '+' + self.spec['toolchain'], '--edition=2024', str(HERE / 'export_expressions.rs'),
                      '-C', 'lto=thin', '-C', 'opt-level=3', '-C', 'codegen-units=1', '--extern',
                      'hephaestus_core=' + str(core), '-L', 'dependency=' + str(core.parent), '-o', str(exporter)], cohort + '-export-build', env=env)
        _, expression_file, _ = self.command([str(exporter)], cohort + '-expressions')
        require(sha256(expression_file) == identity['expression_sha256'], 'actual compiled expression map mismatch')
        exported = exports(expression_file)
        _, manifest_file, _ = self.command([str(preserved)], cohort + '-cpu-manifest')
        cpu = manifest(manifest_file)
        require(cpu['meta']['revision'] == identity['revision'] and cpu['meta']['expression_sha256'] == identity['expression_sha256'], 'CPU binary identity mismatch')
        for op in self.spec['operations']:
            require(cpu['expressions'][op] == exported[op, 'Wgsl'], 'benchmark/library expression mismatch')
        _, features, _ = self.command(['cargo', '+' + self.spec['toolchain'], 'tree', '--manifest-path', str(checkout / 'Cargo.toml'),
                                      '-p', 'hephaestus-wgpu', '--edges', 'features', '--locked', '--offline'], cohort + '-features', env=env)
        feature_hash = hashlib.sha256()
        with features.open() as stream:
            for line in stream:
                normalized = line.replace(str(checkout), 'COHORT').replace('v0.27.0', 'vWORKSPACE').replace('v0.27.1', 'vWORKSPACE')
                feature_hash.update(normalized.encode())
        item = {'revision': identity['revision'], 'tree': identity['tree'], 'binary': self.relative(preserved),
                'binary_sha256': sha256(preserved), 'expression_file': self.relative(expression_file),
                'expression_sha256': sha256(expression_file), 'cpu_manifest': self.relative(manifest_file),
                'cpu_manifest_sha256': sha256(manifest_file), 'fixture_sha256': cpu['fixture_sha256'],
                'features_sha256': feature_hash.hexdigest(), 'external_graph_sha256': graph,
                'harness_sha256': sha256(bench), 'overlay_status': status}
        self.record['builds'][cohort] = item
        self.save()
        return cpu, exported

    def campaign(self, manifests):
        predicates = json.loads((HERE / self.spec['expected_baseline_predicates']).read_text())
        identity = None
        runs = []
        for index, cohort in enumerate(self.spec['order']):
            build = self.record['builds'][cohort]
            binary = self.artifacts / build['binary']
            require(sha256(binary) == build['binary_sha256'], 'copied binary changed')
            env = os.environ.copy()
            env['ACTIVATION_DEVICE_AUTHORIZATION'] = 'root-approved-metal-only'
            self.record['device_attempts'] += 1
            row = {'index': index, 'cohort': cohort, 'command_index': len(self.record['commands']), 'status': 'STARTED_OUTCOME_PENDING'}
            self.record['campaign'].append(row)
            self.save()
            item, stdout, stderr = self.command([str(binary), '--run-device'], f'run-{index}-{cohort}',
                                               cap=self.spec['budgets']['invocation_seconds'], env=env, check=False)
            row.update(stdout=self.relative(stdout), stderr=self.relative(stderr), exit_code=item['exit_code'])
            self.save()
            with stderr.open() as stream:
                unavailable = any('ACQUISITION_FAILURE\t' in line for line in stream)
            if unavailable or item['exit_code'] == 2:
                row['status'] = 'ACQUISITION_FAILURE_STOPPED_NO_RETRY'
                raise Stopped('ACQUISITION_FAILURE_UNMEASURED', 'production Metal acquisition failed; no retry/fallback')
            require(item['exit_code'] in [0, 1], 'runtime error, no retry')
            checked = measured_run(stdout, manifests[cohort], self.spec, cohort, predicates, self.record['diagnostic_identity'])
            require(item['exit_code'] == (1 if checked['failures'] else 0), 'exit/correctness mismatch')
            if identity is not None:
                require(identity == checked['identity'], 'BHHB adapter/feature/limit mismatch')
            identity = checked['identity']
            row['classified_baseline_failures'] = checked['failures']
            row['verified_samples'] = len(checked['timings'])
            row['status'] = 'VERIFIED_WITH_EXPECTED_BASELINE_FAILURES' if checked['failures'] else 'VERIFIED'
            runs.append((cohort, checked))
            self.save()
        self.record['timing_summary'] = timing_summary(runs)

    def finish(self):
        self.record['completed_utc'] = datetime.datetime.now(datetime.timezone.utc).isoformat()
        records, size = [], 0
        for file in sorted(self.artifacts.rglob('*')):
            if not file.is_file() or file.name in ['manifest.json', 'activation-result.json']:
                continue
            require(not file.is_symlink(), 'artifact symlink is forbidden')
            size += file.stat().st_size
            records.append({'path': self.relative(file), 'bytes': file.stat().st_size, 'sha256': sha256(file)})
        require(size <= self.spec['budgets']['artifact_bytes'], 'narrow artifact budget exceeded')
        atomic_json(self.packet / 'manifest.json', {'files': records, 'total_bytes': size})
        self.record['artifact_manifest_sha256'] = sha256(self.packet / 'manifest.json')
        self.save()


def main():
    parser = argparse.ArgumentParser()
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--hosted', action='store_true')
    mode.add_argument('--prepare-only', action='store_true')
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--work-root', type=Path, required=True)
    parser.add_argument('--artifacts', type=Path, required=True)
    args = parser.parse_args()
    driver = None
    try:
        driver = Driver(args)
        driver.check_budget()
        driver.source_identity()
        manifests, maps = {}, {}
        for cohort in ['base', 'head']:
            manifests[cohort], maps[cohort] = driver.build(cohort)
        require(manifests['base']['fixture_sha256'] == manifests['head']['fixture_sha256'], 'base/head reference fixture mismatch')
        builds = driver.record['builds']
        require(builds['base']['features_sha256'] == builds['head']['features_sha256'], 'base/head feature graph mismatch')
        changed = {op for (op, dialect), expression in maps['base'].items() if expression != maps['head'][op, dialect]}
        require(changed == set(driver.spec['operations']), 'changed expression set not covered by benchmark')
        from predict_baseline import generate
        base = builds['base']
        generated = generate(driver.artifacts / base['expression_file'], driver.artifacts / base['cpu_manifest'])
        retained = json.loads((HERE / driver.spec['expected_baseline_predicates']).read_text())
        require(generated == retained, 'baseline exception predicates differ from fixed reviewed CPU source model')
        driver.record['baseline_predicates_sha256'] = sha256(HERE / driver.spec['expected_baseline_predicates'])
        if args.hosted:
            driver.campaign(manifests)
            driver.record['status'] = 'MEASURED_COHORT_REVIEW_PENDING'
        else:
            driver.record['status'] = 'CPU_PREPARED_NO_DEVICE_EXECUTION'
        driver.finish()
        print(driver.record['status'])
        return 0
    except (InvalidEvidence, OSError, ValueError, KeyError) as error:
        if driver is not None:
            driver.record['status'] = error.status if isinstance(error, Stopped) else 'UNMEASURED_OR_INVALID'
            driver.record['error'] = str(error)
            driver.save()
            driver.finish()
        print(str(error), file=sys.stderr)
        return 2


if __name__ == '__main__':
    sys.exit(main())
