"""Strict CPU-only validation of exact activation manifests and measured records."""
import argparse
import collections
import datetime
import hashlib
import json
import math
from pathlib import Path
import re
import struct
import statistics
import tomllib

OPERATIONS = {'ErfcOp', 'GeluOp', 'GeluGradOp', 'GeluTanhGradOp', 'SoftplusOp', 'SiluOp', 'MishOp', 'MishGradOp'}
DIALECTS = {'Wgsl', 'CudaC', 'HipC'}
MEASURED_OPERATIONS = {'GeluOp', 'GeluGradOp', 'GeluTanhGradOp'}


class InvalidEvidence(ValueError):
    """Evidence is incomplete, unexpected, or not bound to its stated inputs."""


def require(condition, message):
    if not condition:
        raise InvalidEvidence(message)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b''):
            digest.update(chunk)
    return digest.hexdigest()


def manifest(path):
    """Read bounded seed/reference data, not million-element readback arrays."""
    data = {'meta': {}, 'fixtures': {}, 'inputs': {}, 'expressions': {}}
    digest = hashlib.sha256()
    with Path(path).open() as stream:
        for line in stream:
            require(len(line) < 1 << 19, 'oversized manifest record')
            fields = line.rstrip('\n').split('\t')
            if fields[0] == 'META':
                require(len(fields) == 3, 'malformed manifest metadata')
                require(fields[1] not in data['meta'], 'duplicate manifest metadata')
                data['meta'][fields[1]] = fields[2]
            elif fields[0] == 'FIXTURE':
                require(len(fields) == 6, 'malformed fixture')
                key = tuple(fields[1:3])
                require(key not in data['fixtures'], 'duplicate fixture')
                data['fixtures'][key] = tuple(map(int, fields[3:]))
            elif fields[0] == 'INPUT':
                require(len(fields) == 8, 'invalid input record')
                key = tuple(fields[1:3])
                row = (int(fields[3]), int(fields[4]), int(fields[5]), float(fields[6]),
                       None if fields[7] == 'none' else float(fields[7]))
                rows = data['inputs'].setdefault(key, [])
                require(row[0] == len(rows), 'nonconsecutive seed indices')
                require(len(rows) < 8192, 'seed bound exceeded')
                rows.append(row)
                digest.update(line.encode())
            elif fields[0] == 'EXPR':
                require(len(fields) == 3 and fields[1] not in data['expressions'], 'malformed/duplicate manifest expression')
                data['expressions'][fields[1]] = json.loads(fields[2])
            else:
                raise InvalidEvidence('unexpected CPU manifest row ' + fields[0])
    data['fixture_sha256'] = digest.hexdigest()
    require(data['meta'].get('status') == 'CPU_MANIFEST_ONLY_NO_DEVICE_ACQUISITION', 'not a CPU-only manifest')
    require(len(data['fixtures']) == 9 and set(data['fixtures']) == set(data['inputs']), 'incomplete nine-fixture manifest')
    for key, rows in data['inputs'].items():
        require(len(rows) == data['fixtures'][key][0], 'fixture/input count mismatch')
        require(sum(row[1] for row in rows) % (1 << 64) == data['fixtures'][key][1], 'fixture input checksum mismatch')
        require(sum(row[2] for row in rows) % (1 << 64) == data['fixtures'][key][2], 'fixture reference checksum mismatch')
    require(set(data['expressions']) == MEASURED_OPERATIONS, 'incomplete manifest expression map')
    return data


def exports(path):
    result = {}
    with Path(path).open() as stream:
        for line in stream:
            fields = line.rstrip('\n').split('\t', 2)
            require(len(fields) == 3, 'malformed expression export')
            op, dialect, expression = fields
            require((op, dialect) not in result, 'duplicate expression export')
            result[op, dialect] = json.loads(expression)
    require(set(result) == {(op, dialect) for op in OPERATIONS for dialect in DIALECTS}, 'expected exact eight operators times three dialects')
    return result


def double(bits):
    return struct.unpack('>d', int(bits).to_bytes(8, 'big'))[0]


def single(bits):
    return struct.unpack('>f', int(bits).to_bytes(4, 'big'))[0]


def ulps(got, reference):
    if math.isnan(reference):
        return 0.0 if math.isnan(got) else math.inf
    if math.isinf(reference):
        return 0.0 if got == reference else math.inf
    if not math.isfinite(got):
        return math.inf
    exponent = math.frexp(abs(reference))[1] - 1 if reference else -126
    spacing = max(2.0 ** (exponent - 23), 2.0 ** -149)
    return abs(got - reference) / spacing


def failure_class(got, reference, bound, absolute):
    if math.isnan(got):
        return 'nan'
    if math.isinf(got):
        return 'infinite'
    selected = reference == 0.0 or abs(reference) >= 2.0 ** -126
    if not selected:
        return None
    relative_failed = ulps(got, reference) > bound
    absolute_failed = absolute is not None and abs(got - reference) > absolute
    if not (relative_failed or absolute_failed):
        return None
    if got == 0.0:
        return 'zero'
    return 'finite_absolute' if absolute_failed else 'finite_relative'


def field_value(line, marker):
    """Extract a Rust Debug field while preserving nested structs/quoted text."""
    match = re.search(r'(?<![A-Za-z0-9_])' + re.escape(marker) + '=', line)
    require(match is not None, 'missing diagnostic field ' + marker)
    start = match.end()
    prefix = re.match(r'[A-Za-z_]\w*\s+[\{\(\[]', line[start:])
    end = start + len(prefix.group()) - 1 if prefix else start
    depth = 0
    quoted = False
    escaped = False
    while end < len(line):
        char = line[end]
        if quoted:
            if escaped:
                escaped = False
            elif char == '\\':
                escaped = True
            elif char == '"':
                quoted = False
        elif char == '"':
            quoted = True
        elif char in '({[':
            depth += 1
        elif char in ')}]':
            depth -= 1
            require(depth >= 0, 'unbalanced diagnostic field')
        elif char.isspace() and depth == 0:
            break
        end += 1
    require(not quoted and depth == 0 and end > start, 'unterminated/empty diagnostic field')
    return line[start:end]



def adapter_identity(value):
    """Canonical Rust Debug identity; only Option::Some changes representation."""
    require(isinstance(value, str), 'malformed adapter identity')
    if value.startswith('Some('):
        require(value.endswith(')'), 'malformed adapter identity')
        value = value[5:-1]
    require(value.startswith('AdapterInfo {') and value.endswith('}'), 'malformed adapter identity')
    stack, quoted, escaped = [], False, False
    opening = len('AdapterInfo ')
    for index in range(opening, len(value)):
        char = value[index]
        if quoted:
            if escaped:
                escaped = False
            elif char == '\\':
                escaped = True
            elif char == '"':
                quoted = False
        elif char == '"':
            quoted = True
        elif char in '({[':
            stack.append(char)
        elif char in ')}]':
            require(stack and stack.pop() == {')': '(', '}': '{', ']': '['}[char], 'malformed adapter identity')
            require(stack or index == len(value) - 1, 'malformed adapter identity')
    require(not quoted and not stack, 'malformed adapter identity')
    return value


def cohort_identity(meta, diagnostic=None):
    identity = {key: meta.get(key) for key in ['adapter', 'features', 'device_limits']}
    require(all(identity.values()), 'missing device identity')
    adapter = adapter_identity(identity['adapter'])
    require('backend: Metal' in adapter and
            any('device_type: ' + item in adapter for item in ['IntegratedGpu', 'DiscreteGpu']), 'not identified Metal adapter type; physical Metal is unverified')
    if diagnostic is not None:
        expected = dict(diagnostic, adapter=adapter_identity(diagnostic.get('adapter')))
        require(dict(identity, adapter=adapter) == expected, 'witness/benchmark cohort mismatch')
    return identity


def diagnostic_identity(path):
    identities = []
    witnesses = 0
    with Path(path).open() as stream:
        for line in stream:
            if 'observed adapter; absent native compiler/driver versions remain unknown' in line:
                identities.append({'adapter': field_value(line, 'adapter'),
                                   'features': field_value(line, 'enabled_features'),
                                   'device_limits': field_value(line, 'device_limits')})
            if 'exact intrinsic witness result' in line:
                # Tracing span attributes repeat target_ulp before the result message.
                payload = line.partition('exact intrinsic witness result ')[2]
                for marker in ['error_ulp', 'target_ulp']:
                    matches = re.findall(r'(?<![A-Za-z0-9_])' + marker + '=', payload)
                    require(len(matches) <= 1, 'duplicate diagnostic field ' + marker)
                error = float(field_value(payload, 'error_ulp'))
                bound = float(field_value(payload, 'target_ulp'))
                require(math.isfinite(error) and error <= bound, 'focused witness exceeds unchanged target')
                witnesses += 1
    require(len(identities) == 1 and witnesses == 6, 'missing exact diagnostic identity/six results')
    adapter_identity(identities[0]['adapter'])
    return identities[0]


def measured_run(path, cpu, spec, cohort, predicates, diagnostic=None):
    expected_cases = {(op, band, length) for op in spec['operations'] for band in spec['bands'] for length in spec['sizes']}
    meta, fixtures, checks, timings, expressions = {}, {}, {}, {}, {}
    details = collections.Counter()
    detail_by_input = collections.Counter()
    total_final = None
    expected = {(row['op'], row['band'], row['input_bits'], row['class']) for row in predicates['predicates']}
    require(predicates['fixture_sha256'] == cpu['fixture_sha256'], 'baseline predicates bind another fixture/scorer')
    require(predicates['expression_sha256'] == spec['cohorts']['base']['expression_sha256'], 'baseline predicates bind another source expression')
    references = {key: {row[1]: row for row in rows} for key, rows in cpu['inputs'].items()}
    with Path(path).open() as stream:
        for line in stream:
            require(len(line) < 1 << 19, 'oversized measured record')
            f = line.rstrip('\n').split('\t')
            kind = f[0]
            require(total_final is None, 'record follows final result')
            if kind == 'META':
                require(len(f) == 3, 'malformed run metadata')
                require(f[1] not in meta, 'duplicate run metadata')
                meta[f[1]] = f[2]
            elif kind == 'EXPR':
                require(len(f) == 3 and f[1] not in expressions, 'malformed/duplicate runtime expression')
                require(cpu['expressions'].get(f[1]) == json.loads(f[2]), 'runtime expression mismatch')
                expressions[f[1]] = json.loads(f[2])
            elif kind == 'FIXTURE':
                require(len(f) == 6 and tuple(f[1:3]) not in fixtures, 'malformed/duplicate runtime fixture')
                fixtures[tuple(f[1:3])] = tuple(map(int, f[3:]))
            elif kind == 'FAILURE_DETAIL':
                require(len(f) == 11, 'malformed failure detail')
                op, band, length, phase, bits, failure, count, output_bits, reference_bits = f[1:10]
                length, bits, count = int(length), int(bits), int(count)
                require((op, band, length) in expected_cases and phase in ['before', 'after'], 'unexpected failure case')
                require(count > 0, 'empty failure detail')
                ref = references[op, band].get(bits)
                require(ref is not None and ref[2] == int(reference_bits), 'failure reference/input mismatch')
                require(failure_class(single(output_bits), double(reference_bits), ref[3], ref[4]) == failure, 'failure class misreports first output')
                require(cohort == 'base' and (op, band, bits, failure) in expected,
                        'unexpected baseline/candidate failure: ' + repr((op, band, bits, failure)))
                key = op, band, length, phase
                details[key] += count
                detail_by_input[key, bits] += count
            elif kind == 'FIRST_FAILURE':
                continue  # Complete per-input/class counts, not this sample, govern classification.
            elif kind == 'CHECK':
                require(len(f) == 10, 'malformed check')
                key = f[1], f[2], int(f[3]), f[4]
                require(key[:3] in expected_cases and key[3] in ['before', 'after'] and key not in checks, 'duplicate/unexpected check')
                samples = sum(sample[:3] == key[:3] for sample in timings)
                require(samples == (spec['samples'] if key[3] == 'after' else 0), 'check/sample order mismatch')
                checks[key] = int(f[5])
                require(0 <= checks[key] <= key[2], 'impossible failure count')
            elif kind == 'TIMING':
                require(len(f) == 7, 'malformed timing')
                key = f[1], f[2], int(f[3]), int(f[4])
                require(key[:3] in expected_cases and 0 <= key[3] < spec['samples'] and key not in timings, 'duplicate/unexpected sample')
                require((*key[:3], 'before') in checks and (*key[:3], 'after') not in checks, 'sample outside computed checks')
                require(int(f[5]) == spec['batch'] and int(f[6]) > 0, 'batch/elapsed mismatch')
                timings[key] = int(f[6])
            elif kind == 'FINAL':
                require(len(f) == 3 and total_final is None and f[1] == 'computed_value_failures', 'duplicate/faulted final result')
                total_final = int(f[2])
            else:
                raise InvalidEvidence('unexpected measured output row ' + kind)
    require(fixtures == cpu['fixtures'], 'runtime fixture mismatch')
    require(expressions == cpu['expressions'], 'runtime expression map incomplete')
    for name in ['revision', 'expression_sha256', 'harness_sha256', 'validation_head']:
        require(meta.get(name) == cpu['meta'].get(name), 'run/binary metadata mismatch: ' + name)
    require(meta.get('revision') == spec['cohorts'][cohort]['revision'], 'wrong comparison revision')
    require(len(checks) == len(expected_cases) * 2 and len(timings) == len(expected_cases) * spec['samples'], 'incomplete checks/samples')
    require(total_final == sum(checks.values()), 'final failure count mismatch')
    require(all(details[key] == count for key, count in checks.items()), 'unclassified failure/check count mismatch')
    for ((op, band, length, phase), bits), count in detail_by_input.items():
        rows = cpu['inputs'][op, band]
        maximum = sum(1 + (length - 1 - row[0]) // len(rows) for row in rows if row[1] == bits and row[0] < length)
        require(count <= maximum, 'per-input failures exceed actual repeated inputs')
    require(cohort == 'base' or total_final == 0, 'candidate numerical failure')
    identity = cohort_identity(meta, diagnostic)
    return {'identity': identity, 'failures': total_final, 'timings': timings}


def artifact_files(directory, manifest_path):
    root = Path(directory).resolve()
    records = json.loads(Path(manifest_path).read_text())['files']
    paths = set()
    for record in records:
        require(record['path'] not in paths, 'duplicate artifact path')
        paths.add(record['path'])
        file = root / record['path']
        require(not file.is_symlink() and root in file.resolve().parents, 'artifact path escapes root')
        require(file.is_file() and file.stat().st_size == record['bytes'] and sha256(file) == record['sha256'], 'artifact hash/size mismatch')
    return len(records)


def campaign_layout(spec):
    require(spec['order'] == ['base', 'head', 'head', 'base'], 'mandatory BHHB order changed')
    require(spec['replication_order'] == ['head', 'base', 'base', 'head'], 'reversed HBBH order changed')
    return [(block_index * 4 + slot, name, slot, cohort)
            for block_index, (name, order) in enumerate([('mandatory', spec['order']), ('reversed', spec['replication_order'])])
            for slot, cohort in enumerate(order)]


def replication_admission(spec, remaining):
    required = len(spec['replication_order']) * spec['budgets']['invocation_seconds']
    return {'required_remaining_phase_seconds': required, 'observed_remaining_phase_seconds': remaining,
            'sufficient': remaining >= required}


def phase_span(row, limit):
    start, end = row['phase_started_seconds'], row['phase_finished_seconds']
    require(all(type(x) in [int, float] and math.isfinite(x) for x in [start, end])
            and 0 <= start <= end <= limit, 'invalid campaign phase span')
    for key in ['started_utc', 'finished_utc']:
        require(datetime.datetime.fromisoformat(row[key]).utcoffset() == datetime.timedelta(0), 'campaign UTC missing')
    return start, end


def campaign_runs(root, result, spec, parsed, predicates, diagnostic):
    layout, campaign = campaign_layout(spec), result['campaign']
    require(len(campaign) == result['device_attempts'] == len(layout), 'incomplete replicated campaign')
    blocks = result['campaign_blocks']
    require(len(blocks) == 2, 'incomplete campaign blocks')
    limit, spans, previous = spec['budgets']['phase_seconds'], [], 0
    for index, block in enumerate(blocks):
        require(block['name'] == ['mandatory', 'reversed'][index] and type(block['start_index']) is int and block['start_index'] == index * 4
                and block['status'] == 'VERIFIED', 'invalid campaign block')
        start, end = phase_span(block, limit)
        require(start >= previous, 'overlapping campaign blocks')
        spans.append((start, end))
        previous = end
    require(blocks[1]['admission'] == replication_admission(spec, limit - spans[1][0])
            and blocks[1]['admission']['sufficient'], 'invalid reversed-block admission')
    verified, identity, seen_commands, seen_outputs, previous = [], None, set(), set(), 0
    for row, (index, block, slot, cohort) in zip(campaign, layout):
        require(type(row['index']) is int and type(row['slot']) is int
                and (row['index'], row['block'], row['slot'], row['cohort']) == (index, block, slot, cohort), 'campaign order/labels mismatch')
        start, end = phase_span(row, limit)
        low, high = spans[index // 4]
        require(low <= start <= end <= high and start >= previous, 'campaign process span mismatch')
        previous = end
        require(row['binary_sha256'] == result['builds'][cohort]['binary_sha256'], 'campaign binary mismatch')
        command_index = row['command_index']
        require(type(command_index) is int and 0 <= command_index < len(result['commands'])
                and command_index not in seen_commands, 'campaign command index mismatch')
        command = result['commands'][command_index]
        require(command['name'] == f'run-{index}-{cohort}' and command['exit_code'] == row['exit_code']
                and command['timed_out'] is False and command['stdout'] == row['stdout'] and command['stderr'] == row['stderr']
                and len(command['args']) == 2 and command['args'][1] == '--run-device'
                and Path(command['args'][0]).name == Path(result['builds'][cohort]['binary']).name, 'campaign command mismatch')
        require(not {row['stdout'], row['stderr']} & seen_outputs and row['stdout'] != row['stderr'], 'duplicate campaign output')
        seen_commands.add(command_index)
        seen_outputs.update([row['stdout'], row['stderr']])
        checked = measured_run(root / row['stdout'], parsed[cohort], spec, cohort, predicates, diagnostic)
        require(row['exit_code'] == (1 if checked['failures'] else 0)
                and row['classified_baseline_failures'] == checked['failures'] and row['verified_samples'] == len(checked['timings'])
                and row['status'] == ('VERIFIED_WITH_EXPECTED_BASELINE_FAILURES' if checked['failures'] else 'VERIFIED'), 'campaign exit/correctness mismatch')
        require(identity is None or identity == checked['identity'], 'campaign adapter mismatch')
        identity = checked['identity']
        verified.append((cohort, checked))
    device_commands = {i for i, command in enumerate(result['commands'])
                       if command['name'].startswith('run-') or '--run-device' in command['args']}
    require(device_commands == seen_commands, 'unreferenced or missing device command')
    summaries = {name: timing_summary(verified[start:start + 4]) for name, start in [('mandatory', 0), ('reversed', 4)]}
    require(result['timing_summary'] == summaries['mandatory'] and result['block_timing_summaries'] == summaries,
            'block timing summary differs from raw samples')
    return verified


def timing_summary(runs):
    """Raw samples remain primary; summarize dispatch+completion, without a gate."""
    groups = collections.defaultdict(lambda: {'base': [], 'head': []})
    for cohort, run in runs:
        for (op, band, length, _), elapsed in run['timings'].items():
            groups[op, band, length][cohort].append(elapsed)
    rows = []
    for (op, band, length), cohorts in sorted(groups.items()):
        row = {'op': op, 'band': band, 'elements': length}
        for cohort, samples in cohorts.items():
            median = statistics.median(samples)
            row[cohort] = {'samples': len(samples), 'median_batch_ns': median,
                           'minimum_batch_ns': min(samples), 'maximum_batch_ns': max(samples),
                           'median_absolute_deviation_ns': statistics.median(abs(x - median) for x in samples)}
        row['head_over_base_median_ratio'] = row['head']['median_batch_ns'] / row['base']['median_batch_ns']
        rows.append(row)
    return rows


def packet(directory):
    """Independently reparse retained source/binary/check/timing bindings."""
    root = Path(directory)
    result = json.loads((root / 'activation-result.json').read_text())
    require(result['performance_accepted'] is False, 'packet must not auto-accept performance')
    index = root / 'activation/manifest.json'
    require(sha256(index) == result['artifact_manifest_sha256'], 'artifact index hash mismatch')
    count = artifact_files(root, index)
    indexed = {row['path'] for row in json.loads(index.read_text())['files']}
    references = [build[key] for build in result['builds'].values() for key in ['binary', 'expression_file', 'cpu_manifest']]
    references += [row[key] for row in result['campaign'] for key in ['stdout', 'stderr'] if key in row]
    require(all(path in indexed for path in references), 'unindexed build/campaign artifact')
    spec = json.loads((root / 'activation/comparison_spec.json').read_text())
    parsed, expression_maps = {}, {}
    for cohort, build in result['builds'].items():
        require(cohort in spec['cohorts'], 'unexpected comparison cohort')
        wanted = spec['cohorts'][cohort]
        require(all(build[k] == wanted[k] for k in ['revision', 'tree', 'expression_sha256']), 'build source identity mismatch')
        require(sha256(root / build['binary']) == build['binary_sha256'], 'binary hash mismatch')
        packages = tomllib.loads((root / ('activation/' + cohort + '-Cargo.lock')).read_text())['package']
        graph = sorted((p['name'], p['version'], p['source'], p.get('checksum')) for p in packages if 'source' in p)
        graph_digest = hashlib.sha256(json.dumps(graph, separators=(',', ':')).encode()).hexdigest()
        require(graph_digest == wanted['external_package_graph_sha256'] == build['external_graph_sha256'] and len(graph) == wanted['external_packages'], 'retained lock external graph mismatch')
        require(sha256(root / build['expression_file']) == wanted['expression_sha256'], 'compiled export hash mismatch')
        cpu = manifest(root / build['cpu_manifest'])
        require(sha256(root / build['cpu_manifest']) == build['cpu_manifest_sha256'], 'CPU manifest hash mismatch')
        require(cpu['fixture_sha256'] == build['fixture_sha256'], 'fixture/scorer digest mismatch')
        require(cpu['meta']['revision'] == wanted['revision'] and cpu['meta']['expression_sha256'] == wanted['expression_sha256'], 'CPU source identity mismatch')
        require(cpu['meta']['harness_sha256'] == sha256(root / 'activation/activation_paths.rs') == build['harness_sha256'], 'harness digest mismatch')
        require(cpu['meta']['validation_head'] == result['validation_source']['head'], 'validation source binding mismatch')
        expression_maps[cohort] = exports(root / build['expression_file'])
        require(all(cpu['expressions'][op] == expression_maps[cohort][op, 'Wgsl'] for op in spec['operations']), 'CPU/compiled expressions differ')
        parsed[cohort] = cpu
    if len(parsed) == 2:
        require(parsed['base']['fixture_sha256'] == parsed['head']['fixture_sha256'], 'cohort fixture mismatch')
        require(result['builds']['base']['features_sha256'] == result['builds']['head']['features_sha256'], 'cohort feature mismatch')
        changed = {key for key in expression_maps['base'] if expression_maps['base'][key] != expression_maps['head'][key]}
        require(changed == {(op, 'Wgsl') for op in spec['operations']}, 'uncovered changed expression')
        require(result['builds']['base']['binary_sha256'] != result['builds']['head']['binary_sha256'], 'cohort binaries are not distinct')
    verified = []
    campaign = result['campaign']
    if result['status'] == 'MEASURED_REPLICATED_COHORT_REVIEW_PENDING':
        diagnostic = diagnostic_identity(root / 'witness-nextest.log')
        predicates = json.loads((root / 'activation/baseline_failures.json').read_text())
        verified = campaign_runs(root, result, spec, parsed, predicates, diagnostic)
    elif result['status'] == 'CPU_PREPARED_NO_DEVICE_EXECUTION':
        require(len(parsed) == 2 and result['device_attempts'] == 0 and not campaign, 'CPU preparation includes device attempts')
    else:
        require(not result['status'].startswith('MEASURED'), 'unsupported measured campaign status')
    return {'artifact_files_verified': count, 'status': result['status'], 'measured_invocations_verified': len(verified),
            'performance_accepted': False, 'note': 'Measured cohort requires independent performance assessment; no portable intrinsic qualification.'}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--artifacts', type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(packet(args.artifacts)))


if __name__ == '__main__':
    main()
