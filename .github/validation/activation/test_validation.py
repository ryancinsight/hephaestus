"""CPU-only parser/numeric unit tests; no backend is mocked or acquired.

Temporary record specimens exercise parsers, not activation accuracy or timing.
Their IEEE powers of two have exact known bits; all case/sample dimensions come
from the unchanged comparison spec. They are never retained as device evidence.
"""
import sys
sys.dont_write_bytecode = True

import ast
import copy
import hashlib
import importlib.util
import itertools
import json
import math
import os
from pathlib import Path
import struct
import subprocess
import tempfile
import unittest

HERE = Path(__file__).resolve().parent
MODULE = importlib.util.spec_from_file_location('verify', HERE / 'verify.py')
verify = importlib.util.module_from_spec(MODULE)
sys.modules[MODULE.name] = verify
MODULE.loader.exec_module(verify)
DRIVER_MODULE = importlib.util.spec_from_file_location('activation_run', HERE / 'run.py')
driver = importlib.util.module_from_spec(DRIVER_MODULE)
DRIVER_MODULE.loader.exec_module(driver)


def bits(value, code):
    return int.from_bytes(struct.pack('>' + code, value), 'big')


def record(*fields):
    return '\t'.join(map(str, fields)) + '\n'


class TemporaryRecords(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix='activation-parser-unit-')
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)

    def write(self, name, content):
        path = self.root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content)
        return path

    def rejected(self, expression, function, *args, **kwargs):
        with self.assertRaisesRegex(verify.InvalidEvidence, expression):
            function(*args, **kwargs)


class NumericContract(unittest.TestCase):
    def test_ieee_bit_decoding(self):
        for code, decoder, values in [('f', verify.single, [1.0, -2.0, 2.0 ** -149]),
                                       ('d', verify.double, [1.0, -2.0, 2.0 ** -1022])]:
            for value in values:
                with self.subTest(code=code, value=value):
                    self.assertEqual(decoder(bits(value, code)), value)
        self.assertEqual(math.copysign(1, verify.single(0x80000000)), -1)
        self.assertTrue(math.isnan(verify.single(0x7fc00001)))

    def test_ulps_exact_normal_subnormal_and_sign_boundaries(self):
        # Binary32 spacing is 2**(e-23), bounded below by 2**-149.
        cases = [(1.0, 1.0, 0.0), (1.0 + 2.0 ** -23, 1.0, 1.0),
                 (-2.0 - 2.0 ** -22, -2.0, 1.0),
                 (0.0, 2.0 ** -126, float(2 ** 23)),
                 (2.0 ** -149, 0.0, 1.0),
                 (0.0, 2.0 ** -150, 0.5), (-0.0, 0.0, 0.0)]
        for got, reference, expected in cases:
            with self.subTest(got=got, reference=reference):
                self.assertEqual(verify.ulps(got, reference), expected)

    def test_ulps_nonfinite_matches_harness_contract(self):
        cases = [(math.nan, math.nan, 0.0), (math.inf, math.inf, 0.0),
                 (-math.inf, -math.inf, 0.0), (1.0, math.nan, math.inf),
                 (math.inf, 1.0, math.inf), (1.0, math.inf, math.inf),
                 (-math.inf, math.inf, math.inf)]
        for got, reference, expected in cases:
            with self.subTest(got=got, reference=reference):
                self.assertEqual(verify.ulps(got, reference), expected)

    def test_failure_classes_and_strict_boundaries(self):
        spacing = 2.0 ** -23
        cases = [(math.nan, 1.0, 32, None, 'nan'),
                 (math.inf, 1.0, 32, None, 'infinite'),
                 (-math.inf, 2.0 ** -149, 32, None, 'infinite'),
                 (0.0, 2.0 ** -126, 256, None, 'zero'),
                 (1.0 + 32 * spacing, 1.0, 32, None, None),
                 (1.0 + 33 * spacing, 1.0, 32, None, 'finite_relative'),
                 (1.0 + spacing, 1.0, math.inf, spacing, None),
                 (1.0 + 2 * spacing, 1.0, math.inf, spacing, 'finite_absolute'),
                 (1.0 + 33 * spacing, 1.0, 32, spacing, 'finite_absolute'),
                 (0.0, 2.0 ** -149, 0, 0, None),
                 (math.nan, 2.0 ** -149, 256, None, 'nan'),
                 (-0.0, 0.0, 16, 3e-7, None)]
        for got, reference, bound, absolute, expected in cases:
            with self.subTest(got=got, reference=reference, bound=bound):
                self.assertEqual(verify.failure_class(got, reference, bound, absolute), expected)


class RecordContract(TemporaryRecords):
    def setUp(self):
        super().setUp()
        self.spec = json.loads((HERE / 'comparison_spec.json').read_text())
        self.meta = {'revision': self.spec['cohorts']['base']['revision'],
                     'expression_sha256': self.spec['cohorts']['base']['expression_sha256'],
                     'harness_sha256': verify.sha256(HERE / 'activation_paths.rs'),
                     'validation_head': self.spec['cohorts']['head']['revision']}
        self.identity = {
            'adapter': 'Some(AdapterInfo { name: "PARSER UNIT ONLY; no device", '
                       'device_type: IntegratedGpu, backend: Metal })',
            'features': 'Features(0x5)',
            'device_limits': 'Limits { nested: (8, [16, 32]), max_buffer_size: 4194304 }'}
        self.cpu_lines = [record('META', k, v) for k, v in self.meta.items()]
        self.cpu_lines += [record('META', 'status', 'CPU_MANIFEST_ONLY_NO_DEVICE_ACQUISITION')]
        for op in self.spec['operations']:
            self.cpu_lines.append(record('EXPR', op, json.dumps({'parser_unit_only': op})))
            for band in self.spec['bands']:
                inputs = [(bits(value, 'f'), bits(value, 'd')) for value in [1.0, 2.0]]
                self.cpu_lines.append(record('FIXTURE', op, band, len(inputs),
                                             sum(x for x, _ in inputs), sum(y for _, y in inputs)))
                for index, (value, reference) in enumerate(inputs):
                    self.cpu_lines.append(record('INPUT', op, band, index, value, reference,
                                                 16 if op == 'GeluGradOp' else 32, 'none'))
        self.cpu = verify.manifest(self.write('cpu.tsv', ''.join(self.cpu_lines)))
        self.predicates = {'fixture_sha256': self.cpu['fixture_sha256'],
                           'expression_sha256': self.meta['expression_sha256'], 'predicates': []}

    def measured_lines(self):
        lines = [record('META', k, v) for k, v in {**self.meta, **self.identity}.items()]
        lines += [line for line in self.cpu_lines if line.startswith(('EXPR\t', 'FIXTURE\t'))]
        for op, band, length in itertools.product(self.spec['operations'], self.spec['bands'], self.spec['sizes']):
            lines.append(record('CHECK', op, band, length, 'before', 0, 0.0, 0.0, 0, length * bits(1.0, 'f')))
            for sample in range(self.spec['samples']):
                # Analytic parser integers; these are not measured nanoseconds.
                lines.append(record('TIMING', op, band, length, sample, self.spec['batch'], length * (sample + 1)))
            lines.append(record('CHECK', op, band, length, 'after', 0, 0.0, 0.0, 0, length * bits(1.0, 'f')))
        return lines + [record('FINAL', 'computed_value_failures', 0)]

    def parse(self, lines, **kwargs):
        return verify.measured_run(self.write('parser-unit.tsv', ''.join(lines)), self.cpu,
                                   self.spec, 'base', self.predicates, **kwargs)

    def test_manifest_computes_input_digest_and_fixture_values(self):
        input_bytes = ''.join(line for line in self.cpu_lines if line.startswith('INPUT\t')).encode()
        self.assertEqual(self.cpu['fixture_sha256'], hashlib.sha256(input_bytes).hexdigest())
        self.assertEqual(len(self.cpu['fixtures']), 9)
        self.assertEqual(self.cpu['inputs']['GeluOp', 'ordinary'][1],
                         (1, 0x40000000, 0x4000000000000000, 32.0, None))
        self.assertEqual(self.cpu['fixtures']['GeluOp', 'ordinary'],
                         (2, 0x7f800000, 0x7ff0000000000000))

    def test_manifest_rejects_duplicate_missing_and_nonconsecutive_rows(self):
        fixture = next(line for line in self.cpu_lines if line.startswith('FIXTURE\t'))
        input_row = next(line for line in self.cpu_lines if line.startswith('INPUT\t'))
        for lines, reason in [(self.cpu_lines + [fixture], 'duplicate fixture'),
                              (self.cpu_lines + [self.cpu_lines[0]], 'duplicate manifest metadata'),
                              ([x for x in self.cpu_lines if x != fixture], 'incomplete'),
                              ([x for x in self.cpu_lines if x != input_row], 'nonconsecutive')]:
            with self.subTest(reason=reason):
                self.rejected(reason, verify.manifest, self.write('bad-cpu.tsv', ''.join(lines)))

    def test_manifest_rejects_fixture_checksum_corruption(self):
        lines = list(self.cpu_lines)
        index = next(i for i, line in enumerate(lines) if line.startswith('FIXTURE\t'))
        fields = lines[index].rstrip('\n').split('\t')
        fields[4] = str(int(fields[4]) + 1)
        lines[index] = record(*fields)
        text = ''.join(lines)
        self.assertNotEqual(text, ''.join(self.cpu_lines))
        self.rejected('checksum', verify.manifest, self.write('bad-cpu.tsv', text))

    def test_manifest_rejects_malformed_and_incomplete_expression_rows(self):
        for suffix, reason in [('META\trevision\n', 'malformed|invalid|duplicate'),
                               ('EXPR\tGeluOp\n', 'malformed|invalid|duplicate')]:
            with self.subTest(suffix=suffix):
                self.rejected(reason, verify.manifest, self.write('bad-cpu.tsv', ''.join(self.cpu_lines) + suffix))
        lines = [line for line in self.cpu_lines if not line.startswith('EXPR\tGeluOp\t')]
        self.rejected('expression|incomplete', verify.manifest, self.write('bad-cpu.tsv', ''.join(lines)))

    def test_parser_returns_exact_all_case_sample_values(self):
        result = self.parse(self.measured_lines(), diagnostic=self.identity)
        expected = {(op, band, length, sample): length * (sample + 1)
                    for op, band, length, sample in itertools.product(
                        self.spec['operations'], self.spec['bands'], self.spec['sizes'], range(self.spec['samples']))}
        self.assertEqual(result, {'identity': self.identity, 'failures': 0, 'timings': expected})
        self.assertEqual(len(expected), 162)

    def test_rejects_missing_duplicate_and_invalid_samples(self):
        lines = self.measured_lines()
        timing = next(line for line in lines if line.startswith('TIMING\t'))
        for replacement, reason in [([], 'check/sample order mismatch'), ([timing, timing], 'duplicate/unexpected sample'),
                                    ([timing.replace('\t0\t8\t', '\t9\t8\t')], 'duplicate/unexpected sample'),
                                    ([timing.replace('\t8\t65536', '\t7\t65536')], 'batch/elapsed'),
                                    ([timing.rsplit('\t', 1)[0] + '\t0\n'], 'batch/elapsed'),
                                    (['TIMING\tGeluOp\n'], 'malformed timing')]:
            with self.subTest(reason=reason, replacement=replacement):
                index = lines.index(timing)
                self.rejected(reason, self.parse, lines[:index] + replacement + lines[index + 1:])

    def test_rejects_incomplete_and_impossible_failure_counts(self):
        lines = self.measured_lines()
        check = next(line for line in lines if line.startswith('CHECK\t'))
        for changed, reason in [(lines[:-1], 'final failure count'),
                                ([x for x in lines if x != check], 'sample outside computed checks'),
                                (lines[:-1] + [check, lines[-1]], 'duplicate/unexpected check'),
                                ([x.replace('\tbefore\t0\t', '\tbefore\t65537\t') if x == check else x for x in lines], 'impossible failure count'),
                                ([x.replace('\tbefore\t0\t', '\tbefore\t1\t') if x == check else x for x in lines], 'final failure count')]:
            with self.subTest(reason=reason):
                self.rejected(reason, self.parse, changed)

    def test_rejects_wrong_check_order_and_records_after_final(self):
        lines = self.measured_lines()
        first, second = [i for i, line in enumerate(lines) if line.startswith('CHECK\t')][:2]
        wrong = list(lines)
        wrong[first], wrong[second] = wrong[second], wrong[first]
        self.rejected('check/sample order mismatch', self.parse, wrong)
        self.rejected('record follows final', self.parse, lines + [record('META', 'parser_unit_only', 'late')])

    def test_rejects_source_predicate_fixture_and_identity_mismatches(self):
        for key in ['fixture_sha256', 'expression_sha256']:
            original = self.predicates[key]
            self.predicates[key] = hashlib.sha256(key.encode()).hexdigest()
            with self.subTest(binding=key):
                self.rejected('predicates bind another', self.parse, self.measured_lines())
            self.predicates[key] = original
        for key in ['revision', 'expression_sha256', 'harness_sha256', 'validation_head']:
            with self.subTest(metadata=key):
                lines = [record('META', key, 'parser-unit-tampered') if x.startswith('META\t' + key + '\t') else x
                         for x in self.measured_lines()]
                self.rejected('metadata mismatch', self.parse, lines)
        changed = copy.deepcopy(self.identity)
        changed['features'] = 'Features(0x6)'
        self.rejected('cohort mismatch', self.parse, self.measured_lines(), diagnostic=changed)
        lines = [x.replace('backend: Metal', 'backend: Vulkan') for x in self.measured_lines()]
        self.rejected('physical Metal', self.parse, lines)
        fixture = next(x for x in self.measured_lines() if x.startswith('FIXTURE\t'))
        self.rejected('fixture mismatch', self.parse, [x for x in self.measured_lines() if x != fixture])

    def test_runtime_expressions_and_fixtures_are_bound_without_duplicates(self):
        lines = self.measured_lines()
        expression = next(x for x in lines if x.startswith('EXPR\t'))
        self.rejected('runtime expression mismatch', self.parse,
                      [x.replace('parser_unit_only', 'tampered_unit_record') if x == expression else x for x in lines])
        for kind in ['EXPR', 'FIXTURE']:
            with self.subTest(kind=kind):
                duplicate = next(x for x in lines if x.startswith(kind + '\t'))
                self.rejected('duplicate', self.parse, lines[:-1] + [duplicate, lines[-1]])

    def failed_lines(self, count=1):
        lines = self.measured_lines()
        op, band, length, phase = 'GeluOp', 'ordinary', self.spec['sizes'][0], 'before'
        self.predicates['predicates'] = [{'op': op, 'band': band, 'input_bits': bits(1.0, 'f'), 'class': 'zero'}]
        lines = [record('CHECK', op, band, length, phase, count, 2 ** 23, 1.0, 0, length * bits(1.0, 'f'))
                 if x.startswith(record('CHECK', op, band, length, phase).rstrip('\n') + '\t') else x for x in lines]
        lines[-1] = record('FINAL', 'computed_value_failures', count)
        detail = record('FAILURE_DETAIL', op, band, length, phase, bits(1.0, 'f'), 'zero', count, 0,
                        bits(1.0, 'd'), float(2 ** 23))
        index = next(i for i, line in enumerate(lines) if line.startswith('CHECK\t'))
        return lines[:index] + [detail] + lines[index:], detail

    def test_failure_detail_counts_and_first_output_are_computed(self):
        lines, detail = self.failed_lines()
        self.assertEqual(self.parse(lines)['failures'], 1)
        self.rejected('unclassified', self.parse, [x for x in lines if x != detail])
        self.rejected('failure class', self.parse, [x.replace('\tzero\t', '\tfinite_relative\t') for x in lines])
        self.rejected('reference/input mismatch', self.parse,
                      [x.replace(str(bits(1.0, 'd')), str(bits(2.0, 'd'))) if x == detail else x for x in lines])
        too_many, _ = self.failed_lines(self.spec['sizes'][0] // 2 + 1)
        self.rejected('per-input failures exceed', self.parse, too_many)

    def test_rejects_malformed_failure_detail(self):
        lines, detail = self.failed_lines()
        self.rejected('malformed failure detail', self.parse,
                      [x.rsplit('\t', 1)[0] + '\n' if x == detail else x for x in lines])

    def test_predicate_map_cannot_excuse_unlisted_or_candidate_failures(self):
        lines, _ = self.failed_lines()
        self.predicates['predicates'] = []
        self.rejected('unexpected baseline/candidate failure', self.parse, lines)
        self.failed_lines()
        self.rejected('unexpected baseline/candidate failure', verify.measured_run,
                      self.write('candidate-parser-unit.tsv', ''.join(lines)), self.cpu,
                      self.spec, 'head', self.predicates)


class ExportContract(TemporaryRecords):
    def setUp(self):
        super().setUp()
        self.operators = ['ErfcOp', 'GeluOp', 'GeluGradOp', 'GeluTanhGradOp',
                          'SoftplusOp', 'SiluOp', 'MishOp', 'MishGradOp']
        self.dialects = ['Wgsl', 'CudaC', 'HipC']
        self.expected = {(op, dialect): {'parser_unit_only': [op, dialect]}
                         for op, dialect in itertools.product(self.operators, self.dialects)}
        self.lines = [record(op, dialect, json.dumps(expression))
                      for (op, dialect), expression in self.expected.items()]

    def test_exact_operator_dialect_values(self):
        self.assertEqual(verify.exports(self.write('exports.tsv', ''.join(self.lines))), self.expected)

    def test_rejects_missing_duplicate_and_wrong_domain_exports(self):
        for lines, reason in [(self.lines[:-1], 'eight operators'),
                              (self.lines + [self.lines[0]], 'duplicate expression'),
                              ([x.replace('HipC\t', 'ParserUnitDialect\t') for x in self.lines],
                               'operator|dialect|export')]:
            with self.subTest(reason=reason):
                self.rejected(reason, verify.exports, self.write('bad-exports.tsv', ''.join(lines)))


class DiagnosticContract(TemporaryRecords):
    def diagnostic(self, count=6):
        identity = {'adapter': 'Some(AdapterInfo { name: "PARSER UNIT \\\"quoted\\\" (x)", '
                               'device_type: DiscreteGpu, backend: Metal })',
                    'features': 'Features(0x15)', 'device_limits': 'Limits { nested: (8, [16, 32]) }'}
        line = 'observed adapter; absent native compiler/driver versions remain unknown '
        line += ' '.join(f'{name}={identity[key]}' for name, key in
                         [('adapter', 'adapter'), ('enabled_features', 'features'), ('device_limits', 'device_limits')]) + '\n'
        return identity, [line] + [f'2026-10-20T23:25:01Z INFO witness{{target_ulp=3}}:assert: contracts: '
            f'exact intrinsic witness result error_ulp={index / 2} target_ulp=3\n' for index in range(count)]

    def test_nested_quoted_debug_fields_and_six_witnesses(self):
        identity, lines = self.diagnostic()
        self.assertEqual(verify.diagnostic_identity(self.write('diagnostic.txt', ''.join(lines))), identity)
        boundary = [x.replace('error_ulp=2.5', 'error_ulp=3') for x in lines]
        self.assertEqual(verify.diagnostic_identity(self.write('boundary.txt', ''.join(boundary))), identity)
        self.assertEqual(verify.field_value('a=Limits { tuple: (2, [4, 8]) } b=3', 'a'),
                         'Limits { tuple: (2, [4, 8]) }')
        self.rejected('missing diagnostic field', verify.field_value, 'adapter=Some(7)', 'enabled_features')

    def test_rejects_missing_duplicate_excessive_nonfinite_witnesses(self):
        _, lines = self.diagnostic()
        for changed, reason in [(lines[:-1], 'six results'), (lines + [lines[0]], 'identity/six'),
                                (lines + [lines[-1]], 'six results'),
                                ([x.replace('error_ulp=2.5', 'error_ulp=3.5') for x in lines], 'exceeds'),
                                ([x.replace('error_ulp=2.5', 'error_ulp=NaN') for x in lines], 'exceeds'),
                                ([x.replace(' target_ulp=3\n', '\n') for x in lines], 'missing diagnostic field'),
                                ([x.replace(' target_ulp=3\n', ' target_ulp=3 target_ulp=4\n') for x in lines], 'duplicate diagnostic field'),
                                ([x.replace('error_ulp=2.5', 'error_ulp=2.5 error_ulp=1') for x in lines], 'duplicate diagnostic field')]:
            with self.subTest(reason=reason):
                self.rejected(reason, verify.diagnostic_identity, self.write('bad-diagnostic.txt', ''.join(changed)))

    def test_rejects_unterminated_debug_identity(self):
        for text in ['adapter=Some(AdapterInfo { name: "unfinished', 'adapter=Some([7]']:
            with self.subTest(text=text):
                self.rejected('unterminated|unbalanced|malformed', verify.field_value, text, 'adapter')

    def test_debug_field_names_do_not_match_suffixes(self):
        self.assertEqual(verify.field_value('other_adapter=Some(7) adapter=Some(11)', 'adapter'), 'Some(11)')


class ArtifactContract(TemporaryRecords):
    def manifest(self, path):
        payload = {'files': [{'path': path.relative_to(self.root).as_posix(),
                              'bytes': path.stat().st_size, 'sha256': hashlib.sha256(path.read_bytes()).hexdigest()}]}
        manifest = self.write('manifest.json', json.dumps(payload))
        return manifest, payload

    def test_real_file_hash_and_same_size_corruption(self):
        path = self.write('activation/data.tsv', 'INPUT\t1\t2\t4\n')
        manifest, payload = self.manifest(path)
        self.assertEqual(verify.sha256(path), payload['files'][0]['sha256'])
        self.assertEqual(verify.artifact_files(self.root, manifest), 1)
        path.write_bytes(path.read_bytes().replace(b'4', b'8'))
        self.rejected('hash/size mismatch', verify.artifact_files, self.root, manifest)

    def test_real_size_corruption_and_missing_file(self):
        path = self.write('activation/data.tsv', 'exact artifact\n')
        manifest, _ = self.manifest(path)
        path.write_text('exact artifact\nextra\n')
        self.rejected('hash/size mismatch', verify.artifact_files, self.root, manifest)
        path.unlink()
        self.rejected('hash/size mismatch', verify.artifact_files, self.root, manifest)

    def test_path_traversal_absolute_and_symlink_escapes(self):
        outer = self.write('outer.tsv', 'outside artifact root\n')
        root = self.root / 'artifacts'
        root.mkdir()
        (root / 'linked.tsv').symlink_to(outer)
        (root / 'directory').symlink_to(self.root, target_is_directory=True)
        for entry in ['../outer.tsv', str(outer), 'linked.tsv', 'directory/outer.tsv']:
            with self.subTest(entry=entry):
                manifest = self.write('paths.json', json.dumps({'files': [{'path': entry,
                    'bytes': outer.stat().st_size, 'sha256': verify.sha256(outer)}]}))
                self.rejected('escapes root', verify.artifact_files, root, manifest)


class DriverContract(TemporaryRecords):
    def test_shallow_merge_fetches_missing_fixed_object_before_production_diff(self):
        origin, checkout = self.root / 'origin', self.root / 'shallow-merge'
        origin.mkdir()
        env = {**os.environ, 'GIT_CONFIG_NOSYSTEM': '1', 'GIT_CONFIG_GLOBAL': os.devnull,
               'GIT_AUTHOR_NAME': 'Parser Unit', 'GIT_COMMITTER_NAME': 'Parser Unit',
               'GIT_AUTHOR_EMAIL': 'parser-unit@example.invalid', 'GIT_COMMITTER_EMAIL': 'parser-unit@example.invalid'}
        def git(*args, data=None, check=True):
            return subprocess.run(['git', '-c', 'protocol.file.allow=always', *args], input=data,
                                  capture_output=True, text=True, env=env, timeout=10, check=check)
        git('init', '-q', str(origin))
        commits = []
        for value in range(1, 5):
            blob = git('-C', str(origin), 'hash-object', '-w', '--stdin', data=f'pub const VALUE: u32 = {value};\n').stdout.strip()
            tree = git('-C', str(origin), 'mktree', data=f'100644 blob {blob}\tlib.rs\n').stdout.strip()
            for component in ['src', 'core', 'crates']:
                tree = git('-C', str(origin), 'mktree', data=f'040000 tree {tree}\t{component}\n').stdout.strip()
            parents = [] if not commits else ([commits[0]] if value < 4 else commits[1:3])
            parent_args = list(itertools.chain.from_iterable(('-p', parent) for parent in parents))
            commits.append(git('-C', str(origin), 'commit-tree', tree, *parent_args, data=f'parser unit {value}\n').stdout.strip())
        git('-C', str(origin), 'update-ref', 'refs/heads/main', commits[-1])
        git('-C', str(origin), 'symbolic-ref', 'HEAD', 'refs/heads/main')
        git('clone', '--depth=1', origin.as_uri(), str(checkout))
        fixed = commits[1]
        self.assertNotEqual(git('-C', str(checkout), 'cat-file', '-e', fixed + '^{commit}', check=False).returncode, 0)
        self.assertEqual(git('-C', str(checkout), 'rev-parse', '--is-shallow-repository').stdout.strip(), 'true')
        fetches = []
        def execute(args, name, **kwargs):
            actual = list(args)
            if 'fetch' in actual:
                remote = 'https://github.com/ryancinsight/hephaestus.git'
                self.assertIn(remote, actual)
                self.assertIn('--depth=1', actual)
                self.assertEqual(actual[-1], fixed)
                actual[actual.index(remote)] = origin.as_uri()  # Authorized local transport; real Git fetch.
                fetches.append(tuple(actual))
            result = git(*actual[1:], check=kwargs.get('check', True))
            return {'exit_code': result.returncode}, None, None
        self.rejected('missing|local', driver.ensure_comparison_object, checkout, fixed, False, execute)
        self.assertEqual(fetches, [])
        driver.ensure_comparison_object(checkout, fixed, True, execute)
        self.assertEqual(len(fetches), 1)
        self.assertEqual(git('-C', str(checkout), 'cat-file', '-p', fixed + ':crates/core/src/lib.rs').stdout,
                         'pub const VALUE: u32 = 2;\n')
        self.assertEqual(git('-C', str(checkout), 'diff', '--name-only', fixed, 'HEAD', '--', 'crates').stdout,
                         'crates/core/src/lib.rs\n')
        driver.ensure_comparison_object(checkout, fixed, True, execute)
        self.assertEqual(len(fetches), 1)

    def test_installed_context_checks_every_hosted_identity_field(self):
        environ = {'GITHUB_ACTIONS': 'true', 'GITHUB_EVENT_NAME': 'pull_request',
                   'GITHUB_REPOSITORY': 'ryancinsight/hephaestus', 'GITHUB_RUN_ID': '123456789',
                   'HEPHAESTUS_HOSTED_ACTIVATION_INTENT': 'reviewed-metal-validation'}
        self.assertIsNone(driver.installed_context(environ, 'Darwin'))
        for key, reason in [('GITHUB_ACTIONS', 'local process'), ('GITHUB_EVENT_NAME', 'automatic PR'),
                            ('GITHUB_REPOSITORY', 'repository'), ('GITHUB_RUN_ID', 'run identity'),
                            ('HEPHAESTUS_HOSTED_ACTIVATION_INTENT', 'workflow intention')]:
            with self.subTest(key=key):
                self.rejected(reason, driver.installed_context, {**environ, key: 'parser-unit-invalid'}, 'Darwin')
        self.rejected('macOS cohort', driver.installed_context, environ, 'Linux')
        self.rejected('run identity', driver.installed_context, {**environ, 'GITHUB_RUN_ID': '12x34'}, 'Darwin')

    def test_budget_exact_1500_second_remaining_boundary(self):
        spec = json.loads((HERE / 'comparison_spec.json').read_text())
        self.assertEqual(spec['budgets']['phase_seconds'] + spec['budgets']['artifact_reserve_seconds']
                         + spec['budgets']['setup_margin_seconds'], 1500)
        origin = 1_700_000_000
        for elapsed, remaining, sufficient in [(1199, 1501, True), (1200, 1500, True), (1201, 1499, False)]:
            with self.subTest(elapsed=elapsed):
                observed = driver.budget_record(spec, origin, origin + elapsed)
                self.assertEqual(observed, {'origin_scope': 'first user step; runner initialization predates this clock',
                    'origin_epoch': origin, 'observed_epoch': origin + elapsed, 'elapsed_seconds': elapsed,
                    'raw_remaining_seconds': remaining, 'required_remaining_seconds': 1500,
                    'setup_margin_seconds': 60, 'job_budget_seconds': 2700, 'phase_seconds': 1320,
                    'artifact_reserve_seconds': 120, 'sufficient': sufficient})
        for invalid in [0, -1, origin + 1, float(origin)]:
            with self.subTest(origin=invalid):
                self.rejected('invalid/future', driver.budget_record, spec, invalid, origin)

    def test_driver_has_no_native_time_or_sysctl_command_paths(self):
        constants = {node.value for node in ast.walk(ast.parse((HERE / 'run.py').read_text()))
                     if isinstance(node, ast.Constant) and isinstance(node.value, str)}
        self.assertFalse(hasattr(driver, 'memory_record'))
        for prohibited in ['/usr/bin/time', 'sysctl', 'kern.clockrate']:
            with self.subTest(command=prohibited):
                self.assertNotIn(prohibited, constants)

    def test_external_graph_hash_is_canonical_and_excludes_workspace_packages(self):
        checksum = hashlib.sha256(b'parser-unit-alpha').hexdigest()
        alpha = '[[package]]\nname = "alpha"\nversion = "1.2.3"\nsource = "registry+https://unit.invalid/index"\n'
        alpha += 'checksum = "' + checksum + '"\n'
        beta = '[[package]]\nname = "beta"\nversion = "2.3.4"\nsource = "git+https://unit.invalid/repository#123"\n'
        workspace = '[[package]]\nname = "workspace-only"\nversion = "9.8.7"\n'
        # Literal sorted JSON binds the exact separators, tuple order and missing-checksum null.
        canonical = ('[["alpha","1.2.3","registry+https://unit.invalid/index","' + checksum +
                     '"],["beta","2.3.4","git+https://unit.invalid/repository#123",null]]').encode()
        expected = hashlib.sha256(canonical).hexdigest(), 2
        for index, content in enumerate([alpha + beta + workspace, workspace + beta + alpha]):
            with self.subTest(order=index):
                self.assertEqual(driver.external_graph(self.write(f'lock-{index}.toml', content)), expected)
        changed = self.write('changed-lock.toml', (alpha + beta).replace('2.3.4', '2.3.5'))
        changed_canonical = canonical.replace(b'2.3.4', b'2.3.5')
        self.assertEqual(driver.external_graph(changed), (hashlib.sha256(changed_canonical).hexdigest(), 2))


if __name__ == '__main__':
    unittest.main()
