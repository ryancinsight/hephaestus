"""CPU-only source-model tests using real compiled fixture inputs, never a device.

Pass --exports, --manifest, --previous-predicates and --previous-generator from
the accepted preparation packet/source. Missing fixtures are an error, not a skip.
"""
import argparse
import ast
from decimal import Decimal, localcontext
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import sys
import unittest

sys.dont_write_bytecode = True
import predict_baseline as predictor
import verify

HERE = Path(__file__).resolve().parent
FIXTURES = None


def tuple_key(row):
    return row['op'], row['band'], row['input_bits'], row['class']


class ReductionContract(unittest.TestCase):
    def test_log2e_is_independently_rounded_real_constant(self):
        proof = predictor.prove_log2e()
        self.assertEqual(proof['bits'], '3fb8aa3b')
        with localcontext() as context:
            context.prec = 100
            reference = Decimal(1) / Decimal(2).ln()
            lower, upper = map(Decimal, proof['real_interval'])
            self.assertLessEqual(lower, reference)
            self.assertLessEqual(reference, upper)

    def test_general_reduction_formula_on_unrelated_arguments(self):
        for value in [-40.0, -1.0, -0.125, 0.0, 0.125, 1.0, 40.0]:
            with self.subTest(argument=value):
                expected = math.exp2(predictor.rounded(value * predictor.LOG2_E))
                self.assertEqual(predictor.exp_model(value, True), expected)
                self.assertEqual(predictor.exp_model(value, False), math.exp(value))
                self.assertEqual(predictor.prove_reduction(predictor.word(value))['status'],
                                 'BOUNDED_FINITE_RESULT')

    def test_active_arguments_follow_selection_and_condition_dependencies(self):
        args = set()
        predictor.Expression('select(exp(88.0), exp(-1.0), x < 0.0)').evaluate(-1.0, False, True, args)
        self.assertEqual(args, {predictor.word(-1.0)})
        args.clear()
        predictor.Expression('select(exp(1.0), exp(2.0), exp(x) > 1.0)').evaluate(0.0, False, True, args)
        self.assertEqual(args, {predictor.word(0.0), predictor.word(1.0)})

    def test_intrinsic_error_proof_precedes_separate_ftz_accounting(self):
        row = predictor.prove_reduction(predictor.word(-100.0))
        self.assertEqual(row['status'], 'BOUNDED_FINITE_RESULT')
        self.assertEqual(row['preflush_result_class'], 'subnormal')
        self.assertNotEqual(row['preflush_result_bits'], '00000000')
        self.assertEqual(row['post_ftz_result_bits'], '00000000')
        self.assertLessEqual(Decimal(row['exp_error_upper_ulp']), Decimal(row['exp_allowance_ulp']))
        self.assertLessEqual(Decimal(row['exp2_error_upper_ulp']), Decimal(row['exp2_allowance_ulp']))
        smallest_normal = verify.single(0x00800000)
        largest_subnormal = verify.single(0x007fffff)
        self.assertEqual(predictor.rounded(smallest_normal, True), smallest_normal)
        self.assertEqual(predictor.rounded(largest_subnormal, True), 0.0)

    def test_zero_integer_reduction_and_extreme_finite_arguments(self):
        for value, status in [(0.0, 'BOUNDED_FINITE_RESULT'),
                              (-105.0, 'EXACT_RNE_UNDERFLOW'), (89.0, 'EXACT_RNE_OVERFLOW'),
                              (-1e30, 'EXACT_RNE_UNDERFLOW'), (1e30, 'EXACT_RNE_OVERFLOW')]:
            with self.subTest(argument=value):
                self.assertEqual(predictor.prove_reduction(predictor.word(value))['status'], status)
        value = predictor.rounded(math.log(2))
        proof = predictor.prove_reduction(predictor.word(value))
        self.assertEqual(proof['exp2_error_upper_ulp'], '0')

    def test_nonfinite_arguments_are_explicitly_outside_finite_accuracy_proof(self):
        for bits in [0x7f800000, 0xff800000, 0x7fc00000]:
            with self.subTest(argument_bits=bits):
                row = predictor.prove_reduction(bits)
                self.assertEqual(row['status'], 'NONFINITE_MODELED_SEPARATELY')
                self.assertNotIn('exp_error_upper_ulp', row)

    def test_generator_has_no_observed_input_output_or_log_parameters(self):
        source = (HERE / 'predict_baseline.py').read_text()
        constants = {n.value for n in ast.walk(ast.parse(source)) if isinstance(n, ast.Constant)}
        for forbidden in ['32-run-0-base.stdout', '39516156930', 'FAILURE_DETAIL', 'observed_baseline']:
            self.assertNotIn(forbidden, constants)
        # These regression inputs/results live only in tests, never model branches.
        for forbidden in [0xc14fe3f4, 0xc14905b3, 0xc1446f7a, 0xc13e0c2d,
                          0x81eca23d, 0x81eca23f, 0x85decc13, 0x8874e9b9, 0x8bed8a28]:
            self.assertNotIn(forbidden, constants)


class CompiledFixtureContract(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if FIXTURES is None:
            raise RuntimeError('Actual compiled fixture paths are required; no fixture skip')
        cls.exports, cls.manifest, cls.previous_predicates, previous_source = FIXTURES
        cls.previous = json.loads(cls.previous_predicates.read_text())
        if verify.sha256(previous_source) != cls.previous['generator_sha256']:
            raise RuntimeError('Previous generator source does not match accepted predicate seal')
        module = importlib.util.spec_from_file_location('accepted_previous_predictor', previous_source)
        cls.legacy = importlib.util.module_from_spec(module)
        module.loader.exec_module(cls.legacy)
        cls.cpu = verify.manifest(cls.manifest)
        cls.expressions = verify.exports(cls.exports)
        cls.audit = {}
        cls.generated = predictor.generate(cls.exports, cls.manifest, cls.audit)

    def test_original_profile_output_bits_are_preserved_for_every_fixture_input(self):
        checked = 0
        for (op, band), rows in self.cpu['inputs'].items():
            expression = self.expressions[op, 'Wgsl']
            old, new = self.legacy.Expression(expression), predictor.Expression(expression)
            for flush in [False, True]:
                for _, bits, *_ in rows:
                    value = verify.single(bits)
                    before, after = old.evaluate(value, flush), new.evaluate(value, flush)
                    if math.isnan(before):
                        self.assertTrue(math.isnan(after))
                    else:
                        self.assertEqual(predictor.word(before), predictor.word(after), (op, band, bits, flush))
                    checked += 1
        self.assertEqual(checked, 2 * sum(map(len, self.cpu['inputs'].values())))

    def test_all_prior_predicates_profiles_counts_and_bindings_remain(self):
        before = {tuple_key(row): row for row in self.previous['predicates']}
        after = {tuple_key(row): row for row in self.generated['predicates']}
        self.assertLessEqual(before.keys(), after.keys())
        for key, row in before.items():
            self.assertLessEqual(set(row['source_model_profiles']), set(after[key]['source_model_profiles']))
        for key, count in self.previous['profile_failure_counts'].items():
            self.assertEqual(self.generated['profile_failure_counts'][key], count)
        for key in ['expression_sha256', 'fixture_sha256']:
            self.assertEqual(self.generated[key], self.previous[key])

    def test_whole_fixture_admissibility_is_bound_complete_and_preflush(self):
        records = self.audit['arguments']
        summary = self.generated['exp2_profile_admissibility']
        canonical = json.dumps(records, sort_keys=True, separators=(',', ':')).encode()
        self.assertEqual(summary['proof_records_sha256'], hashlib.sha256(canonical).hexdigest())
        self.assertEqual(summary['argument_count'], len(records))
        self.assertEqual(sum(summary['status_counts'].values()), len(records))
        self.assertEqual(len({r['argument_bits'] for r in records}), len(records))
        for row in records:
            if row['status'] == 'BOUNDED_FINITE_RESULT':
                self.assertLessEqual(Decimal(row['exp_error_upper_ulp']), Decimal(row['exp_allowance_ulp']))
                self.assertLessEqual(Decimal(row['exp2_error_upper_ulp']), Decimal(row['exp2_allowance_ulp']))
            self.assertNotEqual(row['status'], 'UNVERIFIED')
            if row.get('preflush_result_class') == 'subnormal':
                self.assertNotEqual(row['preflush_result_bits'], '00000000')
                self.assertEqual(row['post_ftz_result_bits'], '00000000')

    def test_source_derived_four_class_regressions_do_not_match_device_outputs(self):
        expression = predictor.Expression(self.expressions['GeluOp', 'Wgsl'])
        rows = {r[1]: r for r in self.cpu['inputs']['GeluOp', 'tail']}
        # Expected bits come from the separately sealed exact-RNE/high-precision
        # decomposition, not the observed GPU outputs (which differ by1-3 ULP).
        expected = {0xc14fe3f4: 0x81eca23f, 0xc14905b3: 0x85decc13,
                    0xc1446f7a: 0x8874e9b9, 0xc13e0c2d: 0x8bed8a28}
        generated = {tuple_key(row): row for row in self.generated['predicates']}
        for bits, wanted in expected.items():
            _, _, reference_bits, bound, absolute = rows[bits]
            reference = verify.double(reference_bits)
            self.assertIsNone(verify.failure_class(expression.evaluate(verify.single(bits), False), reference, bound, absolute))
            for flush in [False, True]:
                got = expression.evaluate(verify.single(bits), flush, True)
                self.assertEqual(predictor.word(got), wanted)
                self.assertEqual(verify.failure_class(got, reference, bound, absolute), 'finite_relative')
            self.assertIn(('GeluOp', 'tail', bits, 'finite_relative'), generated)

    def test_generation_is_deterministic_and_matches_committed_map(self):
        again = predictor.generate(self.exports, self.manifest)
        self.assertEqual(again, self.generated)
        self.assertEqual(again, json.loads((HERE / 'baseline_failures.json').read_text()))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ['exports', 'manifest', 'previous-predicates', 'previous-generator']:
        parser.add_argument('--' + name, type=Path, required=True)
    options, rest = parser.parse_known_args()
    FIXTURES = options.exports, options.manifest, options.previous_predicates, options.previous_generator
    unittest.main(argv=[sys.argv[0], *rest])
