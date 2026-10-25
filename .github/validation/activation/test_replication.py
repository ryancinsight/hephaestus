"""CPU-only campaign control tests using the existing analytical TSV fixtures.

The scheduler receives fixture files instead of starting a child process. No
backend is acquired and none of these parser/control records are device evidence.
"""
import copy
import json
from pathlib import Path
import sys
import time
import unittest
from unittest import mock

import test_validation as fixtures

verify = fixtures.verify
driver = fixtures.driver


class ReplicationContract(unittest.TestCase):
    def setUp(self):
        self.new_campaign()

    def new_campaign(self):
        self.records = fixtures.RecordContract()
        self.records.setUp()
        self.addCleanup(self.records.doCleanups)
        self.root = self.records.root
        self.spec = copy.deepcopy(self.records.spec)
        self.spec['replication_order'] = ['head', 'base', 'base', 'head']
        self.manifests = {}
        self.subject = driver.Driver.__new__(driver.Driver)
        self.subject.spec = self.spec
        self.subject.artifacts = self.root
        self.subject.packet = self.root / 'activation'
        self.subject.packet.mkdir()
        self.subject.deadline = time.monotonic() + self.spec['budgets']['phase_seconds']
        self.subject.record = {'builds': {}, 'commands': [], 'campaign': [], 'device_attempts': 0,
                               'diagnostic_identity': self.records.identity, 'performance_accepted': False}
        for cohort in ['base', 'head']:
            cpu = copy.deepcopy(self.records.cpu)
            for key, field in [('revision', 'revision'), ('expression_sha256', 'expression_sha256')]:
                cpu['meta'][key] = self.spec['cohorts'][cohort][field]
            self.manifests[cohort] = cpu
            binary = self.records.write(f'activation/{cohort}-activation_paths', f'PARSER UNIT DATA {cohort}\n')
            self.subject.record['builds'][cohort] = {'binary': str(binary.relative_to(self.root)),
                                                     'binary_sha256': verify.sha256(binary)}
        self.subject.command = self.command
        self.faults = {}
        self.after_command = None

    def command(self, args, name, **kwargs):
        index = len(self.subject.record['commands'])
        cohort = self.subject.record['campaign'][-1]['cohort']
        self.assertEqual(Path(args[0]).name, f'{cohort}-activation_paths')
        self.assertEqual(args[1:], ['--run-device'])
        self.assertEqual(kwargs['cap'], self.spec['budgets']['invocation_seconds'])
        self.assertFalse(kwargs['check'])
        original = self.records.meta
        self.records.meta = self.manifests[cohort]['meta']
        try:
            lines = self.records.failed_lines()[0] if self.faults.get(index) == 'numeric' else self.records.measured_lines()
        finally:
            self.records.meta = original
        out = self.records.write(f'activation/unit-{index}.stdout', ''.join(lines))
        error = 'ACQUISITION_FAILURE\tUNIT ONLY\n' if self.faults.get(index) == 'acquisition' else ''
        err = self.records.write(f'activation/unit-{index}.stderr', error)
        exit_code = {'numeric': 1, 'acquisition': 2, 'runtime': 3, 'timeout': -9}.get(self.faults.get(index), 0)
        item = {'name': name, 'args': args, 'exit_code': exit_code, 'timed_out': self.faults.get(index) == 'timeout',
                'elapsed_seconds': 0.001, 'stdout': str(out.relative_to(self.root)), 'stderr': str(err.relative_to(self.root))}
        self.subject.record['commands'].append(item)
        if self.after_command:
            self.after_command(index, out)
        if item['timed_out']:
            raise driver.Stopped('CHILD_TIMEOUT_UNMEASURED', 'CPU scheduler timeout fixture; no child starts')
        return item, out, err

    def campaign(self):
        self.records.write('baseline_failures.json', json.dumps(self.records.predicates))
        with mock.patch.object(driver, 'HERE', self.root):
            self.subject.campaign(self.manifests)
        return self.subject.record

    def test_complete_campaign_preserves_BHHB_then_HBBH(self):
        result = self.campaign()
        self.assertEqual([r['cohort'] for r in result['campaign']],
                         ['base', 'head', 'head', 'base', 'head', 'base', 'base', 'head'])
        self.assertEqual(result['device_attempts'], 8)
        runs = self.verified(result)
        self.assertEqual(sum(len(run['timings']) for _, run in runs), 1296)
        self.assertEqual(result['timing_summary'], verify.timing_summary(runs[:4]))
        self.assertEqual(result['block_timing_summaries']['reversed'], verify.timing_summary(runs[4:]))
        self.assertEqual([(row['index'], row['block'], row['slot']) for row in result['campaign']],
                         [(i, 'mandatory' if i < 4 else 'reversed', i % 4) for i in range(8)])

    def verified(self, result=None):
        return verify.campaign_runs(self.root, result or self.subject.record, self.spec,
                                    self.manifests, self.records.predicates, self.records.identity)

    def test_layout_rejects_changed_mandatory_and_reversed_orders(self):
        for key in ['order', 'replication_order']:
            spec = copy.deepcopy(self.spec)
            spec[key][0], spec[key][1] = spec[key][1], spec[key][0]
            with self.subTest(key=key), self.assertRaises(verify.InvalidEvidence):
                verify.campaign_layout(spec)

    def test_admission_exact_derived_boundary_and_unchanged_entry_guard(self):
        for remaining, sufficient in [(479.999, False), (480, True), (480.001, True)]:
            with self.subTest(remaining=remaining):
                self.assertEqual(verify.replication_admission(self.spec, remaining),
                                 {'required_remaining_phase_seconds': 480,
                                  'observed_remaining_phase_seconds': remaining, 'sufficient': sufficient})
        self.assertEqual(driver.budget_record(self.spec, 1000, 2200)['required_remaining_seconds'], 1500)

    def test_insufficient_extra_budget_preserves_only_mandatory_block(self):
        self.subject.deadline = time.monotonic() + 479
        original = self.subject.deadline
        with self.assertRaisesRegex(driver.Stopped, 'no fifth acquisition'):
            self.campaign()
        result = self.subject.record
        self.assertEqual(self.subject.deadline, original)
        self.assertEqual(result['device_attempts'], 4)
        self.assertEqual(result['campaign_blocks'][0]['status'], 'VERIFIED')
        self.assertEqual(result['campaign_blocks'][1]['status'], 'UNMEASURED_OR_INVALID')
        self.assertEqual(set(result['block_timing_summaries']), {'mandatory'})
        with self.assertRaisesRegex(verify.InvalidEvidence, 'incomplete'):
            self.verified()

    def test_every_extra_slot_stops_after_acquisition_runtime_or_timeout_fault(self):
        for slot in range(4, 8):
            for fault in ['acquisition', 'runtime', 'timeout']:
                with self.subTest(slot=slot, fault=fault):
                    self.new_campaign()
                    self.faults[slot] = fault
                    original = self.subject.deadline
                    with self.assertRaises(verify.InvalidEvidence):
                        self.campaign()
                    self.assertEqual(len(self.subject.record['commands']), slot + 1)
                    self.assertEqual(self.subject.record['device_attempts'], slot + 1)
                    self.assertEqual(self.subject.deadline, original)
                    self.assertEqual(self.subject.record['campaign_blocks'][0]['status'], 'VERIFIED')
                    self.assertEqual(self.subject.record['campaign_blocks'][1]['status'], 'UNMEASURED_OR_INVALID')

    def test_head_slots_four_and_seven_cannot_use_baseline_predicates(self):
        for slot in [4, 7]:
            with self.subTest(slot=slot):
                self.new_campaign()
                self.records.failed_lines()  # Retain the exact allowed baseline predicate.
                self.faults[slot] = 'numeric'
                with self.assertRaisesRegex(verify.InvalidEvidence, 'unexpected baseline/candidate failure'):
                    self.campaign()
                self.assertEqual(self.subject.record['device_attempts'], slot + 1)

    def test_mandatory_fault_is_never_rescued_by_added_block(self):
        for slot in range(4):
            with self.subTest(slot=slot):
                self.new_campaign()
                self.faults[slot] = 'runtime'
                with self.assertRaises(verify.InvalidEvidence):
                    self.campaign()
                self.assertEqual(self.subject.record['device_attempts'], slot + 1)
                self.assertEqual(len(self.subject.record['campaign_blocks']), 1)
                self.assertEqual(self.subject.record['block_timing_summaries'], {})

    def test_baseline_slots_five_and_six_use_fixed_predicates(self):
        for slot in [5, 6]:
            with self.subTest(slot=slot):
                self.new_campaign()
                self.records.failed_lines()
                self.faults[slot] = 'numeric'
                result = self.campaign()
                self.assertEqual(result['campaign'][slot]['classified_baseline_failures'], 1)
                self.assertEqual(result['campaign'][slot]['status'], 'VERIFIED_WITH_EXPECTED_BASELINE_FAILURES')
                self.assertEqual(len(self.verified()), 8)

    def test_unlisted_failure_in_each_new_baseline_slot_stops(self):
        for slot in [5, 6]:
            with self.subTest(slot=slot):
                self.new_campaign()
                self.faults[slot] = 'numeric'
                with self.assertRaisesRegex(verify.InvalidEvidence, 'unexpected baseline/candidate failure'):
                    self.campaign()
                self.assertEqual(self.subject.record['device_attempts'], slot + 1)

    def test_identity_change_in_every_extra_slot_stops(self):
        for slot in range(4, 8):
            with self.subTest(slot=slot):
                self.new_campaign()
                def change(index, path):
                    if index == slot:
                        path.write_text(path.read_text().replace('Features(0x5)', 'Features(0x6)'))
                self.after_command = change
                with self.assertRaisesRegex(verify.InvalidEvidence, 'cohort mismatch'):
                    self.campaign()
                self.assertEqual(self.subject.record['device_attempts'], slot + 1)

    def test_binary_change_between_blocks_prevents_fifth_acquisition(self):
        def change(index, unused):
            if index == 3:
                (self.root / self.subject.record['builds']['head']['binary']).write_text('changed unit data\n')
        self.after_command = change
        with self.assertRaisesRegex(verify.InvalidEvidence, 'copied binary changed'):
            self.campaign()
        self.assertEqual(self.subject.record['device_attempts'], 4)

    def test_complete_verifier_rejects_missing_duplicate_reordered_and_relabelled_rows(self):
        result = self.campaign()
        variants = []
        for index in range(4, 8):
            missing = copy.deepcopy(result)
            missing['campaign'].pop(index)
            variants.append(missing)
            for key, value in [('index', 0), ('slot', 9), ('block', 'mandatory'), ('cohort', 'other')]:
                changed = copy.deepcopy(result)
                changed['campaign'][index][key] = value
                variants.append(changed)
        swapped = copy.deepcopy(result)
        swapped['campaign'][4], swapped['campaign'][5] = swapped['campaign'][5], swapped['campaign'][4]
        variants.append(swapped)
        for index, value in enumerate(variants):
            with self.subTest(variant=index), self.assertRaises(verify.InvalidEvidence):
                self.verified(value)

    def test_complete_verifier_rejects_changed_binary_command_identity_and_summary(self):
        result = self.campaign()
        for key, value in [('binary_sha256', 'changed'), ('command_index', 0), ('exit_code', 1),
                           ('status', 'STARTED_OUTCOME_PENDING'), ('verified_samples', 161),
                           ('classified_baseline_failures', 1), ('stdout', result['campaign'][0]['stdout'])]:
            changed = copy.deepcopy(result)
            changed['campaign'][4][key] = value
            with self.subTest(key=key), self.assertRaises(verify.InvalidEvidence):
                self.verified(changed)
        for block in ['mandatory', 'reversed']:
            changed = copy.deepcopy(result)
            changed['block_timing_summaries'][block][0]['head_over_base_median_ratio'] += 1
            with self.subTest(block=block), self.assertRaisesRegex(verify.InvalidEvidence, 'summary'):
                self.verified(changed)

    def test_complete_verifier_rejects_invalid_time_span_and_admission(self):
        result = self.campaign()
        for key, value in [('phase_started_seconds', -1), ('phase_finished_seconds', 1321),
                           ('phase_started_seconds', float('nan')), ('phase_finished_seconds', False)]:
            changed = copy.deepcopy(result)
            changed['campaign'][4][key] = value
            with self.subTest(key=key, value=value), self.assertRaises(verify.InvalidEvidence):
                self.verified(changed)
        changed = copy.deepcopy(result)
        changed['campaign_blocks'][1]['admission']['sufficient'] = False
        with self.assertRaisesRegex(verify.InvalidEvidence, 'admission'):
            self.verified(changed)

    def test_complete_verifier_rejects_ninth_unreferenced_device_command(self):
        result = self.campaign()
        extra = copy.deepcopy(result['commands'][0])
        extra['name'] = 'run-8-head'
        result['commands'].append(extra)
        with self.assertRaisesRegex(verify.InvalidEvidence, 'unreferenced'):
            self.verified(result)

    def test_real_CPU_child_timeout_uses_existing_phase_deadline(self):
        self.subject.deadline = time.monotonic() + 0.02
        with self.assertRaises(driver.Stopped):
            driver.Driver.command(self.subject, [sys.executable, '-c', 'import time; time.sleep(10)'],
                                  'unit-cpu-timeout', cap=self.spec['budgets']['invocation_seconds'])
        self.assertTrue(self.subject.record['commands'][-1]['timed_out'])
        self.assertEqual(self.subject.record['commands'][-1]['exit_code'], -9)

    def test_packet_rejects_unindexed_dynamic_log(self):
        indexed, _ = fixtures.ArtifactContract.manifest(self.records, self.records.write('indexed.txt', 'unit data\n'))
        (self.subject.packet / 'manifest.json').write_bytes(indexed.read_bytes())
        result = {'builds': {}, 'campaign': [{'stdout': 'not-in-manifest.txt'}], 'performance_accepted': False,
                  'artifact_manifest_sha256': verify.sha256(self.subject.packet / 'manifest.json')}
        self.records.write('activation-result.json', json.dumps(result))
        with self.assertRaisesRegex(verify.InvalidEvidence, 'unindexed'):
            verify.packet(self.root)

    def test_partial_packet_stays_unmeasured_and_old_complete_status_is_rejected(self):
        spec_file = self.records.write('activation/comparison_spec.json', json.dumps(self.spec))
        indexed, _ = fixtures.ArtifactContract.manifest(self.records, spec_file)
        (self.subject.packet / 'manifest.json').write_bytes(indexed.read_bytes())
        result = {'builds': {}, 'campaign': [], 'device_attempts': 0, 'performance_accepted': False,
                  'artifact_manifest_sha256': verify.sha256(self.subject.packet / 'manifest.json'),
                  'status': 'UNMEASURED_OR_INVALID'}
        self.records.write('activation-result.json', json.dumps(result))
        self.assertEqual(verify.packet(self.root)['measured_invocations_verified'], 0)
        result['status'] = 'MEASURED_COHORT_REVIEW_PENDING'
        self.records.write('activation-result.json', json.dumps(result))
        with self.assertRaisesRegex(verify.InvalidEvidence, 'unsupported measured'):
            verify.packet(self.root)


if __name__ == '__main__':
    unittest.main()
