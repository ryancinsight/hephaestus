"""CPU-only protocol regressions from retained automatic Metal run39516156930."""
import copy
import tempfile
import unittest
from pathlib import Path

import verify

ACTUAL = {'adapter': 'AdapterInfo { name: "Apple Paravirtual device", vendor: 0, device: 0, device_type: IntegratedGpu, device_pci_bus_id: "", driver: "", driver_info: "", backend: Metal, subgroup_min_size: 4, subgroup_max_size: 64, transient_saves_memory: Some(true), limit_bucket: None }', 'features': 'Features { features_wgpu: FeaturesWGPU(0x0), features_webgpu: FeaturesWebGPU(0x0) }', 'device_limits': 'DeviceLimits { max_buffer_size: 268435456, max_compute_workgroup_size_x: 256, max_compute_workgroup_size_y: 256, max_compute_workgroup_size_z: 64, max_compute_invocations_per_workgroup: 256, max_compute_workgroup_storage_size: 16352, max_storage_buffers_per_shader_stage: Some(4), max_buffers_and_acceleration_structures_per_shader_stage: Some(24), max_immediate_size: 0 }'}


class AdapterIdentityContract(unittest.TestCase):
    def test_actual_option_and_bare_identities_are_equal(self):
        benchmark = copy.deepcopy(ACTUAL)
        benchmark['adapter'] = 'Some(' + benchmark['adapter'] + ')'
        self.assertNotEqual(benchmark, ACTUAL)
        self.assertEqual(verify.cohort_identity(benchmark, ACTUAL), benchmark)
        self.assertEqual(verify.cohort_identity(ACTUAL, ACTUAL), ACTUAL)

    def test_quoted_brackets_and_nested_metadata_are_preserved(self):
        identity = copy.deepcopy(ACTUAL)
        identity['adapter'] = identity['adapter'].replace('Apple Paravirtual device', r'Quoted } ) \" device [')
        self.assertEqual(verify.adapter_identity('Some(' + identity['adapter'] + ')'), identity['adapter'])
        self.assertIn('transient_saves_memory: Some(true)', verify.adapter_identity(identity['adapter']))

    def test_absent_or_malformed_option_and_debug_records_are_rejected(self):
        bare = ACTUAL['adapter']
        bad = [None, '', 'None', 'Some(None)', 'Some(Some(' + bare + '))',
               'Some(' + bare, 'Some(' + bare + '))', 'Some(' + bare + ') extra',
               'Some(' + bare + ' extra)', ' Some(' + bare + ')',
               bare[:-1], bare + ' trailing', bare.replace('Some(true)', 'Some[true)'),
               bare.replace('Some(true)', 'Some(true]'), bare.replace('"Apple Paravirtual device"', '"unterminated')]
        for value in bad:
            with self.subTest(value=value), self.assertRaises(verify.InvalidEvidence):
                verify.adapter_identity(value)

    def test_changed_adapter_fields_are_not_normalized_away(self):
        changes = [('vendor: 0', 'vendor: 1'), ('name: "Apple Paravirtual device"', 'name: "Changed device"'),
                   ('driver: ""', 'driver: "Changed driver"'), ('driver_info: ""', 'driver_info: "Changed info"'),
                   ('subgroup_min_size: 4', 'subgroup_min_size: 8')]
        for before, after in changes:
            changed = copy.deepcopy(ACTUAL)
            changed['adapter'] = 'Some(' + changed['adapter'].replace(before, after) + ')'
            with self.subTest(field=before), self.assertRaisesRegex(verify.InvalidEvidence, 'cohort mismatch'):
                verify.cohort_identity(changed, ACTUAL)

    def test_changed_features_or_limits_are_rejected(self):
        for key in ['features', 'device_limits']:
            changed = copy.deepcopy(ACTUAL)
            changed[key] += ' changed'
            with self.subTest(field=key), self.assertRaisesRegex(verify.InvalidEvidence, 'cohort mismatch'):
                verify.cohort_identity(changed, ACTUAL)

    def test_absent_identity_components_are_rejected(self):
        for key in ACTUAL:
            for absent in [None, '']:
                changed = copy.deepcopy(ACTUAL)
                changed[key] = absent
                with self.subTest(field=key, value=absent), self.assertRaisesRegex(verify.InvalidEvidence, 'missing device identity'):
                    verify.cohort_identity(changed, ACTUAL)

    def test_existing_metal_and_reported_device_type_guards_remain(self):
        for before, after in [('backend: Metal', 'backend: Vulkan'),
                              ('device_type: IntegratedGpu', 'device_type: Cpu'),
                              ('device_type: IntegratedGpu', 'device_type: Other')]:
            changed = copy.deepcopy(ACTUAL)
            changed['adapter'] = changed['adapter'].replace(before, after)
            with self.subTest(change=after), self.assertRaisesRegex(verify.InvalidEvidence, 'Metal adapter type'):
                verify.cohort_identity(changed, changed)

    def test_diagnostic_accepts_one_typed_identity_and_six_numeric_records(self):
        lines = ['observed adapter; absent native compiler/driver versions remain unknown ' +
                 ' '.join(f'{name}={ACTUAL[key]}' for name, key in [('adapter', 'adapter'),
                    ('enabled_features', 'features'), ('device_limits', 'device_limits')])]
        lines += ['exact intrinsic witness result error_ulp=0 target_ulp=128'] * 6
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / 'diagnostic.log'
            path.write_text('\n'.join(lines) + '\n')
            self.assertEqual(verify.diagnostic_identity(path), ACTUAL)


if __name__ == '__main__':
    unittest.main()
