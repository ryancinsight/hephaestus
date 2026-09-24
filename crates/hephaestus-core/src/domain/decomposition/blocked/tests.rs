mod loop_orchestration_tests {
    //! CPU exercises of the shared blocked loops.
    //!
    //! A loop's orchestration — the panel walk, the bookkeeping, and the
    //! ordering of the region writes and trailing updates — is pure host code;
    //! only the device operations are backend-specific. Implementing those on
    //! the CPU exercises every branch of [`blocked_lu`] and
    //! [`blocked_cholesky`] without a device, so the orchestration is covered
    //! by `cargo test -p hephaestus-core` instead of only by the accelerator
    //! backends' device-executed differentials.
    //!
    //! A buffer is an index into one store. That is what lets `write_region`
    //! and the trailing updates mutate it through `&self`, the way a real
    //! device handle does; a `Vec`-valued buffer could not.

    use std::cell::{Ref, RefCell};

    use super::super::*;
    use crate::domain::error::Result;

    #[derive(Default)]
    struct CpuBackend {
        store: RefCell<Vec<Vec<f32>>>,
        /// Whether the loop finishes a small final tail on the host — the
        /// policy a fixed-launch-cost trailing kernel selects.
        finish_tail_on_cpu: bool,
    }

    impl CpuBackend {
        fn put(&self, data: Vec<f32>) -> usize {
            let mut store = self.store.borrow_mut();
            store.push(data);
            store.len() - 1
        }

        fn read(&self, buffer: usize) -> Ref<'_, Vec<f32>> {
            Ref::map(self.store.borrow(), |store| &store[buffer])
        }

        fn mutate(&self, buffer: usize, edit: impl FnOnce(&mut [f32])) {
            edit(&mut self.store.borrow_mut()[buffer]);
        }
    }

    impl BlockedDecompositionBackend for CpuBackend {
        type Buffer = usize;

        fn alloc(&self, len: usize) -> Result<Self::Buffer> {
            Ok(self.put(vec![0.0; len]))
        }

        fn clone_device(&self, src: &Self::Buffer, len: usize) -> Result<Self::Buffer> {
            let mut data = self.read(*src).clone();
            data.truncate(len);
            Ok(self.put(data))
        }

        fn download_region(
            &self,
            buf: &Self::Buffer,
            region: PanelRegion,
            _scratch: &Self::Buffer,
            out: &mut Vec<f32>,
        ) -> Result<()> {
            let source = self.read(*buf);
            out.clear();
            for row in 0..region.rows {
                let base = (region.row0 + row) * region.stride + region.col0;
                out.extend_from_slice(&source[base..base + region.cols]);
            }
            Ok(())
        }

        fn write_region(
            &self,
            buf: &Self::Buffer,
            region: PanelRegion,
            _scratch: &Self::Buffer,
            data: &[f32],
        ) -> Result<()> {
            self.mutate(*buf, |target| {
                for row in 0..region.rows {
                    let base = (region.row0 + row) * region.stride + region.col0;
                    let src = row * region.cols;
                    target[base..base + region.cols].copy_from_slice(&data[src..src + region.cols]);
                }
            });
            Ok(())
        }

        fn gemm_trailing(&self, buf: &Self::Buffer, spec: TrailingGemm) -> Result<()> {
            self.mutate(*buf, |target| {
                for row in 0..spec.a_rows {
                    for col in 0..spec.b_cols {
                        let mut acc = 0.0f32;
                        for inner in 0..spec.a_cols {
                            acc += target[spec.a_offset + row * spec.a_stride + inner]
                                * target[spec.b_offset + inner * spec.b_stride + col];
                        }
                        target[spec.c_offset + row * spec.c_stride + col] -= acc;
                    }
                }
            });
            Ok(())
        }
    }

    impl BlockedCholeskyBackend for CpuBackend {
        fn syrk_trailing(&self, matrix: &Self::Buffer, spec: TrailingSyrk) -> Result<()> {
            self.mutate(*matrix, |target| {
                for row in 0..spec.trail_extent {
                    for col in 0..=row {
                        let mut acc = 0.0f32;
                        for inner in 0..spec.panel_cols {
                            acc += target[spec.panel_offset + row * spec.panel_stride + inner]
                                * target[spec.panel_offset + col * spec.panel_stride + inner];
                        }
                        target[spec.trail_offset + row * spec.matrix_stride + col] -= acc;
                    }
                }
            });
            Ok(())
        }
    }

    /// `A = [[4,2,1],[2,3,1],[1,1,3]]` has an exact LU: the leading column is
    /// strictly dominant, so no pivot swap occurs, and every multiplier and
    /// Schur entry is dyadic.
    const EXACT_LU: [f32; 9] = [4.0, 2.0, 1.0, 2.0, 3.0, 1.0, 1.0, 1.0, 3.0];

    /// `A = [[1,2,3],[2,5,8],[3,8,14]] = L·Lᵀ` for the integer
    /// `L = [[1,0,0],[2,1,0],[3,2,1]]`, so the Cholesky is exact.
    const EXACT_CHOLESKY: [f32; 9] = [1.0, 2.0, 3.0, 2.0, 5.0, 8.0, 3.0, 8.0, 14.0];

    /// Two panels at `block_size = 2`, so the trailing update runs once.
    const BLOCK: usize = 2;

    #[test]
    fn blocked_lu_reconstructs_its_input() {
        let backend = CpuBackend::default();
        let source = backend.put(EXACT_LU.to_vec());
        let work = backend.clone_device(&source, 9).expect("working copy");
        let result = blocked_lu(&backend, work, 3, BLOCK).expect("blocked LU");

        assert_eq!(result.perm, vec![0, 1, 2], "no pivot swap is required");
        assert_eq!(result.sign, 1, "the identity permutation is even");

        // P·A = L·U, reconstructed from the packed factors. Every value here is
        // dyadic, so the reconstruction is exact.
        for row in 0..3 {
            for col in 0..3 {
                let mut acc = 0.0f32;
                for inner in 0..3 {
                    let l = match inner.cmp(&row) {
                        std::cmp::Ordering::Less => result.host[row * 3 + inner],
                        std::cmp::Ordering::Equal => 1.0,
                        std::cmp::Ordering::Greater => 0.0,
                    };
                    let u = if inner <= col {
                        result.host[inner * 3 + col]
                    } else {
                        0.0
                    };
                    acc += l * u;
                }
                let expected = EXACT_LU[result.perm[row] * 3 + col];
                assert_eq!(acc, expected, "L·U at ({row},{col})");
            }
        }
    }

    #[test]
    fn blocked_cholesky_reconstructs_its_input() {
        let backend = CpuBackend::default();
        let source = backend.put(EXACT_CHOLESKY.to_vec());
        let work = backend.clone_device(&source, 9).expect("working copy");
        let result = blocked_cholesky(&backend, work, 3, BLOCK).expect("blocked Cholesky");

        assert_eq!(result.diagonal, vec![1.0, 1.0, 1.0], "factor diagonal");

        // A = L·Lᵀ from the lower factor. The strict upper triangle of the
        // device buffer is stale by contract, so the reconstruction reads the
        // lower triangle only.
        let lower = backend.read(result.lower);
        for row in 0..3 {
            for col in 0..=row {
                let mut acc = 0.0f32;
                for inner in 0..3 {
                    acc += element(&lower, row, inner) * element(&lower, col, inner);
                }
                assert_eq!(acc, EXACT_CHOLESKY[row * 3 + col], "L·Lᵀ at ({row},{col})");
            }
        }
    }

    /// Lower-triangular element access on the factored buffer.
    fn element(lower: &[f32], row: usize, col: usize) -> f32 {
        if col <= row {
            lower[row * 3 + col]
        } else {
            0.0
        }
    }

    impl BlockedQrBackend for CpuBackend {
        type Reflectors = ();

        fn alloc_reflectors(&self, _len: usize) -> Result<Self::Reflectors> {
            Ok(())
        }

        fn write_flat(&self, buf: &Self::Buffer, data: &[f32]) -> Result<()> {
            self.mutate(*buf, |target| target[..data.len()].copy_from_slice(data));
            Ok(())
        }

        fn householder_trailing(
            &self,
            vectors: &Self::Buffer,
            matrix: &Self::Buffer,
            _reflectors: &Self::Reflectors,
            spec: TrailingHh<'_>,
        ) -> Result<()> {
            // Copy the vectors out before borrowing the matrix mutably: the
            // store is one `RefCell`, so a live `Ref` and a `borrow_mut` on a
            // different entry would still panic.
            let vectors = self.read(*vectors).clone();
            self.mutate(*matrix, |matrix| apply_reflectors(matrix, &vectors, spec));
            Ok(())
        }

        fn finishes_tail_on_cpu(&self) -> bool {
            self.finish_tail_on_cpu
        }
    }

    /// Apply packed Householder reflectors to a matrix's trailing columns.
    ///
    /// Mirrors the device trailing kernels and
    /// [`apply_packed_qr_panel_left`](crate::domain::decomposition::apply_packed_qr_panel_left):
    /// reflector `j`'s vector is its head followed by the packed panel column,
    /// and it acts on rows `panel_start + j .. m` as `C ← (I − β v vᵀ) C`.
    fn apply_reflectors(matrix: &mut [f32], vectors: &[f32], spec: TrailingHh<'_>) {
        let n = spec.matrix_cols;
        let trail_start = spec.panel_start + spec.betas.len();
        for (j, beta) in spec.betas.iter().copied().enumerate() {
            let offset = spec.vector_offsets[j];
            let support = spec.panel_rows - j;
            let row0 = spec.panel_start + j;
            for col in trail_start..n {
                let mut dot = 0.0f32;
                for r in 0..support {
                    dot += vectors[offset + r] * matrix[(row0 + r) * n + col];
                }
                let scale = beta * dot;
                for r in 0..support {
                    matrix[(row0 + r) * n + col] -= scale * vectors[offset + r];
                }
            }
        }
    }

    /// A `4 × 3` fixture: two panels at `block_size = 2`, so the trailing
    /// Householder application runs for the first panel.
    const QR_INPUT: [f32; 12] = [
        1.0, 2.0, 3.0, //
        4.0, 5.0, 6.0, //
        7.0, 8.0, 10.0, //
        1.0, 1.0, 1.0,
    ];

    #[test]
    fn blocked_qr_reconstructs_its_input() {
        let backend = CpuBackend::default();
        let source = backend.put(QR_INPUT.to_vec());
        let work = backend.clone_device(&source, 12).expect("working copy");
        let result = blocked_qr(&backend, work, 4, 3, 2).expect("blocked QR");

        assert_eq!(result.heads.len(), 3, "one reflector per column");
        assert_eq!(result.betas.len(), 3, "one reflector per column");

        // The factored matrix's strict lower triangle is zero: the loop scatters
        // each panel with its sub-diagonal cleared.
        let factored = backend.read(result.r);
        for row in 0..4 {
            for col in 0..row.min(3) {
                assert_eq!(factored[row * 3 + col], 0.0, "R entry ({row},{col})");
            }
        }

        // Rebuilding A from R by re-applying the reflectors in reverse order
        // exercises the loop's trailing updates: every reflector that acts on
        // the trailing columns has to be there, on the right rows and columns,
        // with the right vector and β. R is the upper triangle of the packed
        // matrix, so the reconstruction reads its strictly-lower entries as the
        // packed reflector tails.
        let rebuilt =
            reconstruct_from_reflectors(&result.packed, &result.heads, &result.betas, 4, 3);
        // f32 Householder application carries O(n·ε·‖A‖) error; the largest
        // entry here is 10, so the bound is a few times 1e-6.
        for (index, (got, expected)) in rebuilt.iter().zip(QR_INPUT.iter()).enumerate() {
            assert!(
                (got - expected).abs() < 1e-4,
                "A[{index}] rebuilt as {got}, expected {expected}"
            );
        }
    }

    #[test]
    fn blocked_qr_host_tail_matches_the_device_tail() {
        // `n = 3` at `block_size = 2` leaves a one-column tail on the first
        // panel, which is exactly the width the host path intercepts. Both
        // policies must produce the same factors: the host path applies the
        // panel's reflectors with the core reference and factors the tail with
        // the panel routine, while the device path hands the same reflectors to
        // the trailing kernel.
        let device_tail = CpuBackend::default();
        let source = device_tail.put(QR_INPUT.to_vec());
        let work = device_tail.clone_device(&source, 12).expect("working copy");
        let device = blocked_qr(&device_tail, work, 4, 3, 2).expect("device tail");

        let host_tail = CpuBackend {
            finish_tail_on_cpu: true,
            ..CpuBackend::default()
        };
        let source = host_tail.put(QR_INPUT.to_vec());
        let work = host_tail.clone_device(&source, 12).expect("working copy");
        let host = blocked_qr(&host_tail, work, 4, 3, 2).expect("host tail");

        assert_eq!(host.heads, device.heads, "reflector heads");
        assert_eq!(host.betas, device.betas, "reflector coefficients");
        assert_eq!(host.packed, device.packed, "packed factors");
        assert_eq!(
            host_tail.read(host.r).clone(),
            device_tail.read(device.r).clone(),
            "factored matrices"
        );
    }

    /// Re-apply the packed reflectors to **R** in reverse order to rebuild
    /// **A**, reading `packed`'s strictly-lower entries as the reflector tails.
    fn reconstruct_from_reflectors(
        packed: &[f32],
        heads: &[f32],
        betas: &[f32],
        m: usize,
        n: usize,
    ) -> Vec<f32> {
        let mut rebuilt = vec![0.0f32; m * n];
        for (row, line) in rebuilt.chunks_exact_mut(n).enumerate().take(n) {
            line[row..].copy_from_slice(&packed[row * n + row..(row + 1) * n]);
        }
        for j in (0..n).rev() {
            for col in 0..n {
                let mut dot = heads[j] * rebuilt[j * n + col];
                for r in (j + 1)..m {
                    dot += packed[r * n + j] * rebuilt[r * n + col];
                }
                let scale = betas[j] * dot;
                rebuilt[j * n + col] -= scale * heads[j];
                for r in (j + 1)..m {
                    rebuilt[r * n + col] -= scale * packed[r * n + j];
                }
            }
        }
        rebuilt
    }
}
