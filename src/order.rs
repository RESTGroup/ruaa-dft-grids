//! Point ordering and permutation utilities for molecular grids.
//!
//! A [`GridOrder`] is a bijection between two index spaces: the **sorted
//! slot** (`0..n`, the order a kernel iterates) and the **source index** (the
//! order in which [`crate::grid::GridPlan`] or
//! [`crate::moleculegrid::MolecularGrid`] produced the points). The
//! permutation is stored once and the inverse is kept alongside it, so a
//! consumer can both iterate in the new order and map a point back to its
//! original index.
//!
//! Ordering is a consumer policy, not part of the grid definition:
//!
//! * a spatial order is geometry-only and needs no basis set, which is why
//!   [`GridOrder::boxes`] and [`GridOrder::morton`] take plain coordinates;
//! * a screening table needs the basis set; when the consumer computes the
//!   significance signature itself, [`GridOrder::from_keys`] or
//!   [`GridOrder::from_permutation`] accept that order directly.
//!
//! The crate never applies an order implicitly: the value evaluator walks
//! atom-contiguous runs, so an order is a **whole-grid** operation. Reorder a
//! fully materialised [`crate::moleculegrid::MolecularGrid`] (see
//! [`crate::moleculegrid::MolecularGrid::reorder`]), or evaluate first and
//! permute the resulting per-point arrays. Applying an order to a
//! [`crate::grid::GridPlan`] is deliberately not offered: its point table is
//! the evaluator's index space.

use std::ops::Range;

use crate::error::{Error, Result};

/// A permutation between a grid's current point order and a sorted order.
///
/// `perm()[slot]` is the source index stored in sorted slot `slot`;
/// `inverse()[source]` is the slot that source index ends up in. Every
/// constructor returns a true bijection.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GridOrder {
    perm: Vec<u32>,
    inverse: Vec<u32>,
}

impl GridOrder {
    /// The identity order over `len` points.
    ///
    /// # Errors
    /// [`Error::TooManyPoints`] when `len` exceeds the `u32` index space.
    pub fn identity(len: usize) -> Result<Self> {
        check_len(len)?;
        Ok(Self {
            perm: (0..len as u32).collect(),
            inverse: (0..len as u32).collect(),
        })
    }

    /// Wrap a permutation computed outside the crate.
    ///
    /// `perm[slot]` must be the source index for sorted slot `slot`. The
    /// function validates that every source index in `0..perm.len()` appears
    /// exactly once.
    ///
    /// # Errors
    /// [`Error::PermutationOutOfRange`] for a source index outside `0..len`
    /// and [`Error::PermutationDuplicate`] for a repeated source index.
    pub fn from_permutation(perm: &[u32]) -> Result<Self> {
        let len = perm.len();
        check_len(len)?;
        let mut seen = vec![false; len];
        let mut inverse = vec![0u32; len];
        for (slot, &source) in perm.iter().enumerate() {
            let source_usize = source as usize;
            if source_usize >= len {
                return Err(Error::PermutationOutOfRange {
                    slot,
                    value: source,
                    len,
                });
            }
            if seen[source_usize] {
                return Err(Error::PermutationDuplicate { value: source });
            }
            seen[source_usize] = true;
            inverse[source_usize] = slot as u32;
        }
        Ok(Self {
            perm: perm.to_vec(),
            inverse,
        })
    }

    /// Sort by a caller-supplied `u64` key, stably.
    ///
    /// Equal keys keep their original relative order, so the result is
    /// deterministic. A consumer can build the key from anything it likes
    /// (a significance-signature hash, a block id, a density bucket, ...);
    /// this is the hook for screening-table orders that need the basis set.
    ///
    /// # Errors
    /// [`Error::TooManyPoints`] when the key count exceeds the `u32` index
    /// space.
    pub fn from_keys<I>(keys: I) -> Result<Self>
    where
        I: IntoIterator<Item = u64>,
    {
        let keys: Vec<u64> = keys.into_iter().collect();
        let len = keys.len();
        check_len(len)?;
        let mut perm: Vec<u32> = (0..len as u32).collect();
        // `sort_by_key` is stable, so equal keys keep their original order.
        perm.sort_by_key(|&source| keys[source as usize]);
        let mut inverse = vec![0u32; len];
        for (slot, &source) in perm.iter().enumerate() {
            inverse[source as usize] = slot as u32;
        }
        Ok(Self { perm, inverse })
    }

    /// Sort points into uniform cubic boxes of edge `box_size` (Bohr).
    ///
    /// # Arguments
    ///
    /// * `coords` — `[N, 3]` row-major point coordinates in Bohr.
    /// * `box_size` — box edge length in Bohr.
    ///
    /// The lattice is anchored at the origin, and the boxes are visited in
    /// lexicographic `(ix, iy, iz)` order with the original index as the
    /// final tie-break, so the result is deterministic. This is a
    /// PySCF-style geometry-only grouping with `GROUP_BOX_SIZE = 1.2 Bohr`.
    /// PySCF's `arg_group_grids` anchors the lattice at the molecule bounding
    /// box expanded by 4.2 Bohr and clamps boundary boxes, so the two
    /// permutations are not identical.
    ///
    /// # Errors
    /// [`Error::InvalidArgument`] if `box_size` is not finite and positive or
    /// a coordinate is not finite, and [`Error::TooManyPoints`] when there
    /// are too many points.
    ///
    /// ```
    /// use ruaa_dft_grids::order::GridOrder;
    ///
    /// # fn main() -> ruaa_dft_grids::Result<()> {
    /// let coords = [[0.0, 0.0, 0.0], [2.0, 0.0, 0.0], [0.1, 0.1, 0.1]];
    /// let order = GridOrder::boxes(&coords, 1.2)?;
    /// assert_eq!(order.len(), 3);
    /// // The two points in the origin box come first; the far point is last.
    /// assert_eq!(order.perm(), &[0, 2, 1]);
    /// assert_eq!(order.inverse(), &[0, 2, 1]);
    /// # Ok(())
    /// # }
    /// ```
    pub fn boxes(coords: &[[f64; 3]], box_size: f64) -> Result<Self> {
        if !(box_size.is_finite() && box_size > 0.0) {
            return Err(Error::InvalidArgument {
                argument: "box size",
            });
        }
        check_finite_coords(coords)?;
        check_len(coords.len())?;
        let mut keyed: Vec<(i64, i64, i64, u32)> = coords
            .iter()
            .enumerate()
            .map(|(i, p)| {
                (
                    (p[0] / box_size).floor() as i64,
                    (p[1] / box_size).floor() as i64,
                    (p[2] / box_size).floor() as i64,
                    i as u32,
                )
            })
            .collect();
        keyed.sort_by_key(|&(x, y, z, source)| (x, y, z, source));
        let perm: Vec<u32> = keyed.into_iter().map(|(_, _, _, source)| source).collect();
        check_len(perm.len())?;
        Ok(Self::from_sorted(perm))
    }

    /// Sort points along a Morton (Z-order) space-filling curve.
    ///
    /// # Arguments
    ///
    /// * `coords` — `[N, 3]` row-major point coordinates in Bohr.
    /// * `cell` — Morton cell size in Bohr.
    ///
    /// Coordinates are quantised to a `cell`-sized lattice anchored at the
    /// origin, clamped to 21 bits per axis ($\pm2^20$ cells, about $\pm1.2e6$
    /// Bohr at a 1.2 Bohr cell), then bit-interleaved. Morton order keeps
    /// nearby points close in memory while remaining a single `O(n log n)`
    /// sort; the cell size is the only knob. See [`Self::boxes`] for the
    /// simpler lexicographic variant.
    ///
    /// # Errors
    /// [`Error::InvalidArgument`] if `cell` is not finite and positive or a
    /// coordinate is not finite, and [`Error::TooManyPoints`] when there are
    /// too many points.
    pub fn morton(coords: &[[f64; 3]], cell: f64) -> Result<Self> {
        if !(cell.is_finite() && cell > 0.0) {
            return Err(Error::InvalidArgument {
                argument: "Morton cell size",
            });
        }
        check_finite_coords(coords)?;
        check_len(coords.len())?;
        let mut keyed: Vec<(u64, u32)> = coords
            .iter()
            .enumerate()
            .map(|(i, p)| {
                let x = quantize(p[0], cell);
                let y = quantize(p[1], cell);
                let z = quantize(p[2], cell);
                (morton_code(x, y, z), i as u32)
            })
            .collect();
        keyed.sort_by_key(|&(code, source)| (code, source));
        let perm: Vec<u32> = keyed.into_iter().map(|(_, source)| source).collect();
        check_len(perm.len())?;
        Ok(Self::from_sorted(perm))
    }

    /// Number of points covered by this order.
    #[must_use]
    pub fn len(&self) -> usize {
        self.perm.len()
    }

    /// Whether the order covers no points.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.perm.is_empty()
    }

    /// Source index per sorted slot: `perm()[slot]` is the original index.
    #[must_use]
    pub fn perm(&self) -> &[u32] {
        &self.perm
    }

    /// Slot per source index: `inverse()[source]` is the sorted slot.
    #[must_use]
    pub fn inverse(&self) -> &[u32] {
        &self.inverse
    }

    /// Contiguous ranges of at most `block_size` **sorted slots**.
    ///
    /// The ranges tile `0..len`; the last one may be shorter.
    /// `block_size == 0` is treated as 1. Block size is a consumer policy
    /// (kernels use 56/64 points), so it is passed here rather than stored.
    pub fn blocks(&self, block_size: usize) -> impl Iterator<Item = Range<usize>> + '_ {
        let size = block_size.max(1);
        let len = self.len();
        (0..len)
            .step_by(size)
            .map(move |start| start..(start + size).min(len))
    }

    /// Apply the order to `src`, returning `out[slot] = src[perm[slot]]`.
    ///
    /// # Errors
    /// [`Error::LengthMismatch`] if `src.len()` differs from the order
    /// length.
    pub fn permute<T: Copy>(&self, src: &[T]) -> Result<Vec<T>> {
        if src.len() != self.len() {
            return Err(Error::LengthMismatch {
                expected: self.len(),
                got: src.len(),
            });
        }
        Ok(self
            .perm
            .iter()
            .map(|&source| src[source as usize])
            .collect())
    }

    /// Apply the order into an existing buffer: `out[slot] = src[perm[slot]]`.
    ///
    /// # Errors
    /// [`Error::LengthMismatch`] if `src.len()` or `out.len()` differs from
    /// the order length.
    pub fn permute_into<T: Copy>(&self, src: &[T], out: &mut [T]) -> Result<()> {
        if src.len() != self.len() {
            return Err(Error::LengthMismatch {
                expected: self.len(),
                got: src.len(),
            });
        }
        if out.len() != self.len() {
            return Err(Error::LengthMismatch {
                expected: self.len(),
                got: out.len(),
            });
        }
        for (slot, &source) in self.perm.iter().enumerate() {
            out[slot] = src[source as usize];
        }
        Ok(())
    }

    /// Permute `data` in place using a `visited` bitmask; no second copy of
    /// the data is allocated.
    ///
    /// The permutation is applied by following cycles and rotating each one,
    /// so exactly `len - cycles` swaps are performed.
    ///
    /// # Errors
    /// [`Error::LengthMismatch`] if `data.len()` differs from the order
    /// length.
    pub fn permute_in_place<T>(&self, data: &mut [T]) -> Result<()> {
        if data.len() != self.len() {
            return Err(Error::LengthMismatch {
                expected: self.len(),
                got: data.len(),
            });
        }
        let mut visited = vec![false; data.len()];
        for start in 0..data.len() {
            if visited[start] {
                continue;
            }
            let mut slot = start;
            loop {
                visited[slot] = true;
                let next = self.perm[slot] as usize;
                if visited[next] {
                    break;
                }
                data.swap(slot, next);
                slot = next;
            }
        }
        Ok(())
    }

    fn from_sorted(perm: Vec<u32>) -> Self {
        let len = perm.len();
        assert!(
            len <= u32::MAX as usize,
            "grid order supports at most 2^32 - 1 points"
        );
        let mut inverse = vec![0u32; len];
        for (slot, &source) in perm.iter().enumerate() {
            inverse[source as usize] = slot as u32;
        }
        Self { perm, inverse }
    }
}

/// Validate that `len` fits the `u32` permutation representation.
fn check_finite_coords(coords: &[[f64; 3]]) -> Result<()> {
    if coords.iter().flatten().any(|value| !value.is_finite()) {
        return Err(Error::InvalidArgument {
            argument: "point coordinates",
        });
    }
    Ok(())
}

fn check_len(len: usize) -> Result<()> {
    if len > u32::MAX as usize {
        return Err(Error::TooManyPoints {
            points: len,
            max: u32::MAX as usize,
        });
    }
    Ok(())
}

/// 21 bits per axis: the Morton code fits in 63 bits.
const MORTON_BITS: i64 = 21;
const MORTON_MAX: i64 = (1 << MORTON_BITS) - 1;
const MORTON_BIAS: i64 = 1 << (MORTON_BITS - 1);

fn quantize(v: f64, cell: f64) -> u64 {
    // `as i64` saturates, so NaN and out-of-range coordinates clamp instead
    // of wrapping.
    let q = (v / cell).floor() as i64;
    q.saturating_add(MORTON_BIAS).clamp(0, MORTON_MAX) as u64
}

/// Interleave the low 21 bits of `x` into every third bit.
fn spread_bits(x: u64) -> u64 {
    let mut x = x & 0x1f_ffff;
    x = (x | (x << 32)) & 0x001f_0000_0000_ffff;
    x = (x | (x << 16)) & 0x001f_0000_ff00_00ff;
    x = (x | (x << 8)) & 0x100f_00f0_0f00_f00f;
    x = (x | (x << 4)) & 0x10c3_0c30_c30c_30c3;
    x = (x | (x << 2)) & 0x1249_2492_4924_9249;
    x
}

fn morton_code(x: u64, y: u64, z: u64) -> u64 {
    spread_bits(x) | (spread_bits(y) << 1) | (spread_bits(z) << 2)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn assert_bijection(order: &GridOrder) {
        let len = order.len();
        let mut seen = vec![false; len];
        for (slot, &source) in order.perm().iter().enumerate() {
            let source = source as usize;
            assert!(source < len);
            assert!(!seen[source], "source {source} repeated");
            seen[source] = true;
            assert_eq!(order.inverse()[source] as usize, slot);
        }
        assert!(seen.iter().all(|&s| s));
    }

    #[test]
    fn identity_round_trip() {
        let order = GridOrder::identity(5).unwrap();
        assert_eq!(order.perm(), [0, 1, 2, 3, 4]);
        assert_eq!(order.inverse(), [0, 1, 2, 3, 4]);
        assert_eq!(
            order.permute(&[10, 11, 12, 13, 14]).unwrap(),
            vec![10, 11, 12, 13, 14]
        );
        assert_bijection(&order);
        assert!(GridOrder::identity(0).unwrap().is_empty());
    }

    #[test]
    fn from_permutation_validates() {
        let order = GridOrder::from_permutation(&[2, 0, 1]).unwrap();
        assert_eq!(order.perm(), [2, 0, 1]);
        assert_eq!(order.inverse(), [1, 2, 0]);

        assert_eq!(
            GridOrder::from_permutation(&[0, 0]).unwrap_err(),
            Error::PermutationDuplicate { value: 0 }
        );
        assert_eq!(
            GridOrder::from_permutation(&[0, 2]).unwrap_err(),
            Error::PermutationOutOfRange {
                slot: 1,
                value: 2,
                len: 2
            }
        );
    }

    #[test]
    fn keys_sort_stably() {
        let order = GridOrder::from_keys([5u64, 1, 5, 0, 1]).unwrap();
        // keys: 5,1,5,0,1 -> stable order: index 3 (0), 1 (1), 4 (1), 0 (5), 2 (5)
        assert_eq!(order.perm(), [3, 1, 4, 0, 2]);
        assert_bijection(&order);
        assert!(GridOrder::from_keys(std::iter::empty()).unwrap().is_empty());
    }

    #[test]
    fn boxes_are_lexicographic_and_deterministic() {
        let coords = [
            [0.1, 0.1, 0.1],
            [0.2, 0.1, 0.1],
            [5.1, 0.1, 0.1],
            [0.1, 5.1, 0.1],
            [5.1, 5.1, 5.1],
        ];
        let order = GridOrder::boxes(&coords, 1.0).unwrap();
        assert_eq!(order.perm(), [0, 1, 3, 2, 4]);
        assert_bijection(&order);
        assert_eq!(GridOrder::boxes(&coords, 1.0).unwrap(), order);
    }

    #[test]
    fn morton_is_a_bijection_and_deterministic() {
        let coords = [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [-1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [1.0, 1.0, 1.0],
        ];
        let order = GridOrder::morton(&coords, 1.0).unwrap();
        assert_bijection(&order);
        assert_eq!(GridOrder::morton(&coords, 1.0).unwrap(), order);
        // Coincident points keep their original relative order.
        let same = [[1.0, 2.0, 3.0]; 4];
        assert_eq!(GridOrder::morton(&same, 1.0).unwrap().perm(), [0, 1, 2, 3]);
    }

    #[test]
    fn blocks_tile_all_slots() {
        let order = GridOrder::identity(10).unwrap();
        let blocks: Vec<_> = order.blocks(4).collect();
        assert_eq!(blocks, [0..4, 4..8, 8..10]);
        assert_eq!(order.blocks(0).count(), 10);
        assert_eq!(GridOrder::identity(0).unwrap().blocks(4).count(), 0);
    }

    #[test]
    fn permute_variants_agree() {
        let order = GridOrder::from_permutation(&[3, 0, 4, 1, 2]).unwrap();
        let src: Vec<i32> = (0..5).map(|i| 10 + i).collect();
        let expected = vec![13, 10, 14, 11, 12];
        assert_eq!(order.permute(&src).unwrap(), expected);

        let mut out = vec![0; 5];
        order.permute_into(&src, &mut out).unwrap();
        assert_eq!(out, expected);

        let mut in_place = src.clone();
        order.permute_in_place(&mut in_place).unwrap();
        assert_eq!(in_place, expected);
    }

    #[test]
    fn permute_rejects_wrong_length() {
        let order = GridOrder::identity(3).unwrap();
        assert_eq!(
            order.permute(&[1, 2]),
            Err(Error::LengthMismatch {
                expected: 3,
                got: 2
            })
        );
        let mut out = [0; 2];
        assert_eq!(
            order.permute_into(&[1, 2, 3], &mut out),
            Err(Error::LengthMismatch {
                expected: 3,
                got: 2
            })
        );
    }

    /// Invalid geometry-only constructors report a typed argument error.
    #[test]
    fn invalid_box_and_cell_are_errors() {
        let coords = [[0.0, 0.0, 0.0]];
        assert_eq!(
            GridOrder::boxes(&coords, 0.0),
            Err(Error::InvalidArgument {
                argument: "box size"
            })
        );
        assert_eq!(
            GridOrder::morton(&coords, f64::NAN),
            Err(Error::InvalidArgument {
                argument: "Morton cell size"
            })
        );
    }

    #[test]
    fn non_finite_coordinates_are_rejected() {
        for bad in [
            [f64::NAN, 0.0, 0.0],
            [f64::INFINITY, 0.0, 0.0],
            [0.0, f64::NEG_INFINITY, 0.0],
        ] {
            let coords = [bad];
            assert_eq!(
                GridOrder::boxes(&coords, 1.0),
                Err(Error::InvalidArgument {
                    argument: "point coordinates"
                })
            );
            assert_eq!(
                GridOrder::morton(&coords, 1.0),
                Err(Error::InvalidArgument {
                    argument: "point coordinates"
                })
            );
        }
        let huge = [[f64::MAX, -f64::MAX, f64::MAX]];
        assert!(GridOrder::morton(&huge, 1.0).is_ok());
    }
}
