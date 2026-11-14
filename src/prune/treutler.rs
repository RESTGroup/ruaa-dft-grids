//! PySCF `gen_grid.treutler_prune`: index-fraction pruning independent of
//! the element.

/// PySCF `gen_grid.treutler_prune`.
pub fn treutler_prune(rads: &[f64], n_ang: usize) -> Vec<usize> {
    let nr = rads.len();
    (0..nr)
        .map(|i| {
            if i < nr / 3 {
                14
            } else if i < nr / 2 {
                50
            } else {
                n_ang
            }
        })
        .collect()
}
