//! PySCF atomic-grid builder.

use crate::angular::angular_grid;
use crate::atomgrid::AtomicGrid;
use crate::error::Result;
use crate::grid::PyscfGridSpec;
use crate::pyscf::presets::level_defaults;

/// Build one PySCF atomic grid for `z`.
///
/// `spec.atom_grid` overrides `(n_rad, n_ang)`; otherwise
/// [`level_defaults`] is used. Weights are PySCF's unpartitioned
/// `quadrature_weights`: `w_rad = 4*pi*r^2*dr` times a sum-to-1 angular
/// weight, so they match PySCF's `gen_atomic_grids` volume elements.
///
/// # Errors
/// Propagates the PySCF level/element error from [`level_defaults`], the
/// radial $\xi$ table error from [`crate::radial::Radial::grid`], and the
/// pruning error from [`crate::prune::Prune::sizes`].
pub fn build_pyscf_atom_grid(z: u32, spec: &PyscfGridSpec) -> Result<AtomicGrid> {
    spec.validate()?;
    if !(1..=118).contains(&z) {
        return Err(crate::Error::UnsupportedElement { z });
    }
    let (n_rad, n_ang) = match spec.atom_grid.get(&z).copied() {
        Some(override_pair) => override_pair,
        None => level_defaults(z, spec.level)?,
    };
    let n_ang = crate::pyscf::presets::angular_count(n_ang)?;
    let (rads, wrads) = spec.radial.grid_for_element(n_rad, z)?;
    let ratios: Vec<f64> = if spec.prune.needs_ratios() {
        let xi = spec.radial.xi_for_element(z)?;
        crate::radial::ratios(spec.radial, &rads, xi)
    } else {
        Vec::new()
    };
    let context = crate::prune::PruneContext::for_pyscf(z, &rads, &ratios, n_ang);
    let angs = crate::prune::sizes_checked(&spec.prune, &context, n_rad)?.sizes;
    crate::atomgrid::assemble_atomic_grid(z, &rads, &wrads, |shell| angular_grid(angs[shell]))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::grid::PyscfGridSpecBuilder;
    use crate::pyscf::presets::level_defaults as lvl;

    #[test]
    fn pyscf_atom_grid_has_metadata() {
        let preset = PyscfGridSpec::default();
        let g = build_pyscf_atom_grid(6, &preset).unwrap();
        assert_eq!(g.nr(), 75);
        assert_eq!(g.shell_angular.len(), 75);
        assert_eq!(g.points.len(), g.quad_weights.len());
        let (n_rad, _) = lvl(6, 3).unwrap();
        assert_eq!(g.nr(), n_rad);
        assert!(g.total_quad_weight() > 0.0);
    }

    #[test]
    fn per_z_override_is_used() {
        let preset = PyscfGridSpecBuilder::default()
            .atom_grid(std::collections::BTreeMap::from([(6, (20, 50))]))
            .build()
            .unwrap();
        let g = build_pyscf_atom_grid(6, &preset).unwrap();
        assert_eq!(g.nr(), 20);
        assert!(g.points.len() >= g.nr());
    }

    #[test]
    fn per_z_override_accepts_order_alias_and_trivial_rule() {
        use crate::prune::Prune;

        let preset = PyscfGridSpecBuilder::default()
            .prune(Prune::None)
            .atom_grid(std::collections::BTreeMap::from([(6, (20, 11))]))
            .build()
            .unwrap();
        let g = build_pyscf_atom_grid(6, &preset).unwrap();
        assert_eq!(g.nr(), 20);
        assert!(g.shell_angular.iter().all(|&n| n == 50));

        let preset = PyscfGridSpecBuilder::default()
            .prune(Prune::None)
            .atom_grid(std::collections::BTreeMap::from([(6, (10, 1))]))
            .build()
            .unwrap();
        let g = build_pyscf_atom_grid(6, &preset).unwrap();
        assert_eq!(g.nr(), 10);
        assert_eq!(g.points.len(), 10);
        assert!(g.shell_angular.iter().all(|&n| n == 1));
    }

    #[test]
    fn ladder_prune_is_available_to_pyscf() {
        use crate::prune::{LadderPrune, Prune};

        let bounds = [
            [0.25, 0.5, 1.0, 4.5],
            [0.1667, 0.5, 0.9, 3.5],
            [0.1, 0.4, 0.8, 2.5],
            [0.1, 0.4, 0.8, 2.5],
        ];
        let preset = PyscfGridSpecBuilder::default()
            .prune(Prune::Ladder(LadderPrune::new(
                [6, 26, 50, 110, 50],
                110,
                bounds,
            )))
            .build()
            .unwrap();
        let g = build_pyscf_atom_grid(6, &preset).unwrap();
        assert_eq!(g.nr(), 75);
        assert!(
            g.shell_angular
                .iter()
                .all(|&n| [6, 26, 50, 110].contains(&n))
        );
    }
}
