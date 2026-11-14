//! End-to-end checks for the crate-wide [`ruaa_dft_grids::Error`].
//!
//! Every invalid input on the grid-construction path comes back as a typed
//! error, and the error must be the one a caller can match on (not a wrapped
//! string).

use ruaa_dft_grids::grid::{GridPlan, PyscfGridSpec, PyscfGridSpecBuilder};
use ruaa_dft_grids::moleculegrid::build_molecular_grid;
use ruaa_dft_grids::preset::{ElementPreset, GridPreset};
use ruaa_dft_grids::prune::Prune;
use ruaa_dft_grids::radial::Radial;
use ruaa_dft_grids::{Error, Result};

fn h2() -> [(u32, [f64; 3]); 2] {
    [(1, [-0.7, 0.0, 0.0]), (1, [0.7, 0.0, 0.0])]
}

#[test]
fn plan_build_rejects_unknown_elements() -> Result<()> {
    assert_eq!(
        GridPlan::build(&[(0, [0.0; 3])], PyscfGridSpec::default()).unwrap_err(),
        Error::UnsupportedElement { z: 0 }
    );
    assert_eq!(
        GridPlan::build(&[(119, [0.0; 3])], PyscfGridSpec::default()).unwrap_err(),
        Error::UnsupportedElement { z: 119 }
    );
    Ok(())
}

/// A data-driven preset without an entry for the element names the preset
/// table instead of panicking.
#[test]
fn preset_reports_a_missing_element() -> Result<()> {
    let preset =
        GridPreset::new(Radial::TreutlerAhlrichs).with_default(ElementPreset::fixed(20, 1.0, 50));
    assert_eq!(
        GridPlan::build(&[(119, [0.0; 3])], preset).unwrap_err(),
        Error::UnsupportedElement { z: 119 }
    );
    Ok(())
}

#[test]
fn pyscf_rejects_unknown_levels() -> Result<()> {
    assert_eq!(
        PyscfGridSpecBuilder::default()
            .level(11)
            .build()
            .unwrap_err(),
        Error::UnsupportedLevel { level: 11 }
    );
    Ok(())
}

/// SG1 pruning only covers H..Ar; a heavier element is a typed error.
#[test]
fn pyscf_sg1_prune_domain_is_checked() -> Result<()> {
    let spec = PyscfGridSpecBuilder::default()
        .prune(Prune::Sg1)
        .build()
        .unwrap();
    assert_eq!(
        GridPlan::build(&[(26, [0.0; 3])], spec).unwrap_err(),
        Error::MissingElementData {
            z: 26,
            table: "PySCF SG1",
        }
    );
    Ok(())
}

/// A per-element `(n_rad, n_ang)` override must name a Lebedev point count or
/// a PySCF `LEBEDEV_ORDER` key; a value that is neither is a typed error.
#[test]
fn pyscf_rejects_unsupported_lebedev_counts() -> Result<()> {
    let error = PyscfGridSpecBuilder::default()
        .atom_grid(std::collections::BTreeMap::from([(1, (20, 72))]))
        .build()
        .unwrap_err();
    assert_eq!(error, Error::UnsupportedLebedevGrid { points: 72 });
    Ok(())
}

/// The full-materialisation entry point propagates the same error.
#[test]
fn molecular_grid_propagates_errors() -> Result<()> {
    let mut spec = PyscfGridSpec::default();
    spec.level = 10;
    assert_eq!(
        build_molecular_grid(&h2(), spec).unwrap_err(),
        Error::UnsupportedLevel { level: 10 }
    );
    Ok(())
}

/// Errors compose with `?` inside caller code and expose a stable message.
#[test]
fn errors_propagate_with_question_mark() -> Result<()> {
    fn build_water() -> Result<usize> {
        build_molecular_grid(
            &[(8, [0.0, 0.0, 0.0]), (1, [0.0, 1.43, 1.11])],
            PyscfGridSpec::from_level(0)?,
        )
        .map(|grid| grid.len())
    }

    let n = build_water()?;
    assert!(n > 1000);

    let err = Error::MissingElementData {
        z: 60,
        table: "grid preset",
    };
    assert_eq!(err.to_string(), "no grid preset data for Z = 60");
    let boxed: Box<dyn std::error::Error> = Box::new(err);
    assert!(boxed.source().is_none());
    Ok(())
}
