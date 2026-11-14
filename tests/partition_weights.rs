//! Reusable partition weights at caller-supplied points.

use ruaa_dft_grids::Error;
use ruaa_dft_grids::partition::{PartitionConfig, PartitionScheme, RadiiConfig};

fn water() -> (Vec<u32>, Vec<[f64; 3]>) {
    let atoms: Vec<(u32, [f64; 3])> = vec![
        (8, [0.0, 0.0, 0.0]),
        (1, [0.0, 1.43, 1.11]),
        (1, [0.0, -1.43, 1.11]),
    ];
    (
        atoms.iter().map(|a| a.0).collect(),
        atoms.iter().map(|a| a.1).collect(),
    )
}

fn points() -> Vec<[f64; 3]> {
    vec![
        [0.0, 0.0, 0.0],
        [0.0, 0.5, 0.25],
        [0.0, -0.75, 0.5],
        [1.2, -0.4, 0.9],
        [-2.0, 1.5, -0.3],
        [3.1, 2.2, 1.7],
    ]
}

fn check_config(config: PartitionConfig) {
    let (charges, centers) = water();
    let points = points();
    let evaluator = config.weights_evaluator(&centers, &charges).unwrap();
    assert_eq!(evaluator.atom_count(), centers.len());
    assert_eq!(evaluator.config(), &config);

    let serial = evaluator.weights(&points).unwrap();
    assert_eq!(serial.len(), points.len() * centers.len());
    for (index, point) in points.iter().enumerate() {
        let want = config.weights(*point, &centers, &charges).unwrap();
        let row = &serial[index * centers.len()..(index + 1) * centers.len()];
        for (a, b) in row.iter().zip(&want) {
            assert_eq!(a.to_bits(), b.to_bits(), "point {index}");
        }
    }

    let mut filled = vec![0.0; points.len() * centers.len()];
    evaluator.fill_points(&points, &mut filled).unwrap();
    assert_eq!(filled, serial);

    let mut parallel = vec![0.0; points.len() * centers.len()];
    evaluator.fill_points_par(&points, &mut parallel).unwrap();
    assert_eq!(parallel, serial);

    for (index, point) in points.iter().enumerate() {
        let at = evaluator.weights_at(*point).unwrap();
        assert_eq!(
            at,
            serial[index * centers.len()..(index + 1) * centers.len()]
        );
    }

    let mut one = vec![0.0; centers.len()];
    evaluator.fill_point(points[1], &mut one).unwrap();
    assert_eq!(one, serial[centers.len()..2 * centers.len()]);

    let empty = evaluator.weights(&[]).unwrap();
    assert!(empty.is_empty());
}

#[test]
fn classic_becke_matches_the_reference_path() {
    check_config(PartitionConfig::new(
        PartitionScheme::Becke,
        RadiiConfig::pyscf(),
    ));
}

#[test]
fn stratmann_matches_the_reference_path() {
    check_config(PartitionConfig::new(
        PartitionScheme::Stratmann,
        RadiiConfig::pyscf(),
    ));
}

#[test]
fn lko_matches_the_reference_path() {
    check_config(PartitionConfig::new(
        PartitionScheme::Lko,
        RadiiConfig::pyscf(),
    ));
}

#[test]
fn mbecke_matches_the_reference_path() {
    check_config(PartitionConfig::new(
        PartitionScheme::MBecke,
        RadiiConfig::pyscf(),
    ));
}

#[test]
fn length_and_geometry_errors_are_reported() {
    let (charges, centers) = water();
    let config = PartitionConfig::new(PartitionScheme::Becke, RadiiConfig::pyscf());
    let evaluator = config.weights_evaluator(&centers, &charges).unwrap();

    let mut too_short = vec![0.0; 2];
    assert!(matches!(
        evaluator.fill_points(&points(), &mut too_short),
        Err(Error::LengthMismatch { .. })
    ));
    assert!(matches!(
        evaluator.fill_points_par(&points(), &mut too_short),
        Err(Error::LengthMismatch { .. })
    ));
    assert!(matches!(
        evaluator.fill_point([0.0; 3], &mut [0.0; 2]),
        Err(Error::LengthMismatch { .. })
    ));

    let err = config
        .weights_evaluator(&centers[..2], &charges)
        .unwrap_err();
    assert!(matches!(err, Error::LengthMismatch { .. }));
}
