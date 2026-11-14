//! Weights-only throughput benchmark for the classic Becke partition.
//!
//! ```text
//! cargo run --release --example bench_weights -- \
//!     <molecule.xyz> <level> [batch] [becke|stratmann|gaussian_ss|mbecke|lko]
//! ```
//!
//! The XYZ file is in Angstrom; the grid is PySCF level `level` with the
//! selected partition and the PySCF Bragg/Treutler radii, so the same source
//! can be built against another kernel for an A/B run.
//!
//! Only partition evaluation is timed: grid construction and the plan build
//! are excluded, and the quadrature/partition arrays are reused across
//! batches.

use std::time::Instant;

use ruaa_dft_grids::grid::{BatchBuffer, GridPlan, GridSpec, PyscfGridSpecBuilder};
use ruaa_dft_grids::partition::{
    PartitionConfig, PartitionEvaluator, PartitionScheme, RadiiConfig, Workspace,
};

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    assert!(
        args.len() >= 2,
        "usage: bench_weights <molecule.xyz> <level> [batch]"
    );
    let path = &args[0];
    let level: u8 = args[1].parse().unwrap();
    let batch: usize = args.get(2).map_or(4096, |s| s.parse().unwrap());
    let scheme = match args.get(3).map(String::as_str) {
        Some("stratmann") => PartitionScheme::Stratmann,
        Some("gaussian_ss") => PartitionScheme::GaussianSs,
        Some("mbecke") => PartitionScheme::MBecke,
        Some("lko") => PartitionScheme::Lko,
        _ => PartitionScheme::Becke,
    };
    let atoms = read_xyz(path);
    let centers: Vec<[f64; 3]> = atoms.iter().map(|a| a.1).collect();
    let charges: Vec<u32> = atoms.iter().map(|a| a.0).collect();

    let spec: GridSpec = PyscfGridSpecBuilder::default()
        .level(level)
        .partition(PartitionConfig::new(scheme, RadiiConfig::pyscf()))
        .build()
        .unwrap()
        .into();
    let plan = GridPlan::build(&atoms, &spec).unwrap();

    let evaluator =
        PartitionEvaluator::from_parts(&centers, &charges, spec.partition().clone()).unwrap();
    let mut ws = Workspace::new();
    let mut out = BatchBuffer::new();
    let ranges: Vec<_> = plan.batches(batch).collect();
    let mut checksum = 0.0;

    // warm-up
    if let Some(r) = ranges.first() {
        evaluator
            .fill_plan_par(&plan, r.clone(), &mut ws, &mut out)
            .unwrap();
        checksum += out.weights().iter().sum::<f64>();
    }
    let t = Instant::now();
    for r in &ranges {
        evaluator
            .fill_plan_par(&plan, r.clone(), &mut ws, &mut out)
            .unwrap();
        checksum += out.weights().iter().sum::<f64>();
    }
    let elapsed = t.elapsed().as_secs_f64();

    let points = plan.len();
    let pair_ops = points as f64 * (charges.len() * charges.len()) as f64 / 2.0;
    println!("file            {path}");
    println!("atoms           {}", charges.len());
    println!("points          {points}");
    println!("batch           {batch}");
    println!("threads         {}", rayon::current_num_threads());
    println!("level           {level}");
    println!("scheme          {scheme:?}");
    println!("seconds         {elapsed:.6}");
    println!("ns/point        {:.3}", elapsed * 1e9 / points as f64);
    println!("Gpair/s         {:.3}", pair_ops / elapsed / 1e9);
    println!("checksum        {checksum:.12e}");
}

fn read_xyz(path: &str) -> Vec<(u32, [f64; 3])> {
    let text = std::fs::read_to_string(path).expect("read xyz");
    let mut lines = text.lines();
    let n: usize = lines
        .next()
        .expect("xyz atom count")
        .trim()
        .parse()
        .unwrap();
    let _comment = lines.next();
    let mut atoms = Vec::with_capacity(n);
    for line in lines.take(n) {
        let mut it = line.split_whitespace();
        let sym = it.next().expect("symbol");
        let x: f64 = it.next().unwrap().parse().unwrap();
        let y: f64 = it.next().unwrap().parse().unwrap();
        let z: f64 = it.next().unwrap().parse().unwrap();
        let scale = 1.0 / ruaa_dft_grids::BOHR;
        atoms.push((atomic_number(sym), [x * scale, y * scale, z * scale]));
    }
    atoms
}

fn atomic_number(sym: &str) -> u32 {
    const TABLE: &[(&str, u32)] = &[
        ("H", 1),
        ("He", 2),
        ("Li", 3),
        ("Be", 4),
        ("B", 5),
        ("C", 6),
        ("N", 7),
        ("O", 8),
        ("F", 9),
        ("Ne", 10),
        ("Na", 11),
        ("Mg", 12),
        ("Al", 13),
        ("Si", 14),
        ("P", 15),
        ("S", 16),
        ("Cl", 17),
        ("Ar", 18),
        ("K", 19),
        ("Ca", 20),
        ("Fe", 26),
        ("Zn", 30),
        ("Br", 35),
        ("I", 53),
    ];
    TABLE
        .iter()
        .find_map(|&(s, z)| (s == sym).then_some(z))
        .unwrap_or_else(|| panic!("unknown element symbol {sym}"))
}
