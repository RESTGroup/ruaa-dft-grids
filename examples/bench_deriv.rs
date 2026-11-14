//! Gradient/Hessian throughput benchmark for the partition schemes.
//!
//! ```text
//! cargo run --release --example bench_deriv -- \
//!     <molecule.xyz> <level> <grad|hess|both|combined> [batch] [nsel] [points] [block] [auto|lane|pointwise] [becke|stratmann|gaussian_ss|mbecke|lko]
//! ```
//!
//! `nsel` restricts the differentiated rows to the first `nsel` atoms (all
//! when omitted); `points` limits how many grid points are evaluated, which
//! is what makes the `O(natm^3)` Hessian measurable on a large molecule.
//! `combined` evaluates the gradient and the Hessian in one fused pass;
//! `both` runs the two dedicated paths for comparison.  Only partition
//! derivative evaluation is timed.
use ruaa_dft_grids::partition::DerivPoints;

use std::time::Instant;

use ruaa_dft_grids::grid::{GridPlan, GridSpec, PyscfGridSpecBuilder};
use ruaa_dft_grids::partition::{
    DerivContract, DerivOptions, DerivOutput, DerivRequest, HessianBackend, PartitionConfig,
    PartitionDerivEvaluator, PartitionScheme, RadiiConfig,
};

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    assert!(
        args.len() >= 3,
        "usage: bench_deriv <molecule.xyz> <level> <grad|hess|both|combined> [batch] [nsel] [points] [block] [auto|lane|pointwise]"
    );
    let path = &args[0];
    let level: u8 = args[1].parse().unwrap();
    let mode = args[2].as_str();
    let batch: usize = args.get(3).map_or(1024, |s| s.parse().unwrap());
    let atoms = read_xyz(path);
    let n = atoms.len();
    let nsel: usize = args.get(4).map_or(n, |s| s.parse().unwrap());
    let block_points: usize = args.get(6).map_or(0, |s| s.parse().unwrap());
    let scheme = match args.get(8).map(String::as_str) {
        Some("stratmann") => PartitionScheme::Stratmann,
        Some("gaussian_ss") => PartitionScheme::GaussianSs,
        Some("mbecke") => PartitionScheme::MBecke,
        Some("lko") => PartitionScheme::Lko,
        _ => PartitionScheme::Becke,
    };
    let hessian_backend = match args.get(7).map(String::as_str) {
        Some("lane") => HessianBackend::Lane,
        Some("pointwise") => HessianBackend::Pointwise,
        _ => HessianBackend::Auto,
    };
    let selection = (nsel < n).then(|| (0..nsel).collect::<Vec<_>>());
    let charges: Vec<u32> = atoms.iter().map(|a| a.0).collect();
    let centers: Vec<[f64; 3]> = atoms.iter().map(|a| a.1).collect();

    let spec: GridSpec = PyscfGridSpecBuilder::default()
        .level(level)
        .partition(PartitionConfig::new(scheme, RadiiConfig::pyscf()))
        .build()
        .unwrap()
        .into();
    let plan = GridPlan::build(&atoms, &spec).unwrap();
    let ev = PartitionDerivEvaluator::from_parts(
        &centers,
        &charges,
        spec.partition().clone(),
        DerivOptions {
            atoms: selection.clone(),
            block_points,
            worker_memory_bytes: 0,
            hessian_backend,
        },
    )
    .unwrap();

    let available = plan.len();
    // A full-grid gradient is output-sized (nsel*3*points doubles); default to
    // a bounded sample so `bench_deriv ... grad` cannot accidentally allocate
    // gigabytes.  The Hessian is O(n^3) per point and defaults to 64 points.
    let default_points = if mode == "hess" {
        64
    } else {
        available.min(200_000)
    };
    let points: usize = args.get(5).map_or(default_points, |s| {
        s.parse::<usize>().unwrap().min(available)
    });
    let ranges: Vec<_> = plan
        .batches(batch)
        .take_while(|r| r.start < points)
        .map(|r| r.start..r.end.min(points))
        .collect();

    println!("file            {path}");
    println!("atoms           {n}");
    println!("selected        {}", selection.as_ref().map_or(n, Vec::len));
    println!("points_available {available}");
    println!("points          {points}");
    println!("batch           {batch}");
    println!("threads         {}", rayon::current_num_threads());
    println!("level           {level}");
    println!("hessian_backend {hessian_backend:?}");

    let contract: Vec<f64> = (0..batch).map(|k| 0.5 + 0.001 * k as f64).collect();

    if mode == "grad" || mode == "both" {
        let req = DerivRequest {
            weights: true,
            dw: true,
            ..Default::default()
        };
        let mut out = DerivOutput::new();
        let mut checksum = 0.0;
        if let Some(r) = ranges.first() {
            ev.fill(DerivPoints::plan(&plan, r.clone()), &req, &mut out)
                .unwrap();
            checksum += out.dw().iter().sum::<f64>();
        }
        let t = Instant::now();
        for r in &ranges {
            ev.fill(DerivPoints::plan(&plan, r.clone()), &req, &mut out)
                .unwrap();
            checksum += out.dw().iter().sum::<f64>();
        }
        let elapsed = t.elapsed().as_secs_f64();
        let nsel = selection.as_ref().map_or(n, Vec::len);
        let ops = points as f64 * (n * n) as f64 / 2.0 * (1.0 + nsel as f64 * 6.0 / n as f64);
        println!("grad_seconds    {elapsed:.6}");
        println!("grad_ns/point   {:.3}", elapsed * 1e9 / points as f64);
        println!("grad_Gop/s      {:.3}", ops / elapsed / 1e9);
        println!("grad_checksum   {checksum:.12e}");
    }

    if mode == "hess" || mode == "both" {
        let mut out = DerivOutput::new();
        let mut checksum = 0.0;
        let make_req = |r: &std::ops::Range<usize>| DerivRequest {
            ddc: Some(DerivContract::exact(&contract[..r.len()], 1)),
            ..Default::default()
        };
        if let Some(r) = ranges.first() {
            ev.fill(DerivPoints::plan(&plan, r.clone()), &make_req(r), &mut out)
                .unwrap();
            checksum += out.ddc().unwrap().ddc().iter().sum::<f64>();
        }
        let t = Instant::now();
        for r in &ranges {
            ev.fill(DerivPoints::plan(&plan, r.clone()), &make_req(r), &mut out)
                .unwrap();
            checksum += out.ddc().unwrap().ddc().iter().sum::<f64>();
        }
        let elapsed = t.elapsed().as_secs_f64();
        let nsel = selection.as_ref().map_or(n, Vec::len);
        let n3 = 3 * n;
        // dominant rank-one cross rows: nsel*3 * n * n3
        let ops = points as f64 * (nsel * 3 * n * n3) as f64;
        println!("hess_seconds    {elapsed:.6}");
        println!("hess_ns/point   {:.3}", elapsed * 1e9 / points as f64);
        println!("hess_Gop/s      {:.3}", ops / elapsed / 1e9);
        println!("hess_checksum   {checksum:.12e}");
    }

    if mode == "combined" {
        let mut out = DerivOutput::new();
        let mut checksum = 0.0;
        let make_req = |r: &std::ops::Range<usize>| DerivRequest {
            weights: true,
            dw: true,
            dc: Some(DerivContract::exact(&contract[..r.len()], 1)),
            ddc: Some(DerivContract::exact(&contract[..r.len()], 1)),
            ..Default::default()
        };
        if let Some(r) = ranges.first() {
            ev.fill(DerivPoints::plan(&plan, r.clone()), &make_req(r), &mut out)
                .unwrap();
            checksum += out.dw().iter().sum::<f64>();
            checksum += out.dc().unwrap().dc().iter().sum::<f64>();
            checksum += out.ddc().unwrap().ddc().iter().sum::<f64>();
        }
        let t = Instant::now();
        for r in &ranges {
            ev.fill(DerivPoints::plan(&plan, r.clone()), &make_req(r), &mut out)
                .unwrap();
            checksum += out.dw().iter().sum::<f64>();
            checksum += out.dc().unwrap().dc().iter().sum::<f64>();
            checksum += out.ddc().unwrap().ddc().iter().sum::<f64>();
        }
        let elapsed = t.elapsed().as_secs_f64();
        println!("both_seconds    {elapsed:.6}");
        println!("both_ns/point   {:.3}", elapsed * 1e9 / points as f64);
        println!("both_checksum   {checksum:.12e}");
    }
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
