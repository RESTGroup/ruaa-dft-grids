//! Random-order stress benchmark for grid-native partition derivatives.
//!
//! ```text
//! cargo run --release --example bench_grid_permute -- \
//!     <molecule.xyz> <level> [grad_points] [hess_points] [seed] [scheme] [rows]
//! ```
//!
//! A contiguous sample of the materialised grid is evaluated three ways:
//! in its build order, after a seeded random permutation, and through the
//! plan-based range path.  The permuted result is mapped back through the
//! permutation and compared with the sorted result; every case is timed with
//! the process CPU time so parallel utilisation is visible.
use ruaa_dft_grids::partition::DerivPoints;

use std::time::Instant;

use ruaa_dft_grids::grid::{GridPlan, GridSpec, PyscfGridSpecBuilder};
use ruaa_dft_grids::moleculegrid::{MolecularGrid, PostPrune, build_molecular_grid};
use ruaa_dft_grids::order::GridOrder;
use ruaa_dft_grids::partition::{
    DerivContract, DerivOptions, DerivOutput, DerivRequest, PartitionConfig, PartitionScheme,
    RadiiConfig,
};

struct Rng(u64);

impl Rng {
    fn next_u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    fn below(&mut self, bound: usize) -> usize {
        (self.next_u64() % bound as u64) as usize
    }
}

fn random_order(len: usize, seed: u64) -> GridOrder {
    let mut perm: Vec<u32> = (0..len as u32).collect();
    let mut rng = Rng(seed);
    for i in (1..len).rev() {
        let j = rng.below(i + 1);
        perm.swap(i, j);
    }
    GridOrder::from_permutation(&perm).expect("random permutation")
}

/// Shuffle the atom-contiguous runs of the grid, keeping each run intact.
fn block_order(grid: &MolecularGrid, seed: u64) -> GridOrder {
    let mut runs: Vec<(usize, usize)> = Vec::new();
    let mut start = 0;
    for i in 1..=grid.len() {
        if i == grid.len() || grid.atom_idx()[i] != grid.atom_idx()[start] {
            runs.push((start, i - start));
            start = i;
        }
    }
    let mut rng = Rng(seed);
    for i in (1..runs.len()).rev() {
        let j = rng.below(i + 1);
        runs.swap(i, j);
    }
    let mut perm = Vec::with_capacity(grid.len());
    for &(start, len) in &runs {
        perm.extend(start as u32..(start + len) as u32);
    }
    GridOrder::from_permutation(&perm).expect("block permutation")
}

fn make_order(mode: &str, grid: &MolecularGrid, len: usize, seed: u64) -> GridOrder {
    match mode {
        "identity" => GridOrder::identity(len).expect("identity order"),
        "blocks" => block_order(grid, seed),
        "boxes" => GridOrder::boxes(grid.coords(), 1.2).expect("box order"),
        _ => random_order(len, seed),
    }
}

#[cfg(target_os = "linux")]
fn cpu_seconds() -> f64 {
    let stat = std::fs::read_to_string("/proc/self/stat").expect("read /proc/self/stat");
    let tail = &stat[stat.rfind(')').expect("stat comm") + 2..];
    let fields: Vec<&str> = tail.split_whitespace().collect();
    let utime: u64 = fields[11].parse().expect("utime");
    let stime: u64 = fields[12].parse().expect("stime");
    (utime + stime) as f64 / 100.0
}

#[cfg(not(target_os = "linux"))]
fn cpu_seconds() -> f64 {
    0.0
}

struct Sample {
    wall: f64,
    cpu: f64,
}

fn sample(f: &mut impl FnMut()) -> Sample {
    let cpu0 = cpu_seconds();
    let start = Instant::now();
    f();
    Sample {
        wall: start.elapsed().as_secs_f64(),
        cpu: cpu_seconds() - cpu0,
    }
}

/// Report the minimum wall time of `samples` and the aggregate utilisation.
fn report(label: &str, points: usize, samples: &[Sample]) {
    let wall = samples.iter().map(|s| s.wall).fold(f64::INFINITY, f64::min);
    let cpu: f64 = samples.iter().map(|s| s.cpu).sum();
    let total: f64 = samples.iter().map(|s| s.wall).sum();
    println!(
        "  {label:<16} min wall {wall:8.4}s  cpu {cpu:8.3}s ({:8.2} ns/pt)  util {:6.1}%  wall ns/pt {:9.2}",
        cpu * 1e9 / (samples.len() * points) as f64,
        cpu / total * 100.0,
        wall * 1e9 / points as f64
    );
}

/// Fraction of eight-point lanes whose points name more than one atom.
fn mixed_lanes(grid: &MolecularGrid, block: usize) -> f64 {
    let mut mixed = 0usize;
    let mut total = 0usize;
    let mut start = 0;
    while start < grid.len() {
        let end = (start + block).min(grid.len());
        let mut b = start;
        while b < end {
            let e = (b + 8).min(end);
            let first = grid.atom_idx()[b];
            if (b + 1..e).any(|i| grid.atom_idx()[i] != first) {
                mixed += 1;
            }
            total += 1;
            b += 8;
        }
        start = end;
    }
    mixed as f64 / total.max(1) as f64
}

fn bits_diff(got: &[f64], want: &[f64]) -> (usize, f64) {
    assert_eq!(got.len(), want.len());
    let mut count = 0;
    let mut max_abs = 0.0_f64;
    for (&a, &b) in got.iter().zip(want) {
        if a.to_bits() != b.to_bits() {
            count += 1;
            max_abs = max_abs.max((a - b).abs());
        }
    }
    (count, max_abs)
}

/// Maximum absolute difference and the same difference relative to the
/// largest magnitude in `want`.
fn value_diff(got: &[f64], want: &[f64]) -> (f64, f64) {
    assert_eq!(got.len(), want.len());
    let mut max_abs = 0.0_f64;
    let mut max_want = 0.0_f64;
    for (&a, &b) in got.iter().zip(want) {
        max_abs = max_abs.max((a - b).abs());
        max_want = max_want.max(b.abs());
    }
    (max_abs, max_abs / max_want.max(1e-300))
}

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    assert!(
        args.len() >= 2,
        "usage: bench_grid_permute <molecule.xyz> <level> [grad_points] [hess_points] [seed] [scheme] [rows] [random|blocks|boxes|identity]"
    );
    let path = &args[0];
    let level: u8 = args[1].parse().unwrap();
    let grad_points_arg: usize = args.get(2).map_or(50_000, |s| s.parse().unwrap());
    let hess_points_arg: usize = args.get(3).map_or(512, |s| s.parse().unwrap());
    let seed: u64 = args.get(4).map_or(0x5EED_1234, |s| s.parse().unwrap());
    let scheme = match args.get(5).map(String::as_str) {
        Some("stratmann") => PartitionScheme::Stratmann,
        Some("mbecke") => PartitionScheme::MBecke,
        Some("lko") => PartitionScheme::Lko,
        _ => PartitionScheme::Becke,
    };
    let rows_arg: usize = args.get(6).map_or(0, |s| s.parse().unwrap());
    let order_mode = args.get(7).map_or("random", String::as_str);

    let atoms = read_xyz(path);
    let n = atoms.len();
    let rows = if rows_arg == 0 || rows_arg >= n {
        n
    } else {
        rows_arg
    };
    let selection = (rows < n).then(|| (0..rows).collect::<Vec<_>>());

    let spec: GridSpec = PyscfGridSpecBuilder::default()
        .level(level)
        .partition(PartitionConfig::new(scheme, RadiiConfig::pyscf()))
        .build()
        .unwrap()
        .into();
    let plan = GridPlan::build(&atoms, &spec).unwrap();
    let grid = build_molecular_grid(&atoms, &spec).unwrap();
    let points = grid.len();
    assert_eq!(points, plan.len());

    let grad_points = grad_points_arg.min(points);
    let hess_points = hess_points_arg.min(grad_points);

    // Keep one sample of the materialised grid: the gradient sample first,
    // then a smaller prefix for the Hessian.
    let mut grad_sorted = grid;
    let discarded = grad_sorted
        .split_off_pruned(PostPrune::KeepIndices((0..grad_points as u32).collect()))
        .unwrap();
    drop(discarded);
    let hess_sorted = grad_sorted
        .clone()
        .split(PostPrune::KeepIndices((0..hess_points as u32).collect()))
        .unwrap()
        .retained;

    let grad_order = make_order(order_mode, &grad_sorted, grad_points, seed);
    let hess_order = make_order(
        order_mode,
        &hess_sorted,
        hess_points,
        seed ^ 0xA5A5_5A5A_1234_5678,
    );
    let mut grad_perm = grad_sorted.clone();
    grad_perm.reorder(&grad_order).unwrap();
    let mut hess_perm = hess_sorted.clone();
    hess_perm.reorder(&hess_order).unwrap();

    let ev = grad_sorted
        .partition_deriv_evaluator(DerivOptions {
            atoms: selection.clone(),
            ..Default::default()
        })
        .unwrap();

    println!("file            {path}");
    println!("atoms           {n}");
    println!("rows            {rows}");
    println!("points          {points}");
    println!("grad_points     {grad_points}");
    println!("hess_points     {hess_points}");
    println!("scheme          {scheme:?}");
    println!("order           {order_mode}");
    println!("threads         {}", rayon::current_num_threads());
    let block = ev.gradient_block();
    println!(
        "mixed lanes     sorted {:.1}%  ordered {:.1}%  (gradient block {block})",
        mixed_lanes(&grad_sorted, block) * 100.0,
        mixed_lanes(&grad_perm, block) * 100.0
    );

    let reps: usize = std::env::var("BENCH_REPS")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(7);
    let grad_req = DerivRequest {
        weights: true,
        dw: true,
        ..Default::default()
    };
    let mut out_sorted = DerivOutput::new();
    let mut out_perm = DerivOutput::new();
    let mut out_plan = DerivOutput::new();

    let mut run_sorted = || {
        ev.fill(
            DerivPoints::all_grid(&grad_sorted),
            &grad_req,
            &mut out_sorted,
        )
        .unwrap();
        std::hint::black_box(out_sorted.dw().last());
    };
    let mut run_perm = || {
        ev.fill(DerivPoints::all_grid(&grad_perm), &grad_req, &mut out_perm)
            .unwrap();
        std::hint::black_box(out_perm.dw().last());
    };
    let mut run_plan = || {
        ev.fill(
            DerivPoints::plan(&plan, 0..grad_points),
            &grad_req,
            &mut out_plan,
        )
        .unwrap();
        std::hint::black_box(out_plan.dw().last());
    };

    println!("gradient");
    run_sorted();
    run_perm();
    run_plan();
    let mut sorted_samples = Vec::with_capacity(reps);
    let mut perm_samples = Vec::with_capacity(reps);
    let mut plan_samples = Vec::with_capacity(reps);
    for _ in 0..reps {
        sorted_samples.push(sample(&mut run_sorted));
        perm_samples.push(sample(&mut run_perm));
        plan_samples.push(sample(&mut run_plan));
    }
    report("grid sorted", grad_points, &sorted_samples);
    report("grid permuted", grad_points, &perm_samples);
    report("plan range", grad_points, &plan_samples);
    let s = min_wall(&sorted_samples);
    let p = min_wall(&perm_samples);
    let r = min_wall(&plan_samples);
    println!(
        "  ratios           perm/sorted {:.3}  plan/sorted {:.3}",
        p / s,
        r / s
    );

    let perm = grad_order.perm();
    let rows_out = out_perm.dw().len() / grad_points;
    let (mut w_bits, mut w_max) = (0, 0.0_f64);
    for (slot, &source) in perm.iter().enumerate() {
        let source = source as usize;
        let a = out_perm.partition_weights()[slot];
        let b = out_sorted.partition_weights()[source];
        if a.to_bits() != b.to_bits() {
            w_bits += 1;
            w_max = w_max.max((a - b).abs());
        }
    }
    let (mut dw_bits, mut dw_max) = (0, 0.0_f64);
    for (slot, &source) in perm.iter().enumerate() {
        let source = source as usize;
        for row in 0..rows_out {
            let a = out_perm.dw()[row * grad_points + slot];
            let b = out_sorted.dw()[row * grad_points + source];
            if a.to_bits() != b.to_bits() {
                dw_bits += 1;
                dw_max = dw_max.max((a - b).abs());
            }
        }
    }
    println!(
        "  reorder check    weights {w_bits}/{grad_points} differ (max {w_max:.3e})  dw {dw_bits}/{} differ (max {dw_max:.3e})",
        out_sorted.dw().len()
    );
    let (plan_w_bits, plan_w_max) =
        bits_diff(out_plan.partition_weights(), out_sorted.partition_weights());
    let (plan_dw_bits, plan_dw_max) = bits_diff(out_plan.dw(), out_sorted.dw());
    println!(
        "  plan check       weights {plan_w_bits}/{grad_points} differ (max {plan_w_max:.3e})  dw {plan_dw_bits}/{} differ (max {plan_dw_max:.3e})",
        out_sorted.dw().len()
    );

    drop(out_sorted);
    drop(out_perm);
    drop(out_plan);

    let coefficients = vec![1.0; hess_points];
    let hess_req = DerivRequest {
        dc: Some(DerivContract::exact(&coefficients, 1)),
        ddc: Some(DerivContract::exact(&coefficients, 1)),
        ..Default::default()
    };
    let mut hess_out = DerivOutput::new();
    let mut hess_perm_out = DerivOutput::new();
    let mut hess_plan_out = DerivOutput::new();

    let mut run_hess = || {
        hess_out.zero_contracts();
        ev.fill(
            DerivPoints::all_grid(&hess_sorted),
            &hess_req,
            &mut hess_out,
        )
        .unwrap();
        std::hint::black_box(hess_out.ddc().unwrap().ddc()[0]);
    };
    let mut run_hess_perm = || {
        hess_perm_out.zero_contracts();
        ev.fill(
            DerivPoints::all_grid(&hess_perm),
            &hess_req,
            &mut hess_perm_out,
        )
        .unwrap();
        std::hint::black_box(hess_perm_out.ddc().unwrap().ddc()[0]);
    };
    let mut run_hess_plan = || {
        hess_plan_out.zero_contracts();
        ev.fill(
            DerivPoints::plan(&plan, 0..hess_points),
            &hess_req,
            &mut hess_plan_out,
        )
        .unwrap();
        std::hint::black_box(hess_plan_out.ddc().unwrap().ddc()[0]);
    };

    println!("hessian");
    run_hess();
    run_hess_perm();
    run_hess_plan();
    let mut hs_samples = Vec::with_capacity(reps);
    let mut hp_samples = Vec::with_capacity(reps);
    let mut hr_samples = Vec::with_capacity(reps);
    for _ in 0..reps {
        hs_samples.push(sample(&mut run_hess));
        hp_samples.push(sample(&mut run_hess_perm));
        hr_samples.push(sample(&mut run_hess_plan));
    }
    report("grid sorted", hess_points, &hs_samples);
    report("grid permuted", hess_points, &hp_samples);
    report("plan range", hess_points, &hr_samples);
    let hs = min_wall(&hs_samples);
    let hp = min_wall(&hp_samples);
    let hr = min_wall(&hr_samples);
    println!(
        "  ratios           perm/sorted {:.3}  plan/sorted {:.3}",
        hp / hs,
        hr / hs
    );

    let (ddc_abs, ddc_rel) = value_diff(
        hess_perm_out.ddc().unwrap().ddc(),
        hess_out.ddc().unwrap().ddc(),
    );
    let (dc_abs, dc_rel) = value_diff(
        hess_perm_out.dc().unwrap().dc(),
        hess_out.dc().unwrap().dc(),
    );
    println!(
        "  reorder check    ddc max abs {ddc_abs:.3e} (rel {ddc_rel:.3e})  dc max abs {dc_abs:.3e} (rel {dc_rel:.3e})"
    );
    let (plan_ddc_abs, plan_ddc_rel) = value_diff(
        hess_plan_out.ddc().unwrap().ddc(),
        hess_out.ddc().unwrap().ddc(),
    );
    let (plan_dc_abs, plan_dc_rel) = value_diff(
        hess_plan_out.dc().unwrap().dc(),
        hess_out.dc().unwrap().dc(),
    );
    println!(
        "  plan check       ddc max abs {plan_ddc_abs:.3e} (rel {plan_ddc_rel:.3e})  dc max abs {plan_dc_abs:.3e} (rel {plan_dc_rel:.3e})"
    );

    drop(hess_out);
    drop(hess_perm_out);
    drop(hess_plan_out);

    // Per-point Hessian tensors, mapped back and compared bit for bit.
    let ddw_points = hess_points.min(64);
    let ddw_sorted = hess_sorted
        .clone()
        .split(PostPrune::KeepIndices((0..ddw_points as u32).collect()))
        .unwrap()
        .retained;
    let ddw_order = make_order(
        order_mode,
        &ddw_sorted,
        ddw_points,
        seed ^ 0x0F0F_0F0F_DEAD_BEEF,
    );
    let mut ddw_perm = ddw_sorted.clone();
    ddw_perm.reorder(&ddw_order).unwrap();
    let ddw_req = DerivRequest {
        ddw: true,
        ..Default::default()
    };
    let mut a = DerivOutput::new();
    let mut b = DerivOutput::new();
    ev.fill(DerivPoints::all_grid(&ddw_sorted), &ddw_req, &mut a)
        .unwrap();
    ev.fill(DerivPoints::all_grid(&ddw_perm), &ddw_req, &mut b)
        .unwrap();
    let cols = a.ddw().len() / ddw_points;
    let (mut ddw_bits, mut ddw_max) = (0, 0.0_f64);
    for (slot, &source) in ddw_order.perm().iter().enumerate() {
        let source = source as usize;
        for c in 0..cols {
            let x = b.ddw()[c * ddw_points + slot];
            let y = a.ddw()[c * ddw_points + source];
            if x.to_bits() != y.to_bits() {
                ddw_bits += 1;
                ddw_max = ddw_max.max((x - y).abs());
            }
        }
    }
    println!(
        "  ddw check        {ddw_bits}/{} differ (max {ddw_max:.3e}) on {ddw_points} points",
        a.ddw().len()
    );
}

fn min_wall(samples: &[Sample]) -> f64 {
    samples.iter().map(|s| s.wall).fold(f64::INFINITY, f64::min)
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
