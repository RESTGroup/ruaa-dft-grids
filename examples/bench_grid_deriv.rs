//! Grid-native derivative throughput and parallel CPU utilisation.
//!
//! ```text
//! cargo run --release --example bench_grid_deriv -- [waters] [level] [repeat]
//! ```
//!
//! Builds a water cluster, evaluates the grid-native gradient or the
//! contracted Hessian on the whole grid, and reports wall time and the
//! process CPU time from `/proc/self/stat`.  The same evaluation is run on the
//! plan-based path for comparison and inside a one-thread Rayon pool for the
//! speedup.
use ruaa_dft_grids::partition::DerivPoints;

use std::time::Instant;

use ruaa_dft_grids::grid::{GridPlan, GridSpec, PyscfGridSpecBuilder};
use ruaa_dft_grids::moleculegrid::build_molecular_grid;
use ruaa_dft_grids::partition::{
    DerivContract, DerivOptions, DerivOutput, DerivRequest, PartitionConfig, PartitionScheme,
    RadiiConfig,
};

fn cluster(waters: usize) -> Vec<(u32, [f64; 3])> {
    let mut atoms = Vec::with_capacity(3 * waters);
    for i in 0..waters {
        let z = i as f64 * 5.0;
        atoms.push((8, [0.0, 0.0, z]));
        atoms.push((1, [0.0, 1.43, z + 1.11]));
        atoms.push((1, [0.0, -1.43, z + 1.11]));
    }
    atoms
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

fn bench(label: &str, repeat: usize, mut f: impl FnMut()) {
    f();
    let cpu0 = cpu_seconds();
    let start = Instant::now();
    for _ in 0..repeat {
        f();
    }
    let wall = start.elapsed().as_secs_f64();
    let cpu = cpu_seconds() - cpu0;
    println!(
        "{label:<26} wall {wall:8.3}s  cpu {cpu:8.3}s  util {:6.1}%  per-iter {:7.3}s",
        cpu / wall * 100.0,
        wall / repeat as f64
    );
}

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let waters: usize = args.first().map_or(10, |s| s.parse().unwrap());
    let level: u8 = args.get(1).map_or(1, |s| s.parse().unwrap());
    let repeat: usize = args.get(2).map_or(3, |s| s.parse().unwrap());
    let run_serial = args.get(3).is_none_or(|s| s != "0");

    let atoms = cluster(waters);
    let n = atoms.len();
    let spec: GridSpec = PyscfGridSpecBuilder::default()
        .level(level)
        .partition(PartitionConfig::new(
            PartitionScheme::Becke,
            RadiiConfig::pyscf(),
        ))
        .build()
        .unwrap()
        .into();
    let plan = GridPlan::build(&atoms, &spec).unwrap();
    let grid = build_molecular_grid(&atoms, &spec).unwrap();
    let points = grid.len();
    let ev = grid
        .partition_deriv_evaluator(DerivOptions::default())
        .unwrap();

    println!("waters {waters}  atoms {n}  level {level}  points {points}");
    println!(
        "rayon threads {}  plan points {}",
        rayon::current_num_threads(),
        plan.len()
    );

    let grad_req = DerivRequest {
        weights: true,
        dw: true,
        ..Default::default()
    };
    let mut out = DerivOutput::new();
    bench("grid grad (ambient)", repeat, || {
        ev.fill(DerivPoints::all_grid(&grid), &grad_req, &mut out)
            .unwrap();
        std::hint::black_box(out.dw().first());
    });

    let mut out_plan = DerivOutput::new();
    bench("plan grad_par (ambient)", repeat, || {
        ev.fill(
            DerivPoints::plan(&plan, 0..plan.len()),
            &grad_req,
            &mut out_plan,
        )
        .unwrap();
        std::hint::black_box(out_plan.dw().first());
    });

    let coefficients = vec![1.0; points];

    let weights_req = DerivRequest {
        weights: true,
        ..Default::default()
    };
    let mut weights_out = DerivOutput::new();
    bench("grid weights (ambient)", repeat, || {
        ev.fill(DerivPoints::all_grid(&grid), &weights_req, &mut weights_out)
            .unwrap();
        std::hint::black_box(weights_out.partition_weights().first());
    });

    let dc_req = DerivRequest {
        dc: Some(DerivContract::exact(&coefficients, 1)),
        ..Default::default()
    };
    let mut dc_out = DerivOutput::new();
    bench("grid dc (ambient)", repeat, || {
        dc_out.zero_contracts();
        ev.fill(DerivPoints::all_grid(&grid), &dc_req, &mut dc_out)
            .unwrap();
        std::hint::black_box(dc_out.dc().unwrap().dc().first());
    });

    let ddc_req = DerivRequest {
        ddc: Some(DerivContract::exact(&coefficients, 1)),
        ..Default::default()
    };
    let mut hess = DerivOutput::new();
    bench("grid ddc (ambient)", repeat, || {
        hess.zero_contracts();
        ev.fill(DerivPoints::all_grid(&grid), &ddc_req, &mut hess)
            .unwrap();
        std::hint::black_box(hess.ddc().unwrap().ddc().first());
    });

    let mut hess_plan = DerivOutput::new();
    bench("plan ddc_par (ambient)", repeat, || {
        hess_plan.zero_contracts();
        ev.fill(
            DerivPoints::plan(&plan, 0..plan.len()),
            &ddc_req,
            &mut hess_plan,
        )
        .unwrap();
        std::hint::black_box(hess_plan.ddc().unwrap().ddc().first());
    });

    if run_serial {
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(1)
            .build()
            .unwrap();
        let mut serial = DerivOutput::new();
        bench("grid grad (1 thread)", repeat, || {
            pool.install(|| ev.fill(DerivPoints::all_grid(&grid), &grad_req, &mut serial))
                .unwrap();
            std::hint::black_box(serial.dw().first());
        });

        let mut serial_hess = DerivOutput::new();
        bench("grid ddc (1 thread)", repeat, || {
            serial_hess.zero_contracts();
            pool.install(|| ev.fill(DerivPoints::all_grid(&grid), &ddc_req, &mut serial_hess))
                .unwrap();
            std::hint::black_box(serial_hess.ddc().unwrap().ddc().first());
        });
    }
}
