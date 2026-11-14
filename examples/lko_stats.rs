//! Diagnostic: LKO screening-set statistics on a real grid.
//!
//! ```text
//! cargo run --release --example lko_stats -- <molecule.xyz> [level] [points]
//! ```
//!
//! Prints the mean/p99 of `|S|` (live centres, `d_i <= d_min + 5`) and
//! `|T|` (candidate partners, `d_i <= d_min + 10`), the mean `|S| |T|` pair
//! bound, and the fraction of points whose generating atom is not live
//! (weight exactly zero, pair loop skipped).

use ruaa_dft_grids::grid::{GridPlan, GridSpec, PyscfGridSpecBuilder};
use ruaa_dft_grids::partition::{PartitionConfig, PartitionScheme, RadiiConfig};

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    assert!(
        !args.is_empty(),
        "usage: lko_stats <molecule.xyz> [level] [points]"
    );
    let level: u8 = args.get(1).map_or(3, |s| s.parse().unwrap());
    let atoms = read_xyz(&args[0]);
    let spec: GridSpec = PyscfGridSpecBuilder::default()
        .level(level)
        .partition(PartitionConfig::new(
            PartitionScheme::Lko,
            RadiiConfig::pyscf(),
        ))
        .build()
        .unwrap()
        .into();
    let plan = GridPlan::build(&atoms, &spec).unwrap();
    let centers: Vec<[f64; 3]> = atoms.iter().map(|a| a.1).collect();
    let limit: usize = args
        .get(2)
        .map_or(plan.len(), |s| s.parse::<usize>().unwrap().min(plan.len()));
    let n = centers.len();
    let mut s_sizes = Vec::with_capacity(limit);
    let mut t_sizes = Vec::with_capacity(limit);
    let mut dead = 0usize;
    let mut dist = vec![0.0; n];
    // Union-pair statistics for the eight-point lane design.
    let mut lane_points = 0usize;
    let mut lane_pairs = 0usize;
    let mut lane_occ = 0usize;
    let mut s_masks = vec![0u8; n];
    let mut t_masks = vec![0u8; n];
    let mut block: Vec<([f64; 3], usize)> = Vec::with_capacity(8);
    for g in 0..limit {
        let local = plan.local_coord(g);
        let ia = plan.parent_of(g) as usize;
        let c = centers[ia];
        let p = [c[0] + local[0], c[1] + local[1], c[2] + local[2]];
        let mut dmin = f64::INFINITY;
        for (k, center) in centers.iter().enumerate() {
            let d = dist3(p, *center);
            dist[k] = d;
            dmin = dmin.min(d);
        }
        let s_cut = dmin + 5.0 + 1e-9;
        let t_cut = dmin + 10.0 + 1e-9;
        let ns = dist.iter().filter(|&&d| d <= s_cut).count();
        let nt = dist.iter().filter(|&&d| d <= t_cut).count();
        if dist[ia] > s_cut {
            dead += 1;
        }
        s_sizes.push(ns);
        t_sizes.push(nt);
        // Lane statistics: gather up to eight consecutive points.
        block.push((p, ia));
        if block.len() == 8 || g + 1 == limit {
            s_masks.iter_mut().for_each(|m| *m = 0);
            t_masks.iter_mut().for_each(|m| *m = 0);
            for (l, &(bp, bia)) in block.iter().enumerate() {
                let _ = bia;
                let mut bd = vec![0.0; n];
                let mut bdmin = f64::INFINITY;
                for (k, center) in centers.iter().enumerate() {
                    let d = dist3(bp, *center);
                    bd[k] = d;
                    bdmin = bdmin.min(d);
                }
                let sc = bdmin + 5.0 + 1e-9;
                let tc = bdmin + 10.0 + 1e-9;
                for k in 0..n {
                    if bd[k] <= tc {
                        t_masks[k] |= 1 << l;
                        if bd[k] <= sc {
                            s_masks[k] |= 1 << l;
                        }
                    }
                }
            }
            // Union T list.
            let union_t: Vec<usize> = (0..n).filter(|&k| t_masks[k] != 0).collect();
            for (a, &i) in union_t.iter().enumerate() {
                for &j in &union_t[..a] {
                    let need_i = s_masks[i] & t_masks[j];
                    let need_j = s_masks[j] & t_masks[i];
                    let need = need_i | need_j;
                    if need != 0 {
                        lane_pairs += 1;
                        lane_occ += need.count_ones() as usize;
                    }
                }
            }
            lane_points += block.len();
            block.clear();
        }
    }
    let n_points = limit;
    let pct = |v: &mut Vec<usize>, q: f64| -> usize {
        v.sort_unstable();
        v[((v.len() as f64 - 1.0) * q).round() as usize]
    };
    let mean = |v: &[usize]| v.iter().sum::<usize>() as f64 / v.len() as f64;
    let n = limit;
    println!("molecule        {}", args[0]);
    println!("atoms           {}", centers.len());
    println!("points          {} of {}", n, plan.len());
    println!("S_mean          {:.2}", mean(&s_sizes));
    println!("S_p99           {}", pct(&mut s_sizes, 0.99));
    println!("T_mean          {:.2}", mean(&t_sizes));
    println!("T_p99           {}", pct(&mut t_sizes, 0.99));
    println!(
        "pairs_mean      {:.1}",
        s_sizes
            .iter()
            .zip(&t_sizes)
            .map(|(&a, &b)| (a * b) as f64)
            .sum::<f64>()
            / n as f64
    );
    println!("dead_center     {:.4}", dead as f64 / n as f64);
    println!(
        "lane_pairs_pt   {:.1}",
        lane_pairs as f64 / lane_points as f64
    );
    println!("lane_occ        {:.2}", lane_occ as f64 / lane_pairs as f64);
    let sparse_pairs: f64 = s_sizes
        .iter()
        .zip(&t_sizes)
        .map(|(&a, &b)| (a * b) as f64)
        .sum::<f64>()
        / n_points as f64;
    println!(
        "lane_vs_sparse  {:.2}",
        (lane_pairs as f64 / lane_points as f64) / sparse_pairs
    );
}

#[inline]
fn dist3(a: [f64; 3], b: [f64; 3]) -> f64 {
    let dx = a[0] - b[0];
    let dy = a[1] - b[1];
    let dz = a[2] - b[2];
    (dx * dx + dy * dy + dz * dz).sqrt()
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
