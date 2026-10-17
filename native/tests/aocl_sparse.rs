//! AOCL-Sparse's GMRES against photonoxide's own solvers, on this machine. Skipped, and saying
//! so, where AOCL-Sparse isn't found; with `PHOTONOXIDE_REQUIRE_AOCL_SPARSE` set (a runner
//! that built it) not finding it is a failure.

use std::sync::Arc;

use num_complex::Complex64 as c64;
use photonoxide::backend::{
    self, Choice, Form, IterativeSolver, Matrix, OwnPreconditioner, RowMatrix,
};
use photonoxide::fdfd::{
    Axis, Boundaries3d, Edges, Formulation, Grid3d, IterativeSolver3d, Stopping,
};
use photonoxide::units::Wavelength;
use photonoxide_native::{AoclSparse, aocl_sparse, offer_iterative, smoke_test_iterative};

fn aocl() -> Option<Arc<AoclSparse>> {
    match AoclSparse::load() {
        Ok(a) => Some(Arc::new(a)),
        Err(reason) => {
            assert!(
                std::env::var_os("PHOTONOXIDE_REQUIRE_AOCL_SPARSE").is_none(),
                "AOCL-Sparse is required here and isn't found: {reason}"
            );
            println!("skipped: AOCL-Sparse isn't here: {reason}");
            None
        }
    }
}

#[test]
fn it_finds_aocl_sparse_and_says_what_it_is() {
    let probe = aocl_sparse::probe();
    println!("{probe}");
    let Some(aocl) = aocl() else { return };
    let c = aocl.capabilities();
    println!("{c:?}, 64-bit integers: {}", aocl.wide_integers());
    assert_eq!(probe.version.as_deref(), Some(aocl.version()));
    assert_eq!(c.name, "aocl-sparse");
    assert!(
        c.version.starts_with(|ch: char| ch.is_ascii_digit()),
        "{}",
        c.version
    );
    assert!(c.complex && !c.symmetric && !c.transpose && !c.deterministic);
    assert!(c.licence.contains("MIT"));
    assert!(aocl.own_solver().unwrap().contains("GMRES"));
    // it runs none of photonoxide's own recurrences
    let (s, r, v) = (vec![0, 1], vec![0], vec![c64::new(1.0, 0.0)]);
    let m = Matrix::new(1, &s, &r, &v, Form::General).unwrap();
    assert!(
        aocl.qmr_run(&m, &[c64::new(1.0, 0.0)], Stopping::default())
            .is_err()
    );
    assert!(!aocl.runs_multigrid());
}

#[test]
fn it_passes_the_smoke_test() {
    let Some(aocl) = aocl() else { return };
    let difference = smoke_test_iterative(aocl.as_ref()).unwrap();
    println!("largest difference from photonoxide's direct solve: {difference:.1e}");
    assert!(difference < 1e-9);
}

/// A Helmholtz operator with loss on a grid, general (an asymmetric coupling along x), by
/// rows.
fn helmholtz(cells: [usize; 3]) -> (Vec<usize>, Vec<usize>, Vec<c64>) {
    let [mx, my, mz] = cells;
    let n = mx * my * mz;
    let strides = [1, mx, mx * my];
    let (mut starts, mut columns, mut values) = (vec![0], Vec::new(), Vec::new());
    for i in 0..n {
        let at = [i % mx, i / mx % my, i / (mx * my)];
        let mut row: Vec<(usize, c64)> = Vec::new();
        for axis in 0..3 {
            if at[axis] > 0 {
                row.push((i - strides[axis], c64::new(1.0, 0.0)));
            }
            if at[axis] + 1 < cells[axis] {
                let v = if axis == 0 {
                    c64::new(1.0, 0.1)
                } else {
                    c64::new(1.0, 0.0)
                };
                row.push((i + strides[axis], v));
            }
        }
        row.push((i, c64::new(-5.7, 0.3 + 0.001 * (i % 11) as f64)));
        row.sort_by_key(|&(c, _)| c);
        for (c, v) in row {
            columns.push(c);
            values.push(v);
        }
        starts.push(columns.len());
    }
    (starts, columns, values)
}

fn relative(a: &[c64], b: &[c64]) -> f64 {
    let scale = b.iter().map(|x| x.norm()).fold(0.0, f64::max);
    a.iter()
        .zip(b)
        .map(|(a, b)| (a - b).norm())
        .fold(0.0, f64::max)
        / scale
}

#[test]
fn its_gmres_solves_to_the_tolerance_plain_and_with_its_ilu() {
    let Some(aocl) = aocl() else { return };
    let own = backend::direct(&Choice::Photonoxide).unwrap();
    for cells in [[30, 30, 1], [12, 10, 8]] {
        let (s, c, v) = helmholtz(cells);
        let n = s.len() - 1;
        let rows = RowMatrix::new(n, &s, &c, &v).unwrap();
        let b: Vec<c64> = (0..n)
            .map(|k| c64::new(1.0, k as f64 * 1e-3).powi((k % 3) as i32))
            .collect();
        // the rows of A are the columns of Aᵀ: photonoxide's direct solve, transposed
        let transposed = Matrix::new(n, &s, &c, &v, Form::General).unwrap();
        let exact = own
            .analyse(&transposed)
            .unwrap()
            .unwrap()
            .factorize(&transposed)
            .unwrap()
            .solve_transpose(&b)
            .unwrap();
        let norm = |x: &[c64]| x.iter().map(|z| z.norm_sqr()).sum::<f64>().sqrt();
        for tolerance in [1e-6, 1e-10] {
            let stopping = Stopping {
                tolerance,
                max_iterations: 5000,
            };
            let mut taken = Vec::new();
            for preconditioner in [OwnPreconditioner::None, OwnPreconditioner::Ilu0] {
                let (x, how) = aocl.solve(rows, preconditioner, &b, stopping).unwrap();
                let ax = rows.apply(&x);
                let r: Vec<c64> = b.iter().zip(&ax).map(|(p, q)| p - q).collect();
                let residual = norm(&r) / norm(&b);
                let d = relative(&x, &exact);
                println!(
                    "{cells:?} to {tolerance:e}, {preconditioner:?}: {} iterations, a true \
                     residual of {residual:.1e} (it says {:.1e}), {d:.1e} from the direct solve",
                    how.iterations, how.residual
                );
                // what it says is what is: its residual is the true one, to rounding
                assert!(residual <= 1.01 * tolerance, "{residual:e}");
                assert!(d < 1e3 * tolerance, "{d:e}");
                assert_eq!(
                    how.history.len(),
                    how.iterations,
                    "a residual per iteration"
                );
                assert!(how.history.last().copied().unwrap() <= 1.01 * tolerance);
                taken.push(how.iterations);
            }
            // its ILU(0) is worth having
            assert!(taken[1] < taken[0], "{taken:?}");
        }
        // the limit is a limit, and said
        let short = Stopping {
            tolerance: 1e-12,
            max_iterations: 3,
        };
        let e = aocl
            .solve(rows, OwnPreconditioner::None, &b, short)
            .unwrap_err()
            .to_string();
        println!("{e}");
        assert!(e.contains("didn't reach"), "{e}");
        // and a right-hand side of another size is refused
        assert!(
            aocl.solve(rows, OwnPreconditioner::None, &b[1..], Stopping::default())
                .is_err()
        );
    }
}

/// The 3D FDFD solver with `aocl-sparse` chosen by name: the field photonoxide's QMR gives,
/// with PMLs (the symmetric similar matrix) and a Bloch side (the general one), plain and, on
/// Shin and Fan's operator, with ILU(0).
#[test]
fn the_3d_solver_runs_on_it_by_name() {
    let Some(aocl) = aocl() else { return };
    assert!(offer_iterative(aocl).unwrap());
    let (n, h) = (10, 0.05);
    let grid = Grid3d {
        nx: n,
        ny: n,
        nz: n,
        dx: h,
        dy: h,
        dz: h,
        x0: -0.25,
        y0: -0.25,
        z0: -0.25,
    };
    let cube = |x: f64, y: f64, z: f64| {
        let inside = x.abs() < 0.1 && y.abs() < 0.1 && z.abs() < 0.1;
        c64::new(if inside { 12.0 } else { 2.0 }, 0.0)
    };
    let lam = Wavelength::um(1.55).unwrap();
    let mut source = vec![c64::new(0.0, 0.0); grid.unknowns()];
    source[grid.index(Axis::Z, (5, 5, 5))] = c64::new(1.0, 0.0);
    let stopping = Stopping {
        tolerance: 1e-9,
        max_iterations: 20_000,
    };
    let by_name = Choice::Named("aocl-sparse".into());
    let bloch = Boundaries3d {
        x: Edges::Bloch { k: 0.8 },
        ..Boundaries3d::pml(3)
    };
    for (name, boundaries) in [("PMLs", Boundaries3d::pml(3)), ("Bloch along x", bloch)] {
        for (formulation, ilu) in [(Formulation::CurlCurl, false), (Formulation::ShinFan, true)] {
            let new = || {
                let solver =
                    IterativeSolver3d::new(grid, lam, cube, boundaries, formulation).unwrap();
                if ilu {
                    solver.with_ilu().unwrap()
                } else {
                    solver
                }
            };
            let clock = std::time::Instant::now();
            let (ours, how_ours) = new().solve(&source, stopping).unwrap();
            let ours_seconds = clock.elapsed().as_secs_f64();
            let clock = std::time::Instant::now();
            let outcome = new()
                .with_iterative_backend(&by_name)
                .unwrap()
                .solve(&source, stopping);
            let seconds = clock.elapsed().as_secs_f64();
            match outcome {
                Ok((theirs, how)) => {
                    let d = relative(theirs.values(), ours.values());
                    println!(
                        "{name}, ILU(0) {ilu} ({} unknowns): photonoxide's QMR {} iterations in \
                         {ours_seconds:.2} s, AOCL-Sparse's GMRES {} in {seconds:.2} s, residual \
                         {:.1e}, {d:.1e} apart",
                        grid.unknowns(),
                        how_ours.iterations,
                        how.iterations,
                        how.residual
                    );
                    assert!(how.residual <= 1e-9, "{how:?}");
                    assert!(d < 1e-5, "{name}, ILU(0) {ilu}: {d:e}");
                }
                // restarted GMRES may stall where QMR doesn't: said, not hidden
                Err(e) => {
                    println!(
                        "{name}, ILU(0) {ilu}: photonoxide's QMR {} iterations in \
                         {ours_seconds:.2} s; AOCL-Sparse's GMRES gave up after {seconds:.2} s: {e}",
                        how_ours.iterations
                    );
                    assert!(e.to_string().contains("didn't reach"), "{e}");
                }
            }
        }
    }
}
