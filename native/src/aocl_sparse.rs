//! AMD's AOCL-Sparse as an iterative backend with a solver of its own (#189): its restarted
//! GMRES on complex matrices, plain or with its ILU(0), through its iterative solver suite
//! (`aoclsparse_itsol_z_solve`).
//!
//! Written from AOCL-Sparse 5.3.2's public headers (`aoclsparse_solvers.h`,
//! `aoclsparse_auxiliary.h`, `aoclsparse_analysis.h`, `aoclsparse_types.h`) and its example
//! `sample_itsol_z_gmres.cpp`; the library is MIT-licensed, installed by the user and never
//! redistributed.
//!
//! - **What it is to photonoxide.** Not photonoxide's QMR on another library's kernels, as the
//!   GPU backend is, but another method: where photonoxide would run QMR, plain or with its
//!   ILU(0), this backend, chosen by name, runs AOCL-Sparse's GMRES, restarted every
//!   [`RESTART`] steps, plain or with AOCL-Sparse's own ILU(0)
//!   ([`IterativeSolver::own_solver`]). photonoxide computes the true residual of what comes
//!   back and refuses a solution above the tolerance. `auto` never picks it.
//! - **The interface** needs no structure's layout: the matrix, its descriptor and the
//!   solver's handle are opaque, and the solver's settings are set by name
//!   (`aoclsparse_itsol_option_set`): the method, the preconditioner, the restart, the
//!   iteration limit, and the relative tolerance on ‖b − A x‖ / ‖b‖ (the absolute one set to
//!   zero).
//! - **Integers.** AOCL-Sparse is built with 32-bit integers (LP64) or 64-bit ones (ILP64),
//!   under the same names. Which, is found by asking it for a 1 × 1 matrix both ways: the
//!   row starts of one are not valid row starts read as the other. Either build is used.
//!   Integers passed by value are passed as 64-bit ones, which a 32-bit build reads the low
//!   half of.
//! - **Each solve** makes the matrix (a copy of the rows, with the library's integers), its
//!   descriptor and a handle, and destroys them: the ILU(0) is the library's to compute, once
//!   per solve.
//! - **History:** the library's residual after each iteration (`rinfo[0]` over `rinfo[1]`),
//!   by its monitoring callback.
//! - **Threads** are OpenMP's (`OMP_NUM_THREADS`). Not declared deterministic.

use std::ffi::{CStr, CString, c_char, c_int, c_void};
use std::path::PathBuf;
use std::sync::Mutex;

use num_complex::Complex64 as c64;
use photonoxide::Result;
use photonoxide::backend::{
    Capabilities, IterativeSolver, Matrix, OwnPreconditioner, QmrRun, RowMatrix, Threads,
};
use photonoxide::fdfd::{Convergence, Stopping};

use crate::Probe;
use crate::discovery::Spec;
use crate::library::{Library, error, load};

/// The steps of GMRES between restarts: photonoxide's own default for its GMRES.
pub const RESTART: usize = 40;

/// AOCL-Sparse's library.
pub const AOCL_SPARSE: Spec = Spec {
    name: "AOCL-Sparse",
    files: if cfg!(windows) {
        &["aoclsparse.dll"]
    } else if cfg!(target_os = "linux") {
        &["libaoclsparse.so"]
    } else {
        &[]
    },
    variables: &["PHOTONOXIDE_AOCL_SPARSE"],
    install,
    wheel: None,
};

/// Where AMD's installers and `AOCL_ROOT` put it.
fn install() -> Vec<PathBuf> {
    let mut roots: Vec<PathBuf> = std::env::var_os("AOCL_ROOT")
        .map(PathBuf::from)
        .into_iter()
        .collect();
    if cfg!(windows) {
        roots.push(PathBuf::from(r"C:\Program Files\AMD\AOCL-Windows"));
    }
    roots
        .into_iter()
        .flat_map(|root| {
            [
                root.join("lib"),
                root.join("lib_LP64"),
                root.join("amd-sparse").join("lib").join("LP64"),
                root.join("amd-sparse").join("lib").join("ILP64"),
            ]
        })
        .collect()
}

type Handle = *mut c_void;
/// `aoclsparse_create_zcsr(mat, base, M, N, nnz, row_ptr, col_idx, val)`.
type CreateZcsr = unsafe extern "C" fn(
    *mut Handle,
    c_int,
    i64,
    i64,
    i64,
    *mut c_void,
    *mut c_void,
    *mut c64,
) -> c_int;
type DestroyAt = unsafe extern "C" fn(*mut Handle) -> c_int;
type Make = unsafe extern "C" fn(*mut Handle) -> c_int;
type Destroy = unsafe extern "C" fn(Handle) -> c_int;
type SetEnum = unsafe extern "C" fn(Handle, c_int) -> c_int;
/// `aoclsparse_set_lu_smoother_hint(mat, trans, descr, expected_no_of_calls)`.
type Hint = unsafe extern "C" fn(Handle, c_int, Handle, i64) -> c_int;
type Optimize = unsafe extern "C" fn(Handle) -> c_int;
/// `aoclsparse_itsol_option_set(handle, option, value)`.
type OptionSet = unsafe extern "C" fn(Handle, *const c_char, *const c_char) -> c_int;
type ItsolDestroy = unsafe extern "C" fn(*mut Handle);
/// `monit(n, x, r, rinfo, udata)`: nonzero to stop.
type Monit = unsafe extern "C" fn(i64, *const c64, *const c64, *mut f64, *mut c_void) -> i64;
/// `precond(flag, n, u, v, udata)`, never given.
type Precond = unsafe extern "C" fn(i64, i64, *const c64, *mut c64, *mut c_void) -> i64;
/// `aoclsparse_itsol_z_solve(handle, n, mat, descr, b, x, rinfo, precond, monit, udata)`.
type Solve = unsafe extern "C" fn(
    Handle,
    i64,
    Handle,
    Handle,
    *const c64,
    *mut c64,
    *mut f64,
    Option<Precond>,
    Option<Monit>,
    *mut c_void,
) -> c_int;
type Version = unsafe extern "C" fn() -> *const c_char;

// aoclsparse_types.h
const SUCCESS: c_int = 0;
const MAXIT: c_int = 7;
const BASE_ZERO: c_int = 0;
const GENERAL: c_int = 0;
const OPERATION_NONE: c_int = 111;

/// An `aoclsparse_status`, in words.
fn status(code: c_int) -> &'static str {
    match code {
        0 => "success",
        1 => "not implemented",
        2 => "an invalid pointer",
        3 => "an invalid size",
        4 => "an internal error",
        5 => "an invalid value",
        6 => "an invalid index",
        7 => "the iteration limit was reached",
        8 => "stopped by the caller",
        9 => "the wrong type",
        10 => "memory couldn't be had",
        11 => "a numerical error",
        12 => "an invalid operation",
        13 => "unsorted input",
        14 => "the kernel asked for isn't there",
        _ => "a status its header doesn't name",
    }
}

fn refused(what: &str, code: c_int) -> photonoxide::Error {
    error(format!(
        "AOCL-Sparse's {what}: {} (status {code})",
        status(code)
    ))
}

/// The library's integers.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Width {
    /// 32-bit (LP64).
    Narrow,
    /// 64-bit (ILP64).
    Wide,
}

/// Indices as the library's integers, in 8-byte words (so aligned for either).
fn integers(width: Width, values: impl ExactSizeIterator<Item = usize>) -> Result<Vec<u64>> {
    match width {
        Width::Wide => Ok(values.map(|v| v as u64).collect()),
        Width::Narrow => {
            let mut words = vec![0u64; values.len().div_ceil(2)];
            for (k, v) in values.enumerate() {
                let v = u32::try_from(v)
                    .ok()
                    .filter(|v| *v <= i32::MAX as u32)
                    .ok_or_else(|| {
                        error(format!(
                            "this AOCL-Sparse's integers are 32-bit: {v} is too large"
                        ))
                    })?;
                // the low half first, as the library reads them on the little-endian
                // machines it runs on
                words[k / 2] |= u64::from(v) << (32 * (k % 2));
            }
            Ok(words)
        }
    }
}

struct Api {
    create: CreateZcsr,
    destroy: DestroyAt,
    make_descr: Make,
    destroy_descr: Destroy,
    set_type: SetEnum,
    set_base: SetEnum,
    hint: Option<Hint>,
    optimize: Option<Optimize>,
    init: Make,
    option: OptionSet,
    solve: Solve,
    end: ItsolDestroy,
    version: String,
    width: Width,
    // last: the functions above must not outlive it
    _library: Library,
}

/// One call into AOCL-Sparse's solver at a time.
static CALLS: Mutex<()> = Mutex::new(());

/// What a solve's monitoring callback is given.
struct Watch {
    history: Vec<f64>,
}

/// `monit`: the residual after this iteration, relative to the right-hand side's norm.
unsafe extern "C" fn monit(
    _n: i64,
    _x: *const c64,
    _r: *const c64,
    rinfo: *mut f64,
    udata: *mut c_void,
) -> i64 {
    if !rinfo.is_null() && !udata.is_null() {
        // SAFETY: rinfo is the solver's array of 100 doubles; udata is the Watch the solve
        // below passes, alive and not otherwise used while the solver runs
        unsafe {
            let (residual, norm) = (*rinfo, *rinfo.add(1));
            let watch = &mut *udata.cast::<Watch>();
            watch.history.push(if norm > 0.0 {
                residual / norm
            } else {
                residual
            });
        }
    }
    0
}

impl Api {
    /// A 1 × 1 matrix asked for with these integers: whether the library takes them.
    fn takes(&self, create: CreateZcsr, destroy: DestroyAt, width: Width) -> bool {
        let Ok(mut starts) = integers(width, [0usize, 1].into_iter()) else {
            return false;
        };
        // (padded: a 64-bit build reading 32-bit starts reads two words)
        starts.resize(2, 0);
        let mut columns = vec![0u64; 1];
        let mut value = [c64::new(2.0, 1.0)];
        let mut matrix: Handle = std::ptr::null_mut();
        // SAFETY: create_zcsr's arguments; the arrays hold M + 1 starts and nnz columns of
        // either width (16 and 8 bytes), and outlive the matrix, destroyed here
        unsafe {
            let code = create(
                &mut matrix,
                BASE_ZERO,
                1,
                1,
                1,
                starts.as_mut_ptr().cast(),
                columns.as_mut_ptr().cast(),
                value.as_mut_ptr(),
            );
            if !matrix.is_null() {
                destroy(&mut matrix);
            }
            code == SUCCESS
        }
    }
}

fn open(library: Library) -> Result<Api> {
    // SAFETY: each type is the function's declaration in AOCL-Sparse 5.3.2's headers, its
    // integers passed by value as 64-bit ones (see the module's notes)
    let mut api = unsafe {
        Api {
            create: library.function("aoclsparse_create_zcsr")?,
            destroy: library.function("aoclsparse_destroy")?,
            make_descr: library.function("aoclsparse_create_mat_descr")?,
            destroy_descr: library.function("aoclsparse_destroy_mat_descr")?,
            set_type: library.function("aoclsparse_set_mat_type")?,
            set_base: library.function("aoclsparse_set_mat_index_base")?,
            hint: library.function("aoclsparse_set_lu_smoother_hint").ok(),
            optimize: library.function("aoclsparse_optimize").ok(),
            init: library.function("aoclsparse_itsol_z_init")?,
            option: library.function("aoclsparse_itsol_option_set")?,
            solve: library.function("aoclsparse_itsol_z_solve")?,
            end: library.function("aoclsparse_itsol_destroy")?,
            version: String::new(),
            width: Width::Narrow,
            _library: library,
        }
    };
    // SAFETY: `const char *aoclsparse_get_version()`, a static string
    let version = unsafe { api._library.function::<Version>("aoclsparse_get_version") }
        .ok()
        .map(|f| {
            // SAFETY: as above
            let text = unsafe { f() };
            if text.is_null() {
                String::new()
            } else {
                // SAFETY: a C string that lives as long as the library
                unsafe { CStr::from_ptr(text) }
                    .to_string_lossy()
                    .into_owned()
            }
        })
        .unwrap_or_default();
    api.version = version_of(&version);
    api.width = if api.takes(api.create, api.destroy, Width::Narrow) {
        Width::Narrow
    } else if api.takes(api.create, api.destroy, Width::Wide) {
        Width::Wide
    } else {
        return Err(error(
            "AOCL-Sparse made no 1 × 1 matrix with 32-bit integers nor with 64-bit ones",
        ));
    };
    Ok(api)
}

/// The version in what the library says of itself ("AOCL-Sparse 5.3.2 Build 20260820").
fn version_of(text: &str) -> String {
    text.split_whitespace()
        .find(|word| word.starts_with(|c: char| c.is_ascii_digit()) && word.contains('.'))
        .unwrap_or("unknown")
        .to_owned()
}

/// AOCL-Sparse's GMRES, as photonoxide's registry has it.
pub struct AoclSparse {
    api: Api,
}

impl AoclSparse {
    /// AOCL-Sparse found and loaded, with its complex iterative solver.
    ///
    /// # Errors
    ///
    /// Why not: not found, not loadable (its BLAS, LAPACK and AOCL-Utils must load with it),
    /// or without the solver.
    pub fn load() -> std::result::Result<AoclSparse, String> {
        let (discovery, library) = load(&AOCL_SPARSE, None, |l| {
            // SAFETY: only resolved; the type is the solver's
            unsafe { l.function::<Solve>("aoclsparse_itsol_z_solve") }.map(drop)
        });
        let library = library.ok_or_else(|| discovery.reason())?;
        let api = open(library).map_err(|e| e.to_string())?;
        Ok(AoclSparse { api })
    }

    /// Its version.
    pub fn version(&self) -> &str {
        &self.api.version
    }

    /// Whether this build's integers are 64-bit (ILP64).
    pub fn wide_integers(&self) -> bool {
        self.api.width == Width::Wide
    }

    fn set(&self, handle: Handle, option: &str, value: &str) -> Result<()> {
        let (name, text) = (
            CString::new(option).map_err(|_| error("an option's name with a NUL"))?,
            CString::new(value).map_err(|_| error("an option's value with a NUL"))?,
        );
        // SAFETY: a handle made by aoclsparse_itsol_z_init and two C strings
        let code = unsafe { (self.api.option)(handle, name.as_ptr(), text.as_ptr()) };
        if code == SUCCESS {
            Ok(())
        } else {
            Err(refused(&format!("option {option:?} = {value:?}"), code))
        }
    }
}

/// AOCL-Sparse: where it was found, its version and its integers.
pub fn probe() -> Probe {
    let (discovery, library) = load(&AOCL_SPARSE, None, |l| {
        // SAFETY: only resolved; the type is the solver's
        unsafe { l.function::<Solve>("aoclsparse_itsol_z_solve") }.map(drop)
    });
    let opened = library.and_then(|l| open(l).ok());
    Probe {
        discovery,
        version: opened.as_ref().map(|api| api.version.clone()),
        details: opened
            .map(|api| {
                vec![match api.width {
                    Width::Narrow => "32-bit integers (LP64)".to_owned(),
                    Width::Wide => "64-bit integers (ILP64)".to_owned(),
                }]
            })
            .unwrap_or_default(),
    }
}

impl IterativeSolver for AoclSparse {
    fn capabilities(&self) -> Capabilities {
        let mut c = Capabilities::new(
            "aocl-sparse",
            self.api.version.clone(),
            "MIT (installed by the user)",
        );
        c.symmetric = false;
        c.transpose = false;
        c.threads = Threads::Library("OpenMP's (OMP_NUM_THREADS)".into());
        c.deterministic = false;
        c
    }

    fn qmr_run(&self, _: &Matrix<'_>, _: &[c64], _: Stopping) -> Result<QmrRun> {
        Err(error(
            "AOCL-Sparse runs its own GMRES, not photonoxide's QMR",
        ))
    }

    fn own_solver(&self) -> Option<String> {
        Some(format!("AOCL-Sparse's GMRES({RESTART})"))
    }

    fn solve(
        &self,
        matrix: RowMatrix<'_>,
        preconditioner: OwnPreconditioner,
        b: &[c64],
        stopping: Stopping,
    ) -> Result<(Vec<c64>, Convergence)> {
        let n = matrix.rows();
        if matrix.columns() != n || b.len() != n {
            return Err(error(format!(
                "a square matrix and a right-hand side of its size: {n} rows, {} columns, {} \
                 values",
                matrix.columns(),
                b.len()
            )));
        }
        let positive = stopping.tolerance.is_finite() && stopping.tolerance > 0.0;
        if !positive || stopping.max_iterations == 0 {
            return Err(error("a positive tolerance and at least one iteration"));
        }
        let api = &self.api;
        let mut starts = integers(api.width, matrix.starts().iter().copied())?;
        let mut columns = integers(api.width, matrix.indices().iter().copied())?;
        // (at least a word each, so that the pointers are to something)
        starts.resize(starts.len().max(1), 0);
        columns.resize(columns.len().max(1), 0);
        let mut values = matrix.values().to_vec();
        let mut x = vec![c64::new(0.0, 0.0); n];
        let mut rinfo = [0.0f64; 100];
        let mut watch = Watch {
            history: Vec::new(),
        };
        let (mut a, mut descr, mut handle): (Handle, Handle, Handle) = (
            std::ptr::null_mut(),
            std::ptr::null_mut(),
            std::ptr::null_mut(),
        );
        let _one = CALLS.lock().unwrap_or_else(|p| p.into_inner());
        let outcome = (|| -> Result<c_int> {
            // SAFETY: create_zcsr's arguments: n + 1 starts and as many columns and values as
            // the last start, of the library's integers, alive until the matrix is destroyed
            let code = unsafe {
                (api.create)(
                    &mut a,
                    BASE_ZERO,
                    n as i64,
                    n as i64,
                    values.len() as i64,
                    starts.as_mut_ptr().cast(),
                    columns.as_mut_ptr().cast(),
                    values.as_mut_ptr(),
                )
            };
            if code != SUCCESS {
                return Err(refused("matrix", code));
            }
            // SAFETY: a descriptor made, then set by the header's enums
            let code = unsafe { (api.make_descr)(&mut descr) };
            if code != SUCCESS {
                return Err(refused("descriptor", code));
            }
            // SAFETY: as above
            unsafe {
                (api.set_type)(descr, GENERAL);
                (api.set_base)(descr, BASE_ZERO);
            }
            // SAFETY: a handle for complex doubles
            let code = unsafe { (api.init)(&mut handle) };
            if code != SUCCESS {
                return Err(refused("solver", code));
            }
            self.set(handle, "iterative method", "GMRES")?;
            self.set(
                handle,
                "gmres preconditioner",
                match preconditioner {
                    OwnPreconditioner::None => "None",
                    OwnPreconditioner::Ilu0 => "ILU0",
                },
            )?;
            self.set(handle, "gmres restart iterations", &RESTART.to_string())?;
            self.set(
                handle,
                "gmres iteration limit",
                &stopping.max_iterations.min(i32::MAX as usize).to_string(),
            )?;
            self.set(
                handle,
                "gmres rel tolerance",
                &format!("{:e}", stopping.tolerance),
            )?;
            self.set(handle, "gmres abs tolerance", "0")?;
            if preconditioner == OwnPreconditioner::Ilu0
                && let (Some(hint), Some(optimize)) = (api.hint, api.optimize)
            {
                // SAFETY: the matrix and descriptor made above; the hint says an ILU's
                // solves are coming, as the library's examples do before its solver
                unsafe {
                    hint(a, OPERATION_NONE, descr, stopping.max_iterations as i64);
                    optimize(a);
                }
            }
            // SAFETY: the solver's arguments: b and x of n values, rinfo of 100, no
            // preconditioner of ours, the monitor and its Watch, which outlives the call
            Ok(unsafe {
                (api.solve)(
                    handle,
                    n as i64,
                    a,
                    descr,
                    b.as_ptr(),
                    x.as_mut_ptr(),
                    rinfo.as_mut_ptr(),
                    None,
                    Some(monit),
                    std::ptr::from_mut(&mut watch).cast(),
                )
            })
        })();
        // SAFETY: each made above or still null, destroyed once
        unsafe {
            if !handle.is_null() {
                (api.end)(&mut handle);
            }
            if !descr.is_null() {
                (api.destroy_descr)(descr);
            }
            if !a.is_null() {
                (api.destroy)(&mut a);
            }
        }
        let code = outcome?;
        let iterations = rinfo[30].max(0.0) as usize;
        let residual = if rinfo[1] > 0.0 {
            rinfo[0] / rinfo[1]
        } else {
            rinfo[0]
        };
        if code == MAXIT {
            return Err(error(format!(
                "AOCL-Sparse's GMRES didn't reach {:e} in {iterations} iterations: its residual \
                 is {residual:e}",
                stopping.tolerance
            )));
        }
        if code != SUCCESS {
            return Err(refused("GMRES", code));
        }
        Ok((
            x,
            Convergence {
                iterations: iterations.max(watch.history.len()),
                residual,
                history: watch.history,
            },
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn indices_are_laid_out_as_the_librarys_integers() {
        let wide = integers(Width::Wide, [0usize, 3, 7].into_iter()).unwrap();
        assert_eq!(wide, [0, 3, 7]);
        // two 32-bit integers a word, the first in its low half
        let narrow = integers(Width::Narrow, [1usize, 2, 3].into_iter()).unwrap();
        assert_eq!(narrow, [1 | 2 << 32, 3]);
        let bytes: Vec<u8> = narrow.iter().flat_map(|w| w.to_le_bytes()).collect();
        assert_eq!(&bytes[..12], [1, 0, 0, 0, 2, 0, 0, 0, 3, 0, 0, 0]);
        // the starts of a 1 × 1 matrix: neither width's are the other's
        let as_narrow = integers(Width::Narrow, [0usize, 1].into_iter()).unwrap();
        assert_eq!(as_narrow, [1 << 32], "read as 64-bit: a start that isn't 0");
        let as_wide = integers(Width::Wide, [0usize, 1].into_iter()).unwrap();
        assert_eq!(as_wide[0] as u32, 0);
        assert_eq!(
            (as_wide[0] >> 32) as u32,
            0,
            "read as 32-bit: two starts of 0"
        );
        assert!(integers(Width::Narrow, [1usize << 31].into_iter()).is_err());
    }

    #[test]
    fn a_status_and_a_version_are_read() {
        assert_eq!(status(7), "the iteration limit was reached");
        assert!(
            refused("GMRES", 11)
                .to_string()
                .contains("a numerical error")
        );
        assert_eq!(version_of("AOCL-Sparse 5.3.2 Build 20260820"), "5.3.2");
        assert_eq!(version_of(""), "unknown");
    }

    #[test]
    fn it_says_why_where_it_isnt() {
        if let Err(reason) = AoclSparse::load() {
            assert!(!reason.is_empty());
        }
    }
}
