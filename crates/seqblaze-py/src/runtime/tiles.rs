//! Lazy tile bridge (CPU or CUDA); returned columns independently own their storage.
use super::{
    cancel, column, dispatch, dispatch_i32, error, oriented_work_source, parse_grain,
    resolve_device, set_primary_and_details, Erase, Keys, Numeric, PyIndexColumn, PySeqTable,
    RuntimeKernel,
};
use pyo3::{prelude::*, types::PyDict};
use seqblaze_core::{
    detail::DetailColumns,
    exec::Strategy,
    sched::Workers,
    sink::tiles::{TileOutput, TileStream},
    validate::EngineError,
};
use std::sync::Arc;

/// A tile with its keys erased to the runtime's key column.
struct Tile {
    keys: Keys,
    details: Option<DetailColumns>,
    passed: Option<Vec<u8>>,
    shape: (usize, usize),
    rows: (usize, usize),
    cols: (usize, usize),
}

fn erase<S: Erase>(tile: Option<TileOutput<S>>) -> Option<Tile> {
    tile.map(|tile| Tile {
        keys: S::keys(tile.columns.dist),
        details: tile.columns.details,
        passed: tile.columns.passed,
        shape: tile.shape,
        rows: tile.rows,
        cols: tile.cols,
    })
}

#[pyclass(module = "seqblaze._core", name = "_TileStream")]
pub struct PyTileStream {
    stream: TileStream,
    kernel: RuntimeKernel,
    device: Option<seqblaze_cuda::Device>,
    /// Device tables for this iterator: uploaded at the first `next()` of a
    /// pair space with work (never for n_pairs == 0), freed on completion,
    /// close, failure or drop.
    session: Option<seqblaze_cuda::TileSession>,
    /// Host workers of the stream (routed pairs on the device path).
    workers: Workers,
    /// Distinct work items routed to the host so far (device path only).
    host_pairs: usize,
    /// The public kernel name when progress lines are on, captured at creation.
    progress: Option<String>,
}

impl PyTileStream {
    /// Close after a failed step, with its position and counters rewound.
    fn fail_step(&mut self, mark: seqblaze_core::sink::tiles::TileMark) {
        self.stream.fail_step(mark);
        self.session = None;
    }
    fn next_tile(&mut self) -> Result<Option<Tile>, EngineError> {
        let kernel = &self.kernel;
        let result = match &self.device {
            Some(device) => {
                // The stream's validated per-call tables (None once complete).
                let inputs = self
                    .stream
                    .inputs()
                    .filter(|(_, _, work)| work.n_pairs > 0)
                    .map(|(a, b, _)| (Arc::clone(a), Arc::clone(b)));
                // A zero-work pair space (n_pairs == 0) never creates a session:
                // no upload, allocation or launch; the host path scores nothing.
                if let (None, Some((a, b))) = (&self.session, &inputs) {
                    // The device pack is built once, for the session only; `_native_tiles`
                    // rejects a device for a kernel the backend lacks.
                    let call = kernel.device_call()?;
                    match seqblaze_cuda::TileSession::new(device, call, (&**a, &**b), self.workers)
                    {
                        Ok(session) => self.session = Some(session),
                        Err(failure) => {
                            // Inside the next() guard: retire, drop after it.
                            self.stream.retire();
                            return Err(failure);
                        }
                    }
                }
                let stream = &mut self.stream;
                match (self.session.as_mut(), inputs) {
                    (Some(session), Some((a, b))) => {
                        let result = match kernel {
                            // K2's int64 keys (primer3-cuda.md §7): the int64-key port it
                            // shares with `tm` on Pairs.
                            RuntimeKernel::DuplexPrimer3(k) => {
                                let k: &seqblaze_thermo::DimerPrimer3 = k;
                                stream
                                    .next_tile_batch(k, &mut session.bind_i64(k, (&*a, &*b)))
                                    .map(erase)
                            }
                            // K3: the 64-bit-key port (duplex-pf-cuda.md §7.2).
                            RuntimeKernel::DuplexPf(k) => {
                                let k: &seqblaze_thermo::DuplexPf = k;
                                stream
                                    .next_tile_batch(k, &mut session.bind_pf(k, (&*a, &*b)))
                                    .map(erase)
                            }
                            // tm on Pairs: the int64-key port (tm-pairs-cuda.md §6, §7).
                            RuntimeKernel::TmPairs(k) => {
                                let k: &seqblaze_thermo::TmPairs = k;
                                stream
                                    .next_tile_batch(k, &mut session.bind_i64(k, (&*a, &*b)))
                                    .map(erase)
                            }
                            _ => dispatch_i32!(
                                kernel,
                                |k| stream
                                    .next_tile_batch(k, &mut session.bind(k, (&*a, &*b)))
                                    .map(erase),
                                Err(kernel.cuda_unavailable())
                            ),
                        };
                        self.host_pairs = session.host_pairs();
                        result
                    }
                    _ => dispatch!(kernel, |k| stream.next_tile(k).map(erase)),
                }
            }
            None => {
                let stream = &mut self.stream;
                dispatch!(kernel, |k| stream.next_tile(k).map(erase))
            }
        };
        if result.is_err() || self.stream.complete || self.stream.closed {
            self.session = None;
        }
        result
    }
}

#[pymethods]
impl PyTileStream {
    fn __iter__(slf: PyRef<'_, Self>) -> PyRef<'_, Self> {
        slf
    }
    fn __next__<'py>(&mut self, py: Python<'py>) -> PyResult<Option<Bound<'py, PyDict>>> {
        // Each next() is a call of its own with its own cancel scope
        // (cancellation-progress.md §3.7); E0 for each next(), never between
        // yields (§7.3 item 3).
        let call = cancel::Call::new(self.progress.as_deref());
        let mark = self.stream.mark();
        let tile = call.detach(py, || self.next_tile());
        // A stream that completed or failed released its tables only now,
        // outside the guard: a producer's release callback runs in the
        // caller's FP environment (§7.3).
        self.stream.drop_retired();
        // A handler's exception stored during the step, or a signal still
        // pending after its last poll, is a next() failure (D86 (6)): terminal,
        // whatever the computation returned, and the step's tile is neither
        // returned nor counted (cancellation-progress.md §2 rule 5).
        let raised = call.exception().or_else(|| match &tile {
            Ok(Some(_)) => py.check_signals().err(),
            _ => None,
        });
        if let Some(raised) = raised {
            if tile.is_ok() {
                self.fail_step(mark);
            }
            return Err(raised);
        }
        let tile = tile.map_err(|e| error(py, e))?;
        let Some(tile) = tile else {
            return Ok(None);
        };
        let result = (|| {
            let columns = PyDict::new(py);
            set_primary_and_details(py, &columns, &self.kernel, tile.keys, tile.details)?;
            columns.set_item("passed", column(py, Numeric::Bool(tile.passed.unwrap()))?)?;
            let result = PyDict::new(py);
            result.set_item("columns", columns)?;
            result.set_item("shape", tile.shape)?;
            result.set_item("rows", tile.rows)?;
            result.set_item("cols", tile.cols)?;
            Ok(Some(result))
        })();
        if result.is_err() {
            // A tile that cannot be published is not yielded either.
            self.fail_step(mark);
        }
        result
    }
    fn close(&mut self) {
        self.stream.close();
        self.session = None;
    }
    #[getter]
    fn shape(&self) -> (usize, usize) {
        self.stream.shape
    }
    #[getter]
    fn report<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let stream = &self.stream;
        let result = PyDict::new(py);
        let strategy = if stream.strategy == Strategy::Full {
            "full"
        } else {
            "bounded"
        };
        let info = PyDict::new(py);
        info.set_item("requested", strategy)?;
        info.set_item("effective", strategy)?;
        info.set_item("fallback_reason", py.None())?;
        result.set_item("strategy", info)?;
        let counts = PyDict::new(py);
        counts.set_item("certified", stream.counts.certified)?;
        counts.set_item("pruned", stream.counts.pruned)?;
        counts.set_item("full", stream.counts.full)?;
        result.set_item("counts", counts)?;
        result.set_item("threads", stream.threads)?;
        let flags = PyDict::new(py);
        flags.set_item("n_illegal_seqs", stream.validation.n_illegal_seqs)?;
        result.set_item("flags", flags)?;
        let scans = PyDict::new(py);
        scans.set_item("a", stream.validation.scans[0])?;
        scans.set_item("b", stream.validation.scans[1])?;
        result.set_item("validation_scans", scans)?;
        let tiles = PyDict::new(py);
        tiles.set_item("evaluations", stream.evaluations)?;
        tiles.set_item("mirrored", stream.replayed)?;
        tiles.set_item("yielded", stream.yielded)?;
        tiles.set_item("complete", stream.complete)?;
        tiles.set_item("closed", stream.closed)?;
        result.set_item("tiles", tiles)?;
        if self.device.is_some() {
            result.set_item("host_pairs", self.host_pairs)?;
        }
        Ok(result)
    }
}

/// Trailing arguments as `_native_run`: `kernel`, `grain`, `device`, `mode`,
/// `details`, `orientation`, `align`, `duplex`.
#[pyfunction(signature=(a,b,kind,i,j,self_pairs,check_bases,strategy,threads,tile_shape,kernel="hamming",grain="auto",device=None,mode="global",details=Vec::new(),orientation=None,align=None,duplex=None,progress=None))]
#[allow(clippy::too_many_arguments)]
pub fn _native_tiles(
    py: Python<'_>,
    a: PyRef<'_, PySeqTable>,
    b: PyRef<'_, PySeqTable>,
    kind: &str,
    i: Option<PyRef<'_, PyIndexColumn>>,
    j: Option<PyRef<'_, PyIndexColumn>>,
    self_pairs: bool,
    check_bases: bool,
    strategy: &str,
    threads: usize,
    tile_shape: (usize, usize),
    kernel: &str,
    grain: &str,
    device: Option<u64>,
    mode: &str,
    details: Vec<String>,
    orientation: Option<&str>,
    align: Option<Bound<'_, PyDict>>,
    duplex: Option<Bound<'_, PyAny>>,
    progress: Option<&str>,
) -> PyResult<PyTileStream> {
    let kernel = RuntimeKernel::configure(
        kernel,
        mode,
        &details,
        check_bases,
        align.as_ref(),
        duplex.as_ref(),
    )
    .map_err(|e| error(py, e))?;
    let workers = Workers {
        limit: threads,
        grain: parse_grain(py, grain)?,
    };
    kernel.check_backend(device).map_err(|e| error(py, e))?;
    // Device resolution precedes pair and sequence validation (cuda-backend.md).
    let device = resolve_device(py, device)?;
    let work = oriented_work_source(
        py,
        kind,
        i,
        j,
        self_pairs,
        orientation,
        a.table.len(),
        b.table.len(),
    )?;
    let (a, b) = (
        a.table.host().map_err(|e| error(py, e))?,
        b.table.host().map_err(|e| error(py, e))?,
    );
    let strategy = if strategy == "full" {
        Strategy::Full
    } else {
        Strategy::Bounded
    };
    // Creation is a call of its own (whole-input preflight; §3.7).
    let call = cancel::Call::new(progress);
    let created = call.detach(py, || {
        dispatch!(&kernel, |k| TileStream::with_workers(
            k,
            a,
            b,
            work,
            tile_shape,
            workers,
            strategy,
            check_bases,
        ))
    });
    let stream = call.finish(py, created)?;
    Ok(PyTileStream {
        stream,
        kernel,
        device,
        session: None,
        workers,
        host_pairs: 0,
        progress: progress.map(String::from),
    })
}

pub fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_class::<PyTileStream>()?;
    module.add_function(wrap_pyfunction!(_native_tiles, module)?)?;
    Ok(())
}
