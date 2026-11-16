//! A lazy, device-owned tile stream. Creation validates and hashes the whole
//! input once; next computes only the current tile, with bounded pair scratch.
use super::{
    cancel::Call,
    device_lifetime::Detached,
    device_run::{self, Prepared},
    dispatch, error, parse_grain, resolve_device, PySeqTable, RuntimeKernel,
};
use pyo3::{prelude::*, types::PyDict};
use seqblaze_core::{
    cancel::Stage,
    detail::dense_request_bytes,
    exec::{ExecOptions, Strategy},
    kernel::PairDistanceKernel,
    sink::Sink,
    validate::EngineError,
};
use seqblaze_cuda::{
    resident_run::ResidentRun,
    resident_tiles::{TileDomain, TilePlan},
};

struct State {
    prepared: Prepared,
    _stream_owner: Option<Py<PyAny>>,
    plan: Detached<TilePlan>,
}
#[pyclass(module = "seqblaze._core", name = "_DeviceTileStream")]
pub(super) struct PyDeviceTileStream {
    state: Option<Detached<State>>,
    kernel: RuntimeKernel,
    shape: [usize; 2],
    tile: [usize; 2],
    row: usize,
    col: usize,
    full: usize,
    mirrored: usize,
    evaluations: usize,
    yielded: usize,
    complete: bool,
    closed: bool,
    n_illegal: usize,
    shared: bool,
    strategy: String,
    threads: usize,
    detail_bytes: usize,
    input_digests: Py<PyDict>,
    /// The public kernel name when progress lines are on (cancellation-progress.md §4.1).
    progress: Option<String>,
}
/// Position and counters before a `next()` step (restored when it fails).
#[derive(Clone, Copy)]
struct Mark {
    row: usize,
    col: usize,
    full: usize,
    mirrored: usize,
    evaluations: usize,
    yielded: usize,
    complete: bool,
}
struct Tile {
    output: ResidentRun,
    rows: (usize, usize),
    cols: (usize, usize),
}
fn increment(a: usize, b: usize) -> Result<usize, EngineError> {
    a.checked_add(b)
        .ok_or_else(|| EngineError::Overflow("tile execution count exceeds platform limits".into()))
}
impl PyDeviceTileStream {
    fn compute(&mut self) -> Result<Option<Tile>, EngineError> {
        let Some(state) = self.state.as_ref() else {
            return Ok(None);
        };
        if self.closed || self.complete {
            return Ok(None);
        }
        state.prepared.lease.execution().check_thread()?;
        let rows = self.tile[0].min(self.shape[0] - self.row);
        let cols = self.tile[1].min(self.shape[1] - self.col);
        let cells = rows.checked_mul(cols).ok_or_else(|| {
            EngineError::Overflow("tile shape product exceeds platform limit".into())
        })?;
        cells.checked_mul(5 + self.detail_bytes).ok_or_else(|| {
            EngineError::Overflow("tile byte count exceeds platform limit".into())
        })?;
        let key_width = if matches!(
            self.kernel,
            RuntimeKernel::DuplexPf(_)
                | RuntimeKernel::DuplexPrimer3(_)
                | RuntimeKernel::TmPairs(_)
        ) {
            8
        } else {
            4
        };
        if cells
            .checked_mul(key_width)
            .is_none_or(|bytes| bytes > isize::MAX as usize)
        {
            return Err(EngineError::Overflow(
                "tile distance buffer exceeds addressable allocation size".into(),
            ));
        }
        let p = &state.prepared;
        let b = p.rows_b.as_ref().unwrap_or(&p.rows_a);
        let (_, fields, binary) = device_run::requested(&self.kernel);
        let region = [self.row, self.col, rows, cols];
        let grid = self.shape[0]
            .div_ceil(self.tile[0])
            .saturating_mul(self.shape[1].div_ceil(self.tile[1]));
        let stage = Stage::Tile {
            index: self.yielded.saturating_add(1),
            of: grid,
        };
        let (output, counts) = seqblaze_cuda::resident_run::run_tile(
            &p.lease,
            &p.rows_a,
            b,
            &p.original,
            &state.plan,
            region,
            p.call.clone(),
            &fields,
            p.pf,
            binary,
            stage,
        )?;
        // A handler's exception stored by a poll in the tile's trailing waits
        // (after its last boundary) fails this step before anything is counted.
        seqblaze_core::cancel::check()?;
        let next_full = increment(self.full, counts.full)?;
        let next_mirrored = increment(self.mirrored, counts.mirrored)?;
        let next_evaluations = increment(self.evaluations, counts.evaluations)?;
        let next_yielded = increment(self.yielded, 1)?;
        self.full = next_full;
        self.mirrored = next_mirrored;
        self.evaluations = next_evaluations;
        self.yielded = next_yielded;
        let tile = Tile {
            output,
            rows: (self.row, self.row + rows),
            cols: (self.col, self.col + cols),
        };
        if self.col + cols == self.shape[1] {
            self.col = 0;
            self.row += rows;
        } else {
            self.col += cols;
        }
        self.complete = self.row == self.shape[0];
        Ok(Some(tile))
    }
    fn retire(&mut self) {
        self.state = None;
    }
    fn mark(&self) -> Mark {
        Mark {
            row: self.row,
            col: self.col,
            full: self.full,
            mirrored: self.mirrored,
            evaluations: self.evaluations,
            yielded: self.yielded,
            complete: self.complete,
        }
    }
    /// A failed step is terminal and counts nothing (cancellation-progress.md
    /// §2 rule 5): position and counters go back to `mark`, then the stream closes.
    fn fail_step(&mut self, mark: Mark) {
        self.row = mark.row;
        self.col = mark.col;
        self.full = mark.full;
        self.mirrored = mark.mirrored;
        self.evaluations = mark.evaluations;
        self.yielded = mark.yielded;
        self.complete = mark.complete;
        self.closed = true;
        self.retire();
    }
}
#[pymethods]
impl PyDeviceTileStream {
    fn __iter__(slf: PyRef<'_, Self>) -> PyRef<'_, Self> {
        slf
    }
    fn __next__<'py>(&mut self, py: Python<'py>) -> PyResult<Option<Bound<'py, PyDict>>> {
        if let Err(error) = py.check_signals() {
            self.closed = true;
            self.retire();
            return Err(error);
        }
        // One cancel scope per next() (cancellation-progress.md §3.7): its polls
        // add to the existing signal checks before and after the compute.
        let call = Call::new(self.progress.as_deref());
        let mark = self.mark();
        let result = call.detach(py, || {
            let tile = self.compute()?;
            // Testing builds: a refused trace reservation is terminal (F4 round 3).
            seqblaze_cuda::interop::execution_stream::resident_launch_streams_check()?;
            Ok(tile)
        });
        // A stored handler exception, or a signal still pending after the
        // compute, is terminal like any failing next() (D86 (6)), and the
        // step's tile is neither returned nor counted (§2 rule 5).
        let raised = call.exception().or_else(|| match &result {
            Ok(Some(_)) => py.check_signals().err(),
            _ => None,
        });
        // Retire source/producer Python owners only after leaving native E0.
        if let Some(raised) = raised {
            let computed = result.is_ok();
            drop(result);
            if computed {
                self.fail_step(mark);
            } else {
                self.closed = true;
                self.retire();
            }
            return Err(raised);
        }
        if result.is_err() {
            self.closed = true;
            self.retire();
        }
        let tile = result.map_err(|e| error(py, e))?.map(Detached::new);
        if tile.is_some() {
            if let Some(state) = self.state.as_ref() {
                py.detach(|| state.prepared.lease.execution().publish_packs());
            }
        }
        if self.complete {
            self.retire();
        }
        let Some(mut tile) = tile else {
            return Ok(None);
        };
        let (primary, fields, _) = device_run::requested(&self.kernel);
        let result = (|| {
            let columns = device_run::publish_columns(
                py,
                &self.kernel,
                Sink::Matrix,
                primary,
                &fields,
                std::mem::take(&mut tile.output.columns),
                std::mem::take(&mut tile.output.ragged),
            )?;
            let value = PyDict::new(py);
            value.set_item("columns", columns)?;
            value.set_item("shape", (tile.output.shape[0], tile.output.shape[1]))?;
            value.set_item("rows", tile.rows)?;
            value.set_item("cols", tile.cols)?;
            Ok(Some(value))
        })();
        if result.is_err() {
            // A tile that cannot be published is not yielded either.
            self.fail_step(mark);
        }
        result
    }
    fn close(&mut self) {
        self.closed = true;
        self.retire();
    }
    #[getter]
    fn shape(&self) -> (usize, usize) {
        (self.shape[0], self.shape[1])
    }
    #[getter]
    fn input_digests<'py>(&self, py: Python<'py>) -> Bound<'py, PyDict> {
        self.input_digests.bind(py).clone()
    }
    #[getter]
    fn report<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let result = PyDict::new(py);
        let strategy = PyDict::new(py);
        strategy.set_item("requested", &self.strategy)?;
        strategy.set_item("effective", &self.strategy)?;
        strategy.set_item("fallback_reason", py.None())?;
        result.set_item("strategy", strategy)?;
        let counts = PyDict::new(py);
        counts.set_item("certified", 0)?;
        counts.set_item("pruned", 0)?;
        counts.set_item("full", self.full)?;
        result.set_item("counts", counts)?;
        result.set_item("threads", self.threads)?;
        result.set_item("host_pairs", 0)?;
        let flags = PyDict::new(py);
        flags.set_item("n_illegal_seqs", self.n_illegal)?;
        result.set_item("flags", flags)?;
        let scans = PyDict::new(py);
        scans.set_item("a", 1)?;
        scans.set_item("b", usize::from(!self.shared))?;
        result.set_item("validation_scans", scans)?;
        let tiles = PyDict::new(py);
        tiles.set_item("evaluations", self.evaluations)?;
        tiles.set_item("mirrored", self.mirrored)?;
        tiles.set_item("yielded", self.yielded)?;
        tiles.set_item("complete", self.complete)?;
        tiles.set_item("closed", self.closed)?;
        result.set_item("tiles", tiles)?;
        Ok(result)
    }
}
#[pyfunction(signature=(a,b,kind,i,j,self_pairs,check_bases,strategy,threads,tile_shape,kernel="hamming",grain="auto",device=None,mode="global",details=Vec::new(),orientation=None,align=None,duplex=None,stream=1,stream_owner=None,progress=None))]
#[allow(clippy::too_many_arguments)]
fn _native_resident_tiles<'py>(
    py: Python<'py>,
    a: PyRef<'py, PySeqTable>,
    b: PyRef<'py, PySeqTable>,
    kind: &str,
    i: Option<Bound<'py, PyAny>>,
    j: Option<Bound<'py, PyAny>>,
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
    align: Option<Bound<'py, PyDict>>,
    duplex: Option<Bound<'py, PyAny>>,
    stream: u64,
    stream_owner: Option<Py<PyAny>>,
    progress: Option<String>,
) -> PyResult<PyDeviceTileStream> {
    // Creation's cancel scope (cancellation-progress.md §3.7).
    let scope = Call::new(progress.as_deref());
    #[cfg(feature = "testing")]
    seqblaze_cuda::interop::execution_stream::resident_launch_streams_reset();
    let kernel = RuntimeKernel::configure(
        kernel,
        mode,
        &details,
        check_bases,
        align.as_ref(),
        duplex.as_ref(),
    )
    .map_err(|e| error(py, e))?;
    let grain = parse_grain(py, grain)?;
    kernel.check_backend(device).map_err(|e| error(py, e))?;
    let device = resolve_device(py, device)?.ok_or_else(|| {
        error(
            py,
            EngineError::Device("resident tiles require a CUDA device".into()),
        )
    })?;
    if tile_shape.0 == 0 || tile_shape.1 == 0 || threads == 0 {
        return Err(error(
            py,
            EngineError::Parameter("tile dimensions and threads must be positive".into()),
        ));
    }
    let detail_bytes = dispatch!(&kernel, |k| dense_request_bytes(
        k.detail_fields(),
        k.detail_request()
    ))
    .map_err(|e| error(py, e))?;
    let options = ExecOptions {
        threads,
        grain,
        sink: Sink::Matrix,
        cutoff: None,
        k: None,
        check_bases,
        strategy: if strategy == "full" {
            Strategy::Full
        } else {
            Strategy::Bounded
        },
    };
    let mut prepared = device_run::prepare(
        py,
        &kernel,
        &a,
        &b,
        kind,
        i.as_ref(),
        j.as_ref(),
        self_pairs,
        orientation,
        &options,
        &device,
        stream,
        &scope,
        false,
    )?;
    let domain = match kind {
        "zip" => TileDomain::Zip,
        "index" => TileDomain::Explicit,
        "within" => match orientation {
            Some("both") => TileDomain::WithinBoth { self_pairs },
            Some("as_given") => TileDomain::WithinAsGiven { self_pairs },
            _ => TileDomain::SymmetricWithin { self_pairs },
        },
        _ => TileDomain::Cross,
    };
    let tile = [tile_shape.0, tile_shape.1];
    let plan = scope
        .region(py, || {
            TilePlan::new(&mut prepared.lease, &prepared.original, domain, tile)
        })
        .map(Detached::new)?;
    if let Some(raised) = scope.exception() {
        return Err(raised);
    }
    let shape = [prepared.original.n_a, prepared.original.n_b];
    let complete = shape[0] == 0 || shape[1] == 0;
    let input_digests = prepared.input_digests.clone_ref(py);
    let n_illegal = prepared.n_illegal;
    let shared = prepared.shared;
    device_run::trace_check(py)?;
    let state = if complete {
        None
    } else {
        Some(Detached::new(State {
            prepared,
            plan,
            _stream_owner: stream_owner,
        }))
    };
    Ok(PyDeviceTileStream {
        state,
        kernel,
        shape,
        tile,
        row: 0,
        col: 0,
        full: 0,
        mirrored: 0,
        evaluations: 0,
        yielded: 0,
        complete,
        closed: false,
        n_illegal,
        shared,
        strategy: strategy.into(),
        threads,
        detail_bytes,
        input_digests,
        progress,
    })
}
pub(super) fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_class::<PyDeviceTileStream>()?;
    module.add_function(wrap_pyfunction!(_native_resident_tiles, module)?)?;
    Ok(())
}
