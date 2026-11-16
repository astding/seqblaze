//! `tm` on a Pairs input on the device (docs/wiki/gpu/tm-pairs-cuda.md): the register
//! stage (`sb_reg` in kernels/register.cuh, one thread per pair) and the int64-key path
//! of the CUDA backend (§7): the Each, Hits and RowStats sinks on int64 keys, the
//! batch-distance port on int64 keys for TopK, Matrix and Tiles, and [`run`].
//!
//! The sink code below reads keys from the session's int64 key buffer and is
//! kernel-agnostic; the one stage it calls is the register stage, and the detail
//! fields come from the register state ([`Reg`]).
//!
//! A stage computes the key of every item of a chunk and, in fields mode, every
//! requested field; Each with a cutoff also writes `passed` and masks the failing rows
//! in the kernel (QTP3). Every requested output of every item is written. Nothing is
//! routed: `host_pairs` stays 0 (§9).
use std::sync::Arc;

use cudarc::driver::{CudaSlice, CudaStream, DeviceRepr, PushKernelArg};
use seqblaze_core::{
    detail::{
        try_reserve, try_reserve_exact, DetailColumns, DetailData, DetailField, DetailRequest,
    },
    exec::ExecOptions,
    kernel::{BatchDistance, CellClass, DistanceVisit, PairDistanceKernel, Score},
    seq::SeqTable,
    sink::{matrix, topk, try_filled, Columns, Counts, RowState, RowStatsColumns, RunOutput, Sink},
    validate::{EngineError, ValidationReport},
    work::{Orientation, PairKind, WorkSource},
};

use crate::{
    device::{guard, Device},
    plan::{reg_chunk, Request},
    session::{
        dev_alloc, failure_ordinal, ptr, Chunk, Host, SbRows, SbSpace, Scratch, Session,
        TileSession, BOTH, LIST, RECT, TRI, ZIP,
    },
    DeviceCall, DeviceRun, RegisterCall, REG_INT32_SLOT, REG_SLOTS,
};

/// #[repr(C)] mirror of `SbReg` in kernels/register.cuh; all fields are 8 bytes.
#[repr(C)]
#[derive(Clone, Copy, Default)]
struct SbReg {
    full: u64,
    lam: u64,
    init: u64,
    init_sym: u64,
    end_l: u64,
    end_r: u64,
    stack: u64,
    i11: u64,
    i22: u64,
    gen: u64,
    loop_out: u64,
    loop_in: u64,
    s: u64,
    kappa4: u64,
    kappa1: u64,
    key: u64,
    tm_c: u64,
    dg_kcal: u64,
    dg_dcal: u64,
    dh_kcal: u64,
    ds_cal_per_k: u64,
    passed: u64,
    has_cutoff: u64,
    cutoff: u64,
    mask: u64,
    defect: u64,
    /// Pack format 2 (nndb2024-dna.md §5.4.6): N_P and the device address of CLS.
    n_p: u64,
    cls: u64,
}

// SAFETY: plain #[repr(C)] struct of u64 fields in the CUDA struct's order; passed by
// value as a kernel parameter.
unsafe impl DeviceRepr for SbReg {}

impl SbReg {
    /// The output address of slot `slot` (T1's field order, `REG_SLOTS`).
    fn slot(&mut self, slot: usize) -> &mut u64 {
        match slot {
            0 => &mut self.tm_c,
            1 => &mut self.dg_kcal,
            2 => &mut self.dg_dcal,
            3 => &mut self.dh_kcal,
            _ => &mut self.ds_cal_per_k,
        }
    }
}

/// Table indices in the device pack's order (duplex-cuda.md §4: INIT, INIT_SYM, END_L,
/// END_R, STACK, BULGE, I11, I12, I21, I22, GEN, LOOP_OUT, LOOP_IN, STATES (index 13,
/// never read here; nndb2024-dna.md §5.4.2), then the total).
const INIT: usize = 0;
const INIT_SYM: usize = 1;
const END_L: usize = 2;
const END_R: usize = 3;
const STACK: usize = 4;
const I11: usize = 6;
const I22: usize = 9;
const GEN: usize = 10;
const LOOP_OUT: usize = 11;
const LOOP_IN: usize = 12;
const TOTAL: usize = 14;

/// Per-call register state of a session: the uploaded full keys, the kernel
/// parameters without the per-chunk addresses, and the field buffers.
pub(crate) struct Reg {
    params: SbReg,
    _pack: Arc<crate::pack_cache::Pack<i64, u8>>,
    /// The ⊥ defect slot (QTP6): u64::MAX until an item reads a ⊥ cell.
    defect: CudaSlice<u64>,
    field_index: [usize; REG_SLOTS],
    floats: [Scratch<u64>; REG_SLOTS],
    dcal: Scratch<i32>,
    passed: Scratch<u8>,
}

/// Host copies of a stage's fields over `rows` items: the float64 slots as bit
/// patterns, `dg_dcal` as int32; None for a field not requested.
struct Fields {
    floats: [Option<Vec<u64>>; REG_SLOTS],
    dcal: Option<Vec<i32>>,
}

impl Reg {
    /// Upload the full keys once per call (tm-pairs-cuda.md §4).
    pub(crate) fn new(
        (stream, device): (&Arc<CudaStream>, &Device),
        call: &Arc<RegisterCall>,
    ) -> Result<Self, EngineError> {
        Self::new_on((stream, device), call, None)
    }
    pub(crate) fn new_on(
        (stream, device): (&Arc<CudaStream>, &Device),
        call: &Arc<RegisterCall>,
        execution: Option<&crate::interop::execution_stream::BorrowedExecutionStream>,
    ) -> Result<Self, EngineError> {
        let fail = |what: &str, error| failure_ordinal(device, what, error);
        let o = &call.offsets;
        let total = o[TOTAL];
        if !call.layout_ok() || call.full.len() != total || total < 2 {
            return Err(EngineError::Device(
                "the tm-on-Pairs device pack is inconsistent; use device='cpu'".into(),
            ));
        }
        let mut flat = Vec::new();
        try_reserve_exact(&mut flat, 4 * total, "tm-on-Pairs device pack")?;
        flat.extend(call.full.iter().flatten().copied());
        let pack = crate::pack_cache::Pack::get(
            stream,
            device,
            execution,
            crate::pack_cache::Kind::Register,
            [crate::pack_cache::bytes(&flat), &call.cls],
            || {
                let full = crate::interop::Retained::on_owned_stream(
                    stream,
                    device,
                    crate::interop::execution_stream::upload_on(stream, execution, &flat)
                        .map_err(|e| fail("uploading the tm-on-Pairs pack", e))?,
                );
                let cls = crate::interop::execution_stream::upload_on(stream, execution, &call.cls)
                    .map_err(|e| fail("uploading the tm-on-Pairs pack", e))?;
                Ok::<_, EngineError>((full.into_inner(), cls))
            },
        )?;
        let [full, cls] = pack.addresses();
        let defect = crate::interop::Retained::on_owned_stream(
            stream,
            device,
            match execution {
                // A constant fill on the resident path (S5 (a)).
                Some(execution) => execution.alloc_filled::<u64>(1, 0xFF),
                None => crate::interop::execution_stream::upload_on(stream, None, &[u64::MAX]),
            }
            .map_err(|e| fail("uploading the defect slot", e))?,
        );
        let at = |table: usize| o[table] as u64;
        let params = SbReg {
            full,
            lam: call.loop_max as u64,
            init: at(INIT),
            init_sym: at(INIT_SYM),
            end_l: at(END_L),
            end_r: at(END_R),
            stack: at(STACK),
            i11: at(I11),
            i22: at(I22),
            gen: at(GEN),
            loop_out: at(LOOP_OUT),
            loop_in: at(LOOP_IN),
            s: call.s_bits,
            kappa4: call.kappa4_bits,
            kappa1: call.kappa1_bits,
            defect: ptr(&defect, stream),
            n_p: call.n_p as u64,
            cls,
            ..SbReg::default()
        };
        Ok(Self {
            params,
            _pack: pack,
            defect: defect.into_inner(),
            field_index: call.fields,
            floats: std::array::from_fn(|_| Scratch::new()),
            dcal: Scratch::new(),
            passed: Scratch::new(),
        })
    }

    /// Resident register stage: publish fields from their device allocations,
    /// retaining the original arithmetic kernel and its exact full-cell pack.
    pub(crate) fn resident_stage(
        &mut self,
        lease: &crate::interop::DeviceLease,
        a: &crate::interop::BoundDeviceRows,
        b: &crate::interop::BoundDeviceRows,
        space: crate::resident::ResidentSpace,
    ) -> Result<crate::resident::ResidentChunk, EngineError> {
        use crate::{
            interop::DeviceDType,
            resident::{ResidentChunk, ResidentField},
            resident_native,
        };
        let n = usize::try_from(space.n)
            .map_err(|_| internal("resident register chunk exceeds the platform"))?;
        let keys = lease.output(DeviceDType::I64, vec![n])?;
        let mut fields = Vec::new();
        try_reserve_exact(&mut fields, REG_SLOTS, "resident register fields")?;
        let names = ["tm_c", "dg_kcal", "dg_dcal", "dh_kcal", "ds_cal_per_k"];
        let mut outputs = Vec::new();
        try_reserve_exact(&mut outputs, REG_SLOTS, "resident register owners")?;
        let mut params = self.params;
        params.key = keys.address();
        for (slot, name) in names.into_iter().enumerate() {
            let dtype = if slot == REG_INT32_SLOT {
                DeviceDType::I32
            } else {
                DeviceDType::F64
            };
            let output = lease.output(dtype, vec![n])?;
            *params.slot(slot) = output.address();
            outputs.push((name, dtype, output));
        }
        if n != 0 {
            let defect_byte = if register_defect_fires() { 0 } else { 0xFF };
            lease
                .execution()
                .fill(&mut self.defect, 1, defect_byte)
                .map_err(|e| {
                    failure_ordinal(lease.device(), "initializing resident register defect", e)
                })?;
            let table = resident_native::tables(a, b);
            let abi = space.abi();
            let function = &lease.device().0.kernels.reg;
            let mut launch = lease.execution().launch_builder(function);
            launch.arg(&table).arg(&abi).arg(&params);
            // SAFETY: all validated rows, immutable full pack and n fresh outputs remain owned through this same-stream launch.
            unsafe { launch.launch(resident_native::grid(lease, function, space.n)?) }
                .map_err(|e| failure_ordinal(lease.device(), "launching resident register", e))?;
            let mut defect = [0u64];
            lease
                .execution()
                .memcpy_dtoh(&self.defect, &mut defect[..])
                .map_err(|e| {
                    failure_ordinal(
                        lease.device(),
                        "reading resident register defect control",
                        e,
                    )
                })?;
            lease.synchronize()?;
            if defect[0] != u64::MAX {
                return Err(internal("register device read a bottom parameter cell"));
            }
        }
        for (name, dtype, output) in outputs {
            // SAFETY: sb_reg writes every requested field exactly once per item.
            fields.push(ResidentField {
                name: name.into(),
                values: unsafe { output.finish() }?,
                mask_bits: if dtype == DeviceDType::I32 {
                    i32::MAX as u64
                } else {
                    0x7ff8000000000000
                },
            });
        }
        // SAFETY: sb_reg writes every key; an empty logical output reads no bytes.
        Ok(ResidentChunk {
            space,
            keys: unsafe { keys.finish() }?,
            fields,
        })
    }
    /// Whether the device fills every requested detail field: all five T1 fields are
    /// device fields.
    pub(crate) fn delivers(&self, request: DetailRequest) -> bool {
        !request.is_empty()
            && (0..32).all(|field| !request.contains(field) || self.field_index.contains(&field))
    }

    /// Which of the five slots the request holds.
    fn wanted(&self, request: DetailRequest) -> [bool; REG_SLOTS] {
        self.field_index.map(|field| request.contains(field))
    }

    fn host_fields(&self, request: DetailRequest, rows: usize) -> Result<Fields, EngineError> {
        let wanted = self.wanted(request);
        let mut floats: [Option<Vec<u64>>; REG_SLOTS] = std::array::from_fn(|_| None);
        for (slot, column) in floats.iter_mut().enumerate() {
            if wanted[slot] && slot != REG_INT32_SLOT {
                *column = Some(try_filled(rows, 0u64, "device float field column")?);
            }
        }
        let dcal = if wanted[REG_INT32_SLOT] {
            Some(try_filled(rows, 0i32, "device int32 field column")?)
        } else {
            None
        };
        Ok(Fields { floats, dcal })
    }

    /// Detail columns of the call from whole host vectors, in the kernel's field order:
    /// float64 columns from their bit patterns, nothing else in between.
    fn columns(
        &self,
        fields: &'static [DetailField],
        request: DetailRequest,
        values: Fields,
    ) -> DetailColumns {
        let mut columns = DetailColumns::new(fields, request);
        for (slot, values) in values.floats.into_iter().enumerate() {
            if let Some(values) = values {
                let values = values.into_iter().map(f64::from_bits).collect();
                columns.set_column(self.field_index[slot], DetailData::Float64(values));
            }
        }
        if let Some(values) = values.dcal {
            columns.set_column(self.field_index[REG_INT32_SLOT], DetailData::Int32(values));
        }
        columns
    }
}

fn internal(what: &str) -> EngineError {
    EngineError::Device(format!(
        "internal error: {what}; no result was produced; use device='cpu'"
    ))
}

/// What a stage writes besides the keys, and where its fields go (tm-pairs-cuda.md §6,
/// QTP2).
#[derive(Clone, Copy)]
pub(crate) enum StageFields {
    /// The keys only (Hits' selection, RowStats, a port call without details).
    Keys,
    /// The requested fields, downloaded per chunk into whole-call columns (Each).
    Columns(DetailRequest),
    /// The requested fields, fetched as each chunk's own detail rows (Hits' detail
    /// pass, TopK, Matrix, Tiles). The chunk also closes at `REG_ROW_ITEMS` items, so
    /// the host copies of one chunk stay O(`REG_ROW_ITEMS`) (§14, host memory of Matrix).
    Rows(DetailRequest),
}

impl StageFields {
    fn request(self) -> Option<DetailRequest> {
        match self {
            Self::Keys => None,
            Self::Columns(request) | Self::Rows(request) => Some(request),
        }
    }
}

impl Session {
    /// Stage the chunk starting at item `pos`: the int64 key of every item in
    /// `dist64[0..n]` and, unless `fields` is `Keys`, the requested fields in the
    /// register field buffers. `each` (Each with a cutoff): also `passed` and the Each
    /// masks.
    pub(crate) fn reg_stage(
        &mut self,
        host: Host<'_, i64>,
        request: &Request<'_>,
        pos: usize,
        fields: StageFields,
        each: Option<i64>,
    ) -> Result<Chunk, EngineError> {
        let mut reg = self
            .reg
            .take()
            .ok_or_else(|| internal("an int64 stage without its device pack"))?;
        let result = self.reg_stage_with(&mut reg, host, request, pos, fields, each);
        self.reg = Some(reg);
        result
    }

    fn reg_stage_with(
        &mut self,
        reg: &mut Reg,
        host: Host<'_, i64>,
        request: &Request<'_>,
        pos: usize,
        fields: StageFields,
        each: Option<i64>,
    ) -> Result<Chunk, EngineError> {
        let settings = self.settings;
        let mut space = SbSpace {
            k0: pos as u64,
            ..SbSpace::default()
        };
        // The one length of an implicit cross or within shape (equal lengths are
        // validated), else per-pair lengths; planning never changes a result.
        let uniform = match request {
            Request::Range { work, .. } => match &work.kind {
                PairKind::Zip => {
                    space.kind = ZIP;
                    None
                }
                PairKind::Index { .. } => {
                    space.kind = LIST;
                    (space.li, space.lj) = self.index_columns(work)?;
                    None
                }
                PairKind::Cross | PairKind::OneToMany => {
                    space.kind = RECT;
                    space.width = work.shape.1 as u64;
                    Some(self.a.len(0))
                }
                PairKind::Within { .. }
                | PairKind::WithinOriented {
                    orientation: Orientation::AsGiven,
                    ..
                } => {
                    space.kind = TRI;
                    Some(self.a.len(0))
                }
                PairKind::WithinOriented {
                    orientation: Orientation::Both,
                    self_pairs,
                } => {
                    space.kind = BOTH;
                    space.skip = u64::from(!*self_pairs);
                    space.width = (work.shape.0 - usize::from(!*self_pairs)) as u64;
                    Some(self.a.len(0))
                }
            },
            Request::Pairs(_) => {
                space.kind = LIST;
                None
            }
            Request::Rect { r0, c0, width, .. } => {
                space.kind = RECT;
                space.r0 = *r0 as u64;
                space.c0 = *c0 as u64;
                space.width = *width as u64;
                Some(self.a.len(*r0))
            }
        };
        let a = &self.a;
        let rows = matches!(fields, StageFields::Rows(_));
        let n = reg_chunk(request.end() - pos, &settings, rows, uniform, |k| {
            a.len(Session::item_row(request, pos + k))
        });
        space.n = n as u64;
        if let Request::Pairs(pairs) = request {
            // Explicit pairs are uploaded per chunk and addressed from 0.
            self.upload_pairs(&pairs[pos..pos + n], &mut space)?;
        }
        if space.kind == TRI {
            let Request::Range { work, .. } = request else {
                unreachable!("only within ranges use row segments")
            };
            self.upload_segments(work, pos, n, &mut space)?;
        }
        let stream = Arc::clone(&self.stream);
        let alloc = |error| failure_ordinal(&self.device, "allocation", error);
        let mut params = reg.params;
        params.key = ptr(self.dist64.ensure(&stream, n).map_err(alloc)?, &stream);
        // (address, bytes per item) of every buffer this stage writes.
        let mut written = vec![(params.key, 8usize)];
        if let Some(request) = fields.request() {
            let wanted = reg.wanted(request);
            for (slot, &want) in wanted.iter().enumerate() {
                if !want {
                    continue;
                }
                let (address, width) = if slot == REG_INT32_SLOT {
                    (ptr(reg.dcal.ensure(&stream, n).map_err(alloc)?, &stream), 4)
                } else {
                    let buffer = reg.floats[slot].ensure(&stream, n).map_err(alloc)?;
                    (ptr(buffer, &stream), 8)
                };
                *params.slot(slot) = address;
                written.push((address, width));
            }
        }
        if let Some(cutoff) = each {
            params.passed = ptr(reg.passed.ensure(&stream, n).map_err(alloc)?, &stream);
            params.has_cutoff = 1;
            params.cutoff = cutoff as u64;
            params.mask = 1;
            written.push((params.passed, 1));
        }
        // Testing builds with a poison byte: refill every buffer this stage writes, every
        // chunk, so an item the kernel misses shows the poison byte, never a previous
        // chunk's value (the allocation-time fill covers the first chunk only; QTP9).
        #[cfg(feature = "testing")]
        if let Some(byte) = crate::launch::poison() {
            for &(address, width) in &written {
                self.fill(address, u32::from(byte), n * width, false)?;
            }
            let bytes: usize = written.iter().map(|&(_, width)| n * width).sum();
            crate::launch::count_poisoned(bytes as u64);
        }
        {
            let tables = self.tables();
            let config = self.grid(n as u64);
            let mut launch = stream.launch_builder(&self.device.0.kernels.reg);
            launch.arg(&tables).arg(&space).arg(&params);
            // SAFETY: matches sb_reg; the key, field and passed buffers hold n items,
            // the full keys every offset the kernel reads, the defect slot one u64.
            unsafe { launch.launch(config) }.map_err(|e| self.fail("launching sb_reg", e))?;
        }
        // QTP6: a ⊥ read (unreachable by rule F) is recomputed on the CPU kernel, which
        // raises its internal error; nothing is published.
        let mut slot = [0u64];
        self.download(&reg.defect, 1, &mut slot)?;
        self.sync()?;
        if slot[0] != u64::MAX {
            return Err(self.reg_defect(host, request, pos, slot[0] as usize));
        }
        Ok(Chunk { n, space })
    }

    /// The CPU kernel's error for the item `q` (an item number of the chunk at `pos`)
    /// that read a ⊥ cell on the device.
    fn reg_defect(
        &self,
        host: Host<'_, i64>,
        request: &Request<'_>,
        pos: usize,
        q: usize,
    ) -> EngineError {
        let (i, j) = match request {
            Request::Range { work, .. } => match work.range(q, q + 1).next() {
                Some(pair) => pair,
                None => return internal("a device defect outside its range"),
            },
            Request::Pairs(pairs) => pairs[pos + q],
            Request::Rect { r0, c0, width, .. } => (r0 + q / width, c0 + q % width),
        };
        match host.kernel.evaluate(host.a.get(i), host.b.get(j), None) {
            Err(error) => error,
            Ok(_) => internal(&format!(
                "the device read a ⊥ cell for pair ({i}, {j}) that the CPU kernel does not read"
            )),
        }
    }

    /// Append the staged chunk's keys to `out` (after a stage).
    fn reg_fetch_keys(&self, n: usize, out: &mut Vec<i64>) -> Result<(), EngineError> {
        let base = out.len();
        try_reserve(out, n, "device key chunk")?;
        out.resize(base + n, 0);
        self.download(self.dist64.get(), n, &mut out[base..])
            .and_then(|()| self.sync())
    }

    /// Copy the staged chunk's fields into `values[pos..pos + n]` (no sync).
    fn reg_download(&self, values: &mut Fields, pos: usize, n: usize) -> Result<(), EngineError> {
        let reg = self
            .reg
            .as_ref()
            .ok_or_else(|| internal("no register state"))?;
        for (slot, column) in values.floats.iter_mut().enumerate() {
            if let Some(column) = column.as_mut() {
                self.download(reg.floats[slot].get(), n, &mut column[pos..pos + n])?;
            }
        }
        if let Some(column) = values.dcal.as_mut() {
            self.download(reg.dcal.get(), n, &mut column[pos..pos + n])?;
        }
        Ok(())
    }

    /// The staged chunk's fields as detail rows (a fields-mode stage).
    fn reg_fetch_details(
        &self,
        host: Host<'_, i64>,
        n: usize,
    ) -> Result<DetailColumns, EngineError> {
        let reg = self
            .reg
            .as_ref()
            .ok_or_else(|| internal("no register state"))?;
        let request = host.kernel.detail_request();
        let mut values = reg.host_fields(request, n)?;
        self.reg_download(&mut values, 0, n)?;
        self.sync()?;
        Ok(reg.columns(host.kernel.detail_fields(), request, values))
    }

    fn reg_delivers(&self, request: DetailRequest) -> bool {
        self.reg.as_ref().is_some_and(|reg| reg.delivers(request))
    }

    /// Each (§6): one fields-mode stage per chunk; with a cutoff the kernel writes
    /// `passed` and the masks. The key column is kept only when the kernel asks for it
    /// (`each_keys`; never for `tm` on Pairs), masked to `Score::MASK` on failing rows.
    pub(crate) fn each64(
        &mut self,
        host: Host<'_, i64>,
        work: &WorkSource,
        cutoff: Option<i64>,
    ) -> Result<Columns<i64>, EngineError> {
        let n_pairs = work.n_pairs;
        let request = host.kernel.detail_request();
        let detailed = !request.is_empty();
        if detailed && !self.reg_delivers(request) {
            return Err(internal("a device Each without its detail fields"));
        }
        let keys = !detailed || host.kernel.each_keys();
        let mut dist = if keys {
            try_filled(n_pairs, 0i64, "device each key column")?
        } else {
            Vec::new()
        };
        let mut passed = match cutoff {
            Some(_) => Some(try_filled(n_pairs, 0u8, "device each passed column")?),
            None => None,
        };
        let mut values = if detailed {
            Some(
                self.reg
                    .as_ref()
                    .expect("checked above")
                    .host_fields(request, n_pairs)?,
            )
        } else {
            None
        };
        let range = Request::Range { work, end: n_pairs };
        let fields = if detailed {
            StageFields::Columns(request)
        } else {
            StageFields::Keys
        };
        let mut pos = 0;
        while pos < n_pairs {
            let chunk = self.reg_stage(host, &range, pos, fields, cutoff)?;
            let n = chunk.n;
            if keys {
                self.download(self.dist64.get(), n, &mut dist[pos..pos + n])?;
            }
            if let Some(passed) = passed.as_mut() {
                let reg = self.reg.as_ref().expect("staged above");
                self.download(reg.passed.get(), n, &mut passed[pos..pos + n])?;
            }
            if let Some(values) = values.as_mut() {
                self.reg_download(values, pos, n)?;
            }
            // The next stage reuses the device buffers.
            self.sync()?;
            pos += n;
        }
        if let (true, Some(passed)) = (keys, passed.as_deref()) {
            for (key, &ok) in dist.iter_mut().zip(passed) {
                if ok == 0 {
                    *key = i64::MASK;
                }
            }
        }
        let details = match values {
            Some(values) => {
                let reg = self.reg.as_ref().expect("staged above");
                Some(reg.columns(host.kernel.detail_fields(), request, values))
            }
            None => None,
        };
        Ok(Columns {
            dist,
            passed,
            details,
            ..Columns::default()
        })
    }

    /// Hits (§6): key-mode stages and G1's count, scan and ordered write on int64 keys;
    /// `index` lists through the host sort and dedup; then one fields-mode pass over
    /// the final hit list, whose keys must equal the selection keys (QTP7).
    pub(crate) fn hits64(
        &mut self,
        host: Host<'_, i64>,
        work: &WorkSource,
        cutoff: Option<i64>,
    ) -> Result<Columns<i64>, EngineError> {
        let request = host.kernel.detail_request();
        let (mut ii, mut jj, mut dd) = (Vec::<i64>::new(), Vec::<i64>::new(), Vec::<i64>::new());
        let mut counts_dev = Scratch::<u64>::new();
        let mut offsets_dev = Scratch::<u64>::new();
        let mut out_i = Scratch::<i64>::new();
        let mut out_j = Scratch::<i64>::new();
        let mut out_d = Scratch::<i64>::new();
        let (has_cutoff, limit) = (i32::from(cutoff.is_some()), cutoff.unwrap_or(0));
        let range = Request::Range {
            work,
            end: work.n_pairs,
        };
        let mut pos = 0;
        while pos < work.n_pairs {
            let chunk = self.reg_stage(host, &range, pos, StageFields::Keys, None)?;
            let (n, count) = (chunk.n, chunk.n as u64);
            let (blocks, config) = self.tiles(n);
            let dist_ptr = ptr(self.dist64.get(), &self.stream);
            let counts_ptr = ptr(
                counts_dev
                    .ensure(&self.stream, blocks)
                    .map_err(|e| self.fail("allocation", e))?,
                &self.stream,
            );
            let mut launch = self
                .stream
                .launch_builder(&self.device.0.kernels.hits_count_i64);
            launch
                .arg(&dist_ptr)
                .arg(&count)
                .arg(&has_cutoff)
                .arg(&limit)
                .arg(&counts_ptr);
            // SAFETY: matches sb_hits_count_i64; one count slot per block.
            unsafe { launch.launch(config) }
                .map_err(|e| self.fail("launching sb_hits_count_i64", e))?;
            let mut block_counts = try_filled(blocks, 0u64, "hit block counts")?;
            let mut offsets = Vec::new();
            try_reserve_exact(&mut offsets, blocks, "hit block offsets")?;
            self.download(counts_dev.get(), blocks, &mut block_counts)?;
            self.sync()?;
            let mut total = 0u64;
            for value in block_counts {
                offsets.push(total);
                total += value;
            }
            if total > 0 {
                let hits = total as usize;
                let stream = &self.stream;
                let alloc = |error| failure_ordinal(&self.device, "allocation", error);
                let offsets_ptr =
                    Session::upload(stream, &mut offsets_dev, &offsets).map_err(alloc)?;
                let i_ptr = ptr(out_i.ensure(stream, hits).map_err(alloc)?, stream);
                let j_ptr = ptr(out_j.ensure(stream, hits).map_err(alloc)?, stream);
                let d_ptr = ptr(out_d.ensure(stream, hits).map_err(alloc)?, stream);
                let space = chunk.space;
                let mut launch = stream.launch_builder(&self.device.0.kernels.hits_write_i64);
                launch
                    .arg(&dist_ptr)
                    .arg(&count)
                    .arg(&has_cutoff)
                    .arg(&limit)
                    .arg(&offsets_ptr)
                    .arg(&space)
                    .arg(&i_ptr)
                    .arg(&j_ptr)
                    .arg(&d_ptr);
                // SAFETY: matches sb_hits_write_i64; outputs hold `total` entries.
                unsafe { launch.launch(config) }
                    .map_err(|e| self.fail("launching sb_hits_write_i64", e))?;
                let base = dd.len();
                // Every hit column is reserved before any is written
                // (host-allocation-failures §12.6 rule 8).
                try_reserve(&mut ii, hits, "device hits i column")?;
                try_reserve(&mut jj, hits, "device hits j column")?;
                try_reserve(&mut dd, hits, "device hits key column")?;
                ii.resize(base + hits, 0);
                jj.resize(base + hits, 0);
                dd.resize(base + hits, 0);
                self.download(out_i.get(), hits, &mut ii[base..])?;
                self.download(out_j.get(), hits, &mut jj[base..])?;
                self.download(out_d.get(), hits, &mut dd[base..])?;
                self.sync()?;
            }
            pos += n;
        }
        let mut columns = if matches!(work.kind, PairKind::Index { .. }) {
            // Explicit lists: the CPU's host sort by (i, j) and dedup.
            let mut hits: Vec<(usize, usize, i64)> = Vec::new();
            try_reserve_exact(&mut hits, dd.len(), "device hit records")?;
            hits.extend(
                ii.into_iter()
                    .zip(jj)
                    .zip(dd)
                    .map(|((i, j), d)| (i as usize, j as usize, d)),
            );
            Columns::from_hits(hits)?
        } else {
            // Implicit shapes: work-ID order is (i, j) order and unique.
            Columns {
                dist: dd,
                i: Some(ii),
                j: Some(jj),
                ..Columns::default()
            }
        };
        if !request.is_empty() {
            if !self.reg_delivers(request) {
                return Err(internal("a device Hits without its detail fields"));
            }
            columns.details = Some(self.reg_hit_details(host, &columns)?);
        }
        Ok(columns)
    }

    /// The fields of the final hit list in its order (a fields-mode pass); each detail
    /// key must equal the hit's selection key, otherwise nothing is published (QTP7).
    fn reg_hit_details(
        &mut self,
        host: Host<'_, i64>,
        columns: &Columns<i64>,
    ) -> Result<DetailColumns, EngineError> {
        let (i, j) = (
            columns.i.as_deref().unwrap_or(&[]),
            columns.j.as_deref().unwrap_or(&[]),
        );
        let mut pairs = Vec::new();
        try_reserve_exact(&mut pairs, i.len(), "tm-on-Pairs hit list")?;
        pairs.extend(i.iter().zip(j).map(|(&i, &j)| (i as usize, j as usize)));
        let request = host.kernel.detail_request();
        let mut rows = DetailColumns::new(host.kernel.detail_fields(), request);
        rows.reserve(pairs.len())?;
        let list = Request::Pairs(&pairs);
        let mut keys = Vec::new();
        let mut pos = 0;
        while pos < pairs.len() {
            let chunk = self.reg_stage(host, &list, pos, StageFields::Rows(request), None)?;
            keys.clear();
            self.reg_fetch_keys(chunk.n, &mut keys)?;
            for (t, &key) in keys.iter().enumerate() {
                if key != columns.dist[pos + t] {
                    let (i, j) = pairs[pos + t];
                    return Err(EngineError::Device(format!(
                        "tm on Pairs Hits: the detail pass gives key {key} for pair ({i}, {j}) \
                         but the selection pass gave {}; no result was published; use \
                         device='cpu'",
                        columns.dist[pos + t]
                    )));
                }
            }
            let mut chunk_rows = self.reg_fetch_details(host, chunk.n)?;
            rows.append(&mut chunk_rows)?;
            pos += chunk.n;
        }
        Ok(rows)
    }

    /// RowStats (§6): key-mode stages in two sweeps with the int64 sink kernels; `best`
    /// starts at exactly `Score::MASK`.
    pub(crate) fn rowstats64(
        &mut self,
        host: Host<'_, i64>,
        work: &WorkSource,
        cutoff: Option<i64>,
    ) -> Result<RowStatsColumns<i64>, EngineError> {
        let rows = work.shape.0;
        let alloc = |error| failure_ordinal(&self.device, "allocation", error);
        // SAFETY (all four): initialised by the fills below before any kernel reads them.
        let best = unsafe { dev_alloc::<i64>(&self.stream, rows.max(1)) }.map_err(alloc)?;
        let count = unsafe { dev_alloc::<u64>(&self.stream, rows.max(1)) }.map_err(alloc)?;
        let nbest = unsafe { dev_alloc::<u64>(&self.stream, rows.max(1)) }.map_err(alloc)?;
        let arg = unsafe { dev_alloc::<u64>(&self.stream, rows.max(1)) }.map_err(alloc)?;
        let first = match work.first_occurrences()? {
            Some(mask) => {
                let host_mask: Vec<u8> = mask.into_iter().map(u8::from).collect();
                let mut scratch = Scratch::<u8>::new();
                Session::upload(&self.stream, &mut scratch, &host_mask).map_err(alloc)?;
                Some(scratch)
            }
            None => None,
        };
        let stream = Arc::clone(&self.stream);
        let table = SbRows {
            mirror: u64::from(work.mirrors_within()),
            first: first.as_ref().map_or(0, |mask| ptr(mask.get(), &stream)),
            best: ptr(&best, &stream),
            count: ptr(&count, &stream),
            nbest: ptr(&nbest, &stream),
            arg: ptr(&arg, &stream),
        };
        if rows > 0 {
            let (n, value) = (rows as u64, i64::MASK);
            let config = self.grid(n);
            let mut launch = stream.launch_builder(&self.device.0.kernels.fill_i64);
            launch.arg(&table.best).arg(&n).arg(&value);
            // SAFETY: matches sb_fill_i64; `best` holds `rows` int64 values.
            unsafe { launch.launch(config) }.map_err(|e| self.fail("launching sb_fill_i64", e))?;
        }
        self.fill(table.count, 0, rows * 8, false)?;
        self.fill(table.nbest, 0, rows * 8, false)?;
        self.fill(table.arg, 0xFF, rows * 8, false)?;
        let (has_cutoff, limit) = (i32::from(cutoff.is_some()), cutoff.unwrap_or(0));
        let range = Request::Range {
            work,
            end: work.n_pairs,
        };
        for sweep in [0, 1] {
            let mut pos = 0;
            while pos < work.n_pairs {
                let chunk = self.reg_stage(host, &range, pos, StageFields::Keys, None)?;
                let dist_ptr = ptr(self.dist64.get(), &stream);
                let config = self.grid(chunk.n as u64);
                let (kernel, name) = if sweep == 0 {
                    (
                        &self.device.0.kernels.rowstats_min_i64,
                        "sb_rowstats_min_i64",
                    )
                } else {
                    (
                        &self.device.0.kernels.rowstats_arg_i64,
                        "sb_rowstats_arg_i64",
                    )
                };
                let mut launch = stream.launch_builder(kernel);
                launch
                    .arg(&dist_ptr)
                    .arg(&chunk.space)
                    .arg(&has_cutoff)
                    .arg(&limit)
                    .arg(&table);
                // SAFETY: matches the int64 rowstats kernels; row arrays hold `rows` entries.
                unsafe { launch.launch(config) }
                    .map_err(|e| self.fail(&format!("launching {name}"), e))?;
                // The next stage rewrites the key buffer: finish this sweep's reads first.
                self.sync()?;
                pos += chunk.n;
            }
        }
        // Every download buffer is reserved before any is written
        // (host-allocation-failures §12.6 rule 8).
        let (mut host_best, mut host_count) = (Vec::<i64>::new(), Vec::<u64>::new());
        let (mut host_nbest, mut host_arg) = (Vec::<u64>::new(), Vec::<u64>::new());
        try_reserve_exact(&mut host_best, rows, "rowstats best download")?;
        try_reserve_exact(&mut host_count, rows, "rowstats count download")?;
        try_reserve_exact(&mut host_nbest, rows, "rowstats n_best download")?;
        try_reserve_exact(&mut host_arg, rows, "rowstats argbest download")?;
        host_best.resize(rows, 0);
        host_count.resize(rows, 0);
        host_nbest.resize(rows, 0);
        host_arg.resize(rows, 0);
        self.download(&best, rows, &mut host_best)?;
        self.download(&count, rows, &mut host_count)?;
        self.download(&nbest, rows, &mut host_nbest)?;
        self.download(&arg, rows, &mut host_arg)?;
        self.sync()?;
        let mut states = Vec::new();
        try_reserve_exact(&mut states, rows, "rowstats states")?;
        for row in 0..rows {
            states.push(if host_count[row] == 0 {
                RowState::default()
            } else {
                let partner = i64::try_from(host_arg[row]).map_err(|_| {
                    EngineError::Device(format!("device rowstats argbest is invalid in row {row}"))
                })?;
                RowState::from_summary(host_best[row], partner, host_nbest[row], host_count[row])?
            });
        }
        RowStatsColumns::from_states(states, cutoff.is_some())
    }
}

/// A session bound to one int64-key call's kernel and tables: the batch-distance port
/// on int64 keys (TopK, Matrix, Tiles), with the device's fields as detail rows.
pub(crate) struct Bound64<'s> {
    session: &'s mut Session,
    host: Host<'s, i64>,
    keys: Vec<i64>,
}

impl Bound64<'_> {
    /// A port stage's fields: the chunk's detail rows when asked, else the keys only.
    fn fields(&self, details: bool) -> StageFields {
        if details {
            StageFields::Rows(self.host.kernel.detail_request())
        } else {
            StageFields::Keys
        }
    }

    /// Detail rows of the staged chunk, or a refusal when the device cannot deliver
    /// the request (never for `tm` on Pairs).
    fn chunk_rows(&self, fields: bool, n: usize) -> Result<DetailColumns, EngineError> {
        if !fields {
            return Err(EngineError::Parameter(
                "this device call cannot deliver the requested detail fields".into(),
            ));
        }
        self.session.reg_fetch_details(self.host, n)
    }
}

impl BatchDistance<i64> for Bound64<'_> {
    fn delivers_details(&self) -> bool {
        self.session.reg_delivers(self.host.kernel.detail_request())
    }

    fn range(
        &mut self,
        work: &WorkSource,
        start: usize,
        end: usize,
        details: bool,
        visit: &mut DistanceVisit<'_, i64>,
    ) -> Result<(), EngineError> {
        let request = Request::Range { work, end };
        let details = details && self.delivers_details();
        let fields = self.fields(details);
        let mut pos = start;
        while pos < end {
            let chunk = self
                .session
                .reg_stage(self.host, &request, pos, fields, None)?;
            let mut keys = std::mem::take(&mut self.keys);
            keys.clear();
            let fetched = self.session.reg_fetch_keys(chunk.n, &mut keys);
            let rows = match fetched {
                Ok(()) if details => self.session.reg_fetch_details(self.host, chunk.n).map(Some),
                Ok(()) => Ok(None),
                Err(error) => Err(error),
            };
            let result = rows.and_then(|rows| visit(pos, &keys, rows.as_ref()));
            self.keys = keys;
            result?;
            pos += chunk.n;
        }
        Ok(())
    }

    fn pairs(
        &mut self,
        pairs: &[(usize, usize)],
        out: &mut Vec<i64>,
        mut details: Option<&mut DetailColumns>,
    ) -> Result<(), EngineError> {
        let request = Request::Pairs(pairs);
        let fields = details.is_some() && self.delivers_details();
        try_reserve(out, pairs.len(), "device batch keys")?;
        let mut pos = 0;
        while pos < pairs.len() {
            let chunk =
                self.session
                    .reg_stage(self.host, &request, pos, self.fields(fields), None)?;
            self.session.reg_fetch_keys(chunk.n, out)?;
            if let Some(rows) = details.as_deref_mut() {
                let mut chunk_rows = self.chunk_rows(fields, chunk.n)?;
                rows.append(&mut chunk_rows)?;
            }
            pos += chunk.n;
        }
        Ok(())
    }

    fn rect(
        &mut self,
        rows: (usize, usize),
        cols: (usize, usize),
        class: &CellClass<'_>,
        out: &mut Vec<i64>,
        mut details: Option<&mut DetailColumns>,
    ) -> Result<(), EngineError> {
        let width = cols.1 - cols.0;
        let cells = (rows.1 - rows.0) * width;
        let request = Request::Rect {
            r0: rows.0,
            c0: cols.0,
            width,
            end: cells,
            class: Some(class),
        };
        let fields = details.is_some() && self.delivers_details();
        try_reserve(out, cells, "device batch keys")?;
        let mut pos = 0;
        while pos < cells {
            let chunk =
                self.session
                    .reg_stage(self.host, &request, pos, self.fields(fields), None)?;
            self.session.reg_fetch_keys(chunk.n, out)?;
            if let Some(target) = details.as_deref_mut() {
                let mut chunk_rows = self.chunk_rows(fields, chunk.n)?;
                target.append(&mut chunk_rows)?;
            }
            pos += chunk.n;
        }
        Ok(())
    }
}

/// Device compute of `tm` on Pairs for every non-Tiles sink (`crate::run_i64`, which K2
/// with the primer3 preset shares through `p3::run`; K3 has its own 64-bit-key entry,
/// `crate::run_pf`).
pub(crate) fn run(
    device: &Device,
    kernel: &dyn PairDistanceKernel<i64>,
    call: DeviceCall,
    work: &WorkSource,
    (a, b): (&SeqTable, &SeqTable),
    options: &ExecOptions<i64>,
    validation: ValidationReport,
) -> Result<DeviceRun<i64>, EngineError> {
    guard(|| {
        if !matches!(call, DeviceCall::TmPairs(_)) {
            return Err(internal(
                "the tm-on-Pairs int64-key CUDA entry serves tm on Pairs only",
            ));
        }
        let mut session = Session::new(device, call, (a, b), options.workers(), false)?;
        let host = Host { kernel, a, b };
        let output = match options.sink {
            Sink::TopK => {
                let mut port = Bound64 {
                    session: &mut session,
                    host,
                    keys: Vec::new(),
                };
                topk::run_batch(kernel, (a, b), &mut port, work, options, validation)?
            }
            Sink::Matrix => {
                let mut port = Bound64 {
                    session: &mut session,
                    host,
                    keys: Vec::new(),
                };
                matrix::run_batch(kernel, (a, b), &mut port, work, options, validation)?
            }
            sink => {
                let columns = match sink {
                    Sink::Each => session.each64(host, work, options.cutoff)?,
                    Sink::Hits => session.hits64(host, work, options.cutoff)?,
                    _ => Columns {
                        rowstats: Some(session.rowstats64(host, work, options.cutoff)?),
                        ..Columns::default()
                    },
                };
                RunOutput {
                    columns,
                    shape: work.shape,
                    counts: Counts {
                        certified: 0,
                        pruned: 0,
                        full: work.n_pairs,
                    },
                    threads: options.threads,
                    n_illegal_seqs: validation.n_illegal_seqs,
                    validation_scans: validation.scans,
                    topk_replay: None,
                }
            }
        };
        Ok(DeviceRun {
            output,
            host_pairs: session.host_pairs(),
        })
    })
}

impl TileSession {
    /// The int64-key batch-distance port for one `next()`, bound to the call's kernel and
    /// the stream's validated tables: `tm` on Pairs (this file) or K2 `duplex` with the
    /// primer3 preset (`p3.rs`, primer3-cuda.md §7), chosen by the session's call. Each
    /// keeps its own stage and sink kernels.
    pub fn bind_i64<'s>(
        &'s mut self,
        kernel: &'s dyn PairDistanceKernel<i64>,
        (a, b): (&'s SeqTable, &'s SeqTable),
    ) -> TileBatch64<'s> {
        let host = Host { kernel, a, b };
        if self.0.p3.is_some() {
            TileBatch64(Port64::P3(crate::p3::Bound64 {
                session: &mut self.0,
                host,
            }))
        } else {
            TileBatch64(Port64::Reg(Bound64 {
                session: &mut self.0,
                host,
                keys: Vec::new(),
            }))
        }
    }
}

/// An int64-key tile session bound for one `next()`; every call runs under `guard`.
pub struct TileBatch64<'s>(Port64<'s>);

/// The port of a [`TileBatch64`]: `tm` on Pairs or K2 with the primer3 preset.
enum Port64<'s> {
    Reg(Bound64<'s>),
    P3(crate::p3::Bound64<'s>),
}

impl BatchDistance<i64> for TileBatch64<'_> {
    fn delivers_details(&self) -> bool {
        match &self.0 {
            Port64::Reg(port) => port.delivers_details(),
            Port64::P3(port) => port.delivers_details(),
        }
    }

    fn range(
        &mut self,
        work: &WorkSource,
        start: usize,
        end: usize,
        details: bool,
        visit: &mut DistanceVisit<'_, i64>,
    ) -> Result<(), EngineError> {
        guard(|| match &mut self.0 {
            Port64::Reg(port) => port.range(work, start, end, details, visit),
            Port64::P3(port) => port.range(work, start, end, details, visit),
        })
    }

    fn pairs(
        &mut self,
        pairs: &[(usize, usize)],
        out: &mut Vec<i64>,
        details: Option<&mut DetailColumns>,
    ) -> Result<(), EngineError> {
        guard(|| match &mut self.0 {
            Port64::Reg(port) => port.pairs(pairs, out, details),
            Port64::P3(port) => port.pairs(pairs, out, details),
        })
    }

    fn rect(
        &mut self,
        rows: (usize, usize),
        cols: (usize, usize),
        class: &CellClass<'_>,
        out: &mut Vec<i64>,
        details: Option<&mut DetailColumns>,
    ) -> Result<(), EngineError> {
        guard(|| match &mut self.0 {
            Port64::Reg(port) => port.rect(rows, cols, class, out, details),
            Port64::P3(port) => port.rect(rows, cols, class, out, details),
        })
    }
}

#[cfg(feature = "testing")]
static REGISTER_DEFECT: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
#[cfg(feature = "testing")]
pub fn set_register_defect(count: u64) {
    REGISTER_DEFECT.store(count, std::sync::atomic::Ordering::Relaxed);
}
fn register_defect_fires() -> bool {
    #[cfg(feature = "testing")]
    {
        use std::sync::atomic::Ordering::Relaxed;
        let mut left = REGISTER_DEFECT.load(Relaxed);
        while left != 0 {
            match REGISTER_DEFECT.compare_exchange_weak(left, left - 1, Relaxed, Relaxed) {
                Ok(_) => return true,
                Err(next) => left = next,
            }
        }
    }
    false
}
