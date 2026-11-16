//! Unary device input -> device output on the same generic resident sinks.
//! This new module is deliberately unregistered during the first ABI compile;
//! compatibility view reads additionally require the reviewed family hook patch.
use crate::{
    interop::{BoundDeviceRows, DeviceColumn, DeviceDType, DeviceLease, DeviceOutput},
    resident::{
        ResidentChunk, ResidentField, ResidentFieldSpec, ResidentProducer, ResidentSinkConfig,
        ResidentSpace, ResidentWork,
    },
    resident_native::grid,
    resident_sinks,
    session::failure_ordinal,
    UnaryCall,
};
use cudarc::driver::{DeviceRepr, PushKernelArg};
use seqblaze_core::{sink::Sink, validate::EngineError};
#[repr(C)]
#[derive(Clone, Copy, Default)]
struct Table {
    data: u64,
    off: u64,
    view: u64,
}
#[repr(C)]
#[derive(Clone, Copy, Default)]
struct Output {
    slots: [u64; 5],
    passed: u64,
}
// SAFETY: exact plain-u64 mirrors of SbUnaryTable and SbTmOut.
unsafe impl DeviceRepr for Table {}
unsafe impl DeviceRepr for Output {}
const _: () = {
    assert!(std::mem::size_of::<Table>() == 24);
    assert!(std::mem::size_of::<Output>() == 48);
};

fn spec(name: &str, dtype: DeviceDType) -> ResidentFieldSpec {
    ResidentFieldSpec {
        name: name.into(),
        dtype,
        mask_bits: if dtype == DeviceDType::F64 {
            0x7ff8000000000000
        } else if dtype == DeviceDType::Bool {
            0
        } else {
            i32::MAX as u64
        },
    }
}
pub struct UnaryProducer<'a> {
    table: &'a BoundDeviceRows,
    call: UnaryCall,
    next: u64,
    n: u64,
    specs: Vec<ResidentFieldSpec>,
    stream: u64,
    max_length: u64,
    chunk_cap: u64,
    hairpin: Option<crate::resident_p3::ResidentPrimer3Stage>,
    cutoff: Option<i64>,
    passed: Option<DeviceColumn>,
}
impl<'a> UnaryProducer<'a> {
    pub fn new(
        lease: &DeviceLease,
        table: &'a BoundDeviceRows,
        call: UnaryCall,
        requested: &[&str],
        max_length: u64,
        cutoff: Option<i64>,
    ) -> Result<Self, EngineError> {
        table.ensure_lease(lease)?;
        let all = match &call {
            UnaryCall::Tm(_) => vec![
                spec("tm_c", DeviceDType::F64),
                spec("dg_kcal", DeviceDType::F64),
                spec("dg_dcal", DeviceDType::I32),
                spec("dh_kcal", DeviceDType::F64),
                spec("ds_cal_per_k", DeviceDType::F64),
            ],
            UnaryCall::TmBiopython(_) | UnaryCall::TmPrimer3(_) => {
                vec![spec("tm_c", DeviceDType::F64)]
            }
            UnaryCall::Hairpin(_) => vec![
                spec("found", DeviceDType::Bool),
                spec("dg_kcal", DeviceDType::F64),
                spec("dh_kcal", DeviceDType::F64),
                spec("ds_cal_per_k", DeviceDType::F64),
                spec("tm_c", DeviceDType::F64),
            ],
        };
        let hairpin = if let UnaryCall::Hairpin(pack) = &call {
            Some(crate::resident_p3::ResidentPrimer3Stage::new(lease, pack)?)
        } else {
            None
        };
        let mut specs = Vec::new();
        specs
            .try_reserve_exact(all.len())
            .map_err(|_| EngineError::Device("unable to retain unary resident fields".into()))?;
        specs.extend(
            all.into_iter()
                .filter(|field| requested.contains(&field.name.as_str())),
        );
        Ok(Self {
            table,
            call,
            next: 0,
            n: table.rows.rows,
            specs,
            stream: lease.stream_handle(),
            max_length,
            chunk_cap: crate::launch::settings().pairs_per_chunk.clamp(1, 65536),
            hairpin,
            cutoff,
            passed: None,
        })
    }
    fn stage(
        &mut self,
        lease: &DeviceLease,
        first: u64,
        n: u64,
    ) -> Result<ResidentChunk, EngineError> {
        let mut all = if let Some(stage) = &self.hairpin {
            let fields = stage.hairpin(
                lease,
                self.table,
                first,
                n,
                self.max_length,
                (&self.specs, self.cutoff),
            )?;
            self.passed = fields.passed;
            vec![
                ResidentField {
                    name: "found".into(),
                    values: fields.found,
                    mask_bits: 0,
                },
                ResidentField {
                    name: "dg_kcal".into(),
                    values: fields.dg_kcal,
                    mask_bits: 0x7ff8000000000000,
                },
                ResidentField {
                    name: "dh_kcal".into(),
                    values: fields.dh_kcal,
                    mask_bits: 0x7ff8000000000000,
                },
                ResidentField {
                    name: "ds_cal_per_k".into(),
                    values: fields.ds_cal_per_k,
                    mask_bits: 0x7ff8000000000000,
                },
                ResidentField {
                    name: "tm_c".into(),
                    values: fields.tm_c,
                    mask_bits: 0x7ff8000000000000,
                },
            ]
        } else {
            let native = matches!(self.call, UnaryCall::Tm(_));
            let names = ["tm_c", "dg_kcal", "dg_dcal", "dh_kcal", "ds_cal_per_k"];
            let mut outputs: Vec<(&str, DeviceDType, DeviceOutput)> = Vec::new();
            outputs
                .try_reserve_exact(if native { 5 } else { 1 })
                .map_err(|_| EngineError::Device("unable to retain unary device outputs".into()))?;
            let mut out = Output::default();
            for (slot, name) in names
                .into_iter()
                .enumerate()
                .take(if native { 5 } else { 1 })
            {
                let dtype = if slot == 2 {
                    DeviceDType::I32
                } else {
                    DeviceDType::F64
                };
                let column = lease.output(dtype, vec![n as usize])?;
                #[cfg(feature = "testing")]
                if let Some(byte) = crate::launch::poison() {
                    let bytes = n as usize * dtype.width();
                    // SAFETY: this output owns exactly n elements; the native
                    // stage overwrites them on the same execution stream.
                    unsafe { lease.execution().memset(column.address(), byte, bytes) }.map_err(
                        |e| failure_ordinal(lease.device(), "refilling unary output", e),
                    )?;
                    crate::launch::count_poisoned(bytes as u64);
                }
                out.slots[slot] = column.address();
                outputs.push((name, dtype, column));
            }
            if n != 0 {
                let table = Table {
                    view: self.table.address(),
                    ..Table::default()
                };
                let no = 0i32;
                let bits = 0u64;
                let stage = match &self.call {
                    UnaryCall::Tm(_) => &lease.device().0.kernels.tm,
                    UnaryCall::TmBiopython(_) => &lease.device().0.kernels.tm_biopython,
                    UnaryCall::TmPrimer3(_) => &lease.device().0.kernels.tm_primer3,
                    UnaryCall::Hairpin(_) => unreachable!(),
                };
                let mut launch = lease.execution().launch_builder(stage);
                launch.arg(&table).arg(&first).arg(&n);
                match &self.call {
                    UnaryCall::Tm(pack) => {
                        launch.arg(pack);
                    }
                    UnaryCall::TmBiopython(pack) => {
                        launch.arg(pack);
                    }
                    UnaryCall::TmPrimer3(pack) => {
                        launch.arg(pack);
                    }
                    UnaryCall::Hairpin(_) => unreachable!(),
                };
                launch.arg(&out).arg(&no).arg(&bits).arg(&no);
                // SAFETY: the immutable pack's native stage reads the retained
                // row view, initializes every requested output and applies no mask.
                unsafe { launch.launch(grid(lease, stage, n)?) }.map_err(|e| {
                    failure_ordinal(lease.device(), "launching unary resident stage", e)
                })?;
            }
            let mut all = Vec::new();
            all.try_reserve_exact(outputs.len())
                .map_err(|_| EngineError::Device("unable to publish unary device fields".into()))?;
            for (name, dtype, output) in outputs {
                // SAFETY: native stage initialized all logical output bytes.
                all.push(ResidentField {
                    name: name.into(),
                    values: unsafe { output.finish() }?,
                    mask_bits: if dtype == DeviceDType::F64 {
                        0x7ff8000000000000
                    } else if dtype == DeviceDType::Bool {
                        0
                    } else {
                        i32::MAX as u64
                    },
                });
            }
            all
        };
        let primary = if self.hairpin.is_some() {
            "dg_kcal"
        } else {
            "tm_c"
        };
        let tm = all
            .iter()
            .find(|field| field.name == primary)
            .ok_or_else(|| {
                EngineError::Device("unary resident stage omitted its primary".into())
            })?;
        let keys = lease.output(DeviceDType::I64, vec![n as usize])?;
        if n != 0 {
            let stage = &lease.device().0.kernels.resident_unary_keys;
            let input = tm.values.address();
            let output = keys.address();
            let mut launch = lease.execution().launch_builder(stage);
            let decreasing = u64::from(self.hairpin.is_none());
            launch.arg(&input).arg(&n).arg(&output).arg(&decreasing);
            // SAFETY: exactly n complete tm bit patterns map to n signed keys.
            unsafe { launch.launch(grid(lease, stage, n)?) }
                .map_err(|e| failure_ordinal(lease.device(), "mapping unary selecting keys", e))?;
        }
        let mut fields = Vec::new();
        fields
            .try_reserve_exact(self.specs.len())
            .map_err(|_| EngineError::Device("unable to retain unary resident fields".into()))?;
        for spec in &self.specs {
            if let Some(at) = all.iter().position(|field| field.name == spec.name) {
                fields.push(all.remove(at));
            } else {
                return Err(EngineError::Device(
                    "unary resident field schema changed".into(),
                ));
            }
        }
        // SAFETY: key stage initializes every nonempty key on the same stream.
        Ok(ResidentChunk {
            space: ResidentSpace {
                kind: 0,
                k0: first,
                n,
                ..ResidentSpace::default()
            },
            keys: unsafe { keys.finish() }?,
            fields,
        })
    }
}
impl ResidentProducer for UnaryProducer<'_> {
    fn key_dtype(&self) -> DeviceDType {
        DeviceDType::I64
    }
    fn fields(&self) -> &[ResidentFieldSpec] {
        &self.specs
    }
    fn passed(&self) -> Option<&DeviceColumn> {
        self.passed.as_ref()
    }
    fn next_chunk(&mut self, lease: &DeviceLease) -> Result<Option<ResidentChunk>, EngineError> {
        if self.stream != lease.stream_handle() {
            return Err(EngineError::Device(
                "unary producer changed its consumer stream".into(),
            ));
        }
        if self.next == self.n {
            return Ok(None);
        }
        let n = (self.n - self.next).min(self.chunk_cap);
        let chunk = self.stage(lease, self.next, n)?;
        self.next += n;
        Ok(Some(chunk))
    }
    fn reset(&mut self) -> Result<(), EngineError> {
        self.next = 0;
        Ok(())
    }
}
pub fn run(
    lease: &DeviceLease,
    table: &BoundDeviceRows,
    call: UnaryCall,
    fields: &[&str],
    sink: Sink,
    cutoff: Option<i64>,
    max_length: u64,
) -> Result<Vec<ResidentField>, EngineError> {
    if !matches!(sink, Sink::Each | Sink::Hits) {
        return Err(EngineError::Parameter(
            "unary resident calls support Each and Hits".into(),
        ));
    }
    let n = usize::try_from(table.rows.rows)
        .map_err(|_| EngineError::Device("unary input exceeds platform limits".into()))?;
    let mut producer = UnaryProducer::new(lease, table, call, fields, max_length, cutoff)?;
    let work = ResidentWork {
        space: ResidentSpace {
            kind: 0,
            n: n as u64,
            ..ResidentSpace::default()
        },
        n_a: n,
        n_b: n,
        original_items: n,
        mirror: false,
        owners: Vec::new(),
    };
    let config = ResidentSinkConfig {
        sink,
        cutoff,
        k: 0,
        mirror_symmetric: false,
        tile: None,
    };
    let mut result = if sink == Sink::Each && cutoff.is_none() {
        resident_sinks::each_fields(
            lease,
            &work,
            &mut producer,
            seqblaze_core::cancel::Stage::Sequences,
        )?
    } else {
        resident_sinks::run_staged(
            lease,
            &lease.device().0.resident_sinks,
            &work,
            &mut producer,
            &config,
            None,
            seqblaze_core::cancel::Stage::Sequences,
        )?
    };
    result
        .columns
        .retain(|field| field.name != "j" && !field.name.starts_with("__"));
    lease.synchronize()?;
    Ok(result.columns)
}
