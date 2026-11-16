//! Build without feature `cuda`: the same API. `device()` always fails, so
//! these types are never constructed; their methods fail the same way.
use seqblaze_core::{
    detail::DetailColumns,
    kernel::{BatchDistance, CellClass, DistanceVisit, PairDistanceKernel},
    sched::Workers,
    seq::SeqTable,
    validate::EngineError,
    work::WorkSource,
};

use crate::{DeviceCall, DeviceInfo};

/// Never constructed in this build.
#[derive(Clone, Debug)]
pub struct Device {
    info: DeviceInfo,
}

impl Device {
    pub fn info(&self) -> &DeviceInfo {
        &self.info
    }

    /// Never reached: this build constructs no device.
    pub fn p3_wave(&self) -> [u64; 2] {
        [0, 0]
    }

    /// Never reached: this build constructs no device.
    pub fn resident_wave(
        &self,
        _kernel: usize,
        _staged: u64,
        _shared_band: u64,
    ) -> Result<u64, EngineError> {
        Err(unavailable(self.info.ordinal))
    }
}

/// Never constructed in this build.
pub struct TileSession {
    ordinal: usize,
}

impl TileSession {
    pub fn new(
        device: &Device,
        _call: DeviceCall,
        _tables: (&SeqTable, &SeqTable),
        _workers: Workers,
    ) -> Result<Self, EngineError> {
        Err(unavailable(device.info.ordinal))
    }

    pub fn bind<'s>(
        &'s mut self,
        _kernel: &'s dyn PairDistanceKernel,
        _tables: (&'s SeqTable, &'s SeqTable),
    ) -> TileBatch<'s> {
        TileBatch {
            ordinal: self.ordinal,
            _session: std::marker::PhantomData,
        }
    }

    pub fn bind_pf<'s>(
        &'s mut self,
        _kernel: &'s dyn PairDistanceKernel<i64>,
        _tables: (&'s SeqTable, &'s SeqTable),
    ) -> PfTileBatch<'s> {
        Batch64 {
            ordinal: self.ordinal,
            _session: std::marker::PhantomData,
        }
    }

    pub fn bind_i64<'s>(
        &'s mut self,
        _kernel: &'s dyn PairDistanceKernel<i64>,
        _tables: (&'s SeqTable, &'s SeqTable),
    ) -> TileBatch64<'s> {
        Batch64 {
            ordinal: self.ordinal,
            _session: std::marker::PhantomData,
        }
    }

    pub fn host_pairs(&self) -> usize {
        0
    }
}

/// K3's (`bind_pf`) and the int64-key (`bind_i64`: `tm` on Pairs, K2 with the primer3
/// preset) tile batches: one stub type in this build, two port types in a CUDA build.
pub type PfTileBatch<'s> = Batch64<'s>;
pub type TileBatch64<'s> = Batch64<'s>;

/// Never constructed in this build.
pub struct Batch64<'s> {
    ordinal: usize,
    _session: std::marker::PhantomData<&'s mut TileSession>,
}

impl BatchDistance<i64> for Batch64<'_> {
    fn delivers_details(&self) -> bool {
        false
    }
    fn range(
        &mut self,
        _: &WorkSource,
        _: usize,
        _: usize,
        _: bool,
        _: &mut DistanceVisit<'_, i64>,
    ) -> Result<(), EngineError> {
        Err(unavailable(self.ordinal))
    }
    fn pairs(
        &mut self,
        _: &[(usize, usize)],
        _: &mut Vec<i64>,
        _: Option<&mut DetailColumns>,
    ) -> Result<(), EngineError> {
        Err(unavailable(self.ordinal))
    }
    fn rect(
        &mut self,
        _: (usize, usize),
        _: (usize, usize),
        _: &CellClass<'_>,
        _: &mut Vec<i64>,
        _: Option<&mut DetailColumns>,
    ) -> Result<(), EngineError> {
        Err(unavailable(self.ordinal))
    }
}

/// Never constructed in this build.
pub struct TileBatch<'s> {
    ordinal: usize,
    _session: std::marker::PhantomData<&'s mut TileSession>,
}

impl BatchDistance for TileBatch<'_> {
    fn delivers_details(&self) -> bool {
        false
    }
    fn range(
        &mut self,
        _: &WorkSource,
        _: usize,
        _: usize,
        _: bool,
        _: &mut DistanceVisit<'_>,
    ) -> Result<(), EngineError> {
        Err(unavailable(self.ordinal))
    }
    fn pairs(
        &mut self,
        _: &[(usize, usize)],
        _: &mut Vec<i32>,
        _: Option<&mut DetailColumns>,
    ) -> Result<(), EngineError> {
        Err(unavailable(self.ordinal))
    }
    fn rect(
        &mut self,
        _: (usize, usize),
        _: (usize, usize),
        _: &CellClass<'_>,
        _: &mut Vec<i32>,
        _: Option<&mut DetailColumns>,
    ) -> Result<(), EngineError> {
        Err(unavailable(self.ordinal))
    }
}

pub(crate) fn unavailable(ordinal: usize) -> EngineError {
    EngineError::Device(format!(
        "device='cuda:{ordinal}' is unavailable: this seqblaze build has no CUDA backend \
         (built without CUDA; build_info()['cuda_compiled'] is False); use device='cpu', or \
         a build with the cuda feature (make build DEV_FEATURES=testing,cuda)"
    ))
}
