//! Resident pair geometry. Only explicit caller-supplied indices and O(n)
//! triangular row segments are uploaded; implicit pair products are never built.
use crate::{
    interop::{DeviceColumn, DeviceDType, DeviceLease},
    resident::{ResidentSpace, ResidentWork},
};
use seqblaze_core::{
    sink::Sink,
    validate::EngineError,
    work::{Orientation, PairKind, WorkSource},
};

fn reserve<T>(n: usize) -> Result<Vec<T>, EngineError> {
    let mut values = Vec::new();
    seqblaze_core::detail::try_reserve_exact(&mut values, n, "resident pair controls")?;
    Ok(values)
}
fn upload_i64(lease: &DeviceLease, values: &[i64]) -> Result<DeviceColumn, EngineError> {
    let mut bytes = reserve(
        values
            .len()
            .checked_mul(8)
            .ok_or_else(|| EngineError::Device("resident index bytes overflow".into()))?,
    )?;
    for &value in values {
        bytes.extend_from_slice(&value.to_le_bytes());
    }
    lease.upload_bytes(DeviceDType::I64, vec![values.len()], &bytes)
}
fn upload_u64(lease: &DeviceLease, values: &[u64]) -> Result<DeviceColumn, EngineError> {
    let mut bytes = reserve(
        values
            .len()
            .checked_mul(8)
            .ok_or_else(|| EngineError::Device("resident segment bytes overflow".into()))?,
    )?;
    for &value in values {
        bytes.extend_from_slice(&value.to_le_bytes());
    }
    lease.upload_bytes(DeviceDType::U64, vec![values.len()], &bytes)
}

pub fn from_host_work(
    lease: &DeviceLease,
    work: &WorkSource,
    sink: Sink,
) -> Result<ResidentWork, EngineError> {
    let mut space = ResidentSpace {
        n: work.n_pairs as u64,
        ..ResidentSpace::default()
    };
    let mut owners = reserve(3)?;
    match &work.kind {
        PairKind::Zip => space.kind = 0,
        PairKind::Cross | PairKind::OneToMany => {
            space.kind = 2;
            space.width = work.shape.1 as u64;
        }
        PairKind::WithinOriented {
            self_pairs,
            orientation: Orientation::Both,
        } => {
            space.kind = 4;
            space.skip = u64::from(!*self_pairs);
            space.width = work.shape.0.saturating_sub(usize::from(!*self_pairs)) as u64;
        }
        PairKind::Within { .. }
        | PairKind::WithinOriented {
            orientation: Orientation::AsGiven,
            ..
        } => {
            space.kind = 3;
            let mut k = reserve(work.shape.0)?;
            let mut i = reserve(work.shape.0)?;
            let mut j = reserve(work.shape.0)?;
            for (row, column, first) in work
                .row_segment_iter(0, work.n_pairs)
                .expect("triangular kind")
            {
                k.push(first as u64);
                i.push(row as i64);
                j.push(column as i64);
            }
            space.nseg = k.len() as u64;
            let dk = upload_u64(lease, &k)?;
            let di = upload_i64(lease, &i)?;
            let dj = upload_i64(lease, &j)?;
            space.seg_k = dk.address();
            space.seg_i = di.address();
            space.seg_j = dj.address();
            owners.extend([dk, di, dj]);
        }
        PairKind::Index { i, j } => {
            space.kind = 1;
            let mut pairs = reserve(i.len())?;
            pairs.extend((0..i.len()).map(|at| (i.get(at) as i64, j.get(at) as i64)));
            if sink != Sink::Each {
                pairs.sort_unstable();
                pairs.dedup();
            }
            let mut left = reserve(pairs.len())?;
            let mut right = reserve(pairs.len())?;
            for (i, j) in pairs {
                left.push(i);
                right.push(j);
            }
            space.n = left.len() as u64;
            let li = upload_i64(lease, &left)?;
            let lj = upload_i64(lease, &right)?;
            space.li = li.address();
            space.lj = lj.address();
            owners.extend([li, lj]);
        }
    }
    Ok(ResidentWork {
        space,
        n_a: work.shape.0,
        n_b: work.shape.1,
        original_items: work.n_pairs,
        mirror: work.mirrors_within(),
        owners,
    })
}

/// Use already normalized, call-owned device indices. No table payload or index
/// values are downloaded; original occurrence count survives canonicalization.
pub fn from_device_indices(
    lease: &mut DeviceLease,
    i: &DeviceColumn,
    j: &DeviceColumn,
    shape: [usize; 2],
    sink: Sink,
) -> Result<ResidentWork, EngineError> {
    crate::resident_indices::validate_pair_indices(lease, i, j, shape)?;
    let original_items = i.shape()[0];
    let (i, j) = if sink == Sink::Each {
        (i.clone(), j.clone())
    } else {
        crate::resident_indices::canonical_pairs(lease, i, j)?
    };
    Ok(ResidentWork {
        space: ResidentSpace {
            kind: 1,
            n: i.shape()[0] as u64,
            li: i.address(),
            lj: j.address(),
            ..ResidentSpace::default()
        },
        n_a: shape[0],
        n_b: shape[1],
        original_items,
        mirror: false,
        owners: vec![i, j],
    })
}
