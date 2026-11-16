//! Python extension module `seqblaze._core`.

use pyo3::prelude::*;
use pyo3::types::PyDict;

mod runtime;

/// Describe this build: version, SIMD level detected at runtime, compiled backends.
#[pyfunction]
fn build_info(py: Python<'_>) -> PyResult<Bound<'_, PyDict>> {
    let info = seqblaze_core::build_info();
    let d = PyDict::new(py);
    d.set_item("version", info.version)?;
    d.set_item("git", info.git)?;
    d.set_item("git_dirty", info.git_dirty)?;
    d.set_item("testing", cfg!(feature = "testing"))?;
    d.set_item("target_arch", info.target_arch)?;
    d.set_item("simd", info.simd)?;
    // Core cannot see the CUDA crate; the composition root reports it.
    d.set_item(
        "cuda_compiled",
        info.cuda_compiled || seqblaze_cuda::COMPILED,
    )?;
    d.set_item("nvcc_version", seqblaze_cuda::nvcc_version())?;
    d.set_item("nvcc_flags", seqblaze_cuda::nvcc_flags())?;
    d.set_item("debug_assertions", info.debug_assertions)?;
    Ok(d)
}

#[pyfunction]
fn _available_threads() -> usize {
    std::thread::available_parallelism().map_or(1, usize::from)
}

#[pymodule]
fn _core(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add("__version__", seqblaze_core::VERSION)?;
    m.add_function(wrap_pyfunction!(build_info, m)?)?;
    runtime::register(m)?;
    m.add_function(wrap_pyfunction!(_available_threads, m)?)?;
    #[cfg(feature = "testing")]
    {
        for suffix in [
            "seqs",
            "seqs_numpy",
            "seqs_arrow",
            "seqs_arrow_stream",
            "indices",
            "run",
            "tiles",
            "cuda_launch",
            "cuda_poison",
            "cuda_poisoned_bytes",
            "cuda_resident_poison_stats",
            "cuda_resident_depth_cap",
            "cuda_p3_wave",
            "cuda_resident_launch",
            "cuda_resident_wave",
            "cuda_pf_capture",
            "cuda_unary_refill_stats",
            "tm_ln",
            "cuda_f64",
            "duplex_pack",
            "duplex_evaluate",
            "duplex_pf_scaled",
            "duplex_pf_weight",
            "duplex_pf_log",
            "duplex_pf_zhat",
            "duplex_bound",
            "duplex_bound_certificate",
            "duplex_bound_fault",
            "duplex_bound_evals",
            "duplex_bound_certificate_builds",
            "duplex_bound_device_costs",
            "duplex_pf_bound",
            "duplex_pf_bound_certificate",
            "cuda_f53",
            "primer3_device_plan",
            "alloc_fault",
            "alloc_faults_fired",
            "align_lanes",
            "edit_lanes",
            "cancel_probe",
            "cancel_probe_state",
            "cancel_probe_release",
        ] {
            m.add(
                format!("_testing_{suffix}"),
                m.getattr(format!("_native_{suffix}"))?,
            )?;
        }
        #[cfg(feature = "cuda")]
        for suffix in [
            "resident_launch_streams",
            "resident_launch_streams_reset",
            "cuda_pool_stats",
            "cuda_register_defect",
            "cuda_pool_trim",
            "cuda_pool_fail",
            "cuda_pool_fails_fired",
            "cuda_pool_fallback",
            "cuda_device_synchronize",
            "cuda_wait_stats",
            "cuda_wait_stats_reset",
            "cuda_stream_misuse_count",
        ] {
            m.add(
                format!("_testing_{suffix}"),
                m.getattr(format!("_native_{suffix}"))?,
            )?;
        }
        m.add("_testing_enabled", true)?;
    }
    Ok(())
}
