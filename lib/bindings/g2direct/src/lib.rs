use pyo3::prelude::*;
use std::fmt::Display;
use std::sync::OnceLock;

mod hicache;

type RuntimeProvider = fn() -> tokio::runtime::Handle;
static RUNTIME_PROVIDER: OnceLock<RuntimeProvider> = OnceLock::new();

pub fn add_to_module(
    module: &Bound<'_, PyModule>,
    runtime_provider: RuntimeProvider,
) -> PyResult<()> {
    let current = RUNTIME_PROVIDER.get_or_init(|| runtime_provider);
    if !std::ptr::fn_addr_eq(*current, runtime_provider) {
        return Err(pyo3::exceptions::PyRuntimeError::new_err(
            "G2Direct is already attached to a different runtime provider",
        ));
    }
    hicache::add_to_module(module)
}

fn get_current_tokio_handle() -> tokio::runtime::Handle {
    RUNTIME_PROVIDER
        .get()
        .expect("G2Direct module must be initialized before creating a transport")()
}

fn to_pyerr(error: impl Display) -> PyErr {
    pyo3::exceptions::PyException::new_err(error.to_string())
}

#[cfg(feature = "python-extension")]
fn standalone_tokio_handle() -> tokio::runtime::Handle {
    static RUNTIME: OnceLock<tokio::runtime::Runtime> = OnceLock::new();
    RUNTIME
        .get_or_init(|| {
            let config = dynamo_runtime::RuntimeConfig::from_settings()
                .expect("failed to build G2Direct runtime config");
            tokio::runtime::Builder::new_multi_thread()
                .worker_threads(
                    config
                        .num_worker_threads
                        .unwrap_or_else(|| std::thread::available_parallelism().unwrap().get()),
                )
                .max_blocking_threads(config.max_blocking_threads)
                .enable_all()
                .build()
                .expect("failed to build G2Direct Tokio runtime")
        })
        .handle()
        .clone()
}

#[cfg(feature = "python-extension")]
#[pymodule]
fn _core(module: &Bound<'_, PyModule>) -> PyResult<()> {
    let handle = standalone_tokio_handle();
    let _guard = handle.enter();
    dynamo_runtime::logging::init();
    add_to_module(module, standalone_tokio_handle)
}
