use pyo3::proc_macro::pymodule;
use pyo3::types::PyModule;
use pyo3::{PyResult, Python};

#[pymodule]
fn syntaxdot(_py: Python, _m: &PyModule) -> PyResult<()> {
    Ok(())
}
