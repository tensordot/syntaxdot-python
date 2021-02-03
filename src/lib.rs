use pyo3::proc_macro::pymodule;
use pyo3::types::PyModule;
use pyo3::{PyResult, Python};

use crate::annotator::PyAnnotator;
use crate::sentence::{PySentence, PySentenceIter, PyToken};

mod annotator;

mod sentence;

#[pymodule]
fn syntaxdot(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_class::<PyAnnotator>()?;
    m.add_class::<PySentence>()?;
    m.add_class::<PySentenceIter>()?;
    m.add_class::<PyToken>()?;
    Ok(())
}
