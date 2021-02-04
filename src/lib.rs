use pyo3::prelude::*;
use pyo3::wrap_pyfunction;

use crate::annotator::PyAnnotator;
use crate::sentence::{PySentence, PySentenceIter, PyToken};

mod annotator;

mod sentence;

/// Query the version of SyntaxDot.
#[pyfunction]
pub fn syntaxdot_version() -> &'static str {
    syntaxdot::VERSION
}

#[pymodule]
fn syntaxdot(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(syntaxdot_version, m)?)?;
    m.add_class::<PyAnnotator>()?;
    m.add_class::<PySentence>()?;
    m.add_class::<PySentenceIter>()?;
    m.add_class::<PyToken>()?;
    Ok(())
}
