use std::sync::Arc;

use conllu::graph::DepTriple;
use conllu::token::Token;
use pyo3::exceptions::PyIndexError;
use pyo3::prelude::*;
use pyo3::{PyIterProtocol, PyObjectProtocol, PySequenceProtocol};
use std::collections::BTreeMap;
use std::ops::Deref;
use syntaxdot_tokenizers::SentenceWithPieces;

#[pyclass(name = "Sentence")]
pub struct PySentence {
    inner: Arc<SentenceWithPieces>,
}

#[pyproto]
impl PyIterProtocol for PySentence {
    fn __iter__(slf: PyRef<Self>) -> PyResult<Py<PySentenceIter>> {
        let iter = PySentenceIter {
            sentence: slf.inner.clone(),
            idx: 1,
        };

        Py::new(slf.py(), iter)
    }
}

#[pyproto]
impl PySequenceProtocol for PySentence {
    fn __len__(&self) -> usize {
        self.inner.sentence.len() - 1
    }

    fn __getitem__(&self, idx: isize) -> PyResult<PyToken> {
        let len_without_root = self.inner.sentence.len() as isize - 1;

        let idx = if idx < 0 { len_without_root + idx } else { idx };

        if idx >= 0 && idx < len_without_root {
            Ok(PyToken {
                sentence: self.inner.clone(),
                idx: idx as usize + 1,
            })
        } else {
            Err(PyIndexError::new_err("token index out of range"))
        }
    }
}

impl From<SentenceWithPieces> for PySentence {
    fn from(sentence: SentenceWithPieces) -> Self {
        PySentence {
            inner: Arc::new(sentence),
        }
    }
}

#[pyclass(name = "SentenceIter")]
pub struct PySentenceIter {
    sentence: Arc<SentenceWithPieces>,
    idx: usize,
}

#[pyproto]
impl PyIterProtocol for PySentenceIter {
    fn __iter__(slf: PyRef<Self>) -> PyRef<Self> {
        slf
    }

    fn __next__(mut slf: PyRefMut<Self>) -> Option<PyToken> {
        if slf.idx < slf.sentence.sentence.len() {
            let token = PyToken {
                sentence: slf.sentence.clone(),
                idx: slf.idx,
            };
            slf.idx += 1;
            Some(token)
        } else {
            None
        }
    }
}

#[pyclass(name = "Token")]
pub struct PyToken {
    sentence: Arc<SentenceWithPieces>,
    idx: usize,
}

impl PyToken {
    fn dep_triple(&self) -> Option<DepTriple<&str>> {
        self.sentence.sentence.dep_graph().head(self.idx)
    }

    fn token(&self) -> &Token {
        self.sentence.as_ref().sentence[self.idx].token().unwrap()
    }
}

#[pymethods]
impl PyToken {
    fn form(&self) -> &str {
        self.token().form()
    }

    #[getter]
    fn get_features(&self) -> BTreeMap<String, String> {
        self.token().features().deref().to_owned()
    }

    #[getter]
    fn get_head(&self) -> Option<usize> {
        self.dep_triple().map(|triple| triple.head())
    }

    #[getter]
    fn get_relation(&self) -> Option<String> {
        // XXX: we convert to an owned string here, because the lifetime of
        // relation is bound to the triple. But it shouldn't.
        self.dep_triple()
            .and_then(|triple| triple.relation().map(ToOwned::to_owned))
    }

    #[getter]
    fn get_upos(&self) -> Option<&str> {
        self.token().upos()
    }

    #[getter]
    fn get_xpos(&self) -> Option<&str> {
        self.token().xpos()
    }
}

#[pyproto]
impl PyObjectProtocol<'_> for PyToken {
    fn __repr__(&self) -> PyResult<String> {
        let token = self.token();
        Ok(format!("Token(form=\"{}\")", token.form()))
    }
}
