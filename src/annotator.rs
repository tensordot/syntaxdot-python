use std::fs::File;
use std::io;
use std::io::BufReader;
use std::path::Path;

use pyo3::exceptions::{PyIOError, PyRuntimeError};
use pyo3::prelude::*;
use pyo3::PyObjectProtocol;
use syntaxdot::config::{BiaffineParserConfig, Config, PretrainConfig, TomlRead};
use syntaxdot::encoders::Encoders;
use syntaxdot::error::SyntaxDotError;
use syntaxdot::model::bert::BertModel;
use syntaxdot::tagger::Tagger;
use syntaxdot_encoders::dependency::ImmutableDependencyEncoder;
use syntaxdot_tch_ext::RootExt;
use syntaxdot_tokenizers::{SentenceWithPieces, Tokenize};
use syntaxdot_transformers::TransformerError;
use tch::nn::VarStore;
use tch::Device;
use thiserror::Error;
use udgraph::graph::Sentence;
use udgraph::token::Token;

use crate::sentence::PySentence;

#[pyclass(name = "Annotator")]
pub struct PyAnnotator {
    config_path: String,
    tagger: Tagger,
    tokenizer: Box<dyn Tokenize>,
}

#[pymethods]
impl PyAnnotator {
    #[new]
    fn new(config_path: &str) -> PyResult<Self> {
        Ok(Self::load(Device::Cpu, config_path)?)
    }

    #[args(batch_size = "32")]
    fn annotate(
        &self,
        sentences: Vec<Vec<String>>,
        batch_size: usize,
    ) -> PyResult<Vec<PySentence>> {
        let sentences = sentences
            .into_iter()
            .map(|sentence| sentence.into_iter().map(Token::new).collect::<Sentence>())
            .collect::<Vec<_>>();

        let annotated_sentences = self.annotate_sentences(sentences, batch_size)?;

        Ok(annotated_sentences
            .into_iter()
            .map(PySentence::from)
            .collect())
    }
}

#[pyproto]
impl PyObjectProtocol<'_> for PyAnnotator {
    fn __repr__(&self) -> PyResult<String> {
        Ok(format!("Annotator(\"{}\")", self.config_path))
    }
}

impl PyAnnotator {
    fn load<P>(device: Device, config_path: P) -> PyResult<Self>
    where
        P: AsRef<Path>,
    {
        let r = BufReader::new(File::open(&config_path).map_err(|err| {
            PyIOError::new_err(format!(
                "Cannot open syntaxdot config file `{}`: {}",
                config_path.as_ref().to_string_lossy(),
                err
            ))
        })?);
        let mut config = Config::from_toml_read(r).map_err(|err| {
            PyIOError::new_err(format!(
                "Cannot read configuration from {}: {}",
                config_path.as_ref().to_string_lossy(),
                err
            ))
        })?;
        config.relativize_paths(&config_path).map_err(|err| {
            PyIOError::new_err(format!(
                "Cannot relativize configuration file paths: {}",
                err
            ))
        })?;

        let biaffine_decoder = config
            .biaffine
            .as_ref()
            .map(|config| load_biaffine_decoder(config))
            .transpose()?;
        let encoders = load_encoders(&config)
            .map_err(|err| PyIOError::new_err(format!("Cannot load encoders: {}", err)))?;
        let tokenizer = load_tokenizer(&config)
            .map_err(|err| PyIOError::new_err(format!("Cannot load tokenizer: {}", err)))?;
        let pretrain_config = load_pretrain_config(&config).map_err(|err| {
            PyIOError::new_err(format!(
                "Cannot pretrain config from {}: {}",
                config.model.pretrain_config, err
            ))
        })?;

        let mut vs = VarStore::new(device);

        let model = BertModel::new(
            vs.root_ext(|_| 0),
            &pretrain_config,
            config.biaffine.as_ref(),
            biaffine_decoder
                .as_ref()
                .map(ImmutableDependencyEncoder::n_relations)
                .unwrap_or(0),
            &encoders,
            config.model.pooler,
            0.0,
            config.model.position_embeddings.clone(),
        )
        .map_err(|err| PyRuntimeError::new_err(format!("Cannot construct BERT model: {}", err)))?;

        vs.load(&config.model.parameters).map_err(|err| {
            PyRuntimeError::new_err(format!(
                "Cannot load model parameters from {}: {}",
                config.model.parameters, err
            ))
        })?;

        vs.freeze();

        let tagger = Tagger::new(device, model, biaffine_decoder, encoders);

        Ok(PyAnnotator {
            config_path: config_path.as_ref().to_string_lossy().into_owned(),
            tagger,
            tokenizer,
        })
    }

    pub fn annotate_sentences(
        &self,
        sentences: impl IntoIterator<Item = Sentence>,
        batch_size: usize,
    ) -> PyResult<Vec<SentenceWithPieces>> where {
        let mut sentences_with_pieces = sentences
            .into_iter()
            .map(|s| self.tokenizer.tokenize(s))
            .collect::<Vec<_>>();

        // Sort sentences by length.
        let mut sent_refs: Vec<_> = sentences_with_pieces.iter_mut().collect();
        sent_refs.sort_unstable_by_key(|s| s.pieces.len());

        // Split in batches, tag, and merge results.
        for batch in sent_refs.chunks_mut(batch_size) {
            self.tagger.tag_sentences(batch).map_err(|err| {
                PyRuntimeError::new_err(format!("Cannot annotate sentences: {}", err))
            })?;
        }

        Ok(sentences_with_pieces)
    }
}

#[derive(Debug, Error)]
pub enum AnnotatorError {
    #[error("Cannot construct BERT model: {0}")]
    Transformer(#[from] TransformerError),

    #[error("{0}: {1}")]
    IO(String, io::Error),

    #[error("Cannot deserialize encoders from `{0}`: {1}")]
    LoadEncoders(String, serde_yaml::Error),

    #[error("Cannot load model parameters: {0}")]
    LoadParameters(#[from] tch::TchError),

    #[error(transparent)]
    SyntaxDot(#[from] SyntaxDotError),
}

pub fn load_pretrain_config(config: &Config) -> Result<PretrainConfig, AnnotatorError> {
    Ok(config.model.pretrain_config()?)
}

fn load_biaffine_decoder(config: &BiaffineParserConfig) -> PyResult<ImmutableDependencyEncoder> {
    let f = File::open(&config.labels).map_err(|err| {
        PyIOError::new_err(format!(
            "Cannot open biaffine label file {}: {}",
            config.labels, err
        ))
    })?;

    let encoder: ImmutableDependencyEncoder = serde_yaml::from_reader(&f).map_err(|err| {
        PyIOError::new_err(format!(
            "Cannot load biaffine parser labels from {}: {}",
            config.labels.clone(),
            err
        ))
    })?;

    Ok(encoder)
}

fn load_encoders(config: &Config) -> Result<Encoders, AnnotatorError> {
    let f = File::open(&config.labeler.labels).map_err(|err| {
        AnnotatorError::IO(
            format!("Cannot open label file: {}", config.labeler.labels),
            err,
        )
    })?;

    Ok(serde_yaml::from_reader(&f)
        .map_err(|err| AnnotatorError::LoadEncoders(config.labeler.labels.clone(), err))?)
}

pub fn load_tokenizer(config: &Config) -> Result<Box<dyn Tokenize>, AnnotatorError> {
    Ok(config.tokenizer()?)
}
