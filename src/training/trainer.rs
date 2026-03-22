use crate::tokenizer::tokenizer::{Tokenizer, TokenizerError};
use std::error::Error;
use std::fmt::{Display, Formatter};
use std::fs;
use std::io;
use std::path::{Path, PathBuf};

#[derive(Debug)]
pub enum TrainerError {
    Io { path: PathBuf, source: io::Error },
    Tokenizer(TokenizerError),
    InvalidBlockSize(usize),
}

impl Display for TrainerError {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Io { path, source } => {
                write!(
                    f,
                    "failed to read dataset file '{}': {source}",
                    path.display()
                )
            }
            Self::Tokenizer(source) => write!(f, "tokenizer error: {source}"),
            Self::InvalidBlockSize(size) => {
                write!(f, "invalid block size {size}: block size must be > 0")
            }
        }
    }
}

impl Error for TrainerError {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        match self {
            Self::Io { source, .. } => Some(source),
            Self::Tokenizer(source) => Some(source),
            Self::InvalidBlockSize(_) => None,
        }
    }
}

impl From<TokenizerError> for TrainerError {
    fn from(value: TokenizerError) -> Self {
        Self::Tokenizer(value)
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TrainingSample {
    pub x: Vec<usize>,
    pub y: Vec<usize>,
}

#[derive(Debug, Clone)]
pub struct Dataset {
    raw_text: String,
    tokenizer: Tokenizer,
    token_ids: Vec<usize>,
}

impl Dataset {
    pub fn from_text(text: impl Into<String>) -> Result<Self, TrainerError> {
        let raw_text = text.into();
        let tokenizer = Tokenizer::build_from_text(&raw_text);
        let token_ids = tokenizer.encode(&raw_text)?;

        Ok(Self {
            raw_text,
            tokenizer,
            token_ids,
        })
    }

    pub fn from_file(path: impl AsRef<Path>) -> Result<Self, TrainerError> {
        let path = path.as_ref();
        let raw_text = load_text_file(path)?;
        Self::from_text(raw_text)
    }

    pub fn char_count(&self) -> usize {
        self.raw_text.chars().count()
    }

    pub fn token_count(&self) -> usize {
        self.token_ids.len()
    }

    pub fn vocab_size(&self) -> usize {
        self.tokenizer.vocab_size()
    }

    pub fn tokenizer(&self) -> &Tokenizer {
        &self.tokenizer
    }

    pub fn token_ids(&self) -> &[usize] {
        &self.token_ids
    }

    pub fn decode_prefix(&self, max_tokens: usize) -> Result<String, TrainerError> {
        let end = max_tokens.min(self.token_ids.len());
        Ok(self.tokenizer.decode(&self.token_ids[..end])?)
    }

    pub fn sample_count(&self, block_size: usize) -> usize {
        if block_size == 0 || self.token_ids.len() <= block_size {
            return 0;
        }
        self.token_ids.len() - block_size
    }

    pub fn make_training_samples(
        &self,
        block_size: usize,
    ) -> Result<Vec<TrainingSample>, TrainerError> {
        if block_size == 0 {
            return Err(TrainerError::InvalidBlockSize(block_size));
        }

        let count = self.sample_count(block_size);
        let mut samples = Vec::with_capacity(count);

        for start in 0..count {
            let end = start + block_size + 1;
            let window = &self.token_ids[start..end];
            samples.push(TrainingSample {
                x: window[..block_size].to_vec(),
                y: window[1..].to_vec(),
            });
        }

        Ok(samples)
    }
}

#[derive(Debug, Clone, Copy)]
pub struct Trainer {
    block_size: usize,
}

impl Trainer {
    pub fn new(block_size: usize) -> Result<Self, TrainerError> {
        if block_size == 0 {
            return Err(TrainerError::InvalidBlockSize(block_size));
        }

        Ok(Self { block_size })
    }

    pub fn block_size(&self) -> usize {
        self.block_size
    }

    pub fn build_samples(&self, dataset: &Dataset) -> Result<Vec<TrainingSample>, TrainerError> {
        dataset.make_training_samples(self.block_size)
    }
}

pub fn load_text_file(path: impl AsRef<Path>) -> Result<String, TrainerError> {
    let path = path.as_ref();
    fs::read_to_string(path).map_err(|source| TrainerError::Io {
        path: path.to_path_buf(),
        source,
    })
}

#[cfg(test)]
mod tests {
    use super::{Dataset, Trainer, TrainerError};

    #[test]
    fn dataset_from_text_builds_tokens() {
        let dataset = Dataset::from_text("hello\n").expect("dataset should build");

        assert_eq!(dataset.char_count(), 6);
        assert_eq!(dataset.token_count(), 6);
        assert!(dataset.vocab_size() >= 5);
    }

    #[test]
    fn trainer_builds_shifted_samples() {
        let dataset = Dataset::from_text("abcd").expect("dataset should build");
        let trainer = Trainer::new(2).expect("trainer should build");

        let samples = trainer
            .build_samples(&dataset)
            .expect("samples should be generated");

        assert_eq!(samples.len(), 2);
        assert_eq!(samples[0].x, dataset.token_ids()[0..2].to_vec());
        assert_eq!(samples[0].y, dataset.token_ids()[1..3].to_vec());
        assert_eq!(samples[1].x, dataset.token_ids()[1..3].to_vec());
        assert_eq!(samples[1].y, dataset.token_ids()[2..4].to_vec());
    }

    #[test]
    fn invalid_block_size_returns_error() {
        let err = Trainer::new(0).expect_err("zero block size must fail");
        assert!(matches!(err, TrainerError::InvalidBlockSize(0)));
    }

    #[test]
    fn decode_prefix_respects_limit() {
        let dataset = Dataset::from_text("abcdef").expect("dataset should build");
        let decoded = dataset.decode_prefix(3).expect("decode should work");

        assert_eq!(decoded.chars().count(), 3);
    }
}
