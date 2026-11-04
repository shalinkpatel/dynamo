use std::path::Path;
use std::str::FromStr;
use std::sync::Arc;

use anyhow::{Result, anyhow, bail};

#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub enum Backend {
    HuggingFace,
    #[default]
    Fastokens,
}

impl FromStr for Backend {
    type Err = anyhow::Error;

    fn from_str(value: &str) -> Result<Self> {
        match value {
            "huggingface" => Ok(Self::HuggingFace),
            "fastokens" => Ok(Self::Fastokens),
            _ => bail!("unknown tokenizer backend '{value}'; expected huggingface or fastokens"),
        }
    }
}

// Both backends share the loaded vocabulary between corpus and replay workers.
#[derive(Clone)]
pub enum Tokenizer {
    HuggingFace(Arc<tokenizers::Tokenizer>),
    Fastokens(Arc<fastokens::Tokenizer>),
}

impl Tokenizer {
    pub fn from_file(path: &Path, backend: Backend) -> Result<Self> {
        match backend {
            Backend::HuggingFace => tokenizers::Tokenizer::from_file(path)
                .map(|tok| Self::HuggingFace(Arc::new(tok)))
                .map_err(|e| anyhow!("Hugging Face tokenizer load failed: {e}")),
            Backend::Fastokens => fastokens::Tokenizer::from_file(path)
                .map(|tok| Self::Fastokens(Arc::new(tok)))
                .map_err(|e| anyhow!("fastokens-b10 tokenizer load failed: {e}; try --tokenizer-backend huggingface")),
        }
    }

    pub fn encode(&self, text: &str) -> Result<Vec<u32>> {
        // Reconstruction uses raw completion text: do not insert BOS/EOS tokens.
        match self {
            Self::HuggingFace(tok) => tok
                .encode(text, false)
                .map(|encoding| encoding.get_ids().to_vec())
                .map_err(|e| anyhow!("tokenizer encode failed: {e}")),
            Self::Fastokens(tok) => tok
                .encode_with_special_tokens(text, false)
                .map_err(|e| anyhow!("tokenizer encode failed: {e}")),
        }
    }

    pub fn decode(&self, ids: &[u32]) -> Result<String> {
        // Keep special tokens already present in the corpus, matching the original path.
        match self {
            Self::HuggingFace(tok) => tok
                .decode(ids, false)
                .map_err(|e| anyhow!("tokenizer decode failed: {e}")),
            Self::Fastokens(tok) => tok
                .decode(ids, false)
                .map_err(|e| anyhow!("tokenizer decode failed: {e}")),
        }
    }
}
