use std::collections::{BTreeSet, HashMap};
use std::error::Error;
use std::fmt::{Display, Formatter};

const DEFAULT_BOS_TOKEN: char = '^';

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TokenizerError {
    UnknownChar(char),
    UnknownId(usize),
}

impl Display for TokenizerError {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::UnknownChar(ch) => {
                write!(f, "character '{ch}' not found in tokenizer vocabulary")
            }
            Self::UnknownId(id) => write!(f, "token id {id} not found in tokenizer vocabulary"),
        }
    }
}

impl Error for TokenizerError {}

#[derive(Debug, Clone)]
pub struct Tokenizer {
    stoi: HashMap<char, usize>,
    itos: Vec<char>,
    bos_id: usize,
    bos_token: char,
}

impl Tokenizer {
    pub fn build_from_text(text: &str) -> Self {
        Self::build_from_text_with_bos(text, DEFAULT_BOS_TOKEN)
    }

    pub fn build_from_text_with_bos(text: &str, bos_token: char) -> Self {
        let mut symbols = BTreeSet::new();
        for ch in text.chars() {
            symbols.insert(ch);
        }
        symbols.insert(bos_token);

        let itos: Vec<char> = symbols.into_iter().collect();
        let mut stoi = HashMap::with_capacity(itos.len());
        for (idx, ch) in itos.iter().copied().enumerate() {
            stoi.insert(ch, idx);
        }

        let bos_id = *stoi
            .get(&bos_token)
            .expect("BOS token must exist in vocabulary");

        Self {
            stoi,
            itos,
            bos_id,
            bos_token,
        }
    }

    pub fn encode(&self, text: &str) -> Result<Vec<usize>, TokenizerError> {
        let mut out = Vec::with_capacity(text.chars().count());
        for ch in text.chars() {
            let id = self
                .stoi
                .get(&ch)
                .copied()
                .ok_or(TokenizerError::UnknownChar(ch))?;
            out.push(id);
        }
        Ok(out)
    }

    pub fn encode_with_bos(&self, text: &str) -> Result<Vec<usize>, TokenizerError> {
        let mut out = Vec::with_capacity(text.chars().count() + 1);
        out.push(self.bos_id);
        out.extend(self.encode(text)?);
        Ok(out)
    }

    pub fn decode(&self, ids: &[usize]) -> Result<String, TokenizerError> {
        let mut out = String::with_capacity(ids.len());
        for &id in ids {
            let ch = self
                .itos
                .get(id)
                .copied()
                .ok_or(TokenizerError::UnknownId(id))?;
            out.push(ch);
        }
        Ok(out)
    }

    pub fn decode_without_bos(&self, ids: &[usize]) -> Result<String, TokenizerError> {
        let mut out = String::with_capacity(ids.len());
        for &id in ids {
            if id == self.bos_id {
                continue;
            }
            let ch = self
                .itos
                .get(id)
                .copied()
                .ok_or(TokenizerError::UnknownId(id))?;
            out.push(ch);
        }
        Ok(out)
    }

    pub fn vocab_size(&self) -> usize {
        self.itos.len()
    }

    pub fn bos_id(&self) -> usize {
        self.bos_id
    }

    pub fn bos_token(&self) -> char {
        self.bos_token
    }

    pub fn token_id(&self, ch: char) -> Option<usize> {
        self.stoi.get(&ch).copied()
    }

    pub fn token_for_id(&self, id: usize) -> Option<char> {
        self.itos.get(id).copied()
    }

    pub fn vocabulary(&self) -> &[char] {
        &self.itos
    }
}

#[cfg(test)]
mod tests {
    use super::{Tokenizer, TokenizerError};

    #[test]
    fn tokenizer_includes_bos_token() {
        let tokenizer = Tokenizer::build_from_text("hello");

        assert_eq!(tokenizer.token_for_id(tokenizer.bos_id()), Some('^'));
        assert!(tokenizer.vocab_size() >= 5);
    }

    #[test]
    fn encode_decode_roundtrip() {
        let tokenizer = Tokenizer::build_from_text("abc\n");
        let text = "cab\n";

        let ids = tokenizer.encode(text).expect("encode should work");
        let decoded = tokenizer.decode(&ids).expect("decode should work");

        assert_eq!(decoded, text);
    }

    #[test]
    fn encode_with_bos_prefixes_sequence() {
        let tokenizer = Tokenizer::build_from_text("abc");

        let ids = tokenizer
            .encode_with_bos("cab")
            .expect("encode_with_bos should work");

        assert_eq!(ids[0], tokenizer.bos_id());

        let decoded = tokenizer
            .decode_without_bos(&ids)
            .expect("decode_without_bos should work");
        assert_eq!(decoded, "cab");
    }

    #[test]
    fn unknown_character_returns_error() {
        let tokenizer = Tokenizer::build_from_text("abc");
        let err = tokenizer.encode("z").expect_err("unknown char should fail");

        assert_eq!(err, TokenizerError::UnknownChar('z'));
    }

    #[test]
    fn unknown_id_returns_error() {
        let tokenizer = Tokenizer::build_from_text("abc");
        let invalid = tokenizer.vocab_size() + 10;
        let err = tokenizer
            .decode(&[invalid])
            .expect_err("unknown id should fail");

        assert_eq!(err, TokenizerError::UnknownId(invalid));
    }
}
