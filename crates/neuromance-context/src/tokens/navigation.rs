//! Token-level navigation: position mapping, search, and range extraction.

use regex::Regex;
use serde::{Deserialize, Serialize};

use super::TokenCounter;
use crate::error::TokenCounterError;

/// Information about a single token including its position in the text.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TokenInfo {
    /// The index of this token in the sequence (0-based)
    pub index: usize,

    /// The string representation of this token
    pub token: String,

    /// The token ID from the tokenizer vocabulary
    pub token_id: u32,

    /// Starting character position in the original text
    pub char_start: usize,

    /// Ending character position in the original text
    pub char_end: usize,
}

/// Tokenized text with full position mapping information.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TokenizedText {
    /// The original text
    pub text: String,

    /// Information about each token
    pub tokens: Vec<TokenInfo>,
}

impl TokenizedText {
    /// Returns the token at the given index.
    pub fn get_token(&self, index: usize) -> Option<&TokenInfo> {
        self.tokens.get(index)
    }

    /// Returns the token that contains the given character position.
    pub fn token_at_char_position(&self, char_pos: usize) -> Option<&TokenInfo> {
        self.tokens
            .iter()
            .find(|t| t.char_start <= char_pos && char_pos < t.char_end)
    }

    /// Returns the total number of tokens.
    pub fn token_count(&self) -> usize {
        self.tokens.len()
    }

    /// Returns the character range for a token range.
    pub fn char_range_for_tokens(
        &self,
        start_token: usize,
        end_token: usize,
    ) -> Option<(usize, usize)> {
        if start_token >= self.tokens.len()
            || end_token > self.tokens.len()
            || start_token >= end_token
        {
            return None;
        }

        Some((
            self.tokens[start_token].char_start,
            self.tokens[end_token - 1].char_end,
        ))
    }
}

/// A search match result with both character and token position information.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SearchMatch {
    /// The matched text
    pub matched_text: String,

    /// Starting character position in the text
    pub char_start: usize,

    /// Ending character position in the text
    pub char_end: usize,

    /// Starting token index (if the match starts within a token boundary)
    pub token_start: Option<usize>,

    /// Ending token index (if the match ends within a token boundary)
    pub token_end: Option<usize>,
}

impl SearchMatch {
    /// Returns the token range if both start and end are available.
    pub fn token_range(&self) -> Option<(usize, usize)> {
        match (self.token_start, self.token_end) {
            (Some(start), Some(end)) => Some((start, end + 1)),
            _ => None,
        }
    }

    /// Returns the character length of the match.
    pub fn char_length(&self) -> usize {
        self.char_end - self.char_start
    }
}

impl TokenCounter {
    /// Tokenizes text and returns detailed token information with position mappings.
    ///
    /// # Errors
    ///
    /// Returns an error if tokenization fails.
    pub fn tokenize_with_positions(&self, text: &str) -> Result<TokenizedText, TokenCounterError> {
        let encoding =
            self.tokenizer
                .encode(text, false)
                .map_err(|e| TokenCounterError::Tokenizer {
                    context: "Failed to tokenize text".to_string(),
                    source: e,
                })?;

        let tokens = encoding
            .get_tokens()
            .iter()
            .enumerate()
            .map(|(idx, token)| {
                let offsets = encoding.get_offsets()[idx];
                TokenInfo {
                    index: idx,
                    token: token.clone(),
                    token_id: encoding.get_ids()[idx],
                    char_start: offsets.0,
                    char_end: offsets.1,
                }
            })
            .collect();

        Ok(TokenizedText {
            text: text.to_string(),
            tokens,
        })
    }

    /// Searches for a pattern in text and returns matches with their token positions.
    ///
    /// # Errors
    ///
    /// Returns an error if tokenization or regex compilation fails.
    pub fn search_with_token_positions(
        &self,
        text: &str,
        pattern: &str,
    ) -> Result<Vec<SearchMatch>, TokenCounterError> {
        let tokenized = self.tokenize_with_positions(text)?;
        let regex = Regex::new(pattern)?;

        let matches = regex
            .find_iter(text)
            .map(|match_result| {
                let char_start = match_result.start();
                let char_end = match_result.end();

                // Find token range that covers this match
                let token_start = tokenized
                    .tokens
                    .iter()
                    .find(|t| t.char_start <= char_start && char_start < t.char_end)
                    .map(|t| t.index);

                let token_end = tokenized
                    .tokens
                    .iter()
                    .rev()
                    .find(|t| t.char_start < char_end && char_end <= t.char_end)
                    .map(|t| t.index);

                SearchMatch {
                    matched_text: match_result.as_str().to_string(),
                    char_start,
                    char_end,
                    token_start,
                    token_end,
                }
            })
            .collect();

        Ok(matches)
    }

    /// Extracts a token range from text and returns the corresponding substring.
    ///
    /// # Errors
    ///
    /// Returns an error if tokenization fails or token range is invalid.
    pub fn extract_token_range(
        &self,
        text: &str,
        start_token: usize,
        end_token: usize,
    ) -> Result<String, TokenCounterError> {
        let tokenized = self.tokenize_with_positions(text)?;

        if start_token >= tokenized.tokens.len() || end_token > tokenized.tokens.len() {
            return Err(TokenCounterError::TokenRange(format!(
                "{}-{} out of bounds (text has {} tokens)",
                start_token,
                end_token,
                tokenized.tokens.len()
            )));
        }

        if start_token >= end_token {
            return Err(TokenCounterError::TokenRange(format!(
                "start ({}) must be less than end ({})",
                start_token, end_token
            )));
        }

        let char_start = tokenized.tokens[start_token].char_start;
        let char_end = tokenized.tokens[end_token - 1].char_end;

        Ok(text[char_start..char_end].to_string())
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use super::*;
    use crate::test_support::create_test_tokenizer;

    #[test]
    fn test_tokenize_with_positions_offline() {
        let counter = TokenCounter::from_tokenizer(create_test_tokenizer());

        let result = counter.tokenize_with_positions("hello world").unwrap();
        assert!(!result.tokens.is_empty());
        assert_eq!(result.text, "hello world");
        assert_eq!(result.token_count(), result.tokens.len());
    }

    #[test]
    fn test_tokenized_text_helpers() {
        let counter = TokenCounter::from_tokenizer(create_test_tokenizer());

        let result = counter.tokenize_with_positions("hello world").unwrap();

        assert!(result.get_token(0).is_some());
        assert!(result.get_token(999).is_none());
    }

    #[test]
    fn test_token_at_char_position_boundaries() {
        let counter = TokenCounter::from_tokenizer(create_test_tokenizer());
        let tokenized = counter.tokenize_with_positions("hello world").unwrap();

        // A position inside the first token resolves to that token.
        let first_start = tokenized.tokens[0].char_start;
        assert!(tokenized.token_at_char_position(first_start).is_some());

        // A position past the end of the text maps to no token.
        assert!(
            tokenized
                .token_at_char_position("hello world".len())
                .is_none()
        );
    }

    #[test]
    fn test_char_range_for_tokens_invalid() {
        let counter = TokenCounter::from_tokenizer(create_test_tokenizer());
        let tokenized = counter.tokenize_with_positions("hello world").unwrap();
        let n = tokenized.token_count();

        assert!(
            tokenized.char_range_for_tokens(1, 1).is_none(),
            "start == end"
        );
        assert!(
            tokenized.char_range_for_tokens(0, n + 1).is_none(),
            "end past len"
        );
        assert!(
            tokenized.char_range_for_tokens(n, n).is_none(),
            "start at len"
        );
        assert!(
            tokenized.char_range_for_tokens(0, n).is_some(),
            "valid range"
        );
    }

    #[test]
    fn test_search_with_invalid_regex() {
        let counter = TokenCounter::from_tokenizer(create_test_tokenizer());
        let err = counter
            .search_with_token_positions("hello world", "[invalid")
            .unwrap_err();
        assert!(matches!(err, TokenCounterError::Regex(_)));
    }

    #[test]
    fn test_extract_token_range_errors() {
        let counter = TokenCounter::from_tokenizer(create_test_tokenizer());
        let text = "hello world";
        let n = counter.tokenize_with_positions(text).unwrap().token_count();

        let out_of_bounds = counter.extract_token_range(text, 0, n + 1).unwrap_err();
        assert!(matches!(out_of_bounds, TokenCounterError::TokenRange(_)));

        let inverted = counter.extract_token_range(text, 1, 1).unwrap_err();
        assert!(matches!(inverted, TokenCounterError::TokenRange(_)));

        // A valid range round-trips back to the original text.
        let extracted = counter.extract_token_range(text, 0, n).unwrap();
        assert_eq!(extracted, text);
    }
}
