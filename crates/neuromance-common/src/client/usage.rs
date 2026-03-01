use serde::{Deserialize, Serialize};

/// Token usage statistics for a completion request.
///
/// Tracks the number of tokens consumed by the prompt and completion,
/// along with optional cost information and detailed breakdowns.
///
/// # Note
///
/// Different providers may count tokens differently. The `total_tokens`
/// should always equal `prompt_tokens + completion_tokens`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Usage {
    /// Number of tokens in the input prompt.
    #[serde(alias = "input_tokens")]
    pub prompt_tokens: u32,
    /// Number of tokens generated in the completion.
    #[serde(alias = "output_tokens")]
    pub completion_tokens: u32,
    /// Total tokens used (prompt + completion).
    pub total_tokens: u32,
    /// Estimated cost in USD for this request (if available).
    pub cost: Option<f64>,
    /// Detailed breakdown of input token usage.
    #[serde(
        skip_serializing_if = "Option::is_none",
        alias = "prompt_tokens_details"
    )]
    pub input_tokens_details: Option<InputTokensDetails>,
    /// Detailed breakdown of output token usage.
    #[serde(
        skip_serializing_if = "Option::is_none",
        alias = "completion_tokens_details"
    )]
    pub output_tokens_details: Option<OutputTokensDetails>,
}

/// Detailed breakdown of input token usage.
///
/// Provides additional information about how input tokens were processed,
/// including cache utilization.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct InputTokensDetails {
    /// Number of tokens served from cache rather than processed fresh.
    ///
    /// Cached tokens are typically cheaper and faster to process.
    /// For `OpenAI`: automatic prefix caching.
    /// For Anthropic: prompt caching (`cache_read_input_tokens`).
    #[serde(default)]
    pub cached_tokens: u32,

    /// Number of tokens written to cache (Anthropic-specific).
    ///
    /// These tokens incur a +25% cost for cache creation but enable
    /// future cache reads at -90% cost.
    #[serde(default, skip_serializing_if = "is_zero")]
    pub cache_creation_tokens: u32,
}

/// Helper for serde `skip_serializing_if`.
#[allow(clippy::trivially_copy_pass_by_ref)]
const fn is_zero(val: &u32) -> bool {
    *val == 0
}

/// Detailed breakdown of output token usage.
///
/// Provides additional information about token usage in the model's response,
/// including reasoning tokens for models that support chain-of-thought.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OutputTokensDetails {
    /// Number of tokens used for internal reasoning (e.g., chain-of-thought).
    ///
    /// Some models generate reasoning tokens that are not part of the final response.
    pub reasoning_tokens: u32,
}

impl Usage {
    /// Fraction of input tokens served from cache (0.0-1.0).
    ///
    /// Returns `None` if `prompt_tokens` is zero.
    #[must_use]
    pub fn cache_hit_ratio(&self) -> Option<f64> {
        if self.prompt_tokens == 0 {
            return None;
        }
        let cached = self
            .input_tokens_details
            .as_ref()
            .map_or(0, |d| d.cached_tokens);
        Some(f64::from(cached) / f64::from(self.prompt_tokens))
    }
}

/// Aggregate cache statistics across multiple LLM requests.
///
/// Tracks cumulative token counts and cache hit rates to monitor
/// prompt caching effectiveness over a session or agent run.
///
/// Access via `core.cache_metrics` after running tool loops.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct CacheMetrics {
    /// Sum of `prompt_tokens` across all recorded requests.
    #[serde(default)]
    pub total_input_tokens: u64,
    /// Sum of `cached_tokens` (tokens read from cache).
    #[serde(default)]
    pub total_cached_tokens: u64,
    /// Sum of `cache_creation_tokens` (tokens written to cache).
    #[serde(default)]
    pub total_cache_creation_tokens: u64,
    /// Sum of `completion_tokens` across all recorded requests.
    #[serde(default)]
    pub total_output_tokens: u64,
    /// Number of requests where `cached_tokens > 0`.
    #[serde(default)]
    pub requests_with_cache_hits: u32,
    /// Total number of requests recorded.
    #[serde(default)]
    pub total_requests: u32,
}

impl CacheMetrics {
    /// Record token usage from a single LLM response.
    pub fn record(&mut self, usage: &Usage) {
        self.total_input_tokens += u64::from(usage.prompt_tokens);
        self.total_output_tokens += u64::from(usage.completion_tokens);
        self.total_requests += 1;

        if let Some(ref details) = usage.input_tokens_details {
            self.total_cached_tokens += u64::from(details.cached_tokens);
            self.total_cache_creation_tokens += u64::from(details.cache_creation_tokens);
            if details.cached_tokens > 0 {
                self.requests_with_cache_hits += 1;
            }
        }
    }

    /// Fraction of total input tokens served from cache (0.0-1.0).
    ///
    /// Returns `None` if no input tokens have been recorded.
    #[must_use]
    pub fn cache_hit_ratio(&self) -> Option<f64> {
        if self.total_input_tokens == 0 {
            return None;
        }
        #[allow(clippy::cast_precision_loss)]
        Some(self.total_cached_tokens as f64 / self.total_input_tokens as f64)
    }

    /// Fraction of requests that had at least one cache hit (0.0-1.0).
    ///
    /// Returns `None` if no requests have been recorded.
    #[must_use]
    pub fn request_hit_rate(&self) -> Option<f64> {
        if self.total_requests == 0 {
            return None;
        }
        Some(f64::from(self.requests_with_cache_hits) / f64::from(self.total_requests))
    }
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used)]
    #![allow(clippy::expect_used)]

    use super::*;

    #[test]
    fn usage_cache_hit_ratio_with_cache() {
        let usage = Usage {
            prompt_tokens: 100,
            completion_tokens: 50,
            total_tokens: 150,
            cost: None,
            input_tokens_details: Some(InputTokensDetails {
                cached_tokens: 80,
                cache_creation_tokens: 0,
            }),
            output_tokens_details: None,
        };
        let ratio = usage.cache_hit_ratio().unwrap();
        assert!((ratio - 0.8).abs() < f64::EPSILON);
    }

    #[test]
    fn usage_cache_hit_ratio_without_details() {
        let usage = Usage {
            prompt_tokens: 100,
            completion_tokens: 50,
            total_tokens: 150,
            cost: None,
            input_tokens_details: None,
            output_tokens_details: None,
        };
        let ratio = usage.cache_hit_ratio().unwrap();
        assert!(ratio.abs() < f64::EPSILON);
    }

    #[test]
    fn usage_cache_hit_ratio_zero_prompt_tokens() {
        let usage = Usage {
            prompt_tokens: 0,
            completion_tokens: 50,
            total_tokens: 50,
            cost: None,
            input_tokens_details: None,
            output_tokens_details: None,
        };
        assert!(usage.cache_hit_ratio().is_none());
    }

    #[test]
    fn cache_metrics_default_is_zeroed() {
        let m = CacheMetrics::default();
        assert_eq!(m.total_input_tokens, 0);
        assert_eq!(m.total_cached_tokens, 0);
        assert_eq!(m.total_cache_creation_tokens, 0);
        assert_eq!(m.requests_with_cache_hits, 0);
        assert_eq!(m.total_requests, 0);
        assert!(m.cache_hit_ratio().is_none());
        assert!(m.request_hit_rate().is_none());
    }

    #[test]
    fn cache_metrics_record_accumulates() {
        let mut m = CacheMetrics::default();

        // First request: 80 of 100 cached
        m.record(&Usage {
            prompt_tokens: 100,
            completion_tokens: 20,
            total_tokens: 120,
            cost: None,
            input_tokens_details: Some(InputTokensDetails {
                cached_tokens: 80,
                cache_creation_tokens: 10,
            }),
            output_tokens_details: None,
        });

        assert_eq!(m.total_input_tokens, 100);
        assert_eq!(m.total_cached_tokens, 80);
        assert_eq!(m.total_cache_creation_tokens, 10);
        assert_eq!(m.requests_with_cache_hits, 1);
        assert_eq!(m.total_requests, 1);

        // Second request: no cache hit
        m.record(&Usage {
            prompt_tokens: 200,
            completion_tokens: 30,
            total_tokens: 230,
            cost: None,
            input_tokens_details: Some(InputTokensDetails {
                cached_tokens: 0,
                cache_creation_tokens: 50,
            }),
            output_tokens_details: None,
        });

        assert_eq!(m.total_input_tokens, 300);
        assert_eq!(m.total_cached_tokens, 80);
        assert_eq!(m.total_cache_creation_tokens, 60);
        assert_eq!(m.requests_with_cache_hits, 1);
        assert_eq!(m.total_requests, 2);
    }

    #[test]
    fn cache_metrics_ratios() {
        let mut m = CacheMetrics::default();

        m.record(&Usage {
            prompt_tokens: 100,
            completion_tokens: 10,
            total_tokens: 110,
            cost: None,
            input_tokens_details: Some(InputTokensDetails {
                cached_tokens: 50,
                cache_creation_tokens: 0,
            }),
            output_tokens_details: None,
        });

        m.record(&Usage {
            prompt_tokens: 100,
            completion_tokens: 10,
            total_tokens: 110,
            cost: None,
            input_tokens_details: None,
            output_tokens_details: None,
        });

        // 50 cached / 200 total = 0.25
        let ratio = m.cache_hit_ratio().unwrap();
        assert!((ratio - 0.25).abs() < f64::EPSILON);

        // 1 of 2 requests had cache hits = 0.5
        let rate = m.request_hit_rate().unwrap();
        assert!((rate - 0.5).abs() < f64::EPSILON);
    }
}
