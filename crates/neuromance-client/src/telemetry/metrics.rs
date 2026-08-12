//! The two GenAI client metrics, recorded through the global meter.
//!
//! These are the only measurements in this workspace that go through
//! OpenTelemetry rather than the `metrics` crate. See the
//! `neuromance-runtime::telemetry` module docs for why, and for the rule that
//! nothing is emitted to both pipelines.

use std::sync::OnceLock;

use opentelemetry::KeyValue;
use opentelemetry::metrics::Histogram;

use neuromance_common::client::Usage;
use neuromance_common::telemetry::genai;

use super::attrs::GenAiAttrs;

/// The instruments, built once against whatever meter provider is global.
pub struct GenAiInstruments {
    token_usage: Histogram<u64>,
    operation_duration: Histogram<f64>,
}

impl GenAiInstruments {
    /// Build a fresh set of instruments from the global meter provider.
    ///
    /// Bucket boundaries come from the conventions, not the SDK defaults: the
    /// default buckets top out far below a large prompt and would put every
    /// real token count in one overflow bucket.
    fn new() -> Self {
        let meter = opentelemetry::global::meter("neuromance-client");
        Self {
            token_usage: meter
                .u64_histogram(genai::METRIC_TOKEN_USAGE)
                .with_description("Number of input and output tokens used by the model")
                .with_unit("{token}")
                .with_boundaries(genai::TOKEN_USAGE_BUCKETS.to_vec())
                .build(),
            operation_duration: meter
                .f64_histogram(genai::METRIC_OPERATION_DURATION)
                .with_description("GenAI operation duration")
                .with_unit("s")
                .with_boundaries(genai::OPERATION_DURATION_BUCKETS.to_vec())
                .build(),
        }
    }

    /// Record the prompt and completion token counts as separate series.
    pub fn record_token_usage(
        &self,
        attrs: &GenAiAttrs,
        response_model: Option<&str>,
        usage: &Usage,
    ) {
        self.record_tokens(
            attrs,
            response_model,
            genai::token_type::INPUT,
            usage.prompt_tokens,
        );
        self.record_tokens(
            attrs,
            response_model,
            genai::token_type::OUTPUT,
            usage.completion_tokens,
        );
    }

    /// Record prompt tokens alone.
    ///
    /// An embeddings call generates no completion, so recording a zero-valued
    /// output series for it would put a spurious sample in every histogram
    /// bucket query that spans both operations.
    pub fn record_input_tokens(
        &self,
        attrs: &GenAiAttrs,
        response_model: Option<&str>,
        prompt_tokens: u32,
    ) {
        self.record_tokens(
            attrs,
            response_model,
            genai::token_type::INPUT,
            prompt_tokens,
        );
    }

    fn record_tokens(
        &self,
        attrs: &GenAiAttrs,
        response_model: Option<&str>,
        token_type: &'static str,
        tokens: u32,
    ) {
        let mut kv = attrs.metric_kv(response_model, None);
        kv.push(KeyValue::new(genai::TOKEN_TYPE, token_type));
        self.token_usage.record(u64::from(tokens), &kv);
    }

    /// Record how long the operation took, whether or not it succeeded.
    pub fn record_duration(
        &self,
        attrs: &GenAiAttrs,
        response_model: Option<&str>,
        error_type: Option<&str>,
        seconds: f64,
    ) {
        self.operation_duration
            .record(seconds, &attrs.metric_kv(response_model, error_type));
    }
}

/// The process-wide instruments.
///
/// Built lazily on first use, which in a runtime process is always after
/// `telemetry::try_init` has installed the meter provider. A library consumer
/// that never installs one gets no-op instruments and pays an atomic load.
pub fn instruments() -> &'static GenAiInstruments {
    static INSTRUMENTS: OnceLock<GenAiInstruments> = OnceLock::new();
    INSTRUMENTS.get_or_init(GenAiInstruments::new)
}

#[cfg(test)]
mod tests {
    #![allow(clippy::expect_used)]

    use super::*;
    use neuromance_common::client::Config;
    use opentelemetry::metrics::MeterProvider as _;
    use opentelemetry_sdk::metrics::{
        InMemoryMetricExporter, PeriodicReader, SdkMeterProvider, data::AggregatedMetrics,
        data::MetricData,
    };

    /// Build instruments against a local provider rather than the global one,
    /// so these assertions do not depend on which test in the binary ran first.
    fn local_instruments() -> (SdkMeterProvider, InMemoryMetricExporter, GenAiInstruments) {
        let exporter = InMemoryMetricExporter::default();
        let provider = SdkMeterProvider::builder()
            .with_reader(PeriodicReader::builder(exporter.clone()).build())
            .build();
        let meter = provider.meter("test");
        let instruments = GenAiInstruments {
            token_usage: meter
                .u64_histogram(genai::METRIC_TOKEN_USAGE)
                .with_unit("{token}")
                .with_boundaries(genai::TOKEN_USAGE_BUCKETS.to_vec())
                .build(),
            operation_duration: meter
                .f64_histogram(genai::METRIC_OPERATION_DURATION)
                .with_unit("s")
                .with_boundaries(genai::OPERATION_DURATION_BUCKETS.to_vec())
                .build(),
        };
        (provider, exporter, instruments)
    }

    fn attrs() -> GenAiAttrs {
        GenAiAttrs::chat(
            &Config::new("openai", "gpt-4o"),
            &neuromance_common::client::ChatRequest::new(Vec::new()),
        )
    }

    /// Sum of a histogram's recorded values, keyed by the `gen_ai.token.type`
    /// attribute so input and output series stay distinguishable.
    fn token_sums(exporter: &InMemoryMetricExporter) -> Vec<(String, u64)> {
        let mut sums = Vec::new();
        for resource_metric in exporter.get_finished_metrics().unwrap_or_default() {
            for scope in resource_metric.scope_metrics() {
                for metric in scope.metrics() {
                    if metric.name() != genai::METRIC_TOKEN_USAGE {
                        continue;
                    }
                    let AggregatedMetrics::U64(MetricData::Histogram(hist)) = metric.data() else {
                        continue;
                    };
                    for point in hist.data_points() {
                        let token_type = point
                            .attributes()
                            .find(|kv| kv.key.as_str() == genai::TOKEN_TYPE)
                            .map(|kv| kv.value.to_string())
                            .unwrap_or_default();
                        sums.push((token_type, point.sum()));
                    }
                }
            }
        }
        sums.sort();
        sums
    }

    /// Prompt and completion tokens must land in separate series. Recording
    /// them together would make the cost of an operation unrecoverable.
    #[test]
    fn test_token_usage_splits_input_from_output() {
        let (provider, exporter, instruments) = local_instruments();
        let usage = Usage {
            prompt_tokens: 900,
            completion_tokens: 40,
            total_tokens: 940,
            cost: None,
            input_tokens_details: None,
            output_tokens_details: None,
        };

        instruments.record_token_usage(&attrs(), Some("gpt-4o-2024-08-06"), &usage);
        provider.force_flush().expect("flush");

        assert_eq!(
            token_sums(&exporter),
            vec![("input".to_string(), 900), ("output".to_string(), 40)]
        );
    }

    /// A failed operation still costs latency, and the error must be on the
    /// sample or a failing provider looks fast rather than broken.
    #[test]
    fn test_duration_carries_the_error_type_on_failure() {
        let (provider, exporter, instruments) = local_instruments();
        instruments.record_duration(&attrs(), None, Some("rate_limited"), 1.5);
        provider.force_flush().expect("flush");

        let found = exporter
            .get_finished_metrics()
            .unwrap_or_default()
            .iter()
            .flat_map(|rm| rm.scope_metrics().collect::<Vec<_>>())
            .flat_map(|scope| scope.metrics().collect::<Vec<_>>())
            .filter(|metric| metric.name() == genai::METRIC_OPERATION_DURATION)
            .any(|metric| {
                let AggregatedMetrics::F64(MetricData::Histogram(hist)) = metric.data() else {
                    return false;
                };
                hist.data_points().any(|point| {
                    point.attributes().any(|kv| {
                        kv.key.as_str() == genai::ERROR_TYPE
                            && kv.value.to_string() == "rate_limited"
                    })
                })
            });

        assert!(found, "duration sample should carry error.type");
    }
}
