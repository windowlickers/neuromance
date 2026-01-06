//! Metadata tracking for context transformations.
//!
//! This module provides types for tracking what transformations have been
//! applied to a context as it moves through the state machine.

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Metadata tracking all transformations applied to a context.
///
/// This maintains a history of all state transitions and transformations,
/// allowing you to audit what operations were performed on the context.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ContextMetadata {
    /// Ordered list of transformations applied
    transformations: Vec<TransformationRecord>,

    /// Additional custom metadata
    custom: HashMap<String, serde_json::Value>,
}

impl ContextMetadata {
    /// Creates a new empty metadata instance.
    pub fn new() -> Self {
        Self {
            transformations: Vec::new(),
            custom: HashMap::new(),
        }
    }

    /// Adds a transformation record to the history.
    pub fn add_transformation<T: Serialize>(&mut self, operation: impl Into<String>, details: &T) {
        let record = TransformationRecord {
            operation: operation.into(),
            timestamp: Utc::now(),
            details: serde_json::to_value(details).ok(),
        };
        self.transformations.push(record);
    }

    /// Returns the number of transformations applied.
    pub fn transformation_count(&self) -> usize {
        self.transformations.len()
    }

    /// Returns a slice of all transformation records.
    pub fn transformations(&self) -> &[TransformationRecord] {
        &self.transformations
    }

    /// Adds a custom metadata key-value pair.
    pub fn add_custom(&mut self, key: impl Into<String>, value: serde_json::Value) {
        self.custom.insert(key.into(), value);
    }

    /// Gets a custom metadata value by key.
    pub fn get_custom(&self, key: &str) -> Option<&serde_json::Value> {
        self.custom.get(key)
    }

    /// Returns all custom metadata.
    pub fn custom(&self) -> &HashMap<String, serde_json::Value> {
        &self.custom
    }

    /// Checks if a specific operation has been applied.
    pub fn has_operation(&self, operation: &str) -> bool {
        self.transformations
            .iter()
            .any(|r| r.operation == operation)
    }

    /// Gets all records for a specific operation.
    pub fn get_operation_records(&self, operation: &str) -> Vec<&TransformationRecord> {
        self.transformations
            .iter()
            .filter(|r| r.operation == operation)
            .collect()
    }
}

impl Default for ContextMetadata {
    fn default() -> Self {
        Self::new()
    }
}

/// Record of a single transformation applied to a context.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TransformationRecord {
    /// Name of the operation performed
    pub operation: String,

    /// When this transformation was applied
    pub timestamp: DateTime<Utc>,

    /// Optional details about the transformation (JSON)
    pub details: Option<serde_json::Value>,
}

impl TransformationRecord {
    /// Creates a new transformation record.
    pub fn new(operation: impl Into<String>) -> Self {
        Self {
            operation: operation.into(),
            timestamp: Utc::now(),
            details: None,
        }
    }

    /// Adds details to this record.
    pub fn with_details(mut self, details: serde_json::Value) -> Self {
        self.details = Some(details);
        self
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_metadata_creation() {
        let metadata = ContextMetadata::new();
        assert_eq!(metadata.transformation_count(), 0);
        assert!(metadata.custom().is_empty());
    }

    #[test]
    fn test_add_transformation() {
        let mut metadata = ContextMetadata::new();
        metadata.add_transformation("filter", &"test");

        assert_eq!(metadata.transformation_count(), 1);
        assert_eq!(metadata.transformations()[0].operation, "filter");
    }

    #[test]
    fn test_custom_metadata() {
        let mut metadata = ContextMetadata::new();
        metadata.add_custom("key", serde_json::json!("value"));

        assert_eq!(
            metadata.get_custom("key"),
            Some(&serde_json::json!("value"))
        );
    }

    #[test]
    fn test_has_operation() {
        let mut metadata = ContextMetadata::new();
        metadata.add_transformation("filter", &());
        metadata.add_transformation("transform", &());

        assert!(metadata.has_operation("filter"));
        assert!(metadata.has_operation("transform"));
        assert!(!metadata.has_operation("validate"));
    }

    #[test]
    fn test_get_operation_records() {
        let mut metadata = ContextMetadata::new();
        metadata.add_transformation("filter", &1);
        metadata.add_transformation("filter", &2);
        metadata.add_transformation("transform", &3);

        let filter_records = metadata.get_operation_records("filter");
        assert_eq!(filter_records.len(), 2);

        let transform_records = metadata.get_operation_records("transform");
        assert_eq!(transform_records.len(), 1);
    }

    #[test]
    fn test_transformation_record() {
        let record =
            TransformationRecord::new("test").with_details(serde_json::json!({"foo": "bar"}));

        assert_eq!(record.operation, "test");
        assert!(record.details.is_some());
    }
}
