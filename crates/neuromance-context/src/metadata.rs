//! Metadata tracking for context transformations.
//!
//! This module provides types for tracking what transformations have been
//! applied to a context as it moves through the state machine.

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// The fixed set of operations the context state machine can perform.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Operation {
    Filter,
    SkipFilter,
    Transform,
    TransformWithPipeline,
    Compact,
    SkipTransform,
    Ready,
}

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
    pub fn add_transformation<T: Serialize>(&mut self, operation: Operation, details: &T) {
        let record = TransformationRecord {
            operation,
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
    pub fn has_operation(&self, operation: Operation) -> bool {
        self.transformations
            .iter()
            .any(|r| r.operation == operation)
    }

    /// Gets all records for a specific operation.
    pub fn get_operation_records(&self, operation: Operation) -> Vec<&TransformationRecord> {
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
    /// The operation performed
    pub operation: Operation,

    /// When this transformation was applied
    pub timestamp: DateTime<Utc>,

    /// Optional details about the transformation (JSON)
    pub details: Option<serde_json::Value>,
}

impl TransformationRecord {
    /// Creates a new transformation record.
    pub fn new(operation: Operation) -> Self {
        Self {
            operation,
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
        metadata.add_transformation(Operation::Filter, &"test");

        assert_eq!(metadata.transformation_count(), 1);
        assert_eq!(metadata.transformations()[0].operation, Operation::Filter);
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
        metadata.add_transformation(Operation::Filter, &());
        metadata.add_transformation(Operation::Transform, &());

        assert!(metadata.has_operation(Operation::Filter));
        assert!(metadata.has_operation(Operation::Transform));
        assert!(!metadata.has_operation(Operation::Ready));
    }

    #[test]
    fn test_get_operation_records() {
        let mut metadata = ContextMetadata::new();
        metadata.add_transformation(Operation::Filter, &1);
        metadata.add_transformation(Operation::Filter, &2);
        metadata.add_transformation(Operation::Transform, &3);

        let filter_records = metadata.get_operation_records(Operation::Filter);
        assert_eq!(filter_records.len(), 2);

        let transform_records = metadata.get_operation_records(Operation::Transform);
        assert_eq!(transform_records.len(), 1);
    }

    #[test]
    fn test_transformation_record() {
        let record = TransformationRecord::new(Operation::Filter)
            .with_details(serde_json::json!({"foo": "bar"}));

        assert_eq!(record.operation, Operation::Filter);
        assert!(record.details.is_some());
    }
}
