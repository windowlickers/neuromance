use std::collections::HashMap;
use std::sync::{Arc, RwLock};

use anyhow::Result;
use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::ToolImplementation;
use neuromance_common::tools::{Function, Property, Tool};

/// Status of a todo item
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(rename_all = "snake_case")]
pub enum TodoStatus {
    Pending,
    InProgress,
    Completed,
}

/// A single todo item
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TodoItem {
    pub content: String,
    pub status: TodoStatus,
    pub active_form: String,
}

/// Shared todo list storage
type TodoStorage = Arc<RwLock<Vec<TodoItem>>>;

/// Tool for reading the current todo list
pub struct TodoReadTool {
    storage: TodoStorage,
}

impl TodoReadTool {
    pub fn new(storage: TodoStorage) -> Self {
        Self { storage }
    }
}

impl Default for TodoReadTool {
    fn default() -> Self {
        Self {
            storage: Arc::new(RwLock::new(Vec::new())),
        }
    }
}

#[async_trait]
impl ToolImplementation for TodoReadTool {
    fn get_definition(&self) -> Tool {
        Tool {
            r#type: "function".to_string(),
            function: Function {
                name: "read_todos".to_string(),
                description: "Read the current todo list to see task progress and what's planned."
                    .to_string(),
                parameters: serde_json::json!({
                    "type": "object",
                    "properties": {},
                }),
            },
        }
    }

    async fn execute(&self, _args: &Value) -> Result<String> {
        let todos = self.storage.read().unwrap();

        if todos.is_empty() {
            return Ok("TODO LIST: (empty)".to_string());
        }

        let mut response = String::from("TODO LIST:\n");
        for todo in todos.iter() {
            let status_symbol = match todo.status {
                TodoStatus::Pending => "[ ]",
                TodoStatus::InProgress => "[→]",
                TodoStatus::Completed => "[✓]",
            };
            response.push_str(&format!("{} {}\n", status_symbol, todo.content));
        }

        Ok(response)
    }

    fn is_auto_approved(&self) -> bool {
        true
    }
}

/// Tool for writing/updating the todo list
pub struct TodoWriteTool {
    storage: TodoStorage,
}

impl TodoWriteTool {
    pub fn new(storage: TodoStorage) -> Self {
        Self { storage }
    }
}

impl Default for TodoWriteTool {
    fn default() -> Self {
        Self {
            storage: Arc::new(RwLock::new(Vec::new())),
        }
    }
}

#[async_trait]
impl ToolImplementation for TodoWriteTool {
    fn get_definition(&self) -> Tool {
        let mut properties = HashMap::new();
        properties.insert(
            "todos".to_string(),
            Property {
                prop_type: "array".to_string(),
                description: "Array of todo items with content, status (pending/in_progress/completed), and active_form (present continuous form of the task)".to_string(),
            },
        );

        Tool {
            r#type: "function".to_string(),
            function: Function {
                name: "write_todos".to_string(),
                description: "Update the todo list to track task progress. Each todo should have 'content' (imperative form like 'Fix bug'), 'status' (pending/in_progress/completed), and 'active_form' (present continuous like 'Fixing bug'). Exactly one task must be in_progress.".to_string(),
                parameters: serde_json::json!({
                    "type": "object",
                    "properties": properties,
                    "required": vec!["todos".to_string()],
                }),
            },
        }
    }

    async fn execute(&self, args: &Value) -> Result<String> {
        let obj = args
            .as_object()
            .ok_or_else(|| anyhow::anyhow!("Expected object arguments"))?;

        let todos_value = obj
            .get("todos")
            .ok_or_else(|| anyhow::anyhow!("Missing 'todos' parameter"))?;

        let todos: Vec<TodoItem> = serde_json::from_value(todos_value.clone())
            .map_err(|e| anyhow::anyhow!("Invalid todo items format: {}", e))?;

        // Validate that exactly one task is in_progress
        let in_progress_count = todos
            .iter()
            .filter(|t| t.status == TodoStatus::InProgress)
            .count();
        if in_progress_count != 1 {
            return Err(anyhow::anyhow!(
                "Exactly one task must be in_progress, found {}",
                in_progress_count
            ));
        }

        // Update the stored todos
        *self.storage.write().unwrap() = todos.clone();

        // Format the response
        let mut response = String::from("TODO LIST UPDATED:\n");
        for todo in &todos {
            let status_symbol = match todo.status {
                TodoStatus::Pending => "[ ]",
                TodoStatus::InProgress => "[→]",
                TodoStatus::Completed => "[✓]",
            };
            response.push_str(&format!("{} {}\n", status_symbol, todo.content));
        }

        Ok(response)
    }

    fn is_auto_approved(&self) -> bool {
        true
    }
}

/// Create a pair of TodoRead and TodoWrite tools that share the same storage
pub fn create_todo_tools() -> (TodoReadTool, TodoWriteTool) {
    let storage = Arc::new(RwLock::new(Vec::new()));
    (
        TodoReadTool::new(Arc::clone(&storage)),
        TodoWriteTool::new(storage),
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[tokio::test]
    async fn test_todo_read_empty() {
        let (read_tool, _) = create_todo_tools();

        let result = read_tool.execute(&json!({})).await.unwrap();
        assert!(result.contains("(empty)"));
    }

    #[tokio::test]
    async fn test_todo_write_and_read() {
        let (read_tool, write_tool) = create_todo_tools();

        let args = json!({
            "todos": [
                {
                    "content": "Create contact form component",
                    "status": "in_progress",
                    "active_form": "Creating contact form component"
                },
                {
                    "content": "Add form validation",
                    "status": "pending",
                    "active_form": "Adding form validation"
                }
            ]
        });

        let write_result = write_tool.execute(&args).await.unwrap();
        assert!(write_result.contains("TODO LIST UPDATED:"));
        assert!(write_result.contains("[→] Create contact form component"));

        let read_result = read_tool.execute(&json!({})).await.unwrap();
        assert!(read_result.contains("[→] Create contact form component"));
        assert!(read_result.contains("[ ] Add form validation"));
    }

    #[tokio::test]
    async fn test_todo_write_requires_one_in_progress() {
        let (_, write_tool) = create_todo_tools();

        let args = json!({
            "todos": [
                {
                    "content": "Task 1",
                    "status": "pending",
                    "active_form": "Doing task 1"
                }
            ]
        });

        let result = write_tool.execute(&args).await;
        assert!(result.is_err());
        assert!(
            result
                .unwrap_err()
                .to_string()
                .contains("Exactly one task must be in_progress")
        );
    }

    #[tokio::test]
    async fn test_todo_update_progress() {
        let (read_tool, write_tool) = create_todo_tools();

        // Initial state
        let args = json!({
            "todos": [
                {
                    "content": "Task 1",
                    "status": "in_progress",
                    "active_form": "Doing task 1"
                },
                {
                    "content": "Task 2",
                    "status": "pending",
                    "active_form": "Doing task 2"
                }
            ]
        });
        write_tool.execute(&args).await.unwrap();

        // Update: Task 1 completed, Task 2 in progress
        let args = json!({
            "todos": [
                {
                    "content": "Task 1",
                    "status": "completed",
                    "active_form": "Doing task 1"
                },
                {
                    "content": "Task 2",
                    "status": "in_progress",
                    "active_form": "Doing task 2"
                }
            ]
        });
        write_tool.execute(&args).await.unwrap();

        let read_result = read_tool.execute(&json!({})).await.unwrap();
        assert!(read_result.contains("[✓] Task 1"));
        assert!(read_result.contains("[→] Task 2"));
    }
}
