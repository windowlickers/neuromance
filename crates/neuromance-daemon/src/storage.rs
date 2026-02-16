//! Conversation storage and persistence.
//!
//! Conversations are stored as JSON files in `~/.local/share/neuromance/conversations/`.
//! Each conversation is saved to `<uuid>.json`. The active conversation ID is tracked
//! in a `current` file.
//!
//! ## File Layout
//!
//! ```text
//! ~/.local/share/neuromance/
//! ├── conversations/
//! │   ├── abc12345-6789-....json
//! │   └── def67890-abcd-....json
//! ├── current              (active conversation UUID)
//! ├── bookmarks.json       (bookmark name -> UUID mapping)
//! └── neuromance.sock      (Unix domain socket)
//! ```

use std::collections::HashMap;
use std::fs;
use std::os::unix::fs::{DirBuilderExt, PermissionsExt};
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};

use neuromance_common::Conversation;
use serde::{Deserialize, Serialize};
use tracing::{debug, instrument};
use uuid::Uuid;

use crate::error::{DaemonError, Result};

/// Manages conversation storage on disk.
pub struct Storage {
    /// Conversations directory
    conversations_dir: PathBuf,

    /// Active conversation file path
    current_file: PathBuf,

    /// Bookmarks file path
    bookmarks_file: PathBuf,

    /// Socket file path
    socket_path: PathBuf,

    /// PID file path
    pid_file: PathBuf,

    /// Serializes bookmark read-modify-write operations.
    /// `std::sync::Mutex` is correct here because Storage methods run on
    /// the blocking thread pool via `spawn_blocking`, not the async runtime.
    bookmarks_lock: Mutex<()>,
}

/// Bookmark mapping stored in `bookmarks.json`.
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
struct Bookmarks {
    /// Map of bookmark name -> conversation UUID
    map: HashMap<String, String>,
}

impl Storage {
    /// Creates a new storage manager.
    ///
    /// Initializes the storage directory structure if it doesn't exist.
    ///
    /// # Errors
    ///
    /// Returns an error if:
    /// - The home directory cannot be determined
    /// - Directory creation fails
    pub fn new() -> Result<Self> {
        let data_dir = dirs::data_local_dir()
            .ok_or_else(|| DaemonError::Storage("Failed to determine data directory".to_string()))?
            .join("neuromance");

        let conversations_dir = data_dir.join("conversations");
        let current_file = data_dir.join("current");
        let bookmarks_file = data_dir.join("bookmarks.json");
        let socket_path = data_dir.join("neuromance.sock");
        let pid_file = data_dir.join("neuromance.pid");

        // Create directories with restricted permissions (owner-only)
        fs::DirBuilder::new()
            .recursive(true)
            .mode(0o700)
            .create(&conversations_dir)
            .map_err(|e| {
                DaemonError::Storage(format!(
                    "Failed to create conversations directory: {e}"
                ))
            })?;

        // Harden existing installs: ensure data dir is owner-only
        fs::set_permissions(
            &data_dir,
            fs::Permissions::from_mode(0o700),
        )
        .map_err(|e| {
            DaemonError::Storage(format!(
                "Failed to set data directory permissions: {e}"
            ))
        })?;

        Ok(Self {
            conversations_dir,
            current_file,
            bookmarks_file,
            socket_path,
            pid_file,
            bookmarks_lock: Mutex::new(()),
        })
    }

    /// Runs a synchronous closure on the blocking thread pool.
    ///
    /// All async callers should use this to avoid blocking
    /// tokio worker threads with file I/O.
    ///
    /// # Errors
    ///
    /// Returns an error if the blocking task panics or the closure fails.
    pub async fn run<F, T>(self: &Arc<Self>, f: F) -> Result<T>
    where
        F: FnOnce(&Self) -> Result<T> + Send + 'static,
        T: Send + 'static,
    {
        let this = Arc::clone(self);
        tokio::task::spawn_blocking(move || f(&this))
            .await
            .map_err(|e| {
                DaemonError::Storage(format!("Task join error: {e}"))
            })?
    }

    /// Returns the socket path for the Unix domain socket.
    #[must_use]
    pub fn socket_path(&self) -> &Path {
        &self.socket_path
    }

    /// Writes the daemon PID to the PID file.
    ///
    /// # Errors
    ///
    /// Returns an error if file writing fails.
    pub fn write_pid(&self, pid: u32) -> Result<()> {
        fs::write(&self.pid_file, pid.to_string())?;
        Ok(())
    }

    /// Reads the PID from the PID file, if it exists.
    ///
    /// Returns `None` if the file doesn't exist or can't be parsed.
    #[must_use]
    pub fn read_pid(&self) -> Option<u32> {
        fs::read_to_string(&self.pid_file)
            .ok()
            .and_then(|s| s.trim().parse().ok())
    }

    /// Removes the PID file.
    ///
    /// # Errors
    ///
    /// Returns an error if file deletion fails (unless file doesn't exist).
    pub fn remove_pid(&self) -> Result<()> {
        if self.pid_file.exists() {
            fs::remove_file(&self.pid_file)?;
        }
        Ok(())
    }

    /// Saves a conversation to disk.
    ///
    /// Uses atomic write (write to temp file, then rename) for safety.
    ///
    /// # Errors
    ///
    /// Returns an error if serialization or file I/O fails.
    #[instrument(skip(self, conversation), fields(conversation_id = %conversation.id, message_count = conversation.messages.len()))]
    pub fn save_conversation(&self, conversation: &Conversation) -> Result<()> {
        let path = self.conversation_path(&conversation.id);
        let json = serde_json::to_string_pretty(conversation)?;

        // Atomic write: write to temp file, then rename
        let temp_path = path.with_extension("tmp");
        fs::write(&temp_path, &json)?;
        fs::rename(&temp_path, &path)?;

        debug!(
            conversation_id = %conversation.id,
            path = %path.display(),
            size_bytes = json.len(),
            "Saved conversation"
        );

        Ok(())
    }

    /// Loads a conversation from disk.
    ///
    /// # Errors
    ///
    /// Returns an error if:
    /// - The file doesn't exist
    /// - Deserialization fails
    #[instrument(skip(self), fields(conversation_id = %id))]
    pub fn load_conversation(&self, id: &Uuid) -> Result<Conversation> {
        let path = self.conversation_path(id);

        if !path.exists() {
            return Err(DaemonError::ConversationNotFound(id.to_string()));
        }

        let json = fs::read_to_string(&path)?;
        let conversation: Conversation = serde_json::from_str(&json)?;

        debug!(
            conversation_id = %id,
            message_count = conversation.messages.len(),
            "Loaded conversation"
        );

        Ok(conversation)
    }

    /// Lists all conversation IDs.
    ///
    /// # Errors
    ///
    /// Returns an error if directory reading fails.
    pub fn list_conversations(&self) -> Result<Vec<Uuid>> {
        let mut ids = Vec::new();

        for entry in fs::read_dir(&self.conversations_dir)? {
            let entry = entry?;
            let path = entry.path();

            if path.extension().is_some_and(|ext| ext == "json")
                && let Some(stem) = path.file_stem()
                && let Some(stem_str) = stem.to_str()
                && let Ok(id) = Uuid::parse_str(stem_str)
            {
                ids.push(id);
            }
        }

        Ok(ids)
    }

    /// Gets the active conversation ID.
    ///
    /// Returns `None` if no active conversation is set.
    ///
    /// # Errors
    ///
    /// Returns an error if file reading or UUID parsing fails.
    pub fn get_active_conversation(&self) -> Result<Option<Uuid>> {
        if !self.current_file.exists() {
            return Ok(None);
        }

        let id_str = fs::read_to_string(&self.current_file)?.trim().to_string();

        if id_str.is_empty() {
            return Ok(None);
        }

        let id = Uuid::parse_str(&id_str)
            .map_err(|e| DaemonError::Storage(format!("Invalid UUID in current file: {e}")))?;

        Ok(Some(id))
    }

    /// Sets the active conversation ID.
    ///
    /// # Errors
    ///
    /// Returns an error if file writing fails.
    pub fn set_active_conversation(&self, id: &Uuid) -> Result<()> {
        let temp_path = self.current_file.with_extension("tmp");
        fs::write(&temp_path, id.to_string())?;
        fs::rename(&temp_path, &self.current_file)?;
        Ok(())
    }

    /// Loads bookmarks from disk.
    ///
    /// Returns an empty map if the bookmarks file doesn't exist.
    ///
    /// # Errors
    ///
    /// Returns an error if deserialization fails.
    pub fn load_bookmarks(&self) -> Result<HashMap<String, String>> {
        if !self.bookmarks_file.exists() {
            return Ok(HashMap::new());
        }

        let json = fs::read_to_string(&self.bookmarks_file)?;
        let bookmarks: Bookmarks = serde_json::from_str(&json)?;

        Ok(bookmarks.map)
    }

    /// Saves bookmarks to disk.
    ///
    /// # Errors
    ///
    /// Returns an error if serialization or file writing fails.
    pub fn save_bookmarks(&self, map: &HashMap<String, String>) -> Result<()> {
        let bookmarks = Bookmarks { map: map.clone() };
        let json = serde_json::to_string_pretty(&bookmarks)?;
        let temp_path = self.bookmarks_file.with_extension("tmp");
        fs::write(&temp_path, json)?;
        fs::rename(&temp_path, &self.bookmarks_file)?;
        Ok(())
    }

    /// Sets a bookmark for a conversation.
    ///
    /// # Errors
    ///
    /// Returns an error if:
    /// - The bookmark already exists
    /// - Saving fails
    pub fn set_bookmark(&self, name: &str, conversation_id: &Uuid) -> Result<()> {
        let _guard = self
            .bookmarks_lock
            .lock()
            .map_err(|e| DaemonError::Storage(format!("Bookmark lock poisoned: {e}")))?;

        let mut bookmarks = self.load_bookmarks()?;

        if bookmarks.contains_key(name) {
            return Err(DaemonError::BookmarkExists(name.to_string()));
        }

        bookmarks.insert(name.to_string(), conversation_id.to_string());
        self.save_bookmarks(&bookmarks)?;

        Ok(())
    }

    /// Removes a bookmark.
    ///
    /// # Errors
    ///
    /// Returns an error if:
    /// - The bookmark doesn't exist
    /// - Saving fails
    pub fn remove_bookmark(&self, name: &str) -> Result<()> {
        let _guard = self
            .bookmarks_lock
            .lock()
            .map_err(|e| DaemonError::Storage(format!("Bookmark lock poisoned: {e}")))?;

        let mut bookmarks = self.load_bookmarks()?;

        if !bookmarks.contains_key(name) {
            return Err(DaemonError::BookmarkNotFound(name.to_string()));
        }

        bookmarks.remove(name);
        self.save_bookmarks(&bookmarks)?;

        Ok(())
    }

    /// Resolves a conversation ID from various formats.
    ///
    /// Supports:
    /// - Full UUID
    /// - Short hash (7+ characters)
    /// - Bookmark name
    ///
    /// Resolution order: bookmark → full UUID → short hash prefix
    ///
    /// # Errors
    ///
    /// Returns an error if the conversation cannot be found.
    #[instrument(skip(self), fields(input = %id_or_name))]
    pub fn resolve_conversation_id(&self, id_or_name: &str) -> Result<Uuid> {
        let bookmarks = self.load_bookmarks()?;

        // Try bookmark lookup first
        if let Some(id_str) = bookmarks.get(id_or_name) {
            let id = Uuid::parse_str(id_str)
                .map_err(|_| DaemonError::InvalidConversationId(id_or_name.to_string()))?;
            debug!(resolved_id = %id, method = "bookmark", "Resolved conversation ID");
            return Ok(id);
        }

        // Try full UUID parse
        if let Ok(id) = Uuid::parse_str(id_or_name) {
            debug!(resolved_id = %id, method = "full_uuid", "Resolved conversation ID");
            return Ok(id);
        }

        // Try short hash prefix match (git-style)
        if id_or_name.len() >= 7 {
            let conversations = self.list_conversations()?;
            let matches: Vec<Uuid> = conversations
                .into_iter()
                .filter(|id| id.to_string().starts_with(id_or_name))
                .collect();

            match matches.len() {
                0 => {}
                1 => {
                    debug!(resolved_id = %matches[0], method = "short_hash", "Resolved conversation ID");
                    return Ok(matches[0]);
                }
                _ => {
                    return Err(DaemonError::InvalidConversationId(format!(
                        "Ambiguous short hash: {id_or_name}"
                    )));
                }
            }
        }

        Err(DaemonError::ConversationNotFound(id_or_name.to_string()))
    }

    /// Gets all bookmarks for a conversation.
    ///
    /// # Errors
    ///
    /// Returns an error if bookmark loading fails.
    pub fn get_conversation_bookmarks(&self, id: &Uuid) -> Result<Vec<String>> {
        let bookmarks = self.load_bookmarks()?;
        let id_str = id.to_string();

        let names: Vec<String> = bookmarks
            .iter()
            .filter_map(|(name, conv_id)| {
                if conv_id == &id_str {
                    Some(name.clone())
                } else {
                    None
                }
            })
            .collect();

        Ok(names)
    }

    /// Deletes a conversation file from disk.
    ///
    /// # Errors
    ///
    /// Returns an error if:
    /// - The conversation file doesn't exist
    /// - File deletion fails
    #[instrument(skip(self), fields(conversation_id = %id))]
    pub fn delete_conversation(&self, id: &Uuid) -> Result<()> {
        let path = self.conversation_path(id);

        if !path.exists() {
            return Err(DaemonError::ConversationNotFound(id.to_string()));
        }

        fs::remove_file(&path)?;
        debug!(conversation_id = %id, "Deleted conversation file");

        Ok(())
    }

    /// Clears the active conversation marker.
    ///
    /// Removes the `current` file. No-op if no active conversation is set.
    ///
    /// # Errors
    ///
    /// Returns an error if file deletion fails.
    pub fn clear_active_conversation(&self) -> Result<()> {
        if self.current_file.exists() {
            fs::remove_file(&self.current_file)?;
        }
        Ok(())
    }

    /// Removes all bookmarks pointing to a conversation.
    ///
    /// Returns the names of removed bookmarks.
    ///
    /// # Errors
    ///
    /// Returns an error if bookmark loading or saving fails.
    pub fn remove_bookmarks_for_conversation(&self, id: &Uuid) -> Result<Vec<String>> {
        let _guard = self
            .bookmarks_lock
            .lock()
            .map_err(|e| DaemonError::Storage(format!("Bookmark lock poisoned: {e}")))?;

        let mut bookmarks = self.load_bookmarks()?;
        let id_str = id.to_string();

        let removed: Vec<String> = bookmarks
            .iter()
            .filter(|(_, conv_id)| *conv_id == &id_str)
            .map(|(name, _)| name.clone())
            .collect();

        if !removed.is_empty() {
            for name in &removed {
                bookmarks.remove(name);
            }
            self.save_bookmarks(&bookmarks)?;
        }

        Ok(removed)
    }

    /// Returns the path to a conversation file.
    fn conversation_path(&self, id: &Uuid) -> PathBuf {
        self.conversations_dir.join(format!("{id}.json"))
    }

    /// Creates a test storage instance with a temporary directory.
    ///
    /// Only available for testing within the crate.
    #[cfg(test)]
    pub(crate) fn new_test(data_dir: &Path) -> Self {
        let conversations_dir = data_dir.join("conversations");
        fs::create_dir_all(&conversations_dir).ok();

        Self {
            conversations_dir,
            current_file: data_dir.join("current"),
            bookmarks_file: data_dir.join("bookmarks.json"),
            socket_path: data_dir.join("neuromance.sock"),
            pid_file: data_dir.join("neuromance.pid"),
            bookmarks_lock: Mutex::new(()),
        }
    }
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used)]
    #![allow(clippy::expect_used)]

    use super::*;
    use tempfile::TempDir;

    fn setup_test_storage() -> (Storage, TempDir) {
        let temp_dir = TempDir::new().unwrap();
        let data_dir = temp_dir.path().to_path_buf();

        let conversations_dir = data_dir.join("conversations");
        fs::create_dir_all(&conversations_dir).unwrap();

        let storage = Storage {
            conversations_dir,
            current_file: data_dir.join("current"),
            bookmarks_file: data_dir.join("bookmarks.json"),
            socket_path: data_dir.join("neuromance.sock"),
            pid_file: data_dir.join("neuromance.pid"),
            bookmarks_lock: Mutex::new(()),
        };

        (storage, temp_dir)
    }

    #[test]
    fn test_save_and_load_conversation() {
        let (storage, _temp) = setup_test_storage();

        let conv = Conversation::new().with_title("Test Conversation");
        storage.save_conversation(&conv).unwrap();

        let loaded = storage.load_conversation(&conv.id).unwrap();
        assert_eq!(loaded.id, conv.id);
        assert_eq!(loaded.title, conv.title);
    }

    #[test]
    fn test_list_conversations() {
        let (storage, _temp) = setup_test_storage();

        let conv1 = Conversation::new();
        let conv2 = Conversation::new();

        storage.save_conversation(&conv1).unwrap();
        storage.save_conversation(&conv2).unwrap();

        let ids = storage.list_conversations().unwrap();
        assert_eq!(ids.len(), 2);
        assert!(ids.contains(&conv1.id));
        assert!(ids.contains(&conv2.id));
    }

    #[test]
    fn test_active_conversation() {
        let (storage, _temp) = setup_test_storage();

        assert!(storage.get_active_conversation().unwrap().is_none());

        let conv_id = Uuid::new_v4();
        storage.set_active_conversation(&conv_id).unwrap();

        assert_eq!(storage.get_active_conversation().unwrap(), Some(conv_id));
    }

    #[test]
    fn test_bookmarks() {
        let (storage, _temp) = setup_test_storage();

        let conv_id = Uuid::new_v4();

        storage.set_bookmark("my-conv", &conv_id).unwrap();

        let bookmarks = storage.load_bookmarks().unwrap();
        assert_eq!(bookmarks.get("my-conv"), Some(&conv_id.to_string()));

        // Duplicate bookmark should fail
        assert!(storage.set_bookmark("my-conv", &conv_id).is_err());

        storage.remove_bookmark("my-conv").unwrap();
        let bookmarks = storage.load_bookmarks().unwrap();
        assert!(!bookmarks.contains_key("my-conv"));
    }

    #[test]
    fn test_resolve_conversation_id() {
        let (storage, _temp) = setup_test_storage();

        let conv = Conversation::new();
        storage.save_conversation(&conv).unwrap();
        storage.set_bookmark("test-bookmark", &conv.id).unwrap();

        // Resolve by full UUID
        let resolved = storage
            .resolve_conversation_id(&conv.id.to_string())
            .unwrap();
        assert_eq!(resolved, conv.id);

        // Resolve by bookmark
        let resolved = storage.resolve_conversation_id("test-bookmark").unwrap();
        assert_eq!(resolved, conv.id);

        // Resolve by short hash
        let short_hash = &conv.id.to_string()[..7];
        let resolved = storage.resolve_conversation_id(short_hash).unwrap();
        assert_eq!(resolved, conv.id);
    }

    #[test]
    fn test_delete_conversation() {
        let (storage, _temp) = setup_test_storage();

        let conv = Conversation::new().with_title("To Delete");
        storage.save_conversation(&conv).unwrap();

        // Verify it exists
        assert!(storage.load_conversation(&conv.id).is_ok());

        // Delete it
        storage.delete_conversation(&conv.id).unwrap();

        // Verify it's gone
        assert!(storage.load_conversation(&conv.id).is_err());
    }

    #[test]
    fn test_delete_conversation_not_found() {
        let (storage, _temp) = setup_test_storage();
        let id = Uuid::new_v4();
        assert!(matches!(
            storage.delete_conversation(&id),
            Err(DaemonError::ConversationNotFound(_))
        ));
    }

    #[test]
    fn test_clear_active_conversation() {
        let (storage, _temp) = setup_test_storage();

        let conv_id = Uuid::new_v4();
        storage.set_active_conversation(&conv_id).unwrap();
        assert!(storage.get_active_conversation().unwrap().is_some());

        storage.clear_active_conversation().unwrap();
        assert!(storage.get_active_conversation().unwrap().is_none());
    }

    #[test]
    fn test_clear_active_conversation_noop_when_unset() {
        let (storage, _temp) = setup_test_storage();
        // Should not error when no active conversation
        storage.clear_active_conversation().unwrap();
    }

    #[test]
    fn test_remove_bookmarks_for_conversation() {
        let (storage, _temp) = setup_test_storage();

        let conv_id = Uuid::new_v4();
        let other_id = Uuid::new_v4();

        storage.set_bookmark("bookmark1", &conv_id).unwrap();
        storage.set_bookmark("bookmark2", &conv_id).unwrap();
        storage.set_bookmark("other", &other_id).unwrap();

        let removed = storage.remove_bookmarks_for_conversation(&conv_id).unwrap();
        assert_eq!(removed.len(), 2);
        assert!(removed.contains(&"bookmark1".to_string()));
        assert!(removed.contains(&"bookmark2".to_string()));

        // Other bookmark should remain
        let remaining = storage.load_bookmarks().unwrap();
        assert_eq!(remaining.len(), 1);
        assert!(remaining.contains_key("other"));
    }

    #[test]
    fn test_remove_bookmarks_for_conversation_none() {
        let (storage, _temp) = setup_test_storage();
        let conv_id = Uuid::new_v4();

        let removed = storage.remove_bookmarks_for_conversation(&conv_id).unwrap();
        assert!(removed.is_empty());
    }

    #[test]
    fn test_get_conversation_bookmarks() {
        let (storage, _temp) = setup_test_storage();

        let conv_id = Uuid::new_v4();
        storage.set_bookmark("bookmark1", &conv_id).unwrap();
        storage.set_bookmark("bookmark2", &conv_id).unwrap();

        let bookmarks = storage.get_conversation_bookmarks(&conv_id).unwrap();
        assert_eq!(bookmarks.len(), 2);
        assert!(bookmarks.contains(&"bookmark1".to_string()));
        assert!(bookmarks.contains(&"bookmark2".to_string()));
    }
}
