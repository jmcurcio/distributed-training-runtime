// rust/runtime-core/src/storage/local.rs

//! Local filesystem storage backend implementation.
//!
//! This module provides a storage backend that uses the local filesystem.
//! It supports both buffered I/O and memory-mapped I/O for efficient
//! access to large files.

use std::fs::{self, File, OpenOptions};
use std::io::{BufReader, BufWriter, Cursor, Read, Seek, SeekFrom, Write};
use std::path::{Path, PathBuf};

use memmap2::Mmap;

use super::traits::{ObjectMeta, StorageBackend, StorageReader, StorageWriter};
use crate::config::StorageConfig;
use crate::error::{Result, RuntimeError};

/// Local filesystem storage backend.
///
/// This backend stores objects as files on the local filesystem. It supports
/// both buffered I/O for small files and memory-mapped I/O for large files.
pub struct LocalStorage {
    /// Base path for all storage operations.
    base_path: PathBuf,
    /// Buffer size for buffered I/O operations.
    buffer_size: usize,
    /// Whether to use memory-mapped I/O.
    use_mmap: bool,
    /// File size threshold above which to use mmap.
    mmap_threshold: u64,
}

impl LocalStorage {
    /// Creates a new `LocalStorage` instance from configuration.
    ///
    /// # Errors
    ///
    /// Returns an error if the base path cannot be created.
    pub fn new(config: &StorageConfig) -> Result<Self> {
        let base_path = config.base_path.clone();

        // Create base directory if it doesn't exist
        if !base_path.exists() {
            fs::create_dir_all(&base_path).map_err(|e| {
                RuntimeError::storage_with_source(&base_path, "failed to create base directory", e)
            })?;
        }

        Ok(Self {
            base_path,
            buffer_size: config.buffer_size,
            use_mmap: config.use_mmap,
            mmap_threshold: config.mmap_threshold,
        })
    }

    /// Resolves a path relative to the base path.
    fn resolve_path(&self, path: &Path) -> PathBuf {
        if path.is_absolute() {
            path.to_path_buf()
        } else {
            self.base_path.join(path)
        }
    }
}

impl StorageBackend for LocalStorage {
    fn exists(&self, path: &Path) -> Result<bool> {
        let full_path = self.resolve_path(path);
        Ok(full_path.exists())
    }

    fn metadata(&self, path: &Path) -> Result<ObjectMeta> {
        let full_path = self.resolve_path(path);
        let meta = fs::metadata(&full_path).map_err(|e| {
            RuntimeError::storage_with_source(&full_path, "failed to read metadata", e)
        })?;

        Ok(ObjectMeta {
            size: meta.len(),
            modified: meta.modified().ok(),
            is_dir: meta.is_dir(),
        })
    }

    fn open_read(&self, path: &Path) -> Result<Box<dyn StorageReader>> {
        let full_path = self.resolve_path(path);
        let file = File::open(&full_path)
            .map_err(|e| RuntimeError::storage_with_source(&full_path, "failed to open file", e))?;

        let meta = file.metadata().map_err(|e| {
            RuntimeError::storage_with_source(&full_path, "failed to read file metadata", e)
        })?;
        let size = meta.len();

        // Use mmap for large files if enabled
        if self.use_mmap && size >= self.mmap_threshold {
            // SAFETY: The file is opened read-only and we maintain the Mmap
            // for the lifetime of the reader.
            let mmap = unsafe { Mmap::map(&file) }.map_err(|e| {
                RuntimeError::storage_with_source(&full_path, "failed to memory-map file", e)
            })?;

            Ok(Box::new(MmapReader::new(mmap)))
        } else {
            Ok(Box::new(LocalReader::new(file, size, self.buffer_size)))
        }
    }

    fn open_write(&self, path: &Path) -> Result<Box<dyn StorageWriter>> {
        let full_path = self.resolve_path(path);

        // Create parent directories if needed
        if let Some(parent) = full_path.parent() {
            if !parent.exists() {
                fs::create_dir_all(parent).map_err(|e| {
                    RuntimeError::storage_with_source(
                        parent,
                        "failed to create parent directories",
                        e,
                    )
                })?;
            }
        }

        let file = OpenOptions::new()
            .write(true)
            .create(true)
            .truncate(true)
            .open(&full_path)
            .map_err(|e| {
                RuntimeError::storage_with_source(&full_path, "failed to create file", e)
            })?;

        Ok(Box::new(LocalWriter::new(file, self.buffer_size)))
    }

    fn delete(&self, path: &Path) -> Result<()> {
        let full_path = self.resolve_path(path);

        if full_path.is_dir() {
            fs::remove_dir_all(&full_path).map_err(|e| {
                RuntimeError::storage_with_source(&full_path, "failed to delete directory", e)
            })
        } else {
            fs::remove_file(&full_path).map_err(|e| {
                RuntimeError::storage_with_source(&full_path, "failed to delete file", e)
            })
        }
    }

    fn list(&self, prefix: &Path) -> Result<Vec<String>> {
        let full_path = self.resolve_path(prefix);

        if !full_path.exists() {
            return Ok(Vec::new());
        }

        if !full_path.is_dir() {
            return Err(RuntimeError::storage(&full_path, "path is not a directory"));
        }

        let mut entries = Vec::new();

        for entry in fs::read_dir(&full_path).map_err(|e| {
            RuntimeError::storage_with_source(&full_path, "failed to read directory", e)
        })? {
            let entry = entry.map_err(|e| {
                RuntimeError::storage_with_source(&full_path, "failed to read directory entry", e)
            })?;

            if let Some(name) = entry.file_name().to_str() {
                entries.push(name.to_string());
            }
        }

        entries.sort();
        Ok(entries)
    }

    fn rename(&self, from: &Path, to: &Path) -> Result<()> {
        let from_path = self.resolve_path(from);
        let to_path = self.resolve_path(to);

        // Create parent directories for destination if needed
        if let Some(parent) = to_path.parent() {
            if !parent.exists() {
                fs::create_dir_all(parent).map_err(|e| {
                    RuntimeError::storage_with_source(
                        parent,
                        "failed to create parent directories",
                        e,
                    )
                })?;
            }
        }

        fs::rename(&from_path, &to_path).map_err(|e| {
            RuntimeError::storage_with_source(
                &from_path,
                format!("failed to rename to {}", to_path.display()),
                e,
            )
        })
    }

    fn create_dir_all(&self, path: &Path) -> Result<()> {
        let full_path = self.resolve_path(path);
        fs::create_dir_all(&full_path).map_err(|e| {
            RuntimeError::storage_with_source(&full_path, "failed to create directories", e)
        })
    }
}

/// Buffered file reader for local storage.
struct LocalReader {
    reader: BufReader<File>,
    size: u64,
}

impl LocalReader {
    fn new(file: File, size: u64, buffer_size: usize) -> Self {
        Self {
            reader: BufReader::with_capacity(buffer_size, file),
            size,
        }
    }
}

impl Read for LocalReader {
    fn read(&mut self, buf: &mut [u8]) -> std::io::Result<usize> {
        self.reader.read(buf)
    }
}

impl Seek for LocalReader {
    fn seek(&mut self, pos: SeekFrom) -> std::io::Result<u64> {
        self.reader.seek(pos)
    }
}

impl StorageReader for LocalReader {
    fn size(&self) -> u64 {
        self.size
    }

    fn read_range(&mut self, start: u64, length: usize) -> Result<Vec<u8>> {
        self.seek(SeekFrom::Start(start))
            .map_err(|e| RuntimeError::Storage {
                path: PathBuf::from("<file>"),
                message: format!("failed to seek to position {start}"),
                source: Some(e),
            })?;

        let mut buf = vec![0u8; length];
        self.read_exact(&mut buf)
            .map_err(|e| RuntimeError::Storage {
                path: PathBuf::from("<file>"),
                message: format!("failed to read {length} bytes at position {start}"),
                source: Some(e),
            })?;

        Ok(buf)
    }
}

/// Memory-mapped file reader for local storage.
struct MmapReader {
    mmap: Mmap,
    cursor: Cursor<Vec<u8>>,
}

impl MmapReader {
    fn new(mmap: Mmap) -> Self {
        // Create a cursor that shares the mmap data
        // We use a cursor over the mmap slice for seeking
        Self {
            cursor: Cursor::new(Vec::new()),
            mmap,
        }
    }
}

impl Read for MmapReader {
    fn read(&mut self, buf: &mut [u8]) -> std::io::Result<usize> {
        let pos = self.cursor.position() as usize;
        let remaining = &self.mmap[pos..];
        let to_read = buf.len().min(remaining.len());

        if to_read == 0 {
            return Ok(0);
        }

        buf[..to_read].copy_from_slice(&remaining[..to_read]);
        self.cursor.set_position((pos + to_read) as u64);
        Ok(to_read)
    }
}

impl Seek for MmapReader {
    fn seek(&mut self, pos: SeekFrom) -> std::io::Result<u64> {
        let new_pos = match pos {
            SeekFrom::Start(offset) => offset as i64,
            SeekFrom::End(offset) => self.mmap.len() as i64 + offset,
            SeekFrom::Current(offset) => self.cursor.position() as i64 + offset,
        };

        if new_pos < 0 {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidInput,
                "seek to negative position",
            ));
        }

        let new_pos = new_pos as u64;
        self.cursor.set_position(new_pos);
        Ok(new_pos)
    }
}

impl StorageReader for MmapReader {
    fn size(&self) -> u64 {
        self.mmap.len() as u64
    }

    fn read_range(&mut self, start: u64, length: usize) -> Result<Vec<u8>> {
        let start = start as usize;
        let end = start + length;

        if end > self.mmap.len() {
            return Err(RuntimeError::storage(
                "<mmap>",
                format!(
                    "read range {}..{} exceeds file size {}",
                    start,
                    end,
                    self.mmap.len()
                ),
            ));
        }

        Ok(self.mmap[start..end].to_vec())
    }
}

/// Buffered file writer for local storage.
struct LocalWriter {
    writer: BufWriter<File>,
}

impl LocalWriter {
    fn new(file: File, buffer_size: usize) -> Self {
        Self {
            writer: BufWriter::with_capacity(buffer_size, file),
        }
    }
}

impl Write for LocalWriter {
    fn write(&mut self, buf: &[u8]) -> std::io::Result<usize> {
        self.writer.write(buf)
    }

    fn flush(&mut self) -> std::io::Result<()> {
        self.writer.flush()
    }
}

impl StorageWriter for LocalWriter {
    fn finish(mut self: Box<Self>) -> Result<()> {
        self.writer.flush().map_err(|e| RuntimeError::Storage {
            path: PathBuf::from("<file>"),
            message: "failed to flush writer".to_string(),
            source: Some(e),
        })?;

        // Sync to disk
        self.writer
            .get_ref()
            .sync_all()
            .map_err(|e| RuntimeError::Storage {
                path: PathBuf::from("<file>"),
                message: "failed to sync file to disk".to_string(),
                source: Some(e),
            })?;

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    fn create_test_storage() -> (LocalStorage, TempDir) {
        let temp_dir = TempDir::new().unwrap();
        let config = StorageConfig {
            base_path: temp_dir.path().to_path_buf(),
            buffer_size: 4096,
            use_mmap: true,
            mmap_threshold: 1024, // Low threshold for testing
            ..Default::default()
        };
        let storage = LocalStorage::new(&config).unwrap();
        (storage, temp_dir)
    }

    #[test]
    fn test_new_creates_base_dir() {
        let temp_dir = TempDir::new().unwrap();
        let new_base = temp_dir.path().join("new_subdir");

        let config = StorageConfig {
            base_path: new_base.clone(),
            ..Default::default()
        };

        let _storage = LocalStorage::new(&config).unwrap();
        assert!(new_base.exists());
    }

    #[test]
    fn test_exists_file() {
        let (storage, _temp) = create_test_storage();

        // File doesn't exist yet
        assert!(!storage.exists(Path::new("test.txt")).unwrap());

        // Create a file
        let mut writer = storage.open_write(Path::new("test.txt")).unwrap();
        writer.write_all(b"hello").unwrap();
        writer.finish().unwrap();

        // Now it exists
        assert!(storage.exists(Path::new("test.txt")).unwrap());
    }

    #[test]
    fn test_exists_directory() {
        let (storage, _temp) = create_test_storage();

        // Directory doesn't exist yet
        assert!(!storage.exists(Path::new("subdir")).unwrap());

        // Create directory
        storage.create_dir_all(Path::new("subdir")).unwrap();

        // Now it exists
        assert!(storage.exists(Path::new("subdir")).unwrap());
    }

    #[test]
    fn test_metadata_file() {
        let (storage, _temp) = create_test_storage();

        // Write a file
        let data = b"hello world";
        let mut writer = storage.open_write(Path::new("test.txt")).unwrap();
        writer.write_all(data).unwrap();
        writer.finish().unwrap();

        // Check metadata
        let meta = storage.metadata(Path::new("test.txt")).unwrap();
        assert_eq!(meta.size, data.len() as u64);
        assert!(!meta.is_dir);
        assert!(meta.modified.is_some());
    }

    #[test]
    fn test_metadata_directory() {
        let (storage, _temp) = create_test_storage();

        // Create directory
        storage.create_dir_all(Path::new("subdir")).unwrap();

        // Check metadata
        let meta = storage.metadata(Path::new("subdir")).unwrap();
        assert!(meta.is_dir);
    }

    #[test]
    fn test_metadata_not_found() {
        let (storage, _temp) = create_test_storage();

        let result = storage.metadata(Path::new("nonexistent.txt"));
        assert!(result.is_err());
    }

    #[test]
    fn test_write_and_read_small_file() {
        let (storage, _temp) = create_test_storage();

        // Write a small file (below mmap threshold)
        let data = b"hello world";
        let mut writer = storage.open_write(Path::new("small.txt")).unwrap();
        writer.write_all(data).unwrap();
        writer.finish().unwrap();

        // Read it back
        let mut reader = storage.open_read(Path::new("small.txt")).unwrap();
        let mut buf = Vec::new();
        reader.read_to_end(&mut buf).unwrap();

        assert_eq!(buf, data);
        assert_eq!(reader.size(), data.len() as u64);
    }

    #[test]
    fn test_write_and_read_large_file() {
        let (storage, _temp) = create_test_storage();

        // Write a large file (above mmap threshold of 1024 bytes)
        let data: Vec<u8> = (0..2048).map(|i| (i % 256) as u8).collect();
        let mut writer = storage.open_write(Path::new("large.bin")).unwrap();
        writer.write_all(&data).unwrap();
        writer.finish().unwrap();

        // Read it back (should use mmap)
        let mut reader = storage.open_read(Path::new("large.bin")).unwrap();
        let mut buf = Vec::new();
        reader.read_to_end(&mut buf).unwrap();

        assert_eq!(buf, data);
        assert_eq!(reader.size(), data.len() as u64);
    }

    #[test]
    fn test_read_range_small_file() {
        let (storage, _temp) = create_test_storage();

        // Write a small file
        let data = b"hello world";
        let mut writer = storage.open_write(Path::new("small.txt")).unwrap();
        writer.write_all(data).unwrap();
        writer.finish().unwrap();

        // Read a range
        let mut reader = storage.open_read(Path::new("small.txt")).unwrap();
        let range = reader.read_range(6, 5).unwrap();

        assert_eq!(range, b"world");
    }

    #[test]
    fn test_read_range_large_file() {
        let (storage, _temp) = create_test_storage();

        // Write a large file
        let data: Vec<u8> = (0..2048).map(|i| (i % 256) as u8).collect();
        let mut writer = storage.open_write(Path::new("large.bin")).unwrap();
        writer.write_all(&data).unwrap();
        writer.finish().unwrap();

        // Read a range (should use mmap)
        let mut reader = storage.open_read(Path::new("large.bin")).unwrap();
        let range = reader.read_range(100, 50).unwrap();

        assert_eq!(range, &data[100..150]);
    }

    #[test]
    fn test_read_range_out_of_bounds() {
        let (storage, _temp) = create_test_storage();

        // Write a large file to use mmap
        let data: Vec<u8> = (0..2048).map(|i| (i % 256) as u8).collect();
        let mut writer = storage.open_write(Path::new("large.bin")).unwrap();
        writer.write_all(&data).unwrap();
        writer.finish().unwrap();

        // Try to read beyond bounds
        let mut reader = storage.open_read(Path::new("large.bin")).unwrap();
        let result = reader.read_range(2000, 100);

        assert!(result.is_err());
    }

    #[test]
    fn test_seek() {
        let (storage, _temp) = create_test_storage();

        // Write a file
        let data = b"0123456789";
        let mut writer = storage.open_write(Path::new("test.txt")).unwrap();
        writer.write_all(data).unwrap();
        writer.finish().unwrap();

        // Test seeking
        let mut reader = storage.open_read(Path::new("test.txt")).unwrap();

        // Seek from start
        assert_eq!(reader.seek(SeekFrom::Start(5)).unwrap(), 5);
        let mut buf = [0u8; 1];
        reader.read_exact(&mut buf).unwrap();
        assert_eq!(buf[0], b'5');

        // Seek from current
        assert_eq!(reader.seek(SeekFrom::Current(2)).unwrap(), 8);
        reader.read_exact(&mut buf).unwrap();
        assert_eq!(buf[0], b'8');

        // Seek from end
        assert_eq!(reader.seek(SeekFrom::End(-3)).unwrap(), 7);
        reader.read_exact(&mut buf).unwrap();
        assert_eq!(buf[0], b'7');
    }

    #[test]
    fn test_delete_file() {
        let (storage, _temp) = create_test_storage();

        // Create a file
        let mut writer = storage.open_write(Path::new("test.txt")).unwrap();
        writer.write_all(b"hello").unwrap();
        writer.finish().unwrap();

        assert!(storage.exists(Path::new("test.txt")).unwrap());

        // Delete it
        storage.delete(Path::new("test.txt")).unwrap();

        assert!(!storage.exists(Path::new("test.txt")).unwrap());
    }

    #[test]
    fn test_delete_directory() {
        let (storage, _temp) = create_test_storage();

        // Create a directory with files
        storage.create_dir_all(Path::new("subdir")).unwrap();
        let mut writer = storage.open_write(Path::new("subdir/file.txt")).unwrap();
        writer.write_all(b"hello").unwrap();
        writer.finish().unwrap();

        assert!(storage.exists(Path::new("subdir")).unwrap());

        // Delete the directory
        storage.delete(Path::new("subdir")).unwrap();

        assert!(!storage.exists(Path::new("subdir")).unwrap());
    }

    #[test]
    fn test_delete_not_found() {
        let (storage, _temp) = create_test_storage();

        let result = storage.delete(Path::new("nonexistent.txt"));
        assert!(result.is_err());
    }

    #[test]
    fn test_list_empty_directory() {
        let (storage, _temp) = create_test_storage();

        storage.create_dir_all(Path::new("empty")).unwrap();

        let entries = storage.list(Path::new("empty")).unwrap();
        assert!(entries.is_empty());
    }

    #[test]
    fn test_list_with_files() {
        let (storage, _temp) = create_test_storage();

        // Create some files
        storage.create_dir_all(Path::new("dir")).unwrap();

        for name in &["c.txt", "a.txt", "b.txt"] {
            let mut writer = storage.open_write(&Path::new("dir").join(name)).unwrap();
            writer.write_all(b"data").unwrap();
            writer.finish().unwrap();
        }

        // List should be sorted
        let entries = storage.list(Path::new("dir")).unwrap();
        assert_eq!(entries, vec!["a.txt", "b.txt", "c.txt"]);
    }

    #[test]
    fn test_list_nonexistent() {
        let (storage, _temp) = create_test_storage();

        let entries = storage.list(Path::new("nonexistent")).unwrap();
        assert!(entries.is_empty());
    }

    #[test]
    fn test_list_file_not_dir() {
        let (storage, _temp) = create_test_storage();

        // Create a file
        let mut writer = storage.open_write(Path::new("file.txt")).unwrap();
        writer.write_all(b"data").unwrap();
        writer.finish().unwrap();

        // List on a file should error
        let result = storage.list(Path::new("file.txt"));
        assert!(result.is_err());
    }

    #[test]
    fn test_rename_file() {
        let (storage, _temp) = create_test_storage();

        // Create a file
        let mut writer = storage.open_write(Path::new("old.txt")).unwrap();
        writer.write_all(b"hello").unwrap();
        writer.finish().unwrap();

        // Rename it
        storage
            .rename(Path::new("old.txt"), Path::new("new.txt"))
            .unwrap();

        assert!(!storage.exists(Path::new("old.txt")).unwrap());
        assert!(storage.exists(Path::new("new.txt")).unwrap());

        // Verify content
        let mut reader = storage.open_read(Path::new("new.txt")).unwrap();
        let mut buf = Vec::new();
        reader.read_to_end(&mut buf).unwrap();
        assert_eq!(buf, b"hello");
    }

    #[test]
    fn test_rename_creates_parent_dirs() {
        let (storage, _temp) = create_test_storage();

        // Create a file
        let mut writer = storage.open_write(Path::new("file.txt")).unwrap();
        writer.write_all(b"hello").unwrap();
        writer.finish().unwrap();

        // Rename to a nested path
        storage
            .rename(Path::new("file.txt"), Path::new("a/b/c/file.txt"))
            .unwrap();

        assert!(!storage.exists(Path::new("file.txt")).unwrap());
        assert!(storage.exists(Path::new("a/b/c/file.txt")).unwrap());
    }

    #[test]
    fn test_rename_not_found() {
        let (storage, _temp) = create_test_storage();

        let result = storage.rename(Path::new("nonexistent.txt"), Path::new("new.txt"));
        assert!(result.is_err());
    }

    #[test]
    fn test_create_dir_all() {
        let (storage, _temp) = create_test_storage();

        storage.create_dir_all(Path::new("a/b/c/d")).unwrap();

        assert!(storage.exists(Path::new("a/b/c/d")).unwrap());
        let meta = storage.metadata(Path::new("a/b/c/d")).unwrap();
        assert!(meta.is_dir);
    }

    #[test]
    fn test_create_dir_all_existing() {
        let (storage, _temp) = create_test_storage();

        storage.create_dir_all(Path::new("existing")).unwrap();
        // Should not error when called again
        storage.create_dir_all(Path::new("existing")).unwrap();
    }

    #[test]
    fn test_write_creates_parent_dirs() {
        let (storage, _temp) = create_test_storage();

        // Write to a nested path
        let mut writer = storage
            .open_write(Path::new("nested/path/file.txt"))
            .unwrap();
        writer.write_all(b"hello").unwrap();
        writer.finish().unwrap();

        assert!(storage.exists(Path::new("nested/path/file.txt")).unwrap());
    }

    #[test]
    fn test_overwrite_file() {
        let (storage, _temp) = create_test_storage();

        // Write initial content
        let mut writer = storage.open_write(Path::new("file.txt")).unwrap();
        writer.write_all(b"initial").unwrap();
        writer.finish().unwrap();

        // Overwrite
        let mut writer = storage.open_write(Path::new("file.txt")).unwrap();
        writer.write_all(b"new").unwrap();
        writer.finish().unwrap();

        // Verify new content
        let mut reader = storage.open_read(Path::new("file.txt")).unwrap();
        let mut buf = Vec::new();
        reader.read_to_end(&mut buf).unwrap();
        assert_eq!(buf, b"new");
    }

    #[test]
    fn test_mmap_disabled() {
        let temp_dir = TempDir::new().unwrap();
        let config = StorageConfig {
            base_path: temp_dir.path().to_path_buf(),
            buffer_size: 4096,
            use_mmap: false, // Disable mmap
            mmap_threshold: 1024,
            ..Default::default()
        };
        let storage = LocalStorage::new(&config).unwrap();

        // Write a large file
        let data: Vec<u8> = (0..2048).map(|i| (i % 256) as u8).collect();
        let mut writer = storage.open_write(Path::new("large.bin")).unwrap();
        writer.write_all(&data).unwrap();
        writer.finish().unwrap();

        // Should still work without mmap
        let mut reader = storage.open_read(Path::new("large.bin")).unwrap();
        let mut buf = Vec::new();
        reader.read_to_end(&mut buf).unwrap();

        assert_eq!(buf, data);
    }

    #[test]
    fn test_object_safety() {
        let (storage, _temp) = create_test_storage();

        // Verify StorageBackend can be used as a trait object
        let backend: Box<dyn StorageBackend> = Box::new(storage);

        // Create a file through the trait object
        let mut writer = backend.open_write(Path::new("test.txt")).unwrap();
        writer.write_all(b"hello").unwrap();
        writer.finish().unwrap();

        assert!(backend.exists(Path::new("test.txt")).unwrap());
    }
}
