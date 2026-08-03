use std::alloc::{alloc_zeroed, dealloc, Layout};
use std::fs::File;
#[cfg(any(target_os = "linux", target_os = "android", windows))]
use std::fs::OpenOptions;
use std::io;
use std::path::Path;
use std::ptr::NonNull;

use tokio_util::sync::CancellationToken;
use zeroize::{Zeroize, Zeroizing};

use crate::error::{PowerError, Result};

use super::WeightReadStrategy;

pub(super) const RANGE_READ_CHUNK_BYTES: usize = 1024 * 1024;
const MIN_DIRECT_ALIGNMENT: usize = 4 * 1024;

pub(super) struct WeightFileReader {
    buffered: File,
    cache_bypass: Option<File>,
    direct: Option<File>,
    verified_bytes: u64,
    io_block_size: u64,
}

impl WeightFileReader {
    pub(super) fn open(
        path: &Path,
        verified_bytes: u64,
        strategy: WeightReadStrategy,
    ) -> Result<Self> {
        let buffered = File::open(path)?;
        let metadata = buffered.metadata()?;
        if !metadata.is_file() || metadata.len() != verified_bytes {
            return Err(PowerError::InvalidFormat(
                "verified weight file identity changed before range indexing".to_string(),
            ));
        }
        let io_block_size = platform_block_size(&buffered, &metadata, strategy)?;
        let cache_bypass = if strategy == WeightReadStrategy::PositionalCacheBypass {
            Some(open_cache_bypass(path)?)
        } else {
            None
        };
        let direct = if strategy == WeightReadStrategy::PositionalDirect {
            Some(open_direct(path)?)
        } else {
            None
        };
        Ok(Self {
            buffered,
            cache_bypass,
            direct,
            verified_bytes,
            io_block_size,
        })
    }

    pub(super) fn io_block_size(&self) -> u64 {
        self.io_block_size
    }

    pub(super) fn read_range(
        &self,
        strategy: WeightReadStrategy,
        offset: u64,
        bytes: u64,
        cancellation: &CancellationToken,
    ) -> Result<Zeroizing<Vec<u8>>> {
        let end = offset
            .checked_add(bytes)
            .ok_or_else(|| PowerError::InvalidFormat("tensor byte range overflowed".to_string()))?;
        if end > self.verified_bytes {
            return Err(PowerError::InvalidFormat(
                "tensor byte range exceeds its verified source file".to_string(),
            ));
        }
        match strategy {
            WeightReadStrategy::Mmap => Err(PowerError::InvalidRequest(
                "mmap tensor reads do not use the positional range reader".to_string(),
            )),
            WeightReadStrategy::PositionalBuffered => {
                read_buffered(&self.buffered, offset, bytes, cancellation)
            }
            WeightReadStrategy::PositionalCacheBypass => {
                let cache_bypass = self.cache_bypass.as_ref().ok_or_else(|| {
                    PowerError::BackendNotAvailable(
                        "cache-bypass weight reads were not opened for this source".to_string(),
                    )
                })?;
                read_buffered(cache_bypass, offset, bytes, cancellation)
            }
            WeightReadStrategy::PositionalDirect => {
                let direct = self.direct.as_ref().ok_or_else(|| {
                    PowerError::BackendNotAvailable(
                        "direct weight reads were not opened for this source".to_string(),
                    )
                })?;
                read_direct(
                    direct,
                    offset,
                    bytes,
                    self.verified_bytes,
                    self.io_block_size,
                    cancellation,
                )
            }
        }
    }
}

#[cfg(target_os = "macos")]
pub(super) fn open_cache_bypass(path: &Path) -> Result<File> {
    let file = File::open(path).map_err(cache_bypass_open_error)?;
    set_macos_file_flag(&file, libc::F_NOCACHE, 1).map_err(cache_bypass_open_error)?;
    Ok(file)
}

#[cfg(target_os = "macos")]
fn set_macos_file_flag(file: &File, command: libc::c_int, value: libc::c_int) -> io::Result<()> {
    use std::os::fd::AsRawFd;

    loop {
        // SAFETY: `file` owns a valid descriptor for the duration of the call;
        // this macOS fcntl command accepts one integer argument.
        let result = unsafe { libc::fcntl(file.as_raw_fd(), command, value) };
        if result != -1 {
            return Ok(());
        }
        let error = io::Error::last_os_error();
        if error.kind() != io::ErrorKind::Interrupted {
            return Err(error);
        }
    }
}

#[cfg(not(target_os = "macos"))]
pub(super) fn open_cache_bypass(_path: &Path) -> Result<File> {
    Err(PowerError::BackendNotAvailable(
        "cache-bypass weight reads are supported only on macOS".to_string(),
    ))
}

#[cfg(target_os = "macos")]
fn cache_bypass_open_error(error: io::Error) -> PowerError {
    PowerError::BackendNotAvailable(format!(
        "macOS cache-bypass weight reads are unsupported by the selected storage source ({:?})",
        error.kind()
    ))
}

fn read_buffered(
    file: &File,
    offset: u64,
    bytes: u64,
    cancellation: &CancellationToken,
) -> Result<Zeroizing<Vec<u8>>> {
    let length = usize::try_from(bytes).map_err(|_| {
        PowerError::InvalidFormat("tensor byte length exceeds the host address range".to_string())
    })?;
    let mut output = Zeroizing::new(vec![0_u8; length]);
    read_exact_loop(
        output.as_mut_slice(),
        offset,
        cancellation,
        |buffer, position| read_at(file, buffer, position),
    )?;
    Ok(output)
}

fn read_exact_loop<F>(
    output: &mut [u8],
    offset: u64,
    cancellation: &CancellationToken,
    mut read_once: F,
) -> Result<()>
where
    F: FnMut(&mut [u8], u64) -> io::Result<usize>,
{
    let mut filled = 0_usize;
    while filled < output.len() {
        check_cancelled(cancellation)?;
        let chunk_end = filled
            .saturating_add(RANGE_READ_CHUNK_BYTES)
            .min(output.len());
        let mut chunk_filled = filled;
        while chunk_filled < chunk_end {
            check_cancelled(cancellation)?;
            let position = offset
                .checked_add(u64::try_from(chunk_filled).map_err(|_| {
                    PowerError::InvalidFormat("tensor read position overflowed".to_string())
                })?)
                .ok_or_else(|| {
                    PowerError::InvalidFormat("tensor read position overflowed".to_string())
                })?;
            match read_once(&mut output[chunk_filled..chunk_end], position) {
                Ok(0) => {
                    return Err(PowerError::InvalidFormat(
                        "verified weight file was truncated during a positional read".to_string(),
                    ));
                }
                Ok(read) => chunk_filled = chunk_filled.saturating_add(read),
                Err(error) if error.kind() == io::ErrorKind::Interrupted => continue,
                Err(error) => return Err(error.into()),
            }
        }
        filled = chunk_end;
    }
    check_cancelled(cancellation)
}

fn read_direct(
    file: &File,
    offset: u64,
    bytes: u64,
    verified_bytes: u64,
    block_size: u64,
    cancellation: &CancellationToken,
) -> Result<Zeroizing<Vec<u8>>> {
    let output_len = usize::try_from(bytes).map_err(|_| {
        PowerError::InvalidFormat("tensor byte length exceeds the host address range".to_string())
    })?;
    let alignment = direct_alignment(block_size)?;
    let scratch_len = align_up(RANGE_READ_CHUNK_BYTES, alignment)?;
    let mut scratch = AlignedBuffer::new(scratch_len, alignment)?;
    let mut output = Zeroizing::new(vec![0_u8; output_len]);
    let mut filled = 0_usize;

    while filled < output.len() {
        check_cancelled(cancellation)?;
        let absolute = offset
            .checked_add(u64::try_from(filled).map_err(|_| {
                PowerError::InvalidFormat("direct tensor read position overflowed".to_string())
            })?)
            .ok_or_else(|| {
                PowerError::InvalidFormat("direct tensor read position overflowed".to_string())
            })?;
        let aligned_offset = absolute - absolute % u64::try_from(alignment).unwrap_or(u64::MAX);
        let prefix = usize::try_from(absolute - aligned_offset).map_err(|_| {
            PowerError::InvalidFormat("direct tensor read prefix overflowed".to_string())
        })?;
        scratch.as_mut_slice().zeroize();
        let requested = prefix
            .checked_add(output.len() - filled)
            .ok_or_else(|| PowerError::InvalidFormat("direct read size overflowed".to_string()))?
            .min(scratch.as_slice().len());
        let read_len = align_up(requested, alignment)?.min(scratch.as_slice().len());
        let read = loop {
            match read_at(
                file,
                &mut scratch.as_mut_slice()[..read_len],
                aligned_offset,
            ) {
                Ok(read) => break read,
                Err(error) if error.kind() == io::ErrorKind::Interrupted => {
                    check_cancelled(cancellation)?;
                }
                Err(error) => return Err(direct_read_error(error)),
            }
        };
        if read <= prefix {
            let source_end = aligned_offset.saturating_add(u64::try_from(read).unwrap_or(u64::MAX));
            let message = if source_end < verified_bytes {
                "verified weight file returned an unusable short direct read"
            } else {
                "verified weight file was truncated during a direct read"
            };
            return Err(PowerError::InvalidFormat(message.to_string()));
        }
        let available = read - prefix;
        let copied = available.min(output.len() - filled);
        output[filled..filled + copied]
            .copy_from_slice(&scratch.as_slice()[prefix..prefix + copied]);
        filled += copied;
    }
    check_cancelled(cancellation)?;
    Ok(output)
}

fn check_cancelled(cancellation: &CancellationToken) -> Result<()> {
    if cancellation.is_cancelled() {
        Err(PowerError::InferenceFailed(
            "weight positional read was cancelled".to_string(),
        ))
    } else {
        Ok(())
    }
}

fn direct_alignment(block_size: u64) -> Result<usize> {
    let block_size = usize::try_from(block_size).map_err(|_| {
        PowerError::BackendNotAvailable(
            "direct weight read block size exceeds the host address range".to_string(),
        )
    })?;
    let candidate = block_size.max(MIN_DIRECT_ALIGNMENT);
    if !candidate.is_power_of_two() {
        return Err(PowerError::BackendNotAvailable(
            "direct weight read alignment is not a supported power of two".to_string(),
        ));
    }
    Ok(candidate)
}

fn align_up(value: usize, alignment: usize) -> Result<usize> {
    value
        .checked_add(alignment.saturating_sub(1))
        .map(|adjusted| adjusted / alignment * alignment)
        .ok_or_else(|| PowerError::InvalidFormat("direct read size overflowed".to_string()))
}

struct AlignedBuffer {
    pointer: NonNull<u8>,
    layout: Layout,
}

impl AlignedBuffer {
    fn new(bytes: usize, alignment: usize) -> Result<Self> {
        let layout = Layout::from_size_align(bytes, alignment).map_err(|_| {
            PowerError::BackendNotAvailable(
                "direct weight read alignment is unsupported".to_string(),
            )
        })?;
        // SAFETY: `layout` is non-zero, valid, and retained for deallocation.
        let pointer = unsafe { alloc_zeroed(layout) };
        let pointer = NonNull::new(pointer).ok_or_else(|| {
            PowerError::InferenceFailed(
                "failed to allocate the bounded direct weight read buffer".to_string(),
            )
        })?;
        Ok(Self { pointer, layout })
    }

    fn as_slice(&self) -> &[u8] {
        // SAFETY: the allocation is valid for exactly `layout.size()` bytes.
        unsafe { std::slice::from_raw_parts(self.pointer.as_ptr(), self.layout.size()) }
    }

    fn as_mut_slice(&mut self) -> &mut [u8] {
        // SAFETY: this object owns the allocation exclusively.
        unsafe { std::slice::from_raw_parts_mut(self.pointer.as_ptr(), self.layout.size()) }
    }
}

impl Drop for AlignedBuffer {
    fn drop(&mut self) {
        self.as_mut_slice().zeroize();
        // SAFETY: `pointer` was allocated with this exact layout and has not
        // been deallocated yet.
        unsafe { dealloc(self.pointer.as_ptr(), self.layout) };
    }
}

#[cfg(unix)]
fn read_at(file: &File, buffer: &mut [u8], offset: u64) -> io::Result<usize> {
    use std::os::unix::fs::FileExt;
    file.read_at(buffer, offset)
}

#[cfg(windows)]
fn read_at(file: &File, buffer: &mut [u8], offset: u64) -> io::Result<usize> {
    use std::os::windows::fs::FileExt;
    file.seek_read(buffer, offset)
}

#[cfg(not(any(unix, windows)))]
fn read_at(_file: &File, _buffer: &mut [u8], _offset: u64) -> io::Result<usize> {
    Err(io::Error::new(
        io::ErrorKind::Unsupported,
        "positional file reads are unsupported on this platform",
    ))
}

#[cfg(any(target_os = "linux", target_os = "android"))]
fn open_direct(path: &Path) -> Result<File> {
    use std::os::unix::fs::OpenOptionsExt;
    OpenOptions::new()
        .read(true)
        .custom_flags(libc::O_DIRECT | libc::O_CLOEXEC)
        .open(path)
        .map_err(direct_open_error)
}

#[cfg(windows)]
fn open_direct(path: &Path) -> Result<File> {
    use std::os::windows::fs::OpenOptionsExt;
    use windows_sys::Win32::Storage::FileSystem::FILE_FLAG_NO_BUFFERING;
    OpenOptions::new()
        .read(true)
        .custom_flags(FILE_FLAG_NO_BUFFERING)
        .open(path)
        .map_err(direct_open_error)
}

#[cfg(not(any(target_os = "linux", target_os = "android", windows)))]
fn open_direct(_path: &Path) -> Result<File> {
    Err(PowerError::BackendNotAvailable(
        "direct weight reads are unsupported on this platform".to_string(),
    ))
}

#[cfg(any(target_os = "linux", target_os = "android", windows))]
fn direct_open_error(error: io::Error) -> PowerError {
    PowerError::BackendNotAvailable(format!(
        "direct weight reads are unsupported by the selected storage source ({:?})",
        error.kind()
    ))
}

fn direct_read_error(error: io::Error) -> PowerError {
    match error.kind() {
        io::ErrorKind::InvalidInput | io::ErrorKind::Unsupported => {
            PowerError::BackendNotAvailable(format!(
                "direct weight reads are unsupported by the selected storage source ({:?})",
                error.kind()
            ))
        }
        _ => PowerError::Io(error),
    }
}

#[cfg(unix)]
fn platform_block_size(
    _file: &File,
    metadata: &std::fs::Metadata,
    _strategy: WeightReadStrategy,
) -> Result<u64> {
    use std::os::unix::fs::MetadataExt;
    Ok(metadata
        .blksize()
        .max(u64::try_from(MIN_DIRECT_ALIGNMENT).unwrap_or(4096)))
}

#[cfg(windows)]
fn platform_block_size(
    file: &File,
    _metadata: &std::fs::Metadata,
    strategy: WeightReadStrategy,
) -> Result<u64> {
    use std::mem::{size_of, MaybeUninit};
    use std::os::windows::io::AsRawHandle;
    use windows_sys::Win32::Storage::FileSystem::{
        FileStorageInfo, GetFileInformationByHandleEx, FILE_STORAGE_INFO,
    };

    let mut information = MaybeUninit::<FILE_STORAGE_INFO>::zeroed();
    // SAFETY: the file handle remains valid for the call, and the output
    // pointer names a correctly sized, writable FILE_STORAGE_INFO allocation.
    let result = unsafe {
        GetFileInformationByHandleEx(
            file.as_raw_handle(),
            FileStorageInfo,
            information.as_mut_ptr().cast(),
            u32::try_from(size_of::<FILE_STORAGE_INFO>()).unwrap_or(u32::MAX),
        )
    };
    if result == 0 {
        if strategy == WeightReadStrategy::PositionalDirect {
            return Err(PowerError::BackendNotAvailable(format!(
                "Windows could not query direct-I/O alignment ({:?})",
                io::Error::last_os_error().kind()
            )));
        }
        return Ok(u64::try_from(MIN_DIRECT_ALIGNMENT).unwrap_or(4096));
    }
    // SAFETY: a successful GetFileInformationByHandleEx initialized the full
    // FILE_STORAGE_INFO value.
    let information = unsafe { information.assume_init() };
    let alignment = [
        information.LogicalBytesPerSector,
        information.PhysicalBytesPerSectorForAtomicity,
        information.PhysicalBytesPerSectorForPerformance,
        information.FileSystemEffectivePhysicalBytesPerSectorForAtomicity,
    ]
    .into_iter()
    .map(u64::from)
    .max()
    .unwrap_or_default()
    .max(u64::try_from(MIN_DIRECT_ALIGNMENT).unwrap_or(4096));
    Ok(alignment)
}

#[cfg(not(any(unix, windows)))]
fn platform_block_size(
    _file: &File,
    _metadata: &std::fs::Metadata,
    _strategy: WeightReadStrategy,
) -> Result<u64> {
    Ok(u64::try_from(MIN_DIRECT_ALIGNMENT).unwrap_or(4096))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn exact_loop_accepts_honest_short_and_interrupted_reads() {
        let source = (0_u8..64).collect::<Vec<_>>();
        let mut output = vec![0_u8; source.len()];
        let mut calls = 0_usize;
        read_exact_loop(
            &mut output,
            0,
            &CancellationToken::new(),
            |buffer, offset| {
                calls += 1;
                if calls == 2 {
                    return Err(io::Error::from(io::ErrorKind::Interrupted));
                }
                let offset = usize::try_from(offset).unwrap();
                let read = buffer.len().min(7).min(source.len() - offset);
                buffer[..read].copy_from_slice(&source[offset..offset + read]);
                Ok(read)
            },
        )
        .unwrap();
        assert_eq!(output, source);
        assert!(calls > 2);
    }

    #[test]
    fn exact_loop_reports_cancellation_and_truncation() {
        let cancelled = CancellationToken::new();
        cancelled.cancel();
        assert!(matches!(
            read_exact_loop(&mut [0_u8; 8], 0, &cancelled, |_, _| Ok(8)),
            Err(PowerError::InferenceFailed(_))
        ));

        assert!(matches!(
            read_exact_loop(&mut [0_u8; 8], 0, &CancellationToken::new(), |_, _| Ok(0)),
            Err(PowerError::InvalidFormat(_))
        ));
    }

    #[test]
    fn exact_loop_observes_cancellation_between_chunks() {
        let cancellation = CancellationToken::new();
        let trigger = cancellation.clone();
        let mut output = vec![0_u8; RANGE_READ_CHUNK_BYTES + 1];
        let mut calls = 0_usize;
        let result = read_exact_loop(&mut output, 0, &cancellation, |buffer, _| {
            calls += 1;
            buffer.fill(7);
            trigger.cancel();
            Ok(buffer.len())
        });

        assert!(matches!(result, Err(PowerError::InferenceFailed(_))));
        assert_eq!(calls, 1);
    }

    #[test]
    fn alignment_arithmetic_is_bounded() {
        assert_eq!(direct_alignment(512).unwrap(), 4096);
        assert!(direct_alignment(6144).is_err());
        assert_eq!(align_up(1_048_577, 4096).unwrap(), 1_052_672);
        assert!(align_up(usize::MAX, 4096).is_err());
    }
}
