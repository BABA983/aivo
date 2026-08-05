//! Cross-platform "open this URL in the user's default browser".
//!
//! Best-effort only; the caller must be prepared to fall back to printing
//! the URL and asking the user to paste the callback URL back.

#[cfg(any(target_os = "macos", target_os = "linux"))]
use std::process::{Command, Stdio};

#[cfg(target_os = "macos")]
pub fn open_url(url: &str) -> std::io::Result<()> {
    Command::new("open")
        .arg(url)
        .stdin(Stdio::null())
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .spawn()
        .map(|_| ())
}

#[cfg(target_os = "linux")]
pub fn open_url(url: &str) -> std::io::Result<()> {
    if let Ok(child) = Command::new("xdg-open")
        .arg(url)
        .stdin(Stdio::null())
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .spawn()
    {
        drop(child);
        return Ok(());
    }
    Command::new("sensible-browser")
        .arg(url)
        .stdin(Stdio::null())
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .spawn()
        .map(|_| ())
}

#[cfg(target_os = "windows")]
pub fn open_url(url: &str) -> std::io::Result<()> {
    // `cmd /C start` reparses `&`/`%xx`; `explorer.exe` rejects query-string
    // URLs. ShellExecuteW passes the URL through intact.
    use std::os::windows::ffi::OsStrExt;
    let wide = |s: &str| {
        std::ffi::OsStr::new(s)
            .encode_wide()
            .chain(std::iter::once(0))
            .collect::<Vec<u16>>()
    };
    let operation = wide("open");
    let file = wide(url);
    const SW_SHOWNORMAL: i32 = 1;
    let code = unsafe {
        windows_sys::Win32::UI::Shell::ShellExecuteW(
            std::ptr::null_mut(),
            operation.as_ptr(),
            file.as_ptr(),
            std::ptr::null(),
            std::ptr::null(),
            SW_SHOWNORMAL,
        )
    } as isize;
    // success is > 32; anything else an SE_ERR_* code
    if code > 32 {
        Ok(())
    } else {
        Err(std::io::Error::other(format!(
            "ShellExecuteW failed (code {code})"
        )))
    }
}

#[cfg(not(any(target_os = "macos", target_os = "linux", target_os = "windows")))]
pub fn open_url(_url: &str) -> std::io::Result<()> {
    Err(std::io::Error::new(
        std::io::ErrorKind::Unsupported,
        "no known browser launcher for this platform",
    ))
}
