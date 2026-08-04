//! Process-global flags set by `--transform` / `--transparent`.
//!
//! `ACTIVE` mirrors `http_debug::is_debug_active()` in role: both opt pi into
//! the responses-to-chat router branch (instead of the direct-to-upstream
//! default). `--debug` does so implicitly because its JSONL logger needs
//! the router to capture pi's traffic; `--transform` does so explicitly
//! to normalize SSE for upstreams that omit `finish_reason`.
//! Pi defaults to on (`--transparent` opts out); the default is set at the
//! run dispatch (src/run.rs), so the `aivo start` picker stays direct.
//!
//! `TRANSPARENT` is the non-pi face of `--transparent`: a force-direct
//! override consulted by `router_selection`; `--debug` wins over it.

use std::sync::atomic::{AtomicBool, Ordering};

static ACTIVE: AtomicBool = AtomicBool::new(false);
static TRANSPARENT: AtomicBool = AtomicBool::new(false);

pub fn set_active(active: bool) {
    ACTIVE.store(active, Ordering::SeqCst);
}

pub fn is_active() -> bool {
    ACTIVE.load(Ordering::SeqCst)
}

pub fn set_transparent(active: bool) {
    TRANSPARENT.store(active, Ordering::SeqCst);
}

pub fn is_transparent() -> bool {
    TRANSPARENT.load(Ordering::SeqCst)
}
