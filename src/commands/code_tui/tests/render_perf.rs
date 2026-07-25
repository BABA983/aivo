//! Render-path timing probes, not assertions: run with
//! `cargo test --release --features __internal_test_fast_crypto render_perf -- --ignored --nocapture`
//! and compare the printed per-frame costs across changes.

use super::super::*;
use super::helpers::make_test_app;
use ratatui::Terminal;
use ratatui::backend::TestBackend;
use std::time::Instant;

/// A realistic assistant reply: prose, a fenced code block, and a list.
fn reply_block(i: usize) -> String {
    format!(
        "Looking at request #{i}, the router resolves the provider in three steps \
and falls back to the models cache when the probe times out.\n\n\
```rust\nfn resolve_{i}(key: &ApiKey) -> Route {{\n    let probe = probe_provider(key);\n    \
match probe {{\n        Ok(route) => route,\n        Err(_) => cached_route(key),\n    }}\n}}\n```\n\n\
- probe the native endpoint first\n- then the OpenAI-compatible bridge\n- finally the cached route from the last run\n\n\
The important part is that step {i} never blocks the event loop.",
    )
}

fn seed_large_history(app: &mut CodeTuiApp, exchanges: usize) {
    for i in 0..exchanges {
        app.history.push(ChatMessage {
            model: None,
            role: "user".to_string(),
            content: format!("question {i}: how does the provider router pick a route?"),
            reasoning_content: None,
            attachments: vec![],
        });
        app.history.push(ChatMessage {
            model: None,
            role: "assistant".to_string(),
            content: reply_block(i),
            reasoning_content: None,
            attachments: vec![],
        });
    }
}

fn time_frames(
    label: &str,
    app: &mut CodeTuiApp,
    terminal: &mut Terminal<TestBackend>,
    frames: u32,
    mut per_frame: impl FnMut(&mut CodeTuiApp),
) {
    // Warm the caches so the loop measures steady-state frames.
    terminal.draw(|frame| app.render(frame)).unwrap();
    let start = Instant::now();
    for _ in 0..frames {
        per_frame(app);
        app.frame_tick = app.frame_tick.wrapping_add(1);
        terminal.draw(|frame| app.render(frame)).unwrap();
    }
    let total = start.elapsed();
    println!(
        "{label}: {frames} frames in {total:?} → {:?}/frame",
        total / frames
    );
}

/// Steady-state frame with a large committed history and nothing animating —
/// the cost a keystroke repaint pays in a long session.
#[test]
#[ignore = "timing probe, run with --nocapture"]
fn bench_idle_frame_large_history() {
    for exchanges in [25usize, 100, 400] {
        let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
        let mut app = make_test_app(tx, rx);
        seed_large_history(&mut app, exchanges);
        let mut terminal = Terminal::new(TestBackend::new(120, 40)).unwrap();
        let label = format!("idle {exchanges}-exchange history");
        time_frames(&label, &mut app, &mut terminal, 120, |_| {});
    }
}

/// Spinner-animation frame mid-turn: large history plus a sizeable streamed
/// reply, no new content — the ~60fps steady state while a turn runs.
#[test]
#[ignore = "timing probe, run with --nocapture"]
fn bench_animating_frame_mid_turn() {
    let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
    let mut app = make_test_app(tx, rx);
    seed_large_history(&mut app, 100);
    app.sending = true;
    app.request_started_at = Some(Instant::now());
    app.pending_response = (0..40).map(reply_block).collect::<Vec<_>>().join("\n\n");
    let mut terminal = Terminal::new(TestBackend::new(120, 40)).unwrap();
    time_frames("animating mid-turn", &mut app, &mut terminal, 120, |_| {});
}

/// Where a typewriter tick's cost goes: markdown render vs styled wrap vs
/// plain prepass of the volatile tail, at a large reply size.
#[test]
#[ignore = "timing probe, run with --nocapture"]
fn bench_tail_stage_split() {
    use super::super::render::{wrap_plain_lines, wrap_transcript};
    let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
    let mut app = make_test_app(tx, rx);
    seed_large_history(&mut app, 1);
    app.sending = true;
    app.pending_response = (0..150).map(reply_block).collect::<Vec<_>>().join("\n\n");
    println!("reply bytes: {}", app.pending_response.len());
    let width = 116u16;

    let start = Instant::now();
    let mut blocks = (Vec::new(), Vec::new());
    for _ in 0..30 {
        blocks = app.volatile_tail_blocks(width);
    }
    println!(
        "volatile_tail_blocks (markdown): {:?}/call",
        start.elapsed() / 30
    );

    let (lines, bars) = blocks;
    let start = Instant::now();
    for _ in 0..30 {
        std::hint::black_box(wrap_transcript(&lines, &bars, width));
    }
    println!(
        "wrap_transcript (styled wrap): {:?}/call",
        start.elapsed() / 30
    );

    let plain: Vec<String> = lines.iter().map(|l| l.plain.clone()).collect();
    let start = Instant::now();
    for _ in 0..30 {
        std::hint::black_box(wrap_plain_lines(&plain, width).len());
    }
    println!(
        "wrap_plain_lines (prepass): {:?}/call",
        start.elapsed() / 30
    );
}

/// Scaling curve of the tail markdown render + wrap across reply sizes.
#[test]
#[ignore = "timing probe, run with --nocapture"]
fn bench_tail_scaling_curve() {
    use super::super::render::wrap_transcript;
    let width = 116u16;
    for blocks in [25usize, 50, 100, 150, 200] {
        let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
        let mut app = make_test_app(tx, rx);
        seed_large_history(&mut app, 1);
        app.sending = true;
        app.pending_response = (0..blocks)
            .map(reply_block)
            .collect::<Vec<_>>()
            .join("\n\n");
        let bytes = app.pending_response.len();
        let start = Instant::now();
        let (lines, bars) = app.volatile_tail_blocks(width);
        let md = start.elapsed();
        let start = Instant::now();
        std::hint::black_box(wrap_transcript(&lines, &bars, width));
        let wrap = start.elapsed();
        println!(
            "{blocks} blocks ({bytes} B, {} lines): md {md:?}, wrap {wrap:?}",
            lines.len()
        );
    }
}

/// Typewriter frame: each tick reveals a slice of the buffered stream, so the
/// volatile tail re-renders — the cost that scales with reply length.
#[test]
#[ignore = "timing probe, run with --nocapture"]
fn bench_typewriter_frame() {
    let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
    let mut app = make_test_app(tx, rx);
    seed_large_history(&mut app, 20);
    app.sending = true;
    app.request_started_at = Some(Instant::now());
    // A long reply already on screen, still typing out more — the worst case
    // for a per-tick tail re-render, which used to be O(reply) here.
    app.pending_response = (0..150).map(reply_block).collect::<Vec<_>>().join("\n\n");
    app.incoming_buffer = (150..200).map(reply_block).collect::<Vec<_>>().join("\n\n");
    let mut terminal = Terminal::new(TestBackend::new(120, 40)).unwrap();
    time_frames(
        "typewriter stream (74KB shown)",
        &mut app,
        &mut terminal,
        120,
        |app| {
            app.tick_typewriter();
        },
    );
}
