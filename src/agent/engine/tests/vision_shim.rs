//! Vision fallback: description substitution in outgoing copies.

use super::super::conversation::substitute_image_parts;
use super::super::*;
use crate::services::vision_describe::image_hash;
use serde_json::json;

fn image_part(b64: &str) -> Value {
    json!({"type": "image_url", "image_url": {"url": format!("data:image/png;base64,{b64}")}})
}

fn multimodal(text: &str, b64: &str) -> Value {
    json!([{ "type": "text", "text": text }, image_part(b64)])
}

fn engine_with_image_turn(b64: &str) -> AgentEngine {
    let mut e = AgentEngine::new("/tmp", "m", "", &[], &[], 0, 0);
    e.begin_user_turn(multimodal("look at this", b64), "look at this".into());
    e
}

#[test]
fn substitutes_only_when_enabled_and_keeps_history_intact() {
    let mut engine = engine_with_image_turn("aGVsbG8=");
    engine.insert_image_description(image_hash("aGVsbG8="), "[Image] desc".into());

    let untouched = engine.outgoing_messages();
    assert!(
        untouched.iter().any(|m| m["content"]
            .as_array()
            .is_some_and(|p| p.iter().any(|x| x["type"] == "image_url"))),
        "flag off: image part must survive"
    );

    engine.set_image_substitution(true);
    let out = engine.outgoing_messages();
    let user_parts = out
        .iter()
        .rev()
        .find(|m| m["role"] == "user")
        .and_then(|m| m["content"].as_array())
        .cloned()
        .unwrap();
    assert!(
        user_parts.iter().all(|p| p["type"] != "image_url"),
        "flag on: image part must be substituted"
    );
    assert!(
        user_parts
            .iter()
            .any(|p| p["type"] == "text" && p["text"] == "[Image] desc"),
        "description text part missing: {user_parts:?}"
    );
    assert!(
        engine.messages.iter().any(|m| m["content"]
            .as_array()
            .is_some_and(|p| p.iter().any(|x| x["type"] == "image_url"))),
        "self.messages must keep the image"
    );
}

#[test]
fn cache_miss_leaves_image_part_untouched() {
    let mut engine = engine_with_image_turn("aGVsbG8=");
    engine.set_image_substitution(true);
    let out = engine.outgoing_messages();
    assert!(
        out.iter().any(|m| m["content"]
            .as_array()
            .is_some_and(|p| p.iter().any(|x| x["type"] == "image_url"))),
        "no cached description → part must not be dropped or rewritten"
    );
}

#[test]
fn substitute_ignores_plain_string_messages() {
    let mut messages = vec![json!({"role": "user", "content": "plain text"})];
    let mut cache = std::collections::HashMap::new();
    cache.insert(image_hash("aGVsbG8="), "desc".to_string());
    substitute_image_parts(&mut messages, &cache);
    assert_eq!(messages[0]["content"], "plain text");
}

#[test]
fn undescribed_images_dedups_and_skips_cached() {
    let mut engine = AgentEngine::new("/tmp", "m", "", &[], &[], 0, 0);
    engine.begin_user_turn(multimodal("one", "aGVsbG8="), "one".into());
    // Same image again plus a new one — the repeat must not describe twice.
    engine.begin_user_turn(
        json!([image_part("aGVsbG8="), image_part("d29ybGQ=")]),
        "two".into(),
    );

    let todo = engine.undescribed_images(None);
    assert_eq!(todo.len(), 2, "unique images only: {todo:?}");

    engine.insert_image_description(image_hash("aGVsbG8="), "cached".into());
    let todo = engine.undescribed_images(None);
    assert_eq!(todo.len(), 1);
    assert_eq!(todo[0].0, image_hash("d29ybGQ="));
}

#[test]
fn undescribed_images_scans_pending_content() {
    let engine = AgentEngine::new("/tmp", "m", "", &[], &[], 0, 0);
    let pending = multimodal("incoming", "aGVsbG8=");
    let todo = engine.undescribed_images(Some(&pending));
    assert_eq!(todo.len(), 1);
    assert_eq!(todo[0].0, image_hash("aGVsbG8="));
    assert!(todo[0].1.starts_with("data:image/png;base64,"));
}
