//! Vision fallback: dispatch tri-state, refusal wording, describe events.

use super::super::*;
use super::helpers::*;
use crate::services::session_store::VisionFallbackMode;
use crate::services::vision_describe::DESCRIBE_EXHAUSTED;

fn image_attachment() -> MessageAttachment {
    MessageAttachment {
        name: "shot.png".to_string(),
        mime_type: "image/png".to_string(),
        storage: AttachmentStorage::Inline {
            data: "iVBOR".to_string(),
        },
    }
}

#[tokio::test]
async fn exhausted_latch_refuses_with_quota_wording() {
    let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
    let mut app = make_test_app(tx, rx);
    app.model_image_input = Some(false);
    app.vision_fallback = VisionFallbackMode::Gateway;
    app.draft_attachments.push(image_attachment());

    let guard = crate::services::vision_describe::TEST_DESCRIBE_LOCK
        .lock()
        .await;
    DESCRIBE_EXHAUSTED.store(true, std::sync::atomic::Ordering::Relaxed);
    let dispatched = app.dispatch_user_message("look".to_string(), None).await;
    DESCRIBE_EXHAUSTED.store(false, std::sync::atomic::Ordering::Relaxed);
    drop(guard);
    dispatched.unwrap();

    let msg = notice_text(&app);
    assert!(msg.contains("quota used up"), "got: {msg}");
    assert!(!app.sending, "nothing went out");
    assert_eq!(app.draft_attachments.len(), 1, "attachment retained");
}

#[tokio::test]
async fn custom_mode_with_missing_key_refuses_with_hint() {
    let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
    let mut app = make_test_app(tx, rx);
    app.model_image_input = Some(false);
    app.vision_fallback = VisionFallbackMode::Custom;
    app.vision_fallback_custom = Some(("gone-key-id".to_string(), "gemini-2.5-flash".to_string()));
    app.draft_attachments.push(image_attachment());

    app.dispatch_user_message("look".to_string(), None)
        .await
        .unwrap();

    let msg = notice_text(&app);
    assert!(msg.contains("can't read images"), "got: {msg}");
    assert!(
        msg.contains("/config"),
        "custom refusal points at /config: {msg}"
    );
    assert!(!app.sending);
}

/// The per-model upstream would hijack the main chat's routing.
#[tokio::test]
async fn custom_describer_matching_active_model_is_rejected() {
    let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
    let mut app = make_test_app(tx, rx);
    app.model = "same-model".to_string();
    app.vision_fallback = VisionFallbackMode::Custom;
    app.vision_fallback_custom = Some(("some-key".to_string(), "same-model".to_string()));
    assert!(app.resolve_describer().await.is_err());
}

#[tokio::test]
async fn mixed_attachments_keep_plain_refusal() {
    let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
    let mut app = make_test_app(tx, rx);
    app.model_image_input = Some(false);
    app.vision_fallback = VisionFallbackMode::Gateway;
    app.draft_attachments.push(image_attachment());
    app.draft_attachments.push(MessageAttachment {
        name: "doc.pdf".to_string(),
        mime_type: "application/pdf".to_string(),
        storage: AttachmentStorage::Inline {
            data: "JVBER".to_string(),
        },
    });

    app.dispatch_user_message("look".to_string(), None)
        .await
        .unwrap();

    let msg = notice_text(&app);
    assert!(msg.contains("can't read images"), "got: {msg}");
    assert!(!app.sending);
    assert_eq!(app.draft_attachments.len(), 2, "both attachments retained");
}

#[tokio::test]
async fn known_vision_and_unknown_models_bypass_the_shim() {
    let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
    let mut app = make_test_app(tx, rx);
    pin_to_plain_chat(&mut app);
    app.dispatch_user_message("follow-up".to_string(), None)
        .await
        .unwrap();
    assert!(app.sending, "plain-chat turn went out");
    assert!(
        notice_text(&app).contains("plain"),
        "unknown-vision notice unchanged: {}",
        notice_text(&app)
    );
}

#[tokio::test]
async fn image_described_event_populates_session_cache() {
    let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
    let mut app = make_test_app(tx.clone(), rx);
    tx.send(RuntimeEvent::ImageDescribed {
        hash: "abc123".to_string(),
        text: "[Image] a red button".to_string(),
    })
    .unwrap();
    app.handle_runtime_events().await.unwrap();
    assert_eq!(
        app.vision_descriptions.get("abc123").map(String::as_str),
        Some("[Image] a red button")
    );
}

#[tokio::test]
async fn vision_model_flag_resolution_forms() {
    let dir = crate::test_sandbox::tmp("aivo-test");
    let store = SessionStore::with_path(dir.join("config.json"));
    let or_id = store
        .add_key_with_protocol("or", "https://openrouter.ai/api/v1", None, "sk-x")
        .await
        .unwrap();
    let active = ApiKey::new_with_protocol(
        "active".to_string(),
        "test".to_string(),
        "https://api.anthropic.com".to_string(),
        None,
        String::new(),
    );

    let resolve = async |spec: &str| match parse_vision_flag(spec) {
        VisionFlag::Describer { key, model } => {
            resolve_vision_model_override(&store, &active, "active-model", key, model).await
        }
        _ => Err("picker".to_string()),
    };

    let (key_id, model) = resolve("gemini-2.5-flash")
        .await
        .expect("bare model uses the active key");
    assert_eq!(key_id, active.id);
    assert_eq!(model, "gemini-2.5-flash");

    let (key_id, _) = resolve("or::gemini-2.5-flash")
        .await
        .expect("key:: half resolves by name");
    assert_eq!(key_id, or_id);

    let err = resolve("nope::x").await.expect_err("unknown key");
    assert!(err.contains("no key named"), "{err}");

    let err = resolve("deepseek-chat").await.expect_err("text-only model");
    assert!(err.contains("isn't a vision model"), "{err}");

    assert!(matches!(parse_vision_flag("or::"), VisionFlag::KeyPicker(q) if q == "or"));
    assert!(matches!(parse_vision_flag(""), VisionFlag::Picker));
}

#[tokio::test]
async fn apply_vision_describer_persists_or_refuses() {
    let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
    let mut app = make_test_app(tx, rx);
    let key = app.key.clone();

    app.apply_vision_describer(key.clone(), "deepseek-chat".to_string())
        .await;
    assert_ne!(app.vision_fallback, VisionFallbackMode::Custom);
    assert!(
        notice_text(&app).contains("isn't a vision model"),
        "{}",
        notice_text(&app)
    );

    app.model = "same-model".to_string();
    app.apply_vision_describer(key.clone(), "same-model".to_string())
        .await;
    assert_ne!(app.vision_fallback, VisionFallbackMode::Custom);
    assert!(
        notice_text(&app).contains("can't be the active model"),
        "{}",
        notice_text(&app)
    );

    app.apply_vision_describer(key.clone(), "gemini-2.5-flash".to_string())
        .await;
    assert_eq!(app.vision_fallback, VisionFallbackMode::Custom);
    let persisted = app.session_store.get_chat_toggles().await;
    assert_eq!(
        persisted.vision_fallback_custom,
        Some((key.id.clone(), "gemini-2.5-flash".to_string()))
    );
    assert_eq!(persisted.vision_fallback, VisionFallbackMode::Custom);
    match &app.overlay {
        Overlay::Config(state) => assert_eq!(
            state.items[state.selected].setting,
            ConfigSetting::VisionFallback
        ),
        other => panic!("expected config overlay, got {}", overlay_name(other)),
    }
}

#[tokio::test]
async fn config_custom_opens_key_then_model_picker() {
    let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
    let mut app = make_test_app(tx, rx);
    app.session_store
        .add_key_with_protocol("or", "https://openrouter.ai/api/v1", None, "sk-x")
        .await
        .unwrap();
    // A second key keeps the key stage in play (one key would skip it).
    let zhipu_id = app
        .session_store
        .add_key_with_protocol("zhipu", "https://api.z.ai/api/paas/v4", None, "sk-z")
        .await
        .unwrap();
    app.open_config_overlay();
    let row = match &app.overlay {
        Overlay::Config(state) => state
            .items
            .iter()
            .position(|i| i.setting == ConfigSetting::VisionFallback)
            .expect("vision row present"),
        _ => panic!("expected config overlay"),
    };

    // custom with NO stored pair: opens the picker, mode unchanged.
    app.cycle_config_setting(row, CycleDir::Enter).await;
    assert_eq!(
        app.vision_fallback,
        VisionFallbackMode::Gateway,
        "unchanged"
    );
    match &app.overlay {
        Overlay::Picker(picker) => assert!(matches!(
            picker.kind,
            PickerKind::Key {
                target: KeySelectionTarget::VisionDescriber
            }
        )),
        other => panic!("expected key picker, got {}", overlay_name(other)),
    }

    // Esc backs out to /config, mode still unchanged.
    app.handle_key(KeyEvent::new(KeyCode::Esc, KeyModifiers::NONE))
        .await
        .unwrap();
    match &app.overlay {
        Overlay::Config(state) => assert_eq!(
            state.items[state.selected].setting,
            ConfigSetting::VisionFallback
        ),
        other => panic!("expected config overlay, got {}", overlay_name(other)),
    }
    assert_eq!(app.vision_fallback, VisionFallbackMode::Gateway);

    // With a stored pair it's a PLAIN selection — no picker.
    app.vision_fallback_custom = Some((zhipu_id, "glm-4.6v".to_string()));
    app.cycle_config_setting(row, CycleDir::Enter).await;
    assert_eq!(app.vision_fallback, VisionFallbackMode::Custom, "selected");
    assert!(
        matches!(app.overlay, Overlay::Config(_)),
        "no picker on plain selection"
    );

    // Enter on the ACTIVE custom re-opens the picker.
    app.cycle_config_setting(row, CycleDir::Enter).await;
    match &app.overlay {
        Overlay::Picker(picker) => {
            assert!(matches!(
                picker.kind,
                PickerKind::Key {
                    target: KeySelectionTarget::VisionDescriber
                }
            ));
            assert_eq!(picker.selected, 1, "stored key pre-selected");
        }
        other => panic!("expected key picker, got {}", overlay_name(other)),
    }

    // The secret must be DECRYPTED — an encrypted one 401s the live /v1/models
    // fetch and reads as "No models available for this provider".
    app.activate_picker_selection(0).await.unwrap();
    match &app.overlay {
        Overlay::Picker(picker) => match &picker.kind {
            PickerKind::Model {
                target: ModelSelectionTarget::VisionDescriber(key),
                ..
            } => assert_eq!(key.key.as_str(), "sk-x", "secret must be decrypted"),
            _ => panic!("expected a vision-describer model picker"),
        },
        other => panic!("expected model picker, got {}", overlay_name(other)),
    }

    // Tab/Space on the active custom ADVANCES to off — only Enter re-opens it.
    app.open_config_overlay();
    app.cycle_config_setting(row, CycleDir::Next).await;
    assert_eq!(app.vision_fallback, VisionFallbackMode::Off, "tab advances");
    assert!(
        matches!(app.overlay, Overlay::Config(_)),
        "tab must not open the picker"
    );
    assert_eq!(app.config_segments(ConfigSetting::VisionFallback).active, 2);

    app.vision_fallback = VisionFallbackMode::Custom;
    app.step_config_setting(row, 1).await;
    assert_eq!(app.vision_fallback, VisionFallbackMode::Off);
}

/// aivo's own gateway rejects `image_url` parts in serde wording.
#[test]
fn image_input_rejection_matches_gateway_serde_wording() {
    use super::super::event_loop_impl::is_image_input_rejection;
    assert!(is_image_input_rejection(
        "API returned 400 Bad Request — {\"error\":{\"message\":\"Failed to deserialize the JSON \
body into the target type: messages[0]: unknown variant `image_url`, expected `text` at line 1\"}}"
    ));
    assert!(is_image_input_rejection("this model has no image input"));
    assert!(!is_image_input_rejection("rate limit exceeded"));
    assert!(!is_image_input_rejection("unknown variant `foo`"));
}

#[tokio::test]
async fn image_rejection_learns_text_only_for_the_session() {
    let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
    let mut app = make_test_app(tx, rx);
    app.model = "aivo/starter".to_string();
    app.model_image_input = None; // snapshot-unknown
    // Off → no auto-retry; the learning itself is what we assert.
    app.vision_fallback = VisionFallbackMode::Off;
    app.history.push(ChatMessage {
        model: None,
        role: "user".to_string(),
        content: "what's in this".to_string(),
        reasoning_content: None,
        attachments: vec![image_attachment()],
    });
    app.pending_submit = Some(PendingSubmission {
        content: "what's in this".to_string(),
        attachments: vec![image_attachment()],
    });

    app.finish_failed_response(
        "API returned 400 Bad Request — messages[0]: unknown variant `image_url`, expected `text`"
            .to_string(),
    )
    .await;

    assert_eq!(app.model_image_input, Some(false), "capability learned");
    assert_eq!(app.draft, "what's in this", "composer restored");
    assert_eq!(app.draft_attachments.len(), 1);
    assert!(
        notice_text(&app).contains("can't read images"),
        "{}",
        notice_text(&app)
    );
}

#[test]
fn describer_picker_filters_to_vision_models_cheapest_first() {
    use super::super::event_loop_impl::filter_vision_choices;
    let choice = |id: &str| ModelChoice {
        label: id.to_string(),
        id: id.to_string(),
    };
    let filtered = filter_vision_choices(vec![
        choice("gpt-4o"),                 // vision, pricey
        choice("gemini-2.5-flash"),       // vision, mid
        choice("deepseek-chat"),          // snapshot: text-only
        choice("bytedance/seedream-4.0"), // image-GEN model, unknown to the snapshot
        choice("gemini-2.5-flash-lite"),  // vision, cheapest
    ]);
    let ids: Vec<&str> = filtered.iter().map(|m| m.id.as_str()).collect();
    assert_eq!(
        ids,
        ["gemini-2.5-flash-lite", "gemini-2.5-flash", "gpt-4o"],
        "confirmed vision models only, price-ascending"
    );

    let unknown_only = filter_vision_choices(vec![choice("my-local-llava"), choice("mystery")]);
    assert_eq!(unknown_only.len(), 2, "all-unknown catalogs pass through");
}

#[tokio::test]
async fn attaching_an_image_announces_the_describer() {
    let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
    let mut app = make_test_app(tx, rx);
    app.model_image_input = Some(false);
    app.vision_fallback = VisionFallbackMode::Custom;
    app.vision_fallback_custom = Some(("k1".to_string(), "gemini-2.5-flash-lite".to_string()));

    let hinted = app.with_vision_attach_hint("Pasted image: shot.png".to_string(), true);
    assert!(
        hinted.contains("described via gemini-2.5-flash-lite"),
        "{hinted}"
    );

    let plain = app.with_vision_attach_hint("Queued file: a.pdf".to_string(), false);
    assert_eq!(plain, "Queued file: a.pdf");
    app.vision_fallback = VisionFallbackMode::Off;
    let off = app.with_vision_attach_hint("Pasted image: shot.png".to_string(), true);
    assert_eq!(off, "Pasted image: shot.png");
}

#[tokio::test]
async fn config_row_description_shows_current_describer() {
    let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
    let mut app = make_test_app(tx, rx);
    app.vision_fallback = VisionFallbackMode::Custom;
    app.vision_fallback_custom = Some(("k1".to_string(), "glm-4.6v".to_string()));
    app.open_config_overlay();
    let Overlay::Config(state) = &app.overlay else {
        panic!("expected config overlay");
    };
    let row = state
        .items
        .iter()
        .find(|i| i.setting == ConfigSetting::VisionFallback)
        .unwrap();
    assert!(row.description.contains("glm-4.6v"), "{}", row.description);
}

#[tokio::test]
async fn config_reverse_cycle_and_row_wrap() {
    let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
    let mut app = make_test_app(tx, rx);
    app.vision_fallback_custom = Some(("k1".to_string(), "glm-4.6v".to_string()));
    app.open_config_overlay();

    // BackTab reaches the overlay, not the global mode chord.
    app.handle_key(KeyEvent::new(KeyCode::BackTab, KeyModifiers::NONE))
        .await
        .unwrap();
    assert_eq!(
        app.theme,
        UiTheme::Light,
        "reverse cycle wrapped dark→light"
    );
    assert!(!app.agent_auto_approve, "mode chord must not fire");

    // gateway wraps back to off, then custom (pair present, so no picker).
    let row = match &app.overlay {
        Overlay::Config(state) => state
            .items
            .iter()
            .position(|i| i.setting == ConfigSetting::VisionFallback)
            .unwrap(),
        _ => panic!("expected config overlay"),
    };
    app.cycle_config_setting(row, CycleDir::Prev).await;
    assert_eq!(app.vision_fallback, VisionFallbackMode::Off);
    app.cycle_config_setting(row, CycleDir::Prev).await;
    assert_eq!(app.vision_fallback, VisionFallbackMode::Custom);
    assert!(matches!(app.overlay, Overlay::Config(_)), "no picker");

    if let Overlay::Config(state) = &mut app.overlay {
        state.selected = 0;
        state.select_prev();
        assert_eq!(state.selected, state.items.len() - 1, "top wraps to bottom");
        state.select_next();
        assert_eq!(state.selected, 0, "bottom wraps to top");
    }
}

#[tokio::test]
async fn key_only_flag_spec_opens_model_picker() {
    let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
    let mut app = make_test_app(tx, rx);
    app.session_store
        .add_key_with_protocol("vercel", "https://ai-gateway.vercel.sh/v1", None, "sk-v")
        .await
        .unwrap();

    app.open_vision_picker_for_key("vercel").await;
    match &app.overlay {
        Overlay::Picker(picker) => match &picker.kind {
            PickerKind::Model {
                target: ModelSelectionTarget::VisionDescriber(key),
                ..
            } => assert_eq!(key.key.as_str(), "sk-v", "decrypted, straight to models"),
            _ => panic!("expected a vision-describer model picker"),
        },
        other => panic!("expected model picker, got {}", overlay_name(other)),
    }

    app.overlay = Overlay::None;
    app.open_vision_picker_for_key("nope").await;
    assert!(matches!(app.overlay, Overlay::None), "no picker on miss");
    assert!(
        notice_text(&app).contains("no key named"),
        "{}",
        notice_text(&app)
    );
}

#[tokio::test]
async fn single_key_skips_to_model_picker() {
    let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
    let mut app = make_test_app(tx, rx);
    app.session_store
        .add_key_with_protocol("or", "https://openrouter.ai/api/v1", None, "sk-x")
        .await
        .unwrap();

    app.open_vision_key_picker().await;
    match &app.overlay {
        Overlay::Picker(picker) => match &picker.kind {
            PickerKind::Model {
                target: ModelSelectionTarget::VisionDescriber(key),
                ..
            } => assert_eq!(key.key.as_str(), "sk-x", "decrypted, straight to models"),
            _ => panic!("expected a vision-describer model picker"),
        },
        other => panic!("expected model picker, got {}", overlay_name(other)),
    }
}

fn overlay_name(overlay: &Overlay) -> &'static str {
    match overlay {
        Overlay::None => "none",
        Overlay::Picker(_) => "picker",
        Overlay::Config(_) => "config",
        _ => "other",
    }
}

#[test]
fn seed_turns_carry_descriptions_or_placeholders() {
    use super::super::runtime_impl::agent_seed_turns;
    let history = vec![
        ChatMessage {
            model: None,
            role: "user".to_string(),
            content: "what's in this".to_string(),
            reasoning_content: None,
            attachments: vec![image_attachment()],
        },
        ChatMessage {
            model: None,
            role: "user".to_string(),
            content: "and this".to_string(),
            reasoning_content: None,
            attachments: vec![MessageAttachment {
                name: "other.png".to_string(),
                mime_type: "image/png".to_string(),
                storage: AttachmentStorage::Inline {
                    data: "b3RoZXI=".to_string(),
                },
            }],
        },
    ];
    let mut cache = std::collections::HashMap::new();
    cache.insert(
        crate::services::vision_describe::image_hash("iVBOR"),
        "[Image] a red button".to_string(),
    );
    let seed = agent_seed_turns(&history, &cache);
    assert!(
        seed[0].1.contains("a red button"),
        "described: {}",
        seed[0].1
    );
    assert!(
        seed[1].1.contains("[image attached: other.png"),
        "placeholder: {}",
        seed[1].1
    );
}

#[tokio::test]
async fn describe_failed_restores_composer_and_resets_turn() {
    let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
    let mut app = make_test_app(tx.clone(), rx);
    app.history.push(ChatMessage {
        model: None,
        role: "user".to_string(),
        content: "what's in this".to_string(),
        reasoning_content: None,
        attachments: vec![image_attachment()],
    });
    app.agent_turn_indices.insert(0);
    app.pending_submit = Some(PendingSubmission {
        content: "what's in this".to_string(),
        attachments: vec![image_attachment()],
    });
    app.sending = true;
    app.request_started_at = Some(std::time::Instant::now());

    tx.send(RuntimeEvent::DescribeFailed {
        message: "image describe is temporarily down".to_string(),
    })
    .unwrap();
    app.handle_runtime_events().await.unwrap();

    assert!(!app.sending, "turn reset");
    assert!(app.request_started_at.is_none());
    assert!(app.history.is_empty(), "user row popped");
    assert!(app.agent_turn_indices.is_empty(), "dispatch flag removed");
    assert_eq!(app.draft, "what's in this", "draft text restored");
    assert_eq!(app.draft_attachments.len(), 1, "attachment restored");
    assert!(notice_text(&app).contains("temporarily down"));
}
