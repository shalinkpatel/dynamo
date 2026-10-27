use super::*;

#[test]
fn summary_lists_kinds_in_declaration_order_and_skips_zeroes() {
    let mut losses = Losses::default();
    losses.record(LossKind::RequestFieldDropped, "cache_control", "dropped");
    losses.record(LossKind::ServerToolDropped, "tools[0]", "dropped");
    losses.record(LossKind::RequestFieldDropped, "service_tier", "dropped");
    assert_eq!(
        losses.summary(),
        "server_tool_dropped=1,request_field_dropped=2"
    );
    assert_eq!(
        count_by_kind(&losses.into_vec()),
        vec![
            (LossKind::ServerToolDropped, 1),
            (LossKind::RequestFieldDropped, 2)
        ]
    );
}

#[test]
fn labels_are_unique_snake_case_and_cover_every_kind() {
    let labels: std::collections::HashSet<_> = LossKind::VARIANTS
        .iter()
        .map(|kind| kind.as_label())
        .collect();
    assert_eq!(labels.len(), LossKind::VARIANTS.len());
    for label in labels {
        assert!(
            label.chars().all(|c| c.is_ascii_lowercase() || c == '_'),
            "{label}"
        );
    }
}

/// The metric label and the serialized `kind` come from two derives; they must stay one name.
#[test]
fn label_matches_the_serialized_kind() {
    for kind in LossKind::VARIANTS {
        assert_eq!(serde_json::to_value(kind).unwrap(), kind.as_label());
    }
}

#[test]
fn record_bounds_client_supplied_text() {
    let mut losses = Losses::default();
    losses.record(
        LossKind::RequestFieldDropped,
        "k".repeat(500),
        "d".repeat(500),
    );
    let loss = &losses.into_vec()[0];
    assert!(loss.field.starts_with(&"k".repeat(LOSS_TEXT_LIMIT)));
    assert!(loss.field.contains("truncated"), "{}", loss.field);
    assert!(loss.detail.contains("truncated"), "{}", loss.detail);
}

#[test]
fn every_kind_declares_an_outcome() {
    for kind in LossKind::VARIANTS {
        assert!(!kind.outcome().is_empty(), "{kind}");
    }
}
