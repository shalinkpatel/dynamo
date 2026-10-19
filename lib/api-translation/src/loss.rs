//! What adaptation did not carry through. Every place ingress drops, skips, degrades, or folds a
//! part of the client's request records a [`Loss`] instead of (or as well as) logging it, so a
//! consumer can answer "what did this request lose becoming Chat Completions?" from data: the
//! record rides [`crate::request::AdaptedRequest::losses`], each one is handed to
//! [`crate::hooks::IngressHooks::on_loss`], and the ingress stage line carries the per-kind count.
//!
//! `kind` is a closed enum whose label is a `&'static str`, so it is safe as a metric label.
//! `field` is a JSON path within the request (bounded, client-controlled key names only) and
//! `detail` is a bounded human sentence; neither carries message content.

use std::fmt;

use strum::VariantArray;

/// Loss categories. Adding a variant is additive for consumers that count by kind.
#[derive(
    Debug,
    Clone,
    Copy,
    PartialEq,
    Eq,
    Hash,
    serde::Serialize,
    strum::IntoStaticStr,
    strum::VariantArray,
)]
#[serde(rename_all = "snake_case")]
#[strum(serialize_all = "snake_case")]
pub enum LossKind {
    /// A server-tool-shaped `tools[]` entry was dropped: nothing on this endpoint executes it.
    ServerToolDropped,
    /// A `tool_choice` demanding a tool that was dropped or is unsupported degraded to `auto`.
    ToolChoiceDegraded,
    /// A request field with no Chat Completions meaning was dropped.
    RequestFieldDropped,
    /// A `baseten` extension member this crate does not model was dropped.
    ExtensionFieldDropped,
    /// A message content block with no Chat Completions translation was skipped.
    ContentBlockSkipped,
    /// An oversized unknown `tool_result` block was replaced by a placeholder.
    ToolResultBlockOmitted,
    /// A replayed Responses input item with no translation was skipped.
    InputItemSkipped,
    /// A replayed Responses `compaction` item was ignored.
    ReplayItemIgnored,
    /// A replayed `agent_message` had no plaintext content and was dropped.
    AgentMessageDropped,
    /// A Responses `include` this stack cannot produce was ignored.
    IncludeIgnored,
    /// A duplicate tool declaration was dropped (first declaration kept).
    DuplicateToolDropped,
    /// A replayed function_call/mcp_call with non-JSON `arguments` was carried verbatim, not parsed.
    NonJsonEchoedArguments,
}

impl LossKind {
    /// snake_case label, `&'static` so it can be a metric label.
    pub fn as_label(self) -> &'static str {
        self.into()
    }
}

impl fmt::Display for LossKind {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_label())
    }
}

/// One thing adaptation did not carry through.
#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize)]
pub struct Loss {
    pub kind: LossKind,
    /// JSON path of the affected member within the client's request (`tools[2]`, `cache_control`,
    /// `messages[3].content[1]`, `input[4]`). Bounded; key names only.
    pub field: String,
    /// What happened to it, in one bounded sentence. Never message content.
    pub detail: String,
}

/// Ceiling on `field`/`detail`: both may embed client-supplied key names.
const LOSS_TEXT_LIMIT: usize = 160;

/// The losses recorded while adapting one request.
#[derive(Debug, Default)]
pub struct Losses {
    entries: Vec<Loss>,
}

impl Losses {
    /// Record one loss. Also logs it at WARN under `event_name = "ingress.loss"`, so a plain
    /// deployment with no counter still has a trace.
    pub(crate) fn record(
        &mut self,
        kind: LossKind,
        field: impl Into<String>,
        detail: impl Into<String>,
    ) {
        let field = crate::util::truncate(&field.into(), LOSS_TEXT_LIMIT);
        let detail = crate::util::truncate(&detail.into(), LOSS_TEXT_LIMIT);
        tracing::warn!(
            event_name = "ingress.loss",
            kind = kind.as_label(),
            field = %field,
            "{detail}"
        );
        self.entries.push(Loss {
            kind,
            field,
            detail,
        });
    }

    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    pub fn len(&self) -> usize {
        self.entries.len()
    }

    pub fn iter(&self) -> impl Iterator<Item = &Loss> {
        self.entries.iter()
    }

    pub fn into_vec(self) -> Vec<Loss> {
        self.entries
    }

    /// `kind=n` pairs in [`LossKind::VARIANTS`] order, kinds with no occurrences omitted; the shape the
    /// stage line carries.
    pub fn summary(&self) -> String {
        LossKind::VARIANTS
            .iter()
            .filter_map(|kind| {
                let n = self
                    .entries
                    .iter()
                    .filter(|loss| loss.kind == *kind)
                    .count();
                (n > 0).then(|| format!("{}={n}", kind.as_label()))
            })
            .collect::<Vec<_>>()
            .join(",")
    }
}

/// Per-kind occurrence counts over a recorded list, for consumers that hold the `Vec<Loss>`.
pub fn count_by_kind(losses: &[Loss]) -> Vec<(LossKind, usize)> {
    LossKind::VARIANTS
        .iter()
        .filter_map(|kind| {
            let n = losses.iter().filter(|loss| loss.kind == *kind).count();
            (n > 0).then_some((*kind, n))
        })
        .collect()
}

#[cfg(test)]
#[path = "loss_test.rs"]
mod tests;
