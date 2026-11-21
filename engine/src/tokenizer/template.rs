use super::{PromptMessage, Thinking, ToolSpec, pyjson, pystr, toolcall::ToolGrammar};
use crate::EngError;
use minijinja::{Error, ErrorKind, Value};

/// Prefix marking a mid-list system message demoted to a user turn. The
/// marker is informational for the model, not a trust boundary: ordinary
/// user text can carry the same bytes.
const MID_SYSTEM_MARKER: &str = "[system] ";

/// Escapes user/tool ChatML delimiters for Qwen/Nemotron; Flint keeps its pinned template.
pub struct ChatTemplate {
    env: minijinja::Environment<'static>,
    pub(super) grammar: ToolGrammar,
    chatml_delimiters: bool,
    /// The probe decided at load that demoting mid-list system messages
    /// to marked user turns in place fixes this template.
    pub(super) demote_mid_system: bool,
    /// `ENGINESRV_THINK_REPLAY`: resent assistant reasoning reaches the
    /// template (the strip below stays off) and the template's own keep/drop
    /// rule decides which turns replay it. On by default ONLY for the
    /// pinned Qwen3.6 template bytes (the one template with an A/B);
    /// `ENGINESRV_THINK_REPLAY=0` rolls back to the pre-feature bytes.
    pub(super) think_replay: bool,
    /// `ENGINESRV_NOTICE_TOOLWRAP`: demoted notices that follow an assistant
    /// row render in the template's tool-response shape so its
    /// last_query_index scan keeps the tool loop's think blocks. Requires
    /// think replay. On by default with it; `=0` rolls back to plain
    /// demoted notices.
    pub(super) notice_toolwrap: bool,
}

/// The instructions one render of a chat template may execute. A real template needs a few
/// thousand per message; the bound ends a template that loops over huge ranges, which a GGUF
/// header can carry from any publisher.
const RENDER_FUEL: u64 = 20_000_000;

impl ChatTemplate {
    pub(crate) fn from_str(source: &str) -> Result<Self, EngError> {
        Self::build(source, true)
    }

    /// The same template without the load line, for a probe that renders it once.
    pub(crate) fn probe(source: &str) -> Result<Self, EngError> {
        Self::build(source, false)
    }

    fn build(source: &str, announce: bool) -> Result<Self, EngError> {
        let mut env = minijinja::Environment::new();
        env.set_fuel(Some(RENDER_FUEL));
        // The reference renders with trim_blocks and lstrip_blocks enabled.
        env.set_trim_blocks(true);
        env.set_lstrip_blocks(true);
        env.add_global("raise_exception", Value::from_function(raise_exception));
        env.add_filter("tojson", pyjson::tojson_filter);
        env.add_filter("trim", pystr::trim_filter);
        env.set_unknown_method_callback(unknown_method);
        // The engine loads one template per process. The owned source keeps the
        // environment free of lifetimes without leaking the bytes.
        let grammar = ToolGrammar::for_template(source);
        let normalized = normalize_newlines(source);
        let template = if grammar == ToolGrammar::Inkling {
            inkling_template(&normalized)
        } else {
            without_generation_tags(&normalized)
        };
        env.add_template_owned("chat", template).map_err(|error| {
            EngError::Checkpoint(format!("chat_template.jinja syntax error: {error}"))
        })?;
        let demote_mid_system = demotion_fixes_mid_system(&env);
        // Only the pinned Qwen3.6 template is eligible for the replay
        // default: it is the one template with an A/B. Other templates that
        // also pass the demotion probe (Flash-Next does) carry different
        // keep/drop rules — e.g. `preserve_thinking` undefined replays ALL
        // resent reasoning — and stay off until each has its own evidence.
        let replay_eligible = Self::replay_eligible(source);
        let replay_env_set = std::env::var_os("ENGINESRV_THINK_REPLAY").is_some();
        let (think_replay, notice_toolwrap) = Self::resolve_replay_flags(
            demote_mid_system && replay_eligible,
            Self::env_allows("ENGINESRV_THINK_REPLAY"),
            Self::env_allows("ENGINESRV_NOTICE_TOOLWRAP"),
        );
        if announce && (think_replay || replay_env_set) && !cfg!(test) {
            // The one load line an operator needs to confirm a rollback.
            eprintln!(
                "chat template: think replay {}, notice wrap {}{}",
                on_off(think_replay),
                on_off(notice_toolwrap),
                if replay_eligible {
                    ""
                } else {
                    " (template not replay-eligible)"
                },
            );
        }
        Ok(Self {
            env,
            grammar,
            chatml_delimiters: source.contains("<|im_start|>") && source.contains("<|im_end|>"),
            demote_mid_system,
            think_replay,
            notice_toolwrap,
        })
    }

    /// The replay default's eligibility: byte-exact equality with the
    /// pinned Qwen3.6 template, the one template with an A/B. Every other
    /// template — Flint, Flash-Next, synthetic — stays off until it carries
    /// its own evidence.
    pub(super) fn replay_eligible(source: &str) -> bool {
        source == include_str!("../../testdata/chat_template.jinja")
    }

    /// Pure resolution of the replay flags so the coupling is testable
    /// without touching process-global environment: replay needs an
    /// ELIGIBLE template (the caller passes demotion AND the pinned-template
    /// eligibility check), and the notice wrap rides on replay and never
    /// enables it alone. Probe-passing but ineligible templates (Flash-Next
    /// among them) stay off until each carries its own evidence.
    pub(super) fn resolve_replay_flags(
        demote_mid_system: bool,
        think_env: bool,
        wrap_env: bool,
    ) -> (bool, bool) {
        let think_replay = demote_mid_system && think_env;
        (think_replay, think_replay && wrap_env)
    }

    /// Test-only: forces both replay flags off so server-side pins do not
    /// depend on the ambient environment (the fields are private to the
    /// tokenizer tree).
    #[cfg(test)]
    pub fn force_flags_off(&mut self) {
        self.think_replay = false;
        self.notice_toolwrap = false;
    }

    /// Test-only: forces both replay flags on so config-override pins do
    /// not depend on the ambient environment or the template's own
    /// default resolution.
    #[cfg(test)]
    pub fn force_flags_on(&mut self) {
        self.think_replay = true;
        self.notice_toolwrap = true;
    }

    fn env_allows(name: &str) -> bool {
        crate::env_switch::env_allows_default_on(name)
    }

    pub fn render(
        &self,
        msgs: &[PromptMessage],
        tools: &[ToolSpec],
        thinking: Thinking,
    ) -> Result<String, EngError> {
        self.render_with(msgs, tools, thinking, true)
    }

    /// For a template that appends its generation prompt, this is the text up
    /// to the last message boundary.
    pub fn render_without_generation_prompt(
        &self,
        msgs: &[PromptMessage],
        tools: &[ToolSpec],
        thinking: Thinking,
    ) -> Result<String, EngError> {
        self.render_with(msgs, tools, thinking, false)
    }

    /// Renders without a generation prompt, with each tool call also given in the OpenAI shape
    /// (`function` holding `name` and `arguments`) that some templates read. For probing a
    /// template's own markers; serving renders through [`Self::render`].
    pub(crate) fn render_openai_calls(
        &self,
        msgs: &[PromptMessage],
        tools: &[ToolSpec],
        thinking: Thinking,
    ) -> Result<String, EngError> {
        self.render_inner(msgs, tools, thinking, false, true)
    }

    fn render_with(
        &self,
        msgs: &[PromptMessage],
        tools: &[ToolSpec],
        thinking: Thinking,
        generation_prompt: bool,
    ) -> Result<String, EngError> {
        self.render_inner(msgs, tools, thinking, generation_prompt, false)
    }

    fn render_inner(
        &self,
        msgs: &[PromptMessage],
        tools: &[ToolSpec],
        thinking: Thinking,
        generation_prompt: bool,
        openai_calls: bool,
    ) -> Result<String, EngError> {
        if msgs.is_empty() {
            return Err(EngError::BadRequest("message list is empty".to_string()));
        }
        let context = build_context(msgs, tools, thinking, generation_prompt, openai_calls, self)?;
        let template = self.env.get_template("chat").map_err(render_defect)?;
        template.render(context).map_err(render_error)
    }
}

/// Training-oriented chat templates mark the assistant span with `{% generation %}` and
/// `{% endgeneration %}`, a tag outside the base template language that renders nothing. The tags become an
/// always-true `if` block so the dash and whitespace controls of the tag keep their effect.
fn without_generation_tags(source: &str) -> String {
    let mut out = String::with_capacity(source.len());
    let mut rest = source;
    while let Some(open) = rest.find("{%") {
        let (before, tag_and_after) = rest.split_at(open);
        out.push_str(before);
        let Some(close) = tag_and_after.find("%}") else {
            out.push_str(tag_and_after);
            return out;
        };
        let tag = &tag_and_after[..close + 2];
        let inner = &tag[2..tag.len() - 2];
        let name = inner.trim_matches(|c: char| c == '-' || c == '+' || c.is_whitespace());
        let replacement = match name {
            "generation" => Some(("generation", "if true")),
            "endgeneration" => Some(("endgeneration", "endif")),
            _ => None,
        };
        match replacement {
            Some((word, with)) => out.push_str(&tag.replacen(word, with, 1)),
            None => out.push_str(tag),
        }
        rest = &tag_and_after[close + 2..];
    }
    out.push_str(rest);
    out
}

// Jinja turns every newline sequence in template text into "\n".
fn normalize_newlines(source: &str) -> String {
    source.replace("\r\n", "\n").replace('\r', "\n")
}

// Hugging Face's generation blocks annotate assistant spans for training. MiniJinja
// only needs the rendered text. Substitute a true block with the same whitespace controls.
fn inkling_template(source: &str) -> String {
    source
        .replace("{%- generation -%}", "{%- if true -%}")
        .replace("{%- endgeneration -%}", "{%- endif -%}")
}

/// Two probe renders at load decide whether demoting a mid-list system
/// message to a marked user turn fixes this template: [user, system, user]
/// must be rejected by the template itself, and the demoted candidate
/// [user, user, user] — with the marked row in place — must render
/// successfully. Demotion is enabled only when the rewrite makes the
/// rejected shape render, so templates rejecting the probe for an
/// unrelated reason (a required leading system message, user/assistant
/// alternation that skips system rows) reject the demoted candidate too
/// and stay untouched. Known limit: a template that requires a leading
/// system message and also rejects mid-list ones stays off — no pinned
/// template does both.
fn demotion_fixes_mid_system(env: &minijinja::Environment<'static>) -> bool {
    let demoted = format!("{MID_SYSTEM_MARKER}b");
    probe_raises(env, &[("user", "a"), ("system", "b"), ("user", "c")])
        && probe_renders(env, &[("user", "a"), ("user", &demoted), ("user", "c")])
}

fn probe_raises(env: &minijinja::Environment<'static>, rows: &[(&str, &str)]) -> bool {
    matches!(probe_render(env, rows), Err(error) if is_template_raise(&error))
}

fn probe_renders(env: &minijinja::Environment<'static>, rows: &[(&str, &str)]) -> bool {
    probe_render(env, rows).is_ok()
}

fn probe_render(env: &minijinja::Environment<'static>, rows: &[(&str, &str)]) -> Result<(), Error> {
    let messages: Vec<_> = rows
        .iter()
        .map(|(role, content)| {
            serde_json::json!({
                "role": role,
                "content": content,
                "name": null,
                "tool_call_id": null,
                "tool_calls": [],
            })
        })
        .collect();
    let context = serde_json::json!({
        "messages": messages,
        "tools": [],
        "add_generation_prompt": false,
    });
    let template = env.get_template("chat")?;
    template.render(Value::from_serialize(context)).map(|_| ())
}

// Marks an error that `raise_exception` produced. The template author chose to
// reject this request, so the cause is the request. Every other render error is
// a template or engine defect.
#[derive(Debug)]
struct TemplateRaised;

impl std::fmt::Display for TemplateRaised {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter.write_str("raised by raise_exception")
    }
}

impl std::error::Error for TemplateRaised {}

fn raise_exception(message: String) -> Result<Value, Error> {
    Err(Error::new(ErrorKind::InvalidOperation, message).with_source(TemplateRaised))
}

fn is_template_raise(error: &Error) -> bool {
    let mut source = std::error::Error::source(error);
    while let Some(cause) = source {
        if cause.is::<TemplateRaised>() {
            return true;
        }
        source = cause.source();
    }
    false
}

fn unknown_method(
    state: &minijinja::State,
    value: &Value,
    method: &str,
    args: &[Value],
) -> Result<Value, Error> {
    if value.kind() == minijinja::value::ValueKind::Map && method == "items" {
        let _: () = minijinja::value::from_args(args)?;
        return state.apply_filter("items", std::slice::from_ref(value));
    }
    if value.kind() == minijinja::value::ValueKind::Map && method == "get" {
        // Python dict.get shape, but a missing key with no default yields
        // undefined, not Python's None, on purpose: the Bolt template's falsy
        // chains treat undefined and None alike, and `is undefined` stays
        // distinguishable from `is none`.
        let (key, default): (Value, Option<Value>) = minijinja::value::from_args(args)?;
        let found = value.get_item(&key)?;
        if found.is_undefined() {
            return Ok(default.unwrap_or(Value::UNDEFINED));
        }
        return Ok(found);
    }
    let Some(text) = value.as_str() else {
        return Err(unknown_name(method));
    };
    pystr::call(text, method, args).unwrap_or_else(|| Err(unknown_name(method)))
}

fn on_off(flag: bool) -> &'static str {
    if flag { "on" } else { "off" }
}

fn unknown_name(method: &str) -> Error {
    Error::new(
        ErrorKind::UnknownMethod,
        format!("unknown template method {method}"),
    )
}

fn build_context(
    msgs: &[PromptMessage],
    tools: &[ToolSpec],
    thinking: Thinking,
    generation_prompt: bool,
    openai_calls: bool,
    template: &ChatTemplate,
) -> Result<Value, EngError> {
    let mut context = serde_json::Map::new();
    let mut messages = message_context(msgs, openai_calls)?;
    let rows = messages
        .as_array_mut()
        .expect("message_context returns an array");
    if template.grammar == ToolGrammar::Inkling {
        if !tools.is_empty()
            || msgs
                .iter()
                .any(|msg| !msg.tool_calls.is_empty() || matches!(msg.role, super::Role::Tool))
        {
            return Err(EngError::BadRequest(
                "Inkling 2 Mini experimental serving accepts text chat without tools".into(),
            ));
        }
        for row in rows.iter_mut() {
            if row["role"] == "developer" {
                row["role"] = serde_json::json!("system");
            }
        }
        match thinking {
            Thinking::Default => {}
            Thinking::On => {
                context.insert("reasoning_effort".into(), serde_json::json!("high"));
            }
            Thinking::Off => {
                context.insert("reasoning_effort".into(), serde_json::json!("none"));
            }
        }
    }
    if template.grammar == ToolGrammar::Qwen {
        // Legacy clients have always rendered developer messages as system messages and
        // ignored reasoning replay. Keep that contract unless think replay is enabled:
        // with replay the template's own keep/drop rule decides which assistant turns
        // show their reasoning, and stripping here would hide the in-loop replay.
        for row in rows.iter_mut() {
            if row["role"] == "developer" {
                row["role"] = serde_json::json!("system");
            }
            if !template.think_replay {
                row.as_object_mut()
                    .expect("message row")
                    .remove("reasoning_content");
            }
        }
    }
    if template.demote_mid_system {
        // Some templates (the real Qwen3.6 one among them) reject a system
        // message that is not first — and Claude Code injects its environment
        // and token-budget notices exactly there. Demote in place to a marked
        // user turn. The rendered prefix stays byte-identical across turns
        // while the demoted rows stay in history with unchanged content. The
        // one break: appending a demoted row to a request that left a tool
        // loop open (assistant rows after its last user-role row) moves the
        // Qwen template's last_query_index onto the new row, and those
        // assistants re-render without their think wrapper — the same thing
        // a new real user query does there — so the prefix cache falls back
        // to an earlier boundary. Claude Code ends every request with a
        // system row (environment context on turn 1, its token-budget
        // notice after), so no assistant ever sits after its last user-role
        // row and the stable shape holds.
        let mut demoted: Vec<usize> = Vec::new();
        for (index, row) in rows.iter_mut().enumerate() {
            if index > 0 && row["role"] == "system" {
                row["role"] = serde_json::json!("user");
                row["content"] = serde_json::Value::String(match row["content"].as_str() {
                    Some(text) => format!("{MID_SYSTEM_MARKER}{text}"),
                    // A system row without text still renders as a marked,
                    // visibly empty user turn.
                    None => MID_SYSTEM_MARKER.trim_end().to_string(),
                });
                demoted.push(index);
            }
        }
        if template.notice_toolwrap {
            // The pinned template keeps an assistant turn's think block only
            // while the turn sits after its `last_query_index` — the last
            // user row that is not a tool response. Demoted notices are user
            // rows, so un-wrapped they move that index past the tool-loop
            // assistants and the resent turns lose the think block the engine
            // decoded — the mid-session prefix loss the n9 dogfood session
            // showed as full re-prefills. Wrapping a notice that follows an
            // assistant row in the template's own tool-response shape keeps
            // the scan at the real user query. The rule is position-stable:
            // once an assistant row precedes the notice, every later request
            // wraps it the same way, so the rendered prefix still extends
            // byte-for-byte. Rollback: ENGINESRV_NOTICE_TOOLWRAP=0.
            // Wrap only rows the demotion loop actually demoted, never
            // user text that merely begins with the marker: a user string
            // that happens to start with it is untrusted content, not a
            // notice the server produced.
            let mut saw_assistant = false;
            for (index, row) in rows.iter_mut().enumerate() {
                if row["role"] == "assistant" {
                    saw_assistant = true;
                } else if saw_assistant && demoted.contains(&index) {
                    let text = row["content"].as_str().expect("demoted rows carry text");
                    row["content"] = serde_json::Value::String(format!(
                        "<tool_response>\n{text}\n</tool_response>"
                    ));
                }
            }
        }
    }
    if template.grammar == ToolGrammar::Qwen && template.chatml_delimiters {
        for row in rows.iter_mut() {
            if matches!(row["role"].as_str(), Some("user" | "tool"))
                && let Some(text) = row["content"].as_str()
            {
                // Escape untrusted delimiters without touching trusted template headers.
                row["content"] = serde_json::Value::String(
                    text.replace("<|im_start|>", "&lt;|im_start|&gt;")
                        .replace("<|im_end|>", "&lt;|im_end|&gt;")
                        .replace("<|endoftext|>", "&lt;|endoftext|&gt;"),
                );
            }
        }
    }
    context.insert("messages".to_string(), messages);
    context.insert("tools".to_string(), to_value(tools, "tools")?);
    context.insert(
        "add_generation_prompt".to_string(),
        serde_json::json!(generation_prompt),
    );
    if !matches!(thinking, Thinking::Default) {
        context.insert(
            "enable_thinking".to_string(),
            serde_json::json!(matches!(thinking, Thinking::On)),
        );
    }
    Ok(Value::from_serialize(serde_json::Value::Object(context)))
}

fn message_context(
    msgs: &[PromptMessage],
    openai_calls: bool,
) -> Result<serde_json::Value, EngError> {
    let mut messages = to_value(msgs, "messages")?;
    let rows = messages
        .as_array_mut()
        .ok_or_else(|| EngError::Checkpoint("messages template context failed".to_string()))?;
    for row in rows {
        let Some(calls) = row
            .get_mut("tool_calls")
            .and_then(serde_json::Value::as_array_mut)
        else {
            continue;
        };
        for call in calls {
            let text = call
                .get("arguments")
                .and_then(serde_json::Value::as_str)
                .ok_or_else(|| {
                    EngError::BadRequest("tool call arguments must be JSON text".to_string())
                })?;
            let parsed: serde_json::Value = serde_json::from_str(text).map_err(|error| {
                EngError::BadRequest(format!("tool call arguments failed: {error}"))
            })?;
            if !parsed.is_object() {
                return Err(EngError::BadRequest(
                    "tool call arguments must be a JSON object".to_string(),
                ));
            }
            call["arguments"] = parsed;
            if openai_calls {
                call["function"] = serde_json::json!({
                    "name": call["name"].clone(),
                    "arguments": call["arguments"].clone(),
                });
            }
        }
    }
    Ok(messages)
}

fn to_value<T: serde::Serialize + ?Sized>(
    value: &T,
    what: &str,
) -> Result<serde_json::Value, EngError> {
    serde_json::to_value(value)
        .map_err(|error| EngError::Checkpoint(format!("{what} template context failed: {error}")))
}

fn render_defect(error: Error) -> EngError {
    EngError::Checkpoint(format!("chat template failed: {error}"))
}

fn render_error(error: Error) -> EngError {
    if is_template_raise(&error) {
        EngError::BadRequest(format!("chat template rejected the request: {error}"))
    } else {
        render_defect(error)
    }
}

#[cfg(test)]
mod generation_tag_tests {
    use super::without_generation_tags;

    #[test]
    fn generation_tags_become_an_always_true_block_that_keeps_the_dash_controls() {
        assert_eq!(
            without_generation_tags("a {%- generation %} b {%- endgeneration -%} c"),
            "a {%- if true %} b {%- endif -%} c"
        );
        assert_eq!(
            without_generation_tags("{% generation %}x{% endgeneration %}"),
            "{% if true %}x{% endif %}"
        );
    }

    #[test]
    fn other_tags_and_unclosed_tags_pass_through_unchanged() {
        let source =
            "{% if generation %}{{ generation }}{% endif %} {% for i in x %}{% endfor %} {%";
        assert_eq!(without_generation_tags(source), source);
    }

    #[test]
    fn a_template_without_the_tags_keeps_its_bytes() {
        let source = include_str!("../../testdata/chat_template.jinja");
        assert_eq!(without_generation_tags(source), source);
    }
}

#[cfg(test)]
mod limit_tests {
    use super::ChatTemplate;
    use crate::tokenizer::{PromptMessage, Role, Thinking};

    fn user(text: &str) -> PromptMessage {
        PromptMessage {
            role: Role::User,
            content: Some(text.to_owned()),
            reasoning_content: None,
            name: None,
            tool_call_id: None,
            tool_calls: Vec::new(),
        }
    }

    #[test]
    fn a_template_that_loops_over_huge_ranges_stops_with_an_error_instead_of_running() {
        let template = ChatTemplate::from_str(
            "{% for i in range(100000) %}{% for j in range(100000) %}x{% endfor %}{% endfor %}",
        )
        .expect("the template parses");
        let started = std::time::Instant::now();
        let error = template
            .render(&[user("hi")], &[], Thinking::Default)
            .expect_err("the render runs out of fuel");
        assert!(started.elapsed().as_secs() < 30, "{:?}", started.elapsed());
        assert!(
            error.describe().to_lowercase().contains("fuel"),
            "{}",
            error.describe()
        );
    }

    #[test]
    fn a_long_conversation_renders_inside_the_fuel() {
        let template = ChatTemplate::from_str(
            "{% for m in messages %}<|{{ m.role }}|>{{ m.content }}\n{% endfor %}",
        )
        .expect("the template parses");
        let messages: Vec<PromptMessage> = (0..5000).map(|i| user(&format!("turn {i}"))).collect();
        let text = template
            .render(&messages, &[], Thinking::Default)
            .expect("renders");
        assert!(text.contains("turn 4999"));
    }
}

#[cfg(test)]
mod inkling_tests {
    use super::*;

    #[test]
    fn inkling_generation_annotations_keep_their_whitespace_controls() {
        let source = "a \n{%- generation -%}\n {{ value }} \n{%- endgeneration -%}\n b";
        let mut env = minijinja::Environment::new();
        assert!(env.add_template("original", source).is_err());
        let rewritten = inkling_template(source);
        env.add_template("rewritten", &rewritten).unwrap();
        let text = env
            .get_template("rewritten")
            .unwrap()
            .render(minijinja::context!(value => "answer"))
            .unwrap();
        assert_eq!(text, "aanswerb");
    }
}
