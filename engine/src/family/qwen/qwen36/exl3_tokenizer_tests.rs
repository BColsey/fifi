//! The pack's `tokenizer.json` against the pinned one the engine serves with.
//!
//! The engine encodes with the pinned tokenizer and a pack embeds the ids it was trained with, so
//! the two must give every id the same meaning. The files differ in bytes: the pack writes each
//! merge as one string, the pinned file as a pair, and the pack lacks seven audio special tokens
//! the text engine never emits. This compares what matters: the vocabulary, the merge list, the
//! added tokens the pack has, and the pipeline stages.

use std::path::Path;

use serde_json::Value;

fn merge_text(merge: &Value) -> String {
    match merge {
        Value::String(text) => text.clone(),
        Value::Array(pair) => pair
            .iter()
            .map(|part| part.as_str().unwrap_or_default())
            .collect::<Vec<_>>()
            .join(" "),
        other => other.to_string(),
    }
}

/// Why `pack` would give an id another meaning than `pinned`, or `None` when it would not.
fn disagreement(pinned: &Value, pack: &Value) -> Option<String> {
    for stage in ["normalizer", "pre_tokenizer", "post_processor", "decoder"] {
        if pinned[stage] != pack[stage] {
            return Some(format!("the {stage} differs"));
        }
    }
    if pinned["model"]["vocab"] != pack["model"]["vocab"] {
        return Some("the vocabulary differs".into());
    }
    let merges = |file: &Value| -> Vec<String> {
        file["model"]["merges"]
            .as_array()
            .map(|list| list.iter().map(merge_text).collect())
            .unwrap_or_default()
    };
    if merges(pinned) != merges(pack) {
        return Some("the merge list differs".into());
    }
    let added = |file: &Value| -> Vec<(u64, String, bool)> {
        file["added_tokens"]
            .as_array()
            .map(|list| {
                list.iter()
                    .map(|t| {
                        (
                            t["id"].as_u64().unwrap_or(u64::MAX),
                            t["content"].as_str().unwrap_or_default().to_owned(),
                            t["special"].as_bool().unwrap_or(false),
                        )
                    })
                    .collect()
            })
            .unwrap_or_default()
    };
    let known = added(pinned);
    added(pack)
        .into_iter()
        .find(|token| !known.contains(token))
        .map(|token| format!("the added token {token:?} is not in the pinned file"))
}

fn sample(merges: Value, extra: Option<Value>) -> Value {
    let mut tokens = vec![serde_json::json!({"id": 3, "content": "<|a|>", "special": true})];
    tokens.extend(extra);
    serde_json::json!({
        "normalizer": null, "pre_tokenizer": {"type": "ByteLevel"},
        "post_processor": null, "decoder": {"type": "ByteLevel"},
        "model": {"vocab": {"a": 0, "b": 1, "ab": 2}, "merges": merges},
        "added_tokens": tokens,
    })
}

#[test]
fn merges_as_strings_or_pairs_and_fewer_added_tokens_agree() {
    let extra = serde_json::json!({"id": 4, "content": "<|b|>", "special": true});
    let pinned = sample(serde_json::json!([["a", "b"]]), Some(extra));
    let pack = sample(serde_json::json!(["a b"]), None);
    assert_eq!(disagreement(&pinned, &pack), None);
}

#[test]
fn every_change_that_moves_an_id_is_reported() {
    let pinned = sample(serde_json::json!([["a", "b"]]), None);
    let mut cases: Vec<(&str, Value)> = Vec::new();
    cases.push(("merge", sample(serde_json::json!(["b a"]), None)));
    let mut vocab = sample(serde_json::json!(["a b"]), None);
    vocab["model"]["vocab"]["ab"] = serde_json::json!(5);
    cases.push(("vocabulary", vocab));
    let extra = serde_json::json!({"id": 9, "content": "<|new|>", "special": true});
    cases.push((
        "added token",
        sample(serde_json::json!(["a b"]), Some(extra)),
    ));
    let moved = serde_json::json!({"id": 3, "content": "<|a|>", "special": false});
    let mut flag = sample(serde_json::json!(["a b"]), None);
    flag["added_tokens"] = serde_json::json!([moved]);
    cases.push(("special flag", flag));
    let mut stage = sample(serde_json::json!(["a b"]), None);
    stage["pre_tokenizer"] = serde_json::json!({"type": "Whitespace"});
    cases.push(("pipeline", stage));
    for (what, pack) in cases {
        assert!(disagreement(&pinned, &pack).is_some(), "{what}");
    }
}

/// `FIFI_EXL3_CHECKPOINT` is a pack directory and `FIFI_TOKENIZER_DIR` the pinned tokenizer one.
#[test]
#[ignore = "reads the tokenizer.json of a real pack and of the pinned directory"]
fn the_real_pack_tokenizer_agrees_with_the_pinned_one() {
    let read = |dir: &str| -> Value {
        let path = Path::new(&std::env::var(dir).unwrap_or_else(|_| panic!("{dir}")))
            .join("tokenizer.json");
        serde_json::from_slice(&std::fs::read(path).expect("read tokenizer.json")).unwrap()
    };
    let (pinned, pack) = (read("FIFI_TOKENIZER_DIR"), read("FIFI_EXL3_CHECKPOINT"));
    assert_eq!(disagreement(&pinned, &pack), None);
}
