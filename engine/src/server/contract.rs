//! Test helper: checks server JSON against `api/openapi.yaml`.
//!
//! The crate has no YAML parser, so the helper reads `testdata/openapi.json`.
//! The OpenAPI fixture script writes that file from the contract and records the hash of the YAML bytes.

use jsonschema::Validator;
use serde_json::{Value, json};
use std::sync::OnceLock;

const FIXTURE: &str = include_str!("testdata/openapi.json");

fn fixture() -> &'static Value {
    static PARSED: OnceLock<Value> = OnceLock::new();
    PARSED.get_or_init(|| serde_json::from_str(FIXTURE).expect("parse openapi.json"))
}

fn validator(pointer: &str) -> Validator {
    let document = fixture();
    let schema = json!({
        "$ref": format!("#{pointer}"),
        "components": document["components"],
        "paths": document["paths"],
    });
    jsonschema::validator_for(&schema).expect("compile contract schema")
}

fn errors(pointer: &str, instance: &Value) -> Vec<String> {
    validator(pointer)
        .iter_errors(instance)
        .map(|error| error.to_string())
        .collect()
}

pub(super) fn assert_schema(name: &str, instance: &Value) {
    let found = errors(&format!("/components/schemas/{name}"), instance);
    assert!(
        found.is_empty(),
        "{name} violations: {found:?} in {instance}"
    );
}

pub(super) fn assert_pointer(pointer: &str, instance: &Value) {
    let found = errors(pointer, instance);
    assert!(
        found.is_empty(),
        "{pointer} violations: {found:?} in {instance}"
    );
}

pub(super) fn matches_schema(name: &str, instance: &Value) -> bool {
    errors(&format!("/components/schemas/{name}"), instance).is_empty()
}

#[cfg(test)]
mod tests {
    use super::{FIXTURE, assert_pointer, assert_schema, matches_schema};
    use serde_json::{Value, json};
    use sha2::{Digest, Sha256};

    #[test]
    fn the_fixture_hash_matches_the_yaml_contract() {
        let yaml = std::fs::read(concat!(env!("CARGO_MANIFEST_DIR"), "/../api/openapi.yaml"))
            .expect("read api/openapi.yaml");
        let expected = format!("{:x}", Sha256::digest(&yaml));
        let fixture: Value = serde_json::from_str(FIXTURE).expect("parse openapi.json");
        assert_eq!(
            fixture["x-source-sha256"], expected,
            "stale fixture: regenerate it from api/openapi.yaml with the OpenAPI fixture script"
        );
    }

    /// The (path, method) pairs router.rs registers, scanned from every
    /// `.route("<path>", <method>(...))` call — including chained methods
    /// like `get(a).post(b)`, which axum requires for a path answering
    /// more than one method. Whitespace and line breaks inside the call
    /// are tolerated, and comment lines are dropped so a documented route
    /// is never scanned.
    fn served_routes(source: &str) -> Vec<(String, String)> {
        let code: String = source
            .lines()
            .filter(|line| !line.trim_start().starts_with("//"))
            .collect::<Vec<_>>()
            .join("\n");
        let mut served = Vec::new();
        let mut rest = code.as_str();
        while let Some(at) = rest.find(".route(") {
            rest = &rest[at + ".route(".len()..];
            let after_open = rest.trim_start();
            let Some(after_path) = after_open.strip_prefix('"') else {
                continue;
            };
            let Some(path_end) = after_path.find('"') else {
                continue;
            };
            let path = after_path[..path_end].to_string();
            let Some(comma) = after_path[path_end + 1..].find(',') else {
                continue;
            };
            let after_comma = after_path[path_end + 1 + comma + 1..].trim_start();
            let method_len = after_comma
                .find(|c: char| !c.is_ascii_alphanumeric() && c != '_')
                .unwrap_or(after_comma.len());
            served.push((path, after_comma[..method_len].to_string()));
            rest = &after_comma[method_len..];
            // A chained method follows as `).post(`; this file's route
            // arguments are path-qualified identifiers, so the first `)`
            // after each argument closes that method's call.
            let mut chain = rest.trim_start();
            while chain.starts_with('(')
                && let Some(close) = chain.find(')')
            {
                chain = chain[close + 1..].trim_start();
                if !chain.starts_with('.') {
                    break;
                }
                chain = &chain[1..];
                let name_len = chain
                    .find(|c: char| !c.is_ascii_alphanumeric() && c != '_')
                    .unwrap_or(chain.len());
                let method = &chain[..name_len];
                if matches!(method, "get" | "post" | "delete" | "put" | "patch" | "head") {
                    served.push((after_path[..path_end].to_string(), method.to_string()));
                }
                chain = &chain[name_len..];
            }
        }
        served
    }

    #[test]
    fn every_router_route_is_in_the_contract() {
        let source = std::fs::read(concat!(env!("CARGO_MANIFEST_DIR"), "/src/server/router.rs"))
            .expect("read src/server/router.rs");
        let served = served_routes(&String::from_utf8_lossy(&source));
        // A silent scan gap (an unmatched format, a moved file) cannot pass.
        assert!(
            served.len() >= 15,
            "the router scan found {} routes; the scan is broken",
            served.len()
        );
        // The web shell serves files, not the API contract.
        let non_contract = ["/", "/assets/{*path}"];
        let contract = super::fixture()["paths"]
            .as_object()
            .expect("contract paths");
        let undocumented: Vec<&str> = served
            .iter()
            .map(|(path, _)| path.as_str())
            .filter(|route| !non_contract.contains(route) && !contract.contains_key(*route))
            .collect();
        assert!(
            undocumented.is_empty(),
            "routes the router serves but api/openapi.yaml does not describe: {undocumented:?}; \
             add them to the contract and regenerate the fixture"
        );
        let unrouted: Vec<&String> = contract
            .keys()
            .filter(|path| {
                !served
                    .iter()
                    .any(|(served, _)| served.as_str() == path.as_str())
            })
            .collect();
        assert!(
            unrouted.is_empty(),
            "contract paths the router does not serve: {unrouted:?}"
        );
        // Methods too: every served (path, method) pair is declared in the
        // contract, and every declared operation is served.
        let mut served_ops: Vec<(String, String)> = served
            .into_iter()
            .filter(|(path, _)| !non_contract.contains(&path.as_str()))
            .collect();
        let mut declared: Vec<(String, String)> = contract
            .iter()
            .flat_map(|(path, item)| {
                item.as_object()
                    .expect("a path item")
                    .keys()
                    .map(|method| (path.clone(), method.clone()))
                    .collect::<Vec<_>>()
            })
            .collect();
        served_ops.sort();
        declared.sort();
        assert_eq!(
            served_ops, declared,
            "served operations and contract operations differ"
        );
    }

    #[test]
    fn a_body_that_follows_the_contract_passes() {
        let body = json!({"error": {"message": "bad", "code": "bad_request"}});
        assert_schema("Error", &body);
        assert!(matches_schema("Error", &body));
    }

    #[test]
    fn a_body_that_breaks_the_contract_fails() {
        let missing_code = json!({"error": {"message": "bad"}});
        assert!(!matches_schema("Error", &missing_code));
        let unknown_code = json!({"error": {"message": "bad", "code": "teapot"}});
        assert!(!matches_schema("Error", &unknown_code));
    }

    #[test]
    fn a_path_pointer_selects_a_response_schema() {
        let pointer = "/paths/~1health/get/responses/200/content/application~1json/schema";
        assert_pointer(pointer, &json!({"status": "ok"}));
        assert!(
            super::errors(pointer, &json!({"status": "down"})).len() == 1,
            "an unknown status must fail"
        );
    }
}
