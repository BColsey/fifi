//! The default path is the path of main. K-row work (`draft_verify = krow`) must not change what the
//! eager and host windows and the one-row step do. Behaviour is covered elsewhere: `draw_route` sends a
//! window the K-row path did not run to the default draw (`krow_tests`), and the fake of
//! `window_logic_krow_tests` runs the default windows with the K-row path off.
//!
//! What is pinned here is the one edit of a default-path function: `draw` of the eager window gains a
//! call that is taken only for an open K-row window. The pin reads the source like
//! `krow_step_pin_tests` (comments and white space removed) and checks that the greedy return comes
//! first, that the host rows are still drawn as before, and that the call is the only addition.

const EAGER_WINDOW: &str = include_str!("cuda/eager_window.rs");

fn code(source: &str) -> String {
    let mut out = String::new();
    for line in source.lines() {
        let line = line.split("//").next().unwrap_or("");
        out.extend(line.chars().filter(|c| !c.is_whitespace()));
    }
    out
}

fn body<'a>(code: &'a str, name: &str) -> &'a str {
    let start = ["(", "<"]
        .iter()
        .filter_map(|next| code.find(&format!("fn{name}{next}")))
        .min()
        .unwrap_or_else(|| panic!("`fn {name}` not found"));
    let open = code[start..].find('{').expect("a body") + start;
    let mut depth = 0usize;
    for (offset, c) in code[open..].char_indices() {
        match c {
            '{' => depth += 1,
            '}' => {
                depth -= 1;
                if depth == 0 {
                    return &code[open + 1..open + offset];
                }
            }
            _ => {}
        }
    }
    panic!("unbalanced braces in `{name}`")
}

#[test]
fn the_step_draw_gains_one_call_and_keeps_the_greedy_return_and_the_host_rows_as_they_were() {
    let source = code(EAGER_WINDOW);
    let draw = body(&source, "draw");
    let at = |needle: &str| {
        draw.find(&code(needle))
            .unwrap_or_else(|| panic!("`{needle}` not found in the draw"))
    };
    // The greedy request still answers from the device argmax, first.
    let greedy = at("if sampler.is_greedy() {
            return window.argmax.get(row).copied().ok_or_else(no_window);
        }");
    // The addition: one call, and only that call, before the host rows.
    let added = at(
        "if let Some(drawn) = self.draw_from_krow_window(row, sampler, sampler_vocab) {
            return drawn;
        }",
    );
    // The host rows of an eager or host window, as before.
    let host = at("let logits = window.rows.get(row).ok_or_else(no_window)?;
        sample_logits(Self::OPERATION, logits, sampler, sampler_vocab)");
    assert!(greedy < added && added < host);
    assert_eq!(draw.matches("draw_from_krow_window").count(), 1);
    // Nothing else of the K-row stage is reachable from the default draw.
    assert!(
        !draw.replace("draw_from_krow_window", "").contains("krow"),
        "{draw}"
    );
}
