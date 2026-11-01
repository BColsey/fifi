//! Drafts from the committed tokens alone: the n-gram draft and the hybrid long-passage draft.
//!
//! `RepeatIndex` keeps the tokens of one session and, for each of two tail lengths, the position
//! that followed the latest earlier occurrence of every tail. A draft is the run of tokens after
//! that position. Hash collisions cannot produce a wrong draft that looks right: the match length
//! is counted on the tokens themselves, and a draft is only a guess that the target verifies.

use std::collections::HashMap;

/// Longest match length the backward count measures; a longer repeat is the same to the caller.
const MATCH_CAP: usize = 512;

/// A guess for the tokens after the pending token.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct Draft {
    pub(crate) tokens: Vec<u32>,
    /// How many tokens of the context's tail the earlier occurrence repeats.
    pub(crate) match_len: usize,
}

/// The latest-occurrence index of a token stream.
pub(crate) struct RepeatIndex {
    tokens: Vec<u32>,
    short_len: usize,
    long_len: usize,
    short: HashMap<u64, u32>,
    long: HashMap<u64, u32>,
}

fn gram_key(gram: &[u32]) -> u64 {
    // FNV-1a over the little-endian words.
    gram.iter().fold(0xcbf2_9ce4_8422_2325_u64, |hash, &token| {
        token.to_le_bytes().iter().fold(hash, |h, &b| {
            (h ^ u64::from(b)).wrapping_mul(0x0100_0000_01b3)
        })
    })
}

impl RepeatIndex {
    /// An empty index. `short_len` is the n-gram tail length (at least 1); `long_len` is the
    /// hybrid tail length, 0 for none.
    pub(crate) fn new(short_len: usize, long_len: usize) -> Self {
        Self {
            tokens: Vec::new(),
            short_len: short_len.max(1),
            long_len,
            short: HashMap::new(),
            long: HashMap::new(),
        }
    }

    pub(crate) fn len(&self) -> usize {
        self.tokens.len()
    }

    pub(crate) fn clear(&mut self) {
        self.tokens.clear();
        self.short.clear();
        self.long.clear();
    }

    /// Appends one committed token. The tail that ends just before it is registered with the
    /// token as its follower, so the tail that ends AT the new token is not yet findable and a
    /// lookup never matches itself.
    pub(crate) fn push(&mut self, token: u32) {
        let at = self.tokens.len();
        self.tokens.push(token);
        Self::register(&mut self.short, &self.tokens, self.short_len, at);
        if self.long_len > 0 {
            Self::register(&mut self.long, &self.tokens, self.long_len, at);
        }
    }

    fn register(map: &mut HashMap<u64, u32>, tokens: &[u32], n: usize, follower: usize) {
        if follower < n {
            return;
        }
        if let Ok(position) = u32::try_from(follower) {
            map.insert(gram_key(&tokens[follower - n..follower]), position);
        }
    }

    /// Brings the index to `history`: appends the new tail when the index looks like a prefix of
    /// it, rebuilds from scratch otherwise (a new conversation or a rewound session). The test
    /// samples the first, middle and last indexed token instead of comparing the whole prefix, so a
    /// call costs O(new tokens). A stale index after a lucky sample would only make worse guesses:
    /// the target verifies every draft.
    pub(crate) fn sync(&mut self, history: &[u32]) {
        let held = self.tokens.len();
        let samples = [0, held / 2, held.saturating_sub(1)];
        let prefix = held <= history.len()
            && (held == 0 || samples.iter().all(|&i| history[i] == self.tokens[i]));
        if !prefix {
            self.clear();
        }
        for &token in &history[self.tokens.len()..] {
            self.push(token);
        }
    }

    /// The follower position and the match length of the tail of `n` tokens, if an earlier
    /// occurrence exists.
    fn lookup(&self, map: &HashMap<u64, u32>, n: usize) -> Option<(usize, usize)> {
        let len = self.tokens.len();
        if n == 0 || len < n {
            return None;
        }
        let follower = *map.get(&gram_key(&self.tokens[len - n..]))? as usize;
        let mut matched = 0;
        while matched < MATCH_CAP
            && matched < follower
            && self.tokens[follower - 1 - matched] == self.tokens[len - 1 - matched]
        {
            matched += 1;
        }
        (matched >= n).then_some((follower, matched))
    }

    fn draft_from(&self, found: Option<(usize, usize)>, max_tokens: usize) -> Option<Draft> {
        let (follower, match_len) = found?;
        let end = (follower + max_tokens).min(self.tokens.len());
        (end > follower).then(|| Draft {
            tokens: self.tokens[follower..end].to_vec(),
            match_len,
        })
    }

    /// The tokens after the latest earlier occurrence of the last `short_len` tokens, at most
    /// `max_tokens` of them.
    pub(crate) fn ngram(&self, max_tokens: usize) -> Option<Draft> {
        self.draft_from(self.lookup(&self.short, self.short_len), max_tokens)
    }

    /// The tokens after the latest earlier occurrence of the last `long_len` tokens, at most
    /// `max_tokens` of them. `None` when the hybrid draft is off.
    pub(crate) fn hybrid(&self, max_tokens: usize) -> Option<Draft> {
        self.draft_from(self.lookup(&self.long, self.long_len), max_tokens)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn index(short: usize, long: usize, tokens: &[u32]) -> RepeatIndex {
        let mut index = RepeatIndex::new(short, long);
        tokens.iter().for_each(|&t| index.push(t));
        index
    }

    #[test]
    fn a_repeated_tail_drafts_its_followers() {
        let index = index(2, 0, &[1, 2, 3, 4, 9, 1, 2]);
        let draft = index.ngram(3).expect("draft");
        assert_eq!(draft.tokens, [3, 4, 9]);
        assert_eq!(draft.match_len, 2);
    }

    #[test]
    fn the_latest_occurrence_wins() {
        // The tail [4, 5] occurs at 0 (follower 3) and at 3 (follower 9).
        let index = index(2, 0, &[4, 5, 3, 4, 5, 9, 4, 5]);
        assert_eq!(index.ngram(1).expect("draft").tokens, [9]);
    }

    #[test]
    fn a_tail_never_matches_itself() {
        assert!(index(2, 0, &[1, 2, 3, 4]).ngram(4).is_none());
        assert!(index(3, 0, &[1, 2]).ngram(4).is_none(), "too short");
        assert!(index(3, 0, &[]).ngram(4).is_none(), "empty");
    }

    #[test]
    fn a_draft_stops_at_the_end_of_the_context() {
        // The latest earlier [7, 7] is followed by the last token only.
        let index = index(2, 0, &[7, 7, 7, 7]);
        assert_eq!(index.ngram(8).expect("draft").tokens, [7]);
    }

    #[test]
    fn the_match_length_counts_past_the_tail() {
        let index = index(2, 6, &[1, 2, 3, 4, 5, 6, 8, 1, 2, 3, 4, 5, 6]);
        let ngram = index.ngram(1).expect("ngram");
        assert_eq!((ngram.tokens.clone(), ngram.match_len), (vec![8], 6));
        let hybrid = index.hybrid(4).expect("hybrid");
        assert_eq!((hybrid.tokens, hybrid.match_len), (vec![8, 1, 2, 3], 6));
    }

    #[test]
    fn the_hybrid_draft_needs_the_whole_long_tail() {
        // Only five tokens repeat; a six-token tail was never seen before.
        let index = index(2, 6, &[9, 1, 2, 3, 4, 5, 7, 8, 1, 2, 3, 4, 5]);
        assert!(index.hybrid(4).is_none());
        assert!(index.ngram(4).is_some());
    }

    #[test]
    fn the_hybrid_draft_is_off_without_a_long_tail() {
        assert!(index(2, 0, &[1, 2, 3, 1, 2]).hybrid(4).is_none());
    }

    #[test]
    fn a_colliding_key_cannot_pass_a_false_match() {
        let mut index = RepeatIndex::new(2, 0);
        [1, 2, 3, 4].iter().for_each(|&t| index.push(t));
        // Point the key of the current tail [3, 4] at a place that holds other tokens.
        let key = gram_key(&[3, 4]);
        index.short.insert(key, 2);
        assert!(index.ngram(2).is_none());
    }

    #[test]
    fn sync_appends_a_longer_history_and_rebuilds_a_different_one() {
        let mut index = RepeatIndex::new(2, 0);
        index.sync(&[1, 2, 3]);
        assert_eq!(index.len(), 3);
        index.sync(&[1, 2, 3, 1, 2]);
        assert_eq!(index.ngram(1).expect("draft").tokens, [3]);
        index.sync(&[5, 6, 5, 6]);
        assert_eq!(index.len(), 4);
        assert_eq!(index.ngram(1).expect("draft").tokens, [5]);
        index.sync(&[9, 9, 9, 9]);
        assert_eq!(
            index.len(),
            4,
            "a history that differs at the same length rebuilds"
        );
        assert_eq!(index.ngram(1).expect("draft").tokens, [9]);
        index.sync(&[5, 6]);
        assert_eq!(index.len(), 2, "a shorter history rebuilds");
        assert!(index.ngram(1).is_none());
    }
}
