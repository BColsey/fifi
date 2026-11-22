//! Request-local limits for a speculation policy that can use fewer draft rows.

#[derive(Default)]
pub(crate) struct DraftBudget {
    cap: Option<usize>,
    weak_windows: u8,
}

impl DraftBudget {
    pub(super) fn cap(&self, configured: usize) -> usize {
        self.cap.unwrap_or(configured).min(configured)
    }

    pub(super) fn suspended(&self) -> bool {
        self.cap == Some(0)
    }

    pub(super) fn observe(&mut self, configured: usize, drafted: usize, accepted: usize) {
        debug_assert!(accepted <= drafted && drafted > 0);
        let old = self.cap(configured);
        self.weak_windows = if accepted < drafted.div_ceil(2) {
            self.weak_windows.saturating_add(1)
        } else {
            0
        };
        self.cap = Some(if self.weak_windows >= 2 {
            0
        } else if accepted == drafted {
            old.saturating_mul(2).min(configured)
        } else {
            // A rejection keeps one probe beyond the prefix that paid off.
            old.min(accepted + 1)
        });
    }
}
