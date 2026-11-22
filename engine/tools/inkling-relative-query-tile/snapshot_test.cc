#include "snapshot_plan.h"

#include <algorithm>
#include <cassert>
#include <cstdio>
#include <vector>

namespace {

void check(int first, int width, int capacity, bool restored) {
    assert(snapshot::eligible(true, first, first, width, capacity));
    std::vector<int> ring(capacity, -1), final;
    // Values stand in for row identities; distinct branch identities at the
    // same absolute position catch accidental use of stale same-cursor K/V.
    auto identity = [first, restored](int pos) { return 2 * pos + (restored && pos < first); };
    for (int pos = std::max(0, first - capacity); pos < first; ++pos) {
        ring[pos % capacity] = identity(pos);
    }
    final = ring;
    for (int q = 0; q < width; ++q) {
        const int pos = first + q, slot = pos % capacity;
        final[slot] = identity(pos);
        for (int d = 0; d < std::min(pos + 1, capacity); ++d) {
            const auto src = snapshot::source(q, slot, d, capacity);
            const int row = src.staged ? identity(first + src.row) : ring[src.row];
            assert(row == final[(pos - d) % capacity]);
            assert(row == identity(pos - d));
        }
    }
    std::vector<bool> written(capacity, false);
    for (int row = snapshot::first_commit(width, capacity); row < width; ++row) {
        const int slot = (first + row) % capacity;
        assert(!written[slot]);
        written[slot] = true;
        ring[slot] = identity(first + row);
    }
    assert(ring == final);
}

}  // namespace

int main() {
    using snapshot::eligible;
    assert(!eligible(false, 0, 0, 2, 16));
    assert(!eligible(true, -1, 0, 2, 16));  // invalidated lineage; must reseed
    assert(!eligible(true, 16, 8, 2, 16));  // rewind
    assert(!eligible(true, 7, 8, 2, 16));   // missing history
    assert(!eligible(true, -1, -1, 2, 16));
    assert(!snapshot::valid_interval(0, 0));
    assert(snapshot::valid_interval(0, 1));
    assert(snapshot::valid_interval(0, 65));  // otherwise valid: serial fallback
    assert(!eligible(true, 0, 0, 1, 16));
    assert(!eligible(true, 0, 0, 65, 16));
    assert(!eligible(true, 0, 0, 2, 0));
    assert(!eligible(true, 0, 0, 2, snapshot::kLastPosition + 2));
    assert(!eligible(true, snapshot::kLastPosition, snapshot::kLastPosition, 2, 16));
    assert(eligible(true, snapshot::kLastPosition - 1, snapshot::kLastPosition - 1, 2, 16));
    for (int capacity = 1; capacity <= 12; ++capacity) {
        for (int first = 0; first < 3 * capacity; ++first) {
            for (int width = 2; width <= snapshot::kLimit; ++width) {
                check(first, width, capacity, false);
                check(first, width, capacity, true);
            }
        }
    }
    check(0, 64, 768, false);
    check(448, 64, 768, false);
    check(1007, 64, 1024, true);
    check(snapshot::kLastPosition - 1, 2, 16, true);
    std::puts("PASS snapshot topology: 29,488 cases (sources and final ring), plus eligibility");
}
