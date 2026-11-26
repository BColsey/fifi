#ifndef INFERENCE_CUDA_RESIDENT_RELATIVE_ATTENTION_TILE_H
#define INFERENCE_CUDA_RESIDENT_RELATIVE_ATTENTION_TILE_H

#ifdef __CUDACC__
#define INK_TILE_HD __host__ __device__
#else
#define INK_TILE_HD
#endif

namespace snapshot {

constexpr int kLimit = 512;
constexpr int kMaxStride = (1 << 20) / 4;  // words; bounds row offset arithmetic
constexpr int kLastPosition = (1 << 30) - 2;  // native relative attention position limit

inline bool valid_interval(int first, int count) {
    return first >= 0 && count >= 1 && count - 1 <= kLastPosition &&
           first <= kLastPosition - (count - 1);
}

// The caller owns the visit's ring exclusively, has certified its lineage, and has
// already queued all writes that establish this cursor on the same stream.
// Cursor equality alone is insufficient after a rewind or divergent prefix restore.
inline bool eligible(bool enabled, int cursor, int first, int count, int capacity) {
    return enabled && cursor == first && count >= 2 && count <= kLimit &&
           valid_interval(first, count) && capacity >= 1 && capacity <= kLastPosition + 1;
}

struct Source {
    bool staged;
    int row;
};

// Only used for 0 <= distance < min(position + 1, capacity). Every CTA sees the
// original ring; visible rows appended by the tile are read from immutable K/V.
INK_TILE_HD inline Source source(int query, int slot, int distance, int capacity) {
    if (distance <= query) return {true, query - distance};
    const int row = slot - distance;
    return {false, row < 0 ? row + capacity : row};
}

// A tile may wrap more than once. Publish only its last writer to each slot.
INK_TILE_HD inline int first_commit(int count, int capacity) {
    return count > capacity ? count - capacity : 0;
}

}  // namespace snapshot

#undef INK_TILE_HD
#endif
