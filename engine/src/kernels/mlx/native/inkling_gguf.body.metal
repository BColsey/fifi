// One SIMD group projects one output row directly from its GGML bytes.
const uint row = thread_position_in_grid.y;
const uint lane = thread_position_in_grid.x;
const device uchar* packed = payload + row * ROW_BYTES;
float sum = 0.0f;
if constexpr (FORMAT == 19) {
    for (uint part = lane; part < WIDTH / 8; part += 32) {
        const device uchar* block = packed + 50 * (part / 32);
        const uint sub = part % 32;
        const uint group = sub / 4;
        const ushort bits = ushort(block[0]) | (ushort(block[1]) << 8);
        const uint high = uint(block[34 + 2 * group]) | (uint(block[35 + 2 * group]) << 8);
        const uint index = uint(block[2 + sub]) | (((high >> (3 * (sub % 4))) & 7) << 8);
        const uint grid = codes[index];
        const float scale = float(as_type<half>(bits)) * float(2 * ((high >> 12) & 7) + 1);
        const float delta = high & 0x8000 ? -0.125f : 0.125f;
        for (uint j = 0; j < 8; ++j) {
            const float w = scale * (float((grid >> (2 * j)) & 3) - 1.0f + delta);
            sum += activation[part * 8 + j] * w;
        }
    }
} else if constexpr (FORMAT == 16 || FORMAT == 17 || FORMAT == 22) {
    constexpr uint block_bytes = FORMAT == 16 ? 66 : FORMAT == 17 ? 74 : 82;
    for (uint part = lane; part < WIDTH / 8; part += 32) {
        const device uchar* block = packed + block_bytes * (part / 32);
        const uint sub = part % 32;
        const uint group = sub / 4;
        const uint l = sub % 4;
        const ushort bits = ushort(block[0]) | (ushort(block[1]) << 8);
        uint index;
        uint signs;
        uint scale_code;
        if constexpr (FORMAT == 16) {
            index = block[2 + 8 * group + l];
            const device uchar* high = block + 6 + 8 * group;
            const uint word = uint(high[0]) | (uint(high[1]) << 8)
                            | (uint(high[2]) << 16) | (uint(high[3]) << 24);
            signs = (word >> (7 * l)) & 127;
            scale_code = word >> 28;
        } else if constexpr (FORMAT == 17) {
            const uint v = uint(block[2 + 2 * sub]) | (uint(block[3 + 2 * sub]) << 8);
            index = v & 511;
            signs = v >> 9;
            scale_code = (uint(block[66 + group]) >> (4 * (l / 2))) & 15;
        } else {
            index = uint(block[2 + sub]) | (((uint(block[66 + group]) >> (2 * l)) & 3) << 8);
            signs = block[34 + sub];
            scale_code = (uint(block[74 + group]) >> (4 * (l / 2))) & 15;
        }
        if constexpr (FORMAT == 16 || FORMAT == 17) {
            // ggml's 128 sign entries append the bit that gives even parity.
            signs |= (popcount(signs) & 1) << 7;
        }
        const float scale = float(as_type<half>(bits)) * (0.5f + float(scale_code)) * 0.25f;
        const uint low_grid = codes[2 * index];
        const uint high_grid = codes[2 * index + 1];
        for (uint j = 0; j < 8; ++j) {
            const uint grid = j < 4 ? low_grid : high_grid;
            const float w = scale * float((grid >> (8 * (j % 4))) & 255);
            sum += activation[part * 8 + j] * (((signs >> j) & 1) ? -w : w);
        }
    }
} else if constexpr (FORMAT == 42) {
    for (uint part = lane; part < WIDTH / 4; part += 32) {
        const device uchar* block = packed + 18 * (part / 16);
        const ushort bits = ushort(block[0]) | (ushort(block[1]) << 8);
        const float scale = float(as_type<half>(bits));
        const uint values = block[2 + part % 16];
        for (uint j = 0; j < 4; ++j) {
            const float w = scale * (float((values >> (2 * j)) & 3) - 1.0f);
            sum += activation[part * 4 + j] * w;
        }
    }
} else if constexpr (FORMAT == 40) {
    const float e2m1[16] = {
        0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f,
        0.0f, -0.5f, -1.0f, -1.5f, -2.0f, -3.0f, -4.0f, -6.0f
    };
    for (uint part = lane; part < WIDTH / 8; part += 32) {
        const device uchar* block = packed + 36 * (part / 8);
        const uint group = (part % 8) / 2;
        const uint shift = 4 * (part % 2);
        const uint code = block[group];
        const uint exponent = (code >> 3) & 15;
        const uint fraction = code & 7;
        const float scale = code == 127 ? 0.0f : code == 255 ? 480.0f
            : exponent == 0 ? float(fraction) / 512.0f
                            : metal::ldexp(1.0f + float(fraction) * 0.125f, int(exponent) - 7);
        for (uint j = 0; j < 8; ++j) {
            const uint q = (block[4 + group * 8 + j] >> shift) & 15;
            sum += activation[part * 8 + j] * (e2m1[q] * scale);
        }
    }
}
sum = simd_sum(sum);
if (lane == 0) result[row] = sum;
