#pragma once
#include <type_traits>
// Optional CTA-local codebook for SM75 expert WMMA. Grid entries come
// from FiFi's existing decoder; no new quantization or arithmetic order.
template<int Codec> struct InklingCodebook {
    using Word=std::conditional_t<Codec==19,unsigned,uint64_t>;
    static constexpr int count=Codec==19?2048:Codec==16?256:Codec==17?512:1024;
    Word words[count];
    void __device__ init(){
        for(int i=threadIdx.x;i<count;i+=blockDim.x){
            if constexpr(Codec==19)words[i]=iq1s_grid_gpu[i];
            else if constexpr(Codec==16)words[i]=iq2xxs_grid[i];
            else if constexpr(Codec==17)words[i]=iq2xs_grid[i];
            else words[i]=iq2s_grid[i];}
        __syncthreads();}
    float __device__ scalar(const uint8_t *b,int i)const{
        const int r=i/32,s=i%32/8,l=i%8;const float d=fifi_gguf::half_value(b);
        if constexpr(Codec==19){const unsigned hi=fifi_gguf::read16(b+34+2*r),word=words[b[2+4*r+s]|(((hi>>(3*s))&7)<<8)];
            int code=(word>>(8*(l%4)+4*(l/4)))&15;return (d*float(1+2*((hi>>12)&7)))*(float(code-1)+((hi&0x8000)?-0.125f:0.125f));
        }else{unsigned idx,signs;float scale;
            if constexpr(Codec==16){unsigned v=fifi_gguf::read32(b+6+8*r);idx=b[2+8*r+s];signs=ksigns_iq2xs[(v>>(7*s))&127];scale=d*(0.5f+float(v>>28))*0.25f;}
            else if constexpr(Codec==17){unsigned v=fifi_gguf::read16(b+2+2*(4*r+s));idx=v&511;signs=ksigns_iq2xs[v>>9];scale=d*(0.5f+float(fifi_gguf::nibble(b[66+r],s/2)))*0.25f;}
            else{idx=b[2+4*r+s]|((unsigned(b[66+r])<<(8-2*s))&768);signs=b[34+4*r+s];scale=d*(0.5f+float(fifi_gguf::nibble(b[74+r],s/2)))*0.25f;}
            return fifi_gguf::grid_value(words[idx],l,scale,signs);}
    }
    void __device__ octet(const uint8_t *row,int c,__half *out)const{
        constexpr int bytes=Codec==17?74:Codec==19?50:82;
        const auto *b=row+(c/256)*bytes;int r=c%256/32,s=c%32/8;
        float d=fifi_gguf::half_value(b);
        if constexpr(Codec==19){unsigned hi=fifi_gguf::read16(b+34+2*r);unsigned word=words[b[2+4*r+s]|(((hi>>(3*s))&7)<<8)];
            float scale=d*float(1+2*((hi>>12)&7));float delta=(hi&0x8000)?-0.125f:0.125f;
#pragma unroll
            for(int j=0;j<8;j++){int q=(word>>(8*(j%4)+4*(j/4)))&15;out[j]=__float2half_rn(scale*(float(q-1)+delta));}
        }else{unsigned idx,signs;float scale;
            if constexpr(Codec==17){unsigned v=fifi_gguf::read16(b+2+2*(4*r+s));idx=v&511;signs=ksigns_iq2xs[v>>9];scale=d*(0.5f+float(fifi_gguf::nibble(b[66+r],s/2)))*0.25f;}
            else {idx=b[2+4*r+s]|((unsigned(b[66+r])<<(8-2*s))&768);signs=b[34+4*r+s];scale=d*(0.5f+float(fifi_gguf::nibble(b[74+r],s/2)))*0.25f;}
            auto word=words[idx];
#pragma unroll
            for(int j=0;j<8;j++)out[j]=__float2half_rn(fifi_gguf::grid_value(word,j,scale,signs));}
    }
};
