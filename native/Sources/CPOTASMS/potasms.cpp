#include "potasms.h"
#include "utf8proc.h"
#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <new>
#include <memory>
#include <string>
#include <vector>

namespace {
#include "unicode15_properties.inc"
constexpr size_t kMaxText = 16384;
constexpr size_t kMaxNormalized = 65536;
constexpr size_t kMaxBuckets = 1048576;

template<size_t N> bool contains(const UnicodeRange (&ranges)[N], int32_t c) {
    auto it = std::lower_bound(ranges, ranges + N, c,
                              [](const UnicodeRange &r, int32_t x) { return r.last < x; });
    return it != ranges + N && it->first <= c;
}
bool whitespace(int32_t c) {
    // Unicode White_Space plus U+001C..001F, matching legacy Python semantics.
    return (c >= 9 && c <= 13) || (c >= 0x1c && c <= 0x20) || c == 0x85 ||
        c == 0xa0 || c == 0x1680 || (c >= 0x2000 && c <= 0x200a) ||
        c == 0x2028 || c == 0x2029 || c == 0x202f || c == 0x205f || c == 0x3000;
}
void append(std::string &s, int32_t c) {
    uint8_t bytes[4]; auto n = utf8proc_encode_char(c, bytes);
    s.append(reinterpret_cast<char *>(bytes), static_cast<size_t>(n));
}
std::vector<int32_t> scalars(const std::string &s) {
    std::vector<int32_t> out;
    for (size_t i = 0; i < s.size();) {
        int32_t c; auto n = utf8proc_iterate(reinterpret_cast<const uint8_t *>(s.data()+i), s.size()-i, &c);
        if (n <= 0) throw static_cast<int>(POTA_INVALID);
        out.push_back(c); i += static_cast<size_t>(n);
    }
    return out;
}
std::string mapped(const uint8_t *s, size_t n, utf8proc_option_t options) {
    uint8_t *out = nullptr;
    auto length = utf8proc_map(s ? s : reinterpret_cast<const uint8_t *>(""), n, &out, options);
    if (length < 0) throw static_cast<int>(length == UTF8PROC_ERROR_NOMEM ? POTA_NO_MEMORY : POTA_INVALID);
    std::unique_ptr<uint8_t, decltype(&std::free)> owned(out, &std::free);
    if (static_cast<size_t>(length) > kMaxNormalized) throw static_cast<int>(POTA_TOO_LARGE);
    std::string result(reinterpret_cast<char *>(out), static_cast<size_t>(length));
    return result;
}
std::string normalize(const uint8_t *text, size_t length, int mode) {
    if ((!text && length) || mode < POTA_CLEAN || mode > POTA_CASEFOLD_V2) throw static_cast<int>(POTA_INVALID);
    if (length > kMaxText) throw static_cast<int>(POTA_TOO_LARGE);
    auto s = mapped(text, length, static_cast<utf8proc_option_t>(UTF8PROC_STABLE | UTF8PROC_COMPOSE | UTF8PROC_COMPAT));
    std::string clean; bool pending = false;
    for (auto c : scalars(s)) {
        if (c == 0x200b || c == 0xfeff) continue;
        if (whitespace(c)) { if (!clean.empty()) pending = true; continue; }
        if (pending) { clean.push_back(' '); pending = false; }
        append(clean, c);
    }
    if (mode == POTA_CLEAN) return clean;
    if (mode == POTA_CASEFOLD_V2)
        return mapped(reinterpret_cast<const uint8_t *>(clean.data()), clean.size(), UTF8PROC_CASEFOLD);
    // Locale-independent full lower, including the two default special cases.
    auto chars = scalars(clean); std::string lower;
    for (size_t i = 0; i < chars.size(); ++i) {
        auto c = chars[i];
        if (c == 0x130) { append(lower, 0x69); append(lower, 0x307); continue; }
        if (c == 0x3a3) {
            bool before = false, after = false;
            for (size_t j = i; j > 0;) {
                auto prev = chars[--j];
                if (contains(kCaseIgnorable, prev)) continue;
                before = contains(kCased, prev); break;
            }
            for (size_t j = i+1; j < chars.size(); ++j) {
                if (contains(kCaseIgnorable, chars[j])) continue;
                after = contains(kCased, chars[j]); break;
            }
            append(lower, before && !after ? 0x3c2 : 0x3c3);
        } else append(lower, utf8proc_tolower(c));
    }
    return lower;
}
std::vector<uint32_t> ids(const uint8_t *text, size_t length, int mode,
                          uint32_t buckets, uint32_t min_n, uint32_t max_n) {
    if (!buckets || buckets > kMaxBuckets || min_n != 2 || max_n != 5) throw static_cast<int>(POTA_INVALID);
    auto s = "^" + normalize(text, length, mode) + "$";
    auto chars = scalars(s); std::vector<std::string> encoded;
    for (auto c : chars) { std::string one; append(one, c); encoded.push_back(one); }
    std::vector<uint32_t> result;
    result.reserve(chars.size()*4);
    for (uint32_t n = min_n; n <= max_n; ++n)
        for (size_t start = 0; start+n <= chars.size(); ++start) {
            uint32_t hash = 2166136261u;
            for (size_t j = start; j < start+n; ++j)
                for (unsigned char b : encoded[j]) { hash ^= b; hash *= 16777619u; }
            result.push_back(hash % buckets);
        }
    if (result.empty()) result.push_back(0);
    return result;
}
template<typename F> int protect(F f) noexcept {
    try { f(); return POTA_OK; }
    catch (int status) { return status; }
    catch (const std::bad_alloc &) { return POTA_NO_MEMORY; }
    catch (...) { return POTA_INVALID; }
}
template<typename T> void copy_out(const std::vector<T> &in, T **out, size_t *count) {
    auto p = static_cast<T *>(std::malloc(std::max<size_t>(1, in.size()*sizeof(T))));
    if (!p) throw static_cast<int>(POTA_NO_MEMORY);
    std::memcpy(p, in.data(), in.size()*sizeof(T)); *out = p; *count = in.size();
}
bool valid_features(const pota_feature *f, size_t n, size_t buckets) {
    if ((!f && n) || n > kMaxNormalized*4 || !buckets || buckets > kMaxBuckets) return false;
    for (size_t i=0; i<n; ++i)
        if (f[i].bucket >= buckets || !std::isfinite(f[i].value) || f[i].value < 0 || f[i].value > 1) return false;
    return true;
}
bool valid_model(const int8_t *w, size_t n, float scale, float bias, float delta_bias) {
    return w && n && n <= kMaxBuckets && std::isfinite(scale) && scale > 0 && scale <= 100 &&
        std::isfinite(bias) && std::abs(bias) <= 1000 && std::isfinite(delta_bias) && std::abs(delta_bias) <= 16;
}
bool valid_embedding_features(const pota_feature *f,size_t n,size_t dimensions) {
    if (!f || !n || !dimensions || dimensions>4096 || n>dimensions) return false;
    double squared=0;
    for (size_t i=0;i<n;++i) {
        if (f[i].bucket>=dimensions || !std::isfinite(f[i].value) ||
            std::abs(f[i].value)>1 || (i && f[i-1].bucket>=f[i].bucket)) return false;
        squared+=static_cast<double>(f[i].value)*f[i].value;
    }
    return std::abs(squared-1)<=1e-5;
}
}

extern "C" {
const char *pota_version(void) { return "0.1.0-utf8proc2.8.0-unicode15.0.0"; }
const char *pota_feature_contract(void) { return "unicode15-nfkc-casefold-fnv1a-char-2-5-l2-v2"; }
void pota_free(void *memory) { std::free(memory); }
int pota_normalize(const uint8_t *text, size_t length, int mode, uint8_t **out, size_t *n) {
    if (!out || !n) return POTA_INVALID;
    *out=nullptr; *n=0;
    return protect([&] { auto s=normalize(text,length,mode); std::vector<uint8_t> b(s.begin(),s.end()); copy_out(b,out,n); });
}
int pota_bucket_ids(const uint8_t *text, size_t length, int mode, uint32_t buckets,
                    uint32_t min_n, uint32_t max_n, uint32_t **out, size_t *n) {
    if (!out || !n) return POTA_INVALID;
    *out=nullptr; *n=0;
    return protect([&] { copy_out(ids(text,length,mode,buckets,min_n,max_n),out,n); });
}
int pota_features(const uint8_t *text, size_t length, int mode, uint32_t buckets,
                  uint32_t min_n, uint32_t max_n, pota_feature **out, size_t *n) {
    if (!out || !n) return POTA_INVALID;
    *out=nullptr; *n=0;
    return protect([&] {
        auto all=ids(text,length,mode,buckets,min_n,max_n); std::sort(all.begin(),all.end());
        std::vector<pota_feature> f; double squared=0;
        for (size_t i=0; i<all.size();) {
            size_t j=i+1; while (j<all.size() && all[j]==all[i]) ++j;
            double count=j-i; squared+=count*count; f.push_back({all[i],static_cast<float>(count)}); i=j;
        }
        float norm=static_cast<float>(std::sqrt(std::max(squared,1.0)));
        for (auto &x:f) x.value/=norm;
        copy_out(f,out,n);
    });
}
float pota_sigmoid(float x) {
    if (x>=0) return 1.0f/(1.0f+std::exp(-x));
    float e=std::exp(x); return e/(1.0f+e);
}
int pota_logit(const pota_feature *f, size_t n, const int8_t *w, size_t buckets,
               float scale, float bias, const float *delta, float db, float *out) {
    if (!out) return POTA_INVALID;
    *out=0;
    if (!valid_features(f,n,buckets) || !valid_model(w,buckets,scale,bias,db)) return POTA_INVALID;
    float value=bias+db;
    for(size_t i=0;i<n;++i) {
        float d=delta?delta[i]:0;
        if (!std::isfinite(d) || std::abs(d)>16) return POTA_INVALID;
        value += f[i].value * (static_cast<float>(w[f[i].bucket])*scale+d);
    }
    if (!std::isfinite(value)) return POTA_INVALID;
    *out=value; return POTA_OK;
}
int pota_float_logit(const pota_feature *f,size_t n,const float *w,size_t dimensions,
                     float bias,const float *delta,float db,float *out) {
    if (!out) return POTA_INVALID;
    *out=0;
    if (!valid_embedding_features(f,n,dimensions) || !std::isfinite(bias) ||
        std::abs(bias)>1000 || !std::isfinite(db) || std::abs(db)>16) return POTA_INVALID;
    if (w) for (size_t i=0;i<dimensions;++i)
        if (!std::isfinite(w[i]) || std::abs(w[i])>1000) return POTA_INVALID;
    float value=bias+db;
    for (size_t i=0;i<n;++i) {
        const float d=delta?delta[i]:0;
        if (!std::isfinite(d) || std::abs(d)>16) return POTA_INVALID;
        value+=f[i].value*((w?w[f[i].bucket]:0)+d);
    }
    if (!std::isfinite(value)) return POTA_INVALID;
    *out=value;return POTA_OK;
}
int pota_pa2_margin_sample(const pota_feature *f,size_t n,float *delta,size_t dimensions,
                           float base,uint8_t label,float c,float margin,float limit) {
    if (!delta || label>1 || !valid_embedding_features(f,n,dimensions) ||
        !std::isfinite(base) || std::abs(base)>1000 || !std::isfinite(c) || c<=0 || c>1000 ||
        !std::isfinite(margin) || margin<=0 || margin>10 ||
        !std::isfinite(limit) || limit<=0 || limit>16) return POTA_INVALID;
    float z=base;double norm=0;
    for (size_t i=0;i<n;++i) {
        const float d=delta[f[i].bucket];
        if (!std::isfinite(d) || std::abs(d)>16) return POTA_INVALID;
        z+=f[i].value*d;norm+=static_cast<double>(f[i].value)*f[i].value;
    }
    if (!std::isfinite(z)) return POTA_INVALID;
    const float sign=label?1.0f:-1.0f;
    const float tau=std::max(0.0f,margin-sign*z)/(static_cast<float>(norm)+1.0f/(2*c));
    if (!std::isfinite(tau)) return POTA_INVALID;
    for (size_t i=0;i<n;++i)
        if (!std::isfinite(delta[f[i].bucket]+tau*sign*f[i].value)) return POTA_INVALID;
    for (size_t i=0;i<n;++i) {
        auto &d=delta[f[i].bucket];d=std::max(-limit,std::min(limit,d+tau*sign*f[i].value));
    }
    return POTA_OK;
}
int pota_anchor_sample(const pota_feature *f,size_t n,float *delta,size_t buckets,float rate) {
    if (!delta || !n || !valid_features(f,n,buckets) || !std::isfinite(rate) || rate<=0 || rate>1)
        return POTA_INVALID;
    float z=0; double norm=0;
    for(size_t i=0;i<n;++i) {
        if (i && f[i-1].bucket>=f[i].bucket) return POTA_INVALID;
        float d=delta[f[i].bucket];
        if (!std::isfinite(d) || std::abs(d)>16) return POTA_INVALID;
        z += f[i].value*d;
        norm += static_cast<double>(f[i].value)*f[i].value;
    }
    if (!std::isfinite(z) || std::abs(norm-1)>1e-5) return POTA_INVALID;
    for(size_t i=0;i<n;++i) {
        float value=delta[f[i].bucket]-rate*z*f[i].value;
        if (!std::isfinite(value) || std::abs(value)>16) return POTA_INVALID;
    }
    for(size_t i=0;i<n;++i) delta[f[i].bucket]-=rate*z*f[i].value;
    return POTA_OK;
}
int pota_pa2_sample(const pota_feature *f,size_t n,const int8_t *w,size_t buckets,
                    float scale,float bias,float *delta,float db,float boundary,
                    uint8_t label,float c,float margin,float limit) {
    if (!delta || !n || label>1 || !valid_features(f,n,buckets) ||
        !valid_model(w,buckets,scale,bias,db) || !std::isfinite(boundary) || std::abs(boundary)>120 ||
        !std::isfinite(c) || c<=0 || c>1000 || !std::isfinite(margin) || margin<=0 || margin>10 ||
        !std::isfinite(limit) || limit<=0 || limit>16) return POTA_INVALID;
    float z=bias+db-boundary; double norm=0;
    for (size_t i=0;i<n;++i) {
        if (i && f[i-1].bucket>=f[i].bucket) return POTA_INVALID;
        float d=delta[f[i].bucket];
        if (!std::isfinite(d) || std::abs(d)>16) return POTA_INVALID;
        z+=f[i].value*(static_cast<float>(w[f[i].bucket])*scale+d);
        norm+=static_cast<double>(f[i].value)*f[i].value;
    }
    if (!std::isfinite(z) || std::abs(norm-1)>1e-5) return POTA_INVALID;
    const float sign=label?1.0f:-1.0f;
    const float tau=std::max(0.0f,margin-sign*z)/(static_cast<float>(norm)+1.0f/(2*c));
    if (!std::isfinite(tau)) return POTA_INVALID;
    for (size_t i=0;i<n;++i) {
        const float value=delta[f[i].bucket]+tau*sign*f[i].value;
        if (!std::isfinite(value)) return POTA_INVALID;
    }
    for (size_t i=0;i<n;++i) {
        auto &d=delta[f[i].bucket];
        d=std::max(-limit,std::min(limit,d+tau*sign*f[i].value));
    }
    return POTA_OK;
}
int pota_sgd_sample(const pota_feature *f,size_t n,const int8_t *w,size_t buckets,
                    float scale,float bias,float *delta,float *db,uint8_t label,
                    float rate,float l2,float limit) {
    if (!delta || !db || label>1 || !valid_features(f,n,buckets) ||
        !valid_model(w,buckets,scale,bias,*db) || !std::isfinite(rate) || rate<=0 || rate>1 ||
        !std::isfinite(l2) || l2<0 || l2>1 || !std::isfinite(limit) || limit<=0 || limit>16) return POTA_INVALID;
    float z=bias+*db;
    for(size_t i=0;i<n;++i) {
        float d=delta[f[i].bucket];
        if(!std::isfinite(d) || std::abs(d)>16) return POTA_INVALID;
        z += f[i].value * (static_cast<float>(w[f[i].bucket])*scale+d);
    }
    float error=pota_sigmoid(z)-label;
    for(size_t i=0;i<n;++i) {
        auto &d=delta[f[i].bucket];
        d=std::max(-limit,std::min(limit,d-rate*(error*f[i].value+l2*d)));
    }
    *db=std::max(-limit,std::min(limit,*db-rate*error*0.1f)); return POTA_OK;
}
}
