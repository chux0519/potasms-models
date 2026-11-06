#include "include/potasms.h"
#include "../CUtf8Proc/utf8proc.h"
#define PCRE2_CODE_UNIT_WIDTH 8
#define PCRE2_STATIC 1
#include "../CPCRE2/pcre2.h"
#include <cstdlib>
#include <cstring>
#include <memory>
#include <string>
#include <vector>
#include <new>
#include <cstddef>

namespace {
template<class F> int guarded(F f) noexcept {
    try { f(); return POTA_OK; }
    catch (int status) { return status; }
    catch (const std::bad_alloc &) { return POTA_NO_MEMORY; }
    catch (...) { return POTA_INVALID; }
}
void require(bool condition) { if (!condition) throw int(POTA_INVALID); }
std::string unicode(const uint8_t *text, size_t size, bool fold) {
    require(text || size == 0);
    uint8_t *out = nullptr;
    auto options = UTF8PROC_STABLE | UTF8PROC_COMPOSE | UTF8PROC_COMPAT;
    if (fold) options |= UTF8PROC_CASEFOLD;
    auto n = utf8proc_map(text ? text : reinterpret_cast<const uint8_t *>(""), size, &out,
                         static_cast<utf8proc_option_t>(options));
    if (n < 0) throw int(n == UTF8PROC_ERROR_NOMEM ? POTA_NO_MEMORY : POTA_INVALID);
    std::unique_ptr<uint8_t, decltype(&std::free)> owned(out, &std::free);
    if (n > 65536) throw int(POTA_TOO_LARGE);
    return std::string(reinterpret_cast<char *>(out), size_t(n));
}
// Same cleaning contract as imported/model input: whitespace/zero-width cleanup,
// NFKC and optional case folding. Existing POTA_EXACT remains strict.
std::string exactClean(const uint8_t *text, size_t size, bool sensitive) {
    uint8_t *out = nullptr; size_t count = 0;
    int result = pota_normalize(text, size, sensitive ? POTA_CLEAN : POTA_CASEFOLD_V2, &out, &count);
    std::unique_ptr<void, decltype(&pota_free)> owned(out, &pota_free);
    if (result != POTA_OK) throw result;
    return std::string(reinterpret_cast<char *>(out), count);
}
std::string phone(const std::string &input) {
    std::string digits; bool plus = false, seen = false;
    for (unsigned char c : input) {
        if (c >= '0' && c <= '9') { digits.push_back(c); seen = true; }
        else if (c == '+' && !seen && !plus) { plus = true; seen = true; }
        else if (c != ' ' && c != '-' && c != '(' && c != ')' && c != '.') return {};
    }
    if (digits.size() < 3 || digits.size() > 15) return {};
    // Only bare 10-digit US numbers acquire a country code. Never guess by suffix.
    if (!plus && digits.size() == 10) digits = "1" + digits;
    return digits;
}
struct Memory {
    size_t used = 0;
    struct alignas(std::max_align_t) Header { size_t bytes; };
    static void *allocate(PCRE2_SIZE size, void *p) {
        auto &m = *static_cast<Memory *>(p);
        if (size > 2*1024*1024 || m.used > 2*1024*1024-size) return nullptr;
        auto *h = static_cast<Header *>(std::malloc(sizeof(Header)+size));
        if (!h) return nullptr;
        h->bytes = size; m.used += size; return h+1;
    }
    static void release(void *p, void *state) {
        if (!p) return;
        auto *h = static_cast<Header *>(p)-1;
        static_cast<Memory *>(state)->used -= h->bytes; std::free(h);
    }
};
struct Regex {
    Memory memory;
    pcre2_general_context *general = nullptr;
    pcre2_compile_context *compile = nullptr;
    pcre2_code *code = nullptr;
    pcre2_match_context *context = nullptr;
    pcre2_match_data *data = nullptr;
    ~Regex() {
        pcre2_match_data_free(data); pcre2_match_context_free(context);
        pcre2_code_free(code); pcre2_compile_context_free(compile);
        pcre2_general_context_free(general);
    }
    void prepare(const pota_rule &r) {
        general = pcre2_general_context_create(Memory::allocate, Memory::release, &memory);
        if (!general) throw int(POTA_NO_MEMORY);
        compile = pcre2_compile_context_create(general);
        if (!compile) throw int(POTA_NO_MEMORY);
        pcre2_set_parens_nest_limit(compile, 32);
        pcre2_set_max_pattern_length(compile, 512);
        int error = 0; PCRE2_SIZE offset = 0;
        code = pcre2_compile(r.pattern, r.pattern_length,
            PCRE2_UTF | PCRE2_UCP | PCRE2_AUTO_CALLOUT | PCRE2_NEVER_BACKSLASH_C |
            (r.case_sensitive ? 0 : PCRE2_CASELESS), &error, &offset, compile);
        if (!code) throw int(POTA_INVALID);
    }
    bool match(const uint8_t *text, size_t size, uint32_t &budget) {
        context = pcre2_match_context_create(general);
        data = pcre2_match_data_create_from_pattern(code, general);
        if (!context || !data) throw int(POTA_NO_MEMORY);
        pcre2_set_match_limit(context, 20000);
        pcre2_set_depth_limit(context, 128);
        pcre2_set_heap_limit(context, 512);
        pcre2_set_callout(context, [](pcre2_callout_block *, void *p) -> int {
            auto &left = *static_cast<uint32_t *>(p);
            if (!left) return PCRE2_ERROR_CALLOUT;
            --left; return 0;
        }, &budget);
        int result = pcre2_match(code, text ? text : reinterpret_cast<const uint8_t *>(""), size, 0, 0, data, context);
        if (result == PCRE2_ERROR_NOMATCH) return false;
        if (result < 0) throw int(POTA_RULE_LIMIT);
        return true;
    }
};
void validate(const pota_rule &r, bool compileRegex) {
    require(r.kind >= POTA_CONTAINS && r.kind <= POTA_NORMALIZED_EXACT && r.target >= 0 && r.target <= 1 &&
            r.action >= 0 && r.action <= 1 && r.enabled >= 0 && r.enabled <= 1 &&
            r.case_sensitive >= 0 && r.case_sensitive <= 1 && r.pattern && r.pattern_length > 0 && r.pattern_length <= 512);
    auto pattern = unicode(r.pattern, r.pattern_length, !r.case_sensitive);
    require(pattern.find_first_not_of(" \t\r\n") != std::string::npos && pattern.find('\0') == std::string::npos);
    if (r.kind == POTA_NORMALIZED_EXACT) require(!exactClean(r.pattern, r.pattern_length, r.case_sensitive).empty());
    if (r.kind == POTA_PHONE) require(r.target == 1 && !phone(pattern).empty());
    if (r.kind == POTA_REGEX && compileRegex) { Regex regex; regex.prepare(r); }
}
}
extern "C" int pota_rule_validate(const pota_rule *rule) {
    return guarded([&] { require(rule); validate(*rule, true); });
}
extern "C" int pota_rules_match(const pota_rule *rules, size_t count,
    const uint8_t *sender, size_t sender_length, const uint8_t *body, size_t body_length, int32_t *matched) {
    if (matched) *matched = -1;
    return guarded([&] {
        require(matched && (rules || count == 0) && (sender || !sender_length) && (body || !body_length));
        if (count > 100 || sender_length > 256 || body_length > 16384) throw int(POTA_TOO_LARGE);
        unicode(sender, sender_length, false); unicode(body, body_length, false); // strict UTF-8, never truncate
        size_t regexCount = 0;
        for (size_t i=0; i<count; ++i) { validate(rules[i], false); if (rules[i].kind == POTA_REGEX) ++regexCount; }
        if (regexCount > 32) throw int(POTA_TOO_LARGE);
        uint32_t budget = 100000;
        for (int action=0; action<=1; ++action) for (size_t i=0; i<count; ++i) {
            const auto &r = rules[i]; if (!r.enabled || r.action != action) continue;
            auto *text = r.target ? sender : body; auto size = r.target ? sender_length : body_length;
            // Missing sender never matches even a regex such as ^$.
            if (r.target && size == 0) continue;
            bool found = false;
            if (r.kind == POTA_REGEX) { Regex regex; regex.prepare(r); found = regex.match(text, size, budget); }
            else if (r.kind == POTA_NORMALIZED_EXACT) {
                found = exactClean(r.pattern, r.pattern_length, r.case_sensitive) == exactClean(text, size, r.case_sensitive);
            } else {
                auto pattern = unicode(r.pattern, r.pattern_length, !r.case_sensitive);
                auto value = unicode(text, size, !r.case_sensitive);
                if (r.kind == POTA_PHONE) { value = phone(value); pattern = phone(pattern); found = !value.empty() && value == pattern; }
                else if (r.kind == POTA_EXACT) found = value == pattern;
                else if (r.kind == POTA_PREFIX) found = value.compare(0, pattern.size(), pattern) == 0;
                else found = value.find(pattern) != std::string::npos;
            }
            if (found) { *matched = static_cast<int32_t>(i); return; }
        }
    });
}
