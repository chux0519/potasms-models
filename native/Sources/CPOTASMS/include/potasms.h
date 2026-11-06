#ifndef POTASMS_H
#define POTASMS_H
#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Stable C ABI, no platform frameworks, exceptions, or ownership crossing ABI.
 * Allocated outputs must be freed with pota_free. Input strings are UTF-8 with
 * explicit byte lengths; embedded NUL is supported. All functions are reentrant.
 * Text is capped at 16384 bytes; callers must fail open on errors, never truncate.
 */
enum { POTA_OK = 0, POTA_INVALID = 1, POTA_TOO_LARGE = 2, POTA_NO_MEMORY = 3 };
enum { POTA_CLEAN = 0, POTA_LEGACY_LOWER = 1, POTA_CASEFOLD_V2 = 2 };
typedef struct { uint32_t bucket; float value; } pota_feature;

/* Explicit Keep rules precede Junk rules; array order breaks ties. No match=-1.
 * Regex is PCRE2 UTF/UCP without JIT, with bounded memory and work. Errors must
 * yield no opinion, never a Junk decision. Maximum 100 rules, 32 regex rules.
 * Strings are UTF-8. Patterns <=512 bytes; body <=16384; sender <=256. */
/* NORMALIZED_EXACT uses import/model text cleaning (spacing/zero-width,
 * NFKC and optional case folding), with no word/number/URL generalization.
 * Existing EXACT semantics and the struct/function ABI stay unchanged. */
enum { POTA_RULE_LIMIT = 4 };
enum { POTA_CONTAINS = 0, POTA_EXACT = 1, POTA_PREFIX = 2, POTA_REGEX = 3, POTA_PHONE = 4, POTA_NORMALIZED_EXACT = 5 };
typedef struct {
    int32_t kind, target, action, enabled, case_sensitive; /* target: body=0 sender=1; Keep=0 Junk=1 */
    const uint8_t *pattern;
    size_t pattern_length;
} pota_rule;
int pota_rule_validate(const pota_rule *rule);
int pota_rules_match(const pota_rule *rules, size_t count,
                     const uint8_t *sender, size_t sender_length,
                     const uint8_t *body, size_t body_length, int32_t *matched_index);

const char *pota_version(void);
const char *pota_feature_contract(void);
void pota_free(void *memory);
int pota_normalize(const uint8_t *text, size_t length, int mode,
                   uint8_t **output, size_t *output_length);
int pota_bucket_ids(const uint8_t *text, size_t length, int mode,
                    uint32_t buckets, uint32_t min_n, uint32_t max_n,
                    uint32_t **output, size_t *count);
int pota_features(const uint8_t *text, size_t length, int mode,
                  uint32_t buckets, uint32_t min_n, uint32_t max_n,
                  pota_feature **output, size_t *count);
/* aligned_delta has one value per feature, or may be NULL for base-only. */
int pota_logit(const pota_feature *features, size_t count,
               const int8_t *weights, size_t buckets, float scale, float bias,
               const float *aligned_delta, float bias_delta, float *output);
float pota_sigmoid(float logit);
/* Signed, sorted, L2-unit embedding features (1..4096 dimensions). Unlike the
 * character API this accepts negative coordinates. Float weights may be NULL
 * to compute only a personal delta; aligned_delta has one entry per feature. */
int pota_float_logit(const pota_feature *features, size_t count,
                     const float *weights, size_t dimensions, float bias,
                     const float *aligned_delta, float bias_delta, float *output);
/* PA-II in an explicit calibrated decision-margin domain. The base margin is
 * fixed by the caller; only dense_delta is updated, never the head or bias. */
int pota_pa2_margin_sample(const pota_feature *features, size_t count,
                           float *dense_delta, size_t dimensions, float base_margin,
                           uint8_t label, float aggressiveness, float margin,
                           float maximum_delta);
/* One bounded SGD update; dense_delta has buckets entries. All inputs are
 * validated before mutation. Callers own replay order, acceptance and rollback. */
int pota_sgd_sample(const pota_feature *features, size_t count,
                    const int8_t *weights, size_t buckets, float scale, float bias,
                    float *dense_delta, float *bias_delta, uint8_t label,
                    float learning_rate, float l2, float maximum_delta);
/* One PA-II decision-margin update, with fixed bias and inference threshold.
 * dense_delta has buckets entries. Features must be unique/sorted and L2 unit.
 * Validates before mutation; callers own norm projection and quality reports. */
int pota_pa2_sample(const pota_feature *features, size_t count,
                    const int8_t *weights, size_t buckets, float scale, float bias,
                    float *dense_delta, float bias_delta, float decision_logit,
                    uint8_t label, float aggressiveness, float margin, float maximum_delta);
/* Pull a personal logit change toward zero on an L2-normalized replay example.
 * Does not change base weights or bias. Validates all inputs before mutation. */
int pota_anchor_sample(const pota_feature *features, size_t count,
                       float *dense_delta, size_t buckets, float rate);
#ifdef __cplusplus
}
#endif
#endif
