#include "sqlite-vec.h"

#include <assert.h>
#include <errno.h>
#include <float.h>
#include <inttypes.h>
#include <limits.h>
#include <math.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#ifndef SQLITE_CORE
#include "sqlite3ext.h"
SQLITE_EXTENSION_INIT1
#else
#include "sqlite3.h"
#endif

#ifndef SQLITE_VEC_ENABLE_DISKANN
#define SQLITE_VEC_ENABLE_DISKANN 1
#endif

typedef int8_t i8;
typedef uint8_t u8;
typedef int16_t i16;
typedef int32_t i32;
typedef sqlite3_int64 i64;
typedef uint32_t u32;
typedef uint64_t u64;
typedef float f32;
typedef size_t usize;

#ifndef UNUSED_PARAMETER
#define UNUSED_PARAMETER(X) (void)(X)
#endif

// sqlite3_vtab_in() was added in SQLite version 3.38 (2022-02-22)
// https://www.sqlite.org/changes.html#version_3_38_0
#if SQLITE_VERSION_NUMBER >= 3038000
#define COMPILER_SUPPORTS_VTAB_IN 1
#endif

#ifndef SQLITE_VEC_EXPERIMENTAL_IVF_ENABLE
#define SQLITE_VEC_EXPERIMENTAL_IVF_ENABLE 0
#endif

#ifndef SQLITE_VEC_ENABLE_HNSW
#define SQLITE_VEC_ENABLE_HNSW 1
#endif

#ifndef SQLITE_SUBTYPE
#define SQLITE_SUBTYPE 0x000100000
#endif

#ifndef SQLITE_RESULT_SUBTYPE
#define SQLITE_RESULT_SUBTYPE 0x001000000
#endif

#ifndef SQLITE_INDEX_CONSTRAINT_LIMIT
#define SQLITE_INDEX_CONSTRAINT_LIMIT 73
#endif

#ifndef SQLITE_INDEX_CONSTRAINT_OFFSET
#define SQLITE_INDEX_CONSTRAINT_OFFSET 74
#endif

#define countof(x) (sizeof(x) / sizeof((x)[0]))
#define min(a, b) (((a) <= (b)) ? (a) : (b))

// Locale-independent strtod for parsing JSON floats.
// strtod(3) respects LC_NUMERIC, so "1.5" fails in French/German locales
// where the decimal separator is a comma. This parser always uses '.'.
static double strtod_c(const char *str, char **endptr) {
  const char *p = str;
  double result = 0.0;
  int sign = 1;
  int has_digits = 0;

  while (*p == ' ' || *p == '\t' || *p == '\n' || *p == '\r') {
    p++;
  }

  if (*p == '-') {
    sign = -1;
    p++;
  } else if (*p == '+') {
    p++;
  }

  while (*p >= '0' && *p <= '9') {
    result = result * 10.0 + (*p - '0');
    p++;
    has_digits = 1;
  }

  if (*p == '.') {
    double fraction = 0.0;
    double divisor = 1.0;
    p++;
    while (*p >= '0' && *p <= '9') {
      fraction = fraction * 10.0 + (*p - '0');
      divisor *= 10.0;
      p++;
      has_digits = 1;
    }
    result += fraction / divisor;
  }

  if ((*p == 'e' || *p == 'E') && has_digits) {
    int exp_sign = 1;
    int exponent = 0;
    p++;
    if (*p == '-') {
      exp_sign = -1;
      p++;
    } else if (*p == '+') {
      p++;
    }
    while (*p >= '0' && *p <= '9') {
      exponent = exponent * 10 + (*p - '0');
      p++;
    }
    if (exponent > 0) {
      double exp_mult = pow(10.0, (double)exponent);
      if (exp_sign == 1) {
        result *= exp_mult;
      } else {
        result /= exp_mult;
      }
    }
  }

  if (endptr) {
    *endptr = (char *)(has_digits ? p : str);
  }

  if (result == HUGE_VAL || result == -HUGE_VAL) {
    errno = ERANGE;
  }

  return sign * result;
}

#ifndef SQLITE_VEC_ENABLE_RESCORE
#define SQLITE_VEC_ENABLE_RESCORE 1
#endif

enum VectorElementType {
  // clang-format off
  SQLITE_VEC_ELEMENT_TYPE_FLOAT32 = 223 + 0,
  SQLITE_VEC_ELEMENT_TYPE_BIT     = 223 + 1,
  SQLITE_VEC_ELEMENT_TYPE_INT8    = 223 + 2,
  // clang-format on
};

#ifdef SQLITE_VEC_ENABLE_AVX
#include <immintrin.h>
#define PORTABLE_ALIGN32 __attribute__((aligned(32)))
#define PORTABLE_ALIGN64 __attribute__((aligned(64)))

static f32 l2_sqr_float_avx(const void *pVect1v, const void *pVect2v,
                            const void *qty_ptr) {
  f32 *pVect1 = (f32 *)pVect1v;
  f32 *pVect2 = (f32 *)pVect2v;
  size_t qty = *((size_t *)qty_ptr);
  f32 PORTABLE_ALIGN32 TmpRes[8];
  size_t qty16 = qty >> 4;

  const f32 *pEnd1 = pVect1 + (qty16 << 4);

  __m256 diff, v1, v2;
  __m256 sum = _mm256_set1_ps(0);

  while (pVect1 < pEnd1) {
    v1 = _mm256_loadu_ps(pVect1);
    pVect1 += 8;
    v2 = _mm256_loadu_ps(pVect2);
    pVect2 += 8;
    diff = _mm256_sub_ps(v1, v2);
    sum = _mm256_add_ps(sum, _mm256_mul_ps(diff, diff));

    v1 = _mm256_loadu_ps(pVect1);
    pVect1 += 8;
    v2 = _mm256_loadu_ps(pVect2);
    pVect2 += 8;
    diff = _mm256_sub_ps(v1, v2);
    sum = _mm256_add_ps(sum, _mm256_mul_ps(diff, diff));
  }

  _mm256_store_ps(TmpRes, sum);
  return sqrt(TmpRes[0] + TmpRes[1] + TmpRes[2] + TmpRes[3] + TmpRes[4] +
              TmpRes[5] + TmpRes[6] + TmpRes[7]);
}
#endif

#ifdef SQLITE_VEC_ENABLE_NEON
#include <arm_neon.h>

#define PORTABLE_ALIGN32 __attribute__((aligned(32)))

// thx https://github.com/nmslib/hnswlib/pull/299/files
static f32 l2_sqr_float_neon(const void *pVect1v, const void *pVect2v,
                             const void *qty_ptr) {
  f32 *pVect1 = (f32 *)pVect1v;
  f32 *pVect2 = (f32 *)pVect2v;
  size_t qty = *((size_t *)qty_ptr);
  size_t qty16 = qty >> 4;

  const f32 *pEnd1 = pVect1 + (qty16 << 4);

  float32x4_t diff, v1, v2;
  float32x4_t sum0 = vdupq_n_f32(0);
  float32x4_t sum1 = vdupq_n_f32(0);
  float32x4_t sum2 = vdupq_n_f32(0);
  float32x4_t sum3 = vdupq_n_f32(0);

  while (pVect1 < pEnd1) {
    v1 = vld1q_f32(pVect1);
    pVect1 += 4;
    v2 = vld1q_f32(pVect2);
    pVect2 += 4;
    diff = vsubq_f32(v1, v2);
    sum0 = vfmaq_f32(sum0, diff, diff);

    v1 = vld1q_f32(pVect1);
    pVect1 += 4;
    v2 = vld1q_f32(pVect2);
    pVect2 += 4;
    diff = vsubq_f32(v1, v2);
    sum1 = vfmaq_f32(sum1, diff, diff);

    v1 = vld1q_f32(pVect1);
    pVect1 += 4;
    v2 = vld1q_f32(pVect2);
    pVect2 += 4;
    diff = vsubq_f32(v1, v2);
    sum2 = vfmaq_f32(sum2, diff, diff);

    v1 = vld1q_f32(pVect1);
    pVect1 += 4;
    v2 = vld1q_f32(pVect2);
    pVect2 += 4;
    diff = vsubq_f32(v1, v2);
    sum3 = vfmaq_f32(sum3, diff, diff);
  }

  f32 sum_scalar =
      vaddvq_f32(vaddq_f32(vaddq_f32(sum0, sum1), vaddq_f32(sum2, sum3)));
  const f32 *pEnd2 = pVect1 + (qty - (qty16 << 4));
  while (pVect1 < pEnd2) {
    f32 diff = *pVect1 - *pVect2;
    sum_scalar += diff * diff;
    pVect1++;
    pVect2++;
  }

  return sqrt(sum_scalar);
}

// Sums are kept in double, as distance_cosine_float() keeps them: each
// float32x4 load is widened to two float64x2 halves before it is multiplied
// and added.
static f32 cosine_float_neon(const void *pVect1v, const void *pVect2v,
                              const void *qty_ptr) {
  f32 *pVect1 = (f32 *)pVect1v;
  f32 *pVect2 = (f32 *)pVect2v;
  size_t qty = *((size_t *)qty_ptr);
  size_t qty16 = qty >> 4;
  const f32 *pEnd1 = pVect1 + (qty16 << 4);

  float64x2_t dot0 = vdupq_n_f64(0), dot1 = vdupq_n_f64(0);
  float64x2_t dot2 = vdupq_n_f64(0), dot3 = vdupq_n_f64(0);
  float64x2_t amag0 = vdupq_n_f64(0), amag1 = vdupq_n_f64(0);
  float64x2_t amag2 = vdupq_n_f64(0), amag3 = vdupq_n_f64(0);
  float64x2_t bmag0 = vdupq_n_f64(0), bmag1 = vdupq_n_f64(0);
  float64x2_t bmag2 = vdupq_n_f64(0), bmag3 = vdupq_n_f64(0);

  while (pVect1 < pEnd1) {
    float32x4_t v1, v2;
    float64x2_t a, b;

    v1 = vld1q_f32(pVect1); pVect1 += 4;
    v2 = vld1q_f32(pVect2); pVect2 += 4;
    a = vcvt_f64_f32(vget_low_f32(v1));
    b = vcvt_f64_f32(vget_low_f32(v2));
    dot0 = vfmaq_f64(dot0, a, b);
    amag0 = vfmaq_f64(amag0, a, a);
    bmag0 = vfmaq_f64(bmag0, b, b);
    a = vcvt_high_f64_f32(v1);
    b = vcvt_high_f64_f32(v2);
    dot0 = vfmaq_f64(dot0, a, b);
    amag0 = vfmaq_f64(amag0, a, a);
    bmag0 = vfmaq_f64(bmag0, b, b);

    v1 = vld1q_f32(pVect1); pVect1 += 4;
    v2 = vld1q_f32(pVect2); pVect2 += 4;
    a = vcvt_f64_f32(vget_low_f32(v1));
    b = vcvt_f64_f32(vget_low_f32(v2));
    dot1 = vfmaq_f64(dot1, a, b);
    amag1 = vfmaq_f64(amag1, a, a);
    bmag1 = vfmaq_f64(bmag1, b, b);
    a = vcvt_high_f64_f32(v1);
    b = vcvt_high_f64_f32(v2);
    dot1 = vfmaq_f64(dot1, a, b);
    amag1 = vfmaq_f64(amag1, a, a);
    bmag1 = vfmaq_f64(bmag1, b, b);

    v1 = vld1q_f32(pVect1); pVect1 += 4;
    v2 = vld1q_f32(pVect2); pVect2 += 4;
    a = vcvt_f64_f32(vget_low_f32(v1));
    b = vcvt_f64_f32(vget_low_f32(v2));
    dot2 = vfmaq_f64(dot2, a, b);
    amag2 = vfmaq_f64(amag2, a, a);
    bmag2 = vfmaq_f64(bmag2, b, b);
    a = vcvt_high_f64_f32(v1);
    b = vcvt_high_f64_f32(v2);
    dot2 = vfmaq_f64(dot2, a, b);
    amag2 = vfmaq_f64(amag2, a, a);
    bmag2 = vfmaq_f64(bmag2, b, b);

    v1 = vld1q_f32(pVect1); pVect1 += 4;
    v2 = vld1q_f32(pVect2); pVect2 += 4;
    a = vcvt_f64_f32(vget_low_f32(v1));
    b = vcvt_f64_f32(vget_low_f32(v2));
    dot3 = vfmaq_f64(dot3, a, b);
    amag3 = vfmaq_f64(amag3, a, a);
    bmag3 = vfmaq_f64(bmag3, b, b);
    a = vcvt_high_f64_f32(v1);
    b = vcvt_high_f64_f32(v2);
    dot3 = vfmaq_f64(dot3, a, b);
    amag3 = vfmaq_f64(amag3, a, a);
    bmag3 = vfmaq_f64(bmag3, b, b);
  }

  double dot_s = vaddvq_f64(vaddq_f64(vaddq_f64(dot0, dot1), vaddq_f64(dot2, dot3)));
  double amag_s = vaddvq_f64(vaddq_f64(vaddq_f64(amag0, amag1), vaddq_f64(amag2, amag3)));
  double bmag_s = vaddvq_f64(vaddq_f64(vaddq_f64(bmag0, bmag1), vaddq_f64(bmag2, bmag3)));

  const f32 *pEnd2 = pVect1 + (qty - (qty16 << 4));
  while (pVect1 < pEnd2) {
    dot_s += (double)*pVect1 * (double)*pVect2;
    amag_s += (double)*pVect1 * (double)*pVect1;
    bmag_s += (double)*pVect2 * (double)*pVect2;
    pVect1++; pVect2++;
  }

  return 1 - (dot_s / (sqrt(amag_s) * sqrt(bmag_s)));
}

static f32 l2_sqr_int8_neon(const void *pVect1v, const void *pVect2v,
                            const void *qty_ptr) {
  i8 *pVect1 = (i8 *)pVect1v;
  i8 *pVect2 = (i8 *)pVect2v;
  size_t qty = *((size_t *)qty_ptr);

  const i8 *pEnd1 = pVect1 + qty;
  i32 sum_scalar = 0;

  while (pVect1 < pEnd1 - 7) {
    // loading 8 at a time
    int8x8_t v1 = vld1_s8(pVect1);
    int8x8_t v2 = vld1_s8(pVect2);
    pVect1 += 8;
    pVect2 += 8;

    // widen i8 to i16 for subtraction
    int16x8_t v1_wide = vmovl_s8(v1);
    int16x8_t v2_wide = vmovl_s8(v2);
    int16x8_t diff = vsubq_s16(v1_wide, v2_wide);

    // widening multiply: i16*i16 -> i32 to avoid i16 overflow
    // (diff can be up to 255, so diff*diff can be up to 65025 > INT16_MAX)
    int32x4_t sq_lo = vmull_s16(vget_low_s16(diff), vget_low_s16(diff));
    int32x4_t sq_hi = vmull_s16(vget_high_s16(diff), vget_high_s16(diff));
    int32x4_t sum = vaddq_s32(sq_lo, sq_hi);

    sum_scalar += vgetq_lane_s32(sum, 0) + vgetq_lane_s32(sum, 1) +
                  vgetq_lane_s32(sum, 2) + vgetq_lane_s32(sum, 3);
  }

  // handle leftovers
  while (pVect1 < pEnd1) {
    i16 diff = (i16)*pVect1 - (i16)*pVect2;
    sum_scalar += diff * diff;
    pVect1++;
    pVect2++;
  }

  return sqrtf(sum_scalar);
}

static i32 l1_int8_neon(const void *pVect1v, const void *pVect2v,
                        const void *qty_ptr) {
  i8 *pVect1 = (i8 *)pVect1v;
  i8 *pVect2 = (i8 *)pVect2v;
  size_t qty = *((size_t *)qty_ptr);

  const int8_t *pEnd1 = pVect1 + qty;

  uint32x4_t acc1 = vdupq_n_u32(0);
  uint32x4_t acc2 = vdupq_n_u32(0);
  uint32x4_t acc3 = vdupq_n_u32(0);
  uint32x4_t acc4 = vdupq_n_u32(0);

  while (pVect1 < pEnd1 - 63) {
    int8x16_t v1 = vld1q_s8(pVect1);
    int8x16_t v2 = vld1q_s8(pVect2);
    // Absolute differences can reach 255, so widen them as unsigned bytes.
    uint8x16_t diff1 = vreinterpretq_u8_s8(vabdq_s8(v1, v2));
    acc1 = vaddq_u32(acc1, vpaddlq_u16(vpaddlq_u8(diff1)));

    v1 = vld1q_s8(pVect1 + 16);
    v2 = vld1q_s8(pVect2 + 16);
    uint8x16_t diff2 = vreinterpretq_u8_s8(vabdq_s8(v1, v2));
    acc2 = vaddq_u32(acc2, vpaddlq_u16(vpaddlq_u8(diff2)));

    v1 = vld1q_s8(pVect1 + 32);
    v2 = vld1q_s8(pVect2 + 32);
    uint8x16_t diff3 = vreinterpretq_u8_s8(vabdq_s8(v1, v2));
    acc3 = vaddq_u32(acc3, vpaddlq_u16(vpaddlq_u8(diff3)));

    v1 = vld1q_s8(pVect1 + 48);
    v2 = vld1q_s8(pVect2 + 48);
    uint8x16_t diff4 = vreinterpretq_u8_s8(vabdq_s8(v1, v2));
    acc4 = vaddq_u32(acc4, vpaddlq_u16(vpaddlq_u8(diff4)));

    pVect1 += 64;
    pVect2 += 64;
  }

  while (pVect1 < pEnd1 - 15) {
    int8x16_t v1 = vld1q_s8(pVect1);
    int8x16_t v2 = vld1q_s8(pVect2);
    uint8x16_t diff = vreinterpretq_u8_s8(vabdq_s8(v1, v2));
    acc1 = vaddq_u32(acc1, vpaddlq_u16(vpaddlq_u8(diff)));
    pVect1 += 16;
    pVect2 += 16;
  }

  uint32x4_t acc = vaddq_u32(vaddq_u32(acc1, acc2), vaddq_u32(acc3, acc4));

  int32_t sum = 0;
  while (pVect1 < pEnd1) {
    int32_t diff = abs((int32_t)*pVect1 - (int32_t)*pVect2);
    sum += diff;
    pVect1++;
    pVect2++;
  }

  return vaddvq_u32(acc) + sum;
}

static double l1_f32_neon(const void *pVect1v, const void *pVect2v,
                          const void *qty_ptr) {
  f32 *pVect1 = (f32 *)pVect1v;
  f32 *pVect2 = (f32 *)pVect2v;
  size_t qty = *((size_t *)qty_ptr);

  const f32 *pEnd1 = pVect1 + qty;
  float64x2_t acc = vdupq_n_f64(0);

  while (pVect1 < pEnd1 - 3) {
    float32x4_t v1 = vld1q_f32(pVect1);
    float32x4_t v2 = vld1q_f32(pVect2);
    pVect1 += 4;
    pVect2 += 4;

    // f32x4 -> f64x2 pad for overflow
    float64x2_t low_diff = vabdq_f64(vcvt_f64_f32(vget_low_f32(v1)),
                                     vcvt_f64_f32(vget_low_f32(v2)));
    float64x2_t high_diff =
        vabdq_f64(vcvt_high_f64_f32(v1), vcvt_high_f64_f32(v2));

    acc = vaddq_f64(acc, vaddq_f64(low_diff, high_diff));
  }

  double sum = 0;
  while (pVect1 < pEnd1) {
    sum += fabs((double)*pVect1 - (double)*pVect2);
    pVect1++;
    pVect2++;
  }

  return vaddvq_f64(acc) + sum;
}
#endif

static f32 l2_sqr_float(const void *pVect1v, const void *pVect2v,
                        const void *qty_ptr) {
  f32 *pVect1 = (f32 *)pVect1v;
  f32 *pVect2 = (f32 *)pVect2v;
  size_t qty = *((size_t *)qty_ptr);

  f32 res = 0;
  for (size_t i = 0; i < qty; i++) {
    f32 t = *pVect1 - *pVect2;
    pVect1++;
    pVect2++;
    res += t * t;
  }
  return sqrt(res);
}

static f32 l2_sqr_int8(const void *pA, const void *pB, const void *pD) {
  i8 *a = (i8 *)pA;
  i8 *b = (i8 *)pB;
  size_t d = *((size_t *)pD);

  f32 res = 0;
  for (size_t i = 0; i < d; i++) {
    f32 t = *a - *b;
    a++;
    b++;
    res += t * t;
  }
  return sqrt(res);
}

static f32 distance_l2_sqr_float(const void *a, const void *b, const void *d) {
#ifdef SQLITE_VEC_ENABLE_NEON
  if ((*(const size_t *)d) > 16) {
    return l2_sqr_float_neon(a, b, d);
  }
#endif
#ifdef SQLITE_VEC_ENABLE_AVX
  if (((*(const size_t *)d) % 16 == 0)) {
    return l2_sqr_float_avx(a, b, d);
  }
#endif
  return l2_sqr_float(a, b, d);
}

static f32 distance_l2_sqr_int8(const void *a, const void *b, const void *d) {
#ifdef SQLITE_VEC_ENABLE_NEON
  if ((*(const size_t *)d) > 7) {
    return l2_sqr_int8_neon(a, b, d);
  }
#endif
  return l2_sqr_int8(a, b, d);
}

static i32 l1_int8(const void *pA, const void *pB, const void *pD) {
  i8 *a = (i8 *)pA;
  i8 *b = (i8 *)pB;
  size_t d = *((size_t *)pD);

  i32 res = 0;
  for (size_t i = 0; i < d; i++) {
    res += abs(*a - *b);
    a++;
    b++;
  }

  return res;
}

static i32 distance_l1_int8(const void *a, const void *b, const void *d) {
#ifdef SQLITE_VEC_ENABLE_NEON
  if ((*(const size_t *)d) > 15) {
    return l1_int8_neon(a, b, d);
  }
#endif
  return l1_int8(a, b, d);
}

static double l1_f32(const void *pA, const void *pB, const void *pD) {
  f32 *a = (f32 *)pA;
  f32 *b = (f32 *)pB;
  size_t d = *((size_t *)pD);

  double res = 0;
  for (size_t i = 0; i < d; i++) {
    res += fabs((double)*a - (double)*b);
    a++;
    b++;
  }

  return res;
}

static double distance_l1_f32(const void *a, const void *b, const void *d) {
#ifdef SQLITE_VEC_ENABLE_NEON
  if ((*(const size_t *)d) > 3) {
    return l1_f32_neon(a, b, d);
  }
#endif
  return l1_f32(a, b, d);
}

#ifdef SQLITE_VEC_ENABLE_NEON
/**
 * Single-precision NEON cosine. Fast, but squares of very small (or large)
 * elements underflow (or overflow) f32. Returns 0 and leaves *out untouched
 * when that may have happened, so the caller can fall back to the double
 * kernel; returns 1 otherwise.
 */
static int cosine_float_neon_f32(const f32 *a, const f32 *b, size_t n, f32 *out) {
  float32x4_t dot0 = vdupq_n_f32(0), dot1 = vdupq_n_f32(0);
  float32x4_t aa0 = vdupq_n_f32(0), aa1 = vdupq_n_f32(0);
  float32x4_t bb0 = vdupq_n_f32(0), bb1 = vdupq_n_f32(0);
  size_t i = 0;
  for (; i + 8 <= n; i += 8) {
    float32x4_t x0 = vld1q_f32(a + i), y0 = vld1q_f32(b + i);
    float32x4_t x1 = vld1q_f32(a + i + 4), y1 = vld1q_f32(b + i + 4);
    dot0 = vfmaq_f32(dot0, x0, y0);
    dot1 = vfmaq_f32(dot1, x1, y1);
    aa0 = vfmaq_f32(aa0, x0, x0);
    aa1 = vfmaq_f32(aa1, x1, x1);
    bb0 = vfmaq_f32(bb0, y0, y0);
    bb1 = vfmaq_f32(bb1, y1, y1);
  }
  f32 dot = vaddvq_f32(vaddq_f32(dot0, dot1));
  f32 aa = vaddvq_f32(vaddq_f32(aa0, aa1));
  f32 bb = vaddvq_f32(vaddq_f32(bb0, bb1));
  for (; i < n; i++) {
    dot += a[i] * b[i];
    aa += a[i] * a[i];
    bb += b[i] * b[i];
  }
  // aa/bb >= 1e-30 rules out underflow that matters; finiteness rules out
  // overflow. Anything else goes to the exact double kernel.
  if (!(aa >= 1e-30f && bb >= 1e-30f && isfinite(aa) && isfinite(bb) && isfinite(dot)))
    return 0;
  *out = (f32)(1.0 - (double)dot / (sqrt((double)aa) * sqrt((double)bb)));
  return 1;
}
#endif

static f32 distance_cosine_float(const void *pVect1v, const void *pVect2v,
                                 const void *qty_ptr) {
#ifdef SQLITE_VEC_ENABLE_NEON
  if ((*(const size_t *)qty_ptr) > 16) {
    f32 d;
    if (cosine_float_neon_f32(pVect1v, pVect2v, *(const size_t *)qty_ptr, &d)) return d;
    return cosine_float_neon(pVect1v, pVect2v, qty_ptr);
  }
#endif
  f32 *pVect1 = (f32 *)pVect1v;
  f32 *pVect2 = (f32 *)pVect2v;
  size_t qty = *((size_t *)qty_ptr);

  // Accumulate in double: the square of a small (or large) f32 element
  // underflows (or overflows) f32, which made the magnitude 0 (or Inf) and
  // the distance +-Inf, NaN or plain wrong for nonzero finite vectors. Every
  // f32 product is representable in double.
  double dot = 0;
  double aMag = 0;
  double bMag = 0;
  for (size_t i = 0; i < qty; i++) {
    dot += (double)*pVect1 * (double)*pVect2;
    aMag += (double)*pVect1 * (double)*pVect1;
    bMag += (double)*pVect2 * (double)*pVect2;
    pVect1++;
    pVect2++;
  }
  return 1 - (dot / (sqrt(aMag) * sqrt(bMag)));
}
static f32 cosine_int8(const void *pA, const void *pB, const void *pD) {
  i8 *a = (i8 *)pA;
  i8 *b = (i8 *)pB;
  size_t d = *((size_t *)pD);

  f32 dot = 0;
  f32 aMag = 0;
  f32 bMag = 0;
  for (size_t i = 0; i < d; i++) {
    dot += *a * *b;
    aMag += *a * *a;
    bMag += *b * *b;
    a++;
    b++;
  }
  return 1 - (dot / (sqrt(aMag) * sqrt(bMag)));
}

#ifdef SQLITE_VEC_ENABLE_NEON
// Products of int8 elements are summed exactly: a 32-bit lane holds one pass's
// products, but 131,072 squares of -128 overflow a 32-bit sum, so each pass's
// lanes are added into 64-bit lanes.
static f32 cosine_int8_neon(const void *pA, const void *pB, const void *pD) {
  const i8 *a = (const i8 *)pA;
  const i8 *b = (const i8 *)pB;
  size_t d = *((const size_t *)pD);
  const i8 *aEnd = a + d;

  int64x2_t dot_acc = vdupq_n_s64(0);
  int64x2_t aMag_acc = vdupq_n_s64(0);
  int64x2_t bMag_acc = vdupq_n_s64(0);

  while (a < aEnd - 31) {
    int32x4_t dot_acc1 = vdupq_n_s32(0);
    int32x4_t dot_acc2 = vdupq_n_s32(0);
    int32x4_t aMag_acc1 = vdupq_n_s32(0);
    int32x4_t aMag_acc2 = vdupq_n_s32(0);
    int32x4_t bMag_acc1 = vdupq_n_s32(0);
    int32x4_t bMag_acc2 = vdupq_n_s32(0);
    int8x16_t va1 = vld1q_s8(a);
    int8x16_t vb1 = vld1q_s8(b);
    int16x8_t a1_lo = vmovl_s8(vget_low_s8(va1));
    int16x8_t a1_hi = vmovl_s8(vget_high_s8(va1));
    int16x8_t b1_lo = vmovl_s8(vget_low_s8(vb1));
    int16x8_t b1_hi = vmovl_s8(vget_high_s8(vb1));

    dot_acc1 = vmlal_s16(dot_acc1, vget_low_s16(a1_lo), vget_low_s16(b1_lo));
    dot_acc1 = vmlal_s16(dot_acc1, vget_high_s16(a1_lo), vget_high_s16(b1_lo));
    dot_acc2 = vmlal_s16(dot_acc2, vget_low_s16(a1_hi), vget_low_s16(b1_hi));
    dot_acc2 = vmlal_s16(dot_acc2, vget_high_s16(a1_hi), vget_high_s16(b1_hi));

    aMag_acc1 = vmlal_s16(aMag_acc1, vget_low_s16(a1_lo), vget_low_s16(a1_lo));
    aMag_acc1 = vmlal_s16(aMag_acc1, vget_high_s16(a1_lo), vget_high_s16(a1_lo));
    aMag_acc2 = vmlal_s16(aMag_acc2, vget_low_s16(a1_hi), vget_low_s16(a1_hi));
    aMag_acc2 = vmlal_s16(aMag_acc2, vget_high_s16(a1_hi), vget_high_s16(a1_hi));

    bMag_acc1 = vmlal_s16(bMag_acc1, vget_low_s16(b1_lo), vget_low_s16(b1_lo));
    bMag_acc1 = vmlal_s16(bMag_acc1, vget_high_s16(b1_lo), vget_high_s16(b1_lo));
    bMag_acc2 = vmlal_s16(bMag_acc2, vget_low_s16(b1_hi), vget_low_s16(b1_hi));
    bMag_acc2 = vmlal_s16(bMag_acc2, vget_high_s16(b1_hi), vget_high_s16(b1_hi));

    int8x16_t va2 = vld1q_s8(a + 16);
    int8x16_t vb2 = vld1q_s8(b + 16);
    int16x8_t a2_lo = vmovl_s8(vget_low_s8(va2));
    int16x8_t a2_hi = vmovl_s8(vget_high_s8(va2));
    int16x8_t b2_lo = vmovl_s8(vget_low_s8(vb2));
    int16x8_t b2_hi = vmovl_s8(vget_high_s8(vb2));

    dot_acc1 = vmlal_s16(dot_acc1, vget_low_s16(a2_lo), vget_low_s16(b2_lo));
    dot_acc1 = vmlal_s16(dot_acc1, vget_high_s16(a2_lo), vget_high_s16(b2_lo));
    dot_acc2 = vmlal_s16(dot_acc2, vget_low_s16(a2_hi), vget_low_s16(b2_hi));
    dot_acc2 = vmlal_s16(dot_acc2, vget_high_s16(a2_hi), vget_high_s16(b2_hi));

    aMag_acc1 = vmlal_s16(aMag_acc1, vget_low_s16(a2_lo), vget_low_s16(a2_lo));
    aMag_acc1 = vmlal_s16(aMag_acc1, vget_high_s16(a2_lo), vget_high_s16(a2_lo));
    aMag_acc2 = vmlal_s16(aMag_acc2, vget_low_s16(a2_hi), vget_low_s16(a2_hi));
    aMag_acc2 = vmlal_s16(aMag_acc2, vget_high_s16(a2_hi), vget_high_s16(a2_hi));

    bMag_acc1 = vmlal_s16(bMag_acc1, vget_low_s16(b2_lo), vget_low_s16(b2_lo));
    bMag_acc1 = vmlal_s16(bMag_acc1, vget_high_s16(b2_lo), vget_high_s16(b2_lo));
    bMag_acc2 = vmlal_s16(bMag_acc2, vget_low_s16(b2_hi), vget_low_s16(b2_hi));
    bMag_acc2 = vmlal_s16(bMag_acc2, vget_high_s16(b2_hi), vget_high_s16(b2_hi));

    dot_acc = vpadalq_s32(dot_acc, vaddq_s32(dot_acc1, dot_acc2));
    aMag_acc = vpadalq_s32(aMag_acc, vaddq_s32(aMag_acc1, aMag_acc2));
    bMag_acc = vpadalq_s32(bMag_acc, vaddq_s32(bMag_acc1, bMag_acc2));
    a += 32;
    b += 32;
  }

  while (a < aEnd - 15) {
    int32x4_t dot_acc1 = vdupq_n_s32(0);
    int32x4_t aMag_acc1 = vdupq_n_s32(0);
    int32x4_t bMag_acc1 = vdupq_n_s32(0);
    int8x16_t va = vld1q_s8(a);
    int8x16_t vb = vld1q_s8(b);
    int16x8_t a_lo = vmovl_s8(vget_low_s8(va));
    int16x8_t a_hi = vmovl_s8(vget_high_s8(va));
    int16x8_t b_lo = vmovl_s8(vget_low_s8(vb));
    int16x8_t b_hi = vmovl_s8(vget_high_s8(vb));

    dot_acc1 = vmlal_s16(dot_acc1, vget_low_s16(a_lo), vget_low_s16(b_lo));
    dot_acc1 = vmlal_s16(dot_acc1, vget_high_s16(a_lo), vget_high_s16(b_lo));
    dot_acc1 = vmlal_s16(dot_acc1, vget_low_s16(a_hi), vget_low_s16(b_hi));
    dot_acc1 = vmlal_s16(dot_acc1, vget_high_s16(a_hi), vget_high_s16(b_hi));

    aMag_acc1 = vmlal_s16(aMag_acc1, vget_low_s16(a_lo), vget_low_s16(a_lo));
    aMag_acc1 = vmlal_s16(aMag_acc1, vget_high_s16(a_lo), vget_high_s16(a_lo));
    aMag_acc1 = vmlal_s16(aMag_acc1, vget_low_s16(a_hi), vget_low_s16(a_hi));
    aMag_acc1 = vmlal_s16(aMag_acc1, vget_high_s16(a_hi), vget_high_s16(a_hi));

    bMag_acc1 = vmlal_s16(bMag_acc1, vget_low_s16(b_lo), vget_low_s16(b_lo));
    bMag_acc1 = vmlal_s16(bMag_acc1, vget_high_s16(b_lo), vget_high_s16(b_lo));
    bMag_acc1 = vmlal_s16(bMag_acc1, vget_low_s16(b_hi), vget_low_s16(b_hi));
    bMag_acc1 = vmlal_s16(bMag_acc1, vget_high_s16(b_hi), vget_high_s16(b_hi));

    dot_acc = vpadalq_s32(dot_acc, dot_acc1);
    aMag_acc = vpadalq_s32(aMag_acc, aMag_acc1);
    bMag_acc = vpadalq_s32(bMag_acc, bMag_acc1);
    a += 16;
    b += 16;
  }

  i64 dot = vaddvq_s64(dot_acc);
  i64 aMag = vaddvq_s64(aMag_acc);
  i64 bMag = vaddvq_s64(bMag_acc);

  while (a < aEnd) {
    dot += (i32)*a * (i32)*b;
    aMag += (i32)*a * (i32)*a;
    bMag += (i32)*b * (i32)*b;
    a++;
    b++;
  }

  return 1.0f - ((f32)dot / (sqrtf((f32)aMag) * sqrtf((f32)bMag)));
}
#endif

static f32 distance_cosine_int8(const void *a, const void *b, const void *d) {
#ifdef SQLITE_VEC_ENABLE_NEON
  if ((*(const size_t *)d) > 15) {
    return cosine_int8_neon(a, b, d);
  }
#endif
  return cosine_int8(a, b, d);
}

// https://github.com/facebookresearch/faiss/blob/77e2e79cd0a680adc343b9840dd865da724c579e/faiss/utils/hamming_distance/common.h#L34
static u8 hamdist_table[256] = {
    0, 1, 1, 2, 1, 2, 2, 3, 1, 2, 2, 3, 2, 3, 3, 4, 1, 2, 2, 3, 2, 3, 3, 4,
    2, 3, 3, 4, 3, 4, 4, 5, 1, 2, 2, 3, 2, 3, 3, 4, 2, 3, 3, 4, 3, 4, 4, 5,
    2, 3, 3, 4, 3, 4, 4, 5, 3, 4, 4, 5, 4, 5, 5, 6, 1, 2, 2, 3, 2, 3, 3, 4,
    2, 3, 3, 4, 3, 4, 4, 5, 2, 3, 3, 4, 3, 4, 4, 5, 3, 4, 4, 5, 4, 5, 5, 6,
    2, 3, 3, 4, 3, 4, 4, 5, 3, 4, 4, 5, 4, 5, 5, 6, 3, 4, 4, 5, 4, 5, 5, 6,
    4, 5, 5, 6, 5, 6, 6, 7, 1, 2, 2, 3, 2, 3, 3, 4, 2, 3, 3, 4, 3, 4, 4, 5,
    2, 3, 3, 4, 3, 4, 4, 5, 3, 4, 4, 5, 4, 5, 5, 6, 2, 3, 3, 4, 3, 4, 4, 5,
    3, 4, 4, 5, 4, 5, 5, 6, 3, 4, 4, 5, 4, 5, 5, 6, 4, 5, 5, 6, 5, 6, 6, 7,
    2, 3, 3, 4, 3, 4, 4, 5, 3, 4, 4, 5, 4, 5, 5, 6, 3, 4, 4, 5, 4, 5, 5, 6,
    4, 5, 5, 6, 5, 6, 6, 7, 3, 4, 4, 5, 4, 5, 5, 6, 4, 5, 5, 6, 5, 6, 6, 7,
    4, 5, 5, 6, 5, 6, 6, 7, 5, 6, 6, 7, 6, 7, 7, 8};

#ifdef SQLITE_VEC_ENABLE_NEON
static f32 distance_hamming_neon(const u8 *a, const u8 *b, size_t n_bytes) {
  const u8 *pEnd = a + n_bytes;

  uint32x4_t acc1 = vdupq_n_u32(0);
  uint32x4_t acc2 = vdupq_n_u32(0);
  uint32x4_t acc3 = vdupq_n_u32(0);
  uint32x4_t acc4 = vdupq_n_u32(0);

  while (a <= pEnd - 64) {
    uint8x16_t v1 = vld1q_u8(a);
    uint8x16_t v2 = vld1q_u8(b);
    acc1 = vaddq_u32(acc1, vpaddlq_u16(vpaddlq_u8(vcntq_u8(veorq_u8(v1, v2)))));

    v1 = vld1q_u8(a + 16);
    v2 = vld1q_u8(b + 16);
    acc2 = vaddq_u32(acc2, vpaddlq_u16(vpaddlq_u8(vcntq_u8(veorq_u8(v1, v2)))));

    v1 = vld1q_u8(a + 32);
    v2 = vld1q_u8(b + 32);
    acc3 = vaddq_u32(acc3, vpaddlq_u16(vpaddlq_u8(vcntq_u8(veorq_u8(v1, v2)))));

    v1 = vld1q_u8(a + 48);
    v2 = vld1q_u8(b + 48);
    acc4 = vaddq_u32(acc4, vpaddlq_u16(vpaddlq_u8(vcntq_u8(veorq_u8(v1, v2)))));

    a += 64;
    b += 64;
  }

  while (a <= pEnd - 16) {
    uint8x16_t v1 = vld1q_u8(a);
    uint8x16_t v2 = vld1q_u8(b);
    acc1 = vaddq_u32(acc1, vpaddlq_u16(vpaddlq_u8(vcntq_u8(veorq_u8(v1, v2)))));
    a += 16;
    b += 16;
  }

  acc1 = vaddq_u32(acc1, acc2);
  acc3 = vaddq_u32(acc3, acc4);
  acc1 = vaddq_u32(acc1, acc3);
  u32 sum = vaddvq_u32(acc1);

  while (a < pEnd) {
    sum += hamdist_table[*a ^ *b];
    a++;
    b++;
  }

  return (f32)sum;
}
#endif

#ifdef SQLITE_VEC_ENABLE_AVX
/**
 * AVX2 Hamming distance using VPSHUFB-based popcount.
 * Processes 32 bytes (256 bits) per iteration.
 */
static f32 distance_hamming_avx2(const u8 *a, const u8 *b, size_t n_bytes) {
  const u8 *pEnd = a + n_bytes;

  // VPSHUFB lookup table: popcount of low nibble
  const __m256i lookup = _mm256_setr_epi8(
      0,1,1,2,1,2,2,3,1,2,2,3,2,3,3,4,
      0,1,1,2,1,2,2,3,1,2,2,3,2,3,3,4);
  const __m256i low_mask = _mm256_set1_epi8(0x0f);

  __m256i acc = _mm256_setzero_si256();

  while (a <= pEnd - 32) {
    __m256i va = _mm256_loadu_si256((const __m256i *)a);
    __m256i vb = _mm256_loadu_si256((const __m256i *)b);
    __m256i xored = _mm256_xor_si256(va, vb);

    // VPSHUFB popcount: split into nibbles, lookup each
    __m256i lo = _mm256_and_si256(xored, low_mask);
    __m256i hi = _mm256_and_si256(_mm256_srli_epi16(xored, 4), low_mask);
    __m256i popcnt = _mm256_add_epi8(_mm256_shuffle_epi8(lookup, lo),
                                      _mm256_shuffle_epi8(lookup, hi));

    // Horizontal sum: u8 -> u64 via sad against zero
    acc = _mm256_add_epi64(acc, _mm256_sad_epu8(popcnt, _mm256_setzero_si256()));
    a += 32;
    b += 32;
  }

  // Horizontal sum of 4 x u64 lanes
  u64 tmp[4];
  _mm256_storeu_si256((__m256i *)tmp, acc);
  u32 sum = (u32)(tmp[0] + tmp[1] + tmp[2] + tmp[3]);

  // Scalar tail
  while (a < pEnd) {
    u8 x = *a ^ *b;
    x = x - ((x >> 1) & 0x55);
    x = (x & 0x33) + ((x >> 2) & 0x33);
    sum += (x + (x >> 4)) & 0x0F;
    a++;
    b++;
  }

  return (f32)sum;
}
#endif

static f32 distance_hamming_u8(u8 *a, u8 *b, size_t n) {
  int same = 0;
  for (unsigned long i = 0; i < n; i++) {
    same += hamdist_table[a[i] ^ b[i]];
  }
  return (f32)same;
}

#ifdef _MSC_VER
#if !defined(__clang__) && (defined(_M_ARM) || defined(_M_ARM64))
// From
// https://github.com/ngtcp2/ngtcp2/blob/b64f1e77b5e0d880b93d31f474147fae4a1d17cc/lib/ngtcp2_ringbuf.c,
// line 34-43
static unsigned int __builtin_popcountl(unsigned int x) {
  unsigned int c = 0;
  for (; x; ++c) {
    x &= x - 1;
  }
  return c;
}
#else
#include <intrin.h>
#define __builtin_popcountl __popcnt64
#endif
#endif

static f32 distance_hamming_u64(const u8 *a, const u8 *b, size_t n) {
  int same = 0;
  for (unsigned long i = 0; i < n; i++) {
    u64 va, vb;
    memcpy(&va, a + i * sizeof(u64), sizeof(u64));
    memcpy(&vb, b + i * sizeof(u64), sizeof(u64));
    same += __builtin_popcountl(va ^ vb);
  }
  return (f32)same;
}

/**
 * @brief Calculate the hamming distance between two bitvectors.
 *
 * @param a - first bitvector, MUST have d dimensions
 * @param b - second bitvector, MUST have d dimensions
 * @param d - pointer to size_t, MUST be divisible by CHAR_BIT
 * @return f32
 */
static f32 distance_hamming(const void *a, const void *b, const void *d) {
  size_t dimensions = *((size_t *)d);
  size_t n_bytes = dimensions / CHAR_BIT;

#ifdef SQLITE_VEC_ENABLE_NEON
  if (dimensions >= 128) {
    return distance_hamming_neon((const u8 *)a, (const u8 *)b, n_bytes);
  }
#endif
#ifdef SQLITE_VEC_ENABLE_AVX
  if (n_bytes >= 32) {
    return distance_hamming_avx2((const u8 *)a, (const u8 *)b, n_bytes);
  }
#endif

  if ((dimensions % 64) == 0) {
    return distance_hamming_u64((const u8 *)a, (const u8 *)b, n_bytes / sizeof(u64));
  }
  return distance_hamming_u8((u8 *)a, (u8 *)b, n_bytes);
}

/**
 * Negative inner product, so smaller = more similar like every other metric.
 * Meant for vectors whose magnitude carries meaning (MIPS); for unit vectors
 * it ranks identically to cosine without the two extra norms.
 */
static f32 dot_float(const f32 *a, const f32 *b, size_t d) {
  f32 dot = 0;
  for (size_t i = 0; i < d; i++) dot += a[i] * b[i];
  return dot;
}

#ifdef SQLITE_VEC_ENABLE_NEON
static f32 dot_float_neon(const f32 *a, const f32 *b, size_t d) {
  float32x4_t s0 = vdupq_n_f32(0), s1 = vdupq_n_f32(0);
  float32x4_t s2 = vdupq_n_f32(0), s3 = vdupq_n_f32(0);
  size_t i = 0;
  for (; i + 16 <= d; i += 16) {
    s0 = vfmaq_f32(s0, vld1q_f32(a + i), vld1q_f32(b + i));
    s1 = vfmaq_f32(s1, vld1q_f32(a + i + 4), vld1q_f32(b + i + 4));
    s2 = vfmaq_f32(s2, vld1q_f32(a + i + 8), vld1q_f32(b + i + 8));
    s3 = vfmaq_f32(s3, vld1q_f32(a + i + 12), vld1q_f32(b + i + 12));
  }
  f32 dot = vaddvq_f32(vaddq_f32(vaddq_f32(s0, s1), vaddq_f32(s2, s3)));
  for (; i < d; i++) dot += a[i] * b[i];
  return dot;
}
#endif

static f32 distance_dot_float(const void *a, const void *b, const void *d) {
  size_t dims = *((const size_t *)d);
#ifdef SQLITE_VEC_ENABLE_NEON
  if (dims >= 16) return -dot_float_neon((const f32 *)a, (const f32 *)b, dims);
#endif
  return -dot_float((const f32 *)a, (const f32 *)b, dims);
}

static f32 distance_dot_int8(const void *a, const void *b, const void *d) {
  const i8 *x = (const i8 *)a, *y = (const i8 *)b;
  size_t dims = *((const size_t *)d);
  i32 dot = 0;
  for (size_t i = 0; i < dims; i++) dot += (i32)x[i] * (i32)y[i];
  return -(f32)dot;
}

/**
 * Jaccard distance over set bits: 1 - |a AND b| / |a OR b|. 0 if both empty.
 */
static f32 distance_jaccard(const void *a, const void *b, const void *d) {
  const u8 *x = (const u8 *)a, *y = (const u8 *)b;
  size_t n_bytes = *((const size_t *)d) / CHAR_BIT;
  u32 inter = 0, uni = 0;
  for (size_t i = 0; i < n_bytes; i++) {
    inter += __builtin_popcountl((unsigned long)(x[i] & y[i]));
    uni += __builtin_popcountl((unsigned long)(x[i] | y[i]));
  }
  return uni == 0 ? 0.0f : 1.0f - (f32)inter / (f32)uni;
}

#ifdef SQLITE_VEC_TEST
f32 _test_distance_l2_sqr_float(const f32 *a, const f32 *b, size_t dims) {
  return distance_l2_sqr_float(a, b, &dims);
}
f32 _test_distance_cosine_float(const f32 *a, const f32 *b, size_t dims) {
  return distance_cosine_float(a, b, &dims);
}
f32 _test_distance_hamming(const u8 *a, const u8 *b, size_t dims) {
  return distance_hamming(a, b, &dims);
}
#endif

// from SQLite source:
// https://github.com/sqlite/sqlite/blob/a509a90958ddb234d1785ed7801880ccb18b497e/src/json.c#L153
static const char vecJsonIsSpaceX[] = {
    0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,

    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
};

#define vecJsonIsspace(x) (vecJsonIsSpaceX[(unsigned char)x])

typedef void (*vector_cleanup)(void *p);

void vector_cleanup_noop(void *_) { UNUSED_PARAMETER(_); }

#define JSON_SUBTYPE 74

void vtab_set_error(sqlite3_vtab *pVTab, const char *zFormat, ...) {
  va_list args;
  sqlite3_free(pVTab->zErrMsg);
  va_start(args, zFormat);
  pVTab->zErrMsg = sqlite3_vmprintf(zFormat, args);
  va_end(args);
}
struct Array {
  size_t element_size;
  size_t length;
  size_t capacity;
  void *z;
};

/**
 * @brief Initial an array with the given element size and capacity.
 *
 * @param array
 * @param element_size
 * @param init_capacity
 * @return SQLITE_OK on success, error code on failure. Only error is
 * SQLITE_NOMEM
 */
int array_init(struct Array *array, size_t element_size, size_t init_capacity) {
  int sz = element_size * init_capacity;
  void *z = sqlite3_malloc(sz);
  if (!z) {
    return SQLITE_NOMEM;
  }
  memset(z, 0, sz);

  array->element_size = element_size;
  array->length = 0;
  array->capacity = init_capacity;
  array->z = z;
  return SQLITE_OK;
}

int array_append(struct Array *array, const void *element) {
  if (array->length == array->capacity) {
    size_t new_capacity = array->capacity * 2 + 100;
    void *z = sqlite3_realloc64(array->z, array->element_size * new_capacity);
    if (z) {
      array->capacity = new_capacity;
      array->z = z;
    } else {
      return SQLITE_NOMEM;
    }
  }
  memcpy(&((unsigned char *)array->z)[array->length * array->element_size],
         element, array->element_size);
  array->length++;
  return SQLITE_OK;
}

void array_cleanup(struct Array *array) {
  if (!array)
    return;
  array->element_size = 0;
  array->length = 0;
  array->capacity = 0;
  sqlite3_free(array->z);
  array->z = NULL;
}

char *vector_subtype_name(int subtype) {
  switch (subtype) {
  case SQLITE_VEC_ELEMENT_TYPE_FLOAT32:
    return "float32";
  case SQLITE_VEC_ELEMENT_TYPE_INT8:
    return "int8";
  case SQLITE_VEC_ELEMENT_TYPE_BIT:
    return "bit";
  }
  return "";
}
char *type_name(int type) {
  switch (type) {
  case SQLITE_INTEGER:
    return "INTEGER";
  case SQLITE_BLOB:
    return "BLOB";
  case SQLITE_TEXT:
    return "TEXT";
  case SQLITE_FLOAT:
    return "FLOAT";
  case SQLITE_NULL:
    return "NULL";
  }
  return "";
}

typedef void (*fvec_cleanup)(void *vector);

void fvec_cleanup_noop(void *_) { UNUSED_PARAMETER(_); }

static int fvec_from_value(sqlite3_value *value, f32 **vector,
                           size_t *dimensions, fvec_cleanup *cleanup,
                           char **pzErr) {
  int value_type = sqlite3_value_type(value);

  if (value_type == SQLITE_BLOB) {
    const void *blob = sqlite3_value_blob(value);
    int bytes = sqlite3_value_bytes(value);
    if (bytes == 0) {
      *pzErr = sqlite3_mprintf("zero-length vectors are not supported.");
      return SQLITE_ERROR;
    }
    if ((bytes % sizeof(f32)) != 0) {
      *pzErr = sqlite3_mprintf("invalid float32 vector BLOB length. Must be "
                               "divisible by %d, found %d",
                               sizeof(f32), bytes);
      return SQLITE_ERROR;
    }
    f32 *buf = sqlite3_malloc(bytes);
    if (!buf) {
      *pzErr = sqlite3_mprintf("out of memory");
      return SQLITE_NOMEM;
    }
    memcpy(buf, blob, bytes);
    size_t n = bytes / sizeof(f32);
    for (size_t i = 0; i < n; i++) {
      if (isnan(buf[i]) || isinf(buf[i])) {
        *pzErr = sqlite3_mprintf(
            "invalid float32 vector: element %d is %s",
            (int)i, isnan(buf[i]) ? "NaN" : "Inf");
        sqlite3_free(buf);
        return SQLITE_ERROR;
      }
    }
    *vector = buf;
    *dimensions = n;
    *cleanup = sqlite3_free;
    return SQLITE_OK;
  }

  if (value_type == SQLITE_TEXT) {
    const char *source = (const char *)sqlite3_value_text(value);
    int source_len = sqlite3_value_bytes(value);
    if (source_len == 0) {
      *pzErr = sqlite3_mprintf("zero-length vectors are not supported.");
      return SQLITE_ERROR;
    }
    int i = 0;

    struct Array x;
    int rc = array_init(&x, sizeof(f32), ceil(source_len / 2.0));
    if (rc != SQLITE_OK) {
      return rc;
    }

    // advance leading whitespace to first '['
    while (i < source_len) {
      if (vecJsonIsspace(source[i])) {
        i++;
        continue;
      }
      if (source[i] == '[') {
        break;
      }

      *pzErr = sqlite3_mprintf(
          "JSON array parsing error: Input does not start with '['");
      array_cleanup(&x);
      return SQLITE_ERROR;
    }
    if (source[i] != '[') {
      *pzErr = sqlite3_mprintf(
          "JSON array parsing error: Input does not start with '['");
      array_cleanup(&x);
      return SQLITE_ERROR;
    }
    int offset = i + 1;

    while (offset < source_len) {
      char *ptr = (char *)&source[offset];
      char *endptr;

      errno = 0;
      double result = strtod_c(ptr, &endptr);
      if ((errno != 0 && result == 0) // some interval error?
          || (errno == ERANGE &&
              (result == HUGE_VAL || result == -HUGE_VAL)) // too big / smalls
      ) {
        sqlite3_free(x.z);
        *pzErr = sqlite3_mprintf("JSON parsing error");
        return SQLITE_ERROR;
      }

      if (endptr == ptr) {
        if (*ptr != ']') {
          sqlite3_free(x.z);
          *pzErr = sqlite3_mprintf("JSON parsing error");
          return SQLITE_ERROR;
        }
        goto done;
      }

      f32 res = (f32)result;
      if (isnan(res) || isinf(res)) {
        sqlite3_free(x.z);
        *pzErr = sqlite3_mprintf(
            "invalid float32 vector: element %d is %s",
            (int)x.length, isnan(res) ? "NaN" : "Inf");
        return SQLITE_ERROR;
      }
      array_append(&x, (const void *)&res);

      offset += (endptr - ptr);
      while (offset < source_len) {
        if (vecJsonIsspace(source[offset])) {
          offset++;
          continue;
        }
        if (source[offset] == ',') {
          offset++;
          continue;
        }
        if (source[offset] == ']')
          goto done;
        break;
      }
    }

  done:

    if (x.length > 0) {
      *vector = (f32 *)x.z;
      *dimensions = x.length;
      *cleanup = sqlite3_free;
      return SQLITE_OK;
    }
    sqlite3_free(x.z);
    *pzErr = sqlite3_mprintf("zero-length vectors are not supported.");
    return SQLITE_ERROR;
  }

  *pzErr = sqlite3_mprintf(
      "Input must have type BLOB (compact format) or TEXT (JSON), found %s",
      type_name(value_type));
  return SQLITE_ERROR;
}

static int bitvec_from_value(sqlite3_value *value, u8 **vector,
                             size_t *dimensions, vector_cleanup *cleanup,
                             char **pzErr) {
  int value_type = sqlite3_value_type(value);
  if (value_type == SQLITE_BLOB) {
    const void *blob = sqlite3_value_blob(value);
    int bytes = sqlite3_value_bytes(value);
    if (bytes == 0) {
      *pzErr = sqlite3_mprintf("zero-length vectors are not supported.");
      return SQLITE_ERROR;
    }
    *vector = (u8 *)blob;
    *dimensions = bytes * CHAR_BIT;
    *cleanup = vector_cleanup_noop;
    return SQLITE_OK;
  }
  *pzErr = sqlite3_mprintf("Unknown type for bitvector.");
  return SQLITE_ERROR;
}

static int int8_vec_from_value(sqlite3_value *value, i8 **vector,
                               size_t *dimensions, vector_cleanup *cleanup,
                               char **pzErr) {
  int value_type = sqlite3_value_type(value);
  if (value_type == SQLITE_BLOB) {
    const void *blob = sqlite3_value_blob(value);
    int bytes = sqlite3_value_bytes(value);
    if (bytes == 0) {
      *pzErr = sqlite3_mprintf("zero-length vectors are not supported.");
      return SQLITE_ERROR;
    }
    *vector = (i8 *)blob;
    *dimensions = bytes;
    *cleanup = vector_cleanup_noop;
    return SQLITE_OK;
  }

  if (value_type == SQLITE_TEXT) {
    const char *source = (const char *)sqlite3_value_text(value);
    int source_len = sqlite3_value_bytes(value);
    int i = 0;

    if (source_len == 0) {
      *pzErr = sqlite3_mprintf("zero-length vectors are not supported.");
      return SQLITE_ERROR;
    }

    struct Array x;
    int rc = array_init(&x, sizeof(i8), ceil(source_len / 2.0));
    if (rc != SQLITE_OK) {
      return rc;
    }

    // advance leading whitespace to first '['
    while (i < source_len) {
      if (vecJsonIsspace(source[i])) {
        i++;
        continue;
      }
      if (source[i] == '[') {
        break;
      }

      *pzErr = sqlite3_mprintf(
          "JSON array parsing error: Input does not start with '['");
      array_cleanup(&x);
      return SQLITE_ERROR;
    }
    if (source[i] != '[') {
      *pzErr = sqlite3_mprintf(
          "JSON array parsing error: Input does not start with '['");
      array_cleanup(&x);
      return SQLITE_ERROR;
    }
    int offset = i + 1;

    while (offset < source_len) {
      char *ptr = (char *)&source[offset];
      char *endptr;

      errno = 0;
      long result = strtol(ptr, &endptr, 10);
      if ((errno != 0 && result == 0) ||
          (errno == ERANGE && (result == LONG_MAX || result == LONG_MIN))) {
        sqlite3_free(x.z);
        *pzErr = sqlite3_mprintf("JSON parsing error");
        return SQLITE_ERROR;
      }

      if (endptr == ptr) {
        if (*ptr != ']') {
          sqlite3_free(x.z);
          *pzErr = sqlite3_mprintf("JSON parsing error");
          return SQLITE_ERROR;
        }
        goto done;
      }

      if (result < INT8_MIN || result > INT8_MAX) {
        sqlite3_free(x.z);
        *pzErr =
            sqlite3_mprintf("JSON parsing error: value out of range for int8");
        return SQLITE_ERROR;
      }

      i8 res = (i8)result;
      array_append(&x, (const void *)&res);

      offset += (endptr - ptr);
      while (offset < source_len) {
        if (vecJsonIsspace(source[offset])) {
          offset++;
          continue;
        }
        if (source[offset] == ',') {
          offset++;
          continue;
        }
        if (source[offset] == ']')
          goto done;
        break;
      }
    }

  done:

    if (x.length > 0) {
      *vector = (i8 *)x.z;
      *dimensions = x.length;
      *cleanup = (vector_cleanup)sqlite3_free;
      return SQLITE_OK;
    }
    sqlite3_free(x.z);
    *pzErr = sqlite3_mprintf("zero-length vectors are not supported.");
    return SQLITE_ERROR;
  }

  *pzErr = sqlite3_mprintf("Unknown type for int8 vector.");
  return SQLITE_ERROR;
}

/**
 * @brief Extract a vector from a sqlite3_value. Can be a float32, int8, or bit
 * vector.
 *
 * @param value: the sqlite3_value to read from.
 * @param vector: Output pointer to vector data.
 * @param dimensions: Output number of dimensions
 * @param dimensions: Output vector element type
 * @param cleanup
 * @param pzErrorMessage
 * @return int SQLITE_OK on success, error code otherwise
 */
int vector_from_value(sqlite3_value *value, void **vector, size_t *dimensions,
                      enum VectorElementType *element_type,
                      vector_cleanup *cleanup, char **pzErrorMessage) {
  int subtype = sqlite3_value_subtype(value);
  if (!subtype || (subtype == SQLITE_VEC_ELEMENT_TYPE_FLOAT32) ||
      (subtype == JSON_SUBTYPE)) {
    int rc = fvec_from_value(value, (f32 **)vector, dimensions,
                             (fvec_cleanup *)cleanup, pzErrorMessage);
    if (rc == SQLITE_OK) {
      *element_type = SQLITE_VEC_ELEMENT_TYPE_FLOAT32;
    }
    return rc;
  }

  if (subtype == SQLITE_VEC_ELEMENT_TYPE_BIT) {
    int rc = bitvec_from_value(value, (u8 **)vector, dimensions, cleanup,
                               pzErrorMessage);
    if (rc == SQLITE_OK) {
      *element_type = SQLITE_VEC_ELEMENT_TYPE_BIT;
    }
    return rc;
  }
  if (subtype == SQLITE_VEC_ELEMENT_TYPE_INT8) {
    int rc = int8_vec_from_value(value, (i8 **)vector, dimensions, cleanup,
                                 pzErrorMessage);
    if (rc == SQLITE_OK) {
      *element_type = SQLITE_VEC_ELEMENT_TYPE_INT8;
    }
    return rc;
  }
  *pzErrorMessage = sqlite3_mprintf("Unknown subtype: %d", subtype);
  return SQLITE_ERROR;
}

int ensure_vector_match(sqlite3_value *aValue, sqlite3_value *bValue, void **a,
                        void **b, enum VectorElementType *element_type,
                        size_t *dimensions, vector_cleanup *outACleanup,
                        vector_cleanup *outBCleanup, char **outError) {
  int rc;
  enum VectorElementType aType, bType;
  size_t aDims, bDims;
  char *error = NULL;
  vector_cleanup aCleanup, bCleanup;

  rc = vector_from_value(aValue, a, &aDims, &aType, &aCleanup, &error);
  if (rc != SQLITE_OK) {
    *outError = sqlite3_mprintf("Error reading 1st vector: %s", error);
    sqlite3_free(error);
    return SQLITE_ERROR;
  }

  rc = vector_from_value(bValue, b, &bDims, &bType, &bCleanup, &error);
  if (rc != SQLITE_OK) {
    *outError = sqlite3_mprintf("Error reading 2nd vector: %s", error);
    sqlite3_free(error);
    aCleanup(*a);
    return SQLITE_ERROR;
  }

  if (aType != bType) {
    *outError =
        sqlite3_mprintf("Vector type mistmatch. First vector has type %s, "
                        "while the second has type %s.",
                        vector_subtype_name(aType), vector_subtype_name(bType));
    aCleanup(*a);
    bCleanup(*b);
    return SQLITE_ERROR;
  }
  if (aDims != bDims) {
    *outError = sqlite3_mprintf(
        "Vector dimension mistmatch. First vector has %ld dimensions, "
        "while the second has %ld dimensions.",
        aDims, bDims);
    aCleanup(*a);
    bCleanup(*b);
    return SQLITE_ERROR;
  }
  *element_type = aType;
  *dimensions = aDims;
  *outACleanup = aCleanup;
  *outBCleanup = bCleanup;
  return SQLITE_OK;
}

int _cmp(const void *a, const void *b) {
  i64 x = *(const i64 *)a;
  i64 y = *(const i64 *)b;
  return (x > y) - (x < y);
}

// Filtered KNN, one policy for every index: a `rowid in (...)` filter of at
// most VEC0_KNN_EXACT_FILTER_MAX rows ranks each named row by its exact
// distance, since a selective filter starves an index search of matches; a
// larger one is applied while the index is searched. rowidsIn is
// sorted (vec0_rowids_in_collect), and NULL means no filter.
#define VEC0_KNN_EXACT_FILTER_MAX 4096

static int vec0_rowids_rank_exact(const struct Array *rowidsIn) {
  return rowidsIn && rowidsIn->length <= VEC0_KNN_EXACT_FILTER_MAX;
}

static int vec0_rowids_contains(const struct Array *rowidsIn, i64 rowid) {
  return !rowidsIn ||
         bsearch(&rowid, rowidsIn->z, rowidsIn->length, sizeof(i64), _cmp) != NULL;
}

#pragma region scalar functions
static void vec_f32(sqlite3_context *context, int argc, sqlite3_value **argv) {
  assert(argc == 1);
  int rc;
  f32 *vector = NULL;
  size_t dimensions;
  fvec_cleanup cleanup;
  char *errmsg;
  rc = fvec_from_value(argv[0], &vector, &dimensions, &cleanup, &errmsg);
  if (rc != SQLITE_OK) {
    sqlite3_result_error(context, errmsg, -1);
    sqlite3_free(errmsg);
    return;
  }
  sqlite3_result_blob(context, vector, dimensions * sizeof(f32),
                      (void (*)(void *))cleanup);
  sqlite3_result_subtype(context, SQLITE_VEC_ELEMENT_TYPE_FLOAT32);
}

static void vec_bit(sqlite3_context *context, int argc, sqlite3_value **argv) {
  assert(argc == 1);
  int rc;
  u8 *vector;
  size_t dimensions;
  vector_cleanup cleanup;
  char *errmsg;
  rc = bitvec_from_value(argv[0], &vector, &dimensions, &cleanup, &errmsg);
  if (rc != SQLITE_OK) {
    sqlite3_result_error(context, errmsg, -1);
    sqlite3_free(errmsg);
    return;
  }
  sqlite3_result_blob(context, vector, dimensions / CHAR_BIT, SQLITE_TRANSIENT);
  sqlite3_result_subtype(context, SQLITE_VEC_ELEMENT_TYPE_BIT);
  cleanup(vector);
}
static void vec_int8(sqlite3_context *context, int argc, sqlite3_value **argv) {
  assert(argc == 1);
  int rc;
  i8 *vector;
  size_t dimensions;
  vector_cleanup cleanup;
  char *errmsg;
  rc = int8_vec_from_value(argv[0], &vector, &dimensions, &cleanup, &errmsg);
  if (rc != SQLITE_OK) {
    sqlite3_result_error(context, errmsg, -1);
    sqlite3_free(errmsg);
    return;
  }
  sqlite3_result_blob(context, vector, dimensions, SQLITE_TRANSIENT);
  sqlite3_result_subtype(context, SQLITE_VEC_ELEMENT_TYPE_INT8);
  cleanup(vector);
}

static void vec_length(sqlite3_context *context, int argc,
                       sqlite3_value **argv) {
  assert(argc == 1);
  int rc;
  void *vector;
  size_t dimensions;
  vector_cleanup cleanup;
  char *errmsg;
  enum VectorElementType elementType;
  rc = vector_from_value(argv[0], &vector, &dimensions, &elementType, &cleanup,
                         &errmsg);
  if (rc != SQLITE_OK) {
    sqlite3_result_error(context, errmsg, -1);
    sqlite3_free(errmsg);
    return;
  }
  sqlite3_result_int64(context, dimensions);
  cleanup(vector);
}

static void vec_distance_cosine(sqlite3_context *context, int argc,
                                sqlite3_value **argv) {
  assert(argc == 2);
  int rc;
  void *a = NULL, *b = NULL;
  size_t dimensions;
  vector_cleanup aCleanup, bCleanup;
  char *error;
  enum VectorElementType elementType;
  rc = ensure_vector_match(argv[0], argv[1], &a, &b, &elementType, &dimensions,
                           &aCleanup, &bCleanup, &error);
  if (rc != SQLITE_OK) {
    sqlite3_result_error(context, error, -1);
    sqlite3_free(error);
    return;
  }

  switch (elementType) {
  case SQLITE_VEC_ELEMENT_TYPE_BIT: {
    sqlite3_result_error(
        context, "Cannot calculate cosine distance between two bitvectors.",
        -1);
    goto finish;
  }
  case SQLITE_VEC_ELEMENT_TYPE_FLOAT32: {
    f32 result = distance_cosine_float(a, b, &dimensions);
    sqlite3_result_double(context, result);
    goto finish;
  }
  case SQLITE_VEC_ELEMENT_TYPE_INT8: {
    f32 result = distance_cosine_int8(a, b, &dimensions);
    sqlite3_result_double(context, result);
    goto finish;
  }
  }

finish:
  aCleanup(a);
  bCleanup(b);
  return;
}

static void vec_distance_l2(sqlite3_context *context, int argc,
                            sqlite3_value **argv) {
  assert(argc == 2);
  int rc;
  void *a = NULL, *b = NULL;
  size_t dimensions;
  vector_cleanup aCleanup, bCleanup;
  char *error;
  enum VectorElementType elementType;
  rc = ensure_vector_match(argv[0], argv[1], &a, &b, &elementType, &dimensions,
                           &aCleanup, &bCleanup, &error);
  if (rc != SQLITE_OK) {
    sqlite3_result_error(context, error, -1);
    sqlite3_free(error);
    return;
  }

  switch (elementType) {
  case SQLITE_VEC_ELEMENT_TYPE_BIT: {
    sqlite3_result_error(
        context, "Cannot calculate L2 distance between two bitvectors.", -1);
    goto finish;
  }
  case SQLITE_VEC_ELEMENT_TYPE_FLOAT32: {
    f32 result = distance_l2_sqr_float(a, b, &dimensions);
    sqlite3_result_double(context, result);
    goto finish;
  }
  case SQLITE_VEC_ELEMENT_TYPE_INT8: {
    f32 result = distance_l2_sqr_int8(a, b, &dimensions);
    sqlite3_result_double(context, result);
    goto finish;
  }
  }

finish:
  aCleanup(a);
  bCleanup(b);
  return;
}

static void vec_distance_l1(sqlite3_context *context, int argc,
                            sqlite3_value **argv) {
  assert(argc == 2);
  int rc;
  void *a, *b;
  size_t dimensions;
  vector_cleanup aCleanup, bCleanup;
  char *error;
  enum VectorElementType elementType;
  rc = ensure_vector_match(argv[0], argv[1], &a, &b, &elementType, &dimensions,
                           &aCleanup, &bCleanup, &error);
  if (rc != SQLITE_OK) {
    sqlite3_result_error(context, error, -1);
    sqlite3_free(error);
    return;
  }

  switch (elementType) {
  case SQLITE_VEC_ELEMENT_TYPE_BIT: {
    sqlite3_result_error(
        context, "Cannot calculate L1 distance between two bitvectors.", -1);
    goto finish;
  }
  case SQLITE_VEC_ELEMENT_TYPE_FLOAT32: {
    double result = distance_l1_f32(a, b, &dimensions);
    sqlite3_result_double(context, result);
    goto finish;
  }
  case SQLITE_VEC_ELEMENT_TYPE_INT8: {
    i64 result = distance_l1_int8(a, b, &dimensions);
    sqlite3_result_int(context, result);
    goto finish;
  }
  }

finish:
  aCleanup(a);
  bCleanup(b);
  return;
}

static void vec_distance_hamming(sqlite3_context *context, int argc,
                                 sqlite3_value **argv) {
  assert(argc == 2);
  int rc;
  void *a = NULL, *b = NULL;
  size_t dimensions;
  vector_cleanup aCleanup, bCleanup;
  char *error;
  enum VectorElementType elementType;
  rc = ensure_vector_match(argv[0], argv[1], &a, &b, &elementType, &dimensions,
                           &aCleanup, &bCleanup, &error);
  if (rc != SQLITE_OK) {
    sqlite3_result_error(context, error, -1);
    sqlite3_free(error);
    return;
  }

  switch (elementType) {
  case SQLITE_VEC_ELEMENT_TYPE_BIT: {
    sqlite3_result_double(context, distance_hamming(a, b, &dimensions));
    goto finish;
  }
  case SQLITE_VEC_ELEMENT_TYPE_FLOAT32: {
    sqlite3_result_error(
        context,
        "Cannot calculate hamming distance between two float32 vectors.", -1);
    goto finish;
  }
  case SQLITE_VEC_ELEMENT_TYPE_INT8: {
    sqlite3_result_error(
        context, "Cannot calculate hamming distance between two int8 vectors.",
        -1);
    goto finish;
  }
  }

finish:
  aCleanup(a);
  bCleanup(b);
  return;
}

static void vec_distance_dot(sqlite3_context *context, int argc,
                             sqlite3_value **argv) {
  assert(argc == 2);
  void *a = NULL, *b = NULL;
  size_t dimensions;
  vector_cleanup aCleanup, bCleanup;
  char *error;
  enum VectorElementType elementType;
  int rc = ensure_vector_match(argv[0], argv[1], &a, &b, &elementType,
                               &dimensions, &aCleanup, &bCleanup, &error);
  if (rc != SQLITE_OK) {
    sqlite3_result_error(context, error, -1);
    sqlite3_free(error);
    return;
  }
  switch (elementType) {
  case SQLITE_VEC_ELEMENT_TYPE_BIT:
    sqlite3_result_error(
        context, "Cannot calculate dot distance between two bitvectors.", -1);
    break;
  case SQLITE_VEC_ELEMENT_TYPE_FLOAT32:
    sqlite3_result_double(context, distance_dot_float(a, b, &dimensions));
    break;
  case SQLITE_VEC_ELEMENT_TYPE_INT8:
    sqlite3_result_double(context, distance_dot_int8(a, b, &dimensions));
    break;
  }
  aCleanup(a);
  bCleanup(b);
}

static void vec_distance_jaccard(sqlite3_context *context, int argc,
                                 sqlite3_value **argv) {
  assert(argc == 2);
  void *a = NULL, *b = NULL;
  size_t dimensions;
  vector_cleanup aCleanup, bCleanup;
  char *error;
  enum VectorElementType elementType;
  int rc = ensure_vector_match(argv[0], argv[1], &a, &b, &elementType,
                               &dimensions, &aCleanup, &bCleanup, &error);
  if (rc != SQLITE_OK) {
    sqlite3_result_error(context, error, -1);
    sqlite3_free(error);
    return;
  }
  if (elementType == SQLITE_VEC_ELEMENT_TYPE_BIT) {
    sqlite3_result_double(context, distance_jaccard(a, b, &dimensions));
  } else {
    sqlite3_result_error(
        context, "Jaccard distance is only defined for bitvectors.", -1);
  }
  aCleanup(a);
  bCleanup(b);
}

size_t vector_byte_size(enum VectorElementType elementType,
                               size_t dimensions);

static void vec_concat(sqlite3_context *context, int argc,
                       sqlite3_value **argv) {
  assert(argc == 2);
  void *a = NULL, *b = NULL;
  size_t aDims, bDims;
  vector_cleanup aCleanup, bCleanup;
  char *error;
  enum VectorElementType aType, bType;
  if (vector_from_value(argv[0], &a, &aDims, &aType, &aCleanup, &error) !=
      SQLITE_OK) {
    sqlite3_result_error(context, error, -1);
    sqlite3_free(error);
    return;
  }
  if (vector_from_value(argv[1], &b, &bDims, &bType, &bCleanup, &error) !=
      SQLITE_OK) {
    sqlite3_result_error(context, error, -1);
    sqlite3_free(error);
    aCleanup(a);
    return;
  }
  if (aType != bType) {
    sqlite3_result_error(context,
                         "Cannot concatenate vectors of different types.", -1);
    goto finish;
  }
  size_t aBytes = vector_byte_size(aType, aDims);
  size_t bBytes = vector_byte_size(bType, bDims);
  u8 *out = sqlite3_malloc64(aBytes + bBytes);
  if (!out) {
    sqlite3_result_error_nomem(context);
    goto finish;
  }
  memcpy(out, a, aBytes);
  memcpy(out + aBytes, b, bBytes);
  sqlite3_result_blob64(context, out, aBytes + bBytes, sqlite3_free);
  sqlite3_result_subtype(context, aType);
finish:
  aCleanup(a);
  bCleanup(b);
}

char *vec_type_name(enum VectorElementType elementType) {
  switch (elementType) {
  case SQLITE_VEC_ELEMENT_TYPE_FLOAT32:
    return "float32";
  case SQLITE_VEC_ELEMENT_TYPE_INT8:
    return "int8";
  case SQLITE_VEC_ELEMENT_TYPE_BIT:
    return "bit";
  }
  return "";
}

static void vec_type(sqlite3_context *context, int argc, sqlite3_value **argv) {
  assert(argc == 1);
  void *vector;
  size_t dimensions;
  vector_cleanup cleanup;
  char *pzError;
  enum VectorElementType elementType;
  int rc = vector_from_value(argv[0], &vector, &dimensions, &elementType,
                             &cleanup, &pzError);
  if (rc != SQLITE_OK) {
    sqlite3_result_error(context, pzError, -1);
    sqlite3_free(pzError);
    return;
  }
  sqlite3_result_text(context, vec_type_name(elementType), -1, SQLITE_STATIC);
  cleanup(vector);
}
/**
 * Sign-bit quantization shared by the IVF and DiskANN binary quantizers and
 * vec_quantize_binary(): bit i is set when src[i] > 0. (The rescore bit
 * quantizer sets it for src[i] >= 0; its stored format keeps that rule.)
 */
static void vec0_quantize_binary_f32(const f32 *src, u8 *dst, size_t n) {
  memset(dst, 0, n / CHAR_BIT);
  for (size_t i = 0; i < n; i++) {
    if (src[i] > 0.0f) dst[i / CHAR_BIT] |= (u8)(1 << (i % CHAR_BIT));
  }
}

/**
 * Map [-1, 1] onto int8 [-128, 127], clamping outside values (and NaN) so the
 * float -> int8 conversion is always defined. Shared by the rescore and
 * DiskANN int8 quantizers.
 */
static void vec0_quantize_int8_unit(const f32 *src, i8 *dst, size_t n) {
  const f32 step = 2.0f / 255.0f;
  for (size_t i = 0; i < n; i++) {
    f32 v = (src[i] + 1.0f) / step - 128.0f;
    if (!(v <= 127.0f)) v = 127.0f;
    if (!(v >= -128.0f)) v = -128.0f;
    dst[i] = (i8)v;
  }
}

static void vec_quantize_binary(sqlite3_context *context, int argc,
                                sqlite3_value **argv) {
  assert(argc == 1);
  void *vector;
  size_t dimensions;
  vector_cleanup vectorCleanup;
  char *pzError;
  enum VectorElementType elementType;
  int rc = vector_from_value(argv[0], &vector, &dimensions, &elementType,
                             &vectorCleanup, &pzError);
  if (rc != SQLITE_OK) {
    sqlite3_result_error(context, pzError, -1);
    sqlite3_free(pzError);
    return;
  }

  if (dimensions <= 0) {
    sqlite3_result_error(context, "Zero length vectors are not supported.", -1);
    goto cleanup;
    return;
  }
  if ((dimensions % CHAR_BIT) != 0) {
    sqlite3_result_error(
        context,
        "Binary quantization requires vectors with a length divisible by 8",
        -1);
    goto cleanup;
    return;
  }

  int sz = dimensions / CHAR_BIT;
  u8 *out = sqlite3_malloc(sz);
  if (!out) {
    sqlite3_result_error_code(context, SQLITE_NOMEM);
    goto cleanup;
    return;
  }
  memset(out, 0, sz);

  switch (elementType) {
  case SQLITE_VEC_ELEMENT_TYPE_FLOAT32: {
    vec0_quantize_binary_f32((const f32 *)vector, out, dimensions);
    break;
  }
  case SQLITE_VEC_ELEMENT_TYPE_INT8: {
    for (size_t i = 0; i < dimensions; i++) {
      int res = ((i8 *)vector)[i] > 0;
      out[i / 8] |= (res << (i % 8));
    }
    break;
  }
  case SQLITE_VEC_ELEMENT_TYPE_BIT: {
    sqlite3_result_error(context,
                         "Can only binary quantize float or int8 vectors", -1);
    sqlite3_free(out);
    return;
  }
  }
  sqlite3_result_blob(context, out, sz, sqlite3_free);
  sqlite3_result_subtype(context, SQLITE_VEC_ELEMENT_TYPE_BIT);

cleanup:
  vectorCleanup(vector);
}

static void vec_quantize_int8(sqlite3_context *context, int argc,
                              sqlite3_value **argv) {
  assert(argc == 2);
  f32 *srcVector;
  size_t dimensions;
  fvec_cleanup srcCleanup;
  char *err;
  i8 *out = NULL;
  int rc = fvec_from_value(argv[0], &srcVector, &dimensions, &srcCleanup, &err);
  if (rc != SQLITE_OK) {
    sqlite3_result_error(context, err, -1);
    sqlite3_free(err);
    return;
  }

  int sz = dimensions * sizeof(i8);
  out = sqlite3_malloc(sz);
  if (!out) {
    sqlite3_result_error_nomem(context);
    goto cleanup;
  }
  memset(out, 0, sz);

  if ((sqlite3_value_type(argv[1]) != SQLITE_TEXT) ||
      (sqlite3_value_bytes(argv[1]) != strlen("unit")) ||
      (sqlite3_stricmp((const char *)sqlite3_value_text(argv[1]), "unit") !=
       0)) {
    sqlite3_result_error(
        context, "2nd argument to vec_quantize_int8() must be 'unit'.", -1);
    sqlite3_free(out);
    goto cleanup;
  }
  f32 step = (1.0 - (-1.0)) / 255;
  for (size_t i = 0; i < dimensions; i++) {
    double val = ((srcVector[i] - (-1.0)) / step) - 128;
    if (!(val <= 127.0)) val = 127.0;   /* also clamps NaN */
    if (!(val >= -128.0)) val = -128.0;
    out[i] = (i8)val;
  }

  sqlite3_result_blob(context, out, dimensions * sizeof(i8), sqlite3_free);
  sqlite3_result_subtype(context, SQLITE_VEC_ELEMENT_TYPE_INT8);

cleanup:
  srcCleanup(srcVector);
}

static void vec_add(sqlite3_context *context, int argc, sqlite3_value **argv) {
  assert(argc == 2);
  int rc;
  void *a = NULL, *b = NULL;
  size_t dimensions;
  vector_cleanup aCleanup, bCleanup;
  char *error;
  enum VectorElementType elementType;
  rc = ensure_vector_match(argv[0], argv[1], &a, &b, &elementType, &dimensions,
                           &aCleanup, &bCleanup, &error);
  if (rc != SQLITE_OK) {
    sqlite3_result_error(context, error, -1);
    sqlite3_free(error);
    return;
  }

  switch (elementType) {
  case SQLITE_VEC_ELEMENT_TYPE_BIT: {
    sqlite3_result_error(context, "Cannot add two bitvectors together.", -1);
    goto finish;
  }
  case SQLITE_VEC_ELEMENT_TYPE_FLOAT32: {
    size_t outSize = dimensions * sizeof(f32);
    f32 *out = sqlite3_malloc(outSize);
    if (!out) {
      sqlite3_result_error_nomem(context);
      goto finish;
    }
    memset(out, 0, outSize);
    for (size_t i = 0; i < dimensions; i++) {
      out[i] = ((f32 *)a)[i] + ((f32 *)b)[i];
    }
    sqlite3_result_blob(context, out, outSize, sqlite3_free);
    sqlite3_result_subtype(context, SQLITE_VEC_ELEMENT_TYPE_FLOAT32);
    goto finish;
  }
  case SQLITE_VEC_ELEMENT_TYPE_INT8: {
    size_t outSize = dimensions * sizeof(i8);
    i8 *out = sqlite3_malloc(outSize);
    if (!out) {
      sqlite3_result_error_nomem(context);
      goto finish;
    }
    memset(out, 0, outSize);
    for (size_t i = 0; i < dimensions; i++) {
      out[i] = ((i8 *)a)[i] + ((i8 *)b)[i];
    }
    sqlite3_result_blob(context, out, outSize, sqlite3_free);
    sqlite3_result_subtype(context, SQLITE_VEC_ELEMENT_TYPE_INT8);
    goto finish;
  }
  }
finish:
  aCleanup(a);
  bCleanup(b);
  return;
}
static void vec_sub(sqlite3_context *context, int argc, sqlite3_value **argv) {
  assert(argc == 2);
  int rc;
  void *a = NULL, *b = NULL;
  size_t dimensions;
  vector_cleanup aCleanup, bCleanup;
  char *error;
  enum VectorElementType elementType;
  rc = ensure_vector_match(argv[0], argv[1], &a, &b, &elementType, &dimensions,
                           &aCleanup, &bCleanup, &error);
  if (rc != SQLITE_OK) {
    sqlite3_result_error(context, error, -1);
    sqlite3_free(error);
    return;
  }

  switch (elementType) {
  case SQLITE_VEC_ELEMENT_TYPE_BIT: {
    sqlite3_result_error(context, "Cannot subtract two bitvectors together.",
                         -1);
    goto finish;
  }
  case SQLITE_VEC_ELEMENT_TYPE_FLOAT32: {
    size_t outSize = dimensions * sizeof(f32);
    f32 *out = sqlite3_malloc(outSize);
    if (!out) {
      sqlite3_result_error_nomem(context);
      goto finish;
    }
    memset(out, 0, outSize);
    for (size_t i = 0; i < dimensions; i++) {
      out[i] = ((f32 *)a)[i] - ((f32 *)b)[i];
    }
    sqlite3_result_blob(context, out, outSize, sqlite3_free);
    sqlite3_result_subtype(context, SQLITE_VEC_ELEMENT_TYPE_FLOAT32);
    goto finish;
  }
  case SQLITE_VEC_ELEMENT_TYPE_INT8: {
    size_t outSize = dimensions * sizeof(i8);
    i8 *out = sqlite3_malloc(outSize);
    if (!out) {
      sqlite3_result_error_nomem(context);
      goto finish;
    }
    memset(out, 0, outSize);
    for (size_t i = 0; i < dimensions; i++) {
      out[i] = ((i8 *)a)[i] - ((i8 *)b)[i];
    }
    sqlite3_result_blob(context, out, outSize, sqlite3_free);
    sqlite3_result_subtype(context, SQLITE_VEC_ELEMENT_TYPE_INT8);
    goto finish;
  }
  }
finish:
  aCleanup(a);
  bCleanup(b);
  return;
}
static void vec_slice(sqlite3_context *context, int argc,
                      sqlite3_value **argv) {
  assert(argc == 3);

  void *vector;
  size_t dimensions;
  vector_cleanup cleanup;
  char *err;
  enum VectorElementType elementType;

  int rc = vector_from_value(argv[0], &vector, &dimensions, &elementType,
                             &cleanup, &err);
  if (rc != SQLITE_OK) {
    sqlite3_result_error(context, err, -1);
    sqlite3_free(err);
    return;
  }

  int start = sqlite3_value_int(argv[1]);
  int end = sqlite3_value_int(argv[2]);

  if (start < 0) {
    sqlite3_result_error(context,
                         "slice 'start' index must be a postive number.", -1);
    goto done;
  }
  if (end < 0) {
    sqlite3_result_error(context, "slice 'end' index must be a postive number.",
                         -1);
    goto done;
  }
  if (((size_t)start) > dimensions) {
    sqlite3_result_error(
        context, "slice 'start' index is greater than the number of dimensions",
        -1);
    goto done;
  }
  if (((size_t)end) > dimensions) {
    sqlite3_result_error(
        context, "slice 'end' index is greater than the number of dimensions",
        -1);
    goto done;
  }
  if (start > end) {
    sqlite3_result_error(context,
                         "slice 'start' index is greater than 'end' index", -1);
    goto done;
  }
  if (start == end) {
    sqlite3_result_error(context,
                         "slice 'start' index is equal to the 'end' index, "
                         "vectors must have non-zero length",
                         -1);
    goto done;
  }
  size_t n = end - start;

  switch (elementType) {
  case SQLITE_VEC_ELEMENT_TYPE_FLOAT32: {
    int outSize = n * sizeof(f32);
    f32 *out = sqlite3_malloc(outSize);
    if (!out) {
      sqlite3_result_error_nomem(context);
      goto done;
    }
    memset(out, 0, outSize);
    for (size_t i = 0; i < n; i++) {
      out[i] = ((f32 *)vector)[start + i];
    }
    sqlite3_result_blob(context, out, outSize, sqlite3_free);
    sqlite3_result_subtype(context, SQLITE_VEC_ELEMENT_TYPE_FLOAT32);
    goto done;
  }
  case SQLITE_VEC_ELEMENT_TYPE_INT8: {
    int outSize = n * sizeof(i8);
    i8 *out = sqlite3_malloc(outSize);
    if (!out) {
      sqlite3_result_error_nomem(context);
      return;
    }
    memset(out, 0, outSize);
    for (size_t i = 0; i < n; i++) {
      out[i] = ((i8 *)vector)[start + i];
    }
    sqlite3_result_blob(context, out, outSize, sqlite3_free);
    sqlite3_result_subtype(context, SQLITE_VEC_ELEMENT_TYPE_INT8);
    goto done;
  }
  case SQLITE_VEC_ELEMENT_TYPE_BIT: {
    if ((start % CHAR_BIT) != 0) {
      sqlite3_result_error(context, "start index must be divisible by 8.", -1);
      goto done;
    }
    if ((end % CHAR_BIT) != 0) {
      sqlite3_result_error(context, "end index must be divisible by 8.", -1);
      goto done;
    }
    int outSize = n / CHAR_BIT;
    u8 *out = sqlite3_malloc(outSize);
    if (!out) {
      sqlite3_result_error_nomem(context);
      return;
    }
    memset(out, 0, outSize);
    for (size_t i = 0; i < n / CHAR_BIT; i++) {
      out[i] = ((u8 *)vector)[(start / CHAR_BIT) + i];
    }
    sqlite3_result_blob(context, out, outSize, sqlite3_free);
    sqlite3_result_subtype(context, SQLITE_VEC_ELEMENT_TYPE_BIT);
    goto done;
  }
  }
done:
  cleanup(vector);
}

static void vec_to_json(sqlite3_context *context, int argc,
                        sqlite3_value **argv) {
  assert(argc == 1);
  void *vector;
  size_t dimensions;
  vector_cleanup cleanup;
  char *err;
  enum VectorElementType elementType;

  int rc = vector_from_value(argv[0], &vector, &dimensions, &elementType,
                             &cleanup, &err);
  if (rc != SQLITE_OK) {
    sqlite3_result_error(context, err, -1);
    sqlite3_free(err);
    return;
  }

  sqlite3_str *str = sqlite3_str_new(sqlite3_context_db_handle(context));
  sqlite3_str_appendall(str, "[");
  for (size_t i = 0; i < dimensions; i++) {
    if (i != 0) {
      sqlite3_str_appendall(str, ",");
    }
    if (elementType == SQLITE_VEC_ELEMENT_TYPE_FLOAT32) {
      f32 value = ((f32 *)vector)[i];
      if (isnan(value)) {
        sqlite3_str_appendall(str, "null");
      } else {
        sqlite3_str_appendf(str, "%f", value);
      }

    } else if (elementType == SQLITE_VEC_ELEMENT_TYPE_INT8) {
      sqlite3_str_appendf(str, "%d", ((i8 *)vector)[i]);
    } else if (elementType == SQLITE_VEC_ELEMENT_TYPE_BIT) {
      u8 b = (((u8 *)vector)[i / 8] >> (i % CHAR_BIT)) & 1;
      sqlite3_str_appendf(str, "%d", b);
    }
  }
  sqlite3_str_appendall(str, "]");
  int len = sqlite3_str_length(str);
  char *s = sqlite3_str_finish(str);
  if (s) {
    sqlite3_result_text(context, s, len, sqlite3_free);
    sqlite3_result_subtype(context, JSON_SUBTYPE);
  } else {
    sqlite3_result_error_nomem(context);
  }
  cleanup(vector);
}

static void vec_normalize(sqlite3_context *context, int argc,
                          sqlite3_value **argv) {
  assert(argc == 1);
  void *vector;
  size_t dimensions;
  vector_cleanup cleanup;
  char *err;
  enum VectorElementType elementType;

  int rc = vector_from_value(argv[0], &vector, &dimensions, &elementType,
                             &cleanup, &err);
  if (rc != SQLITE_OK) {
    sqlite3_result_error(context, err, -1);
    sqlite3_free(err);
    return;
  }

  if (elementType != SQLITE_VEC_ELEMENT_TYPE_FLOAT32) {
    sqlite3_result_error(
        context, "only float32 vectors are supported when normalizing", -1);
    cleanup(vector);
    return;
  }

  int outSize = dimensions * sizeof(f32);
  f32 *out = sqlite3_malloc(outSize);
  if (!out) {
    cleanup(vector);
    sqlite3_result_error_code(context, SQLITE_NOMEM);
    return;
  }
  memset(out, 0, outSize);

  f32 *v = (f32 *)vector;

  // Accumulate in double so that squaring small or large f32 elements does
  // not underflow or overflow, see distance_cosine_float().
  double norm = 0;
  for (size_t i = 0; i < dimensions; i++) {
    norm += (double)v[i] * (double)v[i];
  }
  norm = sqrt(norm);
  for (size_t i = 0; i < dimensions; i++) {
    out[i] = v[i] / norm;
  }

  sqlite3_result_blob(context, out, dimensions * sizeof(f32), sqlite3_free);
  sqlite3_result_subtype(context, SQLITE_VEC_ELEMENT_TYPE_FLOAT32);
  cleanup(vector);
}

static void _static_text_func(sqlite3_context *context, int argc,
                              sqlite3_value **argv) {
  UNUSED_PARAMETER(argc);
  UNUSED_PARAMETER(argv);
  sqlite3_result_text(context, sqlite3_user_data(context), -1, SQLITE_STATIC);
}

#pragma endregion

enum Vec0TokenType {
  TOKEN_TYPE_IDENTIFIER,
  TOKEN_TYPE_DIGIT,
  TOKEN_TYPE_LBRACKET,
  TOKEN_TYPE_RBRACKET,
  TOKEN_TYPE_PLUS,
  TOKEN_TYPE_EQ,
  TOKEN_TYPE_LPAREN,
  TOKEN_TYPE_RPAREN,
  TOKEN_TYPE_COMMA,
};
struct Vec0Token {
  enum Vec0TokenType token_type;
  char *start;
  char *end;
};

int is_alpha(char x) {
  return (x >= 'a' && x <= 'z') || (x >= 'A' && x <= 'Z');
}
int is_digit(char x) { return (x >= '0' && x <= '9'); }
int is_whitespace(char x) {
  return x == ' ' || x == '\t' || x == '\n' || x == '\r';
}

#define VEC0_TOKEN_RESULT_EOF 1
#define VEC0_TOKEN_RESULT_SOME 2
#define VEC0_TOKEN_RESULT_ERROR 3

int vec0_token_next(char *start, char *end, struct Vec0Token *out) {
  char *ptr = start;
  while (ptr < end) {
    char curr = *ptr;
    if (is_whitespace(curr)) {
      ptr++;
      continue;
    } else if (curr == '+') {
      ptr++;
      out->start = ptr;
      out->end = ptr;
      out->token_type = TOKEN_TYPE_PLUS;
      return VEC0_TOKEN_RESULT_SOME;
    } else if (curr == '[') {
      ptr++;
      out->start = ptr;
      out->end = ptr;
      out->token_type = TOKEN_TYPE_LBRACKET;
      return VEC0_TOKEN_RESULT_SOME;
    } else if (curr == ']') {
      ptr++;
      out->start = ptr;
      out->end = ptr;
      out->token_type = TOKEN_TYPE_RBRACKET;
      return VEC0_TOKEN_RESULT_SOME;
    } else if (curr == '=') {
      ptr++;
      out->start = ptr;
      out->end = ptr;
      out->token_type = TOKEN_TYPE_EQ;
      return VEC0_TOKEN_RESULT_SOME;
    } else if (curr == '(') {
      ptr++;
      out->start = ptr;
      out->end = ptr;
      out->token_type = TOKEN_TYPE_LPAREN;
      return VEC0_TOKEN_RESULT_SOME;
    } else if (curr == ')') {
      ptr++;
      out->start = ptr;
      out->end = ptr;
      out->token_type = TOKEN_TYPE_RPAREN;
      return VEC0_TOKEN_RESULT_SOME;
    } else if (curr == ',') {
      ptr++;
      out->start = ptr;
      out->end = ptr;
      out->token_type = TOKEN_TYPE_COMMA;
      return VEC0_TOKEN_RESULT_SOME;
    } else if (is_alpha(curr)) {
      char *start = ptr;
      while (ptr < end && (is_alpha(*ptr) || is_digit(*ptr) || *ptr == '_')) {
        ptr++;
      }
      out->start = start;
      out->end = ptr;
      out->token_type = TOKEN_TYPE_IDENTIFIER;
      return VEC0_TOKEN_RESULT_SOME;
    } else if (is_digit(curr)) {
      char *start = ptr;
      while (ptr < end && (is_digit(*ptr))) {
        ptr++;
      }
      out->start = start;
      out->end = ptr;
      out->token_type = TOKEN_TYPE_DIGIT;
      return VEC0_TOKEN_RESULT_SOME;
    } else {
      return VEC0_TOKEN_RESULT_ERROR;
    }
  }
  return VEC0_TOKEN_RESULT_EOF;
}

struct Vec0Scanner {
  char *start;
  char *end;
  char *ptr;
};

void vec0_scanner_init(struct Vec0Scanner *scanner, const char *source,
                       int source_length) {
  scanner->start = (char *)source;
  scanner->end = (char *)source + source_length;
  scanner->ptr = (char *)source;
}
int vec0_scanner_next(struct Vec0Scanner *scanner, struct Vec0Token *out) {
  int rc = vec0_token_next(scanner->start, scanner->end, out);
  if (rc == VEC0_TOKEN_RESULT_SOME) {
    scanner->start = out->end;
  }
  return rc;
}

int vec0_parse_table_option(const char *source, int source_length,
                            char **out_key, int *out_key_length,
                            char **out_value, int *out_value_length) {
  int rc;
  struct Vec0Scanner scanner;
  struct Vec0Token token;
  char *key;
  char *value;
  int keyLength, valueLength;

  vec0_scanner_init(&scanner, source, source_length);

  rc = vec0_scanner_next(&scanner, &token);
  if (rc != VEC0_TOKEN_RESULT_SOME ||
      token.token_type != TOKEN_TYPE_IDENTIFIER) {
    return SQLITE_EMPTY;
  }
  key = token.start;
  keyLength = token.end - token.start;

  rc = vec0_scanner_next(&scanner, &token);
  if (rc != VEC0_TOKEN_RESULT_SOME || token.token_type != TOKEN_TYPE_EQ) {
    return SQLITE_EMPTY;
  }

  rc = vec0_scanner_next(&scanner, &token);
  if (rc != VEC0_TOKEN_RESULT_SOME ||
      !((token.token_type == TOKEN_TYPE_IDENTIFIER) ||
        (token.token_type == TOKEN_TYPE_DIGIT))) {
    return SQLITE_ERROR;
  }
  value = token.start;
  valueLength = token.end - token.start;

  rc = vec0_scanner_next(&scanner, &token);
  if (rc == VEC0_TOKEN_RESULT_EOF) {
    *out_key = key;
    *out_key_length = keyLength;
    *out_value = value;
    *out_value_length = valueLength;
    return SQLITE_OK;
  }
  return SQLITE_ERROR;
}

/**
 * Parse `key = 'string'` (SQL quoting: '' is a quote). Returns SQLITE_EMPTY
 * when source is not this key, SQLITE_ERROR when the value is malformed, and
 * SQLITE_OK with a sqlite3_malloc'd copy of the string in *out.
 */
static int vec0_parse_string_option(const char *source, const char *key, char **out) {
  const char *p = source;
  size_t keyLength = strlen(key);
  while (*p == ' ' || *p == '\t' || *p == '\n') p++;
  if (sqlite3_strnicmp(p, key, (int)keyLength) != 0) return SQLITE_EMPTY;
  p += keyLength;
  while (*p == ' ' || *p == '\t') p++;
  if (*p++ != '=') return SQLITE_EMPTY;
  while (*p == ' ' || *p == '\t') p++;
  if (*p++ != '\'') return SQLITE_ERROR;
  sqlite3_str *str = sqlite3_str_new(NULL);
  for (;; p++) {
    if (*p == 0) { sqlite3_free(sqlite3_str_finish(str)); return SQLITE_ERROR; }
    if (*p == '\'') {
      if (p[1] != '\'') break;
      p++;
    }
    sqlite3_str_appendchar(str, 1, *p);
  }
  p++;
  while (*p == ' ' || *p == '\t' || *p == '\n') p++;
  char *z = sqlite3_str_finish(str);
  if (*p != 0 || !z) { sqlite3_free(z); return z ? SQLITE_ERROR : SQLITE_NOMEM; }
  *out = z;
  return SQLITE_OK;
}

/**
 * @brief Parse an argv[i] entry of a vec0 virtual table definition, and see if
 * it's a PARTITION KEY definition.
 *
 * @param source: argv[i] source string
 * @param source_length: length of the source string
 * @param out_column_name: If it is a partition key, the output column name. Same lifetime
 * as source, points to specific char *
 * @param out_column_name_length: Length of out_column_name in bytes
 * @param out_column_type: SQLITE_TEXT or SQLITE_INTEGER.
 * @return int: SQLITE_EMPTY if not a PK, SQLITE_OK if it is.
 */
int vec0_parse_partition_key_definition(const char *source, int source_length,
                                 char **out_column_name,
                                 int *out_column_name_length,
                                 int *out_column_type) {
  struct Vec0Scanner scanner;
  struct Vec0Token token;
  char *column_name;
  int column_name_length;
  int column_type;
  vec0_scanner_init(&scanner, source, source_length);

  // Check first token is identifier, will be the column name
  int rc = vec0_scanner_next(&scanner, &token);
  if (rc != VEC0_TOKEN_RESULT_SOME ||
      token.token_type != TOKEN_TYPE_IDENTIFIER) {
    return SQLITE_EMPTY;
  }

  column_name = token.start;
  column_name_length = token.end - token.start;

  // Check the next token matches "text" or "integer", as column type
  rc = vec0_scanner_next(&scanner, &token);
  if (rc != VEC0_TOKEN_RESULT_SOME ||
      token.token_type != TOKEN_TYPE_IDENTIFIER) {
    return SQLITE_EMPTY;
  }
  if (sqlite3_strnicmp(token.start, "text", token.end - token.start) == 0) {
    column_type = SQLITE_TEXT;
  } else if (sqlite3_strnicmp(token.start, "int", token.end - token.start) ==
                 0 ||
             sqlite3_strnicmp(token.start, "integer",
                              token.end - token.start) == 0) {
    column_type = SQLITE_INTEGER;
  } else {
    return SQLITE_EMPTY;
  }

  // Check the next token is identifier and matches "partition"
  rc = vec0_scanner_next(&scanner, &token);
  if (rc != VEC0_TOKEN_RESULT_SOME ||
      token.token_type != TOKEN_TYPE_IDENTIFIER) {
    return SQLITE_EMPTY;
  }
  if (sqlite3_strnicmp(token.start, "partition", token.end - token.start) != 0) {
    return SQLITE_EMPTY;
  }

  // Check the next token is identifier and matches "key"
  rc = vec0_scanner_next(&scanner, &token);
  if (rc != VEC0_TOKEN_RESULT_SOME ||
      token.token_type != TOKEN_TYPE_IDENTIFIER) {
    return SQLITE_EMPTY;
  }
  if (sqlite3_strnicmp(token.start, "key", token.end - token.start) != 0) {
    return SQLITE_EMPTY;
  }

  *out_column_name = column_name;
  *out_column_name_length = column_name_length;
  *out_column_type = column_type;

  return SQLITE_OK;
}

/**
 * @brief Parse an argv[i] entry of a vec0 virtual table definition, and see if
 * it's an auxiliar column definition, ie `+[name] [type]` like `+contents text`
 *
 * @param source: argv[i] source string
 * @param source_length: length of the source string
 * @param out_column_name: If it is a partition key, the output column name. Same lifetime
 * as source, points to specific char *
 * @param out_column_name_length: Length of out_column_name in bytes
 * @param out_column_type: SQLITE_TEXT, SQLITE_INTEGER, SQLITE_FLOAT, or SQLITE_BLOB.
 * @return int: SQLITE_EMPTY if not an aux column, SQLITE_OK if it is.
 */
int vec0_parse_auxiliary_column_definition(const char *source, int source_length,
                                 char **out_column_name,
                                 int *out_column_name_length,
                                 int *out_column_type) {
  struct Vec0Scanner scanner;
  struct Vec0Token token;
  char *column_name;
  int column_name_length;
  int column_type;
  vec0_scanner_init(&scanner, source, source_length);

  // Check first token is '+', which denotes aux columns
  int rc = vec0_scanner_next(&scanner, &token);
  if (rc != VEC0_TOKEN_RESULT_SOME ||
      token.token_type != TOKEN_TYPE_PLUS) {
    return SQLITE_EMPTY;
  }

  rc = vec0_scanner_next(&scanner, &token);
  if (rc != VEC0_TOKEN_RESULT_SOME ||
      token.token_type != TOKEN_TYPE_IDENTIFIER) {
    return SQLITE_EMPTY;
  }

  column_name = token.start;
  column_name_length = token.end - token.start;

  // Check the next token matches "text" or "integer", as column type
  rc = vec0_scanner_next(&scanner, &token);
  if (rc != VEC0_TOKEN_RESULT_SOME ||
      token.token_type != TOKEN_TYPE_IDENTIFIER) {
    return SQLITE_EMPTY;
  }
  if (sqlite3_strnicmp(token.start, "text", token.end - token.start) == 0) {
    column_type = SQLITE_TEXT;
  } else if (sqlite3_strnicmp(token.start, "int", token.end - token.start) ==
                 0 ||
             sqlite3_strnicmp(token.start, "integer",
                              token.end - token.start) == 0) {
    column_type = SQLITE_INTEGER;
  } else if (sqlite3_strnicmp(token.start, "float", token.end - token.start) ==
                 0 ||
             sqlite3_strnicmp(token.start, "double",
                              token.end - token.start) == 0) {
    column_type = SQLITE_FLOAT;
  } else if (sqlite3_strnicmp(token.start, "blob", token.end - token.start) ==0) {
    column_type = SQLITE_BLOB;
  } else {
    return SQLITE_EMPTY;
  }

  *out_column_name = column_name;
  *out_column_name_length = column_name_length;
  *out_column_type = column_type;

  return SQLITE_OK;
}

typedef enum {
  VEC0_METADATA_COLUMN_KIND_BOOLEAN,
  VEC0_METADATA_COLUMN_KIND_INTEGER,
  VEC0_METADATA_COLUMN_KIND_FLOAT,
  VEC0_METADATA_COLUMN_KIND_TEXT,
  // future: blob, date, datetime
} vec0_metadata_column_kind;

/**
 * @brief Parse an argv[i] entry of a vec0 virtual table definition, and see if
 * it's an metadata column definition, ie `[name] [type]` like `is_released boolean`
 *
 * @param source: argv[i] source string
 * @param source_length: length of the source string
 * @param out_column_name: If it is a metadata column, the output column name. Same lifetime
 * as source, points to specific char *
 * @param out_column_name_length: Length of out_column_name in bytes
 * @param out_column_type: one of vec0_metadata_column_kind
 * @return int: SQLITE_EMPTY if not an metadata column, SQLITE_OK if it is.
 */
int vec0_parse_metadata_column_definition(const char *source, int source_length,
                                 char **out_column_name,
                                 int *out_column_name_length,
                                 vec0_metadata_column_kind *out_column_type) {
  struct Vec0Scanner scanner;
  struct Vec0Token token;
  char *column_name;
  int column_name_length;
  vec0_metadata_column_kind column_type;
  int rc;
  vec0_scanner_init(&scanner, source, source_length);

  rc = vec0_scanner_next(&scanner, &token);
  if (rc != VEC0_TOKEN_RESULT_SOME ||
      token.token_type != TOKEN_TYPE_IDENTIFIER) {
    return SQLITE_EMPTY;
  }

  column_name = token.start;
  column_name_length = token.end - token.start;

  // Check the next token matches a valid metadata type
  rc = vec0_scanner_next(&scanner, &token);
  if (rc != VEC0_TOKEN_RESULT_SOME ||
      token.token_type != TOKEN_TYPE_IDENTIFIER) {
    return SQLITE_EMPTY;
  }
  char * t = token.start;
  int n = token.end - token.start;
  if (sqlite3_strnicmp(t, "boolean", n) == 0 || sqlite3_strnicmp(t, "bool", n) == 0) {
    column_type = VEC0_METADATA_COLUMN_KIND_BOOLEAN;
  }else if (sqlite3_strnicmp(t, "int64", n) == 0 || sqlite3_strnicmp(t, "integer64", n) == 0 || sqlite3_strnicmp(t, "integer", n) == 0 || sqlite3_strnicmp(t, "int", n) == 0) {
    column_type = VEC0_METADATA_COLUMN_KIND_INTEGER;
  }else if (sqlite3_strnicmp(t, "float", n) == 0 || sqlite3_strnicmp(t, "double", n) == 0 || sqlite3_strnicmp(t, "float64", n) == 0 || sqlite3_strnicmp(t, "f64", n) == 0) {
    column_type = VEC0_METADATA_COLUMN_KIND_FLOAT;
  } else if (sqlite3_strnicmp(t, "text", n) == 0) {
    column_type = VEC0_METADATA_COLUMN_KIND_TEXT;
  } else {
    return SQLITE_EMPTY;
  }

  *out_column_name = column_name;
  *out_column_name_length = column_name_length;
  *out_column_type = column_type;

  return SQLITE_OK;
}

/**
 * @brief Parse an argv[i] entry of a vec0 virtual table definition, and see if
 * it's a PRIMARY KEY definition.
 *
 * @param source: argv[i] source string
 * @param source_length: length of the source string
 * @param out_column_name: If it is a PK, the output column name. Same lifetime
 * as source, points to specific char *
 * @param out_column_name_length: Length of out_column_name in bytes
 * @param out_column_type: SQLITE_TEXT or SQLITE_INTEGER.
 * @return int: SQLITE_EMPTY if not a PK, SQLITE_OK if it is.
 */
int vec0_parse_primary_key_definition(const char *source, int source_length,
                                 char **out_column_name,
                                 int *out_column_name_length,
                                 int *out_column_type) {
  struct Vec0Scanner scanner;
  struct Vec0Token token;
  char *column_name;
  int column_name_length;
  int column_type;
  vec0_scanner_init(&scanner, source, source_length);

  // Check first token is identifier, will be the column name
  int rc = vec0_scanner_next(&scanner, &token);
  if (rc != VEC0_TOKEN_RESULT_SOME ||
      token.token_type != TOKEN_TYPE_IDENTIFIER) {
    return SQLITE_EMPTY;
  }

  column_name = token.start;
  column_name_length = token.end - token.start;

  // Check the next token matches "text" or "integer", as column type
  rc = vec0_scanner_next(&scanner, &token);
  if (rc != VEC0_TOKEN_RESULT_SOME ||
      token.token_type != TOKEN_TYPE_IDENTIFIER) {
    return SQLITE_EMPTY;
  }
  if (sqlite3_strnicmp(token.start, "text", token.end - token.start) == 0) {
    column_type = SQLITE_TEXT;
  } else if (sqlite3_strnicmp(token.start, "int", token.end - token.start) ==
                 0 ||
             sqlite3_strnicmp(token.start, "integer",
                              token.end - token.start) == 0) {
    column_type = SQLITE_INTEGER;
  } else {
    return SQLITE_EMPTY;
  }

  // Check the next token is identifier and matches "primary"
  rc = vec0_scanner_next(&scanner, &token);
  if (rc != VEC0_TOKEN_RESULT_SOME ||
      token.token_type != TOKEN_TYPE_IDENTIFIER) {
    return SQLITE_EMPTY;
  }
  if (sqlite3_strnicmp(token.start, "primary", token.end - token.start) != 0) {
    return SQLITE_EMPTY;
  }

  // Check the next token is identifier and matches "key"
  rc = vec0_scanner_next(&scanner, &token);
  if (rc != VEC0_TOKEN_RESULT_SOME ||
      token.token_type != TOKEN_TYPE_IDENTIFIER) {
    return SQLITE_EMPTY;
  }
  if (sqlite3_strnicmp(token.start, "key", token.end - token.start) != 0) {
    return SQLITE_EMPTY;
  }

  *out_column_name = column_name;
  *out_column_name_length = column_name_length;
  *out_column_type = column_type;

  return SQLITE_OK;
}

enum Vec0DistanceMetrics {
  VEC0_DISTANCE_METRIC_L2 = 1,
  VEC0_DISTANCE_METRIC_COSINE = 2,
  VEC0_DISTANCE_METRIC_L1 = 3,
  VEC0_DISTANCE_METRIC_DOT = 4,
};

typedef f32 (*vec0_distance_fn)(const void *a, const void *b, const void *dims);

static f32 distance_l1_f32_as_f32(const void *a, const void *b, const void *d) {
  return (f32)distance_l1_f32(a, b, d);
}
static f32 distance_l1_int8_as_f32(const void *a, const void *b, const void *d) {
  return (f32)distance_l1_int8(a, b, d);
}

/**
 * The distance kernel for an element type and metric: the one table every
 * exact distance in vec0 goes through. Hot loops resolve it once.
 */
static vec0_distance_fn vec0_distance_resolve(enum VectorElementType elementType,
                                              enum Vec0DistanceMetrics metric) {
  switch (elementType) {
    case SQLITE_VEC_ELEMENT_TYPE_FLOAT32:
      switch (metric) {
        case VEC0_DISTANCE_METRIC_L2: return distance_l2_sqr_float;
        case VEC0_DISTANCE_METRIC_COSINE: return distance_cosine_float;
        case VEC0_DISTANCE_METRIC_L1: return distance_l1_f32_as_f32;
        case VEC0_DISTANCE_METRIC_DOT: return distance_dot_float;
      }
      break;
    case SQLITE_VEC_ELEMENT_TYPE_INT8:
      switch (metric) {
        case VEC0_DISTANCE_METRIC_L2: return distance_l2_sqr_int8;
        case VEC0_DISTANCE_METRIC_COSINE: return distance_cosine_int8;
        case VEC0_DISTANCE_METRIC_L1: return distance_l1_int8_as_f32;
        case VEC0_DISTANCE_METRIC_DOT: return distance_dot_int8;
      }
      break;
    case SQLITE_VEC_ELEMENT_TYPE_BIT:
      return distance_hamming;
  }
  return distance_l2_sqr_float;  // unreachable: types and metrics are validated
}

/**
 * Distance between two full-precision vectors of the given element type and
 * metric. Shared by every index implementation.
 */
static f32 vec0_distance_full(
    const void *a, const void *b, size_t dimensions,
    enum VectorElementType elementType,
    enum Vec0DistanceMetrics metric) {
  return vec0_distance_resolve(elementType, metric)(a, b, &dimensions);
}

enum Vec0IndexType {
  VEC0_INDEX_TYPE_FLAT = 1,
#if SQLITE_VEC_ENABLE_RESCORE
  VEC0_INDEX_TYPE_RESCORE = 2,
#endif
  VEC0_INDEX_TYPE_IVF = 3,
  VEC0_INDEX_TYPE_DISKANN = 4,
  VEC0_INDEX_TYPE_HNSW = 5,
};

#if SQLITE_VEC_ENABLE_RESCORE
enum Vec0RescoreQuantizerType {
  VEC0_RESCORE_QUANTIZER_BIT = 1,
  VEC0_RESCORE_QUANTIZER_INT8 = 2,
};

struct Vec0RescoreConfig {
  enum Vec0RescoreQuantizerType quantizer_type;
  int oversample;          // CREATE-time default
  int oversample_search;   // runtime override (0 = use default)
};
#endif

#if SQLITE_VEC_EXPERIMENTAL_IVF_ENABLE
enum Vec0IvfQuantizer {
  VEC0_IVF_QUANTIZER_NONE = 0,
  VEC0_IVF_QUANTIZER_INT8 = 1,
  VEC0_IVF_QUANTIZER_BINARY = 2,
  VEC0_IVF_QUANTIZER_RABITQ = 3,
  VEC0_IVF_QUANTIZER_FP16 = 4,
  VEC0_IVF_QUANTIZER_BF16 = 5,
  VEC0_IVF_QUANTIZER_INT4 = 6,
};

struct Vec0IvfConfig {
  int nlist;       // number of centroids (0 = deferred)
  int nprobe;      // cells to probe at query time
  int quantizer;   // VEC0_IVF_QUANTIZER_NONE / INT8 / BINARY
  int oversample;  // >= 1 (1 = no oversampling)
};
#else
struct Vec0IvfConfig { char _unused; };
#endif

// ============================================================
// DiskANN types and constants
// ============================================================

#define VEC0_DISKANN_DEFAULT_N_NEIGHBORS 72
#define VEC0_DISKANN_MAX_N_NEIGHBORS 256
#define VEC0_DISKANN_DEFAULT_SEARCH_LIST_SIZE 128
#define VEC0_DISKANN_DEFAULT_ALPHA 1.2f

/**
 * Quantizer type used for compressing neighbor vectors in the DiskANN graph.
 */
enum Vec0DiskannQuantizerType {
  VEC0_DISKANN_QUANTIZER_BINARY = 1,   // 1 bit per dimension (1/32 compression)
  VEC0_DISKANN_QUANTIZER_INT8   = 2,   // 1 byte per dimension (1/4 compression)
};

/**
 * Configuration for an HNSW index on a single vector column.
 * Parsed from `INDEXED BY hnsw(m=16, ef_construction=200, ef_search=64)`.
 */
struct Vec0HnswConfig {
  int m;                // max neighbors per node above level 0 (2*m at level 0)
  int ef_construction;  // beam width while inserting
  int ef_search;        // beam width while querying; at least k
  int cache_mb;         // per-connection vector/edge cache budget; 0 disables
  int quantizer;        // VEC0_HNSW_QUANTIZER_*: search on codes, rerank exactly
};

/**
 * Configuration for a DiskANN index on a single vector column.
 * Parsed from `INDEXED BY diskann(neighbor_quantizer=binary, n_neighbors=72)`.
 */
struct Vec0DiskannConfig {
  // Quantizer type for neighbor vectors
  enum Vec0DiskannQuantizerType quantizer_type;

  // Maximum number of neighbors per node (R in the paper). Must be divisible by 8.
  int n_neighbors;

  // Search list size (L in the paper) — unified default for both insert and query.
  int search_list_size;

  // Per-path overrides (0 = fall back to search_list_size).
  int search_list_size_search;
  int search_list_size_insert;

  // Alpha parameter for RobustPrune (distance scaling factor, typically 1.0-1.5)
  f32 alpha;

  // Buffer threshold for batched inserts. When > 0, inserts go into a flat
  // buffer table and are flushed into the graph when the buffer reaches this
  // size. 0 = disabled (legacy per-row insert behavior).
  int buffer_threshold;
};

/**
 * Represents a single candidate during greedy beam search.
 * Used in priority queues / sorted arrays during LM-Search.
 */
struct Vec0DiskannCandidate {
  i64 rowid;
  f32 distance;
  int visited;   // 1 if this candidate's neighbors have been explored
  int confirmed; // 1 if full-precision vector was successfully read (node exists)
};

/**
 * Returns the byte size of a quantized vector for the given quantizer type
 * and number of dimensions.
 */
size_t diskann_quantized_vector_byte_size(
    enum Vec0DiskannQuantizerType quantizer_type, size_t dimensions) {
  switch (quantizer_type) {
    case VEC0_DISKANN_QUANTIZER_BINARY:
      return dimensions / CHAR_BIT;  // 1 bit per dimension
    case VEC0_DISKANN_QUANTIZER_INT8:
      return dimensions * sizeof(i8);  // 1 byte per dimension
  }
  return 0;
}

struct VectorColumnDefinition {
  char *name;
  int name_length;
  size_t dimensions;
  enum VectorElementType element_type;
  enum Vec0DistanceMetrics distance_metric;
  enum Vec0IndexType index_type;
#if SQLITE_VEC_ENABLE_RESCORE
  struct Vec0RescoreConfig rescore;
#endif
  struct Vec0IvfConfig ivf;
  struct Vec0DiskannConfig diskann;
  struct Vec0HnswConfig hnsw;
};

/**
 * Compute pairwise distance between two vectors stored in the vec0 table's
 * native format, with the column's element type and metric.
 */
static f32 vec0_compute_distance(struct VectorColumnDefinition *vector_column,
                                 const void *a, const void *b) {
  return vec0_distance_full(a, b, vector_column->dimensions,
                            vector_column->element_type,
                            vector_column->distance_metric);
}

struct Vec0PartitionColumnDefinition {
  int type;
  char * name;
  int name_length;
};

struct Vec0AuxiliaryColumnDefinition {
  int type;
  char * name;
  int name_length;
};
struct Vec0MetadataColumnDefinition {
  vec0_metadata_column_kind kind;
  char * name;
  int name_length;
};

size_t vector_byte_size(enum VectorElementType element_type,
                        size_t dimensions) {
  switch (element_type) {
  case SQLITE_VEC_ELEMENT_TYPE_FLOAT32:
    return dimensions * sizeof(f32);
  case SQLITE_VEC_ELEMENT_TYPE_INT8:
    return dimensions * sizeof(i8);
  case SQLITE_VEC_ELEMENT_TYPE_BIT:
    return dimensions / CHAR_BIT;
  }
  return 0;
}

size_t vector_column_byte_size(struct VectorColumnDefinition column) {
  return vector_byte_size(column.element_type, column.dimensions);
}

#if SQLITE_VEC_ENABLE_RESCORE
/**
 * @brief Parse rescore options from an "INDEXED BY rescore(...)" clause.
 *
 * @param scanner Scanner positioned right after the opening '(' of rescore(...)
 * @param outConfig Output rescore config
 * @param pzErr Error message output
 * @return int SQLITE_OK on success, SQLITE_ERROR on error.
 */
static int vec0_parse_rescore_options(struct Vec0Scanner *scanner,
                                      struct Vec0RescoreConfig *outConfig,
                                      char **pzErr) {
  struct Vec0Token token;
  int rc;
  int hasQuantizer = 0;
  outConfig->oversample = 8;
  outConfig->quantizer_type = 0;

  while (1) {
    rc = vec0_scanner_next(scanner, &token);
    if (rc == VEC0_TOKEN_RESULT_EOF) {
      break;
    }
    // ')' closes rescore options
    if (rc == VEC0_TOKEN_RESULT_SOME && token.token_type == TOKEN_TYPE_RPAREN) {
      break;
    }
    if (rc != VEC0_TOKEN_RESULT_SOME || token.token_type != TOKEN_TYPE_IDENTIFIER) {
      *pzErr = sqlite3_mprintf("Expected option name in rescore(...)");
      return SQLITE_ERROR;
    }

    char *key = token.start;
    int keyLength = token.end - token.start;

    // expect '='
    rc = vec0_scanner_next(scanner, &token);
    if (rc != VEC0_TOKEN_RESULT_SOME || token.token_type != TOKEN_TYPE_EQ) {
      *pzErr = sqlite3_mprintf("Expected '=' after option name in rescore(...)");
      return SQLITE_ERROR;
    }

    // value
    rc = vec0_scanner_next(scanner, &token);
    if (rc != VEC0_TOKEN_RESULT_SOME) {
      *pzErr = sqlite3_mprintf("Expected value after '=' in rescore(...)");
      return SQLITE_ERROR;
    }

    if (sqlite3_strnicmp(key, "quantizer", keyLength) == 0) {
      if (token.token_type != TOKEN_TYPE_IDENTIFIER) {
        *pzErr = sqlite3_mprintf("Expected identifier for quantizer value in rescore(...)");
        return SQLITE_ERROR;
      }
      int valLen = token.end - token.start;
      if (sqlite3_strnicmp(token.start, "bit", valLen) == 0) {
        outConfig->quantizer_type = VEC0_RESCORE_QUANTIZER_BIT;
      } else if (sqlite3_strnicmp(token.start, "int8", valLen) == 0) {
        outConfig->quantizer_type = VEC0_RESCORE_QUANTIZER_INT8;
      } else {
        *pzErr = sqlite3_mprintf("Unknown quantizer type '%.*s' in rescore(...). Expected 'bit' or 'int8'.", valLen, token.start);
        return SQLITE_ERROR;
      }
      hasQuantizer = 1;
    } else if (sqlite3_strnicmp(key, "oversample", keyLength) == 0) {
      if (token.token_type != TOKEN_TYPE_DIGIT) {
        *pzErr = sqlite3_mprintf("Expected integer for oversample value in rescore(...)");
        return SQLITE_ERROR;
      }
      outConfig->oversample = atoi(token.start);
      if (outConfig->oversample <= 0 || outConfig->oversample > 128) {
        *pzErr = sqlite3_mprintf("oversample in rescore(...) must be between 1 and 128, got %d", outConfig->oversample);
        return SQLITE_ERROR;
      }
    } else {
      *pzErr = sqlite3_mprintf("Unknown option '%.*s' in rescore(...)", keyLength, key);
      return SQLITE_ERROR;
    }

    // optional comma between options
    rc = vec0_scanner_next(scanner, &token);
    if (rc == VEC0_TOKEN_RESULT_EOF) {
      break;
    }
    if (rc == VEC0_TOKEN_RESULT_SOME && token.token_type == TOKEN_TYPE_RPAREN) {
      break;
    }
    if (rc == VEC0_TOKEN_RESULT_SOME && token.token_type == TOKEN_TYPE_COMMA) {
      continue;
    }
    // If it's not a comma or rparen, it might be the next key — push back isn't
    // possible with this scanner, so we'll treat unexpected tokens as errors
    *pzErr = sqlite3_mprintf("Unexpected token in rescore(...) options");
    return SQLITE_ERROR;
  }

  if (!hasQuantizer) {
    *pzErr = sqlite3_mprintf("rescore(...) requires a 'quantizer' option (quantizer=bit or quantizer=int8)");
    return SQLITE_ERROR;
  }

  return SQLITE_OK;
}
#endif /* SQLITE_VEC_ENABLE_RESCORE */

/**
 * @brief Parse an vec0 vtab argv[i] column definition and see if
 * it's a vector column defintion, ex `contents_embedding float[768]`.
 *
 * @param source vec0 argv[i] item
 * @param source_length length of source in bytes
 * @param outColumn Output the parse vector column to this struct, if success
 * @return int SQLITE_OK on success, SQLITE_EMPTY is it's not a vector column
 * definition, SQLITE_ERROR on error.
 */
#if SQLITE_VEC_EXPERIMENTAL_IVF_ENABLE
// Forward declaration — defined in sqlite-vec-ivf.c
static int vec0_parse_ivf_options(struct Vec0Scanner *scanner,
                                   struct Vec0IvfConfig *config);
#endif
#if SQLITE_VEC_ENABLE_HNSW
// Forward declaration — defined in sqlite-vec-hnsw.c
static int vec0_parse_hnsw_options(struct Vec0Scanner *scanner,
                                   struct Vec0HnswConfig *config);
#endif

/**
 * Parse the options inside diskann(...) parentheses.
 * Scanner should be positioned right before the '(' token.
 *
 * Recognized options:
 *   neighbor_quantizer = binary | int8       (required)
 *   n_neighbors = <integer>                  (optional, default 72)
 *   search_list_size = <integer>             (optional, default 128)
 */
static int vec0_parse_diskann_options(struct Vec0Scanner *scanner,
                                       struct Vec0DiskannConfig *config) {
  int rc;
  struct Vec0Token token;
  int hasQuantizer = 0;

  // Set defaults
  config->n_neighbors = VEC0_DISKANN_DEFAULT_N_NEIGHBORS;
  config->search_list_size = VEC0_DISKANN_DEFAULT_SEARCH_LIST_SIZE;
  config->search_list_size_search = 0;
  config->search_list_size_insert = 0;
  config->alpha = VEC0_DISKANN_DEFAULT_ALPHA;
  config->buffer_threshold = 0;
  int hasSearchListSize = 0;
  int hasSearchListSizeSplit = 0;

  // Expect '('
  rc = vec0_scanner_next(scanner, &token);
  if (rc != VEC0_TOKEN_RESULT_SOME || token.token_type != TOKEN_TYPE_LPAREN) {
    return SQLITE_ERROR;
  }

  while (1) {
    // key
    rc = vec0_scanner_next(scanner, &token);
    if (rc == VEC0_TOKEN_RESULT_SOME && token.token_type == TOKEN_TYPE_RPAREN) {
      break;  // empty parens or trailing comma
    }
    if (rc != VEC0_TOKEN_RESULT_SOME || token.token_type != TOKEN_TYPE_IDENTIFIER) {
      return SQLITE_ERROR;
    }
    char *optKey = token.start;
    int optKeyLen = token.end - token.start;

    // '='
    rc = vec0_scanner_next(scanner, &token);
    if (rc != VEC0_TOKEN_RESULT_SOME || token.token_type != TOKEN_TYPE_EQ) {
      return SQLITE_ERROR;
    }

    // value (identifier or digit)
    rc = vec0_scanner_next(scanner, &token);
    if (rc != VEC0_TOKEN_RESULT_SOME) {
      return SQLITE_ERROR;
    }
    char *optVal = token.start;
    int optValLen = token.end - token.start;

    if (sqlite3_strnicmp(optKey, "neighbor_quantizer", optKeyLen) == 0) {
      if (sqlite3_strnicmp(optVal, "binary", optValLen) == 0) {
        config->quantizer_type = VEC0_DISKANN_QUANTIZER_BINARY;
      } else if (sqlite3_strnicmp(optVal, "int8", optValLen) == 0) {
        config->quantizer_type = VEC0_DISKANN_QUANTIZER_INT8;
      } else {
        return SQLITE_ERROR;  // unknown quantizer
      }
      hasQuantizer = 1;
    } else if (sqlite3_strnicmp(optKey, "n_neighbors", optKeyLen) == 0) {
      config->n_neighbors = atoi(optVal);
      if (config->n_neighbors <= 0 || (config->n_neighbors % 8) != 0 ||
          config->n_neighbors > VEC0_DISKANN_MAX_N_NEIGHBORS) {
        return SQLITE_ERROR;
      }
    } else if (sqlite3_strnicmp(optKey, "search_list_size_search", optKeyLen) == 0 && optKeyLen == 23) {
      config->search_list_size_search = atoi(optVal);
      if (config->search_list_size_search <= 0) {
        return SQLITE_ERROR;
      }
      hasSearchListSizeSplit = 1;
    } else if (sqlite3_strnicmp(optKey, "search_list_size_insert", optKeyLen) == 0 && optKeyLen == 23) {
      config->search_list_size_insert = atoi(optVal);
      if (config->search_list_size_insert <= 0) {
        return SQLITE_ERROR;
      }
      hasSearchListSizeSplit = 1;
    } else if (sqlite3_strnicmp(optKey, "search_list_size", optKeyLen) == 0) {
      config->search_list_size = atoi(optVal);
      if (config->search_list_size <= 0) {
        return SQLITE_ERROR;
      }
      hasSearchListSize = 1;
    } else if (sqlite3_strnicmp(optKey, "buffer_threshold", optKeyLen) == 0) {
      config->buffer_threshold = atoi(optVal);
      if (config->buffer_threshold < 0) {
        return SQLITE_ERROR;
      }
    } else {
      return SQLITE_ERROR;  // unknown option
    }

    // Expect ',' or ')'
    rc = vec0_scanner_next(scanner, &token);
    if (rc == VEC0_TOKEN_RESULT_SOME && token.token_type == TOKEN_TYPE_RPAREN) {
      break;
    }
    if (rc != VEC0_TOKEN_RESULT_SOME || token.token_type != TOKEN_TYPE_COMMA) {
      return SQLITE_ERROR;
    }
  }

  if (!hasQuantizer) {
    return SQLITE_ERROR;  // neighbor_quantizer is required
  }

  if (hasSearchListSize && hasSearchListSizeSplit) {
    return SQLITE_ERROR;  // cannot mix search_list_size with search_list_size_search/insert
  }

  return SQLITE_OK;
}

int vec0_parse_vector_column(const char *source, int source_length,
                        struct VectorColumnDefinition *outColumn) {
  // parses a vector column definition like so:
  // "abc float[123]", "abc_123 bit[1234]", eetc.
  // https://github.com/asg017/sqlite-vec/issues/46
  int rc;
  struct Vec0Scanner scanner;
  struct Vec0Token token;

  char *name;
  int nameLength;
  enum VectorElementType elementType;
  enum Vec0DistanceMetrics distanceMetric = VEC0_DISTANCE_METRIC_L2;
  enum Vec0IndexType indexType = VEC0_INDEX_TYPE_FLAT;
#if SQLITE_VEC_ENABLE_RESCORE
  struct Vec0RescoreConfig rescoreConfig;
  memset(&rescoreConfig, 0, sizeof(rescoreConfig));
#endif
  struct Vec0IvfConfig ivfConfig;
  memset(&ivfConfig, 0, sizeof(ivfConfig));
  struct Vec0DiskannConfig diskannConfig;
  memset(&diskannConfig, 0, sizeof(diskannConfig));
  struct Vec0HnswConfig hnswConfig;
  memset(&hnswConfig, 0, sizeof(hnswConfig));
  int dimensions;
  vec0_scanner_init(&scanner, source, source_length);

  // starts with an identifier
  rc = vec0_scanner_next(&scanner, &token);

  if (rc != VEC0_TOKEN_RESULT_SOME ||
      token.token_type != TOKEN_TYPE_IDENTIFIER) {
    return SQLITE_EMPTY;
  }

  name = token.start;
  nameLength = token.end - token.start;

  // vector column type comes next: float, int, or bit
  rc = vec0_scanner_next(&scanner, &token);

  if (rc != VEC0_TOKEN_RESULT_SOME ||
      token.token_type != TOKEN_TYPE_IDENTIFIER) {
    return SQLITE_EMPTY;
  }
  // Match the full identifier, not just a prefix: `sqlite3_strnicmp` only
  // compares the given number of bytes, so a bare prefix check would coerce
  // typos and lookalikes (e.g. `float16`, `bitcoin`) to a real type instead of
  // rejecting them.
  const int typeLength = token.end - token.start;
  if ((typeLength == 5 && sqlite3_strnicmp(token.start, "float", 5) == 0) ||
      (typeLength == 7 && sqlite3_strnicmp(token.start, "float32", 7) == 0) ||
      (typeLength == 3 && sqlite3_strnicmp(token.start, "f32", 3) == 0)) {
    elementType = SQLITE_VEC_ELEMENT_TYPE_FLOAT32;
  } else if ((typeLength == 4 && sqlite3_strnicmp(token.start, "int8", 4) == 0) ||
             (typeLength == 2 && sqlite3_strnicmp(token.start, "i8", 2) == 0)) {
    elementType = SQLITE_VEC_ELEMENT_TYPE_INT8;
  } else if (typeLength == 3 && sqlite3_strnicmp(token.start, "bit", 3) == 0) {
    elementType = SQLITE_VEC_ELEMENT_TYPE_BIT;
  } else {
    return SQLITE_EMPTY;
  }

  // left '[' bracket
  rc = vec0_scanner_next(&scanner, &token);
  if (rc != VEC0_TOKEN_RESULT_SOME || token.token_type != TOKEN_TYPE_LBRACKET) {
    return SQLITE_EMPTY;
  }

  // digit, for vector dimension length
  rc = vec0_scanner_next(&scanner, &token);
  if (rc != VEC0_TOKEN_RESULT_SOME || token.token_type != TOKEN_TYPE_DIGIT) {
    return SQLITE_ERROR;
  }
  dimensions = atoi(token.start);
  if (dimensions <= 0) {
    return SQLITE_ERROR;
  }

  // // right ']' bracket
  rc = vec0_scanner_next(&scanner, &token);
  if (rc != VEC0_TOKEN_RESULT_SOME || token.token_type != TOKEN_TYPE_RBRACKET) {
    return SQLITE_ERROR;
  }

  // any other tokens left should be column-level options , ex `key=value`
  // ex `distance_metric=L2 distance_metric=cosine` should error
  while (1) {
    // should be EOF or identifier (option key)
    rc = vec0_scanner_next(&scanner, &token);
    if (rc == VEC0_TOKEN_RESULT_EOF) {
      break;
    }

    if (rc != VEC0_TOKEN_RESULT_SOME ||
        token.token_type != TOKEN_TYPE_IDENTIFIER) {
      return SQLITE_ERROR;
    }

    char *key = token.start;
    int keyLength = token.end - token.start;

    if (sqlite3_strnicmp(key, "distance_metric", keyLength) == 0) {

      if (elementType == SQLITE_VEC_ELEMENT_TYPE_BIT) {
        return SQLITE_ERROR;
      }
      // ensure equal sign after distance_metric
      rc = vec0_scanner_next(&scanner, &token);
      if (rc != VEC0_TOKEN_RESULT_SOME || token.token_type != TOKEN_TYPE_EQ) {
        return SQLITE_ERROR;
      }

      // distance_metric value, an identifier (L2, cosine, etc)
      rc = vec0_scanner_next(&scanner, &token);
      if (rc != VEC0_TOKEN_RESULT_SOME ||
          token.token_type != TOKEN_TYPE_IDENTIFIER) {
        return SQLITE_ERROR;
      }

      char *value = token.start;
      int valueLength = token.end - token.start;
      if (sqlite3_strnicmp(value, "l2", valueLength) == 0) {
        distanceMetric = VEC0_DISTANCE_METRIC_L2;
      } else if (sqlite3_strnicmp(value, "l1", valueLength) == 0) {
        distanceMetric = VEC0_DISTANCE_METRIC_L1;
      } else if (sqlite3_strnicmp(value, "cosine", valueLength) == 0) {
        distanceMetric = VEC0_DISTANCE_METRIC_COSINE;
      } else if (sqlite3_strnicmp(value, "dot", valueLength) == 0) {
        distanceMetric = VEC0_DISTANCE_METRIC_DOT;
      } else {
        return SQLITE_ERROR;
      }
    }
    // INDEXED BY flat() | rescore(...)
    else if (sqlite3_strnicmp(key, "indexed", keyLength) == 0) {
      // expect "by"
      rc = vec0_scanner_next(&scanner, &token);
      if (rc != VEC0_TOKEN_RESULT_SOME ||
          token.token_type != TOKEN_TYPE_IDENTIFIER ||
          sqlite3_strnicmp(token.start, "by", token.end - token.start) != 0) {
        return SQLITE_ERROR;
      }
      // expect index type name
      rc = vec0_scanner_next(&scanner, &token);
      if (rc != VEC0_TOKEN_RESULT_SOME ||
          token.token_type != TOKEN_TYPE_IDENTIFIER) {
        return SQLITE_ERROR;
      }
      int indexNameLen = token.end - token.start;
      if (sqlite3_strnicmp(token.start, "flat", indexNameLen) == 0) {
        indexType = VEC0_INDEX_TYPE_FLAT;
        // expect '('
        rc = vec0_scanner_next(&scanner, &token);
        if (rc != VEC0_TOKEN_RESULT_SOME ||
            token.token_type != TOKEN_TYPE_LPAREN) {
          return SQLITE_ERROR;
        }
        // expect ')'
        rc = vec0_scanner_next(&scanner, &token);
        if (rc != VEC0_TOKEN_RESULT_SOME ||
            token.token_type != TOKEN_TYPE_RPAREN) {
          return SQLITE_ERROR;
        }
      }
#if SQLITE_VEC_ENABLE_RESCORE
      else if (sqlite3_strnicmp(token.start, "rescore", indexNameLen) == 0) {
        indexType = VEC0_INDEX_TYPE_RESCORE;
        if (elementType != SQLITE_VEC_ELEMENT_TYPE_FLOAT32) {
          return SQLITE_ERROR;
        }
        // expect '('
        rc = vec0_scanner_next(&scanner, &token);
        if (rc != VEC0_TOKEN_RESULT_SOME || token.token_type != TOKEN_TYPE_LPAREN) {
          return SQLITE_ERROR;
        }
        char *rescoreErr = NULL;
        rc = vec0_parse_rescore_options(&scanner, &rescoreConfig, &rescoreErr);
        if (rc != SQLITE_OK) {
          if (rescoreErr) sqlite3_free(rescoreErr);
          return SQLITE_ERROR;
        }
        // validate dimensions for bit quantizer
        if (rescoreConfig.quantizer_type == VEC0_RESCORE_QUANTIZER_BIT &&
            (dimensions % CHAR_BIT) != 0) {
          return SQLITE_ERROR;
        }
      }
#endif
      else if (sqlite3_strnicmp(token.start, "ivf", indexNameLen) == 0) {
#if SQLITE_VEC_EXPERIMENTAL_IVF_ENABLE
        indexType = VEC0_INDEX_TYPE_IVF;
        if (elementType != SQLITE_VEC_ELEMENT_TYPE_FLOAT32) {
          return SQLITE_ERROR;
        }
        memset(&ivfConfig, 0, sizeof(ivfConfig));
        rc = vec0_parse_ivf_options(&scanner, &ivfConfig);
        if (rc != SQLITE_OK) {
          return SQLITE_ERROR;
        }
        if ((ivfConfig.quantizer == VEC0_IVF_QUANTIZER_BINARY ||
             ivfConfig.quantizer == VEC0_IVF_QUANTIZER_RABITQ) && (dimensions % 8) != 0) {
          return SQLITE_ERROR;
        }
        if (ivfConfig.quantizer == VEC0_IVF_QUANTIZER_INT4 && (dimensions % 2) != 0) {
          return SQLITE_ERROR;
        }
#else
        return SQLITE_ERROR; // IVF not compiled in
#endif
      } else if (sqlite3_strnicmp(token.start, "diskann", indexNameLen) == 0) {
#if SQLITE_VEC_ENABLE_DISKANN
        indexType = VEC0_INDEX_TYPE_DISKANN;
        rc = vec0_parse_diskann_options(&scanner, &diskannConfig);
        if (rc != SQLITE_OK) {
          return rc;
        }
#else
        return SQLITE_ERROR;
#endif
      } else if (indexNameLen == 4 && sqlite3_strnicmp(token.start, "hnsw", 4) == 0) {
#if SQLITE_VEC_ENABLE_HNSW
        indexType = VEC0_INDEX_TYPE_HNSW;
        rc = vec0_parse_hnsw_options(&scanner, &hnswConfig);
        if (rc != SQLITE_OK) {
          return rc;
        }
#else
        return SQLITE_ERROR;
#endif
      } else {
        // unknown index type
        return SQLITE_ERROR;
      }
    }
    // unknown key
    else {
      return SQLITE_ERROR;
    }
  }

  outColumn->name = sqlite3_mprintf("%.*s", nameLength, name);
  if (!outColumn->name) {
    return SQLITE_ERROR;
  }
  outColumn->name_length = nameLength;
#if SQLITE_VEC_EXPERIMENTAL_IVF_ENABLE
  // RaBitQ estimates inner products, which covers L2 and cosine but not L1
  if (indexType == VEC0_INDEX_TYPE_IVF &&
      ivfConfig.quantizer == VEC0_IVF_QUANTIZER_RABITQ &&
      distanceMetric == VEC0_DISTANCE_METRIC_L1) {
    return SQLITE_ERROR;
  }
#endif
  outColumn->distance_metric = distanceMetric;
  outColumn->element_type = elementType;
  outColumn->dimensions = dimensions;
  outColumn->index_type = indexType;
#if SQLITE_VEC_ENABLE_RESCORE
  outColumn->rescore = rescoreConfig;
#endif
  outColumn->ivf = ivfConfig;
  outColumn->diskann = diskannConfig;
  outColumn->hnsw = hnswConfig;
  return SQLITE_OK;
}

#pragma region vec_each table function

typedef struct vec_each_vtab vec_each_vtab;
struct vec_each_vtab {
  sqlite3_vtab base;
};

typedef struct vec_each_cursor vec_each_cursor;
struct vec_each_cursor {
  sqlite3_vtab_cursor base;
  i64 iRowid;
  enum VectorElementType vector_type;
  void *vector;
  size_t dimensions;
  vector_cleanup cleanup;
};

static int vec_eachConnect(sqlite3 *db, void *pAux, int argc,
                           const char *const *argv, sqlite3_vtab **ppVtab,
                           char **pzErr) {
  UNUSED_PARAMETER(pAux);
  UNUSED_PARAMETER(argc);
  UNUSED_PARAMETER(argv);
  UNUSED_PARAMETER(pzErr);
  vec_each_vtab *pNew;
  int rc;

  rc = sqlite3_declare_vtab(db, "CREATE TABLE x(value, vector hidden)");
#define VEC_EACH_COLUMN_VALUE 0
#define VEC_EACH_COLUMN_VECTOR 1
  if (rc == SQLITE_OK) {
    pNew = sqlite3_malloc(sizeof(*pNew));
    *ppVtab = (sqlite3_vtab *)pNew;
    if (pNew == 0)
      return SQLITE_NOMEM;
    memset(pNew, 0, sizeof(*pNew));
  }
  return rc;
}

static int vec_eachDisconnect(sqlite3_vtab *pVtab) {
  vec_each_vtab *p = (vec_each_vtab *)pVtab;
  sqlite3_free(p);
  return SQLITE_OK;
}

static int vec_eachOpen(sqlite3_vtab *p, sqlite3_vtab_cursor **ppCursor) {
  UNUSED_PARAMETER(p);
  vec_each_cursor *pCur;
  pCur = sqlite3_malloc(sizeof(*pCur));
  if (pCur == 0)
    return SQLITE_NOMEM;
  memset(pCur, 0, sizeof(*pCur));
  *ppCursor = &pCur->base;
  return SQLITE_OK;
}

static int vec_eachClose(sqlite3_vtab_cursor *cur) {
  vec_each_cursor *pCur = (vec_each_cursor *)cur;
  if(pCur->vector) {
    pCur->cleanup(pCur->vector);
  }
  sqlite3_free(pCur);
  return SQLITE_OK;
}

static int vec_eachBestIndex(sqlite3_vtab *pVTab,
                             sqlite3_index_info *pIdxInfo) {
  UNUSED_PARAMETER(pVTab);
  int hasVector = 0;
  for (int i = 0; i < pIdxInfo->nConstraint; i++) {
    const struct sqlite3_index_constraint *pCons = &pIdxInfo->aConstraint[i];
    // printf("i=%d iColumn=%d, op=%d, usable=%d\n", i, pCons->iColumn,
    // pCons->op, pCons->usable);
    switch (pCons->iColumn) {
    case VEC_EACH_COLUMN_VECTOR: {
      if (pCons->op == SQLITE_INDEX_CONSTRAINT_EQ && pCons->usable) {
        hasVector = 1;
        pIdxInfo->aConstraintUsage[i].argvIndex = 1;
        pIdxInfo->aConstraintUsage[i].omit = 1;
      }
      break;
    }
    }
  }
  if (!hasVector) {
    return SQLITE_CONSTRAINT;
  }

  pIdxInfo->estimatedCost = (double)100000;
  pIdxInfo->estimatedRows = 100000;

  return SQLITE_OK;
}

static int vec_eachFilter(sqlite3_vtab_cursor *pVtabCursor, int idxNum,
                          const char *idxStr, int argc, sqlite3_value **argv) {
  UNUSED_PARAMETER(idxNum);
  UNUSED_PARAMETER(idxStr);
  assert(argc == 1);
  vec_each_cursor *pCur = (vec_each_cursor *)pVtabCursor;

  if (pCur->vector) {
    pCur->cleanup(pCur->vector);
    pCur->vector = NULL;
  }

  char *pzErrMsg;
  int rc = vector_from_value(argv[0], &pCur->vector, &pCur->dimensions,
                             &pCur->vector_type, &pCur->cleanup, &pzErrMsg);
  if (rc != SQLITE_OK) {
    sqlite3_free(pzErrMsg);
    return SQLITE_ERROR;
  }
  pCur->iRowid = 0;
  return SQLITE_OK;
}

static int vec_eachRowid(sqlite3_vtab_cursor *cur, sqlite_int64 *pRowid) {
  vec_each_cursor *pCur = (vec_each_cursor *)cur;
  *pRowid = pCur->iRowid;
  return SQLITE_OK;
}

static int vec_eachEof(sqlite3_vtab_cursor *cur) {
  vec_each_cursor *pCur = (vec_each_cursor *)cur;
  return pCur->iRowid >= (i64)pCur->dimensions;
}

static int vec_eachNext(sqlite3_vtab_cursor *cur) {
  vec_each_cursor *pCur = (vec_each_cursor *)cur;
  pCur->iRowid++;
  return SQLITE_OK;
}

static int vec_eachColumn(sqlite3_vtab_cursor *cur, sqlite3_context *context,
                          int i) {
  vec_each_cursor *pCur = (vec_each_cursor *)cur;
  switch (i) {
  case VEC_EACH_COLUMN_VALUE:
    switch (pCur->vector_type) {
    case SQLITE_VEC_ELEMENT_TYPE_FLOAT32: {
      sqlite3_result_double(context, ((f32 *)pCur->vector)[pCur->iRowid]);
      break;
    }
    case SQLITE_VEC_ELEMENT_TYPE_BIT: {
      u8 x = ((u8 *)pCur->vector)[pCur->iRowid / CHAR_BIT];
      sqlite3_result_int(context,
                         (x >> (pCur->iRowid % CHAR_BIT)) & 1);
      break;
    }
    case SQLITE_VEC_ELEMENT_TYPE_INT8: {
      sqlite3_result_int(context, ((i8 *)pCur->vector)[pCur->iRowid]);
      break;
    }
    }

    break;
  }
  return SQLITE_OK;
}

static sqlite3_module vec_eachModule = {
    /* iVersion    */ 0,
    /* xCreate     */ 0,
    /* xConnect    */ vec_eachConnect,
    /* xBestIndex  */ vec_eachBestIndex,
    /* xDisconnect */ vec_eachDisconnect,
    /* xDestroy    */ 0,
    /* xOpen       */ vec_eachOpen,
    /* xClose      */ vec_eachClose,
    /* xFilter     */ vec_eachFilter,
    /* xNext       */ vec_eachNext,
    /* xEof        */ vec_eachEof,
    /* xColumn     */ vec_eachColumn,
    /* xRowid      */ vec_eachRowid,
    /* xUpdate     */ 0,
    /* xBegin      */ 0,
    /* xSync       */ 0,
    /* xCommit     */ 0,
    /* xRollback   */ 0,
    /* xFindMethod */ 0,
    /* xRename     */ 0,
    /* xSavepoint  */ 0,
    /* xRelease    */ 0,
    /* xRollbackTo */ 0,
    /* xShadowName */ 0,
#if SQLITE_VERSION_NUMBER >= 3044000
    /* xIntegrity  */ 0
#endif
};

#pragma endregion

#pragma region vec_file_each table function

// Reads the standard ".fvecs" / ".bvecs" / ".ivecs" vector file formats
// (as used by the SIFT/ANN benchmark datasets): a sequence of records,
// each an int32 dimension count followed by that many little-endian
// elements (float32, uint8, or int32 respectively).

typedef struct vec_file_each_vtab vec_file_each_vtab;
struct vec_file_each_vtab {
  sqlite3_vtab base;
};

typedef struct vec_file_each_cursor vec_file_each_cursor;
struct vec_file_each_cursor {
  sqlite3_vtab_cursor base;
  FILE *file;
  i64 iRowid;
  i64 nRecords;
  i32 dimensions;
  size_t elementSize;
  int isFloat32;
  void *vector;
};

static int vec_file_eachConnect(sqlite3 *db, void *pAux, int argc,
                                const char *const *argv, sqlite3_vtab **ppVtab,
                                char **pzErr) {
  UNUSED_PARAMETER(pAux);
  UNUSED_PARAMETER(argc);
  UNUSED_PARAMETER(argv);
  UNUSED_PARAMETER(pzErr);
  vec_file_each_vtab *pNew;
  int rc;

  rc = sqlite3_declare_vtab(db, "CREATE TABLE x(vector, path hidden)");
#define VEC_FILE_EACH_COLUMN_VECTOR 0
#define VEC_FILE_EACH_COLUMN_PATH 1
  if (rc == SQLITE_OK) {
    pNew = sqlite3_malloc(sizeof(*pNew));
    *ppVtab = (sqlite3_vtab *)pNew;
    if (pNew == 0)
      return SQLITE_NOMEM;
    memset(pNew, 0, sizeof(*pNew));
  }
  return rc;
}

static int vec_file_eachDisconnect(sqlite3_vtab *pVtab) {
  sqlite3_free(pVtab);
  return SQLITE_OK;
}

static int vec_file_eachOpen(sqlite3_vtab *p, sqlite3_vtab_cursor **ppCursor) {
  UNUSED_PARAMETER(p);
  vec_file_each_cursor *pCur;
  pCur = sqlite3_malloc(sizeof(*pCur));
  if (pCur == 0)
    return SQLITE_NOMEM;
  memset(pCur, 0, sizeof(*pCur));
  *ppCursor = &pCur->base;
  return SQLITE_OK;
}

static int vec_file_eachClose(sqlite3_vtab_cursor *cur) {
  vec_file_each_cursor *pCur = (vec_file_each_cursor *)cur;
  if (pCur->file) {
    fclose(pCur->file);
  }
  sqlite3_free(pCur->vector);
  sqlite3_free(pCur);
  return SQLITE_OK;
}

static int vec_file_eachBestIndex(sqlite3_vtab *pVTab,
                                  sqlite3_index_info *pIdxInfo) {
  UNUSED_PARAMETER(pVTab);
  int hasPath = 0;
  for (int i = 0; i < pIdxInfo->nConstraint; i++) {
    const struct sqlite3_index_constraint *pCons = &pIdxInfo->aConstraint[i];
    if (pCons->iColumn == VEC_FILE_EACH_COLUMN_PATH &&
        pCons->op == SQLITE_INDEX_CONSTRAINT_EQ && pCons->usable) {
      hasPath = 1;
      pIdxInfo->aConstraintUsage[i].argvIndex = 1;
      pIdxInfo->aConstraintUsage[i].omit = 1;
    }
  }
  if (!hasPath) {
    return SQLITE_CONSTRAINT;
  }
  pIdxInfo->estimatedCost = (double)100000;
  pIdxInfo->estimatedRows = 100000;
  return SQLITE_OK;
}

static void vec_file_each_set_error(vec_file_each_cursor *pCur,
                                    const char *fmt, const char *arg) {
  sqlite3_free(pCur->base.pVtab->zErrMsg);
  pCur->base.pVtab->zErrMsg = sqlite3_mprintf(fmt, arg);
}

// Read the record under the cursor's file position into pCur->vector.
// Returns SQLITE_OK, SQLITE_DONE at a clean record boundary, or
// SQLITE_ERROR (with zErrMsg set) on a malformed file.
static int vec_file_each_read_record(vec_file_each_cursor *pCur) {
  i32 dim;
  if (fread(&dim, sizeof(i32), 1, pCur->file) != 1) {
    if (feof(pCur->file)) {
      return SQLITE_DONE;
    }
    vec_file_each_set_error(pCur, "error reading vector file: %s",
                            strerror(errno));
    return SQLITE_ERROR;
  }
  if (dim != pCur->dimensions) {
    sqlite3_free(pCur->base.pVtab->zErrMsg);
    pCur->base.pVtab->zErrMsg = sqlite3_mprintf(
        "vector file record %lld has dimension %d, expected %d",
        (long long)pCur->iRowid, dim, pCur->dimensions);
    return SQLITE_ERROR;
  }
  size_t n = (size_t)pCur->dimensions * pCur->elementSize;
  if (n > 0 && fread(pCur->vector, 1, n, pCur->file) != n) {
    sqlite3_free(pCur->base.pVtab->zErrMsg);
    pCur->base.pVtab->zErrMsg = sqlite3_mprintf(
        "truncated record in vector file at record %lld",
        (long long)pCur->iRowid);
    return SQLITE_ERROR;
  }
  return SQLITE_OK;
}

static int vec_file_eachFilter(sqlite3_vtab_cursor *pVtabCursor, int idxNum,
                               const char *idxStr, int argc,
                               sqlite3_value **argv) {
  UNUSED_PARAMETER(idxNum);
  UNUSED_PARAMETER(idxStr);
  assert(argc == 1);
  vec_file_each_cursor *pCur = (vec_file_each_cursor *)pVtabCursor;

  if (pCur->file) {
    fclose(pCur->file);
    pCur->file = NULL;
  }
  sqlite3_free(pCur->vector);
  pCur->vector = NULL;
  pCur->iRowid = 0;
  pCur->nRecords = 0;

  const char *path = (const char *)sqlite3_value_text(argv[0]);
  if (!path) {
    vec_file_each_set_error(pCur, "vec_file_each requires a file path, got %s",
                            "NULL or non-text value");
    return SQLITE_ERROR;
  }

  size_t pathLen = strlen(path);
  if (pathLen > 6 && strcmp(path + pathLen - 6, ".fvecs") == 0) {
    pCur->elementSize = sizeof(f32);
    pCur->isFloat32 = 1;
  } else if (pathLen > 6 && strcmp(path + pathLen - 6, ".bvecs") == 0) {
    pCur->elementSize = sizeof(u8);
    pCur->isFloat32 = 0;
  } else if (pathLen > 6 && strcmp(path + pathLen - 6, ".ivecs") == 0) {
    pCur->elementSize = sizeof(i32);
    pCur->isFloat32 = 0;
  } else {
    sqlite3_free(pCur->base.pVtab->zErrMsg);
    pCur->base.pVtab->zErrMsg = sqlite3_mprintf(
        "unsupported vector file \"%s\": extension must be .fvecs, .bvecs, or .ivecs",
        path);
    return SQLITE_ERROR;
  }

  pCur->file = fopen(path, "rb");
  if (!pCur->file) {
    sqlite3_free(pCur->base.pVtab->zErrMsg);
    pCur->base.pVtab->zErrMsg =
        sqlite3_mprintf("could not open vector file \"%s\": %s", path,
                        strerror(errno));
    return SQLITE_ERROR;
  }

  if (fseek(pCur->file, 0, SEEK_END) != 0) {
    vec_file_each_set_error(pCur, "could not seek in vector file: %s",
                            strerror(errno));
    return SQLITE_ERROR;
  }
  long fileSize = ftell(pCur->file);
  if (fileSize < 0) {
    vec_file_each_set_error(pCur, "could not measure vector file: %s",
                            strerror(errno));
    return SQLITE_ERROR;
  }
  if (fseek(pCur->file, 0, SEEK_SET) != 0) {
    vec_file_each_set_error(pCur, "could not seek in vector file: %s",
                            strerror(errno));
    return SQLITE_ERROR;
  }
  if (fileSize == 0) {
    return SQLITE_OK;
  }
  if (fileSize < 4) {
    sqlite3_free(pCur->base.pVtab->zErrMsg);
    pCur->base.pVtab->zErrMsg = sqlite3_mprintf(
        "vector file \"%s\" is truncated: smaller than one record header", path);
    return SQLITE_ERROR;
  }

  i32 dim;
  if (fread(&dim, sizeof(i32), 1, pCur->file) != 1) {
    vec_file_each_set_error(pCur, "error reading vector file: %s",
                            strerror(errno));
    return SQLITE_ERROR;
  }
  if (dim <= 0) {
    sqlite3_free(pCur->base.pVtab->zErrMsg);
    pCur->base.pVtab->zErrMsg = sqlite3_mprintf(
        "vector file \"%s\" has invalid dimension %d", path, dim);
    return SQLITE_ERROR;
  }
  pCur->dimensions = dim;

  i64 recordSize = 4 + (i64)dim * (i64)pCur->elementSize;
  if (fileSize % recordSize != 0) {
    sqlite3_free(pCur->base.pVtab->zErrMsg);
    pCur->base.pVtab->zErrMsg = sqlite3_mprintf(
        "vector file \"%s\" is truncated: size %lld is not a multiple of the "
        "%lld-byte record size",
        path, (long long)fileSize, (long long)recordSize);
    return SQLITE_ERROR;
  }
  pCur->nRecords = fileSize / recordSize;

  pCur->vector = sqlite3_malloc64((i64)dim * (i64)pCur->elementSize);
  if (!pCur->vector) {
    return SQLITE_NOMEM;
  }
  // The first record's header was already consumed above.
  size_t n = (size_t)dim * pCur->elementSize;
  if (n > 0 && fread(pCur->vector, 1, n, pCur->file) != n) {
    vec_file_each_set_error(pCur, "truncated record in vector file at record %s",
                            "0");
    return SQLITE_ERROR;
  }
  return SQLITE_OK;
}

static int vec_file_eachRowid(sqlite3_vtab_cursor *cur, sqlite_int64 *pRowid) {
  vec_file_each_cursor *pCur = (vec_file_each_cursor *)cur;
  *pRowid = pCur->iRowid;
  return SQLITE_OK;
}

static int vec_file_eachEof(sqlite3_vtab_cursor *cur) {
  vec_file_each_cursor *pCur = (vec_file_each_cursor *)cur;
  return pCur->iRowid >= pCur->nRecords;
}

static int vec_file_eachNext(sqlite3_vtab_cursor *cur) {
  vec_file_each_cursor *pCur = (vec_file_each_cursor *)cur;
  pCur->iRowid++;
  if (pCur->iRowid >= pCur->nRecords) {
    return SQLITE_OK;
  }
  return vec_file_each_read_record(pCur);
}

static int vec_file_eachColumn(sqlite3_vtab_cursor *cur,
                               sqlite3_context *context, int i) {
  vec_file_each_cursor *pCur = (vec_file_each_cursor *)cur;
  if (i == VEC_FILE_EACH_COLUMN_VECTOR) {
    sqlite3_result_blob(context, pCur->vector,
                        (int)((size_t)pCur->dimensions * pCur->elementSize),
                        SQLITE_TRANSIENT);
    if (pCur->isFloat32) {
      sqlite3_result_subtype(context, SQLITE_VEC_ELEMENT_TYPE_FLOAT32);
    }
  }
  return SQLITE_OK;
}

static sqlite3_module vec_file_eachModule = {
    /* iVersion    */ 0,
    /* xCreate     */ 0,
    /* xConnect    */ vec_file_eachConnect,
    /* xBestIndex  */ vec_file_eachBestIndex,
    /* xDisconnect */ vec_file_eachDisconnect,
    /* xDestroy    */ 0,
    /* xOpen       */ vec_file_eachOpen,
    /* xClose      */ vec_file_eachClose,
    /* xFilter     */ vec_file_eachFilter,
    /* xNext       */ vec_file_eachNext,
    /* xEof        */ vec_file_eachEof,
    /* xColumn     */ vec_file_eachColumn,
    /* xRowid      */ vec_file_eachRowid,
    /* xUpdate     */ 0,
    /* xBegin      */ 0,
    /* xSync       */ 0,
    /* xCommit     */ 0,
    /* xRollback   */ 0,
    /* xFindMethod */ 0,
    /* xRename     */ 0,
    /* xSavepoint  */ 0,
    /* xRelease    */ 0,
    /* xRollbackTo */ 0,
    /* xShadowName */ 0,
#if SQLITE_VERSION_NUMBER >= 3044000
    /* xIntegrity  */ 0
#endif
};

#pragma endregion


//////////////////////////////////////////////////////////////////////////////
// Begin inlined: sqlite-vec-sparse.c
//////////////////////////////////////////////////////////////////////////////

/**
 * sqlite-vec-sparse.c — sparse vectors (SPLADE, BM25-style term weights).
 *
 * #include'd into sqlite-vec.c after vec_each and before sqlite3_vec_init().
 *
 * Format: a BLOB of (u32 index, f32 value) pairs, sorted by index, unique,
 * zeros dropped. Little-endian, like every other sqlite-vec vector.
 *
 * Indexing is left to SQLite itself: vec_sparse_each() turns a sparse vector
 * into (idx, value) rows, so an inverted index is a plain WITHOUT ROWID table
 * keyed on (term, doc) and a query is a join + GROUP BY. See reference.yaml.
 */
struct SparseEntry {
  u32 idx;
  f32 value;
};

#define SPARSE_ENTRY_SIZE 8

static int sparse_entry_cmp(const void *a, const void *b) {
  u32 x = ((const struct SparseEntry *)a)->idx;
  u32 y = ((const struct SparseEntry *)b)->idx;
  return (x > y) - (x < y);
}

static void sparse_read(const u8 *blob, int i, u32 *idx, f32 *value) {
  memcpy(idx, blob + i * SPARSE_ENTRY_SIZE, sizeof(u32));
  memcpy(value, blob + i * SPARSE_ENTRY_SIZE + sizeof(u32), sizeof(f32));
}

/**
 * Validate a sparse BLOB: whole entries, strictly increasing indexes.
 * Returns the entry count, or -1 when malformed.
 */
static int sparse_validate(const u8 *blob, int nBytes) {
  if (nBytes % SPARSE_ENTRY_SIZE != 0) return -1;
  int n = nBytes / SPARSE_ENTRY_SIZE;
  u32 prev = 0;
  for (int i = 0; i < n; i++) {
    u32 idx;
    f32 value;
    sparse_read(blob, i, &idx, &value);
    if (i > 0 && idx <= prev) return -1;
    prev = idx;
  }
  return n;
}

static int sparse_from_value(sqlite3_context *context, sqlite3_value *value,
                             const u8 **blob, int *n) {
  if (sqlite3_value_type(value) != SQLITE_BLOB &&
      sqlite3_value_type(value) != SQLITE_NULL) {
    sqlite3_result_error(context, "sparse vector must be a BLOB from vec_sparse()", -1);
    return SQLITE_ERROR;
  }
  *blob = sqlite3_value_blob(value);
  *n = sparse_validate(*blob, sqlite3_value_bytes(value));
  if (*n < 0) {
    sqlite3_result_error(context, "malformed sparse vector", -1);
    return SQLITE_ERROR;
  }
  return SQLITE_OK;
}

static void sparse_result(sqlite3_context *context, struct SparseEntry *e,
                          int n) {
  qsort(e, n, sizeof(*e), sparse_entry_cmp);
  int out = 0;
  for (int i = 0; i < n; i++) {
    if (e[i].value == 0.0f) continue;
    if (out > 0 && e[out - 1].idx == e[i].idx) {
      sqlite3_free(e);
      sqlite3_result_error(context, "duplicate index in sparse vector", -1);
      return;
    }
    e[out++] = e[i];
  }
  u8 *blob = sqlite3_malloc(out * SPARSE_ENTRY_SIZE + 1);
  if (!blob) {
    sqlite3_free(e);
    sqlite3_result_error_nomem(context);
    return;
  }
  for (int i = 0; i < out; i++) {
    memcpy(blob + i * SPARSE_ENTRY_SIZE, &e[i].idx, sizeof(u32));
    memcpy(blob + i * SPARSE_ENTRY_SIZE + sizeof(u32), &e[i].value, sizeof(f32));
  }
  sqlite3_free(e);
  sqlite3_result_blob(context, blob, out * SPARSE_ENTRY_SIZE, sqlite3_free);
}

/**
 * Parse a JSON object of {"index": weight, ...}.
 */
static int sparse_parse_json_object(const char *s, struct SparseEntry **out,
                                    int *nOut, const char **err) {
  int cap = 16, n = 0;
  struct SparseEntry *e = sqlite3_malloc(cap * sizeof(*e));
  if (!e) { *err = "out of memory"; return SQLITE_NOMEM; }
  const char *p = s;
  while (*p == ' ' || *p == '\t' || *p == '\n' || *p == '\r') p++;
  if (*p++ != '{') goto malformed;
  for (;;) {
    while (*p == ' ' || *p == '\t' || *p == '\n' || *p == '\r' || *p == ',') p++;
    if (*p == '}') break;
    if (*p++ != '"') goto malformed;
    char *end;
    errno = 0;
    unsigned long long idx = strtoull(p, &end, 10);
    if (end == p || *end != '"' || errno || idx > UINT32_MAX || *p == '-') goto malformed;
    p = end + 1;
    while (*p == ' ') p++;
    if (*p++ != ':') goto malformed;
    while (*p == ' ') p++;
    double v = strtod_c(p, &end);
    if (end == p) goto malformed;
    p = end;
    if (n == cap) {
      cap *= 2;
      struct SparseEntry *tmp = sqlite3_realloc64(e, cap * sizeof(*e));
      if (!tmp) { sqlite3_free(e); *err = "out of memory"; return SQLITE_NOMEM; }
      e = tmp;
    }
    e[n].idx = (u32)idx;
    e[n].value = (f32)v;
    n++;
  }
  *out = e;
  *nOut = n;
  return SQLITE_OK;
malformed:
  sqlite3_free(e);
  *err = "sparse vector JSON must be an object like {\"12\": 0.5, \"907\": 1.25}";
  return SQLITE_ERROR;
}

/**
 * vec_sparse(x): x is a JSON object {"idx": weight} or any dense float32
 * vector input (JSON array or BLOB), whose nonzero elements are kept.
 */
static void vec_sparse(sqlite3_context *context, int argc, sqlite3_value **argv) {
  assert(argc == 1);
  struct SparseEntry *e = NULL;
  int n = 0;
  if (sqlite3_value_type(argv[0]) == SQLITE_TEXT) {
    const char *s = (const char *)sqlite3_value_text(argv[0]);
    const char *p = s;
    while (*p == ' ' || *p == '\t' || *p == '\n' || *p == '\r') p++;
    if (*p == '{') {
      const char *err;
      int rc = sparse_parse_json_object(s, &e, &n, &err);
      if (rc == SQLITE_NOMEM) { sqlite3_result_error_nomem(context); return; }
      if (rc != SQLITE_OK) { sqlite3_result_error(context, err, -1); return; }
      sparse_result(context, e, n);
      return;
    }
  }
  void *vector;
  size_t dims;
  enum VectorElementType type;
  vector_cleanup cleanup;
  char *err;
  if (vector_from_value(argv[0], &vector, &dims, &type, &cleanup, &err) != SQLITE_OK) {
    sqlite3_result_error(context, err, -1);
    sqlite3_free(err);
    return;
  }
  if (type != SQLITE_VEC_ELEMENT_TYPE_FLOAT32) {
    cleanup(vector);
    sqlite3_result_error(context, "vec_sparse() only converts float32 vectors", -1);
    return;
  }
  e = sqlite3_malloc64((dims ? dims : 1) * sizeof(*e));
  if (!e) { cleanup(vector); sqlite3_result_error_nomem(context); return; }
  for (size_t i = 0; i < dims; i++) {
    f32 v = ((f32 *)vector)[i];
    if (v != 0.0f) { e[n].idx = (u32)i; e[n].value = v; n++; }
  }
  cleanup(vector);
  sparse_result(context, e, n);
}

/**
 * Merge-join over two sorted sparse vectors: dot, and both squared norms.
 */
static void sparse_merge(const u8 *a, int na, const u8 *b, int nb, double *dot,
                         double *aa, double *bb) {
  int i = 0, j = 0;
  *dot = *aa = *bb = 0;
  u32 ia, ib;
  f32 va, vb;
  for (int k = 0; k < na; k++) { sparse_read(a, k, &ia, &va); *aa += (double)va * va; }
  for (int k = 0; k < nb; k++) { sparse_read(b, k, &ib, &vb); *bb += (double)vb * vb; }
  while (i < na && j < nb) {
    sparse_read(a, i, &ia, &va);
    sparse_read(b, j, &ib, &vb);
    if (ia == ib) { *dot += (double)va * vb; i++; j++; }
    else if (ia < ib) i++;
    else j++;
  }
}

static void vec_sparse_distance(sqlite3_context *context, int argc,
                                sqlite3_value **argv, int cosine) {
  assert(argc == 2);
  const u8 *a, *b;
  int na, nb;
  if (sparse_from_value(context, argv[0], &a, &na) != SQLITE_OK) return;
  if (sparse_from_value(context, argv[1], &b, &nb) != SQLITE_OK) return;
  double dot, aa, bb;
  sparse_merge(a, na, b, nb, &dot, &aa, &bb);
  if (!cosine) {
    sqlite3_result_double(context, -dot);
  } else if (aa == 0 || bb == 0) {
    sqlite3_result_null(context);
  } else {
    sqlite3_result_double(context, 1.0 - dot / (sqrt(aa) * sqrt(bb)));
  }
}

static void vec_sparse_distance_dot(sqlite3_context *context, int argc,
                                    sqlite3_value **argv) {
  vec_sparse_distance(context, argc, argv, 0);
}

static void vec_sparse_distance_cosine(sqlite3_context *context, int argc,
                                       sqlite3_value **argv) {
  vec_sparse_distance(context, argc, argv, 1);
}

static void vec_sparse_to_json(sqlite3_context *context, int argc,
                               sqlite3_value **argv) {
  assert(argc == 1);
  const u8 *a;
  int n;
  if (sparse_from_value(context, argv[0], &a, &n) != SQLITE_OK) return;
  sqlite3_str *s = sqlite3_str_new(NULL);
  sqlite3_str_appendchar(s, 1, '{');
  for (int i = 0; i < n; i++) {
    u32 idx;
    f32 v;
    sparse_read(a, i, &idx, &v);
    sqlite3_str_appendf(s, "%s\"%u\":%!.9g", i ? "," : "", idx, (double)v);
  }
  sqlite3_str_appendchar(s, 1, '}');
  int len = sqlite3_str_length(s);
  char *z = sqlite3_str_finish(s);
  if (!z) { sqlite3_result_error_nomem(context); return; }
  sqlite3_result_text(context, z, len, sqlite3_free);
}

#pragma region vec_sparse_each table function

typedef struct vec_sparse_each_cursor vec_sparse_each_cursor;
struct vec_sparse_each_cursor {
  sqlite3_vtab_cursor base;
  i64 iRowid;
  u8 *blob;
  int n;
};

static int vec_sparse_eachConnect(sqlite3 *db, void *pAux, int argc,
                                  const char *const *argv,
                                  sqlite3_vtab **ppVtab, char **pzErr) {
  UNUSED_PARAMETER(pAux);
  UNUSED_PARAMETER(argc);
  UNUSED_PARAMETER(argv);
  UNUSED_PARAMETER(pzErr);
  int rc = sqlite3_declare_vtab(db, "CREATE TABLE x(idx, value, vector hidden)");
#define VEC_SPARSE_EACH_COLUMN_IDX 0
#define VEC_SPARSE_EACH_COLUMN_VALUE 1
#define VEC_SPARSE_EACH_COLUMN_VECTOR 2
  if (rc != SQLITE_OK) return rc;
  sqlite3_vtab *pNew = sqlite3_malloc(sizeof(*pNew));
  if (!pNew) return SQLITE_NOMEM;
  memset(pNew, 0, sizeof(*pNew));
  *ppVtab = pNew;
  return SQLITE_OK;
}

static int vec_sparse_eachDisconnect(sqlite3_vtab *pVtab) {
  sqlite3_free(pVtab);
  return SQLITE_OK;
}

static int vec_sparse_eachOpen(sqlite3_vtab *p, sqlite3_vtab_cursor **ppCursor) {
  UNUSED_PARAMETER(p);
  vec_sparse_each_cursor *pCur = sqlite3_malloc(sizeof(*pCur));
  if (!pCur) return SQLITE_NOMEM;
  memset(pCur, 0, sizeof(*pCur));
  *ppCursor = &pCur->base;
  return SQLITE_OK;
}

static int vec_sparse_eachClose(sqlite3_vtab_cursor *cur) {
  vec_sparse_each_cursor *pCur = (vec_sparse_each_cursor *)cur;
  sqlite3_free(pCur->blob);
  sqlite3_free(pCur);
  return SQLITE_OK;
}

static int vec_sparse_eachBestIndex(sqlite3_vtab *pVTab,
                                    sqlite3_index_info *pIdxInfo) {
  UNUSED_PARAMETER(pVTab);
  for (int i = 0; i < pIdxInfo->nConstraint; i++) {
    const struct sqlite3_index_constraint *c = &pIdxInfo->aConstraint[i];
    if (c->iColumn == VEC_SPARSE_EACH_COLUMN_VECTOR &&
        c->op == SQLITE_INDEX_CONSTRAINT_EQ && c->usable) {
      pIdxInfo->aConstraintUsage[i].argvIndex = 1;
      pIdxInfo->aConstraintUsage[i].omit = 1;
      pIdxInfo->estimatedCost = 100.0;
      pIdxInfo->estimatedRows = 100;
      return SQLITE_OK;
    }
  }
  return SQLITE_CONSTRAINT;
}

static int vec_sparse_eachFilter(sqlite3_vtab_cursor *pVtabCursor, int idxNum,
                                 const char *idxStr, int argc,
                                 sqlite3_value **argv) {
  UNUSED_PARAMETER(idxNum);
  UNUSED_PARAMETER(idxStr);
  assert(argc == 1);
  vec_sparse_each_cursor *pCur = (vec_sparse_each_cursor *)pVtabCursor;
  sqlite3_free(pCur->blob);
  pCur->blob = NULL;
  pCur->n = 0;
  pCur->iRowid = 0;
  int nBytes = sqlite3_value_bytes(argv[0]);
  const u8 *src = sqlite3_value_blob(argv[0]);
  int n = sparse_validate(src, nBytes);
  if (n < 0 || (sqlite3_value_type(argv[0]) != SQLITE_BLOB &&
                sqlite3_value_type(argv[0]) != SQLITE_NULL)) {
    sqlite3_free(pVtabCursor->pVtab->zErrMsg);
    pVtabCursor->pVtab->zErrMsg = sqlite3_mprintf("malformed sparse vector");
    return SQLITE_ERROR;
  }
  if (n > 0) {
    pCur->blob = sqlite3_malloc(nBytes);
    if (!pCur->blob) return SQLITE_NOMEM;
    memcpy(pCur->blob, src, nBytes);
  }
  pCur->n = n;
  return SQLITE_OK;
}

static int vec_sparse_eachRowid(sqlite3_vtab_cursor *cur, sqlite_int64 *pRowid) {
  *pRowid = ((vec_sparse_each_cursor *)cur)->iRowid;
  return SQLITE_OK;
}

static int vec_sparse_eachEof(sqlite3_vtab_cursor *cur) {
  vec_sparse_each_cursor *pCur = (vec_sparse_each_cursor *)cur;
  return pCur->iRowid >= pCur->n;
}

static int vec_sparse_eachNext(sqlite3_vtab_cursor *cur) {
  ((vec_sparse_each_cursor *)cur)->iRowid++;
  return SQLITE_OK;
}

static int vec_sparse_eachColumn(sqlite3_vtab_cursor *cur,
                                 sqlite3_context *context, int i) {
  vec_sparse_each_cursor *pCur = (vec_sparse_each_cursor *)cur;
  u32 idx;
  f32 value;
  sparse_read(pCur->blob, (int)pCur->iRowid, &idx, &value);
  if (i == VEC_SPARSE_EACH_COLUMN_IDX) sqlite3_result_int64(context, idx);
  else if (i == VEC_SPARSE_EACH_COLUMN_VALUE) sqlite3_result_double(context, value);
  return SQLITE_OK;
}

static sqlite3_module vec_sparse_eachModule = {
    /* iVersion    */ 0,
    /* xCreate     */ 0,
    /* xConnect    */ vec_sparse_eachConnect,
    /* xBestIndex  */ vec_sparse_eachBestIndex,
    /* xDisconnect */ vec_sparse_eachDisconnect,
    /* xDestroy    */ 0,
    /* xOpen       */ vec_sparse_eachOpen,
    /* xClose      */ vec_sparse_eachClose,
    /* xFilter     */ vec_sparse_eachFilter,
    /* xNext       */ vec_sparse_eachNext,
    /* xEof        */ vec_sparse_eachEof,
    /* xColumn     */ vec_sparse_eachColumn,
    /* xRowid      */ vec_sparse_eachRowid,
    /* xUpdate     */ 0,
    /* xBegin      */ 0,
    /* xSync       */ 0,
    /* xCommit     */ 0,
    /* xRollback   */ 0,
    /* xFindMethod */ 0,
    /* xRename     */ 0,
    /* xSavepoint  */ 0,
    /* xRelease    */ 0,
    /* xRollbackTo */ 0,
    /* xShadowName */ 0,
#if SQLITE_VERSION_NUMBER >= 3044000
    /* xIntegrity  */ 0
#endif
};

#pragma endregion

/**
 * vec_rrf(rank1, rank2, ...): Reciprocal Rank Fusion score, sum of
 * 1 / (60 + rank) over the non-NULL 1-based ranks. Fuses vec0 KNN ranks with
 * FTS5 bm25() ranks (or any other ranked lists) in plain SQL.
 * ponytail: k is fixed at 60 (the paper's value); add a vec_rrf_k() variant
 * if callers need to tune it.
 */
static void vec_rrf(sqlite3_context *context, int argc, sqlite3_value **argv) {
  double score = 0;
  for (int i = 0; i < argc; i++) {
    if (sqlite3_value_type(argv[i]) == SQLITE_NULL) continue;
    double rank = sqlite3_value_double(argv[i]);
    if (rank < 1) {
      sqlite3_result_error(context, "vec_rrf() ranks start at 1", -1);
      return;
    }
    score += 1.0 / (60.0 + rank);
  }
  sqlite3_result_double(context, score);
}
//////////////////////////////////////////////////////////////////////////////
// End inlined: sqlite-vec-sparse.c
//////////////////////////////////////////////////////////////////////////////

#pragma region vec0 virtual table

#define VEC0_COLUMN_ID 0
#define VEC0_COLUMN_USERN_START 1
#define VEC0_COLUMN_OFFSET_COMMAND 1
#define VEC0_COLUMN_OFFSET_DISTANCE 2
#define VEC0_COLUMN_OFFSET_K 3
#define VEC0_COLUMN_OFFSET_MMR_LAMBDA 4

#define VEC0_SHADOW_INFO_NAME "\"%w\".\"%w_info\""

#define VEC0_SHADOW_CHUNKS_NAME "\"%w\".\"%w_chunks\""
/// 1) schema, 2) original vtab table name
#define VEC0_SHADOW_CHUNKS_CREATE                                              \
  "CREATE TABLE " VEC0_SHADOW_CHUNKS_NAME "("                                  \
  "chunk_id INTEGER PRIMARY KEY AUTOINCREMENT,"                                \
  "size INTEGER NOT NULL,"                                                     \
  "validity BLOB NOT NULL,"                                                    \
  "rowids BLOB NOT NULL"                                                       \
  ");"

#define VEC0_SHADOW_ROWIDS_NAME "\"%w\".\"%w_rowids\""
/// 1) schema, 2) original vtab table name
#define VEC0_SHADOW_ROWIDS_CREATE_BASIC                                        \
  "CREATE TABLE " VEC0_SHADOW_ROWIDS_NAME "("                                  \
  "rowid INTEGER PRIMARY KEY AUTOINCREMENT,"                                   \
  "id,"                                                                        \
  "chunk_id INTEGER,"                                                          \
  "chunk_offset INTEGER"                                                       \
  ");"

// vec0 tables with a text primary keys are still backed by int64 primary keys,
// since a fixed-length rowid is required for vec0 chunks. But we add a new 'id
// text unique' column to emulate a text primary key interface.
#define VEC0_SHADOW_ROWIDS_CREATE_PK_TEXT                                      \
  "CREATE TABLE " VEC0_SHADOW_ROWIDS_NAME "("                                  \
  "rowid INTEGER PRIMARY KEY AUTOINCREMENT,"                                   \
  "id TEXT UNIQUE NOT NULL,"                                                   \
  "chunk_id INTEGER,"                                                          \
  "chunk_offset INTEGER"                                                       \
  ");"

/// 1) schema, 2) original vtab table name
#define VEC0_SHADOW_VECTOR_N_NAME "\"%w\".\"%w_vector_chunks%02d\""

/// 1) schema, 2) original vtab table name
//
// IMPORTANT: "rowid" is declared as PRIMARY KEY but WITHOUT the INTEGER type.
// This means it is NOT a true SQLite rowid alias — the user-defined "rowid"
// column and the internal SQLite rowid (_rowid_) are two separate values.
// When inserting, both must be set explicitly to keep them in sync. See the
// _rowid_ bindings in vec0_new_chunk() and the explanation in
// SHADOW_TABLE_ROWID_QUIRK below.
#define VEC0_SHADOW_VECTOR_N_CREATE                                            \
  "CREATE TABLE " VEC0_SHADOW_VECTOR_N_NAME "("                                \
  "rowid PRIMARY KEY,"                                                         \
  "vectors BLOB NOT NULL"                                                      \
  ");"

#define VEC0_SHADOW_AUXILIARY_NAME "\"%w\".\"%w_auxiliary\""

#define VEC0_SHADOW_METADATA_N_NAME "\"%w\".\"%w_metadatachunks%02d\""
#define VEC0_SHADOW_VECTORS_N_NAME "\"%w\".\"%w_vectors%02d\""
#define VEC0_SHADOW_DISKANN_NODES_N_NAME "\"%w\".\"%w_diskann_nodes%02d\""
#define VEC0_SHADOW_DISKANN_BUFFER_N_NAME "\"%w\".\"%w_diskann_buffer%02d\""
#define VEC0_SHADOW_METADATA_TEXT_DATA_NAME "\"%w\".\"%w_metadatatext%02d\""

#define VEC_INTERAL_ERROR "Internal sqlite-vec error: "
#define REPORT_URL "https://github.com/asg017/sqlite-vec/issues/new"

typedef struct vec0_vtab vec0_vtab;

#define VEC0_MAX_VECTOR_COLUMNS   16
#define VEC0_MAX_PARTITION_COLUMNS 4
#define VEC0_MAX_AUXILIARY_COLUMNS 16
#define VEC0_MAX_METADATA_COLUMNS 16

#define SQLITE_VEC_VEC0_MAX_DIMENSIONS 8192
#define VEC0_METADATA_TEXT_VIEW_BUFFER_LENGTH 16
#define VEC0_METADATA_TEXT_VIEW_DATA_LENGTH 12

typedef enum {
  // vector column, ie "contents_embedding float[1024]"
  SQLITE_VEC0_USER_COLUMN_KIND_VECTOR = 1,

  // partition key column, ie "user_id integer partition key"
  SQLITE_VEC0_USER_COLUMN_KIND_PARTITION = 2,

  //
  SQLITE_VEC0_USER_COLUMN_KIND_AUXILIARY = 3,

  // metadata column that can be filtered, ie "genre text"
  SQLITE_VEC0_USER_COLUMN_KIND_METADATA = 4,
} vec0_user_column_kind;

struct vec0_vtab {
  sqlite3_vtab base;

  // the SQLite connection of the host database
  sqlite3 *db;

  // True if the primary key of the vec0 table has a column type TEXT.
  // Will change the schema of the _rowids table, and insert/query logic.
  int pkIsText;

  // True if the hidden command column (named after the table) exists.
  // Tables created before v0.1.10 or without _info table don't have it.
  int hasCommandColumn;

  // number of defined vector columns.
  int numVectorColumns;

  // number of defined PARTITION KEY columns.
  int numPartitionColumns;

  // number of defined auxiliary columns
  int numAuxiliaryColumns;

  // Storage provider for the auxiliary columns (storage=<name>), or NULL to
  // use the _auxiliary shadow table. See sqlite3_vec_storage_methods.
  const sqlite3_vec_storage_methods *storage;
  sqlite3_vec_storage *storageHandle;

  // number of defined metadata columns
  int numMetadataColumns;


  // Name of the schema the table exists on.
  // Must be freed with sqlite3_free()
  char *schemaName;

  // Name of the table the table exists on.
  // Must be freed with sqlite3_free()
  char *tableName;

  // Name of the _rowids shadow table.
  // Must be freed with sqlite3_free()
  char *shadowRowidsName;

  // Name of the _chunks shadow table.
  // Must be freed with sqlite3_free()
  char *shadowChunksName;

  // contains enum vec0_user_column_kind values for up to
  // numVectorColumns + numPartitionColumns entries
  vec0_user_column_kind user_column_kinds[VEC0_MAX_VECTOR_COLUMNS + VEC0_MAX_PARTITION_COLUMNS + VEC0_MAX_AUXILIARY_COLUMNS + VEC0_MAX_METADATA_COLUMNS];

  uint8_t user_column_idxs[VEC0_MAX_VECTOR_COLUMNS + VEC0_MAX_PARTITION_COLUMNS + VEC0_MAX_AUXILIARY_COLUMNS + VEC0_MAX_METADATA_COLUMNS];


  // Name of all the vector chunk shadow tables.
  // Ex '_vector_chunks00'
  // Only the first numVectorColumns entries will be available.
  // The first numVectorColumns entries must be freed with sqlite3_free()
  char *shadowVectorChunksNames[VEC0_MAX_VECTOR_COLUMNS];

#if SQLITE_VEC_ENABLE_RESCORE
  // Name of all rescore chunk shadow tables, ie `_rescore_chunks00`
  // Only populated for vector columns with rescore enabled.
  // Must be freed with sqlite3_free()
  char *shadowRescoreChunksNames[VEC0_MAX_VECTOR_COLUMNS];

  // Name of all rescore vector shadow tables, ie `_rescore_vectors00`
  // Rowid-keyed table for fast random-access float vector reads during rescore.
  // Only populated for vector columns with rescore enabled.
  // Must be freed with sqlite3_free()
  char *shadowRescoreVectorsNames[VEC0_MAX_VECTOR_COLUMNS];
#endif

  // Name of all metadata chunk shadow tables, ie `_metadatachunks00`
  // Only the first numMetadataColumns entries will be available.
  // The first numMetadataColumns entries must be freed with sqlite3_free()
  char *shadowMetadataChunksNames[VEC0_MAX_METADATA_COLUMNS];

  struct VectorColumnDefinition vector_columns[VEC0_MAX_VECTOR_COLUMNS];
  struct Vec0PartitionColumnDefinition paritition_columns[VEC0_MAX_PARTITION_COLUMNS];
  struct Vec0AuxiliaryColumnDefinition auxiliary_columns[VEC0_MAX_AUXILIARY_COLUMNS];
  struct Vec0MetadataColumnDefinition metadata_columns[VEC0_MAX_METADATA_COLUMNS];

  int chunk_size;

#if SQLITE_VEC_EXPERIMENTAL_IVF_ENABLE
  // IVF cached state per vector column
  struct IvfWriteBuffer *ivfBuffer;
  unsigned ivfBufferGeneration;
  int ivfFlushing;
  char *shadowIvfCellsNames[VEC0_MAX_VECTOR_COLUMNS];   // table name for blob_open
  sqlite3_stmt *stmtIvfCellMeta[VEC0_MAX_VECTOR_COLUMNS]; // find cells with space
  sqlite3_stmt *stmtIvfCellWrite[VEC0_MAX_VECTOR_COLUMNS]; // flush packed cell
  sqlite3_stmt *stmtIvfRowidMapInsert[VEC0_MAX_VECTOR_COLUMNS]; // INSERT INTO rowid_map(rowid,cell_id,slot) VALUES(?,?,?)
  sqlite3_stmt *stmtIvfRowidMapLookup[VEC0_MAX_VECTOR_COLUMNS]; // SELECT cell_id,slot FROM rowid_map WHERE rowid=?
  sqlite3_stmt *stmtIvfRowidMapDelete[VEC0_MAX_VECTOR_COLUMNS]; // DELETE FROM rowid_map WHERE rowid=?
  sqlite3_stmt *stmtIvfCentroidsAll[VEC0_MAX_VECTOR_COLUMNS];   // SELECT centroid_id,centroid FROM centroids
  sqlite3_stmt *stmtIvfVectorsRead[VEC0_MAX_VECTOR_COLUMNS];    // SELECT vector FROM ivf_vectors WHERE rowid=?
#endif

  // select latest chunk from _chunks, getting chunk_id
  sqlite3_stmt *stmtLatestChunk;

  /**
   * Statement to insert a row into the _rowids table, with a rowid.
   * Parameters:
   *    1: int64, rowid to insert
   * Result columns: none
   * SQL: "INSERT INTO _rowids(rowid) VALUES (?)"
   *
   * Must be cleaned up with sqlite3_finalize().
   */
  sqlite3_stmt *stmtRowidsInsertRowid;

  /**
   * Statement to insert a row into the _rowids table, with an id.
   * The id column isn't a tradition primary key, but instead a unique
   * column to handle "text primary key" vec0 tables. The true int64 rowid
   * can be retrieved after inserting with sqlite3_last_rowid().
   *
   * Parameters:
   *    1: text or null, id to insert
   * Result columns: none
   *
   * Must be cleaned up with sqlite3_finalize().
   */
  sqlite3_stmt *stmtRowidsInsertId;

  /**
   * Statement to update the "position" columns chunk_id and chunk_offset for
   * a given _rowids row. Used when the "next available" chunk position is found
   * for a vector.
   *
   * Parameters:
   *    1: int64, chunk_id value
   *    2: int64, chunk_offset value
   *    3: int64, rowid value
   * Result columns: none
   *
   * Must be cleaned up with sqlite3_finalize().
   */
  sqlite3_stmt *stmtRowidsUpdatePosition;

  /**
   * Statement to quickly find the chunk_id + chunk_offset of a given row.
   * Parameters:
   *  1: rowid of the row/vector to lookup
   * Result columns:
   *  0: chunk_id (i64)
   *  1: chunk_offset (i64)
   * SQL: "SELECT id, chunk_id, chunk_offset FROM _rowids WHERE rowid = ?""
   *
   * Must be cleaned up with sqlite3_finalize().
   */
  sqlite3_stmt *stmtRowidsGetChunkPosition;

  sqlite3_stmt *stmtZoneMapRead;
  sqlite3_stmt *stmtZoneMapWrite;

  // === DiskANN additions ===
#if SQLITE_VEC_ENABLE_DISKANN
  // Shadow table names for DiskANN, per vector column
  // e.g., "{schema}"."{table}_vectors{00..15}"
  char *shadowVectorsNames[VEC0_MAX_VECTOR_COLUMNS];

  // e.g., "{schema}"."{table}_diskann_nodes{00..15}"
  char *shadowDiskannNodesNames[VEC0_MAX_VECTOR_COLUMNS];

  // Prepared statements for DiskANN operations (per vector column)
  // These will be lazily prepared on first use.
  sqlite3_stmt *stmtDiskannNodeRead[VEC0_MAX_VECTOR_COLUMNS];
  sqlite3_stmt *stmtDiskannNodeWrite[VEC0_MAX_VECTOR_COLUMNS];
  sqlite3_stmt *stmtDiskannNodeInsert[VEC0_MAX_VECTOR_COLUMNS];
  sqlite3_stmt *stmtVectorsRead[VEC0_MAX_VECTOR_COLUMNS];
  sqlite3_stmt *stmtVectorsInsert[VEC0_MAX_VECTOR_COLUMNS];
#endif
#if SQLITE_VEC_ENABLE_HNSW
  // Prepared statements for HNSW operations (per vector column), lazily
  // prepared on first use.
  sqlite3_stmt *stmtHnswNodeRead[VEC0_MAX_VECTOR_COLUMNS];
  sqlite3_stmt *stmtHnswNodeInsert[VEC0_MAX_VECTOR_COLUMNS];
  sqlite3_stmt *stmtHnswEdgesRead[VEC0_MAX_VECTOR_COLUMNS];
  sqlite3_stmt *stmtHnswEdgesWrite[VEC0_MAX_VECTOR_COLUMNS];
  sqlite3_stmt *stmtHnswEntry[VEC0_MAX_VECTOR_COLUMNS];
  // Per-connection vector/edge cache (sqlite-vec-hnsw.c), per vector column.
  struct HnswCache *hnswCache[VEC0_MAX_VECTOR_COLUMNS];
#endif
  struct Vec0FlatWrite *flatWrites[VEC0_MAX_VECTOR_COLUMNS];
  unsigned flatWriteGeneration;
  int flatFlushing;
};

static int vec0_flat_flush(vec0_vtab *p);
static void vec0_flat_clear(vec0_vtab *p);

#if SQLITE_VEC_ENABLE_RESCORE
// Forward declarations for rescore functions (defined in sqlite-vec-rescore.c,
// included later after all helpers they depend on are defined).
static int rescore_create_tables(vec0_vtab *p, sqlite3 *db, char **pzErr);
static int rescore_drop_tables(vec0_vtab *p);
static int rescore_new_chunk(vec0_vtab *p, i64 chunk_rowid);
static int rescore_on_insert(vec0_vtab *p, i64 chunk_rowid, i64 chunk_offset,
                             i64 rowid, void *vectorDatas[]);
static int rescore_on_delete(vec0_vtab *p, i64 chunk_id, u64 chunk_offset, i64 rowid);
static int rescore_delete_chunk(vec0_vtab *p, i64 chunk_id);
#endif

/**
 * Whether vector column i keeps its vectors in _vector_chunksNN: flat
 * columns, unless a storage provider holds the table's vectors.
 */
static int vec0_column_in_chunks(vec0_vtab *p, int i) {
  return p->vector_columns[i].index_type == VEC0_INDEX_TYPE_FLAT && !p->storage;
}

/**
 * Report a storage provider failure as the vtab's error message.
 */
static int vec0_storage_error(vec0_vtab *p, int rc) {
  if (rc == SQLITE_OK) return rc;
  const char *msg = p->storage->xErrmsg ? p->storage->xErrmsg(p->storageHandle) : NULL;
  vtab_set_error(&p->base, "vec0 storage: %s", msg ? msg : sqlite3_errstr(rc));
  return rc;
}

/**
 * @brief Finalize all the sqlite3_stmt members in a vec0_vtab.
 *
 * @param p vec0_vtab pointer
 */
void vec0_free_resources(vec0_vtab *p) {
  sqlite3_finalize(p->stmtLatestChunk);
  p->stmtLatestChunk = NULL;
  sqlite3_finalize(p->stmtRowidsInsertRowid);
  p->stmtRowidsInsertRowid = NULL;
  sqlite3_finalize(p->stmtRowidsInsertId);
  p->stmtRowidsInsertId = NULL;
  sqlite3_finalize(p->stmtRowidsUpdatePosition);
  p->stmtRowidsUpdatePosition = NULL;
  sqlite3_finalize(p->stmtRowidsGetChunkPosition);
  p->stmtRowidsGetChunkPosition = NULL;
  sqlite3_finalize(p->stmtZoneMapRead);
  p->stmtZoneMapRead = NULL;
  sqlite3_finalize(p->stmtZoneMapWrite);
  p->stmtZoneMapWrite = NULL;

#if SQLITE_VEC_EXPERIMENTAL_IVF_ENABLE
  for (int i = 0; i < VEC0_MAX_VECTOR_COLUMNS; i++) {
    sqlite3_finalize(p->stmtIvfCellMeta[i]); p->stmtIvfCellMeta[i] = NULL;
    sqlite3_finalize(p->stmtIvfCellWrite[i]); p->stmtIvfCellWrite[i] = NULL;
    sqlite3_finalize(p->stmtIvfRowidMapInsert[i]); p->stmtIvfRowidMapInsert[i] = NULL;
    sqlite3_finalize(p->stmtIvfRowidMapLookup[i]); p->stmtIvfRowidMapLookup[i] = NULL;
    sqlite3_finalize(p->stmtIvfRowidMapDelete[i]); p->stmtIvfRowidMapDelete[i] = NULL;
    sqlite3_finalize(p->stmtIvfCentroidsAll[i]); p->stmtIvfCentroidsAll[i] = NULL;
    sqlite3_finalize(p->stmtIvfVectorsRead[i]); p->stmtIvfVectorsRead[i] = NULL;
  }
#endif
#if SQLITE_VEC_ENABLE_DISKANN
  for (int i = 0; i < VEC0_MAX_VECTOR_COLUMNS; i++) {
    sqlite3_finalize(p->stmtDiskannNodeRead[i]); p->stmtDiskannNodeRead[i] = NULL;
    sqlite3_finalize(p->stmtDiskannNodeWrite[i]); p->stmtDiskannNodeWrite[i] = NULL;
    sqlite3_finalize(p->stmtDiskannNodeInsert[i]); p->stmtDiskannNodeInsert[i] = NULL;
    sqlite3_finalize(p->stmtVectorsRead[i]); p->stmtVectorsRead[i] = NULL;
    sqlite3_finalize(p->stmtVectorsInsert[i]); p->stmtVectorsInsert[i] = NULL;
  }
#endif
#if SQLITE_VEC_ENABLE_HNSW
  for (int i = 0; i < VEC0_MAX_VECTOR_COLUMNS; i++) {
    sqlite3_finalize(p->stmtHnswNodeRead[i]); p->stmtHnswNodeRead[i] = NULL;
    sqlite3_finalize(p->stmtHnswNodeInsert[i]); p->stmtHnswNodeInsert[i] = NULL;
    sqlite3_finalize(p->stmtHnswEdgesRead[i]); p->stmtHnswEdgesRead[i] = NULL;
    sqlite3_finalize(p->stmtHnswEdgesWrite[i]); p->stmtHnswEdgesWrite[i] = NULL;
    sqlite3_finalize(p->stmtHnswEntry[i]); p->stmtHnswEntry[i] = NULL;
  }
#endif
}

/**
 * @brief Free all memory and sqlite3_stmt members of a vec0_vtab
 *
 * @param p vec0_vtab pointer
 */
#if SQLITE_VEC_ENABLE_HNSW
static void hnsw_cache_free_all(vec0_vtab *p);  // sqlite-vec-hnsw.c
#endif
#if SQLITE_VEC_EXPERIMENTAL_IVF_ENABLE
static void ivf_buffer_clear(vec0_vtab *p);
#endif
void vec0_free(vec0_vtab *p) {
  vec0_free_resources(p);
  vec0_flat_clear(p);
#if SQLITE_VEC_EXPERIMENTAL_IVF_ENABLE
  ivf_buffer_clear(p);
#endif
  if (p->storage) {
    p->storage->xClose(p->storageHandle);
    p->storage = NULL;
    p->storageHandle = NULL;
  }
#if SQLITE_VEC_ENABLE_HNSW
  hnsw_cache_free_all(p);
#endif

  sqlite3_free(p->schemaName);
  p->schemaName = NULL;
  sqlite3_free(p->tableName);
  p->tableName = NULL;
  sqlite3_free(p->shadowChunksName);
  p->shadowChunksName = NULL;
  sqlite3_free(p->shadowRowidsName);
  p->shadowRowidsName = NULL;

  for (int i = 0; i < p->numVectorColumns; i++) {
    sqlite3_free(p->shadowVectorChunksNames[i]);
    p->shadowVectorChunksNames[i] = NULL;
#if SQLITE_VEC_EXPERIMENTAL_IVF_ENABLE
    sqlite3_free(p->shadowIvfCellsNames[i]);
    p->shadowIvfCellsNames[i] = NULL;
#endif

#if SQLITE_VEC_ENABLE_RESCORE
    sqlite3_free(p->shadowRescoreChunksNames[i]);
    p->shadowRescoreChunksNames[i] = NULL;

    sqlite3_free(p->shadowRescoreVectorsNames[i]);
    p->shadowRescoreVectorsNames[i] = NULL;
#endif

#if SQLITE_VEC_ENABLE_DISKANN
    sqlite3_free(p->shadowVectorsNames[i]);
    p->shadowVectorsNames[i] = NULL;
    sqlite3_free(p->shadowDiskannNodesNames[i]);
    p->shadowDiskannNodesNames[i] = NULL;
#endif

    sqlite3_free(p->vector_columns[i].name);
    p->vector_columns[i].name = NULL;
  }

  for (int i = 0; i < p->numPartitionColumns; i++) {
    sqlite3_free(p->paritition_columns[i].name);
    p->paritition_columns[i].name = NULL;
  }

  for (int i = 0; i < p->numAuxiliaryColumns; i++) {
    sqlite3_free(p->auxiliary_columns[i].name);
    p->auxiliary_columns[i].name = NULL;
  }

  for (int i = 0; i < p->numMetadataColumns; i++) {
    sqlite3_free(p->metadata_columns[i].name);
    p->metadata_columns[i].name = NULL;
  }
}

#if SQLITE_VEC_ENABLE_DISKANN

//////////////////////////////////////////////////////////////////////////////
// Begin inlined: sqlite-vec-diskann.c
//////////////////////////////////////////////////////////////////////////////

// DiskANN algorithm implementation
// This file is #include'd into sqlite-vec.c — not compiled separately.

// ============================================================
// DiskANN node blob encode/decode functions
// ============================================================

/** Compute size of validity bitmap in bytes. */
int diskann_validity_byte_size(int n_neighbors) {
  return n_neighbors / CHAR_BIT;
}

/** Compute size of neighbor IDs blob in bytes. */
size_t diskann_neighbor_ids_byte_size(int n_neighbors) {
  return (size_t)n_neighbors * sizeof(i64);
}

/** Compute size of quantized vectors blob in bytes. */
size_t diskann_neighbor_qvecs_byte_size(
    int n_neighbors, enum Vec0DiskannQuantizerType quantizer_type,
    size_t dimensions) {
  return (size_t)n_neighbors *
         diskann_quantized_vector_byte_size(quantizer_type, dimensions);
}

/**
 * Create empty blobs for a new DiskANN node (all neighbors invalid).
 * Caller must free the returned pointers with sqlite3_free().
 */
int diskann_node_init(
    int n_neighbors, enum Vec0DiskannQuantizerType quantizer_type,
    size_t dimensions,
    u8 **outValidity, int *outValiditySize,
    u8 **outNeighborIds, int *outNeighborIdsSize,
    u8 **outNeighborQvecs, int *outNeighborQvecsSize) {

  int validitySize = diskann_validity_byte_size(n_neighbors);
  size_t idsSize = diskann_neighbor_ids_byte_size(n_neighbors);
  size_t qvecsSize = diskann_neighbor_qvecs_byte_size(
      n_neighbors, quantizer_type, dimensions);

  u8 *validity = sqlite3_malloc(validitySize);
  u8 *ids = sqlite3_malloc(idsSize);
  u8 *qvecs = sqlite3_malloc(qvecsSize);

  if (!validity || !ids || !qvecs) {
    sqlite3_free(validity);
    sqlite3_free(ids);
    sqlite3_free(qvecs);
    return SQLITE_NOMEM;
  }

  memset(validity, 0, validitySize);
  memset(ids, 0, idsSize);
  memset(qvecs, 0, qvecsSize);

  *outValidity = validity;       *outValiditySize = validitySize;
  *outNeighborIds = ids;         *outNeighborIdsSize = (int)idsSize;
  *outNeighborQvecs = qvecs;     *outNeighborQvecsSize = (int)qvecsSize;
  return SQLITE_OK;
}

/** Check if neighbor slot i is valid. */
int diskann_validity_get(const u8 *validity, int i) {
  return (validity[i / CHAR_BIT] >> (i % CHAR_BIT)) & 1;
}

/** Set neighbor slot i as valid (1) or invalid (0). */
void diskann_validity_set(u8 *validity, int i, int value) {
  if (value) {
    validity[i / CHAR_BIT] |= (1 << (i % CHAR_BIT));
  } else {
    validity[i / CHAR_BIT] &= ~(1 << (i % CHAR_BIT));
  }
}

/** Count the number of valid neighbors. */
int diskann_validity_count(const u8 *validity, int n_neighbors) {
  int count = 0;
  for (int i = 0; i < n_neighbors; i++) {
    count += diskann_validity_get(validity, i);
  }
  return count;
}

/** Get the rowid of the neighbor in slot i. */
i64 diskann_neighbor_id_get(const u8 *neighbor_ids, int i) {
  i64 result;
  memcpy(&result, neighbor_ids + i * sizeof(i64), sizeof(i64));
  return result;
}

/** Set the rowid of the neighbor in slot i. */
void diskann_neighbor_id_set(u8 *neighbor_ids, int i, i64 rowid) {
  memcpy(neighbor_ids + i * sizeof(i64), &rowid, sizeof(i64));
}

/** Get a pointer to the quantized vector in slot i (read-only). */
const u8 *diskann_neighbor_qvec_get(
    const u8 *qvecs, int i,
    enum Vec0DiskannQuantizerType quantizer_type, size_t dimensions) {
  size_t qvec_size = diskann_quantized_vector_byte_size(quantizer_type, dimensions);
  return qvecs + (size_t)i * qvec_size;
}

/** Copy a quantized vector into slot i. */
void diskann_neighbor_qvec_set(
    u8 *qvecs, int i, const u8 *src_qvec,
    enum Vec0DiskannQuantizerType quantizer_type, size_t dimensions) {
  size_t qvec_size = diskann_quantized_vector_byte_size(quantizer_type, dimensions);
  memcpy(qvecs + (size_t)i * qvec_size, src_qvec, qvec_size);
}

/**
 * Set neighbor slot i with a rowid and quantized vector, and mark as valid.
 */
void diskann_node_set_neighbor(
    u8 *validity, u8 *neighbor_ids, u8 *qvecs, int i,
    i64 neighbor_rowid, const u8 *neighbor_qvec,
    enum Vec0DiskannQuantizerType quantizer_type, size_t dimensions) {
  diskann_validity_set(validity, i, 1);
  diskann_neighbor_id_set(neighbor_ids, i, neighbor_rowid);
  diskann_neighbor_qvec_set(qvecs, i, neighbor_qvec, quantizer_type, dimensions);
}

/**
 * Clear neighbor slot i (mark invalid, zero out data).
 */
void diskann_node_clear_neighbor(
    u8 *validity, u8 *neighbor_ids, u8 *qvecs, int i,
    enum Vec0DiskannQuantizerType quantizer_type, size_t dimensions) {
  diskann_validity_set(validity, i, 0);
  diskann_neighbor_id_set(neighbor_ids, i, 0);
  size_t qvec_size = diskann_quantized_vector_byte_size(quantizer_type, dimensions);
  memset(qvecs + (size_t)i * qvec_size, 0, qvec_size);
}

/**
 * Quantize a full-precision float32 vector into the target quantizer format.
 * Output buffer must be pre-allocated with diskann_quantized_vector_byte_size() bytes.
 */
int diskann_quantize_vector(
    const f32 *src, size_t dimensions,
    enum Vec0DiskannQuantizerType quantizer_type,
    u8 *out) {

  switch (quantizer_type) {
    case VEC0_DISKANN_QUANTIZER_BINARY:
      vec0_quantize_binary_f32(src, out, dimensions);
      return SQLITE_OK;
    case VEC0_DISKANN_QUANTIZER_INT8:
      vec0_quantize_int8_unit(src, (i8 *)out, dimensions);
      return SQLITE_OK;
  }
  return SQLITE_ERROR;
}

/**
 * Quantize a vector of a DiskANN column's element type into the target
 * quantizer format. A float32 vector goes through diskann_quantize_vector().
 * An int8 vector is its own int8 quantized form, and under the binary
 * quantizer each element above 0 sets its bit, as for a float32 vector.
 * Output buffer must be pre-allocated with diskann_quantized_vector_byte_size() bytes.
 */
static int diskann_quantize_element_vector(
    const void *src, enum VectorElementType element_type, size_t dimensions,
    enum Vec0DiskannQuantizerType quantizer_type, u8 *out) {
  switch (element_type) {
    case SQLITE_VEC_ELEMENT_TYPE_FLOAT32:
      return diskann_quantize_vector((const f32 *)src, dimensions,
                                     quantizer_type, out);
    case SQLITE_VEC_ELEMENT_TYPE_INT8: {
      const i8 *v = (const i8 *)src;
      switch (quantizer_type) {
        case VEC0_DISKANN_QUANTIZER_BINARY:
          memset(out, 0, dimensions / CHAR_BIT);
          for (size_t i = 0; i < dimensions; i++) {
            if (v[i] > 0) {
              out[i / CHAR_BIT] |= (1 << (i % CHAR_BIT));
            }
          }
          return SQLITE_OK;
        case VEC0_DISKANN_QUANTIZER_INT8:
          memcpy(out, v, dimensions);
          return SQLITE_OK;
      }
      break;
    }
    case SQLITE_VEC_ELEMENT_TYPE_BIT:
      break;
  }
  return SQLITE_ERROR;
}

/**
 * Compute approximate distance between a full-precision query vector and a
 * quantized neighbor vector. Used during graph traversal.
 */
/**
 * Compute distance between a pre-quantized query and a quantized neighbor.
 * The caller is responsible for quantizing the query vector once and passing
 * the result here for each neighbor comparison.
 */
static f32 diskann_distance_quantized_precomputed(
    const u8 *query_quantized, const u8 *quantized_neighbor,
    size_t dimensions,
    enum Vec0DiskannQuantizerType quantizer_type,
    enum Vec0DistanceMetrics distance_metric) {

  switch (quantizer_type) {
    case VEC0_DISKANN_QUANTIZER_BINARY:
      return distance_hamming(query_quantized, quantized_neighbor, &dimensions);
    case VEC0_DISKANN_QUANTIZER_INT8:
      return vec0_distance_full(query_quantized, quantized_neighbor, dimensions,
                                SQLITE_VEC_ELEMENT_TYPE_INT8, distance_metric);
  }
  return FLT_MAX;
}

/**
 * Quantize a query vector of the column's element type. Returns allocated
 * buffer (caller must free), or NULL when out of memory.
 */
static u8 *diskann_quantize_query(
    const void *query_vector, enum VectorElementType element_type,
    size_t dimensions, enum Vec0DiskannQuantizerType quantizer_type) {
  size_t qsize = diskann_quantized_vector_byte_size(quantizer_type, dimensions);
  u8 *buf = sqlite3_malloc(qsize);
  if (!buf) return NULL;
  diskann_quantize_element_vector(query_vector, element_type, dimensions,
                                  quantizer_type, buf);
  return buf;
}

// ============================================================
// DiskANN medoid / entry point management
// ============================================================

/**
 * Get the current medoid rowid for the given vector column's DiskANN index.
 * Returns SQLITE_OK with *outMedoid set to the medoid rowid.
 * If the graph is empty, returns SQLITE_OK with *outIsEmpty = 1.
 */
static int diskann_medoid_get(vec0_vtab *p, int vec_col_idx,
                               i64 *outMedoid, int *outIsEmpty) {
  int rc;
  sqlite3_stmt *stmt = NULL;
  char *key = sqlite3_mprintf("diskann_medoid_%02d", vec_col_idx);
  char *zSql = sqlite3_mprintf(
      "SELECT value FROM " VEC0_SHADOW_INFO_NAME " WHERE key = ?",
      p->schemaName, p->tableName);
  if (!key || !zSql) {
    sqlite3_free(key);
    sqlite3_free(zSql);
    return SQLITE_NOMEM;
  }

  rc = sqlite3_prepare_v2(p->db, zSql, -1, &stmt, NULL);
  sqlite3_free(zSql);
  if (rc != SQLITE_OK) {
    sqlite3_free(key);
    return rc;
  }

  sqlite3_bind_text(stmt, 1, key, -1, sqlite3_free);
  rc = sqlite3_step(stmt);
  if (rc == SQLITE_ROW) {
    if (sqlite3_column_type(stmt, 0) == SQLITE_NULL) {
      *outIsEmpty = 1;
    } else {
      *outIsEmpty = 0;
      *outMedoid = sqlite3_column_int64(stmt, 0);
    }
    rc = SQLITE_OK;
  } else {
    rc = SQLITE_ERROR;
  }
  sqlite3_finalize(stmt);
  return rc;
}

/**
 * Set the medoid rowid for the given vector column's DiskANN index.
 * Pass isEmpty = 1 to mark the graph as empty (NULL medoid).
 */
static int diskann_medoid_set(vec0_vtab *p, int vec_col_idx,
                               i64 medoidRowid, int isEmpty) {
  int rc;
  sqlite3_stmt *stmt = NULL;
  char *key = sqlite3_mprintf("diskann_medoid_%02d", vec_col_idx);
  char *zSql = sqlite3_mprintf(
      "UPDATE " VEC0_SHADOW_INFO_NAME " SET value = ?2 WHERE key = ?1",
      p->schemaName, p->tableName);
  if (!key || !zSql) {
    sqlite3_free(key);
    sqlite3_free(zSql);
    return SQLITE_NOMEM;
  }

  rc = sqlite3_prepare_v2(p->db, zSql, -1, &stmt, NULL);
  sqlite3_free(zSql);
  if (rc != SQLITE_OK) {
    sqlite3_free(key);
    return rc;
  }

  sqlite3_bind_text(stmt, 1, key, -1, sqlite3_free);
  if (isEmpty) {
    sqlite3_bind_null(stmt, 2);
  } else {
    sqlite3_bind_int64(stmt, 2, medoidRowid);
  }
  rc = sqlite3_step(stmt);
  sqlite3_finalize(stmt);
  return (rc == SQLITE_DONE) ? SQLITE_OK : SQLITE_ERROR;
}


/**
 * Called when deleting a vector. If the deleted vector was the medoid,
 * pick a new one: the first remaining node of the graph, or set to empty if
 * none remain. A vector still in the buffer has no node, so it cannot be the
 * entry point of a search.
 */
static int diskann_medoid_handle_delete(vec0_vtab *p, int vec_col_idx,
                                          i64 deletedRowid) {
  i64 currentMedoid;
  int isEmpty;
  int rc = diskann_medoid_get(p, vec_col_idx, &currentMedoid, &isEmpty);
  if (rc != SQLITE_OK) return rc;

  if (!isEmpty && currentMedoid == deletedRowid) {
    sqlite3_stmt *stmt = NULL;
    char *zSql = sqlite3_mprintf(
        "SELECT rowid FROM " VEC0_SHADOW_DISKANN_NODES_N_NAME " WHERE rowid != ?1 LIMIT 1",
        p->schemaName, p->tableName, vec_col_idx);
    if (!zSql) return SQLITE_NOMEM;

    rc = sqlite3_prepare_v2(p->db, zSql, -1, &stmt, NULL);
    sqlite3_free(zSql);
    if (rc != SQLITE_OK) return rc;

    sqlite3_bind_int64(stmt, 1, deletedRowid);
    rc = sqlite3_step(stmt);
    if (rc == SQLITE_ROW) {
      i64 newMedoid = sqlite3_column_int64(stmt, 0);
      sqlite3_finalize(stmt);
      return diskann_medoid_set(p, vec_col_idx, newMedoid, 0);
    }
    sqlite3_finalize(stmt);
    if (rc != SQLITE_DONE) return rc;
    return diskann_medoid_set(p, vec_col_idx, -1, 1);
  }
  return SQLITE_OK;
}

// ============================================================
// DiskANN database I/O helpers
// ============================================================

/**
 * Read a node's full data from _diskann_nodes.
 * Returns blobs that must be freed by the caller with sqlite3_free().
 */
static int diskann_node_read(vec0_vtab *p, int vec_col_idx, i64 rowid,
                              u8 **outValidity, int *outValiditySize,
                              u8 **outNeighborIds, int *outNeighborIdsSize,
                              u8 **outQvecs, int *outQvecsSize) {
  int rc;
  if (!p->stmtDiskannNodeRead[vec_col_idx]) {
    char *zSql = sqlite3_mprintf(
        "SELECT neighbors_validity, neighbor_ids, neighbor_quantized_vectors "
        "FROM " VEC0_SHADOW_DISKANN_NODES_N_NAME " WHERE rowid = ?",
        p->schemaName, p->tableName, vec_col_idx);
    if (!zSql) return SQLITE_NOMEM;
    rc = sqlite3_prepare_v2(p->db, zSql, -1,
                             &p->stmtDiskannNodeRead[vec_col_idx], NULL);
    sqlite3_free(zSql);
    if (rc != SQLITE_OK) return rc;
  }

  sqlite3_stmt *stmt = p->stmtDiskannNodeRead[vec_col_idx];
  sqlite3_reset(stmt);
  sqlite3_bind_int64(stmt, 1, rowid);

  rc = sqlite3_step(stmt);
  if (rc != SQLITE_ROW) {
    rc = SQLITE_ERROR;
    goto done;
  }

  int vs = sqlite3_column_bytes(stmt, 0);
  int is = sqlite3_column_bytes(stmt, 1);
  int qs = sqlite3_column_bytes(stmt, 2);

  // Validate blob sizes against config expectations to detect truncated /
  // corrupt data before any caller iterates using cfg->n_neighbors.
  {
    struct VectorColumnDefinition *col = &p->vector_columns[vec_col_idx];
    struct Vec0DiskannConfig *cfg = &col->diskann;
    int expectedVs = diskann_validity_byte_size(cfg->n_neighbors);
    int expectedIs = (int)diskann_neighbor_ids_byte_size(cfg->n_neighbors);
    int expectedQs = (int)diskann_neighbor_qvecs_byte_size(
        cfg->n_neighbors, cfg->quantizer_type, col->dimensions);
    if (vs != expectedVs || is != expectedIs || qs != expectedQs) {
      rc = SQLITE_CORRUPT;
      goto done;
    }
  }

  u8 *v = sqlite3_malloc(vs);
  u8 *ids = sqlite3_malloc(is);
  u8 *qv = sqlite3_malloc(qs);
  if (!v || !ids || !qv) {
    sqlite3_free(v);
    sqlite3_free(ids);
    sqlite3_free(qv);
    rc = SQLITE_NOMEM;
    goto done;
  }

  const void *blobV = sqlite3_column_blob(stmt, 0);
  const void *blobIds = sqlite3_column_blob(stmt, 1);
  const void *blobQv = sqlite3_column_blob(stmt, 2);
  if (!blobV || !blobIds || !blobQv) {
    sqlite3_free(v);
    sqlite3_free(ids);
    sqlite3_free(qv);
    rc = SQLITE_ERROR;
    goto done;
  }
  memcpy(v, blobV, vs);
  memcpy(ids, blobIds, is);
  memcpy(qv, blobQv, qs);

  *outValidity = v;       *outValiditySize = vs;
  *outNeighborIds = ids;  *outNeighborIdsSize = is;
  *outQvecs = qv;         *outQvecsSize = qs;
  rc = SQLITE_OK;

done:
  // A cached statement left on a row stays in progress until the next
  // commit, and a connection with a statement in progress refuses VACUUM.
  sqlite3_reset(stmt);
  return rc;
}

/**
 * Write (INSERT OR REPLACE) a node's data to _diskann_nodes.
 */
static int diskann_node_write(vec0_vtab *p, int vec_col_idx, i64 rowid,
                               const u8 *validity, int validitySize,
                               const u8 *neighborIds, int neighborIdsSize,
                               const u8 *qvecs, int qvecsSize) {
  int rc;
  if (!p->stmtDiskannNodeWrite[vec_col_idx]) {
    char *zSql = sqlite3_mprintf(
        "INSERT OR REPLACE INTO " VEC0_SHADOW_DISKANN_NODES_N_NAME
        " (rowid, neighbors_validity, neighbor_ids, neighbor_quantized_vectors) "
        "VALUES (?, ?, ?, ?)",
        p->schemaName, p->tableName, vec_col_idx);
    if (!zSql) return SQLITE_NOMEM;
    rc = sqlite3_prepare_v2(p->db, zSql, -1,
                             &p->stmtDiskannNodeWrite[vec_col_idx], NULL);
    sqlite3_free(zSql);
    if (rc != SQLITE_OK) return rc;
  }

  sqlite3_stmt *stmt = p->stmtDiskannNodeWrite[vec_col_idx];
  sqlite3_reset(stmt);
  sqlite3_bind_int64(stmt, 1, rowid);
  sqlite3_bind_blob(stmt, 2, validity, validitySize, SQLITE_TRANSIENT);
  sqlite3_bind_blob(stmt, 3, neighborIds, neighborIdsSize, SQLITE_TRANSIENT);
  sqlite3_bind_blob(stmt, 4, qvecs, qvecsSize, SQLITE_TRANSIENT);

  rc = sqlite3_step(stmt);
  return (rc == SQLITE_DONE) ? SQLITE_OK : SQLITE_ERROR;
}

/**
 * Read the full-precision vector for a given rowid from _vectors.
 * Caller must free *outVector with sqlite3_free().
 */
static int diskann_vector_read(vec0_vtab *p, int vec_col_idx, i64 rowid,
                                void **outVector, int *outVectorSize) {
  int rc;
  if (!p->stmtVectorsRead[vec_col_idx]) {
    char *zSql = sqlite3_mprintf(
        "SELECT vector FROM " VEC0_SHADOW_VECTORS_N_NAME " WHERE rowid = ?",
        p->schemaName, p->tableName, vec_col_idx);
    if (!zSql) return SQLITE_NOMEM;
    rc = sqlite3_prepare_v2(p->db, zSql, -1,
                             &p->stmtVectorsRead[vec_col_idx], NULL);
    sqlite3_free(zSql);
    if (rc != SQLITE_OK) return rc;
  }

  sqlite3_stmt *stmt = p->stmtVectorsRead[vec_col_idx];
  sqlite3_reset(stmt);
  sqlite3_bind_int64(stmt, 1, rowid);

  rc = sqlite3_step(stmt);
  if (rc != SQLITE_ROW) {
    rc = SQLITE_ERROR;
    goto done;
  }

  int sz = sqlite3_column_bytes(stmt, 0);
  const void *blob = sqlite3_column_blob(stmt, 0);
  if (!blob || sz == 0) {
    rc = SQLITE_ERROR;
    goto done;
  }
  void *vec = sqlite3_malloc(sz);
  if (!vec) {
    rc = SQLITE_NOMEM;
    goto done;
  }
  memcpy(vec, blob, sz);

  *outVector = vec;
  *outVectorSize = sz;
  rc = SQLITE_OK;

done:
  // As in diskann_node_read(): no cached statement stays on a row.
  sqlite3_reset(stmt);
  return rc;
}

/**
 * Write a full-precision vector to _vectors.
 */
static int diskann_vector_write(vec0_vtab *p, int vec_col_idx, i64 rowid,
                                 const void *vector, int vectorSize) {
  int rc;
  if (!p->stmtVectorsInsert[vec_col_idx]) {
    char *zSql = sqlite3_mprintf(
        "INSERT OR REPLACE INTO " VEC0_SHADOW_VECTORS_N_NAME
        " (rowid, vector) VALUES (?, ?)",
        p->schemaName, p->tableName, vec_col_idx);
    if (!zSql) return SQLITE_NOMEM;
    rc = sqlite3_prepare_v2(p->db, zSql, -1,
                             &p->stmtVectorsInsert[vec_col_idx], NULL);
    sqlite3_free(zSql);
    if (rc != SQLITE_OK) return rc;
  }

  sqlite3_stmt *stmt = p->stmtVectorsInsert[vec_col_idx];
  sqlite3_reset(stmt);
  sqlite3_bind_int64(stmt, 1, rowid);
  sqlite3_bind_blob(stmt, 2, vector, vectorSize, SQLITE_TRANSIENT);

  rc = sqlite3_step(stmt);
  return (rc == SQLITE_DONE) ? SQLITE_OK : SQLITE_ERROR;
}

// ============================================================
// DiskANN search data structures
// ============================================================

/**
 * A sorted candidate list for greedy beam search.
 */
struct DiskannCandidateList {
  struct Vec0DiskannCandidate *items;
  int count;
  int capacity;
};

static int diskann_candidate_list_init(struct DiskannCandidateList *list, int capacity) {
  list->items = sqlite3_malloc(capacity * sizeof(struct Vec0DiskannCandidate));
  if (!list->items) return SQLITE_NOMEM;
  list->count = 0;
  list->capacity = capacity;
  return SQLITE_OK;
}

static void diskann_candidate_list_free(struct DiskannCandidateList *list) {
  sqlite3_free(list->items);
  list->items = NULL;
  list->count = 0;
  list->capacity = 0;
}

/**
 * Insert a candidate into the sorted list, maintaining sort order by distance.
 * Deduplicates by rowid. If at capacity and new candidate is worse, discards it.
 * Returns 1 if inserted, 0 if discarded.
 */
static int diskann_candidate_list_insert(
    struct DiskannCandidateList *list, i64 rowid, f32 distance) {

  // Check for duplicate
  for (int i = 0; i < list->count; i++) {
    if (list->items[i].rowid == rowid) {
      // Update distance if better
      if (distance < list->items[i].distance) {
        list->items[i].distance = distance;
        // Re-sort this item into position
        struct Vec0DiskannCandidate tmp = list->items[i];
        int j = i - 1;
        while (j >= 0 && list->items[j].distance > tmp.distance) {
          list->items[j + 1] = list->items[j];
          j--;
        }
        list->items[j + 1] = tmp;
      }
      return 1;
    }
  }

  // If at capacity, check if new candidate is better than worst
  if (list->count >= list->capacity) {
    if (distance >= list->items[list->count - 1].distance) {
      return 0;  // Discard
    }
    list->count--;  // Make room by dropping the worst
  }

  // Binary search for insertion point
  int lo = 0, hi = list->count;
  while (lo < hi) {
    int mid = (lo + hi) / 2;
    if (list->items[mid].distance < distance) {
      lo = mid + 1;
    } else {
      hi = mid;
    }
  }

  // Shift elements to make room
  memmove(&list->items[lo + 1], &list->items[lo],
          (list->count - lo) * sizeof(struct Vec0DiskannCandidate));

  list->items[lo].rowid = rowid;
  list->items[lo].distance = distance;
  list->items[lo].visited = 0;
  list->items[lo].confirmed = 0;
  list->count++;
  return 1;
}

/**
 * Give a visited candidate its exact distance and confirm it (its vector
 * exists). The exact distance replaces the quantized one whether it is larger
 * or smaller, and the candidate moves to its place in the order. A candidate
 * the list dropped after it was visited comes back, still visited, when its
 * exact distance ranks inside the list.
 */
static void diskann_candidate_list_confirm(
    struct DiskannCandidateList *list, i64 rowid, f32 distance) {
  int i = 0;
  while (i < list->count && list->items[i].rowid != rowid) i++;
  if (i == list->count) {
    if (!diskann_candidate_list_insert(list, rowid, distance)) return;
    i = 0;
    while (list->items[i].rowid != rowid) i++;
    list->items[i].visited = 1;
  }

  struct Vec0DiskannCandidate item = list->items[i];
  item.distance = distance;
  item.confirmed = 1;
  while (i > 0 && list->items[i - 1].distance > distance) {
    list->items[i] = list->items[i - 1];
    i--;
  }
  while (i + 1 < list->count && list->items[i + 1].distance < distance) {
    list->items[i] = list->items[i + 1];
    i++;
  }
  list->items[i] = item;
}

/**
 * Find the closest unvisited candidate. Returns its index, or -1 if none.
 */
static int diskann_candidate_list_next_unvisited(
    const struct DiskannCandidateList *list) {
  for (int i = 0; i < list->count; i++) {
    if (!list->items[i].visited) return i;
  }
  return -1;
}



/**
 * Simple hash set for tracking visited rowids during search.
 * Uses open addressing with linear probing.
 */
struct DiskannVisitedSet {
  i64 *slots;
  int capacity;
  int count;
  int has_zero;
};

static int diskann_visited_set_init(struct DiskannVisitedSet *set, int capacity) {
  memset(set, 0, sizeof(*set));
  // Round up to power of 2
  int cap = 16;
  while (cap < capacity) {
    if (cap > INT_MAX / 2) return SQLITE_NOMEM;
    cap *= 2;
  }
  set->slots = sqlite3_malloc64((sqlite3_uint64)cap * sizeof(i64));
  if (!set->slots) return SQLITE_NOMEM;
  memset(set->slots, 0, cap * sizeof(i64));
  set->capacity = cap;
  set->count = 0;
  return SQLITE_OK;
}

static void diskann_visited_set_free(struct DiskannVisitedSet *set) {
  sqlite3_free(set->slots);
  set->slots = NULL;
  set->capacity = 0;
  set->count = 0;
  set->has_zero = 0;
}

static int diskann_visited_set_contains(const struct DiskannVisitedSet *set, i64 rowid) {
  if (rowid == 0) return set->has_zero;
  int mask = set->capacity - 1;
  int idx = (int)(((u64)rowid * 0x9E3779B97F4A7C15ULL) >> 32) & mask;
  for (int i = 0; i < set->capacity; i++) {
    int slot = (idx + i) & mask;
    if (set->slots[slot] == 0) return 0;
    if (set->slots[slot] == rowid) return 1;
  }
  return 0;
}

// Returns 1 for a new visit, 0 for a duplicate, -1 on allocation failure.
static int diskann_visited_set_insert(struct DiskannVisitedSet *set, i64 rowid) {
  if (rowid == 0) {
    if (set->has_zero) return 0;
    set->has_zero = 1;
    set->count++;
    return 1;
  }
  if (diskann_visited_set_contains(set, rowid)) return 0;
  if (set->count >= set->capacity / 2) {
    if (set->capacity > INT_MAX / 2) return -1;
    int capacity = set->capacity * 2;
    i64 *slots = sqlite3_malloc64((sqlite3_uint64)capacity * sizeof(i64));
    if (!slots) return -1;
    memset(slots, 0, (size_t)capacity * sizeof(i64));
    for (int i = 0; i < set->capacity; i++) {
      i64 entry = set->slots[i];
      if (entry == 0) continue;
      int idx = (int)(((u64)entry * 0x9E3779B97F4A7C15ULL) >> 32) & (capacity - 1);
      while (slots[idx] != 0) idx = (idx + 1) & (capacity - 1);
      slots[idx] = entry;
    }
    sqlite3_free(set->slots);
    set->slots = slots;
    set->capacity = capacity;
  }
  int mask = set->capacity - 1;
  int idx = (int)(((u64)rowid * 0x9E3779B97F4A7C15ULL) >> 32) & mask;
  for (int i = 0; i < set->capacity; i++) {
    int slot = (idx + i) & mask;
    if (set->slots[slot] == 0) {
      set->slots[slot] = rowid;
      set->count++;
      return 1;
    }
    if (set->slots[slot] == rowid) return 0;  // Already there
  }
  return -1;
}

// ============================================================
// DiskANN greedy beam search (LM-Search)
// ============================================================

/**
 * Perform LM-Search: greedy beam search over the DiskANN graph.
 * Follows Algorithm 1 from the LM-DiskANN paper.
 *
 * With `allow` (a sorted rowid filter, NULL for none) the beam still walks
 * every node, and the output is the k nearest allowed nodes it visited.
 * ponytail: the beam is not widened for a selective filter, so it can return
 * fewer than k rows; grow searchListSize by selectivity if that bites.
 */
static int diskann_search(
    vec0_vtab *p, int vec_col_idx,
    const void *queryVector, size_t dimensions,
    enum VectorElementType elementType,
    int k, int searchListSize, const struct Array *allow,
    i64 *outRowids, f32 *outDistances, int *outCount) {

  struct VectorColumnDefinition *col = &p->vector_columns[vec_col_idx];
  struct Vec0DiskannConfig *cfg = &col->diskann;
  int rc;

  if (searchListSize <= 0) {
    searchListSize = cfg->search_list_size_search > 0 ? cfg->search_list_size_search : cfg->search_list_size;
  }
  if (searchListSize < k) {
    searchListSize = k;
  }

  // 1. Get the medoid (entry point)
  i64 medoid;
  int isEmpty;
  rc = diskann_medoid_get(p, vec_col_idx, &medoid, &isEmpty);
  if (rc != SQLITE_OK) return rc;
  if (isEmpty) {
    *outCount = 0;
    return SQLITE_OK;
  }

  // 2. Compute distance from query to medoid using full-precision vector
  void *medoidVector = NULL;
  int medoidVectorSize;
  rc = diskann_vector_read(p, vec_col_idx, medoid, &medoidVector, &medoidVectorSize);
  if (rc != SQLITE_OK) return rc;

  f32 medoidDist = vec0_distance_full(queryVector, medoidVector,
                                          dimensions, elementType,
                                          col->distance_metric);
  sqlite3_free(medoidVector);

  // 3. Initialize candidate list and visited set
  struct DiskannCandidateList candidates;
  rc = diskann_candidate_list_init(&candidates, searchListSize);
  if (rc != SQLITE_OK) return rc;

  struct DiskannVisitedSet visited;
  rc = diskann_visited_set_init(&visited, searchListSize);
  if (rc != SQLITE_OK) {
    diskann_candidate_list_free(&candidates);
    return rc;
  }

  // Seed with medoid (confirmed — we already read its vector above)
  diskann_candidate_list_insert(&candidates, medoid, medoidDist);
  candidates.items[0].confirmed = 1;
  int nAllowed = 0;  // with `allow`: outRowids holds the nearest allowed visits

  // Pre-quantize query vector once for all quantized distance comparisons
  u8 *queryQuantized = diskann_quantize_query(queryVector, elementType,
                                              dimensions, cfg->quantizer_type);
  if (!queryQuantized) {
    diskann_candidate_list_free(&candidates);
    diskann_visited_set_free(&visited);
    return SQLITE_NOMEM;
  }

  // 4. Greedy beam search loop (Algorithm 1 from LM-DiskANN paper)
  while (1) {
    int nextIdx = diskann_candidate_list_next_unvisited(&candidates);
    if (nextIdx < 0) break;

    struct Vec0DiskannCandidate *current = &candidates.items[nextIdx];
    current->visited = 1;
    i64 currentRowid = current->rowid;
    int inserted = diskann_visited_set_insert(&visited, currentRowid);
    if (inserted < 0) {
      rc = SQLITE_NOMEM;
      goto cleanup;
    }
    if (!inserted) continue;

    // Read the node's neighbor data
    u8 *validity = NULL, *neighborIds = NULL, *qvecs = NULL;
    int validitySize, neighborIdsSize, qvecsSize;
    rc = diskann_node_read(p, vec_col_idx, currentRowid,
                            &validity, &validitySize,
                            &neighborIds, &neighborIdsSize,
                            &qvecs, &qvecsSize);
    if (rc != SQLITE_OK) {
      continue;  // Skip if node doesn't exist
    }

    // Insert all valid neighbors with approximate (quantized) distances
    for (int i = 0; i < cfg->n_neighbors; i++) {
      if (!diskann_validity_get(validity, i)) continue;

      i64 neighborRowid = diskann_neighbor_id_get(neighborIds, i);

      if (diskann_visited_set_contains(&visited, neighborRowid)) continue;

      const u8 *neighborQvec = diskann_neighbor_qvec_get(
          qvecs, i, cfg->quantizer_type, dimensions);

      f32 approxDist = diskann_distance_quantized_precomputed(
          queryQuantized, neighborQvec, dimensions,
          cfg->quantizer_type, col->distance_metric);

      diskann_candidate_list_insert(&candidates, neighborRowid, approxDist);
    }

    sqlite3_free(validity);
    sqlite3_free(neighborIds);
    sqlite3_free(qvecs);

    // Paper line 13: Re-rank p* using full-precision distance
    // We already have exact distance for medoid; for others, update now
    void *fullVec = NULL;
    int fullVecSize;
    rc = diskann_vector_read(p, vec_col_idx, currentRowid, &fullVec, &fullVecSize);
    if (rc == SQLITE_OK) {
      f32 exactDist = vec0_distance_full(queryVector, fullVec,
                                             dimensions, elementType,
                                             col->distance_metric);
      sqlite3_free(fullVec);
      diskann_candidate_list_confirm(&candidates, currentRowid, exactDist);
      if (allow && vec0_rowids_contains(allow, currentRowid) &&
          (nAllowed < k || exactDist < outDistances[nAllowed - 1])) {
        int i = nAllowed < k ? nAllowed++ : nAllowed - 1;
        while (i > 0 && outDistances[i - 1] > exactDist) {
          outRowids[i] = outRowids[i - 1];
          outDistances[i] = outDistances[i - 1];
          i--;
        }
        outRowids[i] = currentRowid;
        outDistances[i] = exactDist;
      }
    }
    // If vector read failed, candidate stays unconfirmed (stale edge to deleted node)
  }

  // 5. Output results — only include confirmed candidates (whose vectors exist)
  int resultCount = allow ? nAllowed : 0;
  for (int i = 0; !allow && i < candidates.count && resultCount < k; i++) {
    if (candidates.items[i].confirmed) {
      outRowids[resultCount] = candidates.items[i].rowid;
      outDistances[resultCount] = candidates.items[i].distance;
      resultCount++;
    }
  }
  *outCount = resultCount;

  rc = SQLITE_OK;
cleanup:
  sqlite3_free(queryQuantized);
  diskann_candidate_list_free(&candidates);
  diskann_visited_set_free(&visited);
  return rc;
}

// ============================================================
// DiskANN RobustPrune (Algorithm 4 from LM-DiskANN paper)
// ============================================================

/**
 * RobustPrune: Select up to max_neighbors neighbors for node p from a
 * candidate set, using alpha-pruning for diversity.
 *
 * Following Algorithm 4 (LM-Prune):
 *   C = C union N_out(p) \ {p}
 *   N_out(p) = empty
 *   while C not empty:
 *     p* = argmin d(p, c) for c in C
 *     N_out(p).insert(p*)
 *     if |N_out(p)| >= R: break
 *     for each p' in C:
 *       if alpha * d(p*, p') <= d(p, p'): remove p' from C
 */
/**
 * Pure function: given pre-sorted candidates and a distance matrix, select
 * up to max_neighbors using alpha-pruning. inter_distances is a flattened
 * num_candidates x num_candidates matrix where inter_distances[i*num_candidates+j]
 * = d(candidate_i, candidate_j). p_distances[i] = d(p, candidate_i), already sorted.
 * outSelected[i] = 1 if selected. Returns count of selected.
 */
int diskann_prune_select(
    const f32 *inter_distances, const f32 *p_distances,
    int num_candidates, f32 alpha, int max_neighbors,
    int *outSelected, int *outCount) {

  if (num_candidates == 0) {
    *outCount = 0;
    return SQLITE_OK;
  }

  u8 *active = sqlite3_malloc(num_candidates);
  if (!active) return SQLITE_NOMEM;
  memset(active, 1, num_candidates);
  memset(outSelected, 0, num_candidates * sizeof(int));

  int selectedCount = 0;

  for (int round = 0; round < num_candidates && selectedCount < max_neighbors; round++) {
    int bestIdx = -1;
    for (int i = 0; i < num_candidates; i++) {
      if (active[i]) { bestIdx = i; break; }
    }
    if (bestIdx < 0) break;

    outSelected[bestIdx] = 1;
    selectedCount++;
    active[bestIdx] = 0;

    for (int i = 0; i < num_candidates; i++) {
      if (!active[i]) continue;
      f32 dist_best_to_i = inter_distances[bestIdx * num_candidates + i];
      if (alpha * dist_best_to_i <= p_distances[i]) {
        active[i] = 0;
      }
    }
  }

  *outCount = selectedCount;
  sqlite3_free(active);
  return SQLITE_OK;
}

static int diskann_robust_prune(
    vec0_vtab *p, int vec_col_idx,
    i64 p_rowid, const void *p_vector,
    i64 *candidates, f32 *candidate_distances, int num_candidates,
    f32 alpha, int max_neighbors,
    i64 *outNeighborRowids, int *outNeighborCount) {

  struct VectorColumnDefinition *col = &p->vector_columns[vec_col_idx];
  int rc;

  // Remove p itself from candidates
  for (int i = 0; i < num_candidates; i++) {
    if (candidates[i] == p_rowid) {
      candidates[i] = candidates[num_candidates - 1];
      candidate_distances[i] = candidate_distances[num_candidates - 1];
      num_candidates--;
      break;
    }
  }

  if (num_candidates == 0) {
    *outNeighborCount = 0;
    return SQLITE_OK;
  }

  // Sort candidates by distance to p (ascending) - insertion sort
  for (int i = 1; i < num_candidates; i++) {
    f32 tmpDist = candidate_distances[i];
    i64 tmpRowid = candidates[i];
    int j = i - 1;
    while (j >= 0 && candidate_distances[j] > tmpDist) {
      candidate_distances[j + 1] = candidate_distances[j];
      candidates[j + 1] = candidates[j];
      j--;
    }
    candidate_distances[j + 1] = tmpDist;
    candidates[j + 1] = tmpRowid;
  }

  // Active flags
  u8 *active = sqlite3_malloc(num_candidates);
  if (!active) return SQLITE_NOMEM;
  memset(active, 1, num_candidates);

  // Cache full-precision vectors for inter-candidate distance
  void **candidateVectors = sqlite3_malloc(num_candidates * sizeof(void *));
  if (!candidateVectors) {
    sqlite3_free(active);
    return SQLITE_NOMEM;
  }
  memset(candidateVectors, 0, num_candidates * sizeof(void *));

  int selectedCount = 0;

  for (int round = 0; round < num_candidates && selectedCount < max_neighbors; round++) {
    // Find closest active candidate
    int bestIdx = -1;
    for (int i = 0; i < num_candidates; i++) {
      if (active[i]) { bestIdx = i; break; }
    }
    if (bestIdx < 0) break;

    // Select this candidate
    outNeighborRowids[selectedCount] = candidates[bestIdx];
    selectedCount++;
    active[bestIdx] = 0;

    // Load selected candidate's vector
    if (!candidateVectors[bestIdx]) {
      int vecSize;
      rc = diskann_vector_read(p, vec_col_idx, candidates[bestIdx],
                                &candidateVectors[bestIdx], &vecSize);
      if (rc != SQLITE_OK) continue;
    }

    // Alpha-prune: remove candidates covered by the selected neighbor
    for (int i = 0; i < num_candidates; i++) {
      if (!active[i]) continue;

      if (!candidateVectors[i]) {
        int vecSize;
        rc = diskann_vector_read(p, vec_col_idx, candidates[i],
                                  &candidateVectors[i], &vecSize);
        if (rc != SQLITE_OK) continue;
      }

      f32 dist_selected_to_i = vec0_distance_full(
          candidateVectors[bestIdx], candidateVectors[i],
          col->dimensions, col->element_type, col->distance_metric);

      if (alpha * dist_selected_to_i <= candidate_distances[i]) {
        active[i] = 0;
      }
    }
  }

  *outNeighborCount = selectedCount;

  for (int i = 0; i < num_candidates; i++) {
    sqlite3_free(candidateVectors[i]);
  }
  sqlite3_free(candidateVectors);
  sqlite3_free(active);

  return SQLITE_OK;
}

/**
 * After RobustPrune selects neighbors, build the node blobs and write to DB.
 * Quantizes each neighbor's vector and packs into the node format.
 */
static int diskann_write_pruned_neighbors(
    vec0_vtab *p, int vec_col_idx, i64 nodeRowid,
    const i64 *neighborRowids, int neighborCount) {

  struct VectorColumnDefinition *col = &p->vector_columns[vec_col_idx];
  struct Vec0DiskannConfig *cfg = &col->diskann;
  int rc;

  u8 *validity, *neighborIds, *qvecs;
  int validitySize, neighborIdsSize, qvecsSize;
  rc = diskann_node_init(cfg->n_neighbors, cfg->quantizer_type,
                          col->dimensions,
                          &validity, &validitySize,
                          &neighborIds, &neighborIdsSize,
                          &qvecs, &qvecsSize);
  if (rc != SQLITE_OK) return rc;

  size_t qvecSize = diskann_quantized_vector_byte_size(
      cfg->quantizer_type, col->dimensions);
  u8 *qvec = sqlite3_malloc(qvecSize);
  if (!qvec) {
    sqlite3_free(validity);
    sqlite3_free(neighborIds);
    sqlite3_free(qvecs);
    return SQLITE_NOMEM;
  }

  for (int i = 0; i < neighborCount && i < cfg->n_neighbors; i++) {
    void *neighborVec = NULL;
    int neighborVecSize;
    rc = diskann_vector_read(p, vec_col_idx, neighborRowids[i],
                              &neighborVec, &neighborVecSize);
    if (rc != SQLITE_OK) continue;

    diskann_quantize_element_vector(neighborVec, col->element_type,
                                    col->dimensions, cfg->quantizer_type, qvec);

    diskann_node_set_neighbor(validity, neighborIds, qvecs, i,
                               neighborRowids[i], qvec,
                               cfg->quantizer_type, col->dimensions);

    sqlite3_free(neighborVec);
  }
  sqlite3_free(qvec);

  rc = diskann_node_write(p, vec_col_idx, nodeRowid,
                           validity, validitySize,
                           neighborIds, neighborIdsSize,
                           qvecs, qvecsSize);

  sqlite3_free(validity);
  sqlite3_free(neighborIds);
  sqlite3_free(qvecs);
  return rc;
}

// ============================================================
// DiskANN insert (Algorithm 2 from LM-DiskANN paper)
// ============================================================

/**
 * Add a reverse edge: make target_rowid a neighbor of node_rowid.
 * If node is full, run RobustPrune.
 */
static int diskann_add_reverse_edge(
    vec0_vtab *p, int vec_col_idx,
    i64 node_rowid, i64 target_rowid, const void *target_vector) {

  struct VectorColumnDefinition *col = &p->vector_columns[vec_col_idx];
  struct Vec0DiskannConfig *cfg = &col->diskann;
  int rc;

  u8 *validity = NULL, *neighborIds = NULL, *qvecs = NULL;
  int validitySize, neighborIdsSize, qvecsSize;
  rc = diskann_node_read(p, vec_col_idx, node_rowid,
                          &validity, &validitySize,
                          &neighborIds, &neighborIdsSize,
                          &qvecs, &qvecsSize);
  if (rc != SQLITE_OK) return rc;

  int currentCount = diskann_validity_count(validity, cfg->n_neighbors);

  // Check if target is already a neighbor
  for (int i = 0; i < cfg->n_neighbors; i++) {
    if (diskann_validity_get(validity, i) &&
        diskann_neighbor_id_get(neighborIds, i) == target_rowid) {
      sqlite3_free(validity);
      sqlite3_free(neighborIds);
      sqlite3_free(qvecs);
      return SQLITE_OK;
    }
  }

  if (currentCount < cfg->n_neighbors) {
    // Room available: find first empty slot
    for (int i = 0; i < cfg->n_neighbors; i++) {
      if (!diskann_validity_get(validity, i)) {
        size_t qvecSize = diskann_quantized_vector_byte_size(
            cfg->quantizer_type, col->dimensions);
        u8 *qvec = sqlite3_malloc(qvecSize);
        if (!qvec) {
          sqlite3_free(validity);
          sqlite3_free(neighborIds);
          sqlite3_free(qvecs);
          return SQLITE_NOMEM;
        }

        diskann_quantize_element_vector(target_vector, col->element_type,
                                        col->dimensions, cfg->quantizer_type,
                                        qvec);

        diskann_node_set_neighbor(validity, neighborIds, qvecs, i,
                                   target_rowid, qvec,
                                   cfg->quantizer_type, col->dimensions);
        sqlite3_free(qvec);
        break;
      }
    }

    rc = diskann_node_write(p, vec_col_idx, node_rowid,
                             validity, validitySize,
                             neighborIds, neighborIdsSize,
                             qvecs, qvecsSize);
  } else {
    // Full: lazy replacement — use quantized distances to find the worst
    // existing neighbor and replace it if target is closer. This avoids
    // reading all neighbors' float vectors (the expensive RobustPrune path).

    // Quantize the node's vector and the target vector for comparison
    void *nodeVector = NULL;
    int nodeVecSize;
    rc = diskann_vector_read(p, vec_col_idx, node_rowid,
                              &nodeVector, &nodeVecSize);
    if (rc != SQLITE_OK) {
      sqlite3_free(validity);
      sqlite3_free(neighborIds);
      sqlite3_free(qvecs);
      return rc;
    }

    // Quantize target for node-level comparison
    size_t qvecSize = diskann_quantized_vector_byte_size(
        cfg->quantizer_type, col->dimensions);
    u8 *targetQ = sqlite3_malloc(qvecSize);
    u8 *nodeQ = sqlite3_malloc(qvecSize);
    if (!targetQ || !nodeQ) {
      sqlite3_free(targetQ);
      sqlite3_free(nodeQ);
      sqlite3_free(nodeVector);
      sqlite3_free(validity);
      sqlite3_free(neighborIds);
      sqlite3_free(qvecs);
      return SQLITE_NOMEM;
    }

    diskann_quantize_element_vector(target_vector, col->element_type,
                                    col->dimensions, cfg->quantizer_type,
                                    targetQ);
    diskann_quantize_element_vector(nodeVector, col->element_type,
                                    col->dimensions, cfg->quantizer_type, nodeQ);

    // Compute quantized distance from node to target
    f32 targetDist = diskann_distance_quantized_precomputed(
        nodeQ, targetQ, col->dimensions,
        cfg->quantizer_type, col->distance_metric);

    // Find the worst (farthest) existing neighbor using quantized distances
    int worstIdx = -1;
    f32 worstDist = -1.0f;
    for (int i = 0; i < cfg->n_neighbors; i++) {
      if (!diskann_validity_get(validity, i)) continue;
      const u8 *nqvec = diskann_neighbor_qvec_get(
          qvecs, i, cfg->quantizer_type, col->dimensions);
      f32 d = diskann_distance_quantized_precomputed(
          nodeQ, nqvec, col->dimensions,
          cfg->quantizer_type, col->distance_metric);
      if (d > worstDist) {
        worstDist = d;
        worstIdx = i;
      }
    }

    // Replace worst neighbor if target is closer
    if (worstIdx >= 0 && targetDist < worstDist) {
      diskann_node_set_neighbor(validity, neighborIds, qvecs, worstIdx,
                                 target_rowid, targetQ,
                                 cfg->quantizer_type, col->dimensions);
      rc = diskann_node_write(p, vec_col_idx, node_rowid,
                               validity, validitySize,
                               neighborIds, neighborIdsSize,
                               qvecs, qvecsSize);
    } else {
      rc = SQLITE_OK;  // target is farther than all existing neighbors, skip
    }

    sqlite3_free(targetQ);
    sqlite3_free(nodeQ);
    sqlite3_free(nodeVector);
  }

  sqlite3_free(validity);
  sqlite3_free(neighborIds);
  sqlite3_free(qvecs);
  return rc;
}

// ============================================================
// DiskANN buffer helpers (for batched inserts)
// ============================================================

/**
 * Insert a vector into the _diskann_buffer table.
 */
static int diskann_buffer_write(vec0_vtab *p, int vec_col_idx,
                                 i64 rowid, const void *vector, int vectorSize) {
  sqlite3_stmt *stmt = NULL;
  char *zSql = sqlite3_mprintf(
      "INSERT INTO " VEC0_SHADOW_DISKANN_BUFFER_N_NAME
      " (rowid, vector) VALUES (?, ?)",
      p->schemaName, p->tableName, vec_col_idx);
  if (!zSql) return SQLITE_NOMEM;
  int rc = sqlite3_prepare_v2(p->db, zSql, -1, &stmt, NULL);
  sqlite3_free(zSql);
  if (rc != SQLITE_OK) return rc;
  sqlite3_bind_int64(stmt, 1, rowid);
  sqlite3_bind_blob(stmt, 2, vector, vectorSize, SQLITE_STATIC);
  rc = sqlite3_step(stmt);
  sqlite3_finalize(stmt);
  return (rc == SQLITE_DONE) ? SQLITE_OK : SQLITE_ERROR;
}

/**
 * Delete a vector from the _diskann_buffer table.
 */
static int diskann_buffer_delete(vec0_vtab *p, int vec_col_idx, i64 rowid) {
  sqlite3_stmt *stmt = NULL;
  char *zSql = sqlite3_mprintf(
      "DELETE FROM " VEC0_SHADOW_DISKANN_BUFFER_N_NAME " WHERE rowid = ?",
      p->schemaName, p->tableName, vec_col_idx);
  if (!zSql) return SQLITE_NOMEM;
  int rc = sqlite3_prepare_v2(p->db, zSql, -1, &stmt, NULL);
  sqlite3_free(zSql);
  if (rc != SQLITE_OK) return rc;
  sqlite3_bind_int64(stmt, 1, rowid);
  rc = sqlite3_step(stmt);
  sqlite3_finalize(stmt);
  return (rc == SQLITE_DONE) ? SQLITE_OK : SQLITE_ERROR;
}

/**
 * Check if a rowid exists in the _diskann_buffer table.
 * Returns SQLITE_OK and sets *exists to 1 if found, 0 if not.
 */
static int diskann_buffer_exists(vec0_vtab *p, int vec_col_idx,
                                  i64 rowid, int *exists) {
  sqlite3_stmt *stmt = NULL;
  char *zSql = sqlite3_mprintf(
      "SELECT 1 FROM " VEC0_SHADOW_DISKANN_BUFFER_N_NAME " WHERE rowid = ?",
      p->schemaName, p->tableName, vec_col_idx);
  if (!zSql) return SQLITE_NOMEM;
  int rc = sqlite3_prepare_v2(p->db, zSql, -1, &stmt, NULL);
  sqlite3_free(zSql);
  if (rc != SQLITE_OK) return rc;
  sqlite3_bind_int64(stmt, 1, rowid);
  rc = sqlite3_step(stmt);
  *exists = (rc == SQLITE_ROW) ? 1 : 0;
  sqlite3_finalize(stmt);
  return SQLITE_OK;
}

/**
 * Get the count of rows in the _diskann_buffer table.
 */
static int diskann_buffer_count(vec0_vtab *p, int vec_col_idx, i64 *count) {
  sqlite3_stmt *stmt = NULL;
  char *zSql = sqlite3_mprintf(
      "SELECT count(*) FROM " VEC0_SHADOW_DISKANN_BUFFER_N_NAME,
      p->schemaName, p->tableName, vec_col_idx);
  if (!zSql) return SQLITE_NOMEM;
  int rc = sqlite3_prepare_v2(p->db, zSql, -1, &stmt, NULL);
  sqlite3_free(zSql);
  if (rc != SQLITE_OK) return rc;
  rc = sqlite3_step(stmt);
  if (rc == SQLITE_ROW) {
    *count = sqlite3_column_int64(stmt, 0);
    sqlite3_finalize(stmt);
    return SQLITE_OK;
  }
  sqlite3_finalize(stmt);
  return SQLITE_ERROR;
}

// Forward declaration: diskann_insert_graph does the actual graph insertion
static int diskann_insert_graph(vec0_vtab *p, int vec_col_idx,
                                 i64 rowid, const void *vector);

/**
 * Flush all buffered vectors into the DiskANN graph.
 * Iterates over _diskann_buffer rows and calls diskann_insert_graph for each.
 */
static int diskann_flush_buffer(vec0_vtab *p, int vec_col_idx) {
  sqlite3_stmt *stmt = NULL;
  char *zSql = sqlite3_mprintf(
      "SELECT rowid, vector FROM " VEC0_SHADOW_DISKANN_BUFFER_N_NAME,
      p->schemaName, p->tableName, vec_col_idx);
  if (!zSql) return SQLITE_NOMEM;
  int rc = sqlite3_prepare_v2(p->db, zSql, -1, &stmt, NULL);
  sqlite3_free(zSql);
  if (rc != SQLITE_OK) return rc;

  while ((rc = sqlite3_step(stmt)) == SQLITE_ROW) {
    i64 rowid = sqlite3_column_int64(stmt, 0);
    const void *vector = sqlite3_column_blob(stmt, 1);
    if (!vector) continue;
    // Note: vector is already written to _vectors table, so
    // diskann_insert_graph will skip re-writing it (vector already exists).
    // We call the graph-only insert path.
    int insertRc = diskann_insert_graph(p, vec_col_idx, rowid, vector);
    if (insertRc != SQLITE_OK) {
      sqlite3_finalize(stmt);
      return insertRc;
    }
  }
  sqlite3_finalize(stmt);

  // Clear the buffer
  zSql = sqlite3_mprintf(
      "DELETE FROM " VEC0_SHADOW_DISKANN_BUFFER_N_NAME,
      p->schemaName, p->tableName, vec_col_idx);
  if (!zSql) return SQLITE_NOMEM;
  rc = sqlite3_prepare_v2(p->db, zSql, -1, &stmt, NULL);
  sqlite3_free(zSql);
  if (rc != SQLITE_OK) return rc;
  rc = sqlite3_step(stmt);
  sqlite3_finalize(stmt);
  return (rc == SQLITE_DONE) ? SQLITE_OK : SQLITE_ERROR;
}

/**
 * Insert a new vector into the DiskANN graph (graph-only path).
 * The vector must already be written to _vectors table.
 * This is the core graph insertion logic (Algorithm 2: LM-Insert).
 */
static int diskann_insert_graph(vec0_vtab *p, int vec_col_idx,
                                 i64 rowid, const void *vector) {
  struct VectorColumnDefinition *col = &p->vector_columns[vec_col_idx];
  struct Vec0DiskannConfig *cfg = &col->diskann;
  int rc;

  // Handle first insert (empty graph)
  i64 medoid;
  int isEmpty;
  rc = diskann_medoid_get(p, vec_col_idx, &medoid, &isEmpty);
  if (rc != SQLITE_OK) return rc;

  if (isEmpty) {
    u8 *validity, *neighborIds, *qvecs;
    int validitySize, neighborIdsSize, qvecsSize;
    rc = diskann_node_init(cfg->n_neighbors, cfg->quantizer_type,
                            col->dimensions,
                            &validity, &validitySize,
                            &neighborIds, &neighborIdsSize,
                            &qvecs, &qvecsSize);
    if (rc != SQLITE_OK) return rc;

    rc = diskann_node_write(p, vec_col_idx, rowid,
                             validity, validitySize,
                             neighborIds, neighborIdsSize,
                             qvecs, qvecsSize);
    sqlite3_free(validity);
    sqlite3_free(neighborIds);
    sqlite3_free(qvecs);
    if (rc != SQLITE_OK) return rc;

    return diskann_medoid_set(p, vec_col_idx, rowid, 0);
  }

  // Search for nearest neighbors
  int L = cfg->search_list_size_insert > 0 ? cfg->search_list_size_insert : cfg->search_list_size;
  i64 *searchRowids = sqlite3_malloc(L * sizeof(i64));
  f32 *searchDistances = sqlite3_malloc(L * sizeof(f32));
  if (!searchRowids || !searchDistances) {
    sqlite3_free(searchRowids);
    sqlite3_free(searchDistances);
    return SQLITE_NOMEM;
  }

  int searchCount;
  rc = diskann_search(p, vec_col_idx, vector, col->dimensions,
                       col->element_type, L, L, NULL,
                       searchRowids, searchDistances, &searchCount);
  if (rc != SQLITE_OK) {
    sqlite3_free(searchRowids);
    sqlite3_free(searchDistances);
    return rc;
  }

  // RobustPrune to select neighbors for x
  i64 *selectedNeighbors = sqlite3_malloc(cfg->n_neighbors * sizeof(i64));
  int selectedCount = 0;
  if (!selectedNeighbors) {
    sqlite3_free(searchRowids);
    sqlite3_free(searchDistances);
    return SQLITE_NOMEM;
  }

  rc = diskann_robust_prune(p, vec_col_idx, rowid, vector,
                             searchRowids, searchDistances, searchCount,
                             cfg->alpha, cfg->n_neighbors,
                             selectedNeighbors, &selectedCount);
  sqlite3_free(searchRowids);
  sqlite3_free(searchDistances);
  if (rc != SQLITE_OK) {
    sqlite3_free(selectedNeighbors);
    return rc;
  }

  // Write x's node with selected neighbors
  rc = diskann_write_pruned_neighbors(p, vec_col_idx, rowid,
                                       selectedNeighbors, selectedCount);
  if (rc != SQLITE_OK) {
    sqlite3_free(selectedNeighbors);
    return rc;
  }

  // Add bidirectional edges
  for (int i = 0; i < selectedCount; i++) {
    diskann_add_reverse_edge(p, vec_col_idx,
                              selectedNeighbors[i], rowid, vector);
  }

  sqlite3_free(selectedNeighbors);
  return SQLITE_OK;
}

/**
 * Insert a new vector into the DiskANN index (Algorithm 2: LM-Insert).
 * When buffer_threshold > 0, vectors are buffered and flushed in batch.
 */
static int diskann_insert(vec0_vtab *p, int vec_col_idx,
                           i64 rowid, const void *vector) {
  struct VectorColumnDefinition *col = &p->vector_columns[vec_col_idx];
  struct Vec0DiskannConfig *cfg = &col->diskann;
  int rc;
  size_t vectorSize = vector_column_byte_size(*col);

  // 1. Write full-precision vector to _vectors table (always needed for queries)
  rc = diskann_vector_write(p, vec_col_idx, rowid, vector, (int)vectorSize);
  if (rc != SQLITE_OK) return rc;

  // 2. If buffering is enabled, write to buffer instead of graph
  if (cfg->buffer_threshold > 0) {
    rc = diskann_buffer_write(p, vec_col_idx, rowid, vector, (int)vectorSize);
    if (rc != SQLITE_OK) return rc;

    i64 count;
    rc = diskann_buffer_count(p, vec_col_idx, &count);
    if (rc != SQLITE_OK) return rc;

    if (count >= cfg->buffer_threshold) {
      return diskann_flush_buffer(p, vec_col_idx);
    }
    return SQLITE_OK;
  }

  // 3. Legacy per-row insert directly into graph
  return diskann_insert_graph(p, vec_col_idx, rowid, vector);
}

/**
 * Returns 1 if ALL vector columns in this table are DiskANN-indexed.
 */
// ============================================================
// DiskANN delete (Algorithm 3 from LM-DiskANN paper)
// ============================================================

static int diskann_node_delete(vec0_vtab *p, int vec_col_idx, i64 rowid) {
  sqlite3_stmt *stmt = NULL;
  char *zSql = sqlite3_mprintf(
      "DELETE FROM " VEC0_SHADOW_DISKANN_NODES_N_NAME " WHERE rowid = ?",
      p->schemaName, p->tableName, vec_col_idx);
  if (!zSql) return SQLITE_NOMEM;
  int rc = sqlite3_prepare_v2(p->db, zSql, -1, &stmt, NULL);
  sqlite3_free(zSql);
  if (rc != SQLITE_OK) return rc;
  sqlite3_bind_int64(stmt, 1, rowid);
  rc = sqlite3_step(stmt);
  sqlite3_finalize(stmt);
  return (rc == SQLITE_DONE) ? SQLITE_OK : SQLITE_ERROR;
}

static int diskann_vector_delete(vec0_vtab *p, int vec_col_idx, i64 rowid) {
  sqlite3_stmt *stmt = NULL;
  char *zSql = sqlite3_mprintf(
      "DELETE FROM " VEC0_SHADOW_VECTORS_N_NAME " WHERE rowid = ?",
      p->schemaName, p->tableName, vec_col_idx);
  if (!zSql) return SQLITE_NOMEM;
  int rc = sqlite3_prepare_v2(p->db, zSql, -1, &stmt, NULL);
  sqlite3_free(zSql);
  if (rc != SQLITE_OK) return rc;
  sqlite3_bind_int64(stmt, 1, rowid);
  rc = sqlite3_step(stmt);
  sqlite3_finalize(stmt);
  return (rc == SQLITE_DONE) ? SQLITE_OK : SQLITE_ERROR;
}

/**
 * Repair graph after deleting a node. Following Algorithm 3 (LM-Delete):
 * For each neighbor n of the deleted node, add deleted node's other neighbors
 * to n's candidate set, then remove the deleted node from n's neighbor list.
 * Uses simple slot replacement rather than full RobustPrune for performance.
 */
static int diskann_repair_reverse_edges(
    vec0_vtab *p, int vec_col_idx, i64 deleted_rowid,
    const i64 *deleted_neighbors, int deleted_neighbor_count) {

  struct VectorColumnDefinition *col = &p->vector_columns[vec_col_idx];
  struct Vec0DiskannConfig *cfg = &col->diskann;
  int rc;

  // For each neighbor of the deleted node, fix their neighbor list
  for (int dn = 0; dn < deleted_neighbor_count; dn++) {
    i64 nodeRowid = deleted_neighbors[dn];

    u8 *validity = NULL, *neighborIds = NULL, *qvecs = NULL;
    int vs, nis, qs;
    rc = diskann_node_read(p, vec_col_idx, nodeRowid,
                            &validity, &vs, &neighborIds, &nis, &qvecs, &qs);
    if (rc != SQLITE_OK) continue;

    // Find and clear the deleted node's slot
    int clearedSlot = -1;
    for (int i = 0; i < cfg->n_neighbors; i++) {
      if (diskann_validity_get(validity, i) &&
          diskann_neighbor_id_get(neighborIds, i) == deleted_rowid) {
        diskann_node_clear_neighbor(validity, neighborIds, qvecs, i,
                                     cfg->quantizer_type, col->dimensions);
        clearedSlot = i;
        break;
      }
    }

    if (clearedSlot >= 0) {
      // Try to fill the cleared slot with one of the deleted node's other neighbors
      for (int di = 0; di < deleted_neighbor_count; di++) {
        i64 candidate = deleted_neighbors[di];
        if (candidate == nodeRowid || candidate == deleted_rowid) continue;

        // Check not already a neighbor
        int alreadyNeighbor = 0;
        for (int ni = 0; ni < cfg->n_neighbors; ni++) {
          if (diskann_validity_get(validity, ni) &&
              diskann_neighbor_id_get(neighborIds, ni) == candidate) {
            alreadyNeighbor = 1;
            break;
          }
        }
        if (alreadyNeighbor) continue;

        // Load, quantize, and set
        void *candidateVec = NULL;
        int cvs;
        rc = diskann_vector_read(p, vec_col_idx, candidate, &candidateVec, &cvs);
        if (rc != SQLITE_OK) continue;

        size_t qvecSize = diskann_quantized_vector_byte_size(
            cfg->quantizer_type, col->dimensions);
        u8 *qvec = sqlite3_malloc(qvecSize);
        if (qvec) {
          diskann_quantize_element_vector(candidateVec, col->element_type,
                                          col->dimensions, cfg->quantizer_type,
                                          qvec);
          diskann_node_set_neighbor(validity, neighborIds, qvecs, clearedSlot,
                                     candidate, qvec,
                                     cfg->quantizer_type, col->dimensions);
          sqlite3_free(qvec);
        }
        sqlite3_free(candidateVec);
        break;
      }

      rc = diskann_node_write(p, vec_col_idx, nodeRowid,
                               validity, vs, neighborIds, nis, qvecs, qs);
    }

    sqlite3_free(validity);
    sqlite3_free(neighborIds);
    sqlite3_free(qvecs);
    if (rc != SQLITE_OK) return rc;
  }

  return SQLITE_OK;
}

/**
 * Delete a vector from the DiskANN graph (Algorithm 3: LM-Delete).
 * If the vector is in the buffer (not yet flushed), just remove from buffer.
 */
/**
 * Scan all nodes and clear any neighbor slot referencing deleted_rowid.
 * This removes stale reverse edges that the forward-edge repair misses,
 * preventing data leaks (deleted rowid + quantized vector lingering in
 * other nodes' blobs).
 */
static int diskann_scrub_deleted_rowid(
    vec0_vtab *p, int vec_col_idx, i64 deleted_rowid) {

  struct VectorColumnDefinition *col = &p->vector_columns[vec_col_idx];
  struct Vec0DiskannConfig *cfg = &col->diskann;
  int rc;
  sqlite3_stmt *stmt = NULL;

  // Lightweight scan: only read validity + neighbor_ids to find matches
  char *zSql = sqlite3_mprintf(
      "SELECT rowid, neighbors_validity, neighbor_ids "
      "FROM " VEC0_SHADOW_DISKANN_NODES_N_NAME,
      p->schemaName, p->tableName, vec_col_idx);
  if (!zSql) return SQLITE_NOMEM;
  rc = sqlite3_prepare_v2(p->db, zSql, -1, &stmt, NULL);
  sqlite3_free(zSql);
  if (rc != SQLITE_OK) return rc;

  // Collect rowids that need updating (avoid modifying while iterating)
  i64 *dirty = NULL;
  int nDirty = 0, capDirty = 0;

  while (sqlite3_step(stmt) == SQLITE_ROW) {
    const u8 *validity = (const u8 *)sqlite3_column_blob(stmt, 1);
    const u8 *ids = (const u8 *)sqlite3_column_blob(stmt, 2);
    int idsBytes = sqlite3_column_bytes(stmt, 2);
    if (!validity || !ids) continue;

    int nSlots = idsBytes / (int)sizeof(i64);
    if (nSlots > cfg->n_neighbors) nSlots = cfg->n_neighbors;

    for (int i = 0; i < nSlots; i++) {
      if (!diskann_validity_get(validity, i)) continue;
      i64 nid = diskann_neighbor_id_get(ids, i);
      if (nid == deleted_rowid) {
        i64 nodeRowid = sqlite3_column_int64(stmt, 0);
        // Add to dirty list
        if (nDirty >= capDirty) {
          capDirty = capDirty ? capDirty * 2 : 16;
          i64 *tmp = sqlite3_realloc64(dirty, capDirty * sizeof(i64));
          if (!tmp) { sqlite3_free(dirty); sqlite3_finalize(stmt); return SQLITE_NOMEM; }
          dirty = tmp;
        }
        dirty[nDirty++] = nodeRowid;
        break;  // one match per node is enough
      }
    }
  }
  sqlite3_finalize(stmt);

  // Now do full read/clear/write for each dirty node
  for (int d = 0; d < nDirty; d++) {
    u8 *val = NULL, *nids = NULL, *qvecs = NULL;
    int vs, nis, qs;
    rc = diskann_node_read(p, vec_col_idx, dirty[d],
                            &val, &vs, &nids, &nis, &qvecs, &qs);
    if (rc != SQLITE_OK) continue;

    int modified = 0;
    for (int i = 0; i < cfg->n_neighbors; i++) {
      if (diskann_validity_get(val, i) &&
          diskann_neighbor_id_get(nids, i) == deleted_rowid) {
        diskann_node_clear_neighbor(val, nids, qvecs, i,
                                     cfg->quantizer_type, col->dimensions);
        modified = 1;
      }
    }

    if (modified) {
      rc = diskann_node_write(p, vec_col_idx, dirty[d],
                               val, vs, nids, nis, qvecs, qs);
    }

    sqlite3_free(val);
    sqlite3_free(nids);
    sqlite3_free(qvecs);
    if (rc != SQLITE_OK) break;
  }

  sqlite3_free(dirty);
  return rc;
}

static int diskann_delete(vec0_vtab *p, int vec_col_idx, i64 rowid) {
  struct VectorColumnDefinition *col = &p->vector_columns[vec_col_idx];
  struct Vec0DiskannConfig *cfg = &col->diskann;
  int rc;

  // Check if this rowid is in the buffer (not yet in graph)
  if (cfg->buffer_threshold > 0) {
    int inBuffer = 0;
    rc = diskann_buffer_exists(p, vec_col_idx, rowid, &inBuffer);
    if (rc != SQLITE_OK) return rc;
    if (inBuffer) {
      // Just remove from buffer and _vectors, no graph repair needed
      rc = diskann_buffer_delete(p, vec_col_idx, rowid);
      if (rc == SQLITE_OK) {
        rc = diskann_vector_delete(p, vec_col_idx, rowid);
      }
      return rc;
    }
  }

  // 1. Read the node to get its neighbor list
  u8 *delValidity = NULL, *delNeighborIds = NULL, *delQvecs = NULL;
  int dvs, dnis, dqs;
  rc = diskann_node_read(p, vec_col_idx, rowid,
                          &delValidity, &dvs, &delNeighborIds, &dnis,
                          &delQvecs, &dqs);
  if (rc != SQLITE_OK) {
    return SQLITE_OK;  // Node doesn't exist, nothing to do
  }

  i64 *deletedNeighbors = sqlite3_malloc(cfg->n_neighbors * sizeof(i64));
  int deletedNeighborCount = 0;
  if (!deletedNeighbors) {
    sqlite3_free(delValidity);
    sqlite3_free(delNeighborIds);
    sqlite3_free(delQvecs);
    return SQLITE_NOMEM;
  }

  for (int i = 0; i < cfg->n_neighbors; i++) {
    if (diskann_validity_get(delValidity, i)) {
      deletedNeighbors[deletedNeighborCount++] =
          diskann_neighbor_id_get(delNeighborIds, i);
    }
  }

  sqlite3_free(delValidity);
  sqlite3_free(delNeighborIds);
  sqlite3_free(delQvecs);

  // 2. Repair reverse edges
  rc = diskann_repair_reverse_edges(p, vec_col_idx, rowid,
                                     deletedNeighbors, deletedNeighborCount);
  sqlite3_free(deletedNeighbors);

  // 3. Delete node and vector
  if (rc == SQLITE_OK) {
    rc = diskann_node_delete(p, vec_col_idx, rowid);
  }
  if (rc == SQLITE_OK) {
    rc = diskann_vector_delete(p, vec_col_idx, rowid);
  }

  // 4. Handle medoid deletion
  if (rc == SQLITE_OK) {
    rc = diskann_medoid_handle_delete(p, vec_col_idx, rowid);
  }

  // 5. Scrub stale reverse edges — removes deleted rowid + quantized vector
  //    from any node that still references it (data leak prevention)
  if (rc == SQLITE_OK) {
    rc = diskann_scrub_deleted_rowid(p, vec_col_idx, rowid);
  }

  return rc;
}

static int vec0_all_columns_diskann(vec0_vtab *p) {
  for (int i = 0; i < p->numVectorColumns; i++) {
    if (p->vector_columns[i].index_type != VEC0_INDEX_TYPE_DISKANN) return 0;
  }
  return p->numVectorColumns > 0;
}

// ============================================================================
// Command dispatch
// ============================================================================

static int diskann_handle_command(vec0_vtab *p, const char *command) {
  int col_idx = -1;
  for (int i = 0; i < p->numVectorColumns; i++) {
    if (p->vector_columns[i].index_type == VEC0_INDEX_TYPE_DISKANN) { col_idx = i; break; }
  }
  if (col_idx < 0) return SQLITE_EMPTY;

  struct Vec0DiskannConfig *cfg = &p->vector_columns[col_idx].diskann;

  if (strncmp(command, "search_list_size_search=", 24) == 0) {
    int val = atoi(command + 24);
    if (val < 1) { vtab_set_error(&p->base, "search_list_size_search must be >= 1"); return SQLITE_ERROR; }
    cfg->search_list_size_search = val;
    return SQLITE_OK;
  }
  if (strncmp(command, "search_list_size_insert=", 24) == 0) {
    int val = atoi(command + 24);
    if (val < 1) { vtab_set_error(&p->base, "search_list_size_insert must be >= 1"); return SQLITE_ERROR; }
    cfg->search_list_size_insert = val;
    return SQLITE_OK;
  }
  if (strncmp(command, "search_list_size=", 17) == 0) {
    int val = atoi(command + 17);
    if (val < 1) { vtab_set_error(&p->base, "search_list_size must be >= 1"); return SQLITE_ERROR; }
    cfg->search_list_size = val;
    return SQLITE_OK;
  }
  return SQLITE_EMPTY;
}

#ifdef SQLITE_VEC_TEST
// Expose internal DiskANN data structures and functions for unit testing.

int _test_diskann_candidate_list_init(struct DiskannCandidateList *list, int capacity) {
  return diskann_candidate_list_init(list, capacity);
}
void _test_diskann_candidate_list_free(struct DiskannCandidateList *list) {
  diskann_candidate_list_free(list);
}
int _test_diskann_candidate_list_insert(struct DiskannCandidateList *list, long long rowid, float distance) {
  return diskann_candidate_list_insert(list, (i64)rowid, (f32)distance);
}
int _test_diskann_candidate_list_next_unvisited(const struct DiskannCandidateList *list) {
  return diskann_candidate_list_next_unvisited(list);
}
int _test_diskann_candidate_list_count(const struct DiskannCandidateList *list) {
  return list->count;
}
long long _test_diskann_candidate_list_rowid(const struct DiskannCandidateList *list, int i) {
  return (long long)list->items[i].rowid;
}
float _test_diskann_candidate_list_distance(const struct DiskannCandidateList *list, int i) {
  return (float)list->items[i].distance;
}
void _test_diskann_candidate_list_set_visited(struct DiskannCandidateList *list, int i) {
  list->items[i].visited = 1;
}

int _test_diskann_visited_set_init(struct DiskannVisitedSet *set, int capacity) {
  return diskann_visited_set_init(set, capacity);
}
void _test_diskann_visited_set_free(struct DiskannVisitedSet *set) {
  diskann_visited_set_free(set);
}
int _test_diskann_visited_set_contains(const struct DiskannVisitedSet *set, long long rowid) {
  return diskann_visited_set_contains(set, (i64)rowid);
}
int _test_diskann_visited_set_insert(struct DiskannVisitedSet *set, long long rowid) {
  return diskann_visited_set_insert(set, (i64)rowid);
}
#endif /* SQLITE_VEC_TEST */
//////////////////////////////////////////////////////////////////////////////
// End inlined: sqlite-vec-diskann.c
//////////////////////////////////////////////////////////////////////////////

#else
static int vec0_all_columns_diskann(vec0_vtab *p) { (void)p; return 0; }
#endif

int vec0_num_defined_user_columns(vec0_vtab *p) {
  return p->numVectorColumns + p->numPartitionColumns + p->numAuxiliaryColumns + p->numMetadataColumns;
}

/**
 * @brief Returns the index of the distance hidden column for the given vec0
 * table.
 *
 * @param p vec0 table
 * @return int
 */
int vec0_column_command_idx(vec0_vtab *p) {
  // Command column is the first hidden column (right after user columns)
  return VEC0_COLUMN_USERN_START + vec0_num_defined_user_columns(p);
}

int vec0_column_distance_idx(vec0_vtab *p) {
  int base = VEC0_COLUMN_USERN_START + vec0_num_defined_user_columns(p);
  return base + (p->hasCommandColumn ? 1 : 0);
}

int vec0_column_k_idx(vec0_vtab *p) {
  int base = VEC0_COLUMN_USERN_START + vec0_num_defined_user_columns(p);
  return base + (p->hasCommandColumn ? 2 : 1);
}

/**
 * Returns the column index for the hidden "mmr_lambda" column.
 */
int vec0_column_mmr_lambda_idx(vec0_vtab *p) {
  int base = VEC0_COLUMN_USERN_START + vec0_num_defined_user_columns(p);
  return base + (p->hasCommandColumn ? 3 : 2);
}

/**
 * Returns 1 if the given column-based index is a valid vector column,
 * 0 otherwise.
 */
int vec0_column_idx_is_vector(vec0_vtab *pVtab, int column_idx) {
  return column_idx >= VEC0_COLUMN_USERN_START &&
         column_idx <= (VEC0_COLUMN_USERN_START + vec0_num_defined_user_columns(pVtab) - 1) &&
         pVtab->user_column_kinds[column_idx - VEC0_COLUMN_USERN_START] == SQLITE_VEC0_USER_COLUMN_KIND_VECTOR;
}

/**
 * Returns the vector index of the given user column index.
 * ONLY call if validated with vec0_column_idx_is_vector before
 */
int vec0_column_idx_to_vector_idx(vec0_vtab *pVtab, int column_idx) {
  UNUSED_PARAMETER(pVtab);
  return pVtab->user_column_idxs[column_idx - VEC0_COLUMN_USERN_START];
}
/**
 * Returns 1 if the given column-based index is a "partition key" column,
 * 0 otherwise.
 */
int vec0_column_idx_is_partition(vec0_vtab *pVtab, int column_idx) {
  return column_idx >= VEC0_COLUMN_USERN_START &&
         column_idx <= (VEC0_COLUMN_USERN_START + vec0_num_defined_user_columns(pVtab) - 1) &&
         pVtab->user_column_kinds[column_idx - VEC0_COLUMN_USERN_START] == SQLITE_VEC0_USER_COLUMN_KIND_PARTITION;
}

/**
 * Returns the partition column index of the given user column index.
 * ONLY call if validated with vec0_column_idx_is_vector before
 */
int vec0_column_idx_to_partition_idx(vec0_vtab *pVtab, int column_idx) {
  UNUSED_PARAMETER(pVtab);
  return pVtab->user_column_idxs[column_idx - VEC0_COLUMN_USERN_START];
}

/**
 * Returns 1 if the given column-based index is a auxiliary column,
 * 0 otherwise.
 */
int vec0_column_idx_is_auxiliary(vec0_vtab *pVtab, int column_idx) {
  return column_idx >= VEC0_COLUMN_USERN_START &&
         column_idx <= (VEC0_COLUMN_USERN_START + vec0_num_defined_user_columns(pVtab) - 1) &&
         pVtab->user_column_kinds[column_idx - VEC0_COLUMN_USERN_START] == SQLITE_VEC0_USER_COLUMN_KIND_AUXILIARY;
}

/**
 * Returns the auxiliary column index of the given user column index.
 * ONLY call if validated with vec0_column_idx_to_partition_idx before
 */
int vec0_column_idx_to_auxiliary_idx(vec0_vtab *pVtab, int column_idx) {
  UNUSED_PARAMETER(pVtab);
  return pVtab->user_column_idxs[column_idx - VEC0_COLUMN_USERN_START];
}

/**
 * Returns 1 if the given column-based index is a metadata column,
 * 0 otherwise.
 */
int vec0_column_idx_is_metadata(vec0_vtab *pVtab, int column_idx) {
  return column_idx >= VEC0_COLUMN_USERN_START &&
         column_idx <= (VEC0_COLUMN_USERN_START + vec0_num_defined_user_columns(pVtab) - 1) &&
         pVtab->user_column_kinds[column_idx - VEC0_COLUMN_USERN_START] == SQLITE_VEC0_USER_COLUMN_KIND_METADATA;
}

/**
 * Returns the metadata column index of the given user column index.
 * ONLY call if validated with vec0_column_idx_is_metadata before
 */
int vec0_column_idx_to_metadata_idx(vec0_vtab *pVtab, int column_idx) {
  UNUSED_PARAMETER(pVtab);
  return pVtab->user_column_idxs[column_idx - VEC0_COLUMN_USERN_START];
}

/**
 * @brief Retrieve the chunk_id, chunk_offset, and possible "id" value
 * of a vec0_vtab row with the provided rowid
 *
 * @param p vec0_vtab
 * @param rowid the rowid of the row to query
 * @param id output, optional sqlite3_value to provide the id.
 *            Useful for text PK rows. Must be freed with sqlite3_value_free()
 * @param chunk_id output, the chunk_id the row belongs to
 * @param chunk_offset  output, the offset within the chunk the row belongs to
 * @return SQLITE_ROW on success, error code otherwise. SQLITE_EMPTY if row DNE
 */
int vec0_get_chunk_position(vec0_vtab *p, i64 rowid, sqlite3_value **id,
                            i64 *chunk_id, i64 *chunk_offset) {
  int rc;

  if (!p->stmtRowidsGetChunkPosition) {
    const char *zSql =
        sqlite3_mprintf("SELECT id, chunk_id, chunk_offset "
                        "FROM " VEC0_SHADOW_ROWIDS_NAME " WHERE rowid = ?",
                        p->schemaName, p->tableName);
    if (!zSql) {
      rc = SQLITE_NOMEM;
      goto cleanup;
    }
    rc = sqlite3_prepare_v2(p->db, zSql, -1, &p->stmtRowidsGetChunkPosition, 0);
    sqlite3_free((void *)zSql);
    if (rc != SQLITE_OK) {
      vtab_set_error(
          &p->base, VEC_INTERAL_ERROR
          "could not initialize 'rowids get chunk position' statement");
      goto cleanup;
    }
  }

  sqlite3_bind_int64(p->stmtRowidsGetChunkPosition, 1, rowid);
  rc = sqlite3_step(p->stmtRowidsGetChunkPosition);
  // special case: when no results, return SQLITE_EMPTY to convey "that chunk
  // position doesnt exist"
  if (rc == SQLITE_DONE) {
    rc = SQLITE_EMPTY;
    goto cleanup;
  }
  if (rc != SQLITE_ROW) {
    goto cleanup;
  }

  if (id) {
    sqlite3_value *value =
        sqlite3_column_value(p->stmtRowidsGetChunkPosition, 0);
    *id = sqlite3_value_dup(value);
    if (!*id) {
      rc = SQLITE_NOMEM;
      goto cleanup;
    }
  }

  if (chunk_id) {
    *chunk_id = sqlite3_column_int64(p->stmtRowidsGetChunkPosition, 1);
  }
  if (chunk_offset) {
    *chunk_offset = sqlite3_column_int64(p->stmtRowidsGetChunkPosition, 2);
  }

  rc = SQLITE_OK;

cleanup:
  sqlite3_reset(p->stmtRowidsGetChunkPosition);
  sqlite3_clear_bindings(p->stmtRowidsGetChunkPosition);
  return rc;
}

/**
 * @brief Return the id value from the _rowids table where _rowids.rowid =
 * rowid.
 *
 * @param pVtab: vec0 table to query
 * @param rowid: rowid of the row to query.
 * @param out: A dup'ed sqlite3_value of the id column. Might be null.
 *                         Must be cleaned up with sqlite3_value_free().
 * @returns SQLITE_OK on success, error code on failure
 */
int vec0_get_id_value_from_rowid(vec0_vtab *pVtab, i64 rowid,
                                 sqlite3_value **out) {
  // PERF: different strategy than get_chunk_position?
  return vec0_get_chunk_position((vec0_vtab *)pVtab, rowid, out, NULL, NULL);
}

int vec0_rowid_from_id(vec0_vtab *p, sqlite3_value *valueId, i64 *rowid) {
  sqlite3_stmt *stmt = NULL;
  int rc;
  char *zSql;
  zSql = sqlite3_mprintf("SELECT rowid"
                         " FROM " VEC0_SHADOW_ROWIDS_NAME " WHERE id = ?",
                         p->schemaName, p->tableName);
  if (!zSql) {
    rc = SQLITE_NOMEM;
    goto cleanup;
  }
  rc = sqlite3_prepare_v2(p->db, zSql, -1, &stmt, NULL);
  sqlite3_free(zSql);
  if (rc != SQLITE_OK) {
    goto cleanup;
  }
  sqlite3_bind_value(stmt, 1, valueId);
  rc = sqlite3_step(stmt);
  if (rc == SQLITE_DONE) {
    rc = SQLITE_EMPTY;
    goto cleanup;
  }
  if (rc != SQLITE_ROW) {
    goto cleanup;
  }
  *rowid = sqlite3_column_int64(stmt, 0);
  rc = sqlite3_step(stmt);
  if (rc != SQLITE_DONE) {
    goto cleanup;
  }

  rc = SQLITE_OK;

cleanup:
  sqlite3_finalize(stmt);
  return rc;
}

#if COMPILER_SUPPORTS_VTAB_IN
/**
 * Read a `rowid in (...)` (or `id in (...)`) KNN argument, a literal list or a
 * subquery, into a sorted array of rowids for vec0_rowids_contains(). An id
 * the table does not have is skipped. The caller frees *out with
 * array_cleanup() and sqlite3_free(), also on error.
 */
static int vec0_rowids_in_collect(vec0_vtab *p, sqlite3_value *in,
                                  struct Array **out) {
  sqlite3_value *item;
  struct Array *a = sqlite3_malloc(sizeof(*a));
  *out = a;
  if (!a) return SQLITE_NOMEM;
  memset(a, 0, sizeof(*a));
  int rc = array_init(a, sizeof(i64), 32);
  if (rc != SQLITE_OK) return rc;
  for (rc = sqlite3_vtab_in_first(in, &item); rc == SQLITE_OK && item;
       rc = sqlite3_vtab_in_next(in, &item)) {
    i64 rowid;
    if (p->pkIsText) {
      rc = vec0_rowid_from_id(p, item, &rowid);
      if (rc == SQLITE_EMPTY) continue;
      if (rc != SQLITE_OK) return rc;
    } else {
      rowid = sqlite3_value_int64(item);
    }
    rc = array_append(a, &rowid);
    if (rc != SQLITE_OK) return rc;
  }
  if (rc != SQLITE_DONE) return rc;
  qsort(a->z, a->length, a->element_size, _cmp);
  return SQLITE_OK;
}
#endif

int vec0_result_id(vec0_vtab *p, sqlite3_context *context, i64 rowid) {
  if (!p->pkIsText) {
    sqlite3_result_int64(context, rowid);
    return SQLITE_OK;
  }
  sqlite3_value *valueId;
  int rc = vec0_get_id_value_from_rowid(p, rowid, &valueId);
  if (rc != SQLITE_OK) {
    return rc;
  }
  if (!valueId) {
    sqlite3_result_error_nomem(context);
  } else {
    sqlite3_result_value(context, valueId);
    sqlite3_value_free(valueId);
  }
  return SQLITE_OK;
}

/**
 * @brief
 *
 * @param pVtab: virtual table to query
 * @param rowid: row to lookup
 * @param vector_column_idx: which vector column to query
 * @param outVector: Output pointer to the vector buffer.
 *                    Must be sqlite3_free()'ed.
 * @param outVectorSize: Pointer to a int where the size of outVector
 *                       will be stored.
 * @return int SQLITE_OK on success.
 */
#if SQLITE_VEC_EXPERIMENTAL_IVF_ENABLE
// Forward declaration — defined in sqlite-vec-ivf.c (included later)
static int ivf_get_vector_data(vec0_vtab *p, i64 rowid, int col_idx,
                                void **outVector, int *outVectorSize);
#endif
#if SQLITE_VEC_ENABLE_HNSW
// Forward declarations — defined in sqlite-vec-hnsw.c (included later)
static int hnsw_get_vector_data(vec0_vtab *p, i64 rowid, int col,
                                void **outVector, int *outVectorSize);
#endif

int vec0_get_vector_data(vec0_vtab *pVtab, i64 rowid, int vector_column_idx,
                         void **outVector, int *outVectorSize) {
  vec0_vtab *p = pVtab;
  int rc, brc;
  i64 chunk_id;
  i64 chunk_offset;
  assert((vector_column_idx >= 0) &&
         (vector_column_idx < pVtab->numVectorColumns));

  // Every index type registers its rows in _rowids, so a rowid with no
  // position there is no row, whichever table holds the column's vectors.
  rc = vec0_get_chunk_position(pVtab, rowid, NULL, &chunk_id, &chunk_offset);
  if (rc == SQLITE_EMPTY) {
    vtab_set_error(&pVtab->base, "Could not find a row with rowid %lld", rowid);
    return rc;
  }
  if (rc != SQLITE_OK) {
    return rc;
  }

  if (p->storage) {
    const void *data;
    int n;
    rc = vec0_storage_error(p, p->storage->xRead(p->storageHandle, rowid,
                                                 p->numAuxiliaryColumns + vector_column_idx,
                                                 &data, &n));
    if (rc != SQLITE_OK) return rc;
    if (!data || n != (int)vector_column_byte_size(p->vector_columns[vector_column_idx])) {
      vtab_set_error(&p->base, "vec0 storage: no vector for rowid %lld", rowid);
      return SQLITE_ERROR;
    }
    *outVector = sqlite3_malloc(n);
    if (!*outVector) return SQLITE_NOMEM;
    memcpy(*outVector, data, n);
    if (outVectorSize) *outVectorSize = n;
    return SQLITE_OK;
  }

#if SQLITE_VEC_ENABLE_DISKANN
  // DiskANN fast path: read from _vectors table
  if (p->vector_columns[vector_column_idx].index_type == VEC0_INDEX_TYPE_DISKANN) {
    void *vec = NULL;
    int vecSize;
    rc = diskann_vector_read(p, vector_column_idx, rowid, &vec, &vecSize);
    if (rc != SQLITE_OK) {
      vtab_set_error(&pVtab->base,
                     "Could not fetch vector data for %lld from DiskANN vectors table",
                     rowid);
      return SQLITE_ERROR;
    }
    *outVector = vec;
    if (outVectorSize) *outVectorSize = vecSize;
    return SQLITE_OK;
  }
#endif

#if SQLITE_VEC_ENABLE_HNSW
  if (p->vector_columns[vector_column_idx].index_type == VEC0_INDEX_TYPE_HNSW) {
    return hnsw_get_vector_data(p, rowid, vector_column_idx, outVector, outVectorSize);
  }
#endif

#if SQLITE_VEC_EXPERIMENTAL_IVF_ENABLE
  // IVF-indexed columns store vectors in _ivf_cells, not _vector_chunks
  if (p->vector_columns[vector_column_idx].index_type == VEC0_INDEX_TYPE_IVF) {
    return ivf_get_vector_data(p, rowid, vector_column_idx, outVector, outVectorSize);
  }
#endif
  size_t size;
  void *buf = NULL;
  int blobOffset;
  sqlite3_blob *vectorBlob = NULL;

#if SQLITE_VEC_ENABLE_RESCORE
  // Rescore columns store float vectors in _rescore_vectors (rowid-keyed)
  if (p->vector_columns[vector_column_idx].index_type == VEC0_INDEX_TYPE_RESCORE) {
    size = vector_column_byte_size(p->vector_columns[vector_column_idx]);
    rc = sqlite3_blob_open(p->db, p->schemaName,
                           p->shadowRescoreVectorsNames[vector_column_idx],
                           "vector", rowid, 0, &vectorBlob);
    if (rc != SQLITE_OK) {
      vtab_set_error(&pVtab->base,
                     "Could not fetch vector data for %lld from rescore vectors",
                     rowid);
      rc = SQLITE_ERROR;
      goto cleanup;
    }
    buf = sqlite3_malloc(size);
    if (!buf) {
      rc = SQLITE_NOMEM;
      goto cleanup;
    }
    rc = sqlite3_blob_read(vectorBlob, buf, size, 0);
    if (rc != SQLITE_OK) {
      sqlite3_free(buf);
      buf = NULL;
      rc = SQLITE_ERROR;
      goto cleanup;
    }
    *outVector = buf;
    if (outVectorSize) {
      *outVectorSize = size;
    }
    rc = SQLITE_OK;
    goto cleanup;
  }
#endif /* SQLITE_VEC_ENABLE_RESCORE */

  rc = sqlite3_blob_open(p->db, p->schemaName,
                         p->shadowVectorChunksNames[vector_column_idx],
                         "vectors", chunk_id, 0, &vectorBlob);

  if (rc != SQLITE_OK) {
    vtab_set_error(&pVtab->base,
                   "Could not fetch vector data for %lld, opening blob failed",
                   rowid);
    rc = SQLITE_ERROR;
    goto cleanup;
  }

  size = vector_column_byte_size(pVtab->vector_columns[vector_column_idx]);
  blobOffset = chunk_offset * size;

  buf = sqlite3_malloc(size);
  if (!buf) {
    rc = SQLITE_NOMEM;
    goto cleanup;
  }

  rc = sqlite3_blob_read(vectorBlob, buf, size, blobOffset);
  if (rc != SQLITE_OK) {
    sqlite3_free(buf);
    buf = NULL;
    vtab_set_error(
        &pVtab->base,
        "Could not fetch vector data for %lld, reading from blob failed",
        rowid);
    rc = SQLITE_ERROR;
    goto cleanup;
  }

  *outVector = buf;
  if (outVectorSize) {
    *outVectorSize = size;
  }
  rc = SQLITE_OK;

cleanup:
  brc = sqlite3_blob_close(vectorBlob);
  if ((rc == SQLITE_OK) && (brc != SQLITE_OK)) {
    vtab_set_error(
        &p->base, VEC_INTERAL_ERROR
        "unknown error, could not close vector blob, please file an issue");
    return brc;
  }

  return rc;
}

/**
 * @brief Retrieve the sqlite3_value of the i'th partition value for the given row.
 *
 * @param pVtab - the vec0_vtab in questions
 * @param rowid - rowid of target row
 * @param partition_idx - which partition column to retrieve
 * @param outValue - output sqlite3_value
 * @return int - SQLITE_OK on success, otherwise error code
 */
int vec0_get_partition_value_for_rowid(vec0_vtab *pVtab, i64 rowid, int partition_idx, sqlite3_value ** outValue) {
  int rc;
  i64 chunk_id;
  i64 chunk_offset;
  rc = vec0_get_chunk_position(pVtab, rowid, NULL, &chunk_id, &chunk_offset);
  if(rc != SQLITE_OK) {
    return rc;
  }
  sqlite3_stmt * stmt = NULL;
  char * zSql = sqlite3_mprintf("SELECT partition%02d FROM " VEC0_SHADOW_CHUNKS_NAME " WHERE chunk_id = ?", partition_idx, pVtab->schemaName, pVtab->tableName);
  if(!zSql) {
    return SQLITE_NOMEM;
  }
  rc = sqlite3_prepare_v2(pVtab->db, zSql, -1, &stmt, NULL);
  sqlite3_free(zSql);
  if(rc != SQLITE_OK) {
    return rc;
  }
  sqlite3_bind_int64(stmt, 1, chunk_id);
  rc = sqlite3_step(stmt);
  if(rc != SQLITE_ROW) {
    rc = SQLITE_ERROR;
    goto done;
  }
  *outValue = sqlite3_value_dup(sqlite3_column_value(stmt, 0));
  if(!*outValue) {
    rc = SQLITE_NOMEM;
    goto done;
  }
  rc = SQLITE_OK;

  done:
    sqlite3_finalize(stmt);
    return rc;

}

/**
 * @brief Get the value of an auxiliary column for the given rowid
 *
 * @param pVtab vec0_vtab
 * @param rowid the rowid of the row to lookup
 * @param auxiliary_idx aux index of the column we care about
 * @param outValue Output sqlite3_value to store
 * @return int SQLITE_OK on success, error code otherwise
 */
int vec0_get_auxiliary_value_for_rowid(vec0_vtab *pVtab, i64 rowid, int auxiliary_idx, sqlite3_value ** outValue);

/**
 * Set auxiliary column auxiliary_idx of rowid as the result of context, from
 * the storage provider or the _auxiliary shadow table.
 */
static void vec0_result_auxiliary(vec0_vtab *pVtab, sqlite3_context *context,
                                  i64 rowid, int auxiliary_idx) {
  if (pVtab->storage) {
    int rc = pVtab->storage->xColumn(pVtab->storageHandle, rowid, auxiliary_idx, context);
    if (rc != SQLITE_OK) {
      const char *msg = pVtab->storage->xErrmsg ? pVtab->storage->xErrmsg(pVtab->storageHandle) : NULL;
      if (msg) sqlite3_result_error(context, msg, -1);
      else sqlite3_result_error_code(context, rc);
    }
    return;
  }
  sqlite3_value * v;
  int rc = vec0_get_auxiliary_value_for_rowid(pVtab, rowid, auxiliary_idx, &v);
  if(rc == SQLITE_OK) {
    sqlite3_result_value(context, v);
    sqlite3_value_free(v);
  }else {
    sqlite3_result_error_code(context, rc);
  }
}

int vec0_get_auxiliary_value_for_rowid(vec0_vtab *pVtab, i64 rowid, int auxiliary_idx, sqlite3_value ** outValue) {
  int rc;
  sqlite3_stmt * stmt = NULL;
  char * zSql = sqlite3_mprintf("SELECT value%02d FROM " VEC0_SHADOW_AUXILIARY_NAME " WHERE rowid = ?", auxiliary_idx, pVtab->schemaName, pVtab->tableName);
  if(!zSql) {
    return SQLITE_NOMEM;
  }
  rc = sqlite3_prepare_v2(pVtab->db, zSql, -1, &stmt, NULL);
  sqlite3_free(zSql);
  if(rc != SQLITE_OK) {
    return rc;
  }
  sqlite3_bind_int64(stmt, 1, rowid);
  rc = sqlite3_step(stmt);
  if(rc != SQLITE_ROW) {
    rc = SQLITE_ERROR;
    goto done;
  }
  *outValue = sqlite3_value_dup(sqlite3_column_value(stmt, 0));
  if(!*outValue) {
    rc = SQLITE_NOMEM;
    goto done;
  }
  rc = SQLITE_OK;

  done:
    sqlite3_finalize(stmt);
    return rc;
}

/**
 * @brief Result the given metadata value for the given row and metadata column index.
 * Will traverse the metadatachunksNN table with BLOB I/0 for the given rowid.
 *
 * @param p
 * @param rowid
 * @param metadata_idx
 * @param context
 * @return int
 */
int vec0_result_metadata_value_for_rowid(vec0_vtab *p, i64 rowid, int metadata_idx, sqlite3_context * context) {
  int rc;
  i64 chunk_id;
  i64 chunk_offset;
  rc = vec0_get_chunk_position(p, rowid, NULL, &chunk_id, &chunk_offset);
  if(rc != SQLITE_OK) {
    return rc;
  }
  sqlite3_blob * blobValue;
  rc = sqlite3_blob_open(p->db, p->schemaName, p->shadowMetadataChunksNames[metadata_idx], "data", chunk_id, 0, &blobValue);
  if(rc != SQLITE_OK) {
    return rc;
  }

  switch(p->metadata_columns[metadata_idx].kind) {
    case VEC0_METADATA_COLUMN_KIND_BOOLEAN: {
      u8 block;
      rc = sqlite3_blob_read(blobValue, &block, sizeof(block), chunk_offset / CHAR_BIT);
      if(rc != SQLITE_OK) {
        goto done;
      }
      int value = block >> ((chunk_offset % CHAR_BIT)) & 1;
      sqlite3_result_int(context, value);
      break;
    }
    case VEC0_METADATA_COLUMN_KIND_INTEGER: {
      i64 value;
      rc = sqlite3_blob_read(blobValue, &value, sizeof(value), chunk_offset * sizeof(i64));
      if(rc != SQLITE_OK) {
        goto done;
      }
      sqlite3_result_int64(context, value);
      break;
    }
    case VEC0_METADATA_COLUMN_KIND_FLOAT: {
      double value;
      rc = sqlite3_blob_read(blobValue, &value, sizeof(value), chunk_offset * sizeof(double));
      if(rc != SQLITE_OK) {
        goto done;
      }
      sqlite3_result_double(context, value);
      break;
    }
    case VEC0_METADATA_COLUMN_KIND_TEXT: {
      u8 view[VEC0_METADATA_TEXT_VIEW_BUFFER_LENGTH];
      rc = sqlite3_blob_read(blobValue, &view, VEC0_METADATA_TEXT_VIEW_BUFFER_LENGTH, chunk_offset * VEC0_METADATA_TEXT_VIEW_BUFFER_LENGTH);
      if(rc != SQLITE_OK) {
        goto done;
      }
      int length = ((int *)view)[0];
      if(length <= VEC0_METADATA_TEXT_VIEW_DATA_LENGTH) {
        sqlite3_result_text(context, (const char*) (view + 4), length, SQLITE_TRANSIENT);
      }
      else {
        sqlite3_stmt * stmt;
        const char * zSql = sqlite3_mprintf("SELECT data FROM " VEC0_SHADOW_METADATA_TEXT_DATA_NAME " WHERE rowid = ?", p->schemaName, p->tableName, metadata_idx);
        if(!zSql) {
          rc = SQLITE_ERROR;
          goto done;
        }
        rc = sqlite3_prepare_v2(p->db, zSql, -1, &stmt, NULL);
        sqlite3_free((void *) zSql);
        if(rc != SQLITE_OK) {
          goto done;
        }
        sqlite3_bind_int64(stmt, 1, rowid);
        rc = sqlite3_step(stmt);
        if(rc != SQLITE_ROW) {
          sqlite3_finalize(stmt);
          rc = SQLITE_ERROR;
          goto done;
        }
        sqlite3_result_value(context, sqlite3_column_value(stmt, 0));
        sqlite3_finalize(stmt);
        rc = SQLITE_OK;
      }
      break;
    }
  }
  done:
    // blobValue is read-only, will not fail on close
    sqlite3_blob_close(blobValue);
    return rc;

}

int vec0_get_latest_chunk_rowid(vec0_vtab *p, i64 *chunk_rowid, sqlite3_value ** partitionKeyValues) {
  int rc;
  const char *zSql;
  // lazy initialize stmtLatestChunk when needed. May be cleared during xSync()
  if (!p->stmtLatestChunk) {
    if(p->numPartitionColumns > 0) {
      sqlite3_str * s = sqlite3_str_new(NULL);
      sqlite3_str_appendf(s, "SELECT max(rowid) FROM " VEC0_SHADOW_CHUNKS_NAME " WHERE ",
                           p->schemaName, p->tableName);

      for(int i = 0; i < p->numPartitionColumns; i++) {
        if(i != 0) {
          sqlite3_str_appendall(s, " AND ");
        }
        sqlite3_str_appendf(s, " partition%02d = ? ", i);
      }
      zSql = sqlite3_str_finish(s);
    }else {
      zSql = sqlite3_mprintf("SELECT max(rowid) FROM " VEC0_SHADOW_CHUNKS_NAME,
                           p->schemaName, p->tableName);
    }

    if (!zSql) {
      rc = SQLITE_NOMEM;
      goto cleanup;
    }
    rc = sqlite3_prepare_v2(p->db, zSql, -1, &p->stmtLatestChunk, 0);
    sqlite3_free((void *)zSql);
    if (rc != SQLITE_OK) {
      // IMP: V21406_05476
      vtab_set_error(&p->base, VEC_INTERAL_ERROR
                     "could not initialize 'latest chunk' statement");
      goto cleanup;
    }
  }

  for(int i = 0; i < p->numPartitionColumns; i++) {
    sqlite3_bind_value(p->stmtLatestChunk, i+1, (partitionKeyValues[i]));
  }

  rc = sqlite3_step(p->stmtLatestChunk);
  if (rc != SQLITE_ROW) {
    // IMP: V31559_15629
    vtab_set_error(&p->base, VEC_INTERAL_ERROR "Could not find latest chunk");
    rc = SQLITE_ERROR;
    goto cleanup;
  }
  if(sqlite3_column_type(p->stmtLatestChunk, 0) == SQLITE_NULL){
    rc = SQLITE_EMPTY;
    goto cleanup;
  }
  *chunk_rowid = sqlite3_column_int64(p->stmtLatestChunk, 0);
  rc = sqlite3_step(p->stmtLatestChunk);
  if (rc != SQLITE_DONE) {
    vtab_set_error(&p->base,
                   VEC_INTERAL_ERROR
                   "unknown result code when closing out stmtLatestChunk. "
                   "Please file an issue: " REPORT_URL,
                   p->schemaName, p->shadowChunksName);
    goto cleanup;
  }
  rc = SQLITE_OK;

cleanup:
  if (p->stmtLatestChunk) {
    sqlite3_reset(p->stmtLatestChunk);
    sqlite3_clear_bindings(p->stmtLatestChunk);
  }
  return rc;
}

int vec0_rowids_insert_rowid(vec0_vtab *p, i64 rowid) {
  int rc = SQLITE_OK;
  int entered = 0;
  UNUSED_PARAMETER(entered); // temporary
  if (!p->stmtRowidsInsertRowid) {
    const char *zSql =
        sqlite3_mprintf("INSERT INTO " VEC0_SHADOW_ROWIDS_NAME "(rowid)"
                        "VALUES (?);",
                        p->schemaName, p->tableName);
    if (!zSql) {
      rc = SQLITE_NOMEM;
      goto cleanup;
    }
    rc = sqlite3_prepare_v2(p->db, zSql, -1, &p->stmtRowidsInsertRowid, 0);
    sqlite3_free((void *)zSql);
    if (rc != SQLITE_OK) {
      vtab_set_error(&p->base, VEC_INTERAL_ERROR
                     "could not initialize 'insert rowids' statement");
      goto cleanup;
    }
  }

#if SQLITE_THREADSAFE
  if (sqlite3_mutex_enter) {
    sqlite3_mutex_enter(sqlite3_db_mutex(p->db));
    entered = 1;
  }
#endif
  sqlite3_bind_int64(p->stmtRowidsInsertRowid, 1, rowid);
  rc = sqlite3_step(p->stmtRowidsInsertRowid);

  if (rc != SQLITE_DONE) {
    if (sqlite3_extended_errcode(p->db) == SQLITE_CONSTRAINT_PRIMARYKEY) {
      // IMP: V17090_01160
      vtab_set_error(&p->base, "UNIQUE constraint failed on %s primary key",
                     p->tableName);
    } else {
      // IMP: V04679_21517
      vtab_set_error(&p->base,
                     "Error inserting rowid into rowids shadow table: %s",
                     sqlite3_errmsg(sqlite3_db_handle(p->stmtRowidsInsertId)));
    }
    rc = SQLITE_ERROR;
    goto cleanup;
  }

  rc = SQLITE_OK;

cleanup:
  if (p->stmtRowidsInsertRowid) {
    sqlite3_reset(p->stmtRowidsInsertRowid);
    sqlite3_clear_bindings(p->stmtRowidsInsertRowid);
  }

#if SQLITE_THREADSAFE
  if (sqlite3_mutex_leave && entered) {
    sqlite3_mutex_leave(sqlite3_db_mutex(p->db));
  }
#endif
  return rc;
}

int vec0_rowids_insert_id(vec0_vtab *p, sqlite3_value *idValue, i64 *rowid) {
  int rc = SQLITE_OK;
  int entered = 0;
  UNUSED_PARAMETER(entered); // temporary
  if (!p->stmtRowidsInsertId) {
    const char *zSql =
        sqlite3_mprintf("INSERT INTO " VEC0_SHADOW_ROWIDS_NAME "(id)"
                        "VALUES (?);",
                        p->schemaName, p->tableName);
    if (!zSql) {
      rc = SQLITE_NOMEM;
      goto complete;
    }
    rc = sqlite3_prepare_v2(p->db, zSql, -1, &p->stmtRowidsInsertId, 0);
    sqlite3_free((void *)zSql);
    if (rc != SQLITE_OK) {
      vtab_set_error(&p->base, VEC_INTERAL_ERROR
                     "could not initialize 'insert rowids id' statement");
      goto complete;
    }
  }

#if SQLITE_THREADSAFE
  if (sqlite3_mutex_enter) {
    sqlite3_mutex_enter(sqlite3_db_mutex(p->db));
    entered = 1;
  }
#endif

  if (idValue) {
    sqlite3_bind_value(p->stmtRowidsInsertId, 1, idValue);
  }
  rc = sqlite3_step(p->stmtRowidsInsertId);

  if (rc != SQLITE_DONE) {
    if (sqlite3_extended_errcode(p->db) == SQLITE_CONSTRAINT_UNIQUE) {
      // IMP: V20497_04568
      vtab_set_error(&p->base, "UNIQUE constraint failed on %s primary key",
                     p->tableName);
    } else {
      // IMP: V24016_08086
      // IMP: V15177_32015
      vtab_set_error(&p->base,
                     "Error inserting id into rowids shadow table: %s",
                     sqlite3_errmsg(sqlite3_db_handle(p->stmtRowidsInsertId)));
    }
    rc = SQLITE_ERROR;
    goto complete;
  }

  *rowid = sqlite3_last_insert_rowid(p->db);
  rc = SQLITE_OK;

complete:
  if (p->stmtRowidsInsertId) {
    sqlite3_reset(p->stmtRowidsInsertId);
    sqlite3_clear_bindings(p->stmtRowidsInsertId);
  }

#if SQLITE_THREADSAFE
  if (sqlite3_mutex_leave && entered) {
    sqlite3_mutex_leave(sqlite3_db_mutex(p->db));
  }
#endif
  return rc;
}

int vec0_metadata_chunk_size(vec0_metadata_column_kind kind, int chunk_size) {
  switch(kind) {
    case VEC0_METADATA_COLUMN_KIND_BOOLEAN:
      return chunk_size / 8;
    case VEC0_METADATA_COLUMN_KIND_INTEGER:
      return chunk_size * sizeof(i64);
    case VEC0_METADATA_COLUMN_KIND_FLOAT:
      return chunk_size * sizeof(double);
    case VEC0_METADATA_COLUMN_KIND_TEXT:
      return chunk_size * VEC0_METADATA_TEXT_VIEW_BUFFER_LENGTH;
  }
  return 0;
}

typedef union {
  i64 i[2];
  double f[2];
} Vec0ZoneMap;

static int vec0_ensure_stmt(vec0_vtab *p, sqlite3_stmt **pStmt, const char *fmt, int col_idx);

// State: 0 = missing/malformed, 1 = new empty chunk, 2 = populated bounds.
static int vec0_zonemap_read(vec0_vtab *p, int column, i64 chunk,
                            Vec0ZoneMap *bounds, int *state) {
  *state = 0;
  if (!p->hasCommandColumn) return SQLITE_OK;
  int rc = vec0_ensure_stmt(p, &p->stmtZoneMapRead,
                           "SELECT value FROM " VEC0_SHADOW_INFO_NAME " WHERE key=?1", 0);
  if (rc != SQLITE_OK) return rc;
  char key[64];
  sqlite3_snprintf(sizeof(key), key, "zonemap_%02d_%lld", column, chunk);
  sqlite3_stmt *stmt = p->stmtZoneMapRead;
  rc = sqlite3_bind_text(stmt, 1, key, -1, SQLITE_STATIC);
  if (rc == SQLITE_OK) {
    rc = sqlite3_step(stmt);
    if (rc == SQLITE_ROW) {
      if (sqlite3_column_type(stmt, 0) == SQLITE_NULL) {
        *state = 1;
      } else if (sqlite3_column_type(stmt, 0) == SQLITE_BLOB &&
                 sqlite3_column_bytes(stmt, 0) == sizeof(*bounds)) {
        const void *data = sqlite3_column_blob(stmt, 0);
        if (!data) { rc = SQLITE_NOMEM; goto done; }
        memcpy(bounds, data, sizeof(*bounds));
        *state = 2;
      }
      rc = SQLITE_OK;
    } else if (rc == SQLITE_DONE) {
      rc = SQLITE_OK;
    }
  }
done:
  sqlite3_reset(stmt);
  sqlite3_clear_bindings(stmt);
  return rc;
}

static int vec0_zonemap_store(vec0_vtab *p, int column, i64 chunk,
                             const Vec0ZoneMap *bounds) {
  if (!p->hasCommandColumn) return SQLITE_OK;
  int rc = vec0_ensure_stmt(p, &p->stmtZoneMapWrite,
                           "INSERT OR REPLACE INTO " VEC0_SHADOW_INFO_NAME
                           "(key,value) VALUES (?1,?2)", 0);
  if (rc != SQLITE_OK) return rc;
  char key[64];
  sqlite3_snprintf(sizeof(key), key, "zonemap_%02d_%lld", column, chunk);
  sqlite3_stmt *stmt = p->stmtZoneMapWrite;
  rc = sqlite3_bind_text(stmt, 1, key, -1, SQLITE_STATIC);
  if (rc == SQLITE_OK) {
    rc = bounds ? sqlite3_bind_blob(stmt, 2, bounds, sizeof(*bounds), SQLITE_STATIC)
                : sqlite3_bind_null(stmt, 2);
  }
  if (rc == SQLITE_OK) {
    rc = sqlite3_step(stmt);
    if (rc == SQLITE_DONE) rc = SQLITE_OK;
  }
  sqlite3_reset(stmt);
  sqlite3_clear_bindings(stmt);
  return rc;
}

static int vec0_zonemap_widen(vec0_vtab *p, int column, i64 chunk, sqlite3_value *value) {
  // shortcut: deletions leave wide bounds; rebuild only if pruning loss warrants the write cost.
  vec0_metadata_column_kind kind = p->metadata_columns[column].kind;
  if (kind == VEC0_METADATA_COLUMN_KIND_TEXT) return SQLITE_OK;
  Vec0ZoneMap bounds;
  int state;
  int rc = vec0_zonemap_read(p, column, chunk, &bounds, &state);
  // A missing entry may describe an old, populated chunk: never seed it from one row.
  if (rc != SQLITE_OK || !state) return rc;
  if (kind == VEC0_METADATA_COLUMN_KIND_FLOAT) {
    double v = sqlite3_value_double(value);
    if (state == 1) bounds.f[0] = bounds.f[1] = v;
    else if (!(bounds.f[0] <= bounds.f[1])) return SQLITE_OK;
    else if (v < bounds.f[0]) bounds.f[0] = v;
    else if (v > bounds.f[1]) bounds.f[1] = v;
    else return SQLITE_OK;
  } else {
    i64 v = sqlite3_value_int64(value);
    if (state == 1) bounds.i[0] = bounds.i[1] = v;
    else if (bounds.i[0] > bounds.i[1]) return SQLITE_OK;
    else if (v < bounds.i[0]) bounds.i[0] = v;
    else if (v > bounds.i[1]) bounds.i[1] = v;
    else return SQLITE_OK;
  }
  return vec0_zonemap_store(p, column, chunk, &bounds);
}

int vec0_rowids_update_position(vec0_vtab *p, i64 rowid, i64 chunk_rowid,
                                i64 chunk_offset) {
  int rc = SQLITE_OK;

  if (!p->stmtRowidsUpdatePosition) {
    const char *zSql = sqlite3_mprintf(" UPDATE " VEC0_SHADOW_ROWIDS_NAME
                                       " SET chunk_id = ?, chunk_offset = ?"
                                       " WHERE rowid = ?",
                                       p->schemaName, p->tableName);
    if (!zSql) {
      rc = SQLITE_NOMEM;
      goto cleanup;
    }
    rc = sqlite3_prepare_v2(p->db, zSql, -1, &p->stmtRowidsUpdatePosition, 0);
    sqlite3_free((void *)zSql);
    if (rc != SQLITE_OK) {
      vtab_set_error(&p->base, VEC_INTERAL_ERROR
                     "could not initialize 'update rowids position' statement");
      goto cleanup;
    }
  }

  sqlite3_bind_int64(p->stmtRowidsUpdatePosition, 1, chunk_rowid);
  sqlite3_bind_int64(p->stmtRowidsUpdatePosition, 2, chunk_offset);
  sqlite3_bind_int64(p->stmtRowidsUpdatePosition, 3, rowid);

  rc = sqlite3_step(p->stmtRowidsUpdatePosition);
  if (rc != SQLITE_DONE) {
    // IMP: V21925_05995
    vtab_set_error(&p->base,
                   VEC_INTERAL_ERROR
                   "could not update rowids position for rowid=%lld, "
                   "chunk_rowid=%lld, chunk_offset=%lld",
                   rowid, chunk_rowid, chunk_offset);
    rc = SQLITE_ERROR;
    goto cleanup;
  }
  rc = SQLITE_OK;

cleanup:
  if (p->stmtRowidsUpdatePosition) {
    sqlite3_reset(p->stmtRowidsUpdatePosition);
    sqlite3_clear_bindings(p->stmtRowidsUpdatePosition);
  }

  return rc;
}

/**
 * @brief Adds a new chunk for the vec0 table, and the corresponding vector
 * chunks.
 *
 * Inserts a new row into the _chunks table, with blank data, and uses that new
 * rowid to insert new blank rows into _vector_chunksXX tables.
 *
 * @param p: vec0 table to add new chunk
 * @param paritionKeyValues: Array of partition key valeus for the new chunk, if available
 * @param chunk_rowid: Output pointer, if not NULL, then will be filled with the
 * new chunk rowid.
 * @return int SQLITE_OK on success, error code otherwise.
 */
int vec0_new_chunk(vec0_vtab *p, sqlite3_value ** partitionKeyValues, i64 *chunk_rowid) {
  int rc;
  char *zSql;
  sqlite3_stmt *stmt;
  i64 rowid;

  // Step 1: Insert a new row in _chunks, capture that new rowid
  if(p->numPartitionColumns > 0) {
    sqlite3_str * s = sqlite3_str_new(NULL);
    sqlite3_str_appendf(s, "INSERT INTO " VEC0_SHADOW_CHUNKS_NAME, p->schemaName, p->tableName);
    sqlite3_str_appendall(s, "(size, validity, rowids");
    for(int i = 0; i < p->numPartitionColumns; i++) {
      sqlite3_str_appendf(s, ", partition%02d", i);
    }
    sqlite3_str_appendall(s, ") VALUES (?, ?, ?");
    for(int i = 0; i < p->numPartitionColumns; i++) {
      sqlite3_str_appendall(s, ", ?");
    }
    sqlite3_str_appendall(s, ")");

    zSql = sqlite3_str_finish(s);
  }else {
    zSql = sqlite3_mprintf("INSERT INTO " VEC0_SHADOW_CHUNKS_NAME
                         "(size, validity, rowids) "
                         "VALUES (?, ?, ?);",
                         p->schemaName, p->tableName);
  }

  if (!zSql) {
    return SQLITE_NOMEM;
  }
  rc = sqlite3_prepare_v2(p->db, zSql, -1, &stmt, NULL);
  sqlite3_free(zSql);
  if (rc != SQLITE_OK) {
    sqlite3_finalize(stmt);
    return rc;
  }

#if SQLITE_THREADSAFE
  if (sqlite3_mutex_enter) {
    sqlite3_mutex_enter(sqlite3_db_mutex(p->db));
  }
#endif

  sqlite3_bind_int64(stmt, 1, p->chunk_size);               // size
  sqlite3_bind_zeroblob(stmt, 2, p->chunk_size / CHAR_BIT); // validity bitmap
  sqlite3_bind_zeroblob(stmt, 3, p->chunk_size * sizeof(i64)); // rowids

  for(int i = 0; i < p->numPartitionColumns; i++) {
    sqlite3_bind_value(stmt, 4 + i, partitionKeyValues[i]);
  }

  rc = sqlite3_step(stmt);
  int failed = rc != SQLITE_DONE;
  rowid = sqlite3_last_insert_rowid(p->db);
#if SQLITE_THREADSAFE
  if (sqlite3_mutex_leave) {
    sqlite3_mutex_leave(sqlite3_db_mutex(p->db));
  }
#endif
  sqlite3_finalize(stmt);
  if (failed) {
    return SQLITE_ERROR;
  }

  // Step 2: Create new vector chunks for each vector column, with
  //          that new chunk_rowid.
  //
  // SHADOW_TABLE_ROWID_QUIRK: The _vector_chunksNN and _metadatachunksNN
  // shadow tables declare "rowid PRIMARY KEY" without the INTEGER type, so
  // the user-defined "rowid" column is NOT an alias for the internal SQLite
  // rowid (_rowid_). When only appending rows these two happen to stay in
  // sync, but after a chunk is deleted (vec0Update_Delete_DeleteChunk)
  // and a new one is created, the auto-assigned _rowid_ can diverge from the
  // user "rowid" value. Since sqlite3_blob_open() addresses rows by internal
  // _rowid_, we must explicitly set BOTH _rowid_ and "rowid" to the same
  // value so that later blob operations can find the row.
  //
  // The correct long-term fix is changing the schema to
  //   "rowid INTEGER PRIMARY KEY"
  // which makes it a true alias, but that would break existing databases.

  for (int i = 0; i < vec0_num_defined_user_columns(p); i++) {
    if(p->user_column_kinds[i] != SQLITE_VEC0_USER_COLUMN_KIND_VECTOR) {
      continue;
    }
    int vector_column_idx = p->user_column_idxs[i];

    // Non-FLAT columns (rescore, IVF, DiskANN) don't use _vector_chunks
    if (!vec0_column_in_chunks(p, vector_column_idx)) {
      continue;
    }

    i64 vectorsSize =
        p->chunk_size * vector_column_byte_size(p->vector_columns[vector_column_idx]);

    // See SHADOW_TABLE_ROWID_QUIRK above for why _rowid_ and rowid are both set.
    zSql = sqlite3_mprintf("INSERT INTO " VEC0_SHADOW_VECTOR_N_NAME
                           "(_rowid_, rowid, vectors)"
                           "VALUES (?, ?, ?)",
                           p->schemaName, p->tableName, vector_column_idx);
    if (!zSql) {
      return SQLITE_NOMEM;
    }
    rc = sqlite3_prepare_v2(p->db, zSql, -1, &stmt, NULL);
    sqlite3_free(zSql);

    if (rc != SQLITE_OK) {
      sqlite3_finalize(stmt);
      return rc;
    }

    sqlite3_bind_int64(stmt, 1, rowid);  // _rowid_ (internal SQLite rowid)
    sqlite3_bind_int64(stmt, 2, rowid);  // rowid   (user-defined column)
    sqlite3_bind_zeroblob64(stmt, 3, vectorsSize);

    rc = sqlite3_step(stmt);
    sqlite3_finalize(stmt);
    if (rc != SQLITE_DONE) {
      return rc;
    }
  }

#if SQLITE_VEC_ENABLE_RESCORE
  // Create new rescore chunks for each rescore-enabled vector column
  rc = rescore_new_chunk(p, rowid);
  if (rc != SQLITE_OK) {
    return rc;
  }
#endif

  // Step 3: Create new metadata chunks for each metadata column
  for (int i = 0; i < vec0_num_defined_user_columns(p); i++) {
    if(p->user_column_kinds[i] != SQLITE_VEC0_USER_COLUMN_KIND_METADATA) {
      continue;
    }
    int metadata_column_idx = p->user_column_idxs[i];
    // See SHADOW_TABLE_ROWID_QUIRK above for why _rowid_ and rowid are both set.
    zSql = sqlite3_mprintf("INSERT INTO " VEC0_SHADOW_METADATA_N_NAME
                           "(_rowid_, rowid, data)"
                           "VALUES (?, ?, ?)",
                           p->schemaName, p->tableName, metadata_column_idx);
    if (!zSql) {
      return SQLITE_NOMEM;
    }
    rc = sqlite3_prepare_v2(p->db, zSql, -1, &stmt, NULL);
    sqlite3_free(zSql);

    if (rc != SQLITE_OK) {
      sqlite3_finalize(stmt);
      return rc;
    }

    sqlite3_bind_int64(stmt, 1, rowid);  // _rowid_ (internal SQLite rowid)
    sqlite3_bind_int64(stmt, 2, rowid);  // rowid   (user-defined column)
    sqlite3_bind_zeroblob64(stmt, 3, vec0_metadata_chunk_size(p->metadata_columns[metadata_column_idx].kind, p->chunk_size));

    rc = sqlite3_step(stmt);
    sqlite3_finalize(stmt);
    if (rc != SQLITE_DONE) {
      return rc;
    }
    if (p->metadata_columns[metadata_column_idx].kind != VEC0_METADATA_COLUMN_KIND_TEXT) {
      rc = vec0_zonemap_store(p, metadata_column_idx, rowid, NULL);
      if (rc != SQLITE_OK) return rc;
    }
  }


  if (chunk_rowid) {
    *chunk_rowid = rowid;
  }

  return SQLITE_OK;
}

struct vec0_query_fullscan_data {
  // Every rowid, in storage order as of the scan's start.
  sqlite3_stmt *rowids_stmt;
  i8 done;
  // Per flat vector column, a handle on the vectors blob of the chunk the scan
  // last read from. A chunk's blob holds chunk_size vectors, and the first read
  // through a newly opened handle walks the blob's overflow pages up to the
  // row's offset, so a scan keeps it open across the chunk's rows instead of
  // opening it once per row.
  sqlite3_blob *vectorBlobs[VEC0_MAX_VECTOR_COLUMNS];
  i64 vectorBlobChunkIds[VEC0_MAX_VECTOR_COLUMNS];
};
void vec0_query_fullscan_data_clear(
    struct vec0_query_fullscan_data *fullscan_data) {
  if (!fullscan_data)
    return;

  if (fullscan_data->rowids_stmt) {
    sqlite3_finalize(fullscan_data->rowids_stmt);
    fullscan_data->rowids_stmt = NULL;
  }
  for (int i = 0; i < VEC0_MAX_VECTOR_COLUMNS; i++) {
    sqlite3_blob_close(fullscan_data->vectorBlobs[i]);
    fullscan_data->vectorBlobs[i] = NULL;
  }
}

struct vec0_query_knn_data {
  i64 k;
  i64 k_used;
  // Array of rowids of size k. Must be freed with sqlite3_free().
  i64 *rowids;
  // Array of distances of size k. Must be freed with sqlite3_free().
  f32 *distances;
  i64 current_idx;
};
void vec0_query_knn_data_clear(struct vec0_query_knn_data *knn_data) {
  if (!knn_data)
    return;

  if (knn_data->rowids) {
    sqlite3_free(knn_data->rowids);
    knn_data->rowids = NULL;
  }
  if (knn_data->distances) {
    sqlite3_free(knn_data->distances);
    knn_data->distances = NULL;
  }
}

struct vec0_query_point_data {
  i64 rowid;
  void *vectors[VEC0_MAX_VECTOR_COLUMNS];
  int done;
};
void vec0_query_point_data_clear(struct vec0_query_point_data *point_data) {
  if (!point_data)
    return;
  for (int i = 0; i < VEC0_MAX_VECTOR_COLUMNS; i++) {
    sqlite3_free(point_data->vectors[i]);
    point_data->vectors[i] = NULL;
  }
}

typedef enum {
  // If any values are updated, please update the ARCHITECTURE.md docs accordingly!

 VEC0_QUERY_PLAN_FULLSCAN = '1',
 VEC0_QUERY_PLAN_POINT = '2',
 VEC0_QUERY_PLAN_KNN = '3',
} vec0_query_plan;

typedef struct vec0_cursor vec0_cursor;
struct vec0_cursor {
  sqlite3_vtab_cursor base;

  vec0_query_plan query_plan;
  struct vec0_query_fullscan_data *fullscan_data;
  struct vec0_query_knn_data *knn_data;
  struct vec0_query_point_data *point_data;
};

void vec0_cursor_clear(vec0_cursor *pCur) {
  if (pCur->fullscan_data) {
    vec0_query_fullscan_data_clear(pCur->fullscan_data);
    sqlite3_free(pCur->fullscan_data);
    pCur->fullscan_data = NULL;
  }
  if (pCur->knn_data) {
    vec0_query_knn_data_clear(pCur->knn_data);
    sqlite3_free(pCur->knn_data);
    pCur->knn_data = NULL;
  }
  if (pCur->point_data) {
    vec0_query_point_data_clear(pCur->point_data);
    sqlite3_free(pCur->point_data);
    pCur->point_data = NULL;
  }
}

/**
 * Prepare a statement once and cache it in *pStmt. `fmt` is formatted with
 * the schema name, table name and vector column index, in that order.
 */
static int vec0_ensure_stmt(vec0_vtab *p, sqlite3_stmt **pStmt, const char *fmt,
                            int col_idx) {
  if (*pStmt) return SQLITE_OK;
  char *zSql = sqlite3_mprintf(fmt, p->schemaName, p->tableName, col_idx);
  if (!zSql) return SQLITE_NOMEM;
  int rc = sqlite3_prepare_v2(p->db, zSql, -1, pStmt, NULL);
  sqlite3_free(zSql);
  return rc;
}

/** Run a one-off statement whose only parameter is a rowid. */
static int vec0_exec_rowid(vec0_vtab *p, const char *fmt, int col_idx, i64 rowid) {
  sqlite3_stmt *s = NULL;
  char *zSql = sqlite3_mprintf(fmt, p->schemaName, p->tableName, col_idx);
  if (!zSql) return SQLITE_NOMEM;
  int rc = sqlite3_prepare_v2(p->db, zSql, -1, &s, NULL);
  sqlite3_free(zSql);
  if (rc != SQLITE_OK) return rc;
  sqlite3_bind_int64(s, 1, rowid);
  rc = sqlite3_step(s);
  sqlite3_finalize(s);
  return rc == SQLITE_DONE ? SQLITE_OK : rc;
}

// IVF index implementation — #include'd here after all struct/helper definitions
#if SQLITE_VEC_EXPERIMENTAL_IVF_ENABLE

//////////////////////////////////////////////////////////////////////////////
// Begin inlined: sqlite-vec-ivf-kmeans.c
//////////////////////////////////////////////////////////////////////////////

/**
 * sqlite-vec-ivf-kmeans.c — Pure k-means clustering algorithm.
 *
 * No SQLite dependency. Operates on float arrays in memory.
 * #include'd into sqlite-vec.c after struct definitions.
 */
// When opened standalone in an editor, pull in types so the LSP is happy.
// When #include'd from sqlite-vec.c, SQLITE_VEC_H is already defined.

#include <float.h>
#include <string.h>

#define VEC0_IVF_KMEANS_MAX_ITER     25
#define VEC0_IVF_KMEANS_DEFAULT_SEED  0

// Simple xorshift32 PRNG
static uint32_t ivf_xorshift32(uint32_t *state) {
  uint32_t x = *state;
  x ^= x << 13;
  x ^= x >> 17;
  x ^= x << 5;
  *state = x;
  return x;
}

// L2 squared distance between two float vectors
static float ivf_l2_dist(const float *a, const float *b, int D) {
  float sum = 0.0f;
  for (int d = 0; d < D; d++) {
    float diff = a[d] - b[d];
    sum += diff * diff;
  }
  return sum;
}

// Find nearest centroid for a single vector. Returns centroid index.
static int ivf_nearest_centroid(const float *vec, const float *centroids,
                                int D, int k) {
  float min_dist = FLT_MAX;
  int best = 0;
  for (int c = 0; c < k; c++) {
    float dist = ivf_l2_dist(vec, &centroids[c * D], D);
    if (dist < min_dist) {
      min_dist = dist;
      best = c;
    }
  }
  return best;
}

/**
 * K-means++ initialization.
 * Picks k initial centroids from the data with probability proportional
 * to squared distance from nearest existing centroid.
 */
static int ivf_kmeans_init_plusplus(const float *vectors, int N, int D,
                                    int k, uint32_t seed, float *centroids) {
  if (N <= 0 || k <= 0 || D <= 0)
    return -1;
  if (seed == 0)
    seed = 42;

  // Pick first centroid randomly
  int first = ivf_xorshift32(&seed) % N;
  memcpy(centroids, &vectors[first * D], D * sizeof(float));

  if (k == 1)
    return 0;

  // Allocate distance array
  float *dists = sqlite3_malloc64((i64)N * sizeof(float));
  if (!dists)
    return -1;

  for (int c = 1; c < k; c++) {
    // Compute D(x) = distance to nearest existing centroid
    double total = 0.0;
    for (int i = 0; i < N; i++) {
      float d = ivf_l2_dist(&vectors[i * D], &centroids[(c - 1) * D], D);
      if (c == 1 || d < dists[i]) {
        dists[i] = d;
      }
      total += dists[i];
    }

    // Weighted random selection
    if (total <= 0.0) {
      // All distances zero — pick randomly
      int pick = ivf_xorshift32(&seed) % N;
      memcpy(&centroids[c * D], &vectors[pick * D], D * sizeof(float));
    } else {
      double threshold = ((double)ivf_xorshift32(&seed) / (double)0xFFFFFFFF) * total;
      double cumulative = 0.0;
      int pick = N - 1;
      for (int i = 0; i < N; i++) {
        cumulative += dists[i];
        if (cumulative >= threshold) {
          pick = i;
          break;
        }
      }
      memcpy(&centroids[c * D], &vectors[pick * D], D * sizeof(float));
    }
  }

  sqlite3_free(dists);
  return 0;
}

/**
 * Lloyd's k-means algorithm.
 *
 * @param vectors   N*D float array (row-major)
 * @param N         number of vectors
 * @param D         dimensionality
 * @param k         number of clusters
 * @param max_iter  maximum iterations
 * @param seed      PRNG seed for initialization
 * @param out_centroids  output: k*D float array (caller-allocated)
 * @return 0 on success, -1 on error
 */
static int ivf_kmeans(const float *vectors, int N, int D, int k,
                       int max_iter, uint32_t seed, float *out_centroids) {
  if (N <= 0 || D <= 0 || k <= 0)
    return -1;

  // Clamp k to N
  if (k > N)
    k = N;

  // Allocate working memory
  int *assignments = sqlite3_malloc64((i64)N * sizeof(int));
  float *new_centroids = sqlite3_malloc64((i64)k * D * sizeof(float));
  int *counts = sqlite3_malloc64((i64)k * sizeof(int));

  if (!assignments || !new_centroids || !counts) {
    sqlite3_free(assignments);
    sqlite3_free(new_centroids);
    sqlite3_free(counts);
    return -1;
  }

  memset(assignments, -1, N * sizeof(int));

  // Initialize centroids via k-means++
  if (ivf_kmeans_init_plusplus(vectors, N, D, k, seed, out_centroids) != 0) {
    sqlite3_free(assignments);
    sqlite3_free(new_centroids);
    sqlite3_free(counts);
    return -1;
  }

  for (int iter = 0; iter < max_iter; iter++) {
    // Assignment step
    int changed = 0;
    for (int i = 0; i < N; i++) {
      int nearest = ivf_nearest_centroid(&vectors[i * D], out_centroids, D, k);
      if (nearest != assignments[i]) {
        assignments[i] = nearest;
        changed++;
      }
    }
    if (changed == 0)
      break;

    // Update step
    memset(new_centroids, 0, (size_t)k * D * sizeof(float));
    memset(counts, 0, k * sizeof(int));

    for (int i = 0; i < N; i++) {
      int c = assignments[i];
      counts[c]++;
      for (int d = 0; d < D; d++) {
        new_centroids[c * D + d] += vectors[i * D + d];
      }
    }

    for (int c = 0; c < k; c++) {
      if (counts[c] == 0) {
        // Empty cluster: reassign to farthest point from its nearest centroid
        float max_dist = -1.0f;
        int farthest = 0;
        for (int i = 0; i < N; i++) {
          float d = ivf_l2_dist(&vectors[i * D],
                                &out_centroids[assignments[i] * D], D);
          if (d > max_dist) {
            max_dist = d;
            farthest = i;
          }
        }
        memcpy(&out_centroids[c * D], &vectors[farthest * D],
               D * sizeof(float));
      } else {
        for (int d = 0; d < D; d++) {
          out_centroids[c * D + d] = new_centroids[c * D + d] / counts[c];
        }
      }
    }
  }

  sqlite3_free(assignments);
  sqlite3_free(new_centroids);
  sqlite3_free(counts);
  return 0;
}
//////////////////////////////////////////////////////////////////////////////
// End inlined: sqlite-vec-ivf-kmeans.c
//////////////////////////////////////////////////////////////////////////////


//////////////////////////////////////////////////////////////////////////////
// Begin inlined: sqlite-vec-ivf.c
//////////////////////////////////////////////////////////////////////////////

/**
 * sqlite-vec-ivf.c — IVF (Inverted File Index) for sqlite-vec.
 *
 * #include'd into sqlite-vec.c after struct definitions and before vec0_init().
 *
 * Storage: fixed-size packed blob cells (capped at IVF_CELL_MAX_VECTORS).
 * Multiple cell rows per centroid. cell_id is auto-increment rowid,
 * centroid_id is indexed for lookup. This keeps blobs small (~200KB)
 * and avoids expensive overflow page traversal on insert.
 */
#ifdef SQLITE_VEC_TEST
#define IVF_STATIC
#else
#define IVF_STATIC static
#endif

// When opened standalone in an editor, pull in sqlite-vec.c so the LSP
// can resolve all types (vec0_vtab, VectorColumnDefinition, etc.).
// When #include'd from sqlite-vec.c, SQLITE_VEC_H is already defined.

#define VEC0_IVF_DEFAULT_NLIST  128
#define VEC0_IVF_DEFAULT_NPROBE  10
#define VEC0_IVF_MAX_NLIST    65536
#define VEC0_IVF_CELL_MAX_VECTORS 64  // ~200KB per cell at 768-dim f32
#define VEC0_IVF_UNASSIGNED_CENTROID_ID (-1)

#define VEC0_SHADOW_IVF_CENTROIDS_NAME "\"%w\".\"%w_ivf_centroids%02d\""
#define VEC0_SHADOW_IVF_CELLS_NAME     "\"%w\".\"%w_ivf_cells%02d\""
#define VEC0_SHADOW_IVF_ROWID_MAP_NAME "\"%w\".\"%w_ivf_rowid_map%02d\""
#define VEC0_SHADOW_IVF_VECTORS_NAME   "\"%w\".\"%w_ivf_vectors%02d\""

// ============================================================================
// Parser
// ============================================================================

static int vec0_parse_ivf_options(struct Vec0Scanner *scanner,
                                   struct Vec0IvfConfig *config) {
  struct Vec0Token token;
  int rc;
  config->nlist = VEC0_IVF_DEFAULT_NLIST;
  config->nprobe = -1;
  config->quantizer = VEC0_IVF_QUANTIZER_NONE;
  config->oversample = 1;
  int nprobe_explicit = 0;

  rc = vec0_scanner_next(scanner, &token);
  if (rc != VEC0_TOKEN_RESULT_SOME || token.token_type != TOKEN_TYPE_LPAREN)
    return SQLITE_ERROR;

  rc = vec0_scanner_next(scanner, &token);
  if (rc == VEC0_TOKEN_RESULT_SOME && token.token_type == TOKEN_TYPE_RPAREN) {
    config->nprobe = VEC0_IVF_DEFAULT_NPROBE;
    return SQLITE_OK;
  }

  while (1) {
    if (rc != VEC0_TOKEN_RESULT_SOME || token.token_type != TOKEN_TYPE_IDENTIFIER)
      return SQLITE_ERROR;
    char *key = token.start;
    int keyLength = token.end - token.start;

    rc = vec0_scanner_next(scanner, &token);
    if (rc != VEC0_TOKEN_RESULT_SOME || token.token_type != TOKEN_TYPE_EQ)
      return SQLITE_ERROR;

    // Read value — can be digit or identifier
    rc = vec0_scanner_next(scanner, &token);
    if (rc != VEC0_TOKEN_RESULT_SOME) return SQLITE_ERROR;
    if (token.token_type != TOKEN_TYPE_DIGIT &&
        token.token_type != TOKEN_TYPE_IDENTIFIER)
      return SQLITE_ERROR;

    char *val = token.start;
    int valLength = token.end - token.start;

    if (sqlite3_strnicmp(key, "nlist", keyLength) == 0) {
      if (token.token_type != TOKEN_TYPE_DIGIT) return SQLITE_ERROR;
      int v = atoi(val);
      if (v < 0 || v > VEC0_IVF_MAX_NLIST) return SQLITE_ERROR;
      config->nlist = v;
    } else if (sqlite3_strnicmp(key, "nprobe", keyLength) == 0) {
      if (token.token_type != TOKEN_TYPE_DIGIT) return SQLITE_ERROR;
      int v = atoi(val);
      if (v < 1 || v > VEC0_IVF_MAX_NLIST) return SQLITE_ERROR;
      config->nprobe = v;
      nprobe_explicit = 1;
    } else if (sqlite3_strnicmp(key, "quantizer", keyLength) == 0) {
      if (token.token_type != TOKEN_TYPE_IDENTIFIER) return SQLITE_ERROR;
      if (sqlite3_strnicmp(val, "none", valLength) == 0) {
        config->quantizer = VEC0_IVF_QUANTIZER_NONE;
      } else if (sqlite3_strnicmp(val, "int8", valLength) == 0) {
        config->quantizer = VEC0_IVF_QUANTIZER_INT8;
      } else if (sqlite3_strnicmp(val, "binary", valLength) == 0) {
        config->quantizer = VEC0_IVF_QUANTIZER_BINARY;
      } else if (sqlite3_strnicmp(val, "rabitq", valLength) == 0) {
        config->quantizer = VEC0_IVF_QUANTIZER_RABITQ;
      } else if (sqlite3_strnicmp(val, "fp16", valLength) == 0) {
        config->quantizer = VEC0_IVF_QUANTIZER_FP16;
      } else if (sqlite3_strnicmp(val, "bf16", valLength) == 0) {
        config->quantizer = VEC0_IVF_QUANTIZER_BF16;
      } else if (sqlite3_strnicmp(val, "int4", valLength) == 0) {
        config->quantizer = VEC0_IVF_QUANTIZER_INT4;
      } else {
        return SQLITE_ERROR;
      }
    } else if (sqlite3_strnicmp(key, "oversample", keyLength) == 0) {
      if (token.token_type != TOKEN_TYPE_DIGIT) return SQLITE_ERROR;
      int v = atoi(val);
      if (v < 1) return SQLITE_ERROR;
      config->oversample = v;
    } else {
      return SQLITE_ERROR;
    }

    rc = vec0_scanner_next(scanner, &token);
    if (rc != VEC0_TOKEN_RESULT_SOME) return SQLITE_ERROR;
    if (token.token_type == TOKEN_TYPE_RPAREN) break;
    if (token.token_type != TOKEN_TYPE_COMMA) return SQLITE_ERROR;
    rc = vec0_scanner_next(scanner, &token);
  }

  if (config->nprobe < 0) config->nprobe = VEC0_IVF_DEFAULT_NPROBE;
  if (config->nlist > 0 && config->nprobe > config->nlist) {
    if (nprobe_explicit) return SQLITE_ERROR;
    config->nprobe = config->nlist;
  }

  // Validation: oversample > 1 only makes sense with quantization
  if (config->oversample > 1 && config->quantizer == VEC0_IVF_QUANTIZER_NONE) {
    return SQLITE_ERROR;
  }

  return SQLITE_OK;
}

// ============================================================================
// Helpers
// ============================================================================

/**
 * Size of a stored vector in bytes, accounting for quantization.
 */
static int ivf_vec_size(vec0_vtab *p, int col_idx) {
  int D = (int)p->vector_columns[col_idx].dimensions;
  switch (p->vector_columns[col_idx].ivf.quantizer) {
  case VEC0_IVF_QUANTIZER_INT8:    return D;
  case VEC0_IVF_QUANTIZER_BINARY:  return D / 8;
  case VEC0_IVF_QUANTIZER_RABITQ:  return D / 8 + 2 * (int)sizeof(float);
  case VEC0_IVF_QUANTIZER_FP16:
  case VEC0_IVF_QUANTIZER_BF16:    return D * 2;
  case VEC0_IVF_QUANTIZER_INT4:    return D / 2;
  default:                          return D * (int)sizeof(float);
  }
}

/**
 * Size of the full-precision vector in bytes (always float32).
 */
static int ivf_full_vec_size(vec0_vtab *p, int col_idx) {
  return (int)(p->vector_columns[col_idx].dimensions * sizeof(float));
}

/**
 * Quantize float32 vector to int8.
 * Uses unit normalization: clamp to [-1,1], scale to [-127,127].
 */
IVF_STATIC void ivf_quantize_int8(const float *src, int8_t *dst, int D) {
  for (int i = 0; i < D; i++) {
    float v = src[i];
    if (v > 1.0f) v = 1.0f;
    if (v < -1.0f) v = -1.0f;
    dst[i] = (int8_t)(v * 127.0f);
  }
}

/**
 * Quantize float32 vector to binary (sign-bit quantization).
 * Each bit = 1 if src[i] > 0, else 0.
 */
IVF_STATIC void ivf_quantize_binary(const float *src, uint8_t *dst, int D) {
  vec0_quantize_binary_f32(src, dst, (size_t)D);
}

/**
 * Random orthogonal transform shared by RaBitQ data and query vectors:
 * 3 rounds of seeded sign flips + Walsh-Hadamard butterflies. Every
 * butterfly is an orthonormal 2x2, so norms and inner products survive.
 * ponytail: fixed seed (all tables share one rotation), store a per-table
 * seed if adversarial data ever matters.
 */
IVF_STATIC void ivf_rabitq_rotate(float *v, int D) {
  const float h = 0.70710678118654752f;
  for (int round = 0; round < 3; round++) {
    uint32_t x = 0x9E3779B9u ^ (uint32_t)(round + 1) * 0x85EBCA6Bu;
    for (int i = 0; i < D; i++) {
      x ^= x << 13; x ^= x >> 17; x ^= x << 5;
      if (x & 1) v[i] = -v[i];
    }
    for (int stride = 1; stride < D; stride <<= 1) {
      for (int i = 0; i + stride < D; i++) {
        if (i & stride) continue;
        float a = v[i], b = v[i + stride];
        v[i] = (a + b) * h;
        v[i + stride] = (a - b) * h;
      }
    }
  }
}

/**
 * RaBitQ code (Gao & Long, SIGMOD'24), centered at the origin:
 *   [D/8 sign bits of P*x][f32 ||x||][f32 x0 = <P*x/||x||, sign(P*x)/sqrt(D)>]
 * x0 is the correction factor that makes the inner-product estimate unbiased.
 * ponytail: origin-centered, not centroid-centered like PRISM; codes are
 * centroid-independent so cells move freely. Center per-cell if recall needs it.
 */
IVF_STATIC void ivf_quantize_rabitq(const float *src, uint8_t *dst, int D) {
  float *r = sqlite3_malloc(D * (int)sizeof(float));
  float norm = 0.0f, abs_sum = 0.0f, x0 = 1.0f;
  memset(dst, 0, D / 8);
  if (r) {
    memcpy(r, src, D * sizeof(float));
    ivf_rabitq_rotate(r, D);
    for (int i = 0; i < D; i++) {
      norm += r[i] * r[i];
      abs_sum += fabsf(r[i]);
      if (r[i] > 0.0f) dst[i / 8] |= (1 << (i % 8));
    }
    sqlite3_free(r);
  }
  norm = sqrtf(norm);
  if (norm > 0.0f) x0 = abs_sum / (sqrtf((float)D) * norm);
  memcpy(dst + D / 8, &norm, sizeof(float));
  memcpy(dst + D / 8 + sizeof(float), &x0, sizeof(float));
}

/**
 * RaBitQ query side: [D f32 of P*q/||q||][f32 ||q||]. Size D*4+4.
 */
static int ivf_rabitq_query_size(int D) { return (D + 1) * (int)sizeof(float); }

IVF_STATIC void ivf_rabitq_prepare_query(const float *src, float *dst, int D) {
  memcpy(dst, src, D * sizeof(float));
  ivf_rabitq_rotate(dst, D);
  float norm = 0.0f;
  for (int i = 0; i < D; i++) norm += dst[i] * dst[i];
  norm = sqrtf(norm);
  if (norm > 0.0f) for (int i = 0; i < D; i++) dst[i] /= norm;
  dst[D] = norm;
}

/**
 * Estimated distance between a prepared query and a RaBitQ code:
 *   <o,q> ~= <sign(P*x)/sqrt(D), P*q/||q||> / x0
 */
IVF_STATIC float ivf_rabitq_distance(const float *q, const uint8_t *code,
                                     int D, int metric) {
  float dot = 0.0f;
  for (int i = 0; i < D; i++) {
    dot += (code[i / 8] & (1 << (i % 8))) ? q[i] : -q[i];
  }
  float norm, x0;
  memcpy(&norm, code + D / 8, sizeof(float));
  memcpy(&x0, code + D / 8 + sizeof(float), sizeof(float));
  float est = dot / (sqrtf((float)D) * x0);
  if (est > 1.0f) est = 1.0f;
  if (est < -1.0f) est = -1.0f;
  if (metric == VEC0_DISTANCE_METRIC_COSINE) return 1.0f - est;
  float qn = q[D];
  if (metric == VEC0_DISTANCE_METRIC_DOT) return -norm * qn * est;
  return norm * norm + qn * qn - 2.0f * norm * qn * est;
}

/**
 * IEEE half precision, round-to-nearest-even. Values past 65504 become inf.
 */
IVF_STATIC uint16_t ivf_f32_to_f16(float f) {
  uint32_t x;
  memcpy(&x, &f, 4);
  uint32_t sign = (x >> 16) & 0x8000;
  uint32_t mag = x & 0x7FFFFFFF;
  if (mag >= 0x7F800000) return (uint16_t)(sign | 0x7C00 | (mag > 0x7F800000 ? 0x200 : 0));
  if (mag >= 0x477FF000) return (uint16_t)(sign | 0x7C00);  // overflow -> inf
  if (mag < 0x38800000) {                                    // subnormal / zero
    if (mag < 0x33000000) return (uint16_t)sign;
    uint32_t m = (mag & 0x7FFFFF) | 0x800000;
    int shift = 113 - (int)(mag >> 23);
    uint32_t h = m >> (shift + 13);
    uint32_t rem = m & ((1u << (shift + 13)) - 1), half = 1u << (shift + 12);
    if (rem > half || (rem == half && (h & 1))) h++;
    return (uint16_t)(sign | h);
  }
  uint32_t h = ((mag - 0x38000000) >> 13);
  uint32_t rem = mag & 0x1FFF;
  if (rem > 0x1000 || (rem == 0x1000 && (h & 1))) h++;
  return (uint16_t)(sign | h);
}

IVF_STATIC float ivf_f16_to_f32(uint16_t h) {
  uint32_t sign = (uint32_t)(h & 0x8000) << 16;
  uint32_t exp = (h >> 10) & 0x1F, man = h & 0x3FF, x;
  if (exp == 0) {
    float f = (float)man * (1.0f / 16777216.0f);  // man * 2^-24
    return sign ? -f : f;
  }
  if (exp == 31) x = sign | 0x7F800000 | (man << 13);
  else x = sign | ((exp + 112) << 23) | (man << 13);
  float f;
  memcpy(&f, &x, 4);
  return f;
}

/** bfloat16: top 16 bits of float32, round-to-nearest-even. */
IVF_STATIC uint16_t ivf_f32_to_bf16(float f) {
  uint32_t x;
  memcpy(&x, &f, 4);
  if ((x & 0x7FFFFFFF) > 0x7F800000) return (uint16_t)((x >> 16) | 0x40);  // NaN
  x += 0x7FFF + ((x >> 16) & 1);
  return (uint16_t)(x >> 16);
}

IVF_STATIC float ivf_bf16_to_f32(uint16_t h) {
  uint32_t x = (uint32_t)h << 16;
  float f;
  memcpy(&f, &x, 4);
  return f;
}

/**
 * int4: clamp to [-1,1] like int8, scale to [-7,7], two per byte
 * (low nibble = even index). Same unit-range assumption as int8.
 */
IVF_STATIC void ivf_quantize_int4(const float *src, uint8_t *dst, int D) {
  for (int i = 0; i < D; i += 2) {
    int q[2];
    for (int j = 0; j < 2; j++) {
      float v = src[i + j];
      if (v > 1.0f) v = 1.0f;
      if (v < -1.0f) v = -1.0f;
      q[j] = (int)lrintf(v * 7.0f);
    }
    dst[i / 2] = (uint8_t)((q[0] & 0xF) | ((q[1] & 0xF) << 4));
  }
}

static inline float ivf_int4_at(const uint8_t *code, int i) {
  int nib = (code[i / 2] >> ((i & 1) * 4)) & 0xF;
  return (float)(nib >= 8 ? nib - 16 : nib) * (1.0f / 7.0f);
}

/**
 * Asymmetric distance between a float32 query and an fp16/bf16/int4 code:
 * the code is decoded element by element, never materialized.
 */
static float ivf_decoded_distance(const float *q, const uint8_t *code, int D,
                                  int quantizer, int metric) {
  double dot = 0, qq = 0, xx = 0, l1 = 0, l2 = 0;
  for (int i = 0; i < D; i++) {
    float x;
    if (quantizer == VEC0_IVF_QUANTIZER_INT4) {
      x = ivf_int4_at(code, i);
    } else {
      uint16_t h;
      memcpy(&h, code + 2 * i, 2);
      x = quantizer == VEC0_IVF_QUANTIZER_FP16 ? ivf_f16_to_f32(h) : ivf_bf16_to_f32(h);
    }
    float d = q[i] - x;
    dot += (double)q[i] * x;
    qq += (double)q[i] * q[i];
    xx += (double)x * x;
    l1 += fabsf(d);
    l2 += (double)d * d;
  }
  switch (metric) {
  case VEC0_DISTANCE_METRIC_COSINE:
    return (qq == 0 || xx == 0) ? 1.0f : (float)(1.0 - dot / (sqrt(qq) * sqrt(xx)));
  case VEC0_DISTANCE_METRIC_L1:
    return (float)l1;
  case VEC0_DISTANCE_METRIC_DOT:
    return (float)-dot;
  default:
    return (float)l2;
  }
}

/**
 * Quantize a float32 vector to the target type based on config.
 * dst must be pre-allocated to ivf_vec_size() bytes.
 * If quantizer=none, copies src as-is.
 */
static void ivf_quantize(vec0_vtab *p, int col_idx,
                          const float *src, void *dst) {
  int D = (int)p->vector_columns[col_idx].dimensions;
  switch (p->vector_columns[col_idx].ivf.quantizer) {
  case VEC0_IVF_QUANTIZER_INT8:
    ivf_quantize_int8(src, (int8_t *)dst, D);
    break;
  case VEC0_IVF_QUANTIZER_BINARY:
    ivf_quantize_binary(src, (uint8_t *)dst, D);
    break;
  case VEC0_IVF_QUANTIZER_RABITQ:
    ivf_quantize_rabitq(src, (uint8_t *)dst, D);
    break;
  case VEC0_IVF_QUANTIZER_FP16:
  case VEC0_IVF_QUANTIZER_BF16: {
    int fp16 = p->vector_columns[col_idx].ivf.quantizer == VEC0_IVF_QUANTIZER_FP16;
    for (int i = 0; i < D; i++) {
      uint16_t h = fp16 ? ivf_f32_to_f16(src[i]) : ivf_f32_to_bf16(src[i]);
      memcpy((uint8_t *)dst + 2 * i, &h, 2);
    }
    break;
  }
  case VEC0_IVF_QUANTIZER_INT4:
    ivf_quantize_int4(src, (uint8_t *)dst, D);
    break;
  default:
    memcpy(dst, src, D * sizeof(float));
    break;
  }
}

/**
 * Query-side representation passed as `a` to ivf_distance(). Same as the
 * stored code for symmetric quantizers; RaBitQ is asymmetric (float query
 * vs. bit code). Caller frees.
 */
static void *ivf_query_rep(vec0_vtab *p, int col_idx, const float *src) {
  int D = (int)p->vector_columns[col_idx].dimensions;
  if (p->vector_columns[col_idx].ivf.quantizer == VEC0_IVF_QUANTIZER_RABITQ) {
    float *q = sqlite3_malloc(ivf_rabitq_query_size(D));
    if (q) ivf_rabitq_prepare_query(src, q, D);
    return q;
  }
  switch (p->vector_columns[col_idx].ivf.quantizer) {
  case VEC0_IVF_QUANTIZER_FP16:
  case VEC0_IVF_QUANTIZER_BF16:
  case VEC0_IVF_QUANTIZER_INT4: {
    float *q = sqlite3_malloc(D * (int)sizeof(float));
    if (q) memcpy(q, src, D * sizeof(float));
    return q;
  }
  }
  void *q = sqlite3_malloc(ivf_vec_size(p, col_idx));
  if (q) ivf_quantize(p, col_idx, src, q);
  return q;
}

// Forward declaration
static float ivf_distance(vec0_vtab *p, int col_idx, const void *a, const void *b);

/**
 * Find nearest centroid. Works with quantized or float centroids.
 * vec and centroids must be in the same representation (both quantized or both float).
 * vecSize = size of one vector in bytes.
 */
static int ivf_find_nearest_centroid(vec0_vtab *p, int col_idx,
                                      const void *vec, const void *centroids,
                                      int vecSize, int k) {
  float min_dist = FLT_MAX;
  int best = 0;
  const unsigned char *cdata = (const unsigned char *)centroids;
  for (int c = 0; c < k; c++) {
    float dist = ivf_distance(p, col_idx, vec, cdata + c * vecSize);
    if (dist < min_dist) { min_dist = dist; best = c; }
  }
  return best;
}


/**
 * Compute distance between two quantized vectors.
 * For int8: uses L2 or cosine on int8.
 * For binary: uses hamming distance.
 * For none: the column's exact distance.
 */
static float ivf_distance(vec0_vtab *p, int col_idx,
                           const void *a, const void *b) {
  size_t dims = p->vector_columns[col_idx].dimensions;
  switch (p->vector_columns[col_idx].ivf.quantizer) {
  case VEC0_IVF_QUANTIZER_INT8:
    return distance_l2_sqr_int8(a, b, &dims);
  case VEC0_IVF_QUANTIZER_BINARY:
    return distance_hamming(a, b, &dims);
  case VEC0_IVF_QUANTIZER_RABITQ:
    return ivf_rabitq_distance((const float *)a, (const uint8_t *)b, (int)dims,
                               p->vector_columns[col_idx].distance_metric);
  case VEC0_IVF_QUANTIZER_FP16:
  case VEC0_IVF_QUANTIZER_BF16:
  case VEC0_IVF_QUANTIZER_INT4:
    return ivf_decoded_distance((const float *)a, (const uint8_t *)b, (int)dims,
                                p->vector_columns[col_idx].ivf.quantizer,
                                p->vector_columns[col_idx].distance_metric);
  default:
    return vec0_compute_distance(&p->vector_columns[col_idx], a, b);
  }
}

static int ivf_exec(vec0_vtab *p, const char *fmt, int col_idx) {
  sqlite3_stmt *stmt = NULL;
  char *zSql = sqlite3_mprintf(fmt, p->schemaName, p->tableName, col_idx);
  if (!zSql) return SQLITE_NOMEM;
  int rc = sqlite3_prepare_v2(p->db, zSql, -1, &stmt, NULL);
  sqlite3_free(zSql);
  if (rc != SQLITE_OK) return rc;
  rc = sqlite3_step(stmt);
  sqlite3_finalize(stmt);
  return rc == SQLITE_DONE ? SQLITE_OK : rc;
}

static int ivf_is_trained(vec0_vtab *p, int col_idx, int *trained) {
  sqlite3_stmt *stmt = NULL;
  *trained = 0;
  char *zSql = sqlite3_mprintf(
      "SELECT value FROM " VEC0_SHADOW_INFO_NAME " WHERE key = 'ivf_trained_%d'",
      p->schemaName, p->tableName, col_idx);
  if (!zSql) return SQLITE_NOMEM;
  int rc = sqlite3_prepare_v2(p->db, zSql, -1, &stmt, NULL);
  sqlite3_free(zSql);
  if (rc != SQLITE_OK) return rc;
  rc = sqlite3_step(stmt);
  if (rc == SQLITE_ROW) *trained = (sqlite3_column_int(stmt, 0) == 1);
  sqlite3_finalize(stmt);
  return (rc == SQLITE_ROW || rc == SQLITE_DONE) ? SQLITE_OK : rc;
}

// ============================================================================
// Cell operations — fixed-size cells, multiple rows per centroid
// ============================================================================

#define VEC0_IVF_BUFFER_CELLS 64
#define VEC0_IVF_BUFFER_BYTES (8 * 1024 * 1024)

struct IvfBufferedCell {
  i64 id, centroid;
  int col, n, vectorBytes;
  unsigned char validity[VEC0_IVF_CELL_MAX_VECTORS / 8];
  i64 rowids[VEC0_IVF_CELL_MAX_VECTORS];
  unsigned char vectors[];
};

struct IvfWriteBuffer {
  struct IvfBufferedCell *cells[VEC0_IVF_BUFFER_CELLS];
  int n;
  sqlite3_uint64 bytes;
};

static void ivf_buffer_free(struct IvfWriteBuffer *b) {
  if (!b) return;
  for (int i = 0; i < b->n; i++) sqlite3_free(b->cells[i]);
  sqlite3_free(b);
}

static void ivf_buffer_clear(vec0_vtab *p) {
  p->ivfBufferGeneration++;
  ivf_buffer_free(p->ivfBuffer);
  p->ivfBuffer = NULL;
}

static struct IvfBufferedCell *ivf_buffer_find(vec0_vtab *p, int col, i64 id) {
  if (!p->ivfBuffer) return NULL;
  for (int i = 0; i < p->ivfBuffer->n; i++) {
    struct IvfBufferedCell *c = p->ivfBuffer->cells[i];
    if (c->col == col && c->id == id) return c;
  }
  return NULL;
}

static int ivf_flush(vec0_vtab *p) {
  struct IvfWriteBuffer *b = p->ivfBuffer;
  if (!b) return SQLITE_OK;
  // sqlite3_step can invoke rollback callbacks: the active flush owns its cells.
  p->ivfBuffer = NULL;
  p->ivfFlushing = 1;
  unsigned generation = p->ivfBufferGeneration;
  int rc = SQLITE_OK;
  for (int i = 0; rc == SQLITE_OK && i < b->n; i++) {
    struct IvfBufferedCell *c = b->cells[i];
    rc = vec0_ensure_stmt(p, &p->stmtIvfCellWrite[c->col],
        "UPDATE " VEC0_SHADOW_IVF_CELLS_NAME
        " SET n_vectors=?, validity=?, rowids=?, vectors=? WHERE cell_id=?", c->col);
    if (rc != SQLITE_OK) break;
    sqlite3_stmt *s = p->stmtIvfCellWrite[c->col];
    sqlite3_reset(s);
    sqlite3_bind_int(s, 1, c->n);
    sqlite3_bind_blob(s, 2, c->validity, sizeof(c->validity), SQLITE_TRANSIENT);
    sqlite3_bind_blob(s, 3, c->rowids, sizeof(c->rowids), SQLITE_TRANSIENT);
    sqlite3_bind_blob(s, 4, c->vectors, c->vectorBytes, SQLITE_TRANSIENT);
    sqlite3_bind_int64(s, 5, c->id);
    rc = sqlite3_step(s);
    if (rc == SQLITE_DONE) rc = sqlite3_changes(p->db) == 1 ? SQLITE_OK : SQLITE_CORRUPT;
    if (rc != SQLITE_OK && rc != SQLITE_INTERRUPT)
      vtab_set_error(&p->base, "Could not flush IVF cell %lld: %s", c->id, sqlite3_errmsg(p->db));
    sqlite3_reset(s);
    sqlite3_clear_bindings(s);
  }
  p->ivfFlushing = 0;
  if (rc == SQLITE_OK || generation != p->ivfBufferGeneration) ivf_buffer_free(b);
  else p->ivfBuffer = b;
  return rc;
}

static int ivf_buffer_load(vec0_vtab *p, int col, i64 id, struct IvfBufferedCell **out) {
  struct IvfBufferedCell *c = ivf_buffer_find(p, col, id);
  if (c) { *out = c; return SQLITE_OK; }
  int vectorBytes = VEC0_IVF_CELL_MAX_VECTORS * ivf_vec_size(p, col);
  sqlite3_uint64 bytes = sizeof(*c) + (sqlite3_uint64)vectorBytes;
  if (p->ivfBuffer && (p->ivfBuffer->n == VEC0_IVF_BUFFER_CELLS ||
                       p->ivfBuffer->bytes + bytes > VEC0_IVF_BUFFER_BYTES)) {
    int rc = ivf_flush(p);
    if (rc != SQLITE_OK) return rc;
  }
  c = sqlite3_malloc64(bytes);
  if (!c) return SQLITE_NOMEM;
  char *sql = sqlite3_mprintf("SELECT centroid_id,n_vectors,validity,rowids,vectors FROM "
                            VEC0_SHADOW_IVF_CELLS_NAME " WHERE cell_id=?",
                            p->schemaName, p->tableName, col);
  sqlite3_stmt *s = NULL;
  int rc = sql ? sqlite3_prepare_v2(p->db, sql, -1, &s, NULL) : SQLITE_NOMEM;
  sqlite3_free(sql);
  if (rc == SQLITE_OK) {
    sqlite3_bind_int64(s, 1, id);
    rc = sqlite3_step(s);
    if (rc == SQLITE_ROW) {
      c->centroid = sqlite3_column_int64(s, 0);
      c->n = sqlite3_column_int(s, 1);
      if (c->n < 0 || c->n > VEC0_IVF_CELL_MAX_VECTORS ||
          sqlite3_column_bytes(s, 2) != sizeof(c->validity) ||
          sqlite3_column_bytes(s, 3) != sizeof(c->rowids) ||
          sqlite3_column_bytes(s, 4) != vectorBytes) {
        rc = SQLITE_CORRUPT;
      } else {
        memcpy(c->validity, sqlite3_column_blob(s, 2), sizeof(c->validity));
        memcpy(c->rowids, sqlite3_column_blob(s, 3), sizeof(c->rowids));
        memcpy(c->vectors, sqlite3_column_blob(s, 4), vectorBytes);
        int live = 0;
        for (int slot = 0; slot < VEC0_IVF_CELL_MAX_VECTORS; slot++)
          live += !!(c->validity[slot / 8] & (1 << (slot % 8)));
        rc = live == c->n ? SQLITE_OK : SQLITE_CORRUPT;
      }
    } else if (rc == SQLITE_DONE) rc = SQLITE_CORRUPT;
  }
  sqlite3_finalize(s);
  if (rc != SQLITE_OK) { sqlite3_free(c); return rc; }
  if (!p->ivfBuffer) {
    p->ivfBuffer = sqlite3_malloc(sizeof(*p->ivfBuffer));
    if (!p->ivfBuffer) { sqlite3_free(c); return SQLITE_NOMEM; }
    memset(p->ivfBuffer, 0, sizeof(*p->ivfBuffer));
    p->ivfBuffer->bytes = sizeof(*p->ivfBuffer);
  }
  c->id = id; c->col = col; c->vectorBytes = vectorBytes;
  p->ivfBuffer->cells[p->ivfBuffer->n++] = c;
  p->ivfBuffer->bytes += bytes;
  *out = c;
  return SQLITE_OK;
}

/**
 * Create a new cell row. Returns the new cell_id (rowid) via *out_cell_id.
 */
static int ivf_cell_create(vec0_vtab *p, int col_idx, i64 centroid_id,
                            i64 *out_cell_id) {
  sqlite3_stmt *stmt = NULL;
  int rc;
  if (out_cell_id) *out_cell_id = 0;
  int cap = VEC0_IVF_CELL_MAX_VECTORS;
  int vecSize = ivf_vec_size(p, col_idx);
  char *zSql = sqlite3_mprintf(
      "INSERT INTO " VEC0_SHADOW_IVF_CELLS_NAME
      " (centroid_id, n_vectors, validity, rowids, vectors) VALUES (?, 0, ?, ?, ?)",
      p->schemaName, p->tableName, col_idx);
  if (!zSql) return SQLITE_NOMEM;
  rc = sqlite3_prepare_v2(p->db, zSql, -1, &stmt, NULL);
  sqlite3_free(zSql);
  if (rc != SQLITE_OK) return rc;
  sqlite3_bind_int64(stmt, 1, centroid_id);
  sqlite3_bind_zeroblob(stmt, 2, cap / 8);
  sqlite3_bind_zeroblob(stmt, 3, cap * (int)sizeof(i64));
  sqlite3_bind_zeroblob(stmt, 4, cap * vecSize);
  rc = sqlite3_step(stmt);
  sqlite3_finalize(stmt);
  if (rc != SQLITE_DONE) return rc;
  if (out_cell_id) *out_cell_id = sqlite3_last_insert_rowid(p->db);
  return SQLITE_OK;
}

/**
 * Find a cell with space for the given centroid, or create one.
 * Returns cell_id (rowid); buffered counts override persisted counts.
 */
static int ivf_cell_find_or_create(vec0_vtab *p, int col_idx, i64 centroid_id,
                                     i64 *out_cell_id) {
  if (p->ivfBuffer) {
    for (int i = 0; i < p->ivfBuffer->n; i++) {
      struct IvfBufferedCell *c = p->ivfBuffer->cells[i];
      if (c->col == col_idx && c->centroid == centroid_id && c->n < VEC0_IVF_CELL_MAX_VECTORS) {
        *out_cell_id = c->id;
        return SQLITE_OK;
      }
    }
  }
  int rc = vec0_ensure_stmt(p, &p->stmtIvfCellMeta[col_idx],
      "SELECT cell_id FROM " VEC0_SHADOW_IVF_CELLS_NAME
      " WHERE centroid_id=? AND n_vectors<?", col_idx);
  if (rc != SQLITE_OK) return rc;
  sqlite3_stmt *stmt = p->stmtIvfCellMeta[col_idx];
  sqlite3_reset(stmt);
  sqlite3_bind_int64(stmt, 1, centroid_id);
  sqlite3_bind_int(stmt, 2, VEC0_IVF_CELL_MAX_VECTORS);
  while ((rc = sqlite3_step(stmt)) == SQLITE_ROW) {
    i64 id = sqlite3_column_int64(stmt, 0);
    struct IvfBufferedCell *c = ivf_buffer_find(p, col_idx, id);
    if (c && c->n == VEC0_IVF_CELL_MAX_VECTORS) continue;
    *out_cell_id = id;
    sqlite3_reset(stmt);
    return SQLITE_OK;
  }
  sqlite3_reset(stmt);
  if (rc != SQLITE_DONE) return rc;
  return ivf_cell_create(p, col_idx, centroid_id, out_cell_id);
}

/**
 * Insert into an unused slot; live count is not an offset after deletion.
 * Cell must have space (n_vectors < VEC0_IVF_CELL_MAX_VECTORS).
 */
static int ivf_cell_insert(vec0_vtab *p, int col_idx, i64 centroid_id,
                            i64 rowid, const void *vectorData, int vectorSize) {
  int rc;
  i64 cell_id;
  rc = ivf_cell_find_or_create(p, col_idx, centroid_id, &cell_id);
  if (rc != SQLITE_OK) return rc;

  struct IvfBufferedCell *c;
  rc = ivf_buffer_load(p, col_idx, cell_id, &c);
  if (rc != SQLITE_OK) return rc;
  int slot = 0;
  while (slot < VEC0_IVF_CELL_MAX_VECTORS && (c->validity[slot / 8] & (1 << (slot % 8)))) slot++;
  if (slot == VEC0_IVF_CELL_MAX_VECTORS || vectorSize != ivf_vec_size(p, col_idx)) return SQLITE_CORRUPT;
  c->validity[slot / 8] |= (1 << (slot % 8));
  c->rowids[slot] = rowid;
  memcpy(c->vectors + slot * vectorSize, vectorData, vectorSize);
  c->n++;

  // Insert rowid_map (cached)
  rc = vec0_ensure_stmt(p, &p->stmtIvfRowidMapInsert[col_idx],
      "INSERT INTO " VEC0_SHADOW_IVF_ROWID_MAP_NAME
      " (rowid, cell_id, slot) VALUES (?, ?, ?)", col_idx);
  if (rc != SQLITE_OK) return rc;
  sqlite3_stmt *s = p->stmtIvfRowidMapInsert[col_idx];
  sqlite3_reset(s);
  sqlite3_bind_int64(s, 1, rowid);
  sqlite3_bind_int64(s, 2, cell_id);
  sqlite3_bind_int(s, 3, slot);
  rc = sqlite3_step(s);
  if (rc != SQLITE_DONE) return rc;

  return SQLITE_OK;
}

// ============================================================================
// Shadow tables
// ============================================================================

static int ivf_create_shadow_tables(vec0_vtab *p, int col_idx) {
  sqlite3_stmt *stmt = NULL;
  int rc;
  char *zSql;

  zSql = sqlite3_mprintf(
      "CREATE TABLE " VEC0_SHADOW_IVF_CENTROIDS_NAME
      " (centroid_id INTEGER PRIMARY KEY, centroid BLOB NOT NULL)",
      p->schemaName, p->tableName, col_idx);
  if (!zSql) return SQLITE_NOMEM;
  rc = sqlite3_prepare_v2(p->db, zSql, -1, &stmt, NULL); sqlite3_free(zSql);
  if (rc != SQLITE_OK || sqlite3_step(stmt) != SQLITE_DONE) { sqlite3_finalize(stmt); return SQLITE_ERROR; }
  sqlite3_finalize(stmt);

  // cell_id is the rowid. A centroid's cells are found through the index of
  // the UNIQUE constraint, which ALTER TABLE RENAME renames with its table; an
  // index from CREATE INDEX would keep the old table's name, and xRename
  // cannot drop it ("database table is locked").
  zSql = sqlite3_mprintf(
      "CREATE TABLE " VEC0_SHADOW_IVF_CELLS_NAME
      " (cell_id INTEGER PRIMARY KEY,"
      "  centroid_id INTEGER NOT NULL,"
      "  n_vectors INTEGER NOT NULL DEFAULT 0,"
      "  validity BLOB NOT NULL,"
      "  rowids BLOB NOT NULL,"
      "  vectors BLOB NOT NULL,"
      "  UNIQUE (centroid_id, cell_id))",
      p->schemaName, p->tableName, col_idx);
  if (!zSql) return SQLITE_NOMEM;
  rc = sqlite3_prepare_v2(p->db, zSql, -1, &stmt, NULL); sqlite3_free(zSql);
  if (rc != SQLITE_OK || sqlite3_step(stmt) != SQLITE_DONE) { sqlite3_finalize(stmt); return SQLITE_ERROR; }
  sqlite3_finalize(stmt);

  zSql = sqlite3_mprintf(
      "CREATE TABLE " VEC0_SHADOW_IVF_ROWID_MAP_NAME
      " (rowid INTEGER PRIMARY KEY, cell_id INTEGER NOT NULL, slot INTEGER NOT NULL)",
      p->schemaName, p->tableName, col_idx);
  if (!zSql) return SQLITE_NOMEM;
  rc = sqlite3_prepare_v2(p->db, zSql, -1, &stmt, NULL); sqlite3_free(zSql);
  if (rc != SQLITE_OK || sqlite3_step(stmt) != SQLITE_DONE) { sqlite3_finalize(stmt); return SQLITE_ERROR; }
  sqlite3_finalize(stmt);

  // _ivf_vectors — full-precision KV store (only when quantizer != none)
  if (p->vector_columns[col_idx].ivf.quantizer != VEC0_IVF_QUANTIZER_NONE) {
    zSql = sqlite3_mprintf(
        "CREATE TABLE " VEC0_SHADOW_IVF_VECTORS_NAME
        " (rowid INTEGER PRIMARY KEY, vector BLOB NOT NULL)",
        p->schemaName, p->tableName, col_idx);
    if (!zSql) return SQLITE_NOMEM;
    rc = sqlite3_prepare_v2(p->db, zSql, -1, &stmt, NULL); sqlite3_free(zSql);
    if (rc != SQLITE_OK || sqlite3_step(stmt) != SQLITE_DONE) { sqlite3_finalize(stmt); return SQLITE_ERROR; }
    sqlite3_finalize(stmt);
  }

  zSql = sqlite3_mprintf(
      "INSERT INTO " VEC0_SHADOW_INFO_NAME " (key, value) VALUES ('ivf_trained_%d', '0')",
      p->schemaName, p->tableName, col_idx);
  if (!zSql) return SQLITE_NOMEM;
  rc = sqlite3_prepare_v2(p->db, zSql, -1, &stmt, NULL); sqlite3_free(zSql);
  if (rc != SQLITE_OK || sqlite3_step(stmt) != SQLITE_DONE) { sqlite3_finalize(stmt); return SQLITE_ERROR; }
  sqlite3_finalize(stmt);

  return SQLITE_OK;
}

static int ivf_drop_shadow_tables(vec0_vtab *p, int col_idx) {
  ivf_exec(p, "DROP TABLE IF EXISTS " VEC0_SHADOW_IVF_CENTROIDS_NAME, col_idx);
  ivf_exec(p, "DROP TABLE IF EXISTS " VEC0_SHADOW_IVF_CELLS_NAME, col_idx);
  ivf_exec(p, "DROP TABLE IF EXISTS " VEC0_SHADOW_IVF_ROWID_MAP_NAME, col_idx);
  ivf_exec(p, "DROP TABLE IF EXISTS " VEC0_SHADOW_IVF_VECTORS_NAME, col_idx);
  return SQLITE_OK;
}

// ============================================================================
// Insert / Delete
// ============================================================================

static int ivf_insert(vec0_vtab *p, int col_idx, i64 rowid,
                       const void *vectorData, int vectorSize) {
  UNUSED_PARAMETER(vectorSize);
  int quantizer = p->vector_columns[col_idx].ivf.quantizer;
  int qvecSize = ivf_vec_size(p, col_idx);
  int rc;

  // Quantize the input vector (or copy as-is if no quantization)
  void *qvec = sqlite3_malloc(qvecSize);
  if (!qvec) return SQLITE_NOMEM;
  ivf_quantize(p, col_idx, (const float *)vectorData, qvec);

  int trained;
  rc = ivf_is_trained(p, col_idx, &trained);
  if (rc != SQLITE_OK) { sqlite3_free(qvec); return rc; }
  if (!trained) {
    rc = ivf_cell_insert(p, col_idx, VEC0_IVF_UNASSIGNED_CENTROID_ID,
                          rowid, qvec, qvecSize);
  } else {
    // Find nearest centroid using quantized distance
    int best_centroid = -1;
    float min_dist = FLT_MAX;
    void *qrep = ivf_query_rep(p, col_idx, (const float *)vectorData);
    if (!qrep) { sqlite3_free(qvec); return SQLITE_NOMEM; }

    rc = vec0_ensure_stmt(p, &p->stmtIvfCentroidsAll[col_idx],
        "SELECT centroid_id, centroid FROM " VEC0_SHADOW_IVF_CENTROIDS_NAME, col_idx);
    if (rc != SQLITE_OK) { sqlite3_free(qrep); sqlite3_free(qvec); return rc; }
    sqlite3_stmt *stmt = p->stmtIvfCentroidsAll[col_idx];
    sqlite3_reset(stmt);
    while (sqlite3_step(stmt) == SQLITE_ROW) {
      int cid = sqlite3_column_int(stmt, 0);
      const void *c = sqlite3_column_blob(stmt, 1);
      int cBytes = sqlite3_column_bytes(stmt, 1);
      if (!c || cBytes != qvecSize) continue;
      float dist = ivf_distance(p, col_idx, qrep, c);
      if (dist < min_dist) { min_dist = dist; best_centroid = cid; }
    }
    sqlite3_free(qrep);
    if (best_centroid < 0) { sqlite3_free(qvec); return SQLITE_ERROR; }

    rc = ivf_cell_insert(p, col_idx, best_centroid, rowid, qvec, qvecSize);
  }

  sqlite3_free(qvec);
  if (rc != SQLITE_OK) return rc;

  // Store full-precision vector in KV table when quantized
  if (quantizer != VEC0_IVF_QUANTIZER_NONE) {
    sqlite3_stmt *stmt = NULL;
    char *zSql = sqlite3_mprintf(
        "INSERT INTO " VEC0_SHADOW_IVF_VECTORS_NAME " (rowid, vector) VALUES (?, ?)",
        p->schemaName, p->tableName, col_idx);
    if (!zSql) return SQLITE_NOMEM;
    rc = sqlite3_prepare_v2(p->db, zSql, -1, &stmt, NULL); sqlite3_free(zSql);
    if (rc != SQLITE_OK) return rc;
    sqlite3_bind_int64(stmt, 1, rowid);
    sqlite3_bind_blob(stmt, 2, vectorData, ivf_full_vec_size(p, col_idx), SQLITE_STATIC);
    rc = sqlite3_step(stmt);
    sqlite3_finalize(stmt);
    if (rc != SQLITE_DONE) return SQLITE_ERROR;
  }

  return SQLITE_OK;
}

static int ivf_delete(vec0_vtab *p, int col_idx, i64 rowid) {
  int rc;
  i64 cell_id = 0;
  int slot = -1;

  rc = vec0_ensure_stmt(p, &p->stmtIvfRowidMapLookup[col_idx],
      "SELECT cell_id, slot FROM " VEC0_SHADOW_IVF_ROWID_MAP_NAME
      " WHERE rowid = ?", col_idx);
  if (rc != SQLITE_OK) return rc;
  sqlite3_stmt *s = p->stmtIvfRowidMapLookup[col_idx];
  sqlite3_reset(s);
  sqlite3_bind_int64(s, 1, rowid);
  rc = sqlite3_step(s);
  if (rc == SQLITE_ROW) {
    cell_id = sqlite3_column_int64(s, 0);
    slot = sqlite3_column_int(s, 1);
  } else if (rc != SQLITE_DONE) {
    return rc;
  }
  sqlite3_reset(s);
  if (slot < 0) return SQLITE_OK;

  struct IvfBufferedCell *c;
  rc = ivf_buffer_load(p, col_idx, cell_id, &c);
  if (rc != SQLITE_OK) return rc;
  if (slot >= VEC0_IVF_CELL_MAX_VECTORS || !(c->validity[slot / 8] & (1 << (slot % 8))) ||
      c->rowids[slot] != rowid) return SQLITE_CORRUPT;
  c->validity[slot / 8] &= ~(1 << (slot % 8));
  c->n--;

  // Delete from rowid_map
  rc = vec0_ensure_stmt(p, &p->stmtIvfRowidMapDelete[col_idx],
      "DELETE FROM " VEC0_SHADOW_IVF_ROWID_MAP_NAME " WHERE rowid = ?", col_idx);
  if (rc != SQLITE_OK) return rc;
  sqlite3_stmt *sd = p->stmtIvfRowidMapDelete[col_idx];
  sqlite3_reset(sd);
  sqlite3_bind_int64(sd, 1, rowid);
  rc = sqlite3_step(sd);
  if (rc != SQLITE_DONE) return rc;

  // Delete from _ivf_vectors (full-precision KV) when quantized
  if (p->vector_columns[col_idx].ivf.quantizer != VEC0_IVF_QUANTIZER_NONE) {
    sqlite3_stmt *stmtDelVec = NULL;
    char *zSql = sqlite3_mprintf(
        "DELETE FROM " VEC0_SHADOW_IVF_VECTORS_NAME " WHERE rowid = ?",
        p->schemaName, p->tableName, col_idx);
    if (!zSql) return SQLITE_NOMEM;
    rc = sqlite3_prepare_v2(p->db, zSql, -1, &stmtDelVec, NULL);
    sqlite3_free(zSql);
    if (rc != SQLITE_OK) return rc;
    sqlite3_bind_int64(stmtDelVec, 1, rowid);
    rc = sqlite3_step(stmtDelVec);
    sqlite3_finalize(stmtDelVec);
    if (rc != SQLITE_DONE) return rc;
  }

  return SQLITE_OK;
}

// ============================================================================
// Point query
// ============================================================================

/**
 * Read a quantized column's full-precision vector from _ivf_vectors.
 */
static int ivf_full_vector_read(vec0_vtab *p, i64 rowid, int col_idx,
                                 void **outVector, int *outVectorSize) {
  int vecSize = ivf_full_vec_size(p, col_idx);
  int rc = vec0_ensure_stmt(p, &p->stmtIvfVectorsRead[col_idx],
      "SELECT vector FROM " VEC0_SHADOW_IVF_VECTORS_NAME " WHERE rowid = ?",
      col_idx);
  if (rc != SQLITE_OK) return rc;
  sqlite3_stmt *s = p->stmtIvfVectorsRead[col_idx];
  sqlite3_reset(s);
  sqlite3_bind_int64(s, 1, rowid);
  rc = sqlite3_step(s);
  if (rc == SQLITE_ROW) {
    void *buf = NULL;
    if (sqlite3_column_bytes(s, 0) != vecSize) {
      rc = SQLITE_CORRUPT;
    } else if (!(buf = sqlite3_malloc(vecSize))) {
      rc = SQLITE_NOMEM;
    } else {
      memcpy(buf, sqlite3_column_blob(s, 0), vecSize);
      *outVector = buf;
      if (outVectorSize) *outVectorSize = vecSize;
      rc = SQLITE_OK;
    }
  } else if (rc == SQLITE_DONE) {
    rc = SQLITE_EMPTY;
  }
  // A cached statement left on a row stays in progress until the next commit,
  // and a connection with a statement in progress refuses VACUUM.
  sqlite3_reset(s);
  return rc;
}

static int ivf_get_vector_data(vec0_vtab *p, i64 rowid, int col_idx,
                                void **outVector, int *outVectorSize) {
  int rc = ivf_flush(p);
  if (rc != SQLITE_OK) return rc;
  int vecSize = ivf_vec_size(p, col_idx);
  i64 cell_id = 0;
  int slot = -1;

  // A quantized column's cells hold its quantized vectors, which are not the
  // vectors it was given.
  if (p->vector_columns[col_idx].ivf.quantizer != VEC0_IVF_QUANTIZER_NONE) {
    return ivf_full_vector_read(p, rowid, col_idx, outVector, outVectorSize);
  }

  rc = vec0_ensure_stmt(p, &p->stmtIvfRowidMapLookup[col_idx],
      "SELECT cell_id, slot FROM " VEC0_SHADOW_IVF_ROWID_MAP_NAME
      " WHERE rowid = ?", col_idx);
  if (rc != SQLITE_OK) return rc;
  sqlite3_stmt *s = p->stmtIvfRowidMapLookup[col_idx];
  sqlite3_reset(s);
  sqlite3_bind_int64(s, 1, rowid);
  rc = sqlite3_step(s);
  if (rc == SQLITE_DONE) return SQLITE_EMPTY;
  if (rc != SQLITE_ROW) return rc;
  cell_id = sqlite3_column_int64(s, 0);
  slot = sqlite3_column_int(s, 1);
  // A cached statement left on a row stays in progress until the next commit,
  // and a connection with a statement in progress refuses VACUUM.
  sqlite3_reset(s);

  void *buf = sqlite3_malloc(vecSize);
  if (!buf) return SQLITE_NOMEM;

  sqlite3_blob *blob = NULL;
  rc = sqlite3_blob_open(p->db, p->schemaName, p->shadowIvfCellsNames[col_idx],
                          "vectors", cell_id, 0, &blob);
  if (rc != SQLITE_OK) { sqlite3_free(buf); return rc; }
  rc = sqlite3_blob_read(blob, buf, vecSize, slot * vecSize);
  sqlite3_blob_close(blob);
  if (rc != SQLITE_OK) { sqlite3_free(buf); return rc; }

  *outVector = buf;
  if (outVectorSize) *outVectorSize = vecSize;
  return SQLITE_OK;
}

// ============================================================================
// Centroid commands
// ============================================================================

static int ivf_load_all_vectors(vec0_vtab *p, int col_idx,
                                 float **out_vectors, i64 **out_rowids, int *out_N) {
  sqlite3_stmt *stmt = NULL;
  int rc;
  int D = (int)p->vector_columns[col_idx].dimensions;
  int vecSize = D * (int)sizeof(float);
  int quantizer = p->vector_columns[col_idx].ivf.quantizer;

  // When quantized, load full-precision vectors from _ivf_vectors KV table
  if (quantizer != VEC0_IVF_QUANTIZER_NONE) {
    int total = 0;
    char *zSql = sqlite3_mprintf(
        "SELECT count(*) FROM " VEC0_SHADOW_IVF_VECTORS_NAME,
        p->schemaName, p->tableName, col_idx);
    if (!zSql) return SQLITE_NOMEM;
    rc = sqlite3_prepare_v2(p->db, zSql, -1, &stmt, NULL); sqlite3_free(zSql);
    if (rc != SQLITE_OK) return rc;
    rc = sqlite3_step(stmt);
    if (rc == SQLITE_ROW) total = sqlite3_column_int(stmt, 0);
    sqlite3_finalize(stmt);
    if (rc != SQLITE_ROW) return rc;
    if (total == 0) { *out_vectors = NULL; *out_rowids = NULL; *out_N = 0; return SQLITE_OK; }

    float *vectors = sqlite3_malloc64((i64)total * D * sizeof(float));
    i64 *rowids = sqlite3_malloc64((i64)total * sizeof(i64));
    if (!vectors || !rowids) { sqlite3_free(vectors); sqlite3_free(rowids); return SQLITE_NOMEM; }

    int idx = 0;
    zSql = sqlite3_mprintf(
        "SELECT rowid, vector FROM " VEC0_SHADOW_IVF_VECTORS_NAME,
        p->schemaName, p->tableName, col_idx);
    if (!zSql) { sqlite3_free(vectors); sqlite3_free(rowids); return SQLITE_NOMEM; }
    rc = sqlite3_prepare_v2(p->db, zSql, -1, &stmt, NULL); sqlite3_free(zSql);
    if (rc != SQLITE_OK) { sqlite3_free(vectors); sqlite3_free(rowids); return rc; }
    while ((rc = sqlite3_step(stmt)) == SQLITE_ROW && idx < total) {
      rowids[idx] = sqlite3_column_int64(stmt, 0);
      const void *blob = sqlite3_column_blob(stmt, 1);
      int blobBytes = sqlite3_column_bytes(stmt, 1);
      if (blob && blobBytes == vecSize) {
        memcpy(&vectors[idx * D], blob, vecSize);
        idx++;
      }
    }
    sqlite3_finalize(stmt);
    if (rc != SQLITE_ROW && rc != SQLITE_DONE) {
      sqlite3_free(vectors); sqlite3_free(rowids);
      return rc;
    }
    *out_vectors = vectors; *out_rowids = rowids; *out_N = idx;
    return SQLITE_OK;
  }

  // Non-quantized: load from cells (existing path)

  // Count total
  int total = 0;
  char *zSql = sqlite3_mprintf(
      "SELECT COALESCE(SUM(n_vectors),0) FROM " VEC0_SHADOW_IVF_CELLS_NAME,
      p->schemaName, p->tableName, col_idx);
  if (!zSql) return SQLITE_NOMEM;
  rc = sqlite3_prepare_v2(p->db, zSql, -1, &stmt, NULL); sqlite3_free(zSql);
  if (rc != SQLITE_OK) return rc;
  rc = sqlite3_step(stmt);
  if (rc == SQLITE_ROW) total = sqlite3_column_int(stmt, 0);
  sqlite3_finalize(stmt);
  if (rc != SQLITE_ROW) return rc;

  if (total == 0) { *out_vectors = NULL; *out_rowids = NULL; *out_N = 0; return SQLITE_OK; }

  float *vectors = sqlite3_malloc64((i64)total * D * sizeof(float));
  i64 *rowids = sqlite3_malloc64((i64)total * sizeof(i64));
  if (!vectors || !rowids) { sqlite3_free(vectors); sqlite3_free(rowids); return SQLITE_NOMEM; }

  int idx = 0;
  zSql = sqlite3_mprintf(
      "SELECT n_vectors, validity, rowids, vectors FROM " VEC0_SHADOW_IVF_CELLS_NAME,
      p->schemaName, p->tableName, col_idx);
  if (!zSql) { sqlite3_free(vectors); sqlite3_free(rowids); return SQLITE_NOMEM; }
  rc = sqlite3_prepare_v2(p->db, zSql, -1, &stmt, NULL); sqlite3_free(zSql);
  if (rc != SQLITE_OK) { sqlite3_free(vectors); sqlite3_free(rowids); return rc; }

  while ((rc = sqlite3_step(stmt)) == SQLITE_ROW) {
    int n = sqlite3_column_int(stmt, 0);
    if (n == 0) continue;
    const unsigned char *val = (const unsigned char *)sqlite3_column_blob(stmt, 1);
    const i64 *rids = (const i64 *)sqlite3_column_blob(stmt, 2);
    const float *vecs = (const float *)sqlite3_column_blob(stmt, 3);
    int valBytes = sqlite3_column_bytes(stmt, 1);
    int ridsBytes = sqlite3_column_bytes(stmt, 2);
    int vecsBytes = sqlite3_column_bytes(stmt, 3);
    if (!val || !rids || !vecs) continue;
    int cap = valBytes * 8;
    // Clamp cap to the number of entries actually backed by the rowids and vectors blobs
    if (ridsBytes / (int)sizeof(i64) < cap) cap = ridsBytes / (int)sizeof(i64);
    if (vecsBytes / vecSize < cap) cap = vecsBytes / vecSize;
    for (int i = 0; i < cap && idx < total; i++) {
      if (val[i / 8] & (1 << (i % 8))) {
        rowids[idx] = rids[i];
        memcpy(&vectors[idx * D], &vecs[i * D], vecSize);
        idx++;
      }
    }
  }
  sqlite3_finalize(stmt);
  if (rc != SQLITE_DONE) {
    sqlite3_free(vectors); sqlite3_free(rowids);
    return rc;
  }
  *out_vectors = vectors; *out_rowids = rowids; *out_N = idx;
  return SQLITE_OK;
}

static void ivf_invalidate_cached(vec0_vtab *p, int col_idx) {
  sqlite3_finalize(p->stmtIvfCellMeta[col_idx]); p->stmtIvfCellMeta[col_idx] = NULL;
  sqlite3_finalize(p->stmtIvfCentroidsAll[col_idx]); p->stmtIvfCentroidsAll[col_idx] = NULL;
  sqlite3_finalize(p->stmtIvfCellWrite[col_idx]); p->stmtIvfCellWrite[col_idx] = NULL;
  sqlite3_finalize(p->stmtIvfRowidMapInsert[col_idx]); p->stmtIvfRowidMapInsert[col_idx] = NULL;
}

/**
 * Write one cell of a centroid's vectors with compute-centroids' INSERT into
 * _ivf_cells, then map each of its rowids to its slot with the INSERT into
 * _ivf_rowid_map.
 */
static int ivf_train_write_cell(vec0_vtab *p, sqlite3_stmt *stmtCell,
                                 sqlite3_stmt *stmtMap, int centroid_id,
                                 int n, const unsigned char *val,
                                 const i64 *rids, const unsigned char *vecs,
                                 int cap, int qvecSize) {
  sqlite3_reset(stmtCell);
  sqlite3_bind_int(stmtCell, 1, centroid_id);
  sqlite3_bind_int(stmtCell, 2, n);
  sqlite3_bind_blob(stmtCell, 3, val, cap / 8, SQLITE_TRANSIENT);
  sqlite3_bind_blob(stmtCell, 4, rids, cap * (int)sizeof(i64), SQLITE_TRANSIENT);
  sqlite3_bind_blob(stmtCell, 5, vecs, cap * qvecSize, SQLITE_TRANSIENT);
  int rc = sqlite3_step(stmtCell);
  if (rc != SQLITE_DONE) return rc;
  i64 cell_id = sqlite3_last_insert_rowid(p->db);

  for (int s = 0; s < n; s++) {
    sqlite3_reset(stmtMap);
    sqlite3_bind_int64(stmtMap, 1, rids[s]);
    sqlite3_bind_int64(stmtMap, 2, cell_id);
    sqlite3_bind_int(stmtMap, 3, s);
    rc = sqlite3_step(stmtMap);
    if (rc != SQLITE_DONE) return rc;
  }
  return SQLITE_OK;
}

static int ivf_cmd_compute_centroids(vec0_vtab *p, int col_idx, int nlist_override,
                                      int max_iter, uint32_t seed) {
  int rc;
  int D = (int)p->vector_columns[col_idx].dimensions;
  int vecSize = D * (int)sizeof(float);
  int quantizer = p->vector_columns[col_idx].ivf.quantizer;
  int nlist = nlist_override > 0 ? nlist_override : p->vector_columns[col_idx].ivf.nlist;
  if (nlist <= 0) { vtab_set_error(&p->base, "nlist must be specified"); return SQLITE_ERROR; }

  float *vectors = NULL; i64 *rowids = NULL; int N = 0;
  rc = ivf_load_all_vectors(p, col_idx, &vectors, &rowids, &N);
  if (rc != SQLITE_OK) return rc;
  if (N == 0) { vtab_set_error(&p->base, "No vectors"); sqlite3_free(vectors); sqlite3_free(rowids); return SQLITE_ERROR; }
  if (nlist > N) nlist = N;

  float *centroids = sqlite3_malloc64((i64)nlist * D * sizeof(float));
  if (!centroids) { sqlite3_free(vectors); sqlite3_free(rowids); return SQLITE_NOMEM; }
  if (ivf_kmeans(vectors, N, D, nlist, max_iter, seed, centroids) != 0) {
    sqlite3_free(vectors); sqlite3_free(rowids); sqlite3_free(centroids); return SQLITE_ERROR;
  }

  // Compute assignments
  int *assignments = sqlite3_malloc64((i64)N * sizeof(int));
  if (!assignments) { sqlite3_free(vectors); sqlite3_free(rowids); sqlite3_free(centroids); return SQLITE_NOMEM; }
  // Assignment uses float32 distances (k-means operates in float32 space)
  for (int i = 0; i < N; i++) {
    float min_d = FLT_MAX;
    int best = 0;
    for (int c = 0; c < nlist; c++) {
      float d = vec0_compute_distance(&p->vector_columns[col_idx], &vectors[i * D], &centroids[c * D]);
      if (d < min_d) { min_d = d; best = c; }
    }
    assignments[i] = best;
  }

  // Invalidate all cached stmts before dropping/recreating tables
  ivf_invalidate_cached(p, col_idx);

  // The INSERT running this command is a write statement in progress, so no
  // savepoint can open here ("cannot open savepoint - SQL statements in
  // progress"). An error returned to it rolls back every write below with the
  // transaction, in autocommit mode or when the error is an interrupt or a
  // memory, I/O or full-disk error; any other error inside BEGIN leaves them,
  // as it leaves the writes of any vec0 statement.
  sqlite3_stmt *stmt = NULL;
  char *zSql;

  // Clear all data
  rc = ivf_exec(p, "DELETE FROM " VEC0_SHADOW_IVF_CENTROIDS_NAME, col_idx);
  if (rc == SQLITE_OK) rc = ivf_exec(p, "DELETE FROM " VEC0_SHADOW_IVF_CELLS_NAME, col_idx);
  if (rc == SQLITE_OK) rc = ivf_exec(p, "DELETE FROM " VEC0_SHADOW_IVF_ROWID_MAP_NAME, col_idx);
  if (rc != SQLITE_OK) goto train_error;

  // Write centroids (quantized if quantizer is set)
  int qvecSize = ivf_vec_size(p, col_idx);
  void *qbuf = sqlite3_malloc(qvecSize > vecSize ? qvecSize : vecSize);
  if (!qbuf) { rc = SQLITE_NOMEM; goto train_error; }

  zSql = sqlite3_mprintf(
      "INSERT INTO " VEC0_SHADOW_IVF_CENTROIDS_NAME " (centroid_id, centroid) VALUES (?, ?)",
      p->schemaName, p->tableName, col_idx);
  if (!zSql) { sqlite3_free(qbuf); rc = SQLITE_NOMEM; goto train_error; }
  rc = sqlite3_prepare_v2(p->db, zSql, -1, &stmt, NULL); sqlite3_free(zSql);
  if (rc != SQLITE_OK) { sqlite3_free(qbuf); goto train_error; }
  for (int i = 0; i < nlist; i++) {
    ivf_quantize(p, col_idx, &centroids[i * D], qbuf);
    sqlite3_reset(stmt);
    sqlite3_bind_int(stmt, 1, i);
    sqlite3_bind_blob(stmt, 2, qbuf, qvecSize, SQLITE_TRANSIENT);
    rc = sqlite3_step(stmt);
    if (rc != SQLITE_DONE) { sqlite3_finalize(stmt); sqlite3_free(qbuf); goto train_error; }
  }
  sqlite3_finalize(stmt);

  // Build cells: group vectors by centroid, create fixed-size cells
  {
    // Prepare INSERT statements
    sqlite3_stmt *stmtCell = NULL;
    zSql = sqlite3_mprintf(
        "INSERT INTO " VEC0_SHADOW_IVF_CELLS_NAME
        " (centroid_id, n_vectors, validity, rowids, vectors) VALUES (?, ?, ?, ?, ?)",
        p->schemaName, p->tableName, col_idx);
    if (!zSql) { rc = SQLITE_NOMEM; goto train_error; }
    rc = sqlite3_prepare_v2(p->db, zSql, -1, &stmtCell, NULL); sqlite3_free(zSql);
    if (rc != SQLITE_OK) goto train_error;

    sqlite3_stmt *stmtMap = NULL;
    zSql = sqlite3_mprintf(
        "INSERT INTO " VEC0_SHADOW_IVF_ROWID_MAP_NAME " (rowid, cell_id, slot) VALUES (?, ?, ?)",
        p->schemaName, p->tableName, col_idx);
    if (!zSql) { sqlite3_finalize(stmtCell); rc = SQLITE_NOMEM; goto train_error; }
    rc = sqlite3_prepare_v2(p->db, zSql, -1, &stmtMap, NULL); sqlite3_free(zSql);
    if (rc != SQLITE_OK) { sqlite3_finalize(stmtCell); goto train_error; }

    int cap = VEC0_IVF_CELL_MAX_VECTORS;
    unsigned char *val = sqlite3_malloc(cap / 8);
    i64 *rids = sqlite3_malloc64((i64)cap * sizeof(i64));
    unsigned char *vecs = sqlite3_malloc64((i64)cap * qvecSize);  // quantized size
    if (!val || !rids || !vecs) {
      sqlite3_free(val); sqlite3_free(rids); sqlite3_free(vecs);
      sqlite3_finalize(stmtCell); sqlite3_finalize(stmtMap);
      sqlite3_free(qbuf);
      rc = SQLITE_NOMEM; goto train_error;
    }

    // Process one centroid at a time
    for (int c = 0; c < nlist && rc == SQLITE_OK; c++) {
      int slot = 0;
      memset(val, 0, cap / 8);
      memset(rids, 0, cap * sizeof(i64));

      for (int i = 0; i < N; i++) {
        if (assignments[i] != c) continue;

        if (slot >= cap) {
          // Flush current cell
          rc = ivf_train_write_cell(p, stmtCell, stmtMap, c, slot, val, rids,
                                    vecs, cap, qvecSize);
          if (rc != SQLITE_OK) break;
          slot = 0;
          memset(val, 0, cap / 8);
          memset(rids, 0, cap * sizeof(i64));
        }

        val[slot / 8] |= (1 << (slot % 8));
        rids[slot] = rowids[i];
        // Quantize float32 vector into cell blob
        ivf_quantize(p, col_idx, &vectors[i * D], &vecs[slot * qvecSize]);
        slot++;
      }

      // Flush remaining
      if (slot > 0 && rc == SQLITE_OK) {
        rc = ivf_train_write_cell(p, stmtCell, stmtMap, c, slot, val, rids,
                                  vecs, cap, qvecSize);
      }
    }

    sqlite3_free(val); sqlite3_free(rids); sqlite3_free(vecs);
    sqlite3_finalize(stmtCell); sqlite3_finalize(stmtMap);
    if (rc != SQLITE_OK) { sqlite3_free(qbuf); goto train_error; }
  }

  sqlite3_free(qbuf);

  // Store full-precision vectors in _ivf_vectors when quantized
  if (quantizer != VEC0_IVF_QUANTIZER_NONE) {
    rc = ivf_exec(p, "DELETE FROM " VEC0_SHADOW_IVF_VECTORS_NAME, col_idx);
    if (rc != SQLITE_OK) goto train_error;
    zSql = sqlite3_mprintf(
        "INSERT INTO " VEC0_SHADOW_IVF_VECTORS_NAME " (rowid, vector) VALUES (?, ?)",
        p->schemaName, p->tableName, col_idx);
    if (!zSql) { rc = SQLITE_NOMEM; goto train_error; }
    rc = sqlite3_prepare_v2(p->db, zSql, -1, &stmt, NULL); sqlite3_free(zSql);
    if (rc != SQLITE_OK) goto train_error;
    for (int i = 0; i < N; i++) {
      sqlite3_reset(stmt);
      sqlite3_bind_int64(stmt, 1, rowids[i]);
      sqlite3_bind_blob(stmt, 2, &vectors[i * D], vecSize, SQLITE_STATIC);
      rc = sqlite3_step(stmt);
      if (rc != SQLITE_DONE) break;
    }
    sqlite3_finalize(stmt);
    if (rc != SQLITE_DONE) goto train_error;
  }

  // Set trained = 1
  rc = ivf_exec(p,
      "INSERT OR REPLACE INTO " VEC0_SHADOW_INFO_NAME " (key, value) VALUES ('ivf_trained_%d', '1')",
      col_idx);
  if (rc != SQLITE_OK) goto train_error;

  sqlite3_free(vectors); sqlite3_free(rowids); sqlite3_free(centroids); sqlite3_free(assignments);
  return SQLITE_OK;

train_error:
  sqlite3_free(vectors); sqlite3_free(rowids); sqlite3_free(centroids); sqlite3_free(assignments);
  return rc;
}

static int ivf_cmd_set_centroid(vec0_vtab *p, int col_idx, int centroid_id,
                                 const void *vectorData, int vectorSize) {
  sqlite3_stmt *stmt = NULL;
  int rc;
  int D = (int)p->vector_columns[col_idx].dimensions;
  if (vectorSize != (int)(D * sizeof(float))) { vtab_set_error(&p->base, "Dimension mismatch"); return SQLITE_ERROR; }

  char *zSql = sqlite3_mprintf(
      "INSERT OR REPLACE INTO " VEC0_SHADOW_IVF_CENTROIDS_NAME " (centroid_id, centroid) VALUES (?, ?)",
      p->schemaName, p->tableName, col_idx);
  if (!zSql) return SQLITE_NOMEM;
  rc = sqlite3_prepare_v2(p->db, zSql, -1, &stmt, NULL); sqlite3_free(zSql);
  if (rc != SQLITE_OK) return rc;
  sqlite3_bind_int(stmt, 1, centroid_id);
  sqlite3_bind_blob(stmt, 2, vectorData, vectorSize, SQLITE_STATIC);
  rc = sqlite3_step(stmt); sqlite3_finalize(stmt);
  if (rc != SQLITE_DONE) return rc;

  rc = ivf_exec(p,
      "INSERT OR REPLACE INTO " VEC0_SHADOW_INFO_NAME " (key, value) VALUES ('ivf_trained_%d', '1')",
      col_idx);
  sqlite3_finalize(p->stmtIvfCentroidsAll[col_idx]); p->stmtIvfCentroidsAll[col_idx] = NULL;
  return rc;
}

static int ivf_cmd_assign_vectors(vec0_vtab *p, int col_idx) {
  int trained;
  int rc = ivf_is_trained(p, col_idx, &trained);
  if (rc != SQLITE_OK) return rc;
  if (!trained) { vtab_set_error(&p->base, "No centroids"); return SQLITE_ERROR; }

  int D = (int)p->vector_columns[col_idx].dimensions;
  int vecSize = D * (int)sizeof(float);
  sqlite3_stmt *stmt = NULL;
  char *zSql;

  // Load centroids
  int nlist = 0;
  float *centroids = NULL;
  zSql = sqlite3_mprintf("SELECT count(*) FROM " VEC0_SHADOW_IVF_CENTROIDS_NAME,
      p->schemaName, p->tableName, col_idx);
  if (!zSql) return SQLITE_NOMEM;
  rc = sqlite3_prepare_v2(p->db, zSql, -1, &stmt, NULL); sqlite3_free(zSql);
  if (rc != SQLITE_OK) return rc;
  rc = sqlite3_step(stmt);
  if (rc == SQLITE_ROW) nlist = sqlite3_column_int(stmt, 0);
  sqlite3_finalize(stmt);
  if (rc != SQLITE_ROW) return rc;
  if (nlist == 0) { vtab_set_error(&p->base, "No centroids"); return SQLITE_ERROR; }

  centroids = sqlite3_malloc64((i64)nlist * D * sizeof(float));
  if (!centroids) return SQLITE_NOMEM;
  zSql = sqlite3_mprintf("SELECT centroid_id, centroid FROM " VEC0_SHADOW_IVF_CENTROIDS_NAME " ORDER BY centroid_id",
      p->schemaName, p->tableName, col_idx);
  if (!zSql) { sqlite3_free(centroids); return SQLITE_NOMEM; }
  rc = sqlite3_prepare_v2(p->db, zSql, -1, &stmt, NULL); sqlite3_free(zSql);
  if (rc != SQLITE_OK) { sqlite3_free(centroids); return rc; }
  { int ci = 0; while ((rc = sqlite3_step(stmt)) == SQLITE_ROW && ci < nlist) {
      const void *b = sqlite3_column_blob(stmt, 1);
      int bBytes = sqlite3_column_bytes(stmt, 1);
      if (b && bBytes == vecSize) memcpy(&centroids[ci * D], b, vecSize);
      ci++;
  }}
  sqlite3_finalize(stmt);
  if (rc != SQLITE_ROW && rc != SQLITE_DONE) { sqlite3_free(centroids); return rc; }

  // Read unassigned cells, re-insert into trained cells
  zSql = sqlite3_mprintf(
      "SELECT rowid, n_vectors, validity, rowids, vectors FROM " VEC0_SHADOW_IVF_CELLS_NAME
      " WHERE centroid_id = %d",
      p->schemaName, p->tableName, col_idx, VEC0_IVF_UNASSIGNED_CENTROID_ID);
  if (!zSql) { sqlite3_free(centroids); return SQLITE_NOMEM; }
  rc = sqlite3_prepare_v2(p->db, zSql, -1, &stmt, NULL); sqlite3_free(zSql);
  if (rc != SQLITE_OK) { sqlite3_free(centroids); return rc; }

  // Invalidate cached stmts since we'll be modifying cells
  ivf_invalidate_cached(p, col_idx);

  while ((rc = sqlite3_step(stmt)) == SQLITE_ROW) {
    int n = sqlite3_column_int(stmt, 1);
    const unsigned char *val = (const unsigned char *)sqlite3_column_blob(stmt, 2);
    const i64 *rids = (const i64 *)sqlite3_column_blob(stmt, 3);
    const float *vecs = (const float *)sqlite3_column_blob(stmt, 4);
    int valBytes = sqlite3_column_bytes(stmt, 2);
    int ridsBytes = sqlite3_column_bytes(stmt, 3);
    int vecsBytes = sqlite3_column_bytes(stmt, 4);
    if (!val || !rids || !vecs) continue;
    int cap = valBytes * 8;
    if (ridsBytes / (int)sizeof(i64) < cap) cap = ridsBytes / (int)sizeof(i64);
    if (vecsBytes / vecSize < cap) cap = vecsBytes / vecSize;

    for (int i = 0; i < cap && n > 0 && rc == SQLITE_ROW; i++) {
      if (!(val[i / 8] & (1 << (i % 8)))) continue;
      n--;
      int cid = ivf_find_nearest_centroid(p, col_idx, &vecs[i * D], centroids, D, nlist);

      // Delete old rowid_map entry
      sqlite3_stmt *sd = NULL;
      char *zd = sqlite3_mprintf("DELETE FROM " VEC0_SHADOW_IVF_ROWID_MAP_NAME " WHERE rowid = ?",
          p->schemaName, p->tableName, col_idx);
      if (!zd) { rc = SQLITE_NOMEM; break; }
      int drc = sqlite3_prepare_v2(p->db, zd, -1, &sd, NULL); sqlite3_free(zd);
      if (drc == SQLITE_OK) {
        sqlite3_bind_int64(sd, 1, rids[i]);
        drc = sqlite3_step(sd);
        if (drc == SQLITE_DONE) drc = SQLITE_OK;
      }
      sqlite3_finalize(sd);

      if (drc == SQLITE_OK) drc = ivf_cell_insert(p, col_idx, cid, rids[i], &vecs[i * D], vecSize);
      if (drc != SQLITE_OK) rc = drc;
    }
    if (rc != SQLITE_ROW) break;
  }
  sqlite3_finalize(stmt);
  if (rc != SQLITE_DONE) { sqlite3_free(centroids); return rc; }

  // Delete unassigned cells
  zSql = sqlite3_mprintf(
      "DELETE FROM " VEC0_SHADOW_IVF_CELLS_NAME " WHERE centroid_id = %d",
      p->schemaName, p->tableName, col_idx, VEC0_IVF_UNASSIGNED_CENTROID_ID);
  if (!zSql) { sqlite3_free(centroids); return SQLITE_NOMEM; }
  rc = sqlite3_prepare_v2(p->db, zSql, -1, &stmt, NULL); sqlite3_free(zSql);
  if (rc == SQLITE_OK) {
    rc = sqlite3_step(stmt);
    if (rc == SQLITE_DONE) rc = SQLITE_OK;
  }
  sqlite3_finalize(stmt);

  sqlite3_free(centroids);
  return rc;
}

static int ivf_cmd_clear_centroids(vec0_vtab *p, int col_idx) {
  float *vectors = NULL; i64 *rowids = NULL; int N = 0;
  int vecSize = ivf_vec_size(p, col_idx);
  int D = (int)p->vector_columns[col_idx].dimensions;
  int rc;

  rc = ivf_load_all_vectors(p, col_idx, &vectors, &rowids, &N);
  if (rc != SQLITE_OK) return rc;

  ivf_invalidate_cached(p, col_idx);

  rc = ivf_exec(p, "DELETE FROM " VEC0_SHADOW_IVF_CENTROIDS_NAME, col_idx);
  if (rc == SQLITE_OK) rc = ivf_exec(p, "DELETE FROM " VEC0_SHADOW_IVF_CELLS_NAME, col_idx);
  if (rc == SQLITE_OK) rc = ivf_exec(p, "DELETE FROM " VEC0_SHADOW_IVF_ROWID_MAP_NAME, col_idx);

  // Re-insert all vectors into unassigned cells
  for (int i = 0; i < N && rc == SQLITE_OK; i++) {
    rc = ivf_cell_insert(p, col_idx, VEC0_IVF_UNASSIGNED_CENTROID_ID,
                         rowids[i], &vectors[i * D], vecSize);
  }

  if (rc == SQLITE_OK) {
    rc = ivf_exec(p,
        "INSERT OR REPLACE INTO " VEC0_SHADOW_INFO_NAME " (key, value) VALUES ('ivf_trained_%d', '0')",
        col_idx);
  }

  sqlite3_free(vectors); sqlite3_free(rowids);
  return rc;
}

// ============================================================================
// KNN Query — scan all cells for probed centroids
// ============================================================================

struct IvfCentroidDist { int id; float dist; };
struct IvfCandidate { i64 rowid; float distance; };

static int ivf_candidate_cmp(const void *a, const void *b) {
  float da = ((const struct IvfCandidate *)a)->distance;
  float db = ((const struct IvfCandidate *)b)->distance;
  if (da < db) return -1;
  if (da > db) return 1;
  return 0;
}

/**
 * Scan cell rows from a prepared statement, computing distances in-memory.
 * The statement must return (n_vectors, validity, rowids, vectors) columns.
 * queryVecQ is the quantized query (same type as cell vectors).
 * qvecSize is the size of one quantized vector in bytes.
 */
static int ivf_scan_cells_from_stmt(vec0_vtab *p, int col_idx,
                                     sqlite3_stmt *stmt,
                                     const void *queryVecQ, int qvecSize,
                                     const struct Array *allow,
                                     struct IvfCandidate **candidates,
                                     int *nCandidates, int *cap) {
  int rc;
  while ((rc = sqlite3_step(stmt)) == SQLITE_ROW) {
    int n = sqlite3_column_int(stmt, 0);
    if (n == 0) continue;
    const unsigned char *validity = (const unsigned char *)sqlite3_column_blob(stmt, 1);
    const i64 *rowids = (const i64 *)sqlite3_column_blob(stmt, 2);
    const unsigned char *vectors = (const unsigned char *)sqlite3_column_blob(stmt, 3);
    int valBytes = sqlite3_column_bytes(stmt, 1);
    int ridsBytes = sqlite3_column_bytes(stmt, 2);
    int vecsBytes = sqlite3_column_bytes(stmt, 3);
    if (!validity || !rowids || !vectors) continue;
    int cell_cap = valBytes * 8;
    if (ridsBytes / (int)sizeof(i64) < cell_cap) cell_cap = ridsBytes / (int)sizeof(i64);
    if (vecsBytes / qvecSize < cell_cap) cell_cap = vecsBytes / qvecSize;

    int found = 0;
    for (int i = 0; i < cell_cap && found < n; i++) {
      if (!(validity[i / 8] & (1 << (i % 8)))) continue;
      found++;
      if (!vec0_rowids_contains(allow, rowids[i])) continue;
      if (*nCandidates >= *cap) {
        *cap *= 2;
        struct IvfCandidate *tmp = sqlite3_realloc64(*candidates, (i64)*cap * sizeof(struct IvfCandidate));
        if (!tmp) return SQLITE_NOMEM;
        *candidates = tmp;
      }
      (*candidates)[*nCandidates].rowid = rowids[i];
      (*candidates)[*nCandidates].distance = ivf_distance(p, col_idx,
          queryVecQ, &vectors[i * qvecSize]);
      (*nCandidates)++;
    }
  }
  return rc == SQLITE_DONE ? SQLITE_OK : rc;
}

/**
 * Rank the rows a `rowid in (...)` constraint names by their exact distances
 * to the query. A rowid the table does not have is skipped.
 */
static int ivf_rank_rowids(vec0_vtab *p, int col_idx, const void *queryVector,
                            struct Array *rowidsIn,
                            struct IvfCandidate **candidates,
                            int *nCandidates, int *cap) {
  for (size_t i = 0; i < rowidsIn->length; i++) {
    i64 rowid = ((const i64 *)rowidsIn->z)[i];
    void *vec = NULL;
    int rc = ivf_get_vector_data(p, rowid, col_idx, &vec, NULL);
    if (rc == SQLITE_EMPTY) continue;
    if (rc != SQLITE_OK) return rc;
    if (*nCandidates >= *cap) {
      *cap *= 2;
      struct IvfCandidate *tmp = sqlite3_realloc64(*candidates, (i64)*cap * sizeof(struct IvfCandidate));
      if (!tmp) { sqlite3_free(vec); return SQLITE_NOMEM; }
      *candidates = tmp;
    }
    (*candidates)[*nCandidates].rowid = rowid;
    (*candidates)[*nCandidates].distance = vec0_compute_distance(&p->vector_columns[col_idx],
        (const float *)queryVector, (const float *)vec);
    (*nCandidates)++;
    sqlite3_free(vec);
  }
  return SQLITE_OK;
}

static int ivf_query_knn(vec0_vtab *p, int col_idx,
                          const void *queryVector, int queryVectorSize,
                          struct Array *arrayRowidsIn,
                          i64 k, struct vec0_query_knn_data *knn_data) {
  UNUSED_PARAMETER(queryVectorSize);
  int rc;
  int nprobe = p->vector_columns[col_idx].ivf.nprobe;
  int trained;
  rc = ivf_is_trained(p, col_idx, &trained);
  if (rc != SQLITE_OK) return rc;
  int quantizer = p->vector_columns[col_idx].ivf.quantizer;
  int oversample = p->vector_columns[col_idx].ivf.oversample;
  int qvecSize = ivf_vec_size(p, col_idx);

  // Quantize query vector for scanning
  void *queryQ = ivf_query_rep(p, col_idx, (const float *)queryVector);
  if (!queryQ) return SQLITE_NOMEM;

  // With oversample, collect more candidates for re-ranking
  i64 collect_k = (oversample > 1) ? k * oversample : k;

  int cap = (collect_k < 1024) ? 1024 : (int)collect_k * 2;
  int nCandidates = 0;
  struct IvfCandidate *candidates = sqlite3_malloc64((i64)cap * sizeof(struct IvfCandidate));
  if (!candidates) { sqlite3_free(queryQ); return SQLITE_NOMEM; }

  int exact = vec0_rowids_rank_exact(arrayRowidsIn);
  if (exact) {
    // A small rowid filter names its candidates, so each is ranked by its
    // exact distance; the probed cells hold only the rows near the query,
    // which the filter may not name. A larger filter is applied to the
    // probed cells.
    // ponytail: a selective filter above VEC0_KNN_EXACT_FILTER_MAX can return
    // fewer than k rows; widen nprobe by selectivity if that bites.
    rc = ivf_rank_rowids(p, col_idx, queryVector, arrayRowidsIn,
                         &candidates, &nCandidates, &cap);
    if (rc != SQLITE_OK) { sqlite3_free(queryQ); sqlite3_free(candidates); return rc; }
  } else if (trained) {
    // Find top nprobe centroids using quantized distance
    int nlist = 0;
    rc = vec0_ensure_stmt(p, &p->stmtIvfCentroidsAll[col_idx],
        "SELECT centroid_id, centroid FROM " VEC0_SHADOW_IVF_CENTROIDS_NAME, col_idx);
    if (rc != SQLITE_OK) { sqlite3_free(queryQ); sqlite3_free(candidates); return rc; }
    sqlite3_stmt *stmt = p->stmtIvfCentroidsAll[col_idx];
    sqlite3_reset(stmt);

    int centroid_cap = 64;
    struct IvfCentroidDist *cd = sqlite3_malloc64(centroid_cap * sizeof(*cd));
    if (!cd) { sqlite3_free(queryQ); sqlite3_free(candidates); return SQLITE_NOMEM; }

    while ((rc = sqlite3_step(stmt)) == SQLITE_ROW) {
      if (nlist >= centroid_cap) {
        centroid_cap *= 2;
        struct IvfCentroidDist *tmp = sqlite3_realloc64(cd, centroid_cap * sizeof(*cd));
        if (!tmp) { sqlite3_free(cd); sqlite3_free(queryQ); sqlite3_free(candidates); return SQLITE_NOMEM; }
        cd = tmp;
      }
      cd[nlist].id = sqlite3_column_int(stmt, 0);
      const void *c = sqlite3_column_blob(stmt, 1);
      int cBytes = sqlite3_column_bytes(stmt, 1);
      // Compare quantized query with quantized centroid
      cd[nlist].dist = (c && cBytes == qvecSize) ? ivf_distance(p, col_idx, queryQ, c) : FLT_MAX;
      nlist++;
    }
    if (rc != SQLITE_DONE) { sqlite3_free(cd); sqlite3_free(queryQ); sqlite3_free(candidates); return rc; }

    int actual_nprobe = nprobe < nlist ? nprobe : nlist;
    for (int i = 0; i < actual_nprobe; i++) {
      int min_j = i;
      for (int j = i + 1; j < nlist; j++) {
        if (cd[j].dist < cd[min_j].dist) min_j = j;
      }
      if (min_j != i) { struct IvfCentroidDist tmp = cd[i]; cd[i] = cd[min_j]; cd[min_j] = tmp; }
    }

    // Scan probed cells + unassigned with quantized distance
    {
      sqlite3_str *s = sqlite3_str_new(NULL);
      sqlite3_str_appendf(s,
          "SELECT n_vectors, validity, rowids, vectors FROM " VEC0_SHADOW_IVF_CELLS_NAME
          " WHERE centroid_id IN (",
          p->schemaName, p->tableName, col_idx);
      for (int i = 0; i < actual_nprobe; i++) {
        if (i > 0) sqlite3_str_appendall(s, ",");
        sqlite3_str_appendf(s, "%d", cd[i].id);
      }
      sqlite3_str_appendf(s, ",%d)", VEC0_IVF_UNASSIGNED_CENTROID_ID);
      char *zSql = sqlite3_str_finish(s);
      if (!zSql) { sqlite3_free(cd); sqlite3_free(queryQ); sqlite3_free(candidates); return SQLITE_NOMEM; }

      sqlite3_stmt *stmtScan = NULL;
      rc = sqlite3_prepare_v2(p->db, zSql, -1, &stmtScan, NULL);
      sqlite3_free(zSql);
      if (rc != SQLITE_OK) { sqlite3_free(cd); sqlite3_free(queryQ); sqlite3_free(candidates); return rc; }

      rc = ivf_scan_cells_from_stmt(p, col_idx, stmtScan, queryQ, qvecSize, arrayRowidsIn,
                                     &candidates, &nCandidates, &cap);
      sqlite3_finalize(stmtScan);
      if (rc != SQLITE_OK) { sqlite3_free(cd); sqlite3_free(queryQ); sqlite3_free(candidates); return rc; }
    }

    sqlite3_free(cd);
  } else {
    // Flat mode: scan only unassigned cells
    sqlite3_stmt *stmtScan = NULL;
    char *zSql = sqlite3_mprintf(
        "SELECT n_vectors, validity, rowids, vectors FROM " VEC0_SHADOW_IVF_CELLS_NAME
        " WHERE centroid_id = %d",
        p->schemaName, p->tableName, col_idx, VEC0_IVF_UNASSIGNED_CENTROID_ID);
    if (!zSql) { sqlite3_free(queryQ); sqlite3_free(candidates); return SQLITE_NOMEM; }
    rc = sqlite3_prepare_v2(p->db, zSql, -1, &stmtScan, NULL); sqlite3_free(zSql);
    if (rc != SQLITE_OK) { sqlite3_free(queryQ); sqlite3_free(candidates); return rc; }
    rc = ivf_scan_cells_from_stmt(p, col_idx, stmtScan, queryQ, qvecSize, arrayRowidsIn,
                                   &candidates, &nCandidates, &cap);
    sqlite3_finalize(stmtScan);
    if (rc != SQLITE_OK) { sqlite3_free(queryQ); sqlite3_free(candidates); return rc; }
  }

  sqlite3_free(queryQ);

  // Sort candidates by quantized distance
  qsort(candidates, nCandidates, sizeof(struct IvfCandidate), ivf_candidate_cmp);

  // Oversample re-ranking: re-score top (oversample*k) with full-precision vectors
  if (oversample > 1 && quantizer != VEC0_IVF_QUANTIZER_NONE && nCandidates > 0 &&
      !exact) {
    i64 rescore_n = collect_k < nCandidates ? collect_k : nCandidates;
    sqlite3_stmt *stmtVec = NULL;
    char *zSql = sqlite3_mprintf(
        "SELECT vector FROM " VEC0_SHADOW_IVF_VECTORS_NAME " WHERE rowid = ?",
        p->schemaName, p->tableName, col_idx);
    if (!zSql) { sqlite3_free(candidates); return SQLITE_NOMEM; }
    rc = sqlite3_prepare_v2(p->db, zSql, -1, &stmtVec, NULL); sqlite3_free(zSql);
    if (rc != SQLITE_OK) { sqlite3_free(candidates); return rc; }
    rc = SQLITE_DONE;
    for (i64 i = 0; i < rescore_n; i++) {
      sqlite3_reset(stmtVec);
      sqlite3_bind_int64(stmtVec, 1, candidates[i].rowid);
      rc = sqlite3_step(stmtVec);
      if (rc == SQLITE_ROW) {
        const float *fullVec = (const float *)sqlite3_column_blob(stmtVec, 0);
        int fullVecBytes = sqlite3_column_bytes(stmtVec, 0);
        if (fullVec && fullVecBytes == (int)p->vector_columns[col_idx].dimensions * (int)sizeof(float)) {
          candidates[i].distance = vec0_compute_distance(&p->vector_columns[col_idx],
              (const float *)queryVector, fullVec);
        }
      } else if (rc != SQLITE_DONE) {
        break;
      }
    }
    sqlite3_finalize(stmtVec);
    if (rc != SQLITE_ROW && rc != SQLITE_DONE) { sqlite3_free(candidates); return rc; }
    // Re-sort after re-scoring
    qsort(candidates, (size_t)rescore_n, sizeof(struct IvfCandidate), ivf_candidate_cmp);
    nCandidates = (int)rescore_n;
  }

  qsort(candidates, nCandidates, sizeof(struct IvfCandidate), ivf_candidate_cmp);
  i64 nResults = nCandidates < k ? nCandidates : k;

  if (nResults == 0) {
    knn_data->rowids = NULL; knn_data->distances = NULL;
    knn_data->k = k; knn_data->k_used = 0; knn_data->current_idx = 0;
    sqlite3_free(candidates); return SQLITE_OK;
  }

  knn_data->rowids = sqlite3_malloc64(nResults * sizeof(i64));
  knn_data->distances = sqlite3_malloc64(nResults * sizeof(f32));
  if (!knn_data->rowids || !knn_data->distances) {
    sqlite3_free(knn_data->rowids); sqlite3_free(knn_data->distances);
    sqlite3_free(candidates); return SQLITE_NOMEM;
  }
  for (i64 i = 0; i < nResults; i++) {
    knn_data->rowids[i] = candidates[i].rowid;
    knn_data->distances[i] = candidates[i].distance;
  }
  knn_data->k = k; knn_data->k_used = nResults; knn_data->current_idx = 0;
  sqlite3_free(candidates);
  return SQLITE_OK;
}

// ============================================================================
// Command dispatch
// ============================================================================

static int ivf_handle_command_inner(vec0_vtab *p, const char *command,
                               int argc, sqlite3_value **argv) {
  UNUSED_PARAMETER(argc);
  int col_idx = -1;
  for (int i = 0; i < p->numVectorColumns; i++) {
    if (p->vector_columns[i].index_type == VEC0_INDEX_TYPE_IVF) { col_idx = i; break; }
  }
  if (col_idx < 0) return SQLITE_EMPTY;

  // nprobe=N — change nprobe at runtime without rebuilding
  if (strncmp(command, "nprobe=", 7) == 0) {
    int new_nprobe = atoi(command + 7);
    if (new_nprobe < 1) {
      vtab_set_error(&p->base, "nprobe must be >= 1");
      return SQLITE_ERROR;
    }
    p->vector_columns[col_idx].ivf.nprobe = new_nprobe;
    return SQLITE_OK;
  }

  if (strcmp(command, "compute-centroids") == 0)
    return ivf_cmd_compute_centroids(p, col_idx, 0, VEC0_IVF_KMEANS_MAX_ITER, VEC0_IVF_KMEANS_DEFAULT_SEED);

  if (strncmp(command, "compute-centroids:", 18) == 0) {
    const char *json = command + 18;
    int nlist = 0, max_iter = VEC0_IVF_KMEANS_MAX_ITER;
    uint32_t seed = VEC0_IVF_KMEANS_DEFAULT_SEED;
    const char *pn = strstr(json, "\"nlist\":"); if (pn) nlist = atoi(pn + 8);
    const char *pi = strstr(json, "\"max_iterations\":"); if (pi) max_iter = atoi(pi + 17);
    const char *ps = strstr(json, "\"seed\":"); if (ps) seed = (uint32_t)atoi(ps + 7);
    return ivf_cmd_compute_centroids(p, col_idx, nlist, max_iter, seed);
  }

  if (strncmp(command, "set-centroid:", 13) == 0) {
    int centroid_id = atoi(command + 13);
    for (int i = 0; i < (int)(p->numVectorColumns + p->numPartitionColumns +
                               p->numAuxiliaryColumns + p->numMetadataColumns); i++) {
      if (p->user_column_kinds[i] == SQLITE_VEC0_USER_COLUMN_KIND_VECTOR &&
          p->user_column_idxs[i] == col_idx) {
        sqlite3_value *v = argv[2 + VEC0_COLUMN_USERN_START + i];
        if (sqlite3_value_type(v) == SQLITE_NULL) { vtab_set_error(&p->base, "set-centroid requires vector"); return SQLITE_ERROR; }
        return ivf_cmd_set_centroid(p, col_idx, centroid_id, sqlite3_value_blob(v), sqlite3_value_bytes(v));
      }
    }
    return SQLITE_ERROR;
  }

  if (strcmp(command, "assign-vectors") == 0) return ivf_cmd_assign_vectors(p, col_idx);
  if (strcmp(command, "clear-centroids") == 0) return ivf_cmd_clear_centroids(p, col_idx);
  return SQLITE_EMPTY;
}

static int ivf_handle_command(vec0_vtab *p, const char *command,
                               int argc, sqlite3_value **argv) {
  int rc = ivf_flush(p);
  if (rc != SQLITE_OK) return rc;
  rc = ivf_handle_command_inner(p, command, argc, argv);
  return rc == SQLITE_OK ? ivf_flush(p) : rc;
}
//////////////////////////////////////////////////////////////////////////////
// End inlined: sqlite-vec-ivf.c
//////////////////////////////////////////////////////////////////////////////

#endif
#if SQLITE_VEC_ENABLE_HNSW

//////////////////////////////////////////////////////////////////////////////
// Begin inlined: sqlite-vec-hnsw.c
//////////////////////////////////////////////////////////////////////////////

/**
 * sqlite-vec-hnsw.c — HNSW (Malkov & Yashunin, 2016) graph index for vec0.
 *
 * #include'd into sqlite-vec.c after struct definitions and before vec0_init().
 *
 *   v float[768] indexed by hnsw(m=16, ef_construction=200, ef_search=64, cache_mb=256)
 *
 * Unlike hnswlib-based extensions, the graph lives in SQLite shadow tables, so
 * it is transactional and crash-safe with no save/load step:
 *
 *   _hnsw_nodesNN (rowid PK, level, vector, UNIQUE(level, rowid))
 *   _hnsw_edgesNN (rowid, level, neighbors BLOB of i64) WITHOUT ROWID
 *
 * The entry point is not stored: it is the node with the highest level, found
 * through the UNIQUE(level, rowid) index, so deleting it needs no fix-up.
 *
 * Speed comes from a per-connection cache laid out like hnswlib's memory
 * index: nodes get dense 32-bit indexes, vectors and level-0 neighbor lists
 * sit in fixed-size chunks (pointers stay valid as the cache grows), and the
 * visited set is a tag array. Changed neighbor lists stay in the cache until
 * xSync, a savepoint boundary, or eviction. Rollback clears the cache after
 * SQLite restores shadow tables; other commits invalidate it by data version.
 *
 * Deletes repair the graph: every out-neighbor of the deleted node re-selects
 * its neighbors from its remaining list plus the deleted node's list. Edges
 * that pointed at the deleted node from elsewhere are skipped at search time.
 */
#define VEC0_HNSW_DEFAULT_M 16
#define VEC0_HNSW_DEFAULT_EF_CONSTRUCTION 200
#define VEC0_HNSW_DEFAULT_EF_SEARCH 64
#define VEC0_HNSW_DEFAULT_CACHE_MB 256
#define VEC0_HNSW_MAX_M 256
#define VEC0_HNSW_MAX_LEVEL 16

enum { VEC0_HNSW_QUANTIZER_NONE = 0, VEC0_HNSW_QUANTIZER_INT8 = 1 };

#define VEC0_SHADOW_HNSW_NODES_NAME "\"%w\".\"%w_hnsw_nodes%02d\""
#define VEC0_SHADOW_HNSW_EDGES_NAME "\"%w\".\"%w_hnsw_edges%02d\""

// ============================================================================
// Parser
// ============================================================================

static int vec0_parse_hnsw_options(struct Vec0Scanner *scanner,
                                   struct Vec0HnswConfig *config) {
  struct Vec0Token token;
  config->m = VEC0_HNSW_DEFAULT_M;
  config->ef_construction = VEC0_HNSW_DEFAULT_EF_CONSTRUCTION;
  config->ef_search = VEC0_HNSW_DEFAULT_EF_SEARCH;
  config->cache_mb = VEC0_HNSW_DEFAULT_CACHE_MB;
  config->quantizer = VEC0_HNSW_QUANTIZER_NONE;

  int rc = vec0_scanner_next(scanner, &token);
  if (rc != VEC0_TOKEN_RESULT_SOME || token.token_type != TOKEN_TYPE_LPAREN)
    return SQLITE_ERROR;
  rc = vec0_scanner_next(scanner, &token);
  if (rc == VEC0_TOKEN_RESULT_SOME && token.token_type == TOKEN_TYPE_RPAREN)
    return SQLITE_OK;

  while (1) {
    if (rc != VEC0_TOKEN_RESULT_SOME || token.token_type != TOKEN_TYPE_IDENTIFIER)
      return SQLITE_ERROR;
    const char *key = token.start;
    int keyLength = token.end - token.start;
    rc = vec0_scanner_next(scanner, &token);
    if (rc != VEC0_TOKEN_RESULT_SOME || token.token_type != TOKEN_TYPE_EQ)
      return SQLITE_ERROR;
    rc = vec0_scanner_next(scanner, &token);
    if (rc != VEC0_TOKEN_RESULT_SOME) return SQLITE_ERROR;
    const char *val = token.start;
    int valLength = token.end - token.start;
    int isDigit = token.token_type == TOKEN_TYPE_DIGIT;
    if (!isDigit && token.token_type != TOKEN_TYPE_IDENTIFIER) return SQLITE_ERROR;
    int v = isDigit ? atoi(token.start) : 0;

#define HNSW_KEY(name) \
  (keyLength == (int)sizeof(name) - 1 && sqlite3_strnicmp(key, name, keyLength) == 0)
#define HNSW_VAL(name) \
  (valLength == (int)sizeof(name) - 1 && sqlite3_strnicmp(val, name, valLength) == 0)
    if (HNSW_KEY("quantizer")) {
      if (isDigit) return SQLITE_ERROR;
      if (HNSW_VAL("none")) config->quantizer = VEC0_HNSW_QUANTIZER_NONE;
      else if (HNSW_VAL("int8")) config->quantizer = VEC0_HNSW_QUANTIZER_INT8;
      else return SQLITE_ERROR;
    } else if (!isDigit) {
      return SQLITE_ERROR;
    } else if (HNSW_KEY("m")) {
      if (v < 2 || v > VEC0_HNSW_MAX_M) return SQLITE_ERROR;
      config->m = v;
    } else if (HNSW_KEY("ef_construction")) {
      if (v < 1) return SQLITE_ERROR;
      config->ef_construction = v;
    } else if (HNSW_KEY("ef_search")) {
      if (v < 1) return SQLITE_ERROR;
      config->ef_search = v;
    } else if (HNSW_KEY("cache_mb")) {
      config->cache_mb = v;
    } else {
      return SQLITE_ERROR;
    }
#undef HNSW_KEY
#undef HNSW_VAL

    rc = vec0_scanner_next(scanner, &token);
    if (rc != VEC0_TOKEN_RESULT_SOME) return SQLITE_ERROR;
    if (token.token_type == TOKEN_TYPE_RPAREN) break;
    if (token.token_type != TOKEN_TYPE_COMMA) return SQLITE_ERROR;
    rc = vec0_scanner_next(scanner, &token);
  }
  if (config->ef_construction < config->m) config->ef_construction = config->m;
  return SQLITE_OK;
}

// ============================================================================
// Heaps of (distance, node index)
// ============================================================================

struct HnswItem {
  f32 d;
  i64 id;  // dense node index inside the graph code, rowid in results
};

struct HnswHeap {
  struct HnswItem *a;
  int n, cap;
  int max;  // 1 = max-heap (farthest on top), 0 = min-heap
};

static int hnsw_heap_before(const struct HnswHeap *h, int i, int j) {
  return h->max ? h->a[i].d > h->a[j].d : h->a[i].d < h->a[j].d;
}

static int hnsw_heap_push(struct HnswHeap *h, f32 d, i64 id) {
  if (h->n == h->cap) {
    int cap = h->cap ? h->cap * 2 : 64;
    struct HnswItem *a = sqlite3_realloc64(h->a, (i64)cap * sizeof(*a));
    if (!a) return SQLITE_NOMEM;
    h->a = a;
    h->cap = cap;
  }
  int i = h->n++;
  h->a[i].d = d;
  h->a[i].id = id;
  while (i > 0) {
    int parent = (i - 1) / 2;
    if (!hnsw_heap_before(h, i, parent)) break;
    struct HnswItem t = h->a[i]; h->a[i] = h->a[parent]; h->a[parent] = t;
    i = parent;
  }
  return SQLITE_OK;
}

static struct HnswItem hnsw_heap_pop(struct HnswHeap *h) {
  struct HnswItem top = h->a[0];
  h->a[0] = h->a[--h->n];
  int i = 0;
  for (;;) {
    int l = 2 * i + 1, r = l + 1, best = i;
    if (l < h->n && hnsw_heap_before(h, l, best)) best = l;
    if (r < h->n && hnsw_heap_before(h, r, best)) best = r;
    if (best == i) break;
    struct HnswItem t = h->a[i]; h->a[i] = h->a[best]; h->a[best] = t;
    i = best;
  }
  return top;
}

/** Sort a max-heap in place into ascending order (heapsort's second half). */
static void hnsw_heap_sort(struct HnswHeap *h) {
  int n = h->n;
  while (h->n > 1) {
    struct HnswItem top = h->a[0];
    struct HnswItem last = h->a[h->n - 1];
    h->n--;
    h->a[0] = last;
    int i = 0;
    for (;;) {
      int l = 2 * i + 1, r = l + 1, best = i;
      if (l < h->n && hnsw_heap_before(h, l, best)) best = l;
      if (r < h->n && hnsw_heap_before(h, r, best)) best = r;
      if (best == i) break;
      struct HnswItem t = h->a[i]; h->a[i] = h->a[best]; h->a[best] = t;
      i = best;
    }
    h->a[h->n] = top;
  }
  h->n = n;
}

/** Insertion sort, nearest first; for the short lists of neighbor selection. */
static void hnsw_sort_small(struct HnswItem *a, int n) {
  for (int i = 1; i < n; i++) {
    struct HnswItem t = a[i];
    int j = i - 1;
    while (j >= 0 && a[j].d > t.d) { a[j + 1] = a[j]; j--; }
    a[j + 1] = t;
  }
}

static int hnsw_item_cmp(const void *a, const void *b) {
  f32 x = ((const struct HnswItem *)a)->d, y = ((const struct HnswItem *)b)->d;
  return (x > y) - (x < y);
}

// ============================================================================
// Per-connection cache, in hnswlib's layout
// ============================================================================

#define HNSW_CHUNK_BITS 10
#define HNSW_CHUNK (1 << HNSW_CHUNK_BITS)

enum {
  HNSW_F_VEC = 1,     // vector loaded
  HNSW_F_ABSENT = 2,  // no such node (deleted, or a dangling edge)
  HNSW_F_L0 = 4,      // level-0 neighbor list loaded
  HNSW_F_CODE = 8,    // int8 search code computed
  HNSW_F_DIRTY = 16, // level-0 list needs a shadow-table write
};

struct HnswUpper {  // neighbor list above level 0
  int32_t idx;
  int level;
  int len;
  int32_t *ids;
  u8 used;
  u8 dirty;
};

struct HnswCache {
  int m0;           // level-0 list capacity (2*m)
  size_t vecBytes;
  size_t codeBytes; // 0 = no search codes; else 8 + dims (see hnsw_encode)
  // rowid -> idx (open addressing; hval holds idx+1, 0 = empty)
  i64 *hkey;
  int32_t *hval;
  int hcap;
  int n, cap;       // nodes known, capacity (a multiple of HNSW_CHUNK)
  i64 *rowid;       // idx -> rowid
  u8 *flags;
  u32 *visit;       // visited tags
  u32 tag;
  u8 **vchunk;      // vectors, HNSW_CHUNK per chunk
  u8 **qchunk;      // search codes, HNSW_CHUNK per chunk (codeBytes > 0)
  int32_t **echunk; // level-0 lists, HNSW_CHUNK slots of [count, ids...]
  int nchunks;
  struct HnswUpper *up;
  int upcap, upn;
  struct HnswHeap cand;  // search scratch, reused across operations
  // Entry point (highest level, then lowest rowid), mirrored from SQL.
  int entryKnown, entryFound, entryLevel;
  int32_t entryIdx;
  i64 bytes;
  int broken;       // a load failed mid-operation: clear at the next boundary
  u32 dataVersion;
  int haveVersion;
  int dirty, flushing;
};

static int hnsw_flush(vec0_vtab *p, int col);

static int hnsw_flush_all(vec0_vtab *p) {
  for (int col = 0; col < p->numVectorColumns; col++) {
    int rc = hnsw_flush(p, col);
    if (rc != SQLITE_OK) return rc;
  }
  return SQLITE_OK;
}

static u64 hnsw_mix(u64 x) {
  x += 0x9E3779B97F4A7C15ull;
  x = (x ^ (x >> 30)) * 0xBF58476D1CE4E5B9ull;
  x = (x ^ (x >> 27)) * 0x94D049BB133111EBull;
  return x ^ (x >> 31);
}

static void hnsw_cache_reset(struct HnswCache *c) {
  for (int i = 0; i < c->nchunks; i++) {
    sqlite3_free(c->vchunk[i]);
    sqlite3_free(c->echunk[i]);
    if (c->qchunk) sqlite3_free(c->qchunk[i]);
  }
  for (int i = 0; i < c->upcap; i++) {
    if (c->up[i].used) sqlite3_free(c->up[i].ids);
  }
  sqlite3_free(c->vchunk);
  sqlite3_free(c->echunk);
  sqlite3_free(c->qchunk);
  sqlite3_free(c->up);
  sqlite3_free(c->hkey);
  sqlite3_free(c->hval);
  sqlite3_free(c->rowid);
  sqlite3_free(c->flags);
  sqlite3_free(c->visit);
  sqlite3_free(c->cand.a);
  int m0 = c->m0;
  size_t vecBytes = c->vecBytes, codeBytes = c->codeBytes;
  memset(c, 0, sizeof(*c));
  c->m0 = m0;
  c->vecBytes = vecBytes;
  c->codeBytes = codeBytes;
}

/** Forget every cached HNSW node of this table. */
static void hnsw_cache_clear_all(vec0_vtab *p) {
  for (int i = 0; i < VEC0_MAX_VECTOR_COLUMNS; i++) {
    if (p->hnswCache[i]) hnsw_cache_reset(p->hnswCache[i]);
  }
}

static void hnsw_cache_rollback_to(vec0_vtab *p) {
  // A nested edge-write rollback must retain every column's pending lists.
  for (int i = 0; i < VEC0_MAX_VECTOR_COLUMNS; i++) {
    if (p->hnswCache[i] && p->hnswCache[i]->flushing) return;
  }
  hnsw_cache_clear_all(p);
}

static void hnsw_cache_free_all(vec0_vtab *p) {
  hnsw_cache_clear_all(p);
  for (int i = 0; i < VEC0_MAX_VECTOR_COLUMNS; i++) {
    sqlite3_free(p->hnswCache[i]);
    p->hnswCache[i] = NULL;
  }
}

static inline u8 *hnsw_vecp(const struct HnswCache *c, int32_t idx) {
  return c->vchunk[idx >> HNSW_CHUNK_BITS] + (size_t)(idx & (HNSW_CHUNK - 1)) * c->vecBytes;
}

static inline u8 *hnsw_codep(const struct HnswCache *c, int32_t idx) {
  return c->qchunk[idx >> HNSW_CHUNK_BITS] + (size_t)(idx & (HNSW_CHUNK - 1)) * c->codeBytes;
}

static inline int32_t *hnsw_slot(const struct HnswCache *c, int32_t idx) {
  return c->echunk[idx >> HNSW_CHUNK_BITS] + (size_t)(idx & (HNSW_CHUNK - 1)) * (c->m0 + 1);
}

static int hnsw_grow(struct HnswCache *c) {
  int cap = c->cap + HNSW_CHUNK;
  i64 *rowid = sqlite3_realloc64(c->rowid, (i64)cap * sizeof(i64));
  if (!rowid) return SQLITE_NOMEM;
  c->rowid = rowid;
  u8 *flags = sqlite3_realloc64(c->flags, cap);
  if (!flags) return SQLITE_NOMEM;
  c->flags = flags;
  u32 *visit = sqlite3_realloc64(c->visit, (i64)cap * sizeof(u32));
  if (!visit) return SQLITE_NOMEM;
  c->visit = visit;
  memset(c->visit + c->cap, 0, HNSW_CHUNK * sizeof(u32));
  u8 **vchunk = sqlite3_realloc64(c->vchunk, (i64)(c->nchunks + 1) * sizeof(u8 *));
  if (!vchunk) return SQLITE_NOMEM;
  c->vchunk = vchunk;
  int32_t **echunk = sqlite3_realloc64(c->echunk, (i64)(c->nchunks + 1) * sizeof(int32_t *));
  if (!echunk) return SQLITE_NOMEM;
  c->echunk = echunk;
  u8 *q = NULL;
  if (c->codeBytes) {
    u8 **qchunk = sqlite3_realloc64(c->qchunk, (i64)(c->nchunks + 1) * sizeof(u8 *));
    if (!qchunk) return SQLITE_NOMEM;
    c->qchunk = qchunk;
    q = sqlite3_malloc64((i64)HNSW_CHUNK * c->codeBytes);
    if (!q) return SQLITE_NOMEM;
  }
  u8 *v = sqlite3_malloc64((i64)HNSW_CHUNK * c->vecBytes);
  int32_t *e = sqlite3_malloc64((i64)HNSW_CHUNK * (c->m0 + 1) * sizeof(int32_t));
  if (!v || !e) {
    sqlite3_free(q);
    sqlite3_free(v);
    sqlite3_free(e);
    return SQLITE_NOMEM;
  }
  if (c->codeBytes) c->qchunk[c->nchunks] = q;
  c->vchunk[c->nchunks] = v;
  c->echunk[c->nchunks] = e;
  c->nchunks++;
  c->cap = cap;
  c->bytes += (i64)HNSW_CHUNK *
              ((i64)c->vecBytes + (i64)c->codeBytes + (c->m0 + 1) * 4 + 8 + 1 + 4 + 12);
  return SQLITE_OK;
}

/** Dense index for rowid, allocating an unloaded placeholder if new. */
static int hnsw_idx(struct HnswCache *c, i64 rowid, int32_t *out) {
  if ((c->n + 1) * 10 >= c->hcap * 7) {
    int hcap = c->hcap ? c->hcap * 2 : 1024;
    i64 *hkey = sqlite3_malloc64((i64)hcap * sizeof(i64));
    int32_t *hval = sqlite3_malloc64((i64)hcap * sizeof(int32_t));
    if (!hkey || !hval) {
      sqlite3_free(hkey);
      sqlite3_free(hval);
      return SQLITE_NOMEM;
    }
    memset(hval, 0, (size_t)hcap * sizeof(int32_t));
    u64 mask = (u64)hcap - 1;
    for (int32_t i = 0; i < c->n; i++) {
      u64 s = hnsw_mix((u64)c->rowid[i]) & mask;
      while (hval[s]) s = (s + 1) & mask;
      hkey[s] = c->rowid[i];
      hval[s] = i + 1;
    }
    sqlite3_free(c->hkey);
    sqlite3_free(c->hval);
    c->hkey = hkey;
    c->hval = hval;
    c->bytes += (i64)(hcap - c->hcap) * 12;
    c->hcap = hcap;
  }
  u64 mask = (u64)c->hcap - 1;
  u64 s = hnsw_mix((u64)rowid) & mask;
  while (c->hval[s]) {
    if (c->hkey[s] == rowid) {
      *out = c->hval[s] - 1;
      return SQLITE_OK;
    }
    s = (s + 1) & mask;
  }
  if (c->n == c->cap) {
    int rc = hnsw_grow(c);
    if (rc != SQLITE_OK) return rc;
  }
  int32_t idx = c->n++;
  c->rowid[idx] = rowid;
  c->flags[idx] = 0;
  c->hkey[s] = rowid;
  c->hval[s] = idx + 1;
  *out = idx;
  return SQLITE_OK;
}

static struct HnswUpper *hnsw_upper(struct HnswCache *c, int32_t idx, int level, int create) {
  if (create && (c->upn + 1) * 10 >= c->upcap * 7) {
    int cap = c->upcap ? c->upcap * 2 : 256;
    struct HnswUpper *up = sqlite3_malloc64((i64)cap * sizeof(*up));
    if (!up) return NULL;
    memset(up, 0, (size_t)cap * sizeof(*up));
    for (int i = 0; i < c->upcap; i++) {
      if (!c->up[i].used) continue;
      u64 s = hnsw_mix((u64)c->up[i].idx * 31 + (u64)c->up[i].level) & ((u64)cap - 1);
      while (up[s].used) s = (s + 1) & ((u64)cap - 1);
      up[s] = c->up[i];
    }
    sqlite3_free(c->up);
    c->up = up;
    c->upcap = cap;
  }
  if (!c->upcap) return NULL;
  u64 mask = (u64)c->upcap - 1;
  u64 s = hnsw_mix((u64)idx * 31 + (u64)level) & mask;
  while (c->up[s].used) {
    if (c->up[s].idx == idx && c->up[s].level == level) return &c->up[s];
    s = (s + 1) & mask;
  }
  if (!create) return NULL;
  memset(&c->up[s], 0, sizeof(c->up[s]));
  c->up[s].used = 1;
  c->up[s].idx = idx;
  c->up[s].level = level;
  c->upn++;
  return &c->up[s];
}

struct HnswCtx {
  vec0_vtab *p;
  int col;
  int m;
  size_t dims;
  enum VectorElementType type;
  enum Vec0DistanceMetrics metric;
  vec0_distance_fn dist;  // resolved once per operation
  int scol;               // storage-provider column of the vectors, or -1
  int defer;              // rebuild: node writes go to the cache only too
  struct HnswCache *c;
};

static int hnsw_data_version(vec0_vtab *p, u32 *v) {
  return sqlite3_file_control(p->db, p->schemaName, SQLITE_FCNTL_DATA_VERSION, v);
}

/** Our own commit moved the data version; the cache already mirrors it. */
static void hnsw_cache_after_commit(vec0_vtab *p) {
  for (int i = 0; i < VEC0_MAX_VECTOR_COLUMNS; i++) {
    struct HnswCache *c = p->hnswCache[i];
    u32 v;
    if (c && c->haveVersion && hnsw_data_version(p, &v) == SQLITE_OK) c->dataVersion = v;
  }
}

static void hnsw_cache_code(struct HnswCtx *x, int32_t idx);

static int hnsw_preload_row(void *pArg, sqlite3_int64 rowid, const void *data, int n) {
  struct HnswCtx *x = pArg;
  if (!data || (size_t)n != x->c->vecBytes) return SQLITE_OK;
  int32_t idx;
  int rc = hnsw_idx(x->c, rowid, &idx);
  if (rc != SQLITE_OK) return rc;
  memcpy(hnsw_vecp(x->c, idx), data, x->c->vecBytes);
  x->c->flags[idx] |= HNSW_F_VEC;
  hnsw_cache_code(x, idx);
  return SQLITE_OK;
}

/**
 * Vectors held by a storage provider: load them all with one sequential
 * scan when the index fits the cache budget, instead of one read per node.
 */
static int hnsw_preload(struct HnswCtx *x) {
  vec0_vtab *p = x->p;
  struct VectorColumnDefinition *def = &p->vector_columns[x->col];
  if (def->hnsw.cache_mb <= 0) return SQLITE_OK;
  sqlite3_stmt *s = NULL;
  char *zSql = sqlite3_mprintf("SELECT count(*) FROM " VEC0_SHADOW_HNSW_NODES_NAME,
                               p->schemaName, p->tableName, x->col);
  if (!zSql) return SQLITE_NOMEM;
  int rc = sqlite3_prepare_v2(p->db, zSql, -1, &s, NULL);
  sqlite3_free(zSql);
  if (rc != SQLITE_OK) return rc;
  i64 nodes = sqlite3_step(s) == SQLITE_ROW ? sqlite3_column_int64(s, 0) : 0;
  sqlite3_finalize(s);
  i64 perNode = (i64)x->c->vecBytes + (i64)x->c->codeBytes + (x->c->m0 + 1) * 4 + 48;
  if (nodes == 0 || nodes * perNode > (i64)def->hnsw.cache_mb * 1024 * 1024) return SQLITE_OK;
  rc = vec0_storage_error(p, p->storage->xScan(p->storageHandle, x->scol, hnsw_preload_row, x));
  if (rc != SQLITE_OK) x->c->broken = 1;
  return rc;
}

/**
 * Start an operation: drop the cache if another connection committed since
 * our last operation, if a previous load failed, or if it is over budget.
 * Never cleared mid-operation, so pointers into it stay valid until hnsw_end().
 */
static int hnsw_begin(struct HnswCtx *x, vec0_vtab *p, int col) {
  struct VectorColumnDefinition *def = &p->vector_columns[col];
  memset(x, 0, sizeof(*x));
  x->p = p;
  x->col = col;
  x->m = def->hnsw.m;
  x->dims = def->dimensions;
  x->type = def->element_type;
  x->metric = def->distance_metric;
  x->dist = vec0_distance_resolve(x->type, x->metric);
  x->scol = p->storage ? p->numAuxiliaryColumns + col : -1;
  if (!p->hnswCache[col]) {
    struct HnswCache *c = sqlite3_malloc(sizeof(*c));
    if (!c) return SQLITE_NOMEM;
    memset(c, 0, sizeof(*c));
    c->m0 = 2 * def->hnsw.m;
    c->vecBytes = vector_column_byte_size(*def);
    if (def->hnsw.quantizer == VEC0_HNSW_QUANTIZER_INT8 &&
        def->element_type == SQLITE_VEC_ELEMENT_TYPE_FLOAT32 &&
        def->distance_metric != VEC0_DISTANCE_METRIC_L1)
      c->codeBytes = 8 + def->dimensions;
    p->hnswCache[col] = c;
  }
  x->c = p->hnswCache[col];
  u32 v = 0;
  int known = hnsw_data_version(p, &v) == SQLITE_OK;
  if (!known || !x->c->haveVersion || v != x->c->dataVersion || x->c->broken ||
      x->c->bytes > (i64)def->hnsw.cache_mb * 1024 * 1024) {
    int rc = hnsw_flush(p, col);
    if (rc != SQLITE_OK) return rc;
    hnsw_cache_reset(x->c);
  }
  if (x->scol >= 0 && x->c->n == 0) return hnsw_preload(x);
  return SQLITE_OK;
}

/** Finish an operation; the cache stays valid until the data version moves. */
static void hnsw_end(struct HnswCtx *x) {
  u32 v;
  int keep = !x->c->broken && x->p->vector_columns[x->col].hnsw.cache_mb > 0 &&
             hnsw_data_version(x->p, &v) == SQLITE_OK;
  if (!keep) {
    if (!x->c->dirty) hnsw_cache_reset(x->c);
    return;
  }
  x->c->haveVersion = 1;
  x->c->dataVersion = v;
}

static int hnsw_m_max(struct HnswCtx *x, int level) {
  return level == 0 ? 2 * x->m : x->m;
}

// ============================================================================
// Storage (reads see the cache, neighbor writes coalesce until a boundary)
// ============================================================================

static void hnsw_encode(const f32 *v, size_t dims, u8 *out);

/** Encode a just-cached vector, so searches never touch the float copy. */
static void hnsw_cache_code(struct HnswCtx *x, int32_t idx) {
  struct HnswCache *c = x->c;
  if (!c->codeBytes) return;
  hnsw_encode((const f32 *)hnsw_vecp(c, idx), x->dims, hnsw_codep(c, idx));
  c->flags[idx] |= HNSW_F_CODE;
}

/** The node's vector, or NULL when the node does not exist. Stable pointer. */
static int hnsw_vector(struct HnswCtx *x, int32_t idx, const void **out) {
  struct HnswCache *c = x->c;
  u8 f = c->flags[idx];
  if (f & HNSW_F_ABSENT) { *out = NULL; return SQLITE_OK; }
  if (f & HNSW_F_VEC) { *out = hnsw_vecp(c, idx); return SQLITE_OK; }
  vec0_vtab *p = x->p;
  if (x->scol >= 0) {
    const void *data;
    int n;
    int rc = vec0_storage_error(p, p->storage->xRead(p->storageHandle, c->rowid[idx], x->scol, &data, &n));
    if (rc != SQLITE_OK) { c->broken = 1; return rc; }
    if (data && (size_t)n == c->vecBytes) {
      memcpy(hnsw_vecp(c, idx), data, c->vecBytes);
      c->flags[idx] |= HNSW_F_VEC;
      hnsw_cache_code(x, idx);
      *out = hnsw_vecp(c, idx);
    } else {
      c->flags[idx] |= HNSW_F_ABSENT;
      *out = NULL;
    }
    return SQLITE_OK;
  }
  int rc = vec0_ensure_stmt(p, &p->stmtHnswNodeRead[x->col],
      "SELECT vector FROM " VEC0_SHADOW_HNSW_NODES_NAME " WHERE rowid = ?", x->col);
  if (rc == SQLITE_OK) {
    sqlite3_stmt *s = p->stmtHnswNodeRead[x->col];
    sqlite3_reset(s);
    sqlite3_bind_int64(s, 1, c->rowid[idx]);
    rc = sqlite3_step(s);
    if (rc == SQLITE_ROW) {
      if ((size_t)sqlite3_column_bytes(s, 0) != c->vecBytes) {
        rc = SQLITE_CORRUPT;
      } else {
        memcpy(hnsw_vecp(c, idx), sqlite3_column_blob(s, 0), c->vecBytes);
        c->flags[idx] |= HNSW_F_VEC;
        hnsw_cache_code(x, idx);
        rc = SQLITE_OK;
      }
    } else if (rc == SQLITE_DONE) {
      c->flags[idx] |= HNSW_F_ABSENT;
      rc = SQLITE_OK;
    }
    sqlite3_reset(s);
  }
  if (rc != SQLITE_OK) {
    c->broken = 1;
    return rc;
  }
  *out = (c->flags[idx] & HNSW_F_ABSENT) ? NULL : hnsw_vecp(c, idx);
  return SQLITE_OK;
}

/** Distance from `q` to node idx; FLT_MAX when the node is gone. */
static int hnsw_distance(struct HnswCtx *x, const void *q, int32_t idx, f32 *d) {
  const void *v;
  int rc = hnsw_vector(x, idx, &v);
  if (rc != SQLITE_OK) return rc;
  *d = v ? x->dist(q, v, &x->dims) : FLT_MAX;
  return SQLITE_OK;
}

/**
 * int8 search code: [f32 scale][f32 squared norm][i8 * dims], with the scale
 * chosen per vector (max |x| / 127) so no global range is needed. The norm is
 * of the dequantized vector, so code-to-code distances are self-consistent.
 */
static void hnsw_encode(const f32 *v, size_t dims, u8 *out) {
  f32 maxabs = 0;
  for (size_t i = 0; i < dims; i++) {
    f32 a = fabsf(v[i]);
    if (a > maxabs) maxabs = a;
  }
  f32 scale = maxabs > 0 ? maxabs / 127.0f : 1.0f;
  i8 *q = (i8 *)(out + 8);
  i32 sq = 0;
  for (size_t i = 0; i < dims; i++) {
    i32 c = (i32)lrintf(v[i] / scale);
    if (c > 127) c = 127;
    if (c < -127) c = -127;
    q[i] = (i8)c;
    sq += c * c;
  }
  f32 norm2 = (f32)sq * scale * scale;
  memcpy(out, &scale, 4);
  memcpy(out + 4, &norm2, 4);
}

static i32 hnsw_dot_i8(const i8 *a, const i8 *b, size_t n) {
  size_t i = 0;
  i32 s = 0;
#if defined(SQLITE_VEC_ENABLE_NEON) && defined(__ARM_FEATURE_DOTPROD)
  int32x4_t acc0 = vdupq_n_s32(0), acc1 = vdupq_n_s32(0);
  for (; i + 32 <= n; i += 32) {
    acc0 = vdotq_s32(acc0, vld1q_s8(a + i), vld1q_s8(b + i));
    acc1 = vdotq_s32(acc1, vld1q_s8(a + i + 16), vld1q_s8(b + i + 16));
  }
  for (; i + 16 <= n; i += 16) acc0 = vdotq_s32(acc0, vld1q_s8(a + i), vld1q_s8(b + i));
  s = vaddvq_s32(vaddq_s32(acc0, acc1));
#endif
  for (; i < n; i++) s += (i32)a[i] * (i32)b[i];
  return s;
}

/** Approximate distance between two codes, in the column's metric. */
static f32 hnsw_code_distance(const u8 *a, const u8 *b, size_t dims,
                              enum Vec0DistanceMetrics metric) {
  f32 sa, na, sb, nb;
  memcpy(&sa, a, 4); memcpy(&na, a + 4, 4);
  memcpy(&sb, b, 4); memcpy(&nb, b + 4, 4);
  f32 dot = sa * sb * (f32)hnsw_dot_i8((const i8 *)(a + 8), (const i8 *)(b + 8), dims);
  switch (metric) {
  case VEC0_DISTANCE_METRIC_COSINE:
    return (na > 0 && nb > 0) ? 1.0f - dot / sqrtf(na * nb) : 1.0f;
  case VEC0_DISTANCE_METRIC_DOT:
    return -dot;
  default: {
    f32 d = na + nb - 2.0f * dot;
    return d > 0 ? sqrtf(d) : 0.0f;  // vec0 reports L2, not its square
  }
  }
}

/** The node's search code, or NULL when the node does not exist. */
static int hnsw_code(struct HnswCtx *x, int32_t idx, const u8 **out) {
  struct HnswCache *c = x->c;
  if (c->flags[idx] & HNSW_F_CODE) { *out = hnsw_codep(c, idx); return SQLITE_OK; }
  const void *v;
  int rc = hnsw_vector(x, idx, &v);
  if (rc != SQLITE_OK || !v) { *out = NULL; return rc; }
  hnsw_encode((const f32 *)v, x->dims, hnsw_codep(c, idx));
  c->flags[idx] |= HNSW_F_CODE;
  *out = hnsw_codep(c, idx);
  return SQLITE_OK;
}

/**
 * Distance used while searching: exact on vectors, or approximate on codes
 * when `qcode` is set (queries on an int8-quantized column).
 */
static int hnsw_search_distance(struct HnswCtx *x, const void *q, const u8 *qcode,
                                int32_t idx, f32 *d) {
  if (!qcode) return hnsw_distance(x, q, idx, d);
  const u8 *code;
  int rc = hnsw_code(x, idx, &code);
  if (rc != SQLITE_OK) return rc;
  *d = code ? hnsw_code_distance(qcode, code, x->dims, x->metric) : FLT_MAX;
  return SQLITE_OK;
}

/**
 * Neighbors of idx at level, as dense indexes. *ids is borrowed: valid until
 * the next write of that list or the end of the operation.
 */
static int hnsw_edges_read(struct HnswCtx *x, int32_t idx, int level,
                           const int32_t **ids, int *n) {
  struct HnswCache *c = x->c;
  if (level == 0 && (c->flags[idx] & HNSW_F_L0)) {
    int32_t *slot = hnsw_slot(c, idx);
    *ids = slot + 1;
    *n = slot[0];
    return SQLITE_OK;
  }
  if (level > 0) {
    struct HnswUpper *u = hnsw_upper(c, idx, level, 0);
    if (u) {
      *ids = u->ids;
      *n = u->len;
      return SQLITE_OK;
    }
  }

  vec0_vtab *p = x->p;
  int rc = vec0_ensure_stmt(p, &p->stmtHnswEdgesRead[x->col],
      "SELECT neighbors FROM " VEC0_SHADOW_HNSW_EDGES_NAME
      " WHERE rowid = ? AND level = ?", x->col);
  i64 *rowids = NULL;
  int count = 0;
  if (rc == SQLITE_OK) {
    sqlite3_stmt *s = p->stmtHnswEdgesRead[x->col];
    sqlite3_reset(s);
    sqlite3_bind_int64(s, 1, c->rowid[idx]);
    sqlite3_bind_int(s, 2, level);
    rc = sqlite3_step(s);
    if (rc == SQLITE_ROW) {
      int bytes = sqlite3_column_bytes(s, 0);
      count = bytes / (int)sizeof(i64);
      if (bytes % (int)sizeof(i64)) {
        rc = SQLITE_CORRUPT;
      } else if (count > 0 && !(rowids = sqlite3_malloc(bytes))) {
        rc = SQLITE_NOMEM;
      } else {
        if (count > 0) memcpy(rowids, sqlite3_column_blob(s, 0), bytes);
        rc = SQLITE_OK;
      }
    } else if (rc == SQLITE_DONE) {
      rc = SQLITE_OK;
    }
    sqlite3_reset(s);
  }
  int cap = hnsw_m_max(x, level);
  if (count > cap) count = cap;  // a longer list is corrupt; keep the head

  int32_t *dst = NULL;
  if (rc == SQLITE_OK) {
    if (level == 0) {
      dst = hnsw_slot(c, idx) + 1;
    } else {
      struct HnswUpper *u = hnsw_upper(c, idx, level, 1);
      dst = count > 0 ? sqlite3_malloc64((i64)count * sizeof(int32_t)) : NULL;
      if (!u || (count > 0 && !dst)) {
        sqlite3_free(dst);
        rc = SQLITE_NOMEM;
      } else {
        u->ids = dst;
        u->len = count;
        c->bytes += (i64)count * 4 + (i64)sizeof(*u);
      }
    }
  }
  for (int i = 0; rc == SQLITE_OK && i < count; i++) {
    int32_t j;
    rc = hnsw_idx(c, rowids[i], &j);
    if (rc == SQLITE_OK) dst[i] = j;
  }
  sqlite3_free(rowids);
  if (rc != SQLITE_OK) {
    c->broken = 1;
    return rc;
  }
  if (level == 0) {
    hnsw_slot(c, idx)[0] = count;
    c->flags[idx] |= HNSW_F_L0;
  }
  *ids = dst;
  *n = count;
  return SQLITE_OK;
}

static int hnsw_edges_store(vec0_vtab *p, int col, struct HnswCache *c,
                            int32_t idx, int level, const int32_t *ids, int n) {
  i64 rowids[2 * VEC0_HNSW_MAX_M];
  for (int i = 0; i < n; i++) rowids[i] = c->rowid[ids[i]];
  int rc = vec0_ensure_stmt(p, &p->stmtHnswEdgesWrite[col],
      "INSERT OR REPLACE INTO " VEC0_SHADOW_HNSW_EDGES_NAME
      " (rowid, level, neighbors) VALUES (?, ?, ?)", col);
  if (rc != SQLITE_OK) return rc;
  sqlite3_stmt *s = p->stmtHnswEdgesWrite[col];
  sqlite3_reset(s);
  sqlite3_bind_int64(s, 1, c->rowid[idx]);
  sqlite3_bind_int(s, 2, level);
  sqlite3_bind_blob(s, 3, rowids, n * (int)sizeof(i64), SQLITE_STATIC);
  rc = sqlite3_step(s);
  if (rc != SQLITE_DONE)
    vtab_set_error(&p->base, "Could not write HNSW neighbor list: %s", sqlite3_errmsg(p->db));
  sqlite3_reset(s);
  return rc == SQLITE_DONE ? SQLITE_OK : rc;
}

/** Flush final lists; retain dirty state on failure so commit cannot lose it. */
static int hnsw_flush(vec0_vtab *p, int col) {
  struct HnswCache *c = p->hnswCache[col];
  if (!c || !c->dirty || c->flushing) return SQLITE_OK;
  if (c->broken) return SQLITE_ABORT;
  c->flushing = 1;
  int rc = SQLITE_OK;
  for (int32_t idx = 0; rc == SQLITE_OK && idx < c->n; idx++) {
    if (c->flags[idx] & HNSW_F_DIRTY) {
      int32_t *slot = hnsw_slot(c, idx);
      rc = hnsw_edges_store(p, col, c, idx, 0, slot + 1, slot[0]);
    }
  }
  for (int i = 0; rc == SQLITE_OK && i < c->upcap; i++) {
    struct HnswUpper *u = &c->up[i];
    if (u->used && u->dirty)
      rc = hnsw_edges_store(p, col, c, u->idx, u->level, u->ids, u->len);
  }
  c->flushing = 0;
  if (rc == SQLITE_OK) {
    for (int32_t idx = 0; idx < c->n; idx++) c->flags[idx] &= ~HNSW_F_DIRTY;
    for (int i = 0; i < c->upcap; i++) c->up[i].dirty = 0;
    c->dirty = 0;
  }
  return rc;
}

/** Update cached lists immediately; coalesce their writes until a boundary. */
static int hnsw_edges_write(struct HnswCtx *x, int32_t idx, int level, const int32_t *ids, int n) {
  struct HnswCache *c = x->c;
  int deferred = x->defer || x->p->vector_columns[x->col].hnsw.cache_mb > 0;
  if (!deferred) {
    int rc = hnsw_edges_store(x->p, x->col, c, idx, level, ids, n);
    if (rc != SQLITE_OK) return rc;
  }
  if (level == 0) {
    int32_t *slot = hnsw_slot(c, idx);
    memmove(slot + 1, ids, (size_t)n * sizeof(int32_t));
    slot[0] = n;
    c->flags[idx] |= HNSW_F_L0;
    if (deferred) { c->flags[idx] |= HNSW_F_DIRTY; c->dirty = 1; }
    return SQLITE_OK;
  }
  struct HnswUpper *u = hnsw_upper(c, idx, level, 1);
  int32_t *copy = n > 0 ? sqlite3_malloc64((i64)n * sizeof(int32_t)) : NULL;
  if (!u || (n > 0 && !copy)) {
    sqlite3_free(copy);
    c->broken = 1;  // the cache no longer mirrors the table
    return SQLITE_NOMEM;
  }
  if (n > 0) memcpy(copy, ids, (size_t)n * sizeof(int32_t));
  c->bytes += (i64)(n - u->len) * 4;
  sqlite3_free(u->ids);
  u->ids = copy;
  u->len = n;
  if (deferred) { u->dirty = 1; c->dirty = 1; }
  return SQLITE_OK;
}

/** Entry point: the highest-level node. *found = 0 when the graph is empty. */
static int hnsw_entry(struct HnswCtx *x, int32_t *idx, int *level, int *found) {
  struct HnswCache *c = x->c;
  if (c->entryKnown) {
    *found = c->entryFound;
    *idx = c->entryIdx;
    *level = c->entryLevel;
    return SQLITE_OK;
  }
  vec0_vtab *p = x->p;
  int rc = vec0_ensure_stmt(p, &p->stmtHnswEntry[x->col],
      "SELECT rowid, level FROM " VEC0_SHADOW_HNSW_NODES_NAME
      " ORDER BY level DESC, rowid LIMIT 1", x->col);
  if (rc != SQLITE_OK) return rc;
  sqlite3_stmt *s = p->stmtHnswEntry[x->col];
  sqlite3_reset(s);
  rc = sqlite3_step(s);
  *found = rc == SQLITE_ROW;
  i64 rowid = 0;
  if (*found) {
    rowid = sqlite3_column_int64(s, 0);
    *level = sqlite3_column_int(s, 1);
  }
  sqlite3_reset(s);
  if (rc != SQLITE_ROW && rc != SQLITE_DONE) return rc;
  if (*found) {
    rc = hnsw_idx(c, rowid, idx);
    if (rc != SQLITE_OK) return rc;
  }
  c->entryKnown = 1;
  c->entryFound = *found;
  c->entryIdx = *found ? *idx : 0;
  c->entryLevel = *found ? *level : 0;
  return SQLITE_OK;
}

static int hnsw_exec(vec0_vtab *p, const char *fmt, int col) {
  char *zSql = sqlite3_mprintf(fmt, p->schemaName, p->tableName, col);
  if (!zSql) return SQLITE_NOMEM;
  int rc = sqlite3_exec(p->db, zSql, NULL, NULL, NULL);
  sqlite3_free(zSql);
  return rc;
}

static int hnsw_create_shadow_tables(vec0_vtab *p, int col) {
  // UNIQUE(level, rowid) gives the entry-point index a name that ALTER TABLE
  // RENAME carries along (a CREATE INDEX would keep the old table's name).
  int rc = hnsw_exec(p,
      "CREATE TABLE " VEC0_SHADOW_HNSW_NODES_NAME
      " (rowid INTEGER PRIMARY KEY, level INTEGER NOT NULL, vector BLOB NOT NULL,"
      "  UNIQUE (level, rowid))", col);
  if (rc != SQLITE_OK) return rc;
  return hnsw_exec(p,
      "CREATE TABLE " VEC0_SHADOW_HNSW_EDGES_NAME
      " (rowid INTEGER NOT NULL, level INTEGER NOT NULL, neighbors BLOB NOT NULL,"
      "  PRIMARY KEY (rowid, level)) WITHOUT ROWID", col);
}

static int hnsw_drop_shadow_tables(vec0_vtab *p, int col) {
  int rc = hnsw_exec(p, "DROP TABLE IF EXISTS " VEC0_SHADOW_HNSW_NODES_NAME, col);
  if (rc != SQLITE_OK) return rc;
  return hnsw_exec(p, "DROP TABLE IF EXISTS " VEC0_SHADOW_HNSW_EDGES_NAME, col);
}

// ============================================================================
// Graph algorithms (on dense indexes)
// ============================================================================

static int hnsw_allowed(struct HnswCtx *x, const struct Array *a, int32_t idx) {
  return vec0_rowids_contains(a, x->c->rowid[idx]);
}

static void hnsw_visit_reset(struct HnswCache *c) {
  if (++c->tag == 0) {
    memset(c->visit, 0, (size_t)c->cap * sizeof(u32));
    c->tag = 1;
  }
}

/**
 * Greedy beam search on one level (Algorithm 2). `eps` seeds the search.
 * Results (at most ef, nearest first) land in *out. With `allow`, only those
 * rowids may enter the results, though every node may be traversed.
 */
static int hnsw_search_level(struct HnswCtx *x, const void *q, const u8 *qcode,
                             const struct HnswItem *eps, int nEps, int ef,
                             int level, const struct Array *allow,
                             struct HnswItem **out, int *nOut) {
  struct HnswCache *c = x->c;
  struct HnswHeap *candp = &c->cand, res = {0};
  candp->n = 0;
  candp->max = 0;
#define cand (*candp)
  res.max = 1;
  int rc = SQLITE_OK;
  hnsw_visit_reset(c);

  for (int i = 0; i < nEps && rc == SQLITE_OK; i++) {
    int32_t e = (int32_t)eps[i].id;
    if (c->visit[e] == c->tag || eps[i].d == FLT_MAX) continue;
    c->visit[e] = c->tag;
    rc = hnsw_heap_push(&cand, eps[i].d, e);
    if (rc == SQLITE_OK && hnsw_allowed(x, allow, e)) rc = hnsw_heap_push(&res, eps[i].d, e);
  }

  while (rc == SQLITE_OK && cand.n > 0) {
    struct HnswItem cur = hnsw_heap_pop(&cand);
    if (res.n >= ef && cur.d > res.a[0].d) break;
    const int32_t *nbrs;
    int n;
    rc = hnsw_edges_read(x, (int32_t)cur.id, level, &nbrs, &n);
    // ponytail: no software prefetch; measured 2-3% slower on Apple M-series
    // (the hardware prefetcher covers this). Re-measure on x86.
    for (int i = 0; rc == SQLITE_OK && i < n; i++) {
      int32_t nb = nbrs[i];
      if (c->visit[nb] == c->tag) continue;
      c->visit[nb] = c->tag;
      f32 d;
      rc = hnsw_search_distance(x, q, qcode, nb, &d);
      if (rc != SQLITE_OK || d == FLT_MAX) continue;
      if (res.n < ef || d < res.a[0].d) {
        rc = hnsw_heap_push(&cand, d, nb);
        if (rc == SQLITE_OK && hnsw_allowed(x, allow, nb)) {
          rc = hnsw_heap_push(&res, d, nb);
          if (rc == SQLITE_OK && res.n > ef) hnsw_heap_pop(&res);
        }
      }
    }
  }

#undef cand
  if (rc != SQLITE_OK) {
    sqlite3_free(res.a);
    return rc;
  }
  hnsw_heap_sort(&res);
  *out = res.a;
  *nOut = res.n;
  return SQLITE_OK;
}

/**
 * Neighbor selection heuristic (Algorithm 4, as in hnswlib): walking the
 * candidates nearest-first, keep one only if it is closer to the base than to
 * every neighbor already kept. Spreads edges across directions.
 * `cands` must be sorted nearest-first. Writes up to `m` indexes to `out`.
 */
static int hnsw_select(struct HnswCtx *x, const struct HnswItem *cands, int n,
                       int m, int32_t *out, int *nOut) {
  int kept = 0, rc = SQLITE_OK;
  const void *keptVec[2 * VEC0_HNSW_MAX_M];
  for (int i = 0; i < n && kept < m; i++) {
    const void *v;
    rc = hnsw_vector(x, (int32_t)cands[i].id, &v);
    if (rc != SQLITE_OK) break;
    if (!v) continue;
    int good = 1;
    for (int j = 0; j < kept && good; j++) {
      if (x->dist(v, keptVec[j], &x->dims) < cands[i].d)
        good = 0;
    }
    if (good) {
      keptVec[kept] = v;
      out[kept++] = (int32_t)cands[i].id;
    }
  }
  *nOut = kept;
  return rc;
}

/**
 * Re-select `node`'s neighbors at `level` from `ids` (which may contain
 * duplicates, the node itself, or `exclude`), then write them. `ids` must not
 * be the cached list of `node` itself.
 */
static int hnsw_reselect(struct HnswCtx *x, int32_t node, int level, const int32_t *ids,
                         int n, int32_t exclude) {
  const void *base;
  int rc = hnsw_vector(x, node, &base);
  if (rc != SQLITE_OK || !base) return rc;
  struct HnswItem *items = sqlite3_malloc64((i64)(n > 0 ? n : 1) * sizeof(*items));
  if (!items) return SQLITE_NOMEM;
  int m = 0;
  for (int i = 0; i < n && rc == SQLITE_OK; i++) {
    if (ids[i] == node || ids[i] == exclude) continue;
    int dup = 0;
    for (int j = 0; j < m && !dup; j++) dup = items[j].id == ids[i];
    if (dup) continue;
    f32 d;
    rc = hnsw_distance(x, base, ids[i], &d);
    if (rc == SQLITE_OK && d != FLT_MAX) { items[m].d = d; items[m].id = ids[i]; m++; }
  }
  int32_t chosen[2 * VEC0_HNSW_MAX_M];
  int nChosen = 0;
  if (rc == SQLITE_OK) {
    hnsw_sort_small(items, m);
    rc = hnsw_select(x, items, m, hnsw_m_max(x, level), chosen, &nChosen);
  }
  if (rc == SQLITE_OK) rc = hnsw_edges_write(x, node, level, chosen, nChosen);
  sqlite3_free(items);
  return rc;
}

/** Level drawn from rowid, so a rebuild reproduces the same graph shape. */
static int hnsw_random_level(struct HnswCtx *x, i64 rowid) {
  double u = (double)(hnsw_mix((u64)rowid) >> 11) * (1.0 / 9007199254740992.0);
  if (u <= 0) u = 1e-300;
  int level = (int)(-log(u) / log((double)x->m));
  return level > VEC0_HNSW_MAX_LEVEL ? VEC0_HNSW_MAX_LEVEL : level;
}

/** Greedy descent (ef=1) from the entry point down to level `stop` + 1. */
static int hnsw_descend(struct HnswCtx *x, const void *q, const u8 *qcode, int32_t entry,
                        int entryLevel, int stop, struct HnswItem *ep) {
  ep->id = entry;
  int rc = hnsw_search_distance(x, q, qcode, entry, &ep->d);
  for (int level = entryLevel; rc == SQLITE_OK && level > stop; level--) {
    for (int changed = 1; changed && rc == SQLITE_OK;) {
      changed = 0;
      const int32_t *nbrs;
      int n;
      rc = hnsw_edges_read(x, (int32_t)ep->id, level, &nbrs, &n);
      for (int i = 0; rc == SQLITE_OK && i < n; i++) {
        f32 d;
        int32_t nb = nbrs[i];
        rc = hnsw_search_distance(x, q, qcode, nb, &d);
        if (rc == SQLITE_OK && d < ep->d) { ep->d = d; ep->id = nb; changed = 1; }
      }
    }
  }
  return rc;
}

// ============================================================================
// Insert / delete / query
// ============================================================================

/** Cache a node with empty neighbor lists, as written by an insert. */
static int hnsw_node_cache(struct HnswCtx *x, int32_t idx, int level, const void *vector) {
  struct HnswCache *c = x->c;
  memcpy(hnsw_vecp(c, idx), vector, c->vecBytes);
  hnsw_slot(c, idx)[0] = 0;
  c->flags[idx] = HNSW_F_VEC | HNSW_F_L0;
  hnsw_cache_code(x, idx);
  for (int l = 1; l <= level; l++) {
    struct HnswUpper *u = hnsw_upper(c, idx, l, 1);
    if (!u) { c->broken = 1; return SQLITE_NOMEM; }
    sqlite3_free(u->ids);
    u->ids = NULL;
    u->len = 0;
  }
  return SQLITE_OK;
}

static int hnsw_node_write(struct HnswCtx *x, int32_t idx, int level, const void *vector) {
  struct HnswCache *c = x->c;
  vec0_vtab *p = x->p;
  if (x->defer) return hnsw_node_cache(x, idx, level, vector);  // row already exists
  int rc = vec0_ensure_stmt(p, &p->stmtHnswNodeInsert[x->col],
      "INSERT INTO " VEC0_SHADOW_HNSW_NODES_NAME " (rowid, level, vector) VALUES (?, ?, ?)",
      x->col);
  if (rc != SQLITE_OK) return rc;
  sqlite3_stmt *s = p->stmtHnswNodeInsert[x->col];
  sqlite3_reset(s);
  sqlite3_bind_int64(s, 1, c->rowid[idx]);
  sqlite3_bind_int(s, 2, level);
  if (x->scol >= 0) sqlite3_bind_zeroblob(s, 3, 0);  // the provider holds the vector
  else sqlite3_bind_blob(s, 3, vector, (int)c->vecBytes, SQLITE_STATIC);
  rc = sqlite3_step(s);
  sqlite3_reset(s);
  if (rc != SQLITE_DONE) return rc;
  return hnsw_node_cache(x, idx, level, vector);
}

static int hnsw_insert_graph(struct HnswCtx *x, i64 rowid, const void *vector) {
  vec0_vtab *p = x->p;
  int rc, found, entryLevel = 0;
  int32_t entry = 0, idx;
  struct HnswItem *w = NULL;

  rc = hnsw_entry(x, &entry, &entryLevel, &found);
  if (rc == SQLITE_OK) rc = hnsw_idx(x->c, rowid, &idx);
  if (rc != SQLITE_OK) return rc;
  int level = hnsw_random_level(x, rowid);
  rc = hnsw_node_write(x, idx, level, vector);
  if (rc != SQLITE_OK) return rc;
  // Same order as the entry SQL: highest level, then lowest rowid.
  struct HnswCache *c = x->c;
  if (!found || level > entryLevel || (level == entryLevel && rowid < c->rowid[entry])) {
    c->entryKnown = 1;
    c->entryFound = 1;
    c->entryIdx = idx;
    c->entryLevel = level;
  }
  if (!found) return SQLITE_OK;  // first node: entry point, no edges

  struct HnswItem ep;
  rc = hnsw_descend(x, vector, NULL, entry, entryLevel, level, &ep);
  int efc = p->vector_columns[x->col].hnsw.ef_construction;
  const struct HnswItem *eps = &ep;
  int nEps = 1;
  int32_t chosen[VEC0_HNSW_MAX_M], grown[2 * VEC0_HNSW_MAX_M + 1];

  for (int l = level < entryLevel ? level : entryLevel; rc == SQLITE_OK && l >= 0; l--) {
    struct HnswItem *next;
    int nNext;
    rc = hnsw_search_level(x, vector, NULL, eps, nEps, efc, l, NULL, &next, &nNext);
    if (rc != SQLITE_OK) break;
    sqlite3_free(w);
    w = next;
    eps = w;
    nEps = nNext;

    int nChosen;
    rc = hnsw_select(x, w, nNext, x->m, chosen, &nChosen);
    if (rc == SQLITE_OK) rc = hnsw_edges_write(x, idx, l, chosen, nChosen);

    // Reverse edges, shrinking any list that overflows.
    for (int i = 0; rc == SQLITE_OK && i < nChosen; i++) {
      const int32_t *nbrs;
      int n;
      rc = hnsw_edges_read(x, chosen[i], l, &nbrs, &n);
      if (rc != SQLITE_OK) break;
      memcpy(grown, nbrs, (size_t)n * sizeof(int32_t));
      grown[n++] = idx;
      rc = n > hnsw_m_max(x, l) ? hnsw_reselect(x, chosen[i], l, grown, n, -1)
                                : hnsw_edges_write(x, chosen[i], l, grown, n);
    }
  }
  sqlite3_free(w);
  return rc;
}

static int hnsw_delete_graph(struct HnswCtx *x, i64 rowid) {
  vec0_vtab *p = x->p;
  struct HnswCache *c = x->c;
  int rc, level = -1;
  int32_t *mine = NULL, *merged = NULL, idx;

  {
    sqlite3_stmt *s = NULL;
    char *zSql = sqlite3_mprintf("SELECT level FROM " VEC0_SHADOW_HNSW_NODES_NAME
                                 " WHERE rowid = ?", p->schemaName, p->tableName, x->col);
    if (!zSql) return SQLITE_NOMEM;
    rc = sqlite3_prepare_v2(p->db, zSql, -1, &s, NULL);
    sqlite3_free(zSql);
    if (rc != SQLITE_OK) return rc;
    sqlite3_bind_int64(s, 1, rowid);
    if (sqlite3_step(s) == SQLITE_ROW) level = sqlite3_column_int(s, 0);
    rc = sqlite3_finalize(s);
    if (rc != SQLITE_OK) return rc;
  }
  if (level < 0) return SQLITE_OK;
  rc = hnsw_idx(c, rowid, &idx);

  for (int l = 0; rc == SQLITE_OK && l <= level; l++) {
    const int32_t *cached;
    int nMine;
    rc = hnsw_edges_read(x, idx, l, &cached, &nMine);
    if (rc != SQLITE_OK) break;
    // copy: repairing neighbors rewrites cached lists
    sqlite3_free(mine);
    mine = sqlite3_malloc64((i64)(nMine > 0 ? nMine : 1) * sizeof(int32_t));
    if (!mine) { rc = SQLITE_NOMEM; break; }
    memcpy(mine, cached, (size_t)nMine * sizeof(int32_t));
    for (int i = 0; rc == SQLITE_OK && i < nMine; i++) {
      const int32_t *theirs;
      int nTheirs;
      rc = hnsw_edges_read(x, mine[i], l, &theirs, &nTheirs);
      if (rc != SQLITE_OK) break;
      merged = sqlite3_malloc64((i64)(nTheirs + nMine + 1) * sizeof(int32_t));
      if (!merged) { rc = SQLITE_NOMEM; break; }
      memcpy(merged, theirs, (size_t)nTheirs * sizeof(int32_t));
      memcpy(merged + nTheirs, mine, (size_t)nMine * sizeof(int32_t));
      rc = hnsw_reselect(x, mine[i], l, merged, nTheirs + nMine, idx);
      sqlite3_free(merged);
      merged = NULL;
    }
  }
  if (rc == SQLITE_OK)
    rc = vec0_exec_rowid(p, "DELETE FROM " VEC0_SHADOW_HNSW_EDGES_NAME " WHERE rowid = ?", x->col, rowid);
  if (rc == SQLITE_OK)
    rc = vec0_exec_rowid(p, "DELETE FROM " VEC0_SHADOW_HNSW_NODES_NAME " WHERE rowid = ?", x->col, rowid);
  if (rc == SQLITE_OK) {
    // mirror the deletes: absent node, empty lists
    if (c->entryKnown && c->entryFound && c->entryIdx == idx) c->entryKnown = 0;
    c->flags[idx] = HNSW_F_ABSENT | HNSW_F_L0;
    hnsw_slot(c, idx)[0] = 0;
    for (int l = 1; l <= level; l++) {
      struct HnswUpper *u = hnsw_upper(c, idx, l, 0);
      if (u) { sqlite3_free(u->ids); u->ids = NULL; u->len = 0; u->dirty = 0; }
    }
  }
  sqlite3_free(mine);
  sqlite3_free(merged);
  return rc;
}

static int hnsw_insert(vec0_vtab *p, int col, i64 rowid, const void *vector) {
  struct HnswCtx x;
  int rc = hnsw_begin(&x, p, col);
  if (rc != SQLITE_OK) return rc;
  rc = hnsw_insert_graph(&x, rowid, vector);
  if (rc != SQLITE_OK) x.c->broken = 1;  // a partial write may be cached
  hnsw_end(&x);
  return rc;
}

static int hnsw_delete(vec0_vtab *p, int col, i64 rowid) {
  struct HnswCtx x;
  int rc = hnsw_begin(&x, p, col);
  if (rc != SQLITE_OK) return rc;
  rc = hnsw_delete_graph(&x, rowid);
  if (rc != SQLITE_OK) x.c->broken = 1;
  hnsw_end(&x);
  return rc;
}

/** UPDATE of the vector: unlink the node and link it again. */
static int hnsw_update(vec0_vtab *p, int col, i64 rowid, const void *vector) {
  struct HnswCtx x;
  int rc = hnsw_begin(&x, p, col);
  if (rc != SQLITE_OK) return rc;
  rc = hnsw_delete_graph(&x, rowid);
  if (rc == SQLITE_OK) rc = hnsw_insert_graph(&x, rowid, vector);
  if (rc != SQLITE_OK) x.c->broken = 1;
  hnsw_end(&x);
  return rc;
}

static int hnsw_get_vector_data(vec0_vtab *p, i64 rowid, int col,
                                void **outVector, int *outVectorSize) {
  struct HnswCtx x;
  int rc = hnsw_begin(&x, p, col);
  if (rc != SQLITE_OK) return rc;
  int32_t idx;
  const void *v = NULL;
  rc = hnsw_idx(x.c, rowid, &idx);
  if (rc == SQLITE_OK) rc = hnsw_vector(&x, idx, &v);
  if (rc == SQLITE_OK && !v) rc = SQLITE_EMPTY;
  if (rc == SQLITE_OK) {
    *outVector = sqlite3_malloc64(x.c->vecBytes);
    if (!*outVector) rc = SQLITE_NOMEM;
    else memcpy(*outVector, v, x.c->vecBytes);
    if (outVectorSize) *outVectorSize = (int)x.c->vecBytes;
  }
  hnsw_end(&x);
  return rc;
}

static int hnsw_query_knn(vec0_vtab *p, int col, const void *query,
                          const struct Array *rowidsIn, i64 k,
                          struct vec0_query_knn_data *knn) {
  struct HnswCtx x;
  memset(knn, 0, sizeof(*knn));
  knn->k = k;
  int rc = hnsw_begin(&x, p, col);
  if (rc != SQLITE_OK) return rc;
  struct HnswItem *res = NULL;
  int nRes = 0;

  if (vec0_rowids_rank_exact(rowidsIn)) {
    res = sqlite3_malloc64((i64)(rowidsIn->length ? rowidsIn->length : 1) * sizeof(*res));
    if (!res) rc = SQLITE_NOMEM;
    for (size_t i = 0; rc == SQLITE_OK && i < rowidsIn->length; i++) {
      int32_t idx;
      f32 d;
      rc = hnsw_idx(x.c, ((const i64 *)rowidsIn->z)[i], &idx);
      if (rc == SQLITE_OK) rc = hnsw_distance(&x, query, idx, &d);
      if (rc == SQLITE_OK && d != FLT_MAX) { res[nRes].d = d; res[nRes].id = idx; nRes++; }
    }
    if (rc == SQLITE_OK) qsort(res, nRes, sizeof(*res), hnsw_item_cmp);
  } else {
    int32_t entry;
    int entryLevel, found;
    rc = hnsw_entry(&x, &entry, &entryLevel, &found);
    if (rc == SQLITE_OK && found) {
      u8 qcodeBuf[8 + SQLITE_VEC_VEC0_MAX_DIMENSIONS];
      const u8 *qcode = NULL;
      if (x.c->codeBytes) {
        hnsw_encode((const f32 *)query, x.dims, qcodeBuf);
        qcode = qcodeBuf;
      }
      struct HnswItem ep;
      rc = hnsw_descend(&x, query, qcode, entry, entryLevel, 0, &ep);
      int ef = p->vector_columns[col].hnsw.ef_search;
      if (ef < k) ef = (int)k;
      if (rc == SQLITE_OK)
        rc = hnsw_search_level(&x, query, qcode, &ep, 1, ef, 0, rowidsIn, &res, &nRes);
      // Codes ranked the candidates; exact distances decide the answer.
      for (int i = 0; qcode && rc == SQLITE_OK && i < nRes; i++)
        rc = hnsw_distance(&x, query, (int32_t)res[i].id, &res[i].d);
      if (qcode && rc == SQLITE_OK) hnsw_sort_small(res, nRes);
    }
  }

  if (rc == SQLITE_OK && nRes > 0) {
    i64 n = nRes < k ? nRes : k;
    knn->rowids = sqlite3_malloc64(n * sizeof(i64));
    knn->distances = sqlite3_malloc64(n * sizeof(f32));
    if (!knn->rowids || !knn->distances) {
      sqlite3_free(knn->rowids);
      sqlite3_free(knn->distances);
      knn->rowids = NULL;
      knn->distances = NULL;
      rc = SQLITE_NOMEM;
    } else {
      for (i64 i = 0; i < n; i++) {
        knn->rowids[i] = x.c->rowid[res[i].id];
        knn->distances[i] = res[i].d;
      }
      knn->k_used = n;
    }
  }
  sqlite3_free(res);
  hnsw_end(&x);
  return rc;
}

/**
 * Relink every node from scratch, in rowid order, with all neighbor-list
 * writes held in the cache, then write every list once. One statement, so a
 * failure rolls the whole rebuild back. Needs the whole graph in memory for
 * the duration, whatever cache_mb says.
 */
static int hnsw_rebuild(vec0_vtab *p, int col) {
  struct HnswCtx x;
  int rc = hnsw_begin(&x, p, col);
  if (rc != SQLITE_OK) return rc;
  rc = hnsw_flush(p, col);
  if (rc != SQLITE_OK) return rc;
  hnsw_cache_reset(x.c);
  x.c->entryKnown = 1;  // nothing is linked yet, whatever the nodes table says
  x.c->entryFound = 0;
  x.defer = 1;
  int *levels = NULL;
  int nLevels = 0;
  sqlite3_stmt *s = NULL;
  rc = hnsw_exec(p, "DELETE FROM " VEC0_SHADOW_HNSW_EDGES_NAME, col);
  if (rc == SQLITE_OK) {
    char *zSql = sqlite3_mprintf("SELECT rowid, level, vector FROM " VEC0_SHADOW_HNSW_NODES_NAME
                                 " ORDER BY rowid", p->schemaName, p->tableName, col);
    rc = zSql ? sqlite3_prepare_v2(p->db, zSql, -1, &s, NULL) : SQLITE_NOMEM;
    sqlite3_free(zSql);
  }
  while (rc == SQLITE_OK) {
    int step = sqlite3_step(s);
    if (step == SQLITE_DONE) break;
    if (step != SQLITE_ROW) { rc = step; break; }
    i64 rowid = sqlite3_column_int64(s, 0);
    const void *vec = sqlite3_column_blob(s, 2);
    int vecBytes = sqlite3_column_bytes(s, 2);
    if (x.scol >= 0) {  // stored by the provider
      rc = vec0_storage_error(p, p->storage->xRead(p->storageHandle, rowid, x.scol, &vec, &vecBytes));
      if (rc != SQLITE_OK) break;
    }
    if (!vec || (size_t)vecBytes != x.c->vecBytes) { rc = SQLITE_CORRUPT; break; }
    // xRead's buffer lasts only until the next provider call: keep a copy.
    u8 vecCopy[SQLITE_VEC_VEC0_MAX_DIMENSIONS * sizeof(f32)];
    memcpy(vecCopy, vec, x.c->vecBytes);
    vec = vecCopy;
    if (sqlite3_column_int(s, 1) != hnsw_random_level(&x, rowid)) { rc = SQLITE_CORRUPT; break; }
    rc = hnsw_insert_graph(&x, rowid, vec);
  }
  sqlite3_finalize(s);

  // Write every list once. Node levels come back from the cache shape.
  if (rc == SQLITE_OK) {
    nLevels = x.c->n;
    levels = sqlite3_malloc64((i64)(nLevels ? nLevels : 1) * sizeof(int));
    if (!levels) rc = SQLITE_NOMEM;
  }
  for (int32_t i = 0; rc == SQLITE_OK && i < nLevels; i++) {
    levels[i] = (x.c->flags[i] & HNSW_F_ABSENT) ? -1 : hnsw_random_level(&x, x.c->rowid[i]);
  }
  for (int32_t i = 0; rc == SQLITE_OK && i < nLevels; i++) {
    for (int l = 0; rc == SQLITE_OK && l <= levels[i]; l++) {
      const int32_t *ids;
      int n;
      int32_t copy[2 * VEC0_HNSW_MAX_M];
      rc = hnsw_edges_read(&x, i, l, &ids, &n);
      if (rc == SQLITE_OK) {
        memcpy(copy, ids, (size_t)n * sizeof(int32_t));
        rc = hnsw_edges_write(&x, i, l, copy, n);
      }
    }
  }
  sqlite3_free(levels);
  x.defer = 0;
  if (rc == SQLITE_OK) rc = hnsw_flush(p, col);
  if (rc != SQLITE_OK) x.c->broken = 1;
  hnsw_end(&x);
  return rc;
}

/**
 * Runtime tuning, e.g. insert into t(t) values ('ef_search=128').
 * Applies to every HNSW column of the table, for this connection.
 */
static int hnsw_handle_command(vec0_vtab *p, const char *command) {
  if (strcmp(command, "hnsw-rebuild") == 0) {
    int handled = 0;
    for (int i = 0; i < p->numVectorColumns; i++) {
      if (p->vector_columns[i].index_type != VEC0_INDEX_TYPE_HNSW) continue;
      int rc = hnsw_rebuild(p, i);
      if (rc != SQLITE_OK) return rc;
      handled = 1;
    }
    return handled ? SQLITE_OK : SQLITE_EMPTY;
  }
  const char *eq = strchr(command, '=');
  if (!eq) return SQLITE_EMPTY;
  size_t keyLen = (size_t)(eq - command);
  int is_ef = keyLen == 9 && strncmp(command, "ef_search", 9) == 0;
  int is_cache = keyLen == 8 && strncmp(command, "cache_mb", 8) == 0;
  if (!is_ef && !is_cache) return SQLITE_EMPTY;
  int min = is_ef ? 1 : 0;
  char *end;
  long v = strtol(eq + 1, &end, 10);
  if (end == eq + 1 || *end || v < min || v > INT_MAX) {
    vtab_set_error(&p->base, "%.*s must be an integer >= %d", (int)keyLen, command, min);
    return SQLITE_ERROR;
  }
  int handled = 0;
  for (int i = 0; i < p->numVectorColumns; i++) {
    if (p->vector_columns[i].index_type != VEC0_INDEX_TYPE_HNSW) continue;
    if (is_ef) p->vector_columns[i].hnsw.ef_search = (int)v;
    else p->vector_columns[i].hnsw.cache_mb = (int)v;
    handled = 1;
  }
  if (is_cache) {
    int rc = hnsw_flush_all(p);
    if (rc != SQLITE_OK) return rc;
    hnsw_cache_clear_all(p);
  }
  return handled ? SQLITE_OK : SQLITE_EMPTY;
}
//////////////////////////////////////////////////////////////////////////////
// End inlined: sqlite-vec-hnsw.c
//////////////////////////////////////////////////////////////////////////////

#endif

//////////////////////////////////////////////////////////////////////////////
// Begin inlined: sqlite-vec-graph.c
//////////////////////////////////////////////////////////////////////////////

/**
 * sqlite-vec-graph.c — breadth-first traversal over ordinary SQLite edge tables.
 *
 * #include'd into sqlite-vec.c; graph_register() runs from sqlite3_vec_init().
 *
 *   select node, depth from graph_bfs('edges', 'src', 'dst', :start, :max_depth);
 *   select step, node from graph_shortest_path('edges', 'src', 'dst', :a, :b);
 *
 * Edges are directed src -> dst. Any table or view works, so an undirected
 * graph is a `UNION ALL` view over both directions.
 *
 * Each connection caches a compressed sparse row (CSR) copy of up to
 * GRAPH_CACHE_SLOTS (table, src, dst) graphs, laid out like LadybugDB's
 * adjacency: nodes get dense 32-bit indexes, one offsets array and one
 * neighbor array sized exactly once (no per-batch slack), and a tag array for
 * the visited set. A cached graph is reused only while the main pager's data
 * version (moves on every commit, ours included) and sqlite3_total_changes64()
 * (moves on our own row writes, also to temp tables) are unchanged.
 *
 * Inside a write transaction the graph is rebuilt for every statement and not
 * cached: ROLLBACK TO a savepoint reverts rows without moving either counter.
 */
// Resident adjacency only; build scratch and cursor outputs are not cached.
#define GRAPH_CACHE_BYTES (16 * 1024 * 1024)
#define GRAPH_CACHE_SLOTS 8
#define GRAPH_MAX_NODES 0xFFFFFFF0u

typedef struct GraphCsr GraphCsr;
struct GraphCsr {
  char *zKey;    // "table\x1fsrc\x1fdst"
  u32 dataVersion;
  i64 changes;
  u64 lastUse;
  u32 nNode;
  u32 nEdge;
  i64 *ids;      // dense index -> node id
  u32 *off;      // nNode + 1 offsets into adj
  u32 *adj;      // dense neighbor indexes
  u32 *slot;     // open-addressing id hash: dense index + 1, 0 = empty
  u32 mask;      // slot capacity - 1
  u32 *tag;      // visited set: tag[i] == gen
  u32 *parent;   // valid where tag[i] == gen
  u32 gen;
};

/** One per connection, shared by both modules (passed as pAux). */
typedef struct GraphCache GraphCache;
struct GraphCache {
  GraphCsr *a[GRAPH_CACHE_SLOTS];
  u64 clock;
  int refs;
};

static u64 graph_csr_bytes(GraphCsr *g) {
  if (!g) return 0;
  return sqlite3_msize(g) + sqlite3_msize(g->zKey) + sqlite3_msize(g->ids) +
      sqlite3_msize(g->off) + sqlite3_msize(g->adj) + sqlite3_msize(g->slot) +
      sqlite3_msize(g->tag) + sqlite3_msize(g->parent);
}

static u64 graph_cache_bytes(GraphCache *c) {
  u64 bytes = 0;
  for (int i = 0; i < GRAPH_CACHE_SLOTS; i++) bytes += graph_csr_bytes(c->a[i]);
  return bytes;
}

static void graph_cache_memory(sqlite3_context *ctx, int argc, sqlite3_value **argv) {
  UNUSED_PARAMETER(argc); UNUSED_PARAMETER(argv);
  sqlite3_result_int64(ctx, graph_cache_bytes(sqlite3_user_data(ctx)));
}

static void graph_csr_free(GraphCsr *g) {
  if (!g) return;
  sqlite3_free(g->zKey);
  sqlite3_free(g->ids);
  sqlite3_free(g->off);
  sqlite3_free(g->adj);
  sqlite3_free(g->slot);
  sqlite3_free(g->tag);
  sqlite3_free(g->parent);
  sqlite3_free(g);
}

static void graph_cache_free(void *p) {
  GraphCache *c = p;
  if (--c->refs) return;
  for (int i = 0; i < GRAPH_CACHE_SLOTS; i++) graph_csr_free(c->a[i]);
  sqlite3_free(c);
}

static u32 graph_hash(i64 id) {
  u64 x = (u64)id + 0x9E3779B97F4A7C15ull;
  x = (x ^ (x >> 30)) * 0xBF58476D1CE4E5B9ull;
  x = (x ^ (x >> 27)) * 0x94D049BB133111EBull;
  return (u32)(x ^ (x >> 31));
}

/** Dense index of id, or -1. */
static i64 graph_find(GraphCsr *g, i64 id) {
  if (!g->slot) return -1;
  for (u32 h = graph_hash(id) & g->mask;; h = (h + 1) & g->mask) {
    u32 s = g->slot[h];
    if (s == 0) return -1;
    if (g->ids[s - 1] == id) return s - 1;
  }
}

static int graph_rehash(GraphCsr *g, u32 cap) {
  u32 *slot = sqlite3_malloc64((u64)cap * sizeof(u32));
  if (!slot) return SQLITE_NOMEM;
  memset(slot, 0, (u64)cap * sizeof(u32));
  sqlite3_free(g->slot);
  g->slot = slot;
  g->mask = cap - 1;
  for (u32 i = 0; i < g->nNode; i++) {
    u32 h = graph_hash(g->ids[i]) & g->mask;
    while (slot[h]) h = (h + 1) & g->mask;
    slot[h] = i + 1;
  }
  return SQLITE_OK;
}

/** Dense index of id, adding it if new. */
static int graph_intern(GraphCsr *g, i64 id, u32 *pIdx, u32 *capIds) {
  i64 found = graph_find(g, id);
  if (found >= 0) { *pIdx = (u32)found; return SQLITE_OK; }
  if (g->nNode >= GRAPH_MAX_NODES) return SQLITE_FULL;
  if (g->nNode == *capIds) {
    u32 cap = *capIds ? *capIds * 2 : 1024;
    i64 *ids = sqlite3_realloc64(g->ids, (u64)cap * sizeof(i64));
    if (!ids) return SQLITE_NOMEM;
    g->ids = ids;
    *capIds = cap;
  }
  g->ids[g->nNode++] = id;
  // Keep the hash at most half full.
  if (!g->slot || (u64)g->nNode * 2 > (u64)g->mask + 1) {
    int rc = graph_rehash(g, g->slot ? (g->mask + 1) * 2 : 2048);
    if (rc != SQLITE_OK) return rc;
  } else {
    u32 h = graph_hash(id) & g->mask;
    while (g->slot[h]) h = (h + 1) & g->mask;
    g->slot[h] = g->nNode;
  }
  *pIdx = g->nNode - 1;
  return SQLITE_OK;
}

static int graph_endpoint(sqlite3_stmt *stmt, int col, i64 *out, char **pzErr) {
  int t = sqlite3_column_type(stmt, col);
  if (t == SQLITE_NULL) return SQLITE_DONE;
  if (t != SQLITE_INTEGER) {
    *pzErr = sqlite3_mprintf("graph: edge endpoints must be integers, got %s",
                             t == SQLITE_TEXT ? "TEXT" : t == SQLITE_FLOAT ? "REAL" : "BLOB");
    return SQLITE_ERROR;
  }
  *out = sqlite3_column_int64(stmt, col);
  return SQLITE_OK;
}

/** 1 if zTable (table or view) has column zCol; SQLite would otherwise read a
 * missing double-quoted column as a string literal. */
static int graph_has_column(sqlite3 *db, const char *zTable, const char *zCol) {
  sqlite3_stmt *stmt;
  int found = 0;
  if (sqlite3_prepare_v2(db,
          "SELECT 1 FROM pragma_table_xinfo(?) WHERE name = ? COLLATE NOCASE",
          -1, &stmt, NULL) != SQLITE_OK)
    return 0;
  sqlite3_bind_text(stmt, 1, zTable, -1, SQLITE_STATIC);
  sqlite3_bind_text(stmt, 2, zCol, -1, SQLITE_STATIC);
  found = sqlite3_step(stmt) == SQLITE_ROW;
  sqlite3_finalize(stmt);
  return found;
}

/** Scan the edge table once and build its CSR. Rows with a NULL endpoint are skipped. */
static int graph_build(sqlite3 *db, const char *zTable, const char *zSrc,
                       const char *zDst, GraphCsr **out, char **pzErr) {
  int rc;
  sqlite3_stmt *stmt = NULL;
  u32 *pairs = NULL; // (src, dst) dense pairs, transient
  u64 nPair = 0, capPair = 0;
  u32 capIds = 0;
  GraphCsr *g = sqlite3_malloc(sizeof(*g));
  if (!g) return SQLITE_NOMEM;
  memset(g, 0, sizeof(*g));

  if (!graph_has_column(db, zTable, zSrc) || !graph_has_column(db, zTable, zDst)) {
    *pzErr = sqlite3_mprintf("graph: cannot read edges %s(%s, %s): no such table or column",
                             zTable, zSrc, zDst);
    rc = SQLITE_ERROR;
    goto done;
  }
  // %w quotes identifiers, so the names cannot inject SQL.
  char *zSql = sqlite3_mprintf("SELECT \"%w\", \"%w\" FROM \"%w\"", zSrc, zDst, zTable);
  if (!zSql) { rc = SQLITE_NOMEM; goto done; }
  rc = sqlite3_prepare_v2(db, zSql, -1, &stmt, NULL);
  sqlite3_free(zSql);
  if (rc != SQLITE_OK) {
    *pzErr = sqlite3_mprintf("graph: cannot read edges %s(%s, %s): %s", zTable,
                             zSrc, zDst, sqlite3_errmsg(db));
    goto done;
  }
  while ((rc = sqlite3_step(stmt)) == SQLITE_ROW) {
    i64 s, d;
    u32 si, di;
    int r1 = graph_endpoint(stmt, 0, &s, pzErr);
    if (r1 == SQLITE_ERROR) { rc = r1; goto done; }
    int r2 = r1 == SQLITE_OK ? graph_endpoint(stmt, 1, &d, pzErr) : SQLITE_DONE;
    if (r2 == SQLITE_ERROR) { rc = r2; goto done; }
    if (r1 != SQLITE_OK || r2 != SQLITE_OK) continue;
    if (nPair >= GRAPH_MAX_NODES) { rc = SQLITE_FULL; goto done; }
    if ((rc = graph_intern(g, s, &si, &capIds)) != SQLITE_OK) goto done;
    if ((rc = graph_intern(g, d, &di, &capIds)) != SQLITE_OK) goto done;
    if (nPair == capPair) {
      capPair = capPair ? capPair * 2 : 4096;
      u32 *p = sqlite3_realloc64(pairs, capPair * 2 * sizeof(u32));
      if (!p) { rc = SQLITE_NOMEM; goto done; }
      pairs = p;
    }
    pairs[2 * nPair] = si;
    pairs[2 * nPair + 1] = di;
    nPair++;
  }
  if (rc != SQLITE_DONE) {
    *pzErr = sqlite3_mprintf("graph: reading edges: %s", sqlite3_errmsg(db));
    goto done;
  }

  // Counting sort of pairs by source into exactly-sized CSR arrays.
  g->nEdge = (u32)nPair;
  g->off = sqlite3_malloc64(((u64)g->nNode + 1) * sizeof(u32));
  g->adj = sqlite3_malloc64((nPair ? nPair : 1) * sizeof(u32));
  g->tag = sqlite3_malloc64(((u64)g->nNode + 1) * sizeof(u32));
  g->parent = sqlite3_malloc64(((u64)g->nNode + 1) * sizeof(u32));
  if (!g->off || !g->adj || !g->tag || !g->parent) { rc = SQLITE_NOMEM; goto done; }
  memset(g->off, 0, ((u64)g->nNode + 1) * sizeof(u32));
  memset(g->tag, 0, ((u64)g->nNode + 1) * sizeof(u32));
  for (u64 i = 0; i < nPair; i++) g->off[pairs[2 * i] + 1]++;
  for (u32 i = 0; i < g->nNode; i++) g->off[i + 1] += g->off[i];
  // Reuse tag as the per-source fill cursor, then clear it.
  for (u64 i = 0; i < nPair; i++) {
    u32 s = pairs[2 * i];
    g->adj[g->off[s] + g->tag[s]++] = pairs[2 * i + 1];
  }
  memset(g->tag, 0, ((u64)g->nNode + 1) * sizeof(u32));
  if (g->nNode && capIds > g->nNode) {
    i64 *ids = sqlite3_realloc64(g->ids, (u64)g->nNode * sizeof(i64));
    if (ids) g->ids = ids;
  }
  rc = SQLITE_OK;

done:
  sqlite3_finalize(stmt);
  sqlite3_free(pairs);
  if (rc == SQLITE_FULL && !*pzErr)
    *pzErr = sqlite3_mprintf("graph: more than %u nodes or edges", GRAPH_MAX_NODES);
  if (rc != SQLITE_OK) { graph_csr_free(g); return rc; }
  *out = g;
  return SQLITE_OK;
}

/**
 * The current graph for (table, src, dst). *pFresh is set when the caller owns
 * the result (write transaction) and must free it after the statement's scan.
 */
static int graph_get(sqlite3 *db, GraphCache *c, const char *zTable,
                     const char *zSrc, const char *zDst, GraphCsr **out,
                     int *pFresh, char **pzErr) {
  *pFresh = 0;
  if (sqlite3_txn_state(db, NULL) == SQLITE_TXN_WRITE) {
    *pFresh = 1;
    return graph_build(db, zTable, zSrc, zDst, out, pzErr);
  }
  u32 dv = 0;
  int known = sqlite3_file_control(db, "main", SQLITE_FCNTL_DATA_VERSION, &dv) == SQLITE_OK;
  i64 changes = sqlite3_total_changes64(db);
  char *zKey = sqlite3_mprintf("%s\x1f%s\x1f%s", zTable, zSrc, zDst);
  if (!zKey) return SQLITE_NOMEM;
  int victim = 0;
  for (int i = 0; i < GRAPH_CACHE_SLOTS; i++) {
    GraphCsr *g = c->a[i];
    if (g && strcmp(g->zKey, zKey) == 0) {
      if (known && g->dataVersion == dv && g->changes == changes) {
        sqlite3_free(zKey);
        g->lastUse = ++c->clock;
        *out = g;
        return SQLITE_OK;
      }
      victim = i;
      break;
    }
    if (!g || (c->a[victim] && g->lastUse < c->a[victim]->lastUse)) victim = i;
  }
  GraphCsr *g;
  int rc = graph_build(db, zTable, zSrc, zDst, &g, pzErr);
  if (rc != SQLITE_OK) { sqlite3_free(zKey); return rc; }
  g->zKey = zKey;
  g->dataVersion = dv;
  // An unknown data version never matches again, so the next call rebuilds.
  g->changes = known ? changes : -1;
  g->lastUse = ++c->clock;
  graph_csr_free(c->a[victim]);
  c->a[victim] = NULL;
  u64 bytes = graph_csr_bytes(g);
  if (bytes > GRAPH_CACHE_BYTES) {
    *pFresh = 1;
    *out = g;
    return SQLITE_OK;
  }
  while (graph_cache_bytes(c) + bytes > GRAPH_CACHE_BYTES) {
    int oldest = -1;
    for (int i = 0; i < GRAPH_CACHE_SLOTS; i++)
      if (c->a[i] && (oldest < 0 || c->a[i]->lastUse < c->a[oldest]->lastUse)) oldest = i;
    graph_csr_free(c->a[oldest]);
    c->a[oldest] = NULL;
  }
  c->a[victim] = g;
  *out = g;
  return SQLITE_OK;
}

static u32 graph_next_gen(GraphCsr *g) {
  if (++g->gen == 0) {
    memset(g->tag, 0, ((u64)g->nNode + 1) * sizeof(u32));
    g->gen = 1;
  }
  return g->gen;
}

/* ---- virtual table ---- */

enum { GRAPH_BFS, GRAPH_PATH };

// Column layout shared by both modules: two output columns, then hidden args.
#define GRAPH_COL_OUT0 0
#define GRAPH_COL_OUT1 1
#define GRAPH_COL_TABLE 2
#define GRAPH_COL_SRC 3
#define GRAPH_COL_DST 4
#define GRAPH_COL_ARG0 5 // bfs: start, path: source
#define GRAPH_COL_ARG1 6 // bfs: max_depth (optional), path: target
#define GRAPH_NCOL 7

typedef struct graph_vtab graph_vtab;
struct graph_vtab {
  sqlite3_vtab base;
  sqlite3 *db;
  GraphCache *cache;
  int kind;
};

typedef struct graph_cursor graph_cursor;
struct graph_cursor {
  sqlite3_vtab_cursor base;
  i64 *a;  // output column 0 (bfs: node, path: step)
  i64 *b;  // output column 1 (bfs: depth, path: node)
  i64 n;
  i64 i;
};

static int graph_connect_kind(sqlite3 *db, void *pAux, sqlite3_vtab **ppVtab,
                              int kind) {
  int rc = sqlite3_declare_vtab(
      db, kind == GRAPH_BFS
              ? "CREATE TABLE x(node, depth, edge_table hidden, src_column hidden,"
                " dst_column hidden, start hidden, max_depth hidden)"
              : "CREATE TABLE x(step, node, edge_table hidden, src_column hidden,"
                " dst_column hidden, source hidden, target hidden)");
  if (rc != SQLITE_OK) return rc;
  graph_vtab *p = sqlite3_malloc(sizeof(*p));
  if (!p) return SQLITE_NOMEM;
  memset(p, 0, sizeof(*p));
  p->db = db;
  p->cache = pAux;
  p->kind = kind;
  *ppVtab = &p->base;
  return SQLITE_OK;
}

static int graph_bfsConnect(sqlite3 *db, void *pAux, int argc,
                            const char *const *argv, sqlite3_vtab **ppVtab,
                            char **pzErr) {
  UNUSED_PARAMETER(argc); UNUSED_PARAMETER(argv); UNUSED_PARAMETER(pzErr);
  return graph_connect_kind(db, pAux, ppVtab, GRAPH_BFS);
}

static int graph_pathConnect(sqlite3 *db, void *pAux, int argc,
                             const char *const *argv, sqlite3_vtab **ppVtab,
                             char **pzErr) {
  UNUSED_PARAMETER(argc); UNUSED_PARAMETER(argv); UNUSED_PARAMETER(pzErr);
  return graph_connect_kind(db, pAux, ppVtab, GRAPH_PATH);
}

static int graphDisconnect(sqlite3_vtab *pVtab) {
  sqlite3_free(pVtab);
  return SQLITE_OK;
}

static int graphOpen(sqlite3_vtab *p, sqlite3_vtab_cursor **ppCursor) {
  UNUSED_PARAMETER(p);
  graph_cursor *cur = sqlite3_malloc(sizeof(*cur));
  if (!cur) return SQLITE_NOMEM;
  memset(cur, 0, sizeof(*cur));
  *ppCursor = &cur->base;
  return SQLITE_OK;
}

static int graphClose(sqlite3_vtab_cursor *cur) {
  graph_cursor *c = (graph_cursor *)cur;
  sqlite3_free(c->a);
  sqlite3_free(c->b);
  sqlite3_free(c);
  return SQLITE_OK;
}

/** idxNum is a bitmask of the hidden columns bound, in column order in argv. */
static int graphBestIndex(sqlite3_vtab *pVTab, sqlite3_index_info *info) {
  graph_vtab *p = (graph_vtab *)pVTab;
  int seen[GRAPH_NCOL] = {0};
  int mask = 0;
  for (int i = 0; i < info->nConstraint; i++) {
    const struct sqlite3_index_constraint *c = &info->aConstraint[i];
    if (c->iColumn < GRAPH_COL_TABLE || c->op != SQLITE_INDEX_CONSTRAINT_EQ) continue;
    if (!c->usable) return SQLITE_CONSTRAINT;
    if (seen[c->iColumn]) continue;
    seen[c->iColumn] = i + 1;
    mask |= 1 << c->iColumn;
  }
  int required = (1 << GRAPH_COL_TABLE) | (1 << GRAPH_COL_SRC) |
                 (1 << GRAPH_COL_DST) | (1 << GRAPH_COL_ARG0);
  if (p->kind == GRAPH_PATH) required |= 1 << GRAPH_COL_ARG1;
  if ((mask & required) != required) return SQLITE_CONSTRAINT;
  int argv = 0;
  for (int col = GRAPH_COL_TABLE; col < GRAPH_NCOL; col++) {
    if (!seen[col]) continue;
    info->aConstraintUsage[seen[col] - 1].argvIndex = ++argv;
    info->aConstraintUsage[seen[col] - 1].omit = 1;
  }
  info->idxNum = mask;
  info->estimatedCost = 1000;
  info->estimatedRows = 1000;
  return SQLITE_OK;
}

static int graph_push(graph_cursor *c, i64 *cap, i64 a, i64 b) {
  if (c->n == *cap) {
    i64 nc = *cap ? *cap * 2 : 64;
    i64 *na = sqlite3_realloc64(c->a, nc * sizeof(i64));
    if (!na) return SQLITE_NOMEM;
    c->a = na;
    i64 *nb = sqlite3_realloc64(c->b, nc * sizeof(i64));
    if (!nb) return SQLITE_NOMEM;
    c->b = nb;
    *cap = nc;
  }
  c->a[c->n] = a;
  c->b[c->n] = b;
  c->n++;
  return SQLITE_OK;
}

/** Level-synchronous BFS; rows are (node, depth) in visit order. */
static int graph_run_bfs(graph_cursor *c, GraphCsr *g, i64 start, i64 maxDepth) {
  i64 cap = 0;
  int rc = graph_push(c, &cap, start, 0);
  i64 s = graph_find(g, start);
  if (rc != SQLITE_OK || s < 0) return rc;
  u32 gen = graph_next_gen(g);
  g->tag[s] = gen;
  // The output rows double as the queue: each level is a contiguous range.
  u32 *q = sqlite3_malloc64(((u64)g->nNode + 1) * sizeof(u32));
  if (!q) return SQLITE_NOMEM;
  u64 head = 0, tail = 0;
  q[tail++] = (u32)s;
  for (i64 depth = 1; head < tail && (maxDepth < 0 || depth <= maxDepth); depth++) {
    u64 end = tail;
    for (; head < end; head++) {
      u32 u = q[head];
      for (u32 e = g->off[u]; e < g->off[u + 1]; e++) {
        u32 v = g->adj[e];
        if (g->tag[v] == gen) continue;
        g->tag[v] = gen;
        q[tail++] = v;
        if ((rc = graph_push(c, &cap, g->ids[v], depth)) != SQLITE_OK) goto done;
      }
    }
  }
done:
  sqlite3_free(q);
  return rc;
}

/** BFS from source until target; rows are (step, node) from source to target. */
static int graph_run_path(graph_cursor *c, GraphCsr *g, i64 source, i64 target) {
  i64 cap = 0;
  if (source == target) return graph_push(c, &cap, 0, source);
  i64 s = graph_find(g, source), t = graph_find(g, target);
  if (s < 0 || t < 0) return SQLITE_OK;
  u32 gen = graph_next_gen(g);
  u32 *q = sqlite3_malloc64(((u64)g->nNode + 1) * sizeof(u32));
  if (!q) return SQLITE_NOMEM;
  u64 head = 0, tail = 0;
  g->tag[s] = gen;
  q[tail++] = (u32)s;
  int found = 0;
  while (head < tail && !found) {
    u32 u = q[head++];
    for (u32 e = g->off[u]; e < g->off[u + 1]; e++) {
      u32 v = g->adj[e];
      if (g->tag[v] == gen) continue;
      g->tag[v] = gen;
      g->parent[v] = u;
      if (v == (u32)t) { found = 1; break; }
      q[tail++] = v;
    }
  }
  int rc = SQLITE_OK;
  if (found) {
    // Walk parents back to the source into q, then emit in forward order.
    u64 len = 0;
    for (u32 v = (u32)t;; v = g->parent[v]) {
      q[len++] = v;
      if (v == (u32)s) break;
    }
    for (u64 i = 0; i < len && rc == SQLITE_OK; i++)
      rc = graph_push(c, &cap, (i64)i, g->ids[q[len - 1 - i]]);
  }
  sqlite3_free(q);
  return rc;
}

static int graph_int_arg(graph_vtab *p, sqlite3_value *v, const char *name, i64 *out) {
  if (sqlite3_value_type(v) != SQLITE_INTEGER) {
    p->base.zErrMsg = sqlite3_mprintf("graph: %s must be an integer", name);
    return SQLITE_ERROR;
  }
  *out = sqlite3_value_int64(v);
  return SQLITE_OK;
}

static int graphFilter(sqlite3_vtab_cursor *cur, int idxNum, const char *idxStr,
                       int argc, sqlite3_value **argv) {
  UNUSED_PARAMETER(idxStr);
  graph_cursor *c = (graph_cursor *)cur;
  graph_vtab *p = (graph_vtab *)cur->pVtab;
  c->n = c->i = 0;
  sqlite3_value *arg[GRAPH_NCOL] = {0};
  int j = 0;
  for (int col = GRAPH_COL_TABLE; col < GRAPH_NCOL && j < argc; col++)
    if (idxNum & (1 << col)) arg[col] = argv[j++];

  const char *names[3];
  for (int col = GRAPH_COL_TABLE; col <= GRAPH_COL_DST; col++) {
    names[col - GRAPH_COL_TABLE] = (const char *)sqlite3_value_text(arg[col]);
    if (!names[col - GRAPH_COL_TABLE] || sqlite3_value_type(arg[col]) != SQLITE_TEXT) {
      p->base.zErrMsg = sqlite3_mprintf("graph: table and column names must be text");
      return SQLITE_ERROR;
    }
  }
  i64 x, y = -1;
  int rc = graph_int_arg(p, arg[GRAPH_COL_ARG0], p->kind == GRAPH_BFS ? "start" : "source", &x);
  if (rc != SQLITE_OK) return rc;
  if (arg[GRAPH_COL_ARG1] && !(p->kind == GRAPH_BFS && sqlite3_value_type(arg[GRAPH_COL_ARG1]) == SQLITE_NULL)) {
    rc = graph_int_arg(p, arg[GRAPH_COL_ARG1], p->kind == GRAPH_BFS ? "max_depth" : "target", &y);
    if (rc != SQLITE_OK) return rc;
    if (p->kind == GRAPH_BFS && y < 0) {
      p->base.zErrMsg = sqlite3_mprintf("graph: max_depth must be >= 0");
      return SQLITE_ERROR;
    }
  }

  GraphCsr *g;
  int fresh;
  char *zErr = NULL;
  rc = graph_get(p->db, p->cache, names[0], names[1], names[2], &g, &fresh, &zErr);
  if (rc != SQLITE_OK) {
    p->base.zErrMsg = zErr ? zErr : sqlite3_mprintf("graph: %s", sqlite3_errstr(rc));
    return rc == SQLITE_NOMEM ? rc : SQLITE_ERROR;
  }
  rc = p->kind == GRAPH_BFS ? graph_run_bfs(c, g, x, y) : graph_run_path(c, g, x, y);
  if (fresh) graph_csr_free(g);
  return rc;
}

static int graphNext(sqlite3_vtab_cursor *cur) {
  ((graph_cursor *)cur)->i++;
  return SQLITE_OK;
}

static int graphEof(sqlite3_vtab_cursor *cur) {
  graph_cursor *c = (graph_cursor *)cur;
  return c->i >= c->n;
}

static int graphRowid(sqlite3_vtab_cursor *cur, sqlite_int64 *pRowid) {
  *pRowid = ((graph_cursor *)cur)->i;
  return SQLITE_OK;
}

static int graphColumn(sqlite3_vtab_cursor *cur, sqlite3_context *ctx, int i) {
  graph_cursor *c = (graph_cursor *)cur;
  if (i == GRAPH_COL_OUT0) sqlite3_result_int64(ctx, c->a[c->i]);
  else if (i == GRAPH_COL_OUT1) sqlite3_result_int64(ctx, c->b[c->i]);
  return SQLITE_OK;
}

#if SQLITE_VERSION_NUMBER >= 3044000
#define GRAPH_MODULE_TAIL 0, 0
#else
#define GRAPH_MODULE_TAIL 0
#endif
#define GRAPH_MODULE(connect)                                                  \
  {                                                                            \
    /* iVersion    */ 0, /* xCreate */ 0, connect, graphBestIndex,             \
    graphDisconnect, /* xDestroy */ 0, graphOpen, graphClose, graphFilter,     \
    graphNext, graphEof, graphColumn, graphRowid, /* xUpdate .. */ 0, 0, 0, 0, \
    0, 0, 0, 0, 0, 0, GRAPH_MODULE_TAIL,                                       \
  }

static sqlite3_module graph_bfsModule = GRAPH_MODULE(graph_bfsConnect);
static sqlite3_module graph_pathModule = GRAPH_MODULE(graph_pathConnect);

typedef struct { i64 src, dst, id; } GraphWitnessEdge;
typedef struct { i64 id, edge; u64 parent; } GraphWitnessNode;

static int graph_witness_tick(sqlite3 *db, i64 *work) {
#if SQLITE_VERSION_NUMBER >= 3041000
  if (sqlite3_is_interrupted(db)) return SQLITE_INTERRUPT;
#else
  UNUSED_PARAMETER(db);
#endif
  if (*work == 0) return SQLITE_FULL;
  --*work;
  return SQLITE_OK;
}

/* One arena bounds native payload, including the returned JSON. SQLite VM,
 * allocator bookkeeping and the caller's result copy are outside this budget. */
static void graph_witness(sqlite3_context *ctx, int argc, sqlite3_value **argv) {
  UNUSED_PARAMETER(argc);
  sqlite3 *db = sqlite3_context_db_handle(ctx);
  sqlite3_stmt *stmt = NULL;
  unsigned char *arena = NULL;
  int rc = SQLITE_OK;
  const char *err = NULL;
  const char *names[4];
  for (int i = 0; i < 4; i++) {
    if (sqlite3_value_type(argv[i]) != SQLITE_TEXT) {
      err = "graph_witness: names must be text"; goto done;
    }
    names[i] = (const char *)sqlite3_value_text(argv[i]);
    if (!names[i]) { rc = SQLITE_NOMEM; goto done; }
    if ((int)strlen(names[i]) != sqlite3_value_bytes(argv[i])) {
      err = "graph_witness: names must not contain NUL"; goto done;
    }
  }
  for (int i = 4; i < 8; i++) {
    if (sqlite3_value_type(argv[i]) != SQLITE_INTEGER) {
      err = "graph_witness: endpoints and budgets must be integers"; goto done;
    }
  }
#if SQLITE_VERSION_NUMBER >= 3041000
  if (sqlite3_libversion_number() < 3041000) {
    err = "graph_witness requires SQLite 3.41 or later"; goto done;
  }
#else
  err = "graph_witness requires SQLite 3.41 or later"; goto done;
#endif
  i64 source = sqlite3_value_int64(argv[4]);
  i64 target = sqlite3_value_int64(argv[5]);
  i64 work = sqlite3_value_int64(argv[6]);
  i64 bytes = sqlite3_value_int64(argv[7]);
  if (work < 1 || bytes < 128 || bytes > 64 * 1024 * 1024) {
    err = "graph_witness: positive work and 128..67108864 bytes required"; goto done;
  }
  if ((rc = graph_witness_tick(db, &work)) != SQLITE_OK) goto done;
  for (int i = 1; i < 4; i++) {
    if (!graph_has_column(db, names[0], names[i])) {
      if (sqlite3_errcode(db) == SQLITE_INTERRUPT) { rc = SQLITE_INTERRUPT; goto done; }
      err = "graph_witness: missing table or column"; goto done;
    }
  }
  arena = sqlite3_malloc64((u64)bytes);
  if (!arena) { rc = SQLITE_NOMEM; goto done; }
  if (sqlite3_msize(arena) > (u64)bytes) { rc = SQLITE_TOOBIG; goto done; }
  char *sql = sqlite3_mprintf("SELECT \"%w\",\"%w\",\"%w\" FROM \"%w\"",
                              names[1], names[2], names[3], names[0]);
  if (!sql) { rc = SQLITE_NOMEM; goto done; }
  rc = sqlite3_prepare_v2(db, sql, -1, &stmt, NULL);
  sqlite3_free(sql);
  if (rc != SQLITE_OK) goto done;
  GraphWitnessEdge *edges = (GraphWitnessEdge *)arena;
  u64 ne = 0;
  while ((rc = sqlite3_step(stmt)) == SQLITE_ROW) {
    if ((rc = graph_witness_tick(db, &work)) != SQLITE_OK) goto done;
    for (int i = 0; i < 3; i++) {
      if (sqlite3_column_type(stmt, i) != SQLITE_INTEGER) {
        err = "graph_witness: endpoints and edge IDs must be non-NULL integers";
        goto done;
      }
    }
    i64 id = sqlite3_column_int64(stmt, 2);
    if ((ne + 1) * sizeof(*edges) > (u64)bytes) { rc = SQLITE_TOOBIG; goto done; }
    edges[ne++] = (GraphWitnessEdge){sqlite3_column_int64(stmt, 0),
                                   sqlite3_column_int64(stmt, 1), id};
  }
  if (rc != SQLITE_DONE) goto done;
  sqlite3_finalize(stmt); stmt = NULL;
  u64 used = ne * sizeof(*edges);
  // Half-full hash tables use the same ID hash as the cached CSR. Every probe
  // consumes work, so collisions remain cancellable and bounded by the caller.
  u32 cap = 2;
  while ((u64)cap < 2 * (ne + 1)) cap *= 2;
  u64 scratch = (u64)cap * sizeof(u32) * 2 + ne * sizeof(u32);
  if (used + scratch > (u64)bytes) { rc = SQLITE_TOOBIG; goto done; }
  u32 *slots = (u32 *)(arena + used);
  u32 *heads = slots + cap;
  u32 *next = heads + cap;
  used += scratch;
  // Align the queue's int64 fields after the uint32 scratch arrays.
  used = (used + 7) & ~(u64)7;
  for (u32 i = 0; i < cap; i++) {
    if ((rc = graph_witness_tick(db, &work)) != SQLITE_OK) goto done;
    slots[i] = heads[i] = 0;
  }
  for (u64 e = 0; e < ne; e++) {
    u32 h = graph_hash(edges[e].id) & (cap - 1);
    for (;; h = (h + 1) & (cap - 1)) {
      if ((rc = graph_witness_tick(db, &work)) != SQLITE_OK) goto done;
      if (!slots[h]) { slots[h] = (u32)e + 1; break; }
      if (edges[slots[h] - 1].id == edges[e].id) {
        err = "graph_witness: duplicate edge ID"; goto done;
      }
    }
  }
  // Insert backwards so each source's list retains the SQL scan's tie order.
  for (u64 e = ne; e > 0;) {
    --e;
    u32 h = graph_hash(edges[e].src) & (cap - 1);
    for (;; h = (h + 1) & (cap - 1)) {
      if ((rc = graph_witness_tick(db, &work)) != SQLITE_OK) goto done;
      if (!heads[h] || edges[heads[h] - 1].src == edges[e].src) break;
    }
    next[e] = heads[h];
    heads[h] = (u32)e + 1;
  }
  // Reuse edge-ID validation scratch for visited-node lookup.
  for (u32 i = 0; i < cap; i++) {
    if ((rc = graph_witness_tick(db, &work)) != SQLITE_OK) goto done;
    slots[i] = 0;
  }
  if (used + sizeof(GraphWitnessNode) > (u64)bytes) { rc = SQLITE_TOOBIG; goto done; }
  GraphWitnessNode *nodes = (GraphWitnessNode *)(arena + used);
  nodes[0] = (GraphWitnessNode){source, 0, 0};
  slots[graph_hash(source) & (cap - 1)] = 1;
  u64 nn = 1, head = 0, found = source == target ? 0 : (u64)-1;
  while (head < nn && found == (u64)-1) {
    u32 h = graph_hash(nodes[head].id) & (cap - 1);
    for (;; h = (h + 1) & (cap - 1)) {
      if ((rc = graph_witness_tick(db, &work)) != SQLITE_OK) goto done;
      if (!heads[h] || edges[heads[h] - 1].src == nodes[head].id) break;
    }
    for (u32 link = heads[h]; link; link = next[link - 1]) {
      if ((rc = graph_witness_tick(db, &work)) != SQLITE_OK) goto done;
      u32 e = link - 1;
      u32 slot = graph_hash(edges[e].dst) & (cap - 1);
      for (;; slot = (slot + 1) & (cap - 1)) {
        if ((rc = graph_witness_tick(db, &work)) != SQLITE_OK) goto done;
        if (!slots[slot] || nodes[slots[slot] - 1].id == edges[e].dst) break;
      }
      if (slots[slot]) continue;
      if (used + (nn + 1) * sizeof(*nodes) > (u64)bytes) { rc = SQLITE_TOOBIG; goto done; }
      nodes[nn] = (GraphWitnessNode){edges[e].dst, edges[e].id, head};
      slots[slot] = (u32)nn + 1;
      if (edges[e].dst == target) found = nn;
      nn++;
      if (found != (u64)-1) break;
    }
    head++;
  }
  used += nn * sizeof(*nodes);
  u64 len = 0;
  if (found != (u64)-1) {
    // Reverse only the selected parent chain, then emit it in one forward pass.
    u64 previous = (u64)-1;
    for (u64 n = found;;) {
      if ((rc = graph_witness_tick(db, &work)) != SQLITE_OK) goto done;
      u64 parent = nodes[n].parent;
      nodes[n].parent = previous;
      previous = n;
      len++;
      if (n == 0) break;
      n = parent;
    }
  }
  /* Signed int64 JSON pairs need at most 46 bytes per node. */
  if (used + len * 48 + 3 > (u64)bytes) { rc = SQLITE_TOOBIG; goto done; }
  char *out = (char *)arena + used;
  char *p = out;
  *p++ = '[';
  for (u64 step = 0, n = 0; step < len; step++, n = nodes[n].parent) {
    if ((rc = graph_witness_tick(db, &work)) != SQLITE_OK) goto done;
    sqlite3_snprintf(48, p, step ? ",[%lld,%lld]" : "[%lld,null]",
                     nodes[n].id, nodes[n].edge);
    p += strlen(p);
  }
  *p++ = ']'; *p = 0;
  sqlite3_result_text(ctx, out, (int)(p - out), SQLITE_TRANSIENT);
  rc = SQLITE_OK;
done:
  sqlite3_finalize(stmt);
  sqlite3_free(arena);
  if (err || rc != SQLITE_OK) {
    sqlite3_result_error(ctx, err ? err : rc == SQLITE_FULL ? "graph_witness: work limit exceeded" :
                         rc == SQLITE_TOOBIG ? "graph_witness: memory limit exceeded" : sqlite3_errstr(rc), -1);
    if (!err) sqlite3_result_error_code(ctx, rc);
  }
}

/** Registers graph_bfs and graph_shortest_path with one shared per-connection cache. */
static int graph_register(sqlite3 *db, char **pzErrMsg) {
  if (sqlite3_libversion_number() < 3037000) {
    *pzErrMsg = sqlite3_mprintf("graph_bfs requires SQLite 3.37.0 or later");
    return SQLITE_ERROR;
  }
  GraphCache *c = sqlite3_malloc(sizeof(*c));
  if (!c) return SQLITE_NOMEM;
  memset(c, 0, sizeof(*c));
  // Keep an owner reference until every registration has completed.
  c->refs = 2;
  int rc = sqlite3_create_module_v2(db, "graph_bfs", &graph_bfsModule, c, graph_cache_free);
  if (rc == SQLITE_OK) {
    c->refs++;
    rc = sqlite3_create_module_v2(db, "graph_shortest_path", &graph_pathModule, c,
                                  graph_cache_free);
  }
  if (rc == SQLITE_OK)
    rc = sqlite3_create_function_v2(db, "graph_witness", 8, SQLITE_UTF8 | SQLITE_DIRECTONLY,
                                    NULL, graph_witness, NULL, NULL, NULL);
  if (rc == SQLITE_OK) {
    c->refs++;
    rc = sqlite3_create_function_v2(db, "graph_cache_bytes", 0, SQLITE_UTF8 | SQLITE_DIRECTONLY,
                                    c, graph_cache_memory, NULL, NULL, graph_cache_free);
  }
  graph_cache_free(c);
  if (rc != SQLITE_OK)
    *pzErrMsg = sqlite3_mprintf("Error creating graph modules: %s", sqlite3_errmsg(db));
  return rc;
}
//////////////////////////////////////////////////////////////////////////////
// End inlined: sqlite-vec-graph.c
//////////////////////////////////////////////////////////////////////////////

#define VEC_CONSTRUCTOR_ERROR "vec0 constructor error: "
static int vec0_init(sqlite3 *db, void *pAux, int argc, const char *const *argv,
                     sqlite3_vtab **ppVtab, char **pzErr, bool isCreate) {
  char *storageName = NULL; // storage=<provider>
  char *storageUri = NULL;  // storage_uri='...'

  UNUSED_PARAMETER(pAux);
  vec0_vtab *pNew;
  int rc;
  const char *zSql;

  pNew = sqlite3_malloc(sizeof(*pNew));
  if (pNew == 0)
    return SQLITE_NOMEM;
  memset(pNew, 0, sizeof(*pNew));

  // Declared chunk_size=N for entire table.
  // -1 to use the defualt, otherwise will get re-assigned on `chunk_size=N`
  // option
  int chunk_size = -1;
  int numVectorColumns = 0;
  int numPartitionColumns = 0;
  int numAuxiliaryColumns = 0;
  int numMetadataColumns = 0;
  int user_column_idx = 0;

  // track if a "primary key" column is defined
  char *pkColumnName = NULL;
  int pkColumnNameLength;
  int pkColumnType = SQLITE_INTEGER;

  for (int i = 3; i < argc; i++) {
    struct VectorColumnDefinition vecColumn;
    struct Vec0PartitionColumnDefinition partitionColumn;
    struct Vec0AuxiliaryColumnDefinition auxColumn;
    struct Vec0MetadataColumnDefinition metadataColumn;
    char *cName = NULL;
    int cNameLength;
    int cType;

    // Scenario #1: Constructor argument is a vector column definition, ie `foo float[1024]`
    rc = vec0_parse_vector_column(argv[i], strlen(argv[i]), &vecColumn);
    if (rc == SQLITE_ERROR) {
      *pzErr = sqlite3_mprintf(
          VEC_CONSTRUCTOR_ERROR "could not parse vector column '%s'", argv[i]);
      goto error;
    }
    if (rc == SQLITE_OK) {
      if (numVectorColumns >= VEC0_MAX_VECTOR_COLUMNS) {
        sqlite3_free(vecColumn.name);
        *pzErr = sqlite3_mprintf(VEC_CONSTRUCTOR_ERROR
                                 "Too many provided vector columns, maximum %d",
                                 VEC0_MAX_VECTOR_COLUMNS);
        goto error;
      }

      if (vecColumn.dimensions > SQLITE_VEC_VEC0_MAX_DIMENSIONS) {
        sqlite3_free(vecColumn.name);
        *pzErr = sqlite3_mprintf(
            VEC_CONSTRUCTOR_ERROR
            "Dimension on vector column too large, provided %lld, maximum %lld",
            (i64)vecColumn.dimensions, SQLITE_VEC_VEC0_MAX_DIMENSIONS);
        goto error;
      }

      // DiskANN validation
      if (vecColumn.index_type == VEC0_INDEX_TYPE_DISKANN) {
        if (vecColumn.element_type == SQLITE_VEC_ELEMENT_TYPE_BIT) {
          sqlite3_free(vecColumn.name);
          *pzErr = sqlite3_mprintf(
              VEC_CONSTRUCTOR_ERROR
              "DiskANN index is not supported on bit vector columns");
          goto error;
        }
        if (vecColumn.diskann.quantizer_type == VEC0_DISKANN_QUANTIZER_BINARY &&
            (vecColumn.dimensions % CHAR_BIT) != 0) {
          sqlite3_free(vecColumn.name);
          *pzErr = sqlite3_mprintf(
              VEC_CONSTRUCTOR_ERROR
              "DiskANN with binary quantizer requires dimensions divisible by 8");
          goto error;
        }
      }

      pNew->user_column_kinds[user_column_idx] = SQLITE_VEC0_USER_COLUMN_KIND_VECTOR;
      pNew->user_column_idxs[user_column_idx] = numVectorColumns;
      memcpy(&pNew->vector_columns[numVectorColumns], &vecColumn, sizeof(vecColumn));
      numVectorColumns++;
      pNew->numVectorColumns = numVectorColumns;
      user_column_idx++;

      continue;
    }

    // Scenario #2: Constructor argument is a partition key column definition, ie `user_id text partition key`
    rc = vec0_parse_partition_key_definition(argv[i], strlen(argv[i]), &cName,
                                      &cNameLength, &cType);
    if (rc == SQLITE_OK) {
      if (numPartitionColumns >= VEC0_MAX_PARTITION_COLUMNS) {
        *pzErr = sqlite3_mprintf(
            VEC_CONSTRUCTOR_ERROR
            "More than %d partition key columns were provided",
            VEC0_MAX_PARTITION_COLUMNS);
        goto error;
      }
      partitionColumn.type = cType;
      partitionColumn.name_length = cNameLength;
      partitionColumn.name = sqlite3_mprintf("%.*s", cNameLength, cName);
      if(!partitionColumn.name) {
        rc = SQLITE_NOMEM;
        goto error;
      }

      pNew->user_column_kinds[user_column_idx] = SQLITE_VEC0_USER_COLUMN_KIND_PARTITION;
      pNew->user_column_idxs[user_column_idx] = numPartitionColumns;
      memcpy(&pNew->paritition_columns[numPartitionColumns], &partitionColumn, sizeof(partitionColumn));
      numPartitionColumns++;
      pNew->numPartitionColumns = numPartitionColumns;
      user_column_idx++;
      continue;
    }

    // Scenario #3: Constructor argument is a primary key column definition, ie `article_id text primary key`
    rc = vec0_parse_primary_key_definition(argv[i], strlen(argv[i]), &cName,
                                      &cNameLength, &cType);
    if (rc == SQLITE_OK) {
      if (pkColumnName) {
        *pzErr = sqlite3_mprintf(
            VEC_CONSTRUCTOR_ERROR
            "More than one primary key definition was provided, vec0 only "
            "suports a single primary key column",
            argv[i]);
        goto error;
      }
      pkColumnName = cName;
      pkColumnNameLength = cNameLength;
      pkColumnType = cType;
      continue;
    }

    // Scenario #4: Constructor argument is a auxiliary column definition, ie `+contents text`
    rc = vec0_parse_auxiliary_column_definition(argv[i], strlen(argv[i]), &cName,
                                      &cNameLength, &cType);
    if(rc == SQLITE_OK) {
      if (numAuxiliaryColumns >= VEC0_MAX_AUXILIARY_COLUMNS) {
        *pzErr = sqlite3_mprintf(
            VEC_CONSTRUCTOR_ERROR
            "More than %d auxiliary columns were provided",
            VEC0_MAX_AUXILIARY_COLUMNS);
        goto error;
      }
      auxColumn.type = cType;
      auxColumn.name_length = cNameLength;
      auxColumn.name = sqlite3_mprintf("%.*s", cNameLength, cName);
      if(!auxColumn.name) {
        rc = SQLITE_NOMEM;
        goto error;
      }

      pNew->user_column_kinds[user_column_idx] = SQLITE_VEC0_USER_COLUMN_KIND_AUXILIARY;
      pNew->user_column_idxs[user_column_idx] = numAuxiliaryColumns;
      memcpy(&pNew->auxiliary_columns[numAuxiliaryColumns], &auxColumn, sizeof(auxColumn));
      numAuxiliaryColumns++;
      pNew->numAuxiliaryColumns = numAuxiliaryColumns;
      user_column_idx++;
      continue;
    }

    vec0_metadata_column_kind kind;
    rc = vec0_parse_metadata_column_definition(argv[i], strlen(argv[i]), &cName,
                                      &cNameLength, &kind);
    if(rc == SQLITE_OK) {
      if (numMetadataColumns >= VEC0_MAX_METADATA_COLUMNS) {
        *pzErr = sqlite3_mprintf(
            VEC_CONSTRUCTOR_ERROR
            "More than %d metadata columns were provided",
            VEC0_MAX_METADATA_COLUMNS);
        goto error;
      }
      metadataColumn.kind = kind;
      metadataColumn.name_length = cNameLength;
      metadataColumn.name = sqlite3_mprintf("%.*s", cNameLength, cName);
      if(!metadataColumn.name) {
        rc = SQLITE_NOMEM;
        goto error;
      }

      pNew->user_column_kinds[user_column_idx] = SQLITE_VEC0_USER_COLUMN_KIND_METADATA;
      pNew->user_column_idxs[user_column_idx] = numMetadataColumns;
      memcpy(&pNew->metadata_columns[numMetadataColumns], &metadataColumn, sizeof(metadataColumn));
      numMetadataColumns++;
      pNew->numMetadataColumns = numMetadataColumns;
      user_column_idx++;
      continue;
    }

    // Scenario #4: Constructor argument is a table-level option, ie `chunk_size`

    char *key;
    char *value;
    int keyLength, valueLength;
    {
      char *zStr = NULL;
      int src = vec0_parse_string_option(argv[i], "storage_uri", &zStr);
      if (src == SQLITE_ERROR || src == SQLITE_NOMEM) {
        *pzErr = sqlite3_mprintf(VEC_CONSTRUCTOR_ERROR
                                 "storage_uri must be a single-quoted string");
        goto error;
      }
      if (src == SQLITE_OK) {
        sqlite3_free(storageUri);
        storageUri = zStr;
        continue;
      }
    }
    rc = vec0_parse_table_option(argv[i], strlen(argv[i]), &key, &keyLength,
                                 &value, &valueLength);
    if (rc == SQLITE_ERROR) {
      *pzErr = sqlite3_mprintf(
          VEC_CONSTRUCTOR_ERROR "could not parse table option '%s'", argv[i]);
      goto error;
    }
    if (rc == SQLITE_OK) {
      if (keyLength == 7 && sqlite3_strnicmp(key, "storage", 7) == 0) {
        sqlite3_free(storageName);
        storageName = sqlite3_mprintf("%.*s", valueLength, value);
        if (!storageName) goto error;
      } else if (sqlite3_strnicmp(key, "chunk_size", keyLength) == 0) {
        chunk_size = atoi(value);
        if (chunk_size <= 0) {
          // IMP: V01931_18769
          *pzErr =
              sqlite3_mprintf(VEC_CONSTRUCTOR_ERROR
                              "chunk_size must be a non-zero positive integer");
          goto error;
        }
        if ((chunk_size % 8) != 0) {
          // IMP: V14110_30948
          *pzErr = sqlite3_mprintf(VEC_CONSTRUCTOR_ERROR
                                   "chunk_size must be divisible by 8");
          goto error;
        }
#define SQLITE_VEC_CHUNK_SIZE_MAX 4096
        if (chunk_size > SQLITE_VEC_CHUNK_SIZE_MAX) {
          *pzErr =
              sqlite3_mprintf(VEC_CONSTRUCTOR_ERROR "chunk_size too large");
          goto error;
        }
      } else {
        // IMP: V27642_11712
        *pzErr = sqlite3_mprintf(
            VEC_CONSTRUCTOR_ERROR "Unknown table option: %.*s", keyLength, key);
        goto error;
      }
      continue;
    }

    // Scenario #5: Unknown constructor argument
    *pzErr =
        sqlite3_mprintf(VEC_CONSTRUCTOR_ERROR "Could not parse '%s'", argv[i]);
    goto error;
  }

  if (chunk_size < 0) {
    chunk_size = 1024;
  }

  if (numVectorColumns <= 0) {
    *pzErr = sqlite3_mprintf(VEC_CONSTRUCTOR_ERROR
                             "At least one vector column is required");
    goto error;
  }

#if SQLITE_VEC_ENABLE_RESCORE
  {
    int hasRescore = 0;
    for (int i = 0; i < numVectorColumns; i++) {
      if (pNew->vector_columns[i].index_type == VEC0_INDEX_TYPE_RESCORE) {
        hasRescore = 1;
        break;
      }
    }
    if (hasRescore) {
      if (numMetadataColumns > 0) {
        *pzErr = sqlite3_mprintf(VEC_CONSTRUCTOR_ERROR
            "Metadata columns are not supported with rescore indexes");
        goto error;
      }
      if (numPartitionColumns > 0) {
        *pzErr = sqlite3_mprintf(VEC_CONSTRUCTOR_ERROR
            "Partition key columns are not supported with rescore indexes");
        goto error;
      }
    }
  }
#endif

  // IVF indexes do not support auxiliary, metadata, or partition key columns.
  {
    int has_ivf = 0;
    for (int i = 0; i < numVectorColumns; i++) {
      if (pNew->vector_columns[i].index_type == VEC0_INDEX_TYPE_IVF) {
        has_ivf = 1;
        break;
      }
    }
    if (has_ivf) {
      if (numPartitionColumns > 0) {
        *pzErr = sqlite3_mprintf(VEC_CONSTRUCTOR_ERROR
            "partition key columns are not supported with IVF indexes");
        goto error;
      }
      if (numMetadataColumns > 0) {
        *pzErr = sqlite3_mprintf(VEC_CONSTRUCTOR_ERROR
            "metadata columns are not supported with IVF indexes");
        goto error;
      }
    }
  }

  // Graph-indexed (DiskANN, HNSW) columns cannot coexist with metadata or
  // partition columns: the graph search cannot apply their filters.
  for (int i = 0; i < numVectorColumns; i++) {
    enum Vec0IndexType t = pNew->vector_columns[i].index_type;
    if (t == VEC0_INDEX_TYPE_DISKANN || t == VEC0_INDEX_TYPE_HNSW) {
      const char *name = t == VEC0_INDEX_TYPE_DISKANN ? "DiskANN" : "HNSW";
      if (numMetadataColumns > 0) {
        *pzErr = sqlite3_mprintf(
            VEC_CONSTRUCTOR_ERROR
            "Metadata columns are not supported with %s-indexed vector columns", name);
        goto error;
      }
      if (numPartitionColumns > 0) {
        *pzErr = sqlite3_mprintf(
            VEC_CONSTRUCTOR_ERROR
            "Partition key columns are not supported with %s-indexed vector columns", name);
        goto error;
      }
      break;
    }
  }

  // Determine whether to add the FTS5-style hidden command column.
  // New tables (isCreate) always get it; existing tables only if created
  // with v0.1.10+ (which validated no column name == table name).
  int hasCommandColumn = 0;
  if (isCreate) {
    // Validate no user column name conflicts with the table name
    const char *tblName = argv[2];
    int tblNameLen = (int)strlen(tblName);
    for (int i = 0; i < numVectorColumns; i++) {
      if (pNew->vector_columns[i].name_length == tblNameLen &&
          sqlite3_strnicmp(pNew->vector_columns[i].name, tblName, tblNameLen) == 0) {
        *pzErr = sqlite3_mprintf(
            VEC_CONSTRUCTOR_ERROR
            "column name '%s' conflicts with table name (reserved for command column)",
            tblName);
        goto error;
      }
    }
    for (int i = 0; i < numPartitionColumns; i++) {
      if (pNew->paritition_columns[i].name_length == tblNameLen &&
          sqlite3_strnicmp(pNew->paritition_columns[i].name, tblName, tblNameLen) == 0) {
        *pzErr = sqlite3_mprintf(
            VEC_CONSTRUCTOR_ERROR
            "column name '%s' conflicts with table name (reserved for command column)",
            tblName);
        goto error;
      }
    }
    for (int i = 0; i < numAuxiliaryColumns; i++) {
      if (pNew->auxiliary_columns[i].name_length == tblNameLen &&
          sqlite3_strnicmp(pNew->auxiliary_columns[i].name, tblName, tblNameLen) == 0) {
        *pzErr = sqlite3_mprintf(
            VEC_CONSTRUCTOR_ERROR
            "column name '%s' conflicts with table name (reserved for command column)",
            tblName);
        goto error;
      }
    }
    for (int i = 0; i < numMetadataColumns; i++) {
      if (pNew->metadata_columns[i].name_length == tblNameLen &&
          sqlite3_strnicmp(pNew->metadata_columns[i].name, tblName, tblNameLen) == 0) {
        *pzErr = sqlite3_mprintf(
            VEC_CONSTRUCTOR_ERROR
            "column name '%s' conflicts with table name (reserved for command column)",
            tblName);
        goto error;
      }
    }
    hasCommandColumn = 1;
  } else {
    // xConnect: check _info shadow table for version
    sqlite3_stmt *stmtInfo = NULL;
    char *zInfoSql = sqlite3_mprintf(
        "SELECT value FROM " VEC0_SHADOW_INFO_NAME " WHERE key = 'CREATE_VERSION_PATCH'",
        argv[1], argv[2]);
    if (zInfoSql) {
      int infoRc = sqlite3_prepare_v2(db, zInfoSql, -1, &stmtInfo, NULL);
      sqlite3_free(zInfoSql);
      if (infoRc == SQLITE_OK && sqlite3_step(stmtInfo) == SQLITE_ROW) {
        int patch = sqlite3_column_int(stmtInfo, 0);
        hasCommandColumn = (patch >= 10);  // v0.1.10+
      }
      // If _info doesn't exist or has no version, assume old table
      sqlite3_finalize(stmtInfo);
    }
  }
  pNew->hasCommandColumn = hasCommandColumn;

  sqlite3_str *createStr = sqlite3_str_new(NULL);
  sqlite3_str_appendall(createStr, "CREATE TABLE x(");
  if (pkColumnName) {
    sqlite3_str_appendf(createStr, "\"%.*w\" primary key, ", pkColumnNameLength,
                        pkColumnName);
  } else {
    sqlite3_str_appendall(createStr, "rowid, ");
  }
  for (int i = 0; i < numVectorColumns + numPartitionColumns + numAuxiliaryColumns + numMetadataColumns; i++) {
    switch(pNew->user_column_kinds[i]) {
      case SQLITE_VEC0_USER_COLUMN_KIND_VECTOR: {
        int vector_idx = pNew->user_column_idxs[i];
        sqlite3_str_appendf(createStr, "\"%.*w\", ",
                        pNew->vector_columns[vector_idx].name_length,
                        pNew->vector_columns[vector_idx].name);
        break;
      }
      case SQLITE_VEC0_USER_COLUMN_KIND_PARTITION: {
        int partition_idx = pNew->user_column_idxs[i];
        sqlite3_str_appendf(createStr, "\"%.*w\", ",
                        pNew->paritition_columns[partition_idx].name_length,
                        pNew->paritition_columns[partition_idx].name);
        break;
      }
      case SQLITE_VEC0_USER_COLUMN_KIND_AUXILIARY: {
        int auxiliary_idx = pNew->user_column_idxs[i];
        sqlite3_str_appendf(createStr, "\"%.*w\", ",
                        pNew->auxiliary_columns[auxiliary_idx].name_length,
                        pNew->auxiliary_columns[auxiliary_idx].name);
        break;
      }
      case SQLITE_VEC0_USER_COLUMN_KIND_METADATA: {
        int metadata_idx = pNew->user_column_idxs[i];
        sqlite3_str_appendf(createStr, "\"%.*w\", ",
                        pNew->metadata_columns[metadata_idx].name_length,
                        pNew->metadata_columns[metadata_idx].name);
        break;
      }
    }

  }
  if (hasCommandColumn) {
    sqlite3_str_appendf(createStr, " \"%w\" hidden, distance hidden, k hidden, mmr_lambda hidden) ", argv[2]);
  } else {
    sqlite3_str_appendall(createStr, " distance hidden, k hidden, mmr_lambda hidden) ");
  }
  if (pkColumnName) {
    sqlite3_str_appendall(createStr, "without rowid ");
  }
  zSql = sqlite3_str_finish(createStr);
  if (!zSql) {
    goto error;
  }
  rc = sqlite3_declare_vtab(db, zSql);
  sqlite3_free((void *)zSql);
  if (rc != SQLITE_OK) {
    *pzErr = sqlite3_mprintf(VEC_CONSTRUCTOR_ERROR
                             "could not declare virtual table, '%s'",
                             sqlite3_errmsg(db));
    goto error;
  }

  const char *schemaName = argv[1];
  const char *tableName = argv[2];

  pNew->db = db;
  pNew->pkIsText = pkColumnType == SQLITE_TEXT;
  pNew->schemaName = sqlite3_mprintf("%s", schemaName);
  if (!pNew->schemaName) {
    goto error;
  }
  pNew->tableName = sqlite3_mprintf("%s", tableName);
  if (!pNew->tableName) {
    goto error;
  }
  pNew->shadowRowidsName = sqlite3_mprintf("%s_rowids", tableName);
  if (!pNew->shadowRowidsName) {
    goto error;
  }
  pNew->shadowChunksName = sqlite3_mprintf("%s_chunks", tableName);
  if (!pNew->shadowChunksName) {
    goto error;
  }
  pNew->numVectorColumns = numVectorColumns;
  pNew->numPartitionColumns = numPartitionColumns;
  pNew->numAuxiliaryColumns = numAuxiliaryColumns;
  pNew->numMetadataColumns = numMetadataColumns;

  if (storageUri && !storageName) {
    *pzErr = sqlite3_mprintf(VEC_CONSTRUCTOR_ERROR "storage_uri needs storage=<provider>");
    goto error;
  }
  if (storageName) {
    // Vectors move to the provider, so only indexes that read vectors through
    // vec0 (flat scans, HNSW) can sit on top of it.
    for (int i = 0; i < numVectorColumns; i++) {
      enum Vec0IndexType t = pNew->vector_columns[i].index_type;
      if (t != VEC0_INDEX_TYPE_FLAT && t != VEC0_INDEX_TYPE_HNSW) {
        *pzErr = sqlite3_mprintf(VEC_CONSTRUCTOR_ERROR
                                 "storage=%s supports flat and hnsw vector columns", storageName);
        goto error;
      }
    }
    if (numMetadataColumns > 0 || numPartitionColumns > 0) {
      *pzErr = sqlite3_mprintf(VEC_CONSTRUCTOR_ERROR
                               "storage=%s does not support metadata or partition key columns",
                               storageName);
      goto error;
    }
    const sqlite3_vec_storage_methods *m = NULL;
#if SQLITE_VERSION_NUMBER >= 3044000
    if (sqlite3_libversion_number() >= 3044000) {
      char *zKey = sqlite3_mprintf(SQLITE_VEC_STORAGE_KEY_PREFIX "%s", storageName);
      if (!zKey) goto error;
      m = sqlite3_get_clientdata(db, zKey);
      sqlite3_free(zKey);
    }
#endif
    if (!m || m->iVersion < 1) {
      *pzErr = sqlite3_mprintf(VEC_CONSTRUCTOR_ERROR
                               "storage provider '%s' is not available on this connection "
                               "(the hybrid build, dist/vec0_lance, provides 'lance')",
                               storageName);
      goto error;
    }
    const char *azCol[VEC0_MAX_AUXILIARY_COLUMNS];
    int aType[VEC0_MAX_AUXILIARY_COLUMNS];
    const char *azVec[VEC0_MAX_VECTOR_COLUMNS];
    int aVecBytes[VEC0_MAX_VECTOR_COLUMNS], aVecDims[VEC0_MAX_VECTOR_COLUMNS];
    int nNames = 0, nVecNames = 0;
    for (; nNames < numAuxiliaryColumns; nNames++) {
      struct Vec0AuxiliaryColumnDefinition *c = &pNew->auxiliary_columns[nNames];
      azCol[nNames] = sqlite3_mprintf("%.*s", c->name_length, c->name);
      aType[nNames] = c->type;
      if (!azCol[nNames]) break;
    }
    for (; nVecNames < numVectorColumns; nVecNames++) {
      struct VectorColumnDefinition *c = &pNew->vector_columns[nVecNames];
      azVec[nVecNames] = sqlite3_mprintf("%.*s", c->name_length, c->name);
      aVecBytes[nVecNames] = (int)vector_column_byte_size(*c);
      aVecDims[nVecNames] = c->element_type == SQLITE_VEC_ELEMENT_TYPE_FLOAT32 ? (int)c->dimensions : 0;
      if (!azVec[nVecNames]) break;
    }
    int orc = (nNames < numAuxiliaryColumns || nVecNames < numVectorColumns)
                  ? SQLITE_NOMEM
                  : m->xOpen(db, pNew->schemaName, pNew->tableName, storageUri, isCreate,
                             numAuxiliaryColumns, azCol, aType,
                             numVectorColumns, azVec, aVecBytes, aVecDims,
                             &pNew->storageHandle, pzErr);
    for (int i = 0; i < nNames; i++) sqlite3_free((char *)azCol[i]);
    for (int i = 0; i < nVecNames; i++) sqlite3_free((char *)azVec[i]);
    if (orc != SQLITE_OK) {
      if (!*pzErr) *pzErr = sqlite3_mprintf(VEC_CONSTRUCTOR_ERROR "could not open storage '%s': %s",
                                            storageName, sqlite3_errstr(orc));
      goto error;
    }
    pNew->storage = m;
  }

  for (int i = 0; i < pNew->numVectorColumns; i++) {
    pNew->shadowVectorChunksNames[i] =
        sqlite3_mprintf("%s_vector_chunks%02d", tableName, i);
    if (!pNew->shadowVectorChunksNames[i]) {
      goto error;
    }
#if SQLITE_VEC_ENABLE_RESCORE
    if (pNew->vector_columns[i].index_type == VEC0_INDEX_TYPE_RESCORE) {
      pNew->shadowRescoreChunksNames[i] =
          sqlite3_mprintf("%s_rescore_chunks%02d", tableName, i);
      if (!pNew->shadowRescoreChunksNames[i]) {
        goto error;
      }
      pNew->shadowRescoreVectorsNames[i] =
          sqlite3_mprintf("%s_rescore_vectors%02d", tableName, i);
      if (!pNew->shadowRescoreVectorsNames[i]) {
        goto error;
      }
    }
#endif
#if SQLITE_VEC_ENABLE_DISKANN
    if (pNew->vector_columns[i].index_type == VEC0_INDEX_TYPE_DISKANN) {
      pNew->shadowVectorsNames[i] =
          sqlite3_mprintf("%s_vectors%02d", tableName, i);
      if (!pNew->shadowVectorsNames[i]) {
        goto error;
      }
      pNew->shadowDiskannNodesNames[i] =
          sqlite3_mprintf("%s_diskann_nodes%02d", tableName, i);
      if (!pNew->shadowDiskannNodesNames[i]) {
        goto error;
      }
    }
#endif
  }
#if SQLITE_VEC_EXPERIMENTAL_IVF_ENABLE
  for (int i = 0; i < pNew->numVectorColumns; i++) {
    if (pNew->vector_columns[i].index_type != VEC0_INDEX_TYPE_IVF) continue;
    pNew->shadowIvfCellsNames[i] =
        sqlite3_mprintf("%s_ivf_cells%02d", tableName, i);
    if (!pNew->shadowIvfCellsNames[i]) goto error;
  }
#endif
  for (int i = 0; i < pNew->numMetadataColumns; i++) {
    pNew->shadowMetadataChunksNames[i] =
        sqlite3_mprintf("%s_metadatachunks%02d", tableName, i);
    if (!pNew->shadowMetadataChunksNames[i]) {
      goto error;
    }
  }
  pNew->chunk_size = chunk_size;

  // if xCreate, then create the necessary shadow tables
  if (isCreate) {
    sqlite3_stmt *stmt;
    int rc;

    char * zCreateInfo = sqlite3_mprintf("CREATE TABLE "VEC0_SHADOW_INFO_NAME " (key text primary key, value any)", pNew->schemaName, pNew->tableName);
    if(!zCreateInfo) {
      goto error;
    }
    rc = sqlite3_prepare_v2(db, zCreateInfo, -1, &stmt, NULL);

    sqlite3_free((void *) zCreateInfo);
    if ((rc != SQLITE_OK) || (sqlite3_step(stmt) != SQLITE_DONE)) {
      // TODO(IMP)
      sqlite3_finalize(stmt);
      *pzErr = sqlite3_mprintf("Could not create '_info' shadow table: %s",
                               sqlite3_errmsg(db));
      goto error;
    }
    sqlite3_finalize(stmt);

    char * zSeedInfo = sqlite3_mprintf(
      "INSERT INTO "VEC0_SHADOW_INFO_NAME "(key, value) VALUES "
      "(?1, ?2), (?3, ?4), (?5, ?6), (?7, ?8) ",
      pNew->schemaName, pNew->tableName
    );
    if(!zSeedInfo) {
      goto error;
    }
    rc = sqlite3_prepare_v2(db, zSeedInfo, -1, &stmt, NULL);
    sqlite3_free((void *) zSeedInfo);
    if (rc != SQLITE_OK) {
      // TODO(IMP)
      sqlite3_finalize(stmt);
      *pzErr = sqlite3_mprintf("Could not seed '_info' shadow table: %s",
                               sqlite3_errmsg(db));
      goto error;
    }
    sqlite3_bind_text(stmt, 1, "CREATE_VERSION", -1, SQLITE_STATIC);
    sqlite3_bind_text(stmt, 2, SQLITE_VEC_VERSION, -1, SQLITE_STATIC);
    sqlite3_bind_text(stmt, 3, "CREATE_VERSION_MAJOR", -1, SQLITE_STATIC);
    sqlite3_bind_int(stmt, 4, SQLITE_VEC_VERSION_MAJOR);
    sqlite3_bind_text(stmt, 5, "CREATE_VERSION_MINOR", -1, SQLITE_STATIC);
    sqlite3_bind_int(stmt, 6, SQLITE_VEC_VERSION_MINOR);
    sqlite3_bind_text(stmt, 7, "CREATE_VERSION_PATCH", -1, SQLITE_STATIC);
    sqlite3_bind_int(stmt, 8, SQLITE_VEC_VERSION_PATCH);

    if(sqlite3_step(stmt) != SQLITE_DONE) {
      // TODO(IMP)
      sqlite3_finalize(stmt);
      *pzErr = sqlite3_mprintf("Could not seed '_info' shadow table: %s",
                               sqlite3_errmsg(db));
      goto error;
    }
    sqlite3_finalize(stmt);

#if SQLITE_VEC_ENABLE_DISKANN
    // Seed medoid entries for DiskANN-indexed columns
    for (int i = 0; i < pNew->numVectorColumns; i++) {
      if (pNew->vector_columns[i].index_type != VEC0_INDEX_TYPE_DISKANN) {
        continue;
      }
      char *key = sqlite3_mprintf("diskann_medoid_%02d", i);
      char *zInsert = sqlite3_mprintf(
          "INSERT INTO " VEC0_SHADOW_INFO_NAME "(key, value) VALUES (?1, ?2)",
          pNew->schemaName, pNew->tableName);
      rc = sqlite3_prepare_v2(db, zInsert, -1, &stmt, NULL);
      sqlite3_free(zInsert);
      if (rc != SQLITE_OK) {
        sqlite3_free(key);
        sqlite3_finalize(stmt);
        goto error;
      }
      sqlite3_bind_text(stmt, 1, key, -1, sqlite3_free);
      sqlite3_bind_null(stmt, 2);  // NULL means empty graph
      if (sqlite3_step(stmt) != SQLITE_DONE) {
        sqlite3_finalize(stmt);
        goto error;
      }
      sqlite3_finalize(stmt);
    }
#endif

    // create the _chunks shadow table
    char *zCreateShadowChunks = NULL;
    if(pNew->numPartitionColumns) {
      sqlite3_str * s = sqlite3_str_new(NULL);
      sqlite3_str_appendf(s, "CREATE TABLE " VEC0_SHADOW_CHUNKS_NAME "(", pNew->schemaName, pNew->tableName);
      sqlite3_str_appendall(s, "chunk_id INTEGER PRIMARY KEY AUTOINCREMENT," "size INTEGER NOT NULL,");
      sqlite3_str_appendall(s, "sequence_id integer,");
      for(int i = 0; i < pNew->numPartitionColumns;i++) {
        sqlite3_str_appendf(s, "partition%02d,", i);
      }
      sqlite3_str_appendall(s, "validity BLOB NOT NULL, rowids BLOB NOT NULL);");
      zCreateShadowChunks = sqlite3_str_finish(s);
    }else {
      zCreateShadowChunks = sqlite3_mprintf(VEC0_SHADOW_CHUNKS_CREATE,
                                          pNew->schemaName, pNew->tableName);
    }
    if (!zCreateShadowChunks) {
        goto error;
      }
    rc = sqlite3_prepare_v2(db, zCreateShadowChunks, -1, &stmt, 0);
    sqlite3_free((void *)zCreateShadowChunks);
    if ((rc != SQLITE_OK) || (sqlite3_step(stmt) != SQLITE_DONE)) {
      // IMP: V17740_01811
      sqlite3_finalize(stmt);
      *pzErr = sqlite3_mprintf("Could not create '_chunks' shadow table: %s",
                               sqlite3_errmsg(db));
      goto error;
    }
    sqlite3_finalize(stmt);

    // create the _rowids shadow table
    char *zCreateShadowRowids;
    if (pNew->pkIsText) {
      // adds a "text unique not null" constraint to the id column
      zCreateShadowRowids = sqlite3_mprintf(VEC0_SHADOW_ROWIDS_CREATE_PK_TEXT,
                                            pNew->schemaName, pNew->tableName);
    } else {
      zCreateShadowRowids = sqlite3_mprintf(VEC0_SHADOW_ROWIDS_CREATE_BASIC,
                                            pNew->schemaName, pNew->tableName);
    }
    if (!zCreateShadowRowids) {
      goto error;
    }
    rc = sqlite3_prepare_v2(db, zCreateShadowRowids, -1, &stmt, 0);
    sqlite3_free((void *)zCreateShadowRowids);
    if ((rc != SQLITE_OK) || (sqlite3_step(stmt) != SQLITE_DONE)) {
      // IMP: V11631_28470
      sqlite3_finalize(stmt);
      *pzErr = sqlite3_mprintf("Could not create '_rowids' shadow table: %s",
                               sqlite3_errmsg(db));
      goto error;
    }
    sqlite3_finalize(stmt);

    for (int i = 0; i < pNew->numVectorColumns; i++) {
      // Non-FLAT columns (rescore, IVF, DiskANN) don't use _vector_chunks
      if (!vec0_column_in_chunks(pNew, i))
        continue;
      char *zSql = sqlite3_mprintf(VEC0_SHADOW_VECTOR_N_CREATE,
                                   pNew->schemaName, pNew->tableName, i);
      if (!zSql) {
        goto error;
      }
      rc = sqlite3_prepare_v2(db, zSql, -1, &stmt, 0);
      sqlite3_free((void *)zSql);
      if ((rc != SQLITE_OK) || (sqlite3_step(stmt) != SQLITE_DONE)) {
        // IMP: V25919_09989
        sqlite3_finalize(stmt);
        *pzErr = sqlite3_mprintf(
            "Could not create '_vector_chunks%02d' shadow table: %s", i,
            sqlite3_errmsg(db));
        goto error;
      }
      sqlite3_finalize(stmt);
    }

#if SQLITE_VEC_ENABLE_RESCORE
    rc = rescore_create_tables(pNew, db, pzErr);
    if (rc != SQLITE_OK) {
      goto error;
    }
#endif

#if SQLITE_VEC_ENABLE_HNSW
    for (int i = 0; i < pNew->numVectorColumns; i++) {
      if (pNew->vector_columns[i].index_type != VEC0_INDEX_TYPE_HNSW) continue;
      rc = hnsw_create_shadow_tables(pNew, i);
      if (rc != SQLITE_OK) {
        *pzErr = sqlite3_mprintf("Could not create HNSW shadow tables for column %d", i);
        goto error;
      }
    }
#endif

#if SQLITE_VEC_EXPERIMENTAL_IVF_ENABLE
    // Create IVF shadow tables for IVF-indexed vector columns
    for (int i = 0; i < pNew->numVectorColumns; i++) {
      if (pNew->vector_columns[i].index_type != VEC0_INDEX_TYPE_IVF) continue;
      rc = ivf_create_shadow_tables(pNew, i);
      if (rc != SQLITE_OK) {
        *pzErr = sqlite3_mprintf("Could not create IVF shadow tables for column %d", i);
        goto error;
      }
    }
#endif

#if SQLITE_VEC_ENABLE_DISKANN
    // Create DiskANN shadow tables for indexed vector columns
    for (int i = 0; i < pNew->numVectorColumns; i++) {
      if (pNew->vector_columns[i].index_type != VEC0_INDEX_TYPE_DISKANN) {
        continue;
      }

      // Create _vectors{NN} table
      {
        char *zSql = sqlite3_mprintf(
            "CREATE TABLE " VEC0_SHADOW_VECTORS_N_NAME
            " (rowid INTEGER PRIMARY KEY, vector BLOB NOT NULL);",
            pNew->schemaName, pNew->tableName, i);
        if (!zSql) {
          goto error;
        }
        rc = sqlite3_prepare_v2(db, zSql, -1, &stmt, 0);
        sqlite3_free(zSql);
        if ((rc != SQLITE_OK) || (sqlite3_step(stmt) != SQLITE_DONE)) {
          sqlite3_finalize(stmt);
          *pzErr = sqlite3_mprintf(
              "Could not create '_vectors%02d' shadow table: %s", i,
              sqlite3_errmsg(db));
          goto error;
        }
        sqlite3_finalize(stmt);
      }

      // Create _diskann_nodes{NN} table
      {
        char *zSql = sqlite3_mprintf(
            "CREATE TABLE " VEC0_SHADOW_DISKANN_NODES_N_NAME " ("
            "rowid INTEGER PRIMARY KEY, "
            "neighbors_validity BLOB NOT NULL, "
            "neighbor_ids BLOB NOT NULL, "
            "neighbor_quantized_vectors BLOB NOT NULL"
            ");",
            pNew->schemaName, pNew->tableName, i);
        if (!zSql) {
          goto error;
        }
        rc = sqlite3_prepare_v2(db, zSql, -1, &stmt, 0);
        sqlite3_free(zSql);
        if ((rc != SQLITE_OK) || (sqlite3_step(stmt) != SQLITE_DONE)) {
          sqlite3_finalize(stmt);
          *pzErr = sqlite3_mprintf(
              "Could not create '_diskann_nodes%02d' shadow table: %s", i,
              sqlite3_errmsg(db));
          goto error;
        }
        sqlite3_finalize(stmt);
      }

      // Create _diskann_buffer{NN} table (for batched inserts)
      {
        char *zSql = sqlite3_mprintf(
            "CREATE TABLE " VEC0_SHADOW_DISKANN_BUFFER_N_NAME " ("
            "rowid INTEGER PRIMARY KEY, "
            "vector BLOB NOT NULL"
            ");",
            pNew->schemaName, pNew->tableName, i);
        if (!zSql) {
          goto error;
        }
        rc = sqlite3_prepare_v2(db, zSql, -1, &stmt, 0);
        sqlite3_free(zSql);
        if ((rc != SQLITE_OK) || (sqlite3_step(stmt) != SQLITE_DONE)) {
          sqlite3_finalize(stmt);
          *pzErr = sqlite3_mprintf(
              "Could not create '_diskann_buffer%02d' shadow table: %s", i,
              sqlite3_errmsg(db));
          goto error;
        }
        sqlite3_finalize(stmt);
      }
    }
#endif

    // See SHADOW_TABLE_ROWID_QUIRK in vec0_new_chunk() — same "rowid PRIMARY KEY"
    // without INTEGER type issue applies here.
    for (int i = 0; i < pNew->numMetadataColumns; i++) {
      char *zSql = sqlite3_mprintf("CREATE TABLE " VEC0_SHADOW_METADATA_N_NAME "(rowid PRIMARY KEY, data BLOB NOT NULL);",
                                   pNew->schemaName, pNew->tableName, i);
      if (!zSql) {
        goto error;
      }
      rc = sqlite3_prepare_v2(db, zSql, -1, &stmt, 0);
      sqlite3_free((void *)zSql);
      if ((rc != SQLITE_OK) || (sqlite3_step(stmt) != SQLITE_DONE)) {
        sqlite3_finalize(stmt);
        *pzErr = sqlite3_mprintf(
            "Could not create '_metata_chunks%02d' shadow table: %s", i,
            sqlite3_errmsg(db));
        goto error;
      }
      sqlite3_finalize(stmt);

      if(pNew->metadata_columns[i].kind == VEC0_METADATA_COLUMN_KIND_TEXT) {
        char *zSql = sqlite3_mprintf("CREATE TABLE " VEC0_SHADOW_METADATA_TEXT_DATA_NAME "(rowid PRIMARY KEY, data TEXT);",
                                   pNew->schemaName, pNew->tableName, i);
        if (!zSql) {
          goto error;
        }
        rc = sqlite3_prepare_v2(db, zSql, -1, &stmt, 0);
        sqlite3_free((void *)zSql);
        if ((rc != SQLITE_OK) || (sqlite3_step(stmt) != SQLITE_DONE)) {
          sqlite3_finalize(stmt);
          *pzErr = sqlite3_mprintf(
              "Could not create '_metadatatext%02d' shadow table: %s", i,
              sqlite3_errmsg(db));
          goto error;
        }
        sqlite3_finalize(stmt);

      }
    }

    if(pNew->numAuxiliaryColumns > 0 && !pNew->storage) {
      sqlite3_stmt * stmt;
      sqlite3_str * s = sqlite3_str_new(NULL);
      sqlite3_str_appendf(s, "CREATE TABLE " VEC0_SHADOW_AUXILIARY_NAME "( rowid integer PRIMARY KEY ", pNew->schemaName, pNew->tableName);
      for(int i = 0; i < pNew->numAuxiliaryColumns; i++) {
        sqlite3_str_appendf(s, ", value%02d", i);
      }
      sqlite3_str_appendall(s, ")");
      char *zSql = sqlite3_str_finish(s);
      if(!zSql) {
        goto error;
      }
      rc = sqlite3_prepare_v2(db, zSql, -1, &stmt, NULL);
      if ((rc != SQLITE_OK) || (sqlite3_step(stmt) != SQLITE_DONE)) {
        sqlite3_finalize(stmt);
        *pzErr = sqlite3_mprintf(
            "Could not create auxiliary shadow table: %s",
            sqlite3_errmsg(db));

        goto error;
      }
      sqlite3_finalize(stmt);
    }
  }

  sqlite3_free(storageName);
  sqlite3_free(storageUri);
  *ppVtab = (sqlite3_vtab *)pNew;
  return SQLITE_OK;

error:
  sqlite3_free(storageName);
  sqlite3_free(storageUri);
  vec0_free(pNew);
  sqlite3_free(pNew);
  return SQLITE_ERROR;
}

static int vec0Create(sqlite3 *db, void *pAux, int argc,
                      const char *const *argv, sqlite3_vtab **ppVtab,
                      char **pzErr) {
  return vec0_init(db, pAux, argc, argv, ppVtab, pzErr, true);
}
static int vec0Connect(sqlite3 *db, void *pAux, int argc,
                       const char *const *argv, sqlite3_vtab **ppVtab,
                       char **pzErr) {
  return vec0_init(db, pAux, argc, argv, ppVtab, pzErr, false);
}

static int vec0Disconnect(sqlite3_vtab *pVtab) {
  vec0_vtab *p = (vec0_vtab *)pVtab;
  vec0_free(p);
  sqlite3_free(p);
  return SQLITE_OK;
}
static int vec0Destroy(sqlite3_vtab *pVtab) {
  vec0_vtab *p = (vec0_vtab *)pVtab;
  sqlite3_stmt *stmt = NULL;
  int rc;
  const char *zSql;

  // Free up any sqlite3_stmt, otherwise DROPs on those tables will fail
  vec0_free_resources(p);

  // Let providers refuse DROP before schema changes can invalidate pending state.
  if (p->storage) {
    rc = vec0_storage_error(p, p->storage->xDestroy(p->storageHandle));
    if (rc != SQLITE_OK) goto done;
  }

  // TODO(test) later: can't evidence-of here, bc always gives "SQL logic error" instead of
  // provided error
  zSql = sqlite3_mprintf("DROP TABLE " VEC0_SHADOW_CHUNKS_NAME, p->schemaName,
                         p->tableName);
  rc = sqlite3_prepare_v2(p->db, zSql, -1, &stmt, 0);
  sqlite3_free((void *)zSql);
  if ((rc != SQLITE_OK) || (sqlite3_step(stmt) != SQLITE_DONE)) {
    rc = SQLITE_ERROR;
    vtab_set_error(pVtab, "could not drop chunks shadow table");
    goto done;
  }
  sqlite3_finalize(stmt);

  zSql = sqlite3_mprintf("DROP TABLE " VEC0_SHADOW_INFO_NAME, p->schemaName,
                         p->tableName);
  rc = sqlite3_prepare_v2(p->db, zSql, -1, &stmt, 0);
  sqlite3_free((void *)zSql);
  if ((rc != SQLITE_OK) || (sqlite3_step(stmt) != SQLITE_DONE)) {
    rc = SQLITE_ERROR;
    vtab_set_error(pVtab, "could not drop info shadow table");
    goto done;
  }
  sqlite3_finalize(stmt);

  zSql = sqlite3_mprintf("DROP TABLE " VEC0_SHADOW_ROWIDS_NAME, p->schemaName,
                         p->tableName);
  rc = sqlite3_prepare_v2(p->db, zSql, -1, &stmt, 0);
  sqlite3_free((void *)zSql);
  if ((rc != SQLITE_OK) || (sqlite3_step(stmt) != SQLITE_DONE)) {
    rc = SQLITE_ERROR;
    goto done;
  }
  sqlite3_finalize(stmt);

  for (int i = 0; i < p->numVectorColumns; i++) {
#if SQLITE_VEC_ENABLE_DISKANN
    if (p->vector_columns[i].index_type == VEC0_INDEX_TYPE_DISKANN) {
      // Drop DiskANN shadow tables
      zSql = sqlite3_mprintf("DROP TABLE IF EXISTS " VEC0_SHADOW_VECTORS_N_NAME,
                             p->schemaName, p->tableName, i);
      if (zSql) {
        rc = sqlite3_prepare_v2(p->db, zSql, -1, &stmt, 0);
        sqlite3_free((void *)zSql);
        if ((rc != SQLITE_OK) || (sqlite3_step(stmt) != SQLITE_DONE)) {
          rc = SQLITE_ERROR;
          goto done;
        }
        sqlite3_finalize(stmt);
      }
      zSql = sqlite3_mprintf("DROP TABLE IF EXISTS " VEC0_SHADOW_DISKANN_NODES_N_NAME,
                             p->schemaName, p->tableName, i);
      if (zSql) {
        rc = sqlite3_prepare_v2(p->db, zSql, -1, &stmt, 0);
        sqlite3_free((void *)zSql);
        if ((rc != SQLITE_OK) || (sqlite3_step(stmt) != SQLITE_DONE)) {
          rc = SQLITE_ERROR;
          goto done;
        }
        sqlite3_finalize(stmt);
      }
      zSql = sqlite3_mprintf("DROP TABLE IF EXISTS " VEC0_SHADOW_DISKANN_BUFFER_N_NAME,
                             p->schemaName, p->tableName, i);
      if (zSql) {
        rc = sqlite3_prepare_v2(p->db, zSql, -1, &stmt, 0);
        sqlite3_free((void *)zSql);
        if ((rc != SQLITE_OK) || (sqlite3_step(stmt) != SQLITE_DONE)) {
          rc = SQLITE_ERROR;
          goto done;
        }
        sqlite3_finalize(stmt);
      }
      continue;
    }
#endif
    // Non-FLAT columns (rescore, IVF, DiskANN) don't use _vector_chunks
    if (!vec0_column_in_chunks(p, i))
      continue;
    zSql = sqlite3_mprintf("DROP TABLE \"%w\".\"%w\"", p->schemaName,
                           p->shadowVectorChunksNames[i]);
    rc = sqlite3_prepare_v2(p->db, zSql, -1, &stmt, 0);
    sqlite3_free((void *)zSql);
    if ((rc != SQLITE_OK) || (sqlite3_step(stmt) != SQLITE_DONE)) {
      rc = SQLITE_ERROR;
      goto done;
    }
    sqlite3_finalize(stmt);
  }

  stmt = NULL;
#if SQLITE_VEC_ENABLE_RESCORE
  rc = rescore_drop_tables(p);
  if (rc != SQLITE_OK) {
    goto done;
  }
#endif

#if SQLITE_VEC_ENABLE_HNSW
  for (int i = 0; i < p->numVectorColumns; i++) {
    if (p->vector_columns[i].index_type != VEC0_INDEX_TYPE_HNSW) continue;
    hnsw_drop_shadow_tables(p, i);
  }
#endif

#if SQLITE_VEC_EXPERIMENTAL_IVF_ENABLE
  // Drop IVF shadow tables
  for (int i = 0; i < p->numVectorColumns; i++) {
    if (p->vector_columns[i].index_type != VEC0_INDEX_TYPE_IVF) continue;
    ivf_drop_shadow_tables(p, i);
  }
#endif

  if (!p->storage && p->numAuxiliaryColumns > 0) {
    zSql = sqlite3_mprintf("DROP TABLE " VEC0_SHADOW_AUXILIARY_NAME, p->schemaName, p->tableName);
    rc = sqlite3_prepare_v2(p->db, zSql, -1, &stmt, 0);
    sqlite3_free((void *)zSql);
    if ((rc != SQLITE_OK) || (sqlite3_step(stmt) != SQLITE_DONE)) {
      rc = SQLITE_ERROR;
      goto done;
    }
    sqlite3_finalize(stmt);
  }


  for (int i = 0; i < p->numMetadataColumns; i++) {
    zSql = sqlite3_mprintf("DROP TABLE " VEC0_SHADOW_METADATA_N_NAME, p->schemaName,p->tableName, i);
    rc = sqlite3_prepare_v2(p->db, zSql, -1, &stmt, 0);
    sqlite3_free((void *)zSql);
    if ((rc != SQLITE_OK) || (sqlite3_step(stmt) != SQLITE_DONE)) {
      rc = SQLITE_ERROR;
      goto done;
    }
    sqlite3_finalize(stmt);

    if(p->metadata_columns[i].kind == VEC0_METADATA_COLUMN_KIND_TEXT) {
      zSql = sqlite3_mprintf("DROP TABLE " VEC0_SHADOW_METADATA_TEXT_DATA_NAME, p->schemaName,p->tableName, i);
      rc = sqlite3_prepare_v2(p->db, zSql, -1, &stmt, 0);
      sqlite3_free((void *)zSql);
      if ((rc != SQLITE_OK) || (sqlite3_step(stmt) != SQLITE_DONE)) {
        rc = SQLITE_ERROR;
        goto done;
      }
      sqlite3_finalize(stmt);
    }
  }

  stmt = NULL;
  rc = SQLITE_OK;

done:
  sqlite3_finalize(stmt);
  // SQLite retains the vtab when xDestroy fails; its state must remain usable.
  if (rc == SQLITE_OK) {
    vec0_free(p);
    sqlite3_free(p);
  }
  return rc;
}

static int vec0Open(sqlite3_vtab *p, sqlite3_vtab_cursor **ppCursor) {
  UNUSED_PARAMETER(p);
  vec0_cursor *pCur;
  pCur = sqlite3_malloc(sizeof(*pCur));
  if (pCur == 0)
    return SQLITE_NOMEM;
  memset(pCur, 0, sizeof(*pCur));
  *ppCursor = &pCur->base;
  return SQLITE_OK;
}

static int vec0Close(sqlite3_vtab_cursor *cur) {
  vec0_cursor *pCur = (vec0_cursor *)cur;
  vec0_cursor_clear(pCur);
  sqlite3_free(pCur);
  return SQLITE_OK;
}

// All the different type of "values" provided to argv/argc in vec0Filter.
// These enums denote the use and purpose of all of them.
typedef enum  {
  // If any values are updated, please update the ARCHITECTURE.md docs accordingly!

  // ~~~ KNN QUERIES ~~~ //
  VEC0_IDXSTR_KIND_KNN_MATCH = '{',
  VEC0_IDXSTR_KIND_KNN_K = '}',
  // argv[i] is the OFFSET of a LIMIT-sourced KNN query (`LIMIT n OFFSET m`):
  // vec0 fetches n + m rows and SQLite skips the first m (omit = 0).
  VEC0_IDXSTR_KIND_KNN_OFFSET = '(',
  VEC0_IDXSTR_KIND_KNN_ROWID_IN = '[',
  // argv[i] is a constraint on a PARTITON KEY column in a KNN query
  // 
  VEC0_IDXSTR_KIND_KNN_PARTITON_CONSTRAINT = ']',

  // argv[i] is a constraint on the distance column in a KNN query
  VEC0_IDXSTR_KIND_KNN_DISTANCE_CONSTRAINT = '*',

  // ~~~ POINT QUERIES ~~~ //
  VEC0_IDXSTR_KIND_POINT_ID = '!',

  // ~~~ ??? ~~~ //
  VEC0_IDXSTR_KIND_METADATA_CONSTRAINT = '&',
  VEC0_IDXSTR_KIND_KNN_MMR_LAMBDA = '#',
} vec0_idxstr_kind;

// The different SQLITE_INDEX_CONSTRAINT values that vec0 partition key columns
// support, but as characters that fit nicely in idxstr.
typedef enum  {
  // If any values are updated, please update the ARCHITECTURE.md docs accordingly!

  // Equality constraint on a PARTITON KEY column, ex `user_id = 123`
  VEC0_PARTITION_OPERATOR_EQ = 'a',
  
  // "Greater than" constraint on a PARTITON KEY column, ex `year > 2024`
  VEC0_PARTITION_OPERATOR_GT = 'b',
  
  // "Less than or equal to" constraint on a PARTITON KEY column, ex `year <= 2024`
  VEC0_PARTITION_OPERATOR_LE = 'c',

  // "Less than" constraint on a PARTITON KEY column, ex `year < 2024`
  VEC0_PARTITION_OPERATOR_LT = 'd',
  
  // "Greater than or equal to" constraint on a PARTITON KEY column, ex `year >= 2024`
  VEC0_PARTITION_OPERATOR_GE = 'e',
  
  // "Not equal to" constraint on a PARTITON KEY column, ex `year != 2024`
  VEC0_PARTITION_OPERATOR_NE = 'f',
} vec0_partition_operator;
typedef enum  {
  VEC0_METADATA_OPERATOR_EQ = 'a',
  VEC0_METADATA_OPERATOR_GT = 'b',
  VEC0_METADATA_OPERATOR_LE = 'c',
  VEC0_METADATA_OPERATOR_LT = 'd',
  VEC0_METADATA_OPERATOR_GE = 'e',
  VEC0_METADATA_OPERATOR_NE = 'f',
  VEC0_METADATA_OPERATOR_IN = 'g',
} vec0_metadata_operator;


typedef enum {

  VEC0_DISTANCE_CONSTRAINT_GT = 'a',
  VEC0_DISTANCE_CONSTRAINT_GE = 'b',
  VEC0_DISTANCE_CONSTRAINT_LT = 'c',
  VEC0_DISTANCE_CONSTRAINT_LE = 'd',
} vec0_distance_constraint_operator;

/**
 * Whether constraint i is an argument of a KNN query: a MATCH on a vector
 * column, or `k` or a hidden parameter column declared after it. Filters on
 * rowid, partition key, metadata and distance columns are not;
 * ARCHITECTURE.md's idxStr section says why they are left to the plan.
 */
static int vec0_constraint_is_knn_argument(vec0_vtab *p,
                                           sqlite3_index_info *pIdxInfo,
                                           int i) {
  int iColumn = pIdxInfo->aConstraint[i].iColumn;
  int op = pIdxInfo->aConstraint[i].op;
  if (op == SQLITE_INDEX_CONSTRAINT_MATCH) {
    return vec0_column_idx_is_vector(p, iColumn);
  }
  if (op == SQLITE_INDEX_CONSTRAINT_LIMIT ||
      op == SQLITE_INDEX_CONSTRAINT_OFFSET) {
    return 1;
  }
  return op == SQLITE_INDEX_CONSTRAINT_EQ && iColumn >= vec0_column_k_idx(p);
}

/**
 * @brief Test a single distance value against every distance constraint
 *        encoded in idxStr.
 *
 * vec0BestIndex() sets aConstraintUsage[].omit = 1 on `distance` constraints,
 * which promises SQLite that the vtab applies them itself. Every KNN
 * implementation (FLAT, rescore, IVF, ...) MUST therefore route its candidate
 * distances through this predicate, otherwise the WHERE clause is silently
 * dropped from the query plan and never re-checked by SQLite.
 *
 * @param distance - candidate distance to test
 * @param idxStr - the xBestIndex/xFilter idxStr
 * @param argc, argv - xFilter arguments, parallel to the idxStr entries
 * @returns 1 if the distance satisfies all constraints, 0 otherwise.
 */
static int vec0_distance_constraints_satisfied(f32 distance, const char *idxStr,
                                               int argc,
                                               sqlite3_value **argv) {
  for (int i = 0; i < argc; i++) {
    int idx = 1 + (i * 4);
    if (idxStr[idx + 0] != VEC0_IDXSTR_KIND_KNN_DISTANCE_CONSTRAINT) {
      continue;
    }
    // TODO casts f64 to f32, is that a problem?
    f32 target = (f32)sqlite3_value_double(argv[i]);
    switch ((vec0_distance_constraint_operator)idxStr[idx + 1]) {
    case VEC0_DISTANCE_CONSTRAINT_GT:
      if (!(distance > target))
        return 0;
      break;
    case VEC0_DISTANCE_CONSTRAINT_GE:
      if (!(distance >= target))
        return 0;
      break;
    case VEC0_DISTANCE_CONSTRAINT_LT:
      if (!(distance < target))
        return 0;
      break;
    case VEC0_DISTANCE_CONSTRAINT_LE:
      if (!(distance <= target))
        return 0;
      break;
    }
  }
  return 1;
}

/**
 * Drop the rows of an index backend's KNN result that fail the query's
 * `distance` constraints, keeping the order. vec0BestIndex() omits these
 * constraints from the plan, so SQLite will not re-check them.
 */
static void vec0_knn_apply_distance_constraints(struct vec0_query_knn_data *knn,
                                                const char *idxStr, int argc,
                                                sqlite3_value **argv) {
  i64 kept = 0;
  for (i64 j = 0; j < knn->k_used; j++) {
    if (!vec0_distance_constraints_satisfied(knn->distances[j], idxStr, argc, argv))
      continue;
    knn->rowids[kept] = knn->rowids[j];
    knn->distances[kept] = knn->distances[j];
    kept++;
  }
  knn->k = kept;
  knn->k_used = kept;
}

static int vec0BestIndex(sqlite3_vtab *pVTab, sqlite3_index_info *pIdxInfo) {
  vec0_vtab *p = (vec0_vtab *)pVTab;
  /**
   * Possible query plans are:
   * 1. KNN when:
   *    a) An `MATCH` op on vector column
   *    b) ORDER BY on distance column
   *    c) LIMIT
   *    d) rowid in (...) OPTIONAL
   * 2. Point when:
   *    a) An `EQ` op on rowid column
   * 3. else: fullscan
   *
   */
  int iMatchTerm = -1;
  int iMatchVectorTerm = -1;
  int iLimitTerm = -1;
  int iOffsetTerm = -1;
  int iRowidTerm = -1;
  int iKTerm = -1;
  int iMmrLambdaTerm = -1;
  int iRowidInTerm = -1;
  int hasAuxConstraint = 0;

#ifdef SQLITE_VEC_DEBUG
  printf("pIdxInfo->nOrderBy=%d, pIdxInfo->nConstraint=%d\n", pIdxInfo->nOrderBy, pIdxInfo->nConstraint);
#endif

  // When a KNN argument reads another table of the query, SQLite also weighs
  // plans that scan vec0 before that table, in which the argument is unusable.
  // Such a plan is refused, the way json_each refuses a plan that cannot supply
  // its argument, so the planner takes one that gives vec0 every argument.
  for (int i = 0; i < pIdxInfo->nConstraint; i++) {
    if (pIdxInfo->aConstraint[i].op == SQLITE_INDEX_CONSTRAINT_MATCH &&
        vec0_column_idx_is_vector(p, pIdxInfo->aConstraint[i].iColumn)) {
      for (int j = 0; j < pIdxInfo->nConstraint; j++) {
        if (!pIdxInfo->aConstraint[j].usable &&
            vec0_constraint_is_knn_argument(p, pIdxInfo, j)) {
          return SQLITE_CONSTRAINT;
        }
      }
      break;
    }
  }

  for (int i = 0; i < pIdxInfo->nConstraint; i++) {
    u8 vtabIn = 0;

#if COMPILER_SUPPORTS_VTAB_IN
    if (sqlite3_libversion_number() >= 3038000) {
      vtabIn = sqlite3_vtab_in(pIdxInfo, i, -1);
    }
#endif

#ifdef SQLITE_VEC_DEBUG
    printf("xBestIndex [%d] usable=%d iColumn=%d op=%d vtabin=%d\n", i,
           pIdxInfo->aConstraint[i].usable, pIdxInfo->aConstraint[i].iColumn,
           pIdxInfo->aConstraint[i].op, vtabIn);
#endif
    if (!pIdxInfo->aConstraint[i].usable)
      continue;

    int iColumn = pIdxInfo->aConstraint[i].iColumn;
    int op = pIdxInfo->aConstraint[i].op;

    if (op == SQLITE_INDEX_CONSTRAINT_LIMIT) {
      iLimitTerm = i;
    }
    if (op == SQLITE_INDEX_CONSTRAINT_OFFSET) {
      iOffsetTerm = i;
    }
    if (op == SQLITE_INDEX_CONSTRAINT_MATCH &&
        vec0_column_idx_is_vector(p, iColumn)) {
      if (iMatchTerm > -1) {
        vtab_set_error(
            pVTab, "only 1 MATCH operator is allowed in a single vec0 query");
        return SQLITE_ERROR;
      }
      iMatchTerm = i;
      iMatchVectorTerm = vec0_column_idx_to_vector_idx(p, iColumn);
    }
    if (op == SQLITE_INDEX_CONSTRAINT_EQ && iColumn == VEC0_COLUMN_ID) {
      if (vtabIn) {
        if (iRowidInTerm != -1) {
          vtab_set_error(pVTab, "only 1 'rowid in (..)' operator is allowed in "
                                "a single vec0 query");
          return SQLITE_ERROR;
        }
        iRowidInTerm = i;

      } else {
        iRowidTerm = i;
      }
    }
    if (op == SQLITE_INDEX_CONSTRAINT_EQ && iColumn == vec0_column_k_idx(p)) {
      iKTerm = i;
    }
    if (op == SQLITE_INDEX_CONSTRAINT_EQ && iColumn == vec0_column_mmr_lambda_idx(p)) {
      iMmrLambdaTerm = i;
    }
    if(
      (op != SQLITE_INDEX_CONSTRAINT_LIMIT && op != SQLITE_INDEX_CONSTRAINT_OFFSET)
      && vec0_column_idx_is_auxiliary(p, iColumn)) {
        hasAuxConstraint = 1;
      }
  }

  sqlite3_str *idxStr = sqlite3_str_new(NULL);
  int rc;

  if (iMatchTerm >= 0) {
    if (iLimitTerm < 0 && iKTerm < 0) {
      vtab_set_error(
          pVTab,
          "A LIMIT or 'k = ?' constraint is required on vec0 knn queries.");
      rc = SQLITE_ERROR;
      goto done;
    }
    if (iLimitTerm >= 0 && iKTerm >= 0) {
      vtab_set_error(pVTab, "Only LIMIT or 'k =?' can be provided, not both");
      rc = SQLITE_ERROR;
      goto done;
    }

    if (pIdxInfo->nOrderBy) {
      if (pIdxInfo->nOrderBy > 1) {
        vtab_set_error(pVTab, "Only a single 'ORDER BY distance' clause is "
                              "allowed on vec0 KNN queries");
        rc = SQLITE_ERROR;
      goto done;
      }
      if (pIdxInfo->aOrderBy[0].iColumn != vec0_column_distance_idx(p)) {
        vtab_set_error(pVTab,
                       "Only a single 'ORDER BY distance' clause is allowed on "
                       "vec0 KNN queries, not on other columns");
        rc = SQLITE_ERROR;
      goto done;
      }
      if (pIdxInfo->aOrderBy[0].desc) {
        vtab_set_error(
            pVTab, "Only ascending in ORDER BY distance clause is supported, "
                   "DESC is not supported yet.");
        rc = SQLITE_ERROR;
      goto done;
      }
    }

    if(hasAuxConstraint) {
      // IMP: V25623_09693
      vtab_set_error(pVTab, "An illegal WHERE constraint was provided on a vec0 auxiliary column in a KNN query.");
      rc = SQLITE_ERROR;
      goto done;
    }

    sqlite3_str_appendchar(idxStr, 1, VEC0_QUERY_PLAN_KNN);

    int argvIndex = 1;
    pIdxInfo->aConstraintUsage[iMatchTerm].argvIndex = argvIndex++;
    pIdxInfo->aConstraintUsage[iMatchTerm].omit = 1;
    sqlite3_str_appendchar(idxStr, 1, VEC0_IDXSTR_KIND_KNN_MATCH);
    sqlite3_str_appendchar(idxStr, 3, '_');

    if (iLimitTerm >= 0) {
      pIdxInfo->aConstraintUsage[iLimitTerm].argvIndex = argvIndex++;
      pIdxInfo->aConstraintUsage[iLimitTerm].omit = 1;
    } else {
      pIdxInfo->aConstraintUsage[iKTerm].argvIndex = argvIndex++;
      pIdxInfo->aConstraintUsage[iKTerm].omit = 1;
    }
    sqlite3_str_appendchar(idxStr, 1, VEC0_IDXSTR_KIND_KNN_K);
    sqlite3_str_appendchar(idxStr, 3, '_');

    if (iLimitTerm >= 0 && iOffsetTerm >= 0) {
      // Pass the OFFSET value to xFilter so it can fetch LIMIT + OFFSET
      // rows; omit = 0 keeps SQLite applying the OFFSET itself.
      pIdxInfo->aConstraintUsage[iOffsetTerm].argvIndex = argvIndex++;
      pIdxInfo->aConstraintUsage[iOffsetTerm].omit = 0;
      sqlite3_str_appendchar(idxStr, 1, VEC0_IDXSTR_KIND_KNN_OFFSET);
      sqlite3_str_appendchar(idxStr, 3, '_');
    }

#if COMPILER_SUPPORTS_VTAB_IN
    if (iRowidInTerm >= 0) {
      // already validated as  >= SQLite 3.38 bc iRowidInTerm is only >= 0 when
      // vtabIn == 1
      sqlite3_vtab_in(pIdxInfo, iRowidInTerm, 1);
      pIdxInfo->aConstraintUsage[iRowidInTerm].argvIndex = argvIndex++;
      pIdxInfo->aConstraintUsage[iRowidInTerm].omit = 1;
      sqlite3_str_appendchar(idxStr, 1, VEC0_IDXSTR_KIND_KNN_ROWID_IN);
      sqlite3_str_appendchar(idxStr, 3, '_');
    }
#endif

    // find any PARTITION KEY column constraints
    for (int i = 0; i < pIdxInfo->nConstraint; i++) {
      if (!pIdxInfo->aConstraint[i].usable)
        continue;

      int iColumn = pIdxInfo->aConstraint[i].iColumn;
      int op = pIdxInfo->aConstraint[i].op;
      if(op == SQLITE_INDEX_CONSTRAINT_LIMIT || op == SQLITE_INDEX_CONSTRAINT_OFFSET) {
        continue;
      }
      if(!vec0_column_idx_is_partition(p, iColumn)) {
        continue;
      }

      int partition_idx = vec0_column_idx_to_partition_idx(p, iColumn);
      char value = 0;

      switch(op) {
        case SQLITE_INDEX_CONSTRAINT_EQ: {
          value = VEC0_PARTITION_OPERATOR_EQ;
          break;
        }
        case SQLITE_INDEX_CONSTRAINT_GT: {
          value = VEC0_PARTITION_OPERATOR_GT;
          break;
        }
        case SQLITE_INDEX_CONSTRAINT_LE: {
          value = VEC0_PARTITION_OPERATOR_LE;
          break;
        }
        case SQLITE_INDEX_CONSTRAINT_LT: {
          value = VEC0_PARTITION_OPERATOR_LT;
          break;
        }
        case SQLITE_INDEX_CONSTRAINT_GE: {
          value = VEC0_PARTITION_OPERATOR_GE;
          break;
        }
        case SQLITE_INDEX_CONSTRAINT_NE: {
          value = VEC0_PARTITION_OPERATOR_NE;
          break;
        }
      }

      if(value) {
        pIdxInfo->aConstraintUsage[i].argvIndex = argvIndex++;
        pIdxInfo->aConstraintUsage[i].omit = 1;
        sqlite3_str_appendchar(idxStr, 1, VEC0_IDXSTR_KIND_KNN_PARTITON_CONSTRAINT);
        sqlite3_str_appendchar(idxStr, 1, 'A' + partition_idx);
        sqlite3_str_appendchar(idxStr, 1, value);
        sqlite3_str_appendchar(idxStr, 1, '_');
      }

    }

    // find any metadata column constraints
    for (int i = 0; i < pIdxInfo->nConstraint; i++) {
      if (!pIdxInfo->aConstraint[i].usable)
        continue;

      int iColumn = pIdxInfo->aConstraint[i].iColumn;
      int op = pIdxInfo->aConstraint[i].op;
      if(op == SQLITE_INDEX_CONSTRAINT_LIMIT || op == SQLITE_INDEX_CONSTRAINT_OFFSET) {
        continue;
      }
      if(!vec0_column_idx_is_metadata(p, iColumn)) {
        continue;
      }

      int metadata_idx = vec0_column_idx_to_metadata_idx(p, iColumn);
      char value = 0;

      switch(op) {
        case SQLITE_INDEX_CONSTRAINT_EQ: {
          int vtabIn = 0;
          #if COMPILER_SUPPORTS_VTAB_IN
          if (sqlite3_libversion_number() >= 3038000) {
            vtabIn = sqlite3_vtab_in(pIdxInfo, i, -1);
          }
          if(vtabIn) {
            switch(p->metadata_columns[metadata_idx].kind) {
              case VEC0_METADATA_COLUMN_KIND_FLOAT:
              case VEC0_METADATA_COLUMN_KIND_BOOLEAN: {
                // IMP: V15248_32086
                rc = SQLITE_ERROR;
                vtab_set_error(pVTab, "'xxx in (...)' is only available on INTEGER or TEXT metadata columns.");
                goto done;
                break;
              }
              case VEC0_METADATA_COLUMN_KIND_INTEGER:
              case VEC0_METADATA_COLUMN_KIND_TEXT: {
                break;
              }
            }
            value = VEC0_METADATA_OPERATOR_IN;
            sqlite3_vtab_in(pIdxInfo, i, 1);
          }else
          #endif
           {
            value = VEC0_PARTITION_OPERATOR_EQ;
          }
          break;
        }
        case SQLITE_INDEX_CONSTRAINT_GT: {
          value = VEC0_METADATA_OPERATOR_GT;
          break;
        }
        case SQLITE_INDEX_CONSTRAINT_LE: {
          value = VEC0_METADATA_OPERATOR_LE;
          break;
        }
        case SQLITE_INDEX_CONSTRAINT_LT: {
          value = VEC0_METADATA_OPERATOR_LT;
          break;
        }
        case SQLITE_INDEX_CONSTRAINT_GE: {
          value = VEC0_METADATA_OPERATOR_GE;
          break;
        }
        case SQLITE_INDEX_CONSTRAINT_NE: {
          value = VEC0_METADATA_OPERATOR_NE;
          break;
        }
        default: {
          // IMP: V16511_00582
          rc = SQLITE_ERROR;
          vtab_set_error(pVTab,
          "An illegal WHERE constraint was provided on a vec0 metadata column in a KNN query. "
          "Only one of EQUALS, GREATER_THAN, LESS_THAN_OR_EQUAL, LESS_THAN, GREATER_THAN_OR_EQUAL, NOT_EQUALS is allowed."
          );
          goto done;
        }
      }

      if(p->metadata_columns[metadata_idx].kind == VEC0_METADATA_COLUMN_KIND_BOOLEAN) {
        if(!(value == VEC0_METADATA_OPERATOR_EQ || value == VEC0_METADATA_OPERATOR_NE)) {
          // IMP: V10145_26984
          rc = SQLITE_ERROR;
          vtab_set_error(pVTab, "ONLY EQUALS (=) or NOT_EQUALS (!=) operators are allowed on boolean metadata columns.");
          goto done;
        }
      }

      pIdxInfo->aConstraintUsage[i].argvIndex = argvIndex++;
      pIdxInfo->aConstraintUsage[i].omit = 1;
      sqlite3_str_appendchar(idxStr, 1, VEC0_IDXSTR_KIND_METADATA_CONSTRAINT);
      sqlite3_str_appendchar(idxStr, 1, 'A' + metadata_idx);
      sqlite3_str_appendchar(idxStr, 1, value);
      sqlite3_str_appendchar(idxStr, 1, '_');

    }

    // find any distance column constraints
    for (int i = 0; i < pIdxInfo->nConstraint; i++) {
      if (!pIdxInfo->aConstraint[i].usable)
        continue;

      int iColumn = pIdxInfo->aConstraint[i].iColumn;
      int op = pIdxInfo->aConstraint[i].op;
      if(op == SQLITE_INDEX_CONSTRAINT_LIMIT || op == SQLITE_INDEX_CONSTRAINT_OFFSET) {
        continue;
      }
      if(vec0_column_distance_idx(p) != iColumn) {
        continue;
      }

      char value = 0;
      switch(op) {
        case SQLITE_INDEX_CONSTRAINT_GT: {
          value = VEC0_DISTANCE_CONSTRAINT_GT;
          break;
        }
        case SQLITE_INDEX_CONSTRAINT_GE: {
          value = VEC0_DISTANCE_CONSTRAINT_GE;
          break;
        }
        case SQLITE_INDEX_CONSTRAINT_LT: {
          value = VEC0_DISTANCE_CONSTRAINT_LT;
          break;
        }
        case SQLITE_INDEX_CONSTRAINT_LE: {
          value = VEC0_DISTANCE_CONSTRAINT_LE;
          break;
        }
        default: {
          // IMP TODO
          rc = SQLITE_ERROR;
          vtab_set_error(
            pVTab, 
            "Illegal WHERE constraint on distance column in a KNN query. "
            "Only one of GT, GE, LT, LE constraints are allowed."
          );
          goto done;
        }
      }

      pIdxInfo->aConstraintUsage[i].argvIndex = argvIndex++;
      pIdxInfo->aConstraintUsage[i].omit = 1;
      sqlite3_str_appendchar(idxStr, 1, VEC0_IDXSTR_KIND_KNN_DISTANCE_CONSTRAINT);
      sqlite3_str_appendchar(idxStr, 1, value);
      sqlite3_str_appendchar(idxStr, 1, '_');
      sqlite3_str_appendchar(idxStr, 1, '_');
    }

    if (iMmrLambdaTerm >= 0) {
      pIdxInfo->aConstraintUsage[iMmrLambdaTerm].argvIndex = argvIndex++;
      pIdxInfo->aConstraintUsage[iMmrLambdaTerm].omit = 1;
      sqlite3_str_appendchar(idxStr, 1, VEC0_IDXSTR_KIND_KNN_MMR_LAMBDA);
      sqlite3_str_appendchar(idxStr, 3, '_');
    }

    pIdxInfo->idxNum = iMatchVectorTerm;
    pIdxInfo->estimatedCost = 30.0;
    pIdxInfo->estimatedRows = 10;

  } else if (iRowidTerm >= 0) {
    sqlite3_str_appendchar(idxStr, 1, VEC0_QUERY_PLAN_POINT);
    pIdxInfo->aConstraintUsage[iRowidTerm].argvIndex = 1;
    pIdxInfo->aConstraintUsage[iRowidTerm].omit = 1;
    sqlite3_str_appendchar(idxStr, 1, VEC0_IDXSTR_KIND_POINT_ID);
    sqlite3_str_appendchar(idxStr, 3, '_');
    pIdxInfo->idxNum = pIdxInfo->colUsed;
    pIdxInfo->estimatedCost = 10.0;
    pIdxInfo->estimatedRows = 1;
  } else {
    sqlite3_str_appendchar(idxStr, 1, VEC0_QUERY_PLAN_FULLSCAN);
    pIdxInfo->estimatedCost = 3000000.0;
    pIdxInfo->estimatedRows = 100000;
  }
  pIdxInfo->idxStr = sqlite3_str_finish(idxStr);
  idxStr = NULL;
  if (!pIdxInfo->idxStr) {
    rc = SQLITE_OK;
    goto done;
  }
  pIdxInfo->needToFreeIdxStr = 1;

  rc = SQLITE_OK;

  done:
    if(idxStr) {
      sqlite3_str_finish(idxStr);
    }
    return rc;
}

// forward delcaration bc vec0Filter uses it
static int vec0Next(sqlite3_vtab_cursor *cur);

void merge_sorted_lists(f32 *a, i64 *a_rowids, i64 a_length, f32 *b,
                        i64 *b_rowids, i32 *b_top_idxs, i64 b_length, f32 *out,
                        i64 *out_rowids, i64 out_length, i64 *out_used) {
  // assert((a_length >= out_length) || (b_length >= out_length));
  i64 ptrA = 0;
  i64 ptrB = 0;
  for (int i = 0; i < out_length; i++) {
    if ((ptrA >= a_length) && (ptrB >= b_length)) {
      *out_used = i;
      return;
    }
    if (ptrA >= a_length) {
      out[i] = b[b_top_idxs[ptrB]];
      out_rowids[i] = b_rowids[b_top_idxs[ptrB]];
      ptrB++;
    } else if (ptrB >= b_length) {
      out[i] = a[ptrA];
      out_rowids[i] = a_rowids[ptrA];
      ptrA++;
    } else {
      // A NaN ranks after every number, as in min_idx()
      if (a[ptrA] <= b[b_top_idxs[ptrB]] || isnan(b[b_top_idxs[ptrB]])) {
        out[i] = a[ptrA];
        out_rowids[i] = a_rowids[ptrA];
        ptrA++;
      } else {
        out[i] = b[b_top_idxs[ptrB]];
        out_rowids[i] = b_rowids[b_top_idxs[ptrB]];
        ptrB++;
      }
    }
  }

  *out_used = out_length;
}

u8 *bitmap_new(i32 n) {
  assert(n % 8 == 0);
  u8 *p = sqlite3_malloc(n * sizeof(u8) / CHAR_BIT);
  if (p) {
    memset(p, 0, n * sizeof(u8) / CHAR_BIT);
  }
  return p;
}
u8 *bitmap_new_from(i32 n, u8 *from) {
  assert(n % 8 == 0);
  u8 *p = sqlite3_malloc(n * sizeof(u8) / CHAR_BIT);
  if (p) {
    memcpy(p, from, n / CHAR_BIT);
  }
  return p;
}

void bitmap_copy(u8 *base, u8 *from, i32 n) {
  assert(n % 8 == 0);
  memcpy(base, from, n / CHAR_BIT);
}

void bitmap_and_inplace(u8 *base, u8 *other, i32 n) {
  assert((n % 8) == 0);
  for (int i = 0; i < n / CHAR_BIT; i++) {
    base[i] = base[i] & other[i];
  }
}

void bitmap_set(u8 *bitmap, i32 position, int value) {
  if (value) {
    bitmap[position / CHAR_BIT] |= 1 << (position % CHAR_BIT);
  } else {
    bitmap[position / CHAR_BIT] &= ~(1 << (position % CHAR_BIT));
  }
}

int bitmap_get(u8 *bitmap, i32 position) {
  return (((bitmap[position / CHAR_BIT]) >> (position % CHAR_BIT)) & 1);
}

void bitmap_clear(u8 *bitmap, i32 n) {
  assert((n % 8) == 0);
  memset(bitmap, 0, n / CHAR_BIT);
}

void bitmap_fill(u8 *bitmap, i32 n) {
  assert((n % 8) == 0);
  memset(bitmap, 0xFF, n / CHAR_BIT);
}

int bitmap_any(u8 *bitmap, i32 n) {
  assert((n % 8) == 0);
  for (int i = 0; i < n / CHAR_BIT; i++) {
    if (bitmap[i]) {
      return 1;
    }
  }
  return 0;
}

/**
 * @brief Whether the candidate at index a ranks after the one at index b in
 * min_idx()'s order: by distance, a NaN distance after every number, and
 * between equal distances the smaller index after the larger.
 */
static int min_idx_after(const f32 *distances, i32 a, i32 b) {
  int a_nan = isnan(distances[a]), b_nan = isnan(distances[b]);
  if (a_nan != b_nan) {
    return a_nan;
  }
  if (!a_nan && distances[a] != distances[b]) {
    return distances[a] > distances[b];
  }
  return a < b;
}

/**
 * @brief Finds the k candidates with the smallest distances, and writes their
 * indices to out, nearest first. Equal distances keep the larger index first,
 * and a NaN distance ranks after every number.
 *
 * @param distances input f32 array of size n, the items to consider.
 * @param n: size of distances array.
 * @param candidates: bitmap of size n, the indices that may be chosen.
 * @param out: Output array of size k, will contain at most k element indicies
 * @param k: Size of output array
 * @param k_used: set to the number of indices written to out
 * @return int
 */
int min_idx(const f32 *distances, i32 n, u8 *candidates, i32 *out, i32 k,
            i32 *k_used) {
  assert(k > 0);
  assert(k <= n);

  // Max-heap: O(n log k) in a single pass.
  // out[0..heap_size-1] stores indices; heap ordered by min_idx_after()
  // so out[0] is always the index that ranks LAST in the top-k.
  int heap_size = 0;

  #define HEAP_SIFT_UP(pos) do {                          \
    int _c = (pos);                                       \
    while (_c > 0) {                                      \
      int _p = (_c - 1) / 2;                              \
      if (min_idx_after(distances, out[_c], out[_p])) {   \
        i32 _tmp = out[_p]; out[_p] = out[_c]; out[_c] = _tmp; \
        _c = _p;                                          \
      } else break;                                       \
    }                                                     \
  } while(0)

  #define HEAP_SIFT_DOWN(pos, sz) do {                    \
    int _p = (pos);                                       \
    for (;;) {                                            \
      int _l = 2*_p + 1, _r = 2*_p + 2, _largest = _p;  \
      if (_l < (sz) && min_idx_after(distances, out[_l], out[_largest])) \
        _largest = _l;                                    \
      if (_r < (sz) && min_idx_after(distances, out[_r], out[_largest])) \
        _largest = _r;                                    \
      if (_largest == _p) break;                          \
      i32 _tmp = out[_p]; out[_p] = out[_largest]; out[_largest] = _tmp; \
      _p = _largest;                                      \
    }                                                     \
  } while(0)

  for (int i = 0; i < n; i++) {
    if (!bitmap_get(candidates, i))
      continue;
    if (heap_size < k) {
      out[heap_size] = i;
      heap_size++;
      HEAP_SIFT_UP(heap_size - 1);
    } else if (min_idx_after(distances, out[0], i)) {
      out[0] = i;
      HEAP_SIFT_DOWN(0, heap_size);
    }
  }

  // Heapsort to produce ascending order.
  for (int i = heap_size - 1; i > 0; i--) {
    i32 tmp = out[0]; out[0] = out[i]; out[i] = tmp;
    HEAP_SIFT_DOWN(0, i);
  }

  #undef HEAP_SIFT_UP
  #undef HEAP_SIFT_DOWN

  *k_used = heap_size;
  return SQLITE_OK;
}

int vec0_get_metadata_text_long_value(
  vec0_vtab * p,
  sqlite3_stmt ** stmt,
  int metadata_idx,
  i64 rowid,
  int *n,
  char ** s) {
  int rc;
  if(!(*stmt)) {
    const char * zSql = sqlite3_mprintf("select data from " VEC0_SHADOW_METADATA_TEXT_DATA_NAME " where rowid = ?", p->schemaName, p->tableName, metadata_idx);
    if(!zSql) {
      rc = SQLITE_NOMEM;
      goto done;
    }
    rc = sqlite3_prepare_v2(p->db, zSql, -1, stmt, NULL);
    sqlite3_free( (void *) zSql);
    if(rc != SQLITE_OK) {
      goto done;
    }
  }

  sqlite3_reset(*stmt);
  sqlite3_bind_int64(*stmt, 1, rowid);
  rc = sqlite3_step(*stmt);
  if(rc != SQLITE_ROW) {
    rc = SQLITE_ERROR;
    goto done;
  }
  *s = (char *) sqlite3_column_text(*stmt, 0);
  *n = sqlite3_column_bytes(*stmt, 0);
  rc = SQLITE_OK;
  done:
    return rc;
}

/**
 * @brief Crete at "iterator" (sqlite3_stmt) of chunks with the given constraints
 *
 * Any VEC0_IDXSTR_KIND_KNN_PARTITON_CONSTRAINT values in idxStr/argv will be applied
 * as WHERE constraints in the underlying stmt SQL, and any consumer of the stmt
 * can freely step through the stmt with all constraints satisfied.
 *
 * @param p - vec0_vtab
 * @param idxStr - the xBestIndex/xFilter idxstr containing VEC0_IDXSTR values
 * @param argc - number of argv values from xFilter
 * @param argv - array of sqlite3_value from xFilter
 * @param outStmt - output sqlite3_stmt of chunks with all filters applied
 * @return int SQLITE_OK on success, error code otherwise
 */
int vec0_chunks_iter(vec0_vtab * p, const char * idxStr, int argc, sqlite3_value ** argv, sqlite3_stmt** outStmt) {
  // always null terminated, enforced by SQLite
  int idxStrLength = strlen(idxStr);
  // "1" refers to the initial vec0_query_plan char, 4 is the number of chars per "element"
  int numValueEntries = (idxStrLength-1) / 4;
  assert(argc == numValueEntries);

  int rc;
  sqlite3_str * s = sqlite3_str_new(NULL);
  sqlite3_str_appendf(s, "select chunk_id, validity, rowids "
                         " from " VEC0_SHADOW_CHUNKS_NAME,
                         p->schemaName, p->tableName);

  int appendedWhere = 0;
  for(int i = 0; i < numValueEntries; i++) {
    int idx = 1 + (i * 4);
    char kind = idxStr[idx + 0];
    if(kind != VEC0_IDXSTR_KIND_KNN_PARTITON_CONSTRAINT) {
      continue;
    }

    int partition_idx = idxStr[idx + 1] - 'A';
    int operator = idxStr[idx + 2];
    // idxStr[idx + 3] is just null, a '_' placeholder

    if(!appendedWhere) {
      sqlite3_str_appendall(s, " WHERE ");
      appendedWhere = 1;
    }else {
      sqlite3_str_appendall(s, " AND ");
    }
    switch(operator) {
     case VEC0_PARTITION_OPERATOR_EQ:
      sqlite3_str_appendf(s, " partition%02d = ? ", partition_idx);
      break;
     case VEC0_PARTITION_OPERATOR_GT:
      sqlite3_str_appendf(s, " partition%02d > ? ", partition_idx);
      break;
     case VEC0_PARTITION_OPERATOR_LE:
      sqlite3_str_appendf(s, " partition%02d <= ? ", partition_idx);
      break;
     case VEC0_PARTITION_OPERATOR_LT:
      sqlite3_str_appendf(s, " partition%02d < ? ", partition_idx);
      break;
     case VEC0_PARTITION_OPERATOR_GE:
      sqlite3_str_appendf(s, " partition%02d >= ? ", partition_idx);
      break;
     case VEC0_PARTITION_OPERATOR_NE:
      sqlite3_str_appendf(s, " partition%02d != ? ", partition_idx);
      break;
     default: {
      char * zSql = sqlite3_str_finish(s);
      sqlite3_free(zSql);
      return SQLITE_ERROR;
     }

    }

  }

  char *zSql = sqlite3_str_finish(s);
  if (!zSql) {
    return SQLITE_NOMEM;
  }

  rc = sqlite3_prepare_v2(p->db, zSql, -1, outStmt, NULL);
  sqlite3_free(zSql);
  if(rc != SQLITE_OK) {
    return rc;
  }

  int n = 1;
  for(int i = 0; i < numValueEntries; i++) {
    int idx = 1 + (i * 4);
    char kind = idxStr[idx + 0];
    if(kind != VEC0_IDXSTR_KIND_KNN_PARTITON_CONSTRAINT) {
      continue;
    }
    sqlite3_bind_value(*outStmt, n++, argv[i]);
  }

  return rc;
}

// a single `xxx in (...)` constraint on a metadata column. TEXT or INTEGER only for now.
struct Vec0MetadataIn{
  // index of argv[i]` the constraint is on
  int argv_idx;
  // metadata column index of the constraint, derived from idxStr + argv_idx
  int metadata_idx;
  // array of the copied `(...)` values from sqlite3_vtab_in_first()/sqlite3_vtab_in_next()
  struct Array array;
};

// Array elements for `xxx in (...)` values for a text column. basically just a string
struct Vec0MetadataInTextEntry {
  int n;
  char * zString;
};


int vec0_metadata_filter_text(vec0_vtab * p, sqlite3_value * value, const void * buffer, int size, vec0_metadata_operator op, u8* b, int metadata_idx, int chunk_rowid, struct Array * aMetadataIn, int argv_idx) {
  int rc;
  sqlite3_stmt * stmt = NULL;
  i64 * rowids = NULL;
  sqlite3_blob * rowidsBlob;
  const char * sTarget = (const char *) sqlite3_value_text(value);
  int nTarget = sqlite3_value_bytes(value);


  // TODO(perf): only text metadata news the rowids BLOB. Make it so that
  // rowids BLOB is re-used when multiple fitlers on text columns,
  // ex "name BETWEEN 'a' and 'b'""
  rc = sqlite3_blob_open(p->db, p->schemaName, p->shadowChunksName, "rowids", chunk_rowid, 0, &rowidsBlob);
  if(rc != SQLITE_OK) {
    return rc;
  }
  assert(sqlite3_blob_bytes(rowidsBlob) % sizeof(i64) == 0);
  assert((sqlite3_blob_bytes(rowidsBlob) / sizeof(i64)) == size);

  rowids = sqlite3_malloc(sqlite3_blob_bytes(rowidsBlob));
  if(!rowids) {
    sqlite3_blob_close(rowidsBlob);
    return SQLITE_NOMEM;
  }

  rc = sqlite3_blob_read(rowidsBlob, rowids, sqlite3_blob_bytes(rowidsBlob), 0);
  if(rc != SQLITE_OK) {
    sqlite3_blob_close(rowidsBlob);
    return rc;
  }
  sqlite3_blob_close(rowidsBlob);

  switch(op) {
    int nPrefix;
    char * sPrefix;
    char *sFull;
    int nFull;
    u8 * view;
    case VEC0_METADATA_OPERATOR_EQ: {
      for(int i = 0; i < size; i++) {
        view = &((u8*) buffer)[i * VEC0_METADATA_TEXT_VIEW_BUFFER_LENGTH];
        nPrefix = ((int*) view)[0];
        sPrefix = (char *) &view[4];

        // for EQ the text lengths must match
        if(nPrefix != nTarget) {
          bitmap_set(b, i, 0);
          continue;
        }
        int cmpPrefix = strncmp(sPrefix, sTarget, min(nPrefix, VEC0_METADATA_TEXT_VIEW_DATA_LENGTH));

        // for short strings, use the prefix comparison direclty
        if(nPrefix <= VEC0_METADATA_TEXT_VIEW_DATA_LENGTH) {
          bitmap_set(b, i, cmpPrefix == 0);
          continue;
        }
        // for EQ on longs strings, the prefix must match
        if(cmpPrefix) {
          bitmap_set(b, i, 0);
          continue;
        }
        // consult the full string
        rc = vec0_get_metadata_text_long_value(p, &stmt, metadata_idx, rowids[i], &nFull, &sFull);
        if(rc != SQLITE_OK) {
          goto done;
        }
        if(nPrefix != nFull) {
          rc = SQLITE_ERROR;
          goto done;
        }
        bitmap_set(b, i, strncmp(sFull, sTarget, nFull) == 0);
      }
      break;
    }
    case VEC0_METADATA_OPERATOR_NE: {
      for(int i = 0; i < size; i++) {
        view = &((u8*) buffer)[i * VEC0_METADATA_TEXT_VIEW_BUFFER_LENGTH];
        nPrefix = ((int*) view)[0];
        sPrefix = (char *) &view[4];

        // for NE if text lengths dont match, it never will
        if(nPrefix != nTarget) {
          bitmap_set(b, i, 1);
          continue;
        }

        int cmpPrefix = strncmp(sPrefix, sTarget, min(nPrefix, VEC0_METADATA_TEXT_VIEW_DATA_LENGTH));

        // for short strings, use the prefix comparison direclty
        if(nPrefix <= VEC0_METADATA_TEXT_VIEW_DATA_LENGTH) {
          bitmap_set(b, i, cmpPrefix != 0);
          continue;
        }
        // for NE on longs strings, if prefixes dont match, then long string wont
        if(cmpPrefix) {
          bitmap_set(b, i, 1);
          continue;
        }
        // consult the full string
        rc = vec0_get_metadata_text_long_value(p, &stmt, metadata_idx, rowids[i], &nFull, &sFull);
        if(rc != SQLITE_OK) {
          goto done;
        }
        if(nPrefix != nFull) {
          rc = SQLITE_ERROR;
          goto done;
        }
        bitmap_set(b, i, strncmp(sFull, sTarget, nFull) != 0);
      }
      break;
    }
    case VEC0_METADATA_OPERATOR_GT: {
      for(int i = 0; i < size; i++) {
        view = &((u8*) buffer)[i * VEC0_METADATA_TEXT_VIEW_BUFFER_LENGTH];
        nPrefix = ((int*) view)[0];
        sPrefix = (char *) &view[4];
        int cmpPrefix = strncmp(sPrefix, sTarget, min(min(nPrefix, VEC0_METADATA_TEXT_VIEW_DATA_LENGTH), nTarget));

        if(nPrefix < VEC0_METADATA_TEXT_VIEW_DATA_LENGTH) {
          // if prefix match, check which is longer
          if(cmpPrefix == 0) {
            bitmap_set(b, i, nPrefix > nTarget);
          }
          else {
            bitmap_set(b, i, cmpPrefix > 0);
          }
          continue;
        }
        // TODO(perf): may not need to compare full text in some cases

        rc = vec0_get_metadata_text_long_value(p, &stmt, metadata_idx, rowids[i], &nFull, &sFull);
        if(rc != SQLITE_OK) {
          goto done;
        }
        if(nPrefix != nFull) {
          rc = SQLITE_ERROR;
          goto done;
        }
        bitmap_set(b, i, strncmp(sFull, sTarget, nFull) > 0);
      }
      break;
    }
    case VEC0_METADATA_OPERATOR_GE: {
      for(int i = 0; i < size; i++) {
        view = &((u8*) buffer)[i * VEC0_METADATA_TEXT_VIEW_BUFFER_LENGTH];
        nPrefix = ((int*) view)[0];
        sPrefix = (char *) &view[4];
        int cmpPrefix = strncmp(sPrefix, sTarget, min(min(nPrefix, VEC0_METADATA_TEXT_VIEW_DATA_LENGTH), nTarget));

        if(nPrefix < VEC0_METADATA_TEXT_VIEW_DATA_LENGTH) {
          // if prefix match, check which is longer
          if(cmpPrefix == 0) {
            bitmap_set(b, i, nPrefix >= nTarget);
          }
          else {
            bitmap_set(b, i, cmpPrefix >= 0);
          }
          continue;
        }
        // TODO(perf): may not need to compare full text in some cases

        rc = vec0_get_metadata_text_long_value(p, &stmt, metadata_idx, rowids[i], &nFull, &sFull);
        if(rc != SQLITE_OK) {
          goto done;
        }
        if(nPrefix != nFull) {
          rc = SQLITE_ERROR;
          goto done;
        }
        bitmap_set(b, i, strncmp(sFull, sTarget, nFull) >= 0);
      }
      break;
    }
    case VEC0_METADATA_OPERATOR_LE: {
      for(int i = 0; i < size; i++) {
        view = &((u8*) buffer)[i * VEC0_METADATA_TEXT_VIEW_BUFFER_LENGTH];
        nPrefix = ((int*) view)[0];
        sPrefix = (char *) &view[4];
        int cmpPrefix = strncmp(sPrefix, sTarget, min(min(nPrefix, VEC0_METADATA_TEXT_VIEW_DATA_LENGTH), nTarget));

        if(nPrefix < VEC0_METADATA_TEXT_VIEW_DATA_LENGTH) {
          // if prefix match, check which is longer
          if(cmpPrefix == 0) {
            bitmap_set(b, i, nPrefix <= nTarget);
          }
          else {
            bitmap_set(b, i, cmpPrefix <= 0);
          }
          continue;
        }
        // TODO(perf): may not need to compare full text in some cases

        rc = vec0_get_metadata_text_long_value(p, &stmt, metadata_idx, rowids[i], &nFull, &sFull);
        if(rc != SQLITE_OK) {
          goto done;
        }
        if(nPrefix != nFull) {
          rc = SQLITE_ERROR;
          goto done;
        }
        bitmap_set(b, i, strncmp(sFull, sTarget, nFull) <= 0);
      }
      break;
    }
    case VEC0_METADATA_OPERATOR_LT: {
      for(int i = 0; i < size; i++) {
        view = &((u8*) buffer)[i * VEC0_METADATA_TEXT_VIEW_BUFFER_LENGTH];
        nPrefix = ((int*) view)[0];
        sPrefix = (char *) &view[4];
        int cmpPrefix = strncmp(sPrefix, sTarget, min(min(nPrefix, VEC0_METADATA_TEXT_VIEW_DATA_LENGTH), nTarget));

        if(nPrefix < VEC0_METADATA_TEXT_VIEW_DATA_LENGTH) {
          // if prefix match, check which is longer
          if(cmpPrefix == 0) {
            bitmap_set(b, i, nPrefix < nTarget);
          }
          else {
            bitmap_set(b, i, cmpPrefix < 0);
          }
          continue;
        }
        // TODO(perf): may not need to compare full text in some cases

        rc = vec0_get_metadata_text_long_value(p, &stmt, metadata_idx, rowids[i], &nFull, &sFull);
        if(rc != SQLITE_OK) {
          goto done;
        }
        if(nPrefix != nFull) {
          rc = SQLITE_ERROR;
          goto done;
        }
        bitmap_set(b, i, strncmp(sFull, sTarget, nFull) < 0);
      }
      break;
    }

    case VEC0_METADATA_OPERATOR_IN: {
      int metadataInIdx = -1;
      for(size_t i = 0; i < aMetadataIn->length; i++) {
        struct Vec0MetadataIn * metadataIn = &(((struct Vec0MetadataIn *) aMetadataIn->z)[i]);
        if(metadataIn->argv_idx == argv_idx) {
          metadataInIdx = i;
          break;
        }
      }
      if(metadataInIdx < 0) {
        rc = SQLITE_ERROR;
        goto done;
      }

      struct Vec0MetadataIn * metadataIn = &((struct Vec0MetadataIn *) aMetadataIn->z)[metadataInIdx];
      struct Array * aTarget = &(metadataIn->array);


      int nPrefix;
      char * sPrefix;
      char *sFull;
      int nFull;
      u8 * view;
      for(int i = 0; i < size; i++) {
        view = &((u8*) buffer)[i * VEC0_METADATA_TEXT_VIEW_BUFFER_LENGTH];
        nPrefix = ((int*) view)[0];
        sPrefix = (char *) &view[4];
        for(size_t target_idx = 0; target_idx < aTarget->length; target_idx++) {
          struct Vec0MetadataInTextEntry * entry = &(((struct Vec0MetadataInTextEntry*)aTarget->z)[target_idx]);
          if(entry->n != nPrefix) {
            continue;
          }
          int cmpPrefix = strncmp(sPrefix, entry->zString, min(nPrefix, VEC0_METADATA_TEXT_VIEW_DATA_LENGTH));
          if(nPrefix <= VEC0_METADATA_TEXT_VIEW_DATA_LENGTH) {
            if(cmpPrefix == 0) {
              bitmap_set(b, i, 1);
              break;
            }
            continue;
          }
          if(cmpPrefix) {
            continue;
          }

          rc = vec0_get_metadata_text_long_value(p, &stmt, metadata_idx, rowids[i], &nFull, &sFull);
          if(rc != SQLITE_OK) {
            goto done;
          }
          if(nPrefix != nFull) {
            rc = SQLITE_ERROR;
            goto done;
          }
          if(strncmp(sFull, entry->zString, nFull) == 0) {
            bitmap_set(b, i, 1);
            break;
          }
        }
      }
      break;
    }

  }
  rc = SQLITE_OK;

  done:
    sqlite3_finalize(stmt);
    sqlite3_free(rowids);
    return rc;

}

/**
 * @brief Fill in bitmap of chunk values, whether or not the values match a metadata constraint
 *
 * @param p vec0_vtab
 * @param metadata_idx index of the metatadata column to perfrom constraints on
 * @param value sqlite3_value of the constraints value
 * @param blob sqlite3_blob that is already opened on the metdata column's shadow chunk table
 * @param chunk_rowid rowid of the chunk to calculate on
 * @param b pre-allocated and zero'd out bitmap to write results to
 * @param size size of the chunk
 * @return int SQLITE_OK on success, error code otherwise
 */
int vec0_set_metadata_filter_bitmap(
  vec0_vtab *p,
  int metadata_idx,
  vec0_metadata_operator op,
  sqlite3_value * value,
  sqlite3_blob * blob,
  i64 chunk_rowid,
  u8* b,
  int size,
  struct Array * aMetadataIn, int argv_idx) {
  // TODO: shouldn't this skip in-valid entries from the chunk's  validity bitmap?

  int rc;
  rc = sqlite3_blob_reopen(blob, chunk_rowid);
  if(rc != SQLITE_OK) {
    return rc;
  }

  vec0_metadata_column_kind kind = p->metadata_columns[metadata_idx].kind;
  int szMatch = 0;
  int blobSize = sqlite3_blob_bytes(blob);
  switch(kind) {
    case VEC0_METADATA_COLUMN_KIND_BOOLEAN: {
      szMatch = blobSize == size / CHAR_BIT;
      break;
    }
    case VEC0_METADATA_COLUMN_KIND_INTEGER: {
      szMatch = blobSize == size * sizeof(i64);
      break;
    }
    case VEC0_METADATA_COLUMN_KIND_FLOAT: {
      szMatch = blobSize == size * sizeof(double);
      break;
    }
    case VEC0_METADATA_COLUMN_KIND_TEXT: {
      szMatch = blobSize == size * VEC0_METADATA_TEXT_VIEW_BUFFER_LENGTH;
      break;
    }
  }
  if(!szMatch) {
    return SQLITE_ERROR;
  }
  void * buffer = sqlite3_malloc(blobSize);
  if(!buffer) {
    return SQLITE_NOMEM;
  }
  rc = sqlite3_blob_read(blob, buffer, blobSize, 0);
  if(rc != SQLITE_OK) {
    goto done;
  }
  switch(kind) {
    case VEC0_METADATA_COLUMN_KIND_BOOLEAN: {
      int target = sqlite3_value_int(value);
      if( (target && op == VEC0_METADATA_OPERATOR_EQ) || (!target && op == VEC0_METADATA_OPERATOR_NE)) {
        for(int i = 0; i < size; i++) { bitmap_set(b, i, bitmap_get((u8*) buffer, i)); }
      }
      else {
        for(int i = 0; i < size; i++) { bitmap_set(b, i, !bitmap_get((u8*) buffer, i)); }
      }
      break;
    }
    case VEC0_METADATA_COLUMN_KIND_INTEGER: {
      i64 * array = (i64*) buffer;
      i64 target = sqlite3_value_int64(value);
      switch(op) {
        case VEC0_METADATA_OPERATOR_EQ: {
          for(int i = 0; i < size; i++) { bitmap_set(b, i, array[i] == target); }
          break;
        }
        case VEC0_METADATA_OPERATOR_GT: {
          for(int i = 0; i < size; i++) { bitmap_set(b, i, array[i] > target); }
          break;
        }
        case VEC0_METADATA_OPERATOR_LE: {
          for(int i = 0; i < size; i++) { bitmap_set(b, i, array[i] <= target); }
          break;
        }
        case VEC0_METADATA_OPERATOR_LT: {
          for(int i = 0; i < size; i++) { bitmap_set(b, i, array[i] < target); }
          break;
        }
        case VEC0_METADATA_OPERATOR_GE: {
          for(int i = 0; i < size; i++) { bitmap_set(b, i, array[i] >= target); }
          break;
        }
        case VEC0_METADATA_OPERATOR_NE: {
          for(int i = 0; i < size; i++) { bitmap_set(b, i, array[i] != target); }
          break;
        }
        case VEC0_METADATA_OPERATOR_IN: {
          int metadataInIdx = -1;
          for(size_t i = 0; i < aMetadataIn->length; i++) {
            struct Vec0MetadataIn * metadataIn = &((struct Vec0MetadataIn *) aMetadataIn->z)[i];
            if(metadataIn->argv_idx == argv_idx) {
              metadataInIdx = i;
              break;
            }
          }
          if(metadataInIdx < 0) {
            rc = SQLITE_ERROR;
            goto done;
          }
          struct Vec0MetadataIn * metadataIn = &((struct Vec0MetadataIn *) aMetadataIn->z)[metadataInIdx];
          struct Array * aTarget = &(metadataIn->array);

          // the list was sorted when vec0Filter_knn read it, as a
          // `rowid in (...)` list is
          for(int i = 0; i < size; i++) {
            if(bsearch(&array[i], aTarget->z, aTarget->length, sizeof(i64), _cmp)) {
              bitmap_set(b, i, 1);
            }
          }
          break;
        }
      }
      break;
    }
    case VEC0_METADATA_COLUMN_KIND_FLOAT: {
      double * array = (double*) buffer;
      double target = sqlite3_value_double(value);
      switch(op) {
        case VEC0_METADATA_OPERATOR_EQ: {
          for(int i = 0; i < size; i++) { bitmap_set(b, i, array[i] == target); }
          break;
        }
        case VEC0_METADATA_OPERATOR_GT: {
          for(int i = 0; i < size; i++) { bitmap_set(b, i, array[i] > target); }
          break;
        }
        case VEC0_METADATA_OPERATOR_LE: {
          for(int i = 0; i < size; i++) { bitmap_set(b, i, array[i] <= target); }
          break;
        }
        case VEC0_METADATA_OPERATOR_LT: {
          for(int i = 0; i < size; i++) { bitmap_set(b, i, array[i] < target); }
          break;
        }
        case VEC0_METADATA_OPERATOR_GE: {
          for(int i = 0; i < size; i++) { bitmap_set(b, i, array[i] >= target); }
          break;
        }
        case VEC0_METADATA_OPERATOR_NE: {
          for(int i = 0; i < size; i++) { bitmap_set(b, i, array[i] != target); }
          break;
        }
        case VEC0_METADATA_OPERATOR_IN: {
          // should never be reached
          break;
        }
      }
      break;
    }
    case VEC0_METADATA_COLUMN_KIND_TEXT: {
      rc = vec0_metadata_filter_text(p, value, buffer, size, op, b, metadata_idx, chunk_rowid, aMetadataIn, argv_idx);
      if(rc != SQLITE_OK) {
        goto done;
      }
      break;
    }
  }
  done:
    sqlite3_free(buffer);
    return rc;
}

static int vec0_zonemap_excludes(vec0_vtab *p, i64 chunk, const char *idxStr,
                                int argc, sqlite3_value **argv,
                                struct Array *metadataIn, int *excluded) {
  *excluded = 0;
  for (int i = 0; i < argc; i++) {
    int idx = 1 + i * 4;
    if (idxStr[idx] != VEC0_IDXSTR_KIND_METADATA_CONSTRAINT) continue;
    int column = idxStr[idx + 1] - 'A';
    int op = idxStr[idx + 2];
    vec0_metadata_column_kind kind = p->metadata_columns[column].kind;
    if (kind == VEC0_METADATA_COLUMN_KIND_TEXT) continue;
    Vec0ZoneMap bounds;
    int state;
    int rc = vec0_zonemap_read(p, column, chunk, &bounds, &state);
    if (rc != SQLITE_OK) return rc;
    if (state != 2) continue;
    int below, above, atMin, atMax;
    if (kind == VEC0_METADATA_COLUMN_KIND_FLOAT) {
      if (!(bounds.f[0] <= bounds.f[1])) continue;
      double target = sqlite3_value_double(argv[i]);
      below = target < bounds.f[0]; above = target > bounds.f[1];
      atMin = target == bounds.f[0]; atMax = target == bounds.f[1];
    } else {
      if (bounds.i[0] > bounds.i[1]) continue;
      if (op == VEC0_METADATA_OPERATOR_IN) {
        for (size_t j = 0; j < metadataIn->length; j++) {
          struct Vec0MetadataIn *in = &((struct Vec0MetadataIn *)metadataIn->z)[j];
          if (in->argv_idx != i) continue;
          i64 *values = in->array.z;
          size_t lo = 0, hi = in->array.length;
          while (lo < hi) {
            size_t mid = lo + (hi - lo) / 2;
            if (values[mid] < bounds.i[0]) lo = mid + 1;
            else hi = mid;
          }
          *excluded = lo == in->array.length || values[lo] > bounds.i[1];
          break;
        }
        if (*excluded) return SQLITE_OK;
        continue;
      }
      i64 target = kind == VEC0_METADATA_COLUMN_KIND_BOOLEAN
                     ? !!sqlite3_value_int(argv[i]) : sqlite3_value_int64(argv[i]);
      below = target < bounds.i[0]; above = target > bounds.i[1];
      atMin = target == bounds.i[0]; atMax = target == bounds.i[1];
    }
    switch (op) {
      case VEC0_METADATA_OPERATOR_EQ: *excluded = below || above; break;
      case VEC0_METADATA_OPERATOR_GT: *excluded = above || atMax; break;
      case VEC0_METADATA_OPERATOR_GE: *excluded = above; break;
      case VEC0_METADATA_OPERATOR_LT: *excluded = below || atMin; break;
      case VEC0_METADATA_OPERATOR_LE: *excluded = below; break;
      case VEC0_METADATA_OPERATOR_NE: *excluded = atMin && atMax; break;
      default: break;
    }
    if (*excluded) return SQLITE_OK;
  }
  return SQLITE_OK;
}

/* Keep each vector at its normal chunk offset so distance/ranking uses the
 * same filtered bitmap for whole-chunk and selective reads. */
static int vec0_read_vector_chunk(sqlite3_blob *blob, u8 *data, u8 *selected,
                                 int count, int stride) {
  // Many small BLOB calls can cost more than copying the whole chunk. Keep
  // dense chunks contiguous, and charge 64 KiB of saved copying per read run.
  int kept = 0, runs = 0, previous = 0;
  for (int i = 0; i < count; i++) {
    int current = bitmap_get(selected, i);
    kept += current;
    runs += current && !previous;
    if (kept > count / 8 ||
        (i64)kept * stride + (i64)runs * 65536 >= (i64)count * stride)
      return sqlite3_blob_read(blob, data, count * stride, 0);
    previous = current;
  }
  for (int i = 0; i < count;) {
    if (!bitmap_get(selected, i)) { i++; continue; }
    int start = i++;
    while (i < count && bitmap_get(selected, i)) i++;
    int rc = sqlite3_blob_read(blob, data + start * stride,
                               (i - start) * stride, start * stride);
    if (rc != SQLITE_OK) return rc;
  }
  return SQLITE_OK;
}

int vec0Filter_knn_chunks_iter(vec0_vtab *p, sqlite3_stmt *stmtChunks,
                               struct VectorColumnDefinition *vector_column,
                               int vectorColumnIdx, struct Array *arrayRowidsIn,
                               struct Array * aMetadataIn,
                               const char * idxStr, int argc, sqlite3_value ** argv,
                               void *queryVector, i64 k, i64 **out_topk_rowids,
                               f32 **out_topk_distances, i64 *out_used) {
  // for each chunk, get top min(k, chunk_size) rowid + distances to query vec.
  // then reconcile all topk_chunks for a true top k.
  // output only rowids + distances for now

  int rc = SQLITE_OK;
  sqlite3_blob *blobVectors = NULL;

  void *baseVectors = NULL; // memory: chunk_size * dimensions * element_size

  // OWNED BY CALLER ON SUCCESS
  i64 *topk_rowids = NULL; // memory: k * 4
  // OWNED BY CALLER ON SUCCESS
  f32 *topk_distances = NULL; // memory: k * 4

  i64 *tmp_topk_rowids = NULL;    // memory: k * 4
  f32 *tmp_topk_distances = NULL; // memory: k * 4
  f32 *chunk_distances = NULL;    // memory: chunk_size * 4
  u8 *b = NULL;                   // memory: chunk_size / 8
  i32 *chunk_topk_idxs = NULL;    // memory: k * 4
  u8 *bmRowids = NULL;            // memory: chunk_size / 8
  u8 *bmMetadata = NULL;            // memory: chunk_size / 8
  //                        // total: a lot???

  // 6 * (k * 4) + (k * 2) + (chunk_size / 8) + (chunk_size * dimensions * 4)

  topk_rowids = sqlite3_malloc(k * sizeof(i64));
  if (!topk_rowids) {
    rc = SQLITE_NOMEM;
    goto cleanup;
  }
  memset(topk_rowids, 0, k * sizeof(i64));

  topk_distances = sqlite3_malloc(k * sizeof(f32));
  if (!topk_distances) {
    rc = SQLITE_NOMEM;
    goto cleanup;
  }
  memset(topk_distances, 0, k * sizeof(f32));

  tmp_topk_rowids = sqlite3_malloc(k * sizeof(i64));
  if (!tmp_topk_rowids) {
    rc = SQLITE_NOMEM;
    goto cleanup;
  }
  memset(tmp_topk_rowids, 0, k * sizeof(i64));

  tmp_topk_distances = sqlite3_malloc(k * sizeof(f32));
  if (!tmp_topk_distances) {
    rc = SQLITE_NOMEM;
    goto cleanup;
  }
  memset(tmp_topk_distances, 0, k * sizeof(f32));

  i64 k_used = 0;
  i64 baseVectorsSize = p->chunk_size * vector_column_byte_size(*vector_column);
  baseVectors = sqlite3_malloc(baseVectorsSize);
  if (!baseVectors) {
    rc = SQLITE_NOMEM;
    goto cleanup;
  }

  chunk_distances = sqlite3_malloc(p->chunk_size * sizeof(f32));
  if (!chunk_distances) {
    rc = SQLITE_NOMEM;
    goto cleanup;
  }

  b = bitmap_new(p->chunk_size);
  if (!b) {
    rc = SQLITE_NOMEM;
    goto cleanup;
  }

  chunk_topk_idxs = sqlite3_malloc(k * sizeof(i32));
  if (!chunk_topk_idxs) {
    rc = SQLITE_NOMEM;
    goto cleanup;
  }

  bmRowids = arrayRowidsIn ? bitmap_new(p->chunk_size) : NULL;
  if (arrayRowidsIn && !bmRowids) {
    rc = SQLITE_NOMEM;
    goto cleanup;
  }

  sqlite3_blob * metadataBlobs[VEC0_MAX_METADATA_COLUMNS];
  memset(metadataBlobs, 0, sizeof(sqlite3_blob*) * VEC0_MAX_METADATA_COLUMNS);

  bmMetadata = bitmap_new(p->chunk_size);
  if(!bmMetadata) {
    rc = SQLITE_NOMEM;
    goto cleanup;
  }

  int idxStrLength = strlen(idxStr);
  int numValueEntries = (idxStrLength-1) / 4;
  assert(numValueEntries == argc);
  int hasMetadataFilters = 0;
  int hasDistanceConstraints = 0;
  for(int i = 0; i < argc; i++) {
    int idx = 1 + (i * 4);
    char kind = idxStr[idx + 0];
    if(kind == VEC0_IDXSTR_KIND_METADATA_CONSTRAINT) {
      hasMetadataFilters = 1;
    }
    else if(kind == VEC0_IDXSTR_KIND_KNN_DISTANCE_CONSTRAINT) {
      hasDistanceConstraints = 1;
    }
  }

  while (true) {
    rc = sqlite3_step(stmtChunks);
    if (rc == SQLITE_DONE) {
      break;
    }
    if (rc != SQLITE_ROW) {
      vtab_set_error(&p->base, "chunks iter error");
      rc = SQLITE_ERROR;
      goto cleanup;
    }
    memset(chunk_distances, 0, p->chunk_size * sizeof(f32));
    memset(chunk_topk_idxs, 0, k * sizeof(i32));
    bitmap_clear(b, p->chunk_size);

    i64 chunk_id = sqlite3_column_int64(stmtChunks, 0);
    unsigned char *chunkValidity =
        (unsigned char *)sqlite3_column_blob(stmtChunks, 1);
    i64 validitySize = sqlite3_column_bytes(stmtChunks, 1);
    if (validitySize != p->chunk_size / CHAR_BIT) {
      // IMP: V05271_22109
      vtab_set_error(
          &p->base,
          "chunk validity size doesn't match - expected %lld, found %lld",
          p->chunk_size / CHAR_BIT, validitySize);
      rc = SQLITE_ERROR;
      goto cleanup;
    }

    i64 *chunkRowids = (i64 *)sqlite3_column_blob(stmtChunks, 2);
    i64 rowidsSize = sqlite3_column_bytes(stmtChunks, 2);
    if (rowidsSize != p->chunk_size * sizeof(i64)) {
      // IMP: V02796_19635
      vtab_set_error(&p->base, "rowids size doesn't match");
      vtab_set_error(
          &p->base,
          "chunk rowids size doesn't match - expected %lld, found %lld",
          p->chunk_size * sizeof(i64), rowidsSize);
      rc = SQLITE_ERROR;
      goto cleanup;
    }

    bitmap_copy(b, chunkValidity, p->chunk_size);
    if (arrayRowidsIn) {
      bitmap_clear(bmRowids, p->chunk_size);

      for (int i = 0; i < p->chunk_size; i++) {
        if (!bitmap_get(chunkValidity, i)) {
          continue;
        }
        bitmap_set(bmRowids, i, vec0_rowids_contains(arrayRowidsIn, chunkRowids[i]));
      }
      bitmap_and_inplace(b, bmRowids, p->chunk_size);
    }

    if(hasMetadataFilters) {
      int excluded;
      rc = vec0_zonemap_excludes(p, chunk_id, idxStr, argc, argv, aMetadataIn, &excluded);
      if (rc != SQLITE_OK) goto cleanup;
      if (excluded) continue;
      for(int i = 0; i < argc; i++) {
        int idx = 1 + (i * 4);
        char kind = idxStr[idx + 0];
        if(kind != VEC0_IDXSTR_KIND_METADATA_CONSTRAINT) {
          continue;
        }
        int metadata_idx = idxStr[idx + 1] - 'A';
        int operator = idxStr[idx + 2];

        if(!metadataBlobs[metadata_idx]) {
          rc = sqlite3_blob_open(p->db, p->schemaName, p->shadowMetadataChunksNames[metadata_idx], "data", chunk_id, 0, &metadataBlobs[metadata_idx]);
          vtab_set_error(&p->base, "Could not open metadata blob");
          if(rc != SQLITE_OK) {
            goto cleanup;
          }
        }

        bitmap_clear(bmMetadata, p->chunk_size);
        rc = vec0_set_metadata_filter_bitmap(p, metadata_idx, operator, argv[i], metadataBlobs[metadata_idx], chunk_id, bmMetadata, p->chunk_size, aMetadataIn, i);
        if(rc != SQLITE_OK) {
          vtab_set_error(&p->base, "Could not filter metadata fields");
          if(rc != SQLITE_OK) {
            goto cleanup;
          }
        }
        bitmap_and_inplace(b, bmMetadata, p->chunk_size);
      }
    }

    // A chunk whose rowid and metadata filters keep no row offers no
    // candidate, so its vectors are never read.
    if (!bitmap_any(b, p->chunk_size)) {
      continue;
    }

    // open the vector chunk blob for the current chunk
    rc = sqlite3_blob_open(p->db, p->schemaName,
                           p->shadowVectorChunksNames[vectorColumnIdx],
                           "vectors", chunk_id, 0, &blobVectors);
    if (rc != SQLITE_OK) {
      vtab_set_error(&p->base, "could not open vectors blob for chunk %lld",
                     chunk_id);
      rc = SQLITE_ERROR;
      goto cleanup;
    }

    i64 currentBaseVectorsSize = sqlite3_blob_bytes(blobVectors);
    i64 expectedBaseVectorsSize =
        p->chunk_size * vector_column_byte_size(*vector_column);
    if (currentBaseVectorsSize != expectedBaseVectorsSize) {
      // IMP: V16465_00535
      vtab_set_error(
          &p->base,
          "vectors blob size doesn't match - expected %lld, found %lld",
          expectedBaseVectorsSize, currentBaseVectorsSize);
      rc = SQLITE_ERROR;
      goto cleanup;
    }
    rc = vec0_read_vector_chunk(blobVectors, baseVectors, b, p->chunk_size,
                                vector_column_byte_size(*vector_column));

    if (rc != SQLITE_OK) {
      vtab_set_error(&p->base, "vectors blob read error for %lld", chunk_id);
      rc = SQLITE_ERROR;
      goto cleanup;
    }

    for (int i = 0; i < p->chunk_size; i++) {
      if (!bitmap_get(b, i)) {
        continue;
      };

      size_t stride = vector_column_byte_size(*vector_column);
      f32 result = vec0_distance_full((const u8 *)baseVectors + i * stride,
                                      queryVector, vector_column->dimensions,
                                      vector_column->element_type,
                                      vector_column->distance_metric);

      chunk_distances[i] = result;
    }

    if(hasDistanceConstraints) {
      for(int i = 0; i < p->chunk_size; i++) {
        if(bitmap_get(b, i) &&
           !vec0_distance_constraints_satisfied(chunk_distances[i], idxStr,
                                                argc, argv)) {
          bitmap_set(b, i, 0);
        }
      }
    }

    int used1;
    min_idx(chunk_distances, p->chunk_size, b, chunk_topk_idxs,
            min(k, p->chunk_size), &used1);

    i64 used;
    merge_sorted_lists(topk_distances, topk_rowids, k_used, chunk_distances,
                       chunkRowids, chunk_topk_idxs,
                       min(min(k, p->chunk_size), used1), tmp_topk_distances,
                       tmp_topk_rowids, k, &used);

    for (int i = 0; i < used; i++) {
      topk_rowids[i] = tmp_topk_rowids[i];
      topk_distances[i] = tmp_topk_distances[i];
    }
    k_used = used;
    // blobVectors is always opened with read-only permissions, so this never
    // fails.
    sqlite3_blob_close(blobVectors);
    blobVectors = NULL;
  }

  *out_topk_rowids = topk_rowids;
  *out_topk_distances = topk_distances;
  *out_used = k_used;
  rc = SQLITE_OK;

cleanup:
  if (rc != SQLITE_OK) {
    sqlite3_free(topk_rowids);
    sqlite3_free(topk_distances);
  }
  sqlite3_free(chunk_topk_idxs);
  sqlite3_free(tmp_topk_rowids);
  sqlite3_free(tmp_topk_distances);
  sqlite3_free(b);
  sqlite3_free(bmRowids);
  sqlite3_free(baseVectors);
  sqlite3_free(chunk_distances);
  sqlite3_free(bmMetadata);
  for(int i = 0; i < VEC0_MAX_METADATA_COLUMNS; i++) {
    sqlite3_blob_close(metadataBlobs[i]);
  }
  // blobVectors is always opened with read-only permissions, so this never
  // fails.
  sqlite3_blob_close(blobVectors);
  return rc;
}

#if SQLITE_VEC_ENABLE_RESCORE

//////////////////////////////////////////////////////////////////////////////
// Begin inlined: sqlite-vec-rescore.c
//////////////////////////////////////////////////////////////////////////////

/**
 * sqlite-vec-rescore.c — Rescore index logic for sqlite-vec.
 *
 * This file is #included into sqlite-vec.c after the vec0_vtab definition.
 * All functions receive a vec0_vtab *p and access p->vector_columns[i].rescore.
 *
 * Shadow tables per rescore-enabled vector column:
 *   _rescore_chunks{NN}  — quantized vectors in chunk layout (for coarse scan)
 *   _rescore_vectors{NN} — float vectors keyed by rowid (for fast rescore lookup)
 */

// ============================================================================
// Shadow table lifecycle
// ============================================================================

static int rescore_create_tables(vec0_vtab *p, sqlite3 *db, char **pzErr) {
  for (int i = 0; i < p->numVectorColumns; i++) {
    if (p->vector_columns[i].index_type != VEC0_INDEX_TYPE_RESCORE)
      continue;

    // Quantized chunk table (same layout as _vector_chunks)
    char *zSql = sqlite3_mprintf(
        "CREATE TABLE \"%w\".\"%w_rescore_chunks%02d\""
        "(rowid PRIMARY KEY, vectors BLOB NOT NULL)",
        p->schemaName, p->tableName, i);
    if (!zSql)
      return SQLITE_NOMEM;
    sqlite3_stmt *stmt;
    int rc = sqlite3_prepare_v2(db, zSql, -1, &stmt, 0);
    sqlite3_free(zSql);
    if ((rc != SQLITE_OK) || (sqlite3_step(stmt) != SQLITE_DONE)) {
      *pzErr = sqlite3_mprintf(
          "Could not create '_rescore_chunks%02d' shadow table: %s", i,
          sqlite3_errmsg(db));
      sqlite3_finalize(stmt);
      return SQLITE_ERROR;
    }
    sqlite3_finalize(stmt);

    // Float vector table (rowid-keyed for fast random access)
    zSql = sqlite3_mprintf(
        "CREATE TABLE \"%w\".\"%w_rescore_vectors%02d\""
        "(rowid INTEGER PRIMARY KEY, vector BLOB NOT NULL)",
        p->schemaName, p->tableName, i);
    if (!zSql)
      return SQLITE_NOMEM;
    rc = sqlite3_prepare_v2(db, zSql, -1, &stmt, 0);
    sqlite3_free(zSql);
    if ((rc != SQLITE_OK) || (sqlite3_step(stmt) != SQLITE_DONE)) {
      *pzErr = sqlite3_mprintf(
          "Could not create '_rescore_vectors%02d' shadow table: %s", i,
          sqlite3_errmsg(db));
      sqlite3_finalize(stmt);
      return SQLITE_ERROR;
    }
    sqlite3_finalize(stmt);
  }
  return SQLITE_OK;
}

static int rescore_drop_tables(vec0_vtab *p) {
  for (int i = 0; i < p->numVectorColumns; i++) {
    sqlite3_stmt *stmt;
    int rc;
    char *zSql;

    if (p->shadowRescoreChunksNames[i]) {
      zSql = sqlite3_mprintf("DROP TABLE IF EXISTS \"%w\".\"%w\"",
                              p->schemaName, p->shadowRescoreChunksNames[i]);
      if (!zSql)
        return SQLITE_NOMEM;
      rc = sqlite3_prepare_v2(p->db, zSql, -1, &stmt, 0);
      sqlite3_free(zSql);
      if ((rc != SQLITE_OK) || (sqlite3_step(stmt) != SQLITE_DONE)) {
        sqlite3_finalize(stmt);
        return SQLITE_ERROR;
      }
      sqlite3_finalize(stmt);
    }

    if (p->shadowRescoreVectorsNames[i]) {
      zSql = sqlite3_mprintf("DROP TABLE IF EXISTS \"%w\".\"%w\"",
                              p->schemaName, p->shadowRescoreVectorsNames[i]);
      if (!zSql)
        return SQLITE_NOMEM;
      rc = sqlite3_prepare_v2(p->db, zSql, -1, &stmt, 0);
      sqlite3_free(zSql);
      if ((rc != SQLITE_OK) || (sqlite3_step(stmt) != SQLITE_DONE)) {
        sqlite3_finalize(stmt);
        return SQLITE_ERROR;
      }
      sqlite3_finalize(stmt);
    }
  }
  return SQLITE_OK;
}

static size_t rescore_quantized_byte_size(struct VectorColumnDefinition *col) {
  switch (col->rescore.quantizer_type) {
  case VEC0_RESCORE_QUANTIZER_BIT:
    return col->dimensions / CHAR_BIT;
  case VEC0_RESCORE_QUANTIZER_INT8:
    return col->dimensions;
  default:
    return 0;
  }
}

/**
 * Insert a new chunk row into each _rescore_chunks{NN} table with a zeroblob.
 */
static int rescore_new_chunk(vec0_vtab *p, i64 chunk_rowid) {
  for (int i = 0; i < p->numVectorColumns; i++) {
    if (p->vector_columns[i].index_type != VEC0_INDEX_TYPE_RESCORE)
      continue;
    size_t quantized_size =
        rescore_quantized_byte_size(&p->vector_columns[i]);
    i64 blob_size = (i64)p->chunk_size * (i64)quantized_size;

    char *zSql = sqlite3_mprintf(
        "INSERT INTO \"%w\".\"%w\"(_rowid_, rowid, vectors) VALUES (?, ?, ?)",
        p->schemaName, p->shadowRescoreChunksNames[i]);
    if (!zSql)
      return SQLITE_NOMEM;
    sqlite3_stmt *stmt;
    int rc = sqlite3_prepare_v2(p->db, zSql, -1, &stmt, NULL);
    sqlite3_free(zSql);
    if (rc != SQLITE_OK) {
      sqlite3_finalize(stmt);
      return rc;
    }
    sqlite3_bind_int64(stmt, 1, chunk_rowid);
    sqlite3_bind_int64(stmt, 2, chunk_rowid);
    sqlite3_bind_zeroblob64(stmt, 3, blob_size);
    rc = sqlite3_step(stmt);
    sqlite3_finalize(stmt);
    if (rc != SQLITE_DONE)
      return rc;
  }
  return SQLITE_OK;
}

// ============================================================================
// Quantization
// ============================================================================

static void rescore_quantize_float_to_bit(const float *src, uint8_t *dst,
                                          size_t dimensions) {
  memset(dst, 0, dimensions / CHAR_BIT);
  for (size_t i = 0; i < dimensions; i++) {
    if (src[i] >= 0.0f) {
      dst[i / CHAR_BIT] |= (1 << (i % CHAR_BIT));
    }
  }
}

static void rescore_quantize_float_to_int8(const float *src, int8_t *dst,
                                           size_t dimensions) {
  vec0_quantize_int8_unit(src, dst, dimensions);
}

// ============================================================================
// Insert path
// ============================================================================

/**
 * Quantize float vector to _rescore_chunks and store in _rescore_vectors.
 */
static int rescore_on_insert(vec0_vtab *p, i64 chunk_rowid, i64 chunk_offset,
                             i64 rowid, void *vectorDatas[]) {
  for (int i = 0; i < p->numVectorColumns; i++) {
    if (p->vector_columns[i].index_type != VEC0_INDEX_TYPE_RESCORE)
      continue;

    struct VectorColumnDefinition *col = &p->vector_columns[i];
    size_t qsize = rescore_quantized_byte_size(col);
    size_t fsize = vector_column_byte_size(*col);
    int rc;

    // 1. Write quantized vector to _rescore_chunks blob
    {
      void *qbuf = sqlite3_malloc(qsize);
      if (!qbuf)
        return SQLITE_NOMEM;

      switch (col->rescore.quantizer_type) {
      case VEC0_RESCORE_QUANTIZER_BIT:
        rescore_quantize_float_to_bit((const float *)vectorDatas[i],
                                      (uint8_t *)qbuf, col->dimensions);
        break;
      case VEC0_RESCORE_QUANTIZER_INT8:
        rescore_quantize_float_to_int8((const float *)vectorDatas[i],
                                       (int8_t *)qbuf, col->dimensions);
        break;
      }

      sqlite3_blob *blob = NULL;
      rc = sqlite3_blob_open(p->db, p->schemaName,
                             p->shadowRescoreChunksNames[i], "vectors",
                             chunk_rowid, 1, &blob);
      if (rc != SQLITE_OK) {
        sqlite3_free(qbuf);
        return rc;
      }
      rc = sqlite3_blob_write(blob, qbuf, qsize, chunk_offset * qsize);
      sqlite3_free(qbuf);
      int brc = sqlite3_blob_close(blob);
      if (rc != SQLITE_OK)
        return rc;
      if (brc != SQLITE_OK)
        return brc;
    }

    // 2. Insert float vector into _rescore_vectors (rowid-keyed)
    {
      char *zSql = sqlite3_mprintf(
          "INSERT INTO \"%w\".\"%w\"(rowid, vector) VALUES (?, ?)",
          p->schemaName, p->shadowRescoreVectorsNames[i]);
      if (!zSql)
        return SQLITE_NOMEM;
      sqlite3_stmt *stmt;
      rc = sqlite3_prepare_v2(p->db, zSql, -1, &stmt, NULL);
      sqlite3_free(zSql);
      if (rc != SQLITE_OK) {
        sqlite3_finalize(stmt);
        return rc;
      }
      sqlite3_bind_int64(stmt, 1, rowid);
      sqlite3_bind_blob(stmt, 2, vectorDatas[i], fsize, SQLITE_TRANSIENT);
      rc = sqlite3_step(stmt);
      sqlite3_finalize(stmt);
      if (rc != SQLITE_DONE)
        return SQLITE_ERROR;
    }
  }
  return SQLITE_OK;
}

// ============================================================================
// Delete path
// ============================================================================

/**
 * Zero out quantized vector in _rescore_chunks and delete from _rescore_vectors.
 */
static int rescore_on_delete(vec0_vtab *p, i64 chunk_id, u64 chunk_offset,
                             i64 rowid) {
  for (int i = 0; i < p->numVectorColumns; i++) {
    if (p->vector_columns[i].index_type != VEC0_INDEX_TYPE_RESCORE)
      continue;
    int rc;

    // 1. Zero out quantized data in _rescore_chunks
    {
      size_t qsize = rescore_quantized_byte_size(&p->vector_columns[i]);
      void *zeroBuf = sqlite3_malloc(qsize);
      if (!zeroBuf)
        return SQLITE_NOMEM;
      memset(zeroBuf, 0, qsize);

      sqlite3_blob *blob = NULL;
      rc = sqlite3_blob_open(p->db, p->schemaName,
                             p->shadowRescoreChunksNames[i], "vectors",
                             chunk_id, 1, &blob);
      if (rc != SQLITE_OK) {
        sqlite3_free(zeroBuf);
        return rc;
      }
      rc = sqlite3_blob_write(blob, zeroBuf, qsize, chunk_offset * qsize);
      sqlite3_free(zeroBuf);
      int brc = sqlite3_blob_close(blob);
      if (rc != SQLITE_OK)
        return rc;
      if (brc != SQLITE_OK)
        return brc;
    }

    // 2. Delete from _rescore_vectors
    {
      char *zSql = sqlite3_mprintf(
          "DELETE FROM \"%w\".\"%w\" WHERE rowid = ?",
          p->schemaName, p->shadowRescoreVectorsNames[i]);
      if (!zSql)
        return SQLITE_NOMEM;
      sqlite3_stmt *stmt;
      rc = sqlite3_prepare_v2(p->db, zSql, -1, &stmt, NULL);
      sqlite3_free(zSql);
      if (rc != SQLITE_OK)
        return rc;
      sqlite3_bind_int64(stmt, 1, rowid);
      rc = sqlite3_step(stmt);
      sqlite3_finalize(stmt);
      if (rc != SQLITE_DONE)
        return SQLITE_ERROR;
    }
  }
  return SQLITE_OK;
}

/**
 * Delete a chunk row from _rescore_chunks{NN} tables.
 * (_rescore_vectors rows were already deleted per-row in rescore_on_delete)
 */
static int rescore_delete_chunk(vec0_vtab *p, i64 chunk_id) {
  for (int i = 0; i < p->numVectorColumns; i++) {
    if (!p->shadowRescoreChunksNames[i])
      continue;
    char *zSql = sqlite3_mprintf(
        "DELETE FROM \"%w\".\"%w\" WHERE rowid = ?",
        p->schemaName, p->shadowRescoreChunksNames[i]);
    if (!zSql)
      return SQLITE_NOMEM;
    sqlite3_stmt *stmt;
    int rc = sqlite3_prepare_v2(p->db, zSql, -1, &stmt, NULL);
    sqlite3_free(zSql);
    if (rc != SQLITE_OK)
      return rc;
    sqlite3_bind_int64(stmt, 1, chunk_id);
    rc = sqlite3_step(stmt);
    sqlite3_finalize(stmt);
    if (rc != SQLITE_DONE)
      return SQLITE_ERROR;
  }
  return SQLITE_OK;
}

// ============================================================================
// KNN rescore query
// ============================================================================

/**
 * Phase 1: Coarse scan of quantized chunks → top k*oversample candidates (rowids).
 * Phase 2: For each candidate, blob_open _rescore_vectors by rowid, read float
 *          vector, compute float distance. Sort, return top k.
 *
 * Phase 2 is fast because _rescore_vectors has INTEGER PRIMARY KEY, so
 * sqlite3_blob_open/reopen addresses rows directly by rowid — no index lookup.
 */
static int rescore_knn(vec0_vtab *p, vec0_cursor *pCur,
                       struct VectorColumnDefinition *vector_column,
                       int vectorColumnIdx, struct Array *arrayRowidsIn,
                       struct Array *aMetadataIn, const char *idxStr, int argc,
                       sqlite3_value **argv, void *queryVector, i64 k,
                       struct vec0_query_knn_data *knn_data) {
  (void)pCur;
  (void)aMetadataIn;
  int rc = SQLITE_OK;
  int oversample = vector_column->rescore.oversample_search > 0
      ? vector_column->rescore.oversample_search
      : vector_column->rescore.oversample;
  i64 k_oversample = k * oversample;
  if (k_oversample > 4096)
    k_oversample = 4096;

  size_t qdim = vector_column->dimensions;
  size_t qsize = rescore_quantized_byte_size(vector_column);
  size_t fsize = vector_column_byte_size(*vector_column);

  // Quantize the query vector
  void *quantizedQuery = sqlite3_malloc(qsize);
  if (!quantizedQuery)
    return SQLITE_NOMEM;

  switch (vector_column->rescore.quantizer_type) {
  case VEC0_RESCORE_QUANTIZER_BIT:
    rescore_quantize_float_to_bit((const float *)queryVector,
                                  (uint8_t *)quantizedQuery, qdim);
    break;
  case VEC0_RESCORE_QUANTIZER_INT8:
    rescore_quantize_float_to_int8((const float *)queryVector,
                                   (int8_t *)quantizedQuery, qdim);
    break;
  }

  // Phase 1: Scan quantized chunks for k*oversample candidates
  sqlite3_stmt *stmtChunks = NULL;
  rc = vec0_chunks_iter(p, idxStr, argc, argv, &stmtChunks);
  if (rc != SQLITE_OK) {
    sqlite3_free(quantizedQuery);
    return rc;
  }

  i64 *cand_rowids = sqlite3_malloc(k_oversample * sizeof(i64));
  f32 *cand_distances = sqlite3_malloc(k_oversample * sizeof(f32));
  i64 *tmp_rowids = sqlite3_malloc(k_oversample * sizeof(i64));
  f32 *tmp_distances = sqlite3_malloc(k_oversample * sizeof(f32));
  f32 *chunk_distances = sqlite3_malloc(p->chunk_size * sizeof(f32));
  i32 *chunk_topk_idxs = sqlite3_malloc(k_oversample * sizeof(i32));
  u8 *b = sqlite3_malloc(p->chunk_size / CHAR_BIT);
  u8 *bmRowids = NULL;
  void *baseVectors = sqlite3_malloc((i64)p->chunk_size * (i64)qsize);

  if (!cand_rowids || !cand_distances || !tmp_rowids || !tmp_distances ||
      !chunk_distances || !chunk_topk_idxs || !b || !baseVectors) {
    rc = SQLITE_NOMEM;
    goto cleanup;
  }
  memset(cand_rowids, 0, k_oversample * sizeof(i64));
  memset(cand_distances, 0, k_oversample * sizeof(f32));

  if (arrayRowidsIn) {
    bmRowids = sqlite3_malloc(p->chunk_size / CHAR_BIT);
    if (!bmRowids) {
      rc = SQLITE_NOMEM;
      goto cleanup;
    }
  }

  i64 cand_used = 0;

  while (1) {
    rc = sqlite3_step(stmtChunks);
    if (rc == SQLITE_DONE)
      break;
    if (rc != SQLITE_ROW) {
      rc = SQLITE_ERROR;
      goto cleanup;
    }

    i64 chunk_id = sqlite3_column_int64(stmtChunks, 0);
    unsigned char *chunkValidity =
        (unsigned char *)sqlite3_column_blob(stmtChunks, 1);
    i64 *chunkRowids = (i64 *)sqlite3_column_blob(stmtChunks, 2);
    int validityBytes = sqlite3_column_bytes(stmtChunks, 1);
    int rowidsBytes = sqlite3_column_bytes(stmtChunks, 2);
    if (!chunkValidity || !chunkRowids) {
      rc = SQLITE_ERROR;
      goto cleanup;
    }
    // Validate blob sizes match chunk_size expectations
    if (validityBytes < (p->chunk_size + 7) / 8 ||
        rowidsBytes < p->chunk_size * (int)sizeof(i64)) {
      rc = SQLITE_ERROR;
      goto cleanup;
    }

    memset(chunk_distances, 0, p->chunk_size * sizeof(f32));
    memset(chunk_topk_idxs, 0, k_oversample * sizeof(i32));
    bitmap_copy(b, chunkValidity, p->chunk_size);

    if (arrayRowidsIn) {
      bitmap_clear(bmRowids, p->chunk_size);
      for (int j = 0; j < p->chunk_size; j++) {
        if (!bitmap_get(chunkValidity, j))
          continue;
        bitmap_set(bmRowids, j, vec0_rowids_contains(arrayRowidsIn, chunkRowids[j]));
      }
      bitmap_and_inplace(b, bmRowids, p->chunk_size);
    }

    // Read quantized vectors
    sqlite3_blob *blobQ = NULL;
    rc = sqlite3_blob_open(p->db, p->schemaName,
                           p->shadowRescoreChunksNames[vectorColumnIdx],
                           "vectors", chunk_id, 0, &blobQ);
    if (rc != SQLITE_OK)
      goto cleanup;
    rc = sqlite3_blob_read(blobQ, baseVectors,
                           (i64)p->chunk_size * (i64)qsize, 0);
    sqlite3_blob_close(blobQ);
    if (rc != SQLITE_OK)
      goto cleanup;

    // Compute quantized distances
    for (int j = 0; j < p->chunk_size; j++) {
      if (!bitmap_get(b, j))
        continue;
      f32 dist = FLT_MAX;
      switch (vector_column->rescore.quantizer_type) {
      case VEC0_RESCORE_QUANTIZER_BIT: {
        const u8 *base_j = ((u8 *)baseVectors) + (j * (qdim / CHAR_BIT));
        dist = distance_hamming(base_j, (u8 *)quantizedQuery, &qdim);
        break;
      }
      case VEC0_RESCORE_QUANTIZER_INT8: {
        const i8 *base_j = ((i8 *)baseVectors) + (j * qdim);
        dist = vec0_distance_full(base_j, quantizedQuery, qdim,
                                  SQLITE_VEC_ELEMENT_TYPE_INT8,
                                  vector_column->distance_metric);
        break;
      }
      }
      chunk_distances[j] = dist;
    }

    int used1;
    min_idx(chunk_distances, p->chunk_size, b, chunk_topk_idxs,
            min(k_oversample, p->chunk_size), &used1);

    i64 merged_used;
    merge_sorted_lists(cand_distances, cand_rowids, cand_used, chunk_distances,
                       chunkRowids, chunk_topk_idxs,
                       min(min(k_oversample, p->chunk_size), used1),
                       tmp_distances, tmp_rowids, k_oversample, &merged_used);

    for (i64 j = 0; j < merged_used; j++) {
      cand_rowids[j] = tmp_rowids[j];
      cand_distances[j] = tmp_distances[j];
    }
    cand_used = merged_used;
  }
  rc = SQLITE_OK;

  // Phase 2: Rescore candidates using _rescore_vectors (rowid-keyed)
  if (cand_used == 0) {
    knn_data->current_idx = 0;
    knn_data->k = 0;
    knn_data->rowids = NULL;
    knn_data->distances = NULL;
    knn_data->k_used = 0;
    goto cleanup;
  }
  {
    f32 *float_distances = sqlite3_malloc(cand_used * sizeof(f32));
    void *fBuf = sqlite3_malloc(fsize);
    if (!float_distances || !fBuf) {
      sqlite3_free(float_distances);
      sqlite3_free(fBuf);
      rc = SQLITE_NOMEM;
      goto cleanup;
    }

    // Open blob on _rescore_vectors, then reopen for each candidate rowid.
    // blob_reopen is O(1) for INTEGER PRIMARY KEY tables.
    sqlite3_blob *blobFloat = NULL;
    rc = sqlite3_blob_open(p->db, p->schemaName,
                           p->shadowRescoreVectorsNames[vectorColumnIdx],
                           "vector", cand_rowids[0], 0, &blobFloat);
    if (rc != SQLITE_OK) {
      sqlite3_free(float_distances);
      sqlite3_free(fBuf);
      goto cleanup;
    }

    rc = sqlite3_blob_read(blobFloat, fBuf, fsize, 0);
    if (rc != SQLITE_OK) {
      sqlite3_blob_close(blobFloat);
      sqlite3_free(float_distances);
      sqlite3_free(fBuf);
      goto cleanup;
    }
    float_distances[0] =
        vec0_distance_full(fBuf, queryVector, vector_column->dimensions,
                           vector_column->element_type,
                           vector_column->distance_metric);

    for (i64 j = 1; j < cand_used; j++) {
      rc = sqlite3_blob_reopen(blobFloat, cand_rowids[j]);
      if (rc != SQLITE_OK) {
        sqlite3_blob_close(blobFloat);
        sqlite3_free(float_distances);
        sqlite3_free(fBuf);
        goto cleanup;
      }
      rc = sqlite3_blob_read(blobFloat, fBuf, fsize, 0);
      if (rc != SQLITE_OK) {
        sqlite3_blob_close(blobFloat);
        sqlite3_free(float_distances);
        sqlite3_free(fBuf);
        goto cleanup;
      }
      float_distances[j] =
          vec0_distance_full(fBuf, queryVector, vector_column->dimensions,
                             vector_column->element_type,
                             vector_column->distance_metric);
    }
    sqlite3_blob_close(blobFloat);
    sqlite3_free(fBuf);

    // Sort by float distance
    for (i64 a = 0; a + 1 < cand_used; a++) {
      i64 minIdx = a;
      for (i64 c = a + 1; c < cand_used; c++) {
        if (float_distances[c] < float_distances[minIdx])
          minIdx = c;
      }
      if (minIdx != a) {
        f32 td = float_distances[a];
        float_distances[a] = float_distances[minIdx];
        float_distances[minIdx] = td;
        i64 tr = cand_rowids[a];
        cand_rowids[a] = cand_rowids[minIdx];
        cand_rowids[minIdx] = tr;
      }
    }

    // Apply any `distance` constraints from the WHERE clause. This must happen
    // on the rescored float distances (the coarse quantized distances from
    // phase 1 are not comparable to a user-supplied threshold), and before the
    // top-k truncation below, so that a constraint like `distance > x` can
    // still yield k rows instead of dropping the head of the result set.
    i64 cand_kept = 0;
    for (i64 j = 0; j < cand_used; j++) {
      if (!vec0_distance_constraints_satisfied(float_distances[j], idxStr, argc,
                                               argv)) {
        continue;
      }
      cand_rowids[cand_kept] = cand_rowids[j];
      float_distances[cand_kept] = float_distances[j];
      cand_kept++;
    }
    cand_used = cand_kept;

    // Every candidate was filtered out: return an empty result set. Falling
    // through would sqlite3_malloc(0), which returns NULL and would be
    // misreported as SQLITE_NOMEM below.
    if (cand_used == 0) {
      knn_data->current_idx = 0;
      knn_data->k = 0;
      knn_data->rowids = NULL;
      knn_data->distances = NULL;
      knn_data->k_used = 0;
      sqlite3_free(float_distances);
      goto cleanup;
    }

    i64 result_k = min(k, cand_used);
    i64 *out_rowids = sqlite3_malloc(result_k * sizeof(i64));
    f32 *out_distances = sqlite3_malloc(result_k * sizeof(f32));
    if (!out_rowids || !out_distances) {
      sqlite3_free(out_rowids);
      sqlite3_free(out_distances);
      sqlite3_free(float_distances);
      rc = SQLITE_NOMEM;
      goto cleanup;
    }
    for (i64 j = 0; j < result_k; j++) {
      out_rowids[j] = cand_rowids[j];
      out_distances[j] = float_distances[j];
    }

    knn_data->current_idx = 0;
    knn_data->k = result_k;
    knn_data->rowids = out_rowids;
    knn_data->distances = out_distances;
    knn_data->k_used = result_k;

    sqlite3_free(float_distances);
  }

cleanup:
  sqlite3_finalize(stmtChunks);
  sqlite3_free(quantizedQuery);
  sqlite3_free(cand_rowids);
  sqlite3_free(cand_distances);
  sqlite3_free(tmp_rowids);
  sqlite3_free(tmp_distances);
  sqlite3_free(chunk_distances);
  sqlite3_free(chunk_topk_idxs);
  sqlite3_free(b);
  sqlite3_free(bmRowids);
  sqlite3_free(baseVectors);
  return rc;
}

/**
 * Handle FTS5-style command dispatch for rescore parameters.
 * Returns SQLITE_OK if handled, SQLITE_EMPTY if not a rescore command.
 */
static int rescore_handle_command(vec0_vtab *p, const char *command) {
  if (strncmp(command, "oversample=", 11) == 0) {
    int val = atoi(command + 11);
    if (val < 1) {
      vtab_set_error(&p->base, "oversample must be >= 1");
      return SQLITE_ERROR;
    }
    for (int i = 0; i < p->numVectorColumns; i++) {
      if (p->vector_columns[i].index_type == VEC0_INDEX_TYPE_RESCORE) {
        p->vector_columns[i].rescore.oversample_search = val;
      }
    }
    return SQLITE_OK;
  }
  return SQLITE_EMPTY;
}

#ifdef SQLITE_VEC_TEST
void _test_rescore_quantize_float_to_bit(const float *src, uint8_t *dst, size_t dim) {
  rescore_quantize_float_to_bit(src, dst, dim);
}
void _test_rescore_quantize_float_to_int8(const float *src, int8_t *dst, size_t dim) {
  rescore_quantize_float_to_int8(src, dst, dim);
}
size_t _test_rescore_quantized_byte_size_bit(size_t dimensions) {
  struct VectorColumnDefinition col;
  memset(&col, 0, sizeof(col));
  col.dimensions = dimensions;
  col.rescore.quantizer_type = VEC0_RESCORE_QUANTIZER_BIT;
  return rescore_quantized_byte_size(&col);
}
size_t _test_rescore_quantized_byte_size_int8(size_t dimensions) {
  struct VectorColumnDefinition col;
  memset(&col, 0, sizeof(col));
  col.dimensions = dimensions;
  col.rescore.quantizer_type = VEC0_RESCORE_QUANTIZER_INT8;
  return rescore_quantized_byte_size(&col);
}
#endif
//////////////////////////////////////////////////////////////////////////////
// End inlined: sqlite-vec-rescore.c
//////////////////////////////////////////////////////////////////////////////

#endif

#if SQLITE_VEC_ENABLE_DISKANN
#if COMPILER_SUPPORTS_VTAB_IN
/**
 * Rank the rows a `rowid in (...)` constraint names by their exact distances
 * to the query, keeping the k nearest in ascending order of distance. A rowid
 * or id the table does not have is skipped. The vectors are read from
 * _vectorsNN, which holds the buffered rows as well as the graph's.
 */
static int diskann_knn_rowids_in(vec0_vtab *p, int vec_col_idx,
                                 const struct Array *rowidsIn,
                                 const void *queryVector, size_t dimensions,
                                 enum VectorElementType elementType, i64 k,
                                 i64 *outRowids, f32 *outDistances,
                                 int *outCount) {
  struct VectorColumnDefinition *col = &p->vector_columns[vec_col_idx];
  int vectorSize = (int)vector_column_byte_size(*col);
  sqlite3_stmt *stmt = NULL;
  int count = 0;
  int rc = SQLITE_DONE;
  char *zSql = sqlite3_mprintf(
      "SELECT vector FROM " VEC0_SHADOW_VECTORS_N_NAME " WHERE rowid = ?",
      p->schemaName, p->tableName, vec_col_idx);
  if (!zSql) return SQLITE_NOMEM;
  rc = sqlite3_prepare_v2(p->db, zSql, -1, &stmt, NULL);
  sqlite3_free(zSql);
  if (rc != SQLITE_OK) return rc;

  for (size_t n = 0; n < rowidsIn->length; n++) {
    i64 rowid = ((const i64 *)rowidsIn->z)[n];
    sqlite3_reset(stmt);
    sqlite3_bind_int64(stmt, 1, rowid);
    rc = sqlite3_step(stmt);
    if (rc == SQLITE_DONE) continue;
    if (rc != SQLITE_ROW) break;
    rc = SQLITE_DONE;
    if (sqlite3_column_bytes(stmt, 0) != vectorSize) {
      rc = SQLITE_CORRUPT;
      break;
    }
    f32 distance = vec0_distance_full(queryVector, sqlite3_column_blob(stmt, 0),
                                      dimensions, elementType,
                                      col->distance_metric);

    if (count == k && distance >= outDistances[count - 1]) continue;
    int i = count < k ? count++ : count - 1;
    while (i > 0 && outDistances[i - 1] > distance) {
      outRowids[i] = outRowids[i - 1];
      outDistances[i] = outDistances[i - 1];
      i--;
    }
    outRowids[i] = rowid;
    outDistances[i] = distance;
  }
  sqlite3_finalize(stmt);
  if (rc != SQLITE_DONE) return rc;
  *outCount = count;
  return SQLITE_OK;
}
#endif

/**
 * Handle a KNN query using the DiskANN graph search.
 */
static int vec0Filter_knn_diskann(
    vec0_cursor *pCur, vec0_vtab *p, int idxNum,
    const char *idxStr, int argc, sqlite3_value **argv) {

  int rc;
  int vectorColumnIdx = idxNum;
  struct VectorColumnDefinition *vector_column = &p->vector_columns[vectorColumnIdx];
  struct vec0_query_knn_data *knn_data;

  knn_data = sqlite3_malloc(sizeof(*knn_data));
  if (!knn_data) return SQLITE_NOMEM;
  memset(knn_data, 0, sizeof(*knn_data));

  // Parse query_idx, k_idx and rowid_in_idx from idxStr
  int query_idx = -1;
  int k_idx = -1;
  int offset_idx = -1;
  int rowid_in_idx = -1;
  for (int i = 0; i < argc; i++) {
    if (idxStr[1 + (i * 4)] == VEC0_IDXSTR_KIND_KNN_MATCH) {
      query_idx = i;
    }
    if (idxStr[1 + (i * 4)] == VEC0_IDXSTR_KIND_KNN_K) {
      k_idx = i;
    }
    if (idxStr[1 + (i * 4)] == VEC0_IDXSTR_KIND_KNN_OFFSET) {
      offset_idx = i;
    }
    if (idxStr[1 + (i * 4)] == VEC0_IDXSTR_KIND_KNN_ROWID_IN) {
      rowid_in_idx = i;
    }
  }
  assert(query_idx >= 0);
  assert(k_idx >= 0);

  // Extract query vector
  void *queryVector;
  size_t dimensions;
  enum VectorElementType elementType;
  vector_cleanup queryVectorCleanup = vector_cleanup_noop;
  char *pzError;

  rc = vector_from_value(argv[query_idx], &queryVector, &dimensions,
                          &elementType, &queryVectorCleanup, &pzError);
  if (rc != SQLITE_OK) {
    vtab_set_error(&p->base, "Invalid query vector: %z", pzError);
    sqlite3_free(knn_data);
    return SQLITE_ERROR;
  }

  if (elementType != vector_column->element_type ||
      dimensions != vector_column->dimensions) {
    vtab_set_error(&p->base, "Query vector type/dimension mismatch");
    queryVectorCleanup(queryVector);
    sqlite3_free(knn_data);
    return SQLITE_ERROR;
  }

  i64 k = sqlite3_value_int64(argv[k_idx]);
  if (k <= 0) {
    knn_data->k = 0;
    knn_data->k_used = 0;
    pCur->knn_data = knn_data;
    pCur->query_plan = VEC0_QUERY_PLAN_KNN;
    queryVectorCleanup(queryVector);
    return SQLITE_OK;
  }
  if (offset_idx >= 0) {
    i64 knn_offset = sqlite3_value_int64(argv[offset_idx]);
    if (knn_offset > 0) {
      k += knn_offset;
    }
  }

  // Run DiskANN search
  i64 *resultRowids = sqlite3_malloc(k * sizeof(i64));
  f32 *resultDistances = sqlite3_malloc(k * sizeof(f32));
  if (!resultRowids || !resultDistances) {
    sqlite3_free(resultRowids);
    sqlite3_free(resultDistances);
    queryVectorCleanup(queryVector);
    sqlite3_free(knn_data);
    return SQLITE_NOMEM;
  }

  int resultCount = 0;
  struct Array *rowidsIn = NULL;
  rc = SQLITE_OK;
#if COMPILER_SUPPORTS_VTAB_IN
  if (rowid_in_idx >= 0) {
    rc = vec0_rowids_in_collect(p, argv[rowid_in_idx], &rowidsIn);
  }
#endif
  // A small rowid filter names its candidates, so each is ranked by its
  // exact distance from _vectorsNN, which holds the buffered rows too. A
  // larger one is applied during the graph search and the buffer scan.
  if (rc == SQLITE_OK && vec0_rowids_rank_exact(rowidsIn)) {
    rc = diskann_knn_rowids_in(p, vectorColumnIdx, rowidsIn,
                               queryVector, dimensions, elementType, k,
                               resultRowids, resultDistances, &resultCount);
  } else if (rc == SQLITE_OK) {
    rc = diskann_search(p, vectorColumnIdx, queryVector, dimensions,
                        elementType, (int)k, 0, rowidsIn,
                        resultRowids, resultDistances, &resultCount);
  }

  if (rc != SQLITE_OK) {
    if (rowidsIn) array_cleanup(rowidsIn);
    sqlite3_free(rowidsIn);
    queryVectorCleanup(queryVector);
    sqlite3_free(resultRowids);
    sqlite3_free(resultDistances);
    sqlite3_free(knn_data);
    return rc;
  }

  // Scan _diskann_buffer for any buffered (unflushed) vectors and merge
  // with graph results. This ensures no recall loss for buffered vectors.
  // A small rowid filter has already ranked the buffered rows it names.
  if (!vec0_rowids_rank_exact(rowidsIn)) {
    sqlite3_stmt *bufStmt = NULL;
    char *zSql = sqlite3_mprintf(
        "SELECT rowid, vector FROM " VEC0_SHADOW_DISKANN_BUFFER_N_NAME,
        p->schemaName, p->tableName, vectorColumnIdx);
    if (!zSql) {
      if (rowidsIn) array_cleanup(rowidsIn);
      sqlite3_free(rowidsIn);
      queryVectorCleanup(queryVector);
      sqlite3_free(resultRowids);
      sqlite3_free(resultDistances);
      sqlite3_free(knn_data);
      return SQLITE_NOMEM;
    }
    int bufRc = sqlite3_prepare_v2(p->db, zSql, -1, &bufStmt, NULL);
    sqlite3_free(zSql);
    if (bufRc == SQLITE_OK) {
      while (sqlite3_step(bufStmt) == SQLITE_ROW) {
        i64 bufRowid = sqlite3_column_int64(bufStmt, 0);
        if (!vec0_rowids_contains(rowidsIn, bufRowid)) continue;
        const void *bufVec = sqlite3_column_blob(bufStmt, 1);
        f32 dist = vec0_distance_full(
            queryVector, bufVec, dimensions, elementType,
            vector_column->distance_metric);

        // Check if this buffer vector should replace the worst graph result
        if (resultCount < (int)k) {
          // Still have room, just add it
          resultRowids[resultCount] = bufRowid;
          resultDistances[resultCount] = dist;
          resultCount++;
        } else {
          // Find worst (largest distance) in results
          int worstIdx = 0;
          for (int wi = 1; wi < resultCount; wi++) {
            if (resultDistances[wi] > resultDistances[worstIdx]) {
              worstIdx = wi;
            }
          }
          if (dist < resultDistances[worstIdx]) {
            resultRowids[worstIdx] = bufRowid;
            resultDistances[worstIdx] = dist;
          }
        }
      }
      sqlite3_finalize(bufStmt);
    }
  }

  if (rowidsIn) array_cleanup(rowidsIn);
  sqlite3_free(rowidsIn);
  queryVectorCleanup(queryVector);

  // Sort results by distance (ascending)
  for (int si = 0; si < resultCount - 1; si++) {
    for (int sj = si + 1; sj < resultCount; sj++) {
      if (resultDistances[sj] < resultDistances[si]) {
        f32 tmpD = resultDistances[si];
        resultDistances[si] = resultDistances[sj];
        resultDistances[sj] = tmpD;
        i64 tmpR = resultRowids[si];
        resultRowids[si] = resultRowids[sj];
        resultRowids[sj] = tmpR;
      }
    }
  }

  // Apply any `distance` constraints from the WHERE clause. vec0BestIndex()
  // omits these from the query plan, so SQLite will not re-check them.
  {
    int kept = 0;
    for (int si = 0; si < resultCount; si++) {
      if (!vec0_distance_constraints_satisfied(resultDistances[si], idxStr,
                                               argc, argv)) {
        continue;
      }
      resultRowids[kept] = resultRowids[si];
      resultDistances[kept] = resultDistances[si];
      kept++;
    }
    resultCount = kept;
  }

  knn_data->k = resultCount;
  knn_data->k_used = resultCount;
  knn_data->rowids = resultRowids;
  knn_data->distances = resultDistances;
  knn_data->current_idx = 0;

  pCur->knn_data = knn_data;
  pCur->query_plan = VEC0_QUERY_PLAN_KNN;

  return SQLITE_OK;
}
#endif /* SQLITE_VEC_ENABLE_DISKANN */


/**
 * MMR greedy reranking of KNN results.
 *
 * Loads vectors for top-k candidates, then iteratively selects the
 * candidate with the best MMR score:
 *   MMR(d) = lambda * relevance(d) - (1-lambda) * max_sim(d, S)
 *
 * where relevance = 1 - normalized_distance, and max_sim is the maximum
 * cosine similarity between d and any already-selected result.
 *
 * Reorders topk_rowids and topk_distances in place.
 * After return, the first *out_n_selected entries are the MMR-selected results.
 * out_n_selected may be less than k_target if the greedy selection exhausts
 * candidates early.
 */
static int vec0_mmr_rerank(
    vec0_vtab *p,
    int vectorColumnIdx,
    struct VectorColumnDefinition *vector_column,
    i64 *topk_rowids,
    f32 *topk_distances,
    i64 k_used,
    i64 k_target,
    f32 mmr_lambda,
    i64 *out_n_selected
) {
    int rc = SQLITE_OK;

    // 1. Allocate vector storage for all candidates
    void **vectors = sqlite3_malloc64(k_used * sizeof(void *));
    if (!vectors) return SQLITE_NOMEM;
    memset(vectors, 0, k_used * sizeof(void *));

    f32 *relevance = NULL;
    i64 *out_rowids = NULL;
    f32 *out_distances = NULL;
    f32 *max_sim = NULL;
    u8 *selected = NULL;

    // 2. Load vectors from shadow tables
    for (i64 i = 0; i < k_used; i++) {
        rc = vec0_get_vector_data(p, topk_rowids[i], vectorColumnIdx,
                                  &vectors[i], NULL);
        if (rc != SQLITE_OK) goto cleanup;
    }

    // 3. Normalize distances to [0, 1] for relevance scoring
    f32 max_dist = 0.0f;
    for (i64 i = 0; i < k_used; i++) {
        if (topk_distances[i] > max_dist) max_dist = topk_distances[i];
    }
    if (max_dist < 1e-9f) max_dist = 1.0f;

    relevance = sqlite3_malloc64(k_used * sizeof(f32));
    if (!relevance) { rc = SQLITE_NOMEM; goto cleanup; }
    for (i64 i = 0; i < k_used; i++) {
        relevance[i] = 1.0f - (topk_distances[i] / max_dist);
    }

    // 4. Greedy MMR selection. max_sim[i] is candidate i's highest similarity
    // to the results selected so far, raised against each new selection
    // alone, so a step measures each remaining candidate once.
    out_rowids = sqlite3_malloc64(k_target * sizeof(i64));
    out_distances = sqlite3_malloc64(k_target * sizeof(f32));
    max_sim = sqlite3_malloc64(k_used * sizeof(f32));
    selected = sqlite3_malloc64(k_used);
    if (!out_rowids || !out_distances || !max_sim || !selected) {
        rc = SQLITE_NOMEM; goto cleanup;
    }
    memset(selected, 0, k_used);
    for (i64 i = 0; i < k_used; i++) {
        max_sim[i] = 0.0f;
    }

    i64 n_selected = 0;
    for (i64 step = 0; step < k_target && step < k_used; step++) {
        f32 best_mmr = -FLT_MAX;
        i64 best_idx = -1;

        for (i64 i = 0; i < k_used; i++) {
            if (selected[i]) continue;

            f32 mmr_score = mmr_lambda * relevance[i]
                          - (1.0f - mmr_lambda) * max_sim[i];
            if (mmr_score > best_mmr) {
                best_mmr = mmr_score;
                best_idx = i;
            }
        }

        if (best_idx < 0) break;
        selected[best_idx] = 1;
        out_rowids[step] = topk_rowids[best_idx];
        out_distances[step] = topk_distances[best_idx];
        n_selected++;

        for (i64 i = 0; i < k_used; i++) {
            if (selected[i]) continue;
            f32 d = vec0_compute_distance(vector_column,
                                          vectors[i], vectors[best_idx]);
            f32 sim = 1.0f - (d / max_dist);
            if (sim > max_sim[i]) max_sim[i] = sim;
        }
    }

    // 5. Copy only the actually-selected entries back to input arrays
    for (i64 i = 0; i < n_selected; i++) {
        topk_rowids[i] = out_rowids[i];
        topk_distances[i] = out_distances[i];
    }
    *out_n_selected = n_selected;

cleanup:
    if (vectors) {
        for (i64 i = 0; i < k_used; i++) {
            sqlite3_free(vectors[i]);
        }
        sqlite3_free(vectors);
    }
    sqlite3_free(relevance);
    sqlite3_free(out_rowids);
    sqlite3_free(out_distances);
    sqlite3_free(max_sim);
    sqlite3_free(selected);
    return rc;
}

/**
 * Flat KNN over vectors held by a storage provider: stream the column with
 * xScan and keep the k nearest that pass the rowid filter and the query's
 * distance constraints (so k counts only rows that qualify).
 */
struct vec0_knn_scan {
  struct VectorColumnDefinition *col;
  const void *query;
  vec0_distance_fn dist;
  const struct Array *rowidsIn;
  const char *idxStr;
  int argc;
  sqlite3_value **argv;
  i64 k, n;
  i64 *rowids;
  f32 *distances;
};

static int vec0_knn_scan_row(void *pArg, sqlite3_int64 rowid, const void *data, int nData) {
  struct vec0_knn_scan *s = pArg;
  if (!data || nData != (int)vector_column_byte_size(*s->col)) return SQLITE_OK;
  if (!vec0_rowids_contains(s->rowidsIn, rowid)) return SQLITE_OK;
  f32 d = s->dist(s->query, data, &s->col->dimensions);
  if (!vec0_distance_constraints_satisfied(d, s->idxStr, s->argc, s->argv)) return SQLITE_OK;
  if (s->n == s->k && d >= s->distances[s->n - 1]) return SQLITE_OK;
  i64 i = s->n < s->k ? s->n++ : s->n - 1;
  while (i > 0 && s->distances[i - 1] > d) {
    s->rowids[i] = s->rowids[i - 1];
    s->distances[i] = s->distances[i - 1];
    i--;
  }
  s->rowids[i] = rowid;
  s->distances[i] = d;
  return SQLITE_OK;
}

static int vec0_knn_storage(vec0_vtab *p, int col, const void *query,
                            const struct Array *rowidsIn, i64 k, const char *idxStr,
                            int argc, sqlite3_value **argv,
                            struct vec0_query_knn_data *knn) {
  struct vec0_knn_scan s;
  memset(&s, 0, sizeof(s));
  s.col = &p->vector_columns[col];
  s.query = query;
  s.dist = vec0_distance_resolve(s.col->element_type, s.col->distance_metric);
  s.rowidsIn = rowidsIn;
  s.idxStr = idxStr;
  s.argc = argc;
  s.argv = argv;
  s.k = k;
  s.rowids = sqlite3_malloc64(k * sizeof(i64));
  s.distances = sqlite3_malloc64(k * sizeof(f32));
  if (!s.rowids || !s.distances) {
    sqlite3_free(s.rowids);
    sqlite3_free(s.distances);
    return SQLITE_NOMEM;
  }
  int rc = vec0_storage_error(p, p->storage->xScan(p->storageHandle, p->numAuxiliaryColumns + col,
                                                   vec0_knn_scan_row, &s));
  if (rc != SQLITE_OK) {
    sqlite3_free(s.rowids);
    sqlite3_free(s.distances);
    return rc;
  }
  memset(knn, 0, sizeof(*knn));
  knn->rowids = s.rowids;
  knn->distances = s.distances;
  knn->k = k;
  knn->k_used = s.n;
  return SQLITE_OK;
}

int vec0Filter_knn(vec0_cursor *pCur, vec0_vtab *p, int idxNum,
                   const char *idxStr, int argc, sqlite3_value **argv) {
  assert(argc == (strlen(idxStr)-1) / 4);
  int rc;
  struct vec0_query_knn_data *knn_data;

  int vectorColumnIdx = idxNum;
  struct VectorColumnDefinition *vector_column =
      &p->vector_columns[vectorColumnIdx];

#if SQLITE_VEC_ENABLE_DISKANN
  // DiskANN dispatch
  if (vector_column->index_type == VEC0_INDEX_TYPE_DISKANN) {
    return vec0Filter_knn_diskann(pCur, p, idxNum, idxStr, argc, argv);
  }
#endif

  struct Array *arrayRowidsIn = NULL;
  sqlite3_stmt *stmtChunks = NULL;
  void *queryVector;
  size_t dimensions;
  enum VectorElementType elementType;
  vector_cleanup queryVectorCleanup = vector_cleanup_noop;
  char *pzError;
  knn_data = sqlite3_malloc(sizeof(*knn_data));
  if (!knn_data) {
    return SQLITE_NOMEM;
  }
  memset(knn_data, 0, sizeof(*knn_data));
  // array of `struct Vec0MetadataIn`, IF there are any `xxx in (...)` metadata constraints
  struct Array * aMetadataIn = NULL;

  int query_idx =-1;
  int k_idx = -1;
  int offset_idx = -1;
  int rowid_in_idx = -1;
  int mmr_lambda_idx = -1;
  for(int i = 0; i < argc; i++) {
    if(idxStr[1 + (i*4)] == VEC0_IDXSTR_KIND_KNN_MATCH) {
      query_idx = i;
    }
    if(idxStr[1 + (i*4)] == VEC0_IDXSTR_KIND_KNN_K) {
      k_idx = i;
    }
    if(idxStr[1 + (i*4)] == VEC0_IDXSTR_KIND_KNN_OFFSET) {
      offset_idx = i;
    }
    if(idxStr[1 + (i*4)] == VEC0_IDXSTR_KIND_KNN_ROWID_IN) {
      rowid_in_idx = i;
    }
    if(idxStr[1 + (i*4)] == VEC0_IDXSTR_KIND_KNN_MMR_LAMBDA) {
      mmr_lambda_idx = i;
    }
  }
  assert(query_idx >= 0);
  assert(k_idx >= 0);

  // make sure the query vector matches the vector column (type dimensions etc.)
  rc = vector_from_value(argv[query_idx], &queryVector, &dimensions, &elementType,
                         &queryVectorCleanup, &pzError);

  if (rc != SQLITE_OK) {
    vtab_set_error(&p->base,
                   "Query vector on the \"%.*s\" column is invalid: %z",
                   vector_column->name_length, vector_column->name, pzError);
    rc = SQLITE_ERROR;
    goto cleanup;
  }
  if (elementType != vector_column->element_type) {
    vtab_set_error(
        &p->base,
        "Query vector for the \"%.*s\" column is expected to be of type "
        "%s, but a %s vector was provided.",
        vector_column->name_length, vector_column->name,
        vector_subtype_name(vector_column->element_type),
        vector_subtype_name(elementType));
    rc = SQLITE_ERROR;
    goto cleanup;
  }
  if (dimensions != vector_column->dimensions) {
    vtab_set_error(
        &p->base,
        "Dimension mismatch for query vector for the \"%.*s\" column. "
        "Expected %d dimensions but received %d.",
        vector_column->name_length, vector_column->name,
        vector_column->dimensions, dimensions);
    rc = SQLITE_ERROR;
    goto cleanup;
  }

  i64 k = sqlite3_value_int64(argv[k_idx]);
  if (k < 0) {
    vtab_set_error(
        &p->base, "k value in knn queries must be greater than or equal to 0.");
    rc = SQLITE_ERROR;
    goto cleanup;
  }
#define SQLITE_VEC_VEC0_K_MAX 4096
  if (k > SQLITE_VEC_VEC0_K_MAX) {
    vtab_set_error(
        &p->base,
        "k value in knn query too large, provided %lld and the limit is %lld",
        k, SQLITE_VEC_VEC0_K_MAX);
    rc = SQLITE_ERROR;
    goto cleanup;
  }

  if (k == 0) {
    knn_data->k = 0;
    pCur->knn_data = knn_data;
    pCur->query_plan = VEC0_QUERY_PLAN_KNN;
    rc = SQLITE_OK;
    goto cleanup;
  }

  // `LIMIT n OFFSET m`: SQLite applies the OFFSET (omit = 0), so fetch
  // n + m rows for it to skip into. Negative OFFSET is 0, as in SQLite.
  if (offset_idx >= 0) {
    i64 knn_offset = sqlite3_value_int64(argv[offset_idx]);
    if (knn_offset < 0) {
      knn_offset = 0;
    }
    if (knn_offset > SQLITE_VEC_VEC0_K_MAX - k) {
      vtab_set_error(
          &p->base,
          "k value in knn query too large, provided %lld and the limit is %lld",
          knn_offset >= SQLITE_VEC_VEC0_K_MAX ? knn_offset : k + knn_offset,
          SQLITE_VEC_VEC0_K_MAX);
      rc = SQLITE_ERROR;
      goto cleanup;
    }
    k += knn_offset;
  }

  // MMR: validate lambda and over-fetch candidates
#define SQLITE_VEC_MMR_OVERFETCH_FACTOR 5
  f32 mmr_lambda = -1.0f;
  i64 k_original = k;
  if (mmr_lambda_idx >= 0) {
    mmr_lambda = (f32)sqlite3_value_double(argv[mmr_lambda_idx]);
    if (mmr_lambda < 0.0f || mmr_lambda > 1.0f) {
      vtab_set_error(
          &p->base,
          "mmr_lambda value in knn query must be between 0.0 and 1.0, "
          "provided %f",
          (double)mmr_lambda);
      rc = SQLITE_ERROR;
      goto cleanup;
    }
    if (mmr_lambda < 1.0f) {
      i64 k_internal = k * SQLITE_VEC_MMR_OVERFETCH_FACTOR;
      if (k_internal > SQLITE_VEC_VEC0_K_MAX) k_internal = SQLITE_VEC_VEC0_K_MAX;
      if (k_internal < k) k_internal = k;  // overflow guard
      k = k_internal;
    }
  }

// handle when a `rowid in (...)` operation was provided
// Array of all the rowids that appear in any `rowid in (...)` constraint.
// NULL if none were provided, which means a "full" scan.
#if COMPILER_SUPPORTS_VTAB_IN
  if (rowid_in_idx >= 0) {
    rc = vec0_rowids_in_collect(p, argv[rowid_in_idx], &arrayRowidsIn);
    if (rc != SQLITE_OK) {
      vtab_set_error(&p->base, "error processing rowid in (...) array");
      goto cleanup;
    }
  }
#endif

  #if COMPILER_SUPPORTS_VTAB_IN
  for(int i = 0; i < argc; i++) {
    if(!(idxStr[1 + (i*4)] == VEC0_IDXSTR_KIND_METADATA_CONSTRAINT && idxStr[1 + (i*4) + 2] == VEC0_METADATA_OPERATOR_IN)) {
      continue;
    }
    int metadata_idx = idxStr[1 + (i*4) + 1]  - 'A';
    if(!aMetadataIn) {
      aMetadataIn = sqlite3_malloc(sizeof(*aMetadataIn));
      if(!aMetadataIn) {
        rc = SQLITE_NOMEM;
        goto cleanup;
      }
      memset(aMetadataIn, 0, sizeof(*aMetadataIn));
      rc = array_init(aMetadataIn, sizeof(struct Vec0MetadataIn), 8);
      if(rc != SQLITE_OK) {
        goto cleanup;
      }
    }

    struct Vec0MetadataIn item;
    memset(&item, 0, sizeof(item));
    item.metadata_idx=metadata_idx;
    item.argv_idx = i;

    switch(p->metadata_columns[metadata_idx].kind) {
      case VEC0_METADATA_COLUMN_KIND_INTEGER: {
        rc = array_init(&item.array, sizeof(i64), 16);
        if(rc != SQLITE_OK) {
          goto cleanup;
        }
        sqlite3_value *entry;
        for (rc = sqlite3_vtab_in_first(argv[i], &entry); rc == SQLITE_OK && entry; rc = sqlite3_vtab_in_next(argv[i], &entry)) {
          i64 v = sqlite3_value_int64(entry);
          rc = array_append(&item.array, &v);
          if (rc != SQLITE_OK) {
            goto cleanup;
          }
        }

        if (rc != SQLITE_DONE) {
          vtab_set_error(&p->base, "Error fetching next value in `x in (...)` integer expression");
          goto cleanup;
        }
        qsort(item.array.z, item.array.length, item.array.element_size, _cmp);

        break;
      }
      case VEC0_METADATA_COLUMN_KIND_TEXT: {
        rc = array_init(&item.array, sizeof(struct Vec0MetadataInTextEntry), 16);
        if(rc != SQLITE_OK) {
          goto cleanup;
        }
        sqlite3_value *entry;
        for (rc = sqlite3_vtab_in_first(argv[i], &entry); rc == SQLITE_OK && entry; rc = sqlite3_vtab_in_next(argv[i], &entry)) {
          const char * s = (const char *) sqlite3_value_text(entry);
          int n = sqlite3_value_bytes(entry);

          struct Vec0MetadataInTextEntry entry;
          entry.zString = sqlite3_mprintf("%.*s", n, s);
          if(!entry.zString) {
            rc = SQLITE_NOMEM;
            goto cleanup;
          }
          entry.n = n;
          rc = array_append(&item.array, &entry);
          if (rc != SQLITE_OK) {
            goto cleanup;
          }
        }

        if (rc != SQLITE_DONE) {
          vtab_set_error(&p->base, "Error fetching next value in `x in (...)` text expression");
          goto cleanup;
        }

        break;
      }
      default: {
        vtab_set_error(&p->base, "Internal sqlite-vec error");
        goto cleanup;
      }
    }

    rc = array_append(aMetadataIn, &item);
    if(rc != SQLITE_OK) {
      goto cleanup;
    }
  }
  #endif

#if SQLITE_VEC_ENABLE_RESCORE
  // Dispatch to rescore KNN path if this vector column has rescore enabled
  if (vector_column->index_type == VEC0_INDEX_TYPE_RESCORE) {
    rc = rescore_knn(p, pCur, vector_column, vectorColumnIdx, arrayRowidsIn,
                     aMetadataIn, idxStr, argc, argv, queryVector, k, knn_data);
    if (rc != SQLITE_OK) {
      goto cleanup;
    }
    pCur->knn_data = knn_data;
    pCur->query_plan = VEC0_QUERY_PLAN_KNN;
    rc = SQLITE_OK;
    goto cleanup;
  }
#endif

  if (p->storage && vector_column->index_type == VEC0_INDEX_TYPE_FLAT) {
    rc = vec0_knn_storage(p, vectorColumnIdx, queryVector, arrayRowidsIn, k, idxStr,
                          argc, argv, knn_data);
    if (rc != SQLITE_OK) {
      goto cleanup;
    }
    pCur->knn_data = knn_data;
    pCur->query_plan = VEC0_QUERY_PLAN_KNN;
    rc = SQLITE_OK;
    goto cleanup;
  }

#if SQLITE_VEC_ENABLE_HNSW
  if (vector_column->index_type == VEC0_INDEX_TYPE_HNSW) {
    rc = hnsw_query_knn(p, vectorColumnIdx, queryVector, arrayRowidsIn, k, knn_data);
    if (rc != SQLITE_OK) {
      goto cleanup;
    }
    vec0_knn_apply_distance_constraints(knn_data, idxStr, argc, argv);
    pCur->knn_data = knn_data;
    pCur->query_plan = VEC0_QUERY_PLAN_KNN;
    rc = SQLITE_OK;
    goto cleanup;
  }
#endif

#if SQLITE_VEC_EXPERIMENTAL_IVF_ENABLE
  // IVF dispatch: if vector column has IVF, use IVF query instead of chunk scan
  if (vector_column->index_type == VEC0_INDEX_TYPE_IVF) {
    rc = ivf_query_knn(p, vectorColumnIdx, queryVector,
                       (int)vector_column_byte_size(*vector_column),
                       arrayRowidsIn, k, knn_data);
    if (rc != SQLITE_OK) {
      goto cleanup;
    }
    vec0_knn_apply_distance_constraints(knn_data, idxStr, argc, argv);
    pCur->knn_data = knn_data;
    pCur->query_plan = VEC0_QUERY_PLAN_KNN;
    rc = SQLITE_OK;
    goto cleanup;
  }
#endif

  rc = vec0_chunks_iter(p, idxStr, argc, argv, &stmtChunks);
  if (rc != SQLITE_OK) {
    // IMP: V06942_23781
    vtab_set_error(&p->base, "Error preparing stmtChunk: %s",
                   sqlite3_errmsg(p->db));
    goto cleanup;
  }

  i64 *topk_rowids = NULL;
  f32 *topk_distances = NULL;
  i64 k_used = 0;
  rc = vec0Filter_knn_chunks_iter(p, stmtChunks, vector_column, vectorColumnIdx,
                                  arrayRowidsIn, aMetadataIn, idxStr, argc, argv, queryVector, k, &topk_rowids,
                                  &topk_distances, &k_used);
  if (rc != SQLITE_OK) {
    goto cleanup;
  }

  // MMR reranking: select diverse subset from over-fetched candidates
  if (mmr_lambda >= 0.0f && mmr_lambda < 1.0f && k_used > k_original) {
    i64 n_selected = 0;
    rc = vec0_mmr_rerank(p, vectorColumnIdx, vector_column,
                         topk_rowids, topk_distances, k_used, k_original,
                         mmr_lambda, &n_selected);
    if (rc != SQLITE_OK) goto cleanup;
    k_used = n_selected;
    k = k_original;
  }

  knn_data->current_idx = 0;
  knn_data->k = k;
  knn_data->rowids = topk_rowids;
  knn_data->distances = topk_distances;
  knn_data->k_used = k_used;

  pCur->knn_data = knn_data;
  pCur->query_plan = VEC0_QUERY_PLAN_KNN;
  rc = SQLITE_OK;

cleanup:
  sqlite3_finalize(stmtChunks);
  array_cleanup(arrayRowidsIn);
  sqlite3_free(arrayRowidsIn);
  queryVectorCleanup(queryVector);
  if(aMetadataIn) {
    for(size_t i = 0; i < aMetadataIn->length; i++) {
      struct Vec0MetadataIn* item = &((struct Vec0MetadataIn *) aMetadataIn->z)[i];
      for(size_t j = 0; j < item->array.length; j++) {
        if(p->metadata_columns[item->metadata_idx].kind == VEC0_METADATA_COLUMN_KIND_TEXT) {
          struct Vec0MetadataInTextEntry entry = ((struct Vec0MetadataInTextEntry*)item->array.z)[j];
          sqlite3_free(entry.zString);
        }
      }
      array_cleanup(&item->array);
    }
    array_cleanup(aMetadataIn);
  }

  sqlite3_free(aMetadataIn);

  if (rc != SQLITE_OK) {
    sqlite3_free(knn_data);
  }

  return rc;
}

int vec0Filter_fullscan(vec0_vtab *p, vec0_cursor *pCur) {
  int rc;
  char *zSql;
  struct vec0_query_fullscan_data *fullscan_data;

  fullscan_data = sqlite3_malloc(sizeof(*fullscan_data));
  if (!fullscan_data) {
    return SQLITE_NOMEM;
  }
  memset(fullscan_data, 0, sizeof(*fullscan_data));

  zSql = sqlite3_mprintf(" SELECT rowid "
                         " FROM " VEC0_SHADOW_ROWIDS_NAME
                         " ORDER by chunk_id, chunk_offset ",
                         p->schemaName, p->tableName);
  if (!zSql) {
    rc = SQLITE_NOMEM;
    goto error;
  }
  rc = sqlite3_prepare_v2(p->db, zSql, -1, &fullscan_data->rowids_stmt, NULL);
  sqlite3_free(zSql);
  if (rc != SQLITE_OK) {
    // IMP: V09901_26739
    vtab_set_error(&p->base, "Error preparing rowid scan: %s",
                   sqlite3_errmsg(p->db));
    goto error;
  }

  rc = sqlite3_step(fullscan_data->rowids_stmt);

  // DONE when there's no rowids, ROW when there are, both "success"
  if (!(rc == SQLITE_ROW || rc == SQLITE_DONE)) {
    goto error;
  }

  fullscan_data->done = rc == SQLITE_DONE;
  pCur->query_plan = VEC0_QUERY_PLAN_FULLSCAN;
  pCur->fullscan_data = fullscan_data;
  return SQLITE_OK;

error:
  vec0_query_fullscan_data_clear(fullscan_data);
  sqlite3_free(fullscan_data);
  return rc;
}

int vec0Filter_point(vec0_cursor *pCur, vec0_vtab *p, int argc,
                     sqlite3_value **argv) {
  int rc;
  assert(argc == 1);
  i64 rowid;
  struct vec0_query_point_data *point_data = NULL;

  point_data = sqlite3_malloc(sizeof(*point_data));
  if (!point_data) {
    rc = SQLITE_NOMEM;
    goto error;
  }
  memset(point_data, 0, sizeof(*point_data));

  if (p->pkIsText) {
    rc = vec0_rowid_from_id(p, argv[0], &rowid);
    if (rc == SQLITE_EMPTY) {
      goto eof;
    }
    if (rc != SQLITE_OK) {
      goto error;
    }
  } else {
    rowid = sqlite3_value_int64(argv[0]);
  }

  for (int i = 0; i < p->numVectorColumns; i++) {
    rc = vec0_get_vector_data(p, rowid, i, &point_data->vectors[i], NULL);
    if (rc == SQLITE_EMPTY) {
      goto eof;
    }
    if (rc != SQLITE_OK) {
      goto error;
    }
  }

  point_data->rowid = rowid;
  point_data->done = 0;
  pCur->point_data = point_data;
  pCur->query_plan = VEC0_QUERY_PLAN_POINT;
  return SQLITE_OK;

eof:
  point_data->rowid = rowid;
  point_data->done = 1;
  pCur->point_data = point_data;
  pCur->query_plan = VEC0_QUERY_PLAN_POINT;
  return SQLITE_OK;

error:
  vec0_query_point_data_clear(point_data);
  sqlite3_free(point_data);
  return rc;
}

static int vec0Filter(sqlite3_vtab_cursor *pVtabCursor, int idxNum,
                      const char *idxStr, int argc, sqlite3_value **argv) {
  vec0_vtab *p = (vec0_vtab *)pVtabCursor->pVtab;
  vec0_cursor *pCur = (vec0_cursor *)pVtabCursor;
  vec0_cursor_clear(pCur);
  int frc = vec0_flat_flush(p);
  if (frc != SQLITE_OK) return frc;
#if SQLITE_VEC_EXPERIMENTAL_IVF_ENABLE
  int rc = ivf_flush(p);
  if (rc != SQLITE_OK) return rc;
#endif

  int idxStrLength = strlen(idxStr);
  if(idxStrLength <= 0) {
    return SQLITE_ERROR;
  }
  if((idxStrLength-1) % 4 != 0) {
    return SQLITE_ERROR;
  }
  int numValueEntries = (idxStrLength-1) / 4;
  if(numValueEntries != argc) {
    return SQLITE_ERROR;
  }

  char query_plan = idxStr[0];
  switch(query_plan) {
    case VEC0_QUERY_PLAN_FULLSCAN:
      return vec0Filter_fullscan(p, pCur);
    case VEC0_QUERY_PLAN_KNN: {
      int rc = vec0Filter_knn(pCur, p, idxNum, idxStr, argc, argv);
      // The result rowids are known now: let the provider fetch them at once.
      if (rc == SQLITE_OK && p->storage && p->storage->xPrefetch && pCur->knn_data &&
          pCur->knn_data->k_used > 0) {
        rc = vec0_storage_error(p, p->storage->xPrefetch(p->storageHandle,
                                                         pCur->knn_data->rowids,
                                                         (int)pCur->knn_data->k_used));
      }
      return rc;
    }
    case VEC0_QUERY_PLAN_POINT:
      return vec0Filter_point(pCur, p, argc, argv);
    default:
      vtab_set_error(pVtabCursor->pVtab, "unknown idxStr '%s'", idxStr);
      return SQLITE_ERROR;
  }
}

static int vec0Rowid(sqlite3_vtab_cursor *cur, sqlite_int64 *pRowid) {
  vec0_cursor *pCur = (vec0_cursor *)cur;
  switch (pCur->query_plan) {
  case VEC0_QUERY_PLAN_FULLSCAN: {
    *pRowid = sqlite3_column_int64(pCur->fullscan_data->rowids_stmt, 0);
    return SQLITE_OK;
  }
  case VEC0_QUERY_PLAN_POINT: {
    *pRowid = pCur->point_data->rowid;
    return SQLITE_OK;
  }
  case VEC0_QUERY_PLAN_KNN: {
    vtab_set_error(cur->pVtab,
                   "Internal sqlite-vec error: expected point query plan in "
                   "vec0Rowid, found %d",
                   pCur->query_plan);
    return SQLITE_ERROR;
  }
  }
  return SQLITE_ERROR;
}

static int vec0Next(sqlite3_vtab_cursor *cur) {
  vec0_cursor *pCur = (vec0_cursor *)cur;
  switch (pCur->query_plan) {
  case VEC0_QUERY_PLAN_FULLSCAN: {
    if (!pCur->fullscan_data) {
      return SQLITE_ERROR;
    }
    int rc = sqlite3_step(pCur->fullscan_data->rowids_stmt);
    if (rc == SQLITE_DONE) {
      pCur->fullscan_data->done = 1;
      return SQLITE_OK;
    }
    if (rc == SQLITE_ROW) {
      return SQLITE_OK;
    }
    return SQLITE_ERROR;
  }
  case VEC0_QUERY_PLAN_KNN: {
    if (!pCur->knn_data) {
      return SQLITE_ERROR;
    }

    pCur->knn_data->current_idx++;
    return SQLITE_OK;
  }
  case VEC0_QUERY_PLAN_POINT: {
    if (!pCur->point_data) {
      return SQLITE_ERROR;
    }
    pCur->point_data->done = 1;
    return SQLITE_OK;
  }
  }
  return SQLITE_ERROR;
}

static int vec0Eof(sqlite3_vtab_cursor *cur) {
  vec0_cursor *pCur = (vec0_cursor *)cur;
  switch (pCur->query_plan) {
  case VEC0_QUERY_PLAN_FULLSCAN: {
    if (!pCur->fullscan_data) {
      return 1;
    }
    return pCur->fullscan_data->done;
  }
  case VEC0_QUERY_PLAN_KNN: {
    if (!pCur->knn_data) {
      return 1;
    }
    // return (pCur->knn_data->current_idx >= pCur->knn_data->k) ||
    // (pCur->knn_data->distances[pCur->knn_data->current_idx] == FLT_MAX);
    return (pCur->knn_data->current_idx >= pCur->knn_data->k_used);
  }
  case VEC0_QUERY_PLAN_POINT: {
    if (!pCur->point_data) {
      return 1;
    }
    return pCur->point_data->done;
  }
  }
  return 1;
}

/**
 * @brief Read the vector of the full scan's current row from a flat column,
 * through the handle the scan keeps on the row's chunk.
 *
 * The scan visits rows in (chunk_id, chunk_offset) order, so the handle moves
 * to the next chunk with sqlite3_blob_reopen() once per chunk. A write that
 * replaces the chunk's row expires the handle, and sqlite3_blob_read() answers
 * SQLITE_ABORT (https://sqlite.org/c3ref/blob_open.html). A rollback of a
 * transaction that changed the schema invalidates it, and both
 * sqlite3_blob_read() and sqlite3_blob_reopen() answer SQLITE_ABORT_ROLLBACK
 * when the connection has extended result codes on. Either way the read is
 * retried once on a newly opened handle.
 *
 * The row's position is looked up here rather than taken from the scan's
 * statement: that statement sorts every row when the scan starts, so its
 * positions miss a delete, insert or optimize the same connection runs during
 * the scan.
 *
 * @param outVector set to a buffer the caller frees with sqlite3_free()
 */
static int vec0_fullscan_read_vector(vec0_vtab *p,
                                     struct vec0_query_fullscan_data *f,
                                     int vector_idx, void **outVector,
                                     int *outVectorSize) {
  i64 rowid = sqlite3_column_int64(f->rowids_stmt, 0);
  i64 chunk_id;
  i64 chunk_offset;
  int rc = vec0_get_chunk_position(p, rowid, NULL, &chunk_id, &chunk_offset);
  if (rc == SQLITE_EMPTY) {
    vtab_set_error(&p->base, "Could not find a row with rowid %lld", rowid);
    return SQLITE_ERROR;
  }
  if (rc != SQLITE_OK) {
    return rc;
  }
  int size = vector_column_byte_size(p->vector_columns[vector_idx]);
  void *buf = sqlite3_malloc(size);
  if (!buf) {
    return SQLITE_NOMEM;
  }

  sqlite3_blob **blob = &f->vectorBlobs[vector_idx];
  rc = SQLITE_ABORT;
  for (int attempt = 0; attempt < 2 && (rc & 0xff) == SQLITE_ABORT; attempt++) {
    if (*blob && f->vectorBlobChunkIds[vector_idx] != chunk_id) {
      rc = sqlite3_blob_reopen(*blob, chunk_id);
      if (rc != SQLITE_OK) {
        sqlite3_blob_close(*blob);
        *blob = NULL;
      }
      // An invalidated handle is replaced below; any other failure, such as
      // SQLITE_INTERRUPT, fails the read.
      if (rc != SQLITE_OK && (rc & 0xff) != SQLITE_ABORT) {
        sqlite3_free(buf);
        vtab_set_error(
            &p->base,
            "Could not fetch vector data for %lld, opening blob failed", rowid);
        return SQLITE_ERROR;
      }
    }
    if (!*blob) {
      rc = sqlite3_blob_open(p->db, p->schemaName,
                             p->shadowVectorChunksNames[vector_idx], "vectors",
                             chunk_id, 0, blob);
      if (rc != SQLITE_OK) {
        *blob = NULL;
        sqlite3_free(buf);
        vtab_set_error(
            &p->base,
            "Could not fetch vector data for %lld, opening blob failed", rowid);
        return SQLITE_ERROR;
      }
    }
    f->vectorBlobChunkIds[vector_idx] = chunk_id;
    rc = sqlite3_blob_read(*blob, buf, size, chunk_offset * size);
    if ((rc & 0xff) == SQLITE_ABORT) {
      sqlite3_blob_close(*blob);
      *blob = NULL;
    }
  }
  if (rc != SQLITE_OK) {
    sqlite3_free(buf);
    vtab_set_error(
        &p->base,
        "Could not fetch vector data for %lld, reading from blob failed",
        rowid);
    return SQLITE_ERROR;
  }
  *outVector = buf;
  *outVectorSize = size;
  return SQLITE_OK;
}

static int vec0Column_fullscan(vec0_vtab *pVtab, vec0_cursor *pCur,
                               sqlite3_context *context, int i) {
  if (!pCur->fullscan_data) {
    sqlite3_result_error(
        context, "Internal sqlite-vec error: fullscan_data is NULL.", -1);
    return SQLITE_ERROR;
  }
  i64 rowid = sqlite3_column_int64(pCur->fullscan_data->rowids_stmt, 0);
  if (i == VEC0_COLUMN_ID) {
    return vec0_result_id(pVtab, context, rowid);
  }
  else if (vec0_column_idx_is_vector(pVtab, i)) {
    void *v;
    int sz;
    int vector_idx = vec0_column_idx_to_vector_idx(pVtab, i);
    // Only a flat column keeps its vectors in _vector_chunks; the other index
    // types keep them in tables of their own, read per row.
    int rc =
        vec0_column_in_chunks(pVtab, vector_idx)
            ? vec0_fullscan_read_vector(pVtab, pCur->fullscan_data, vector_idx,
                                        &v, &sz)
            : vec0_get_vector_data(pVtab, rowid, vector_idx, &v, &sz);
    if (rc != SQLITE_OK) {
      return rc;
    }
    sqlite3_result_blob(context, v, sz, sqlite3_free);
    sqlite3_result_subtype(context,
                           pVtab->vector_columns[vector_idx].element_type);

  }
  else if (i == vec0_column_distance_idx(pVtab)) {
    sqlite3_result_null(context);
  }
  else if(vec0_column_idx_is_partition(pVtab, i)) {
    int partition_idx = vec0_column_idx_to_partition_idx(pVtab, i);
    sqlite3_value * v;
    int rc = vec0_get_partition_value_for_rowid(pVtab, rowid, partition_idx, &v);
    if(rc == SQLITE_OK) {
      sqlite3_result_value(context, v);
      sqlite3_value_free(v);
    }else {
      sqlite3_result_error_code(context, rc);
    }
  }
  else if(vec0_column_idx_is_auxiliary(pVtab, i)) {
    int auxiliary_idx = vec0_column_idx_to_auxiliary_idx(pVtab, i);
    vec0_result_auxiliary(pVtab, context, rowid, auxiliary_idx);
  }

  else if(vec0_column_idx_is_metadata(pVtab, i)) {
    if(sqlite3_vtab_nochange(context)) {
      return SQLITE_OK;
    }
    int metadata_idx = vec0_column_idx_to_metadata_idx(pVtab, i);
    int rc = vec0_result_metadata_value_for_rowid(pVtab, rowid, metadata_idx, context);
    if(rc != SQLITE_OK) {
      // IMP: V15466_32305
      const char * zErr = sqlite3_mprintf(
        "Could not extract metadata value for column %.*s at rowid %lld",
        pVtab->metadata_columns[metadata_idx].name_length,
        pVtab->metadata_columns[metadata_idx].name, rowid
      );
      if(zErr) {
        sqlite3_result_error(context, zErr, -1);
        sqlite3_free((void *) zErr);
      }else {
        sqlite3_result_error_nomem(context);
      }
    }
  }

  return SQLITE_OK;
}

static int vec0Column_point(vec0_vtab *pVtab, vec0_cursor *pCur,
                            sqlite3_context *context, int i) {
  if (!pCur->point_data) {
    sqlite3_result_error(context,
                         "Internal sqlite-vec error: point_data is NULL.", -1);
    return SQLITE_ERROR;
  }
  if (i == VEC0_COLUMN_ID) {
    return vec0_result_id(pVtab, context, pCur->point_data->rowid);
  }
  else if (i == vec0_column_distance_idx(pVtab)) {
    sqlite3_result_null(context);
    return SQLITE_OK;
  }
  else if (vec0_column_idx_is_vector(pVtab, i)) {
    if (sqlite3_vtab_nochange(context)) {
      sqlite3_result_null(context);
      return SQLITE_OK;
    }
    int vector_idx = vec0_column_idx_to_vector_idx(pVtab, i);
    sqlite3_result_blob(
        context, pCur->point_data->vectors[vector_idx],
        vector_column_byte_size(pVtab->vector_columns[vector_idx]),
        SQLITE_TRANSIENT);
    sqlite3_result_subtype(context,
                           pVtab->vector_columns[vector_idx].element_type);
    return SQLITE_OK;
  }
  else if(vec0_column_idx_is_partition(pVtab, i)) {
    if(sqlite3_vtab_nochange(context)) {
      return SQLITE_OK;
    }
    int partition_idx = vec0_column_idx_to_partition_idx(pVtab, i);
    i64 rowid = pCur->point_data->rowid;
    sqlite3_value * v;
    int rc = vec0_get_partition_value_for_rowid(pVtab, rowid, partition_idx, &v);
    if(rc == SQLITE_OK) {
      sqlite3_result_value(context, v);
      sqlite3_value_free(v);
    }else {
      sqlite3_result_error_code(context, rc);
    }
  }
  else if(vec0_column_idx_is_auxiliary(pVtab, i)) {
    if(sqlite3_vtab_nochange(context)) {
      return SQLITE_OK;
    }
    i64 rowid = pCur->point_data->rowid;
    int auxiliary_idx = vec0_column_idx_to_auxiliary_idx(pVtab, i);
    vec0_result_auxiliary(pVtab, context, rowid, auxiliary_idx);
  }

  else if(vec0_column_idx_is_metadata(pVtab, i)) {
    if(sqlite3_vtab_nochange(context)) {
      return SQLITE_OK;
    }
    i64 rowid = pCur->point_data->rowid;
    int metadata_idx = vec0_column_idx_to_metadata_idx(pVtab, i);
    int rc = vec0_result_metadata_value_for_rowid(pVtab, rowid, metadata_idx, context);
    if(rc != SQLITE_OK) {
      const char * zErr = sqlite3_mprintf(
        "Could not extract metadata value for column %.*s at rowid %lld",
        pVtab->metadata_columns[metadata_idx].name_length,
        pVtab->metadata_columns[metadata_idx].name, rowid
      );
      if(zErr) {
        sqlite3_result_error(context, zErr, -1);
        sqlite3_free((void *) zErr);
      }else {
        sqlite3_result_error_nomem(context);
      }
    }
  }

  return SQLITE_OK;
}

static int vec0Column_knn(vec0_vtab *pVtab, vec0_cursor *pCur,
                          sqlite3_context *context, int i) {
  if (!pCur->knn_data) {
    sqlite3_result_error(context,
                         "Internal sqlite-vec error: knn_data is NULL.", -1);
    return SQLITE_ERROR;
  }
  if (i == VEC0_COLUMN_ID) {
    i64 rowid = pCur->knn_data->rowids[pCur->knn_data->current_idx];
    return vec0_result_id(pVtab, context, rowid);
  }
  else if (i == vec0_column_distance_idx(pVtab)) {
    sqlite3_result_double(
        context, pCur->knn_data->distances[pCur->knn_data->current_idx]);
    return SQLITE_OK;
  }
  else if (vec0_column_idx_is_vector(pVtab, i)) {
    void *out;
    int sz;
    int vector_idx = vec0_column_idx_to_vector_idx(pVtab, i);
    int rc = vec0_get_vector_data(
        pVtab, pCur->knn_data->rowids[pCur->knn_data->current_idx], vector_idx,
        &out, &sz);
    if (rc != SQLITE_OK) {
      return rc;
    }
    sqlite3_result_blob(context, out, sz, sqlite3_free);
    sqlite3_result_subtype(context,
                           pVtab->vector_columns[vector_idx].element_type);
    return SQLITE_OK;
  }
  else if(vec0_column_idx_is_partition(pVtab, i)) {
    int partition_idx = vec0_column_idx_to_partition_idx(pVtab, i);
    i64 rowid = pCur->knn_data->rowids[pCur->knn_data->current_idx];
    sqlite3_value * v;
    int rc = vec0_get_partition_value_for_rowid(pVtab, rowid, partition_idx, &v);
    if(rc == SQLITE_OK) {
      sqlite3_result_value(context, v);
      sqlite3_value_free(v);
    }else {
      sqlite3_result_error_code(context, rc);
    }
  }
  else if(vec0_column_idx_is_auxiliary(pVtab, i)) {
    int auxiliary_idx = vec0_column_idx_to_auxiliary_idx(pVtab, i);
    i64 rowid = pCur->knn_data->rowids[pCur->knn_data->current_idx];
    vec0_result_auxiliary(pVtab, context, rowid, auxiliary_idx);
  }

  else if(vec0_column_idx_is_metadata(pVtab, i)) {
    int metadata_idx = vec0_column_idx_to_metadata_idx(pVtab, i);
    i64 rowid = pCur->knn_data->rowids[pCur->knn_data->current_idx];
    int rc = vec0_result_metadata_value_for_rowid(pVtab, rowid, metadata_idx, context);
    if(rc != SQLITE_OK) {
      const char * zErr = sqlite3_mprintf(
        "Could not extract metadata value for column %.*s at rowid %lld",
        pVtab->metadata_columns[metadata_idx].name_length,
        pVtab->metadata_columns[metadata_idx].name, rowid
      );
      if(zErr) {
        sqlite3_result_error(context, zErr, -1);
        sqlite3_free((void *) zErr);
      }else {
        sqlite3_result_error_nomem(context);
      }
    }
  }

  return SQLITE_OK;
}

static int vec0Column(sqlite3_vtab_cursor *cur, sqlite3_context *context,
                      int i) {
  vec0_cursor *pCur = (vec0_cursor *)cur;
  vec0_vtab *pVtab = (vec0_vtab *)cur->pVtab;
  int rc = vec0_flat_flush(pVtab);
  if (rc != SQLITE_OK) return rc;
  switch (pCur->query_plan) {
  case VEC0_QUERY_PLAN_FULLSCAN: {
    return vec0Column_fullscan(pVtab, pCur, context, i);
  }
  case VEC0_QUERY_PLAN_KNN: {
    return vec0Column_knn(pVtab, pCur, context, i);
  }
  case VEC0_QUERY_PLAN_POINT: {
    return vec0Column_point(pVtab, pCur, context, i);
  }
  }
  return SQLITE_OK;
}

/**
 * @brief Handles the "insert rowid" step of a row insert operation of a vec0
 * table.
 *
 * This function will insert a new row into the _rowids vec0 shadow table.
 *
 * @param p: virtual table
 * @param idValue: Value containing the inserted rowid/id value.
 * @param rowid: Output rowid, will point to the "real" i64 rowid
 * value that was inserted
 * @return int SQLITE_OK on success, error code on failure
 */
int vec0Update_InsertRowidStep(vec0_vtab *p, sqlite3_value *idValue,
                               i64 *rowid) {

  /**
   * An insert into a vec0 table can happen a few different ways:
   *  1) With default INTEGER primary key: With a supplied i64 rowid
   *  2) With default INTEGER primary key: WITHOUT a supplied rowid
   *  3) With TEXT primary key: supplied text rowid
   */

  int rc;

  // Option 3: vtab has a user-defined TEXT primary key, so ensure a text value
  // is provided.
  if (p->pkIsText) {
    if (sqlite3_value_type(idValue) != SQLITE_TEXT) {
      // IMP: V04200_21039
      vtab_set_error(&p->base,
                     "The %s virtual table was declared with a TEXT primary "
                     "key, but a non-TEXT value was provided in an INSERT.",
                     p->tableName);
      return SQLITE_ERROR;
    }

    return vec0_rowids_insert_id(p, idValue, rowid);
  }

  // Option 1: User supplied a i64 rowid
  if (sqlite3_value_type(idValue) == SQLITE_INTEGER) {
    i64 suppliedRowid = sqlite3_value_int64(idValue);
    rc = vec0_rowids_insert_rowid(p, suppliedRowid);
    if (rc == SQLITE_OK) {
      *rowid = suppliedRowid;
    }
    return rc;
  }

  // Option 2: User did not suppled a rowid

  if (sqlite3_value_type(idValue) != SQLITE_NULL) {
    // IMP: V30855_14925
    vtab_set_error(&p->base,
                   "Only integers are allows for primary key values on %s",
                   p->tableName);
    return SQLITE_ERROR;
  }
  // NULL to get next auto-incremented value
  return vec0_rowids_insert_id(p, NULL, rowid);
}

/**
 * @brief Determines the "next available" chunk position for a newly inserted
 * vec0 row.
 *
 * This operation may insert a new "blank" chunk the _chunks table, if there is
 * no more space in previous chunks.
 *
 * @param p: virtual table
 * @param partitionKeyValues: array of partition key column values, to constrain
 * against any partition key columns.
 * @param chunk_rowid: Output rowid of the chunk in the _chunks virtual table
 * that has the avialabiity.
 * @param chunk_offset: Output the index of the available space insert the
 * chunk, based on the index of the first available validity bit.
 * @param pBlobValidity: Output blob of the validity column of the available
 * chunk. Will be opened with read/write permissions.
 * @param pValidity: Output buffer of the original chunk's validity column.
 *    Needs to be cleaned up with sqlite3_free().
 * @return int SQLITE_OK on success, error code on failure
 */
int vec0Update_InsertNextAvailableStep(
    vec0_vtab *p,
    sqlite3_value ** partitionKeyValues,
    i64 *chunk_rowid, i64 *chunk_offset,
    sqlite3_blob **blobChunksValidity,
    const unsigned char **bufferChunksValidity) {

  int rc;
  i64 validitySize;
  *chunk_offset = -1;

  rc = vec0_get_latest_chunk_rowid(p, chunk_rowid, partitionKeyValues);
  if(rc == SQLITE_EMPTY) {
    goto done;
  }
  if (rc != SQLITE_OK) {
    goto cleanup;
  }

  rc = sqlite3_blob_open(p->db, p->schemaName, p->shadowChunksName, "validity",
                         *chunk_rowid, 1, blobChunksValidity);
  if (rc != SQLITE_OK) {
    // IMP: V22053_06123
    vtab_set_error(&p->base,
                   VEC_INTERAL_ERROR
                   "could not open validity blob on %s.%s.%lld",
                   p->schemaName, p->shadowChunksName, *chunk_rowid);
    goto cleanup;
  }

  validitySize = sqlite3_blob_bytes(*blobChunksValidity);
  if (validitySize != p->chunk_size / CHAR_BIT) {
    // IMP: V29362_13432
    vtab_set_error(&p->base,
                   VEC_INTERAL_ERROR
                   "validity blob size mismatch on "
                   "%s.%s.%lld, expected %lld but received %lld.",
                   p->schemaName, p->shadowChunksName, *chunk_rowid,
                   (i64)(p->chunk_size / CHAR_BIT), validitySize);
    rc = SQLITE_ERROR;
    goto cleanup;
  }

  *bufferChunksValidity = sqlite3_malloc(validitySize);
  if (!(*bufferChunksValidity)) {
    vtab_set_error(&p->base, VEC_INTERAL_ERROR
                   "Could not allocate memory for validity bitmap");
    rc = SQLITE_NOMEM;
    goto cleanup;
  }

  rc = sqlite3_blob_read(*blobChunksValidity, (void *)*bufferChunksValidity,
                         validitySize, 0);

  if (rc != SQLITE_OK) {
    vtab_set_error(&p->base,
                   VEC_INTERAL_ERROR
                   "Could not read validity bitmap for %s.%s.%lld",
                   p->schemaName, p->shadowChunksName, *chunk_rowid);
    goto cleanup;
  }

  // find the next available offset, ie first `0` in the bitmap.
  for (int i = 0; i < validitySize; i++) {
    if ((*bufferChunksValidity)[i] == 0b11111111)
      continue;
    for (int j = 0; j < CHAR_BIT; j++) {
      if (((((*bufferChunksValidity)[i] >> j) & 1) == 0)) {
        *chunk_offset = (i * CHAR_BIT) + j;
        goto done;
      }
    }
  }

done:
  // latest chunk was full, so need to create a new one
  if (*chunk_offset == -1) {
    rc = vec0_new_chunk(p, partitionKeyValues, chunk_rowid);
    if (rc != SQLITE_OK) {
      // IMP: V08441_25279
      vtab_set_error(&p->base,
                     VEC_INTERAL_ERROR "Could not insert a new vector chunk");
      rc = SQLITE_ERROR; // otherwise raises a DatabaseError and not operational
                         // error?
      goto cleanup;
    }
    *chunk_offset = 0;

    // blobChunksValidity and pValidity are stale, pointing to the previous
    // (full) chunk. to re-assign them
    rc = sqlite3_blob_close(*blobChunksValidity);
    sqlite3_free((void *)*bufferChunksValidity);
    *blobChunksValidity = NULL;
    *bufferChunksValidity = NULL;
    if (rc != SQLITE_OK) {
      vtab_set_error(&p->base, VEC_INTERAL_ERROR
                     "unknown error, blobChunksValidity could not be closed, "
                     "please file an issue.");
      rc = SQLITE_ERROR;
      goto cleanup;
    }

    rc = sqlite3_blob_open(p->db, p->schemaName, p->shadowChunksName,
                           "validity", *chunk_rowid, 1, blobChunksValidity);
    if (rc != SQLITE_OK) {
      vtab_set_error(
          &p->base,
          VEC_INTERAL_ERROR
          "Could not open validity blob for newly created chunk %s.%s.%lld",
          p->schemaName, p->shadowChunksName, *chunk_rowid);
      goto cleanup;
    }
    validitySize = sqlite3_blob_bytes(*blobChunksValidity);
    if (validitySize != p->chunk_size / CHAR_BIT) {
      vtab_set_error(&p->base,
                     VEC_INTERAL_ERROR
                     "validity blob size mismatch for newly created chunk "
                     "%s.%s.%lld. Exepcted %lld, got %lld",
                     p->schemaName, p->shadowChunksName, *chunk_rowid,
                     p->chunk_size / CHAR_BIT, validitySize);
      goto cleanup;
    }
    *bufferChunksValidity = sqlite3_malloc(validitySize);
    rc = sqlite3_blob_read(*blobChunksValidity, (void *)*bufferChunksValidity,
                           validitySize, 0);
    if (rc != SQLITE_OK) {
      vtab_set_error(&p->base,
                     VEC_INTERAL_ERROR
                     "could not read validity blob newly created chunk "
                     "%s.%s.%lld",
                     p->schemaName, p->shadowChunksName, *chunk_rowid);
      goto cleanup;
    }
  }

  rc = SQLITE_OK;

cleanup:
  return rc;
}

#define VEC0_FLAT_WRITE_BYTES (64 * 1024)

struct Vec0FlatWrite {
  i64 chunk;
  int offset, n, capacity;
  unsigned char data[];
};

static void vec0_flat_clear(vec0_vtab *p) {
  p->flatWriteGeneration++;
  for (int i = 0; i < VEC0_MAX_VECTOR_COLUMNS; i++) {
    sqlite3_free(p->flatWrites[i]);
    p->flatWrites[i] = NULL;
  }
}

static int vec0_flat_open(vec0_vtab *p, int col, i64 chunk, int write, sqlite3_blob **blob) {
  int rc = sqlite3_blob_open(p->db, p->schemaName, p->shadowVectorChunksNames[col],
                             "vectors", chunk, write, blob);
  if (rc != SQLITE_OK) {
    vtab_set_error(&p->base, "Error opening vector blob at %s.%s.%lld",
                   p->schemaName, p->shadowVectorChunksNames[col], chunk);
    return rc;
  }
  i64 expected = p->chunk_size * vector_column_byte_size(p->vector_columns[col]);
  i64 actual = sqlite3_blob_bytes(*blob);
  if (actual != expected) {
    // IMP: V16386_00456
    vtab_set_error(&p->base, VEC_INTERAL_ERROR
        "vector blob size mismatch on %s.%s.%lld. Expected %lld, actual %lld",
        p->schemaName, p->shadowVectorChunksNames[col], chunk, expected, actual);
    sqlite3_blob_close(*blob);
    *blob = NULL;
    return SQLITE_ERROR;
  }
  return SQLITE_OK;
}

static int vec0_flat_flush_column(vec0_vtab *p, int col) {
  struct Vec0FlatWrite *w = p->flatWrites[col];
  if (!w) return SQLITE_OK;
  // SQLite can call rollback while opening/writing the blob; keep ownership here.
  p->flatWrites[col] = NULL;
  unsigned generation = p->flatWriteGeneration;
  p->flatFlushing++;
  sqlite3_blob *blob = NULL;
  int rc = vec0_flat_open(p, col, w->chunk, 1, &blob);
  if (rc == SQLITE_OK) rc = sqlite3_blob_write(blob, w->data, w->n, w->offset);
  int brc = sqlite3_blob_close(blob);
  if (rc == SQLITE_OK) rc = brc;
  p->flatFlushing--;
  if (rc == SQLITE_OK || generation != p->flatWriteGeneration) sqlite3_free(w);
  else p->flatWrites[col] = w;
  return rc;
}

static int vec0_flat_flush(vec0_vtab *p) {
  for (int col = 0; col < p->numVectorColumns; col++) {
    int rc = vec0_flat_flush_column(p, col);
    if (rc != SQLITE_OK) return rc;
  }
  return SQLITE_OK;
}

static int vec0_flat_write(vec0_vtab *p, int col, i64 chunk, i64 slot, const void *data) {
  int bytes = vector_column_byte_size(p->vector_columns[col]);
  if (slot < 0 || slot >= p->chunk_size) return SQLITE_CORRUPT;
  i64 offset = slot * bytes;
  if (offset > INT_MAX - bytes) return SQLITE_CORRUPT;
  struct Vec0FlatWrite *w = p->flatWrites[col];
  if (w && (w->chunk != chunk || w->offset + w->n != offset || w->n + bytes > w->capacity)) {
    int rc = vec0_flat_flush_column(p, col);
    if (rc != SQLITE_OK) return rc;
    w = NULL;
  }
  if (!w) {
    sqlite3_blob *blob = NULL;
    int rc = vec0_flat_open(p, col, chunk, 0, &blob);
    int brc = sqlite3_blob_close(blob);
    if (rc != SQLITE_OK) return rc;
    if (brc != SQLITE_OK) return brc;
    int capacity = VEC0_FLAT_WRITE_BYTES;
    i64 remaining = (p->chunk_size - slot) * bytes;
    if (remaining < capacity) capacity = remaining;
    w = sqlite3_malloc64(sizeof(*w) + (sqlite3_uint64)capacity);
    if (!w) return SQLITE_NOMEM;
    w->chunk = chunk; w->offset = offset; w->n = 0; w->capacity = capacity;
    p->flatWrites[col] = w;
  }
  memcpy(w->data + w->n, data, bytes);
  w->n += bytes;
  return SQLITE_OK;
}

/**
 * @brief
 *
 * @param p vec0 virtual table
 * @param chunk_rowid: which chunk to write to
 * @param chunk_offset: the offset inside the chunk to write the vector to.
 * @param rowid: the rowid of the inserting row
 * @param vectorDatas: array of the vector data to insert
 * @param blobValidity: writeable validity blob of the row's assigned chunk.
 * @param validity: snapshot buffer of the valdity column from the row's
 * assigned chunk.
 * @return int SQLITE_OK on success, error code on failure
 */
int vec0Update_InsertWriteFinalStep(vec0_vtab *p, i64 chunk_rowid,
                                    i64 chunk_offset, i64 rowid,
                                    void *vectorDatas[],
                                    sqlite3_blob *blobChunksValidity,
                                    const unsigned char *bufferChunksValidity) {
  int rc, brc;
  sqlite3_blob *blobChunksRowids = NULL;

  // mark the validity bit for this row in the chunk's validity bitmap
  // Get the byte offset of the bitmap
  char unsigned bx = bufferChunksValidity[chunk_offset / CHAR_BIT];
  // set the bit at the chunk_offset position inside that byte
  bx = bx | (1 << (chunk_offset % CHAR_BIT));
  // write that 1 byte
  rc = sqlite3_blob_write(blobChunksValidity, &bx, 1, chunk_offset / CHAR_BIT);
  if (rc != SQLITE_OK) {
    vtab_set_error(&p->base, VEC_INTERAL_ERROR "could not mark validity bit ");
    return rc;
  }

  // Go insert the vector data into the vector chunk shadow tables
  for (int i = 0; i < p->numVectorColumns; i++) {
    // Non-FLAT columns (rescore, IVF, DiskANN) don't use _vector_chunks
    if (!vec0_column_in_chunks(p, i))
      continue;

    rc = vec0_flat_write(p, i, chunk_rowid, chunk_offset, vectorDatas[i]);
    if (rc != SQLITE_OK) goto cleanup;
  }

  // write the new rowid to the rowids column of the _chunks table
  rc = sqlite3_blob_open(p->db, p->schemaName, p->shadowChunksName, "rowids",
                         chunk_rowid, 1, &blobChunksRowids);
  if (rc != SQLITE_OK) {
    // IMP: V09221_26060
    vtab_set_error(&p->base,
                   VEC_INTERAL_ERROR "could not open rowids blob on %s.%s.%lld",
                   p->schemaName, p->shadowChunksName, chunk_rowid);
    goto cleanup;
  }
  i64 expected = p->chunk_size * sizeof(i64);
  i64 actual = sqlite3_blob_bytes(blobChunksRowids);
  if (expected != actual) {
    // IMP: V12779_29618
    vtab_set_error(
        &p->base,
        VEC_INTERAL_ERROR
        "rowids blob size mismatch on %s.%s.%lld. Expected %lld, actual %lld",
        p->schemaName, p->shadowChunksName, chunk_rowid, expected, actual);
    rc = SQLITE_ERROR;
    goto cleanup;
  }
  rc = sqlite3_blob_write(blobChunksRowids, &rowid, sizeof(i64),
                          chunk_offset * sizeof(i64));
  if (rc != SQLITE_OK) {
    vtab_set_error(
        &p->base, VEC_INTERAL_ERROR "could not write rowids blob on %s.%s.%lld",
        p->schemaName, p->shadowChunksName, chunk_rowid);
    rc = SQLITE_ERROR;
    goto cleanup;
  }

  // Now with all the vectors inserted, go back and update the _rowids table
  // with the new chunk_rowid/chunk_offset values
  rc = vec0_rowids_update_position(p, rowid, chunk_rowid, chunk_offset);

cleanup:
  brc = sqlite3_blob_close(blobChunksRowids);
  if ((rc == SQLITE_OK) && (brc != SQLITE_OK)) {
    vtab_set_error(
        &p->base, VEC_INTERAL_ERROR "could not close rowids blob on %s.%s.%lld",
        p->schemaName, p->shadowChunksName, chunk_rowid);
    return brc;
  }
  return rc;
}

int vec0_write_metadata_value(vec0_vtab *p, int metadata_column_idx, i64 rowid, i64 chunk_id, i64 chunk_offset, sqlite3_value * v, int isupdate) {
  int rc;
  struct Vec0MetadataColumnDefinition * metadata_column = &p->metadata_columns[metadata_column_idx];
  vec0_metadata_column_kind kind = metadata_column->kind;

  // verify input value matches column type
  switch(kind) {
    case VEC0_METADATA_COLUMN_KIND_BOOLEAN: {
      if(sqlite3_value_type(v) != SQLITE_INTEGER || ((sqlite3_value_int(v) != 0) && (sqlite3_value_int(v) != 1))) {
        rc = SQLITE_ERROR;
        vtab_set_error(&p->base, "Expected 0 or 1 for BOOLEAN metadata column %.*s", metadata_column->name_length, metadata_column->name);
        goto done;
      }
      break;
    }
    case VEC0_METADATA_COLUMN_KIND_INTEGER: {
      if(sqlite3_value_type(v) != SQLITE_INTEGER) {
        rc = SQLITE_ERROR;
        vtab_set_error(&p->base, "Expected integer for INTEGER metadata column %.*s, received %s", metadata_column->name_length, metadata_column->name, type_name(sqlite3_value_type(v)));
        goto done;
      }
      break;
    }
    case VEC0_METADATA_COLUMN_KIND_FLOAT: {
      if(sqlite3_value_type(v) != SQLITE_FLOAT) {
        rc = SQLITE_ERROR;
        vtab_set_error(&p->base, "Expected float for FLOAT metadata column %.*s, received %s", metadata_column->name_length, metadata_column->name, type_name(sqlite3_value_type(v)));
        goto done;
      }
      break;
    }
    case VEC0_METADATA_COLUMN_KIND_TEXT: {
      if(sqlite3_value_type(v) != SQLITE_TEXT) {
        rc = SQLITE_ERROR;
        vtab_set_error(&p->base, "Expected text for TEXT metadata column %.*s, received %s", metadata_column->name_length, metadata_column->name, type_name(sqlite3_value_type(v)));
        goto done;
      }
      break;
    }
  }

  sqlite3_blob * blobValue = NULL;
  rc = sqlite3_blob_open(p->db, p->schemaName, p->shadowMetadataChunksNames[metadata_column_idx], "data", chunk_id, 1, &blobValue);
  if(rc != SQLITE_OK) {
    goto done;
  }

  switch(kind) {
    case VEC0_METADATA_COLUMN_KIND_BOOLEAN: {
      u8 block;
      int value = sqlite3_value_int(v);
      rc = sqlite3_blob_read(blobValue, &block, sizeof(u8), (int) (chunk_offset / CHAR_BIT));
      if(rc != SQLITE_OK) {
        goto done;
      }

      if (value) {
        block |= 1 << (chunk_offset % CHAR_BIT);
      } else {
        block &= ~(1 << (chunk_offset % CHAR_BIT));
      }

      rc = sqlite3_blob_write(blobValue, &block, sizeof(u8), chunk_offset / CHAR_BIT);
      break;
    }
    case VEC0_METADATA_COLUMN_KIND_INTEGER: {
      i64 value = sqlite3_value_int64(v);
      rc = sqlite3_blob_write(blobValue, &value, sizeof(value), chunk_offset * sizeof(i64));
      break;
    }
    case VEC0_METADATA_COLUMN_KIND_FLOAT: {
      double value = sqlite3_value_double(v);
      rc = sqlite3_blob_write(blobValue, &value, sizeof(value), chunk_offset * sizeof(double));
      break;
    }
    case VEC0_METADATA_COLUMN_KIND_TEXT: {
      int prev_n;
      rc = sqlite3_blob_read(blobValue, &prev_n, sizeof(int), chunk_offset * VEC0_METADATA_TEXT_VIEW_BUFFER_LENGTH);
      if(rc != SQLITE_OK) {
        goto done;
      }

      const char * s = (const char *) sqlite3_value_text(v);
      int n = sqlite3_value_bytes(v);
      u8 view[VEC0_METADATA_TEXT_VIEW_BUFFER_LENGTH];
      memset(view, 0, VEC0_METADATA_TEXT_VIEW_BUFFER_LENGTH);
      memcpy(view, &n, sizeof(int));
      memcpy(view+4, s, min(n, VEC0_METADATA_TEXT_VIEW_BUFFER_LENGTH-4));

      rc = sqlite3_blob_write(blobValue, &view, VEC0_METADATA_TEXT_VIEW_BUFFER_LENGTH, chunk_offset * VEC0_METADATA_TEXT_VIEW_BUFFER_LENGTH);
      if(n > VEC0_METADATA_TEXT_VIEW_DATA_LENGTH) {
        const char * zSql;

        if(isupdate && (prev_n > VEC0_METADATA_TEXT_VIEW_DATA_LENGTH)) {
          zSql = sqlite3_mprintf("UPDATE " VEC0_SHADOW_METADATA_TEXT_DATA_NAME " SET data = ?2 WHERE rowid = ?1", p->schemaName, p->tableName, metadata_column_idx);
        }else {
          zSql = sqlite3_mprintf("INSERT INTO " VEC0_SHADOW_METADATA_TEXT_DATA_NAME " (rowid, data) VALUES (?1, ?2)", p->schemaName, p->tableName, metadata_column_idx);
        }
        if(!zSql) {
          rc = SQLITE_NOMEM;
          goto done;
        }
        sqlite3_stmt * stmt;
        rc = sqlite3_prepare_v2(p->db, zSql, -1, &stmt, NULL);
        if(rc != SQLITE_OK) {
          goto done;
        }
        sqlite3_bind_int64(stmt, 1, rowid);
        sqlite3_bind_text(stmt, 2, s, n, SQLITE_STATIC);
        rc = sqlite3_step(stmt);
        sqlite3_finalize(stmt);

        if(rc != SQLITE_DONE) {
          rc = SQLITE_ERROR;
          goto done;
        }
      }
      else if(prev_n > VEC0_METADATA_TEXT_VIEW_DATA_LENGTH) {
        const char * zSql = sqlite3_mprintf("DELETE FROM " VEC0_SHADOW_METADATA_TEXT_DATA_NAME " WHERE rowid = ?", p->schemaName, p->tableName, metadata_column_idx);
        if(!zSql) {
          rc = SQLITE_NOMEM;
          goto done;
        }
        sqlite3_stmt * stmt;
        rc = sqlite3_prepare_v2(p->db, zSql, -1, &stmt, NULL);
        if(rc != SQLITE_OK) {
          goto done;
        }
        sqlite3_bind_int64(stmt, 1, rowid);
        rc = sqlite3_step(stmt);
        sqlite3_finalize(stmt);

        if(rc != SQLITE_DONE) {
          rc = SQLITE_ERROR;
          goto done;
        }
      }
      break;
    }
  }

  if(rc != SQLITE_OK) {

  }
  rc = sqlite3_blob_close(blobValue);
  if(rc != SQLITE_OK) {
    goto done;
  }

  rc = vec0_zonemap_widen(p, metadata_column_idx, chunk_id, v);

  done:
    return rc;
}


/**
 * @brief Handles INSERT INTO operations on a vec0 table.
 *
 * @return int SQLITE_OK on success, otherwise error code on failure
 */
// Forward declaration: needed for INSERT OR REPLACE handling in vec0Update_Insert
int vec0Update_Delete(sqlite3_vtab *pVTab, sqlite3_value *idValue);

int vec0Update_Insert(sqlite3_vtab *pVTab, int argc, sqlite3_value **argv,
                      sqlite_int64 *pRowid) {
  UNUSED_PARAMETER(argc);
  vec0_vtab *p = (vec0_vtab *)pVTab;
  int rc;
  // Rowid for the inserted row, deterimined by the inserted ID + _rowids shadow
  // table
  i64 rowid;

  // Array to hold the vector data of the inserted row. Individual elements will
  // have a lifetime bound to the argv[..] values.
  void *vectorDatas[VEC0_MAX_VECTOR_COLUMNS];
  // Array to hold cleanup functions for vectorDatas[]
  vector_cleanup cleanups[VEC0_MAX_VECTOR_COLUMNS];

  sqlite3_value * partitionKeyValues[VEC0_MAX_PARTITION_COLUMNS];

  // Rowid of the chunk in the _chunks shadow table that the row will be a part
  // of.
  i64 chunk_rowid;
  // offset within the chunk where the rowid belongs
  i64 chunk_offset;

  // a write-able blob of the validity column for the given chunk. Used to mark
  // validity bit
  sqlite3_blob *blobChunksValidity = NULL;
  // buffer for the valididty column for the given chunk. Maybe not needed here?
  const unsigned char *bufferChunksValidity = NULL;
  int numReadVectors = 0;

  // Read all provided partition key values into partitionKeyValues
  for (int i = 0; i < vec0_num_defined_user_columns(p); i++) {
    if(p->user_column_kinds[i] != SQLITE_VEC0_USER_COLUMN_KIND_PARTITION) {
      continue;
    }
    int partition_key_idx = p->user_column_idxs[i];
    partitionKeyValues[partition_key_idx] = argv[2+VEC0_COLUMN_USERN_START + i];

    int new_value_type = sqlite3_value_type(partitionKeyValues[partition_key_idx]);
    if((new_value_type != SQLITE_NULL) && (new_value_type != p->paritition_columns[partition_key_idx].type)) {
      // IMP: V11454_28292
      vtab_set_error(
        pVTab,
        "Parition key type mismatch: The partition key column %.*s has type %s, but %s was provided.",
        p->paritition_columns[partition_key_idx].name_length,
        p->paritition_columns[partition_key_idx].name,
        type_name(p->paritition_columns[partition_key_idx].type),
        type_name(new_value_type)
      );
      rc = SQLITE_ERROR;
      goto cleanup;
    }
  }

  // read all the inserted vectors  into vectorDatas, validate their lengths.
  for (int i = 0; i < vec0_num_defined_user_columns(p); i++) {
    if(p->user_column_kinds[i] != SQLITE_VEC0_USER_COLUMN_KIND_VECTOR) {
      continue;
    }
    int vector_column_idx = p->user_column_idxs[i];
    sqlite3_value *valueVector = argv[2 + VEC0_COLUMN_USERN_START + i];
    size_t dimensions;

    char *pzError;
    enum VectorElementType elementType;
    rc = vector_from_value(valueVector, &vectorDatas[vector_column_idx], &dimensions,
                           &elementType, &cleanups[vector_column_idx], &pzError);
    if (rc != SQLITE_OK) {
      // IMP: V06519_23358
      vtab_set_error(
          pVTab, "Inserted vector for the \"%.*s\" column is invalid: %z",
          p->vector_columns[vector_column_idx].name_length, p->vector_columns[vector_column_idx].name, pzError);
      rc = SQLITE_ERROR;
      goto cleanup;
    }

    numReadVectors++;
    if (elementType != p->vector_columns[vector_column_idx].element_type) {
      // IMP: V08221_25059
      vtab_set_error(
          pVTab,
          "Inserted vector for the \"%.*s\" column is expected to be of type "
          "%s, but a %s vector was provided.",
          p->vector_columns[i].name_length, p->vector_columns[i].name,
          vector_subtype_name(p->vector_columns[i].element_type),
          vector_subtype_name(elementType));
      rc = SQLITE_ERROR;
      goto cleanup;
    }

    if (dimensions != p->vector_columns[vector_column_idx].dimensions) {
      // IMP: V01145_17984
      vtab_set_error(
          pVTab,
          "Dimension mismatch for inserted vector for the \"%.*s\" column. "
          "Expected %d dimensions but received %d.",
          p->vector_columns[vector_column_idx].name_length, p->vector_columns[vector_column_idx].name,
          p->vector_columns[vector_column_idx].dimensions, dimensions);
      rc = SQLITE_ERROR;
      goto cleanup;
    }
  }

  // Cannot insert a value in the hidden "distance" column
  if (sqlite3_value_type(argv[2 + vec0_column_distance_idx(p)]) !=
      SQLITE_NULL) {
    // IMP: V24228_08298
    vtab_set_error(pVTab,
                   "A value was provided for the hidden \"distance\" column.");
    rc = SQLITE_ERROR;
    goto cleanup;
  }
  // Cannot insert a value in the hidden "k" column
  if (sqlite3_value_type(argv[2 + vec0_column_k_idx(p)]) != SQLITE_NULL) {
    // IMP: V11875_28713
    vtab_set_error(pVTab, "A value was provided for the hidden \"k\" column.");
    rc = SQLITE_ERROR;
    goto cleanup;
  }
  // Cannot insert a value in the hidden "mmr_lambda" column
  if (sqlite3_value_type(argv[2 + vec0_column_mmr_lambda_idx(p)]) != SQLITE_NULL) {
    vtab_set_error(pVTab, "A value was provided for the hidden \"mmr_lambda\" column.");
    rc = SQLITE_ERROR;
    goto cleanup;
  }

  // Handle INSERT OR REPLACE: if the conflict resolution is REPLACE and the
  // row already exists, delete the existing row first before inserting.
  if (sqlite3_vtab_on_conflict(p->db) == SQLITE_REPLACE) {
    sqlite3_value *idValue = argv[2 + VEC0_COLUMN_ID];
    int idType = sqlite3_value_type(idValue);
    int existingRowExists = 0;

    if (p->pkIsText && idType == SQLITE_TEXT) {
      i64 existingRowid;
      rc = vec0_rowid_from_id(p, idValue, &existingRowid);
      if (rc == SQLITE_OK) {
        existingRowExists = 1;
      } else if (rc == SQLITE_EMPTY) {
        rc = SQLITE_OK; // row doesn't exist, proceed with normal insert
      } else {
        goto cleanup;
      }
    } else if (!p->pkIsText && idType == SQLITE_INTEGER) {
      i64 existingRowid = sqlite3_value_int64(idValue);
      i64 chunk_id_tmp, chunk_offset_tmp;
      rc = vec0_get_chunk_position(p, existingRowid, NULL, &chunk_id_tmp, &chunk_offset_tmp);
      if (rc == SQLITE_OK) {
        existingRowExists = 1;
      } else if (rc == SQLITE_EMPTY) {
        rc = SQLITE_OK; // row doesn't exist, proceed with normal insert
      } else {
        goto cleanup;
      }
    }

    if (existingRowExists) {
      rc = vec0Update_Delete(pVTab, idValue);
      if (rc != SQLITE_OK) {
        goto cleanup;
      }
    }
  }

  // Step #1: Insert/get a rowid for this row, from the _rowids table.
  rc = vec0Update_InsertRowidStep(p, argv[2 + VEC0_COLUMN_ID], &rowid);
  if (rc != SQLITE_OK) {
    goto cleanup;
  }

  if (!vec0_all_columns_diskann(p)) {
    // Step #2: Find the next "available" position in the _chunks table for this
    // row.
    rc = vec0Update_InsertNextAvailableStep(p, partitionKeyValues,
    &chunk_rowid, &chunk_offset,
                                            &blobChunksValidity,
                                            &bufferChunksValidity);
    if (rc != SQLITE_OK) {
      goto cleanup;
    }

    // Step #3: With the next available chunk position, write out all the vectors
    //          to their specified location.
    rc = vec0Update_InsertWriteFinalStep(p, chunk_rowid, chunk_offset, rowid,
                                         vectorDatas, blobChunksValidity,
                                         bufferChunksValidity);
    if (rc != SQLITE_OK) {
      goto cleanup;
    }
  }

#if SQLITE_VEC_ENABLE_DISKANN
  // Step #4: Insert into DiskANN graph for indexed vector columns
  for (int i = 0; i < p->numVectorColumns; i++) {
    if (p->vector_columns[i].index_type != VEC0_INDEX_TYPE_DISKANN) continue;
    rc = diskann_insert(p, i, rowid, vectorDatas[i]);
    if (rc != SQLITE_OK) {
      goto cleanup;
    }
  }
#endif

#if SQLITE_VEC_ENABLE_RESCORE
  rc = rescore_on_insert(p, chunk_rowid, chunk_offset, rowid, vectorDatas);
  if (rc != SQLITE_OK) {
    goto cleanup;
  }
#endif

#if SQLITE_VEC_ENABLE_HNSW
  for (int i = 0; i < p->numVectorColumns; i++) {
    if (p->vector_columns[i].index_type != VEC0_INDEX_TYPE_HNSW) continue;
    rc = hnsw_insert(p, i, rowid, vectorDatas[i]);
    if (rc != SQLITE_OK) {
      goto cleanup;
    }
  }
#endif

#if SQLITE_VEC_EXPERIMENTAL_IVF_ENABLE
  // Step #4: IVF index insert (if any vector column uses IVF)
  for (int i = 0; i < p->numVectorColumns; i++) {
    if (p->vector_columns[i].index_type != VEC0_INDEX_TYPE_IVF) continue;
    int vecSize = (int)vector_column_byte_size(p->vector_columns[i]);
    rc = ivf_insert(p, i, rowid, vectorDatas[i], vecSize);
    if (rc != SQLITE_OK) {
      goto cleanup;
    }
  }
#endif

  // Validate every auxiliary value first, in auxiliary-column order.
  sqlite3_value *auxValues[VEC0_MAX_AUXILIARY_COLUMNS];
  if(p->numAuxiliaryColumns > 0) {
    for (int i = 0; i < vec0_num_defined_user_columns(p); i++) {
      if(p->user_column_kinds[i] != SQLITE_VEC0_USER_COLUMN_KIND_AUXILIARY) {
        continue;
      }
      int auxiliary_key_idx = p->user_column_idxs[i];
      sqlite3_value * v = argv[2+VEC0_COLUMN_USERN_START + i];
      int v_type = sqlite3_value_type(v);
      if(v_type != SQLITE_NULL && (v_type != p->auxiliary_columns[auxiliary_key_idx].type)) {
        rc = SQLITE_CONSTRAINT;
        vtab_set_error(
          pVTab,
          "Auxiliary column type mismatch: The auxiliary column %.*s has type %s, but %s was provided.",
          p->auxiliary_columns[auxiliary_key_idx].name_length,
          p->auxiliary_columns[auxiliary_key_idx].name,
          type_name(p->auxiliary_columns[auxiliary_key_idx].type),
          type_name(v_type)
        );
        goto cleanup;
      }
      auxValues[auxiliary_key_idx] = v;
    }

    if (!p->storage) {
      sqlite3_stmt *stmt;
      sqlite3_str * s = sqlite3_str_new(NULL);
      sqlite3_str_appendf(s, "INSERT INTO " VEC0_SHADOW_AUXILIARY_NAME "(rowid ", p->schemaName, p->tableName);
      for(int i = 0; i < p->numAuxiliaryColumns; i++) {
        sqlite3_str_appendf(s, ", value%02d", i);
      }
      sqlite3_str_appendall(s, ") VALUES (? ");
      for(int i = 0; i < p->numAuxiliaryColumns; i++) {
        sqlite3_str_appendall(s, ", ?");
      }
      sqlite3_str_appendall(s, ")");
      char * zSql = sqlite3_str_finish(s);
      if(!zSql) {
        rc = SQLITE_NOMEM;
        goto cleanup;
      }
      rc = sqlite3_prepare_v2(p->db, zSql, -1, &stmt, NULL);
      sqlite3_free(zSql);
      if(rc != SQLITE_OK) {
        goto cleanup;
      }
      sqlite3_bind_int64(stmt, 1, rowid);
      for (int i = 0; i < p->numAuxiliaryColumns; i++) {
        // 1-based binds, after the rowid parameter
        sqlite3_bind_value(stmt, 1 + 1 + i, auxValues[i]);
      }
      rc = sqlite3_step(stmt);
      sqlite3_finalize(stmt);
      if(rc != SQLITE_DONE) {
        rc = SQLITE_ERROR;
        goto cleanup;
      }
    }
  }
  if (p->storage) {
    rc = vec0_storage_error(p, p->storage->xPut(p->storageHandle, rowid, auxValues,
                                                (const void *const *)vectorDatas));
    if (rc != SQLITE_OK) goto cleanup;
  }


  for(int i = 0; i < vec0_num_defined_user_columns(p); i++) {
    if(p->user_column_kinds[i] != SQLITE_VEC0_USER_COLUMN_KIND_METADATA) {
      continue;
    }
    int metadata_idx = p->user_column_idxs[i];
    sqlite3_value *v = argv[2 + VEC0_COLUMN_USERN_START + i];
    rc = vec0_write_metadata_value(p, metadata_idx, rowid, chunk_rowid, chunk_offset, v, 0);
    if(rc != SQLITE_OK) {
      goto cleanup;
    }
  }

  *pRowid = rowid;
  rc = SQLITE_OK;

cleanup:
  for (int i = 0; i < numReadVectors; i++) {
    cleanups[i](vectorDatas[i]);
  }
  sqlite3_free((void *)bufferChunksValidity);
  int brc = sqlite3_blob_close(blobChunksValidity);
  if ((rc == SQLITE_OK) && (brc != SQLITE_OK)) {
    vtab_set_error(&p->base,
                   VEC_INTERAL_ERROR "unknown error, blobChunksValidity could "
                                     "not be closed, please file an issue");
    return brc;
  }
  return rc;
}

// Clears the row's validity bit, and sets *chunkEmpty to whether that was the
// chunk's last set bit.
int vec0Update_Delete_ClearValidity(vec0_vtab *p, i64 chunk_id,
                                    u64 chunk_offset, int *chunkEmpty) {
  int rc, brc;
  sqlite3_blob *blobChunksValidity = NULL;
  char unsigned bx;
  int validityOffset = chunk_offset / CHAR_BIT;

  // 2. ensure chunks.validity bit is 1, then set to 0
  rc = sqlite3_blob_open(p->db, p->schemaName, p->shadowChunksName, "validity",
                         chunk_id, 1, &blobChunksValidity);
  if (rc != SQLITE_OK) {
    // IMP: V26002_10073
    vtab_set_error(&p->base, "could not open validity blob for %s.%s.%lld",
                   p->schemaName, p->shadowChunksName, chunk_id);
    return SQLITE_ERROR;
  }
  // will skip the sqlite3_blob_bytes(blobChunksValidity) check for now,
  // the read below would catch it

  rc = sqlite3_blob_read(blobChunksValidity, &bx, sizeof(bx), validityOffset);
  if (rc != SQLITE_OK) {
    // IMP: V21193_05263
    vtab_set_error(
        &p->base, "could not read validity blob for %s.%s.%lld at %d",
        p->schemaName, p->shadowChunksName, chunk_id, validityOffset);
    goto cleanup;
  }
  if (!((bx >> (chunk_offset % CHAR_BIT)) & 1)) {
    // IMP: V21193_05263
    rc = SQLITE_ERROR;
    vtab_set_error(
        &p->base,
        "vec0 deletion error: validity bit is not set for %s.%s.%lld at %d",
        p->schemaName, p->shadowChunksName, chunk_id, validityOffset);
    goto cleanup;
  }
  char unsigned mask = ~(1 << (chunk_offset % CHAR_BIT));
  char result = bx & mask;
  rc = sqlite3_blob_write(blobChunksValidity, &result, sizeof(bx),
                          validityOffset);
  if (rc != SQLITE_OK) {
    vtab_set_error(
        &p->base, "could not write to validity blob for %s.%s.%lld at %d",
        p->schemaName, p->shadowChunksName, chunk_id, validityOffset);
    goto cleanup;
  }

  // Another set bit in this byte keeps the chunk, as it does for most deletes,
  // so the rest of the bitmap is read only once this byte is zero.
  *chunkEmpty = 0;
  if (result == 0) {
    int validitySize = sqlite3_blob_bytes(blobChunksValidity);
    unsigned char *validity = sqlite3_malloc(validitySize);
    if (!validity) {
      rc = SQLITE_NOMEM;
      goto cleanup;
    }
    rc = sqlite3_blob_read(blobChunksValidity, validity, validitySize, 0);
    if (rc == SQLITE_OK) {
      *chunkEmpty = 1;
      for (int i = 0; i < validitySize; i++) {
        if (validity[i] != 0) {
          *chunkEmpty = 0;
          break;
        }
      }
    } else {
      vtab_set_error(&p->base, "could not read validity blob for %s.%s.%lld",
                     p->schemaName, p->shadowChunksName, chunk_id);
    }
    sqlite3_free(validity);
  }

cleanup:

  brc = sqlite3_blob_close(blobChunksValidity);
  if (rc != SQLITE_OK)
    return rc;
  if (brc != SQLITE_OK) {
    vtab_set_error(&p->base,
                   "vec0 deletion error: Error commiting validity blob "
                   "transaction on %s.%s.%lld at %d",
                   p->schemaName, p->shadowChunksName, chunk_id,
                   validityOffset);
    return brc;
  }
  return SQLITE_OK;
}

int vec0Update_Delete_ClearRowid(vec0_vtab *p, i64 chunk_id,
                                  u64 chunk_offset) {
  int rc, brc;
  sqlite3_blob *blobChunksRowids = NULL;
  i64 zero = 0;

  rc = sqlite3_blob_open(p->db, p->schemaName, p->shadowChunksName, "rowids",
                         chunk_id, 1, &blobChunksRowids);
  if (rc != SQLITE_OK) {
    vtab_set_error(&p->base, "could not open rowids blob for %s.%s.%lld",
                   p->schemaName, p->shadowChunksName, chunk_id);
    return SQLITE_ERROR;
  }

  rc = sqlite3_blob_write(blobChunksRowids, &zero, sizeof(zero),
                          chunk_offset * sizeof(i64));
  if (rc != SQLITE_OK) {
    vtab_set_error(&p->base,
                   "could not write to rowids blob for %s.%s.%lld at %llu",
                   p->schemaName, p->shadowChunksName, chunk_id, chunk_offset);
  }

  brc = sqlite3_blob_close(blobChunksRowids);
  if (rc != SQLITE_OK)
    return rc;
  if (brc != SQLITE_OK) {
    vtab_set_error(&p->base,
                   "vec0 deletion error: Error commiting rowids blob "
                   "transaction on %s.%s.%lld at %llu",
                   p->schemaName, p->shadowChunksName, chunk_id, chunk_offset);
    return brc;
  }
  return SQLITE_OK;
}

int vec0Update_Delete_ClearVectors(vec0_vtab *p, i64 chunk_id,
                                    u64 chunk_offset) {
  int rc, brc;
  for (int i = 0; i < p->numVectorColumns; i++) {
    // Non-FLAT columns (rescore, IVF, DiskANN) don't use _vector_chunks
    if (!vec0_column_in_chunks(p, i))
      continue;
    sqlite3_blob *blobVectors = NULL;
    size_t n = vector_column_byte_size(p->vector_columns[i]);

    rc = sqlite3_blob_open(p->db, p->schemaName,
                           p->shadowVectorChunksNames[i], "vectors",
                           chunk_id, 1, &blobVectors);
    if (rc != SQLITE_OK) {
      vtab_set_error(&p->base,
                     "could not open vector blob for %s.%s.%lld column %d",
                     p->schemaName, p->shadowVectorChunksNames[i], chunk_id, i);
      return SQLITE_ERROR;
    }

    void *zeroBuf = sqlite3_malloc(n);
    if (!zeroBuf) {
      sqlite3_blob_close(blobVectors);
      return SQLITE_NOMEM;
    }
    memset(zeroBuf, 0, n);

    rc = sqlite3_blob_write(blobVectors, zeroBuf, n, chunk_offset * n);
    sqlite3_free(zeroBuf);
    if (rc != SQLITE_OK) {
      vtab_set_error(
          &p->base,
          "could not write to vector blob for %s.%s.%lld at %llu column %d",
          p->schemaName, p->shadowVectorChunksNames[i], chunk_id,
          chunk_offset, i);
    }

    brc = sqlite3_blob_close(blobVectors);
    if (rc != SQLITE_OK)
      return rc;
    if (brc != SQLITE_OK) {
      vtab_set_error(&p->base,
                     "vec0 deletion error: Error commiting vector blob "
                     "transaction on %s.%s.%lld column %d",
                     p->schemaName, p->shadowVectorChunksNames[i], chunk_id, i);
      return brc;
    }
  }
  return SQLITE_OK;
}

// Deletes a chunk whose last row a DELETE cleared, with its rows in every
// chunk table.
int vec0Update_Delete_DeleteChunk(vec0_vtab *p, i64 chunk_id) {
  int rc;
  char *zSql;
  sqlite3_stmt *stmt;

  // Delete from _chunks
  zSql = sqlite3_mprintf(
      "DELETE FROM " VEC0_SHADOW_CHUNKS_NAME " WHERE rowid = ?",
      p->schemaName, p->tableName);
  if (!zSql)
    return SQLITE_NOMEM;
  rc = sqlite3_prepare_v2(p->db, zSql, -1, &stmt, NULL);
  sqlite3_free(zSql);
  if (rc != SQLITE_OK)
    return rc;
  sqlite3_bind_int64(stmt, 1, chunk_id);
  rc = sqlite3_step(stmt);
  sqlite3_finalize(stmt);
  if (rc != SQLITE_DONE)
    return SQLITE_ERROR;

  // Delete from each _vector_chunksNN
  for (int i = 0; i < p->numVectorColumns; i++) {
    // Non-FLAT columns (rescore, IVF, DiskANN) don't use _vector_chunks
    if (!vec0_column_in_chunks(p, i))
      continue;
    zSql = sqlite3_mprintf(
        "DELETE FROM " VEC0_SHADOW_VECTOR_N_NAME " WHERE rowid = ?",
        p->schemaName, p->tableName, i);
    if (!zSql)
      return SQLITE_NOMEM;
    rc = sqlite3_prepare_v2(p->db, zSql, -1, &stmt, NULL);
    sqlite3_free(zSql);
    if (rc != SQLITE_OK)
      return rc;
    sqlite3_bind_int64(stmt, 1, chunk_id);
    rc = sqlite3_step(stmt);
    sqlite3_finalize(stmt);
    if (rc != SQLITE_DONE)
      return SQLITE_ERROR;
  }

#if SQLITE_VEC_ENABLE_RESCORE
  rc = rescore_delete_chunk(p, chunk_id);
  if (rc != SQLITE_OK)
    return rc;
#endif

  // Delete from each _metadatachunksNN
  for (int i = 0; i < p->numMetadataColumns; i++) {
    zSql = sqlite3_mprintf(
        "DELETE FROM " VEC0_SHADOW_METADATA_N_NAME " WHERE rowid = ?",
        p->schemaName, p->tableName, i);
    if (!zSql)
      return SQLITE_NOMEM;
    rc = sqlite3_prepare_v2(p->db, zSql, -1, &stmt, NULL);
    sqlite3_free(zSql);
    if (rc != SQLITE_OK)
      return rc;
    sqlite3_bind_int64(stmt, 1, chunk_id);
    rc = sqlite3_step(stmt);
    sqlite3_finalize(stmt);
    if (rc != SQLITE_DONE)
      return SQLITE_ERROR;
  }

  // Remove bounds before this chunk id can be reused.
  if (p->hasCommandColumn && p->numMetadataColumns) {
    zSql = sqlite3_mprintf("DELETE FROM " VEC0_SHADOW_INFO_NAME
                          " WHERE key=?1", p->schemaName, p->tableName);
    if (!zSql) return SQLITE_NOMEM;
    rc = sqlite3_prepare_v2(p->db, zSql, -1, &stmt, NULL);
    sqlite3_free(zSql);
    if (rc != SQLITE_OK) return rc;
    for (int i = 0; i < p->numMetadataColumns; i++) {
      char key[64];
      sqlite3_snprintf(sizeof(key), key, "zonemap_%02d_%lld", i, chunk_id);
      rc = sqlite3_bind_text(stmt, 1, key, -1, SQLITE_STATIC);
      if (rc == SQLITE_OK) rc = sqlite3_step(stmt);
      sqlite3_reset(stmt);
      sqlite3_clear_bindings(stmt);
      if (rc != SQLITE_DONE) break;
    }
    sqlite3_finalize(stmt);
    if (rc != SQLITE_DONE) return rc;
  }

  // Invalidate cached stmtLatestChunk so it gets re-prepared on next insert.
  if (p->stmtLatestChunk) {
    sqlite3_finalize(p->stmtLatestChunk);
    p->stmtLatestChunk = NULL;
  }

  return SQLITE_OK;
}

int vec0Update_Delete_DeleteRowids(vec0_vtab *p, i64 rowid) {
  int rc;
  sqlite3_stmt *stmt = NULL;

  char *zSql =
      sqlite3_mprintf("DELETE FROM " VEC0_SHADOW_ROWIDS_NAME " WHERE rowid = ?",
                      p->schemaName, p->tableName);
  if (!zSql) {
    return SQLITE_NOMEM;
  }

  rc = sqlite3_prepare_v2(p->db, zSql, -1, &stmt, NULL);
  sqlite3_free(zSql);
  if (rc != SQLITE_OK) {
    goto cleanup;
  }
  sqlite3_bind_int64(stmt, 1, rowid);
  rc = sqlite3_step(stmt);
  if (rc != SQLITE_DONE) {
    goto cleanup;
  }
  rc = SQLITE_OK;

cleanup:
  sqlite3_finalize(stmt);
  return rc;
}

int vec0Update_Delete_DeleteAux(vec0_vtab *p, i64 rowid) {
  int rc;
  sqlite3_stmt *stmt = NULL;
  if (p->storage) {
    return vec0_storage_error(p, p->storage->xDelete(p->storageHandle, rowid));
  }

  char *zSql =
      sqlite3_mprintf("DELETE FROM " VEC0_SHADOW_AUXILIARY_NAME " WHERE rowid = ?",
                      p->schemaName, p->tableName);
  if (!zSql) {
    return SQLITE_NOMEM;
  }

  rc = sqlite3_prepare_v2(p->db, zSql, -1, &stmt, NULL);
  sqlite3_free(zSql);
  if (rc != SQLITE_OK) {
    goto cleanup;
  }
  sqlite3_bind_int64(stmt, 1, rowid);
  rc = sqlite3_step(stmt);
  if (rc != SQLITE_DONE) {
    goto cleanup;
  }
  rc = SQLITE_OK;

cleanup:
  sqlite3_finalize(stmt);
  return rc;
}

int vec0Update_Delete_ClearMetadata(vec0_vtab *p, int metadata_idx, i64 rowid, i64 chunk_id,
                                    u64 chunk_offset) {
  int rc;
  sqlite3_blob * blobValue;
  vec0_metadata_column_kind kind = p->metadata_columns[metadata_idx].kind;
  rc = sqlite3_blob_open(p->db, p->schemaName, p->shadowMetadataChunksNames[metadata_idx], "data", chunk_id, 1, &blobValue);
  if(rc != SQLITE_OK) {
    return rc;
  }

  switch(kind) {
    case VEC0_METADATA_COLUMN_KIND_BOOLEAN: {
      u8 block;
      rc = sqlite3_blob_read(blobValue, &block, sizeof(u8), (int) (chunk_offset / CHAR_BIT));
      if(rc != SQLITE_OK) {
        goto done;
      }

      block &= ~(1 << (chunk_offset % CHAR_BIT));
      rc = sqlite3_blob_write(blobValue, &block, sizeof(u8), chunk_offset / CHAR_BIT);
      break;
    }
    case VEC0_METADATA_COLUMN_KIND_INTEGER: {
      i64 v = 0;
      rc = sqlite3_blob_write(blobValue, &v, sizeof(v), chunk_offset * sizeof(i64));
      break;
    }
    case VEC0_METADATA_COLUMN_KIND_FLOAT: {
      double v = 0;
      rc = sqlite3_blob_write(blobValue, &v, sizeof(v), chunk_offset * sizeof(double));
      break;
    }
    case VEC0_METADATA_COLUMN_KIND_TEXT: {
      int n;
      rc = sqlite3_blob_read(blobValue, &n, sizeof(int), chunk_offset * VEC0_METADATA_TEXT_VIEW_BUFFER_LENGTH);
      if(rc != SQLITE_OK) {
        goto done;
      }

      u8 view[VEC0_METADATA_TEXT_VIEW_BUFFER_LENGTH];
      memset(view, 0, VEC0_METADATA_TEXT_VIEW_BUFFER_LENGTH);
      rc = sqlite3_blob_write(blobValue, &view, sizeof(view), chunk_offset * VEC0_METADATA_TEXT_VIEW_BUFFER_LENGTH);
      if(rc != SQLITE_OK) {
        goto done;
      }

      if(n > VEC0_METADATA_TEXT_VIEW_DATA_LENGTH) {
        const char * zSql = sqlite3_mprintf("DELETE FROM " VEC0_SHADOW_METADATA_TEXT_DATA_NAME " WHERE rowid = ?", p->schemaName, p->tableName, metadata_idx);
        if(!zSql) {
          rc = SQLITE_NOMEM;
          goto done;
        }
        sqlite3_stmt * stmt;
        rc = sqlite3_prepare_v2(p->db, zSql, -1, &stmt, NULL);
        if(rc != SQLITE_OK) {
          goto done;
        }
        sqlite3_bind_int64(stmt, 1, rowid);
        rc = sqlite3_step(stmt);
        sqlite3_finalize(stmt);
        if(rc != SQLITE_DONE) {
          rc = SQLITE_ERROR;
          goto done;
        }
        // Fix for https://github.com/asg017/sqlite-vec/issues/274
        // sqlite3_step returns SQLITE_DONE (101) on DML success, but the
        // `done:` epilogue treats anything other than SQLITE_OK as an error.
        // Without this, SQLITE_DONE propagates up to vec0Update_Delete,
        // which aborts the DELETE scan and silently drops remaining rows.
        rc = SQLITE_OK;
      }
      break;
    }
  }
  int rc2;
  done:
  rc2 = sqlite3_blob_close(blobValue);
  if(rc == SQLITE_OK) {
    return rc2;
  }
  return rc;
}

int vec0Update_Delete(sqlite3_vtab *pVTab, sqlite3_value *idValue) {
  vec0_vtab *p = (vec0_vtab *)pVTab;
  int rc = vec0_flat_flush(p);
  if (rc != SQLITE_OK) return rc;
  i64 rowid;
  i64 chunk_id = 0;
  i64 chunk_offset = 0;
  int chunkEmpty = 0;

  if (p->pkIsText) {
    rc = vec0_rowid_from_id(p, idValue, &rowid);
    if (rc != SQLITE_OK) {
      return rc;
    }
  } else {
    rowid = sqlite3_value_int64(idValue);
  }

  // 1. Find chunk position for given rowid
  // 2. Ensure that validity bit for position is 1, then set to 0
  // 3. Zero out rowid in chunks.rowid
  // 4. Zero out vector data in all vector column chunks
  // 5. Delete value in _rowids table

#if SQLITE_VEC_ENABLE_DISKANN
  // DiskANN graph deletion for indexed columns
  for (int i = 0; i < p->numVectorColumns; i++) {
    if (p->vector_columns[i].index_type != VEC0_INDEX_TYPE_DISKANN) continue;
    rc = diskann_delete(p, i, rowid);
    if (rc != SQLITE_OK) {
      return rc;
    }
  }
#endif

  if (!vec0_all_columns_diskann(p)) {
    // 1. get chunk_id and chunk_offset from _rowids
    rc = vec0_get_chunk_position(p, rowid, NULL, &chunk_id, &chunk_offset);
    if (rc != SQLITE_OK) {
      return rc;
    }

    // 2. clear validity bit
    rc =
        vec0Update_Delete_ClearValidity(p, chunk_id, chunk_offset, &chunkEmpty);
    if (rc != SQLITE_OK) {
      return rc;
    }

    // 3. zero out rowid in chunks.rowids
    rc = vec0Update_Delete_ClearRowid(p, chunk_id, chunk_offset);
    if (rc != SQLITE_OK) {
      return rc;
    }

    // 4. zero out any data in vector chunks tables
    rc = vec0Update_Delete_ClearVectors(p, chunk_id, chunk_offset);
    if (rc != SQLITE_OK) {
      return rc;
    }

#if SQLITE_VEC_ENABLE_RESCORE
    // 4b. zero out quantized data in rescore chunk tables, delete from rescore vectors
    rc = rescore_on_delete(p, chunk_id, chunk_offset, rowid);
    if (rc != SQLITE_OK) {
      return rc;
    }
#endif
  }

  // 5. delete from _rowids table
  rc = vec0Update_Delete_DeleteRowids(p, rowid);
  if (rc != SQLITE_OK) {
    return rc;
  }

  // 6. delete any auxiliary rows
  if(p->numAuxiliaryColumns > 0 || p->storage) {
    rc = vec0Update_Delete_DeleteAux(p, rowid);
    if (rc != SQLITE_OK) {
      return rc;
    }
  }

  // 7. delete metadata and reclaim chunk (only when using chunk-based storage)
  if (!vec0_all_columns_diskann(p)) {
    for(int i = 0; i < p->numMetadataColumns; i++) {
      rc = vec0Update_Delete_ClearMetadata(p, i, rowid, chunk_id, chunk_offset);
      if (rc != SQLITE_OK) {
        return rc;
      }
    }

    // 8. reclaim chunk if fully empty
    if (chunkEmpty) {
      rc = vec0Update_Delete_DeleteChunk(p, chunk_id);
      if (rc != SQLITE_OK) {
        return rc;
      }
    }
  }

#if SQLITE_VEC_ENABLE_HNSW
  for (int i = 0; i < p->numVectorColumns; i++) {
    if (p->vector_columns[i].index_type != VEC0_INDEX_TYPE_HNSW) continue;
    rc = hnsw_delete(p, i, rowid);
    if (rc != SQLITE_OK) return rc;
  }
#endif

#if SQLITE_VEC_EXPERIMENTAL_IVF_ENABLE
  // 7. delete from IVF index
  for (int i = 0; i < p->numVectorColumns; i++) {
    if (p->vector_columns[i].index_type != VEC0_INDEX_TYPE_IVF) continue;
    rc = ivf_delete(p, i, rowid);
    if (rc != SQLITE_OK) return rc;
  }
#endif

  return SQLITE_OK;
}

int vec0Update_UpdateAuxColumn(vec0_vtab *p, int auxiliary_column_idx, sqlite3_value * value, i64 rowid) {
  int rc;
  sqlite3_stmt *stmt;
  if (p->storage) {
    return vec0_storage_error(
        p, p->storage->xSet(p->storageHandle, rowid, auxiliary_column_idx, value));
  }
  const char * zSql = sqlite3_mprintf("UPDATE " VEC0_SHADOW_AUXILIARY_NAME " SET value%02d = ? WHERE rowid = ?", p->schemaName, p->tableName, auxiliary_column_idx);
  if(!zSql) {
    return SQLITE_NOMEM;
  }
  rc = sqlite3_prepare_v2(p->db, zSql, -1, &stmt, NULL);
  if(rc != SQLITE_OK) {
    return rc;
  }
  sqlite3_bind_value(stmt, 1, value);
  sqlite3_bind_int64(stmt, 2, rowid);
  rc = sqlite3_step(stmt);
  if(rc != SQLITE_DONE) {
    sqlite3_finalize(stmt);
    return SQLITE_ERROR;
  }
  sqlite3_finalize(stmt);
  return SQLITE_OK;
}

int vec0Update_UpdateVectorColumn(vec0_vtab *p, i64 chunk_id, i64 chunk_offset,
                                  int i, sqlite3_value *valueVector, i64 rowid) {
  int rc;
#if !SQLITE_VEC_ENABLE_RESCORE
  UNUSED_PARAMETER(rowid);
#endif

  char *pzError;
  size_t dimensions;
  enum VectorElementType elementType;
  void *vector;
  vector_cleanup cleanup = vector_cleanup_noop;
  // https://github.com/asg017/sqlite-vec/issues/53
  rc = vector_from_value(valueVector, &vector, &dimensions, &elementType,
                         &cleanup, &pzError);
  if (rc != SQLITE_OK) {
    // IMP: V15203_32042
    vtab_set_error(
        &p->base, "Updated vector for the \"%.*s\" column is invalid: %z",
        p->vector_columns[i].name_length, p->vector_columns[i].name, pzError);
    rc = SQLITE_ERROR;
    goto cleanup;
  }
  if (elementType != p->vector_columns[i].element_type) {
    // IMP: V03643_20481
    vtab_set_error(
        &p->base,
        "Updated vector for the \"%.*s\" column is expected to be of type "
        "%s, but a %s vector was provided.",
        p->vector_columns[i].name_length, p->vector_columns[i].name,
        vector_subtype_name(p->vector_columns[i].element_type),
        vector_subtype_name(elementType));
    rc = SQLITE_ERROR;
    goto cleanup;
  }
  if (dimensions != p->vector_columns[i].dimensions) {
    // IMP: V25739_09810
    vtab_set_error(
        &p->base,
        "Dimension mismatch for new updated vector for the \"%.*s\" column. "
        "Expected %d dimensions but received %d.",
        p->vector_columns[i].name_length, p->vector_columns[i].name,
        p->vector_columns[i].dimensions, dimensions);
    rc = SQLITE_ERROR;
    goto cleanup;
  }

  if (p->storage) {
    rc = vec0_storage_error(p, p->storage->xSetVector(p->storageHandle, rowid, i, vector));
    goto cleanup;
  }

#if SQLITE_VEC_ENABLE_RESCORE
  if (p->vector_columns[i].index_type == VEC0_INDEX_TYPE_RESCORE) {
    // For rescore columns, update _rescore_vectors and _rescore_chunks
    struct VectorColumnDefinition *col = &p->vector_columns[i];
    size_t qsize = rescore_quantized_byte_size(col);
    size_t fsize = vector_column_byte_size(*col);

    // 1. Update quantized chunk
    {
      void *qbuf = sqlite3_malloc(qsize);
      if (!qbuf) { rc = SQLITE_NOMEM; goto cleanup; }
      switch (col->rescore.quantizer_type) {
      case VEC0_RESCORE_QUANTIZER_BIT:
        rescore_quantize_float_to_bit((const float *)vector, (uint8_t *)qbuf, col->dimensions);
        break;
      case VEC0_RESCORE_QUANTIZER_INT8:
        rescore_quantize_float_to_int8((const float *)vector, (int8_t *)qbuf, col->dimensions);
        break;
      }
      sqlite3_blob *blobQ = NULL;
      rc = sqlite3_blob_open(p->db, p->schemaName,
                             p->shadowRescoreChunksNames[i], "vectors",
                             chunk_id, 1, &blobQ);
      if (rc != SQLITE_OK) { sqlite3_free(qbuf); goto cleanup; }
      rc = sqlite3_blob_write(blobQ, qbuf, qsize, chunk_offset * qsize);
      sqlite3_free(qbuf);
      int brc2 = sqlite3_blob_close(blobQ);
      if (rc != SQLITE_OK) goto cleanup;
      if (brc2 != SQLITE_OK) { rc = brc2; goto cleanup; }
    }

    // 2. Update float vector in _rescore_vectors (keyed by user rowid)
    {
      char *zSql = sqlite3_mprintf(
          "UPDATE \"%w\".\"%w\" SET vector = ? WHERE rowid = ?",
          p->schemaName, p->shadowRescoreVectorsNames[i]);
      if (!zSql) { rc = SQLITE_NOMEM; goto cleanup; }
      sqlite3_stmt *stmtUp;
      rc = sqlite3_prepare_v2(p->db, zSql, -1, &stmtUp, NULL);
      sqlite3_free(zSql);
      if (rc != SQLITE_OK) goto cleanup;
      sqlite3_bind_blob(stmtUp, 1, vector, fsize, SQLITE_TRANSIENT);
      sqlite3_bind_int64(stmtUp, 2, rowid);
      rc = sqlite3_step(stmtUp);
      sqlite3_finalize(stmtUp);
      if (rc != SQLITE_DONE) { rc = SQLITE_ERROR; goto cleanup; }
    }

    rc = SQLITE_OK;
    goto cleanup;
  }
#endif

  rc = vec0_flat_write(p, i, chunk_id, chunk_offset, vector);

cleanup:
  cleanup(vector);
  return rc;
}

int vec0Update_Update(sqlite3_vtab *pVTab, int argc, sqlite3_value **argv) {
  UNUSED_PARAMETER(argc);
  vec0_vtab *p = (vec0_vtab *)pVTab;
  int rc;
  i64 chunk_id;
  i64 chunk_offset;

  i64 rowid;
  if (p->pkIsText) {
    const char *a = (const char *)sqlite3_value_text(argv[0]);
    const char *b = (const char *)sqlite3_value_text(argv[1]);
    // IMP: V08886_25725
    if ((sqlite3_value_bytes(argv[0]) != sqlite3_value_bytes(argv[1])) ||
        strncmp(a, b, sqlite3_value_bytes(argv[0])) != 0) {
      vtab_set_error(pVTab,
                     "UPDATEs on vec0 primary key values are not allowed.");
      return SQLITE_ERROR;
    }
    rc = vec0_rowid_from_id(p, argv[0], &rowid);
    if (rc != SQLITE_OK) {
      return rc;
    }
  } else {
    rowid = sqlite3_value_int64(argv[0]);
  }

  // 1) get chunk_id and chunk_offset from _rowids
  rc = vec0_get_chunk_position(p, rowid, NULL, &chunk_id, &chunk_offset);
  if (rc != SQLITE_OK) {
    return rc;
  }

  // 2) update any partition key values
  for (int i = 0; i < vec0_num_defined_user_columns(p); i++) {
    if(p->user_column_kinds[i] != SQLITE_VEC0_USER_COLUMN_KIND_PARTITION) {
      continue;
    }
    sqlite3_value * value = argv[2+VEC0_COLUMN_USERN_START + i];
    if(sqlite3_value_nochange(value)) {
      continue;
    }
    vtab_set_error(pVTab, "UPDATE on partition key columns are not supported yet. ");
    return SQLITE_ERROR;
  }

  // 3) handle auxiliary column updates
  for (int i = 0; i < vec0_num_defined_user_columns(p); i++) {
    if(p->user_column_kinds[i] != SQLITE_VEC0_USER_COLUMN_KIND_AUXILIARY) {
      continue;
    }
    int auxiliary_column_idx = p->user_column_idxs[i];
    sqlite3_value * value = argv[2+VEC0_COLUMN_USERN_START + i];
    if(sqlite3_value_nochange(value)) {
      continue;
    }
    rc = vec0Update_UpdateAuxColumn(p, auxiliary_column_idx, value, rowid);
    if(rc != SQLITE_OK) {
      return SQLITE_ERROR;
    }
  }

  // 4) handle metadata column updates
  for (int i = 0; i < vec0_num_defined_user_columns(p); i++) {
    if(p->user_column_kinds[i] != SQLITE_VEC0_USER_COLUMN_KIND_METADATA) {
      continue;
    }
    int metadata_column_idx = p->user_column_idxs[i];
    sqlite3_value * value = argv[2+VEC0_COLUMN_USERN_START + i];
    if(sqlite3_value_nochange(value)) {
      continue;
    }
    rc = vec0_write_metadata_value(p, metadata_column_idx, rowid, chunk_id, chunk_offset, value, 1);
    if(rc != SQLITE_OK) {
      return rc;
    }
  }

  // 5) iterate over all new vectors, update the vectors
  for (int i = 0; i < vec0_num_defined_user_columns(p); i++) {
    if(p->user_column_kinds[i] != SQLITE_VEC0_USER_COLUMN_KIND_VECTOR) {
      continue;
    }
    int vector_idx = p->user_column_idxs[i];
    sqlite3_value *valueVector = argv[2 + VEC0_COLUMN_USERN_START + i];
    // in vec0Column, we check sqlite3_vtab_nochange() on vector columns.
    // If the vector column isn't being changed, we return NULL;
    // That's not great, that means vector columns can never be NULLABLE
    // (bc we cant distinguish if an updated vector is truly NULL or nochange).
    // Also it means that if someone tries to run `UPDATE v SET X = NULL`,
    // we can't effectively detect and raise an error.
    // A better solution would be to use a custom result_type for "empty",
    // but subtypes don't appear to survive xColumn -> xUpdate, it's always 0.
    // So for now, we'll just use NULL and warn people to not SET X = NULL
    // in the docs.
    if (sqlite3_value_type(valueVector) == SQLITE_NULL) {
      continue;
    }

    // Block vector UPDATE for index types that don't implement it —
    // the DiskANN graph / IVF lists would become stale.
    {
      enum Vec0IndexType idx_type = p->vector_columns[vector_idx].index_type;
      const char *idx_name = NULL;
      if (idx_type == VEC0_INDEX_TYPE_DISKANN) idx_name = "DiskANN";
#if SQLITE_VEC_EXPERIMENTAL_IVF_ENABLE
      else if (idx_type == VEC0_INDEX_TYPE_IVF) idx_name = "IVF";
#endif
      if (idx_name) {
        vtab_set_error(
          &p->base,
          "UPDATE on vector column \"%.*s\" is not supported for %s indexes.",
          p->vector_columns[vector_idx].name_length,
          p->vector_columns[vector_idx].name,
          idx_name);
        return SQLITE_ERROR;
      }
#if SQLITE_VEC_ENABLE_HNSW
      // HNSW re-links the node: delete it from the graph, insert it again.
      if (idx_type == VEC0_INDEX_TYPE_HNSW) {
        struct VectorColumnDefinition *col = &p->vector_columns[vector_idx];
        void *vec;
        size_t dims;
        enum VectorElementType type;
        vector_cleanup cleanup;
        char *err;
        rc = vector_from_value(valueVector, &vec, &dims, &type, &cleanup, &err);
        if (rc != SQLITE_OK) {
          vtab_set_error(&p->base, "Updated vector for the \"%.*s\" column is invalid: %z",
                         col->name_length, col->name, err);
          return SQLITE_ERROR;
        }
        if (type != col->element_type || dims != col->dimensions) {
          cleanup(vec);
          vtab_set_error(&p->base,
                         "Updated vector for the \"%.*s\" column must be %s[%lld]",
                         col->name_length, col->name, vector_subtype_name(col->element_type),
                         (i64)col->dimensions);
          return SQLITE_ERROR;
        }
        rc = hnsw_update(p, vector_idx, rowid, vec);
        if (rc == SQLITE_OK && p->storage)
          rc = vec0_storage_error(p, p->storage->xSetVector(p->storageHandle, rowid, vector_idx, vec));
        cleanup(vec);
        if (rc != SQLITE_OK) return rc;
        continue;
      }
#endif
    }

    rc = vec0Update_UpdateVectorColumn(p, chunk_id, chunk_offset, vector_idx,
                                       valueVector, rowid);
    if (rc != SQLITE_OK) {
      return SQLITE_ERROR;
    }
  }

  return SQLITE_OK;
}

static int vec0Update(sqlite3_vtab *pVTab, int argc, sqlite3_value **argv,
                      sqlite_int64 *pRowid) {
  // DELETE operation
  if (argc == 1 && sqlite3_value_type(argv[0]) != SQLITE_NULL) {
    return vec0Update_Delete(pVTab, argv[0]);
  }
  // INSERT operation
  else if (argc > 1 && sqlite3_value_type(argv[0]) == SQLITE_NULL) {
    vec0_vtab *p = (vec0_vtab *)pVTab;
    // FTS5-style command dispatch via hidden column named after table
    if (p->hasCommandColumn) {
      sqlite3_value *cmdVal = argv[2 + vec0_column_command_idx(p)];
      if (sqlite3_value_type(cmdVal) == SQLITE_TEXT) {
        int frc = vec0_flat_flush(p);
        if (frc != SQLITE_OK) return frc;
        const char *cmd = (const char *)sqlite3_value_text(cmdVal);
        int cmdRc = SQLITE_EMPTY;
#if SQLITE_VEC_ENABLE_RESCORE
        cmdRc = rescore_handle_command(p, cmd);
#endif
#if SQLITE_VEC_EXPERIMENTAL_IVF_ENABLE
        if (cmdRc == SQLITE_EMPTY)
          cmdRc = ivf_handle_command(p, cmd, argc, argv);
#endif
#if SQLITE_VEC_ENABLE_DISKANN
        if (cmdRc == SQLITE_EMPTY)
          cmdRc = diskann_handle_command(p, cmd);
#endif
#if SQLITE_VEC_ENABLE_HNSW
        if (cmdRc == SQLITE_EMPTY)
          cmdRc = hnsw_handle_command(p, cmd);
#endif
        if (cmdRc == SQLITE_EMPTY && p->storage && p->storage->iVersion >= 2 && p->storage->xCommand) {
          cmdRc = p->storage->xCommand(p->storageHandle, cmd);
          if (cmdRc != SQLITE_OK && cmdRc != SQLITE_EMPTY) return vec0_storage_error(p, cmdRc);
        }
        if (cmdRc == SQLITE_EMPTY) {
          vtab_set_error(pVTab, "unknown vec0 command: '%s'", cmd);
          return SQLITE_ERROR;
        }
        return cmdRc;
      }
    }
    return vec0Update_Insert(pVTab, argc, argv, pRowid);
  }
  // UPDATE operation
  else if (argc > 1 && sqlite3_value_type(argv[0]) != SQLITE_NULL) {
    return vec0Update_Update(pVTab, argc, argv);
  } else {
    vtab_set_error(pVTab, "Unrecognized xUpdate operation provided for vec0.");
    return SQLITE_ERROR;
  }
}

static int vec0ShadowName(const char *zName) {
  static const char *azName[] = {
    "rowids", "chunks", "auxiliary", "info",

  // Up to VEC0_MAX_METADATA_COLUMNS
  // TODO be smarter about this man
  "metadatachunks00",
  "metadatachunks01",
  "metadatachunks02",
  "metadatachunks03",
  "metadatachunks04",
  "metadatachunks05",
  "metadatachunks06",
  "metadatachunks07",
  "metadatachunks08",
  "metadatachunks09",
  "metadatachunks10",
  "metadatachunks11",
  "metadatachunks12",
  "metadatachunks13",
  "metadatachunks14",
  "metadatachunks15",

  // Up to
  "metadatatext00",
  "metadatatext01",
  "metadatatext02",
  "metadatatext03",
  "metadatatext04",
  "metadatatext05",
  "metadatatext06",
  "metadatatext07",
  "metadatatext08",
  "metadatatext09",
  "metadatatext10",
  "metadatatext11",
  "metadatatext12",
  "metadatatext13",
  "metadatatext14",
  "metadatatext15",
  };

  for (size_t i = 0; i < sizeof(azName) / sizeof(azName[0]); i++) {
    if (sqlite3_stricmp(zName, azName[i]) == 0)
      return 1;
  }
  //for(size_t i = 0; i < )"vector_chunks", "metadatachunks"
  return 0;
}

static int vec0Begin(sqlite3_vtab *pVTab) {
  UNUSED_PARAMETER(pVTab);
  return SQLITE_OK;
}
static int vec0Sync(sqlite3_vtab *pVTab) {
  vec0_vtab *p = (vec0_vtab *)pVTab;
  int frc = vec0_flat_flush(p);
  if (frc != SQLITE_OK) return frc;
#if SQLITE_VEC_EXPERIMENTAL_IVF_ENABLE
  int irc = ivf_flush(p);
  if (irc != SQLITE_OK) return irc;
#endif
#if SQLITE_VEC_ENABLE_HNSW
  int rc = hnsw_flush_all(p);
  if (rc != SQLITE_OK) return rc;
#endif
  vec0_free_resources(p);
  return p->storage ? vec0_storage_error(p, p->storage->xSync(p->storageHandle)) : SQLITE_OK;
}
static int vec0Commit(sqlite3_vtab *pVTab) {
  vec0_vtab *p = (vec0_vtab *)pVTab;
#if SQLITE_VEC_ENABLE_HNSW
  hnsw_cache_after_commit(p);
#endif
  return p->storage ? vec0_storage_error(p, p->storage->xCommit(p->storageHandle)) : SQLITE_OK;
}
// Rollbacks undo shadow-table writes that in-memory index caches mirror.
static int vec0Rollback(sqlite3_vtab *pVTab) {
  vec0_vtab *p = (vec0_vtab *)pVTab;
  vec0_flat_clear(p);
#if SQLITE_VEC_EXPERIMENTAL_IVF_ENABLE
  ivf_buffer_clear(p);
#endif
#if SQLITE_VEC_ENABLE_HNSW
  hnsw_cache_clear_all(p);
#endif
  return p->storage ? vec0_storage_error(p, p->storage->xRollback(p->storageHandle)) : SQLITE_OK;
}
// Persist older dirty lists before SQLite opens the new rollback boundary.
static int vec0Savepoint(sqlite3_vtab *pVTab, int iSavepoint) {
  vec0_vtab *p = (vec0_vtab *)pVTab;
  int frc = vec0_flat_flush(p);
  if (frc != SQLITE_OK) return frc;
#if SQLITE_VEC_EXPERIMENTAL_IVF_ENABLE
  int irc = ivf_flush(p);
  if (irc != SQLITE_OK) return irc;
#endif
#if SQLITE_VEC_ENABLE_HNSW
  int rc = hnsw_flush_all(p);
  if (rc != SQLITE_OK) return rc;
#endif
  return p->storage
             ? vec0_storage_error(p, p->storage->xSavepoint(p->storageHandle, iSavepoint))
             : SQLITE_OK;
}
static int vec0Release(sqlite3_vtab *pVTab, int iSavepoint) {
  vec0_vtab *p = (vec0_vtab *)pVTab;
  return p->storage
             ? vec0_storage_error(p, p->storage->xRelease(p->storageHandle, iSavepoint))
             : SQLITE_OK;
}
// Called when a statement fails inside a transaction, and on ROLLBACK TO.
static int vec0RollbackTo(sqlite3_vtab *pVTab, int iSavepoint) {
  vec0_vtab *p = (vec0_vtab *)pVTab;
  if (!p->flatFlushing) vec0_flat_clear(p);
#if SQLITE_VEC_EXPERIMENTAL_IVF_ENABLE
  // A nested shadow UPDATE rollback does not undo the pending logical writes.
  if (!p->ivfFlushing) ivf_buffer_clear(p);
#endif
#if SQLITE_VEC_ENABLE_HNSW
  hnsw_cache_rollback_to(p);
#endif
  return p->storage
             ? vec0_storage_error(p, p->storage->xRollbackTo(p->storageHandle, iSavepoint))
             : SQLITE_OK;
}

/**
 * xRename implementation for vec0.
 * Renames all shadow tables to match the new virtual table name,
 * then updates cached table names and finalizes stale prepared statements.
 */
static int vec0Rename(sqlite3_vtab *pVtab, const char *zNew) {
  vec0_vtab *p = (vec0_vtab *)pVtab;
  int rc = vec0_flat_flush(p);
  if (rc != SQLITE_OK) return rc;
#if SQLITE_VEC_EXPERIMENTAL_IVF_ENABLE
  rc = ivf_flush(p);
  if (rc != SQLITE_OK) return rc;
#endif

  // Build a single SQL string with ALTER TABLE RENAME for every shadow table.
  sqlite3_str *s = sqlite3_str_new(p->db);

  // Core shadow tables (always present)
  sqlite3_str_appendf(s,
    "ALTER TABLE \"%w\".\"%w_info\" RENAME TO \"%w_info\";",
    p->schemaName, p->tableName, zNew);
  sqlite3_str_appendf(s,
    "ALTER TABLE \"%w\".\"%w_rowids\" RENAME TO \"%w_rowids\";",
    p->schemaName, p->tableName, zNew);
  sqlite3_str_appendf(s,
    "ALTER TABLE \"%w\".\"%w_chunks\" RENAME TO \"%w_chunks\";",
    p->schemaName, p->tableName, zNew);

  // Auxiliary columns: a storage provider renames its own data; otherwise
  // the _auxiliary shadow table (only if auxiliary columns exist).
  if (p->storage) {
    rc = vec0_storage_error(p, p->storage->xRename(p->storageHandle, zNew));
    if (rc != SQLITE_OK) {
      sqlite3_free(sqlite3_str_finish(s));
      return rc;
    }
  } else if (p->numAuxiliaryColumns > 0) {
    sqlite3_str_appendf(s,
      "ALTER TABLE \"%w\".\"%w_auxiliary\" RENAME TO \"%w_auxiliary\";",
      p->schemaName, p->tableName, zNew);
  }

  // Per-vector-column shadow tables
  for (int i = 0; i < p->numVectorColumns; i++) {
    // Non-FLAT columns (rescore, IVF, DiskANN) don't create _vector_chunks
    // (mirror the guard in vec0_init around VEC0_SHADOW_VECTOR_N_CREATE).
    if (vec0_column_in_chunks(p, i)) {
      sqlite3_str_appendf(s,
        "ALTER TABLE \"%w\".\"%w_vector_chunks%02d\" RENAME TO \"%w_vector_chunks%02d\";",
        p->schemaName, p->tableName, i, zNew, i);
    }

#if SQLITE_VEC_ENABLE_RESCORE
    if (p->shadowRescoreChunksNames[i]) {
      sqlite3_str_appendf(s,
        "ALTER TABLE \"%w\".\"%w_rescore_chunks%02d\" RENAME TO \"%w_rescore_chunks%02d\";",
        p->schemaName, p->tableName, i, zNew, i);
      sqlite3_str_appendf(s,
        "ALTER TABLE \"%w\".\"%w_rescore_vectors%02d\" RENAME TO \"%w_rescore_vectors%02d\";",
        p->schemaName, p->tableName, i, zNew, i);
    }
#endif

#if SQLITE_VEC_ENABLE_DISKANN
    if (p->shadowVectorsNames[i]) {
      sqlite3_str_appendf(s,
        "ALTER TABLE \"%w\".\"%w_vectors%02d\" RENAME TO \"%w_vectors%02d\";",
        p->schemaName, p->tableName, i, zNew, i);
      sqlite3_str_appendf(s,
        "ALTER TABLE \"%w\".\"%w_diskann_nodes%02d\" RENAME TO \"%w_diskann_nodes%02d\";",
        p->schemaName, p->tableName, i, zNew, i);
      sqlite3_str_appendf(s,
        "ALTER TABLE \"%w\".\"%w_diskann_buffer%02d\" RENAME TO \"%w_diskann_buffer%02d\";",
        p->schemaName, p->tableName, i, zNew, i);
    }
#endif
  }

#if SQLITE_VEC_EXPERIMENTAL_IVF_ENABLE
  for (int i = 0; i < p->numVectorColumns; i++) {
    if (p->shadowIvfCellsNames[i]) {
      sqlite3_str_appendf(s,
        "ALTER TABLE \"%w\".\"%w_ivf_centroids%02d\" RENAME TO \"%w_ivf_centroids%02d\";",
        p->schemaName, p->tableName, i, zNew, i);
      sqlite3_str_appendf(s,
        "ALTER TABLE \"%w\".\"%w_ivf_cells%02d\" RENAME TO \"%w_ivf_cells%02d\";",
        p->schemaName, p->tableName, i, zNew, i);
      sqlite3_str_appendf(s,
        "ALTER TABLE \"%w\".\"%w_ivf_rowid_map%02d\" RENAME TO \"%w_ivf_rowid_map%02d\";",
        p->schemaName, p->tableName, i, zNew, i);
      // _ivf_vectors is only created when quantizer != none
      // (mirror ivf_create_shadow_tables in sqlite-vec-ivf.c).
      if (p->vector_columns[i].ivf.quantizer != VEC0_IVF_QUANTIZER_NONE) {
        sqlite3_str_appendf(s,
          "ALTER TABLE \"%w\".\"%w_ivf_vectors%02d\" RENAME TO \"%w_ivf_vectors%02d\";",
          p->schemaName, p->tableName, i, zNew, i);
      }
    }
  }
#endif

#if SQLITE_VEC_ENABLE_HNSW
  for (int i = 0; i < p->numVectorColumns; i++) {
    if (p->vector_columns[i].index_type != VEC0_INDEX_TYPE_HNSW) continue;
    sqlite3_str_appendf(s,
      "ALTER TABLE \"%w\".\"%w_hnsw_nodes%02d\" RENAME TO \"%w_hnsw_nodes%02d\";",
      p->schemaName, p->tableName, i, zNew, i);
    sqlite3_str_appendf(s,
      "ALTER TABLE \"%w\".\"%w_hnsw_edges%02d\" RENAME TO \"%w_hnsw_edges%02d\";",
      p->schemaName, p->tableName, i, zNew, i);
  }
#endif

  // Per-metadata-column shadow tables
  for (int i = 0; i < p->numMetadataColumns; i++) {
    sqlite3_str_appendf(s,
      "ALTER TABLE \"%w\".\"%w_metadatachunks%02d\" RENAME TO \"%w_metadatachunks%02d\";",
      p->schemaName, p->tableName, i, zNew, i);
    if (p->metadata_columns[i].kind == VEC0_METADATA_COLUMN_KIND_TEXT) {
      sqlite3_str_appendf(s,
        "ALTER TABLE \"%w\".\"%w_metadatatext%02d\" RENAME TO \"%w_metadatatext%02d\";",
        p->schemaName, p->tableName, i, zNew, i);
    }
  }

  char *zSql = sqlite3_str_finish(s);
  if (!zSql) {
    return SQLITE_NOMEM;
  }

  rc = sqlite3_exec(p->db, zSql, 0, 0, 0);
  sqlite3_free(zSql);
  if (rc != SQLITE_OK) {
    return rc;
  }

  // Finalize all prepared statements — they reference old table names.
  vec0_free_resources(p);

  // Update cached table name
  sqlite3_free(p->tableName);
  p->tableName = sqlite3_mprintf("%s", zNew);
  if (!p->tableName) return SQLITE_NOMEM;

  // Update cached shadow table names
  sqlite3_free(p->shadowRowidsName);
  p->shadowRowidsName = sqlite3_mprintf("%s_rowids", zNew);

  sqlite3_free(p->shadowChunksName);
  p->shadowChunksName = sqlite3_mprintf("%s_chunks", zNew);

  for (int i = 0; i < p->numVectorColumns; i++) {
    sqlite3_free(p->shadowVectorChunksNames[i]);
    p->shadowVectorChunksNames[i] =
        sqlite3_mprintf("%s_vector_chunks%02d", zNew, i);

#if SQLITE_VEC_ENABLE_RESCORE
    if (p->shadowRescoreChunksNames[i]) {
      sqlite3_free(p->shadowRescoreChunksNames[i]);
      p->shadowRescoreChunksNames[i] =
          sqlite3_mprintf("%s_rescore_chunks%02d", zNew, i);
      sqlite3_free(p->shadowRescoreVectorsNames[i]);
      p->shadowRescoreVectorsNames[i] =
          sqlite3_mprintf("%s_rescore_vectors%02d", zNew, i);
    }
#endif

#if SQLITE_VEC_ENABLE_DISKANN
    if (p->shadowVectorsNames[i]) {
      sqlite3_free(p->shadowVectorsNames[i]);
      p->shadowVectorsNames[i] =
          sqlite3_mprintf("%s_vectors%02d", zNew, i);
      sqlite3_free(p->shadowDiskannNodesNames[i]);
      p->shadowDiskannNodesNames[i] =
          sqlite3_mprintf("%s_diskann_nodes%02d", zNew, i);
    }
#endif
  }

#if SQLITE_VEC_EXPERIMENTAL_IVF_ENABLE
  for (int i = 0; i < p->numVectorColumns; i++) {
    if (p->shadowIvfCellsNames[i]) {
      sqlite3_free(p->shadowIvfCellsNames[i]);
      p->shadowIvfCellsNames[i] =
          sqlite3_mprintf("%s_ivf_cells%02d", zNew, i);
    }
  }
#endif

  for (int i = 0; i < p->numMetadataColumns; i++) {
    sqlite3_free(p->shadowMetadataChunksNames[i]);
    p->shadowMetadataChunksNames[i] =
        sqlite3_mprintf("%s_metadatachunks%02d", zNew, i);
  }

  return SQLITE_OK;
}

static sqlite3_module vec0Module = {
    /* iVersion      */ 3,
    /* xCreate       */ vec0Create,
    /* xConnect      */ vec0Connect,
    /* xBestIndex    */ vec0BestIndex,
    /* xDisconnect   */ vec0Disconnect,
    /* xDestroy      */ vec0Destroy,
    /* xOpen         */ vec0Open,
    /* xClose        */ vec0Close,
    /* xFilter       */ vec0Filter,
    /* xNext         */ vec0Next,
    /* xEof          */ vec0Eof,
    /* xColumn       */ vec0Column,
    /* xRowid        */ vec0Rowid,
    /* xUpdate       */ vec0Update,
    /* xBegin        */ vec0Begin,
    /* xSync         */ vec0Sync,
    /* xCommit       */ vec0Commit,
    /* xRollback     */ vec0Rollback,
    /* xFindFunction */ 0,
    /* xRename       */ vec0Rename,
    /* xSavepoint    */ vec0Savepoint,
    /* xRelease      */ vec0Release,
    /* xRollbackTo   */ vec0RollbackTo,
    /* xShadowName   */ vec0ShadowName,
#if SQLITE_VERSION_NUMBER >= 3044000
    /* xIntegrity    */ 0, // https://github.com/asg017/sqlite-vec/issues/44
#endif
};
#pragma endregion


#ifdef SQLITE_VEC_ENABLE_AVX
#define SQLITE_VEC_DEBUG_BUILD_AVX "avx"
#else
#define SQLITE_VEC_DEBUG_BUILD_AVX ""
#endif
#ifdef SQLITE_VEC_ENABLE_NEON
#define SQLITE_VEC_DEBUG_BUILD_NEON "neon"
#else
#define SQLITE_VEC_DEBUG_BUILD_NEON ""
#endif
#if SQLITE_VEC_ENABLE_RESCORE
#define SQLITE_VEC_DEBUG_BUILD_RESCORE "rescore"
#else
#define SQLITE_VEC_DEBUG_BUILD_RESCORE ""
#endif

#if SQLITE_VEC_EXPERIMENTAL_IVF_ENABLE
#define SQLITE_VEC_DEBUG_BUILD_IVF "ivf"
#else
#define SQLITE_VEC_DEBUG_BUILD_IVF ""
#endif

#if SQLITE_VEC_ENABLE_DISKANN
#define SQLITE_VEC_DEBUG_BUILD_DISKANN "diskann"
#else
#define SQLITE_VEC_DEBUG_BUILD_DISKANN ""
#endif

#define SQLITE_VEC_DEBUG_BUILD                                                 \
  SQLITE_VEC_DEBUG_BUILD_AVX " " SQLITE_VEC_DEBUG_BUILD_NEON " "              \
  SQLITE_VEC_DEBUG_BUILD_RESCORE " " SQLITE_VEC_DEBUG_BUILD_IVF " "           \
  SQLITE_VEC_DEBUG_BUILD_DISKANN

#define SQLITE_VEC_DEBUG_STRING                                                \
  "Version: " SQLITE_VEC_VERSION "\n"                                          \
  "Date: " SQLITE_VEC_DATE "\n"                                                \
  "Commit: " SQLITE_VEC_SOURCE "\n"                                            \
  "Build flags: " SQLITE_VEC_DEBUG_BUILD

SQLITE_VEC_API int sqlite3_vec_init(sqlite3 *db, char **pzErrMsg,
                                    const sqlite3_api_routines *pApi) {
#ifndef SQLITE_CORE
  SQLITE_EXTENSION_INIT2(pApi);
#else
  UNUSED_PARAMETER(pApi);
#endif
  int rc = SQLITE_OK;

#define DEFAULT_FLAGS (SQLITE_UTF8 | SQLITE_INNOCUOUS | SQLITE_DETERMINISTIC)

  rc = sqlite3_create_function_v2(db, "vec_version", 0, DEFAULT_FLAGS,
                                  SQLITE_VEC_VERSION, _static_text_func, NULL,
                                  NULL, NULL);
  if (rc != SQLITE_OK) {
    return rc;
  }
  rc = sqlite3_create_function_v2(db, "vec_debug", 0, DEFAULT_FLAGS,
                                  SQLITE_VEC_DEBUG_STRING, _static_text_func,
                                  NULL, NULL, NULL);
  if (rc != SQLITE_OK) {
    return rc;
  }
  static struct {
    const char *zFName;
    void (*xFunc)(sqlite3_context *, int, sqlite3_value **);
    int nArg;
    int flags;
  } aFunc[] = {
      // clang-format off
    //{"vec_version",         _static_text_func,    0, DEFAULT_FLAGS,                                          (void *) SQLITE_VEC_VERSION },
    //{"vec_debug",           _static_text_func,    0, DEFAULT_FLAGS,                                          (void *) SQLITE_VEC_DEBUG_STRING },
    {"vec_distance_l2",     vec_distance_l2,      2, DEFAULT_FLAGS | SQLITE_SUBTYPE,                         },
    {"vec_distance_l1",     vec_distance_l1,      2, DEFAULT_FLAGS | SQLITE_SUBTYPE,                         },
    {"vec_distance_hamming",vec_distance_hamming, 2, DEFAULT_FLAGS | SQLITE_SUBTYPE,                         },
    {"vec_distance_cosine", vec_distance_cosine,  2, DEFAULT_FLAGS | SQLITE_SUBTYPE,                         },
    {"vec_distance_dot",    vec_distance_dot,     2, DEFAULT_FLAGS | SQLITE_SUBTYPE,                         },
    {"vec_distance_jaccard",vec_distance_jaccard, 2, DEFAULT_FLAGS | SQLITE_SUBTYPE,                         },
    {"vec_concat",          vec_concat,           2, DEFAULT_FLAGS | SQLITE_SUBTYPE | SQLITE_RESULT_SUBTYPE, },
    {"vec_sparse",          vec_sparse,           1, DEFAULT_FLAGS,                                          },
    {"vec_sparse_distance_dot",    vec_sparse_distance_dot,    2, DEFAULT_FLAGS,                         },
    {"vec_sparse_distance_cosine", vec_sparse_distance_cosine, 2, DEFAULT_FLAGS,                         },
    {"vec_sparse_to_json",  vec_sparse_to_json,   1, DEFAULT_FLAGS,                                          },
    {"vec_rrf",             vec_rrf,             -1, DEFAULT_FLAGS,                                          },
    {"vec_length",          vec_length,           1, DEFAULT_FLAGS | SQLITE_SUBTYPE,                         },
    {"vec_type",           vec_type,           1, DEFAULT_FLAGS,                         },
    {"vec_to_json",         vec_to_json,          1, DEFAULT_FLAGS | SQLITE_SUBTYPE | SQLITE_RESULT_SUBTYPE, },
    {"vec_add",             vec_add,              2, DEFAULT_FLAGS | SQLITE_SUBTYPE | SQLITE_RESULT_SUBTYPE, },
    {"vec_sub",             vec_sub,              2, DEFAULT_FLAGS | SQLITE_SUBTYPE | SQLITE_RESULT_SUBTYPE, },
    {"vec_slice",           vec_slice,            3, DEFAULT_FLAGS | SQLITE_SUBTYPE | SQLITE_RESULT_SUBTYPE, },
    {"vec_normalize",       vec_normalize,        1, DEFAULT_FLAGS | SQLITE_SUBTYPE | SQLITE_RESULT_SUBTYPE, },
    {"vec_f32",             vec_f32,              1, DEFAULT_FLAGS | SQLITE_SUBTYPE | SQLITE_RESULT_SUBTYPE, },
    {"vec_bit",             vec_bit,              1, DEFAULT_FLAGS | SQLITE_SUBTYPE | SQLITE_RESULT_SUBTYPE, },
    {"vec_int8",            vec_int8,             1, DEFAULT_FLAGS | SQLITE_SUBTYPE | SQLITE_RESULT_SUBTYPE, },
    {"vec_quantize_int8",     vec_quantize_int8,      2, DEFAULT_FLAGS | SQLITE_SUBTYPE | SQLITE_RESULT_SUBTYPE, },
    {"vec_quantize_binary", vec_quantize_binary,  1, DEFAULT_FLAGS | SQLITE_SUBTYPE | SQLITE_RESULT_SUBTYPE, },
      // clang-format on
  };

  static struct {
    char *name;
    const sqlite3_module *module;
    void *p;
    void (*xDestroy)(void *);
  } aMod[] = {
      // clang-format off
    {"vec0",          &vec0Module,          NULL, NULL},
    {"vec_each",      &vec_eachModule,      NULL, NULL},
    {"vec_file_each", &vec_file_eachModule, NULL, NULL},
    {"vec_sparse_each", &vec_sparse_eachModule, NULL, NULL},
      // clang-format on
  };

  for (unsigned long i = 0; i < countof(aFunc) && rc == SQLITE_OK; i++) {
    rc = sqlite3_create_function_v2(db, aFunc[i].zFName, aFunc[i].nArg,
                                    aFunc[i].flags, NULL, aFunc[i].xFunc, NULL,
                                    NULL, NULL);
    if (rc != SQLITE_OK) {
      *pzErrMsg = sqlite3_mprintf("Error creating function %s: %s",
                                  aFunc[i].zFName, sqlite3_errmsg(db));
      return rc;
    }
  }

  for (unsigned long i = 0; i < countof(aMod) && rc == SQLITE_OK; i++) {
    rc = sqlite3_create_module_v2(db, aMod[i].name, aMod[i].module, NULL, NULL);
    if (rc != SQLITE_OK) {
      *pzErrMsg = sqlite3_mprintf("Error creating module %s: %s", aMod[i].name,
                                  sqlite3_errmsg(db));
      return rc;
    }
  }

  return graph_register(db, pzErrMsg);
}


