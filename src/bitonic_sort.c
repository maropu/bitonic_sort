/*-------------------------------------------------------------------------
 *
 * bitonic_sort.c - 20110612 v0.1.3
 *	  bitonic sort routines
 *
 *-------------------------------------------------------------------------
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <emmintrin.h>
#include <pthread.h>

#include "err_utils.h"

#define ID_TEST_SSE     0x2000000

#define _log2_uint32(_arg1)             \
        ({                              \
                uint32_t        d;      \
                __asm__("bsr %1, %0;" :"=r"(d) :"r"(_arg1));    \
                d;                      \
        })

#define _replace_float_adr(a, b)        \
        ({                              \
                a = (float *)(((intptr_t)a) ^ ((intptr_t)b));   \
                b = (float *)(((intptr_t)b) ^ ((intptr_t)a));   \
                a = (float *)(((intptr_t)a) ^ ((intptr_t)b));   \
        })

#define _zalloc(size)                           \
        ({                                      \
                void *ptr = malloc(size);       \
                if (ptr)                        \
                        memset(ptr, 0x00, size);        \
                ptr;                            \
         })

#define _simd_align(x)          ((((intptr_t)x) + ((0x01 << 4) - 1)) & ~((0x01 << 4) - 1))
#define _ptr_offset(x, y)       (((intptr_t)(x)) - ((intptr_t)(y)))
#define _likely(x)              __builtin_expect(!!(x), 1)

struct _chunk_info {
        float           *d;
        float           *buf;
        uint32_t        chk;
        uint32_t        offset;
};

/* SIMD operations */
static int32_t _simd_supported(int32_t id);
static int32_t _cpuid_supported(void);
inline static void _fast_gap_memcpy(uint8_t *dst, uint8_t *src, uint32_t gap, size_t sz) __attribute__((always_inline));
inline static void _fast_memcpy(void *d, void *s, size_t sz) __attribute__((always_inline));

/* Support functions for bitonic sort */
static void *_bitonic_sort(void *arg);
static void _bitonic_merge(float *d, uint32_t s, float *buf, uint32_t chunk_size);

inline static void _bitonic_sort32(float *d) __attribute__((always_inline));
inline static void _bitonic_merge_kernel16n(float *dest, float *a, float *b, uint32_t s) __attribute__((always_inline));
inline static void _bitonic_sort_kernel4(float *a, float *b) __attribute__((always_inline));
inline static void _bitonic_merge_kernel4(float *a, float *b) __attribute__((always_inline));
inline static void _bitonic_merge_kernel4core(float *a, float *b) __attribute__((always_inline));
inline static void _bitonic_merge_kernel8(float *a, float *b) __attribute__((always_inline));
inline static void _bitonic_merge_kernel8core(float *a, float *b) __attribute__((always_inline));

/* Global variables for the shift of processing */
uint32_t        _enable_fast_memcpy = 0;
uint32_t        _enable_bitonic_sort = 1;

/* Support other functions */
static int32_t _compare_float(const void *a, const void *b);

void
bitonic_sort(float *d, uint32_t s, float *buf, uint32_t chunk_num /* Must be the number of hardware threads */)
{
        int32_t         i;
        uint32_t        chk;
        pthread_t       *tid;
        struct _chunk_info      *chunk_info;

        if (!_enable_bitonic_sort) {
                qsort((void *)d, s, sizeof(float), _compare_float);
        } else {
                doutput(FILE_OUTPUT, "starting ...");

                /* Check supports */
                if (!_cpuid_supported())
                        eoutput("No CPUID instruction");

                if (!_simd_supported(ID_TEST_SSE))
                        eoutput("No SIMD instruction");

                /* Check validation for input parameters */
                if ((_simd_align((intptr_t)d) != ((intptr_t)d)) || (_simd_align((intptr_t)buf) != ((intptr_t)buf)))
                        eoutput("Alignment error of inputer parameters");

                if ((chunk_num >> _log2_uint32(chunk_num)) != 1)
                        eoutput("chunk_num is not 2^x");

                /* Initialization */
                tid = malloc(sizeof(pthread_t) * chunk_num);
                chunk_info = _zalloc(sizeof(struct _chunk_info) * chunk_num);

                if (tid == NULL || chunk_info == NULL)
                        eoutput("Can't allocate memories");

                chk = s / chunk_num;

                for (i = 0; i < chunk_num; i++) {
                        chunk_info[i].d = d;
                        chunk_info[i].buf = buf;
                        chunk_info[i].chk = chk;
                        chunk_info[i].offset = i * chk;

                        doutput(FILE_OUTPUT, "[%u]: offset->%u chk->%u", i, chunk_info[i].offset, chunk_info[i].chk);
                }

                /* Sort individual chunks */
                for (i = 0; i < chunk_num; i++)
                        pthread_create(&tid[i], NULL, _bitonic_sort, &chunk_info[i]);

                /* Synchronized here */
                for (i = 0; i < chunk_num; i++)
                        pthread_join(tid[i], NULL);

                /* Merge left chunks */
                _bitonic_merge(d, s, buf, chk);

                free(chunk_info);
                free(tid);
        }
}

int32_t 
err_check(float *d, uint32_t s)
{
        int32_t i;

        doutput(FILE_OUTPUT, "starting ...");

        for (i = 0; i < s - 1; i++) {
                if (d[i] > d[i + 1])
                        return -1;
        }

        return 0;
}

/* --- Intra Functions Below ---*/

int32_t 
_simd_supported(int32_t id)
{
        int32_t i;

        i = 1;

#ifndef __x86_64__
        __asm__ __volatile__(
                "movl %1, %%eax\n\t"
                "cpuid\n\t"
                "testl %2, %%edx\n\t"
                "jnz yes_\n\t"
                "movl $0, %0\n\t"
                "jz no_\n\t"
        "yes_:\n\t"
                "movl $1, %0\n\t"
        "no_:\n\t"
        : "=m"(i) : "m"(i), "m"(id)
        : "%eax", "%ebx", "%ecx", "%edx");
#endif

        return i;
}

int32_t 
_cpuid_supported(void)
{
        int32_t v;

#ifndef __x86_64__
        __asm__ __volatile__(
                "pushf\n\t"
                "popl %%eax\n\t"
                "movl %%eax, %%ebx\n\t"
                "xor $0x00200000, %%eax\n\t"
                "pushl %%eax\n\t"
                "popf\n\t"
                "pushf\n\t"
                "popl %%eax\n\t"
                "cmpl %%ebx, %%eax\n\t"
                "jz no_cpuid_\n\t"
                "movl $1, %0\n\t"
                "jmp out_\n\t"
        "no_cpuid_:\n\t"
                "movl $0, %0\n\t"
        "out_:\n\t"
        : "=r"(v) : : "%eax", "%ebx");
#else
        v = 1;
#endif

        return v;
}

#define __inregister_sort(b, reg)       \
        do {                            \
                reg[0] = _mm_load_ps(b);                \
                reg[1] = _mm_load_ps(b+4);              \
                reg[2] = _mm_load_ps(b+8);              \
                reg[3] = _mm_load_ps(b+12);             \
\
                reg[4] = _mm_min_ps(reg[0], reg[1]);    \
                reg[5] = _mm_max_ps(reg[0], reg[1]);    \
\
                reg[6] = _mm_min_ps(reg[2], reg[3]);    \
                reg[7] = _mm_max_ps(reg[2], reg[3]);    \
\
                reg[0] = _mm_min_ps(reg[4], reg[6]);    \
                reg[1] = _mm_max_ps(reg[4], reg[6]);    \
\
                reg[2] = _mm_min_ps(reg[5], reg[7]);    \
                reg[3] = _mm_max_ps(reg[5], reg[7]);    \
\
                reg[5] = _mm_min_ps(reg[1], reg[2]);    \
                reg[6] = _mm_max_ps(reg[1], reg[2]);    \
\
                reg[2] = _mm_shuffle_ps(reg[0], reg[5], 0x44);  \
                reg[7] = _mm_shuffle_ps(reg[0], reg[5], 0xee);  \
\
                reg[4] = _mm_shuffle_ps(reg[6], reg[3], 0xee);  \
                reg[1] = _mm_shuffle_ps(reg[6], reg[3], 0x44);  \
\
                reg[0] = _mm_shuffle_ps(reg[2], reg[1], 0xdd);  \
                reg[5] = _mm_shuffle_ps(reg[2], reg[1], 0x88);  \
                reg[6] = _mm_shuffle_ps(reg[7], reg[4], 0xdd);  \
                reg[3] = _mm_shuffle_ps(reg[7], reg[4], 0x88);  \
\
                reg[1] = reg[5];        \
                reg[2] = reg[6];        \
        } while (0);

#define __bitonic_merge_network4_sim4(a1, b1, a2, b2, a3, b3, a4, b4)   \
        do {                                    \
                /* L1 processing */             \
                lo[0] = _mm_min_ps(a1, b1);     \
                hi[0] = _mm_max_ps(a1, b1);     \
                lo[1] = _mm_min_ps(a2, b2);     \
                hi[1] = _mm_max_ps(a2, b2);     \
                lo[2] = _mm_min_ps(a3, b3);     \
                hi[2] = _mm_max_ps(a3, b3);     \
                lo[3] = _mm_min_ps(a4, b4);     \
                hi[3] = _mm_max_ps(a4, b4);     \
\
                a1 = _mm_shuffle_ps(lo[0], hi[0], 0xe4);        \
                b1 = _mm_shuffle_ps(lo[0], hi[0], 0x4e);        \
                a2 = _mm_shuffle_ps(lo[1], hi[1], 0xe4);        \
                b2 = _mm_shuffle_ps(lo[1], hi[1], 0x4e);        \
                a3 = _mm_shuffle_ps(lo[2], hi[2], 0xe4);        \
                b3 = _mm_shuffle_ps(lo[2], hi[2], 0x4e);        \
                a4 = _mm_shuffle_ps(lo[3], hi[3], 0xe4);        \
                b4 = _mm_shuffle_ps(lo[3], hi[3], 0x4e);        \
\
                /* L2 processing */             \
                lo[0] = _mm_min_ps(a1, b1);     \
                hi[0] = _mm_max_ps(a1, b1);     \
                lo[1] = _mm_min_ps(a2, b2);     \
                hi[1] = _mm_max_ps(a2, b2);     \
                lo[2] = _mm_min_ps(a3, b3);     \
                hi[2] = _mm_max_ps(a3, b3);     \
                lo[3] = _mm_min_ps(a4, b4);     \
                hi[3] = _mm_max_ps(a4, b4);     \
\
                a1 = _mm_shuffle_ps(lo[0], hi[0], 0xd8);        \
                b1 = _mm_shuffle_ps(lo[0], hi[0], 0x8d);        \
                a2 = _mm_shuffle_ps(lo[1], hi[1], 0xd8);        \
                b2 = _mm_shuffle_ps(lo[1], hi[1], 0x8d);        \
                a3 = _mm_shuffle_ps(lo[2], hi[2], 0xd8);        \
                b3 = _mm_shuffle_ps(lo[2], hi[2], 0x8d);        \
                a4 = _mm_shuffle_ps(lo[3], hi[3], 0xd8);        \
                b4 = _mm_shuffle_ps(lo[3], hi[3], 0x8d);        \
\
                /* L3 processing */             \
                lo[0] = _mm_min_ps(a1, b1);     \
                hi[0] = _mm_max_ps(a1, b1);     \
                lo[1] = _mm_min_ps(a2, b2);     \
                hi[1] = _mm_max_ps(a2, b2);     \
                lo[2] = _mm_min_ps(a3, b3);     \
                hi[2] = _mm_max_ps(a3, b3);     \
                lo[3] = _mm_min_ps(a4, b4);     \
                hi[3] = _mm_max_ps(a4, b4);     \
\
                a1 = _mm_shuffle_ps(lo[0], hi[0], 0x88);        \
                b1 = _mm_shuffle_ps(lo[0], hi[0], 0xdd);        \
                a2 = _mm_shuffle_ps(lo[1], hi[1], 0x88);        \
                b2 = _mm_shuffle_ps(lo[1], hi[1], 0xdd);        \
                a3 = _mm_shuffle_ps(lo[2], hi[2], 0x88);        \
                b3 = _mm_shuffle_ps(lo[2], hi[2], 0xdd);        \
                a4 = _mm_shuffle_ps(lo[3], hi[3], 0x88);        \
                b4 = _mm_shuffle_ps(lo[3], hi[3], 0xdd);        \
\
                a1 = _mm_shuffle_ps(a1, a1, 0xd8);      \
                b1 = _mm_shuffle_ps(b1, b1, 0xd8);      \
                a2 = _mm_shuffle_ps(a2, a2, 0xd8);      \
                b2 = _mm_shuffle_ps(b2, b2, 0xd8);      \
                a3 = _mm_shuffle_ps(a3, a3, 0xd8);      \
                b3 = _mm_shuffle_ps(b3, b3, 0xd8);      \
                a4 = _mm_shuffle_ps(a4, a4, 0xd8);      \
                b4 = _mm_shuffle_ps(b4, b4, 0xd8);      \
        } while (0);

#define __bitonic_merge_network8_sim2(a1, b1, a2, b2, a3, b3, a4, b4)   \
        do {                                    \
                lo[0] = _mm_min_ps(a1, a2);     \
                hi[0] = _mm_max_ps(a1, a2);     \
                lo[1] = _mm_min_ps(b1, b2);     \
                hi[1] = _mm_max_ps(b1, b2);     \
\
                lo[2] = _mm_min_ps(a3, a4);     \
                hi[2] = _mm_max_ps(a3, a4);     \
                lo[3] = _mm_min_ps(b3, b4);     \
                hi[3] = _mm_max_ps(b3, b4);     \
\
                a1 = lo[0];     \
                b1 = lo[1];     \
                a2 = hi[0];     \
                b2 = hi[1];     \
\
                a3 = lo[2];     \
                b3 = lo[3];     \
                a4 = hi[2];     \
                b4 = hi[3];     \
        } while (0);

void
_bitonic_sort32(float *d)
{
        float   *a;
        __m128  reg1[8];
        __m128  reg2[8];
        __m128  lo[4];
        __m128  hi[4];

        /* first in-regsiter sort 4x4 */
        a = d;
        __inregister_sort(a, reg1);

        /* second In-regsiter sort 4x4 */
        a = d + 16;
        __inregister_sort(a, reg2);

        /* execute four 4x4 merging op. simultaneouly */
        reg2[0] = _mm_shuffle_ps(reg2[0], reg2[0], 0x1b);
        reg2[1] = _mm_shuffle_ps(reg2[1], reg2[1], 0x1b);
        reg2[2] = _mm_shuffle_ps(reg2[2], reg2[2], 0x1b);
        reg2[3] = _mm_shuffle_ps(reg2[3], reg2[3], 0x1b);

        __bitonic_merge_network4_sim4(reg1[0], reg2[0], reg1[1], reg2[1], reg1[2], reg2[2], reg1[3], reg2[3]);

        /* execute two 8x8 merging op. simultaneouly */
        reg1[5] = _mm_shuffle_ps(reg1[1], reg1[1], 0x1b);
        reg2[5] = _mm_shuffle_ps(reg2[1], reg2[1], 0x1b);
        reg1[7] = _mm_shuffle_ps(reg1[3], reg1[3], 0x1b);
        reg2[7] = _mm_shuffle_ps(reg2[3], reg2[3], 0x1b);

        reg1[1] = reg2[5];
        reg2[1] = reg1[5];
        reg1[3] = reg2[7];
        reg2[3] = reg1[7];

        __bitonic_merge_network8_sim2(reg1[0], reg2[0], reg1[1], reg2[1], reg1[2], reg2[2], reg1[3], reg2[3]);
        __bitonic_merge_network4_sim4(reg1[0], reg2[0], reg1[1], reg2[1], reg1[2], reg2[2], reg1[3], reg2[3]);

        /* execute a single 16x16 merging op. */
        reg1[2] = _mm_shuffle_ps(reg1[2], reg1[2], 0x1b);
        reg2[2] = _mm_shuffle_ps(reg2[2], reg2[2], 0x1b);
        reg1[3] = _mm_shuffle_ps(reg1[3], reg1[3], 0x1b);
        reg2[3] = _mm_shuffle_ps(reg2[3], reg2[3], 0x1b);

        lo[0] = _mm_min_ps(reg1[0], reg2[3]);
        hi[0] = _mm_max_ps(reg1[0], reg2[3]);

        lo[1] = _mm_min_ps(reg2[0], reg1[3]);
        hi[1] = _mm_max_ps(reg2[0], reg1[3]);

        lo[2] = _mm_min_ps(reg1[1], reg2[2]);
        hi[2] = _mm_max_ps(reg1[1], reg2[2]);

        lo[3] = _mm_min_ps(reg2[1], reg1[2]);
        hi[3] = _mm_max_ps(reg2[1], reg1[2]);

        reg1[0] = lo[0];
        reg2[0] = lo[1];
        reg1[1] = lo[2];
        reg2[1] = lo[3];
        reg1[2] = hi[3];
        reg2[2] = hi[2];
        reg1[3] = hi[1];
        reg2[3] = hi[0];

        __bitonic_merge_network8_sim2(reg1[0], reg2[0], reg1[1], reg2[1], reg1[2], reg2[2], reg1[3], reg2[3]);
        __bitonic_merge_network4_sim4(reg1[0], reg2[0], reg1[1], reg2[1], reg1[2], reg2[2], reg1[3], reg2[3]);

        _mm_store_ps(&d[0], reg1[0]);
        _mm_store_ps(&d[4], reg2[0]);
        _mm_store_ps(&d[8], reg1[1]);
        _mm_store_ps(&d[12], reg2[1]);
        _mm_store_ps(&d[16], reg1[2]);
        _mm_store_ps(&d[20], reg2[2]);
        _mm_store_ps(&d[24], reg1[3]);
        _mm_store_ps(&d[28], reg2[3]);

#ifdef DEBUG
        int32_t i;

        for (i = 1; i < 32; i++) {
                if (d[i - 1] > d[i])
                        eoutput("unsorted input error");
        }
#endif /* DEBUG */
}

void *
_bitonic_sort(void *arg)
{
        int32_t         i;
        float           *d;
        float           *buf;
        struct _chunk_info      *c;

        /* Initialization */
        c = (struct _chunk_info *)arg;
        d = c->d;
        buf = c->buf;

        doutput(FILE_OUTPUT, "[%u]: offset>%u chk->%u", (uint32_t)pthread_self(), c->offset, c->chk);

	/* Sort individual 32 strips */
        for (i = 0; i < c->chk / 32; i++)
                _bitonic_sort32(&d[c->offset + 32 * i]);

	/* Merge 32-long strips */
        _bitonic_merge(&d[c->offset], c->chk, &buf[c->offset], 32);

#ifdef DEBUG
        for (i = 1; i < c->chk; i++) {
                if (d[c->offset + i - 1] > d[c->offset + i])
                        eoutput("unsorted input error");
        }
#endif /* DEBUG */

        return NULL;
}

void _bitonic_merge(float *d, uint32_t s, float *buf, uint32_t chunk_size)
{
	int     step;
	int     step_size;
	float   *src;
	float   *dest;

        for (step_size = chunk_size, src = d, dest = buf; step_size < s; step_size *= 2) {
                for (step = 0; step < s; step += step_size * 2)
			_bitonic_merge_kernel16n(dest + step, src + step, src + step + step_size, step_size);

                _replace_float_adr(src, dest);
	}

	if(src != d)
                _fast_memcpy(d, src, s * sizeof(float));
}

#define LOAD16(arg)     \
	mb[3] = _mm_load_ps(arg);               \
	mb[2] = _mm_load_ps(arg + 4);           \
	mb[1] = _mm_load_ps(arg + 8);           \
	mb[0] = _mm_load_ps(arg + 12);          \
        arg += 16

void 
_bitonic_merge_kernel16n(float *dest, float *a, float *b /* must not be reversed*/, uint32_t s)
{
	float 		*last_a;
	float 		*last_b;
	float 		*last_dest;
        __m128          ma[4];
        __m128          mb[4];
        __m128          lo[4];
        __m128          hi[4];

        /* Settings for termination */
	last_a = a + s;
	last_b = b + s;
	last_dest = dest + s * 2;

#ifdef DEBUG
	int32_t         i;

        for (i = 0; i < s - 1; i++) {
                if (a[i] > a[i + 1] || b[i] > b[i + 1])
                        eoutput("unsorted input error");
        }
#endif /* DEBUG */

	ma[0] = _mm_load_ps(a); a+=4;
	ma[1] = _mm_load_ps(a); a+=4;
	ma[2] = _mm_load_ps(a); a+=4;
	ma[3] = _mm_load_ps(a); a+=4;

	for(; dest < (last_dest - 16); dest += 16) {
                /* Load either a or b */
                if(_likely(a < last_a)) {
                        if(_likely(b < last_b)) {
                                if(*a < *b) {
                                        LOAD16(a);
                                } else { 
                                        LOAD16(b);
                                }
                        } else {
#ifdef SKEW_OPT
                                if (_likely(dest[15] > a[0])) {
                                        LOAD16(a);
                                } else {
                                        _fast_memcpy(dest + 16, a, _ptr_offset(last_a, a));
                                        break;
                                }
#else
                                LOAD16(a);
#endif /* SKEY_OPT */
                        }
                } else {
#ifdef SKEW_OPT
                        if (_likely(dest[15] > b[0])) {
                                LOAD16(b);
                        } else {
                                _fast_memcpy(dest + 16, b, _ptr_offset(last_b, b));
                                break;
                        }
#else
                        LOAD16(b);
#endif /* SKEY_OPT */
                }

                mb[0] = _mm_shuffle_ps(mb[0], mb[0], 0x1b);
                mb[1] = _mm_shuffle_ps(mb[1], mb[1], 0x1b);
                mb[2] = _mm_shuffle_ps(mb[2], mb[2], 0x1b);
                mb[3] = _mm_shuffle_ps(mb[3], mb[3], 0x1b);

                lo[0] = _mm_min_ps(ma[0], mb[0]);
                hi[0] = _mm_max_ps(ma[0], mb[0]);

                lo[1] = _mm_min_ps(ma[1], mb[1]);
                hi[1] = _mm_max_ps(ma[1], mb[1]);

                lo[2] = _mm_min_ps(ma[2], mb[2]);
                hi[2] = _mm_max_ps(ma[2], mb[2]);

                lo[3] = _mm_min_ps(ma[3], mb[3]);
                hi[3] = _mm_max_ps(ma[3], mb[3]);

                _mm_store_ps(&dest[0], lo[0]);
                _mm_store_ps(&dest[4], lo[1]);
                _mm_store_ps(&dest[8], lo[2]);
                _mm_store_ps(&dest[12], lo[3]);
                _mm_store_ps(&dest[16], hi[2]);
                _mm_store_ps(&dest[20], hi[3]);
                _mm_store_ps(&dest[24], hi[0]);
                _mm_store_ps(&dest[28], hi[1]);

                _bitonic_merge_kernel8core(&dest[0], &dest[8]);
                _bitonic_merge_kernel8core(&dest[16], &dest[24]);
        
		ma[0] = _mm_load_ps(&dest[16]);
		ma[1] = _mm_load_ps(&dest[20]);
		ma[2] = _mm_load_ps(&dest[24]);
		ma[3] = _mm_load_ps(&dest[28]);
	}
}

int32_t
_compare_float(const void *a, const void *b)
{
        return (*(float *)a - *(float *)b);
}

void 
_bitonic_sort_kernel4(float *a, float *b)
{
        __m128          ma[2];
        __m128          mb[2];
        __m128          lo;
        __m128          hi;

        ma[0] = _mm_loadu_ps(a);
        mb[0] = _mm_loadu_ps(b);

        /* In-Register sort */
        ma[1] = _mm_shuffle_ps(ma[0], mb[0], 0x88);
        mb[1] = _mm_shuffle_ps(ma[0], mb[0], 0xdd);

        lo = _mm_min_ps(ma[1], mb[1]);
        hi = _mm_max_ps(ma[1], mb[1]);

        ma[0] = _mm_shuffle_ps(hi, lo, 0xd8);
        mb[0] = _mm_shuffle_ps(hi, lo, 0x8d);

        lo = _mm_min_ps(ma[0], mb[0]);
        hi = _mm_max_ps(ma[0], mb[0]);

        ma[0] = _mm_shuffle_ps(lo, lo, 0xd8);
        mb[0] = _mm_shuffle_ps(hi, hi, 0x72);

        lo = _mm_min_ps(ma[0], mb[0]);
        hi = _mm_max_ps(ma[0], mb[0]);

        ma[0]  = _mm_shuffle_ps(lo, hi, 0x41);
        mb[0] = _mm_shuffle_ps(hi, lo, 0xeb);

        _mm_storeu_ps(a, ma[0]);
        _mm_storeu_ps(b, mb[0]);

        _bitonic_merge_kernel4core(a, b);
}

void 
_bitonic_merge_kernel4(float *a, float *b)
{
        __m128          mb;
#ifdef DEBUG
        int32_t         i;
#endif /* DEBUG */

#ifdef DEBUG
        for (i = 0; i < 3; i++) {
                if (a[i] > a[i + 1] || b[i] > b[i + 1])
                        eoutput("unsorted input error");
        }
#endif /* DEBUG */

        /* Reverse *b */
        mb = _mm_loadu_ps(b);
        mb  = _mm_shuffle_ps(mb, mb, 0x1b);
        _mm_storeu_ps(b, mb);

        _bitonic_merge_kernel4core(a, b);
}

void 
_bitonic_merge_kernel4core(float *a, float *b)
{
        __m128          ma;
        __m128          map;
        __m128          mb;
        __m128          mbp;
        __m128          lo;
        __m128          hi;

        ma = _mm_loadu_ps(a);
        mb = _mm_loadu_ps(b);

        /* L1 processing */
        lo = _mm_min_ps(ma, mb);
        hi = _mm_max_ps(ma, mb);

        map = _mm_shuffle_ps(lo, hi, 0xe4);
        mbp = _mm_shuffle_ps(lo, hi, 0x4e);

        /* L2 processing */
        lo = _mm_min_ps(map, mbp);
        hi = _mm_max_ps(map, mbp);

        map = _mm_shuffle_ps(lo, hi, 0xd8);
        mbp = _mm_shuffle_ps(lo, hi, 0x8d);

        /* L3 processing */
        lo = _mm_min_ps(map, mbp);
        hi = _mm_max_ps(map, mbp);

        map = _mm_shuffle_ps(hi, lo, 0x88);
        mbp = _mm_shuffle_ps(hi, lo, 0xdd);

        map = _mm_shuffle_ps(map, map, 0x72);
        mbp = _mm_shuffle_ps(mbp, mbp, 0x72);

        _mm_storeu_ps(a, map);
        _mm_storeu_ps(b, mbp);
}

void 
_bitonic_merge_kernel8(float *a, float *b)
{
        __m128          mb[3];
#ifdef DEBUG
       int32_t          i;
#endif /* DEBUG */

#ifdef DEBUG
        for (i = 0; i < 7; i++) {
                if (a[i] > a[i + 1] || b[i] > b[i + 1])
                        eoutput("unsorted input error");
        }
#endif /* DEBUG */

        /* Reverse *b */
        mb[0] = _mm_loadu_ps(b);
        mb[1] = _mm_loadu_ps(b + 4);
        mb[2] = _mm_shuffle_ps(mb[1], mb[1], 0x1b);
        mb[1] = _mm_shuffle_ps(mb[0], mb[0], 0x1b);
        _mm_storeu_ps(b, mb[2]);
        _mm_storeu_ps(b + 4, mb[1]);

        _bitonic_merge_kernel8core(a, b);
}

void 
_bitonic_merge_kernel8core(float *a, float *b)
{
        __m128          ma[2];
        __m128          mb[2];
        __m128          lo[2];
        __m128          hi[2];

        ma[0] = _mm_loadu_ps(a);
        mb[0] = _mm_loadu_ps(b);

        ma[1] = _mm_loadu_ps(a + 4);
        mb[1] = _mm_loadu_ps(b + 4);

        lo[0] = _mm_min_ps(ma[0], mb[0]);
        hi[0] = _mm_max_ps(ma[0], mb[0]);

        lo[1] = _mm_min_ps(ma[1], mb[1]);
        hi[1] = _mm_max_ps(ma[1], mb[1]);

        _mm_storeu_ps(&a[0], lo[0]);
        _mm_storeu_ps(&a[4], lo[1]);
        _mm_storeu_ps(&b[0], hi[1]);
        _mm_storeu_ps(&b[4], hi[0]);

        _bitonic_merge_kernel4core(&a[0], &a[4]);
        _bitonic_merge_kernel4core(&b[0], &b[4]);
}

void 
_fast_gap_memcpy(uint8_t *dst, uint8_t *src, uint32_t gap, size_t sz)
{
        uint32_t        i;
        __m128i         m1dst;
        __m128i         m2dst;
        __m128i         m3dst;
        __m128i         m1src;
        __m128i         m2src;

        /* unpack must be 16bit aligned */
        if ((((intptr_t)src) & 0x0f) != 0)
                eoutput("Alignment error");

        m2src = _mm_loadu_si128((__m128i *)&src[-16]);
        
        for (i = 0; i < sz / 32; i++) {
                m1dst = _mm_srli_si128(m2src, gap);

                m1src = _mm_loadu_si128((__m128i *)&src[32 * i]);
                m2src = _mm_loadu_si128((__m128i *)&src[32 * i + 16]);

                m2dst = _mm_slli_si128(m1src, 16 - gap);
                m3dst = _mm_or_si128(m1dst, m2dst);

                _mm_storeu_si128((__m128i *)&dst[32 * i - (16 - gap)], m3dst);

                m1dst = _mm_srli_si128(m1src, gap);
                m2dst = _mm_slli_si128(m2src, 16 - gap);

                m3dst = _mm_or_si128(m1dst, m2dst);

                _mm_storeu_si128((__m128i *)&dst[32 * i + gap], m3dst);
        }
}

void 
_fast_memcpy(void *d, void *s, size_t sz)
{
        uint32_t        i;
        uint32_t        gap;
        uint32_t        left;
        uint32_t        offset;
        uint8_t         *dst;
        uint8_t         *src;
        __m128i         msrc;

        dst = (uint8_t *)d;
        src = (uint8_t *)s;

        if (!_enable_fast_memcpy | (sz < 128)) {
                memcpy((void *)dst, (void *)src, sz);
                return;
        }

        left = sz;
        offset = 0;

        if ((((intptr_t)src) & 0x0f) == (((intptr_t)dst) & 0x0f)) {
                if (((intptr_t)src & 0x0f) != 0) {
                        offset = 0x10 - (((intptr_t)src) & 0x0f);
                        memcpy((void *)&dst[0], (void *)&src[0], offset);
                        left -= offset;
                }

                for (i = 0; i < (sz - offset) / 16; i++, left -= 16) {
                        msrc = _mm_loadu_si128((__m128i *)&src[offset + 16 * i]);
                        _mm_storeu_si128((__m128i *)&dst[offset + 16 * i], msrc);
                }

                memcpy((void *)&dst[offset + 16 * i], (void *)&src[offset + 16 * i], left);
        } else {
                offset = 0x10 - (((intptr_t)src) & 0x0f) + 16;
                gap = 0x10 - (((intptr_t)&dst[offset]) & 0x0f);

                memcpy((void *)&dst[0], (void *)&src[0], offset);

                left -= offset;

                switch (gap) {
                case 1:
                        _fast_gap_memcpy(&dst[offset], &src[offset], 1, left);
                        break;

                case 2:
                        _fast_gap_memcpy(&dst[offset], &src[offset], 2, left);
                        break;

                case 3:
                        _fast_gap_memcpy(&dst[offset], &src[offset], 3, left);
                        break;

                case 4:
                        _fast_gap_memcpy(&dst[offset], &src[offset], 4, left);
                        break;

                case 5:
                        _fast_gap_memcpy(&dst[offset], &src[offset], 5, left);
                        break;

                case 6:
                        _fast_gap_memcpy(&dst[offset], &src[offset], 6, left);
                        break;

                case 7:
                        _fast_gap_memcpy(&dst[offset], &src[offset], 7, left);
                        break;

                case 8:
                        _fast_gap_memcpy(&dst[offset], &src[offset], 8, left);
                        break;

                case 9:
                        _fast_gap_memcpy(&dst[offset], &src[offset], 9, left);
                        break;

                case 10:
                        _fast_gap_memcpy(&dst[offset], &src[offset], 10, left);
                        break;

                case 11:
                        _fast_gap_memcpy(&dst[offset], &src[offset], 11, left);
                        break;

                case 12:
                        _fast_gap_memcpy(&dst[offset], &src[offset], 12, left);
                        break;

                case 13:
                        _fast_gap_memcpy(&dst[offset], &src[offset], 13, left);
                        break;

                case 14:
                        _fast_gap_memcpy(&dst[offset], &src[offset], 14, left);
                        break;

                case 15:
                        _fast_gap_memcpy(&dst[offset], &src[offset], 15, left);
                        break;

                default:
                        eoutput("Out of range: gap");
                        break;
                }

                memcpy((void *)&dst[offset + 32 * ((sz - offset) / 32) + gap - 16],
                                (void *)&src[offset + 32 * ((sz - offset) / 32) + gap - 16], 16 - gap + (sz - offset) % 32);
        }
}

