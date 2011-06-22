/*-------------------------------------------------------------------------
 *
 * err_utils.h - 20110610 v0.1.3
 *	  A header for err_utils.c
 *
 *-------------------------------------------------------------------------
 */

#ifndef ERR_UTILS_H
#define ERR_UTILS_H

#define FILE_OUTPUT             0
#define CONSOLE_OUTPUT          1

#define eoutput(fmt, ...)       \
        do {                    \
                flush_log();    \
                err_print(__func__, __LINE__, fmt, ##__VA_ARGS__); \
        } while (0)

#ifdef DEBUG
#define doutput(flag, fmt, ...) \
        do {                    \
                if ((flag) == FILE_OUTPUT) {                                    \
                        push_log(__func__, __LINE__, fmt, ##__VA_ARGS__);       \
                } else if ((flag) == CONSOLE_OUTPUT) {                          \
                        fprintf(stderr, "%s(%d): ", __func__, __LINE__);        \
                        fprintf(stderr, fmt, ##__VA_ARGS__);                    \
                        fprintf(stderr, "\n");                                  \
                }               \
        } while (0)
#else
#define doutput(fmt, ...)
#endif /* DEBUG */

/* Logging functions */
extern void push_log(const char *func, int32_t line, const char *fmt, ...)
        __attribute__ ((format (printf, 3, 4))); 
extern void flush_log(void);

/* Error functions */
extern void err_print(const char *func, int32_t line, const char *fmt, ...)
        __attribute__ ((format (printf, 3, 4))); 

#endif  /* ERR_UTILS_H */
