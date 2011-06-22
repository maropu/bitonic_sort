/*-------------------------------------------------------------------------
 *
 * err_utils.c - 20110610 v0.1.3
 *	  output routines for  error/debug information 
 *
 *-------------------------------------------------------------------------
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdarg.h>
#include <stdint.h>
#include <signal.h>

static void _init_push_log(void);
static void _sig_handler(int n);

static FILE     *_log;
static int     _init_utils;

void 
err_print(const char *func, int32_t line, const char *fmt, ...)
{
        uint32_t        n;
        char            buf[1024];
        va_list         ap;

        memset(buf, 0x00, 1024);

        va_start(ap, fmt);
        if ((n = vsnprintf(buf, sizeof(buf), fmt, ap)) == -1)
                exit(EXIT_FAILURE);
        va_end(ap);

        buf[n] = '\n';

        fprintf(stderr, "%s(%d): ", func, line);
        fprintf(stderr, "%s", buf);
        exit(EXIT_FAILURE);
}

void 
push_log(const char *func, int32_t line, const char *fmt, ...)
{
        uint32_t        n;
        char            buf[1024];
        va_list         ap;

        memset(buf, 0x00, 1024);

        if (!_init_utils++)
                _init_push_log();

        va_start(ap, fmt);

        if ((n = vsnprintf(buf, sizeof(buf), fmt, ap)) == -1) {
                fprintf(stderr, "_push_log(): Irregal format errors\n");
                exit(EXIT_FAILURE);
        }

        va_end(ap);

        buf[n] = '\n';

        fprintf(_log, "%s(%d): ", func, line);
        fprintf(_log, "%s", buf);
}

void
flush_log(void)
{
        if (_log != NULL) {
                fflush(_log);
                fclose(_log);
        }
}

/* --- Intra functions below */

#define SIGSEGV 11

void
_sig_handler(int n)
{
        //flush_log();
        fprintf(stderr, "SIGSEGV is catched.\n");
        exit(-1);
}

void
_init_push_log(void)
{
        /* Open a file for debug information */
        _log = fopen("output.log", "w");

        if (_log == NULL) {
                fprintf(stderr, "_push_log(): Can't open a log file\n");
                exit(EXIT_FAILURE);
        }

        /* Register sinal functions */
        if (SIG_ERR == signal(SIGSEGV, _sig_handler)) {
                fprintf(stderr, "_push_log(): Can't register signal functions\n");
                exit(EXIT_FAILURE);
        }
}
