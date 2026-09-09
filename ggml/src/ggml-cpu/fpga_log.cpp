#include "fpga_log.h"

#include <algorithm>
#include <cerrno>
#include <cstring>
#include <ctime>
#include <fcntl.h>
#include <unistd.h>

namespace {

constexpr const char * FPGA_LOG_FILE = "/tmp/fpga_debug.log";

int g_flush_every   = 256;
int g_pending_lines = 0;

} // namespace

FILE * fpga_log_fp() {
    static FILE * fp = nullptr;
    if (!fp) {
        // Opening an existing file with fopen("a") includes O_CREAT. Linux
        // fs.protected_regular can reject that operation in sticky /tmp when
        // root runs llama-cli against a log owned by the debian user. Open the
        // existing file without O_CREAT first, then create it only if absent.
        int fd = open(FPGA_LOG_FILE, O_WRONLY | O_APPEND | O_CLOEXEC);
        if (fd < 0 && errno == ENOENT) {
            fd = open(FPGA_LOG_FILE, O_WRONLY | O_APPEND | O_CREAT | O_CLOEXEC, 0644);
        }
        if (fd >= 0) {
            fp = fdopen(fd, "a");
            if (!fp) {
                close(fd);
            }
        }
        if (!fp) {
            const int open_errno = errno;
            fprintf(stderr,
                    "[FPGA][ERROR] cannot open %s; detailed FPGA telemetry is disabled: errno=%d (%s)\n",
                    FPGA_LOG_FILE, open_errno, strerror(open_errno));
            fp = fopen("/dev/null", "w");
            if (!fp) {
                fp = stderr;
            }
        }

        const time_t now = time(nullptr);
        fprintf(fp, "\n============================================================\n");
        fprintf(fp, "[FPGA] ZDMA DDR-to-IP log started at %ld\n", (long) now);
        fprintf(fp, "============================================================\n");
        fflush(fp);
    }
    return fp;
}

void fpga_log_set_flush_every(int flush_every) {
    g_flush_every = std::max(1, flush_every);
}

void fpga_log_finish_line(FILE * fp, bool force_flush) {
    ++g_pending_lines;
    if (force_flush || g_flush_every <= 1 || g_pending_lines >= g_flush_every) {
        fflush(fp);
        g_pending_lines = 0;
    }
}

void fpga_log_vline(const char * tag, bool force_flush, const char * fmt, va_list ap) {
    FILE * fp = fpga_log_fp();
    fprintf(fp, "[FPGA][%s] ", tag ? tag : "INFO");
    vfprintf(fp, fmt, ap);
    fputc('\n', fp);
    fpga_log_finish_line(fp, force_flush);
}
