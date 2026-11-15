/* Measure simultaneous selected-slot Zynq UltraScale+ DDR-controller APM traffic. */
#define _GNU_SOURCE

#include <errno.h>
#include <fcntl.h>
#include <getopt.h>
#include <limits.h>
#include <math.h>
#include <signal.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/file.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <sys/sysmacros.h>
#include <time.h>
#include <unistd.h>

#define DDR_APM_PHYS_ADDR UINT64_C(0xfd0b0000)
#define DDR_APM_MAP_BYTES UINT64_C(0x10000)
#define DDR_APM_SLOT_COUNT 6U
#define DDR_APM_COUNTER_COUNT 10U
#define DEFAULT_UIO "/dev/uio1"
#define DEFAULT_SECONDS 60.0
#define WINDOW_SECONDS 0.100

#define REG_SAMPLE_INTERVAL_HI 0x0020U
#define REG_SAMPLE_INTERVAL_LO 0x0024U
#define REG_SAMPLE_INTERVAL_CTL 0x0028U
#define REG_GLOBAL_INTERRUPT 0x0030U
#define REG_INTERRUPT_STATUS 0x0038U
#define REG_INTERRUPT_ENABLE 0x0034U
#define REG_METRIC_SELECTOR0 0x0044U
#define REG_METRIC_SELECTOR1 0x0048U
#define REG_METRIC_SELECTOR2 0x004cU
#define REG_METRIC_COUNTER0 0x0100U
#define REG_METRIC_COUNTER1 0x0110U
#define REG_CONTROL 0x0300U

#define CTRL_METRIC_ENABLE UINT32_C(0x00000001)
#define CTRL_METRIC_RESET UINT32_C(0x00000002)
#define CTRL_METRIC_EXT_TRIGGER UINT32_C(0x00000004)
#define CTRL_ID_FILTER UINT32_C(0x00000008)
#define CTRL_WRITE_LAT_START UINT32_C(0x00000010)
#define CTRL_WRITE_LAT_END UINT32_C(0x00000020)
#define CTRL_READ_LAT_START UINT32_C(0x00000040)
#define CTRL_READ_LAT_END UINT32_C(0x00000080)
#define CTRL_EVENT_LOG_ENABLE UINT32_C(0x00000100)
#define CTRL_EVENT_LOG_TRIGGER UINT32_C(0x00000200)
#define CTRL_GLOBAL_ENABLE UINT32_C(0x00010000)
#define CTRL_GLOBAL_RESET UINT32_C(0x00020000)
#define CTRL_FIFO_RESET UINT32_C(0x02000000)
#define CTRL_BUSY_MASK (CTRL_METRIC_ENABLE | CTRL_METRIC_RESET | CTRL_METRIC_EXT_TRIGGER | CTRL_ID_FILTER | \
                        CTRL_EVENT_LOG_ENABLE | CTRL_EVENT_LOG_TRIGGER | CTRL_GLOBAL_ENABLE | CTRL_GLOBAL_RESET | \
                        CTRL_FIFO_RESET)

#define SICR_ENABLE UINT32_C(0x00000001)
#define SICR_LOAD UINT32_C(0x00000002)
#define SICR_METRIC_RESET UINT32_C(0x00000100)
#define ISR_COUNTER0_OVERFLOW UINT32_C(0x00000008)
#define ISR_COUNTER1_OVERFLOW UINT32_C(0x00000010)
#define ISR_COUNTER_OVERFLOW_MASK UINT32_C(0x00001ff8)

struct options {
    const char * uio_path;
    unsigned slots[DDR_APM_COUNTER_COUNT / 2U];
    unsigned slot_count;
    int slot_set;
    double seconds;
};

struct mapping {
    int uio_fd;
    size_t map0_bytes;
    size_t map1_bytes;
    volatile uint32_t * regs;
    volatile uint32_t * params;
};

struct saved_state {
    uint32_t selectors[3];
    uint32_t control;
    uint32_t global_interrupt;
    uint32_t interrupt_enable;
    uint32_t sample_interval_hi;
    uint32_t sample_interval_lo;
    uint32_t sample_interval_control;
    int valid;
};

static volatile sig_atomic_t stop_requested;
static struct saved_state saved;

static void on_signal(int signal_number) {
    (void) signal_number;
    stop_requested = 1;
}

static void usage(FILE * stream, const char * argv0) {
    fprintf(stream,
            "Usage: %s (--slot N | --slots N,N,...) [--uio /dev/uioN] [--seconds S]\n"
            "Measure 1..5 unique DDR APM slots (N=0..5) in shared 100 ms windows.\n"
            "Default UIO device: " DEFAULT_UIO "\n", argv0);
}

static int parse_unsigned(const char * text, unsigned * value) {
    char * end = NULL;
    unsigned long parsed;
    if (!text || !*text || text[0] == '-') {
        return -1;
    }
    errno = 0;
    parsed = strtoul(text, &end, 0);
    if (errno || end == text || *end || parsed > UINT_MAX) {
        return -1;
    }
    *value = (unsigned) parsed;
    return 0;
}

/* List syntax is deliberately strict; --slot retains its existing numeric syntax. */
static int parse_slots(const char * text, struct options * options) {
    unsigned count = 0U;
    unsigned seen = 0U;
    if (!text || !*text) return -1;
    while (*text) {
        unsigned slot;
        if (*text < '0' || *text > '5' || count >= DDR_APM_COUNTER_COUNT / 2U) return -1;
        slot = (unsigned) (*text++ - '0');
        if ((seen & (1U << slot)) != 0U || (*text && *text != ',')) return -1;
        seen |= 1U << slot;
        options->slots[count++] = slot;
        if (*text == ',' && !*++text) return -1;
    }
    options->slot_count = count;
    return 0;
}

static int parse_seconds(const char * text, double * value) {
    char * end = NULL;
    double parsed;
    if (!text || !*text || text[0] == '-') {
        return -1;
    }
    errno = 0;
    parsed = strtod(text, &end);
    if (errno || end == text || *end || !(parsed > 0.0) || parsed > 86400.0) {
        return -1;
    }
    *value = parsed;
    return 0;
}

static int parse_options(int argc, char ** argv, struct options * options) {
    static const struct option long_options[] = {
        { "help", no_argument, NULL, 'h' },
        { "slot", required_argument, NULL, 's' },
        { "slots", required_argument, NULL, 'S' },
        { "seconds", required_argument, NULL, 't' },
        { "uio", required_argument, NULL, 'u' },
        { NULL, 0, NULL, 0 }
    };
    int option;
    options->uio_path = DEFAULT_UIO;
    options->slot_count = 0U;
    options->slot_set = 0;
    options->seconds = DEFAULT_SECONDS;
    while ((option = getopt_long(argc, argv, "hs:S:t:u:", long_options, NULL)) != -1) {
        switch (option) {
        case 'h':
            usage(stdout, argv[0]);
            return 1;
        case 's':
            if (options->slot_set || parse_unsigned(optarg, &options->slots[0]) != 0 ||
                options->slots[0] >= DDR_APM_SLOT_COUNT) {
                fprintf(stderr, "Invalid or repeated slot selector; --slot expects 0..%u.\n", DDR_APM_SLOT_COUNT - 1U);
                return -1;
            }
            options->slot_count = 1U;
            options->slot_set = 1;
            break;
        case 'S':
            if (options->slot_set || parse_slots(optarg, options) != 0) {
                fprintf(stderr, "Invalid or repeated slot selector; --slots expects 1..5 unique slots from 0..5.\n");
                return -1;
            }
            options->slot_set = 1;
            break;
        case 't':
            if (parse_seconds(optarg, &options->seconds) != 0) {
                fprintf(stderr, "Invalid --seconds; expected a number in (0, 86400].\n");
                return -1;
            }
            break;
        case 'u':
            options->uio_path = optarg;
            break;
        default:
            usage(stderr, argv[0]);
            return -1;
        }
    }
    if (!options->slot_set || optind != argc) {
        usage(stderr, argv[0]);
        return -1;
    }
    return 0;
}

static int read_text(const char * path, char * buffer, size_t buffer_size) {
    FILE * file;
    size_t length;
    file = fopen(path, "r");
    if (!file || !fgets(buffer, (int) buffer_size, file)) {
        if (file) fclose(file);
        return -1;
    }
    fclose(file);
    length = strlen(buffer);
    while (length > 0U && (buffer[length - 1U] == '\n' || buffer[length - 1U] == '\r' ||
                           buffer[length - 1U] == ' ' || buffer[length - 1U] == '\t')) {
        buffer[--length] = '\0';
    }
    return 0;
}

static int read_hex(const char * path, uint64_t * value) {
    char text[64];
    char * end = NULL;
    unsigned long long parsed;
    if (read_text(path, text, sizeof(text)) != 0) {
        return -1;
    }
    errno = 0;
    parsed = strtoull(text, &end, 0);
    if (errno || end == text || *end) {
        return -1;
    }
    *value = (uint64_t) parsed;
    return 0;
}

static int uio_number(const char * path, char * number, size_t number_size) {
    const char * base;
    size_t length;
    if (!path || strncmp(path, "/dev/uio", 8U) != 0) return -1;
    base = path + 8U;
    length = strlen(base);
    if (length == 0U || length >= number_size) return -1;
    for (size_t i = 0U; i < length; ++i) {
        if (base[i] < '0' || base[i] > '9') return -1;
    }
    memcpy(number, base, length + 1U);
    return 0;
}

static int read_map_property(const char * number, unsigned map, const char * property, uint64_t * value) {
    char path[PATH_MAX];
    if (snprintf(path, sizeof(path), "/sys/class/uio/uio%s/maps/map%u/%s", number, map, property) >= (int) sizeof(path)) {
        return -1;
    }
    return read_hex(path, value);
}

static int validate_opened_uio(const char * path, int fd) {
    char number[32];
    char sysfs_path[PATH_MAX];
    char text[64];
    unsigned int sys_major;
    unsigned int sys_minor;
    struct stat device_stat;
    if (uio_number(path, number, sizeof(number)) != 0 || fstat(fd, &device_stat) != 0 ||
        !S_ISCHR(device_stat.st_mode)) {
        fprintf(stderr, "Opened UIO path is not the expected character device: %s\n", path);
        return -1;
    }
    if (snprintf(sysfs_path, sizeof(sysfs_path), "/sys/class/uio/uio%s/dev", number) >= (int) sizeof(sysfs_path) ||
        read_text(sysfs_path, text, sizeof(text)) != 0 || sscanf(text, "%u:%u", &sys_major, &sys_minor) != 2 ||
        major(device_stat.st_rdev) != sys_major || minor(device_stat.st_rdev) != sys_minor) {
        fprintf(stderr, "Opened device does not match sysfs UIO identity: %s\n", path);
        return -1;
    }
    return 0;
}

static int validate_uio(const char * path, size_t * map0_bytes, size_t * map1_bytes) {
    char number[32];
    char sysfs_path[PATH_MAX];
    char name[64];
    uint64_t address;
    uint64_t size;
    uint64_t offset;
    if (uio_number(path, number, sizeof(number)) != 0) {
        fprintf(stderr, "UIO path must be /dev/uioN: %s\n", path);
        return -1;
    }
    if (snprintf(sysfs_path, sizeof(sysfs_path), "/sys/class/uio/uio%s/name", number) >= (int) sizeof(sysfs_path) ||
        read_text(sysfs_path, name, sizeof(name)) != 0 || strcmp(name, "axi-pmon") != 0) {
        fprintf(stderr, "UIO %s is not named axi-pmon.\n", path);
        return -1;
    }
    if (read_map_property(number, 0U, "addr", &address) != 0 || address != DDR_APM_PHYS_ADDR) {
        fprintf(stderr, "UIO %s map0 is not DDR APM address 0x%llx.\n", path,
                (unsigned long long) DDR_APM_PHYS_ADDR);
        return -1;
    }
    if (read_map_property(number, 0U, "size", &size) != 0 || size < DDR_APM_MAP_BYTES || size > SIZE_MAX ||
        read_map_property(number, 0U, "offset", &offset) != 0 || offset != 0U) {
        fprintf(stderr, "UIO %s map0 must be 64 KiB at offset zero.\n", path);
        return -1;
    }
    *map0_bytes = (size_t) size;
    if (read_map_property(number, 1U, "size", &size) != 0 || size < (uint64_t) getpagesize() || size > SIZE_MAX ||
        read_map_property(number, 1U, "offset", &offset) != 0 || offset != 0U) {
        fprintf(stderr, "UIO %s has no readable APM parameter page (map1).\n", path);
        return -1;
    }
    *map1_bytes = (size_t) size;
    return 0;
}

static void mmio_barrier(void) {
#if defined(__aarch64__) || defined(__arm__)
    __asm__ volatile("dsb sy" ::: "memory");
#else
    __atomic_thread_fence(__ATOMIC_SEQ_CST);
#endif
}

static uint32_t mmio_read(volatile uint32_t * regs, uint32_t offset) {
    uint32_t value;
    mmio_barrier();
    value = regs[offset / sizeof(uint32_t)];
    mmio_barrier();
    return value;
}

static void mmio_write(volatile uint32_t * regs, uint32_t offset, uint32_t value) {
    mmio_barrier();
    regs[offset / sizeof(uint32_t)] = value;
    mmio_barrier();
    (void) mmio_read(regs, offset);
}

static int prepare_counter_window(volatile uint32_t * regs) {
    uint32_t status = mmio_read(regs, REG_INTERRUPT_STATUS);
    if ((status & ~ISR_COUNTER_OVERFLOW_MASK) != 0U) {
        fprintf(stderr, "APM has an unrecognized sticky status before a window: 0x%08x.\n", status);
        return -1;
    }
    if ((status & ISR_COUNTER_OVERFLOW_MASK) != 0U) {
        mmio_write(regs, REG_INTERRUPT_STATUS, status & ISR_COUNTER_OVERFLOW_MASK);
        status = mmio_read(regs, REG_INTERRUPT_STATUS);
        if (status != 0U) {
            fprintf(stderr, "APM sticky status did not clear before a window: 0x%08x.\n", status);
            return -1;
        }
    }
    return 0;
}

static void clear_counter_overflows_for_cleanup(volatile uint32_t * regs) {
    const uint32_t status = mmio_read(regs, REG_INTERRUPT_STATUS);
    if ((status & ISR_COUNTER_OVERFLOW_MASK) != 0U) {
        mmio_write(regs, REG_INTERRUPT_STATUS, status & ISR_COUNTER_OVERFLOW_MASK);
    }
}

static int acquire_lock(struct mapping * mapping) {
    if (flock(mapping->uio_fd, LOCK_EX | LOCK_NB) != 0) {
        fprintf(stderr, "Another client owns the APM UIO device; refusing to configure it.\n");
        return -1;
    }
    return 0;
}

static int map_apm(const char * path, struct mapping * mapping) {
    const long page_size = getpagesize();
    if (validate_uio(path, &mapping->map0_bytes, &mapping->map1_bytes) != 0) return -1;
    mapping->uio_fd = open(path, O_RDWR | O_CLOEXEC);
    if (mapping->uio_fd < 0) {
        fprintf(stderr, "open %s failed: %s\n", path, strerror(errno));
        return -1;
    }
    if (validate_opened_uio(path, mapping->uio_fd) != 0 || acquire_lock(mapping) != 0) {
        close(mapping->uio_fd);
        mapping->uio_fd = -1;
        return -1;
    }
    mapping->regs = mmap(NULL, mapping->map0_bytes, PROT_READ | PROT_WRITE, MAP_SHARED, mapping->uio_fd, 0);
    if (mapping->regs == MAP_FAILED) {
        mapping->regs = NULL;
        fprintf(stderr, "mmap map0 failed: %s\n", strerror(errno));
        return -1;
    }
    mapping->params = mmap(NULL, mapping->map1_bytes, PROT_READ, MAP_SHARED, mapping->uio_fd, page_size);
    if (mapping->params == MAP_FAILED) {
        mapping->params = NULL;
        fprintf(stderr, "mmap map1 failed: %s\n", strerror(errno));
        return -1;
    }
    return 0;
}

static void unmap_apm(struct mapping * mapping) {
    if (mapping->params) munmap((void *) mapping->params, mapping->map1_bytes);
    if (mapping->regs) munmap((void *) mapping->regs, mapping->map0_bytes);
    if (mapping->uio_fd >= 0) (void) flock(mapping->uio_fd, LOCK_UN);
    if (mapping->uio_fd >= 0) close(mapping->uio_fd);
    mapping->params = NULL;
    mapping->regs = NULL;
    mapping->uio_fd = -1;
}

static int check_capabilities(const struct mapping * mapping) {
    const volatile uint32_t * p = mapping->params;
    const uint32_t mode = p[0];
    const uint32_t slots = p[1];
    const uint32_t event_count = p[2];
    const uint32_t counters = p[5];
    const uint32_t metric_width = p[6];
    const uint32_t scale = p[9];
    if (printf("APM mode=%u slots=%u counters=%u metric_width=%u scale=%u event_count=%u\n",
               mode, slots, counters, metric_width, scale, event_count) < 0 ||
        fflush(stdout) != 0 || ferror(stdout)) {
        fprintf(stderr, "stdout write failed; refusing to configure the APM.\n");
        return -1;
    }
    if (mode != 1U || slots < DDR_APM_SLOT_COUNT || event_count == 0U || counters < DDR_APM_COUNTER_COUNT ||
        metric_width != 32U || scale != 1U) {
        fprintf(stderr, "APM capabilities do not match the Zynq DDR monitor contract.\n");
        return -1;
    }
    return 0;
}

static int check_idle(volatile uint32_t * regs) {
    const uint32_t control = mmio_read(regs, REG_CONTROL);
    const uint32_t sample_control = mmio_read(regs, REG_SAMPLE_INTERVAL_CTL);
    const uint32_t global_interrupt = mmio_read(regs, REG_GLOBAL_INTERRUPT);
    const uint32_t interrupt_enable = mmio_read(regs, REG_INTERRUPT_ENABLE);
    const uint32_t interrupt_status = mmio_read(regs, REG_INTERRUPT_STATUS);
    if ((control & CTRL_BUSY_MASK) != 0U || (sample_control & (SICR_ENABLE | SICR_LOAD)) != 0U ||
        global_interrupt != 0U || interrupt_enable != 0U || interrupt_status != 0U) {
        fprintf(stderr, "APM is already configured or active; refusing ownership "
                "(control=0x%08x sample_ctl=0x%08x gie=0x%08x ie=0x%08x is=0x%08x).\n",
                control, sample_control, global_interrupt, interrupt_enable, interrupt_status);
        return -1;
    }
    return 0;
}

static void save_state(volatile uint32_t * regs) {
    saved.selectors[0] = mmio_read(regs, REG_METRIC_SELECTOR0);
    saved.selectors[1] = mmio_read(regs, REG_METRIC_SELECTOR1);
    saved.selectors[2] = mmio_read(regs, REG_METRIC_SELECTOR2);
    saved.control = mmio_read(regs, REG_CONTROL);
    saved.global_interrupt = mmio_read(regs, REG_GLOBAL_INTERRUPT);
    saved.interrupt_enable = mmio_read(regs, REG_INTERRUPT_ENABLE);
    saved.sample_interval_hi = mmio_read(regs, REG_SAMPLE_INTERVAL_HI);
    saved.sample_interval_lo = mmio_read(regs, REG_SAMPLE_INTERVAL_LO);
    saved.sample_interval_control = mmio_read(regs, REG_SAMPLE_INTERVAL_CTL);
    saved.valid = 1;
}

static void restore_state(volatile uint32_t * regs) {
    if (!regs || !saved.valid) return;
    mmio_write(regs, REG_CONTROL, saved.control & ~CTRL_METRIC_ENABLE);
    clear_counter_overflows_for_cleanup(regs);
    mmio_write(regs, REG_METRIC_SELECTOR0, saved.selectors[0]);
    mmio_write(regs, REG_METRIC_SELECTOR1, saved.selectors[1]);
    mmio_write(regs, REG_METRIC_SELECTOR2, saved.selectors[2]);
    mmio_write(regs, REG_SAMPLE_INTERVAL_HI, saved.sample_interval_hi);
    mmio_write(regs, REG_SAMPLE_INTERVAL_LO, saved.sample_interval_lo);
    mmio_write(regs, REG_SAMPLE_INTERVAL_CTL, saved.sample_interval_control);
    mmio_write(regs, REG_INTERRUPT_ENABLE, saved.interrupt_enable);
    mmio_write(regs, REG_GLOBAL_INTERRUPT, saved.global_interrupt);
    mmio_write(regs, REG_CONTROL, saved.control);
    saved.valid = 0;
}

static uint32_t selector(unsigned slot, unsigned metric) {
    return (uint32_t) ((slot << 5U) | metric);
}

static int configure_selectors(volatile uint32_t * regs, const struct options * options) {
    uint32_t packed[3];
    const unsigned count = options->slot_count * 2U;
    if (options->slot_count == 0U || count > DDR_APM_COUNTER_COUNT) return -1;
    memcpy(packed, saved.selectors, sizeof(packed));
    for (unsigned i = 0U; i < count; ++i) {
        const unsigned shift = (i % 4U) * 8U;
        const uint32_t field = selector(options->slots[i / 2U], 2U + i % 2U);
        packed[i / 4U] = (packed[i / 4U] & ~(UINT32_C(0xff) << shift)) | (field << shift);
    }
    for (unsigned i = 0U; i < (count + 3U) / 4U; ++i) {
        const uint32_t offset = REG_METRIC_SELECTOR0 + 4U * i;
        mmio_write(regs, offset, packed[i]);
        if (mmio_read(regs, offset) != packed[i]) {
            fprintf(stderr, "APM selector readback failed; measurement is invalid.\n");
            return -1;
        }
    }
    return 0;
}

static void reset_counters(volatile uint32_t * regs, uint32_t control_without_enable) {
    mmio_write(regs, REG_CONTROL, control_without_enable | CTRL_METRIC_RESET);
    mmio_write(regs, REG_CONTROL, control_without_enable);
}

static double seconds_between(const struct timespec * first, const struct timespec * second) {
    return (double) (second->tv_sec - first->tv_sec) +
           (double) (second->tv_nsec - first->tv_nsec) / 1000000000.0;
}

static int sleep_seconds(double seconds) {
    struct timespec request;
    request.tv_sec = (time_t) seconds;
    request.tv_nsec = (long) ((seconds - (double) request.tv_sec) * 1000000000.0);
    if (request.tv_nsec < 0L) request.tv_nsec = 0L;
    while (!stop_requested && nanosleep(&request, &request) != 0) {
        if (errno != EINTR) return -1;
    }
    return stop_requested ? -1 : 0;
}

static int print_wall_timestamp(const struct timespec * value) {
    struct tm utc;
    char text[64];
    time_t seconds = value->tv_sec;
    if (!gmtime_r(&seconds, &utc) || strftime(text, sizeof(text), "%Y-%m-%dT%H:%M:%S", &utc) == 0U) {
        return fputs("unknown", stdout) < 0 ? -1 : 0;
    }
    return printf("%s.%03ldZ", text, value->tv_nsec / 1000000L) < 0 ? -1 : 0;
}

static int run_measurement(volatile uint32_t * regs, const struct options * options) {
    const uint32_t control_without_enable = saved.control & ~CTRL_METRIC_ENABLE;
    const double seconds = options->seconds;
    double measured_active = 0.0;
    uint64_t total_write[DDR_APM_COUNTER_COUNT / 2U] = { 0U };
    uint64_t total_read[DDR_APM_COUNTER_COUNT / 2U] = { 0U };
    unsigned windows = 0U;
    unsigned valid_windows = 0U;
    int failed = 0;
    struct timespec session_begin;
    struct timespec session_end;
    struct timespec last_window_end;

    if (!isfinite(seconds) || seconds <= 0.0 ||
        clock_gettime(CLOCK_MONOTONIC, &session_begin) != 0) return -1;
    last_window_end = session_begin;
    /* The ownership gate guarantees stopped counters; selectors are never changed live. */
    if (configure_selectors(regs, options) != 0) return -1;
    while (!stop_requested) {
        struct timespec now, wall_timestamp, active_begin, active_end, wall_end;
        uint32_t write_bytes[DDR_APM_COUNTER_COUNT / 2U];
        uint32_t read_bytes[DDR_APM_COUNTER_COUNT / 2U];
        uint32_t status;
        double active_elapsed, window_elapsed, overhead;
        if (clock_gettime(CLOCK_MONOTONIC, &now) != 0) { failed = 1; break; }
        const double session_elapsed = seconds_between(&session_begin, &now);
        if (!isfinite(session_elapsed) || session_elapsed < measured_active ||
            seconds_between(&last_window_end, &now) < 0.0) { failed = 1; break; }
        if (session_elapsed >= seconds) break;
        const double requested_window = seconds - session_elapsed > WINDOW_SECONDS ?
            WINDOW_SECONDS : seconds - session_elapsed;
        if (clock_gettime(CLOCK_REALTIME, &wall_timestamp) != 0 || prepare_counter_window(regs) != 0) {
            failed = 1; break;
        }
        /* Shared reset affects ALL metric counters, including unselected counters. */
        reset_counters(regs, control_without_enable);
        if (clock_gettime(CLOCK_MONOTONIC, &active_begin) != 0) { failed = 1; break; }
        mmio_write(regs, REG_CONTROL, control_without_enable | CTRL_METRIC_ENABLE);
        const int sleep_result = sleep_seconds(requested_window);
        /* One shared stop and readback precedes every selected counter/status read. */
        mmio_write(regs, REG_CONTROL, control_without_enable);
        if (sleep_result != 0 || stop_requested || clock_gettime(CLOCK_MONOTONIC, &active_end) != 0) {
            failed = 1; break;
        }
        status = mmio_read(regs, REG_INTERRUPT_STATUS);
        for (unsigned i = 0U; i < options->slot_count; ++i) {
            write_bytes[i] = mmio_read(regs, REG_METRIC_COUNTER0 + 0x10U * (2U * i));
            read_bytes[i] = mmio_read(regs, REG_METRIC_COUNTER0 + 0x10U * (2U * i + 1U));
        }
        if (clock_gettime(CLOCK_MONOTONIC, &wall_end) != 0) { failed = 1; break; }
        active_elapsed = seconds_between(&active_begin, &active_end);
        window_elapsed = seconds_between(&now, &wall_end);
        overhead = window_elapsed - active_elapsed;
        ++windows;
        if (status != 0U || !isfinite(active_elapsed) || active_elapsed <= 0.0 ||
            !isfinite(window_elapsed) || window_elapsed < active_elapsed ||
            seconds_between(&now, &active_begin) < 0.0 || seconds_between(&active_end, &wall_end) < 0.0) {
            fprintf(stderr, "APM window %u invalid: status=ERROR(0x%08x) active_s=%.9f window_s=%.9f; aggregate is invalid.\n",
                    windows, status, active_elapsed, window_elapsed);
            failed = 1; break;
        }
        for (unsigned i = 0U; i < options->slot_count; ++i) {
            const uint64_t combined = (uint64_t) write_bytes[i] + read_bytes[i];
            if (print_wall_timestamp(&wall_timestamp) != 0 ||
                printf(" slot=%u window=%u active_s=%.9f window_s=%.9f overhead_us=%.3f "
                       "write_bytes=%u read_bytes=%u write_MBps=%.3f read_MBps=%.3f status=0x%08x "
                       "combined_MBps=%.3f\n", options->slots[i], windows, active_elapsed, window_elapsed,
                       overhead * 1000000.0, write_bytes[i], read_bytes[i],
                       (double) write_bytes[i] / active_elapsed / 1000000.0,
                       (double) read_bytes[i] / active_elapsed / 1000000.0, status,
                       (double) combined / active_elapsed / 1000000.0) < 0) {
                failed = 1; break;
            }
        }
        if (failed || fflush(stdout) != 0 || ferror(stdout)) { failed = 1; break; }
        for (unsigned i = 0U; i < options->slot_count; ++i) {
            total_write[i] += write_bytes[i];
            total_read[i] += read_bytes[i];
        }
        last_window_end = wall_end;
        measured_active += active_elapsed;
        ++valid_windows;
    }
    if (stop_requested || clock_gettime(CLOCK_MONOTONIC, &session_end) != 0) failed = 1;
    if (!failed) {
        const double total_wall = seconds_between(&session_begin, &session_end);
        if (valid_windows == 0U || !isfinite(measured_active) || measured_active <= 0.0 ||
            !isfinite(total_wall) || total_wall < measured_active ||
            seconds_between(&last_window_end, &session_end) < 0.0) failed = 1;
        if (!failed) {
            uint64_t selected_write = 0U, selected_read = 0U;
            const double dead_seconds = total_wall - measured_active;
            for (unsigned i = 0U; i < options->slot_count; ++i) {
                selected_write += total_write[i];
                selected_read += total_read[i];
                if (printf("AGGREGATE slot=%u valid_windows=%u target_s=%.3f total_wall_s=%.9f active_s=%.9f "
                           "dead_s=%.9f coverage_pct=%.3f write_bytes=%llu read_bytes=%llu write_MBps=%.3f read_MBps=%.3f "
                           "combined_MBps=%.3f\n",
                           options->slots[i], valid_windows, seconds, total_wall, measured_active, dead_seconds,
                           measured_active / total_wall * 100.0,
                           (unsigned long long) total_write[i], (unsigned long long) total_read[i],
                           (double) total_write[i] / measured_active / 1000000.0,
                           (double) total_read[i] / measured_active / 1000000.0,
                           (double) (total_write[i] + total_read[i]) / measured_active / 1000000.0) < 0) {
                    failed = 1; break;
                }
            }
            if (!failed && options->slot_count > 1U &&
                printf("SELECTED_AGGREGATE scope=selected_ports_system_traffic ports=%u active_s=%.9f "
                       "write_bytes=%llu read_bytes=%llu combined_bytes=%llu write_MBps=%.3f read_MBps=%.3f combined_MBps=%.3f\n",
                       options->slot_count, measured_active, (unsigned long long) selected_write,
                       (unsigned long long) selected_read, (unsigned long long) (selected_write + selected_read),
                       (double) selected_write / measured_active / 1000000.0,
                       (double) selected_read / measured_active / 1000000.0,
                       (double) (selected_write + selected_read) / measured_active / 1000000.0) < 0) failed = 1;
            if (fflush(stdout) != 0 || ferror(stdout) || stop_requested) failed = 1;
        }
    }
    if (failed) fprintf(stderr, "Measurement failed or interrupted; discard all session aggregates.\n");
    return failed ? -1 : 0;
}

int main(int argc, char ** argv) {
    struct options options;
    struct mapping mapping = { -1, 0U, 0U, NULL, NULL };
    struct sigaction action;
    int parse_result;
    int result = 1;

    parse_result = parse_options(argc, argv, &options);
    if (parse_result != 0) return parse_result > 0 ? 0 : 2;
    memset(&action, 0, sizeof(action));
    action.sa_handler = SIG_IGN;
    if (sigemptyset(&action.sa_mask) != 0 || sigaction(SIGPIPE, &action, NULL) != 0) {
        fprintf(stderr, "sigaction(SIGPIPE) failed: %s\n", strerror(errno));
        return 2;
    }
    action.sa_handler = on_signal;
    if (sigaction(SIGINT, &action, NULL) != 0 || sigaction(SIGTERM, &action, NULL) != 0) {
        fprintf(stderr, "sigaction failed: %s\n", strerror(errno));
        return 2;
    }
    if (map_apm(options.uio_path, &mapping) != 0) goto done;
    if (check_capabilities(&mapping) != 0 || check_idle(mapping.regs) != 0) goto done;
    save_state(mapping.regs);
    if (run_measurement(mapping.regs, &options) == 0) result = 0;
    restore_state(mapping.regs);
done:
    if (saved.valid) restore_state(mapping.regs);
    unmap_apm(&mapping);
    return result;
}
