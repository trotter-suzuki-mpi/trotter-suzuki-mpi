extern "C"
{
    const char *__nanos_family __attribute__((weak))  = "master";
    int __nanos_version __attribute__((weak))  = 5015;
    int __mcc_master __attribute__((weak))  = 5015;
    int __mcc_openmp __attribute__((weak))  = 5;
    int __mcc_trunk __attribute__((weak))  = 399;
    int __mcc_worksharing __attribute__((weak))  = 1000;
}
extern "C"
{
    typedef unsigned char __u_char;
    typedef unsigned short int __u_short;
    typedef unsigned int __u_int;
    typedef unsigned long int __u_long;
    typedef signed char __int8_t;
    typedef unsigned char __uint8_t;
    typedef signed short int __int16_t;
    typedef unsigned short int __uint16_t;
    typedef signed int __int32_t;
    typedef unsigned int __uint32_t;
    typedef signed long int __int64_t;
    typedef unsigned long int __uint64_t;
    typedef long int __quad_t;
    typedef unsigned long int __u_quad_t;
    typedef unsigned long int __dev_t;
    typedef unsigned int __uid_t;
    typedef unsigned int __gid_t;
    typedef unsigned long int __ino_t;
    typedef unsigned long int __ino64_t;
    typedef unsigned int __mode_t;
    typedef unsigned long int __nlink_t;
    typedef long int __off_t;
    typedef long int __off64_t;
    typedef int __pid_t;
    typedef struct 
    {
            int __val[2];
    } __fsid_t;
    typedef long int __clock_t;
    typedef unsigned long int __rlim_t;
    typedef unsigned long int __rlim64_t;
    typedef unsigned int __id_t;
    typedef long int __time_t;
    typedef unsigned int __useconds_t;
    typedef long int __suseconds_t;
    typedef int __daddr_t;
    typedef long int __swblk_t;
    typedef int __key_t;
    typedef int __clockid_t;
    typedef void *__timer_t;
    typedef long int __blksize_t;
    typedef long int __blkcnt_t;
    typedef long int __blkcnt64_t;
    typedef unsigned long int __fsblkcnt_t;
    typedef unsigned long int __fsblkcnt64_t;
    typedef unsigned long int __fsfilcnt_t;
    typedef unsigned long int __fsfilcnt64_t;
    typedef long int __ssize_t;
    typedef __off64_t __loff_t;
    typedef __quad_t *__qaddr_t;
    typedef char *__caddr_t;
    typedef long int __intptr_t;
    typedef unsigned int __socklen_t;
    typedef __ssize_t ssize_t;
    typedef long unsigned int size_t;
    typedef __gid_t gid_t;
    typedef __uid_t uid_t;
    typedef __off_t off_t;
    typedef __off64_t off64_t;
    typedef __useconds_t useconds_t;
    typedef __pid_t pid_t;
    typedef __intptr_t intptr_t;
    typedef __socklen_t socklen_t;
    extern int access(__const char *__name, int __type) throw () __attribute__((__nonnull__(1)));
    extern int euidaccess(__const char *__name, int __type) throw () __attribute__((__nonnull__(1)));
    extern int eaccess(__const char *__name, int __type) throw () __attribute__((__nonnull__(1)));
    extern int faccessat(int __fd, __const char *__file, int __type, int __flag) throw () __attribute__((__nonnull__(2)));
    extern __off_t lseek(int __fd, __off_t __offset, int __whence) throw ();
    extern __off64_t lseek64(int __fd, __off64_t __offset, int __whence) throw ();
    extern int close(int __fd);
    extern ssize_t read(int __fd, void *__buf, size_t __nbytes);
    extern ssize_t write(int __fd, __const void *__buf, size_t __n);
    extern ssize_t pread(int __fd, void *__buf, size_t __nbytes, __off_t __offset);
    extern ssize_t pwrite(int __fd, __const void *__buf, size_t __n, __off_t __offset);
    extern ssize_t pread64(int __fd, void *__buf, size_t __nbytes, __off64_t __offset);
    extern ssize_t pwrite64(int __fd, __const void *__buf, size_t __n, __off64_t __offset);
    extern int pipe(int __pipedes[2]) throw ();
    extern int pipe2(int __pipedes[2], int __flags) throw ();
    extern unsigned int alarm(unsigned int __seconds) throw ();
    extern unsigned int sleep(unsigned int __seconds);
    extern __useconds_t ualarm(__useconds_t __value, __useconds_t __interval) throw ();
    extern int usleep(__useconds_t __useconds);
    extern int pause(void);
    extern int chown(__const char *__file, __uid_t __owner, __gid_t __group) throw () __attribute__((__nonnull__(1)));
    extern int fchown(int __fd, __uid_t __owner, __gid_t __group) throw ();
    extern int lchown(__const char *__file, __uid_t __owner, __gid_t __group) throw () __attribute__((__nonnull__(1)));
    extern int fchownat(int __fd, __const char *__file, __uid_t __owner, __gid_t __group, int __flag) throw () __attribute__((__nonnull__(2)));
    extern int chdir(__const char *__path) throw () __attribute__((__nonnull__(1)));
    extern int fchdir(int __fd) throw ();
    extern char *getcwd(char *__buf, size_t __size) throw ();
    extern char *get_current_dir_name(void) throw ();
    extern char *getwd(char *__buf) throw () __attribute__((__nonnull__(1))) __attribute__((__deprecated__));
    extern int dup(int __fd) throw ();
    extern int dup2(int __fd, int __fd2) throw ();
    extern int dup3(int __fd, int __fd2, int __flags) throw ();
    extern char **__environ;
    extern char **environ;
    extern int execve(__const char *__path, char *__const __argv[], char *__const __envp[]) throw () __attribute__((__nonnull__(1, 2)));
    extern int fexecve(int __fd, char *__const __argv[], char *__const __envp[]) throw () __attribute__((__nonnull__(2)));
    extern int execv(__const char *__path, char *__const __argv[]) throw () __attribute__((__nonnull__(1, 2)));
    extern int execle(__const char *__path, __const char *__arg, ...) throw () __attribute__((__nonnull__(1, 2)));
    extern int execl(__const char *__path, __const char *__arg, ...) throw () __attribute__((__nonnull__(1, 2)));
    extern int execvp(__const char *__file, char *__const __argv[]) throw () __attribute__((__nonnull__(1, 2)));
    extern int execlp(__const char *__file, __const char *__arg, ...) throw () __attribute__((__nonnull__(1, 2)));
    extern int execvpe(__const char *__file, char *__const __argv[], char *__const __envp[]) throw () __attribute__((__nonnull__(1, 2)));
    extern int nice(int __inc) throw ();
    extern void _exit(int __status) __attribute__((__noreturn__));
    enum 
    {
        _PC_LINK_MAX, 
        _PC_MAX_CANON, 
        _PC_MAX_INPUT, 
        _PC_NAME_MAX, 
        _PC_PATH_MAX, 
        _PC_PIPE_BUF, 
        _PC_CHOWN_RESTRICTED, 
        _PC_NO_TRUNC, 
        _PC_VDISABLE, 
        _PC_SYNC_IO, 
        _PC_ASYNC_IO, 
        _PC_PRIO_IO, 
        _PC_SOCK_MAXBUF, 
        _PC_FILESIZEBITS, 
        _PC_REC_INCR_XFER_SIZE, 
        _PC_REC_MAX_XFER_SIZE, 
        _PC_REC_MIN_XFER_SIZE, 
        _PC_REC_XFER_ALIGN, 
        _PC_ALLOC_SIZE_MIN, 
        _PC_SYMLINK_MAX, 
        _PC_2_SYMLINKS
    };
    enum 
    {
        _SC_ARG_MAX, 
        _SC_CHILD_MAX, 
        _SC_CLK_TCK, 
        _SC_NGROUPS_MAX, 
        _SC_OPEN_MAX, 
        _SC_STREAM_MAX, 
        _SC_TZNAME_MAX, 
        _SC_JOB_CONTROL, 
        _SC_SAVED_IDS, 
        _SC_REALTIME_SIGNALS, 
        _SC_PRIORITY_SCHEDULING, 
        _SC_TIMERS, 
        _SC_ASYNCHRONOUS_IO, 
        _SC_PRIORITIZED_IO, 
        _SC_SYNCHRONIZED_IO, 
        _SC_FSYNC, 
        _SC_MAPPED_FILES, 
        _SC_MEMLOCK, 
        _SC_MEMLOCK_RANGE, 
        _SC_MEMORY_PROTECTION, 
        _SC_MESSAGE_PASSING, 
        _SC_SEMAPHORES, 
        _SC_SHARED_MEMORY_OBJECTS, 
        _SC_AIO_LISTIO_MAX, 
        _SC_AIO_MAX, 
        _SC_AIO_PRIO_DELTA_MAX, 
        _SC_DELAYTIMER_MAX, 
        _SC_MQ_OPEN_MAX, 
        _SC_MQ_PRIO_MAX, 
        _SC_VERSION, 
        _SC_PAGESIZE, 
        _SC_RTSIG_MAX, 
        _SC_SEM_NSEMS_MAX, 
        _SC_SEM_VALUE_MAX, 
        _SC_SIGQUEUE_MAX, 
        _SC_TIMER_MAX, 
        _SC_BC_BASE_MAX, 
        _SC_BC_DIM_MAX, 
        _SC_BC_SCALE_MAX, 
        _SC_BC_STRING_MAX, 
        _SC_COLL_WEIGHTS_MAX, 
        _SC_EQUIV_CLASS_MAX, 
        _SC_EXPR_NEST_MAX, 
        _SC_LINE_MAX, 
        _SC_RE_DUP_MAX, 
        _SC_CHARCLASS_NAME_MAX, 
        _SC_2_VERSION, 
        _SC_2_C_BIND, 
        _SC_2_C_DEV, 
        _SC_2_FORT_DEV, 
        _SC_2_FORT_RUN, 
        _SC_2_SW_DEV, 
        _SC_2_LOCALEDEF, 
        _SC_PII, 
        _SC_PII_XTI, 
        _SC_PII_SOCKET, 
        _SC_PII_INTERNET, 
        _SC_PII_OSI, 
        _SC_POLL, 
        _SC_SELECT, 
        _SC_UIO_MAXIOV, 
        _SC_IOV_MAX = _SC_UIO_MAXIOV, 
        _SC_PII_INTERNET_STREAM, 
        _SC_PII_INTERNET_DGRAM, 
        _SC_PII_OSI_COTS, 
        _SC_PII_OSI_CLTS, 
        _SC_PII_OSI_M, 
        _SC_T_IOV_MAX, 
        _SC_THREADS, 
        _SC_THREAD_SAFE_FUNCTIONS, 
        _SC_GETGR_R_SIZE_MAX, 
        _SC_GETPW_R_SIZE_MAX, 
        _SC_LOGIN_NAME_MAX, 
        _SC_TTY_NAME_MAX, 
        _SC_THREAD_DESTRUCTOR_ITERATIONS, 
        _SC_THREAD_KEYS_MAX, 
        _SC_THREAD_STACK_MIN, 
        _SC_THREAD_THREADS_MAX, 
        _SC_THREAD_ATTR_STACKADDR, 
        _SC_THREAD_ATTR_STACKSIZE, 
        _SC_THREAD_PRIORITY_SCHEDULING, 
        _SC_THREAD_PRIO_INHERIT, 
        _SC_THREAD_PRIO_PROTECT, 
        _SC_THREAD_PROCESS_SHARED, 
        _SC_NPROCESSORS_CONF, 
        _SC_NPROCESSORS_ONLN, 
        _SC_PHYS_PAGES, 
        _SC_AVPHYS_PAGES, 
        _SC_ATEXIT_MAX, 
        _SC_PASS_MAX, 
        _SC_XOPEN_VERSION, 
        _SC_XOPEN_XCU_VERSION, 
        _SC_XOPEN_UNIX, 
        _SC_XOPEN_CRYPT, 
        _SC_XOPEN_ENH_I18N, 
        _SC_XOPEN_SHM, 
        _SC_2_CHAR_TERM, 
        _SC_2_C_VERSION, 
        _SC_2_UPE, 
        _SC_XOPEN_XPG2, 
        _SC_XOPEN_XPG3, 
        _SC_XOPEN_XPG4, 
        _SC_CHAR_BIT, 
        _SC_CHAR_MAX, 
        _SC_CHAR_MIN, 
        _SC_INT_MAX, 
        _SC_INT_MIN, 
        _SC_LONG_BIT, 
        _SC_WORD_BIT, 
        _SC_MB_LEN_MAX, 
        _SC_NZERO, 
        _SC_SSIZE_MAX, 
        _SC_SCHAR_MAX, 
        _SC_SCHAR_MIN, 
        _SC_SHRT_MAX, 
        _SC_SHRT_MIN, 
        _SC_UCHAR_MAX, 
        _SC_UINT_MAX, 
        _SC_ULONG_MAX, 
        _SC_USHRT_MAX, 
        _SC_NL_ARGMAX, 
        _SC_NL_LANGMAX, 
        _SC_NL_MSGMAX, 
        _SC_NL_NMAX, 
        _SC_NL_SETMAX, 
        _SC_NL_TEXTMAX, 
        _SC_XBS5_ILP32_OFF32, 
        _SC_XBS5_ILP32_OFFBIG, 
        _SC_XBS5_LP64_OFF64, 
        _SC_XBS5_LPBIG_OFFBIG, 
        _SC_XOPEN_LEGACY, 
        _SC_XOPEN_REALTIME, 
        _SC_XOPEN_REALTIME_THREADS, 
        _SC_ADVISORY_INFO, 
        _SC_BARRIERS, 
        _SC_BASE, 
        _SC_C_LANG_SUPPORT, 
        _SC_C_LANG_SUPPORT_R, 
        _SC_CLOCK_SELECTION, 
        _SC_CPUTIME, 
        _SC_THREAD_CPUTIME, 
        _SC_DEVICE_IO, 
        _SC_DEVICE_SPECIFIC, 
        _SC_DEVICE_SPECIFIC_R, 
        _SC_FD_MGMT, 
        _SC_FIFO, 
        _SC_PIPE, 
        _SC_FILE_ATTRIBUTES, 
        _SC_FILE_LOCKING, 
        _SC_FILE_SYSTEM, 
        _SC_MONOTONIC_CLOCK, 
        _SC_MULTI_PROCESS, 
        _SC_SINGLE_PROCESS, 
        _SC_NETWORKING, 
        _SC_READER_WRITER_LOCKS, 
        _SC_SPIN_LOCKS, 
        _SC_REGEXP, 
        _SC_REGEX_VERSION, 
        _SC_SHELL, 
        _SC_SIGNALS, 
        _SC_SPAWN, 
        _SC_SPORADIC_SERVER, 
        _SC_THREAD_SPORADIC_SERVER, 
        _SC_SYSTEM_DATABASE, 
        _SC_SYSTEM_DATABASE_R, 
        _SC_TIMEOUTS, 
        _SC_TYPED_MEMORY_OBJECTS, 
        _SC_USER_GROUPS, 
        _SC_USER_GROUPS_R, 
        _SC_2_PBS, 
        _SC_2_PBS_ACCOUNTING, 
        _SC_2_PBS_LOCATE, 
        _SC_2_PBS_MESSAGE, 
        _SC_2_PBS_TRACK, 
        _SC_SYMLOOP_MAX, 
        _SC_STREAMS, 
        _SC_2_PBS_CHECKPOINT, 
        _SC_V6_ILP32_OFF32, 
        _SC_V6_ILP32_OFFBIG, 
        _SC_V6_LP64_OFF64, 
        _SC_V6_LPBIG_OFFBIG, 
        _SC_HOST_NAME_MAX, 
        _SC_TRACE, 
        _SC_TRACE_EVENT_FILTER, 
        _SC_TRACE_INHERIT, 
        _SC_TRACE_LOG, 
        _SC_LEVEL1_ICACHE_SIZE, 
        _SC_LEVEL1_ICACHE_ASSOC, 
        _SC_LEVEL1_ICACHE_LINESIZE, 
        _SC_LEVEL1_DCACHE_SIZE, 
        _SC_LEVEL1_DCACHE_ASSOC, 
        _SC_LEVEL1_DCACHE_LINESIZE, 
        _SC_LEVEL2_CACHE_SIZE, 
        _SC_LEVEL2_CACHE_ASSOC, 
        _SC_LEVEL2_CACHE_LINESIZE, 
        _SC_LEVEL3_CACHE_SIZE, 
        _SC_LEVEL3_CACHE_ASSOC, 
        _SC_LEVEL3_CACHE_LINESIZE, 
        _SC_LEVEL4_CACHE_SIZE, 
        _SC_LEVEL4_CACHE_ASSOC, 
        _SC_LEVEL4_CACHE_LINESIZE, 
        _SC_IPV6 = _SC_LEVEL1_ICACHE_SIZE + 50, 
        _SC_RAW_SOCKETS, 
        _SC_V7_ILP32_OFF32, 
        _SC_V7_ILP32_OFFBIG, 
        _SC_V7_LP64_OFF64, 
        _SC_V7_LPBIG_OFFBIG, 
        _SC_SS_REPL_MAX, 
        _SC_TRACE_EVENT_NAME_MAX, 
        _SC_TRACE_NAME_MAX, 
        _SC_TRACE_SYS_MAX, 
        _SC_TRACE_USER_EVENT_MAX, 
        _SC_XOPEN_STREAMS, 
        _SC_THREAD_ROBUST_PRIO_INHERIT, 
        _SC_THREAD_ROBUST_PRIO_PROTECT
    };
    enum 
    {
        _CS_PATH, 
        _CS_V6_WIDTH_RESTRICTED_ENVS, 
        _CS_GNU_LIBC_VERSION, 
        _CS_GNU_LIBPTHREAD_VERSION, 
        _CS_V5_WIDTH_RESTRICTED_ENVS, 
        _CS_V7_WIDTH_RESTRICTED_ENVS, 
        _CS_LFS_CFLAGS = 1000, 
        _CS_LFS_LDFLAGS, 
        _CS_LFS_LIBS, 
        _CS_LFS_LINTFLAGS, 
        _CS_LFS64_CFLAGS, 
        _CS_LFS64_LDFLAGS, 
        _CS_LFS64_LIBS, 
        _CS_LFS64_LINTFLAGS, 
        _CS_XBS5_ILP32_OFF32_CFLAGS = 1100, 
        _CS_XBS5_ILP32_OFF32_LDFLAGS, 
        _CS_XBS5_ILP32_OFF32_LIBS, 
        _CS_XBS5_ILP32_OFF32_LINTFLAGS, 
        _CS_XBS5_ILP32_OFFBIG_CFLAGS, 
        _CS_XBS5_ILP32_OFFBIG_LDFLAGS, 
        _CS_XBS5_ILP32_OFFBIG_LIBS, 
        _CS_XBS5_ILP32_OFFBIG_LINTFLAGS, 
        _CS_XBS5_LP64_OFF64_CFLAGS, 
        _CS_XBS5_LP64_OFF64_LDFLAGS, 
        _CS_XBS5_LP64_OFF64_LIBS, 
        _CS_XBS5_LP64_OFF64_LINTFLAGS, 
        _CS_XBS5_LPBIG_OFFBIG_CFLAGS, 
        _CS_XBS5_LPBIG_OFFBIG_LDFLAGS, 
        _CS_XBS5_LPBIG_OFFBIG_LIBS, 
        _CS_XBS5_LPBIG_OFFBIG_LINTFLAGS, 
        _CS_POSIX_V6_ILP32_OFF32_CFLAGS, 
        _CS_POSIX_V6_ILP32_OFF32_LDFLAGS, 
        _CS_POSIX_V6_ILP32_OFF32_LIBS, 
        _CS_POSIX_V6_ILP32_OFF32_LINTFLAGS, 
        _CS_POSIX_V6_ILP32_OFFBIG_CFLAGS, 
        _CS_POSIX_V6_ILP32_OFFBIG_LDFLAGS, 
        _CS_POSIX_V6_ILP32_OFFBIG_LIBS, 
        _CS_POSIX_V6_ILP32_OFFBIG_LINTFLAGS, 
        _CS_POSIX_V6_LP64_OFF64_CFLAGS, 
        _CS_POSIX_V6_LP64_OFF64_LDFLAGS, 
        _CS_POSIX_V6_LP64_OFF64_LIBS, 
        _CS_POSIX_V6_LP64_OFF64_LINTFLAGS, 
        _CS_POSIX_V6_LPBIG_OFFBIG_CFLAGS, 
        _CS_POSIX_V6_LPBIG_OFFBIG_LDFLAGS, 
        _CS_POSIX_V6_LPBIG_OFFBIG_LIBS, 
        _CS_POSIX_V6_LPBIG_OFFBIG_LINTFLAGS, 
        _CS_POSIX_V7_ILP32_OFF32_CFLAGS, 
        _CS_POSIX_V7_ILP32_OFF32_LDFLAGS, 
        _CS_POSIX_V7_ILP32_OFF32_LIBS, 
        _CS_POSIX_V7_ILP32_OFF32_LINTFLAGS, 
        _CS_POSIX_V7_ILP32_OFFBIG_CFLAGS, 
        _CS_POSIX_V7_ILP32_OFFBIG_LDFLAGS, 
        _CS_POSIX_V7_ILP32_OFFBIG_LIBS, 
        _CS_POSIX_V7_ILP32_OFFBIG_LINTFLAGS, 
        _CS_POSIX_V7_LP64_OFF64_CFLAGS, 
        _CS_POSIX_V7_LP64_OFF64_LDFLAGS, 
        _CS_POSIX_V7_LP64_OFF64_LIBS, 
        _CS_POSIX_V7_LP64_OFF64_LINTFLAGS, 
        _CS_POSIX_V7_LPBIG_OFFBIG_CFLAGS, 
        _CS_POSIX_V7_LPBIG_OFFBIG_LDFLAGS, 
        _CS_POSIX_V7_LPBIG_OFFBIG_LIBS, 
        _CS_POSIX_V7_LPBIG_OFFBIG_LINTFLAGS, 
        _CS_V6_ENV, 
        _CS_V7_ENV
    };
    extern long int pathconf(__const char *__path, int __name) throw () __attribute__((__nonnull__(1)));
    extern long int fpathconf(int __fd, int __name) throw ();
    extern long int sysconf(int __name) throw ();
    extern size_t confstr(int __name, char *__buf, size_t __len) throw ();
    extern __pid_t getpid(void) throw ();
    extern __pid_t getppid(void) throw ();
    extern __pid_t getpgrp(void) throw ();
    extern __pid_t __getpgid(__pid_t __pid) throw ();
    extern __pid_t getpgid(__pid_t __pid) throw ();
    extern int setpgid(__pid_t __pid, __pid_t __pgid) throw ();
    extern int setpgrp(void) throw ();
    extern __pid_t setsid(void) throw ();
    extern __pid_t getsid(__pid_t __pid) throw ();
    extern __uid_t getuid(void) throw ();
    extern __uid_t geteuid(void) throw ();
    extern __gid_t getgid(void) throw ();
    extern __gid_t getegid(void) throw ();
    extern int getgroups(int __size, __gid_t __list[]) throw ();
    extern int group_member(__gid_t __gid) throw ();
    extern int setuid(__uid_t __uid) throw ();
    extern int setreuid(__uid_t __ruid, __uid_t __euid) throw ();
    extern int seteuid(__uid_t __uid) throw ();
    extern int setgid(__gid_t __gid) throw ();
    extern int setregid(__gid_t __rgid, __gid_t __egid) throw ();
    extern int setegid(__gid_t __gid) throw ();
    extern int getresuid(__uid_t *__ruid, __uid_t *__euid, __uid_t *__suid) throw ();
    extern int getresgid(__gid_t *__rgid, __gid_t *__egid, __gid_t *__sgid) throw ();
    extern int setresuid(__uid_t __ruid, __uid_t __euid, __uid_t __suid) throw ();
    extern int setresgid(__gid_t __rgid, __gid_t __egid, __gid_t __sgid) throw ();
    extern __pid_t fork(void) throw ();
    extern __pid_t vfork(void) throw ();
    extern char *ttyname(int __fd) throw ();
    extern int ttyname_r(int __fd, char *__buf, size_t __buflen) throw () __attribute__((__nonnull__(2)));
    extern int isatty(int __fd) throw ();
    extern int ttyslot(void) throw ();
    extern int link(__const char *__from, __const char *__to) throw () __attribute__((__nonnull__(1, 2)));
    extern int linkat(int __fromfd, __const char *__from, int __tofd, __const char *__to, int __flags) throw () __attribute__((__nonnull__(2, 4)));
    extern int symlink(__const char *__from, __const char *__to) throw () __attribute__((__nonnull__(1, 2)));
    extern ssize_t readlink(__const char *__restrict __path, char *__restrict __buf, size_t __len) throw () __attribute__((__nonnull__(1, 2)));
    extern int symlinkat(__const char *__from, int __tofd, __const char *__to) throw () __attribute__((__nonnull__(1, 3)));
    extern ssize_t readlinkat(int __fd, __const char *__restrict __path, char *__restrict __buf, size_t __len) throw () __attribute__((__nonnull__(2, 3)));
    extern int unlink(__const char *__name) throw () __attribute__((__nonnull__(1)));
    extern int unlinkat(int __fd, __const char *__name, int __flag) throw () __attribute__((__nonnull__(2)));
    extern int rmdir(__const char *__path) throw () __attribute__((__nonnull__(1)));
    extern __pid_t tcgetpgrp(int __fd) throw ();
    extern int tcsetpgrp(int __fd, __pid_t __pgrp_id) throw ();
    extern char *getlogin(void);
    extern int getlogin_r(char *__name, size_t __name_len) __attribute__((__nonnull__(1)));
    extern int setlogin(__const char *__name) throw () __attribute__((__nonnull__(1)));
    extern "C"
    {
        extern char *optarg;
        extern int optind;
        extern int opterr;
        extern int optopt;
        extern int getopt(int ___argc, char *const *___argv, const char *__shortopts) throw ();
    }
    extern int gethostname(char *__name, size_t __len) throw () __attribute__((__nonnull__(1)));
    extern int sethostname(__const char *__name, size_t __len) throw () __attribute__((__nonnull__(1)));
    extern int sethostid(long int __id) throw ();
    extern int getdomainname(char *__name, size_t __len) throw () __attribute__((__nonnull__(1)));
    extern int setdomainname(__const char *__name, size_t __len) throw () __attribute__((__nonnull__(1)));
    extern int vhangup(void) throw ();
    extern int revoke(__const char *__file) throw () __attribute__((__nonnull__(1)));
    extern int profil(unsigned short int *__sample_buffer, size_t __size, size_t __offset, unsigned int __scale) throw () __attribute__((__nonnull__(1)));
    extern int acct(__const char *__name) throw ();
    extern char *getusershell(void) throw ();
    extern void endusershell(void) throw ();
    extern void setusershell(void) throw ();
    extern int daemon(int __nochdir, int __noclose) throw ();
    extern int chroot(__const char *__path) throw () __attribute__((__nonnull__(1)));
    extern char *getpass(__const char *__prompt) __attribute__((__nonnull__(1)));
    extern int fsync(int __fd);
    extern long int gethostid(void);
    extern void sync(void) throw ();
    extern int getpagesize(void) throw () __attribute__((__const__));
    extern int getdtablesize(void) throw ();
    extern int truncate(__const char *__file, __off_t __length) throw () __attribute__((__nonnull__(1)));
    extern int truncate64(__const char *__file, __off64_t __length) throw () __attribute__((__nonnull__(1)));
    extern int ftruncate(int __fd, __off_t __length) throw ();
    extern int ftruncate64(int __fd, __off64_t __length) throw ();
    extern int brk(void *__addr) throw ();
    extern void *sbrk(intptr_t __delta) throw ();
    extern long int syscall(long int __sysno, ...) throw ();
    extern int lockf(int __fd, int __cmd, __off_t __len);
    extern int lockf64(int __fd, int __cmd, __off64_t __len);
    extern int fdatasync(int __fildes);
    extern char *crypt(__const char *__key, __const char *__salt) throw () __attribute__((__nonnull__(1, 2)));
    extern void encrypt(char *__block, int __edflag) throw () __attribute__((__nonnull__(1)));
    extern void swab(__const void *__restrict __from, void *__restrict __to, ssize_t __n) throw () __attribute__((__nonnull__(1, 2)));
    extern char *ctermid(char *__s) throw ();
}
extern "C"
{
    struct _IO_FILE;
    typedef struct _IO_FILE FILE;
    typedef struct _IO_FILE __FILE;
    typedef struct 
    {
            int __count;
            union 
            {
                    unsigned int __wch;
                    char __wchb[4];
            } __value;
    } __mbstate_t;
    typedef struct 
    {
            __off_t __pos;
            __mbstate_t __state;
    } _G_fpos_t;
    typedef struct 
    {
            __off64_t __pos;
            __mbstate_t __state;
    } _G_fpos64_t;
    typedef int _G_int16_t __attribute__((__mode__(__HI__)));
    typedef int _G_int32_t __attribute__((__mode__(__SI__)));
    typedef unsigned int _G_uint16_t __attribute__((__mode__(__HI__)));
    typedef unsigned int _G_uint32_t __attribute__((__mode__(__SI__)));
    typedef __builtin_va_list __gnuc_va_list;
    struct _IO_jump_t;
    struct _IO_FILE;
    typedef void _IO_lock_t;
    struct _IO_marker
    {
            struct _IO_marker *_next;
            struct _IO_FILE *_sbuf;
            int _pos;
    };
    enum __codecvt_result
    {
        __codecvt_ok, 
        __codecvt_partial, 
        __codecvt_error, 
        __codecvt_noconv
    };
    struct _IO_FILE
    {
            int _flags;
            char *_IO_read_ptr;
            char *_IO_read_end;
            char *_IO_read_base;
            char *_IO_write_base;
            char *_IO_write_ptr;
            char *_IO_write_end;
            char *_IO_buf_base;
            char *_IO_buf_end;
            char *_IO_save_base;
            char *_IO_backup_base;
            char *_IO_save_end;
            struct _IO_marker *_markers;
            struct _IO_FILE *_chain;
            int _fileno;
            int _flags2;
            __off_t _old_offset;
            unsigned short _cur_column;
            signed char _vtable_offset;
            char _shortbuf[1];
            _IO_lock_t *_lock;
            __off64_t _offset;
            void *__pad1;
            void *__pad2;
            void *__pad3;
            void *__pad4;
            size_t __pad5;
            int _mode;
            char _unused2[15 * sizeof(int) - 4 * sizeof(void *) - sizeof(size_t)];
    };
    struct _IO_FILE_plus;
    extern struct _IO_FILE_plus _IO_2_1_stdin_;
    extern struct _IO_FILE_plus _IO_2_1_stdout_;
    extern struct _IO_FILE_plus _IO_2_1_stderr_;
    typedef __ssize_t __io_read_fn(void *__cookie, char *__buf, size_t __nbytes);
    typedef __ssize_t __io_write_fn(void *__cookie, __const char *__buf, size_t __n);
    typedef int __io_seek_fn(void *__cookie, __off64_t *__pos, int __w);
    typedef int __io_close_fn(void *__cookie);
    typedef __io_read_fn cookie_read_function_t;
    typedef __io_write_fn cookie_write_function_t;
    typedef __io_seek_fn cookie_seek_function_t;
    typedef __io_close_fn cookie_close_function_t;
    typedef struct 
    {
            __io_read_fn *read;
            __io_write_fn *write;
            __io_seek_fn *seek;
            __io_close_fn *close;
    } _IO_cookie_io_functions_t;
    typedef _IO_cookie_io_functions_t cookie_io_functions_t;
    struct _IO_cookie_file;
    extern void _IO_cookie_init(struct _IO_cookie_file *__cfile, int __read_write, void *__cookie, _IO_cookie_io_functions_t __fns);
    extern "C"
    {
        extern int __underflow(_IO_FILE *);
        extern int __uflow(_IO_FILE *);
        extern int __overflow(_IO_FILE *, int);
        extern int _IO_getc(_IO_FILE *__fp);
        extern int _IO_putc(int __c, _IO_FILE *__fp);
        extern int _IO_feof(_IO_FILE *__fp) throw ();
        extern int _IO_ferror(_IO_FILE *__fp) throw ();
        extern int _IO_peekc_locked(_IO_FILE *__fp);
        extern void _IO_flockfile(_IO_FILE *) throw ();
        extern void _IO_funlockfile(_IO_FILE *) throw ();
        extern int _IO_ftrylockfile(_IO_FILE *) throw ();
        extern int _IO_vfscanf(_IO_FILE *__restrict , const char *__restrict , __gnuc_va_list, int *__restrict );
        extern int _IO_vfprintf(_IO_FILE *__restrict , const char *__restrict , __gnuc_va_list);
        extern __ssize_t _IO_padn(_IO_FILE *, int, __ssize_t);
        extern size_t _IO_sgetn(_IO_FILE *, void *, size_t);
        extern __off64_t _IO_seekoff(_IO_FILE *, __off64_t, int, int);
        extern __off64_t _IO_seekpos(_IO_FILE *, __off64_t, int);
        extern void _IO_free_backup_area(_IO_FILE *) throw ();
    }
    typedef __gnuc_va_list va_list;
    typedef _G_fpos_t fpos_t;
    typedef _G_fpos64_t fpos64_t;
    extern struct _IO_FILE *stdin;
    extern struct _IO_FILE *stdout;
    extern struct _IO_FILE *stderr;
    extern int remove(__const char *__filename) throw ();
    extern int rename(__const char *__old, __const char *__new) throw ();
    extern int renameat(int __oldfd, __const char *__old, int __newfd, __const char *__new) throw ();
    extern FILE *tmpfile(void);
    extern FILE *tmpfile64(void);
    extern char *tmpnam(char *__s) throw ();
    extern char *tmpnam_r(char *__s) throw ();
    extern char *tempnam(__const char *__dir, __const char *__pfx) throw () __attribute__((__malloc__));
    extern int fclose(FILE *__stream);
    extern int fflush(FILE *__stream);
    extern int fflush_unlocked(FILE *__stream);
    extern int fcloseall(void);
    extern FILE *fopen(__const char *__restrict __filename, __const char *__restrict __modes);
    extern FILE *freopen(__const char *__restrict __filename, __const char *__restrict __modes, FILE *__restrict __stream);
    extern FILE *fopen64(__const char *__restrict __filename, __const char *__restrict __modes);
    extern FILE *freopen64(__const char *__restrict __filename, __const char *__restrict __modes, FILE *__restrict __stream);
    extern FILE *fdopen(int __fd, __const char *__modes) throw ();
    extern FILE *fopencookie(void *__restrict __magic_cookie, __const char *__restrict __modes, _IO_cookie_io_functions_t __io_funcs) throw ();
    extern FILE *fmemopen(void *__s, size_t __len, __const char *__modes) throw ();
    extern FILE *open_memstream(char **__bufloc, size_t *__sizeloc) throw ();
    extern void setbuf(FILE *__restrict __stream, char *__restrict __buf) throw ();
    extern int setvbuf(FILE *__restrict __stream, char *__restrict __buf, int __modes, size_t __n) throw ();
    extern void setbuffer(FILE *__restrict __stream, char *__restrict __buf, size_t __size) throw ();
    extern void setlinebuf(FILE *__stream) throw ();
    extern int fprintf(FILE *__restrict __stream, __const char *__restrict __format, ...);
    extern int printf(__const char *__restrict __format, ...);
    extern int sprintf(char *__restrict __s, __const char *__restrict __format, ...) throw ();
    extern int vfprintf(FILE *__restrict __s, __const char *__restrict __format, __gnuc_va_list __arg);
    extern int vprintf(__const char *__restrict __format, __gnuc_va_list __arg);
    extern int vsprintf(char *__restrict __s, __const char *__restrict __format, __gnuc_va_list __arg) throw ();
    extern int snprintf(char *__restrict __s, size_t __maxlen, __const char *__restrict __format, ...) throw () __attribute__((__format__(__printf__, 3, 4)));
    extern int vsnprintf(char *__restrict __s, size_t __maxlen, __const char *__restrict __format, __gnuc_va_list __arg) throw () __attribute__((__format__(__printf__, 3, 0)));
    extern int vasprintf(char **__restrict __ptr, __const char *__restrict __f, __gnuc_va_list __arg) throw () __attribute__((__format__(__printf__, 2, 0)));
    extern int __asprintf(char **__restrict __ptr, __const char *__restrict __fmt, ...) throw () __attribute__((__format__(__printf__, 2, 3)));
    extern int asprintf(char **__restrict __ptr, __const char *__restrict __fmt, ...) throw () __attribute__((__format__(__printf__, 2, 3)));
    extern int vdprintf(int __fd, __const char *__restrict __fmt, __gnuc_va_list __arg) __attribute__((__format__(__printf__, 2, 0)));
    extern int dprintf(int __fd, __const char *__restrict __fmt, ...) __attribute__((__format__(__printf__, 2, 3)));
    extern int fscanf(FILE *__restrict __stream, __const char *__restrict __format, ...);
    extern int scanf(__const char *__restrict __format, ...);
    extern int sscanf(__const char *__restrict __s, __const char *__restrict __format, ...) throw ();
    extern int vfscanf(FILE *__restrict __s, __const char *__restrict __format, __gnuc_va_list __arg) __attribute__((__format__(__scanf__, 2, 0)));
    extern int vscanf(__const char *__restrict __format, __gnuc_va_list __arg) __attribute__((__format__(__scanf__, 1, 0)));
    extern int vsscanf(__const char *__restrict __s, __const char *__restrict __format, __gnuc_va_list __arg) throw () __attribute__((__format__(__scanf__, 2, 0)));
    extern int fgetc(FILE *__stream);
    extern int getc(FILE *__stream);
    extern int getchar(void);
    extern int getc_unlocked(FILE *__stream);
    extern int getchar_unlocked(void);
    extern int fgetc_unlocked(FILE *__stream);
    extern int fputc(int __c, FILE *__stream);
    extern int putc(int __c, FILE *__stream);
    extern int putchar(int __c);
    extern int fputc_unlocked(int __c, FILE *__stream);
    extern int putc_unlocked(int __c, FILE *__stream);
    extern int putchar_unlocked(int __c);
    extern int getw(FILE *__stream);
    extern int putw(int __w, FILE *__stream);
    extern char *fgets(char *__restrict __s, int __n, FILE *__restrict __stream);
    extern char *gets(char *__s);
    extern char *fgets_unlocked(char *__restrict __s, int __n, FILE *__restrict __stream);
    extern __ssize_t __getdelim(char **__restrict __lineptr, size_t *__restrict __n, int __delimiter, FILE *__restrict __stream);
    extern __ssize_t getdelim(char **__restrict __lineptr, size_t *__restrict __n, int __delimiter, FILE *__restrict __stream);
    extern __ssize_t getline(char **__restrict __lineptr, size_t *__restrict __n, FILE *__restrict __stream);
    extern int fputs(__const char *__restrict __s, FILE *__restrict __stream);
    extern int puts(__const char *__s);
    extern int ungetc(int __c, FILE *__stream);
    extern size_t fread(void *__restrict __ptr, size_t __size, size_t __n, FILE *__restrict __stream);
    extern size_t fwrite(__const void *__restrict __ptr, size_t __size, size_t __n, FILE *__restrict __s);
    extern int fputs_unlocked(__const char *__restrict __s, FILE *__restrict __stream);
    extern size_t fread_unlocked(void *__restrict __ptr, size_t __size, size_t __n, FILE *__restrict __stream);
    extern size_t fwrite_unlocked(__const void *__restrict __ptr, size_t __size, size_t __n, FILE *__restrict __stream);
    extern int fseek(FILE *__stream, long int __off, int __whence);
    extern long int ftell(FILE *__stream);
    extern void rewind(FILE *__stream);
    extern int fseeko(FILE *__stream, __off_t __off, int __whence);
    extern __off_t ftello(FILE *__stream);
    extern int fgetpos(FILE *__restrict __stream, fpos_t *__restrict __pos);
    extern int fsetpos(FILE *__stream, __const fpos_t *__pos);
    extern int fseeko64(FILE *__stream, __off64_t __off, int __whence);
    extern __off64_t ftello64(FILE *__stream);
    extern int fgetpos64(FILE *__restrict __stream, fpos64_t *__restrict __pos);
    extern int fsetpos64(FILE *__stream, __const fpos64_t *__pos);
    extern void clearerr(FILE *__stream) throw ();
    extern int feof(FILE *__stream) throw ();
    extern int ferror(FILE *__stream) throw ();
    extern void clearerr_unlocked(FILE *__stream) throw ();
    extern int feof_unlocked(FILE *__stream) throw ();
    extern int ferror_unlocked(FILE *__stream) throw ();
    extern void perror(__const char *__s);
    extern int sys_nerr;
    extern __const char *__const sys_errlist[];
    extern int _sys_nerr;
    extern __const char *__const _sys_errlist[];
    extern int fileno(FILE *__stream) throw ();
    extern int fileno_unlocked(FILE *__stream) throw ();
    extern FILE *popen(__const char *__command, __const char *__modes);
    extern int pclose(FILE *__stream);
    extern char *ctermid(char *__s) throw ();
    extern char *cuserid(char *__s);
    struct obstack;
    extern int obstack_printf(struct obstack *__restrict __obstack, __const char *__restrict __format, ...) throw () __attribute__((__format__(__printf__, 2, 3)));
    extern int obstack_vprintf(struct obstack *__restrict __obstack, __const char *__restrict __format, __gnuc_va_list __args) throw () __attribute__((__format__(__printf__, 2, 0)));
    extern void flockfile(FILE *__stream) throw ();
    extern int ftrylockfile(FILE *__stream) throw ();
    extern void funlockfile(FILE *__stream) throw ();
    extern __inline __attribute__((__gnu_inline__)) int vprintf(__const char *__restrict __fmt, __gnuc_va_list __arg)
    {
        return vfprintf(stdout, __fmt, __arg);
    }
    extern __inline __attribute__((__gnu_inline__)) int getchar(void)
    {
        return _IO_getc(stdin);
    }
    extern __inline __attribute__((__gnu_inline__)) int fgetc_unlocked(FILE *__fp)
    {
        return (__builtin_expect(((__fp)->_IO_read_ptr >= (__fp)->_IO_read_end), 0) ? __uflow(__fp) : *(unsigned char *) (__fp)->_IO_read_ptr++);
    }
    extern __inline __attribute__((__gnu_inline__)) int getc_unlocked(FILE *__fp)
    {
        return (__builtin_expect(((__fp)->_IO_read_ptr >= (__fp)->_IO_read_end), 0) ? __uflow(__fp) : *(unsigned char *) (__fp)->_IO_read_ptr++);
    }
    extern __inline __attribute__((__gnu_inline__)) int getchar_unlocked(void)
    {
        return (__builtin_expect(((stdin)->_IO_read_ptr >= (stdin)->_IO_read_end), 0) ? __uflow(stdin) : *(unsigned char *) (stdin)->_IO_read_ptr++);
    }
    extern __inline __attribute__((__gnu_inline__)) int putchar(int __c)
    {
        return _IO_putc(__c, stdout);
    }
    extern __inline __attribute__((__gnu_inline__)) int fputc_unlocked(int __c, FILE *__stream)
    {
        return (__builtin_expect(((__stream)->_IO_write_ptr >= (__stream)->_IO_write_end), 0) ? __overflow(__stream, (unsigned char) (__c)) : (unsigned char) (*(__stream)->_IO_write_ptr++ = (__c)));
    }
    extern __inline __attribute__((__gnu_inline__)) int putc_unlocked(int __c, FILE *__stream)
    {
        return (__builtin_expect(((__stream)->_IO_write_ptr >= (__stream)->_IO_write_end), 0) ? __overflow(__stream, (unsigned char) (__c)) : (unsigned char) (*(__stream)->_IO_write_ptr++ = (__c)));
    }
    extern __inline __attribute__((__gnu_inline__)) int putchar_unlocked(int __c)
    {
        return (__builtin_expect(((stdout)->_IO_write_ptr >= (stdout)->_IO_write_end), 0) ? __overflow(stdout, (unsigned char) (__c)) : (unsigned char) (*(stdout)->_IO_write_ptr++ = (__c)));
    }
    extern __inline __attribute__((__gnu_inline__)) __ssize_t getline(char **__lineptr, size_t *__n, FILE *__stream)
    {
        return __getdelim(__lineptr, __n, '\n', __stream);
    }
    extern __inline __attribute__((__gnu_inline__)) int feof_unlocked(FILE *__stream) throw ()
    {
        return (((__stream)->_flags & 0x10) != 0);
    }
    extern __inline __attribute__((__gnu_inline__)) int ferror_unlocked(FILE *__stream) throw ()
    {
        return (((__stream)->_flags & 0x20) != 0);
    }
}
typedef signed char int8_t;
typedef short int int16_t;
typedef int int32_t;
typedef long int int64_t;
typedef unsigned char uint8_t;
typedef unsigned short int uint16_t;
typedef unsigned int uint32_t;
typedef unsigned long int uint64_t;
typedef signed char int_least8_t;
typedef short int int_least16_t;
typedef int int_least32_t;
typedef long int int_least64_t;
typedef unsigned char uint_least8_t;
typedef unsigned short int uint_least16_t;
typedef unsigned int uint_least32_t;
typedef unsigned long int uint_least64_t;
typedef signed char int_fast8_t;
typedef long int int_fast16_t;
typedef long int int_fast32_t;
typedef long int int_fast64_t;
typedef unsigned char uint_fast8_t;
typedef unsigned long int uint_fast16_t;
typedef unsigned long int uint_fast32_t;
typedef unsigned long int uint_fast64_t;
typedef unsigned long int uintptr_t;
typedef long int intmax_t;
typedef unsigned long int uintmax_t;
typedef long int ptrdiff_t;
typedef struct 
{
        void **address;
        ptrdiff_t offset;
        struct 
        {
                bool input : 1;
                bool output : 1;
                bool can_rename : 1;
                bool commutative : 1;
        } flags;
        size_t size;
} nanos_dependence_internal_t;
typedef enum 
{
    NANOS_PRIVATE, 
    NANOS_SHARED
} nanos_sharing_t;
typedef struct 
{
        void *original;
        void *privates;
        void (*bop)(void *, void *);
        void (*vop)(int n, void *, void *);
        void (*cleanup)(void *);
} nanos_reduction_t;
typedef struct 
{
        uint64_t address;
        nanos_sharing_t sharing;
        struct 
        {
                bool input : 1;
                bool output : 1;
        } flags;
        size_t size;
} nanos_copy_data_internal_t;
typedef nanos_dependence_internal_t nanos_dependence_t;
typedef nanos_copy_data_internal_t nanos_copy_data_t;
typedef void *nanos_thread_t;
typedef void *nanos_wd_t;
typedef struct 
{
        int nsect;
        nanos_wd_t lwd[];
} nanos_compound_wd_data_t;
typedef struct 
{
        int n;
} nanos_repeat_n_info_t;
typedef struct 
{
        int lower;
        int upper;
        int step;
        bool last;
        int chunk;
        int stride;
        int thid;
        void *args;
} nanos_loop_info_t;
typedef void *nanos_ws_t;
typedef void *nanos_ws_info_t;
typedef void *nanos_ws_data_t;
typedef void *nanos_ws_item_t;
typedef struct 
{
        int lower_bound;
        int upper_bound;
        int loop_step;
        int chunk_size;
} nanos_ws_info_loop_t;
typedef struct 
{
        int lower;
        int upper;
        bool execute : 1;
        bool last : 1;
} nanos_ws_item_loop_t;
typedef struct nanos_ws_desc
{
        volatile nanos_ws_t ws;
        nanos_ws_data_t data;
        struct nanos_ws_desc *next;
        nanos_thread_t *threads;
        int nths;
} nanos_ws_desc_t;
typedef struct 
{
        bool mandatory_creation : 1;
        bool tied : 1;
        bool reserved0 : 1;
        bool reserved1 : 1;
        bool reserved2 : 1;
        bool reserved3 : 1;
        bool reserved4 : 1;
        bool reserved5 : 1;
} nanos_wd_props_t;
typedef struct 
{
        nanos_thread_t tie_to;
        unsigned int priority;
} nanos_wd_dyn_props_t;
typedef struct 
{
        void *(*factory)(void *arg);
        void *arg;
} nanos_device_t;
typedef enum 
{
    NANOS_STATE_START, 
    NANOS_STATE_END, 
    NANOS_SUBSTATE_START, 
    NANOS_SUBSTATE_END, 
    NANOS_BURST_START, 
    NANOS_BURST_END, 
    NANOS_PTP_START, 
    NANOS_PTP_END, 
    NANOS_POINT, 
    EVENT_TYPES
} nanos_event_type_t;
typedef enum 
{
    NANOS_NOT_CREATED, 
    NANOS_NOT_RUNNING, 
    NANOS_STARTUP, 
    NANOS_SHUTDOWN, 
    NANOS_ERROR, 
    NANOS_IDLE, 
    NANOS_RUNTIME, 
    NANOS_RUNNING, 
    NANOS_SYNCHRONIZATION, 
    NANOS_SCHEDULING, 
    NANOS_CREATION, 
    NANOS_MEM_TRANSFER_IN, 
    NANOS_MEM_TRANSFER_OUT, 
    NANOS_MEM_TRANSFER_LOCAL, 
    NANOS_MEM_TRANSFER_DEVICE_IN, 
    NANOS_MEM_TRANSFER_DEVICE_OUT, 
    NANOS_MEM_TRANSFER_DEVICE_LOCAL, 
    NANOS_CACHE, 
    NANOS_YIELD, 
    NANOS_ACQUIRING_LOCK, 
    NANOS_CONTEXT_SWITCH, 
    NANOS_DEBUG, 
    NANOS_EVENT_STATE_TYPES
} nanos_event_state_value_t;
typedef enum 
{
    NANOS_WD_DOMAIN, 
    NANOS_WD_DEPENDENCY, 
    NANOS_WAIT, 
    NANOS_WD_REMOTE, 
    NANOS_XFER_PUT, 
    NANOS_XFER_GET
} nanos_event_domain_t;
typedef long long nanos_event_id_t;
typedef unsigned int nanos_event_key_t;
typedef unsigned long long nanos_event_value_t;
typedef struct 
{
        nanos_event_key_t key;
        nanos_event_value_t value;
} nanos_event_burst_t;
typedef struct 
{
        nanos_event_state_value_t value;
} nanos_event_state_t;
typedef struct 
{
        unsigned int nkvs;
        nanos_event_key_t *keys;
        nanos_event_value_t *values;
} nanos_event_point_t;
typedef struct 
{
        nanos_event_domain_t domain;
        nanos_event_id_t id;
        unsigned int nkvs;
        nanos_event_key_t *keys;
        nanos_event_value_t *values;
} nanos_event_ptp_t;
typedef struct 
{
        nanos_event_type_t type;
        union 
        {
                nanos_event_burst_t burst;
                nanos_event_state_t state;
                nanos_event_point_t point;
                nanos_event_ptp_t ptp;
        } info;
} nanos_event_t;
typedef enum 
{
    NANOS_LOCK_FREE = 0, 
    NANOS_LOCK_BUSY = 1
} nanos_lock_state_t;
typedef struct nanos_lock_t
{
        volatile nanos_lock_state_t _state;
        nanos_lock_t(nanos_lock_state_t init = NANOS_LOCK_FREE)
            : _state(init) 
        {
        }
} nanos_lock_t;
typedef void (*nanos_translate_args_t)(void *, nanos_wd_t);
typedef void (nanos_init_func_t)(void *);
typedef struct 
{
        nanos_init_func_t *func;
        void *data;
} nanos_init_desc_t;
typedef void *nanos_wg_t;
typedef void *nanos_team_t;
typedef void *nanos_sched_t;
typedef void *nanos_slicer_t;
typedef void *nanos_dd_t;
typedef void *nanos_sync_cond_t;
typedef unsigned int nanos_copy_id_t;
typedef struct nanos_const_wd_definition_tag
{
        nanos_wd_props_t props;
        size_t data_alignment;
        size_t num_copies;
        size_t num_devices;
} nanos_const_wd_definition_t;
typedef struct 
{
        int nthreads;
        void *arch;
} nanos_constraint_t;
typedef enum 
{
    NANOS_OK = 0, 
    NANOS_UNKNOWN_ERR, 
    NANOS_UNIMPLEMENTED
} nanos_err_t;
typedef struct 
{
        void (*outline)(void *);
} nanos_smp_args_t;
extern "C"
{
    extern nanos_wd_t nanos_current_wd_(void);
    extern nanos_wd_t nanos_current_wd(void);
    extern int nanos_get_wd_id_(nanos_wd_t wd);
    extern int nanos_get_wd_id(nanos_wd_t wd);
    extern nanos_slicer_t nanos_find_slicer_(const char *slicer);
    extern nanos_slicer_t nanos_find_slicer(const char *slicer);
    extern nanos_ws_t nanos_find_worksharing_(const char *label);
    extern nanos_ws_t nanos_find_worksharing(const char *label);
    extern nanos_err_t nanos_create_wd_compact_(nanos_wd_t *wd, nanos_const_wd_definition_t *const_data, nanos_wd_dyn_props_t *dyn_props, size_t data_size, void **data, nanos_wg_t wg, nanos_copy_data_t **copies);
    extern nanos_err_t nanos_create_wd_compact(nanos_wd_t *wd, nanos_const_wd_definition_t *const_data, nanos_wd_dyn_props_t *dyn_props, size_t data_size, void **data, nanos_wg_t wg, nanos_copy_data_t **copies);
    extern nanos_err_t nanos_set_translate_function_(nanos_wd_t wd, nanos_translate_args_t translate_args);
    extern nanos_err_t nanos_set_translate_function(nanos_wd_t wd, nanos_translate_args_t translate_args);
    extern nanos_err_t nanos_create_sliced_wd_(nanos_wd_t *uwd, size_t num_devices, nanos_device_t *devices, size_t outline_data_size, int outline_data_align, void **outline_data, nanos_wg_t uwg, nanos_slicer_t slicer, nanos_wd_props_t *props, nanos_wd_dyn_props_t *dyn_props, size_t num_copies, nanos_copy_data_t **copies);
    extern nanos_err_t nanos_create_sliced_wd(nanos_wd_t *uwd, size_t num_devices, nanos_device_t *devices, size_t outline_data_size, int outline_data_align, void **outline_data, nanos_wg_t uwg, nanos_slicer_t slicer, nanos_wd_props_t *props, nanos_wd_dyn_props_t *dyn_props, size_t num_copies, nanos_copy_data_t **copies);
    extern nanos_err_t nanos_submit_(nanos_wd_t wd, size_t num_deps, nanos_dependence_t *deps, nanos_team_t team);
    extern nanos_err_t nanos_submit(nanos_wd_t wd, size_t num_deps, nanos_dependence_t *deps, nanos_team_t team);
    extern nanos_err_t nanos_create_wd_and_run_compact_(nanos_const_wd_definition_t *const_data, nanos_wd_dyn_props_t *dyn_props, size_t data_size, void *data, size_t num_deps, nanos_dependence_t *deps, nanos_copy_data_t *copies, nanos_translate_args_t translate_args);
    extern nanos_err_t nanos_create_wd_and_run_compact(nanos_const_wd_definition_t *const_data, nanos_wd_dyn_props_t *dyn_props, size_t data_size, void *data, size_t num_deps, nanos_dependence_t *deps, nanos_copy_data_t *copies, nanos_translate_args_t translate_args);
    extern nanos_err_t nanos_create_for_(void);
    extern nanos_err_t nanos_create_for(void);
    extern nanos_err_t nanos_set_internal_wd_data_(nanos_wd_t wd, void *data);
    extern nanos_err_t nanos_set_internal_wd_data(nanos_wd_t wd, void *data);
    extern nanos_err_t nanos_get_internal_wd_data_(nanos_wd_t wd, void **data);
    extern nanos_err_t nanos_get_internal_wd_data(nanos_wd_t wd, void **data);
    extern nanos_err_t nanos_yield_(void);
    extern nanos_err_t nanos_yield(void);
    extern nanos_err_t nanos_slicer_get_specific_data_(nanos_slicer_t slicer, void **data);
    extern nanos_err_t nanos_slicer_get_specific_data(nanos_slicer_t slicer, void **data);
    extern nanos_err_t nanos_create_team_(nanos_team_t *team, nanos_sched_t sg, unsigned int *nthreads, nanos_constraint_t *constraints, bool reuse, nanos_thread_t *info);
    extern nanos_err_t nanos_create_team(nanos_team_t *team, nanos_sched_t sg, unsigned int *nthreads, nanos_constraint_t *constraints, bool reuse, nanos_thread_t *info);
    extern nanos_err_t nanos_create_team_mapped_(nanos_team_t *team, nanos_sched_t sg, unsigned int *nthreads, unsigned int *mapping);
    extern nanos_err_t nanos_create_team_mapped(nanos_team_t *team, nanos_sched_t sg, unsigned int *nthreads, unsigned int *mapping);
    extern nanos_err_t nanos_leave_team_();
    extern nanos_err_t nanos_leave_team();
    extern nanos_err_t nanos_end_team_(nanos_team_t team);
    extern nanos_err_t nanos_end_team(nanos_team_t team);
    extern nanos_err_t nanos_team_barrier_(void);
    extern nanos_err_t nanos_team_barrier(void);
    extern nanos_err_t nanos_single_guard_(bool *);
    extern nanos_err_t nanos_single_guard(bool *);
    extern nanos_err_t nanos_enter_sync_init_(bool *b);
    extern nanos_err_t nanos_enter_sync_init(bool *b);
    extern nanos_err_t nanos_wait_sync_init_(void);
    extern nanos_err_t nanos_wait_sync_init(void);
    extern nanos_err_t nanos_release_sync_init_(void);
    extern nanos_err_t nanos_release_sync_init(void);
    extern nanos_err_t nanos_team_get_num_starring_threads_(int *n);
    extern nanos_err_t nanos_team_get_num_starring_threads(int *n);
    extern nanos_err_t nanos_team_get_starring_threads_(int *n, nanos_thread_t *list_of_threads);
    extern nanos_err_t nanos_team_get_starring_threads(int *n, nanos_thread_t *list_of_threads);
    extern nanos_err_t nanos_team_get_num_supporting_threads_(int *n);
    extern nanos_err_t nanos_team_get_num_supporting_threads(int *n);
    extern nanos_err_t nanos_team_get_supporting_threads_(int *n, nanos_thread_t *list_of_threads);
    extern nanos_err_t nanos_team_get_supporting_threads(int *n, nanos_thread_t *list_of_threads);
    extern nanos_err_t nanos_register_reduction_(nanos_reduction_t *red);
    extern nanos_err_t nanos_register_reduction(nanos_reduction_t *red);
    extern nanos_err_t nanos_reduction_get_private_data_(void **copy, void *sink);
    extern nanos_err_t nanos_reduction_get_private_data(void **copy, void *sink);
    extern nanos_err_t nanos_worksharing_create_(nanos_ws_desc_t **wsd, nanos_ws_t ws, nanos_ws_info_t *info, bool *b);
    extern nanos_err_t nanos_worksharing_create(nanos_ws_desc_t **wsd, nanos_ws_t ws, nanos_ws_info_t *info, bool *b);
    extern nanos_err_t nanos_worksharing_next_item_(nanos_ws_desc_t *wsd, nanos_ws_item_t *wsi);
    extern nanos_err_t nanos_worksharing_next_item(nanos_ws_desc_t *wsd, nanos_ws_item_t *wsi);
    extern nanos_err_t nanos_wg_wait_completion_(nanos_wg_t wg, bool avoid_flush);
    extern nanos_err_t nanos_wg_wait_completion(nanos_wg_t wg, bool avoid_flush);
    extern nanos_err_t nanos_create_int_sync_cond_(nanos_sync_cond_t *sync_cond, volatile int *p, int condition);
    extern nanos_err_t nanos_create_int_sync_cond(nanos_sync_cond_t *sync_cond, volatile int *p, int condition);
    extern nanos_err_t nanos_create_bool_sync_cond_(nanos_sync_cond_t *sync_cond, volatile bool *p, bool condition);
    extern nanos_err_t nanos_create_bool_sync_cond(nanos_sync_cond_t *sync_cond, volatile bool *p, bool condition);
    extern nanos_err_t nanos_sync_cond_wait_(nanos_sync_cond_t sync_cond);
    extern nanos_err_t nanos_sync_cond_wait(nanos_sync_cond_t sync_cond);
    extern nanos_err_t nanos_sync_cond_signal_(nanos_sync_cond_t sync_cond);
    extern nanos_err_t nanos_sync_cond_signal(nanos_sync_cond_t sync_cond);
    extern nanos_err_t nanos_destroy_sync_cond_(nanos_sync_cond_t sync_cond);
    extern nanos_err_t nanos_destroy_sync_cond(nanos_sync_cond_t sync_cond);
    extern nanos_err_t nanos_wait_on_(size_t num_deps, nanos_dependence_t *deps);
    extern nanos_err_t nanos_wait_on(size_t num_deps, nanos_dependence_t *deps);
    extern nanos_err_t nanos_init_lock_(nanos_lock_t **lock);
    extern nanos_err_t nanos_init_lock(nanos_lock_t **lock);
    extern nanos_err_t nanos_set_lock_(nanos_lock_t *lock);
    extern nanos_err_t nanos_set_lock(nanos_lock_t *lock);
    extern nanos_err_t nanos_unset_lock_(nanos_lock_t *lock);
    extern nanos_err_t nanos_unset_lock(nanos_lock_t *lock);
    extern nanos_err_t nanos_try_lock_(nanos_lock_t *lock, bool *result);
    extern nanos_err_t nanos_try_lock(nanos_lock_t *lock, bool *result);
    extern nanos_err_t nanos_destroy_lock_(nanos_lock_t *lock);
    extern nanos_err_t nanos_destroy_lock(nanos_lock_t *lock);
    extern nanos_err_t nanos_get_addr_(nanos_copy_id_t copy_id, void **addr, nanos_wd_t cwd);
    extern nanos_err_t nanos_get_addr(nanos_copy_id_t copy_id, void **addr, nanos_wd_t cwd);
    extern nanos_err_t nanos_copy_value_(void *dst, nanos_copy_id_t copy_id, nanos_wd_t cwd);
    extern nanos_err_t nanos_copy_value(void *dst, nanos_copy_id_t copy_id, nanos_wd_t cwd);
    extern nanos_err_t nanos_get_num_running_tasks_(int *num);
    extern nanos_err_t nanos_get_num_running_tasks(int *num);
    extern nanos_err_t nanos_start_scheduler_();
    extern nanos_err_t nanos_start_scheduler();
    extern nanos_err_t nanos_stop_scheduler_();
    extern nanos_err_t nanos_stop_scheduler();
    extern nanos_err_t nanos_scheduler_enabled_(bool *res);
    extern nanos_err_t nanos_scheduler_enabled(bool *res);
    extern nanos_err_t nanos_wait_until_threads_paused_();
    extern nanos_err_t nanos_wait_until_threads_paused();
    extern nanos_err_t nanos_wait_until_threads_unpaused_();
    extern nanos_err_t nanos_wait_until_threads_unpaused();
    extern nanos_err_t nanos_delay_start_();
    extern nanos_err_t nanos_delay_start();
    extern nanos_err_t nanos_start_();
    extern nanos_err_t nanos_start();
    extern nanos_err_t nanos_finish_();
    extern nanos_err_t nanos_finish();
    extern nanos_err_t nanos_malloc_(void **p, size_t size, const char *file, int line);
    extern nanos_err_t nanos_malloc(void **p, size_t size, const char *file, int line);
    extern nanos_err_t nanos_free_(void *p);
    extern nanos_err_t nanos_free(void *p);
    extern void nanos_handle_error_(nanos_err_t err);
    extern void nanos_handle_error(nanos_err_t err);
    extern void *nanos_smp_factory_(void *args);
    extern void *nanos_smp_factory(void *args);
    extern nanos_err_t nanos_instrument_register_key_(nanos_event_key_t *event_key, const char *key, const char *description, bool abort_when_registered);
    extern nanos_err_t nanos_instrument_register_key(nanos_event_key_t *event_key, const char *key, const char *description, bool abort_when_registered);
    extern nanos_err_t nanos_instrument_register_value_(nanos_event_value_t *event_value, const char *key, const char *value, const char *description, bool abort_when_registered);
    extern nanos_err_t nanos_instrument_register_value(nanos_event_value_t *event_value, const char *key, const char *value, const char *description, bool abort_when_registered);
    extern nanos_err_t nanos_instrument_register_value_with_val_(nanos_event_value_t val, const char *key, const char *value, const char *description, bool abort_when_registered);
    extern nanos_err_t nanos_instrument_register_value_with_val(nanos_event_value_t val, const char *key, const char *value, const char *description, bool abort_when_registered);
    extern nanos_err_t nanos_instrument_get_key_(const char *key, nanos_event_key_t *event_key);
    extern nanos_err_t nanos_instrument_get_key(const char *key, nanos_event_key_t *event_key);
    extern nanos_err_t nanos_instrument_get_value_(const char *key, const char *value, nanos_event_value_t *event_value);
    extern nanos_err_t nanos_instrument_get_value(const char *key, const char *value, nanos_event_value_t *event_value);
    extern nanos_err_t nanos_instrument_events_(unsigned int num_events, nanos_event_t events[]);
    extern nanos_err_t nanos_instrument_events(unsigned int num_events, nanos_event_t events[]);
    extern nanos_err_t nanos_instrument_enter_state_(nanos_event_state_value_t state);
    extern nanos_err_t nanos_instrument_enter_state(nanos_event_state_value_t state);
    extern nanos_err_t nanos_instrument_leave_state_(void);
    extern nanos_err_t nanos_instrument_leave_state(void);
    extern nanos_err_t nanos_instrument_enter_burst_(nanos_event_key_t key, nanos_event_value_t value);
    extern nanos_err_t nanos_instrument_enter_burst(nanos_event_key_t key, nanos_event_value_t value);
    extern nanos_err_t nanos_instrument_leave_burst_(nanos_event_key_t key);
    extern nanos_err_t nanos_instrument_leave_burst(nanos_event_key_t key);
    extern nanos_err_t nanos_instrument_point_event_(unsigned int nkvs, nanos_event_key_t *keys, nanos_event_value_t *values);
    extern nanos_err_t nanos_instrument_point_event(unsigned int nkvs, nanos_event_key_t *keys, nanos_event_value_t *values);
    extern nanos_err_t nanos_instrument_ptp_start_(nanos_event_domain_t domain, nanos_event_id_t id, unsigned int nkvs, nanos_event_key_t *keys, nanos_event_value_t *values);
    extern nanos_err_t nanos_instrument_ptp_start(nanos_event_domain_t domain, nanos_event_id_t id, unsigned int nkvs, nanos_event_key_t *keys, nanos_event_value_t *values);
    extern nanos_err_t nanos_instrument_ptp_end_(nanos_event_domain_t domain, nanos_event_id_t id, unsigned int nkvs, nanos_event_key_t *keys, nanos_event_value_t *values);
    extern nanos_err_t nanos_instrument_ptp_end(nanos_event_domain_t domain, nanos_event_id_t id, unsigned int nkvs, nanos_event_key_t *keys, nanos_event_value_t *values);
    extern nanos_err_t nanos_instrument_disable_state_events_(nanos_event_state_value_t state);
    extern nanos_err_t nanos_instrument_disable_state_events(nanos_event_state_value_t state);
    extern nanos_err_t nanos_instrument_enable_state_events_(void);
    extern nanos_err_t nanos_instrument_enable_state_events(void);
    extern nanos_err_t nanos_instrument_close_user_fun_event_();
    extern nanos_err_t nanos_instrument_close_user_fun_event();
    extern nanos_err_t nanos_instrument_enable_(void);
    extern nanos_err_t nanos_instrument_enable(void);
    extern nanos_err_t nanos_instrument_disable_(void);
    extern nanos_err_t nanos_instrument_disable(void);
    void nanos_reduction_int_vop(int, void *, void *);
}
extern "C"
{
    void nanos_reduction_bop_add_char(void *arg1, void *arg2);
    void nanos_reduction_vop_add_char(int i, void *arg1, void *arg2);
    void nanos_reduction_bop_add_uchar(void *arg1, void *arg2);
    void nanos_reduction_vop_add_uchar(int i, void *arg1, void *arg2);
    void nanos_reduction_bop_add_schar(void *arg1, void *arg2);
    void nanos_reduction_vop_add_schar(int i, void *arg1, void *arg2);
    void nanos_reduction_bop_add_short(void *arg1, void *arg2);
    void nanos_reduction_vop_add_short(int i, void *arg1, void *arg2);
    void nanos_reduction_bop_add_ushort(void *arg1, void *arg2);
    void nanos_reduction_vop_add_ushort(int i, void *arg1, void *arg2);
    void nanos_reduction_bop_add_int(void *arg1, void *arg2);
    void nanos_reduction_vop_add_int(int i, void *arg1, void *arg2);
    void nanos_reduction_bop_add_uint(void *arg1, void *arg2);
    void nanos_reduction_vop_add_uint(int i, void *arg1, void *arg2);
    void nanos_reduction_bop_add_long(void *arg1, void *arg2);
    void nanos_reduction_vop_add_long(int i, void *arg1, void *arg2);
    void nanos_reduction_bop_add_ulong(void *arg1, void *arg2);
    void nanos_reduction_vop_add_ulong(int i, void *arg1, void *arg2);
    void nanos_reduction_bop_add_longlong(void *arg1, void *arg2);
    void nanos_reduction_vop_add_longlong(int i, void *arg1, void *arg2);
    void nanos_reduction_bop_add_ulonglong(void *arg1, void *arg2);
    void nanos_reduction_vop_add_ulonglong(int i, void *arg1, void *arg2);
    void nanos_reduction_bop_add__Bool(void *arg1, void *arg2);
    void nanos_reduction_vop_add__Bool(int i, void *arg1, void *arg2);
    void nanos_reduction_bop_add_float(void *arg1, void *arg2);
    void nanos_reduction_vop_add_float(int i, void *arg1, void *arg2);
    void nanos_reduction_bop_add_double(void *arg1, void *arg2);
    void nanos_reduction_vop_add_double(int i, void *arg1, void *arg2);
    void nanos_reduction_bop_add_longdouble(void *arg1, void *arg2);
    void nanos_reduction_vop_add_longdouble(int i, void *arg1, void *arg2);
    void nanos_reduction_bop_sub_char(void *arg1, void *arg2);
    void nanos_reduction_vop_sub_char(int i, void *arg1, void *arg2);
    void nanos_reduction_bop_sub_uchar(void *arg1, void *arg2);
    void nanos_reduction_vop_sub_uchar(int i, void *arg1, void *arg2);
    void nanos_reduction_bop_sub_schar(void *arg1, void *arg2);
    void nanos_reduction_vop_sub_schar(int i, void *arg1, void *arg2);
    void nanos_reduction_bop_sub_short(void *arg1, void *arg2);
    void nanos_reduction_vop_sub_short(int i, void *arg1, void *arg2);
    void nanos_reduction_bop_sub_ushort(void *arg1, void *arg2);
    void nanos_reduction_vop_sub_ushort(int i, void *arg1, void *arg2);
    void nanos_reduction_bop_sub_int(void *arg1, void *arg2);
    void nanos_reduction_vop_sub_int(int i, void *arg1, void *arg2);
    void nanos_reduction_bop_sub_uint(void *arg1, void *arg2);
    void nanos_reduction_vop_sub_uint(int i, void *arg1, void *arg2);
    void nanos_reduction_bop_sub_long(void *arg1, void *arg2);
    void nanos_reduction_vop_sub_long(int i, void *arg1, void *arg2);
    void nanos_reduction_bop_sub_ulong(void *arg1, void *arg2);
    void nanos_reduction_vop_sub_ulong(int i, void *arg1, void *arg2);
    void nanos_reduction_bop_sub_longlong(void *arg1, void *arg2);
    void nanos_reduction_vop_sub_longlong(int i, void *arg1, void *arg2);
    void nanos_reduction_bop_sub_ulonglong(void *arg1, void *arg2);
    void nanos_reduction_vop_sub_ulonglong(int i, void *arg1, void *arg2);
    void nanos_reduction_bop_sub__Bool(void *arg1, void *arg2);
    void nanos_reduction_vop_sub__Bool(int i, void *arg1, void *arg2);
    void nanos_reduction_bop_sub_float(void *arg1, void *arg2);
    void nanos_reduction_vop_sub_float(int i, void *arg1, void *arg2);
    void nanos_reduction_bop_sub_double(void *arg1, void *arg2);
    void nanos_reduction_vop_sub_double(int i, void *arg1, void *arg2);
    void nanos_reduction_bop_sub_longdouble(void *arg1, void *arg2);
    void nanos_reduction_vop_sub_longdouble(int i, void *arg1, void *arg2);
    void nanos_reduction_bop_prod_char(void *arg1, void *arg2);
    void nanos_reduction_vop_prod_char(int i, void *arg1, void *arg2);
    void nanos_reduction_bop_prod_uchar(void *arg1, void *arg2);
    void nanos_reduction_vop_prod_uchar(int i, void *arg1, void *arg2);
    void nanos_reduction_bop_prod_schar(void *arg1, void *arg2);
    void nanos_reduction_vop_prod_schar(int i, void *arg1, void *arg2);
    void nanos_reduction_bop_prod_short(void *arg1, void *arg2);
    void nanos_reduction_vop_prod_short(int i, void *arg1, void *arg2);
    void nanos_reduction_bop_prod_ushort(void *arg1, void *arg2);
    void nanos_reduction_vop_prod_ushort(int i, void *arg1, void *arg2);
    void nanos_reduction_bop_prod_int(void *arg1, void *arg2);
    void nanos_reduction_vop_prod_int(int i, void *arg1, void *arg2);
    void nanos_reduction_bop_prod_uint(void *arg1, void *arg2);
    void nanos_reduction_vop_prod_uint(int i, void *arg1, void *arg2);
    void nanos_reduction_bop_prod_long(void *arg1, void *arg2);
    void nanos_reduction_vop_prod_long(int i, void *arg1, void *arg2);
    void nanos_reduction_bop_prod_ulong(void *arg1, void *arg2);
    void nanos_reduction_vop_prod_ulong(int i, void *arg1, void *arg2);
    void nanos_reduction_bop_prod_longlong(void *arg1, void *arg2);
    void nanos_reduction_vop_prod_longlong(int i, void *arg1, void *arg2);
    void nanos_reduction_bop_prod_ulonglong(void *arg1, void *arg2);
    void nanos_reduction_vop_prod_ulonglong(int i, void *arg1, void *arg2);
    void nanos_reduction_bop_prod__Bool(void *arg1, void *arg2);
    void nanos_reduction_vop_prod__Bool(int i, void *arg1, void *arg2);
    void nanos_reduction_bop_prod_float(void *arg1, void *arg2);
    void nanos_reduction_vop_prod_float(int i, void *arg1, void *arg2);
    void nanos_reduction_bop_prod_double(void *arg1, void *arg2);
    void nanos_reduction_vop_prod_double(int i, void *arg1, void *arg2);
    void nanos_reduction_bop_prod_longdouble(void *arg1, void *arg2);
    void nanos_reduction_vop_prod_longdouble(int i, void *arg1, void *arg2);
    void nanos_reduction_bop_and_char(void *arg1, void *arg2);
    void nanos_reduction_vop_and_char(int i, void *arg1, void *arg2);
    void nanos_reduction_bop_and_uchar(void *arg1, void *arg2);
    void nanos_reduction_vop_and_uchar(int i, void *arg1, void *arg2);
    void nanos_reduction_bop_and_schar(void *arg1, void *arg2);
    void nanos_reduction_vop_and_schar(int i, void *arg1, void *arg2);
    void nanos_reduction_bop_and_short(void *arg1, void *arg2);
    void nanos_reduction_vop_and_short(int i, void *arg1, void *arg2);
    void nanos_reduction_bop_and_ushort(void *arg1, void *arg2);
    void nanos_reduction_vop_and_ushort(int i, void *arg1, void *arg2);
    void nanos_reduction_bop_and_int(void *arg1, void *arg2);
    void nanos_reduction_vop_and_int(int i, void *arg1, void *arg2);
    void nanos_reduction_bop_and_uint(void *arg1, void *arg2);
    void nanos_reduction_vop_and_uint(int i, void *arg1, void *arg2);
    void nanos_reduction_bop_and_long(void *arg1, void *arg2);
    void nanos_reduction_vop_and_long(int i, void *arg1, void *arg2);
    void nanos_reduction_bop_and_ulong(void *arg1, void *arg2);
    void nanos_reduction_vop_and_ulong(int i, void *arg1, void *arg2);
    void nanos_reduction_bop_and_longlong(void *arg1, void *arg2);
    void nanos_reduction_vop_and_longlong(int i, void *arg1, void *arg2);
    void nanos_reduction_bop_and_ulonglong(void *arg1, void *arg2);
    void nanos_reduction_vop_and_ulonglong(int i, void *arg1, void *arg2);
    void nanos_reduction_bop_and__Bool(void *arg1, void *arg2);
    void nanos_reduction_vop_and__Bool(int i, void *arg1, void *arg2);
    void nanos_reduction_bop_or_char(void *arg1, void *arg2);
    void nanos_reduction_vop_or_char(int i, void *arg1, void *arg2);
    void nanos_reduction_bop_or_uchar(void *arg1, void *arg2);
    void nanos_reduction_vop_or_uchar(int i, void *arg1, void *arg2);
    void nanos_reduction_bop_or_schar(void *arg1, void *arg2);
    void nanos_reduction_vop_or_schar(int i, void *arg1, void *arg2);
    void nanos_reduction_bop_or_short(void *arg1, void *arg2);
    void nanos_reduction_vop_or_short(int i, void *arg1, void *arg2);
    void nanos_reduction_bop_or_ushort(void *arg1, void *arg2);
    void nanos_reduction_vop_or_ushort(int i, void *arg1, void *arg2);
    void nanos_reduction_bop_or_int(void *arg1, void *arg2);
    void nanos_reduction_vop_or_int(int i, void *arg1, void *arg2);
    void nanos_reduction_bop_or_uint(void *arg1, void *arg2);
    void nanos_reduction_vop_or_uint(int i, void *arg1, void *arg2);
    void nanos_reduction_bop_or_long(void *arg1, void *arg2);
    void nanos_reduction_vop_or_long(int i, void *arg1, void *arg2);
    void nanos_reduction_bop_or_ulong(void *arg1, void *arg2);
    void nanos_reduction_vop_or_ulong(int i, void *arg1, void *arg2);
    void nanos_reduction_bop_or_longlong(void *arg1, void *arg2);
    void nanos_reduction_vop_or_longlong(int i, void *arg1, void *arg2);
    void nanos_reduction_bop_or_ulonglong(void *arg1, void *arg2);
    void nanos_reduction_vop_or_ulonglong(int i, void *arg1, void *arg2);
    void nanos_reduction_bop_or__Bool(void *arg1, void *arg2);
    void nanos_reduction_vop_or__Bool(int i, void *arg1, void *arg2);
    void nanos_reduction_bop_xor_char(void *arg1, void *arg2);
    void nanos_reduction_vop_xor_char(int i, void *arg1, void *arg2);
    void nanos_reduction_bop_xor_uchar(void *arg1, void *arg2);
    void nanos_reduction_vop_xor_uchar(int i, void *arg1, void *arg2);
    void nanos_reduction_bop_xor_schar(void *arg1, void *arg2);
    void nanos_reduction_vop_xor_schar(int i, void *arg1, void *arg2);
    void nanos_reduction_bop_xor_short(void *arg1, void *arg2);
    void nanos_reduction_vop_xor_short(int i, void *arg1, void *arg2);
    void nanos_reduction_bop_xor_ushort(void *arg1, void *arg2);
    void nanos_reduction_vop_xor_ushort(int i, void *arg1, void *arg2);
    void nanos_reduction_bop_xor_int(void *arg1, void *arg2);
    void nanos_reduction_vop_xor_int(int i, void *arg1, void *arg2);
    void nanos_reduction_bop_xor_uint(void *arg1, void *arg2);
    void nanos_reduction_vop_xor_uint(int i, void *arg1, void *arg2);
    void nanos_reduction_bop_xor_long(void *arg1, void *arg2);
    void nanos_reduction_vop_xor_long(int i, void *arg1, void *arg2);
    void nanos_reduction_bop_xor_ulong(void *arg1, void *arg2);
    void nanos_reduction_vop_xor_ulong(int i, void *arg1, void *arg2);
    void nanos_reduction_bop_xor_longlong(void *arg1, void *arg2);
    void nanos_reduction_vop_xor_longlong(int i, void *arg1, void *arg2);
    void nanos_reduction_bop_xor_ulonglong(void *arg1, void *arg2);
    void nanos_reduction_vop_xor_ulonglong(int i, void *arg1, void *arg2);
    void nanos_reduction_bop_xor__Bool(void *arg1, void *arg2);
    void nanos_reduction_vop_xor__Bool(int i, void *arg1, void *arg2);
    void nanos_reduction_bop_land_char(void *arg1, void *arg2);
    void nanos_reduction_vop_land_char(int i, void *arg1, void *arg2);
    void nanos_reduction_bop_land_uchar(void *arg1, void *arg2);
    void nanos_reduction_vop_land_uchar(int i, void *arg1, void *arg2);
    void nanos_reduction_bop_land_schar(void *arg1, void *arg2);
    void nanos_reduction_vop_land_schar(int i, void *arg1, void *arg2);
    void nanos_reduction_bop_land_short(void *arg1, void *arg2);
    void nanos_reduction_vop_land_short(int i, void *arg1, void *arg2);
    void nanos_reduction_bop_land_ushort(void *arg1, void *arg2);
    void nanos_reduction_vop_land_ushort(int i, void *arg1, void *arg2);
    void nanos_reduction_bop_land_int(void *arg1, void *arg2);
    void nanos_reduction_vop_land_int(int i, void *arg1, void *arg2);
    void nanos_reduction_bop_land_uint(void *arg1, void *arg2);
    void nanos_reduction_vop_land_uint(int i, void *arg1, void *arg2);
    void nanos_reduction_bop_land_long(void *arg1, void *arg2);
    void nanos_reduction_vop_land_long(int i, void *arg1, void *arg2);
    void nanos_reduction_bop_land_ulong(void *arg1, void *arg2);
    void nanos_reduction_vop_land_ulong(int i, void *arg1, void *arg2);
    void nanos_reduction_bop_land_longlong(void *arg1, void *arg2);
    void nanos_reduction_vop_land_longlong(int i, void *arg1, void *arg2);
    void nanos_reduction_bop_land_ulonglong(void *arg1, void *arg2);
    void nanos_reduction_vop_land_ulonglong(int i, void *arg1, void *arg2);
    void nanos_reduction_bop_land__Bool(void *arg1, void *arg2);
    void nanos_reduction_vop_land__Bool(int i, void *arg1, void *arg2);
    void nanos_reduction_bop_land_float(void *arg1, void *arg2);
    void nanos_reduction_vop_land_float(int i, void *arg1, void *arg2);
    void nanos_reduction_bop_land_double(void *arg1, void *arg2);
    void nanos_reduction_vop_land_double(int i, void *arg1, void *arg2);
    void nanos_reduction_bop_land_longdouble(void *arg1, void *arg2);
    void nanos_reduction_vop_land_longdouble(int i, void *arg1, void *arg2);
    void nanos_reduction_bop_lor_char(void *arg1, void *arg2);
    void nanos_reduction_vop_lor_char(int i, void *arg1, void *arg2);
    void nanos_reduction_bop_lor_uchar(void *arg1, void *arg2);
    void nanos_reduction_vop_lor_uchar(int i, void *arg1, void *arg2);
    void nanos_reduction_bop_lor_schar(void *arg1, void *arg2);
    void nanos_reduction_vop_lor_schar(int i, void *arg1, void *arg2);
    void nanos_reduction_bop_lor_short(void *arg1, void *arg2);
    void nanos_reduction_vop_lor_short(int i, void *arg1, void *arg2);
    void nanos_reduction_bop_lor_ushort(void *arg1, void *arg2);
    void nanos_reduction_vop_lor_ushort(int i, void *arg1, void *arg2);
    void nanos_reduction_bop_lor_int(void *arg1, void *arg2);
    void nanos_reduction_vop_lor_int(int i, void *arg1, void *arg2);
    void nanos_reduction_bop_lor_uint(void *arg1, void *arg2);
    void nanos_reduction_vop_lor_uint(int i, void *arg1, void *arg2);
    void nanos_reduction_bop_lor_long(void *arg1, void *arg2);
    void nanos_reduction_vop_lor_long(int i, void *arg1, void *arg2);
    void nanos_reduction_bop_lor_ulong(void *arg1, void *arg2);
    void nanos_reduction_vop_lor_ulong(int i, void *arg1, void *arg2);
    void nanos_reduction_bop_lor_longlong(void *arg1, void *arg2);
    void nanos_reduction_vop_lor_longlong(int i, void *arg1, void *arg2);
    void nanos_reduction_bop_lor_ulonglong(void *arg1, void *arg2);
    void nanos_reduction_vop_lor_ulonglong(int i, void *arg1, void *arg2);
    void nanos_reduction_bop_lor__Bool(void *arg1, void *arg2);
    void nanos_reduction_vop_lor__Bool(int i, void *arg1, void *arg2);
    void nanos_reduction_bop_lor_float(void *arg1, void *arg2);
    void nanos_reduction_vop_lor_float(int i, void *arg1, void *arg2);
    void nanos_reduction_bop_lor_double(void *arg1, void *arg2);
    void nanos_reduction_vop_lor_double(int i, void *arg1, void *arg2);
    void nanos_reduction_bop_lor_longdouble(void *arg1, void *arg2);
    void nanos_reduction_vop_lor_longdouble(int i, void *arg1, void *arg2);
    void nanos_reduction_bop_max_char(void *arg1, void *arg2);
    void nanos_reduction_vop_max_char(int i, void *arg1, void *arg2);
    void nanos_reduction_bop_max_uchar(void *arg1, void *arg2);
    void nanos_reduction_vop_max_uchar(int i, void *arg1, void *arg2);
    void nanos_reduction_bop_max_schar(void *arg1, void *arg2);
    void nanos_reduction_vop_max_schar(int i, void *arg1, void *arg2);
    void nanos_reduction_bop_max_short(void *arg1, void *arg2);
    void nanos_reduction_vop_max_short(int i, void *arg1, void *arg2);
    void nanos_reduction_bop_max_ushort(void *arg1, void *arg2);
    void nanos_reduction_vop_max_ushort(int i, void *arg1, void *arg2);
    void nanos_reduction_bop_max_int(void *arg1, void *arg2);
    void nanos_reduction_vop_max_int(int i, void *arg1, void *arg2);
    void nanos_reduction_bop_max_uint(void *arg1, void *arg2);
    void nanos_reduction_vop_max_uint(int i, void *arg1, void *arg2);
    void nanos_reduction_bop_max_long(void *arg1, void *arg2);
    void nanos_reduction_vop_max_long(int i, void *arg1, void *arg2);
    void nanos_reduction_bop_max_ulong(void *arg1, void *arg2);
    void nanos_reduction_vop_max_ulong(int i, void *arg1, void *arg2);
    void nanos_reduction_bop_max_longlong(void *arg1, void *arg2);
    void nanos_reduction_vop_max_longlong(int i, void *arg1, void *arg2);
    void nanos_reduction_bop_max_ulonglong(void *arg1, void *arg2);
    void nanos_reduction_vop_max_ulonglong(int i, void *arg1, void *arg2);
    void nanos_reduction_bop_max__Bool(void *arg1, void *arg2);
    void nanos_reduction_vop_max__Bool(int i, void *arg1, void *arg2);
    void nanos_reduction_bop_max_float(void *arg1, void *arg2);
    void nanos_reduction_vop_max_float(int i, void *arg1, void *arg2);
    void nanos_reduction_bop_max_double(void *arg1, void *arg2);
    void nanos_reduction_vop_max_double(int i, void *arg1, void *arg2);
    void nanos_reduction_bop_max_longdouble(void *arg1, void *arg2);
    void nanos_reduction_vop_max_longdouble(int i, void *arg1, void *arg2);
    void nanos_reduction_bop_min_char(void *arg1, void *arg2);
    void nanos_reduction_vop_min_char(int i, void *arg1, void *arg2);
    void nanos_reduction_bop_min_uchar(void *arg1, void *arg2);
    void nanos_reduction_vop_min_uchar(int i, void *arg1, void *arg2);
    void nanos_reduction_bop_min_schar(void *arg1, void *arg2);
    void nanos_reduction_vop_min_schar(int i, void *arg1, void *arg2);
    void nanos_reduction_bop_min_short(void *arg1, void *arg2);
    void nanos_reduction_vop_min_short(int i, void *arg1, void *arg2);
    void nanos_reduction_bop_min_ushort(void *arg1, void *arg2);
    void nanos_reduction_vop_min_ushort(int i, void *arg1, void *arg2);
    void nanos_reduction_bop_min_int(void *arg1, void *arg2);
    void nanos_reduction_vop_min_int(int i, void *arg1, void *arg2);
    void nanos_reduction_bop_min_uint(void *arg1, void *arg2);
    void nanos_reduction_vop_min_uint(int i, void *arg1, void *arg2);
    void nanos_reduction_bop_min_long(void *arg1, void *arg2);
    void nanos_reduction_vop_min_long(int i, void *arg1, void *arg2);
    void nanos_reduction_bop_min_ulong(void *arg1, void *arg2);
    void nanos_reduction_vop_min_ulong(int i, void *arg1, void *arg2);
    void nanos_reduction_bop_min_longlong(void *arg1, void *arg2);
    void nanos_reduction_vop_min_longlong(int i, void *arg1, void *arg2);
    void nanos_reduction_bop_min_ulonglong(void *arg1, void *arg2);
    void nanos_reduction_vop_min_ulonglong(int i, void *arg1, void *arg2);
    void nanos_reduction_bop_min__Bool(void *arg1, void *arg2);
    void nanos_reduction_vop_min__Bool(int i, void *arg1, void *arg2);
    void nanos_reduction_bop_min_float(void *arg1, void *arg2);
    void nanos_reduction_vop_min_float(int i, void *arg1, void *arg2);
    void nanos_reduction_bop_min_double(void *arg1, void *arg2);
    void nanos_reduction_vop_min_double(int i, void *arg1, void *arg2);
    void nanos_reduction_bop_min_longdouble(void *arg1, void *arg2);
    void nanos_reduction_vop_min_longdouble(int i, void *arg1, void *arg2);
    void nanos_reduction_default_cleanup_char(void *r);
    void nanos_reduction_default_cleanup_uchar(void *r);
    void nanos_reduction_default_cleanup_schar(void *r);
    void nanos_reduction_default_cleanup_short(void *r);
    void nanos_reduction_default_cleanup_ushort(void *r);
    void nanos_reduction_default_cleanup_int(void *r);
    void nanos_reduction_default_cleanup_uint(void *r);
    void nanos_reduction_default_cleanup_long(void *r);
    void nanos_reduction_default_cleanup_ulong(void *r);
    void nanos_reduction_default_cleanup_longlong(void *r);
    void nanos_reduction_default_cleanup_ulonglong(void *r);
    void nanos_reduction_default_cleanup__Bool(void *r);
    void nanos_reduction_default_cleanup_float(void *r);
    void nanos_reduction_default_cleanup_double(void *r);
    void nanos_reduction_default_cleanup_longdouble(void *r);
}
typedef void *omp_lock_t;
typedef void *omp_nest_lock_t;
typedef enum omp_sched_t
{
    omp_sched_static = 1, 
    omp_sched_dynamic = 2, 
    omp_sched_guided = 3, 
    omp_sched_auto = 4
} omp_sched_t;
extern "C"
{
    extern void omp_set_num_threads(int num_threads);
    extern int omp_get_num_threads(void);
    extern int omp_get_max_threads(void);
    extern int omp_get_thread_num(void);
    extern int omp_get_num_procs(void);
    extern int omp_in_parallel(void);
    extern void omp_set_dynamic(int dynamic_threads);
    extern int omp_get_dynamic(void);
    extern void omp_set_nested(int nested);
    extern int omp_get_nested(void);
    extern int omp_get_thread_limit(void);
    extern void omp_set_max_active_levels(int max_active_levels);
    extern int omp_get_max_active_levels(void);
    extern void omp_set_schedule(omp_sched_t kind, int modifier);
    extern void omp_get_schedule(omp_sched_t *kind, int *modifier);
    extern int omp_get_level(void);
    extern int omp_get_ancestor_thread_num(int level);
    extern int omp_get_team_size(int level);
    extern int omp_get_active_level(void);
    extern void omp_init_lock(omp_lock_t *lock);
    extern void omp_destroy_lock(omp_lock_t *lock);
    extern void omp_set_lock(omp_lock_t *lock);
    extern void omp_unset_lock(omp_lock_t *lock);
    extern int omp_test_lock(omp_lock_t *lock);
    extern void omp_init_nest_lock(omp_nest_lock_t *lock);
    extern void omp_destroy_nest_lock(omp_nest_lock_t *lock);
    extern void omp_set_nest_lock(omp_nest_lock_t *lock);
    extern void omp_unset_nest_lock(omp_nest_lock_t *lock);
    extern int omp_test_nest_lock(omp_nest_lock_t *lock);
    extern double omp_get_wtime(void);
    extern double omp_get_wtick(void);
    extern int omp_in_final(void);
}
extern "C"
{
    nanos_err_t nanos_omp_single(bool *);
    nanos_err_t nanos_omp_barrier(void);
    void nanos_omp_set_interface(void *);
    nanos_err_t nanos_omp_set_implicit(nanos_wd_t uwd);
    int nanos_omp_get_max_threads(void);
    nanos_ws_t nanos_omp_find_worksharing(omp_sched_t kind);
    nanos_err_t nanos_omp_get_schedule(omp_sched_t *kind, int *modifier);
}
namespace std __attribute__((__visibility__("default"))) {
    using ::ptrdiff_t;
    using ::size_t;
}
#pragma GCC visibility push(default)
extern "C++"
{
    namespace std {
        class exception
        {
            public :
                exception() throw ()
                {
                }
                virtual ~exception() throw ();
                virtual const char *what() const throw ();
        };
        class bad_exception : public exception
        {
            public :
                bad_exception() throw ()
                {
                }
                virtual ~bad_exception() throw ();
                virtual const char *what() const throw ();
        };
        typedef void (*terminate_handler)();
        typedef void (*unexpected_handler)();
        terminate_handler set_terminate(terminate_handler) throw ();
        void terminate() __attribute__((__noreturn__));
        unexpected_handler set_unexpected(unexpected_handler) throw ();
        void unexpected() __attribute__((__noreturn__));
        bool uncaught_exception() throw ();
    }
    namespace __gnu_cxx __attribute__((__visibility__("default"))) {
        void __verbose_terminate_handler();
    }
}
#pragma GCC visibility pop
#pragma GCC visibility push(default)
extern "C++"
{
    namespace std {
        class bad_alloc : public exception
        {
            public :
                bad_alloc() throw ()
                {
                }
                virtual ~bad_alloc() throw ();
                virtual const char *what() const throw ();
        };
        struct nothrow_t
        {
        };
        extern const nothrow_t nothrow;
        typedef void (*new_handler)();
        new_handler set_new_handler(new_handler) throw ();
    }
    void *operator new(std::size_t) throw (std::bad_alloc);
    void *operator new[](std::size_t) throw (std::bad_alloc);
    void operator delete(void *) throw ();
    void operator delete[](void *) throw ();
    void *operator new(std::size_t, const std::nothrow_t &) throw ();
    void *operator new[](std::size_t, const std::nothrow_t &) throw ();
    void operator delete(void *, const std::nothrow_t &) throw ();
    void operator delete[](void *, const std::nothrow_t &) throw ();
    inline void *operator new(std::size_t, void *__p) throw ()
    {
        return __p;
    }
    inline void *operator new[](std::size_t, void *__p) throw ()
    {
        return __p;
    }
    inline void operator delete(void *, void *) throw ()
    {
    }
    inline void operator delete[](void *, void *) throw ()
    {
    }
}
#pragma GCC visibility pop
namespace std __attribute__((__visibility__("default"))) {
    template<typename _Alloc >
    class allocator;
    template<class _CharT >
    struct char_traits;
    template<typename _CharT, typename _Traits = char_traits<_CharT>, typename _Alloc = allocator<_CharT> >
    class basic_string;
    template<>
    struct char_traits<char>;
    typedef basic_string<char> string;
    template<>
    struct char_traits<wchar_t>;
    typedef basic_string<wchar_t> wstring;
}
typedef unsigned int wint_t;
typedef __mbstate_t mbstate_t;
extern "C"
{
    struct tm;
    extern wchar_t *wcscpy(wchar_t *__restrict __dest, __const wchar_t *__restrict __src) throw ();
    extern wchar_t *wcsncpy(wchar_t *__restrict __dest, __const wchar_t *__restrict __src, size_t __n) throw ();
    extern wchar_t *wcscat(wchar_t *__restrict __dest, __const wchar_t *__restrict __src) throw ();
    extern wchar_t *wcsncat(wchar_t *__restrict __dest, __const wchar_t *__restrict __src, size_t __n) throw ();
    extern int wcscmp(__const wchar_t *__s1, __const wchar_t *__s2) throw () __attribute__((__pure__));
    extern int wcsncmp(__const wchar_t *__s1, __const wchar_t *__s2, size_t __n) throw () __attribute__((__pure__));
    extern int wcscasecmp(__const wchar_t *__s1, __const wchar_t *__s2) throw ();
    extern int wcsncasecmp(__const wchar_t *__s1, __const wchar_t *__s2, size_t __n) throw ();
    typedef struct __locale_struct
    {
            struct __locale_data *__locales[13];
            const unsigned short int *__ctype_b;
            const int *__ctype_tolower;
            const int *__ctype_toupper;
            const char *__names[13];
    } *__locale_t;
    typedef __locale_t locale_t;
    extern int wcscasecmp_l(__const wchar_t *__s1, __const wchar_t *__s2, __locale_t __loc) throw ();
    extern int wcsncasecmp_l(__const wchar_t *__s1, __const wchar_t *__s2, size_t __n, __locale_t __loc) throw ();
    extern int wcscoll(__const wchar_t *__s1, __const wchar_t *__s2) throw ();
    extern size_t wcsxfrm(wchar_t *__restrict __s1, __const wchar_t *__restrict __s2, size_t __n) throw ();
    extern int wcscoll_l(__const wchar_t *__s1, __const wchar_t *__s2, __locale_t __loc) throw ();
    extern size_t wcsxfrm_l(wchar_t *__s1, __const wchar_t *__s2, size_t __n, __locale_t __loc) throw ();
    extern wchar_t *wcsdup(__const wchar_t *__s) throw () __attribute__((__malloc__));
    extern "C++"
    wchar_t *wcschr(wchar_t *__wcs, wchar_t __wc) throw () __asm ("wcschr") __attribute__((__pure__));
    extern "C++"
    __const wchar_t *wcschr(__const wchar_t *__wcs, wchar_t __wc) throw () __asm ("wcschr") __attribute__((__pure__));
    extern "C++"
    wchar_t *wcsrchr(wchar_t *__wcs, wchar_t __wc) throw () __asm ("wcsrchr") __attribute__((__pure__));
    extern "C++"
    __const wchar_t *wcsrchr(__const wchar_t *__wcs, wchar_t __wc) throw () __asm ("wcsrchr") __attribute__((__pure__));
    extern wchar_t *wcschrnul(__const wchar_t *__s, wchar_t __wc) throw () __attribute__((__pure__));
    extern size_t wcscspn(__const wchar_t *__wcs, __const wchar_t *__reject) throw () __attribute__((__pure__));
    extern size_t wcsspn(__const wchar_t *__wcs, __const wchar_t *__accept) throw () __attribute__((__pure__));
    extern "C++"
    wchar_t *wcspbrk(wchar_t *__wcs, __const wchar_t *__accept) throw () __asm ("wcspbrk") __attribute__((__pure__));
    extern "C++"
    __const wchar_t *wcspbrk(__const wchar_t *__wcs, __const wchar_t *__accept) throw () __asm ("wcspbrk") __attribute__((__pure__));
    extern "C++"
    wchar_t *wcsstr(wchar_t *__haystack, __const wchar_t *__needle) throw () __asm ("wcsstr") __attribute__((__pure__));
    extern "C++"
    __const wchar_t *wcsstr(__const wchar_t *__haystack, __const wchar_t *__needle) throw () __asm ("wcsstr") __attribute__((__pure__));
    extern wchar_t *wcstok(wchar_t *__restrict __s, __const wchar_t *__restrict __delim, wchar_t **__restrict __ptr) throw ();
    extern size_t wcslen(__const wchar_t *__s) throw () __attribute__((__pure__));
    extern "C++"
    wchar_t *wcswcs(wchar_t *__haystack, __const wchar_t *__needle) throw () __asm ("wcswcs") __attribute__((__pure__));
    extern "C++"
    __const wchar_t *wcswcs(__const wchar_t *__haystack, __const wchar_t *__needle) throw () __asm ("wcswcs") __attribute__((__pure__));
    extern size_t wcsnlen(__const wchar_t *__s, size_t __maxlen) throw () __attribute__((__pure__));
    extern "C++"
    wchar_t *wmemchr(wchar_t *__s, wchar_t __c, size_t __n) throw () __asm ("wmemchr") __attribute__((__pure__));
    extern "C++"
    __const wchar_t *wmemchr(__const wchar_t *__s, wchar_t __c, size_t __n) throw () __asm ("wmemchr") __attribute__((__pure__));
    extern int wmemcmp(__const wchar_t *__restrict __s1, __const wchar_t *__restrict __s2, size_t __n) throw () __attribute__((__pure__));
    extern wchar_t *wmemcpy(wchar_t *__restrict __s1, __const wchar_t *__restrict __s2, size_t __n) throw ();
    extern wchar_t *wmemmove(wchar_t *__s1, __const wchar_t *__s2, size_t __n) throw ();
    extern wchar_t *wmemset(wchar_t *__s, wchar_t __c, size_t __n) throw ();
    extern wchar_t *wmempcpy(wchar_t *__restrict __s1, __const wchar_t *__restrict __s2, size_t __n) throw ();
    extern wint_t btowc(int __c) throw ();
    extern int wctob(wint_t __c) throw ();
    extern int mbsinit(__const mbstate_t *__ps) throw () __attribute__((__pure__));
    extern size_t mbrtowc(wchar_t *__restrict __pwc, __const char *__restrict __s, size_t __n, mbstate_t *__p) throw ();
    extern size_t wcrtomb(char *__restrict __s, wchar_t __wc, mbstate_t *__restrict __ps) throw ();
    extern size_t __mbrlen(__const char *__restrict __s, size_t __n, mbstate_t *__restrict __ps) throw ();
    extern size_t mbrlen(__const char *__restrict __s, size_t __n, mbstate_t *__restrict __ps) throw ();
    extern wint_t __btowc_alias(int __c) __asm ("btowc");
    extern __inline __attribute__((__gnu_inline__)) wint_t btowc(int __c) throw ()
    {
        return (__builtin_constant_p(__c) && __c >= '\0' && __c <= '\x7f' ? (wint_t) __c : __btowc_alias(__c));
    }
    extern int __wctob_alias(wint_t __c) __asm ("wctob");
    extern __inline __attribute__((__gnu_inline__)) int wctob(wint_t __wc) throw ()
    {
        return (__builtin_constant_p(__wc) && __wc >= L'\0' && __wc <= L'\x7f' ? (int) __wc : __wctob_alias(__wc));
    }
    extern __inline __attribute__((__gnu_inline__)) size_t mbrlen(__const char *__restrict __s, size_t __n, mbstate_t *__restrict __ps) throw ()
    {
        return (__ps != __null ? mbrtowc(__null, __s, __n, __ps) : __mbrlen(__s, __n, __null));
    }
    extern size_t mbsrtowcs(wchar_t *__restrict __dst, __const char **__restrict __src, size_t __len, mbstate_t *__restrict __ps) throw ();
    extern size_t wcsrtombs(char *__restrict __dst, __const wchar_t **__restrict __src, size_t __len, mbstate_t *__restrict __ps) throw ();
    extern size_t mbsnrtowcs(wchar_t *__restrict __dst, __const char **__restrict __src, size_t __nmc, size_t __len, mbstate_t *__restrict __ps) throw ();
    extern size_t wcsnrtombs(char *__restrict __dst, __const wchar_t **__restrict __src, size_t __nwc, size_t __len, mbstate_t *__restrict __ps) throw ();
    extern int wcwidth(wchar_t __c) throw ();
    extern int wcswidth(__const wchar_t *__s, size_t __n) throw ();
    extern double wcstod(__const wchar_t *__restrict __nptr, wchar_t **__restrict __endptr) throw ();
    extern float wcstof(__const wchar_t *__restrict __nptr, wchar_t **__restrict __endptr) throw ();
    extern long double wcstold(__const wchar_t *__restrict __nptr, wchar_t **__restrict __endptr) throw ();
    extern long int wcstol(__const wchar_t *__restrict __nptr, wchar_t **__restrict __endptr, int __base) throw ();
    extern unsigned long int wcstoul(__const wchar_t *__restrict __nptr, wchar_t **__restrict __endptr, int __base) throw ();
    __extension__
    extern long long int wcstoll(__const wchar_t *__restrict __nptr, wchar_t **__restrict __endptr, int __base) throw ();
    __extension__
    extern unsigned long long int wcstoull(__const wchar_t *__restrict __nptr, wchar_t **__restrict __endptr, int __base) throw ();
    __extension__
    extern long long int wcstoq(__const wchar_t *__restrict __nptr, wchar_t **__restrict __endptr, int __base) throw ();
    __extension__
    extern unsigned long long int wcstouq(__const wchar_t *__restrict __nptr, wchar_t **__restrict __endptr, int __base) throw ();
    extern long int wcstol_l(__const wchar_t *__restrict __nptr, wchar_t **__restrict __endptr, int __base, __locale_t __loc) throw ();
    extern unsigned long int wcstoul_l(__const wchar_t *__restrict __nptr, wchar_t **__restrict __endptr, int __base, __locale_t __loc) throw ();
    __extension__
    extern long long int wcstoll_l(__const wchar_t *__restrict __nptr, wchar_t **__restrict __endptr, int __base, __locale_t __loc) throw ();
    __extension__
    extern unsigned long long int wcstoull_l(__const wchar_t *__restrict __nptr, wchar_t **__restrict __endptr, int __base, __locale_t __loc) throw ();
    extern double wcstod_l(__const wchar_t *__restrict __nptr, wchar_t **__restrict __endptr, __locale_t __loc) throw ();
    extern float wcstof_l(__const wchar_t *__restrict __nptr, wchar_t **__restrict __endptr, __locale_t __loc) throw ();
    extern long double wcstold_l(__const wchar_t *__restrict __nptr, wchar_t **__restrict __endptr, __locale_t __loc) throw ();
    extern wchar_t *wcpcpy(wchar_t *__dest, __const wchar_t *__src) throw ();
    extern wchar_t *wcpncpy(wchar_t *__dest, __const wchar_t *__src, size_t __n) throw ();
    extern __FILE *open_wmemstream(wchar_t **__bufloc, size_t *__sizeloc) throw ();
    extern int fwide(__FILE *__fp, int __mode) throw ();
    extern int fwprintf(__FILE *__restrict __stream, __const wchar_t *__restrict __format, ...);
    extern int wprintf(__const wchar_t *__restrict __format, ...);
    extern int swprintf(wchar_t *__restrict __s, size_t __n, __const wchar_t *__restrict __format, ...) throw ();
    extern int vfwprintf(__FILE *__restrict __s, __const wchar_t *__restrict __format, __gnuc_va_list __arg);
    extern int vwprintf(__const wchar_t *__restrict __format, __gnuc_va_list __arg);
    extern int vswprintf(wchar_t *__restrict __s, size_t __n, __const wchar_t *__restrict __format, __gnuc_va_list __arg) throw ();
    extern int fwscanf(__FILE *__restrict __stream, __const wchar_t *__restrict __format, ...);
    extern int wscanf(__const wchar_t *__restrict __format, ...);
    extern int swscanf(__const wchar_t *__restrict __s, __const wchar_t *__restrict __format, ...) throw ();
    extern int vfwscanf(__FILE *__restrict __s, __const wchar_t *__restrict __format, __gnuc_va_list __arg);
    extern int vwscanf(__const wchar_t *__restrict __format, __gnuc_va_list __arg);
    extern int vswscanf(__const wchar_t *__restrict __s, __const wchar_t *__restrict __format, __gnuc_va_list __arg) throw ();
    extern wint_t fgetwc(__FILE *__stream);
    extern wint_t getwc(__FILE *__stream);
    extern wint_t getwchar(void);
    extern wint_t fputwc(wchar_t __wc, __FILE *__stream);
    extern wint_t putwc(wchar_t __wc, __FILE *__stream);
    extern wint_t putwchar(wchar_t __wc);
    extern wchar_t *fgetws(wchar_t *__restrict __ws, int __n, __FILE *__restrict __stream);
    extern int fputws(__const wchar_t *__restrict __ws, __FILE *__restrict __stream);
    extern wint_t ungetwc(wint_t __wc, __FILE *__stream);
    extern wint_t getwc_unlocked(__FILE *__stream);
    extern wint_t getwchar_unlocked(void);
    extern wint_t fgetwc_unlocked(__FILE *__stream);
    extern wint_t fputwc_unlocked(wchar_t __wc, __FILE *__stream);
    extern wint_t putwc_unlocked(wchar_t __wc, __FILE *__stream);
    extern wint_t putwchar_unlocked(wchar_t __wc);
    extern wchar_t *fgetws_unlocked(wchar_t *__restrict __ws, int __n, __FILE *__restrict __stream);
    extern int fputws_unlocked(__const wchar_t *__restrict __ws, __FILE *__restrict __stream);
    extern size_t wcsftime(wchar_t *__restrict __s, size_t __maxsize, __const wchar_t *__restrict __format, __const struct tm *__restrict __tp) throw ();
    extern size_t wcsftime_l(wchar_t *__restrict __s, size_t __maxsize, __const wchar_t *__restrict __format, __const struct tm *__restrict __tp, __locale_t __loc) throw ();
}
namespace std __attribute__((__visibility__("default"))) {
    using ::mbstate_t;
}
namespace std __attribute__((__visibility__("default"))) {
    using ::wint_t;
    using ::btowc;
    using ::fgetwc;
    using ::fgetws;
    using ::fputwc;
    using ::fputws;
    using ::fwide;
    using ::fwprintf;
    using ::fwscanf;
    using ::getwc;
    using ::getwchar;
    using ::mbrlen;
    using ::mbrtowc;
    using ::mbsinit;
    using ::mbsrtowcs;
    using ::putwc;
    using ::putwchar;
    using ::swprintf;
    using ::swscanf;
    using ::ungetwc;
    using ::vfwprintf;
    using ::vfwscanf;
    using ::vswprintf;
    using ::vswscanf;
    using ::vwprintf;
    using ::vwscanf;
    using ::wcrtomb;
    using ::wcscat;
    using ::wcscmp;
    using ::wcscoll;
    using ::wcscpy;
    using ::wcscspn;
    using ::wcsftime;
    using ::wcslen;
    using ::wcsncat;
    using ::wcsncmp;
    using ::wcsncpy;
    using ::wcsrtombs;
    using ::wcsspn;
    using ::wcstod;
    using ::wcstof;
    using ::wcstok;
    using ::wcstol;
    using ::wcstoul;
    using ::wcsxfrm;
    using ::wctob;
    using ::wmemcmp;
    using ::wmemcpy;
    using ::wmemmove;
    using ::wmemset;
    using ::wprintf;
    using ::wscanf;
    using ::wcschr;
    using ::wcspbrk;
    using ::wcsrchr;
    using ::wcsstr;
    using ::wmemchr;
}
namespace __gnu_cxx __attribute__((__visibility__("default"))) {
    using ::wcstold;
    using ::wcstoll;
    using ::wcstoull;
}
namespace std __attribute__((__visibility__("default"))) {
    using ::__gnu_cxx::wcstold;
    using ::__gnu_cxx::wcstoll;
    using ::__gnu_cxx::wcstoull;
}
namespace std __attribute__((__visibility__("default"))) {
    typedef long streamoff;
    typedef ptrdiff_t streamsize;
    template<typename _StateT >
    class fpos
    {
        private :
            streamoff _M_off;
            _StateT _M_state;
        public :
            fpos()
                : _M_off(0), _M_state() 
            {
            }
            fpos(streamoff __off)
                : _M_off(__off), _M_state() 
            {
            }
            operator streamoff() const
            {
                return _M_off;
            }
            void state(_StateT __st)
            {
                _M_state = __st;
            }
            _StateT state() const
            {
                return _M_state;
            }
            fpos &operator +=(streamoff __off)
            {
                _M_off += __off;
                return *this;
            }
            fpos &operator -=(streamoff __off)
            {
                _M_off -= __off;
                return *this;
            }
            fpos operator +(streamoff __off) const
            {
                fpos __pos(*this);
                __pos += __off;
                return __pos;
            }
            fpos operator -(streamoff __off) const
            {
                fpos __pos(*this);
                __pos -= __off;
                return __pos;
            }
            streamoff operator -(const fpos &__other) const
            {
                return _M_off - __other._M_off;
            }
    };
    template<typename _StateT >
    inline bool operator ==(const fpos<_StateT> &__lhs, const fpos<_StateT> &__rhs)
    {
        return streamoff(__lhs) == streamoff(__rhs);
    }
    template<typename _StateT >
    inline bool operator !=(const fpos<_StateT> &__lhs, const fpos<_StateT> &__rhs)
    {
        return streamoff(__lhs) != streamoff(__rhs);
    }
    typedef fpos<mbstate_t> streampos;
    typedef fpos<mbstate_t> wstreampos;
}
namespace std __attribute__((__visibility__("default"))) {
    template<typename _CharT, typename _Traits = char_traits<_CharT> >
    class basic_ios;
    template<typename _CharT, typename _Traits = char_traits<_CharT> >
    class basic_streambuf;
    template<typename _CharT, typename _Traits = char_traits<_CharT> >
    class basic_istream;
    template<typename _CharT, typename _Traits = char_traits<_CharT> >
    class basic_ostream;
    template<typename _CharT, typename _Traits = char_traits<_CharT> >
    class basic_iostream;
    template<typename _CharT, typename _Traits = char_traits<_CharT>, typename _Alloc = allocator<_CharT> >
    class basic_stringbuf;
    template<typename _CharT, typename _Traits = char_traits<_CharT>, typename _Alloc = allocator<_CharT> >
    class basic_istringstream;
    template<typename _CharT, typename _Traits = char_traits<_CharT>, typename _Alloc = allocator<_CharT> >
    class basic_ostringstream;
    template<typename _CharT, typename _Traits = char_traits<_CharT>, typename _Alloc = allocator<_CharT> >
    class basic_stringstream;
    template<typename _CharT, typename _Traits = char_traits<_CharT> >
    class basic_filebuf;
    template<typename _CharT, typename _Traits = char_traits<_CharT> >
    class basic_ifstream;
    template<typename _CharT, typename _Traits = char_traits<_CharT> >
    class basic_ofstream;
    template<typename _CharT, typename _Traits = char_traits<_CharT> >
    class basic_fstream;
    template<typename _CharT, typename _Traits = char_traits<_CharT> >
    class istreambuf_iterator;
    template<typename _CharT, typename _Traits = char_traits<_CharT> >
    class ostreambuf_iterator;
    class ios_base;
    typedef basic_ios<char> ios;
    typedef basic_streambuf<char> streambuf;
    typedef basic_istream<char> istream;
    typedef basic_ostream<char> ostream;
    typedef basic_iostream<char> iostream;
    typedef basic_stringbuf<char> stringbuf;
    typedef basic_istringstream<char> istringstream;
    typedef basic_ostringstream<char> ostringstream;
    typedef basic_stringstream<char> stringstream;
    typedef basic_filebuf<char> filebuf;
    typedef basic_ifstream<char> ifstream;
    typedef basic_ofstream<char> ofstream;
    typedef basic_fstream<char> fstream;
    typedef basic_ios<wchar_t> wios;
    typedef basic_streambuf<wchar_t> wstreambuf;
    typedef basic_istream<wchar_t> wistream;
    typedef basic_ostream<wchar_t> wostream;
    typedef basic_iostream<wchar_t> wiostream;
    typedef basic_stringbuf<wchar_t> wstringbuf;
    typedef basic_istringstream<wchar_t> wistringstream;
    typedef basic_ostringstream<wchar_t> wostringstream;
    typedef basic_stringstream<wchar_t> wstringstream;
    typedef basic_filebuf<wchar_t> wfilebuf;
    typedef basic_ifstream<wchar_t> wifstream;
    typedef basic_ofstream<wchar_t> wofstream;
    typedef basic_fstream<wchar_t> wfstream;
}
namespace std __attribute__((__visibility__("default"))) {
    void __throw_bad_exception(void) __attribute__((__noreturn__));
    void __throw_bad_alloc(void) __attribute__((__noreturn__));
    void __throw_bad_cast(void) __attribute__((__noreturn__));
    void __throw_bad_typeid(void) __attribute__((__noreturn__));
    void __throw_logic_error(const char *) __attribute__((__noreturn__));
    void __throw_domain_error(const char *) __attribute__((__noreturn__));
    void __throw_invalid_argument(const char *) __attribute__((__noreturn__));
    void __throw_length_error(const char *) __attribute__((__noreturn__));
    void __throw_out_of_range(const char *) __attribute__((__noreturn__));
    void __throw_runtime_error(const char *) __attribute__((__noreturn__));
    void __throw_range_error(const char *) __attribute__((__noreturn__));
    void __throw_overflow_error(const char *) __attribute__((__noreturn__));
    void __throw_underflow_error(const char *) __attribute__((__noreturn__));
    void __throw_ios_failure(const char *) __attribute__((__noreturn__));
    void __throw_system_error(int) __attribute__((__noreturn__));
}
namespace __gnu_cxx __attribute__((__visibility__("default"))) {
    template<typename _Iterator, typename _Container >
    class __normal_iterator;
}
namespace std __attribute__((__visibility__("default"))) {
    struct __true_type
    {
    };
    struct __false_type
    {
    };
    template<bool >
    struct __truth_type
    {
            typedef __false_type __type;
    };
    template<>
    struct __truth_type<true>
    {
            typedef __true_type __type;
    };
    template<class _Sp, class _Tp >
    struct __traitor
    {
            enum 
            {
                __value = bool(_Sp::__value) || bool(_Tp::__value)
            };
            typedef typename __truth_type<__value>::__type __type;
    };
    template<typename , typename  >
    struct __are_same
    {
            enum 
            {
                __value = 0
            };
            typedef __false_type __type;
    };
    template<typename _Tp >
    struct __are_same<_Tp, _Tp>
    {
            enum 
            {
                __value = 1
            };
            typedef __true_type __type;
    };
    template<typename _Tp >
    struct __is_void
    {
            enum 
            {
                __value = 0
            };
            typedef __false_type __type;
    };
    template<>
    struct __is_void<void>
    {
            enum 
            {
                __value = 1
            };
            typedef __true_type __type;
    };
    template<typename _Tp >
    struct __is_integer
    {
            enum 
            {
                __value = 0
            };
            typedef __false_type __type;
    };
    template<>
    struct __is_integer<bool>
    {
            enum 
            {
                __value = 1
            };
            typedef __true_type __type;
    };
    template<>
    struct __is_integer<char>
    {
            enum 
            {
                __value = 1
            };
            typedef __true_type __type;
    };
    template<>
    struct __is_integer<signed char>
    {
            enum 
            {
                __value = 1
            };
            typedef __true_type __type;
    };
    template<>
    struct __is_integer<unsigned char>
    {
            enum 
            {
                __value = 1
            };
            typedef __true_type __type;
    };
    template<>
    struct __is_integer<wchar_t>
    {
            enum 
            {
                __value = 1
            };
            typedef __true_type __type;
    };
    template<>
    struct __is_integer<short>
    {
            enum 
            {
                __value = 1
            };
            typedef __true_type __type;
    };
    template<>
    struct __is_integer<unsigned short>
    {
            enum 
            {
                __value = 1
            };
            typedef __true_type __type;
    };
    template<>
    struct __is_integer<int>
    {
            enum 
            {
                __value = 1
            };
            typedef __true_type __type;
    };
    template<>
    struct __is_integer<unsigned int>
    {
            enum 
            {
                __value = 1
            };
            typedef __true_type __type;
    };
    template<>
    struct __is_integer<long>
    {
            enum 
            {
                __value = 1
            };
            typedef __true_type __type;
    };
    template<>
    struct __is_integer<unsigned long>
    {
            enum 
            {
                __value = 1
            };
            typedef __true_type __type;
    };
    template<>
    struct __is_integer<long long>
    {
            enum 
            {
                __value = 1
            };
            typedef __true_type __type;
    };
    template<>
    struct __is_integer<unsigned long long>
    {
            enum 
            {
                __value = 1
            };
            typedef __true_type __type;
    };
    template<typename _Tp >
    struct __is_floating
    {
            enum 
            {
                __value = 0
            };
            typedef __false_type __type;
    };
    template<>
    struct __is_floating<float>
    {
            enum 
            {
                __value = 1
            };
            typedef __true_type __type;
    };
    template<>
    struct __is_floating<double>
    {
            enum 
            {
                __value = 1
            };
            typedef __true_type __type;
    };
    template<>
    struct __is_floating<long double>
    {
            enum 
            {
                __value = 1
            };
            typedef __true_type __type;
    };
    template<typename _Tp >
    struct __is_pointer
    {
            enum 
            {
                __value = 0
            };
            typedef __false_type __type;
    };
    template<typename _Tp >
    struct __is_pointer<_Tp *>
    {
            enum 
            {
                __value = 1
            };
            typedef __true_type __type;
    };
    template<typename _Tp >
    struct __is_normal_iterator
    {
            enum 
            {
                __value = 0
            };
            typedef __false_type __type;
    };
    template<typename _Iterator, typename _Container >
    struct __is_normal_iterator<__gnu_cxx::__normal_iterator<_Iterator, _Container> >
    {
            enum 
            {
                __value = 1
            };
            typedef __true_type __type;
    };
    template<typename _Tp >
    struct __is_arithmetic : public __traitor<__is_integer<_Tp>, __is_floating<_Tp> >
    {
    };
    template<typename _Tp >
    struct __is_fundamental : public __traitor<__is_void<_Tp>, __is_arithmetic<_Tp> >
    {
    };
    template<typename _Tp >
    struct __is_scalar : public __traitor<__is_arithmetic<_Tp>, __is_pointer<_Tp> >
    {
    };
    template<typename _Tp >
    struct __is_char
    {
            enum 
            {
                __value = 0
            };
            typedef __false_type __type;
    };
    template<>
    struct __is_char<char>
    {
            enum 
            {
                __value = 1
            };
            typedef __true_type __type;
    };
    template<>
    struct __is_char<wchar_t>
    {
            enum 
            {
                __value = 1
            };
            typedef __true_type __type;
    };
    template<typename _Tp >
    struct __is_byte
    {
            enum 
            {
                __value = 0
            };
            typedef __false_type __type;
    };
    template<>
    struct __is_byte<char>
    {
            enum 
            {
                __value = 1
            };
            typedef __true_type __type;
    };
    template<>
    struct __is_byte<signed char>
    {
            enum 
            {
                __value = 1
            };
            typedef __true_type __type;
    };
    template<>
    struct __is_byte<unsigned char>
    {
            enum 
            {
                __value = 1
            };
            typedef __true_type __type;
    };
    template<typename _Tp >
    struct __is_move_iterator
    {
            enum 
            {
                __value = 0
            };
            typedef __false_type __type;
    };
}
namespace __gnu_cxx __attribute__((__visibility__("default"))) {
    template<bool, typename  >
    struct __enable_if
    {
    };
    template<typename _Tp >
    struct __enable_if<true, _Tp>
    {
            typedef _Tp __type;
    };
    template<bool _Cond, typename _Iftrue, typename _Iffalse >
    struct __conditional_type
    {
            typedef _Iftrue __type;
    };
    template<typename _Iftrue, typename _Iffalse >
    struct __conditional_type<false, _Iftrue, _Iffalse>
    {
            typedef _Iffalse __type;
    };
    template<typename _Tp >
    struct __add_unsigned
    {
        private :
            typedef __enable_if<std::__is_integer<_Tp>::__value, _Tp> __if_type;
        public :
            typedef typename __if_type::__type __type;
    };
    template<>
    struct __add_unsigned<char>
    {
            typedef unsigned char __type;
    };
    template<>
    struct __add_unsigned<signed char>
    {
            typedef unsigned char __type;
    };
    template<>
    struct __add_unsigned<short>
    {
            typedef unsigned short __type;
    };
    template<>
    struct __add_unsigned<int>
    {
            typedef unsigned int __type;
    };
    template<>
    struct __add_unsigned<long>
    {
            typedef unsigned long __type;
    };
    template<>
    struct __add_unsigned<long long>
    {
            typedef unsigned long long __type;
    };
    template<>
    struct __add_unsigned<bool>;
    template<>
    struct __add_unsigned<wchar_t>;
    template<typename _Tp >
    struct __remove_unsigned
    {
        private :
            typedef __enable_if<std::__is_integer<_Tp>::__value, _Tp> __if_type;
        public :
            typedef typename __if_type::__type __type;
    };
    template<>
    struct __remove_unsigned<char>
    {
            typedef signed char __type;
    };
    template<>
    struct __remove_unsigned<unsigned char>
    {
            typedef signed char __type;
    };
    template<>
    struct __remove_unsigned<unsigned short>
    {
            typedef short __type;
    };
    template<>
    struct __remove_unsigned<unsigned int>
    {
            typedef int __type;
    };
    template<>
    struct __remove_unsigned<unsigned long>
    {
            typedef long __type;
    };
    template<>
    struct __remove_unsigned<unsigned long long>
    {
            typedef long long __type;
    };
    template<>
    struct __remove_unsigned<bool>;
    template<>
    struct __remove_unsigned<wchar_t>;
    template<typename _Type >
    inline bool __is_null_pointer(_Type *__ptr)
    {
        return __ptr == 0;
    }
    template<typename _Type >
    inline bool __is_null_pointer(_Type)
    {
        return false;
    }
    template<typename _Tp, bool = std::__is_integer<_Tp>::__value >
    struct __promote
    {
            typedef double __type;
    };
    template<typename _Tp >
    struct __promote<_Tp, false>
    {
            typedef _Tp __type;
    };
    template<typename _Tp, typename _Up >
    struct __promote_2
    {
        private :
            typedef typename __promote<_Tp>::__type __type1;
            typedef typename __promote<_Up>::__type __type2;
        public :
            typedef __typeof__ (__type1() + __type2()) __type;
    };
    template<typename _Tp, typename _Up, typename _Vp >
    struct __promote_3
    {
        private :
            typedef typename __promote<_Tp>::__type __type1;
            typedef typename __promote<_Up>::__type __type2;
            typedef typename __promote<_Vp>::__type __type3;
        public :
            typedef __typeof__ (__type1() + __type2() + __type3()) __type;
    };
    template<typename _Tp, typename _Up, typename _Vp, typename _Wp >
    struct __promote_4
    {
        private :
            typedef typename __promote<_Tp>::__type __type1;
            typedef typename __promote<_Up>::__type __type2;
            typedef typename __promote<_Vp>::__type __type3;
            typedef typename __promote<_Wp>::__type __type4;
        public :
            typedef __typeof__ (__type1() + __type2() + __type3() + __type4()) __type;
    };
}
namespace __gnu_cxx __attribute__((__visibility__("default"))) {
    template<typename _Value >
    struct __numeric_traits_integer
    {
            static const _Value __min  = (((_Value) (- 1) < 0) ? (_Value) 1 << (sizeof(_Value) * 8 - ((_Value) (- 1) < 0)) : (_Value) 0);
            static const _Value __max  = (((_Value) (- 1) < 0) ? (((((_Value) 1 << ((sizeof(_Value) * 8 - ((_Value) (- 1) < 0)) - 1)) - 1) << 1) + 1) : ~(_Value) 0);
            static const bool __is_signed  = ((_Value) (- 1) < 0);
            static const int __digits  = (sizeof(_Value) * 8 - ((_Value) (- 1) < 0));
    };
    template<typename _Value >
    const _Value __numeric_traits_integer<_Value>::__min;
    template<typename _Value >
    const _Value __numeric_traits_integer<_Value>::__max;
    template<typename _Value >
    const bool __numeric_traits_integer<_Value>::__is_signed;
    template<typename _Value >
    const int __numeric_traits_integer<_Value>::__digits;
    template<typename _Value >
    struct __numeric_traits_floating
    {
            static const int __max_digits10  = (2 + (std::__are_same<_Value, float>::__value ? 24 : std::__are_same<_Value, double>::__value ? 53 : 64) * 3010 / 10000);
            static const bool __is_signed  = true;
            static const int __digits10  = (std::__are_same<_Value, float>::__value ? 6 : std::__are_same<_Value, double>::__value ? 15 : 18);
            static const int __max_exponent10  = (std::__are_same<_Value, float>::__value ? 38 : std::__are_same<_Value, double>::__value ? 308 : 4932);
    };
    template<typename _Value >
    const int __numeric_traits_floating<_Value>::__max_digits10;
    template<typename _Value >
    const bool __numeric_traits_floating<_Value>::__is_signed;
    template<typename _Value >
    const int __numeric_traits_floating<_Value>::__digits10;
    template<typename _Value >
    const int __numeric_traits_floating<_Value>::__max_exponent10;
    template<typename _Value >
    struct __numeric_traits : public __conditional_type<std::__is_integer<_Value>::__value, __numeric_traits_integer<_Value>, __numeric_traits_floating<_Value> >::__type
    {
    };
}
namespace std __attribute__((__visibility__("default"))) {
    template<typename _Tp >
    inline void swap(_Tp &__a, _Tp &__b)
    {
        _Tp __tmp = (__a);
        __a = (__b);
        __b = (__tmp);
    }
    template<typename _Tp, size_t _Nm >
    inline void swap(_Tp (&__a)[_Nm], _Tp (&__b)[_Nm])
    {
        for (size_t __n = 0;
            __n < _Nm;
            ++__n)
            swap(__a[__n], __b[__n]);
    }
}
namespace std __attribute__((__visibility__("default"))) {
    template<class _T1, class _T2 >
    struct pair
    {
            typedef _T1 first_type;
            typedef _T2 second_type;
            _T1 first;
            _T2 second;
            pair()
                : first(), second() 
            {
            }
            pair(const _T1 &__a, const _T2 &__b)
                : first(__a), second(__b) 
            {
            }
            template<class _U1, class _U2 >
            pair(const pair<_U1, _U2> &__p)
                : first(__p.first), second(__p.second) 
            {
            }
    };
    template<class _T1, class _T2 >
    inline bool operator ==(const pair<_T1, _T2> &__x, const pair<_T1, _T2> &__y)
    {
        return __x.first == __y.first && __x.second == __y.second;
    }
    template<class _T1, class _T2 >
    inline bool operator <(const pair<_T1, _T2> &__x, const pair<_T1, _T2> &__y)
    {
        return __x.first < __y.first || (!(__y.first < __x.first) && __x.second < __y.second);
    }
    template<class _T1, class _T2 >
    inline bool operator !=(const pair<_T1, _T2> &__x, const pair<_T1, _T2> &__y)
    {
        return !(__x == __y);
    }
    template<class _T1, class _T2 >
    inline bool operator >(const pair<_T1, _T2> &__x, const pair<_T1, _T2> &__y)
    {
        return __y < __x;
    }
    template<class _T1, class _T2 >
    inline bool operator <=(const pair<_T1, _T2> &__x, const pair<_T1, _T2> &__y)
    {
        return !(__y < __x);
    }
    template<class _T1, class _T2 >
    inline bool operator >=(const pair<_T1, _T2> &__x, const pair<_T1, _T2> &__y)
    {
        return !(__x < __y);
    }
    template<class _T1, class _T2 >
    inline pair<_T1, _T2> make_pair(_T1 __x, _T2 __y)
    {
        return pair<_T1, _T2>(__x, __y);
    }
}
namespace std __attribute__((__visibility__("default"))) {
    struct input_iterator_tag
    {
    };
    struct output_iterator_tag
    {
    };
    struct forward_iterator_tag : public input_iterator_tag
    {
    };
    struct bidirectional_iterator_tag : public forward_iterator_tag
    {
    };
    struct random_access_iterator_tag : public bidirectional_iterator_tag
    {
    };
    template<typename _Category, typename _Tp, typename _Distance = ptrdiff_t, typename _Pointer = _Tp *, typename _Reference = _Tp & >
    struct iterator
    {
            typedef _Category iterator_category;
            typedef _Tp value_type;
            typedef _Distance difference_type;
            typedef _Pointer pointer;
            typedef _Reference reference;
    };
    template<typename _Iterator >
    struct iterator_traits
    {
            typedef typename _Iterator::iterator_category iterator_category;
            typedef typename _Iterator::value_type value_type;
            typedef typename _Iterator::difference_type difference_type;
            typedef typename _Iterator::pointer pointer;
            typedef typename _Iterator::reference reference;
    };
    template<typename _Tp >
    struct iterator_traits<_Tp *>
    {
            typedef random_access_iterator_tag iterator_category;
            typedef _Tp value_type;
            typedef ptrdiff_t difference_type;
            typedef _Tp *pointer;
            typedef _Tp &reference;
    };
    template<typename _Tp >
    struct iterator_traits<const _Tp *>
    {
            typedef random_access_iterator_tag iterator_category;
            typedef _Tp value_type;
            typedef ptrdiff_t difference_type;
            typedef const _Tp *pointer;
            typedef const _Tp &reference;
    };
    template<typename _Iter >
    inline typename iterator_traits<_Iter>::iterator_category __iterator_category(const _Iter &)
    {
        return typename iterator_traits<_Iter>::iterator_category();
    }
}
namespace std __attribute__((__visibility__("default"))) {
    template<typename _InputIterator >
    inline typename iterator_traits<_InputIterator>::difference_type __distance(_InputIterator __first, _InputIterator __last, input_iterator_tag)
    {
        typename iterator_traits<_InputIterator>::difference_type __n = 0;
        while (__first != __last)
        {
            ++__first;
            ++__n;
        }
        return __n;
    }
    template<typename _RandomAccessIterator >
    inline typename iterator_traits<_RandomAccessIterator>::difference_type __distance(_RandomAccessIterator __first, _RandomAccessIterator __last, random_access_iterator_tag)
    {
        return __last - __first;
    }
    template<typename _InputIterator >
    inline typename iterator_traits<_InputIterator>::difference_type distance(_InputIterator __first, _InputIterator __last)
    {
        return std::__distance(__first, __last, std::__iterator_category(__first));
    }
    template<typename _InputIterator, typename _Distance >
    inline void __advance(_InputIterator &__i, _Distance __n, input_iterator_tag)
    {
        while (__n--)
            ++__i;
    }
    template<typename _BidirectionalIterator, typename _Distance >
    inline void __advance(_BidirectionalIterator &__i, _Distance __n, bidirectional_iterator_tag)
    {
        if (__n > 0)
            while (__n--)
                ++__i;
        else
            while (__n++)
                --__i;
    }
    template<typename _RandomAccessIterator, typename _Distance >
    inline void __advance(_RandomAccessIterator &__i, _Distance __n, random_access_iterator_tag)
    {
        __i += __n;
    }
    template<typename _InputIterator, typename _Distance >
    inline void advance(_InputIterator &__i, _Distance __n)
    {
        typename iterator_traits<_InputIterator>::difference_type __d = __n;
        std::__advance(__i, __d, std::__iterator_category(__i));
    }
}
namespace std __attribute__((__visibility__("default"))) {
    template<typename _Iterator >
    class reverse_iterator : public iterator<typename iterator_traits<_Iterator>::iterator_category, typename iterator_traits<_Iterator>::value_type, typename iterator_traits<_Iterator>::difference_type, typename iterator_traits<_Iterator>::pointer, typename iterator_traits<_Iterator>::reference>
    {
        protected :
            _Iterator current;
        public :
            typedef _Iterator iterator_type;
            typedef typename iterator_traits<_Iterator>::difference_type difference_type;
            typedef typename iterator_traits<_Iterator>::reference reference;
            typedef typename iterator_traits<_Iterator>::pointer pointer;
        public :
            reverse_iterator()
                : current() 
            {
            }
            explicit reverse_iterator(iterator_type __x)
                : current(__x) 
            {
            }
            reverse_iterator(const reverse_iterator &__x)
                : current(__x.current) 
            {
            }
            template<typename _Iter >
            reverse_iterator(const reverse_iterator<_Iter> &__x)
                : current(__x.base()) 
            {
            }
            iterator_type base() const
            {
                return current;
            }
            reference operator *() const
            {
                _Iterator __tmp = current;
                return *--__tmp;
            }
            pointer operator ->() const
            {
                return &(operator *());
            }
            reverse_iterator &operator ++()
            {
                --current;
                return *this;
            }
            reverse_iterator operator ++(int)
            {
                reverse_iterator __tmp = *this;
                --current;
                return __tmp;
            }
            reverse_iterator &operator --()
            {
                ++current;
                return *this;
            }
            reverse_iterator operator --(int)
            {
                reverse_iterator __tmp = *this;
                ++current;
                return __tmp;
            }
            reverse_iterator operator +(difference_type __n) const
            {
                return reverse_iterator(current - __n);
            }
            reverse_iterator &operator +=(difference_type __n)
            {
                current -= __n;
                return *this;
            }
            reverse_iterator operator -(difference_type __n) const
            {
                return reverse_iterator(current + __n);
            }
            reverse_iterator &operator -=(difference_type __n)
            {
                current += __n;
                return *this;
            }
            reference operator [](difference_type __n) const
            {
                return *(*this + __n);
            }
    };
    template<typename _Iterator >
    inline bool operator ==(const reverse_iterator<_Iterator> &__x, const reverse_iterator<_Iterator> &__y)
    {
        return __x.base() == __y.base();
    }
    template<typename _Iterator >
    inline bool operator <(const reverse_iterator<_Iterator> &__x, const reverse_iterator<_Iterator> &__y)
    {
        return __y.base() < __x.base();
    }
    template<typename _Iterator >
    inline bool operator !=(const reverse_iterator<_Iterator> &__x, const reverse_iterator<_Iterator> &__y)
    {
        return !(__x == __y);
    }
    template<typename _Iterator >
    inline bool operator >(const reverse_iterator<_Iterator> &__x, const reverse_iterator<_Iterator> &__y)
    {
        return __y < __x;
    }
    template<typename _Iterator >
    inline bool operator <=(const reverse_iterator<_Iterator> &__x, const reverse_iterator<_Iterator> &__y)
    {
        return !(__y < __x);
    }
    template<typename _Iterator >
    inline bool operator >=(const reverse_iterator<_Iterator> &__x, const reverse_iterator<_Iterator> &__y)
    {
        return !(__x < __y);
    }
    template<typename _Iterator >
    inline typename reverse_iterator<_Iterator>::difference_type operator -(const reverse_iterator<_Iterator> &__x, const reverse_iterator<_Iterator> &__y)
    {
        return __y.base() - __x.base();
    }
    template<typename _Iterator >
    inline reverse_iterator<_Iterator> operator +(typename reverse_iterator<_Iterator>::difference_type __n, const reverse_iterator<_Iterator> &__x)
    {
        return reverse_iterator<_Iterator>(__x.base() - __n);
    }
    template<typename _IteratorL, typename _IteratorR >
    inline bool operator ==(const reverse_iterator<_IteratorL> &__x, const reverse_iterator<_IteratorR> &__y)
    {
        return __x.base() == __y.base();
    }
    template<typename _IteratorL, typename _IteratorR >
    inline bool operator <(const reverse_iterator<_IteratorL> &__x, const reverse_iterator<_IteratorR> &__y)
    {
        return __y.base() < __x.base();
    }
    template<typename _IteratorL, typename _IteratorR >
    inline bool operator !=(const reverse_iterator<_IteratorL> &__x, const reverse_iterator<_IteratorR> &__y)
    {
        return !(__x == __y);
    }
    template<typename _IteratorL, typename _IteratorR >
    inline bool operator >(const reverse_iterator<_IteratorL> &__x, const reverse_iterator<_IteratorR> &__y)
    {
        return __y < __x;
    }
    template<typename _IteratorL, typename _IteratorR >
    inline bool operator <=(const reverse_iterator<_IteratorL> &__x, const reverse_iterator<_IteratorR> &__y)
    {
        return !(__y < __x);
    }
    template<typename _IteratorL, typename _IteratorR >
    inline bool operator >=(const reverse_iterator<_IteratorL> &__x, const reverse_iterator<_IteratorR> &__y)
    {
        return !(__x < __y);
    }
    template<typename _IteratorL, typename _IteratorR >
    inline typename reverse_iterator<_IteratorL>::difference_type operator -(const reverse_iterator<_IteratorL> &__x, const reverse_iterator<_IteratorR> &__y)
    {
        return __y.base() - __x.base();
    }
    template<typename _Container >
    class back_insert_iterator : public iterator<output_iterator_tag, void, void, void, void>
    {
        protected :
            _Container *container;
        public :
            typedef _Container container_type;
            explicit back_insert_iterator(_Container &__x)
                : container(&__x) 
            {
            }
            back_insert_iterator &operator =(typename _Container::const_reference __value)
            {
                container->push_back(__value);
                return *this;
            }
            back_insert_iterator &operator *()
            {
                return *this;
            }
            back_insert_iterator &operator ++()
            {
                return *this;
            }
            back_insert_iterator operator ++(int)
            {
                return *this;
            }
    };
    template<typename _Container >
    inline back_insert_iterator<_Container> back_inserter(_Container &__x)
    {
        return back_insert_iterator<_Container>(__x);
    }
    template<typename _Container >
    class front_insert_iterator : public iterator<output_iterator_tag, void, void, void, void>
    {
        protected :
            _Container *container;
        public :
            typedef _Container container_type;
            explicit front_insert_iterator(_Container &__x)
                : container(&__x) 
            {
            }
            front_insert_iterator &operator =(typename _Container::const_reference __value)
            {
                container->push_front(__value);
                return *this;
            }
            front_insert_iterator &operator *()
            {
                return *this;
            }
            front_insert_iterator &operator ++()
            {
                return *this;
            }
            front_insert_iterator operator ++(int)
            {
                return *this;
            }
    };
    template<typename _Container >
    inline front_insert_iterator<_Container> front_inserter(_Container &__x)
    {
        return front_insert_iterator<_Container>(__x);
    }
    template<typename _Container >
    class insert_iterator : public iterator<output_iterator_tag, void, void, void, void>
    {
        protected :
            _Container *container;
            typename _Container::iterator iter;
        public :
            typedef _Container container_type;
            insert_iterator(_Container &__x, typename _Container::iterator __i)
                : container(&__x), iter(__i) 
            {
            }
            insert_iterator &operator =(typename _Container::const_reference __value)
            {
                iter = container->insert(iter, __value);
                ++iter;
                return *this;
            }
            insert_iterator &operator *()
            {
                return *this;
            }
            insert_iterator &operator ++()
            {
                return *this;
            }
            insert_iterator &operator ++(int)
            {
                return *this;
            }
    };
    template<typename _Container, typename _Iterator >
    inline insert_iterator<_Container> inserter(_Container &__x, _Iterator __i)
    {
        return insert_iterator<_Container>(__x, typename _Container::iterator(__i));
    }
}
namespace __gnu_cxx __attribute__((__visibility__("default"))) {
    using std::iterator_traits;
    using std::iterator;
    template<typename _Iterator, typename _Container >
    class __normal_iterator
    {
        protected :
            _Iterator _M_current;
        public :
            typedef _Iterator iterator_type;
            typedef typename iterator_traits<_Iterator>::iterator_category iterator_category;
            typedef typename iterator_traits<_Iterator>::value_type value_type;
            typedef typename iterator_traits<_Iterator>::difference_type difference_type;
            typedef typename iterator_traits<_Iterator>::reference reference;
            typedef typename iterator_traits<_Iterator>::pointer pointer;
            __normal_iterator()
                : _M_current(_Iterator()) 
            {
            }
            explicit __normal_iterator(const _Iterator &__i)
                : _M_current(__i) 
            {
            }
            template<typename _Iter >
            __normal_iterator(const __normal_iterator<_Iter, typename __enable_if<(std::__are_same<_Iter, typename _Container::pointer>::__value), _Container>::__type> &__i)
                : _M_current(__i.base()) 
            {
            }
            reference operator *() const
            {
                return *_M_current;
            }
            pointer operator ->() const
            {
                return _M_current;
            }
            __normal_iterator &operator ++()
            {
                ++_M_current;
                return *this;
            }
            __normal_iterator operator ++(int)
            {
                return __normal_iterator(_M_current++);
            }
            __normal_iterator &operator --()
            {
                --_M_current;
                return *this;
            }
            __normal_iterator operator --(int)
            {
                return __normal_iterator(_M_current--);
            }
            reference operator [](const difference_type &__n) const
            {
                return _M_current[__n];
            }
            __normal_iterator &operator +=(const difference_type &__n)
            {
                _M_current += __n;
                return *this;
            }
            __normal_iterator operator +(const difference_type &__n) const
            {
                return __normal_iterator(_M_current + __n);
            }
            __normal_iterator &operator -=(const difference_type &__n)
            {
                _M_current -= __n;
                return *this;
            }
            __normal_iterator operator -(const difference_type &__n) const
            {
                return __normal_iterator(_M_current - __n);
            }
            const _Iterator &base() const
            {
                return _M_current;
            }
    };
    template<typename _IteratorL, typename _IteratorR, typename _Container >
    inline bool operator ==(const __normal_iterator<_IteratorL, _Container> &__lhs, const __normal_iterator<_IteratorR, _Container> &__rhs)
    {
        return __lhs.base() == __rhs.base();
    }
    template<typename _Iterator, typename _Container >
    inline bool operator ==(const __normal_iterator<_Iterator, _Container> &__lhs, const __normal_iterator<_Iterator, _Container> &__rhs)
    {
        return __lhs.base() == __rhs.base();
    }
    template<typename _IteratorL, typename _IteratorR, typename _Container >
    inline bool operator !=(const __normal_iterator<_IteratorL, _Container> &__lhs, const __normal_iterator<_IteratorR, _Container> &__rhs)
    {
        return __lhs.base() != __rhs.base();
    }
    template<typename _Iterator, typename _Container >
    inline bool operator !=(const __normal_iterator<_Iterator, _Container> &__lhs, const __normal_iterator<_Iterator, _Container> &__rhs)
    {
        return __lhs.base() != __rhs.base();
    }
    template<typename _IteratorL, typename _IteratorR, typename _Container >
    inline bool operator <(const __normal_iterator<_IteratorL, _Container> &__lhs, const __normal_iterator<_IteratorR, _Container> &__rhs)
    {
        return __lhs.base() < __rhs.base();
    }
    template<typename _Iterator, typename _Container >
    inline bool operator <(const __normal_iterator<_Iterator, _Container> &__lhs, const __normal_iterator<_Iterator, _Container> &__rhs)
    {
        return __lhs.base() < __rhs.base();
    }
    template<typename _IteratorL, typename _IteratorR, typename _Container >
    inline bool operator >(const __normal_iterator<_IteratorL, _Container> &__lhs, const __normal_iterator<_IteratorR, _Container> &__rhs)
    {
        return __lhs.base() > __rhs.base();
    }
    template<typename _Iterator, typename _Container >
    inline bool operator >(const __normal_iterator<_Iterator, _Container> &__lhs, const __normal_iterator<_Iterator, _Container> &__rhs)
    {
        return __lhs.base() > __rhs.base();
    }
    template<typename _IteratorL, typename _IteratorR, typename _Container >
    inline bool operator <=(const __normal_iterator<_IteratorL, _Container> &__lhs, const __normal_iterator<_IteratorR, _Container> &__rhs)
    {
        return __lhs.base() <= __rhs.base();
    }
    template<typename _Iterator, typename _Container >
    inline bool operator <=(const __normal_iterator<_Iterator, _Container> &__lhs, const __normal_iterator<_Iterator, _Container> &__rhs)
    {
        return __lhs.base() <= __rhs.base();
    }
    template<typename _IteratorL, typename _IteratorR, typename _Container >
    inline bool operator >=(const __normal_iterator<_IteratorL, _Container> &__lhs, const __normal_iterator<_IteratorR, _Container> &__rhs)
    {
        return __lhs.base() >= __rhs.base();
    }
    template<typename _Iterator, typename _Container >
    inline bool operator >=(const __normal_iterator<_Iterator, _Container> &__lhs, const __normal_iterator<_Iterator, _Container> &__rhs)
    {
        return __lhs.base() >= __rhs.base();
    }
    template<typename _IteratorL, typename _IteratorR, typename _Container >
    inline typename __normal_iterator<_IteratorL, _Container>::difference_type operator -(const __normal_iterator<_IteratorL, _Container> &__lhs, const __normal_iterator<_IteratorR, _Container> &__rhs)
    {
        return __lhs.base() - __rhs.base();
    }
    template<typename _Iterator, typename _Container >
    inline typename __normal_iterator<_Iterator, _Container>::difference_type operator -(const __normal_iterator<_Iterator, _Container> &__lhs, const __normal_iterator<_Iterator, _Container> &__rhs)
    {
        return __lhs.base() - __rhs.base();
    }
    template<typename _Iterator, typename _Container >
    inline __normal_iterator<_Iterator, _Container> operator +(typename __normal_iterator<_Iterator, _Container>::difference_type __n, const __normal_iterator<_Iterator, _Container> &__i)
    {
        return __normal_iterator<_Iterator, _Container>(__i.base() + __n);
    }
}
namespace std {
    namespace __debug {
    }
}
namespace __gnu_debug {
    using namespace std::__debug;
}
namespace std __attribute__((__visibility__("default"))) {
    template<bool _BoolType >
    struct __iter_swap
    {
            template<typename _ForwardIterator1, typename _ForwardIterator2 >
            static void iter_swap(_ForwardIterator1 __a, _ForwardIterator2 __b)
            {
                typedef typename iterator_traits<_ForwardIterator1>::value_type _ValueType1;
                _ValueType1 __tmp = (*__a);
                *__a = (*__b);
                *__b = (__tmp);
            }
    };
    template<>
    struct __iter_swap<true>
    {
            template<typename _ForwardIterator1, typename _ForwardIterator2 >
            static void iter_swap(_ForwardIterator1 __a, _ForwardIterator2 __b)
            {
                swap(*__a, *__b);
            }
    };
    template<typename _ForwardIterator1, typename _ForwardIterator2 >
    inline void iter_swap(_ForwardIterator1 __a, _ForwardIterator2 __b)
    {
        typedef typename iterator_traits<_ForwardIterator1>::value_type _ValueType1;
        typedef typename iterator_traits<_ForwardIterator2>::value_type _ValueType2;
        typedef typename iterator_traits<_ForwardIterator1>::reference _ReferenceType1;
        typedef typename iterator_traits<_ForwardIterator2>::reference _ReferenceType2;
        std::__iter_swap<__are_same<_ValueType1, _ValueType2>::__value && __are_same<_ValueType1 &, _ReferenceType1>::__value && __are_same<_ValueType2 &, _ReferenceType2>::__value>::iter_swap(__a, __b);
    }
    template<typename _ForwardIterator1, typename _ForwardIterator2 >
    _ForwardIterator2 swap_ranges(_ForwardIterator1 __first1, _ForwardIterator1 __last1, _ForwardIterator2 __first2)
    {
        ;
        for (;
            __first1 != __last1;
            ++__first1 , ++__first2)
            std::iter_swap(__first1, __first2);
        return __first2;
    }
    template<typename _Tp >
    inline const _Tp &min(const _Tp &__a, const _Tp &__b)
    {
        if (__b < __a)
            return __b;
        return __a;
    }
    template<typename _Tp >
    inline const _Tp &max(const _Tp &__a, const _Tp &__b)
    {
        if (__a < __b)
            return __b;
        return __a;
    }
    template<typename _Tp, typename _Compare >
    inline const _Tp &min(const _Tp &__a, const _Tp &__b, _Compare __comp)
    {
        if (__comp(__b, __a))
            return __b;
        return __a;
    }
    template<typename _Tp, typename _Compare >
    inline const _Tp &max(const _Tp &__a, const _Tp &__b, _Compare __comp)
    {
        if (__comp(__a, __b))
            return __b;
        return __a;
    }
    template<typename _Iterator, bool _IsNormal = __is_normal_iterator<_Iterator>::__value >
    struct __niter_base
    {
            static _Iterator __b(_Iterator __it)
            {
                return __it;
            }
    };
    template<typename _Iterator >
    struct __niter_base<_Iterator, true>
    {
            static typename _Iterator::iterator_type __b(_Iterator __it)
            {
                return __it.base();
            }
    };
    template<typename _Iterator, bool _IsMove = __is_move_iterator<_Iterator>::__value >
    struct __miter_base
    {
            static _Iterator __b(_Iterator __it)
            {
                return __it;
            }
    };
    template<typename _Iterator >
    struct __miter_base<_Iterator, true>
    {
            static typename _Iterator::iterator_type __b(_Iterator __it)
            {
                return __it.base();
            }
    };
    template<bool, bool, typename  >
    struct __copy_move
    {
            template<typename _II, typename _OI >
            static _OI __copy_m(_II __first, _II __last, _OI __result)
            {
                for (;
                    __first != __last;
                    ++__result , ++__first)
                    *__result = *__first;
                return __result;
            }
    };
    template<>
    struct __copy_move<false, false, random_access_iterator_tag>
    {
            template<typename _II, typename _OI >
            static _OI __copy_m(_II __first, _II __last, _OI __result)
            {
                typedef typename iterator_traits<_II>::difference_type _Distance;
                for (_Distance __n = __last - __first;
                    __n > 0;
                    --__n)
                {
                    *__result = *__first;
                    ++__first;
                    ++__result;
                }
                return __result;
            }
    };
    template<bool _IsMove >
    struct __copy_move<_IsMove, true, random_access_iterator_tag>
    {
            template<typename _Tp >
            static _Tp *__copy_m(const _Tp *__first, const _Tp *__last, _Tp *__result)
            {
                __builtin_memmove(__result, __first, sizeof(_Tp) * (__last - __first));
                return __result + (__last - __first);
            }
    };
    template<bool _IsMove, typename _II, typename _OI >
    inline _OI __copy_move_a(_II __first, _II __last, _OI __result)
    {
        typedef typename iterator_traits<_II>::value_type _ValueTypeI;
        typedef typename iterator_traits<_OI>::value_type _ValueTypeO;
        typedef typename iterator_traits<_II>::iterator_category _Category;
        const bool __simple = (__is_pod(_ValueTypeI) && __is_pointer<_II>::__value && __is_pointer<_OI>::__value && __are_same<_ValueTypeI, _ValueTypeO>::__value);
        return std::__copy_move<_IsMove, __simple, _Category>::__copy_m(__first, __last, __result);
    }
    template<typename _CharT >
    struct char_traits;
    template<typename _CharT, typename _Traits >
    class istreambuf_iterator;
    template<typename _CharT, typename _Traits >
    class ostreambuf_iterator;
    template<bool _IsMove, typename _CharT >
    typename __gnu_cxx::__enable_if<__is_char<_CharT>::__value, ostreambuf_iterator<_CharT, char_traits<_CharT> > >::__type __copy_move_a2(_CharT *, _CharT *, ostreambuf_iterator<_CharT, char_traits<_CharT> >);
    template<bool _IsMove, typename _CharT >
    typename __gnu_cxx::__enable_if<__is_char<_CharT>::__value, ostreambuf_iterator<_CharT, char_traits<_CharT> > >::__type __copy_move_a2(const _CharT *, const _CharT *, ostreambuf_iterator<_CharT, char_traits<_CharT> >);
    template<bool _IsMove, typename _CharT >
    typename __gnu_cxx::__enable_if<__is_char<_CharT>::__value, _CharT *>::__type __copy_move_a2(istreambuf_iterator<_CharT, char_traits<_CharT> >, istreambuf_iterator<_CharT, char_traits<_CharT> >, _CharT *);
    template<bool _IsMove, typename _II, typename _OI >
    inline _OI __copy_move_a2(_II __first, _II __last, _OI __result)
    {
        return _OI(std::__copy_move_a<_IsMove>(std::__niter_base<_II>::__b(__first), std::__niter_base<_II>::__b(__last), std::__niter_base<_OI>::__b(__result)));
    }
    template<typename _II, typename _OI >
    inline _OI copy(_II __first, _II __last, _OI __result)
    {
        ;
        return (std::__copy_move_a2<__is_move_iterator<_II>::__value>(std::__miter_base<_II>::__b(__first), std::__miter_base<_II>::__b(__last), __result));
    }
    template<bool, bool, typename  >
    struct __copy_move_backward
    {
            template<typename _BI1, typename _BI2 >
            static _BI2 __copy_move_b(_BI1 __first, _BI1 __last, _BI2 __result)
            {
                while (__first != __last)
                    *--__result = *--__last;
                return __result;
            }
    };
    template<>
    struct __copy_move_backward<false, false, random_access_iterator_tag>
    {
            template<typename _BI1, typename _BI2 >
            static _BI2 __copy_move_b(_BI1 __first, _BI1 __last, _BI2 __result)
            {
                typename iterator_traits<_BI1>::difference_type __n;
                for (__n = __last - __first;
                    __n > 0;
                    --__n)
                    *--__result = *--__last;
                return __result;
            }
    };
    template<bool _IsMove >
    struct __copy_move_backward<_IsMove, true, random_access_iterator_tag>
    {
            template<typename _Tp >
            static _Tp *__copy_move_b(const _Tp *__first, const _Tp *__last, _Tp *__result)
            {
                const ptrdiff_t _Num = __last - __first;
                __builtin_memmove(__result - _Num, __first, sizeof(_Tp) * _Num);
                return __result - _Num;
            }
    };
    template<bool _IsMove, typename _BI1, typename _BI2 >
    inline _BI2 __copy_move_backward_a(_BI1 __first, _BI1 __last, _BI2 __result)
    {
        typedef typename iterator_traits<_BI1>::value_type _ValueType1;
        typedef typename iterator_traits<_BI2>::value_type _ValueType2;
        typedef typename iterator_traits<_BI1>::iterator_category _Category;
        const bool __simple = (__is_pod(_ValueType1) && __is_pointer<_BI1>::__value && __is_pointer<_BI2>::__value && __are_same<_ValueType1, _ValueType2>::__value);
        return std::__copy_move_backward<_IsMove, __simple, _Category>::__copy_move_b(__first, __last, __result);
    }
    template<bool _IsMove, typename _BI1, typename _BI2 >
    inline _BI2 __copy_move_backward_a2(_BI1 __first, _BI1 __last, _BI2 __result)
    {
        return _BI2(std::__copy_move_backward_a<_IsMove>(std::__niter_base<_BI1>::__b(__first), std::__niter_base<_BI1>::__b(__last), std::__niter_base<_BI2>::__b(__result)));
    }
    template<typename _BI1, typename _BI2 >
    inline _BI2 copy_backward(_BI1 __first, _BI1 __last, _BI2 __result)
    {
        ;
        return (std::__copy_move_backward_a2<__is_move_iterator<_BI1>::__value>(std::__miter_base<_BI1>::__b(__first), std::__miter_base<_BI1>::__b(__last), __result));
    }
    template<typename _ForwardIterator, typename _Tp >
    inline typename __gnu_cxx::__enable_if<!__is_scalar<_Tp>::__value, void>::__type __fill_a(_ForwardIterator __first, _ForwardIterator __last, const _Tp &__value)
    {
        for (;
            __first != __last;
            ++__first)
            *__first = __value;
    }
    template<typename _ForwardIterator, typename _Tp >
    inline typename __gnu_cxx::__enable_if<__is_scalar<_Tp>::__value, void>::__type __fill_a(_ForwardIterator __first, _ForwardIterator __last, const _Tp &__value)
    {
        const _Tp __tmp = __value;
        for (;
            __first != __last;
            ++__first)
            *__first = __tmp;
    }
    template<typename _Tp >
    inline typename __gnu_cxx::__enable_if<__is_byte<_Tp>::__value, void>::__type __fill_a(_Tp *__first, _Tp *__last, const _Tp &__c)
    {
        const _Tp __tmp = __c;
        __builtin_memset(__first, static_cast<unsigned char >(__tmp), __last - __first);
    }
    template<typename _ForwardIterator, typename _Tp >
    inline void fill(_ForwardIterator __first, _ForwardIterator __last, const _Tp &__value)
    {
        ;
        std::__fill_a(std::__niter_base<_ForwardIterator>::__b(__first), std::__niter_base<_ForwardIterator>::__b(__last), __value);
    }
    template<typename _OutputIterator, typename _Size, typename _Tp >
    inline typename __gnu_cxx::__enable_if<!__is_scalar<_Tp>::__value, _OutputIterator>::__type __fill_n_a(_OutputIterator __first, _Size __n, const _Tp &__value)
    {
        for (;
            __n > 0;
            --__n , ++__first)
            *__first = __value;
        return __first;
    }
    template<typename _OutputIterator, typename _Size, typename _Tp >
    inline typename __gnu_cxx::__enable_if<__is_scalar<_Tp>::__value, _OutputIterator>::__type __fill_n_a(_OutputIterator __first, _Size __n, const _Tp &__value)
    {
        const _Tp __tmp = __value;
        for (;
            __n > 0;
            --__n , ++__first)
            *__first = __tmp;
        return __first;
    }
    template<typename _Size, typename _Tp >
    inline typename __gnu_cxx::__enable_if<__is_byte<_Tp>::__value, _Tp *>::__type __fill_n_a(_Tp *__first, _Size __n, const _Tp &__c)
    {
        std::__fill_a(__first, __first + __n, __c);
        return __first + __n;
    }
    template<typename _OI, typename _Size, typename _Tp >
    inline _OI fill_n(_OI __first, _Size __n, const _Tp &__value)
    {
        return _OI(std::__fill_n_a(std::__niter_base<_OI>::__b(__first), __n, __value));
    }
    template<bool _BoolType >
    struct __equal
    {
            template<typename _II1, typename _II2 >
            static bool equal(_II1 __first1, _II1 __last1, _II2 __first2)
            {
                for (;
                    __first1 != __last1;
                    ++__first1 , ++__first2)
                    if (!(*__first1 == *__first2))
                        return false;
                return true;
            }
    };
    template<>
    struct __equal<true>
    {
            template<typename _Tp >
            static bool equal(const _Tp *__first1, const _Tp *__last1, const _Tp *__first2)
            {
                return !__builtin_memcmp(__first1, __first2, sizeof(_Tp) * (__last1 - __first1));
            }
    };
    template<typename _II1, typename _II2 >
    inline bool __equal_aux(_II1 __first1, _II1 __last1, _II2 __first2)
    {
        typedef typename iterator_traits<_II1>::value_type _ValueType1;
        typedef typename iterator_traits<_II2>::value_type _ValueType2;
        const bool __simple = (__is_integer<_ValueType1>::__value && __is_pointer<_II1>::__value && __is_pointer<_II2>::__value && __are_same<_ValueType1, _ValueType2>::__value);
        return std::__equal<__simple>::equal(__first1, __last1, __first2);
    }
    template<typename , typename  >
    struct __lc_rai
    {
            template<typename _II1, typename _II2 >
            static _II1 __newlast1(_II1, _II1 __last1, _II2, _II2)
            {
                return __last1;
            }
            template<typename _II >
            static bool __cnd2(_II __first, _II __last)
            {
                return __first != __last;
            }
    };
    template<>
    struct __lc_rai<random_access_iterator_tag, random_access_iterator_tag>
    {
            template<typename _RAI1, typename _RAI2 >
            static _RAI1 __newlast1(_RAI1 __first1, _RAI1 __last1, _RAI2 __first2, _RAI2 __last2)
            {
                const typename iterator_traits<_RAI1>::difference_type __diff1 = __last1 - __first1;
                const typename iterator_traits<_RAI2>::difference_type __diff2 = __last2 - __first2;
                return __diff2 < __diff1 ? __first1 + __diff2 : __last1;
            }
            template<typename _RAI >
            static bool __cnd2(_RAI, _RAI)
            {
                return true;
            }
    };
    template<bool _BoolType >
    struct __lexicographical_compare
    {
            template<typename _II1, typename _II2 >
            static bool __lc(_II1, _II1, _II2, _II2);
    };
    template<bool _BoolType >
    template<typename _II1, typename _II2 >
    bool __lexicographical_compare<_BoolType>::__lc(_II1 __first1, _II1 __last1, _II2 __first2, _II2 __last2)
    {
        typedef typename iterator_traits<_II1>::iterator_category _Category1;
        typedef typename iterator_traits<_II2>::iterator_category _Category2;
        typedef std::__lc_rai<_Category1, _Category2> __rai_type;
        __last1 = __rai_type::__newlast1(__first1, __last1, __first2, __last2);
        for (;
            __first1 != __last1 && __rai_type::__cnd2(__first2, __last2);
            ++__first1 , ++__first2)
        {
            if (*__first1 < *__first2)
                return true;
            if (*__first2 < *__first1)
                return false;
        }
        return __first1 == __last1 && __first2 != __last2;
    }
    template<>
    struct __lexicographical_compare<true>
    {
            template<typename _Tp, typename _Up >
            static bool __lc(const _Tp *__first1, const _Tp *__last1, const _Up *__first2, const _Up *__last2)
            {
                const size_t __len1 = __last1 - __first1;
                const size_t __len2 = __last2 - __first2;
                const int __result = __builtin_memcmp(__first1, __first2, std::min(__len1, __len2));
                return __result != 0 ? __result < 0 : __len1 < __len2;
            }
    };
    template<typename _II1, typename _II2 >
    inline bool __lexicographical_compare_aux(_II1 __first1, _II1 __last1, _II2 __first2, _II2 __last2)
    {
        typedef typename iterator_traits<_II1>::value_type _ValueType1;
        typedef typename iterator_traits<_II2>::value_type _ValueType2;
        const bool __simple = (__is_byte<_ValueType1>::__value && __is_byte<_ValueType2>::__value && !__gnu_cxx::__numeric_traits<_ValueType1>::__is_signed && !__gnu_cxx::__numeric_traits<_ValueType2>::__is_signed && __is_pointer<_II1>::__value && __is_pointer<_II2>::__value);
        return std::__lexicographical_compare<__simple>::__lc(__first1, __last1, __first2, __last2);
    }
}
namespace std __attribute__((__visibility__("default"))) {
    template<typename _II1, typename _II2 >
    inline bool equal(_II1 __first1, _II1 __last1, _II2 __first2)
    {
        ;
        return std::__equal_aux(std::__niter_base<_II1>::__b(__first1), std::__niter_base<_II1>::__b(__last1), std::__niter_base<_II2>::__b(__first2));
    }
    template<typename _IIter1, typename _IIter2, typename _BinaryPredicate >
    inline bool equal(_IIter1 __first1, _IIter1 __last1, _IIter2 __first2, _BinaryPredicate __binary_pred)
    {
        ;
        for (;
            __first1 != __last1;
            ++__first1 , ++__first2)
            if (!bool(__binary_pred(*__first1, *__first2)))
                return false;
        return true;
    }
    template<typename _II1, typename _II2 >
    inline bool lexicographical_compare(_II1 __first1, _II1 __last1, _II2 __first2, _II2 __last2)
    {
        typedef typename iterator_traits<_II1>::value_type _ValueType1;
        typedef typename iterator_traits<_II2>::value_type _ValueType2;
        ;
        ;
        return std::__lexicographical_compare_aux(std::__niter_base<_II1>::__b(__first1), std::__niter_base<_II1>::__b(__last1), std::__niter_base<_II2>::__b(__first2), std::__niter_base<_II2>::__b(__last2));
    }
    template<typename _II1, typename _II2, typename _Compare >
    bool lexicographical_compare(_II1 __first1, _II1 __last1, _II2 __first2, _II2 __last2, _Compare __comp)
    {
        typedef typename iterator_traits<_II1>::iterator_category _Category1;
        typedef typename iterator_traits<_II2>::iterator_category _Category2;
        typedef std::__lc_rai<_Category1, _Category2> __rai_type;
        ;
        ;
        __last1 = __rai_type::__newlast1(__first1, __last1, __first2, __last2);
        for (;
            __first1 != __last1 && __rai_type::__cnd2(__first2, __last2);
            ++__first1 , ++__first2)
        {
            if (__comp(*__first1, *__first2))
                return true;
            if (__comp(*__first2, *__first1))
                return false;
        }
        return __first1 == __last1 && __first2 != __last2;
    }
    template<typename _InputIterator1, typename _InputIterator2 >
    pair<_InputIterator1, _InputIterator2> mismatch(_InputIterator1 __first1, _InputIterator1 __last1, _InputIterator2 __first2)
    {
        ;
        while (__first1 != __last1 && *__first1 == *__first2)
        {
            ++__first1;
            ++__first2;
        }
        return pair<_InputIterator1, _InputIterator2>(__first1, __first2);
    }
    template<typename _InputIterator1, typename _InputIterator2, typename _BinaryPredicate >
    pair<_InputIterator1, _InputIterator2> mismatch(_InputIterator1 __first1, _InputIterator1 __last1, _InputIterator2 __first2, _BinaryPredicate __binary_pred)
    {
        ;
        while (__first1 != __last1 && bool(__binary_pred(*__first1, *__first2)))
        {
            ++__first1;
            ++__first2;
        }
        return pair<_InputIterator1, _InputIterator2>(__first1, __first2);
    }
}
namespace __gnu_cxx __attribute__((__visibility__("default"))) {
    template<typename _CharT >
    struct _Char_types
    {
            typedef unsigned long int_type;
            typedef std::streampos pos_type;
            typedef std::streamoff off_type;
            typedef std::mbstate_t state_type;
    };
    template<typename _CharT >
    struct char_traits
    {
            typedef _CharT char_type;
            typedef typename _Char_types<_CharT>::int_type int_type;
            typedef typename _Char_types<_CharT>::pos_type pos_type;
            typedef typename _Char_types<_CharT>::off_type off_type;
            typedef typename _Char_types<_CharT>::state_type state_type;
            static void assign(char_type &__c1, const char_type &__c2)
            {
                __c1 = __c2;
            }
            static bool eq(const char_type &__c1, const char_type &__c2)
            {
                return __c1 == __c2;
            }
            static bool lt(const char_type &__c1, const char_type &__c2)
            {
                return __c1 < __c2;
            }
            static int compare(const char_type *__s1, const char_type *__s2, std::size_t __n);
            static std::size_t length(const char_type *__s);
            static const char_type *find(const char_type *__s, std::size_t __n, const char_type &__a);
            static char_type *move(char_type *__s1, const char_type *__s2, std::size_t __n);
            static char_type *copy(char_type *__s1, const char_type *__s2, std::size_t __n);
            static char_type *assign(char_type *__s, std::size_t __n, char_type __a);
            static char_type to_char_type(const int_type &__c)
            {
                return static_cast<char_type >(__c);
            }
            static int_type to_int_type(const char_type &__c)
            {
                return static_cast<int_type >(__c);
            }
            static bool eq_int_type(const int_type &__c1, const int_type &__c2)
            {
                return __c1 == __c2;
            }
            static int_type eof()
            {
                return static_cast<int_type >((- 1));
            }
            static int_type not_eof(const int_type &__c)
            {
                return !eq_int_type(__c, eof()) ? __c : to_int_type(char_type());
            }
    };
    template<typename _CharT >
    int char_traits<_CharT>::compare(const char_type *__s1, const char_type *__s2, std::size_t __n)
    {
        for (std::size_t __i = 0;
            __i < __n;
            ++__i)
            if (lt(__s1[__i], __s2[__i]))
                return - 1;
            else
                if (lt(__s2[__i], __s1[__i]))
                    return 1;
        return 0;
    }
    template<typename _CharT >
    std::size_t char_traits<_CharT>::length(const char_type *__p)
    {
        std::size_t __i = 0;
        while (!eq(__p[__i], char_type()))
            ++__i;
        return __i;
    }
    template<typename _CharT >
    const typename char_traits<_CharT>::char_type *char_traits<_CharT>::find(const char_type *__s, std::size_t __n, const char_type &__a)
    {
        for (std::size_t __i = 0;
            __i < __n;
            ++__i)
            if (eq(__s[__i], __a))
                return __s + __i;
        return 0;
    }
    template<typename _CharT >
    typename char_traits<_CharT>::char_type *char_traits<_CharT>::move(char_type *__s1, const char_type *__s2, std::size_t __n)
    {
        return static_cast<_CharT * >(__builtin_memmove(__s1, __s2, __n * sizeof(char_type)));
    }
    template<typename _CharT >
    typename char_traits<_CharT>::char_type *char_traits<_CharT>::copy(char_type *__s1, const char_type *__s2, std::size_t __n)
    {
        std::copy(__s2, __s2 + __n, __s1);
        return __s1;
    }
    template<typename _CharT >
    typename char_traits<_CharT>::char_type *char_traits<_CharT>::assign(char_type *__s, std::size_t __n, char_type __a)
    {
        std::fill_n(__s, __n, __a);
        return __s;
    }
}
namespace std __attribute__((__visibility__("default"))) {
    template<class _CharT >
    struct char_traits : public __gnu_cxx::char_traits<_CharT>
    {
    };
    template<>
    struct char_traits<char>
    {
            typedef char char_type;
            typedef int int_type;
            typedef streampos pos_type;
            typedef streamoff off_type;
            typedef mbstate_t state_type;
            static void assign(char_type &__c1, const char_type &__c2)
            {
                __c1 = __c2;
            }
            static bool eq(const char_type &__c1, const char_type &__c2)
            {
                return __c1 == __c2;
            }
            static bool lt(const char_type &__c1, const char_type &__c2)
            {
                return __c1 < __c2;
            }
            static int compare(const char_type *__s1, const char_type *__s2, size_t __n)
            {
                return __builtin_memcmp(__s1, __s2, __n);
            }
            static size_t length(const char_type *__s)
            {
                return __builtin_strlen(__s);
            }
            static const char_type *find(const char_type *__s, size_t __n, const char_type &__a)
            {
                return static_cast<const char_type * >(__builtin_memchr(__s, __a, __n));
            }
            static char_type *move(char_type *__s1, const char_type *__s2, size_t __n)
            {
                return static_cast<char_type * >(__builtin_memmove(__s1, __s2, __n));
            }
            static char_type *copy(char_type *__s1, const char_type *__s2, size_t __n)
            {
                return static_cast<char_type * >(__builtin_memcpy(__s1, __s2, __n));
            }
            static char_type *assign(char_type *__s, size_t __n, char_type __a)
            {
                return static_cast<char_type * >(__builtin_memset(__s, __a, __n));
            }
            static char_type to_char_type(const int_type &__c)
            {
                return static_cast<char_type >(__c);
            }
            static int_type to_int_type(const char_type &__c)
            {
                return static_cast<int_type >(static_cast<unsigned char >(__c));
            }
            static bool eq_int_type(const int_type &__c1, const int_type &__c2)
            {
                return __c1 == __c2;
            }
            static int_type eof()
            {
                return static_cast<int_type >((- 1));
            }
            static int_type not_eof(const int_type &__c)
            {
                return (__c == eof()) ? 0 : __c;
            }
    };
    template<>
    struct char_traits<wchar_t>
    {
            typedef wchar_t char_type;
            typedef wint_t int_type;
            typedef streamoff off_type;
            typedef wstreampos pos_type;
            typedef mbstate_t state_type;
            static void assign(char_type &__c1, const char_type &__c2)
            {
                __c1 = __c2;
            }
            static bool eq(const char_type &__c1, const char_type &__c2)
            {
                return __c1 == __c2;
            }
            static bool lt(const char_type &__c1, const char_type &__c2)
            {
                return __c1 < __c2;
            }
            static int compare(const char_type *__s1, const char_type *__s2, size_t __n)
            {
                return wmemcmp(__s1, __s2, __n);
            }
            static size_t length(const char_type *__s)
            {
                return wcslen(__s);
            }
            static const char_type *find(const char_type *__s, size_t __n, const char_type &__a)
            {
                return wmemchr(__s, __a, __n);
            }
            static char_type *move(char_type *__s1, const char_type *__s2, size_t __n)
            {
                return wmemmove(__s1, __s2, __n);
            }
            static char_type *copy(char_type *__s1, const char_type *__s2, size_t __n)
            {
                return wmemcpy(__s1, __s2, __n);
            }
            static char_type *assign(char_type *__s, size_t __n, char_type __a)
            {
                return wmemset(__s, __a, __n);
            }
            static char_type to_char_type(const int_type &__c)
            {
                return char_type(__c);
            }
            static int_type to_int_type(const char_type &__c)
            {
                return int_type(__c);
            }
            static bool eq_int_type(const int_type &__c1, const int_type &__c2)
            {
                return __c1 == __c2;
            }
            static int_type eof()
            {
                return static_cast<int_type >((0xffffffffu));
            }
            static int_type not_eof(const int_type &__c)
            {
                return eq_int_type(__c, eof()) ? 0 : __c;
            }
    };
}
enum 
{
    __LC_CTYPE = 0, 
    __LC_NUMERIC = 1, 
    __LC_TIME = 2, 
    __LC_COLLATE = 3, 
    __LC_MONETARY = 4, 
    __LC_MESSAGES = 5, 
    __LC_ALL = 6, 
    __LC_PAPER = 7, 
    __LC_NAME = 8, 
    __LC_ADDRESS = 9, 
    __LC_TELEPHONE = 10, 
    __LC_MEASUREMENT = 11, 
    __LC_IDENTIFICATION = 12
};
extern "C"
{
    struct lconv
    {
            char *decimal_point;
            char *thousands_sep;
            char *grouping;
            char *int_curr_symbol;
            char *currency_symbol;
            char *mon_decimal_point;
            char *mon_thousands_sep;
            char *mon_grouping;
            char *positive_sign;
            char *negative_sign;
            char int_frac_digits;
            char frac_digits;
            char p_cs_precedes;
            char p_sep_by_space;
            char n_cs_precedes;
            char n_sep_by_space;
            char p_sign_posn;
            char n_sign_posn;
            char int_p_cs_precedes;
            char int_p_sep_by_space;
            char int_n_cs_precedes;
            char int_n_sep_by_space;
            char int_p_sign_posn;
            char int_n_sign_posn;
    };
    extern char *setlocale(int __category, __const char *__locale) throw ();
    extern struct lconv *localeconv(void) throw ();
    extern __locale_t newlocale(int __category_mask, __const char *__locale, __locale_t __base) throw ();
    extern __locale_t duplocale(__locale_t __dataset) throw ();
    extern void freelocale(__locale_t __dataset) throw ();
    extern __locale_t uselocale(__locale_t __dataset) throw ();
}
namespace std __attribute__((__visibility__("default"))) {
    using ::lconv;
    using ::setlocale;
    using ::localeconv;
}
namespace __gnu_cxx __attribute__((__visibility__("default"))) {
    extern "C"
    __typeof (uselocale) __uselocale;
}
namespace std __attribute__((__visibility__("default"))) {
    typedef __locale_t __c_locale;
    inline int __convert_from_v(const __c_locale &__cloc __attribute__((__unused__)), char *__out, const int __size __attribute__((__unused__)), const char *__fmt, ...)
    {
        __c_locale __old = __gnu_cxx::__uselocale(__cloc);
        __builtin_va_list __args;
        __builtin_va_start(__args, __fmt);
        const int __ret = __builtin_vsnprintf(__out, __size, __fmt, __args);
        __builtin_va_end(__args);
        __gnu_cxx::__uselocale(__old);
        return __ret;
    }
}
extern "C"
{
    enum 
    {
        _ISupper = ((0) < 8 ? ((1 << (0)) << 8) : ((1 << (0)) >> 8)), 
        _ISlower = ((1) < 8 ? ((1 << (1)) << 8) : ((1 << (1)) >> 8)), 
        _ISalpha = ((2) < 8 ? ((1 << (2)) << 8) : ((1 << (2)) >> 8)), 
        _ISdigit = ((3) < 8 ? ((1 << (3)) << 8) : ((1 << (3)) >> 8)), 
        _ISxdigit = ((4) < 8 ? ((1 << (4)) << 8) : ((1 << (4)) >> 8)), 
        _ISspace = ((5) < 8 ? ((1 << (5)) << 8) : ((1 << (5)) >> 8)), 
        _ISprint = ((6) < 8 ? ((1 << (6)) << 8) : ((1 << (6)) >> 8)), 
        _ISgraph = ((7) < 8 ? ((1 << (7)) << 8) : ((1 << (7)) >> 8)), 
        _ISblank = ((8) < 8 ? ((1 << (8)) << 8) : ((1 << (8)) >> 8)), 
        _IScntrl = ((9) < 8 ? ((1 << (9)) << 8) : ((1 << (9)) >> 8)), 
        _ISpunct = ((10) < 8 ? ((1 << (10)) << 8) : ((1 << (10)) >> 8)), 
        _ISalnum = ((11) < 8 ? ((1 << (11)) << 8) : ((1 << (11)) >> 8))
    };
    extern __const unsigned short int **__ctype_b_loc(void) throw () __attribute__((__const));
    extern __const __int32_t **__ctype_tolower_loc(void) throw () __attribute__((__const));
    extern __const __int32_t **__ctype_toupper_loc(void) throw () __attribute__((__const));
    extern int isalnum(int) throw ();
    extern int isalpha(int) throw ();
    extern int iscntrl(int) throw ();
    extern int isdigit(int) throw ();
    extern int islower(int) throw ();
    extern int isgraph(int) throw ();
    extern int isprint(int) throw ();
    extern int ispunct(int) throw ();
    extern int isspace(int) throw ();
    extern int isupper(int) throw ();
    extern int isxdigit(int) throw ();
    extern int tolower(int __c) throw ();
    extern int toupper(int __c) throw ();
    extern int isblank(int) throw ();
    extern int isctype(int __c, int __mask) throw ();
    extern int isascii(int __c) throw ();
    extern int toascii(int __c) throw ();
    extern int _toupper(int) throw ();
    extern int _tolower(int) throw ();
    extern int isalnum_l(int, __locale_t) throw ();
    extern int isalpha_l(int, __locale_t) throw ();
    extern int iscntrl_l(int, __locale_t) throw ();
    extern int isdigit_l(int, __locale_t) throw ();
    extern int islower_l(int, __locale_t) throw ();
    extern int isgraph_l(int, __locale_t) throw ();
    extern int isprint_l(int, __locale_t) throw ();
    extern int ispunct_l(int, __locale_t) throw ();
    extern int isspace_l(int, __locale_t) throw ();
    extern int isupper_l(int, __locale_t) throw ();
    extern int isxdigit_l(int, __locale_t) throw ();
    extern int isblank_l(int, __locale_t) throw ();
    extern int __tolower_l(int __c, __locale_t __l) throw ();
    extern int tolower_l(int __c, __locale_t __l) throw ();
    extern int __toupper_l(int __c, __locale_t __l) throw ();
    extern int toupper_l(int __c, __locale_t __l) throw ();
}
namespace std __attribute__((__visibility__("default"))) {
    using ::isalnum;
    using ::isalpha;
    using ::iscntrl;
    using ::isdigit;
    using ::isgraph;
    using ::islower;
    using ::isprint;
    using ::ispunct;
    using ::isspace;
    using ::isupper;
    using ::isxdigit;
    using ::tolower;
    using ::toupper;
}
namespace std __attribute__((__visibility__("default"))) {
    class locale;
    template<typename _Facet >
    bool has_facet(const locale &) throw ();
    template<typename _Facet >
    const _Facet &use_facet(const locale &);
    template<typename _CharT >
    bool isspace(_CharT, const locale &);
    template<typename _CharT >
    bool isprint(_CharT, const locale &);
    template<typename _CharT >
    bool iscntrl(_CharT, const locale &);
    template<typename _CharT >
    bool isupper(_CharT, const locale &);
    template<typename _CharT >
    bool islower(_CharT, const locale &);
    template<typename _CharT >
    bool isalpha(_CharT, const locale &);
    template<typename _CharT >
    bool isdigit(_CharT, const locale &);
    template<typename _CharT >
    bool ispunct(_CharT, const locale &);
    template<typename _CharT >
    bool isxdigit(_CharT, const locale &);
    template<typename _CharT >
    bool isalnum(_CharT, const locale &);
    template<typename _CharT >
    bool isgraph(_CharT, const locale &);
    template<typename _CharT >
    _CharT toupper(_CharT, const locale &);
    template<typename _CharT >
    _CharT tolower(_CharT, const locale &);
    class ctype_base;
    template<typename _CharT >
    class ctype;
    template<>
    class ctype<char>;
    template<>
    class ctype<wchar_t>;
    template<typename _CharT >
    class ctype_byname;
    class codecvt_base;
    template<typename _InternT, typename _ExternT, typename _StateT >
    class codecvt;
    template<>
    class codecvt<char, char, mbstate_t>;
    template<>
    class codecvt<wchar_t, char, mbstate_t>;
    template<typename _InternT, typename _ExternT, typename _StateT >
    class codecvt_byname;
    template<typename _CharT, typename _InIter = istreambuf_iterator<_CharT> >
    class num_get;
    template<typename _CharT, typename _OutIter = ostreambuf_iterator<_CharT> >
    class num_put;
    template<typename _CharT >
    class numpunct;
    template<typename _CharT >
    class numpunct_byname;
    template<typename _CharT >
    class collate;
    template<typename _CharT >
    class collate_byname;
    class time_base;
    template<typename _CharT, typename _InIter = istreambuf_iterator<_CharT> >
    class time_get;
    template<typename _CharT, typename _InIter = istreambuf_iterator<_CharT> >
    class time_get_byname;
    template<typename _CharT, typename _OutIter = ostreambuf_iterator<_CharT> >
    class time_put;
    template<typename _CharT, typename _OutIter = ostreambuf_iterator<_CharT> >
    class time_put_byname;
    class money_base;
    template<typename _CharT, typename _InIter = istreambuf_iterator<_CharT> >
    class money_get;
    template<typename _CharT, typename _OutIter = ostreambuf_iterator<_CharT> >
    class money_put;
    template<typename _CharT, bool _Intl = false >
    class moneypunct;
    template<typename _CharT, bool _Intl = false >
    class moneypunct_byname;
    class messages_base;
    template<typename _CharT >
    class messages;
    template<typename _CharT >
    class messages_byname;
}
#pragma GCC visibility push(default)
typedef __time_t time_t;
struct timespec
{
        __time_t tv_sec;
        long int tv_nsec;
};
struct sched_param
{
        int __sched_priority;
};
extern "C"
{
    extern int clone(int (*__fn)(void *__arg), void *__child_stack, int __flags, void *__arg, ...) throw ();
    extern int unshare(int __flags) throw ();
    extern int sched_getcpu(void) throw ();
}
struct __sched_param
{
        int __sched_priority;
};
typedef unsigned long int __cpu_mask;
typedef struct 
{
        __cpu_mask __bits[1024 / (8 * sizeof(__cpu_mask))];
} cpu_set_t;
extern "C"
{
    extern int __sched_cpucount(size_t __setsize, const cpu_set_t *__setp) throw ();
    extern cpu_set_t *__sched_cpualloc(size_t __count) throw ();
    extern void __sched_cpufree(cpu_set_t *__set) throw ();
}
extern "C"
{
    extern int sched_setparam(__pid_t __pid, __const struct sched_param *__param) throw ();
    extern int sched_getparam(__pid_t __pid, struct sched_param *__param) throw ();
    extern int sched_setscheduler(__pid_t __pid, int __policy, __const struct sched_param *__param) throw ();
    extern int sched_getscheduler(__pid_t __pid) throw ();
    extern int sched_yield(void) throw ();
    extern int sched_get_priority_max(int __algorithm) throw ();
    extern int sched_get_priority_min(int __algorithm) throw ();
    extern int sched_rr_get_interval(__pid_t __pid, struct timespec *__t) throw ();
    extern int sched_setaffinity(__pid_t __pid, size_t __cpusetsize, __const cpu_set_t *__cpuset) throw ();
    extern int sched_getaffinity(__pid_t __pid, size_t __cpusetsize, cpu_set_t *__cpuset) throw ();
}
extern "C"
{
    typedef __clock_t clock_t;
    typedef __clockid_t clockid_t;
    typedef __timer_t timer_t;
    struct tm
    {
            int tm_sec;
            int tm_min;
            int tm_hour;
            int tm_mday;
            int tm_mon;
            int tm_year;
            int tm_wday;
            int tm_yday;
            int tm_isdst;
            long int tm_gmtoff;
            __const char *tm_zone;
    };
    struct itimerspec
    {
            struct timespec it_interval;
            struct timespec it_value;
    };
    struct sigevent;
    extern clock_t clock(void) throw ();
    extern time_t time(time_t *__timer) throw ();
    extern double difftime(time_t __time1, time_t __time0) throw () __attribute__((__const__));
    extern time_t mktime(struct tm *__tp) throw ();
    extern size_t strftime(char *__restrict __s, size_t __maxsize, __const char *__restrict __format, __const struct tm *__restrict __tp) throw ();
    extern char *strptime(__const char *__restrict __s, __const char *__restrict __fmt, struct tm *__tp) throw ();
    extern size_t strftime_l(char *__restrict __s, size_t __maxsize, __const char *__restrict __format, __const struct tm *__restrict __tp, __locale_t __loc) throw ();
    extern char *strptime_l(__const char *__restrict __s, __const char *__restrict __fmt, struct tm *__tp, __locale_t __loc) throw ();
    extern struct tm *gmtime(__const time_t *__timer) throw ();
    extern struct tm *localtime(__const time_t *__timer) throw ();
    extern struct tm *gmtime_r(__const time_t *__restrict __timer, struct tm *__restrict __tp) throw ();
    extern struct tm *localtime_r(__const time_t *__restrict __timer, struct tm *__restrict __tp) throw ();
    extern char *asctime(__const struct tm *__tp) throw ();
    extern char *ctime(__const time_t *__timer) throw ();
    extern char *asctime_r(__const struct tm *__restrict __tp, char *__restrict __buf) throw ();
    extern char *ctime_r(__const time_t *__restrict __timer, char *__restrict __buf) throw ();
    extern char *__tzname[2];
    extern int __daylight;
    extern long int __timezone;
    extern char *tzname[2];
    extern void tzset(void) throw ();
    extern int daylight;
    extern long int timezone;
    extern int stime(__const time_t *__when) throw ();
    extern time_t timegm(struct tm *__tp) throw ();
    extern time_t timelocal(struct tm *__tp) throw ();
    extern int dysize(int __year) throw () __attribute__((__const__));
    extern int nanosleep(__const struct timespec *__requested_time, struct timespec *__remaining);
    extern int clock_getres(clockid_t __clock_id, struct timespec *__res) throw ();
    extern int clock_gettime(clockid_t __clock_id, struct timespec *__tp) throw ();
    extern int clock_settime(clockid_t __clock_id, __const struct timespec *__tp) throw ();
    extern int clock_nanosleep(clockid_t __clock_id, int __flags, __const struct timespec *__req, struct timespec *__rem);
    extern int clock_getcpuclockid(pid_t __pid, clockid_t *__clock_id) throw ();
    extern int timer_create(clockid_t __clock_id, struct sigevent *__restrict __evp, timer_t *__restrict __timerid) throw ();
    extern int timer_delete(timer_t __timerid) throw ();
    extern int timer_settime(timer_t __timerid, int __flags, __const struct itimerspec *__restrict __value, struct itimerspec *__restrict __ovalue) throw ();
    extern int timer_gettime(timer_t __timerid, struct itimerspec *__value) throw ();
    extern int timer_getoverrun(timer_t __timerid) throw ();
    extern int getdate_err;
    extern struct tm *getdate(__const char *__string);
    extern int getdate_r(__const char *__restrict __string, struct tm *__restrict __resbufp);
}
typedef unsigned long int pthread_t;
typedef union 
{
        char __size[56];
        long int __align;
} pthread_attr_t;
typedef struct __pthread_internal_list
{
        struct __pthread_internal_list *__prev;
        struct __pthread_internal_list *__next;
} __pthread_list_t;
typedef union 
{
        struct __pthread_mutex_s
        {
                int __lock;
                unsigned int __count;
                int __owner;
                unsigned int __nusers;
                int __kind;
                int __spins;
                __pthread_list_t __list;
        } __data;
        char __size[40];
        long int __align;
} pthread_mutex_t;
typedef union 
{
        char __size[4];
        int __align;
} pthread_mutexattr_t;
typedef union 
{
        struct 
        {
                int __lock;
                unsigned int __futex;
                __extension__
                unsigned long long int __total_seq;
                __extension__
                unsigned long long int __wakeup_seq;
                __extension__
                unsigned long long int __woken_seq;
                void *__mutex;
                unsigned int __nwaiters;
                unsigned int __broadcast_seq;
        } __data;
        char __size[48];
        __extension__
        long long int __align;
} pthread_cond_t;
typedef union 
{
        char __size[4];
        int __align;
} pthread_condattr_t;
typedef unsigned int pthread_key_t;
typedef int pthread_once_t;
typedef union 
{
        struct 
        {
                int __lock;
                unsigned int __nr_readers;
                unsigned int __readers_wakeup;
                unsigned int __writer_wakeup;
                unsigned int __nr_readers_queued;
                unsigned int __nr_writers_queued;
                int __writer;
                int __shared;
                unsigned long int __pad1;
                unsigned long int __pad2;
                unsigned int __flags;
        } __data;
        char __size[56];
        long int __align;
} pthread_rwlock_t;
typedef union 
{
        char __size[8];
        long int __align;
} pthread_rwlockattr_t;
typedef volatile int pthread_spinlock_t;
typedef union 
{
        char __size[32];
        long int __align;
} pthread_barrier_t;
typedef union 
{
        char __size[4];
        int __align;
} pthread_barrierattr_t;
typedef long int __jmp_buf[8];
enum 
{
    PTHREAD_CREATE_JOINABLE, 
    PTHREAD_CREATE_DETACHED
};
enum 
{
    PTHREAD_MUTEX_TIMED_NP, 
    PTHREAD_MUTEX_RECURSIVE_NP, 
    PTHREAD_MUTEX_ERRORCHECK_NP, 
    PTHREAD_MUTEX_ADAPTIVE_NP, 
    PTHREAD_MUTEX_NORMAL = PTHREAD_MUTEX_TIMED_NP, 
    PTHREAD_MUTEX_RECURSIVE = PTHREAD_MUTEX_RECURSIVE_NP, 
    PTHREAD_MUTEX_ERRORCHECK = PTHREAD_MUTEX_ERRORCHECK_NP, 
    PTHREAD_MUTEX_DEFAULT = PTHREAD_MUTEX_NORMAL, 
    PTHREAD_MUTEX_FAST_NP = PTHREAD_MUTEX_TIMED_NP
};
enum 
{
    PTHREAD_MUTEX_STALLED, 
    PTHREAD_MUTEX_STALLED_NP = PTHREAD_MUTEX_STALLED, 
    PTHREAD_MUTEX_ROBUST, 
    PTHREAD_MUTEX_ROBUST_NP = PTHREAD_MUTEX_ROBUST
};
enum 
{
    PTHREAD_PRIO_NONE, 
    PTHREAD_PRIO_INHERIT, 
    PTHREAD_PRIO_PROTECT
};
enum 
{
    PTHREAD_RWLOCK_PREFER_READER_NP, 
    PTHREAD_RWLOCK_PREFER_WRITER_NP, 
    PTHREAD_RWLOCK_PREFER_WRITER_NONRECURSIVE_NP, 
    PTHREAD_RWLOCK_DEFAULT_NP = PTHREAD_RWLOCK_PREFER_READER_NP
};
enum 
{
    PTHREAD_INHERIT_SCHED, 
    PTHREAD_EXPLICIT_SCHED
};
enum 
{
    PTHREAD_SCOPE_SYSTEM, 
    PTHREAD_SCOPE_PROCESS
};
enum 
{
    PTHREAD_PROCESS_PRIVATE, 
    PTHREAD_PROCESS_SHARED
};
struct _pthread_cleanup_buffer
{
        void (*__routine)(void *);
        void *__arg;
        int __canceltype;
        struct _pthread_cleanup_buffer *__prev;
};
enum 
{
    PTHREAD_CANCEL_ENABLE, 
    PTHREAD_CANCEL_DISABLE
};
enum 
{
    PTHREAD_CANCEL_DEFERRED, 
    PTHREAD_CANCEL_ASYNCHRONOUS
};
extern "C"
{
    extern int pthread_create(pthread_t *__restrict __newthread, __const pthread_attr_t *__restrict __attr, void *(*__start_routine)(void *), void *__restrict __arg) throw () __attribute__((__nonnull__(1, 3)));
    extern void pthread_exit(void *__retval) __attribute__((__noreturn__));
    extern int pthread_join(pthread_t __th, void **__thread_return);
    extern int pthread_tryjoin_np(pthread_t __th, void **__thread_return) throw ();
    extern int pthread_timedjoin_np(pthread_t __th, void **__thread_return, __const struct timespec *__abstime);
    extern int pthread_detach(pthread_t __th) throw ();
    extern pthread_t pthread_self(void) throw () __attribute__((__const__));
    extern int pthread_equal(pthread_t __thread1, pthread_t __thread2) throw ();
    extern int pthread_attr_init(pthread_attr_t *__attr) throw () __attribute__((__nonnull__(1)));
    extern int pthread_attr_destroy(pthread_attr_t *__attr) throw () __attribute__((__nonnull__(1)));
    extern int pthread_attr_getdetachstate(__const pthread_attr_t *__attr, int *__detachstate) throw () __attribute__((__nonnull__(1, 2)));
    extern int pthread_attr_setdetachstate(pthread_attr_t *__attr, int __detachstate) throw () __attribute__((__nonnull__(1)));
    extern int pthread_attr_getguardsize(__const pthread_attr_t *__attr, size_t *__guardsize) throw () __attribute__((__nonnull__(1, 2)));
    extern int pthread_attr_setguardsize(pthread_attr_t *__attr, size_t __guardsize) throw () __attribute__((__nonnull__(1)));
    extern int pthread_attr_getschedparam(__const pthread_attr_t *__restrict __attr, struct sched_param *__restrict __param) throw () __attribute__((__nonnull__(1, 2)));
    extern int pthread_attr_setschedparam(pthread_attr_t *__restrict __attr, __const struct sched_param *__restrict __param) throw () __attribute__((__nonnull__(1, 2)));
    extern int pthread_attr_getschedpolicy(__const pthread_attr_t *__restrict __attr, int *__restrict __policy) throw () __attribute__((__nonnull__(1, 2)));
    extern int pthread_attr_setschedpolicy(pthread_attr_t *__attr, int __policy) throw () __attribute__((__nonnull__(1)));
    extern int pthread_attr_getinheritsched(__const pthread_attr_t *__restrict __attr, int *__restrict __inherit) throw () __attribute__((__nonnull__(1, 2)));
    extern int pthread_attr_setinheritsched(pthread_attr_t *__attr, int __inherit) throw () __attribute__((__nonnull__(1)));
    extern int pthread_attr_getscope(__const pthread_attr_t *__restrict __attr, int *__restrict __scope) throw () __attribute__((__nonnull__(1, 2)));
    extern int pthread_attr_setscope(pthread_attr_t *__attr, int __scope) throw () __attribute__((__nonnull__(1)));
    extern int pthread_attr_getstackaddr(__const pthread_attr_t *__restrict __attr, void **__restrict __stackaddr) throw () __attribute__((__nonnull__(1, 2))) __attribute__((__deprecated__));
    extern int pthread_attr_setstackaddr(pthread_attr_t *__attr, void *__stackaddr) throw () __attribute__((__nonnull__(1))) __attribute__((__deprecated__));
    extern int pthread_attr_getstacksize(__const pthread_attr_t *__restrict __attr, size_t *__restrict __stacksize) throw () __attribute__((__nonnull__(1, 2)));
    extern int pthread_attr_setstacksize(pthread_attr_t *__attr, size_t __stacksize) throw () __attribute__((__nonnull__(1)));
    extern int pthread_attr_getstack(__const pthread_attr_t *__restrict __attr, void **__restrict __stackaddr, size_t *__restrict __stacksize) throw () __attribute__((__nonnull__(1, 2, 3)));
    extern int pthread_attr_setstack(pthread_attr_t *__attr, void *__stackaddr, size_t __stacksize) throw () __attribute__((__nonnull__(1)));
    extern int pthread_attr_setaffinity_np(pthread_attr_t *__attr, size_t __cpusetsize, __const cpu_set_t *__cpuset) throw () __attribute__((__nonnull__(1, 3)));
    extern int pthread_attr_getaffinity_np(__const pthread_attr_t *__attr, size_t __cpusetsize, cpu_set_t *__cpuset) throw () __attribute__((__nonnull__(1, 3)));
    extern int pthread_getattr_np(pthread_t __th, pthread_attr_t *__attr) throw () __attribute__((__nonnull__(2)));
    extern int pthread_setschedparam(pthread_t __target_thread, int __policy, __const struct sched_param *__param) throw () __attribute__((__nonnull__(3)));
    extern int pthread_getschedparam(pthread_t __target_thread, int *__restrict __policy, struct sched_param *__restrict __param) throw () __attribute__((__nonnull__(2, 3)));
    extern int pthread_setschedprio(pthread_t __target_thread, int __prio) throw ();
    extern int pthread_getname_np(pthread_t __target_thread, char *__buf, size_t __buflen) throw () __attribute__((__nonnull__(2)));
    extern int pthread_setname_np(pthread_t __target_thread, __const char *__name) throw () __attribute__((__nonnull__(2)));
    extern int pthread_getconcurrency(void) throw ();
    extern int pthread_setconcurrency(int __level) throw ();
    extern int pthread_yield(void) throw ();
    extern int pthread_setaffinity_np(pthread_t __th, size_t __cpusetsize, __const cpu_set_t *__cpuset) throw () __attribute__((__nonnull__(3)));
    extern int pthread_getaffinity_np(pthread_t __th, size_t __cpusetsize, cpu_set_t *__cpuset) throw () __attribute__((__nonnull__(3)));
    extern int pthread_once(pthread_once_t *__once_control, void (*__init_routine)(void)) __attribute__((__nonnull__(1, 2)));
    extern int pthread_setcancelstate(int __state, int *__oldstate);
    extern int pthread_setcanceltype(int __type, int *__oldtype);
    extern int pthread_cancel(pthread_t __th);
    extern void pthread_testcancel(void);
    typedef struct 
    {
            struct 
            {
                    __jmp_buf __cancel_jmp_buf;
                    int __mask_was_saved;
            } __cancel_jmp_buf[1];
            void *__pad[4];
    } __pthread_unwind_buf_t __attribute__((__aligned__));
    struct __pthread_cleanup_frame
    {
            void (*__cancel_routine)(void *);
            void *__cancel_arg;
            int __do_it;
            int __cancel_type;
    };
    class __pthread_cleanup_class
    {
            void (*__cancel_routine)(void *);
            void *__cancel_arg;
            int __do_it;
            int __cancel_type;
        public :
            __pthread_cleanup_class(void (*__fct)(void *), void *__arg)
                : __cancel_routine(__fct), __cancel_arg(__arg), __do_it(1) 
            {
            }
            ~__pthread_cleanup_class()
            {
                if (__do_it)
                    __cancel_routine(__cancel_arg);
            }
            void __setdoit(int __newval)
            {
                __do_it = __newval;
            }
            void __defer()
            {
                pthread_setcanceltype(PTHREAD_CANCEL_DEFERRED, &__cancel_type);
            }
            void __restore() const
            {
                pthread_setcanceltype(__cancel_type, 0);
            }
    };
    struct __jmp_buf_tag;
    extern int __sigsetjmp(struct __jmp_buf_tag *__env, int __savemask) throw ();
    extern int pthread_mutex_init(pthread_mutex_t *__mutex, __const pthread_mutexattr_t *__mutexattr) throw () __attribute__((__nonnull__(1)));
    extern int pthread_mutex_destroy(pthread_mutex_t *__mutex) throw () __attribute__((__nonnull__(1)));
    extern int pthread_mutex_trylock(pthread_mutex_t *__mutex) throw () __attribute__((__nonnull__(1)));
    extern int pthread_mutex_lock(pthread_mutex_t *__mutex) throw () __attribute__((__nonnull__(1)));
    extern int pthread_mutex_timedlock(pthread_mutex_t *__restrict __mutex, __const struct timespec *__restrict __abstime) throw () __attribute__((__nonnull__(1, 2)));
    extern int pthread_mutex_unlock(pthread_mutex_t *__mutex) throw () __attribute__((__nonnull__(1)));
    extern int pthread_mutex_getprioceiling(__const pthread_mutex_t *__restrict __mutex, int *__restrict __prioceiling) throw () __attribute__((__nonnull__(1, 2)));
    extern int pthread_mutex_setprioceiling(pthread_mutex_t *__restrict __mutex, int __prioceiling, int *__restrict __old_ceiling) throw () __attribute__((__nonnull__(1, 3)));
    extern int pthread_mutex_consistent(pthread_mutex_t *__mutex) throw () __attribute__((__nonnull__(1)));
    extern int pthread_mutex_consistent_np(pthread_mutex_t *__mutex) throw () __attribute__((__nonnull__(1)));
    extern int pthread_mutexattr_init(pthread_mutexattr_t *__attr) throw () __attribute__((__nonnull__(1)));
    extern int pthread_mutexattr_destroy(pthread_mutexattr_t *__attr) throw () __attribute__((__nonnull__(1)));
    extern int pthread_mutexattr_getpshared(__const pthread_mutexattr_t *__restrict __attr, int *__restrict __pshared) throw () __attribute__((__nonnull__(1, 2)));
    extern int pthread_mutexattr_setpshared(pthread_mutexattr_t *__attr, int __pshared) throw () __attribute__((__nonnull__(1)));
    extern int pthread_mutexattr_gettype(__const pthread_mutexattr_t *__restrict __attr, int *__restrict __kind) throw () __attribute__((__nonnull__(1, 2)));
    extern int pthread_mutexattr_settype(pthread_mutexattr_t *__attr, int __kind) throw () __attribute__((__nonnull__(1)));
    extern int pthread_mutexattr_getprotocol(__const pthread_mutexattr_t *__restrict __attr, int *__restrict __protocol) throw () __attribute__((__nonnull__(1, 2)));
    extern int pthread_mutexattr_setprotocol(pthread_mutexattr_t *__attr, int __protocol) throw () __attribute__((__nonnull__(1)));
    extern int pthread_mutexattr_getprioceiling(__const pthread_mutexattr_t *__restrict __attr, int *__restrict __prioceiling) throw () __attribute__((__nonnull__(1, 2)));
    extern int pthread_mutexattr_setprioceiling(pthread_mutexattr_t *__attr, int __prioceiling) throw () __attribute__((__nonnull__(1)));
    extern int pthread_mutexattr_getrobust(__const pthread_mutexattr_t *__attr, int *__robustness) throw () __attribute__((__nonnull__(1, 2)));
    extern int pthread_mutexattr_getrobust_np(__const pthread_mutexattr_t *__attr, int *__robustness) throw () __attribute__((__nonnull__(1, 2)));
    extern int pthread_mutexattr_setrobust(pthread_mutexattr_t *__attr, int __robustness) throw () __attribute__((__nonnull__(1)));
    extern int pthread_mutexattr_setrobust_np(pthread_mutexattr_t *__attr, int __robustness) throw () __attribute__((__nonnull__(1)));
    extern int pthread_rwlock_init(pthread_rwlock_t *__restrict __rwlock, __const pthread_rwlockattr_t *__restrict __attr) throw () __attribute__((__nonnull__(1)));
    extern int pthread_rwlock_destroy(pthread_rwlock_t *__rwlock) throw () __attribute__((__nonnull__(1)));
    extern int pthread_rwlock_rdlock(pthread_rwlock_t *__rwlock) throw () __attribute__((__nonnull__(1)));
    extern int pthread_rwlock_tryrdlock(pthread_rwlock_t *__rwlock) throw () __attribute__((__nonnull__(1)));
    extern int pthread_rwlock_timedrdlock(pthread_rwlock_t *__restrict __rwlock, __const struct timespec *__restrict __abstime) throw () __attribute__((__nonnull__(1, 2)));
    extern int pthread_rwlock_wrlock(pthread_rwlock_t *__rwlock) throw () __attribute__((__nonnull__(1)));
    extern int pthread_rwlock_trywrlock(pthread_rwlock_t *__rwlock) throw () __attribute__((__nonnull__(1)));
    extern int pthread_rwlock_timedwrlock(pthread_rwlock_t *__restrict __rwlock, __const struct timespec *__restrict __abstime) throw () __attribute__((__nonnull__(1, 2)));
    extern int pthread_rwlock_unlock(pthread_rwlock_t *__rwlock) throw () __attribute__((__nonnull__(1)));
    extern int pthread_rwlockattr_init(pthread_rwlockattr_t *__attr) throw () __attribute__((__nonnull__(1)));
    extern int pthread_rwlockattr_destroy(pthread_rwlockattr_t *__attr) throw () __attribute__((__nonnull__(1)));
    extern int pthread_rwlockattr_getpshared(__const pthread_rwlockattr_t *__restrict __attr, int *__restrict __pshared) throw () __attribute__((__nonnull__(1, 2)));
    extern int pthread_rwlockattr_setpshared(pthread_rwlockattr_t *__attr, int __pshared) throw () __attribute__((__nonnull__(1)));
    extern int pthread_rwlockattr_getkind_np(__const pthread_rwlockattr_t *__restrict __attr, int *__restrict __pref) throw () __attribute__((__nonnull__(1, 2)));
    extern int pthread_rwlockattr_setkind_np(pthread_rwlockattr_t *__attr, int __pref) throw () __attribute__((__nonnull__(1)));
    extern int pthread_cond_init(pthread_cond_t *__restrict __cond, __const pthread_condattr_t *__restrict __cond_attr) throw () __attribute__((__nonnull__(1)));
    extern int pthread_cond_destroy(pthread_cond_t *__cond) throw () __attribute__((__nonnull__(1)));
    extern int pthread_cond_signal(pthread_cond_t *__cond) throw () __attribute__((__nonnull__(1)));
    extern int pthread_cond_broadcast(pthread_cond_t *__cond) throw () __attribute__((__nonnull__(1)));
    extern int pthread_cond_wait(pthread_cond_t *__restrict __cond, pthread_mutex_t *__restrict __mutex) __attribute__((__nonnull__(1, 2)));
    extern int pthread_cond_timedwait(pthread_cond_t *__restrict __cond, pthread_mutex_t *__restrict __mutex, __const struct timespec *__restrict __abstime) __attribute__((__nonnull__(1, 2, 3)));
    extern int pthread_condattr_init(pthread_condattr_t *__attr) throw () __attribute__((__nonnull__(1)));
    extern int pthread_condattr_destroy(pthread_condattr_t *__attr) throw () __attribute__((__nonnull__(1)));
    extern int pthread_condattr_getpshared(__const pthread_condattr_t *__restrict __attr, int *__restrict __pshared) throw () __attribute__((__nonnull__(1, 2)));
    extern int pthread_condattr_setpshared(pthread_condattr_t *__attr, int __pshared) throw () __attribute__((__nonnull__(1)));
    extern int pthread_condattr_getclock(__const pthread_condattr_t *__restrict __attr, __clockid_t *__restrict __clock_id) throw () __attribute__((__nonnull__(1, 2)));
    extern int pthread_condattr_setclock(pthread_condattr_t *__attr, __clockid_t __clock_id) throw () __attribute__((__nonnull__(1)));
    extern int pthread_spin_init(pthread_spinlock_t *__lock, int __pshared) throw () __attribute__((__nonnull__(1)));
    extern int pthread_spin_destroy(pthread_spinlock_t *__lock) throw () __attribute__((__nonnull__(1)));
    extern int pthread_spin_lock(pthread_spinlock_t *__lock) throw () __attribute__((__nonnull__(1)));
    extern int pthread_spin_trylock(pthread_spinlock_t *__lock) throw () __attribute__((__nonnull__(1)));
    extern int pthread_spin_unlock(pthread_spinlock_t *__lock) throw () __attribute__((__nonnull__(1)));
    extern int pthread_barrier_init(pthread_barrier_t *__restrict __barrier, __const pthread_barrierattr_t *__restrict __attr, unsigned int __count) throw () __attribute__((__nonnull__(1)));
    extern int pthread_barrier_destroy(pthread_barrier_t *__barrier) throw () __attribute__((__nonnull__(1)));
    extern int pthread_barrier_wait(pthread_barrier_t *__barrier) throw () __attribute__((__nonnull__(1)));
    extern int pthread_barrierattr_init(pthread_barrierattr_t *__attr) throw () __attribute__((__nonnull__(1)));
    extern int pthread_barrierattr_destroy(pthread_barrierattr_t *__attr) throw () __attribute__((__nonnull__(1)));
    extern int pthread_barrierattr_getpshared(__const pthread_barrierattr_t *__restrict __attr, int *__restrict __pshared) throw () __attribute__((__nonnull__(1, 2)));
    extern int pthread_barrierattr_setpshared(pthread_barrierattr_t *__attr, int __pshared) throw () __attribute__((__nonnull__(1)));
    extern int pthread_key_create(pthread_key_t *__key, void (*__destr_function)(void *)) throw () __attribute__((__nonnull__(1)));
    extern int pthread_key_delete(pthread_key_t __key) throw ();
    extern void *pthread_getspecific(pthread_key_t __key) throw ();
    extern int pthread_setspecific(pthread_key_t __key, __const void *__pointer) throw ();
    extern int pthread_getcpuclockid(pthread_t __thread_id, __clockid_t *__clock_id) throw () __attribute__((__nonnull__(2)));
    extern int pthread_atfork(void (*__prepare)(void), void (*__parent)(void), void (*__child)(void)) throw ();
    extern __inline __attribute__((__gnu_inline__)) int pthread_equal(pthread_t __thread1, pthread_t __thread2) throw ()
    {
        return __thread1 == __thread2;
    }
}
typedef pthread_t __gthread_t;
typedef pthread_key_t __gthread_key_t;
typedef pthread_once_t __gthread_once_t;
typedef pthread_mutex_t __gthread_mutex_t;
typedef pthread_mutex_t __gthread_recursive_mutex_t;
typedef pthread_cond_t __gthread_cond_t;
typedef struct timespec __gthread_time_t;
static __typeof (pthread_once) __gthrw_pthread_once __attribute__((__weakref__("pthread_once")));
static __typeof (pthread_getspecific) __gthrw_pthread_getspecific __attribute__((__weakref__("pthread_getspecific")));
static __typeof (pthread_setspecific) __gthrw_pthread_setspecific __attribute__((__weakref__("pthread_setspecific")));
static __typeof (pthread_create) __gthrw_pthread_create __attribute__((__weakref__("pthread_create")));
static __typeof (pthread_join) __gthrw_pthread_join __attribute__((__weakref__("pthread_join")));
static __typeof (pthread_equal) __gthrw_pthread_equal __attribute__((__weakref__("pthread_equal")));
static __typeof (pthread_self) __gthrw_pthread_self __attribute__((__weakref__("pthread_self")));
static __typeof (pthread_detach) __gthrw_pthread_detach __attribute__((__weakref__("pthread_detach")));
static __typeof (pthread_cancel) __gthrw_pthread_cancel __attribute__((__weakref__("pthread_cancel")));
static __typeof (sched_yield) __gthrw_sched_yield __attribute__((__weakref__("sched_yield")));
static __typeof (pthread_mutex_lock) __gthrw_pthread_mutex_lock __attribute__((__weakref__("pthread_mutex_lock")));
static __typeof (pthread_mutex_trylock) __gthrw_pthread_mutex_trylock __attribute__((__weakref__("pthread_mutex_trylock")));
static __typeof (pthread_mutex_timedlock) __gthrw_pthread_mutex_timedlock __attribute__((__weakref__("pthread_mutex_timedlock")));
static __typeof (pthread_mutex_unlock) __gthrw_pthread_mutex_unlock __attribute__((__weakref__("pthread_mutex_unlock")));
static __typeof (pthread_mutex_init) __gthrw_pthread_mutex_init __attribute__((__weakref__("pthread_mutex_init")));
static __typeof (pthread_mutex_destroy) __gthrw_pthread_mutex_destroy __attribute__((__weakref__("pthread_mutex_destroy")));
static __typeof (pthread_cond_broadcast) __gthrw_pthread_cond_broadcast __attribute__((__weakref__("pthread_cond_broadcast")));
static __typeof (pthread_cond_signal) __gthrw_pthread_cond_signal __attribute__((__weakref__("pthread_cond_signal")));
static __typeof (pthread_cond_wait) __gthrw_pthread_cond_wait __attribute__((__weakref__("pthread_cond_wait")));
static __typeof (pthread_cond_timedwait) __gthrw_pthread_cond_timedwait __attribute__((__weakref__("pthread_cond_timedwait")));
static __typeof (pthread_cond_destroy) __gthrw_pthread_cond_destroy __attribute__((__weakref__("pthread_cond_destroy")));
static __typeof (pthread_key_create) __gthrw_pthread_key_create __attribute__((__weakref__("pthread_key_create")));
static __typeof (pthread_key_delete) __gthrw_pthread_key_delete __attribute__((__weakref__("pthread_key_delete")));
static __typeof (pthread_mutexattr_init) __gthrw_pthread_mutexattr_init __attribute__((__weakref__("pthread_mutexattr_init")));
static __typeof (pthread_mutexattr_settype) __gthrw_pthread_mutexattr_settype __attribute__((__weakref__("pthread_mutexattr_settype")));
static __typeof (pthread_mutexattr_destroy) __gthrw_pthread_mutexattr_destroy __attribute__((__weakref__("pthread_mutexattr_destroy")));
static inline int __gthread_active_p(void)
{
    static void *const __gthread_active_ptr = __extension__ (void *) &__gthrw_pthread_cancel;
    return __gthread_active_ptr != 0;
}
static inline int __gthread_create(__gthread_t *__threadid, void *(*__func)(void *), void *__args)
{
    return __gthrw_pthread_create(__threadid, __null, __func, __args);
}
static inline int __gthread_join(__gthread_t __threadid, void **__value_ptr)
{
    return __gthrw_pthread_join(__threadid, __value_ptr);
}
static inline int __gthread_detach(__gthread_t __threadid)
{
    return __gthrw_pthread_detach(__threadid);
}
static inline int __gthread_equal(__gthread_t __t1, __gthread_t __t2)
{
    return __gthrw_pthread_equal(__t1, __t2);
}
static inline __gthread_t __gthread_self(void)
{
    return __gthrw_pthread_self();
}
static inline int __gthread_yield(void)
{
    return __gthrw_sched_yield();
}
static inline int __gthread_once(__gthread_once_t *__once, void (*__func)(void))
{
    if (__gthread_active_p())
        return __gthrw_pthread_once(__once, __func);
    else
        return - 1;
}
static inline int __gthread_key_create(__gthread_key_t *__key, void (*__dtor)(void *))
{
    return __gthrw_pthread_key_create(__key, __dtor);
}
static inline int __gthread_key_delete(__gthread_key_t __key)
{
    return __gthrw_pthread_key_delete(__key);
}
static inline void *__gthread_getspecific(__gthread_key_t __key)
{
    return __gthrw_pthread_getspecific(__key);
}
static inline int __gthread_setspecific(__gthread_key_t __key, const void *__ptr)
{
    return __gthrw_pthread_setspecific(__key, __ptr);
}
static inline int __gthread_mutex_destroy(__gthread_mutex_t *__mutex)
{
    if (__gthread_active_p())
        return __gthrw_pthread_mutex_destroy(__mutex);
    else
        return 0;
}
static inline int __gthread_mutex_lock(__gthread_mutex_t *__mutex)
{
    if (__gthread_active_p())
        return __gthrw_pthread_mutex_lock(__mutex);
    else
        return 0;
}
static inline int __gthread_mutex_trylock(__gthread_mutex_t *__mutex)
{
    if (__gthread_active_p())
        return __gthrw_pthread_mutex_trylock(__mutex);
    else
        return 0;
}
static inline int __gthread_mutex_timedlock(__gthread_mutex_t *__mutex, const __gthread_time_t *__abs_timeout)
{
    if (__gthread_active_p())
        return __gthrw_pthread_mutex_timedlock(__mutex, __abs_timeout);
    else
        return 0;
}
static inline int __gthread_mutex_unlock(__gthread_mutex_t *__mutex)
{
    if (__gthread_active_p())
        return __gthrw_pthread_mutex_unlock(__mutex);
    else
        return 0;
}
static inline int __gthread_recursive_mutex_lock(__gthread_recursive_mutex_t *__mutex)
{
    return __gthread_mutex_lock(__mutex);
}
static inline int __gthread_recursive_mutex_trylock(__gthread_recursive_mutex_t *__mutex)
{
    return __gthread_mutex_trylock(__mutex);
}
static inline int __gthread_recursive_mutex_timedlock(__gthread_recursive_mutex_t *__mutex, const __gthread_time_t *__abs_timeout)
{
    return __gthread_mutex_timedlock(__mutex, __abs_timeout);
}
static inline int __gthread_recursive_mutex_unlock(__gthread_recursive_mutex_t *__mutex)
{
    return __gthread_mutex_unlock(__mutex);
}
static inline int __gthread_cond_broadcast(__gthread_cond_t *__cond)
{
    return __gthrw_pthread_cond_broadcast(__cond);
}
static inline int __gthread_cond_signal(__gthread_cond_t *__cond)
{
    return __gthrw_pthread_cond_signal(__cond);
}
static inline int __gthread_cond_wait(__gthread_cond_t *__cond, __gthread_mutex_t *__mutex)
{
    return __gthrw_pthread_cond_wait(__cond, __mutex);
}
static inline int __gthread_cond_timedwait(__gthread_cond_t *__cond, __gthread_mutex_t *__mutex, const __gthread_time_t *__abs_timeout)
{
    return __gthrw_pthread_cond_timedwait(__cond, __mutex, __abs_timeout);
}
static inline int __gthread_cond_wait_recursive(__gthread_cond_t *__cond, __gthread_recursive_mutex_t *__mutex)
{
    return __gthread_cond_wait(__cond, __mutex);
}
static inline int __gthread_cond_timedwait_recursive(__gthread_cond_t *__cond, __gthread_recursive_mutex_t *__mutex, const __gthread_time_t *__abs_timeout)
{
    return __gthread_cond_timedwait(__cond, __mutex, __abs_timeout);
}
static inline int __gthread_cond_destroy(__gthread_cond_t *__cond)
{
    return __gthrw_pthread_cond_destroy(__cond);
}
#pragma GCC visibility pop
typedef int _Atomic_word;
namespace __gnu_cxx __attribute__((__visibility__("default"))) {
    static inline _Atomic_word __exchange_and_add(volatile _Atomic_word *__mem, int __val)
    {
        return __sync_fetch_and_add(__mem, __val);
    }
    static inline void __atomic_add(volatile _Atomic_word *__mem, int __val)
    {
        __sync_fetch_and_add(__mem, __val);
    }
    static inline _Atomic_word __exchange_and_add_single(_Atomic_word *__mem, int __val)
    {
        _Atomic_word __result = *__mem;
        *__mem += __val;
        return __result;
    }
    static inline void __atomic_add_single(_Atomic_word *__mem, int __val)
    {
        *__mem += __val;
    }
    static inline _Atomic_word __attribute__((__unused__)) __exchange_and_add_dispatch(_Atomic_word *__mem, int __val)
    {
        if (__gthread_active_p())
            return __exchange_and_add(__mem, __val);
        else
            return __exchange_and_add_single(__mem, __val);
    }
    static inline void __attribute__((__unused__)) __atomic_add_dispatch(_Atomic_word *__mem, int __val)
    {
        if (__gthread_active_p())
            __atomic_add(__mem, __val);
        else
            __atomic_add_single(__mem, __val);
    }
}
namespace __gnu_cxx __attribute__((__visibility__("default"))) {
    using std::size_t;
    using std::ptrdiff_t;
    template<typename _Tp >
    class new_allocator
    {
        public :
            typedef size_t size_type;
            typedef ptrdiff_t difference_type;
            typedef _Tp *pointer;
            typedef const _Tp *const_pointer;
            typedef _Tp &reference;
            typedef const _Tp &const_reference;
            typedef _Tp value_type;
            template<typename _Tp1 >
            struct rebind
            {
                    typedef new_allocator<_Tp1> other;
            };
            new_allocator() throw ()
            {
            }
            new_allocator(const new_allocator &) throw ()
            {
            }
            template<typename _Tp1 >
            new_allocator(const new_allocator<_Tp1> &) throw ()
            {
            }
            ~new_allocator() throw ()
            {
            }
            pointer address(reference __x) const
            {
                return &__x;
            }
            const_pointer address(const_reference __x) const
            {
                return &__x;
            }
            pointer allocate(size_type __n, const void * = 0)
            {
                if (__builtin_expect(__n > this->max_size(), false))
                    std::__throw_bad_alloc();
                return static_cast<_Tp * >(::operator new(__n * sizeof(_Tp)));
            }
            void deallocate(pointer __p, size_type)
            {
                ::operator delete(__p);
            }
            size_type max_size() const throw ()
            {
                return size_t(- 1) / sizeof(_Tp);
            }
            void construct(pointer __p, const _Tp &__val)
            {
                ::new ((void *) __p) _Tp (__val);
            }
            void destroy(pointer __p)
            {
                __p->~_Tp();
            }
    };
    template<typename _Tp >
    inline bool operator ==(const new_allocator<_Tp> &, const new_allocator<_Tp> &)
    {
        return true;
    }
    template<typename _Tp >
    inline bool operator !=(const new_allocator<_Tp> &, const new_allocator<_Tp> &)
    {
        return false;
    }
}
namespace std __attribute__((__visibility__("default"))) {
    template<typename _Tp >
    class allocator;
    template<>
    class allocator<void>
    {
        public :
            typedef size_t size_type;
            typedef ptrdiff_t difference_type;
            typedef void *pointer;
            typedef const void *const_pointer;
            typedef void value_type;
            template<typename _Tp1 >
            struct rebind
            {
                    typedef allocator<_Tp1> other;
            };
    };
    template<typename _Tp >
    class allocator : public __gnu_cxx::new_allocator<_Tp>
    {
        public :
            typedef size_t size_type;
            typedef ptrdiff_t difference_type;
            typedef _Tp *pointer;
            typedef const _Tp *const_pointer;
            typedef _Tp &reference;
            typedef const _Tp &const_reference;
            typedef _Tp value_type;
            template<typename _Tp1 >
            struct rebind
            {
                    typedef allocator<_Tp1> other;
            };
            allocator() throw ()
            {
            }
            allocator(const allocator &__a) throw ()
                : __gnu_cxx::new_allocator<_Tp>(__a) 
            {
            }
            template<typename _Tp1 >
            allocator(const allocator<_Tp1> &) throw ()
            {
            }
            ~allocator() throw ()
            {
            }
    };
    template<typename _T1, typename _T2 >
    inline bool operator ==(const allocator<_T1> &, const allocator<_T2> &)
    {
        return true;
    }
    template<typename _Tp >
    inline bool operator ==(const allocator<_Tp> &, const allocator<_Tp> &)
    {
        return true;
    }
    template<typename _T1, typename _T2 >
    inline bool operator !=(const allocator<_T1> &, const allocator<_T2> &)
    {
        return false;
    }
    template<typename _Tp >
    inline bool operator !=(const allocator<_Tp> &, const allocator<_Tp> &)
    {
        return false;
    }
    extern template class allocator<char>;
    extern template class allocator<wchar_t>;
    template<typename _Alloc, bool = __is_empty(_Alloc) >
    struct __alloc_swap
    {
            static void _S_do_it(_Alloc &, _Alloc &)
            {
            }
    };
    template<typename _Alloc >
    struct __alloc_swap<_Alloc, false>
    {
            static void _S_do_it(_Alloc &__one, _Alloc &__two)
            {
                if (__one != __two)
                    swap(__one, __two);
            }
    };
    template<typename _Alloc, bool = __is_empty(_Alloc) >
    struct __alloc_neq
    {
            static bool _S_do_it(const _Alloc &, const _Alloc &)
            {
                return false;
            }
    };
    template<typename _Alloc >
    struct __alloc_neq<_Alloc, false>
    {
            static bool _S_do_it(const _Alloc &__one, const _Alloc &__two)
            {
                return __one != __two;
            }
    };
}
#pragma GCC visibility push(default)
namespace __cxxabiv1 {
    class __forced_unwind
    {
            virtual ~__forced_unwind() throw ();
            virtual void __pure_dummy()  = 0;
    };
}
#pragma GCC visibility pop
namespace std __attribute__((__visibility__("default"))) {
    template<typename _CharT, typename _Traits >
    inline void __ostream_write(basic_ostream<_CharT, _Traits> &__out, const _CharT *__s, streamsize __n)
    {
        typedef basic_ostream<_CharT, _Traits> __ostream_type;
        typedef typename __ostream_type::ios_base __ios_base;
        const streamsize __put = __out.rdbuf()->sputn(__s, __n);
        if (__put != __n)
            __out.setstate(__ios_base::badbit);
    }
    template<typename _CharT, typename _Traits >
    inline void __ostream_fill(basic_ostream<_CharT, _Traits> &__out, streamsize __n)
    {
        typedef basic_ostream<_CharT, _Traits> __ostream_type;
        typedef typename __ostream_type::ios_base __ios_base;
        const _CharT __c = __out.fill();
        for (;
            __n > 0;
            --__n)
        {
            const typename _Traits::int_type __put = __out.rdbuf()->sputc(__c);
            if (_Traits::eq_int_type(__put, _Traits::eof()))
            {
                __out.setstate(__ios_base::badbit);
                break;
            }
        }
    }
    template<typename _CharT, typename _Traits >
    basic_ostream<_CharT, _Traits> &__ostream_insert(basic_ostream<_CharT, _Traits> &__out, const _CharT *__s, streamsize __n)
    {
        typedef basic_ostream<_CharT, _Traits> __ostream_type;
        typedef typename __ostream_type::ios_base __ios_base;
        typename __ostream_type::sentry __cerb(__out);
        if (__cerb)
        {
            try
            {
                const streamsize __w = __out.width();
                if (__w > __n)
                {
                    const bool __left = ((__out.flags() & __ios_base::adjustfield) == __ios_base::left);
                    if (!__left)
                        __ostream_fill(__out, __w - __n);
                    if (__out.good())
                        __ostream_write(__out, __s, __n);
                    if (__left && __out.good())
                        __ostream_fill(__out, __w - __n);
                }
                else
                    __ostream_write(__out, __s, __n);
                __out.width(0);
            }
            catch (__cxxabiv1::__forced_unwind &)
            {
                __out._M_setstate(__ios_base::badbit);
                throw;
            }
            catch (...)
            {
                __out._M_setstate(__ios_base::badbit);
            }
        }
        return __out;
    }
    extern template ostream &__ostream_insert(ostream &, const char *, streamsize);
    extern template wostream &__ostream_insert(wostream &, const wchar_t *, streamsize);
}
namespace std __attribute__((__visibility__("default"))) {
    template<typename _Arg, typename _Result >
    struct unary_function
    {
            typedef _Arg argument_type;
            typedef _Result result_type;
    };
    template<typename _Arg1, typename _Arg2, typename _Result >
    struct binary_function
    {
            typedef _Arg1 first_argument_type;
            typedef _Arg2 second_argument_type;
            typedef _Result result_type;
    };
    template<typename _Tp >
    struct plus : public binary_function<_Tp, _Tp, _Tp>
    {
            _Tp operator ()(const _Tp &__x, const _Tp &__y) const
            {
                return __x + __y;
            }
    };
    template<typename _Tp >
    struct minus : public binary_function<_Tp, _Tp, _Tp>
    {
            _Tp operator ()(const _Tp &__x, const _Tp &__y) const
            {
                return __x - __y;
            }
    };
    template<typename _Tp >
    struct multiplies : public binary_function<_Tp, _Tp, _Tp>
    {
            _Tp operator ()(const _Tp &__x, const _Tp &__y) const
            {
                return __x * __y;
            }
    };
    template<typename _Tp >
    struct divides : public binary_function<_Tp, _Tp, _Tp>
    {
            _Tp operator ()(const _Tp &__x, const _Tp &__y) const
            {
                return __x / __y;
            }
    };
    template<typename _Tp >
    struct modulus : public binary_function<_Tp, _Tp, _Tp>
    {
            _Tp operator ()(const _Tp &__x, const _Tp &__y) const
            {
                return __x % __y;
            }
    };
    template<typename _Tp >
    struct negate : public unary_function<_Tp, _Tp>
    {
            _Tp operator ()(const _Tp &__x) const
            {
                return - __x;
            }
    };
    template<typename _Tp >
    struct equal_to : public binary_function<_Tp, _Tp, bool>
    {
            bool operator ()(const _Tp &__x, const _Tp &__y) const
            {
                return __x == __y;
            }
    };
    template<typename _Tp >
    struct not_equal_to : public binary_function<_Tp, _Tp, bool>
    {
            bool operator ()(const _Tp &__x, const _Tp &__y) const
            {
                return __x != __y;
            }
    };
    template<typename _Tp >
    struct greater : public binary_function<_Tp, _Tp, bool>
    {
            bool operator ()(const _Tp &__x, const _Tp &__y) const
            {
                return __x > __y;
            }
    };
    template<typename _Tp >
    struct less : public binary_function<_Tp, _Tp, bool>
    {
            bool operator ()(const _Tp &__x, const _Tp &__y) const
            {
                return __x < __y;
            }
    };
    template<typename _Tp >
    struct greater_equal : public binary_function<_Tp, _Tp, bool>
    {
            bool operator ()(const _Tp &__x, const _Tp &__y) const
            {
                return __x >= __y;
            }
    };
    template<typename _Tp >
    struct less_equal : public binary_function<_Tp, _Tp, bool>
    {
            bool operator ()(const _Tp &__x, const _Tp &__y) const
            {
                return __x <= __y;
            }
    };
    template<typename _Tp >
    struct logical_and : public binary_function<_Tp, _Tp, bool>
    {
            bool operator ()(const _Tp &__x, const _Tp &__y) const
            {
                return __x && __y;
            }
    };
    template<typename _Tp >
    struct logical_or : public binary_function<_Tp, _Tp, bool>
    {
            bool operator ()(const _Tp &__x, const _Tp &__y) const
            {
                return __x || __y;
            }
    };
    template<typename _Tp >
    struct logical_not : public unary_function<_Tp, bool>
    {
            bool operator ()(const _Tp &__x) const
            {
                return !__x;
            }
    };
    template<typename _Tp >
    struct bit_and : public binary_function<_Tp, _Tp, _Tp>
    {
            _Tp operator ()(const _Tp &__x, const _Tp &__y) const
            {
                return __x & __y;
            }
    };
    template<typename _Tp >
    struct bit_or : public binary_function<_Tp, _Tp, _Tp>
    {
            _Tp operator ()(const _Tp &__x, const _Tp &__y) const
            {
                return __x | __y;
            }
    };
    template<typename _Tp >
    struct bit_xor : public binary_function<_Tp, _Tp, _Tp>
    {
            _Tp operator ()(const _Tp &__x, const _Tp &__y) const
            {
                return __x ^ __y;
            }
    };
    template<typename _Predicate >
    class unary_negate : public unary_function<typename _Predicate::argument_type, bool>
    {
        protected :
            _Predicate _M_pred;
        public :
            explicit unary_negate(const _Predicate &__x)
                : _M_pred(__x) 
            {
            }
            bool operator ()(const typename _Predicate::argument_type &__x) const
            {
                return !_M_pred(__x);
            }
    };
    template<typename _Predicate >
    inline unary_negate<_Predicate> not1(const _Predicate &__pred)
    {
        return unary_negate<_Predicate>(__pred);
    }
    template<typename _Predicate >
    class binary_negate : public binary_function<typename _Predicate::first_argument_type, typename _Predicate::second_argument_type, bool>
    {
        protected :
            _Predicate _M_pred;
        public :
            explicit binary_negate(const _Predicate &__x)
                : _M_pred(__x) 
            {
            }
            bool operator ()(const typename _Predicate::first_argument_type &__x, const typename _Predicate::second_argument_type &__y) const
            {
                return !_M_pred(__x, __y);
            }
    };
    template<typename _Predicate >
    inline binary_negate<_Predicate> not2(const _Predicate &__pred)
    {
        return binary_negate<_Predicate>(__pred);
    }
    template<typename _Arg, typename _Result >
    class pointer_to_unary_function : public unary_function<_Arg, _Result>
    {
        protected :
            _Result (*_M_ptr)(_Arg);
        public :
            pointer_to_unary_function()
            {
            }
            explicit pointer_to_unary_function(_Result (*__x)(_Arg))
                : _M_ptr(__x) 
            {
            }
            _Result operator ()(_Arg __x) const
            {
                return _M_ptr(__x);
            }
    };
    template<typename _Arg, typename _Result >
    inline pointer_to_unary_function<_Arg, _Result> ptr_fun(_Result (*__x)(_Arg))
    {
        return pointer_to_unary_function<_Arg, _Result>(__x);
    }
    template<typename _Arg1, typename _Arg2, typename _Result >
    class pointer_to_binary_function : public binary_function<_Arg1, _Arg2, _Result>
    {
        protected :
            _Result (*_M_ptr)(_Arg1, _Arg2);
        public :
            pointer_to_binary_function()
            {
            }
            explicit pointer_to_binary_function(_Result (*__x)(_Arg1, _Arg2))
                : _M_ptr(__x) 
            {
            }
            _Result operator ()(_Arg1 __x, _Arg2 __y) const
            {
                return _M_ptr(__x, __y);
            }
    };
    template<typename _Arg1, typename _Arg2, typename _Result >
    inline pointer_to_binary_function<_Arg1, _Arg2, _Result> ptr_fun(_Result (*__x)(_Arg1, _Arg2))
    {
        return pointer_to_binary_function<_Arg1, _Arg2, _Result>(__x);
    }
    template<typename _Tp >
    struct _Identity : public unary_function<_Tp, _Tp>
    {
            _Tp &operator ()(_Tp &__x) const
            {
                return __x;
            }
            const _Tp &operator ()(const _Tp &__x) const
            {
                return __x;
            }
    };
    template<typename _Pair >
    struct _Select1st : public unary_function<_Pair, typename _Pair::first_type>
    {
            typename _Pair::first_type &operator ()(_Pair &__x) const
            {
                return __x.first;
            }
            const typename _Pair::first_type &operator ()(const _Pair &__x) const
            {
                return __x.first;
            }
    };
    template<typename _Pair >
    struct _Select2nd : public unary_function<_Pair, typename _Pair::second_type>
    {
            typename _Pair::second_type &operator ()(_Pair &__x) const
            {
                return __x.second;
            }
            const typename _Pair::second_type &operator ()(const _Pair &__x) const
            {
                return __x.second;
            }
    };
    template<typename _Ret, typename _Tp >
    class mem_fun_t : public unary_function<_Tp *, _Ret>
    {
        public :
            explicit mem_fun_t(_Ret (_Tp::*__pf)())
                : _M_f(__pf) 
            {
            }
            _Ret operator ()(_Tp *__p) const
            {
                return (__p ->* _M_f)();
            }
        private :
            _Ret (_Tp::*_M_f)();
    };
    template<typename _Ret, typename _Tp >
    class const_mem_fun_t : public unary_function<const _Tp *, _Ret>
    {
        public :
            explicit const_mem_fun_t(_Ret (_Tp::*__pf)() const)
                : _M_f(__pf) 
            {
            }
            _Ret operator ()(const _Tp *__p) const
            {
                return (__p ->* _M_f)();
            }
        private :
            _Ret (_Tp::*_M_f)() const;
    };
    template<typename _Ret, typename _Tp >
    class mem_fun_ref_t : public unary_function<_Tp, _Ret>
    {
        public :
            explicit mem_fun_ref_t(_Ret (_Tp::*__pf)())
                : _M_f(__pf) 
            {
            }
            _Ret operator ()(_Tp &__r) const
            {
                return (__r .* _M_f)();
            }
        private :
            _Ret (_Tp::*_M_f)();
    };
    template<typename _Ret, typename _Tp >
    class const_mem_fun_ref_t : public unary_function<_Tp, _Ret>
    {
        public :
            explicit const_mem_fun_ref_t(_Ret (_Tp::*__pf)() const)
                : _M_f(__pf) 
            {
            }
            _Ret operator ()(const _Tp &__r) const
            {
                return (__r .* _M_f)();
            }
        private :
            _Ret (_Tp::*_M_f)() const;
    };
    template<typename _Ret, typename _Tp, typename _Arg >
    class mem_fun1_t : public binary_function<_Tp *, _Arg, _Ret>
    {
        public :
            explicit mem_fun1_t(_Ret (_Tp::*__pf)(_Arg))
                : _M_f(__pf) 
            {
            }
            _Ret operator ()(_Tp *__p, _Arg __x) const
            {
                return (__p ->* _M_f)(__x);
            }
        private :
            _Ret (_Tp::*_M_f)(_Arg);
    };
    template<typename _Ret, typename _Tp, typename _Arg >
    class const_mem_fun1_t : public binary_function<const _Tp *, _Arg, _Ret>
    {
        public :
            explicit const_mem_fun1_t(_Ret (_Tp::*__pf)(_Arg) const)
                : _M_f(__pf) 
            {
            }
            _Ret operator ()(const _Tp *__p, _Arg __x) const
            {
                return (__p ->* _M_f)(__x);
            }
        private :
            _Ret (_Tp::*_M_f)(_Arg) const;
    };
    template<typename _Ret, typename _Tp, typename _Arg >
    class mem_fun1_ref_t : public binary_function<_Tp, _Arg, _Ret>
    {
        public :
            explicit mem_fun1_ref_t(_Ret (_Tp::*__pf)(_Arg))
                : _M_f(__pf) 
            {
            }
            _Ret operator ()(_Tp &__r, _Arg __x) const
            {
                return (__r .* _M_f)(__x);
            }
        private :
            _Ret (_Tp::*_M_f)(_Arg);
    };
    template<typename _Ret, typename _Tp, typename _Arg >
    class const_mem_fun1_ref_t : public binary_function<_Tp, _Arg, _Ret>
    {
        public :
            explicit const_mem_fun1_ref_t(_Ret (_Tp::*__pf)(_Arg) const)
                : _M_f(__pf) 
            {
            }
            _Ret operator ()(const _Tp &__r, _Arg __x) const
            {
                return (__r .* _M_f)(__x);
            }
        private :
            _Ret (_Tp::*_M_f)(_Arg) const;
    };
    template<typename _Ret, typename _Tp >
    inline mem_fun_t<_Ret, _Tp> mem_fun(_Ret (_Tp::*__f)())
    {
        return mem_fun_t<_Ret, _Tp>(__f);
    }
    template<typename _Ret, typename _Tp >
    inline const_mem_fun_t<_Ret, _Tp> mem_fun(_Ret (_Tp::*__f)() const)
    {
        return const_mem_fun_t<_Ret, _Tp>(__f);
    }
    template<typename _Ret, typename _Tp >
    inline mem_fun_ref_t<_Ret, _Tp> mem_fun_ref(_Ret (_Tp::*__f)())
    {
        return mem_fun_ref_t<_Ret, _Tp>(__f);
    }
    template<typename _Ret, typename _Tp >
    inline const_mem_fun_ref_t<_Ret, _Tp> mem_fun_ref(_Ret (_Tp::*__f)() const)
    {
        return const_mem_fun_ref_t<_Ret, _Tp>(__f);
    }
    template<typename _Ret, typename _Tp, typename _Arg >
    inline mem_fun1_t<_Ret, _Tp, _Arg> mem_fun(_Ret (_Tp::*__f)(_Arg))
    {
        return mem_fun1_t<_Ret, _Tp, _Arg>(__f);
    }
    template<typename _Ret, typename _Tp, typename _Arg >
    inline const_mem_fun1_t<_Ret, _Tp, _Arg> mem_fun(_Ret (_Tp::*__f)(_Arg) const)
    {
        return const_mem_fun1_t<_Ret, _Tp, _Arg>(__f);
    }
    template<typename _Ret, typename _Tp, typename _Arg >
    inline mem_fun1_ref_t<_Ret, _Tp, _Arg> mem_fun_ref(_Ret (_Tp::*__f)(_Arg))
    {
        return mem_fun1_ref_t<_Ret, _Tp, _Arg>(__f);
    }
    template<typename _Ret, typename _Tp, typename _Arg >
    inline const_mem_fun1_ref_t<_Ret, _Tp, _Arg> mem_fun_ref(_Ret (_Tp::*__f)(_Arg) const)
    {
        return const_mem_fun1_ref_t<_Ret, _Tp, _Arg>(__f);
    }
}
namespace std __attribute__((__visibility__("default"))) {
    template<typename _Operation >
    class binder1st : public unary_function<typename _Operation::second_argument_type, typename _Operation::result_type>
    {
        protected :
            _Operation op;
            typename _Operation::first_argument_type value;
        public :
            binder1st(const _Operation &__x, const typename _Operation::first_argument_type &__y)
                : op(__x), value(__y) 
            {
            }
            typename _Operation::result_type operator ()(const typename _Operation::second_argument_type &__x) const
            {
                return op(value, __x);
            }
            typename _Operation::result_type operator ()(typename _Operation::second_argument_type &__x) const
            {
                return op(value, __x);
            }
    };
    template<typename _Operation, typename _Tp >
    inline binder1st<_Operation> bind1st(const _Operation &__fn, const _Tp &__x)
    {
        typedef typename _Operation::first_argument_type _Arg1_type;
        return binder1st<_Operation>(__fn, _Arg1_type(__x));
    }
    template<typename _Operation >
    class binder2nd : public unary_function<typename _Operation::first_argument_type, typename _Operation::result_type>
    {
        protected :
            _Operation op;
            typename _Operation::second_argument_type value;
        public :
            binder2nd(const _Operation &__x, const typename _Operation::second_argument_type &__y)
                : op(__x), value(__y) 
            {
            }
            typename _Operation::result_type operator ()(const typename _Operation::first_argument_type &__x) const
            {
                return op(__x, value);
            }
            typename _Operation::result_type operator ()(typename _Operation::first_argument_type &__x) const
            {
                return op(__x, value);
            }
    };
    template<typename _Operation, typename _Tp >
    inline binder2nd<_Operation> bind2nd(const _Operation &__fn, const _Tp &__x)
    {
        typedef typename _Operation::second_argument_type _Arg2_type;
        return binder2nd<_Operation>(__fn, _Arg2_type(__x));
    }
}
namespace std __attribute__((__visibility__("default"))) {
    template<typename _CharT, typename _Traits, typename _Alloc >
    class basic_string
    {
            typedef typename _Alloc::template rebind<_CharT>::other _CharT_alloc_type;
        public :
            typedef _Traits traits_type;
            typedef typename _Traits::char_type value_type;
            typedef _Alloc allocator_type;
            typedef typename _CharT_alloc_type::size_type size_type;
            typedef typename _CharT_alloc_type::difference_type difference_type;
            typedef typename _CharT_alloc_type::reference reference;
            typedef typename _CharT_alloc_type::const_reference const_reference;
            typedef typename _CharT_alloc_type::pointer pointer;
            typedef typename _CharT_alloc_type::const_pointer const_pointer;
            typedef __gnu_cxx::__normal_iterator<pointer, basic_string> iterator;
            typedef __gnu_cxx::__normal_iterator<const_pointer, basic_string> const_iterator;
            typedef std::reverse_iterator<const_iterator> const_reverse_iterator;
            typedef std::reverse_iterator<iterator> reverse_iterator;
        private :
            struct _Rep_base
            {
                    size_type _M_length;
                    size_type _M_capacity;
                    _Atomic_word _M_refcount;
            };
            struct _Rep : _Rep_base
            {
                    typedef typename _Alloc::template rebind<char>::other _Raw_bytes_alloc;
                    static const size_type _S_max_size;
                    static const _CharT _S_terminal;
                    static size_type _S_empty_rep_storage[];
                    static _Rep &_S_empty_rep()
                    {
                        void *__p = reinterpret_cast<void * >(&_S_empty_rep_storage);
                        return *reinterpret_cast<_Rep * >(__p);
                    }
                    bool _M_is_leaked() const
                    {
                        return this->_M_refcount < 0;
                    }
                    bool _M_is_shared() const
                    {
                        return this->_M_refcount > 0;
                    }
                    void _M_set_leaked()
                    {
                        this->_M_refcount = - 1;
                    }
                    void _M_set_sharable()
                    {
                        this->_M_refcount = 0;
                    }
                    void _M_set_length_and_sharable(size_type __n)
                    {
                        if (__builtin_expect(this != &_S_empty_rep(), false))
                        {
                            this->_M_set_sharable();
                            this->_M_length = __n;
                            traits_type::assign(this->_M_refdata()[__n], _S_terminal);
                        }
                    }
                    _CharT *_M_refdata() throw ()
                    {
                        return reinterpret_cast<_CharT * >(this + 1);
                    }
                    _CharT *_M_grab(const _Alloc &__alloc1, const _Alloc &__alloc2)
                    {
                        return (!_M_is_leaked() && __alloc1 == __alloc2) ? _M_refcopy() : _M_clone(__alloc1);
                    }
                    static _Rep *_S_create(size_type, size_type, const _Alloc &);
                    void _M_dispose(const _Alloc &__a)
                    {
                        if (__builtin_expect(this != &_S_empty_rep(), false))
                            if (__gnu_cxx::__exchange_and_add_dispatch(&this->_M_refcount, - 1) <= 0)
                                _M_destroy(__a);
                    }
                    void _M_destroy(const _Alloc &) throw ();
                    _CharT *_M_refcopy() throw ()
                    {
                        if (__builtin_expect(this != &_S_empty_rep(), false))
                            __gnu_cxx::__atomic_add_dispatch(&this->_M_refcount, 1);
                        return _M_refdata();
                    }
                    _CharT *_M_clone(const _Alloc &, size_type __res = 0);
            };
            struct _Alloc_hider : _Alloc
            {
                    _Alloc_hider(_CharT *__dat, const _Alloc &__a)
                        : _Alloc(__a), _M_p(__dat) 
                    {
                    }
                    _CharT *_M_p;
            };
        public :
            static const size_type npos  = static_cast<size_type >(- 1);
        private :
            mutable _Alloc_hider _M_dataplus;
            _CharT *_M_data() const
            {
                return _M_dataplus._M_p;
            }
            _CharT *_M_data(_CharT *__p)
            {
                return (_M_dataplus._M_p = __p);
            }
            _Rep *_M_rep() const
            {
                return &((reinterpret_cast<_Rep * >(_M_data()))[- 1]);
            }
            iterator _M_ibegin() const
            {
                return iterator(_M_data());
            }
            iterator _M_iend() const
            {
                return iterator(_M_data() + this->size());
            }
            void _M_leak()
            {
                if (!_M_rep()->_M_is_leaked())
                    _M_leak_hard();
            }
            size_type _M_check(size_type __pos, const char *__s) const
            {
                if (__pos > this->size())
                    __throw_out_of_range((__s));
                return __pos;
            }
            void _M_check_length(size_type __n1, size_type __n2, const char *__s) const
            {
                if (this->max_size() - (this->size() - __n1) < __n2)
                    __throw_length_error((__s));
            }
            size_type _M_limit(size_type __pos, size_type __off) const
            {
                const bool __testoff = __off < this->size() - __pos;
                return __testoff ? __off : this->size() - __pos;
            }
            bool _M_disjunct(const _CharT *__s) const
            {
                return (less<const _CharT *>()(__s, _M_data()) || less<const _CharT *>()(_M_data() + this->size(), __s));
            }
            static void _M_copy(_CharT *__d, const _CharT *__s, size_type __n)
            {
                if (__n == 1)
                    traits_type::assign(*__d, *__s);
                else
                    traits_type::copy(__d, __s, __n);
            }
            static void _M_move(_CharT *__d, const _CharT *__s, size_type __n)
            {
                if (__n == 1)
                    traits_type::assign(*__d, *__s);
                else
                    traits_type::move(__d, __s, __n);
            }
            static void _M_assign(_CharT *__d, size_type __n, _CharT __c)
            {
                if (__n == 1)
                    traits_type::assign(*__d, __c);
                else
                    traits_type::assign(__d, __n, __c);
            }
            template<class _Iterator >
            static void _S_copy_chars(_CharT *__p, _Iterator __k1, _Iterator __k2)
            {
                for (;
                    __k1 != __k2;
                    ++__k1 , ++__p)
                    traits_type::assign(*__p, *__k1);
            }
            static void _S_copy_chars(_CharT *__p, iterator __k1, iterator __k2)
            {
                _S_copy_chars(__p, __k1.base(), __k2.base());
            }
            static void _S_copy_chars(_CharT *__p, const_iterator __k1, const_iterator __k2)
            {
                _S_copy_chars(__p, __k1.base(), __k2.base());
            }
            static void _S_copy_chars(_CharT *__p, _CharT *__k1, _CharT *__k2)
            {
                _M_copy(__p, __k1, __k2 - __k1);
            }
            static void _S_copy_chars(_CharT *__p, const _CharT *__k1, const _CharT *__k2)
            {
                _M_copy(__p, __k1, __k2 - __k1);
            }
            static int _S_compare(size_type __n1, size_type __n2)
            {
                const difference_type __d = difference_type(__n1 - __n2);
                if (__d > __gnu_cxx::__numeric_traits<int>::__max)
                    return __gnu_cxx::__numeric_traits<int>::__max;
                else
                    if (__d < __gnu_cxx::__numeric_traits<int>::__min)
                        return __gnu_cxx::__numeric_traits<int>::__min;
                    else
                        return int(__d);
            }
            void _M_mutate(size_type __pos, size_type __len1, size_type __len2);
            void _M_leak_hard();
            static _Rep &_S_empty_rep()
            {
                return _Rep::_S_empty_rep();
            }
        public :
            inline basic_string();
            explicit basic_string(const _Alloc &__a);
            basic_string(const basic_string &__str);
            basic_string(const basic_string &__str, size_type __pos, size_type __n = npos);
            basic_string(const basic_string &__str, size_type __pos, size_type __n, const _Alloc &__a);
            basic_string(const _CharT *__s, size_type __n, const _Alloc &__a = _Alloc());
            basic_string(const _CharT *__s, const _Alloc &__a = _Alloc());
            basic_string(size_type __n, _CharT __c, const _Alloc &__a = _Alloc());
            template<class _InputIterator >
            basic_string(_InputIterator __beg, _InputIterator __end, const _Alloc &__a = _Alloc());
            ~basic_string()
            {
                _M_rep()->_M_dispose(this->get_allocator());
            }
            basic_string &operator =(const basic_string &__str)
            {
                return this->assign(__str);
            }
            basic_string &operator =(const _CharT *__s)
            {
                return this->assign(__s);
            }
            basic_string &operator =(_CharT __c)
            {
                this->assign(1, __c);
                return *this;
            }
            iterator begin()
            {
                _M_leak();
                return iterator(_M_data());
            }
            const_iterator begin() const
            {
                return const_iterator(_M_data());
            }
            iterator end()
            {
                _M_leak();
                return iterator(_M_data() + this->size());
            }
            const_iterator end() const
            {
                return const_iterator(_M_data() + this->size());
            }
            reverse_iterator rbegin()
            {
                return reverse_iterator(this->end());
            }
            const_reverse_iterator rbegin() const
            {
                return const_reverse_iterator(this->end());
            }
            reverse_iterator rend()
            {
                return reverse_iterator(this->begin());
            }
            const_reverse_iterator rend() const
            {
                return const_reverse_iterator(this->begin());
            }
        public :
            size_type size() const
            {
                return _M_rep()->_M_length;
            }
            size_type length() const
            {
                return _M_rep()->_M_length;
            }
            size_type max_size() const
            {
                return _Rep::_S_max_size;
            }
            void resize(size_type __n, _CharT __c);
            void resize(size_type __n)
            {
                this->resize(__n, _CharT());
            }
            size_type capacity() const
            {
                return _M_rep()->_M_capacity;
            }
            void reserve(size_type __res_arg = 0);
            void clear()
            {
                _M_mutate(0, this->size(), 0);
            }
            bool empty() const
            {
                return this->size() == 0;
            }
            const_reference operator [](size_type __pos) const
            {
                ;
                return _M_data()[__pos];
            }
            reference operator [](size_type __pos)
            {
                ;
                ;
                _M_leak();
                return _M_data()[__pos];
            }
            const_reference at(size_type __n) const
            {
                if (__n >= this->size())
                    __throw_out_of_range(("basic_string::at"));
                return _M_data()[__n];
            }
            reference at(size_type __n)
            {
                if (__n >= size())
                    __throw_out_of_range(("basic_string::at"));
                _M_leak();
                return _M_data()[__n];
            }
            basic_string &operator +=(const basic_string &__str)
            {
                return this->append(__str);
            }
            basic_string &operator +=(const _CharT *__s)
            {
                return this->append(__s);
            }
            basic_string &operator +=(_CharT __c)
            {
                this->push_back(__c);
                return *this;
            }
            basic_string &append(const basic_string &__str);
            basic_string &append(const basic_string &__str, size_type __pos, size_type __n);
            basic_string &append(const _CharT *__s, size_type __n);
            basic_string &append(const _CharT *__s)
            {
                ;
                return this->append(__s, traits_type::length(__s));
            }
            basic_string &append(size_type __n, _CharT __c);
            template<class _InputIterator >
            basic_string &append(_InputIterator __first, _InputIterator __last)
            {
                return this->replace(_M_iend(), _M_iend(), __first, __last);
            }
            void push_back(_CharT __c)
            {
                const size_type __len = 1 + this->size();
                if (__len > this->capacity() || _M_rep()->_M_is_shared())
                    this->reserve(__len);
                traits_type::assign(_M_data()[this->size()], __c);
                _M_rep()->_M_set_length_and_sharable(__len);
            }
            basic_string &assign(const basic_string &__str);
            basic_string &assign(const basic_string &__str, size_type __pos, size_type __n)
            {
                return this->assign(__str._M_data() + __str._M_check(__pos, "basic_string::assign"), __str._M_limit(__pos, __n));
            }
            basic_string &assign(const _CharT *__s, size_type __n);
            basic_string &assign(const _CharT *__s)
            {
                ;
                return this->assign(__s, traits_type::length(__s));
            }
            basic_string &assign(size_type __n, _CharT __c)
            {
                return _M_replace_aux(size_type(0), this->size(), __n, __c);
            }
            template<class _InputIterator >
            basic_string &assign(_InputIterator __first, _InputIterator __last)
            {
                return this->replace(_M_ibegin(), _M_iend(), __first, __last);
            }
            void insert(iterator __p, size_type __n, _CharT __c)
            {
                this->replace(__p, __p, __n, __c);
            }
            template<class _InputIterator >
            void insert(iterator __p, _InputIterator __beg, _InputIterator __end)
            {
                this->replace(__p, __p, __beg, __end);
            }
            basic_string &insert(size_type __pos1, const basic_string &__str)
            {
                return this->insert(__pos1, __str, size_type(0), __str.size());
            }
            basic_string &insert(size_type __pos1, const basic_string &__str, size_type __pos2, size_type __n)
            {
                return this->insert(__pos1, __str._M_data() + __str._M_check(__pos2, "basic_string::insert"), __str._M_limit(__pos2, __n));
            }
            basic_string &insert(size_type __pos, const _CharT *__s, size_type __n);
            basic_string &insert(size_type __pos, const _CharT *__s)
            {
                ;
                return this->insert(__pos, __s, traits_type::length(__s));
            }
            basic_string &insert(size_type __pos, size_type __n, _CharT __c)
            {
                return _M_replace_aux(_M_check(__pos, "basic_string::insert"), size_type(0), __n, __c);
            }
            iterator insert(iterator __p, _CharT __c)
            {
                ;
                const size_type __pos = __p - _M_ibegin();
                _M_replace_aux(__pos, size_type(0), size_type(1), __c);
                _M_rep()->_M_set_leaked();
                return iterator(_M_data() + __pos);
            }
            basic_string &erase(size_type __pos = 0, size_type __n = npos)
            {
                _M_mutate(_M_check(__pos, "basic_string::erase"), _M_limit(__pos, __n), size_type(0));
                return *this;
            }
            iterator erase(iterator __position)
            {
                ;
                const size_type __pos = __position - _M_ibegin();
                _M_mutate(__pos, size_type(1), size_type(0));
                _M_rep()->_M_set_leaked();
                return iterator(_M_data() + __pos);
            }
            iterator erase(iterator __first, iterator __last);
            basic_string &replace(size_type __pos, size_type __n, const basic_string &__str)
            {
                return this->replace(__pos, __n, __str._M_data(), __str.size());
            }
            basic_string &replace(size_type __pos1, size_type __n1, const basic_string &__str, size_type __pos2, size_type __n2)
            {
                return this->replace(__pos1, __n1, __str._M_data() + __str._M_check(__pos2, "basic_string::replace"), __str._M_limit(__pos2, __n2));
            }
            basic_string &replace(size_type __pos, size_type __n1, const _CharT *__s, size_type __n2);
            basic_string &replace(size_type __pos, size_type __n1, const _CharT *__s)
            {
                ;
                return this->replace(__pos, __n1, __s, traits_type::length(__s));
            }
            basic_string &replace(size_type __pos, size_type __n1, size_type __n2, _CharT __c)
            {
                return _M_replace_aux(_M_check(__pos, "basic_string::replace"), _M_limit(__pos, __n1), __n2, __c);
            }
            basic_string &replace(iterator __i1, iterator __i2, const basic_string &__str)
            {
                return this->replace(__i1, __i2, __str._M_data(), __str.size());
            }
            basic_string &replace(iterator __i1, iterator __i2, const _CharT *__s, size_type __n)
            {
                ;
                return this->replace(__i1 - _M_ibegin(), __i2 - __i1, __s, __n);
            }
            basic_string &replace(iterator __i1, iterator __i2, const _CharT *__s)
            {
                ;
                return this->replace(__i1, __i2, __s, traits_type::length(__s));
            }
            basic_string &replace(iterator __i1, iterator __i2, size_type __n, _CharT __c)
            {
                ;
                return _M_replace_aux(__i1 - _M_ibegin(), __i2 - __i1, __n, __c);
            }
            template<class _InputIterator >
            basic_string &replace(iterator __i1, iterator __i2, _InputIterator __k1, _InputIterator __k2)
            {
                ;
                ;
                typedef typename std::__is_integer<_InputIterator>::__type _Integral;
                return _M_replace_dispatch(__i1, __i2, __k1, __k2, _Integral());
            }
            basic_string &replace(iterator __i1, iterator __i2, _CharT *__k1, _CharT *__k2)
            {
                ;
                ;
                return this->replace(__i1 - _M_ibegin(), __i2 - __i1, __k1, __k2 - __k1);
            }
            basic_string &replace(iterator __i1, iterator __i2, const _CharT *__k1, const _CharT *__k2)
            {
                ;
                ;
                return this->replace(__i1 - _M_ibegin(), __i2 - __i1, __k1, __k2 - __k1);
            }
            basic_string &replace(iterator __i1, iterator __i2, iterator __k1, iterator __k2)
            {
                ;
                ;
                return this->replace(__i1 - _M_ibegin(), __i2 - __i1, __k1.base(), __k2 - __k1);
            }
            basic_string &replace(iterator __i1, iterator __i2, const_iterator __k1, const_iterator __k2)
            {
                ;
                ;
                return this->replace(__i1 - _M_ibegin(), __i2 - __i1, __k1.base(), __k2 - __k1);
            }
        private :
            template<class _Integer >
            basic_string &_M_replace_dispatch(iterator __i1, iterator __i2, _Integer __n, _Integer __val, __true_type)
            {
                return _M_replace_aux(__i1 - _M_ibegin(), __i2 - __i1, __n, __val);
            }
            template<class _InputIterator >
            basic_string &_M_replace_dispatch(iterator __i1, iterator __i2, _InputIterator __k1, _InputIterator __k2, __false_type);
            basic_string &_M_replace_aux(size_type __pos1, size_type __n1, size_type __n2, _CharT __c);
            basic_string &_M_replace_safe(size_type __pos1, size_type __n1, const _CharT *__s, size_type __n2);
            template<class _InIterator >
            static _CharT *_S_construct_aux(_InIterator __beg, _InIterator __end, const _Alloc &__a, __false_type)
            {
                typedef typename iterator_traits<_InIterator>::iterator_category _Tag;
                return _S_construct(__beg, __end, __a, _Tag());
            }
            template<class _Integer >
            static _CharT *_S_construct_aux(_Integer __beg, _Integer __end, const _Alloc &__a, __true_type)
            {
                return _S_construct(static_cast<size_type >(__beg), __end, __a);
            }
            template<class _InIterator >
            static _CharT *_S_construct(_InIterator __beg, _InIterator __end, const _Alloc &__a)
            {
                typedef typename std::__is_integer<_InIterator>::__type _Integral;
                return _S_construct_aux(__beg, __end, __a, _Integral());
            }
            template<class _InIterator >
            static _CharT *_S_construct(_InIterator __beg, _InIterator __end, const _Alloc &__a, input_iterator_tag);
            template<class _FwdIterator >
            static _CharT *_S_construct(_FwdIterator __beg, _FwdIterator __end, const _Alloc &__a, forward_iterator_tag);
            static _CharT *_S_construct(size_type __req, _CharT __c, const _Alloc &__a);
        public :
            size_type copy(_CharT *__s, size_type __n, size_type __pos = 0) const;
            void swap(basic_string &__s);
            const _CharT *c_str() const
            {
                return _M_data();
            }
            const _CharT *data() const
            {
                return _M_data();
            }
            allocator_type get_allocator() const
            {
                return _M_dataplus;
            }
            size_type find(const _CharT *__s, size_type __pos, size_type __n) const;
            size_type find(const basic_string &__str, size_type __pos = 0) const
            {
                return this->find(__str.data(), __pos, __str.size());
            }
            size_type find(const _CharT *__s, size_type __pos = 0) const
            {
                ;
                return this->find(__s, __pos, traits_type::length(__s));
            }
            size_type find(_CharT __c, size_type __pos = 0) const;
            size_type rfind(const basic_string &__str, size_type __pos = npos) const
            {
                return this->rfind(__str.data(), __pos, __str.size());
            }
            size_type rfind(const _CharT *__s, size_type __pos, size_type __n) const;
            size_type rfind(const _CharT *__s, size_type __pos = npos) const
            {
                ;
                return this->rfind(__s, __pos, traits_type::length(__s));
            }
            size_type rfind(_CharT __c, size_type __pos = npos) const;
            size_type find_first_of(const basic_string &__str, size_type __pos = 0) const
            {
                return this->find_first_of(__str.data(), __pos, __str.size());
            }
            size_type find_first_of(const _CharT *__s, size_type __pos, size_type __n) const;
            size_type find_first_of(const _CharT *__s, size_type __pos = 0) const
            {
                ;
                return this->find_first_of(__s, __pos, traits_type::length(__s));
            }
            size_type find_first_of(_CharT __c, size_type __pos = 0) const
            {
                return this->find(__c, __pos);
            }
            size_type find_last_of(const basic_string &__str, size_type __pos = npos) const
            {
                return this->find_last_of(__str.data(), __pos, __str.size());
            }
            size_type find_last_of(const _CharT *__s, size_type __pos, size_type __n) const;
            size_type find_last_of(const _CharT *__s, size_type __pos = npos) const
            {
                ;
                return this->find_last_of(__s, __pos, traits_type::length(__s));
            }
            size_type find_last_of(_CharT __c, size_type __pos = npos) const
            {
                return this->rfind(__c, __pos);
            }
            size_type find_first_not_of(const basic_string &__str, size_type __pos = 0) const
            {
                return this->find_first_not_of(__str.data(), __pos, __str.size());
            }
            size_type find_first_not_of(const _CharT *__s, size_type __pos, size_type __n) const;
            size_type find_first_not_of(const _CharT *__s, size_type __pos = 0) const
            {
                ;
                return this->find_first_not_of(__s, __pos, traits_type::length(__s));
            }
            size_type find_first_not_of(_CharT __c, size_type __pos = 0) const;
            size_type find_last_not_of(const basic_string &__str, size_type __pos = npos) const
            {
                return this->find_last_not_of(__str.data(), __pos, __str.size());
            }
            size_type find_last_not_of(const _CharT *__s, size_type __pos, size_type __n) const;
            size_type find_last_not_of(const _CharT *__s, size_type __pos = npos) const
            {
                ;
                return this->find_last_not_of(__s, __pos, traits_type::length(__s));
            }
            size_type find_last_not_of(_CharT __c, size_type __pos = npos) const;
            basic_string substr(size_type __pos = 0, size_type __n = npos) const
            {
                return basic_string(*this, _M_check(__pos, "basic_string::substr"), __n);
            }
            int compare(const basic_string &__str) const
            {
                const size_type __size = this->size();
                const size_type __osize = __str.size();
                const size_type __len = std::min(__size, __osize);
                int __r = traits_type::compare(_M_data(), __str.data(), __len);
                if (!__r)
                    __r = _S_compare(__size, __osize);
                return __r;
            }
            int compare(size_type __pos, size_type __n, const basic_string &__str) const;
            int compare(size_type __pos1, size_type __n1, const basic_string &__str, size_type __pos2, size_type __n2) const;
            int compare(const _CharT *__s) const;
            int compare(size_type __pos, size_type __n1, const _CharT *__s) const;
            int compare(size_type __pos, size_type __n1, const _CharT *__s, size_type __n2) const;
    };
    template<typename _CharT, typename _Traits, typename _Alloc >
    inline basic_string<_CharT, _Traits, _Alloc>::basic_string()
        : _M_dataplus(_S_empty_rep()._M_refdata(), _Alloc()) 
    {
    }
    template<typename _CharT, typename _Traits, typename _Alloc >
    basic_string<_CharT, _Traits, _Alloc> operator +(const basic_string<_CharT, _Traits, _Alloc> &__lhs, const basic_string<_CharT, _Traits, _Alloc> &__rhs)
    {
        basic_string<_CharT, _Traits, _Alloc> __str(__lhs);
        __str.append(__rhs);
        return __str;
    }
    template<typename _CharT, typename _Traits, typename _Alloc >
    basic_string<_CharT, _Traits, _Alloc> operator +(const _CharT *__lhs, const basic_string<_CharT, _Traits, _Alloc> &__rhs);
    template<typename _CharT, typename _Traits, typename _Alloc >
    basic_string<_CharT, _Traits, _Alloc> operator +(_CharT __lhs, const basic_string<_CharT, _Traits, _Alloc> &__rhs);
    template<typename _CharT, typename _Traits, typename _Alloc >
    inline basic_string<_CharT, _Traits, _Alloc> operator +(const basic_string<_CharT, _Traits, _Alloc> &__lhs, const _CharT *__rhs)
    {
        basic_string<_CharT, _Traits, _Alloc> __str(__lhs);
        __str.append(__rhs);
        return __str;
    }
    template<typename _CharT, typename _Traits, typename _Alloc >
    inline basic_string<_CharT, _Traits, _Alloc> operator +(const basic_string<_CharT, _Traits, _Alloc> &__lhs, _CharT __rhs)
    {
        typedef basic_string<_CharT, _Traits, _Alloc> __string_type;
        typedef typename __string_type::size_type __size_type;
        __string_type __str(__lhs);
        __str.append(__size_type(1), __rhs);
        return __str;
    }
    template<typename _CharT, typename _Traits, typename _Alloc >
    inline bool operator ==(const basic_string<_CharT, _Traits, _Alloc> &__lhs, const basic_string<_CharT, _Traits, _Alloc> &__rhs)
    {
        return __lhs.compare(__rhs) == 0;
    }
    template<typename _CharT >
    inline typename __gnu_cxx::__enable_if<__is_char<_CharT>::__value, bool>::__type operator ==(const basic_string<_CharT> &__lhs, const basic_string<_CharT> &__rhs)
    {
        return (__lhs.size() == __rhs.size() && !std::char_traits<_CharT>::compare(__lhs.data(), __rhs.data(), __lhs.size()));
    }
    template<typename _CharT, typename _Traits, typename _Alloc >
    inline bool operator ==(const _CharT *__lhs, const basic_string<_CharT, _Traits, _Alloc> &__rhs)
    {
        return __rhs.compare(__lhs) == 0;
    }
    template<typename _CharT, typename _Traits, typename _Alloc >
    inline bool operator ==(const basic_string<_CharT, _Traits, _Alloc> &__lhs, const _CharT *__rhs)
    {
        return __lhs.compare(__rhs) == 0;
    }
    template<typename _CharT, typename _Traits, typename _Alloc >
    inline bool operator !=(const basic_string<_CharT, _Traits, _Alloc> &__lhs, const basic_string<_CharT, _Traits, _Alloc> &__rhs)
    {
        return !(__lhs == __rhs);
    }
    template<typename _CharT, typename _Traits, typename _Alloc >
    inline bool operator !=(const _CharT *__lhs, const basic_string<_CharT, _Traits, _Alloc> &__rhs)
    {
        return !(__lhs == __rhs);
    }
    template<typename _CharT, typename _Traits, typename _Alloc >
    inline bool operator !=(const basic_string<_CharT, _Traits, _Alloc> &__lhs, const _CharT *__rhs)
    {
        return !(__lhs == __rhs);
    }
    template<typename _CharT, typename _Traits, typename _Alloc >
    inline bool operator <(const basic_string<_CharT, _Traits, _Alloc> &__lhs, const basic_string<_CharT, _Traits, _Alloc> &__rhs)
    {
        return __lhs.compare(__rhs) < 0;
    }
    template<typename _CharT, typename _Traits, typename _Alloc >
    inline bool operator <(const basic_string<_CharT, _Traits, _Alloc> &__lhs, const _CharT *__rhs)
    {
        return __lhs.compare(__rhs) < 0;
    }
    template<typename _CharT, typename _Traits, typename _Alloc >
    inline bool operator <(const _CharT *__lhs, const basic_string<_CharT, _Traits, _Alloc> &__rhs)
    {
        return __rhs.compare(__lhs) > 0;
    }
    template<typename _CharT, typename _Traits, typename _Alloc >
    inline bool operator >(const basic_string<_CharT, _Traits, _Alloc> &__lhs, const basic_string<_CharT, _Traits, _Alloc> &__rhs)
    {
        return __lhs.compare(__rhs) > 0;
    }
    template<typename _CharT, typename _Traits, typename _Alloc >
    inline bool operator >(const basic_string<_CharT, _Traits, _Alloc> &__lhs, const _CharT *__rhs)
    {
        return __lhs.compare(__rhs) > 0;
    }
    template<typename _CharT, typename _Traits, typename _Alloc >
    inline bool operator >(const _CharT *__lhs, const basic_string<_CharT, _Traits, _Alloc> &__rhs)
    {
        return __rhs.compare(__lhs) < 0;
    }
    template<typename _CharT, typename _Traits, typename _Alloc >
    inline bool operator <=(const basic_string<_CharT, _Traits, _Alloc> &__lhs, const basic_string<_CharT, _Traits, _Alloc> &__rhs)
    {
        return __lhs.compare(__rhs) <= 0;
    }
    template<typename _CharT, typename _Traits, typename _Alloc >
    inline bool operator <=(const basic_string<_CharT, _Traits, _Alloc> &__lhs, const _CharT *__rhs)
    {
        return __lhs.compare(__rhs) <= 0;
    }
    template<typename _CharT, typename _Traits, typename _Alloc >
    inline bool operator <=(const _CharT *__lhs, const basic_string<_CharT, _Traits, _Alloc> &__rhs)
    {
        return __rhs.compare(__lhs) >= 0;
    }
    template<typename _CharT, typename _Traits, typename _Alloc >
    inline bool operator >=(const basic_string<_CharT, _Traits, _Alloc> &__lhs, const basic_string<_CharT, _Traits, _Alloc> &__rhs)
    {
        return __lhs.compare(__rhs) >= 0;
    }
    template<typename _CharT, typename _Traits, typename _Alloc >
    inline bool operator >=(const basic_string<_CharT, _Traits, _Alloc> &__lhs, const _CharT *__rhs)
    {
        return __lhs.compare(__rhs) >= 0;
    }
    template<typename _CharT, typename _Traits, typename _Alloc >
    inline bool operator >=(const _CharT *__lhs, const basic_string<_CharT, _Traits, _Alloc> &__rhs)
    {
        return __rhs.compare(__lhs) <= 0;
    }
    template<typename _CharT, typename _Traits, typename _Alloc >
    inline void swap(basic_string<_CharT, _Traits, _Alloc> &__lhs, basic_string<_CharT, _Traits, _Alloc> &__rhs)
    {
        __lhs.swap(__rhs);
    }
    template<typename _CharT, typename _Traits, typename _Alloc >
    basic_istream<_CharT, _Traits> &operator >>(basic_istream<_CharT, _Traits> &__is, basic_string<_CharT, _Traits, _Alloc> &__str);
    template<>
    basic_istream<char> &operator >>(basic_istream<char> &__is, basic_string<char> &__str);
    template<typename _CharT, typename _Traits, typename _Alloc >
    inline basic_ostream<_CharT, _Traits> &operator <<(basic_ostream<_CharT, _Traits> &__os, const basic_string<_CharT, _Traits, _Alloc> &__str)
    {
        return __ostream_insert(__os, __str.data(), __str.size());
    }
    template<typename _CharT, typename _Traits, typename _Alloc >
    basic_istream<_CharT, _Traits> &getline(basic_istream<_CharT, _Traits> &__is, basic_string<_CharT, _Traits, _Alloc> &__str, _CharT __delim);
    template<typename _CharT, typename _Traits, typename _Alloc >
    inline basic_istream<_CharT, _Traits> &getline(basic_istream<_CharT, _Traits> &__is, basic_string<_CharT, _Traits, _Alloc> &__str)
    {
        return getline(__is, __str, __is.widen('\n'));
    }
    template<>
    basic_istream<char> &getline(basic_istream<char> &__in, basic_string<char> &__str, char __delim);
    template<>
    basic_istream<wchar_t> &getline(basic_istream<wchar_t> &__in, basic_string<wchar_t> &__str, wchar_t __delim);
}
namespace std __attribute__((__visibility__("default"))) {
    template<typename _CharT, typename _Traits, typename _Alloc >
    const typename basic_string<_CharT, _Traits, _Alloc>::size_type basic_string<_CharT, _Traits, _Alloc>::_Rep::_S_max_size = (((npos - sizeof(_Rep_base)) / sizeof(_CharT)) - 1) / 4;
    template<typename _CharT, typename _Traits, typename _Alloc >
    const _CharT basic_string<_CharT, _Traits, _Alloc>::_Rep::_S_terminal = _CharT();
    template<typename _CharT, typename _Traits, typename _Alloc >
    const typename basic_string<_CharT, _Traits, _Alloc>::size_type basic_string<_CharT, _Traits, _Alloc>::npos;
    template<typename _CharT, typename _Traits, typename _Alloc >
    typename basic_string<_CharT, _Traits, _Alloc>::size_type basic_string<_CharT, _Traits, _Alloc>::_Rep::_S_empty_rep_storage[(sizeof(_Rep_base) + sizeof(_CharT) + sizeof(size_type) - 1) / sizeof(size_type)];
    template<typename _CharT, typename _Traits, typename _Alloc >
    template<typename _InIterator >
    _CharT *basic_string<_CharT, _Traits, _Alloc>::_S_construct(_InIterator __beg, _InIterator __end, const _Alloc &__a, input_iterator_tag)
    {
        if (__beg == __end && __a == _Alloc())
            return _S_empty_rep()._M_refdata();
        _CharT __buf[128];
        size_type __len = 0;
        while (__beg != __end && __len < sizeof (__buf) / sizeof(_CharT))
        {
            __buf[__len++] = *__beg;
            ++__beg;
        }
        _Rep *__r = _Rep::_S_create(__len, size_type(0), __a);
        _M_copy(__r->_M_refdata(), __buf, __len);
        try
        {
            while (__beg != __end)
            {
                if (__len == __r->_M_capacity)
                {
                    _Rep *__another = _Rep::_S_create(__len + 1, __len, __a);
                    _M_copy(__another->_M_refdata(), __r->_M_refdata(), __len);
                    __r->_M_destroy(__a);
                    __r = __another;
                }
                __r->_M_refdata()[__len++] = *__beg;
                ++__beg;
            }
        }
        catch (...)
        {
            __r->_M_destroy(__a);
            throw;
        }
        __r->_M_set_length_and_sharable(__len);
        return __r->_M_refdata();
    }
    template<typename _CharT, typename _Traits, typename _Alloc >
    template<typename _InIterator >
    _CharT *basic_string<_CharT, _Traits, _Alloc>::_S_construct(_InIterator __beg, _InIterator __end, const _Alloc &__a, forward_iterator_tag)
    {
        if (__beg == __end && __a == _Alloc())
            return _S_empty_rep()._M_refdata();
        if (__builtin_expect(__gnu_cxx::__is_null_pointer(__beg) && __beg != __end, 0))
            __throw_logic_error(("basic_string::_S_construct NULL not valid"));
        const size_type __dnew = static_cast<size_type >(std::distance(__beg, __end));
        _Rep *__r = _Rep::_S_create(__dnew, size_type(0), __a);
        try
        {
            _S_copy_chars(__r->_M_refdata(), __beg, __end);
        }
        catch (...)
        {
            __r->_M_destroy(__a);
            throw;
        }
        __r->_M_set_length_and_sharable(__dnew);
        return __r->_M_refdata();
    }
    template<typename _CharT, typename _Traits, typename _Alloc >
    _CharT *basic_string<_CharT, _Traits, _Alloc>::_S_construct(size_type __n, _CharT __c, const _Alloc &__a)
    {
        if (__n == 0 && __a == _Alloc())
            return _S_empty_rep()._M_refdata();
        _Rep *__r = _Rep::_S_create(__n, size_type(0), __a);
        if (__n)
            _M_assign(__r->_M_refdata(), __n, __c);
        __r->_M_set_length_and_sharable(__n);
        return __r->_M_refdata();
    }
    template<typename _CharT, typename _Traits, typename _Alloc >
    basic_string<_CharT, _Traits, _Alloc>::basic_string(const basic_string &__str)
        : _M_dataplus(__str._M_rep()->_M_grab(_Alloc(__str.get_allocator()), __str.get_allocator()), __str.get_allocator()) 
    {
    }
    template<typename _CharT, typename _Traits, typename _Alloc >
    basic_string<_CharT, _Traits, _Alloc>::basic_string(const _Alloc &__a)
        : _M_dataplus(_S_construct(size_type(), _CharT(), __a), __a) 
    {
    }
    template<typename _CharT, typename _Traits, typename _Alloc >
    basic_string<_CharT, _Traits, _Alloc>::basic_string(const basic_string &__str, size_type __pos, size_type __n)
        : _M_dataplus(_S_construct(__str._M_data() + __str._M_check(__pos, "basic_string::basic_string"), __str._M_data() + __str._M_limit(__pos, __n) + __pos, _Alloc()), _Alloc()) 
    {
    }
    template<typename _CharT, typename _Traits, typename _Alloc >
    basic_string<_CharT, _Traits, _Alloc>::basic_string(const basic_string &__str, size_type __pos, size_type __n, const _Alloc &__a)
        : _M_dataplus(_S_construct(__str._M_data() + __str._M_check(__pos, "basic_string::basic_string"), __str._M_data() + __str._M_limit(__pos, __n) + __pos, __a), __a) 
    {
    }
    template<typename _CharT, typename _Traits, typename _Alloc >
    basic_string<_CharT, _Traits, _Alloc>::basic_string(const _CharT *__s, size_type __n, const _Alloc &__a)
        : _M_dataplus(_S_construct(__s, __s + __n, __a), __a) 
    {
    }
    template<typename _CharT, typename _Traits, typename _Alloc >
    basic_string<_CharT, _Traits, _Alloc>::basic_string(const _CharT *__s, const _Alloc &__a)
        : _M_dataplus(_S_construct(__s, __s ? __s + traits_type::length(__s) : __s + npos, __a), __a) 
    {
    }
    template<typename _CharT, typename _Traits, typename _Alloc >
    basic_string<_CharT, _Traits, _Alloc>::basic_string(size_type __n, _CharT __c, const _Alloc &__a)
        : _M_dataplus(_S_construct(__n, __c, __a), __a) 
    {
    }
    template<typename _CharT, typename _Traits, typename _Alloc >
    template<typename _InputIterator >
    basic_string<_CharT, _Traits, _Alloc>::basic_string(_InputIterator __beg, _InputIterator __end, const _Alloc &__a)
        : _M_dataplus(_S_construct(__beg, __end, __a), __a) 
    {
    }
    template<typename _CharT, typename _Traits, typename _Alloc >
    basic_string<_CharT, _Traits, _Alloc> &basic_string<_CharT, _Traits, _Alloc>::assign(const basic_string &__str)
    {
        if (_M_rep() != __str._M_rep())
        {
            const allocator_type __a = this->get_allocator();
            _CharT *__tmp = __str._M_rep()->_M_grab(__a, __str.get_allocator());
            _M_rep()->_M_dispose(__a);
            _M_data(__tmp);
        }
        return *this;
    }
    template<typename _CharT, typename _Traits, typename _Alloc >
    basic_string<_CharT, _Traits, _Alloc> &basic_string<_CharT, _Traits, _Alloc>::assign(const _CharT *__s, size_type __n)
    {
        ;
        _M_check_length(this->size(), __n, "basic_string::assign");
        if (_M_disjunct(__s) || _M_rep()->_M_is_shared())
            return _M_replace_safe(size_type(0), this->size(), __s, __n);
        else
        {
            const size_type __pos = __s - _M_data();
            if (__pos >= __n)
                _M_copy(_M_data(), __s, __n);
            else
                if (__pos)
                    _M_move(_M_data(), __s, __n);
            _M_rep()->_M_set_length_and_sharable(__n);
            return *this;
        }
    }
    template<typename _CharT, typename _Traits, typename _Alloc >
    basic_string<_CharT, _Traits, _Alloc> &basic_string<_CharT, _Traits, _Alloc>::append(size_type __n, _CharT __c)
    {
        if (__n)
        {
            _M_check_length(size_type(0), __n, "basic_string::append");
            const size_type __len = __n + this->size();
            if (__len > this->capacity() || _M_rep()->_M_is_shared())
                this->reserve(__len);
            _M_assign(_M_data() + this->size(), __n, __c);
            _M_rep()->_M_set_length_and_sharable(__len);
        }
        return *this;
    }
    template<typename _CharT, typename _Traits, typename _Alloc >
    basic_string<_CharT, _Traits, _Alloc> &basic_string<_CharT, _Traits, _Alloc>::append(const _CharT *__s, size_type __n)
    {
        ;
        if (__n)
        {
            _M_check_length(size_type(0), __n, "basic_string::append");
            const size_type __len = __n + this->size();
            if (__len > this->capacity() || _M_rep()->_M_is_shared())
            {
                if (_M_disjunct(__s))
                    this->reserve(__len);
                else
                {
                    const size_type __off = __s - _M_data();
                    this->reserve(__len);
                    __s = _M_data() + __off;
                }
            }
            _M_copy(_M_data() + this->size(), __s, __n);
            _M_rep()->_M_set_length_and_sharable(__len);
        }
        return *this;
    }
    template<typename _CharT, typename _Traits, typename _Alloc >
    basic_string<_CharT, _Traits, _Alloc> &basic_string<_CharT, _Traits, _Alloc>::append(const basic_string &__str)
    {
        const size_type __size = __str.size();
        if (__size)
        {
            const size_type __len = __size + this->size();
            if (__len > this->capacity() || _M_rep()->_M_is_shared())
                this->reserve(__len);
            _M_copy(_M_data() + this->size(), __str._M_data(), __size);
            _M_rep()->_M_set_length_and_sharable(__len);
        }
        return *this;
    }
    template<typename _CharT, typename _Traits, typename _Alloc >
    basic_string<_CharT, _Traits, _Alloc> &basic_string<_CharT, _Traits, _Alloc>::append(const basic_string &__str, size_type __pos, size_type __n)
    {
        __str._M_check(__pos, "basic_string::append");
        __n = __str._M_limit(__pos, __n);
        if (__n)
        {
            const size_type __len = __n + this->size();
            if (__len > this->capacity() || _M_rep()->_M_is_shared())
                this->reserve(__len);
            _M_copy(_M_data() + this->size(), __str._M_data() + __pos, __n);
            _M_rep()->_M_set_length_and_sharable(__len);
        }
        return *this;
    }
    template<typename _CharT, typename _Traits, typename _Alloc >
    basic_string<_CharT, _Traits, _Alloc> &basic_string<_CharT, _Traits, _Alloc>::insert(size_type __pos, const _CharT *__s, size_type __n)
    {
        ;
        _M_check(__pos, "basic_string::insert");
        _M_check_length(size_type(0), __n, "basic_string::insert");
        if (_M_disjunct(__s) || _M_rep()->_M_is_shared())
            return _M_replace_safe(__pos, size_type(0), __s, __n);
        else
        {
            const size_type __off = __s - _M_data();
            _M_mutate(__pos, 0, __n);
            __s = _M_data() + __off;
            _CharT *__p = _M_data() + __pos;
            if (__s + __n <= __p)
                _M_copy(__p, __s, __n);
            else
                if (__s >= __p)
                    _M_copy(__p, __s + __n, __n);
                else
                {
                    const size_type __nleft = __p - __s;
                    _M_copy(__p, __s, __nleft);
                    _M_copy(__p + __nleft, __p + __n, __n - __nleft);
                }
            return *this;
        }
    }
    template<typename _CharT, typename _Traits, typename _Alloc >
    typename basic_string<_CharT, _Traits, _Alloc>::iterator basic_string<_CharT, _Traits, _Alloc>::erase(iterator __first, iterator __last)
    {
        ;
        const size_type __size = __last - __first;
        if (__size)
        {
            const size_type __pos = __first - _M_ibegin();
            _M_mutate(__pos, __size, size_type(0));
            _M_rep()->_M_set_leaked();
            return iterator(_M_data() + __pos);
        }
        else
            return __first;
    }
    template<typename _CharT, typename _Traits, typename _Alloc >
    basic_string<_CharT, _Traits, _Alloc> &basic_string<_CharT, _Traits, _Alloc>::replace(size_type __pos, size_type __n1, const _CharT *__s, size_type __n2)
    {
        ;
        _M_check(__pos, "basic_string::replace");
        __n1 = _M_limit(__pos, __n1);
        _M_check_length(__n1, __n2, "basic_string::replace");
        bool __left;
        if (_M_disjunct(__s) || _M_rep()->_M_is_shared())
            return _M_replace_safe(__pos, __n1, __s, __n2);
        else
            if ((__left = __s + __n2 <= _M_data() + __pos) || _M_data() + __pos + __n1 <= __s)
            {
                size_type __off = __s - _M_data();
                __left ? __off : (__off += __n2 - __n1);
                _M_mutate(__pos, __n1, __n2);
                _M_copy(_M_data() + __pos, _M_data() + __off, __n2);
                return *this;
            }
            else
            {
                const basic_string __tmp(__s, __n2);
                return _M_replace_safe(__pos, __n1, __tmp._M_data(), __n2);
            }
    }
    template<typename _CharT, typename _Traits, typename _Alloc >
    void basic_string<_CharT, _Traits, _Alloc>::_Rep::_M_destroy(const _Alloc &__a) throw ()
    {
        const size_type __size = sizeof(_Rep_base) + (this->_M_capacity + 1) * sizeof(_CharT);
        _Raw_bytes_alloc(__a).deallocate(reinterpret_cast<char * >(this), __size);
    }
    template<typename _CharT, typename _Traits, typename _Alloc >
    void basic_string<_CharT, _Traits, _Alloc>::_M_leak_hard()
    {
        if (_M_rep() == &_S_empty_rep())
            return;
        if (_M_rep()->_M_is_shared())
            _M_mutate(0, 0, 0);
        _M_rep()->_M_set_leaked();
    }
    template<typename _CharT, typename _Traits, typename _Alloc >
    void basic_string<_CharT, _Traits, _Alloc>::_M_mutate(size_type __pos, size_type __len1, size_type __len2)
    {
        const size_type __old_size = this->size();
        const size_type __new_size = __old_size + __len2 - __len1;
        const size_type __how_much = __old_size - __pos - __len1;
        if (__new_size > this->capacity() || _M_rep()->_M_is_shared())
        {
            const allocator_type __a = get_allocator();
            _Rep *__r = _Rep::_S_create(__new_size, this->capacity(), __a);
            if (__pos)
                _M_copy(__r->_M_refdata(), _M_data(), __pos);
            if (__how_much)
                _M_copy(__r->_M_refdata() + __pos + __len2, _M_data() + __pos + __len1, __how_much);
            _M_rep()->_M_dispose(__a);
            _M_data(__r->_M_refdata());
        }
        else
            if (__how_much && __len1 != __len2)
            {
                _M_move(_M_data() + __pos + __len2, _M_data() + __pos + __len1, __how_much);
            }
        _M_rep()->_M_set_length_and_sharable(__new_size);
    }
    template<typename _CharT, typename _Traits, typename _Alloc >
    void basic_string<_CharT, _Traits, _Alloc>::reserve(size_type __res)
    {
        if (__res != this->capacity() || _M_rep()->_M_is_shared())
        {
            if (__res < this->size())
                __res = this->size();
            const allocator_type __a = get_allocator();
            _CharT *__tmp = _M_rep()->_M_clone(__a, __res - this->size());
            _M_rep()->_M_dispose(__a);
            _M_data(__tmp);
        }
    }
    template<typename _CharT, typename _Traits, typename _Alloc >
    void basic_string<_CharT, _Traits, _Alloc>::swap(basic_string &__s)
    {
        if (_M_rep()->_M_is_leaked())
            _M_rep()->_M_set_sharable();
        if (__s._M_rep()->_M_is_leaked())
            __s._M_rep()->_M_set_sharable();
        if (this->get_allocator() == __s.get_allocator())
        {
            _CharT *__tmp = _M_data();
            _M_data(__s._M_data());
            __s._M_data(__tmp);
        }
        else
        {
            const basic_string __tmp1(_M_ibegin(), _M_iend(), __s.get_allocator());
            const basic_string __tmp2(__s._M_ibegin(), __s._M_iend(), this->get_allocator());
            *this = __tmp2;
            __s = __tmp1;
        }
    }
    template<typename _CharT, typename _Traits, typename _Alloc >
    typename basic_string<_CharT, _Traits, _Alloc>::_Rep *basic_string<_CharT, _Traits, _Alloc>::_Rep::_S_create(size_type __capacity, size_type __old_capacity, const _Alloc &__alloc)
    {
        if (__capacity > _S_max_size)
            __throw_length_error(("basic_string::_S_create"));
        const size_type __pagesize = 4096;
        const size_type __malloc_header_size = 4 * sizeof(void *);
        if (__capacity > __old_capacity && __capacity < 2 * __old_capacity)
            __capacity = 2 * __old_capacity;
        size_type __size = (__capacity + 1) * sizeof(_CharT) + sizeof(_Rep);
        const size_type __adj_size = __size + __malloc_header_size;
        if (__adj_size > __pagesize && __capacity > __old_capacity)
        {
            const size_type __extra = __pagesize - __adj_size % __pagesize;
            __capacity += __extra / sizeof(_CharT);
            if (__capacity > _S_max_size)
                __capacity = _S_max_size;
            __size = (__capacity + 1) * sizeof(_CharT) + sizeof(_Rep);
        }
        void *__place = _Raw_bytes_alloc(__alloc).allocate(__size);
        _Rep *__p = new (__place) _Rep;
        __p->_M_capacity = __capacity;
        __p->_M_set_sharable();
        return __p;
    }
    template<typename _CharT, typename _Traits, typename _Alloc >
    _CharT *basic_string<_CharT, _Traits, _Alloc>::_Rep::_M_clone(const _Alloc &__alloc, size_type __res)
    {
        const size_type __requested_cap = this->_M_length + __res;
        _Rep *__r = _Rep::_S_create(__requested_cap, this->_M_capacity, __alloc);
        if (this->_M_length)
            _M_copy(__r->_M_refdata(), _M_refdata(), this->_M_length);
        __r->_M_set_length_and_sharable(this->_M_length);
        return __r->_M_refdata();
    }
    template<typename _CharT, typename _Traits, typename _Alloc >
    void basic_string<_CharT, _Traits, _Alloc>::resize(size_type __n, _CharT __c)
    {
        const size_type __size = this->size();
        _M_check_length(__size, __n, "basic_string::resize");
        if (__size < __n)
            this->append(__n - __size, __c);
        else
            if (__n < __size)
                this->erase(__n);
    }
    template<typename _CharT, typename _Traits, typename _Alloc >
    template<typename _InputIterator >
    basic_string<_CharT, _Traits, _Alloc> &basic_string<_CharT, _Traits, _Alloc>::_M_replace_dispatch(iterator __i1, iterator __i2, _InputIterator __k1, _InputIterator __k2, __false_type)
    {
        const basic_string __s(__k1, __k2);
        const size_type __n1 = __i2 - __i1;
        _M_check_length(__n1, __s.size(), "basic_string::_M_replace_dispatch");
        return _M_replace_safe(__i1 - _M_ibegin(), __n1, __s._M_data(), __s.size());
    }
    template<typename _CharT, typename _Traits, typename _Alloc >
    basic_string<_CharT, _Traits, _Alloc> &basic_string<_CharT, _Traits, _Alloc>::_M_replace_aux(size_type __pos1, size_type __n1, size_type __n2, _CharT __c)
    {
        _M_check_length(__n1, __n2, "basic_string::_M_replace_aux");
        _M_mutate(__pos1, __n1, __n2);
        if (__n2)
            _M_assign(_M_data() + __pos1, __n2, __c);
        return *this;
    }
    template<typename _CharT, typename _Traits, typename _Alloc >
    basic_string<_CharT, _Traits, _Alloc> &basic_string<_CharT, _Traits, _Alloc>::_M_replace_safe(size_type __pos1, size_type __n1, const _CharT *__s, size_type __n2)
    {
        _M_mutate(__pos1, __n1, __n2);
        if (__n2)
            _M_copy(_M_data() + __pos1, __s, __n2);
        return *this;
    }
    template<typename _CharT, typename _Traits, typename _Alloc >
    basic_string<_CharT, _Traits, _Alloc> operator +(const _CharT *__lhs, const basic_string<_CharT, _Traits, _Alloc> &__rhs)
    {
        ;
        typedef basic_string<_CharT, _Traits, _Alloc> __string_type;
        typedef typename __string_type::size_type __size_type;
        const __size_type __len = _Traits::length(__lhs);
        __string_type __str;
        __str.reserve(__len + __rhs.size());
        __str.append(__lhs, __len);
        __str.append(__rhs);
        return __str;
    }
    template<typename _CharT, typename _Traits, typename _Alloc >
    basic_string<_CharT, _Traits, _Alloc> operator +(_CharT __lhs, const basic_string<_CharT, _Traits, _Alloc> &__rhs)
    {
        typedef basic_string<_CharT, _Traits, _Alloc> __string_type;
        typedef typename __string_type::size_type __size_type;
        __string_type __str;
        const __size_type __len = __rhs.size();
        __str.reserve(__len + 1);
        __str.append(__size_type(1), __lhs);
        __str.append(__rhs);
        return __str;
    }
    template<typename _CharT, typename _Traits, typename _Alloc >
    typename basic_string<_CharT, _Traits, _Alloc>::size_type basic_string<_CharT, _Traits, _Alloc>::copy(_CharT *__s, size_type __n, size_type __pos) const
    {
        _M_check(__pos, "basic_string::copy");
        __n = _M_limit(__pos, __n);
        ;
        if (__n)
            _M_copy(__s, _M_data() + __pos, __n);
        return __n;
    }
    template<typename _CharT, typename _Traits, typename _Alloc >
    typename basic_string<_CharT, _Traits, _Alloc>::size_type basic_string<_CharT, _Traits, _Alloc>::find(const _CharT *__s, size_type __pos, size_type __n) const
    {
        ;
        const size_type __size = this->size();
        const _CharT *__data = _M_data();
        if (__n == 0)
            return __pos <= __size ? __pos : npos;
        if (__n <= __size)
        {
            for (;
                __pos <= __size - __n;
                ++__pos)
                if (traits_type::eq(__data[__pos], __s[0]) && traits_type::compare(__data + __pos + 1, __s + 1, __n - 1) == 0)
                    return __pos;
        }
        return npos;
    }
    template<typename _CharT, typename _Traits, typename _Alloc >
    typename basic_string<_CharT, _Traits, _Alloc>::size_type basic_string<_CharT, _Traits, _Alloc>::find(_CharT __c, size_type __pos) const
    {
        size_type __ret = npos;
        const size_type __size = this->size();
        if (__pos < __size)
        {
            const _CharT *__data = _M_data();
            const size_type __n = __size - __pos;
            const _CharT *__p = traits_type::find(__data + __pos, __n, __c);
            if (__p)
                __ret = __p - __data;
        }
        return __ret;
    }
    template<typename _CharT, typename _Traits, typename _Alloc >
    typename basic_string<_CharT, _Traits, _Alloc>::size_type basic_string<_CharT, _Traits, _Alloc>::rfind(const _CharT *__s, size_type __pos, size_type __n) const
    {
        ;
        const size_type __size = this->size();
        if (__n <= __size)
        {
            __pos = std::min(size_type(__size - __n), __pos);
            const _CharT *__data = _M_data();
            do
            {
                if (traits_type::compare(__data + __pos, __s, __n) == 0)
                    return __pos;
            }
            while (__pos-- > 0);
        }
        return npos;
    }
    template<typename _CharT, typename _Traits, typename _Alloc >
    typename basic_string<_CharT, _Traits, _Alloc>::size_type basic_string<_CharT, _Traits, _Alloc>::rfind(_CharT __c, size_type __pos) const
    {
        size_type __size = this->size();
        if (__size)
        {
            if (--__size > __pos)
                __size = __pos;
            for (++__size;
                __size-- > 0;
                )
                if (traits_type::eq(_M_data()[__size], __c))
                    return __size;
        }
        return npos;
    }
    template<typename _CharT, typename _Traits, typename _Alloc >
    typename basic_string<_CharT, _Traits, _Alloc>::size_type basic_string<_CharT, _Traits, _Alloc>::find_first_of(const _CharT *__s, size_type __pos, size_type __n) const
    {
        ;
        for (;
            __n && __pos < this->size();
            ++__pos)
        {
            const _CharT *__p = traits_type::find(__s, __n, _M_data()[__pos]);
            if (__p)
                return __pos;
        }
        return npos;
    }
    template<typename _CharT, typename _Traits, typename _Alloc >
    typename basic_string<_CharT, _Traits, _Alloc>::size_type basic_string<_CharT, _Traits, _Alloc>::find_last_of(const _CharT *__s, size_type __pos, size_type __n) const
    {
        ;
        size_type __size = this->size();
        if (__size && __n)
        {
            if (--__size > __pos)
                __size = __pos;
            do
            {
                if (traits_type::find(__s, __n, _M_data()[__size]))
                    return __size;
            }
            while (__size-- != 0);
        }
        return npos;
    }
    template<typename _CharT, typename _Traits, typename _Alloc >
    typename basic_string<_CharT, _Traits, _Alloc>::size_type basic_string<_CharT, _Traits, _Alloc>::find_first_not_of(const _CharT *__s, size_type __pos, size_type __n) const
    {
        ;
        for (;
            __pos < this->size();
            ++__pos)
            if (!traits_type::find(__s, __n, _M_data()[__pos]))
                return __pos;
        return npos;
    }
    template<typename _CharT, typename _Traits, typename _Alloc >
    typename basic_string<_CharT, _Traits, _Alloc>::size_type basic_string<_CharT, _Traits, _Alloc>::find_first_not_of(_CharT __c, size_type __pos) const
    {
        for (;
            __pos < this->size();
            ++__pos)
            if (!traits_type::eq(_M_data()[__pos], __c))
                return __pos;
        return npos;
    }
    template<typename _CharT, typename _Traits, typename _Alloc >
    typename basic_string<_CharT, _Traits, _Alloc>::size_type basic_string<_CharT, _Traits, _Alloc>::find_last_not_of(const _CharT *__s, size_type __pos, size_type __n) const
    {
        ;
        size_type __size = this->size();
        if (__size)
        {
            if (--__size > __pos)
                __size = __pos;
            do
            {
                if (!traits_type::find(__s, __n, _M_data()[__size]))
                    return __size;
            }
            while (__size--);
        }
        return npos;
    }
    template<typename _CharT, typename _Traits, typename _Alloc >
    typename basic_string<_CharT, _Traits, _Alloc>::size_type basic_string<_CharT, _Traits, _Alloc>::find_last_not_of(_CharT __c, size_type __pos) const
    {
        size_type __size = this->size();
        if (__size)
        {
            if (--__size > __pos)
                __size = __pos;
            do
            {
                if (!traits_type::eq(_M_data()[__size], __c))
                    return __size;
            }
            while (__size--);
        }
        return npos;
    }
    template<typename _CharT, typename _Traits, typename _Alloc >
    int basic_string<_CharT, _Traits, _Alloc>::compare(size_type __pos, size_type __n, const basic_string &__str) const
    {
        _M_check(__pos, "basic_string::compare");
        __n = _M_limit(__pos, __n);
        const size_type __osize = __str.size();
        const size_type __len = std::min(__n, __osize);
        int __r = traits_type::compare(_M_data() + __pos, __str.data(), __len);
        if (!__r)
            __r = _S_compare(__n, __osize);
        return __r;
    }
    template<typename _CharT, typename _Traits, typename _Alloc >
    int basic_string<_CharT, _Traits, _Alloc>::compare(size_type __pos1, size_type __n1, const basic_string &__str, size_type __pos2, size_type __n2) const
    {
        _M_check(__pos1, "basic_string::compare");
        __str._M_check(__pos2, "basic_string::compare");
        __n1 = _M_limit(__pos1, __n1);
        __n2 = __str._M_limit(__pos2, __n2);
        const size_type __len = std::min(__n1, __n2);
        int __r = traits_type::compare(_M_data() + __pos1, __str.data() + __pos2, __len);
        if (!__r)
            __r = _S_compare(__n1, __n2);
        return __r;
    }
    template<typename _CharT, typename _Traits, typename _Alloc >
    int basic_string<_CharT, _Traits, _Alloc>::compare(const _CharT *__s) const
    {
        ;
        const size_type __size = this->size();
        const size_type __osize = traits_type::length(__s);
        const size_type __len = std::min(__size, __osize);
        int __r = traits_type::compare(_M_data(), __s, __len);
        if (!__r)
            __r = _S_compare(__size, __osize);
        return __r;
    }
    template<typename _CharT, typename _Traits, typename _Alloc >
    int basic_string<_CharT, _Traits, _Alloc>::compare(size_type __pos, size_type __n1, const _CharT *__s) const
    {
        ;
        _M_check(__pos, "basic_string::compare");
        __n1 = _M_limit(__pos, __n1);
        const size_type __osize = traits_type::length(__s);
        const size_type __len = std::min(__n1, __osize);
        int __r = traits_type::compare(_M_data() + __pos, __s, __len);
        if (!__r)
            __r = _S_compare(__n1, __osize);
        return __r;
    }
    template<typename _CharT, typename _Traits, typename _Alloc >
    int basic_string<_CharT, _Traits, _Alloc>::compare(size_type __pos, size_type __n1, const _CharT *__s, size_type __n2) const
    {
        ;
        _M_check(__pos, "basic_string::compare");
        __n1 = _M_limit(__pos, __n1);
        const size_type __len = std::min(__n1, __n2);
        int __r = traits_type::compare(_M_data() + __pos, __s, __len);
        if (!__r)
            __r = _S_compare(__n1, __n2);
        return __r;
    }
    template<typename _CharT, typename _Traits, typename _Alloc >
    basic_istream<_CharT, _Traits> &operator >>(basic_istream<_CharT, _Traits> &__in, basic_string<_CharT, _Traits, _Alloc> &__str)
    {
        typedef basic_istream<_CharT, _Traits> __istream_type;
        typedef basic_string<_CharT, _Traits, _Alloc> __string_type;
        typedef typename __istream_type::ios_base __ios_base;
        typedef typename __istream_type::int_type __int_type;
        typedef typename __string_type::size_type __size_type;
        typedef ctype<_CharT> __ctype_type;
        typedef typename __ctype_type::ctype_base __ctype_base;
        __size_type __extracted = 0;
        typename __ios_base::iostate __err = __ios_base::goodbit;
        typename __istream_type::sentry __cerb(__in, false);
        if (__cerb)
        {
            try
            {
                __str.erase();
                _CharT __buf[128];
                __size_type __len = 0;
                const streamsize __w = __in.width();
                const __size_type __n = __w > 0 ? static_cast<__size_type >(__w) : __str.max_size();
                const __ctype_type &__ct = use_facet<__ctype_type>(__in.getloc());
                const __int_type __eof = _Traits::eof();
                __int_type __c = __in.rdbuf()->sgetc();
                while (__extracted < __n && !_Traits::eq_int_type(__c, __eof) && !__ct.is(__ctype_base::space, _Traits::to_char_type(__c)))
                {
                    if (__len == sizeof (__buf) / sizeof(_CharT))
                    {
                        __str.append(__buf, sizeof (__buf) / sizeof(_CharT));
                        __len = 0;
                    }
                    __buf[__len++] = _Traits::to_char_type(__c);
                    ++__extracted;
                    __c = __in.rdbuf()->snextc();
                }
                __str.append(__buf, __len);
                if (_Traits::eq_int_type(__c, __eof))
                    __err |= __ios_base::eofbit;
                __in.width(0);
            }
            catch (__cxxabiv1::__forced_unwind &)
            {
                __in._M_setstate(__ios_base::badbit);
                throw;
            }
            catch (...)
            {
                __in._M_setstate(__ios_base::badbit);
            }
        }
        if (!__extracted)
            __err |= __ios_base::failbit;
        if (__err)
            __in.setstate(__err);
        return __in;
    }
    template<typename _CharT, typename _Traits, typename _Alloc >
    basic_istream<_CharT, _Traits> &getline(basic_istream<_CharT, _Traits> &__in, basic_string<_CharT, _Traits, _Alloc> &__str, _CharT __delim)
    {
        typedef basic_istream<_CharT, _Traits> __istream_type;
        typedef basic_string<_CharT, _Traits, _Alloc> __string_type;
        typedef typename __istream_type::ios_base __ios_base;
        typedef typename __istream_type::int_type __int_type;
        typedef typename __string_type::size_type __size_type;
        __size_type __extracted = 0;
        const __size_type __n = __str.max_size();
        typename __ios_base::iostate __err = __ios_base::goodbit;
        typename __istream_type::sentry __cerb(__in, true);
        if (__cerb)
        {
            try
            {
                __str.erase();
                const __int_type __idelim = _Traits::to_int_type(__delim);
                const __int_type __eof = _Traits::eof();
                __int_type __c = __in.rdbuf()->sgetc();
                while (__extracted < __n && !_Traits::eq_int_type(__c, __eof) && !_Traits::eq_int_type(__c, __idelim))
                {
                    __str += _Traits::to_char_type(__c);
                    ++__extracted;
                    __c = __in.rdbuf()->snextc();
                }
                if (_Traits::eq_int_type(__c, __eof))
                    __err |= __ios_base::eofbit;
                else
                    if (_Traits::eq_int_type(__c, __idelim))
                    {
                        ++__extracted;
                        __in.rdbuf()->sbumpc();
                    }
                    else
                        __err |= __ios_base::failbit;
            }
            catch (__cxxabiv1::__forced_unwind &)
            {
                __in._M_setstate(__ios_base::badbit);
                throw;
            }
            catch (...)
            {
                __in._M_setstate(__ios_base::badbit);
            }
        }
        if (!__extracted)
            __err |= __ios_base::failbit;
        if (__err)
            __in.setstate(__err);
        return __in;
    }
    extern template class basic_string<char>;
    extern template basic_istream<char> &operator >>(basic_istream<char> &, string &);
    extern template basic_ostream<char> &operator <<(basic_ostream<char> &, const string &);
    extern template basic_istream<char> &getline(basic_istream<char> &, string &, char);
    extern template basic_istream<char> &getline(basic_istream<char> &, string &);
    extern template class basic_string<wchar_t>;
    extern template basic_istream<wchar_t> &operator >>(basic_istream<wchar_t> &, wstring &);
    extern template basic_ostream<wchar_t> &operator <<(basic_ostream<wchar_t> &, const wstring &);
    extern template basic_istream<wchar_t> &getline(basic_istream<wchar_t> &, wstring &, wchar_t);
    extern template basic_istream<wchar_t> &getline(basic_istream<wchar_t> &, wstring &);
}
namespace std __attribute__((__visibility__("default"))) {
    class locale
    {
        public :
            typedef int category;
            class facet;
            class id;
            class _Impl;
            friend class facet;
            friend class _Impl;
            template<typename _Facet >
            friend bool has_facet(const locale &) throw ();
            template<typename _Facet >
            friend const _Facet &use_facet(const locale &);
            template<typename _Cache >
            friend struct __use_cache;
            static const category none  = 0;
            static const category ctype  = 1L << 0;
            static const category numeric  = 1L << 1;
            static const category collate  = 1L << 2;
            static const category time  = 1L << 3;
            static const category monetary  = 1L << 4;
            static const category messages  = 1L << 5;
            static const category all  = (ctype | numeric | collate | time | monetary | messages);
            locale() throw ();
            locale(const locale &__other) throw ();
            explicit locale(const char *__s);
            locale(const locale &__base, const char *__s, category __cat);
            locale(const locale &__base, const locale &__add, category __cat);
            template<typename _Facet >
            locale(const locale &__other, _Facet *__f);
            ~locale() throw ();
            const locale &operator =(const locale &__other) throw ();
            template<typename _Facet >
            locale combine(const locale &__other) const;
            string name() const;
            bool operator ==(const locale &__other) const throw ();
            bool operator !=(const locale &__other) const throw ()
            {
                return !(this->operator ==(__other));
            }
            template<typename _Char, typename _Traits, typename _Alloc >
            bool operator ()(const basic_string<_Char, _Traits, _Alloc> &__s1, const basic_string<_Char, _Traits, _Alloc> &__s2) const;
            static locale global(const locale &);
            static const locale &classic();
        private :
            _Impl *_M_impl;
            static _Impl *_S_classic;
            static _Impl *_S_global;
            static const char *const *const _S_categories;
            enum 
            {
                _S_categories_size = 6 + 6
            };
            static __gthread_once_t _S_once;
            explicit locale(_Impl *) throw ();
            static void _S_initialize();
            static void _S_initialize_once();
            static category _S_normalize_category(category);
            void _M_coalesce(const locale &__base, const locale &__add, category __cat);
    };
    class locale::facet
    {
        private :
            friend class locale;
            friend class locale::_Impl;
            mutable _Atomic_word _M_refcount;
            static __c_locale _S_c_locale;
            static const char _S_c_name[2];
            static __gthread_once_t _S_once;
            static void _S_initialize_once();
        protected :
            explicit facet(size_t __refs = 0) throw ()
                : _M_refcount(__refs ? 1 : 0) 
            {
            }
            virtual ~facet();
            static void _S_create_c_locale(__c_locale &__cloc, const char *__s, __c_locale __old = 0);
            static __c_locale _S_clone_c_locale(__c_locale &__cloc);
            static void _S_destroy_c_locale(__c_locale &__cloc);
            static __c_locale _S_get_c_locale();
            static const char *_S_get_c_name();
        private :
            void _M_add_reference() const throw ()
            {
                __gnu_cxx::__atomic_add_dispatch(&_M_refcount, 1);
            }
            void _M_remove_reference() const throw ()
            {
                if (__gnu_cxx::__exchange_and_add_dispatch(&_M_refcount, - 1) == 1)
                {
                    try
                    {
                        delete this;
                    }
                    catch (...)
                    {
                    }
                }
            }
            facet(const facet &);
            facet &operator =(const facet &);
    };
    class locale::id
    {
        private :
            friend class locale;
            friend class locale::_Impl;
            template<typename _Facet >
            friend const _Facet &use_facet(const locale &);
            template<typename _Facet >
            friend bool has_facet(const locale &) throw ();
            mutable size_t _M_index;
            static _Atomic_word _S_refcount;
            void operator =(const id &);
            id(const id &);
        public :
            id()
            {
            }
            size_t _M_id() const;
    };
    class locale::_Impl
    {
        public :
            friend class locale;
            friend class locale::facet;
            template<typename _Facet >
            friend bool has_facet(const locale &) throw ();
            template<typename _Facet >
            friend const _Facet &use_facet(const locale &);
            template<typename _Cache >
            friend struct __use_cache;
        private :
            _Atomic_word _M_refcount;
            const facet **_M_facets;
            size_t _M_facets_size;
            const facet **_M_caches;
            char **_M_names;
            static const locale::id *const _S_id_ctype[];
            static const locale::id *const _S_id_numeric[];
            static const locale::id *const _S_id_collate[];
            static const locale::id *const _S_id_time[];
            static const locale::id *const _S_id_monetary[];
            static const locale::id *const _S_id_messages[];
            static const locale::id *const *const _S_facet_categories[];
            void _M_add_reference() throw ()
            {
                __gnu_cxx::__atomic_add_dispatch(&_M_refcount, 1);
            }
            void _M_remove_reference() throw ()
            {
                if (__gnu_cxx::__exchange_and_add_dispatch(&_M_refcount, - 1) == 1)
                {
                    try
                    {
                        delete this;
                    }
                    catch (...)
                    {
                    }
                }
            }
            _Impl(const _Impl &, size_t);
            _Impl(const char *, size_t);
            _Impl(size_t) throw ();
            ~_Impl() throw ();
            _Impl(const _Impl &);
            void operator =(const _Impl &);
            bool _M_check_same_name()
            {
                bool __ret = true;
                if (_M_names[1])
                    for (size_t __i = 0;
                        __ret && __i < _S_categories_size - 1;
                        ++__i)
                        __ret = __builtin_strcmp(_M_names[__i], _M_names[__i + 1]) == 0;
                return __ret;
            }
            void _M_replace_categories(const _Impl *, category);
            void _M_replace_category(const _Impl *, const locale::id *const *);
            void _M_replace_facet(const _Impl *, const locale::id *);
            void _M_install_facet(const locale::id *, const facet *);
            template<typename _Facet >
            void _M_init_facet(_Facet *__facet)
            {
                _M_install_facet(&_Facet::id, __facet);
            }
            void _M_install_cache(const facet *, size_t);
    };
    template<typename _Facet >
    bool has_facet(const locale &__loc) throw ();
    template<typename _Facet >
    const _Facet &use_facet(const locale &__loc);
    template<typename _CharT >
    class collate : public locale::facet
    {
        public :
            typedef _CharT char_type;
            typedef basic_string<_CharT> string_type;
        protected :
            __c_locale _M_c_locale_collate;
        public :
            static locale::id id;
            explicit collate(size_t __refs = 0)
                : facet(__refs), _M_c_locale_collate(_S_get_c_locale()) 
            {
            }
            explicit collate(__c_locale __cloc, size_t __refs = 0)
                : facet(__refs), _M_c_locale_collate(_S_clone_c_locale(__cloc)) 
            {
            }
            int compare(const _CharT *__lo1, const _CharT *__hi1, const _CharT *__lo2, const _CharT *__hi2) const
            {
                return this->do_compare(__lo1, __hi1, __lo2, __hi2);
            }
            string_type transform(const _CharT *__lo, const _CharT *__hi) const
            {
                return this->do_transform(__lo, __hi);
            }
            long hash(const _CharT *__lo, const _CharT *__hi) const
            {
                return this->do_hash(__lo, __hi);
            }
            int _M_compare(const _CharT *, const _CharT *) const;
            size_t _M_transform(_CharT *, const _CharT *, size_t) const;
        protected :
            virtual ~collate()
            {
                _S_destroy_c_locale(_M_c_locale_collate);
            }
            virtual int do_compare(const _CharT *__lo1, const _CharT *__hi1, const _CharT *__lo2, const _CharT *__hi2) const;
            virtual string_type do_transform(const _CharT *__lo, const _CharT *__hi) const;
            virtual long do_hash(const _CharT *__lo, const _CharT *__hi) const;
    };
    template<typename _CharT >
    locale::id collate<_CharT>::id;
    template<>
    int collate<char>::_M_compare(const char *, const char *) const;
    template<>
    size_t collate<char>::_M_transform(char *, const char *, size_t) const;
    template<>
    int collate<wchar_t>::_M_compare(const wchar_t *, const wchar_t *) const;
    template<>
    size_t collate<wchar_t>::_M_transform(wchar_t *, const wchar_t *, size_t) const;
    template<typename _CharT >
    class collate_byname : public collate<_CharT>
    {
        public :
            typedef _CharT char_type;
            typedef basic_string<_CharT> string_type;
            explicit collate_byname(const char *__s, size_t __refs = 0)
                : collate<_CharT>(__refs) 
            {
                if (__builtin_strcmp(__s, "C") != 0 && __builtin_strcmp(__s, "POSIX") != 0)
                {
                    this->_S_destroy_c_locale(this->_M_c_locale_collate);
                    this->_S_create_c_locale(this->_M_c_locale_collate, __s);
                }
            }
        protected :
            virtual ~collate_byname()
            {
            }
    };
}
namespace std __attribute__((__visibility__("default"))) {
    template<typename _Facet >
    locale::locale(const locale &__other, _Facet *__f)
    {
        _M_impl = new _Impl (*__other._M_impl, 1);
        try
        {
            _M_impl->_M_install_facet(&_Facet::id, __f);
        }
        catch (...)
        {
            _M_impl->_M_remove_reference();
            throw;
        }
        delete[] _M_impl->_M_names[0];
        _M_impl->_M_names[0] = 0;
    }
    template<typename _Facet >
    locale locale::combine(const locale &__other) const
    {
        _Impl *__tmp = new _Impl (*_M_impl, 1);
        try
        {
            __tmp->_M_replace_facet(__other._M_impl, &_Facet::id);
        }
        catch (...)
        {
            __tmp->_M_remove_reference();
            throw;
        }
        return locale(__tmp);
    }
    template<typename _CharT, typename _Traits, typename _Alloc >
    bool locale::operator ()(const basic_string<_CharT, _Traits, _Alloc> &__s1, const basic_string<_CharT, _Traits, _Alloc> &__s2) const
    {
        typedef std::collate<_CharT> __collate_type;
        const __collate_type &__collate = use_facet<__collate_type>(*this);
        return (__collate.compare(__s1.data(), __s1.data() + __s1.length(), __s2.data(), __s2.data() + __s2.length()) < 0);
    }
    template<typename _Facet >
    bool has_facet(const locale &__loc) throw ()
    {
        const size_t __i = _Facet::id._M_id();
        const locale::facet **__facets = __loc._M_impl->_M_facets;
        return (__i < __loc._M_impl->_M_facets_size && dynamic_cast<const _Facet * >(__facets[__i]));
    }
    template<typename _Facet >
    const _Facet &use_facet(const locale &__loc)
    {
        const size_t __i = _Facet::id._M_id();
        const locale::facet **__facets = __loc._M_impl->_M_facets;
        if (__i >= __loc._M_impl->_M_facets_size || !__facets[__i])
            __throw_bad_cast();
        return dynamic_cast<const _Facet & >(*__facets[__i]);
    }
    template<typename _CharT >
    int collate<_CharT>::_M_compare(const _CharT *, const _CharT *) const
    {
        return 0;
    }
    template<typename _CharT >
    size_t collate<_CharT>::_M_transform(_CharT *, const _CharT *, size_t) const
    {
        return 0;
    }
    template<typename _CharT >
    int collate<_CharT>::do_compare(const _CharT *__lo1, const _CharT *__hi1, const _CharT *__lo2, const _CharT *__hi2) const
    {
        const string_type __one(__lo1, __hi1);
        const string_type __two(__lo2, __hi2);
        const _CharT *__p = __one.c_str();
        const _CharT *__pend = __one.data() + __one.length();
        const _CharT *__q = __two.c_str();
        const _CharT *__qend = __two.data() + __two.length();
        for (;
            ;
            )
        {
            const int __res = _M_compare(__p, __q);
            if (__res)
                return __res;
            __p += char_traits<_CharT>::length(__p);
            __q += char_traits<_CharT>::length(__q);
            if (__p == __pend && __q == __qend)
                return 0;
            else
                if (__p == __pend)
                    return - 1;
                else
                    if (__q == __qend)
                        return 1;
            __p++;
            __q++;
        }
    }
    template<typename _CharT >
    typename collate<_CharT>::string_type collate<_CharT>::do_transform(const _CharT *__lo, const _CharT *__hi) const
    {
        string_type __ret;
        const string_type __str(__lo, __hi);
        const _CharT *__p = __str.c_str();
        const _CharT *__pend = __str.data() + __str.length();
        size_t __len = (__hi - __lo) * 2;
        _CharT *__c = new _CharT [__len];
        try
        {
            for (;
                ;
                )
            {
                size_t __res = _M_transform(__c, __p, __len);
                if (__res >= __len)
                {
                    __len = __res + 1;
                    delete[] __c , __c = 0;
                    __c = new _CharT [__len];
                    __res = _M_transform(__c, __p, __len);
                }
                __ret.append(__c, __res);
                __p += char_traits<_CharT>::length(__p);
                if (__p == __pend)
                    break;
                __p++;
                __ret.push_back(_CharT());
            }
        }
        catch (...)
        {
            delete[] __c;
            throw;
        }
        delete[] __c;
        return __ret;
    }
    template<typename _CharT >
    long collate<_CharT>::do_hash(const _CharT *__lo, const _CharT *__hi) const
    {
        unsigned long __val = 0;
        for (;
            __lo < __hi;
            ++__lo)
            __val = *__lo + ((__val << 7) | (__val >> (__gnu_cxx::__numeric_traits<unsigned long>::__digits - 7)));
        return static_cast<long >(__val);
    }
    extern template class collate<char>;
    extern template class collate_byname<char>;
    extern template const collate<char> &use_facet<collate<char> >(const locale &);
    extern template bool has_facet<collate<char> >(const locale &);
    extern template class collate<wchar_t>;
    extern template class collate_byname<wchar_t>;
    extern template const collate<wchar_t> &use_facet<collate<wchar_t> >(const locale &);
    extern template bool has_facet<collate<wchar_t> >(const locale &);
}
namespace std __attribute__((__visibility__("default"))) {
    enum _Ios_Fmtflags
    {
        _S_boolalpha = 1L << 0, 
        _S_dec = 1L << 1, 
        _S_fixed = 1L << 2, 
        _S_hex = 1L << 3, 
        _S_internal = 1L << 4, 
        _S_left = 1L << 5, 
        _S_oct = 1L << 6, 
        _S_right = 1L << 7, 
        _S_scientific = 1L << 8, 
        _S_showbase = 1L << 9, 
        _S_showpoint = 1L << 10, 
        _S_showpos = 1L << 11, 
        _S_skipws = 1L << 12, 
        _S_unitbuf = 1L << 13, 
        _S_uppercase = 1L << 14, 
        _S_adjustfield = _S_left | _S_right | _S_internal, 
        _S_basefield = _S_dec | _S_oct | _S_hex, 
        _S_floatfield = _S_scientific | _S_fixed, 
        _S_ios_fmtflags_end = 1L << 16
    };
    inline _Ios_Fmtflags operator &(_Ios_Fmtflags __a, _Ios_Fmtflags __b)
    {
        return _Ios_Fmtflags(static_cast<int >(__a) & static_cast<int >(__b));
    }
    inline _Ios_Fmtflags operator |(_Ios_Fmtflags __a, _Ios_Fmtflags __b)
    {
        return _Ios_Fmtflags(static_cast<int >(__a) | static_cast<int >(__b));
    }
    inline _Ios_Fmtflags operator ^(_Ios_Fmtflags __a, _Ios_Fmtflags __b)
    {
        return _Ios_Fmtflags(static_cast<int >(__a) ^ static_cast<int >(__b));
    }
    inline _Ios_Fmtflags &operator |=(_Ios_Fmtflags &__a, _Ios_Fmtflags __b)
    {
        return __a = __a | __b;
    }
    inline _Ios_Fmtflags &operator &=(_Ios_Fmtflags &__a, _Ios_Fmtflags __b)
    {
        return __a = __a & __b;
    }
    inline _Ios_Fmtflags &operator ^=(_Ios_Fmtflags &__a, _Ios_Fmtflags __b)
    {
        return __a = __a ^ __b;
    }
    inline _Ios_Fmtflags operator ~(_Ios_Fmtflags __a)
    {
        return _Ios_Fmtflags(~static_cast<int >(__a));
    }
    enum _Ios_Openmode
    {
        _S_app = 1L << 0, 
        _S_ate = 1L << 1, 
        _S_bin = 1L << 2, 
        _S_in = 1L << 3, 
        _S_out = 1L << 4, 
        _S_trunc = 1L << 5, 
        _S_ios_openmode_end = 1L << 16
    };
    inline _Ios_Openmode operator &(_Ios_Openmode __a, _Ios_Openmode __b)
    {
        return _Ios_Openmode(static_cast<int >(__a) & static_cast<int >(__b));
    }
    inline _Ios_Openmode operator |(_Ios_Openmode __a, _Ios_Openmode __b)
    {
        return _Ios_Openmode(static_cast<int >(__a) | static_cast<int >(__b));
    }
    inline _Ios_Openmode operator ^(_Ios_Openmode __a, _Ios_Openmode __b)
    {
        return _Ios_Openmode(static_cast<int >(__a) ^ static_cast<int >(__b));
    }
    inline _Ios_Openmode &operator |=(_Ios_Openmode &__a, _Ios_Openmode __b)
    {
        return __a = __a | __b;
    }
    inline _Ios_Openmode &operator &=(_Ios_Openmode &__a, _Ios_Openmode __b)
    {
        return __a = __a & __b;
    }
    inline _Ios_Openmode &operator ^=(_Ios_Openmode &__a, _Ios_Openmode __b)
    {
        return __a = __a ^ __b;
    }
    inline _Ios_Openmode operator ~(_Ios_Openmode __a)
    {
        return _Ios_Openmode(~static_cast<int >(__a));
    }
    enum _Ios_Iostate
    {
        _S_goodbit = 0, 
        _S_badbit = 1L << 0, 
        _S_eofbit = 1L << 1, 
        _S_failbit = 1L << 2, 
        _S_ios_iostate_end = 1L << 16
    };
    inline _Ios_Iostate operator &(_Ios_Iostate __a, _Ios_Iostate __b)
    {
        return _Ios_Iostate(static_cast<int >(__a) & static_cast<int >(__b));
    }
    inline _Ios_Iostate operator |(_Ios_Iostate __a, _Ios_Iostate __b)
    {
        return _Ios_Iostate(static_cast<int >(__a) | static_cast<int >(__b));
    }
    inline _Ios_Iostate operator ^(_Ios_Iostate __a, _Ios_Iostate __b)
    {
        return _Ios_Iostate(static_cast<int >(__a) ^ static_cast<int >(__b));
    }
    inline _Ios_Iostate &operator |=(_Ios_Iostate &__a, _Ios_Iostate __b)
    {
        return __a = __a | __b;
    }
    inline _Ios_Iostate &operator &=(_Ios_Iostate &__a, _Ios_Iostate __b)
    {
        return __a = __a & __b;
    }
    inline _Ios_Iostate &operator ^=(_Ios_Iostate &__a, _Ios_Iostate __b)
    {
        return __a = __a ^ __b;
    }
    inline _Ios_Iostate operator ~(_Ios_Iostate __a)
    {
        return _Ios_Iostate(~static_cast<int >(__a));
    }
    enum _Ios_Seekdir
    {
        _S_beg = 0, 
        _S_cur = 1, 
        _S_end = 2, 
        _S_ios_seekdir_end = 1L << 16
    };
    class ios_base
    {
        public :
            class failure : public exception
            {
                public :
                    explicit failure(const string &__str) throw ();
                    virtual ~failure() throw ();
                    virtual const char *what() const throw ();
                private :
                    string _M_msg;
            };
            typedef _Ios_Fmtflags fmtflags;
            static const fmtflags boolalpha  = _S_boolalpha;
            static const fmtflags dec  = _S_dec;
            static const fmtflags fixed  = _S_fixed;
            static const fmtflags hex  = _S_hex;
            static const fmtflags internal  = _S_internal;
            static const fmtflags left  = _S_left;
            static const fmtflags oct  = _S_oct;
            static const fmtflags right  = _S_right;
            static const fmtflags scientific  = _S_scientific;
            static const fmtflags showbase  = _S_showbase;
            static const fmtflags showpoint  = _S_showpoint;
            static const fmtflags showpos  = _S_showpos;
            static const fmtflags skipws  = _S_skipws;
            static const fmtflags unitbuf  = _S_unitbuf;
            static const fmtflags uppercase  = _S_uppercase;
            static const fmtflags adjustfield  = _S_adjustfield;
            static const fmtflags basefield  = _S_basefield;
            static const fmtflags floatfield  = _S_floatfield;
            typedef _Ios_Iostate iostate;
            static const iostate badbit  = _S_badbit;
            static const iostate eofbit  = _S_eofbit;
            static const iostate failbit  = _S_failbit;
            static const iostate goodbit  = _S_goodbit;
            typedef _Ios_Openmode openmode;
            static const openmode app  = _S_app;
            static const openmode ate  = _S_ate;
            static const openmode binary  = _S_bin;
            static const openmode in  = _S_in;
            static const openmode out  = _S_out;
            static const openmode trunc  = _S_trunc;
            typedef _Ios_Seekdir seekdir;
            static const seekdir beg  = _S_beg;
            static const seekdir cur  = _S_cur;
            static const seekdir end  = _S_end;
            typedef int io_state;
            typedef int open_mode;
            typedef int seek_dir;
            typedef std::streampos streampos;
            typedef std::streamoff streamoff;
            enum event
            {
                erase_event, 
                imbue_event, 
                copyfmt_event
            };
            typedef void (*event_callback)(event, ios_base &, int);
            void register_callback(event_callback __fn, int __index);
        protected :
            streamsize _M_precision;
            streamsize _M_width;
            fmtflags _M_flags;
            iostate _M_exception;
            iostate _M_streambuf_state;
            struct _Callback_list
            {
                    _Callback_list *_M_next;
                    ios_base::event_callback _M_fn;
                    int _M_index;
                    _Atomic_word _M_refcount;
                    _Callback_list(ios_base::event_callback __fn, int __index, _Callback_list *__cb)
                        : _M_next(__cb), _M_fn(__fn), _M_index(__index), _M_refcount(0) 
                    {
                    }
                    void _M_add_reference()
                    {
                        __gnu_cxx::__atomic_add_dispatch(&_M_refcount, 1);
                    }
                    int _M_remove_reference()
                    {
                        return __gnu_cxx::__exchange_and_add_dispatch(&_M_refcount, - 1);
                    }
            };
            _Callback_list *_M_callbacks;
            void _M_call_callbacks(event __ev) throw ();
            void _M_dispose_callbacks(void);
            struct _Words
            {
                    void *_M_pword;
                    long _M_iword;
                    _Words()
                        : _M_pword(0), _M_iword(0) 
                    {
                    }
            };
            _Words _M_word_zero;
            enum 
            {
                _S_local_word_size = 8
            };
            _Words _M_local_word[_S_local_word_size];
            int _M_word_size;
            _Words *_M_word;
            _Words &_M_grow_words(int __index, bool __iword);
            locale _M_ios_locale;
            void _M_init();
        public :
            class Init
            {
                    friend class ios_base;
                public :
                    Init();
                    ~Init();
                private :
                    static _Atomic_word _S_refcount;
                    static bool _S_synced_with_stdio;
            };
            fmtflags flags() const
            {
                return _M_flags;
            }
            fmtflags flags(fmtflags __fmtfl)
            {
                fmtflags __old = _M_flags;
                _M_flags = __fmtfl;
                return __old;
            }
            fmtflags setf(fmtflags __fmtfl)
            {
                fmtflags __old = _M_flags;
                _M_flags |= __fmtfl;
                return __old;
            }
            fmtflags setf(fmtflags __fmtfl, fmtflags __mask)
            {
                fmtflags __old = _M_flags;
                _M_flags &= ~__mask;
                _M_flags |= (__fmtfl & __mask);
                return __old;
            }
            void unsetf(fmtflags __mask)
            {
                _M_flags &= ~__mask;
            }
            streamsize precision() const
            {
                return _M_precision;
            }
            streamsize precision(streamsize __prec)
            {
                streamsize __old = _M_precision;
                _M_precision = __prec;
                return __old;
            }
            streamsize width() const
            {
                return _M_width;
            }
            streamsize width(streamsize __wide)
            {
                streamsize __old = _M_width;
                _M_width = __wide;
                return __old;
            }
            static bool sync_with_stdio(bool __sync = true);
            locale imbue(const locale &__loc);
            locale getloc() const
            {
                return _M_ios_locale;
            }
            const locale &_M_getloc() const
            {
                return _M_ios_locale;
            }
            static int xalloc() throw ();
            long &iword(int __ix)
            {
                _Words &__word = (__ix < _M_word_size) ? _M_word[__ix] : _M_grow_words(__ix, true);
                return __word._M_iword;
            }
            void *&pword(int __ix)
            {
                _Words &__word = (__ix < _M_word_size) ? _M_word[__ix] : _M_grow_words(__ix, false);
                return __word._M_pword;
            }
            virtual ~ios_base();
        protected :
            ios_base();
        private :
            ios_base(const ios_base &);
            ios_base &operator =(const ios_base &);
    };
    inline ios_base &boolalpha(ios_base &__base)
    {
        __base.setf(ios_base::boolalpha);
        return __base;
    }
    inline ios_base &noboolalpha(ios_base &__base)
    {
        __base.unsetf(ios_base::boolalpha);
        return __base;
    }
    inline ios_base &showbase(ios_base &__base)
    {
        __base.setf(ios_base::showbase);
        return __base;
    }
    inline ios_base &noshowbase(ios_base &__base)
    {
        __base.unsetf(ios_base::showbase);
        return __base;
    }
    inline ios_base &showpoint(ios_base &__base)
    {
        __base.setf(ios_base::showpoint);
        return __base;
    }
    inline ios_base &noshowpoint(ios_base &__base)
    {
        __base.unsetf(ios_base::showpoint);
        return __base;
    }
    inline ios_base &showpos(ios_base &__base)
    {
        __base.setf(ios_base::showpos);
        return __base;
    }
    inline ios_base &noshowpos(ios_base &__base)
    {
        __base.unsetf(ios_base::showpos);
        return __base;
    }
    inline ios_base &skipws(ios_base &__base)
    {
        __base.setf(ios_base::skipws);
        return __base;
    }
    inline ios_base &noskipws(ios_base &__base)
    {
        __base.unsetf(ios_base::skipws);
        return __base;
    }
    inline ios_base &uppercase(ios_base &__base)
    {
        __base.setf(ios_base::uppercase);
        return __base;
    }
    inline ios_base &nouppercase(ios_base &__base)
    {
        __base.unsetf(ios_base::uppercase);
        return __base;
    }
    inline ios_base &unitbuf(ios_base &__base)
    {
        __base.setf(ios_base::unitbuf);
        return __base;
    }
    inline ios_base &nounitbuf(ios_base &__base)
    {
        __base.unsetf(ios_base::unitbuf);
        return __base;
    }
    inline ios_base &internal(ios_base &__base)
    {
        __base.setf(ios_base::internal, ios_base::adjustfield);
        return __base;
    }
    inline ios_base &left(ios_base &__base)
    {
        __base.setf(ios_base::left, ios_base::adjustfield);
        return __base;
    }
    inline ios_base &right(ios_base &__base)
    {
        __base.setf(ios_base::right, ios_base::adjustfield);
        return __base;
    }
    inline ios_base &dec(ios_base &__base)
    {
        __base.setf(ios_base::dec, ios_base::basefield);
        return __base;
    }
    inline ios_base &hex(ios_base &__base)
    {
        __base.setf(ios_base::hex, ios_base::basefield);
        return __base;
    }
    inline ios_base &oct(ios_base &__base)
    {
        __base.setf(ios_base::oct, ios_base::basefield);
        return __base;
    }
    inline ios_base &fixed(ios_base &__base)
    {
        __base.setf(ios_base::fixed, ios_base::floatfield);
        return __base;
    }
    inline ios_base &scientific(ios_base &__base)
    {
        __base.setf(ios_base::scientific, ios_base::floatfield);
        return __base;
    }
}
namespace std __attribute__((__visibility__("default"))) {
    template<typename _CharT, typename _Traits >
    streamsize __copy_streambufs_eof(basic_streambuf<_CharT, _Traits> *, basic_streambuf<_CharT, _Traits> *, bool &);
    template<typename _CharT, typename _Traits >
    class basic_streambuf
    {
        public :
            typedef _CharT char_type;
            typedef _Traits traits_type;
            typedef typename traits_type::int_type int_type;
            typedef typename traits_type::pos_type pos_type;
            typedef typename traits_type::off_type off_type;
            typedef basic_streambuf<char_type, traits_type> __streambuf_type;
            friend class basic_ios<char_type, traits_type>;
            friend class basic_istream<char_type, traits_type>;
            friend class basic_ostream<char_type, traits_type>;
            friend class istreambuf_iterator<char_type, traits_type>;
            friend class ostreambuf_iterator<char_type, traits_type>;
            friend streamsize __copy_streambufs_eof<>(__streambuf_type *, __streambuf_type *, bool &);
            template<bool _IsMove, typename _CharT2 >
            friend typename __gnu_cxx::__enable_if<__is_char<_CharT2>::__value, _CharT2 *>::__type __copy_move_a2(istreambuf_iterator<_CharT2>, istreambuf_iterator<_CharT2>, _CharT2 *);
            template<typename _CharT2 >
            friend typename __gnu_cxx::__enable_if<__is_char<_CharT2>::__value, istreambuf_iterator<_CharT2> >::__type find(istreambuf_iterator<_CharT2>, istreambuf_iterator<_CharT2>, const _CharT2 &);
            template<typename _CharT2, typename _Traits2 >
            friend basic_istream<_CharT2, _Traits2> &operator >>(basic_istream<_CharT2, _Traits2> &, _CharT2 *);
            template<typename _CharT2, typename _Traits2, typename _Alloc >
            friend basic_istream<_CharT2, _Traits2> &operator >>(basic_istream<_CharT2, _Traits2> &, basic_string<_CharT2, _Traits2, _Alloc> &);
            template<typename _CharT2, typename _Traits2, typename _Alloc >
            friend basic_istream<_CharT2, _Traits2> &getline(basic_istream<_CharT2, _Traits2> &, basic_string<_CharT2, _Traits2, _Alloc> &, _CharT2);
        protected :
            char_type *_M_in_beg;
            char_type *_M_in_cur;
            char_type *_M_in_end;
            char_type *_M_out_beg;
            char_type *_M_out_cur;
            char_type *_M_out_end;
            locale _M_buf_locale;
        public :
            virtual ~basic_streambuf()
            {
            }
            locale pubimbue(const locale &__loc)
            {
                locale __tmp(this->getloc());
                this->imbue(__loc);
                _M_buf_locale = __loc;
                return __tmp;
            }
            locale getloc() const
            {
                return _M_buf_locale;
            }
            __streambuf_type *pubsetbuf(char_type *__s, streamsize __n)
            {
                return this->setbuf(__s, __n);
            }
            pos_type pubseekoff(off_type __off, ios_base::seekdir __way, ios_base::openmode __mode = ios_base::in | ios_base::out)
            {
                return this->seekoff(__off, __way, __mode);
            }
            pos_type pubseekpos(pos_type __sp, ios_base::openmode __mode = ios_base::in | ios_base::out)
            {
                return this->seekpos(__sp, __mode);
            }
            int pubsync()
            {
                return this->sync();
            }
            streamsize in_avail()
            {
                const streamsize __ret = this->egptr() - this->gptr();
                return __ret ? __ret : this->showmanyc();
            }
            int_type snextc()
            {
                int_type __ret = traits_type::eof();
                if (__builtin_expect(!traits_type::eq_int_type(this->sbumpc(), __ret), true))
                    __ret = this->sgetc();
                return __ret;
            }
            int_type sbumpc()
            {
                int_type __ret;
                if (__builtin_expect(this->gptr() < this->egptr(), true))
                {
                    __ret = traits_type::to_int_type(*this->gptr());
                    this->gbump(1);
                }
                else
                    __ret = this->uflow();
                return __ret;
            }
            int_type sgetc()
            {
                int_type __ret;
                if (__builtin_expect(this->gptr() < this->egptr(), true))
                    __ret = traits_type::to_int_type(*this->gptr());
                else
                    __ret = this->underflow();
                return __ret;
            }
            streamsize sgetn(char_type *__s, streamsize __n)
            {
                return this->xsgetn(__s, __n);
            }
            int_type sputbackc(char_type __c)
            {
                int_type __ret;
                const bool __testpos = this->eback() < this->gptr();
                if (__builtin_expect(!__testpos || !traits_type::eq(__c, this->gptr()[- 1]), false))
                    __ret = this->pbackfail(traits_type::to_int_type(__c));
                else
                {
                    this->gbump(- 1);
                    __ret = traits_type::to_int_type(*this->gptr());
                }
                return __ret;
            }
            int_type sungetc()
            {
                int_type __ret;
                if (__builtin_expect(this->eback() < this->gptr(), true))
                {
                    this->gbump(- 1);
                    __ret = traits_type::to_int_type(*this->gptr());
                }
                else
                    __ret = this->pbackfail();
                return __ret;
            }
            int_type sputc(char_type __c)
            {
                int_type __ret;
                if (__builtin_expect(this->pptr() < this->epptr(), true))
                {
                    *this->pptr() = __c;
                    this->pbump(1);
                    __ret = traits_type::to_int_type(__c);
                }
                else
                    __ret = this->overflow(traits_type::to_int_type(__c));
                return __ret;
            }
            streamsize sputn(const char_type *__s, streamsize __n)
            {
                return this->xsputn(__s, __n);
            }
        protected :
            basic_streambuf()
                : _M_in_beg(0), _M_in_cur(0), _M_in_end(0), _M_out_beg(0), _M_out_cur(0), _M_out_end(0), _M_buf_locale(locale()) 
            {
            }
            char_type *eback() const
            {
                return _M_in_beg;
            }
            char_type *gptr() const
            {
                return _M_in_cur;
            }
            char_type *egptr() const
            {
                return _M_in_end;
            }
            void gbump(int __n)
            {
                _M_in_cur += __n;
            }
            void setg(char_type *__gbeg, char_type *__gnext, char_type *__gend)
            {
                _M_in_beg = __gbeg;
                _M_in_cur = __gnext;
                _M_in_end = __gend;
            }
            char_type *pbase() const
            {
                return _M_out_beg;
            }
            char_type *pptr() const
            {
                return _M_out_cur;
            }
            char_type *epptr() const
            {
                return _M_out_end;
            }
            void pbump(int __n)
            {
                _M_out_cur += __n;
            }
            void setp(char_type *__pbeg, char_type *__pend)
            {
                _M_out_beg = _M_out_cur = __pbeg;
                _M_out_end = __pend;
            }
            virtual void imbue(const locale &)
            {
            }
            virtual basic_streambuf<char_type, _Traits> *setbuf(char_type *, streamsize)
            {
                return this;
            }
            virtual pos_type seekoff(off_type, ios_base::seekdir, ios_base::openmode = ios_base::in | ios_base::out)
            {
                return pos_type(off_type(- 1));
            }
            virtual pos_type seekpos(pos_type, ios_base::openmode = ios_base::in | ios_base::out)
            {
                return pos_type(off_type(- 1));
            }
            virtual int sync()
            {
                return 0;
            }
            virtual streamsize showmanyc()
            {
                return 0;
            }
            virtual streamsize xsgetn(char_type *__s, streamsize __n);
            virtual int_type underflow()
            {
                return traits_type::eof();
            }
            virtual int_type uflow()
            {
                int_type __ret = traits_type::eof();
                const bool __testeof = traits_type::eq_int_type(this->underflow(), __ret);
                if (!__testeof)
                {
                    __ret = traits_type::to_int_type(*this->gptr());
                    this->gbump(1);
                }
                return __ret;
            }
            virtual int_type pbackfail(int_type = traits_type::eof())
            {
                return traits_type::eof();
            }
            virtual streamsize xsputn(const char_type *__s, streamsize __n);
            virtual int_type overflow(int_type = traits_type::eof())
            {
                return traits_type::eof();
            }
        public :
            void stossc()
            {
                if (this->gptr() < this->egptr())
                    this->gbump(1);
                else
                    this->uflow();
            }
        private :
            basic_streambuf(const __streambuf_type &__sb)
                : _M_in_beg(__sb._M_in_beg), _M_in_cur(__sb._M_in_cur), _M_in_end(__sb._M_in_end), _M_out_beg(__sb._M_out_beg), _M_out_cur(__sb._M_out_cur), _M_out_end(__sb._M_out_cur), _M_buf_locale(__sb._M_buf_locale) 
            {
            }
            __streambuf_type &operator =(const __streambuf_type &)
            {
                return *this;
            }
            ;
    };
    template<>
    streamsize __copy_streambufs_eof(basic_streambuf<char> *__sbin, basic_streambuf<char> *__sbout, bool &__ineof);
    template<>
    streamsize __copy_streambufs_eof(basic_streambuf<wchar_t> *__sbin, basic_streambuf<wchar_t> *__sbout, bool &__ineof);
}
namespace std __attribute__((__visibility__("default"))) {
    template<typename _CharT, typename _Traits >
    streamsize basic_streambuf<_CharT, _Traits>::xsgetn(char_type *__s, streamsize __n)
    {
        streamsize __ret = 0;
        while (__ret < __n)
        {
            const streamsize __buf_len = this->egptr() - this->gptr();
            if (__buf_len)
            {
                const streamsize __remaining = __n - __ret;
                const streamsize __len = std::min(__buf_len, __remaining);
                traits_type::copy(__s, this->gptr(), __len);
                __ret += __len;
                __s += __len;
                this->gbump(__len);
            }
            if (__ret < __n)
            {
                const int_type __c = this->uflow();
                if (!traits_type::eq_int_type(__c, traits_type::eof()))
                {
                    traits_type::assign(*__s++, traits_type::to_char_type(__c));
                    ++__ret;
                }
                else
                    break;
            }
        }
        return __ret;
    }
    template<typename _CharT, typename _Traits >
    streamsize basic_streambuf<_CharT, _Traits>::xsputn(const char_type *__s, streamsize __n)
    {
        streamsize __ret = 0;
        while (__ret < __n)
        {
            const streamsize __buf_len = this->epptr() - this->pptr();
            if (__buf_len)
            {
                const streamsize __remaining = __n - __ret;
                const streamsize __len = std::min(__buf_len, __remaining);
                traits_type::copy(this->pptr(), __s, __len);
                __ret += __len;
                __s += __len;
                this->pbump(__len);
            }
            if (__ret < __n)
            {
                int_type __c = this->overflow(traits_type::to_int_type(*__s));
                if (!traits_type::eq_int_type(__c, traits_type::eof()))
                {
                    ++__ret;
                    ++__s;
                }
                else
                    break;
            }
        }
        return __ret;
    }
    template<typename _CharT, typename _Traits >
    streamsize __copy_streambufs_eof(basic_streambuf<_CharT, _Traits> *__sbin, basic_streambuf<_CharT, _Traits> *__sbout, bool &__ineof)
    {
        streamsize __ret = 0;
        __ineof = true;
        typename _Traits::int_type __c = __sbin->sgetc();
        while (!_Traits::eq_int_type(__c, _Traits::eof()))
        {
            __c = __sbout->sputc(_Traits::to_char_type(__c));
            if (_Traits::eq_int_type(__c, _Traits::eof()))
            {
                __ineof = false;
                break;
            }
            ++__ret;
            __c = __sbin->snextc();
        }
        return __ret;
    }
    template<typename _CharT, typename _Traits >
    inline streamsize __copy_streambufs(basic_streambuf<_CharT, _Traits> *__sbin, basic_streambuf<_CharT, _Traits> *__sbout)
    {
        bool __ineof;
        return __copy_streambufs_eof(__sbin, __sbout, __ineof);
    }
    extern template class basic_streambuf<char>;
    extern template streamsize __copy_streambufs(basic_streambuf<char> *, basic_streambuf<char> *);
    extern template streamsize __copy_streambufs_eof(basic_streambuf<char> *, basic_streambuf<char> *, bool &);
    extern template class basic_streambuf<wchar_t>;
    extern template streamsize __copy_streambufs(basic_streambuf<wchar_t> *, basic_streambuf<wchar_t> *);
    extern template streamsize __copy_streambufs_eof(basic_streambuf<wchar_t> *, basic_streambuf<wchar_t> *, bool &);
}
typedef unsigned long int wctype_t;
enum 
{
    __ISwupper = 0, 
    __ISwlower = 1, 
    __ISwalpha = 2, 
    __ISwdigit = 3, 
    __ISwxdigit = 4, 
    __ISwspace = 5, 
    __ISwprint = 6, 
    __ISwgraph = 7, 
    __ISwblank = 8, 
    __ISwcntrl = 9, 
    __ISwpunct = 10, 
    __ISwalnum = 11, 
    _ISwupper = ((__ISwupper) < 8 ? (int) ((1UL << (__ISwupper)) << 24) : ((__ISwupper) < 16 ? (int) ((1UL << (__ISwupper)) << 8) : ((__ISwupper) < 24 ? (int) ((1UL << (__ISwupper)) >> 8) : (int) ((1UL << (__ISwupper)) >> 24)))), 
    _ISwlower = ((__ISwlower) < 8 ? (int) ((1UL << (__ISwlower)) << 24) : ((__ISwlower) < 16 ? (int) ((1UL << (__ISwlower)) << 8) : ((__ISwlower) < 24 ? (int) ((1UL << (__ISwlower)) >> 8) : (int) ((1UL << (__ISwlower)) >> 24)))), 
    _ISwalpha = ((__ISwalpha) < 8 ? (int) ((1UL << (__ISwalpha)) << 24) : ((__ISwalpha) < 16 ? (int) ((1UL << (__ISwalpha)) << 8) : ((__ISwalpha) < 24 ? (int) ((1UL << (__ISwalpha)) >> 8) : (int) ((1UL << (__ISwalpha)) >> 24)))), 
    _ISwdigit = ((__ISwdigit) < 8 ? (int) ((1UL << (__ISwdigit)) << 24) : ((__ISwdigit) < 16 ? (int) ((1UL << (__ISwdigit)) << 8) : ((__ISwdigit) < 24 ? (int) ((1UL << (__ISwdigit)) >> 8) : (int) ((1UL << (__ISwdigit)) >> 24)))), 
    _ISwxdigit = ((__ISwxdigit) < 8 ? (int) ((1UL << (__ISwxdigit)) << 24) : ((__ISwxdigit) < 16 ? (int) ((1UL << (__ISwxdigit)) << 8) : ((__ISwxdigit) < 24 ? (int) ((1UL << (__ISwxdigit)) >> 8) : (int) ((1UL << (__ISwxdigit)) >> 24)))), 
    _ISwspace = ((__ISwspace) < 8 ? (int) ((1UL << (__ISwspace)) << 24) : ((__ISwspace) < 16 ? (int) ((1UL << (__ISwspace)) << 8) : ((__ISwspace) < 24 ? (int) ((1UL << (__ISwspace)) >> 8) : (int) ((1UL << (__ISwspace)) >> 24)))), 
    _ISwprint = ((__ISwprint) < 8 ? (int) ((1UL << (__ISwprint)) << 24) : ((__ISwprint) < 16 ? (int) ((1UL << (__ISwprint)) << 8) : ((__ISwprint) < 24 ? (int) ((1UL << (__ISwprint)) >> 8) : (int) ((1UL << (__ISwprint)) >> 24)))), 
    _ISwgraph = ((__ISwgraph) < 8 ? (int) ((1UL << (__ISwgraph)) << 24) : ((__ISwgraph) < 16 ? (int) ((1UL << (__ISwgraph)) << 8) : ((__ISwgraph) < 24 ? (int) ((1UL << (__ISwgraph)) >> 8) : (int) ((1UL << (__ISwgraph)) >> 24)))), 
    _ISwblank = ((__ISwblank) < 8 ? (int) ((1UL << (__ISwblank)) << 24) : ((__ISwblank) < 16 ? (int) ((1UL << (__ISwblank)) << 8) : ((__ISwblank) < 24 ? (int) ((1UL << (__ISwblank)) >> 8) : (int) ((1UL << (__ISwblank)) >> 24)))), 
    _ISwcntrl = ((__ISwcntrl) < 8 ? (int) ((1UL << (__ISwcntrl)) << 24) : ((__ISwcntrl) < 16 ? (int) ((1UL << (__ISwcntrl)) << 8) : ((__ISwcntrl) < 24 ? (int) ((1UL << (__ISwcntrl)) >> 8) : (int) ((1UL << (__ISwcntrl)) >> 24)))), 
    _ISwpunct = ((__ISwpunct) < 8 ? (int) ((1UL << (__ISwpunct)) << 24) : ((__ISwpunct) < 16 ? (int) ((1UL << (__ISwpunct)) << 8) : ((__ISwpunct) < 24 ? (int) ((1UL << (__ISwpunct)) >> 8) : (int) ((1UL << (__ISwpunct)) >> 24)))), 
    _ISwalnum = ((__ISwalnum) < 8 ? (int) ((1UL << (__ISwalnum)) << 24) : ((__ISwalnum) < 16 ? (int) ((1UL << (__ISwalnum)) << 8) : ((__ISwalnum) < 24 ? (int) ((1UL << (__ISwalnum)) >> 8) : (int) ((1UL << (__ISwalnum)) >> 24))))
};
extern "C"
{
    extern int iswalnum(wint_t __wc) throw ();
    extern int iswalpha(wint_t __wc) throw ();
    extern int iswcntrl(wint_t __wc) throw ();
    extern int iswdigit(wint_t __wc) throw ();
    extern int iswgraph(wint_t __wc) throw ();
    extern int iswlower(wint_t __wc) throw ();
    extern int iswprint(wint_t __wc) throw ();
    extern int iswpunct(wint_t __wc) throw ();
    extern int iswspace(wint_t __wc) throw ();
    extern int iswupper(wint_t __wc) throw ();
    extern int iswxdigit(wint_t __wc) throw ();
    extern int iswblank(wint_t __wc) throw ();
    extern wctype_t wctype(__const char *__property) throw ();
    extern int iswctype(wint_t __wc, wctype_t __desc) throw ();
    typedef __const __int32_t *wctrans_t;
    extern wint_t towlower(wint_t __wc) throw ();
    extern wint_t towupper(wint_t __wc) throw ();
}
extern "C"
{
    extern wctrans_t wctrans(__const char *__property) throw ();
    extern wint_t towctrans(wint_t __wc, wctrans_t __desc) throw ();
    extern int iswalnum_l(wint_t __wc, __locale_t __locale) throw ();
    extern int iswalpha_l(wint_t __wc, __locale_t __locale) throw ();
    extern int iswcntrl_l(wint_t __wc, __locale_t __locale) throw ();
    extern int iswdigit_l(wint_t __wc, __locale_t __locale) throw ();
    extern int iswgraph_l(wint_t __wc, __locale_t __locale) throw ();
    extern int iswlower_l(wint_t __wc, __locale_t __locale) throw ();
    extern int iswprint_l(wint_t __wc, __locale_t __locale) throw ();
    extern int iswpunct_l(wint_t __wc, __locale_t __locale) throw ();
    extern int iswspace_l(wint_t __wc, __locale_t __locale) throw ();
    extern int iswupper_l(wint_t __wc, __locale_t __locale) throw ();
    extern int iswxdigit_l(wint_t __wc, __locale_t __locale) throw ();
    extern int iswblank_l(wint_t __wc, __locale_t __locale) throw ();
    extern wctype_t wctype_l(__const char *__property, __locale_t __locale) throw ();
    extern int iswctype_l(wint_t __wc, wctype_t __desc, __locale_t __locale) throw ();
    extern wint_t towlower_l(wint_t __wc, __locale_t __locale) throw ();
    extern wint_t towupper_l(wint_t __wc, __locale_t __locale) throw ();
    extern wctrans_t wctrans_l(__const char *__property, __locale_t __locale) throw ();
    extern wint_t towctrans_l(wint_t __wc, wctrans_t __desc, __locale_t __locale) throw ();
}
namespace std __attribute__((__visibility__("default"))) {
    using ::wctrans_t;
    using ::wctype_t;
    using ::wint_t;
    using ::iswalnum;
    using ::iswalpha;
    using ::iswblank;
    using ::iswcntrl;
    using ::iswctype;
    using ::iswdigit;
    using ::iswgraph;
    using ::iswlower;
    using ::iswprint;
    using ::iswpunct;
    using ::iswspace;
    using ::iswupper;
    using ::iswxdigit;
    using ::towctrans;
    using ::towlower;
    using ::towupper;
    using ::wctrans;
    using ::wctype;
}
namespace std __attribute__((__visibility__("default"))) {
    struct ctype_base
    {
            typedef const int *__to_type;
            typedef unsigned short mask;
            static const mask upper  = _ISupper;
            static const mask lower  = _ISlower;
            static const mask alpha  = _ISalpha;
            static const mask digit  = _ISdigit;
            static const mask xdigit  = _ISxdigit;
            static const mask space  = _ISspace;
            static const mask print  = _ISprint;
            static const mask graph  = _ISalpha | _ISdigit | _ISpunct;
            static const mask cntrl  = _IScntrl;
            static const mask punct  = _ISpunct;
            static const mask alnum  = _ISalpha | _ISdigit;
    };
}
namespace std __attribute__((__visibility__("default"))) {
    template<typename _CharT, typename _Traits >
    class istreambuf_iterator : public iterator<input_iterator_tag, _CharT, typename _Traits::off_type, _CharT *, _CharT &>
    {
        public :
            typedef _CharT char_type;
            typedef _Traits traits_type;
            typedef typename _Traits::int_type int_type;
            typedef basic_streambuf<_CharT, _Traits> streambuf_type;
            typedef basic_istream<_CharT, _Traits> istream_type;
            template<typename _CharT2 >
            friend typename __gnu_cxx::__enable_if<__is_char<_CharT2>::__value, ostreambuf_iterator<_CharT2> >::__type copy(istreambuf_iterator<_CharT2>, istreambuf_iterator<_CharT2>, ostreambuf_iterator<_CharT2>);
            template<bool _IsMove, typename _CharT2 >
            friend typename __gnu_cxx::__enable_if<__is_char<_CharT2>::__value, _CharT2 *>::__type __copy_move_a2(istreambuf_iterator<_CharT2>, istreambuf_iterator<_CharT2>, _CharT2 *);
            template<typename _CharT2 >
            friend typename __gnu_cxx::__enable_if<__is_char<_CharT2>::__value, istreambuf_iterator<_CharT2> >::__type find(istreambuf_iterator<_CharT2>, istreambuf_iterator<_CharT2>, const _CharT2 &);
        private :
            mutable streambuf_type *_M_sbuf;
            mutable int_type _M_c;
        public :
            istreambuf_iterator() throw ()
                : _M_sbuf(0), _M_c(traits_type::eof()) 
            {
            }
            istreambuf_iterator(istream_type &__s) throw ()
                : _M_sbuf(__s.rdbuf()), _M_c(traits_type::eof()) 
            {
            }
            istreambuf_iterator(streambuf_type *__s) throw ()
                : _M_sbuf(__s), _M_c(traits_type::eof()) 
            {
            }
            char_type operator *() const
            {
                return traits_type::to_char_type(_M_get());
            }
            istreambuf_iterator &operator ++()
            {
                ;
                if (_M_sbuf)
                {
                    _M_sbuf->sbumpc();
                    _M_c = traits_type::eof();
                }
                return *this;
            }
            istreambuf_iterator operator ++(int)
            {
                ;
                istreambuf_iterator __old = *this;
                if (_M_sbuf)
                {
                    __old._M_c = _M_sbuf->sbumpc();
                    _M_c = traits_type::eof();
                }
                return __old;
            }
            bool equal(const istreambuf_iterator &__b) const
            {
                return _M_at_eof() == __b._M_at_eof();
            }
        private :
            int_type _M_get() const
            {
                const int_type __eof = traits_type::eof();
                int_type __ret = __eof;
                if (_M_sbuf)
                {
                    if (!traits_type::eq_int_type(_M_c, __eof))
                        __ret = _M_c;
                    else
                        if (!traits_type::eq_int_type((__ret = _M_sbuf->sgetc()), __eof))
                            _M_c = __ret;
                        else
                            _M_sbuf = 0;
                }
                return __ret;
            }
            bool _M_at_eof() const
            {
                const int_type __eof = traits_type::eof();
                return traits_type::eq_int_type(_M_get(), __eof);
            }
    };
    template<typename _CharT, typename _Traits >
    inline bool operator ==(const istreambuf_iterator<_CharT, _Traits> &__a, const istreambuf_iterator<_CharT, _Traits> &__b)
    {
        return __a.equal(__b);
    }
    template<typename _CharT, typename _Traits >
    inline bool operator !=(const istreambuf_iterator<_CharT, _Traits> &__a, const istreambuf_iterator<_CharT, _Traits> &__b)
    {
        return !__a.equal(__b);
    }
    template<typename _CharT, typename _Traits >
    class ostreambuf_iterator : public iterator<output_iterator_tag, void, void, void, void>
    {
        public :
            typedef _CharT char_type;
            typedef _Traits traits_type;
            typedef basic_streambuf<_CharT, _Traits> streambuf_type;
            typedef basic_ostream<_CharT, _Traits> ostream_type;
            template<typename _CharT2 >
            friend typename __gnu_cxx::__enable_if<__is_char<_CharT2>::__value, ostreambuf_iterator<_CharT2> >::__type copy(istreambuf_iterator<_CharT2>, istreambuf_iterator<_CharT2>, ostreambuf_iterator<_CharT2>);
        private :
            streambuf_type *_M_sbuf;
            bool _M_failed;
        public :
            ostreambuf_iterator(ostream_type &__s) throw ()
                : _M_sbuf(__s.rdbuf()), _M_failed(!_M_sbuf) 
            {
            }
            ostreambuf_iterator(streambuf_type *__s) throw ()
                : _M_sbuf(__s), _M_failed(!_M_sbuf) 
            {
            }
            ostreambuf_iterator &operator =(_CharT __c)
            {
                if (!_M_failed && _Traits::eq_int_type(_M_sbuf->sputc(__c), _Traits::eof()))
                    _M_failed = true;
                return *this;
            }
            ostreambuf_iterator &operator *()
            {
                return *this;
            }
            ostreambuf_iterator &operator ++(int)
            {
                return *this;
            }
            ostreambuf_iterator &operator ++()
            {
                return *this;
            }
            bool failed() const throw ()
            {
                return _M_failed;
            }
            ostreambuf_iterator &_M_put(const _CharT *__ws, streamsize __len)
            {
                if (__builtin_expect(!_M_failed, true) && __builtin_expect(this->_M_sbuf->sputn(__ws, __len) != __len, false))
                    _M_failed = true;
                return *this;
            }
    };
    template<typename _CharT >
    typename __gnu_cxx::__enable_if<__is_char<_CharT>::__value, ostreambuf_iterator<_CharT> >::__type copy(istreambuf_iterator<_CharT> __first, istreambuf_iterator<_CharT> __last, ostreambuf_iterator<_CharT> __result)
    {
        if (__first._M_sbuf && !__last._M_sbuf && !__result._M_failed)
        {
            bool __ineof;
            __copy_streambufs_eof(__first._M_sbuf, __result._M_sbuf, __ineof);
            if (!__ineof)
                __result._M_failed = true;
        }
        return __result;
    }
    template<bool _IsMove, typename _CharT >
    typename __gnu_cxx::__enable_if<__is_char<_CharT>::__value, ostreambuf_iterator<_CharT> >::__type __copy_move_a2(_CharT *__first, _CharT *__last, ostreambuf_iterator<_CharT> __result)
    {
        const streamsize __num = __last - __first;
        if (__num > 0)
            __result._M_put(__first, __num);
        return __result;
    }
    template<bool _IsMove, typename _CharT >
    typename __gnu_cxx::__enable_if<__is_char<_CharT>::__value, ostreambuf_iterator<_CharT> >::__type __copy_move_a2(const _CharT *__first, const _CharT *__last, ostreambuf_iterator<_CharT> __result)
    {
        const streamsize __num = __last - __first;
        if (__num > 0)
            __result._M_put(__first, __num);
        return __result;
    }
    template<bool _IsMove, typename _CharT >
    typename __gnu_cxx::__enable_if<__is_char<_CharT>::__value, _CharT *>::__type __copy_move_a2(istreambuf_iterator<_CharT> __first, istreambuf_iterator<_CharT> __last, _CharT *__result)
    {
        typedef istreambuf_iterator<_CharT> __is_iterator_type;
        typedef typename __is_iterator_type::traits_type traits_type;
        typedef typename __is_iterator_type::streambuf_type streambuf_type;
        typedef typename traits_type::int_type int_type;
        if (__first._M_sbuf && !__last._M_sbuf)
        {
            streambuf_type *__sb = __first._M_sbuf;
            int_type __c = __sb->sgetc();
            while (!traits_type::eq_int_type(__c, traits_type::eof()))
            {
                const streamsize __n = __sb->egptr() - __sb->gptr();
                if (__n > 1)
                {
                    traits_type::copy(__result, __sb->gptr(), __n);
                    __sb->gbump(__n);
                    __result += __n;
                    __c = __sb->underflow();
                }
                else
                {
                    *__result++ = traits_type::to_char_type(__c);
                    __c = __sb->snextc();
                }
            }
        }
        return __result;
    }
    template<typename _CharT >
    typename __gnu_cxx::__enable_if<__is_char<_CharT>::__value, istreambuf_iterator<_CharT> >::__type find(istreambuf_iterator<_CharT> __first, istreambuf_iterator<_CharT> __last, const _CharT &__val)
    {
        typedef istreambuf_iterator<_CharT> __is_iterator_type;
        typedef typename __is_iterator_type::traits_type traits_type;
        typedef typename __is_iterator_type::streambuf_type streambuf_type;
        typedef typename traits_type::int_type int_type;
        if (__first._M_sbuf && !__last._M_sbuf)
        {
            const int_type __ival = traits_type::to_int_type(__val);
            streambuf_type *__sb = __first._M_sbuf;
            int_type __c = __sb->sgetc();
            while (!traits_type::eq_int_type(__c, traits_type::eof()) && !traits_type::eq_int_type(__c, __ival))
            {
                streamsize __n = __sb->egptr() - __sb->gptr();
                if (__n > 1)
                {
                    const _CharT *__p = traits_type::find(__sb->gptr(), __n, __val);
                    if (__p)
                        __n = __p - __sb->gptr();
                    __sb->gbump(__n);
                    __c = __sb->sgetc();
                }
                else
                    __c = __sb->snextc();
            }
            if (!traits_type::eq_int_type(__c, traits_type::eof()))
                __first._M_c = __c;
            else
                __first._M_sbuf = 0;
        }
        return __first;
    }
}
namespace std __attribute__((__visibility__("default"))) {
    template<typename _Tv >
    void __convert_to_v(const char *__in, _Tv &__out, ios_base::iostate &__err, const __c_locale &__cloc);
    template<>
    void __convert_to_v(const char *, float &, ios_base::iostate &, const __c_locale &);
    template<>
    void __convert_to_v(const char *, double &, ios_base::iostate &, const __c_locale &);
    template<>
    void __convert_to_v(const char *, long double &, ios_base::iostate &, const __c_locale &);
    template<typename _CharT, typename _Traits >
    struct __pad
    {
            static void _S_pad(ios_base &__io, _CharT __fill, _CharT *__news, const _CharT *__olds, streamsize __newlen, streamsize __oldlen);
    };
    template<typename _CharT >
    _CharT *__add_grouping(_CharT *__s, _CharT __sep, const char *__gbeg, size_t __gsize, const _CharT *__first, const _CharT *__last);
    template<typename _CharT >
    inline ostreambuf_iterator<_CharT> __write(ostreambuf_iterator<_CharT> __s, const _CharT *__ws, int __len)
    {
        __s._M_put(__ws, __len);
        return __s;
    }
    template<typename _CharT, typename _OutIter >
    inline _OutIter __write(_OutIter __s, const _CharT *__ws, int __len)
    {
        for (int __j = 0;
            __j < __len;
            __j++ , ++__s)
            *__s = __ws[__j];
        return __s;
    }
    template<typename _CharT >
    class __ctype_abstract_base : public locale::facet, public ctype_base
    {
        public :
            typedef _CharT char_type;
            bool is(mask __m, char_type __c) const
            {
                return this->do_is(__m, __c);
            }
            const char_type *is(const char_type *__lo, const char_type *__hi, mask *__vec) const
            {
                return this->do_is(__lo, __hi, __vec);
            }
            const char_type *scan_is(mask __m, const char_type *__lo, const char_type *__hi) const
            {
                return this->do_scan_is(__m, __lo, __hi);
            }
            const char_type *scan_not(mask __m, const char_type *__lo, const char_type *__hi) const
            {
                return this->do_scan_not(__m, __lo, __hi);
            }
            char_type toupper(char_type __c) const
            {
                return this->do_toupper(__c);
            }
            const char_type *toupper(char_type *__lo, const char_type *__hi) const
            {
                return this->do_toupper(__lo, __hi);
            }
            char_type tolower(char_type __c) const
            {
                return this->do_tolower(__c);
            }
            const char_type *tolower(char_type *__lo, const char_type *__hi) const
            {
                return this->do_tolower(__lo, __hi);
            }
            char_type widen(char __c) const
            {
                return this->do_widen(__c);
            }
            const char *widen(const char *__lo, const char *__hi, char_type *__to) const
            {
                return this->do_widen(__lo, __hi, __to);
            }
            char narrow(char_type __c, char __dfault) const
            {
                return this->do_narrow(__c, __dfault);
            }
            const char_type *narrow(const char_type *__lo, const char_type *__hi, char __dfault, char *__to) const
            {
                return this->do_narrow(__lo, __hi, __dfault, __to);
            }
        protected :
            explicit __ctype_abstract_base(size_t __refs = 0)
                : facet(__refs) 
            {
            }
            virtual ~__ctype_abstract_base()
            {
            }
            virtual bool do_is(mask __m, char_type __c) const  = 0;
            virtual const char_type *do_is(const char_type *__lo, const char_type *__hi, mask *__vec) const  = 0;
            virtual const char_type *do_scan_is(mask __m, const char_type *__lo, const char_type *__hi) const  = 0;
            virtual const char_type *do_scan_not(mask __m, const char_type *__lo, const char_type *__hi) const  = 0;
            virtual char_type do_toupper(char_type) const  = 0;
            virtual const char_type *do_toupper(char_type *__lo, const char_type *__hi) const  = 0;
            virtual char_type do_tolower(char_type) const  = 0;
            virtual const char_type *do_tolower(char_type *__lo, const char_type *__hi) const  = 0;
            virtual char_type do_widen(char) const  = 0;
            virtual const char *do_widen(const char *__lo, const char *__hi, char_type *__dest) const  = 0;
            virtual char do_narrow(char_type, char __dfault) const  = 0;
            virtual const char_type *do_narrow(const char_type *__lo, const char_type *__hi, char __dfault, char *__dest) const  = 0;
    };
    template<typename _CharT >
    class ctype : public __ctype_abstract_base<_CharT>
    {
        public :
            typedef _CharT char_type;
            typedef typename __ctype_abstract_base<_CharT>::mask mask;
            static locale::id id;
            explicit ctype(size_t __refs = 0)
                : __ctype_abstract_base<_CharT>(__refs) 
            {
            }
        protected :
            virtual ~ctype();
            virtual bool do_is(mask __m, char_type __c) const;
            virtual const char_type *do_is(const char_type *__lo, const char_type *__hi, mask *__vec) const;
            virtual const char_type *do_scan_is(mask __m, const char_type *__lo, const char_type *__hi) const;
            virtual const char_type *do_scan_not(mask __m, const char_type *__lo, const char_type *__hi) const;
            virtual char_type do_toupper(char_type __c) const;
            virtual const char_type *do_toupper(char_type *__lo, const char_type *__hi) const;
            virtual char_type do_tolower(char_type __c) const;
            virtual const char_type *do_tolower(char_type *__lo, const char_type *__hi) const;
            virtual char_type do_widen(char __c) const;
            virtual const char *do_widen(const char *__lo, const char *__hi, char_type *__dest) const;
            virtual char do_narrow(char_type, char __dfault) const;
            virtual const char_type *do_narrow(const char_type *__lo, const char_type *__hi, char __dfault, char *__dest) const;
    };
    template<typename _CharT >
    locale::id ctype<_CharT>::id;
    template<>
    class ctype<char> : public locale::facet, public ctype_base
    {
        public :
            typedef char char_type;
        protected :
            __c_locale _M_c_locale_ctype;
            bool _M_del;
            __to_type _M_toupper;
            __to_type _M_tolower;
            const mask *_M_table;
            mutable char _M_widen_ok;
            mutable char _M_widen[1 + static_cast<unsigned char >(- 1)];
            mutable char _M_narrow[1 + static_cast<unsigned char >(- 1)];
            mutable char _M_narrow_ok;
        public :
            static locale::id id;
            static const size_t table_size  = 1 + static_cast<unsigned char >(- 1);
            explicit ctype(const mask *__table = 0, bool __del = false, size_t __refs = 0);
            explicit ctype(__c_locale __cloc, const mask *__table = 0, bool __del = false, size_t __refs = 0);
            inline bool is(mask __m, char __c) const;
            inline const char *is(const char *__lo, const char *__hi, mask *__vec) const;
            inline const char *scan_is(mask __m, const char *__lo, const char *__hi) const;
            inline const char *scan_not(mask __m, const char *__lo, const char *__hi) const;
            char_type toupper(char_type __c) const
            {
                return this->do_toupper(__c);
            }
            const char_type *toupper(char_type *__lo, const char_type *__hi) const
            {
                return this->do_toupper(__lo, __hi);
            }
            char_type tolower(char_type __c) const
            {
                return this->do_tolower(__c);
            }
            const char_type *tolower(char_type *__lo, const char_type *__hi) const
            {
                return this->do_tolower(__lo, __hi);
            }
            char_type widen(char __c) const
            {
                if (_M_widen_ok)
                    return _M_widen[static_cast<unsigned char >(__c)];
                this->_M_widen_init();
                return this->do_widen(__c);
            }
            const char *widen(const char *__lo, const char *__hi, char_type *__to) const
            {
                if (_M_widen_ok == 1)
                {
                    __builtin_memcpy(__to, __lo, __hi - __lo);
                    return __hi;
                }
                if (!_M_widen_ok)
                    _M_widen_init();
                return this->do_widen(__lo, __hi, __to);
            }
            char narrow(char_type __c, char __dfault) const
            {
                if (_M_narrow[static_cast<unsigned char >(__c)])
                    return _M_narrow[static_cast<unsigned char >(__c)];
                const char __t = do_narrow(__c, __dfault);
                if (__t != __dfault)
                    _M_narrow[static_cast<unsigned char >(__c)] = __t;
                return __t;
            }
            const char_type *narrow(const char_type *__lo, const char_type *__hi, char __dfault, char *__to) const
            {
                if (__builtin_expect(_M_narrow_ok == 1, true))
                {
                    __builtin_memcpy(__to, __lo, __hi - __lo);
                    return __hi;
                }
                if (!_M_narrow_ok)
                    _M_narrow_init();
                return this->do_narrow(__lo, __hi, __dfault, __to);
            }
            const mask *table() const throw ()
            {
                return _M_table;
            }
            static const mask *classic_table() throw ();
        protected :
            virtual ~ctype();
            virtual char_type do_toupper(char_type) const;
            virtual const char_type *do_toupper(char_type *__lo, const char_type *__hi) const;
            virtual char_type do_tolower(char_type) const;
            virtual const char_type *do_tolower(char_type *__lo, const char_type *__hi) const;
            virtual char_type do_widen(char __c) const
            {
                return __c;
            }
            virtual const char *do_widen(const char *__lo, const char *__hi, char_type *__dest) const
            {
                __builtin_memcpy(__dest, __lo, __hi - __lo);
                return __hi;
            }
            virtual char do_narrow(char_type __c, char) const
            {
                return __c;
            }
            virtual const char_type *do_narrow(const char_type *__lo, const char_type *__hi, char, char *__dest) const
            {
                __builtin_memcpy(__dest, __lo, __hi - __lo);
                return __hi;
            }
        private :
            void _M_narrow_init() const;
            void _M_widen_init() const;
    };
    template<>
    class ctype<wchar_t> : public __ctype_abstract_base<wchar_t>
    {
        public :
            typedef wchar_t char_type;
            typedef wctype_t __wmask_type;
        protected :
            __c_locale _M_c_locale_ctype;
            bool _M_narrow_ok;
            char _M_narrow[128];
            wint_t _M_widen[1 + static_cast<unsigned char >(- 1)];
            mask _M_bit[16];
            __wmask_type _M_wmask[16];
        public :
            static locale::id id;
            explicit ctype(size_t __refs = 0);
            explicit ctype(__c_locale __cloc, size_t __refs = 0);
        protected :
            __wmask_type _M_convert_to_wmask(const mask __m) const;
            virtual ~ctype();
            virtual bool do_is(mask __m, char_type __c) const;
            virtual const char_type *do_is(const char_type *__lo, const char_type *__hi, mask *__vec) const;
            virtual const char_type *do_scan_is(mask __m, const char_type *__lo, const char_type *__hi) const;
            virtual const char_type *do_scan_not(mask __m, const char_type *__lo, const char_type *__hi) const;
            virtual char_type do_toupper(char_type) const;
            virtual const char_type *do_toupper(char_type *__lo, const char_type *__hi) const;
            virtual char_type do_tolower(char_type) const;
            virtual const char_type *do_tolower(char_type *__lo, const char_type *__hi) const;
            virtual char_type do_widen(char) const;
            virtual const char *do_widen(const char *__lo, const char *__hi, char_type *__dest) const;
            virtual char do_narrow(char_type, char __dfault) const;
            virtual const char_type *do_narrow(const char_type *__lo, const char_type *__hi, char __dfault, char *__dest) const;
            void _M_initialize_ctype();
    };
    template<typename _CharT >
    class ctype_byname : public ctype<_CharT>
    {
        public :
            typedef typename ctype<_CharT>::mask mask;
            explicit ctype_byname(const char *__s, size_t __refs = 0);
        protected :
            virtual ~ctype_byname()
            {
            }
            ;
    };
    template<>
    class ctype_byname<char> : public ctype<char>
    {
        public :
            explicit ctype_byname(const char *__s, size_t __refs = 0);
        protected :
            virtual ~ctype_byname();
    };
    template<>
    class ctype_byname<wchar_t> : public ctype<wchar_t>
    {
        public :
            explicit ctype_byname(const char *__s, size_t __refs = 0);
        protected :
            virtual ~ctype_byname();
    };
}
namespace std __attribute__((__visibility__("default"))) {
    bool ctype<char>::is(mask __m, char __c) const
    {
        return _M_table[static_cast<unsigned char >(__c)] & __m;
    }
    const char *ctype<char>::is(const char *__low, const char *__high, mask *__vec) const
    {
        while (__low < __high)
            *__vec++ = _M_table[static_cast<unsigned char >(*__low++)];
        return __high;
    }
    const char *ctype<char>::scan_is(mask __m, const char *__low, const char *__high) const
    {
        while (__low < __high && !(_M_table[static_cast<unsigned char >(*__low)] & __m))
            ++__low;
        return __low;
    }
    const char *ctype<char>::scan_not(mask __m, const char *__low, const char *__high) const
    {
        while (__low < __high && (_M_table[static_cast<unsigned char >(*__low)] & __m) != 0)
            ++__low;
        return __low;
    }
}
namespace std __attribute__((__visibility__("default"))) {
    class __num_base
    {
        public :
            enum 
            {
                _S_ominus, 
                _S_oplus, 
                _S_ox, 
                _S_oX, 
                _S_odigits, 
                _S_odigits_end = _S_odigits + 16, 
                _S_oudigits = _S_odigits_end, 
                _S_oudigits_end = _S_oudigits + 16, 
                _S_oe = _S_odigits + 14, 
                _S_oE = _S_oudigits + 14, 
                _S_oend = _S_oudigits_end
            };
            static const char *_S_atoms_out;
            static const char *_S_atoms_in;
            enum 
            {
                _S_iminus, 
                _S_iplus, 
                _S_ix, 
                _S_iX, 
                _S_izero, 
                _S_ie = _S_izero + 14, 
                _S_iE = _S_izero + 20, 
                _S_iend = 26
            };
            static void _S_format_float(const ios_base &__io, char *__fptr, char __mod);
    };
    template<typename _CharT >
    struct __numpunct_cache : public locale::facet
    {
            const char *_M_grouping;
            size_t _M_grouping_size;
            bool _M_use_grouping;
            const _CharT *_M_truename;
            size_t _M_truename_size;
            const _CharT *_M_falsename;
            size_t _M_falsename_size;
            _CharT _M_decimal_point;
            _CharT _M_thousands_sep;
            _CharT _M_atoms_out[__num_base::_S_oend];
            _CharT _M_atoms_in[__num_base::_S_iend];
            bool _M_allocated;
            __numpunct_cache(size_t __refs = 0)
                : facet(__refs), _M_grouping(__null), _M_grouping_size(0), _M_use_grouping(false), _M_truename(__null), _M_truename_size(0), _M_falsename(__null), _M_falsename_size(0), _M_decimal_point(_CharT()), _M_thousands_sep(_CharT()), _M_allocated(false) 
            {
            }
            ~__numpunct_cache();
            void _M_cache(const locale &__loc);
        private :
            __numpunct_cache &operator =(const __numpunct_cache &);
            explicit __numpunct_cache(const __numpunct_cache &);
    };
    template<typename _CharT >
    __numpunct_cache<_CharT>::~__numpunct_cache()
    {
        if (_M_allocated)
        {
            delete[] _M_grouping;
            delete[] _M_truename;
            delete[] _M_falsename;
        }
    }
    template<typename _CharT >
    class numpunct : public locale::facet
    {
        public :
            typedef _CharT char_type;
            typedef basic_string<_CharT> string_type;
            typedef __numpunct_cache<_CharT> __cache_type;
        protected :
            __cache_type *_M_data;
        public :
            static locale::id id;
            explicit numpunct(size_t __refs = 0)
                : facet(__refs), _M_data(__null) 
            {
                _M_initialize_numpunct();
            }
            explicit numpunct(__cache_type *__cache, size_t __refs = 0)
                : facet(__refs), _M_data(__cache) 
            {
                _M_initialize_numpunct();
            }
            explicit numpunct(__c_locale __cloc, size_t __refs = 0)
                : facet(__refs), _M_data(__null) 
            {
                _M_initialize_numpunct(__cloc);
            }
            char_type decimal_point() const
            {
                return this->do_decimal_point();
            }
            char_type thousands_sep() const
            {
                return this->do_thousands_sep();
            }
            string grouping() const
            {
                return this->do_grouping();
            }
            string_type truename() const
            {
                return this->do_truename();
            }
            string_type falsename() const
            {
                return this->do_falsename();
            }
        protected :
            virtual ~numpunct();
            virtual char_type do_decimal_point() const
            {
                return _M_data->_M_decimal_point;
            }
            virtual char_type do_thousands_sep() const
            {
                return _M_data->_M_thousands_sep;
            }
            virtual string do_grouping() const
            {
                return _M_data->_M_grouping;
            }
            virtual string_type do_truename() const
            {
                return _M_data->_M_truename;
            }
            virtual string_type do_falsename() const
            {
                return _M_data->_M_falsename;
            }
            void _M_initialize_numpunct(__c_locale __cloc = __null);
    };
    template<typename _CharT >
    locale::id numpunct<_CharT>::id;
    template<>
    numpunct<char>::~numpunct();
    template<>
    void numpunct<char>::_M_initialize_numpunct(__c_locale __cloc);
    template<>
    numpunct<wchar_t>::~numpunct();
    template<>
    void numpunct<wchar_t>::_M_initialize_numpunct(__c_locale __cloc);
    template<typename _CharT >
    class numpunct_byname : public numpunct<_CharT>
    {
        public :
            typedef _CharT char_type;
            typedef basic_string<_CharT> string_type;
            explicit numpunct_byname(const char *__s, size_t __refs = 0)
                : numpunct<_CharT>(__refs) 
            {
                if (__builtin_strcmp(__s, "C") != 0 && __builtin_strcmp(__s, "POSIX") != 0)
                {
                    __c_locale __tmp;
                    this->_S_create_c_locale(__tmp, __s);
                    this->_M_initialize_numpunct(__tmp);
                    this->_S_destroy_c_locale(__tmp);
                }
            }
        protected :
            virtual ~numpunct_byname()
            {
            }
    };
    template<typename _CharT, typename _InIter >
    class num_get : public locale::facet
    {
        public :
            typedef _CharT char_type;
            typedef _InIter iter_type;
            static locale::id id;
            explicit num_get(size_t __refs = 0)
                : facet(__refs) 
            {
            }
            iter_type get(iter_type __in, iter_type __end, ios_base &__io, ios_base::iostate &__err, bool &__v) const
            {
                return this->do_get(__in, __end, __io, __err, __v);
            }
            iter_type get(iter_type __in, iter_type __end, ios_base &__io, ios_base::iostate &__err, long &__v) const
            {
                return this->do_get(__in, __end, __io, __err, __v);
            }
            iter_type get(iter_type __in, iter_type __end, ios_base &__io, ios_base::iostate &__err, unsigned short &__v) const
            {
                return this->do_get(__in, __end, __io, __err, __v);
            }
            iter_type get(iter_type __in, iter_type __end, ios_base &__io, ios_base::iostate &__err, unsigned int &__v) const
            {
                return this->do_get(__in, __end, __io, __err, __v);
            }
            iter_type get(iter_type __in, iter_type __end, ios_base &__io, ios_base::iostate &__err, unsigned long &__v) const
            {
                return this->do_get(__in, __end, __io, __err, __v);
            }
            iter_type get(iter_type __in, iter_type __end, ios_base &__io, ios_base::iostate &__err, long long &__v) const
            {
                return this->do_get(__in, __end, __io, __err, __v);
            }
            iter_type get(iter_type __in, iter_type __end, ios_base &__io, ios_base::iostate &__err, unsigned long long &__v) const
            {
                return this->do_get(__in, __end, __io, __err, __v);
            }
            iter_type get(iter_type __in, iter_type __end, ios_base &__io, ios_base::iostate &__err, float &__v) const
            {
                return this->do_get(__in, __end, __io, __err, __v);
            }
            iter_type get(iter_type __in, iter_type __end, ios_base &__io, ios_base::iostate &__err, double &__v) const
            {
                return this->do_get(__in, __end, __io, __err, __v);
            }
            iter_type get(iter_type __in, iter_type __end, ios_base &__io, ios_base::iostate &__err, long double &__v) const
            {
                return this->do_get(__in, __end, __io, __err, __v);
            }
            iter_type get(iter_type __in, iter_type __end, ios_base &__io, ios_base::iostate &__err, void *&__v) const
            {
                return this->do_get(__in, __end, __io, __err, __v);
            }
        protected :
            virtual ~num_get()
            {
            }
            iter_type _M_extract_float(iter_type, iter_type, ios_base &, ios_base::iostate &, string &) const;
            template<typename _ValueT >
            iter_type _M_extract_int(iter_type, iter_type, ios_base &, ios_base::iostate &, _ValueT &) const;
            template<typename _CharT2 >
            typename __gnu_cxx::__enable_if<__is_char<_CharT2>::__value, int>::__type _M_find(const _CharT2 *, size_t __len, _CharT2 __c) const
            {
                int __ret = - 1;
                if (__len <= 10)
                {
                    if (__c >= _CharT2('0') && __c < _CharT2(_CharT2('0') + __len))
                        __ret = __c - _CharT2('0');
                }
                else
                {
                    if (__c >= _CharT2('0') && __c <= _CharT2('9'))
                        __ret = __c - _CharT2('0');
                    else
                        if (__c >= _CharT2('a') && __c <= _CharT2('f'))
                            __ret = 10 + (__c - _CharT2('a'));
                        else
                            if (__c >= _CharT2('A') && __c <= _CharT2('F'))
                                __ret = 10 + (__c - _CharT2('A'));
                }
                return __ret;
            }
            template<typename _CharT2 >
            typename __gnu_cxx::__enable_if<!__is_char<_CharT2>::__value, int>::__type _M_find(const _CharT2 *__zero, size_t __len, _CharT2 __c) const
            {
                int __ret = - 1;
                const char_type *__q = char_traits<_CharT2>::find(__zero, __len, __c);
                if (__q)
                {
                    __ret = __q - __zero;
                    if (__ret > 15)
                        __ret -= 6;
                }
                return __ret;
            }
            virtual iter_type do_get(iter_type, iter_type, ios_base &, ios_base::iostate &, bool &) const;
            virtual iter_type do_get(iter_type __beg, iter_type __end, ios_base &__io, ios_base::iostate &__err, long &__v) const
            {
                return _M_extract_int(__beg, __end, __io, __err, __v);
            }
            virtual iter_type do_get(iter_type __beg, iter_type __end, ios_base &__io, ios_base::iostate &__err, unsigned short &__v) const
            {
                return _M_extract_int(__beg, __end, __io, __err, __v);
            }
            virtual iter_type do_get(iter_type __beg, iter_type __end, ios_base &__io, ios_base::iostate &__err, unsigned int &__v) const
            {
                return _M_extract_int(__beg, __end, __io, __err, __v);
            }
            virtual iter_type do_get(iter_type __beg, iter_type __end, ios_base &__io, ios_base::iostate &__err, unsigned long &__v) const
            {
                return _M_extract_int(__beg, __end, __io, __err, __v);
            }
            virtual iter_type do_get(iter_type __beg, iter_type __end, ios_base &__io, ios_base::iostate &__err, long long &__v) const
            {
                return _M_extract_int(__beg, __end, __io, __err, __v);
            }
            virtual iter_type do_get(iter_type __beg, iter_type __end, ios_base &__io, ios_base::iostate &__err, unsigned long long &__v) const
            {
                return _M_extract_int(__beg, __end, __io, __err, __v);
            }
            virtual iter_type do_get(iter_type, iter_type, ios_base &, ios_base::iostate &__err, float &) const;
            virtual iter_type do_get(iter_type, iter_type, ios_base &, ios_base::iostate &__err, double &) const;
            virtual iter_type do_get(iter_type, iter_type, ios_base &, ios_base::iostate &__err, long double &) const;
            virtual iter_type do_get(iter_type, iter_type, ios_base &, ios_base::iostate &__err, void *&) const;
    };
    template<typename _CharT, typename _InIter >
    locale::id num_get<_CharT, _InIter>::id;
    template<typename _CharT, typename _OutIter >
    class num_put : public locale::facet
    {
        public :
            typedef _CharT char_type;
            typedef _OutIter iter_type;
            static locale::id id;
            explicit num_put(size_t __refs = 0)
                : facet(__refs) 
            {
            }
            iter_type put(iter_type __s, ios_base &__f, char_type __fill, bool __v) const
            {
                return this->do_put(__s, __f, __fill, __v);
            }
            iter_type put(iter_type __s, ios_base &__f, char_type __fill, long __v) const
            {
                return this->do_put(__s, __f, __fill, __v);
            }
            iter_type put(iter_type __s, ios_base &__f, char_type __fill, unsigned long __v) const
            {
                return this->do_put(__s, __f, __fill, __v);
            }
            iter_type put(iter_type __s, ios_base &__f, char_type __fill, long long __v) const
            {
                return this->do_put(__s, __f, __fill, __v);
            }
            iter_type put(iter_type __s, ios_base &__f, char_type __fill, unsigned long long __v) const
            {
                return this->do_put(__s, __f, __fill, __v);
            }
            iter_type put(iter_type __s, ios_base &__f, char_type __fill, double __v) const
            {
                return this->do_put(__s, __f, __fill, __v);
            }
            iter_type put(iter_type __s, ios_base &__f, char_type __fill, long double __v) const
            {
                return this->do_put(__s, __f, __fill, __v);
            }
            iter_type put(iter_type __s, ios_base &__f, char_type __fill, const void *__v) const
            {
                return this->do_put(__s, __f, __fill, __v);
            }
        protected :
            template<typename _ValueT >
            iter_type _M_insert_float(iter_type, ios_base &__io, char_type __fill, char __mod, _ValueT __v) const;
            void _M_group_float(const char *__grouping, size_t __grouping_size, char_type __sep, const char_type *__p, char_type *__new, char_type *__cs, int &__len) const;
            template<typename _ValueT >
            iter_type _M_insert_int(iter_type, ios_base &__io, char_type __fill, _ValueT __v) const;
            void _M_group_int(const char *__grouping, size_t __grouping_size, char_type __sep, ios_base &__io, char_type *__new, char_type *__cs, int &__len) const;
            void _M_pad(char_type __fill, streamsize __w, ios_base &__io, char_type *__new, const char_type *__cs, int &__len) const;
            virtual ~num_put()
            {
            }
            ;
            virtual iter_type do_put(iter_type, ios_base &, char_type __fill, bool __v) const;
            virtual iter_type do_put(iter_type __s, ios_base &__io, char_type __fill, long __v) const
            {
                return _M_insert_int(__s, __io, __fill, __v);
            }
            virtual iter_type do_put(iter_type __s, ios_base &__io, char_type __fill, unsigned long __v) const
            {
                return _M_insert_int(__s, __io, __fill, __v);
            }
            virtual iter_type do_put(iter_type __s, ios_base &__io, char_type __fill, long long __v) const
            {
                return _M_insert_int(__s, __io, __fill, __v);
            }
            virtual iter_type do_put(iter_type __s, ios_base &__io, char_type __fill, unsigned long long __v) const
            {
                return _M_insert_int(__s, __io, __fill, __v);
            }
            virtual iter_type do_put(iter_type, ios_base &, char_type __fill, double __v) const;
            virtual iter_type do_put(iter_type, ios_base &, char_type __fill, long double __v) const;
            virtual iter_type do_put(iter_type, ios_base &, char_type __fill, const void *__v) const;
    };
    template<typename _CharT, typename _OutIter >
    locale::id num_put<_CharT, _OutIter>::id;
    template<typename _CharT >
    inline bool isspace(_CharT __c, const locale &__loc)
    {
        return use_facet<ctype<_CharT> >(__loc).is(ctype_base::space, __c);
    }
    template<typename _CharT >
    inline bool isprint(_CharT __c, const locale &__loc)
    {
        return use_facet<ctype<_CharT> >(__loc).is(ctype_base::print, __c);
    }
    template<typename _CharT >
    inline bool iscntrl(_CharT __c, const locale &__loc)
    {
        return use_facet<ctype<_CharT> >(__loc).is(ctype_base::cntrl, __c);
    }
    template<typename _CharT >
    inline bool isupper(_CharT __c, const locale &__loc)
    {
        return use_facet<ctype<_CharT> >(__loc).is(ctype_base::upper, __c);
    }
    template<typename _CharT >
    inline bool islower(_CharT __c, const locale &__loc)
    {
        return use_facet<ctype<_CharT> >(__loc).is(ctype_base::lower, __c);
    }
    template<typename _CharT >
    inline bool isalpha(_CharT __c, const locale &__loc)
    {
        return use_facet<ctype<_CharT> >(__loc).is(ctype_base::alpha, __c);
    }
    template<typename _CharT >
    inline bool isdigit(_CharT __c, const locale &__loc)
    {
        return use_facet<ctype<_CharT> >(__loc).is(ctype_base::digit, __c);
    }
    template<typename _CharT >
    inline bool ispunct(_CharT __c, const locale &__loc)
    {
        return use_facet<ctype<_CharT> >(__loc).is(ctype_base::punct, __c);
    }
    template<typename _CharT >
    inline bool isxdigit(_CharT __c, const locale &__loc)
    {
        return use_facet<ctype<_CharT> >(__loc).is(ctype_base::xdigit, __c);
    }
    template<typename _CharT >
    inline bool isalnum(_CharT __c, const locale &__loc)
    {
        return use_facet<ctype<_CharT> >(__loc).is(ctype_base::alnum, __c);
    }
    template<typename _CharT >
    inline bool isgraph(_CharT __c, const locale &__loc)
    {
        return use_facet<ctype<_CharT> >(__loc).is(ctype_base::graph, __c);
    }
    template<typename _CharT >
    inline _CharT toupper(_CharT __c, const locale &__loc)
    {
        return use_facet<ctype<_CharT> >(__loc).toupper(__c);
    }
    template<typename _CharT >
    inline _CharT tolower(_CharT __c, const locale &__loc)
    {
        return use_facet<ctype<_CharT> >(__loc).tolower(__c);
    }
}
namespace std __attribute__((__visibility__("default"))) {
    template<typename _Facet >
    struct __use_cache
    {
            const _Facet *operator ()(const locale &__loc) const;
    };
    template<typename _CharT >
    struct __use_cache<__numpunct_cache<_CharT> >
    {
            const __numpunct_cache<_CharT> *operator ()(const locale &__loc) const
            {
                const size_t __i = numpunct<_CharT>::id._M_id();
                const locale::facet **__caches = __loc._M_impl->_M_caches;
                if (!__caches[__i])
                {
                    __numpunct_cache<_CharT> *__tmp = __null;
                    try
                    {
                        __tmp = new __numpunct_cache<_CharT>;
                        __tmp->_M_cache(__loc);
                    }
                    catch (...)
                    {
                        delete __tmp;
                        throw;
                    }
                    __loc._M_impl->_M_install_cache(__tmp, __i);
                }
                return static_cast<const __numpunct_cache<_CharT> * >(__caches[__i]);
            }
    };
    template<typename _CharT >
    void __numpunct_cache<_CharT>::_M_cache(const locale &__loc)
    {
        _M_allocated = true;
        const numpunct<_CharT> &__np = use_facet<numpunct<_CharT> >(__loc);
        _M_grouping_size = __np.grouping().size();
        char *__grouping = new char [_M_grouping_size];
        __np.grouping().copy(__grouping, _M_grouping_size);
        _M_grouping = __grouping;
        _M_use_grouping = (_M_grouping_size && static_cast<signed char >(_M_grouping[0]) > 0 && (_M_grouping[0] != __gnu_cxx::__numeric_traits<char>::__max));
        _M_truename_size = __np.truename().size();
        _CharT *__truename = new _CharT [_M_truename_size];
        __np.truename().copy(__truename, _M_truename_size);
        _M_truename = __truename;
        _M_falsename_size = __np.falsename().size();
        _CharT *__falsename = new _CharT [_M_falsename_size];
        __np.falsename().copy(__falsename, _M_falsename_size);
        _M_falsename = __falsename;
        _M_decimal_point = __np.decimal_point();
        _M_thousands_sep = __np.thousands_sep();
        const ctype<_CharT> &__ct = use_facet<ctype<_CharT> >(__loc);
        __ct.widen(__num_base::_S_atoms_out, __num_base::_S_atoms_out + __num_base::_S_oend, _M_atoms_out);
        __ct.widen(__num_base::_S_atoms_in, __num_base::_S_atoms_in + __num_base::_S_iend, _M_atoms_in);
    }
    bool __verify_grouping(const char *__grouping, size_t __grouping_size, const string &__grouping_tmp);
    template<typename _CharT, typename _InIter >
    _InIter num_get<_CharT, _InIter>::_M_extract_float(_InIter __beg, _InIter __end, ios_base &__io, ios_base::iostate &__err, string &__xtrc) const
    {
        typedef char_traits<_CharT> __traits_type;
        typedef __numpunct_cache<_CharT> __cache_type;
        __use_cache<__cache_type> __uc;
        const locale &__loc = __io._M_getloc();
        const __cache_type *__lc = __uc(__loc);
        const _CharT *__lit = __lc->_M_atoms_in;
        char_type __c = char_type();
        bool __testeof = __beg == __end;
        if (!__testeof)
        {
            __c = *__beg;
            const bool __plus = __c == __lit[__num_base::_S_iplus];
            if ((__plus || __c == __lit[__num_base::_S_iminus]) && !(__lc->_M_use_grouping && __c == __lc->_M_thousands_sep) && !(__c == __lc->_M_decimal_point))
            {
                __xtrc += __plus ? '+' : '-';
                if (++__beg != __end)
                    __c = *__beg;
                else
                    __testeof = true;
            }
        }
        bool __found_mantissa = false;
        int __sep_pos = 0;
        while (!__testeof)
        {
            if ((__lc->_M_use_grouping && __c == __lc->_M_thousands_sep) || __c == __lc->_M_decimal_point)
                break;
            else
                if (__c == __lit[__num_base::_S_izero])
                {
                    if (!__found_mantissa)
                    {
                        __xtrc += '0';
                        __found_mantissa = true;
                    }
                    ++__sep_pos;
                    if (++__beg != __end)
                        __c = *__beg;
                    else
                        __testeof = true;
                }
                else
                    break;
        }
        bool __found_dec = false;
        bool __found_sci = false;
        string __found_grouping;
        if (__lc->_M_use_grouping)
            __found_grouping.reserve(32);
        const char_type *__lit_zero = __lit + __num_base::_S_izero;
        if (!__lc->_M_allocated)
            while (!__testeof)
            {
                const int __digit = _M_find(__lit_zero, 10, __c);
                if (__digit != - 1)
                {
                    __xtrc += '0' + __digit;
                    __found_mantissa = true;
                }
                else
                    if (__c == __lc->_M_decimal_point && !__found_dec && !__found_sci)
                    {
                        __xtrc += '.';
                        __found_dec = true;
                    }
                    else
                        if ((__c == __lit[__num_base::_S_ie] || __c == __lit[__num_base::_S_iE]) && !__found_sci && __found_mantissa)
                        {
                            __xtrc += 'e';
                            __found_sci = true;
                            if (++__beg != __end)
                            {
                                __c = *__beg;
                                const bool __plus = __c == __lit[__num_base::_S_iplus];
                                if (__plus || __c == __lit[__num_base::_S_iminus])
                                    __xtrc += __plus ? '+' : '-';
                                else
                                    continue;
                            }
                            else
                            {
                                __testeof = true;
                                break;
                            }
                        }
                        else
                            break;
                if (++__beg != __end)
                    __c = *__beg;
                else
                    __testeof = true;
            }
        else
            while (!__testeof)
            {
                if (__lc->_M_use_grouping && __c == __lc->_M_thousands_sep)
                {
                    if (!__found_dec && !__found_sci)
                    {
                        if (__sep_pos)
                        {
                            __found_grouping += static_cast<char >(__sep_pos);
                            __sep_pos = 0;
                        }
                        else
                        {
                            __xtrc.clear();
                            break;
                        }
                    }
                    else
                        break;
                }
                else
                    if (__c == __lc->_M_decimal_point)
                    {
                        if (!__found_dec && !__found_sci)
                        {
                            if (__found_grouping.size())
                                __found_grouping += static_cast<char >(__sep_pos);
                            __xtrc += '.';
                            __found_dec = true;
                        }
                        else
                            break;
                    }
                    else
                    {
                        const char_type *__q = __traits_type::find(__lit_zero, 10, __c);
                        if (__q)
                        {
                            __xtrc += '0' + (__q - __lit_zero);
                            __found_mantissa = true;
                            ++__sep_pos;
                        }
                        else
                            if ((__c == __lit[__num_base::_S_ie] || __c == __lit[__num_base::_S_iE]) && !__found_sci && __found_mantissa)
                            {
                                if (__found_grouping.size() && !__found_dec)
                                    __found_grouping += static_cast<char >(__sep_pos);
                                __xtrc += 'e';
                                __found_sci = true;
                                if (++__beg != __end)
                                {
                                    __c = *__beg;
                                    const bool __plus = __c == __lit[__num_base::_S_iplus];
                                    if ((__plus || __c == __lit[__num_base::_S_iminus]) && !(__lc->_M_use_grouping && __c == __lc->_M_thousands_sep) && !(__c == __lc->_M_decimal_point))
                                        __xtrc += __plus ? '+' : '-';
                                    else
                                        continue;
                                }
                                else
                                {
                                    __testeof = true;
                                    break;
                                }
                            }
                            else
                                break;
                    }
                if (++__beg != __end)
                    __c = *__beg;
                else
                    __testeof = true;
            }
        if (__found_grouping.size())
        {
            if (!__found_dec && !__found_sci)
                __found_grouping += static_cast<char >(__sep_pos);
            if (!std::__verify_grouping(__lc->_M_grouping, __lc->_M_grouping_size, __found_grouping))
                __err = ios_base::failbit;
        }
        return __beg;
    }
    template<typename _CharT, typename _InIter >
    template<typename _ValueT >
    _InIter num_get<_CharT, _InIter>::_M_extract_int(_InIter __beg, _InIter __end, ios_base &__io, ios_base::iostate &__err, _ValueT &__v) const
    {
        typedef char_traits<_CharT> __traits_type;
        using __gnu_cxx::__add_unsigned;
        typedef typename __add_unsigned<_ValueT>::__type __unsigned_type;
        typedef __numpunct_cache<_CharT> __cache_type;
        __use_cache<__cache_type> __uc;
        const locale &__loc = __io._M_getloc();
        const __cache_type *__lc = __uc(__loc);
        const _CharT *__lit = __lc->_M_atoms_in;
        char_type __c = char_type();
        const ios_base::fmtflags __basefield = __io.flags() & ios_base::basefield;
        const bool __oct = __basefield == ios_base::oct;
        int __base = __oct ? 8 : (__basefield == ios_base::hex ? 16 : 10);
        bool __testeof = __beg == __end;
        bool __negative = false;
        if (!__testeof)
        {
            __c = *__beg;
            __negative = __c == __lit[__num_base::_S_iminus];
            if ((__negative || __c == __lit[__num_base::_S_iplus]) && !(__lc->_M_use_grouping && __c == __lc->_M_thousands_sep) && !(__c == __lc->_M_decimal_point))
            {
                if (++__beg != __end)
                    __c = *__beg;
                else
                    __testeof = true;
            }
        }
        bool __found_zero = false;
        int __sep_pos = 0;
        while (!__testeof)
        {
            if ((__lc->_M_use_grouping && __c == __lc->_M_thousands_sep) || __c == __lc->_M_decimal_point)
                break;
            else
                if (__c == __lit[__num_base::_S_izero] && (!__found_zero || __base == 10))
                {
                    __found_zero = true;
                    ++__sep_pos;
                    if (__basefield == 0)
                        __base = 8;
                    if (__base == 8)
                        __sep_pos = 0;
                }
                else
                    if (__found_zero && (__c == __lit[__num_base::_S_ix] || __c == __lit[__num_base::_S_iX]))
                    {
                        if (__basefield == 0)
                            __base = 16;
                        if (__base == 16)
                        {
                            __found_zero = false;
                            __sep_pos = 0;
                        }
                        else
                            break;
                    }
                    else
                        break;
            if (++__beg != __end)
            {
                __c = *__beg;
                if (!__found_zero)
                    break;
            }
            else
                __testeof = true;
        }
        const size_t __len = (__base == 16 ? __num_base::_S_iend - __num_base::_S_izero : __base);
        string __found_grouping;
        if (__lc->_M_use_grouping)
            __found_grouping.reserve(32);
        bool __testfail = false;
        bool __testoverflow = false;
        const __unsigned_type __max = (__negative && __gnu_cxx::__numeric_traits<_ValueT>::__is_signed) ? - __gnu_cxx::__numeric_traits<_ValueT>::__min : __gnu_cxx::__numeric_traits<_ValueT>::__max;
        const __unsigned_type __smax = __max / __base;
        __unsigned_type __result = 0;
        int __digit = 0;
        const char_type *__lit_zero = __lit + __num_base::_S_izero;
        if (!__lc->_M_allocated)
            while (!__testeof)
            {
                __digit = _M_find(__lit_zero, __len, __c);
                if (__digit == - 1)
                    break;
                if (__result > __smax)
                    __testoverflow = true;
                else
                {
                    __result *= __base;
                    __testoverflow |= __result > __max - __digit;
                    __result += __digit;
                    ++__sep_pos;
                }
                if (++__beg != __end)
                    __c = *__beg;
                else
                    __testeof = true;
            }
        else
            while (!__testeof)
            {
                if (__lc->_M_use_grouping && __c == __lc->_M_thousands_sep)
                {
                    if (__sep_pos)
                    {
                        __found_grouping += static_cast<char >(__sep_pos);
                        __sep_pos = 0;
                    }
                    else
                    {
                        __testfail = true;
                        break;
                    }
                }
                else
                    if (__c == __lc->_M_decimal_point)
                        break;
                    else
                    {
                        const char_type *__q = __traits_type::find(__lit_zero, __len, __c);
                        if (!__q)
                            break;
                        __digit = __q - __lit_zero;
                        if (__digit > 15)
                            __digit -= 6;
                        if (__result > __smax)
                            __testoverflow = true;
                        else
                        {
                            __result *= __base;
                            __testoverflow |= __result > __max - __digit;
                            __result += __digit;
                            ++__sep_pos;
                        }
                    }
                if (++__beg != __end)
                    __c = *__beg;
                else
                    __testeof = true;
            }
        if (__found_grouping.size())
        {
            __found_grouping += static_cast<char >(__sep_pos);
            if (!std::__verify_grouping(__lc->_M_grouping, __lc->_M_grouping_size, __found_grouping))
                __err = ios_base::failbit;
        }
        if ((!__sep_pos && !__found_zero && !__found_grouping.size()) || __testfail)
        {
            __v = 0;
            __err = ios_base::failbit;
        }
        else
            if (__testoverflow)
            {
                if (__negative && __gnu_cxx::__numeric_traits<_ValueT>::__is_signed)
                    __v = __gnu_cxx::__numeric_traits<_ValueT>::__min;
                else
                    __v = __gnu_cxx::__numeric_traits<_ValueT>::__max;
                __err = ios_base::failbit;
            }
            else
                __v = __negative ? - __result : __result;
        if (__testeof)
            __err |= ios_base::eofbit;
        return __beg;
    }
    template<typename _CharT, typename _InIter >
    _InIter num_get<_CharT, _InIter>::do_get(iter_type __beg, iter_type __end, ios_base &__io, ios_base::iostate &__err, bool &__v) const
    {
        if (!(__io.flags() & ios_base::boolalpha))
        {
            long __l = - 1;
            __beg = _M_extract_int(__beg, __end, __io, __err, __l);
            if (__l == 0 || __l == 1)
                __v = bool(__l);
            else
            {
                __v = true;
                __err = ios_base::failbit;
                if (__beg == __end)
                    __err |= ios_base::eofbit;
            }
        }
        else
        {
            typedef __numpunct_cache<_CharT> __cache_type;
            __use_cache<__cache_type> __uc;
            const locale &__loc = __io._M_getloc();
            const __cache_type *__lc = __uc(__loc);
            bool __testf = true;
            bool __testt = true;
            bool __donef = __lc->_M_falsename_size == 0;
            bool __donet = __lc->_M_truename_size == 0;
            bool __testeof = false;
            size_t __n = 0;
            while (!__donef || !__donet)
            {
                if (__beg == __end)
                {
                    __testeof = true;
                    break;
                }
                const char_type __c = *__beg;
                if (!__donef)
                    __testf = __c == __lc->_M_falsename[__n];
                if (!__testf && __donet)
                    break;
                if (!__donet)
                    __testt = __c == __lc->_M_truename[__n];
                if (!__testt && __donef)
                    break;
                if (!__testt && !__testf)
                    break;
                ++__n;
                ++__beg;
                __donef = !__testf || __n >= __lc->_M_falsename_size;
                __donet = !__testt || __n >= __lc->_M_truename_size;
            }
            if (__testf && __n == __lc->_M_falsename_size && __n)
            {
                __v = false;
                if (__testt && __n == __lc->_M_truename_size)
                    __err = ios_base::failbit;
                else
                    __err = __testeof ? ios_base::eofbit : ios_base::goodbit;
            }
            else
                if (__testt && __n == __lc->_M_truename_size && __n)
                {
                    __v = true;
                    __err = __testeof ? ios_base::eofbit : ios_base::goodbit;
                }
                else
                {
                    __v = false;
                    __err = ios_base::failbit;
                    if (__testeof)
                        __err |= ios_base::eofbit;
                }
        }
        return __beg;
    }
    template<typename _CharT, typename _InIter >
    _InIter num_get<_CharT, _InIter>::do_get(iter_type __beg, iter_type __end, ios_base &__io, ios_base::iostate &__err, float &__v) const
    {
        string __xtrc;
        __xtrc.reserve(32);
        __beg = _M_extract_float(__beg, __end, __io, __err, __xtrc);
        std::__convert_to_v(__xtrc.c_str(), __v, __err, _S_get_c_locale());
        if (__beg == __end)
            __err |= ios_base::eofbit;
        return __beg;
    }
    template<typename _CharT, typename _InIter >
    _InIter num_get<_CharT, _InIter>::do_get(iter_type __beg, iter_type __end, ios_base &__io, ios_base::iostate &__err, double &__v) const
    {
        string __xtrc;
        __xtrc.reserve(32);
        __beg = _M_extract_float(__beg, __end, __io, __err, __xtrc);
        std::__convert_to_v(__xtrc.c_str(), __v, __err, _S_get_c_locale());
        if (__beg == __end)
            __err |= ios_base::eofbit;
        return __beg;
    }
    template<typename _CharT, typename _InIter >
    _InIter num_get<_CharT, _InIter>::do_get(iter_type __beg, iter_type __end, ios_base &__io, ios_base::iostate &__err, long double &__v) const
    {
        string __xtrc;
        __xtrc.reserve(32);
        __beg = _M_extract_float(__beg, __end, __io, __err, __xtrc);
        std::__convert_to_v(__xtrc.c_str(), __v, __err, _S_get_c_locale());
        if (__beg == __end)
            __err |= ios_base::eofbit;
        return __beg;
    }
    template<typename _CharT, typename _InIter >
    _InIter num_get<_CharT, _InIter>::do_get(iter_type __beg, iter_type __end, ios_base &__io, ios_base::iostate &__err, void *&__v) const
    {
        typedef ios_base::fmtflags fmtflags;
        const fmtflags __fmt = __io.flags();
        __io.flags((__fmt & ~ios_base::basefield) | ios_base::hex);
        typedef __gnu_cxx::__conditional_type<(sizeof(void *) <= sizeof(unsigned long)), unsigned long, unsigned long long>::__type _UIntPtrType;
        _UIntPtrType __ul;
        __beg = _M_extract_int(__beg, __end, __io, __err, __ul);
        __io.flags(__fmt);
        __v = reinterpret_cast<void * >(__ul);
        return __beg;
    }
    template<typename _CharT, typename _OutIter >
    void num_put<_CharT, _OutIter>::_M_pad(_CharT __fill, streamsize __w, ios_base &__io, _CharT *__new, const _CharT *__cs, int &__len) const
    {
        __pad<_CharT, char_traits<_CharT> >::_S_pad(__io, __fill, __new, __cs, __w, __len);
        __len = static_cast<int >(__w);
    }
    template<typename _CharT, typename _ValueT >
    int __int_to_char(_CharT *__bufend, _ValueT __v, const _CharT *__lit, ios_base::fmtflags __flags, bool __dec)
    {
        _CharT *__buf = __bufend;
        if (__builtin_expect(__dec, true))
        {
            do
            {
                *--__buf = __lit[(__v % 10) + __num_base::_S_odigits];
                __v /= 10;
            }
            while (__v != 0);
        }
        else
            if ((__flags & ios_base::basefield) == ios_base::oct)
            {
                do
                {
                    *--__buf = __lit[(__v & 0x7) + __num_base::_S_odigits];
                    __v >>= 3;
                }
                while (__v != 0);
            }
            else
            {
                const bool __uppercase = __flags & ios_base::uppercase;
                const int __case_offset = __uppercase ? __num_base::_S_oudigits : __num_base::_S_odigits;
                do
                {
                    *--__buf = __lit[(__v & 0xf) + __case_offset];
                    __v >>= 4;
                }
                while (__v != 0);
            }
        return __bufend - __buf;
    }
    template<typename _CharT, typename _OutIter >
    void num_put<_CharT, _OutIter>::_M_group_int(const char *__grouping, size_t __grouping_size, _CharT __sep, ios_base &, _CharT *__new, _CharT *__cs, int &__len) const
    {
        _CharT *__p = std::__add_grouping(__new, __sep, __grouping, __grouping_size, __cs, __cs + __len);
        __len = __p - __new;
    }
    template<typename _CharT, typename _OutIter >
    template<typename _ValueT >
    _OutIter num_put<_CharT, _OutIter>::_M_insert_int(_OutIter __s, ios_base &__io, _CharT __fill, _ValueT __v) const
    {
        using __gnu_cxx::__add_unsigned;
        typedef typename __add_unsigned<_ValueT>::__type __unsigned_type;
        typedef __numpunct_cache<_CharT> __cache_type;
        __use_cache<__cache_type> __uc;
        const locale &__loc = __io._M_getloc();
        const __cache_type *__lc = __uc(__loc);
        const _CharT *__lit = __lc->_M_atoms_out;
        const ios_base::fmtflags __flags = __io.flags();
        const int __ilen = 5 * sizeof(_ValueT);
        _CharT *__cs = static_cast<_CharT * >(__builtin_alloca(sizeof(_CharT) * __ilen));
        const ios_base::fmtflags __basefield = __flags & ios_base::basefield;
        const bool __dec = (__basefield != ios_base::oct && __basefield != ios_base::hex);
        const __unsigned_type __u = ((__v > 0 || !__dec) ? __unsigned_type(__v) : - __unsigned_type(__v));
        int __len = __int_to_char(__cs + __ilen, __u, __lit, __flags, __dec);
        __cs += __ilen - __len;
        if (__lc->_M_use_grouping)
        {
            _CharT *__cs2 = static_cast<_CharT * >(__builtin_alloca(sizeof(_CharT) * (__len + 1) * 2));
            _M_group_int(__lc->_M_grouping, __lc->_M_grouping_size, __lc->_M_thousands_sep, __io, __cs2 + 2, __cs, __len);
            __cs = __cs2 + 2;
        }
        if (__builtin_expect(__dec, true))
        {
            if (__v >= 0)
            {
                if (bool(__flags & ios_base::showpos) && __gnu_cxx::__numeric_traits<_ValueT>::__is_signed)
                    *--__cs = __lit[__num_base::_S_oplus] , ++__len;
            }
            else
                *--__cs = __lit[__num_base::_S_ominus] , ++__len;
        }
        else
            if (bool(__flags & ios_base::showbase) && __v)
            {
                if (__basefield == ios_base::oct)
                    *--__cs = __lit[__num_base::_S_odigits] , ++__len;
                else
                {
                    const bool __uppercase = __flags & ios_base::uppercase;
                    *--__cs = __lit[__num_base::_S_ox + __uppercase];
                    *--__cs = __lit[__num_base::_S_odigits];
                    __len += 2;
                }
            }
        const streamsize __w = __io.width();
        if (__w > static_cast<streamsize >(__len))
        {
            _CharT *__cs3 = static_cast<_CharT * >(__builtin_alloca(sizeof(_CharT) * __w));
            _M_pad(__fill, __w, __io, __cs3, __cs, __len);
            __cs = __cs3;
        }
        __io.width(0);
        return std::__write(__s, __cs, __len);
    }
    template<typename _CharT, typename _OutIter >
    void num_put<_CharT, _OutIter>::_M_group_float(const char *__grouping, size_t __grouping_size, _CharT __sep, const _CharT *__p, _CharT *__new, _CharT *__cs, int &__len) const
    {
        const int __declen = __p ? __p - __cs : __len;
        _CharT *__p2 = std::__add_grouping(__new, __sep, __grouping, __grouping_size, __cs, __cs + __declen);
        int __newlen = __p2 - __new;
        if (__p)
        {
            char_traits<_CharT>::copy(__p2, __p, __len - __declen);
            __newlen += __len - __declen;
        }
        __len = __newlen;
    }
    template<typename _CharT, typename _OutIter >
    template<typename _ValueT >
    _OutIter num_put<_CharT, _OutIter>::_M_insert_float(_OutIter __s, ios_base &__io, _CharT __fill, char __mod, _ValueT __v) const
    {
        typedef __numpunct_cache<_CharT> __cache_type;
        __use_cache<__cache_type> __uc;
        const locale &__loc = __io._M_getloc();
        const __cache_type *__lc = __uc(__loc);
        const streamsize __prec = __io.precision() < 0 ? 6 : __io.precision();
        const int __max_digits = __gnu_cxx::__numeric_traits<_ValueT>::__digits10;
        int __len;
        char __fbuf[16];
        __num_base::_S_format_float(__io, __fbuf, __mod);
        int __cs_size = __max_digits * 3;
        char *__cs = static_cast<char * >(__builtin_alloca(__cs_size));
        __len = std::__convert_from_v(_S_get_c_locale(), __cs, __cs_size, __fbuf, __prec, __v);
        if (__len >= __cs_size)
        {
            __cs_size = __len + 1;
            __cs = static_cast<char * >(__builtin_alloca(__cs_size));
            __len = std::__convert_from_v(_S_get_c_locale(), __cs, __cs_size, __fbuf, __prec, __v);
        }
        const ctype<_CharT> &__ctype = use_facet<ctype<_CharT> >(__loc);
        _CharT *__ws = static_cast<_CharT * >(__builtin_alloca(sizeof(_CharT) * __len));
        __ctype.widen(__cs, __cs + __len, __ws);
        _CharT *__wp = 0;
        const char *__p = char_traits<char>::find(__cs, __len, '.');
        if (__p)
        {
            __wp = __ws + (__p - __cs);
            *__wp = __lc->_M_decimal_point;
        }
        if (__lc->_M_use_grouping && (__wp || __len < 3 || (__cs[1] <= '9' && __cs[2] <= '9' && __cs[1] >= '0' && __cs[2] >= '0')))
        {
            _CharT *__ws2 = static_cast<_CharT * >(__builtin_alloca(sizeof(_CharT) * __len * 2));
            streamsize __off = 0;
            if (__cs[0] == '-' || __cs[0] == '+')
            {
                __off = 1;
                __ws2[0] = __ws[0];
                __len -= 1;
            }
            _M_group_float(__lc->_M_grouping, __lc->_M_grouping_size, __lc->_M_thousands_sep, __wp, __ws2 + __off, __ws + __off, __len);
            __len += __off;
            __ws = __ws2;
        }
        const streamsize __w = __io.width();
        if (__w > static_cast<streamsize >(__len))
        {
            _CharT *__ws3 = static_cast<_CharT * >(__builtin_alloca(sizeof(_CharT) * __w));
            _M_pad(__fill, __w, __io, __ws3, __ws, __len);
            __ws = __ws3;
        }
        __io.width(0);
        return std::__write(__s, __ws, __len);
    }
    template<typename _CharT, typename _OutIter >
    _OutIter num_put<_CharT, _OutIter>::do_put(iter_type __s, ios_base &__io, char_type __fill, bool __v) const
    {
        const ios_base::fmtflags __flags = __io.flags();
        if ((__flags & ios_base::boolalpha) == 0)
        {
            const long __l = __v;
            __s = _M_insert_int(__s, __io, __fill, __l);
        }
        else
        {
            typedef __numpunct_cache<_CharT> __cache_type;
            __use_cache<__cache_type> __uc;
            const locale &__loc = __io._M_getloc();
            const __cache_type *__lc = __uc(__loc);
            const _CharT *__name = __v ? __lc->_M_truename : __lc->_M_falsename;
            int __len = __v ? __lc->_M_truename_size : __lc->_M_falsename_size;
            const streamsize __w = __io.width();
            if (__w > static_cast<streamsize >(__len))
            {
                const streamsize __plen = __w - __len;
                _CharT *__ps = static_cast<_CharT * >(__builtin_alloca(sizeof(_CharT) * __plen));
                char_traits<_CharT>::assign(__ps, __plen, __fill);
                __io.width(0);
                if ((__flags & ios_base::adjustfield) == ios_base::left)
                {
                    __s = std::__write(__s, __name, __len);
                    __s = std::__write(__s, __ps, __plen);
                }
                else
                {
                    __s = std::__write(__s, __ps, __plen);
                    __s = std::__write(__s, __name, __len);
                }
                return __s;
            }
            __io.width(0);
            __s = std::__write(__s, __name, __len);
        }
        return __s;
    }
    template<typename _CharT, typename _OutIter >
    _OutIter num_put<_CharT, _OutIter>::do_put(iter_type __s, ios_base &__io, char_type __fill, double __v) const
    {
        return _M_insert_float(__s, __io, __fill, char(), __v);
    }
    template<typename _CharT, typename _OutIter >
    _OutIter num_put<_CharT, _OutIter>::do_put(iter_type __s, ios_base &__io, char_type __fill, long double __v) const
    {
        return _M_insert_float(__s, __io, __fill, 'L', __v);
    }
    template<typename _CharT, typename _OutIter >
    _OutIter num_put<_CharT, _OutIter>::do_put(iter_type __s, ios_base &__io, char_type __fill, const void *__v) const
    {
        const ios_base::fmtflags __flags = __io.flags();
        const ios_base::fmtflags __fmt = ~(ios_base::basefield | ios_base::uppercase);
        __io.flags((__flags & __fmt) | (ios_base::hex | ios_base::showbase));
        typedef __gnu_cxx::__conditional_type<(sizeof(const void *) <= sizeof(unsigned long)), unsigned long, unsigned long long>::__type _UIntPtrType;
        __s = _M_insert_int(__s, __io, __fill, reinterpret_cast<_UIntPtrType >(__v));
        __io.flags(__flags);
        return __s;
    }
    template<typename _CharT, typename _Traits >
    void __pad<_CharT, _Traits>::_S_pad(ios_base &__io, _CharT __fill, _CharT *__news, const _CharT *__olds, streamsize __newlen, streamsize __oldlen)
    {
        const size_t __plen = static_cast<size_t >(__newlen - __oldlen);
        const ios_base::fmtflags __adjust = __io.flags() & ios_base::adjustfield;
        if (__adjust == ios_base::left)
        {
            _Traits::copy(__news, __olds, __oldlen);
            _Traits::assign(__news + __oldlen, __plen, __fill);
            return;
        }
        size_t __mod = 0;
        if (__adjust == ios_base::internal)
        {
            const locale &__loc = __io._M_getloc();
            const ctype<_CharT> &__ctype = use_facet<ctype<_CharT> >(__loc);
            if (__ctype.widen('-') == __olds[0] || __ctype.widen('+') == __olds[0])
            {
                __news[0] = __olds[0];
                __mod = 1;
                ++__news;
            }
            else
                if (__ctype.widen('0') == __olds[0] && __oldlen > 1 && (__ctype.widen('x') == __olds[1] || __ctype.widen('X') == __olds[1]))
                {
                    __news[0] = __olds[0];
                    __news[1] = __olds[1];
                    __mod = 2;
                    __news += 2;
                }
        }
        _Traits::assign(__news, __plen, __fill);
        _Traits::copy(__news + __plen, __olds + __mod, __oldlen - __mod);
    }
    template<typename _CharT >
    _CharT *__add_grouping(_CharT *__s, _CharT __sep, const char *__gbeg, size_t __gsize, const _CharT *__first, const _CharT *__last)
    {
        size_t __idx = 0;
        size_t __ctr = 0;
        while (__last - __first > __gbeg[__idx] && static_cast<signed char >(__gbeg[__idx]) > 0 && __gbeg[__idx] != __gnu_cxx::__numeric_traits<char>::__max)
        {
            __last -= __gbeg[__idx];
            __idx < __gsize - 1 ? ++__idx : ++__ctr;
        }
        while (__first != __last)
            *__s++ = *__first++;
        while (__ctr--)
        {
            *__s++ = __sep;
            for (char __i = __gbeg[__idx];
                __i > 0;
                --__i)
                *__s++ = *__first++;
        }
        while (__idx--)
        {
            *__s++ = __sep;
            for (char __i = __gbeg[__idx];
                __i > 0;
                --__i)
                *__s++ = *__first++;
        }
        return __s;
    }
    extern template class numpunct<char>;
    extern template class numpunct_byname<char>;
    extern template class num_get<char>;
    extern template class num_put<char>;
    extern template class ctype_byname<char>;
    extern template const ctype<char> &use_facet<ctype<char> >(const locale &);
    extern template const numpunct<char> &use_facet<numpunct<char> >(const locale &);
    extern template const num_put<char> &use_facet<num_put<char> >(const locale &);
    extern template const num_get<char> &use_facet<num_get<char> >(const locale &);
    extern template bool has_facet<ctype<char> >(const locale &);
    extern template bool has_facet<numpunct<char> >(const locale &);
    extern template bool has_facet<num_put<char> >(const locale &);
    extern template bool has_facet<num_get<char> >(const locale &);
    extern template class numpunct<wchar_t>;
    extern template class numpunct_byname<wchar_t>;
    extern template class num_get<wchar_t>;
    extern template class num_put<wchar_t>;
    extern template class ctype_byname<wchar_t>;
    extern template const ctype<wchar_t> &use_facet<ctype<wchar_t> >(const locale &);
    extern template const numpunct<wchar_t> &use_facet<numpunct<wchar_t> >(const locale &);
    extern template const num_put<wchar_t> &use_facet<num_put<wchar_t> >(const locale &);
    extern template const num_get<wchar_t> &use_facet<num_get<wchar_t> >(const locale &);
    extern template bool has_facet<ctype<wchar_t> >(const locale &);
    extern template bool has_facet<numpunct<wchar_t> >(const locale &);
    extern template bool has_facet<num_put<wchar_t> >(const locale &);
    extern template bool has_facet<num_get<wchar_t> >(const locale &);
}
namespace std __attribute__((__visibility__("default"))) {
    template<typename _Facet >
    inline const _Facet &__check_facet(const _Facet *__f)
    {
        if (!__f)
            __throw_bad_cast();
        return *__f;
    }
    template<typename _CharT, typename _Traits >
    class basic_ios : public ios_base
    {
        public :
            typedef _CharT char_type;
            typedef typename _Traits::int_type int_type;
            typedef typename _Traits::pos_type pos_type;
            typedef typename _Traits::off_type off_type;
            typedef _Traits traits_type;
            typedef ctype<_CharT> __ctype_type;
            typedef num_put<_CharT, ostreambuf_iterator<_CharT, _Traits> > __num_put_type;
            typedef num_get<_CharT, istreambuf_iterator<_CharT, _Traits> > __num_get_type;
        protected :
            basic_ostream<_CharT, _Traits> *_M_tie;
            mutable char_type _M_fill;
            mutable bool _M_fill_init;
            basic_streambuf<_CharT, _Traits> *_M_streambuf;
            const __ctype_type *_M_ctype;
            const __num_put_type *_M_num_put;
            const __num_get_type *_M_num_get;
        public :
            operator void *() const
            {
                return this->fail() ? 0 : const_cast<basic_ios * >(this);
            }
            bool operator !() const
            {
                return this->fail();
            }
            iostate rdstate() const
            {
                return _M_streambuf_state;
            }
            void clear(iostate __state = goodbit);
            void setstate(iostate __state)
            {
                this->clear(this->rdstate() | __state);
            }
            void _M_setstate(iostate __state)
            {
                _M_streambuf_state |= __state;
                if (this->exceptions() & __state)
                    throw;
            }
            bool good() const
            {
                return this->rdstate() == 0;
            }
            bool eof() const
            {
                return (this->rdstate() & eofbit) != 0;
            }
            bool fail() const
            {
                return (this->rdstate() & (badbit | failbit)) != 0;
            }
            bool bad() const
            {
                return (this->rdstate() & badbit) != 0;
            }
            iostate exceptions() const
            {
                return _M_exception;
            }
            void exceptions(iostate __except)
            {
                _M_exception = __except;
                this->clear(_M_streambuf_state);
            }
            explicit basic_ios(basic_streambuf<_CharT, _Traits> *__sb)
                : ios_base(), _M_tie(0), _M_fill(), _M_fill_init(false), _M_streambuf(0), _M_ctype(0), _M_num_put(0), _M_num_get(0) 
            {
                this->init(__sb);
            }
            virtual ~basic_ios()
            {
            }
            basic_ostream<_CharT, _Traits> *tie() const
            {
                return _M_tie;
            }
            basic_ostream<_CharT, _Traits> *tie(basic_ostream<_CharT, _Traits> *__tiestr)
            {
                basic_ostream<_CharT, _Traits> *__old = _M_tie;
                _M_tie = __tiestr;
                return __old;
            }
            basic_streambuf<_CharT, _Traits> *rdbuf() const
            {
                return _M_streambuf;
            }
            basic_streambuf<_CharT, _Traits> *rdbuf(basic_streambuf<_CharT, _Traits> *__sb);
            basic_ios &copyfmt(const basic_ios &__rhs);
            char_type fill() const
            {
                if (!_M_fill_init)
                {
                    _M_fill = this->widen(' ');
                    _M_fill_init = true;
                }
                return _M_fill;
            }
            char_type fill(char_type __ch)
            {
                char_type __old = this->fill();
                _M_fill = __ch;
                return __old;
            }
            locale imbue(const locale &__loc);
            char narrow(char_type __c, char __dfault) const
            {
                return __check_facet(_M_ctype).narrow(__c, __dfault);
            }
            char_type widen(char __c) const
            {
                return __check_facet(_M_ctype).widen(__c);
            }
        protected :
            basic_ios()
                : ios_base(), _M_tie(0), _M_fill(char_type()), _M_fill_init(false), _M_streambuf(0), _M_ctype(0), _M_num_put(0), _M_num_get(0) 
            {
            }
            void init(basic_streambuf<_CharT, _Traits> *__sb);
            void _M_cache_locale(const locale &__loc);
    };
}
namespace std __attribute__((__visibility__("default"))) {
    template<typename _CharT, typename _Traits >
    void basic_ios<_CharT, _Traits>::clear(iostate __state)
    {
        if (this->rdbuf())
            _M_streambuf_state = __state;
        else
            _M_streambuf_state = __state | badbit;
        if (this->exceptions() & this->rdstate())
            __throw_ios_failure(("basic_ios::clear"));
    }
    template<typename _CharT, typename _Traits >
    basic_streambuf<_CharT, _Traits> *basic_ios<_CharT, _Traits>::rdbuf(basic_streambuf<_CharT, _Traits> *__sb)
    {
        basic_streambuf<_CharT, _Traits> *__old = _M_streambuf;
        _M_streambuf = __sb;
        this->clear();
        return __old;
    }
    template<typename _CharT, typename _Traits >
    basic_ios<_CharT, _Traits> &basic_ios<_CharT, _Traits>::copyfmt(const basic_ios &__rhs)
    {
        if (this != &__rhs)
        {
            _Words *__words = (__rhs._M_word_size <= _S_local_word_size) ? _M_local_word : new _Words [__rhs._M_word_size];
            _Callback_list *__cb = __rhs._M_callbacks;
            if (__cb)
                __cb->_M_add_reference();
            _M_call_callbacks(erase_event);
            if (_M_word != _M_local_word)
            {
                delete[] _M_word;
                _M_word = 0;
            }
            _M_dispose_callbacks();
            _M_callbacks = __cb;
            for (int __i = 0;
                __i < __rhs._M_word_size;
                ++__i)
                __words[__i] = __rhs._M_word[__i];
            _M_word = __words;
            _M_word_size = __rhs._M_word_size;
            this->flags(__rhs.flags());
            this->width(__rhs.width());
            this->precision(__rhs.precision());
            this->tie(__rhs.tie());
            this->fill(__rhs.fill());
            _M_ios_locale = __rhs.getloc();
            _M_cache_locale(_M_ios_locale);
            _M_call_callbacks(copyfmt_event);
            this->exceptions(__rhs.exceptions());
        }
        return *this;
    }
    template<typename _CharT, typename _Traits >
    locale basic_ios<_CharT, _Traits>::imbue(const locale &__loc)
    {
        locale __old(this->getloc());
        ios_base::imbue(__loc);
        _M_cache_locale(__loc);
        if (this->rdbuf() != 0)
            this->rdbuf()->pubimbue(__loc);
        return __old;
    }
    template<typename _CharT, typename _Traits >
    void basic_ios<_CharT, _Traits>::init(basic_streambuf<_CharT, _Traits> *__sb)
    {
        ios_base::_M_init();
        _M_cache_locale(_M_ios_locale);
        _M_fill = _CharT();
        _M_fill_init = false;
        _M_tie = 0;
        _M_exception = goodbit;
        _M_streambuf = __sb;
        _M_streambuf_state = __sb ? goodbit : badbit;
    }
    template<typename _CharT, typename _Traits >
    void basic_ios<_CharT, _Traits>::_M_cache_locale(const locale &__loc)
    {
        if (__builtin_expect(has_facet<__ctype_type>(__loc), true))
            _M_ctype = &use_facet<__ctype_type>(__loc);
        else
            _M_ctype = 0;
        if (__builtin_expect(has_facet<__num_put_type>(__loc), true))
            _M_num_put = &use_facet<__num_put_type>(__loc);
        else
            _M_num_put = 0;
        if (__builtin_expect(has_facet<__num_get_type>(__loc), true))
            _M_num_get = &use_facet<__num_get_type>(__loc);
        else
            _M_num_get = 0;
    }
    extern template class basic_ios<char>;
    extern template class basic_ios<wchar_t>;
}
namespace std __attribute__((__visibility__("default"))) {
    template<typename _CharT, typename _Traits >
    class basic_ostream : virtual public basic_ios<_CharT, _Traits>
    {
        public :
            typedef _CharT char_type;
            typedef typename _Traits::int_type int_type;
            typedef typename _Traits::pos_type pos_type;
            typedef typename _Traits::off_type off_type;
            typedef _Traits traits_type;
            typedef basic_streambuf<_CharT, _Traits> __streambuf_type;
            typedef basic_ios<_CharT, _Traits> __ios_type;
            typedef basic_ostream<_CharT, _Traits> __ostream_type;
            typedef num_put<_CharT, ostreambuf_iterator<_CharT, _Traits> > __num_put_type;
            typedef ctype<_CharT> __ctype_type;
            explicit basic_ostream(__streambuf_type *__sb)
            {
                this->init(__sb);
            }
            virtual ~basic_ostream()
            {
            }
            class sentry;
            friend class sentry;
            __ostream_type &operator <<(__ostream_type &(*__pf)(__ostream_type &))
            {
                return __pf(*this);
            }
            __ostream_type &operator <<(__ios_type &(*__pf)(__ios_type &))
            {
                __pf(*this);
                return *this;
            }
            __ostream_type &operator <<(ios_base &(*__pf)(ios_base &))
            {
                __pf(*this);
                return *this;
            }
            __ostream_type &operator <<(long __n)
            {
                return _M_insert(__n);
            }
            __ostream_type &operator <<(unsigned long __n)
            {
                return _M_insert(__n);
            }
            __ostream_type &operator <<(bool __n)
            {
                return _M_insert(__n);
            }
            __ostream_type &operator <<(short __n);
            __ostream_type &operator <<(unsigned short __n)
            {
                return _M_insert(static_cast<unsigned long >(__n));
            }
            __ostream_type &operator <<(int __n);
            __ostream_type &operator <<(unsigned int __n)
            {
                return _M_insert(static_cast<unsigned long >(__n));
            }
            __ostream_type &operator <<(long long __n)
            {
                return _M_insert(__n);
            }
            __ostream_type &operator <<(unsigned long long __n)
            {
                return _M_insert(__n);
            }
            __ostream_type &operator <<(double __f)
            {
                return _M_insert(__f);
            }
            __ostream_type &operator <<(float __f)
            {
                return _M_insert(static_cast<double >(__f));
            }
            __ostream_type &operator <<(long double __f)
            {
                return _M_insert(__f);
            }
            __ostream_type &operator <<(const void *__p)
            {
                return _M_insert(__p);
            }
            __ostream_type &operator <<(__streambuf_type *__sb);
            __ostream_type &put(char_type __c);
            void _M_write(const char_type *__s, streamsize __n)
            {
                const streamsize __put = this->rdbuf()->sputn(__s, __n);
                if (__put != __n)
                    this->setstate(ios_base::badbit);
            }
            __ostream_type &write(const char_type *__s, streamsize __n);
            __ostream_type &flush();
            pos_type tellp();
            __ostream_type &seekp(pos_type);
            __ostream_type &seekp(off_type, ios_base::seekdir);
        protected :
            basic_ostream()
            {
                this->init(0);
            }
            template<typename _ValueT >
            __ostream_type &_M_insert(_ValueT __v);
    };
    template<typename _CharT, typename _Traits >
    class basic_ostream<_CharT, _Traits>::sentry
    {
            bool _M_ok;
            basic_ostream<_CharT, _Traits> &_M_os;
        public :
            explicit sentry(basic_ostream<_CharT, _Traits> &__os);
            ~sentry()
            {
                if (bool(_M_os.flags() & ios_base::unitbuf) && !uncaught_exception())
                {
                    if (_M_os.rdbuf() && _M_os.rdbuf()->pubsync() == - 1)
                        _M_os.setstate(ios_base::badbit);
                }
            }
            operator bool() const
            {
                return _M_ok;
            }
    };
    template<typename _CharT, typename _Traits >
    inline basic_ostream<_CharT, _Traits> &operator <<(basic_ostream<_CharT, _Traits> &__out, _CharT __c)
    {
        return __ostream_insert(__out, &__c, 1);
    }
    template<typename _CharT, typename _Traits >
    inline basic_ostream<_CharT, _Traits> &operator <<(basic_ostream<_CharT, _Traits> &__out, char __c)
    {
        return (__out << __out.widen(__c));
    }
    template<class _Traits >
    inline basic_ostream<char, _Traits> &operator <<(basic_ostream<char, _Traits> &__out, char __c)
    {
        return __ostream_insert(__out, &__c, 1);
    }
    template<class _Traits >
    inline basic_ostream<char, _Traits> &operator <<(basic_ostream<char, _Traits> &__out, signed char __c)
    {
        return (__out << static_cast<char >(__c));
    }
    template<class _Traits >
    inline basic_ostream<char, _Traits> &operator <<(basic_ostream<char, _Traits> &__out, unsigned char __c)
    {
        return (__out << static_cast<char >(__c));
    }
    template<typename _CharT, typename _Traits >
    inline basic_ostream<_CharT, _Traits> &operator <<(basic_ostream<_CharT, _Traits> &__out, const _CharT *__s)
    {
        if (!__s)
            __out.setstate(ios_base::badbit);
        else
            __ostream_insert(__out, __s, static_cast<streamsize >(_Traits::length(__s)));
        return __out;
    }
    template<typename _CharT, typename _Traits >
    basic_ostream<_CharT, _Traits> &operator <<(basic_ostream<_CharT, _Traits> &__out, const char *__s);
    template<class _Traits >
    inline basic_ostream<char, _Traits> &operator <<(basic_ostream<char, _Traits> &__out, const char *__s)
    {
        if (!__s)
            __out.setstate(ios_base::badbit);
        else
            __ostream_insert(__out, __s, static_cast<streamsize >(_Traits::length(__s)));
        return __out;
    }
    template<class _Traits >
    inline basic_ostream<char, _Traits> &operator <<(basic_ostream<char, _Traits> &__out, const signed char *__s)
    {
        return (__out << reinterpret_cast<const char * >(__s));
    }
    template<class _Traits >
    inline basic_ostream<char, _Traits> &operator <<(basic_ostream<char, _Traits> &__out, const unsigned char *__s)
    {
        return (__out << reinterpret_cast<const char * >(__s));
    }
    template<typename _CharT, typename _Traits >
    inline basic_ostream<_CharT, _Traits> &endl(basic_ostream<_CharT, _Traits> &__os)
    {
        return flush(__os.put(__os.widen('\n')));
    }
    template<typename _CharT, typename _Traits >
    inline basic_ostream<_CharT, _Traits> &ends(basic_ostream<_CharT, _Traits> &__os)
    {
        return __os.put(_CharT());
    }
    template<typename _CharT, typename _Traits >
    inline basic_ostream<_CharT, _Traits> &flush(basic_ostream<_CharT, _Traits> &__os)
    {
        return __os.flush();
    }
}
namespace std __attribute__((__visibility__("default"))) {
    template<typename _CharT, typename _Traits >
    basic_ostream<_CharT, _Traits>::sentry::sentry(basic_ostream<_CharT, _Traits> &__os)
        : _M_ok(false), _M_os(__os) 
    {
        if (__os.tie() && __os.good())
            __os.tie()->flush();
        if (__os.good())
            _M_ok = true;
        else
            __os.setstate(ios_base::failbit);
    }
    template<typename _CharT, typename _Traits >
    template<typename _ValueT >
    basic_ostream<_CharT, _Traits> &basic_ostream<_CharT, _Traits>::_M_insert(_ValueT __v)
    {
        sentry __cerb(*this);
        if (__cerb)
        {
            ios_base::iostate __err = ios_base::iostate(ios_base::goodbit);
            try
            {
                const __num_put_type &__np = __check_facet(this->_M_num_put);
                if (__np.put(*this, *this, this->fill(), __v).failed())
                    __err |= ios_base::badbit;
            }
            catch (__cxxabiv1::__forced_unwind &)
            {
                this->_M_setstate(ios_base::badbit);
                throw;
            }
            catch (...)
            {
                this->_M_setstate(ios_base::badbit);
            }
            if (__err)
                this->setstate(__err);
        }
        return *this;
    }
    template<typename _CharT, typename _Traits >
    basic_ostream<_CharT, _Traits> &basic_ostream<_CharT, _Traits>::operator <<(short __n)
    {
        const ios_base::fmtflags __fmt = this->flags() & ios_base::basefield;
        if (__fmt == ios_base::oct || __fmt == ios_base::hex)
            return _M_insert(static_cast<long >(static_cast<unsigned short >(__n)));
        else
            return _M_insert(static_cast<long >(__n));
    }
    template<typename _CharT, typename _Traits >
    basic_ostream<_CharT, _Traits> &basic_ostream<_CharT, _Traits>::operator <<(int __n)
    {
        const ios_base::fmtflags __fmt = this->flags() & ios_base::basefield;
        if (__fmt == ios_base::oct || __fmt == ios_base::hex)
            return _M_insert(static_cast<long >(static_cast<unsigned int >(__n)));
        else
            return _M_insert(static_cast<long >(__n));
    }
    template<typename _CharT, typename _Traits >
    basic_ostream<_CharT, _Traits> &basic_ostream<_CharT, _Traits>::operator <<(__streambuf_type *__sbin)
    {
        ios_base::iostate __err = ios_base::iostate(ios_base::goodbit);
        sentry __cerb(*this);
        if (__cerb && __sbin)
        {
            try
            {
                if (!__copy_streambufs(__sbin, this->rdbuf()))
                    __err |= ios_base::failbit;
            }
            catch (__cxxabiv1::__forced_unwind &)
            {
                this->_M_setstate(ios_base::badbit);
                throw;
            }
            catch (...)
            {
                this->_M_setstate(ios_base::failbit);
            }
        }
        else
            if (!__sbin)
                __err |= ios_base::badbit;
        if (__err)
            this->setstate(__err);
        return *this;
    }
    template<typename _CharT, typename _Traits >
    basic_ostream<_CharT, _Traits> &basic_ostream<_CharT, _Traits>::put(char_type __c)
    {
        sentry __cerb(*this);
        if (__cerb)
        {
            ios_base::iostate __err = ios_base::iostate(ios_base::goodbit);
            try
            {
                const int_type __put = this->rdbuf()->sputc(__c);
                if (traits_type::eq_int_type(__put, traits_type::eof()))
                    __err |= ios_base::badbit;
            }
            catch (__cxxabiv1::__forced_unwind &)
            {
                this->_M_setstate(ios_base::badbit);
                throw;
            }
            catch (...)
            {
                this->_M_setstate(ios_base::badbit);
            }
            if (__err)
                this->setstate(__err);
        }
        return *this;
    }
    template<typename _CharT, typename _Traits >
    basic_ostream<_CharT, _Traits> &basic_ostream<_CharT, _Traits>::write(const _CharT *__s, streamsize __n)
    {
        sentry __cerb(*this);
        if (__cerb)
        {
            try
            {
                _M_write(__s, __n);
            }
            catch (__cxxabiv1::__forced_unwind &)
            {
                this->_M_setstate(ios_base::badbit);
                throw;
            }
            catch (...)
            {
                this->_M_setstate(ios_base::badbit);
            }
        }
        return *this;
    }
    template<typename _CharT, typename _Traits >
    basic_ostream<_CharT, _Traits> &basic_ostream<_CharT, _Traits>::flush()
    {
        ios_base::iostate __err = ios_base::iostate(ios_base::goodbit);
        try
        {
            if (this->rdbuf() && this->rdbuf()->pubsync() == - 1)
                __err |= ios_base::badbit;
        }
        catch (__cxxabiv1::__forced_unwind &)
        {
            this->_M_setstate(ios_base::badbit);
            throw;
        }
        catch (...)
        {
            this->_M_setstate(ios_base::badbit);
        }
        if (__err)
            this->setstate(__err);
        return *this;
    }
    template<typename _CharT, typename _Traits >
    typename basic_ostream<_CharT, _Traits>::pos_type basic_ostream<_CharT, _Traits>::tellp()
    {
        pos_type __ret = pos_type(- 1);
        try
        {
            if (!this->fail())
                __ret = this->rdbuf()->pubseekoff(0, ios_base::cur, ios_base::out);
        }
        catch (__cxxabiv1::__forced_unwind &)
        {
            this->_M_setstate(ios_base::badbit);
            throw;
        }
        catch (...)
        {
            this->_M_setstate(ios_base::badbit);
        }
        return __ret;
    }
    template<typename _CharT, typename _Traits >
    basic_ostream<_CharT, _Traits> &basic_ostream<_CharT, _Traits>::seekp(pos_type __pos)
    {
        ios_base::iostate __err = ios_base::iostate(ios_base::goodbit);
        try
        {
            if (!this->fail())
            {
                const pos_type __p = this->rdbuf()->pubseekpos(__pos, ios_base::out);
                if (__p == pos_type(off_type(- 1)))
                    __err |= ios_base::failbit;
            }
        }
        catch (__cxxabiv1::__forced_unwind &)
        {
            this->_M_setstate(ios_base::badbit);
            throw;
        }
        catch (...)
        {
            this->_M_setstate(ios_base::badbit);
        }
        if (__err)
            this->setstate(__err);
        return *this;
    }
    template<typename _CharT, typename _Traits >
    basic_ostream<_CharT, _Traits> &basic_ostream<_CharT, _Traits>::seekp(off_type __off, ios_base::seekdir __dir)
    {
        ios_base::iostate __err = ios_base::iostate(ios_base::goodbit);
        try
        {
            if (!this->fail())
            {
                const pos_type __p = this->rdbuf()->pubseekoff(__off, __dir, ios_base::out);
                if (__p == pos_type(off_type(- 1)))
                    __err |= ios_base::failbit;
            }
        }
        catch (__cxxabiv1::__forced_unwind &)
        {
            this->_M_setstate(ios_base::badbit);
            throw;
        }
        catch (...)
        {
            this->_M_setstate(ios_base::badbit);
        }
        if (__err)
            this->setstate(__err);
        return *this;
    }
    template<typename _CharT, typename _Traits >
    basic_ostream<_CharT, _Traits> &operator <<(basic_ostream<_CharT, _Traits> &__out, const char *__s)
    {
        if (!__s)
            __out.setstate(ios_base::badbit);
        else
        {
            const size_t __clen = char_traits<char>::length(__s);
            try
            {
                struct __ptr_guard
                {
                        _CharT *__p;
                        __ptr_guard(_CharT *__ip)
                            : __p(__ip) 
                        {
                        }
                        ~__ptr_guard()
                        {
                            delete[] __p;
                        }
                        _CharT *__get()
                        {
                            return __p;
                        }
                } __pg(new _CharT [__clen]);
                _CharT *__ws = __pg.__get();
                for (size_t __i = 0;
                    __i < __clen;
                    ++__i)
                    __ws[__i] = __out.widen(__s[__i]);
                __ostream_insert(__out, __ws, __clen);
            }
            catch (__cxxabiv1::__forced_unwind &)
            {
                __out._M_setstate(ios_base::badbit);
                throw;
            }
            catch (...)
            {
                __out._M_setstate(ios_base::badbit);
            }
        }
        return __out;
    }
    extern template class basic_ostream<char>;
    extern template ostream &endl(ostream &);
    extern template ostream &ends(ostream &);
    extern template ostream &flush(ostream &);
    extern template ostream &operator <<(ostream &, char);
    extern template ostream &operator <<(ostream &, unsigned char);
    extern template ostream &operator <<(ostream &, signed char);
    extern template ostream &operator <<(ostream &, const char *);
    extern template ostream &operator <<(ostream &, const unsigned char *);
    extern template ostream &operator <<(ostream &, const signed char *);
    extern template ostream &ostream::_M_insert(long);
    extern template ostream &ostream::_M_insert(unsigned long);
    extern template ostream &ostream::_M_insert(bool);
    extern template ostream &ostream::_M_insert(long long);
    extern template ostream &ostream::_M_insert(unsigned long long);
    extern template ostream &ostream::_M_insert(double);
    extern template ostream &ostream::_M_insert(long double);
    extern template ostream &ostream::_M_insert(const void *);
    extern template class basic_ostream<wchar_t>;
    extern template wostream &endl(wostream &);
    extern template wostream &ends(wostream &);
    extern template wostream &flush(wostream &);
    extern template wostream &operator <<(wostream &, wchar_t);
    extern template wostream &operator <<(wostream &, char);
    extern template wostream &operator <<(wostream &, const wchar_t *);
    extern template wostream &operator <<(wostream &, const char *);
    extern template wostream &wostream::_M_insert(long);
    extern template wostream &wostream::_M_insert(unsigned long);
    extern template wostream &wostream::_M_insert(bool);
    extern template wostream &wostream::_M_insert(long long);
    extern template wostream &wostream::_M_insert(unsigned long long);
    extern template wostream &wostream::_M_insert(double);
    extern template wostream &wostream::_M_insert(long double);
    extern template wostream &wostream::_M_insert(const void *);
}
namespace std __attribute__((__visibility__("default"))) {
    template<typename _CharT, typename _Traits >
    class basic_istream : virtual public basic_ios<_CharT, _Traits>
    {
        public :
            typedef _CharT char_type;
            typedef typename _Traits::int_type int_type;
            typedef typename _Traits::pos_type pos_type;
            typedef typename _Traits::off_type off_type;
            typedef _Traits traits_type;
            typedef basic_streambuf<_CharT, _Traits> __streambuf_type;
            typedef basic_ios<_CharT, _Traits> __ios_type;
            typedef basic_istream<_CharT, _Traits> __istream_type;
            typedef num_get<_CharT, istreambuf_iterator<_CharT, _Traits> > __num_get_type;
            typedef ctype<_CharT> __ctype_type;
        protected :
            streamsize _M_gcount;
        public :
            explicit basic_istream(__streambuf_type *__sb)
                : _M_gcount(streamsize(0)) 
            {
                this->init(__sb);
            }
            virtual ~basic_istream()
            {
                _M_gcount = streamsize(0);
            }
            class sentry;
            friend class sentry;
            __istream_type &operator >>(__istream_type &(*__pf)(__istream_type &))
            {
                return __pf(*this);
            }
            __istream_type &operator >>(__ios_type &(*__pf)(__ios_type &))
            {
                __pf(*this);
                return *this;
            }
            __istream_type &operator >>(ios_base &(*__pf)(ios_base &))
            {
                __pf(*this);
                return *this;
            }
            __istream_type &operator >>(bool &__n)
            {
                return _M_extract(__n);
            }
            __istream_type &operator >>(short &__n);
            __istream_type &operator >>(unsigned short &__n)
            {
                return _M_extract(__n);
            }
            __istream_type &operator >>(int &__n);
            __istream_type &operator >>(unsigned int &__n)
            {
                return _M_extract(__n);
            }
            __istream_type &operator >>(long &__n)
            {
                return _M_extract(__n);
            }
            __istream_type &operator >>(unsigned long &__n)
            {
                return _M_extract(__n);
            }
            __istream_type &operator >>(long long &__n)
            {
                return _M_extract(__n);
            }
            __istream_type &operator >>(unsigned long long &__n)
            {
                return _M_extract(__n);
            }
            __istream_type &operator >>(float &__f)
            {
                return _M_extract(__f);
            }
            __istream_type &operator >>(double &__f)
            {
                return _M_extract(__f);
            }
            __istream_type &operator >>(long double &__f)
            {
                return _M_extract(__f);
            }
            __istream_type &operator >>(void *&__p)
            {
                return _M_extract(__p);
            }
            __istream_type &operator >>(__streambuf_type *__sb);
            streamsize gcount() const
            {
                return _M_gcount;
            }
            int_type get();
            __istream_type &get(char_type &__c);
            __istream_type &get(char_type *__s, streamsize __n, char_type __delim);
            __istream_type &get(char_type *__s, streamsize __n)
            {
                return this->get(__s, __n, this->widen('\n'));
            }
            __istream_type &get(__streambuf_type &__sb, char_type __delim);
            __istream_type &get(__streambuf_type &__sb)
            {
                return this->get(__sb, this->widen('\n'));
            }
            __istream_type &getline(char_type *__s, streamsize __n, char_type __delim);
            __istream_type &getline(char_type *__s, streamsize __n)
            {
                return this->getline(__s, __n, this->widen('\n'));
            }
            __istream_type &ignore();
            __istream_type &ignore(streamsize __n);
            __istream_type &ignore(streamsize __n, int_type __delim);
            int_type peek();
            __istream_type &read(char_type *__s, streamsize __n);
            streamsize readsome(char_type *__s, streamsize __n);
            __istream_type &putback(char_type __c);
            __istream_type &unget();
            int sync();
            pos_type tellg();
            __istream_type &seekg(pos_type);
            __istream_type &seekg(off_type, ios_base::seekdir);
        protected :
            basic_istream()
                : _M_gcount(streamsize(0)) 
            {
                this->init(0);
            }
            template<typename _ValueT >
            __istream_type &_M_extract(_ValueT &__v);
    };
    template<>
    basic_istream<char> &basic_istream<char>::getline(char_type *__s, streamsize __n, char_type __delim);
    template<>
    basic_istream<char> &basic_istream<char>::ignore(streamsize __n);
    template<>
    basic_istream<char> &basic_istream<char>::ignore(streamsize __n, int_type __delim);
    template<>
    basic_istream<wchar_t> &basic_istream<wchar_t>::getline(char_type *__s, streamsize __n, char_type __delim);
    template<>
    basic_istream<wchar_t> &basic_istream<wchar_t>::ignore(streamsize __n);
    template<>
    basic_istream<wchar_t> &basic_istream<wchar_t>::ignore(streamsize __n, int_type __delim);
    template<typename _CharT, typename _Traits >
    class basic_istream<_CharT, _Traits>::sentry
    {
        public :
            typedef _Traits traits_type;
            typedef basic_streambuf<_CharT, _Traits> __streambuf_type;
            typedef basic_istream<_CharT, _Traits> __istream_type;
            typedef typename __istream_type::__ctype_type __ctype_type;
            typedef typename _Traits::int_type __int_type;
            explicit sentry(basic_istream<_CharT, _Traits> &__is, bool __noskipws = false);
            operator bool() const
            {
                return _M_ok;
            }
        private :
            bool _M_ok;
    };
    template<typename _CharT, typename _Traits >
    basic_istream<_CharT, _Traits> &operator >>(basic_istream<_CharT, _Traits> &__in, _CharT &__c);
    template<class _Traits >
    inline basic_istream<char, _Traits> &operator >>(basic_istream<char, _Traits> &__in, unsigned char &__c)
    {
        return (__in >> reinterpret_cast<char & >(__c));
    }
    template<class _Traits >
    inline basic_istream<char, _Traits> &operator >>(basic_istream<char, _Traits> &__in, signed char &__c)
    {
        return (__in >> reinterpret_cast<char & >(__c));
    }
    template<typename _CharT, typename _Traits >
    basic_istream<_CharT, _Traits> &operator >>(basic_istream<_CharT, _Traits> &__in, _CharT *__s);
    template<>
    basic_istream<char> &operator >>(basic_istream<char> &__in, char *__s);
    template<class _Traits >
    inline basic_istream<char, _Traits> &operator >>(basic_istream<char, _Traits> &__in, unsigned char *__s)
    {
        return (__in >> reinterpret_cast<char * >(__s));
    }
    template<class _Traits >
    inline basic_istream<char, _Traits> &operator >>(basic_istream<char, _Traits> &__in, signed char *__s)
    {
        return (__in >> reinterpret_cast<char * >(__s));
    }
    template<typename _CharT, typename _Traits >
    class basic_iostream : public basic_istream<_CharT, _Traits>, public basic_ostream<_CharT, _Traits>
    {
        public :
            typedef _CharT char_type;
            typedef typename _Traits::int_type int_type;
            typedef typename _Traits::pos_type pos_type;
            typedef typename _Traits::off_type off_type;
            typedef _Traits traits_type;
            typedef basic_istream<_CharT, _Traits> __istream_type;
            typedef basic_ostream<_CharT, _Traits> __ostream_type;
            explicit basic_iostream(basic_streambuf<_CharT, _Traits> *__sb)
                : __istream_type(__sb), __ostream_type(__sb) 
            {
            }
            virtual ~basic_iostream()
            {
            }
        protected :
            basic_iostream()
                : __istream_type(), __ostream_type() 
            {
            }
    };
    template<typename _CharT, typename _Traits >
    basic_istream<_CharT, _Traits> &ws(basic_istream<_CharT, _Traits> &__is);
}
namespace std __attribute__((__visibility__("default"))) {
    template<typename _CharT, typename _Traits >
    basic_istream<_CharT, _Traits>::sentry::sentry(basic_istream<_CharT, _Traits> &__in, bool __noskip)
        : _M_ok(false) 
    {
        ios_base::iostate __err = ios_base::iostate(ios_base::goodbit);
        if (__in.good())
        {
            if (__in.tie())
                __in.tie()->flush();
            if (!__noskip && bool(__in.flags() & ios_base::skipws))
            {
                const __int_type __eof = traits_type::eof();
                __streambuf_type *__sb = __in.rdbuf();
                __int_type __c = __sb->sgetc();
                const __ctype_type &__ct = __check_facet(__in._M_ctype);
                while (!traits_type::eq_int_type(__c, __eof) && __ct.is(ctype_base::space, traits_type::to_char_type(__c)))
                    __c = __sb->snextc();
                if (traits_type::eq_int_type(__c, __eof))
                    __err |= ios_base::eofbit;
            }
        }
        if (__in.good() && __err == ios_base::goodbit)
            _M_ok = true;
        else
        {
            __err |= ios_base::failbit;
            __in.setstate(__err);
        }
    }
    template<typename _CharT, typename _Traits >
    template<typename _ValueT >
    basic_istream<_CharT, _Traits> &basic_istream<_CharT, _Traits>::_M_extract(_ValueT &__v)
    {
        sentry __cerb(*this, false);
        if (__cerb)
        {
            ios_base::iostate __err = ios_base::iostate(ios_base::goodbit);
            try
            {
                const __num_get_type &__ng = __check_facet(this->_M_num_get);
                __ng.get(*this, 0, *this, __err, __v);
            }
            catch (__cxxabiv1::__forced_unwind &)
            {
                this->_M_setstate(ios_base::badbit);
                throw;
            }
            catch (...)
            {
                this->_M_setstate(ios_base::badbit);
            }
            if (__err)
                this->setstate(__err);
        }
        return *this;
    }
    template<typename _CharT, typename _Traits >
    basic_istream<_CharT, _Traits> &basic_istream<_CharT, _Traits>::operator >>(short &__n)
    {
        long __l;
        _M_extract(__l);
        if (!this->fail())
        {
            if (__gnu_cxx::__numeric_traits<short>::__min <= __l && __l <= __gnu_cxx::__numeric_traits<short>::__max)
                __n = short(__l);
            else
                this->setstate(ios_base::failbit);
        }
        return *this;
    }
    template<typename _CharT, typename _Traits >
    basic_istream<_CharT, _Traits> &basic_istream<_CharT, _Traits>::operator >>(int &__n)
    {
        long __l;
        _M_extract(__l);
        if (!this->fail())
        {
            if (__gnu_cxx::__numeric_traits<int>::__min <= __l && __l <= __gnu_cxx::__numeric_traits<int>::__max)
                __n = int(__l);
            else
                this->setstate(ios_base::failbit);
        }
        return *this;
    }
    template<typename _CharT, typename _Traits >
    basic_istream<_CharT, _Traits> &basic_istream<_CharT, _Traits>::operator >>(__streambuf_type *__sbout)
    {
        ios_base::iostate __err = ios_base::iostate(ios_base::goodbit);
        sentry __cerb(*this, false);
        if (__cerb && __sbout)
        {
            try
            {
                bool __ineof;
                if (!__copy_streambufs_eof(this->rdbuf(), __sbout, __ineof))
                    __err |= ios_base::failbit;
                if (__ineof)
                    __err |= ios_base::eofbit;
            }
            catch (__cxxabiv1::__forced_unwind &)
            {
                this->_M_setstate(ios_base::failbit);
                throw;
            }
            catch (...)
            {
                this->_M_setstate(ios_base::failbit);
            }
        }
        else
            if (!__sbout)
                __err |= ios_base::failbit;
        if (__err)
            this->setstate(__err);
        return *this;
    }
    template<typename _CharT, typename _Traits >
    typename basic_istream<_CharT, _Traits>::int_type basic_istream<_CharT, _Traits>::get(void)
    {
        const int_type __eof = traits_type::eof();
        int_type __c = __eof;
        _M_gcount = 0;
        ios_base::iostate __err = ios_base::iostate(ios_base::goodbit);
        sentry __cerb(*this, true);
        if (__cerb)
        {
            try
            {
                __c = this->rdbuf()->sbumpc();
                if (!traits_type::eq_int_type(__c, __eof))
                    _M_gcount = 1;
                else
                    __err |= ios_base::eofbit;
            }
            catch (__cxxabiv1::__forced_unwind &)
            {
                this->_M_setstate(ios_base::badbit);
                throw;
            }
            catch (...)
            {
                this->_M_setstate(ios_base::badbit);
            }
        }
        if (!_M_gcount)
            __err |= ios_base::failbit;
        if (__err)
            this->setstate(__err);
        return __c;
    }
    template<typename _CharT, typename _Traits >
    basic_istream<_CharT, _Traits> &basic_istream<_CharT, _Traits>::get(char_type &__c)
    {
        _M_gcount = 0;
        ios_base::iostate __err = ios_base::iostate(ios_base::goodbit);
        sentry __cerb(*this, true);
        if (__cerb)
        {
            try
            {
                const int_type __cb = this->rdbuf()->sbumpc();
                if (!traits_type::eq_int_type(__cb, traits_type::eof()))
                {
                    _M_gcount = 1;
                    __c = traits_type::to_char_type(__cb);
                }
                else
                    __err |= ios_base::eofbit;
            }
            catch (__cxxabiv1::__forced_unwind &)
            {
                this->_M_setstate(ios_base::badbit);
                throw;
            }
            catch (...)
            {
                this->_M_setstate(ios_base::badbit);
            }
        }
        if (!_M_gcount)
            __err |= ios_base::failbit;
        if (__err)
            this->setstate(__err);
        return *this;
    }
    template<typename _CharT, typename _Traits >
    basic_istream<_CharT, _Traits> &basic_istream<_CharT, _Traits>::get(char_type *__s, streamsize __n, char_type __delim)
    {
        _M_gcount = 0;
        ios_base::iostate __err = ios_base::iostate(ios_base::goodbit);
        sentry __cerb(*this, true);
        if (__cerb)
        {
            try
            {
                const int_type __idelim = traits_type::to_int_type(__delim);
                const int_type __eof = traits_type::eof();
                __streambuf_type *__sb = this->rdbuf();
                int_type __c = __sb->sgetc();
                while (_M_gcount + 1 < __n && !traits_type::eq_int_type(__c, __eof) && !traits_type::eq_int_type(__c, __idelim))
                {
                    *__s++ = traits_type::to_char_type(__c);
                    ++_M_gcount;
                    __c = __sb->snextc();
                }
                if (traits_type::eq_int_type(__c, __eof))
                    __err |= ios_base::eofbit;
            }
            catch (__cxxabiv1::__forced_unwind &)
            {
                this->_M_setstate(ios_base::badbit);
                throw;
            }
            catch (...)
            {
                this->_M_setstate(ios_base::badbit);
            }
        }
        if (__n > 0)
            *__s = char_type();
        if (!_M_gcount)
            __err |= ios_base::failbit;
        if (__err)
            this->setstate(__err);
        return *this;
    }
    template<typename _CharT, typename _Traits >
    basic_istream<_CharT, _Traits> &basic_istream<_CharT, _Traits>::get(__streambuf_type &__sb, char_type __delim)
    {
        _M_gcount = 0;
        ios_base::iostate __err = ios_base::iostate(ios_base::goodbit);
        sentry __cerb(*this, true);
        if (__cerb)
        {
            try
            {
                const int_type __idelim = traits_type::to_int_type(__delim);
                const int_type __eof = traits_type::eof();
                __streambuf_type *__this_sb = this->rdbuf();
                int_type __c = __this_sb->sgetc();
                char_type __c2 = traits_type::to_char_type(__c);
                while (!traits_type::eq_int_type(__c, __eof) && !traits_type::eq_int_type(__c, __idelim) && !traits_type::eq_int_type(__sb.sputc(__c2), __eof))
                {
                    ++_M_gcount;
                    __c = __this_sb->snextc();
                    __c2 = traits_type::to_char_type(__c);
                }
                if (traits_type::eq_int_type(__c, __eof))
                    __err |= ios_base::eofbit;
            }
            catch (__cxxabiv1::__forced_unwind &)
            {
                this->_M_setstate(ios_base::badbit);
                throw;
            }
            catch (...)
            {
                this->_M_setstate(ios_base::badbit);
            }
        }
        if (!_M_gcount)
            __err |= ios_base::failbit;
        if (__err)
            this->setstate(__err);
        return *this;
    }
    template<typename _CharT, typename _Traits >
    basic_istream<_CharT, _Traits> &basic_istream<_CharT, _Traits>::getline(char_type *__s, streamsize __n, char_type __delim)
    {
        _M_gcount = 0;
        ios_base::iostate __err = ios_base::iostate(ios_base::goodbit);
        sentry __cerb(*this, true);
        if (__cerb)
        {
            try
            {
                const int_type __idelim = traits_type::to_int_type(__delim);
                const int_type __eof = traits_type::eof();
                __streambuf_type *__sb = this->rdbuf();
                int_type __c = __sb->sgetc();
                while (_M_gcount + 1 < __n && !traits_type::eq_int_type(__c, __eof) && !traits_type::eq_int_type(__c, __idelim))
                {
                    *__s++ = traits_type::to_char_type(__c);
                    __c = __sb->snextc();
                    ++_M_gcount;
                }
                if (traits_type::eq_int_type(__c, __eof))
                    __err |= ios_base::eofbit;
                else
                {
                    if (traits_type::eq_int_type(__c, __idelim))
                    {
                        __sb->sbumpc();
                        ++_M_gcount;
                    }
                    else
                        __err |= ios_base::failbit;
                }
            }
            catch (__cxxabiv1::__forced_unwind &)
            {
                this->_M_setstate(ios_base::badbit);
                throw;
            }
            catch (...)
            {
                this->_M_setstate(ios_base::badbit);
            }
        }
        if (__n > 0)
            *__s = char_type();
        if (!_M_gcount)
            __err |= ios_base::failbit;
        if (__err)
            this->setstate(__err);
        return *this;
    }
    template<typename _CharT, typename _Traits >
    basic_istream<_CharT, _Traits> &basic_istream<_CharT, _Traits>::ignore(void)
    {
        _M_gcount = 0;
        sentry __cerb(*this, true);
        if (__cerb)
        {
            ios_base::iostate __err = ios_base::iostate(ios_base::goodbit);
            try
            {
                const int_type __eof = traits_type::eof();
                __streambuf_type *__sb = this->rdbuf();
                if (traits_type::eq_int_type(__sb->sbumpc(), __eof))
                    __err |= ios_base::eofbit;
                else
                    _M_gcount = 1;
            }
            catch (__cxxabiv1::__forced_unwind &)
            {
                this->_M_setstate(ios_base::badbit);
                throw;
            }
            catch (...)
            {
                this->_M_setstate(ios_base::badbit);
            }
            if (__err)
                this->setstate(__err);
        }
        return *this;
    }
    template<typename _CharT, typename _Traits >
    basic_istream<_CharT, _Traits> &basic_istream<_CharT, _Traits>::ignore(streamsize __n)
    {
        _M_gcount = 0;
        sentry __cerb(*this, true);
        if (__cerb && __n > 0)
        {
            ios_base::iostate __err = ios_base::iostate(ios_base::goodbit);
            try
            {
                const int_type __eof = traits_type::eof();
                __streambuf_type *__sb = this->rdbuf();
                int_type __c = __sb->sgetc();
                bool __large_ignore = false;
                while (true)
                {
                    while (_M_gcount < __n && !traits_type::eq_int_type(__c, __eof))
                    {
                        ++_M_gcount;
                        __c = __sb->snextc();
                    }
                    if (__n == __gnu_cxx::__numeric_traits<streamsize>::__max && !traits_type::eq_int_type(__c, __eof))
                    {
                        _M_gcount = __gnu_cxx::__numeric_traits<streamsize>::__min;
                        __large_ignore = true;
                    }
                    else
                        break;
                }
                if (__large_ignore)
                    _M_gcount = __gnu_cxx::__numeric_traits<streamsize>::__max;
                if (traits_type::eq_int_type(__c, __eof))
                    __err |= ios_base::eofbit;
            }
            catch (__cxxabiv1::__forced_unwind &)
            {
                this->_M_setstate(ios_base::badbit);
                throw;
            }
            catch (...)
            {
                this->_M_setstate(ios_base::badbit);
            }
            if (__err)
                this->setstate(__err);
        }
        return *this;
    }
    template<typename _CharT, typename _Traits >
    basic_istream<_CharT, _Traits> &basic_istream<_CharT, _Traits>::ignore(streamsize __n, int_type __delim)
    {
        _M_gcount = 0;
        sentry __cerb(*this, true);
        if (__cerb && __n > 0)
        {
            ios_base::iostate __err = ios_base::iostate(ios_base::goodbit);
            try
            {
                const int_type __eof = traits_type::eof();
                __streambuf_type *__sb = this->rdbuf();
                int_type __c = __sb->sgetc();
                bool __large_ignore = false;
                while (true)
                {
                    while (_M_gcount < __n && !traits_type::eq_int_type(__c, __eof) && !traits_type::eq_int_type(__c, __delim))
                    {
                        ++_M_gcount;
                        __c = __sb->snextc();
                    }
                    if (__n == __gnu_cxx::__numeric_traits<streamsize>::__max && !traits_type::eq_int_type(__c, __eof) && !traits_type::eq_int_type(__c, __delim))
                    {
                        _M_gcount = __gnu_cxx::__numeric_traits<streamsize>::__min;
                        __large_ignore = true;
                    }
                    else
                        break;
                }
                if (__large_ignore)
                    _M_gcount = __gnu_cxx::__numeric_traits<streamsize>::__max;
                if (traits_type::eq_int_type(__c, __eof))
                    __err |= ios_base::eofbit;
                else
                    if (traits_type::eq_int_type(__c, __delim))
                    {
                        if (_M_gcount < __gnu_cxx::__numeric_traits<streamsize>::__max)
                            ++_M_gcount;
                        __sb->sbumpc();
                    }
            }
            catch (__cxxabiv1::__forced_unwind &)
            {
                this->_M_setstate(ios_base::badbit);
                throw;
            }
            catch (...)
            {
                this->_M_setstate(ios_base::badbit);
            }
            if (__err)
                this->setstate(__err);
        }
        return *this;
    }
    template<typename _CharT, typename _Traits >
    typename basic_istream<_CharT, _Traits>::int_type basic_istream<_CharT, _Traits>::peek(void)
    {
        int_type __c = traits_type::eof();
        _M_gcount = 0;
        sentry __cerb(*this, true);
        if (__cerb)
        {
            ios_base::iostate __err = ios_base::iostate(ios_base::goodbit);
            try
            {
                __c = this->rdbuf()->sgetc();
                if (traits_type::eq_int_type(__c, traits_type::eof()))
                    __err |= ios_base::eofbit;
            }
            catch (__cxxabiv1::__forced_unwind &)
            {
                this->_M_setstate(ios_base::badbit);
                throw;
            }
            catch (...)
            {
                this->_M_setstate(ios_base::badbit);
            }
            if (__err)
                this->setstate(__err);
        }
        return __c;
    }
    template<typename _CharT, typename _Traits >
    basic_istream<_CharT, _Traits> &basic_istream<_CharT, _Traits>::read(char_type *__s, streamsize __n)
    {
        _M_gcount = 0;
        sentry __cerb(*this, true);
        if (__cerb)
        {
            ios_base::iostate __err = ios_base::iostate(ios_base::goodbit);
            try
            {
                _M_gcount = this->rdbuf()->sgetn(__s, __n);
                if (_M_gcount != __n)
                    __err |= (ios_base::eofbit | ios_base::failbit);
            }
            catch (__cxxabiv1::__forced_unwind &)
            {
                this->_M_setstate(ios_base::badbit);
                throw;
            }
            catch (...)
            {
                this->_M_setstate(ios_base::badbit);
            }
            if (__err)
                this->setstate(__err);
        }
        return *this;
    }
    template<typename _CharT, typename _Traits >
    streamsize basic_istream<_CharT, _Traits>::readsome(char_type *__s, streamsize __n)
    {
        _M_gcount = 0;
        sentry __cerb(*this, true);
        if (__cerb)
        {
            ios_base::iostate __err = ios_base::iostate(ios_base::goodbit);
            try
            {
                const streamsize __num = this->rdbuf()->in_avail();
                if (__num > 0)
                    _M_gcount = this->rdbuf()->sgetn(__s, std::min(__num, __n));
                else
                    if (__num == - 1)
                        __err |= ios_base::eofbit;
            }
            catch (__cxxabiv1::__forced_unwind &)
            {
                this->_M_setstate(ios_base::badbit);
                throw;
            }
            catch (...)
            {
                this->_M_setstate(ios_base::badbit);
            }
            if (__err)
                this->setstate(__err);
        }
        return _M_gcount;
    }
    template<typename _CharT, typename _Traits >
    basic_istream<_CharT, _Traits> &basic_istream<_CharT, _Traits>::putback(char_type __c)
    {
        _M_gcount = 0;
        sentry __cerb(*this, true);
        if (__cerb)
        {
            ios_base::iostate __err = ios_base::iostate(ios_base::goodbit);
            try
            {
                const int_type __eof = traits_type::eof();
                __streambuf_type *__sb = this->rdbuf();
                if (!__sb || traits_type::eq_int_type(__sb->sputbackc(__c), __eof))
                    __err |= ios_base::badbit;
            }
            catch (__cxxabiv1::__forced_unwind &)
            {
                this->_M_setstate(ios_base::badbit);
                throw;
            }
            catch (...)
            {
                this->_M_setstate(ios_base::badbit);
            }
            if (__err)
                this->setstate(__err);
        }
        return *this;
    }
    template<typename _CharT, typename _Traits >
    basic_istream<_CharT, _Traits> &basic_istream<_CharT, _Traits>::unget(void)
    {
        _M_gcount = 0;
        sentry __cerb(*this, true);
        if (__cerb)
        {
            ios_base::iostate __err = ios_base::iostate(ios_base::goodbit);
            try
            {
                const int_type __eof = traits_type::eof();
                __streambuf_type *__sb = this->rdbuf();
                if (!__sb || traits_type::eq_int_type(__sb->sungetc(), __eof))
                    __err |= ios_base::badbit;
            }
            catch (__cxxabiv1::__forced_unwind &)
            {
                this->_M_setstate(ios_base::badbit);
                throw;
            }
            catch (...)
            {
                this->_M_setstate(ios_base::badbit);
            }
            if (__err)
                this->setstate(__err);
        }
        return *this;
    }
    template<typename _CharT, typename _Traits >
    int basic_istream<_CharT, _Traits>::sync(void)
    {
        int __ret = - 1;
        sentry __cerb(*this, true);
        if (__cerb)
        {
            ios_base::iostate __err = ios_base::iostate(ios_base::goodbit);
            try
            {
                __streambuf_type *__sb = this->rdbuf();
                if (__sb)
                {
                    if (__sb->pubsync() == - 1)
                        __err |= ios_base::badbit;
                    else
                        __ret = 0;
                }
            }
            catch (__cxxabiv1::__forced_unwind &)
            {
                this->_M_setstate(ios_base::badbit);
                throw;
            }
            catch (...)
            {
                this->_M_setstate(ios_base::badbit);
            }
            if (__err)
                this->setstate(__err);
        }
        return __ret;
    }
    template<typename _CharT, typename _Traits >
    typename basic_istream<_CharT, _Traits>::pos_type basic_istream<_CharT, _Traits>::tellg(void)
    {
        pos_type __ret = pos_type(- 1);
        try
        {
            if (!this->fail())
                __ret = this->rdbuf()->pubseekoff(0, ios_base::cur, ios_base::in);
        }
        catch (__cxxabiv1::__forced_unwind &)
        {
            this->_M_setstate(ios_base::badbit);
            throw;
        }
        catch (...)
        {
            this->_M_setstate(ios_base::badbit);
        }
        return __ret;
    }
    template<typename _CharT, typename _Traits >
    basic_istream<_CharT, _Traits> &basic_istream<_CharT, _Traits>::seekg(pos_type __pos)
    {
        ios_base::iostate __err = ios_base::iostate(ios_base::goodbit);
        try
        {
            if (!this->fail())
            {
                const pos_type __p = this->rdbuf()->pubseekpos(__pos, ios_base::in);
                if (__p == pos_type(off_type(- 1)))
                    __err |= ios_base::failbit;
            }
        }
        catch (__cxxabiv1::__forced_unwind &)
        {
            this->_M_setstate(ios_base::badbit);
            throw;
        }
        catch (...)
        {
            this->_M_setstate(ios_base::badbit);
        }
        if (__err)
            this->setstate(__err);
        return *this;
    }
    template<typename _CharT, typename _Traits >
    basic_istream<_CharT, _Traits> &basic_istream<_CharT, _Traits>::seekg(off_type __off, ios_base::seekdir __dir)
    {
        ios_base::iostate __err = ios_base::iostate(ios_base::goodbit);
        try
        {
            if (!this->fail())
            {
                const pos_type __p = this->rdbuf()->pubseekoff(__off, __dir, ios_base::in);
                if (__p == pos_type(off_type(- 1)))
                    __err |= ios_base::failbit;
            }
        }
        catch (__cxxabiv1::__forced_unwind &)
        {
            this->_M_setstate(ios_base::badbit);
            throw;
        }
        catch (...)
        {
            this->_M_setstate(ios_base::badbit);
        }
        if (__err)
            this->setstate(__err);
        return *this;
    }
    template<typename _CharT, typename _Traits >
    basic_istream<_CharT, _Traits> &operator >>(basic_istream<_CharT, _Traits> &__in, _CharT &__c)
    {
        typedef basic_istream<_CharT, _Traits> __istream_type;
        typedef typename __istream_type::int_type __int_type;
        typename __istream_type::sentry __cerb(__in, false);
        if (__cerb)
        {
            ios_base::iostate __err = ios_base::iostate(ios_base::goodbit);
            try
            {
                const __int_type __cb = __in.rdbuf()->sbumpc();
                if (!_Traits::eq_int_type(__cb, _Traits::eof()))
                    __c = _Traits::to_char_type(__cb);
                else
                    __err |= (ios_base::eofbit | ios_base::failbit);
            }
            catch (__cxxabiv1::__forced_unwind &)
            {
                __in._M_setstate(ios_base::badbit);
                throw;
            }
            catch (...)
            {
                __in._M_setstate(ios_base::badbit);
            }
            if (__err)
                __in.setstate(__err);
        }
        return __in;
    }
    template<typename _CharT, typename _Traits >
    basic_istream<_CharT, _Traits> &operator >>(basic_istream<_CharT, _Traits> &__in, _CharT *__s)
    {
        typedef basic_istream<_CharT, _Traits> __istream_type;
        typedef basic_streambuf<_CharT, _Traits> __streambuf_type;
        typedef typename _Traits::int_type int_type;
        typedef _CharT char_type;
        typedef ctype<_CharT> __ctype_type;
        streamsize __extracted = 0;
        ios_base::iostate __err = ios_base::iostate(ios_base::goodbit);
        typename __istream_type::sentry __cerb(__in, false);
        if (__cerb)
        {
            try
            {
                streamsize __num = __in.width();
                if (__num <= 0)
                    __num = __gnu_cxx::__numeric_traits<streamsize>::__max;
                const __ctype_type &__ct = use_facet<__ctype_type>(__in.getloc());
                const int_type __eof = _Traits::eof();
                __streambuf_type *__sb = __in.rdbuf();
                int_type __c = __sb->sgetc();
                while (__extracted < __num - 1 && !_Traits::eq_int_type(__c, __eof) && !__ct.is(ctype_base::space, _Traits::to_char_type(__c)))
                {
                    *__s++ = _Traits::to_char_type(__c);
                    ++__extracted;
                    __c = __sb->snextc();
                }
                if (_Traits::eq_int_type(__c, __eof))
                    __err |= ios_base::eofbit;
                *__s = char_type();
                __in.width(0);
            }
            catch (__cxxabiv1::__forced_unwind &)
            {
                __in._M_setstate(ios_base::badbit);
                throw;
            }
            catch (...)
            {
                __in._M_setstate(ios_base::badbit);
            }
        }
        if (!__extracted)
            __err |= ios_base::failbit;
        if (__err)
            __in.setstate(__err);
        return __in;
    }
    template<typename _CharT, typename _Traits >
    basic_istream<_CharT, _Traits> &ws(basic_istream<_CharT, _Traits> &__in)
    {
        typedef basic_istream<_CharT, _Traits> __istream_type;
        typedef basic_streambuf<_CharT, _Traits> __streambuf_type;
        typedef typename __istream_type::int_type __int_type;
        typedef ctype<_CharT> __ctype_type;
        const __ctype_type &__ct = use_facet<__ctype_type>(__in.getloc());
        const __int_type __eof = _Traits::eof();
        __streambuf_type *__sb = __in.rdbuf();
        __int_type __c = __sb->sgetc();
        while (!_Traits::eq_int_type(__c, __eof) && __ct.is(ctype_base::space, _Traits::to_char_type(__c)))
            __c = __sb->snextc();
        if (_Traits::eq_int_type(__c, __eof))
            __in.setstate(ios_base::eofbit);
        return __in;
    }
    extern template class basic_istream<char>;
    extern template istream &ws(istream &);
    extern template istream &operator >>(istream &, char &);
    extern template istream &operator >>(istream &, char *);
    extern template istream &operator >>(istream &, unsigned char &);
    extern template istream &operator >>(istream &, signed char &);
    extern template istream &operator >>(istream &, unsigned char *);
    extern template istream &operator >>(istream &, signed char *);
    extern template istream &istream::_M_extract(unsigned short &);
    extern template istream &istream::_M_extract(unsigned int &);
    extern template istream &istream::_M_extract(long &);
    extern template istream &istream::_M_extract(unsigned long &);
    extern template istream &istream::_M_extract(bool &);
    extern template istream &istream::_M_extract(long long &);
    extern template istream &istream::_M_extract(unsigned long long &);
    extern template istream &istream::_M_extract(float &);
    extern template istream &istream::_M_extract(double &);
    extern template istream &istream::_M_extract(long double &);
    extern template istream &istream::_M_extract(void *&);
    extern template class basic_iostream<char>;
    extern template class basic_istream<wchar_t>;
    extern template wistream &ws(wistream &);
    extern template wistream &operator >>(wistream &, wchar_t &);
    extern template wistream &operator >>(wistream &, wchar_t *);
    extern template wistream &wistream::_M_extract(unsigned short &);
    extern template wistream &wistream::_M_extract(unsigned int &);
    extern template wistream &wistream::_M_extract(long &);
    extern template wistream &wistream::_M_extract(unsigned long &);
    extern template wistream &wistream::_M_extract(bool &);
    extern template wistream &wistream::_M_extract(long long &);
    extern template wistream &wistream::_M_extract(unsigned long long &);
    extern template wistream &wistream::_M_extract(float &);
    extern template wistream &wistream::_M_extract(double &);
    extern template wistream &wistream::_M_extract(long double &);
    extern template wistream &wistream::_M_extract(void *&);
    extern template class basic_iostream<wchar_t>;
}
namespace std __attribute__((__visibility__("default"))) {
    template<typename _CharT, typename _Traits, typename _Alloc >
    class basic_stringbuf : public basic_streambuf<_CharT, _Traits>
    {
        public :
            typedef _CharT char_type;
            typedef _Traits traits_type;
            typedef _Alloc allocator_type;
            typedef typename traits_type::int_type int_type;
            typedef typename traits_type::pos_type pos_type;
            typedef typename traits_type::off_type off_type;
            typedef basic_streambuf<char_type, traits_type> __streambuf_type;
            typedef basic_string<char_type, _Traits, _Alloc> __string_type;
            typedef typename __string_type::size_type __size_type;
        protected :
            ios_base::openmode _M_mode;
            __string_type _M_string;
        public :
            explicit basic_stringbuf(ios_base::openmode __mode = ios_base::in | ios_base::out)
                : __streambuf_type(), _M_mode(__mode), _M_string() 
            {
            }
            explicit basic_stringbuf(const __string_type &__str, ios_base::openmode __mode = ios_base::in | ios_base::out)
                : __streambuf_type(), _M_mode(), _M_string(__str.data(), __str.size()) 
            {
                _M_stringbuf_init(__mode);
            }
            __string_type str() const
            {
                __string_type __ret;
                if (this->pptr())
                {
                    if (this->pptr() > this->egptr())
                        __ret = __string_type(this->pbase(), this->pptr());
                    else
                        __ret = __string_type(this->pbase(), this->egptr());
                }
                else
                    __ret = _M_string;
                return __ret;
            }
            void str(const __string_type &__s)
            {
                _M_string.assign(__s.data(), __s.size());
                _M_stringbuf_init(_M_mode);
            }
        protected :
            void _M_stringbuf_init(ios_base::openmode __mode)
            {
                _M_mode = __mode;
                __size_type __len = 0;
                if (_M_mode & (ios_base::ate | ios_base::app))
                    __len = _M_string.size();
                _M_sync(const_cast<char_type * >(_M_string.data()), 0, __len);
            }
            virtual streamsize showmanyc()
            {
                streamsize __ret = - 1;
                if (_M_mode & ios_base::in)
                {
                    _M_update_egptr();
                    __ret = this->egptr() - this->gptr();
                }
                return __ret;
            }
            virtual int_type underflow();
            virtual int_type pbackfail(int_type __c = traits_type::eof());
            virtual int_type overflow(int_type __c = traits_type::eof());
            virtual __streambuf_type *setbuf(char_type *__s, streamsize __n)
            {
                if (__s && __n >= 0)
                {
                    _M_string.clear();
                    _M_sync(__s, __n, 0);
                }
                return this;
            }
            virtual pos_type seekoff(off_type __off, ios_base::seekdir __way, ios_base::openmode __mode = ios_base::in | ios_base::out);
            virtual pos_type seekpos(pos_type __sp, ios_base::openmode __mode = ios_base::in | ios_base::out);
            void _M_sync(char_type *__base, __size_type __i, __size_type __o);
            void _M_update_egptr()
            {
                const bool __testin = _M_mode & ios_base::in;
                if (this->pptr() && this->pptr() > this->egptr())
                {
                    if (__testin)
                        this->setg(this->eback(), this->gptr(), this->pptr());
                    else
                        this->setg(this->pptr(), this->pptr(), this->pptr());
                }
            }
    };
    template<typename _CharT, typename _Traits, typename _Alloc >
    class basic_istringstream : public basic_istream<_CharT, _Traits>
    {
        public :
            typedef _CharT char_type;
            typedef _Traits traits_type;
            typedef _Alloc allocator_type;
            typedef typename traits_type::int_type int_type;
            typedef typename traits_type::pos_type pos_type;
            typedef typename traits_type::off_type off_type;
            typedef basic_string<_CharT, _Traits, _Alloc> __string_type;
            typedef basic_stringbuf<_CharT, _Traits, _Alloc> __stringbuf_type;
            typedef basic_istream<char_type, traits_type> __istream_type;
        private :
            __stringbuf_type _M_stringbuf;
        public :
            explicit basic_istringstream(ios_base::openmode __mode = ios_base::in)
                : __istream_type(), _M_stringbuf(__mode | ios_base::in) 
            {
                this->init(&_M_stringbuf);
            }
            explicit basic_istringstream(const __string_type &__str, ios_base::openmode __mode = ios_base::in)
                : __istream_type(), _M_stringbuf(__str, __mode | ios_base::in) 
            {
                this->init(&_M_stringbuf);
            }
            ~basic_istringstream()
            {
            }
            __stringbuf_type *rdbuf() const
            {
                return const_cast<__stringbuf_type * >(&_M_stringbuf);
            }
            __string_type str() const
            {
                return _M_stringbuf.str();
            }
            void str(const __string_type &__s)
            {
                _M_stringbuf.str(__s);
            }
    };
    template<typename _CharT, typename _Traits, typename _Alloc >
    class basic_ostringstream : public basic_ostream<_CharT, _Traits>
    {
        public :
            typedef _CharT char_type;
            typedef _Traits traits_type;
            typedef _Alloc allocator_type;
            typedef typename traits_type::int_type int_type;
            typedef typename traits_type::pos_type pos_type;
            typedef typename traits_type::off_type off_type;
            typedef basic_string<_CharT, _Traits, _Alloc> __string_type;
            typedef basic_stringbuf<_CharT, _Traits, _Alloc> __stringbuf_type;
            typedef basic_ostream<char_type, traits_type> __ostream_type;
        private :
            __stringbuf_type _M_stringbuf;
        public :
            explicit basic_ostringstream(ios_base::openmode __mode = ios_base::out)
                : __ostream_type(), _M_stringbuf(__mode | ios_base::out) 
            {
                this->init(&_M_stringbuf);
            }
            explicit basic_ostringstream(const __string_type &__str, ios_base::openmode __mode = ios_base::out)
                : __ostream_type(), _M_stringbuf(__str, __mode | ios_base::out) 
            {
                this->init(&_M_stringbuf);
            }
            ~basic_ostringstream()
            {
            }
            __stringbuf_type *rdbuf() const
            {
                return const_cast<__stringbuf_type * >(&_M_stringbuf);
            }
            __string_type str() const
            {
                return _M_stringbuf.str();
            }
            void str(const __string_type &__s)
            {
                _M_stringbuf.str(__s);
            }
    };
    template<typename _CharT, typename _Traits, typename _Alloc >
    class basic_stringstream : public basic_iostream<_CharT, _Traits>
    {
        public :
            typedef _CharT char_type;
            typedef _Traits traits_type;
            typedef _Alloc allocator_type;
            typedef typename traits_type::int_type int_type;
            typedef typename traits_type::pos_type pos_type;
            typedef typename traits_type::off_type off_type;
            typedef basic_string<_CharT, _Traits, _Alloc> __string_type;
            typedef basic_stringbuf<_CharT, _Traits, _Alloc> __stringbuf_type;
            typedef basic_iostream<char_type, traits_type> __iostream_type;
        private :
            __stringbuf_type _M_stringbuf;
        public :
            explicit basic_stringstream(ios_base::openmode __m = ios_base::out | ios_base::in)
                : __iostream_type(), _M_stringbuf(__m) 
            {
                this->init(&_M_stringbuf);
            }
            explicit basic_stringstream(const __string_type &__str, ios_base::openmode __m = ios_base::out | ios_base::in)
                : __iostream_type(), _M_stringbuf(__str, __m) 
            {
                this->init(&_M_stringbuf);
            }
            ~basic_stringstream()
            {
            }
            __stringbuf_type *rdbuf() const
            {
                return const_cast<__stringbuf_type * >(&_M_stringbuf);
            }
            __string_type str() const
            {
                return _M_stringbuf.str();
            }
            void str(const __string_type &__s)
            {
                _M_stringbuf.str(__s);
            }
    };
}
namespace std __attribute__((__visibility__("default"))) {
    template<class _CharT, class _Traits, class _Alloc >
    typename basic_stringbuf<_CharT, _Traits, _Alloc>::int_type basic_stringbuf<_CharT, _Traits, _Alloc>::pbackfail(int_type __c)
    {
        int_type __ret = traits_type::eof();
        if (this->eback() < this->gptr())
        {
            const bool __testeof = traits_type::eq_int_type(__c, __ret);
            if (!__testeof)
            {
                const bool __testeq = traits_type::eq(traits_type::to_char_type(__c), this->gptr()[- 1]);
                const bool __testout = this->_M_mode & ios_base::out;
                if (__testeq || __testout)
                {
                    this->gbump(- 1);
                    if (!__testeq)
                        *this->gptr() = traits_type::to_char_type(__c);
                    __ret = __c;
                }
            }
            else
            {
                this->gbump(- 1);
                __ret = traits_type::not_eof(__c);
            }
        }
        return __ret;
    }
    template<class _CharT, class _Traits, class _Alloc >
    typename basic_stringbuf<_CharT, _Traits, _Alloc>::int_type basic_stringbuf<_CharT, _Traits, _Alloc>::overflow(int_type __c)
    {
        const bool __testout = this->_M_mode & ios_base::out;
        if (__builtin_expect(!__testout, false))
            return traits_type::eof();
        const bool __testeof = traits_type::eq_int_type(__c, traits_type::eof());
        if (__builtin_expect(__testeof, false))
            return traits_type::not_eof(__c);
        const __size_type __capacity = _M_string.capacity();
        const __size_type __max_size = _M_string.max_size();
        const bool __testput = this->pptr() < this->epptr();
        if (__builtin_expect(!__testput && __capacity == __max_size, false))
            return traits_type::eof();
        const char_type __conv = traits_type::to_char_type(__c);
        if (!__testput)
        {
            const __size_type __opt_len = std::max(__size_type(2 * __capacity), __size_type(512));
            const __size_type __len = std::min(__opt_len, __max_size);
            __string_type __tmp;
            __tmp.reserve(__len);
            if (this->pbase())
                __tmp.assign(this->pbase(), this->epptr() - this->pbase());
            __tmp.push_back(__conv);
            _M_string.swap(__tmp);
            _M_sync(const_cast<char_type * >(_M_string.data()), this->gptr() - this->eback(), this->pptr() - this->pbase());
        }
        else
            *this->pptr() = __conv;
        this->pbump(1);
        return __c;
    }
    template<class _CharT, class _Traits, class _Alloc >
    typename basic_stringbuf<_CharT, _Traits, _Alloc>::int_type basic_stringbuf<_CharT, _Traits, _Alloc>::underflow()
    {
        int_type __ret = traits_type::eof();
        const bool __testin = this->_M_mode & ios_base::in;
        if (__testin)
        {
            _M_update_egptr();
            if (this->gptr() < this->egptr())
                __ret = traits_type::to_int_type(*this->gptr());
        }
        return __ret;
    }
    template<class _CharT, class _Traits, class _Alloc >
    typename basic_stringbuf<_CharT, _Traits, _Alloc>::pos_type basic_stringbuf<_CharT, _Traits, _Alloc>::seekoff(off_type __off, ios_base::seekdir __way, ios_base::openmode __mode)
    {
        pos_type __ret = pos_type(off_type(- 1));
        bool __testin = (ios_base::in & this->_M_mode & __mode) != 0;
        bool __testout = (ios_base::out & this->_M_mode & __mode) != 0;
        const bool __testboth = __testin && __testout && __way != ios_base::cur;
        __testin &= !(__mode & ios_base::out);
        __testout &= !(__mode & ios_base::in);
        const char_type *__beg = __testin ? this->eback() : this->pbase();
        if ((__beg || !__off) && (__testin || __testout || __testboth))
        {
            _M_update_egptr();
            off_type __newoffi = __off;
            off_type __newoffo = __newoffi;
            if (__way == ios_base::cur)
            {
                __newoffi += this->gptr() - __beg;
                __newoffo += this->pptr() - __beg;
            }
            else
                if (__way == ios_base::end)
                    __newoffo = __newoffi += this->egptr() - __beg;
            if ((__testin || __testboth) && __newoffi >= 0 && this->egptr() - __beg >= __newoffi)
            {
                this->gbump((__beg + __newoffi) - this->gptr());
                __ret = pos_type(__newoffi);
            }
            if ((__testout || __testboth) && __newoffo >= 0 && this->egptr() - __beg >= __newoffo)
            {
                this->pbump((__beg + __newoffo) - this->pptr());
                __ret = pos_type(__newoffo);
            }
        }
        return __ret;
    }
    template<class _CharT, class _Traits, class _Alloc >
    typename basic_stringbuf<_CharT, _Traits, _Alloc>::pos_type basic_stringbuf<_CharT, _Traits, _Alloc>::seekpos(pos_type __sp, ios_base::openmode __mode)
    {
        pos_type __ret = pos_type(off_type(- 1));
        const bool __testin = (ios_base::in & this->_M_mode & __mode) != 0;
        const bool __testout = (ios_base::out & this->_M_mode & __mode) != 0;
        const char_type *__beg = __testin ? this->eback() : this->pbase();
        if ((__beg || !off_type(__sp)) && (__testin || __testout))
        {
            _M_update_egptr();
            const off_type __pos(__sp);
            const bool __testpos = (0 <= __pos && __pos <= this->egptr() - __beg);
            if (__testpos)
            {
                if (__testin)
                    this->gbump((__beg + __pos) - this->gptr());
                if (__testout)
                    this->pbump((__beg + __pos) - this->pptr());
                __ret = __sp;
            }
        }
        return __ret;
    }
    template<class _CharT, class _Traits, class _Alloc >
    void basic_stringbuf<_CharT, _Traits, _Alloc>::_M_sync(char_type *__base, __size_type __i, __size_type __o)
    {
        const bool __testin = _M_mode & ios_base::in;
        const bool __testout = _M_mode & ios_base::out;
        char_type *__endg = __base + _M_string.size();
        char_type *__endp = __base + _M_string.capacity();
        if (__base != _M_string.data())
        {
            __endg += __i;
            __i = 0;
            __endp = __endg;
        }
        if (__testin)
            this->setg(__base, __base + __i, __endg);
        if (__testout)
        {
            this->setp(__base, __endp);
            this->pbump(__o);
            if (!__testin)
                this->setg(__endg, __endg, __endg);
        }
    }
    extern template class basic_stringbuf<char>;
    extern template class basic_istringstream<char>;
    extern template class basic_ostringstream<char>;
    extern template class basic_stringstream<char>;
    extern template class basic_stringbuf<wchar_t>;
    extern template class basic_istringstream<wchar_t>;
    extern template class basic_ostringstream<wchar_t>;
    extern template class basic_stringstream<wchar_t>;
}
struct timeval
{
        __time_t tv_sec;
        __suseconds_t tv_usec;
};
typedef int __sig_atomic_t;
typedef struct 
{
        unsigned long int __val[(1024 / (8 * sizeof(unsigned long int)))];
} __sigset_t;
typedef __sigset_t sigset_t;
typedef __suseconds_t suseconds_t;
typedef long int __fd_mask;
typedef struct 
{
        __fd_mask fds_bits[1024 / (8 * (int) sizeof(__fd_mask))];
} fd_set;
typedef __fd_mask fd_mask;
extern "C"
{
    extern int select(int __nfds, fd_set *__restrict __readfds, fd_set *__restrict __writefds, fd_set *__restrict __exceptfds, struct timeval *__restrict __timeout);
    extern int pselect(int __nfds, fd_set *__restrict __readfds, fd_set *__restrict __writefds, fd_set *__restrict __exceptfds, const struct timespec *__restrict __timeout, const __sigset_t *__restrict __sigmask);
}
extern "C"
{
    struct timezone
    {
            int tz_minuteswest;
            int tz_dsttime;
    };
    typedef struct timezone *__restrict __timezone_ptr_t;
    extern int gettimeofday(struct timeval *__restrict __tv, __timezone_ptr_t __tz) throw () __attribute__((__nonnull__(1)));
    extern int settimeofday(__const struct timeval *__tv, __const struct timezone *__tz) throw () __attribute__((__nonnull__(1)));
    extern int adjtime(__const struct timeval *__delta, struct timeval *__olddelta) throw ();
    enum __itimer_which
    {
        ITIMER_REAL = 0, 
        ITIMER_VIRTUAL = 1, 
        ITIMER_PROF = 2
    };
    struct itimerval
    {
            struct timeval it_interval;
            struct timeval it_value;
    };
    typedef int __itimer_which_t;
    extern int getitimer(__itimer_which_t __which, struct itimerval *__value) throw ();
    extern int setitimer(__itimer_which_t __which, __const struct itimerval *__restrict __new, struct itimerval *__restrict __old) throw ();
    extern int utimes(__const char *__file, __const struct timeval __tvp[2]) throw () __attribute__((__nonnull__(1)));
    extern int lutimes(__const char *__file, __const struct timeval __tvp[2]) throw () __attribute__((__nonnull__(1)));
    extern int futimes(int __fd, __const struct timeval __tvp[2]) throw ();
    extern int futimesat(int __fd, __const char *__file, __const struct timeval __tvp[2]) throw ();
}
extern "C"
{
    typedef ptrdiff_t MPI_Aint;
    typedef long long MPI_Offset;
    typedef struct ompi_communicator_t *MPI_Comm;
    typedef struct ompi_datatype_t *MPI_Datatype;
    typedef struct ompi_errhandler_t *MPI_Errhandler;
    typedef struct ompi_file_t *MPI_File;
    typedef struct ompi_group_t *MPI_Group;
    typedef struct ompi_info_t *MPI_Info;
    typedef struct ompi_op_t *MPI_Op;
    typedef struct ompi_request_t *MPI_Request;
    typedef struct ompi_status_public_t MPI_Status;
    typedef struct ompi_win_t *MPI_Win;
    struct ompi_status_public_t
    {
            int MPI_SOURCE;
            int MPI_TAG;
            int MPI_ERROR;
            int _cancelled;
            size_t _ucount;
    };
    typedef struct ompi_status_public_t ompi_status_public_t;
    typedef int (MPI_Copy_function)(MPI_Comm, int, void *, void *, void *, int *);
    typedef int (MPI_Delete_function)(MPI_Comm, int, void *, void *);
    typedef int (MPI_Datarep_extent_function)(MPI_Datatype, MPI_Aint *, void *);
    typedef int (MPI_Datarep_conversion_function)(void *, MPI_Datatype, int, void *, MPI_Offset, void *);
    typedef void (MPI_Comm_errhandler_function)(MPI_Comm *, int *, ...);
    typedef MPI_Comm_errhandler_function MPI_Comm_errhandler_fn;
    typedef void (ompi_file_errhandler_fn)(MPI_File *, int *, ...);
    typedef ompi_file_errhandler_fn MPI_File_errhandler_fn;
    typedef ompi_file_errhandler_fn MPI_File_errhandler_function;
    typedef void (MPI_Win_errhandler_function)(MPI_Win *, int *, ...);
    typedef MPI_Win_errhandler_function MPI_Win_errhandler_fn;
    typedef void (MPI_Handler_function)(MPI_Comm *, int *, ...);
    typedef void (MPI_User_function)(void *, void *, int *, MPI_Datatype *);
    typedef int (MPI_Comm_copy_attr_function)(MPI_Comm, int, void *, void *, void *, int *);
    typedef int (MPI_Comm_delete_attr_function)(MPI_Comm, int, void *, void *);
    typedef int (MPI_Type_copy_attr_function)(MPI_Datatype, int, void *, void *, void *, int *);
    typedef int (MPI_Type_delete_attr_function)(MPI_Datatype, int, void *, void *);
    typedef int (MPI_Win_copy_attr_function)(MPI_Win, int, void *, void *, void *, int *);
    typedef int (MPI_Win_delete_attr_function)(MPI_Win, int, void *, void *);
    typedef int (MPI_Grequest_query_function)(void *, MPI_Status *);
    typedef int (MPI_Grequest_free_function)(void *);
    typedef int (MPI_Grequest_cancel_function)(void *, int);
    enum 
    {
        MPI_TAG_UB, 
        MPI_HOST, 
        MPI_IO, 
        MPI_WTIME_IS_GLOBAL, 
        MPI_APPNUM, 
        MPI_LASTUSEDCODE, 
        MPI_UNIVERSE_SIZE, 
        MPI_WIN_BASE, 
        MPI_WIN_SIZE, 
        MPI_WIN_DISP_UNIT, 
        IMPI_CLIENT_SIZE, 
        IMPI_CLIENT_COLOR, 
        IMPI_HOST_SIZE, 
        IMPI_HOST_COLOR
    };
    enum 
    {
        MPI_IDENT, 
        MPI_CONGRUENT, 
        MPI_SIMILAR, 
        MPI_UNEQUAL
    };
    enum 
    {
        MPI_THREAD_SINGLE, 
        MPI_THREAD_FUNNELED, 
        MPI_THREAD_SERIALIZED, 
        MPI_THREAD_MULTIPLE
    };
    enum 
    {
        MPI_COMBINER_NAMED, 
        MPI_COMBINER_DUP, 
        MPI_COMBINER_CONTIGUOUS, 
        MPI_COMBINER_VECTOR, 
        MPI_COMBINER_HVECTOR_INTEGER, 
        MPI_COMBINER_HVECTOR, 
        MPI_COMBINER_INDEXED, 
        MPI_COMBINER_HINDEXED_INTEGER, 
        MPI_COMBINER_HINDEXED, 
        MPI_COMBINER_INDEXED_BLOCK, 
        MPI_COMBINER_STRUCT_INTEGER, 
        MPI_COMBINER_STRUCT, 
        MPI_COMBINER_SUBARRAY, 
        MPI_COMBINER_DARRAY, 
        MPI_COMBINER_F90_REAL, 
        MPI_COMBINER_F90_COMPLEX, 
        MPI_COMBINER_F90_INTEGER, 
        MPI_COMBINER_RESIZED
    };
    __attribute__((visibility("default"))) int OMPI_C_MPI_TYPE_NULL_DELETE_FN(MPI_Datatype datatype, int type_keyval, void *attribute_val_out, void *extra_state);
    __attribute__((visibility("default"))) int OMPI_C_MPI_TYPE_NULL_COPY_FN(MPI_Datatype datatype, int type_keyval, void *extra_state, void *attribute_val_in, void *attribute_val_out, int *flag);
    __attribute__((visibility("default"))) int OMPI_C_MPI_TYPE_DUP_FN(MPI_Datatype datatype, int type_keyval, void *extra_state, void *attribute_val_in, void *attribute_val_out, int *flag);
    __attribute__((visibility("default"))) int OMPI_C_MPI_COMM_NULL_DELETE_FN(MPI_Comm comm, int comm_keyval, void *attribute_val_out, void *extra_state);
    __attribute__((visibility("default"))) int OMPI_C_MPI_COMM_NULL_COPY_FN(MPI_Comm comm, int comm_keyval, void *extra_state, void *attribute_val_in, void *attribute_val_out, int *flag);
    __attribute__((visibility("default"))) int OMPI_C_MPI_COMM_DUP_FN(MPI_Comm comm, int comm_keyval, void *extra_state, void *attribute_val_in, void *attribute_val_out, int *flag);
    __attribute__((visibility("default"))) int OMPI_C_MPI_NULL_DELETE_FN(MPI_Comm comm, int comm_keyval, void *attribute_val_out, void *extra_state);
    __attribute__((visibility("default"))) int OMPI_C_MPI_NULL_COPY_FN(MPI_Comm comm, int comm_keyval, void *extra_state, void *attribute_val_in, void *attribute_val_out, int *flag);
    __attribute__((visibility("default"))) int OMPI_C_MPI_DUP_FN(MPI_Comm comm, int comm_keyval, void *extra_state, void *attribute_val_in, void *attribute_val_out, int *flag);
    __attribute__((visibility("default"))) int OMPI_C_MPI_WIN_NULL_DELETE_FN(MPI_Win window, int win_keyval, void *attribute_val_out, void *extra_state);
    __attribute__((visibility("default"))) int OMPI_C_MPI_WIN_NULL_COPY_FN(MPI_Win window, int win_keyval, void *extra_state, void *attribute_val_in, void *attribute_val_out, int *flag);
    __attribute__((visibility("default"))) int OMPI_C_MPI_WIN_DUP_FN(MPI_Win window, int win_keyval, void *extra_state, void *attribute_val_in, void *attribute_val_out, int *flag);
    __attribute__((visibility("default"))) extern struct ompi_predefined_communicator_t ompi_mpi_comm_world;
    __attribute__((visibility("default"))) extern struct ompi_predefined_communicator_t ompi_mpi_comm_self;
    __attribute__((visibility("default"))) extern struct ompi_predefined_communicator_t ompi_mpi_comm_null;
    __attribute__((visibility("default"))) extern struct ompi_predefined_group_t ompi_mpi_group_empty;
    __attribute__((visibility("default"))) extern struct ompi_predefined_group_t ompi_mpi_group_null;
    __attribute__((visibility("default"))) extern struct ompi_predefined_request_t ompi_request_null;
    __attribute__((visibility("default"))) extern struct ompi_predefined_op_t ompi_mpi_op_null;
    __attribute__((visibility("default"))) extern struct ompi_predefined_op_t ompi_mpi_op_min;
    __attribute__((visibility("default"))) extern struct ompi_predefined_op_t ompi_mpi_op_max;
    __attribute__((visibility("default"))) extern struct ompi_predefined_op_t ompi_mpi_op_sum;
    __attribute__((visibility("default"))) extern struct ompi_predefined_op_t ompi_mpi_op_prod;
    __attribute__((visibility("default"))) extern struct ompi_predefined_op_t ompi_mpi_op_land;
    __attribute__((visibility("default"))) extern struct ompi_predefined_op_t ompi_mpi_op_band;
    __attribute__((visibility("default"))) extern struct ompi_predefined_op_t ompi_mpi_op_lor;
    __attribute__((visibility("default"))) extern struct ompi_predefined_op_t ompi_mpi_op_bor;
    __attribute__((visibility("default"))) extern struct ompi_predefined_op_t ompi_mpi_op_lxor;
    __attribute__((visibility("default"))) extern struct ompi_predefined_op_t ompi_mpi_op_bxor;
    __attribute__((visibility("default"))) extern struct ompi_predefined_op_t ompi_mpi_op_maxloc;
    __attribute__((visibility("default"))) extern struct ompi_predefined_op_t ompi_mpi_op_minloc;
    __attribute__((visibility("default"))) extern struct ompi_predefined_op_t ompi_mpi_op_replace;
    __attribute__((visibility("default"))) extern struct ompi_predefined_datatype_t ompi_mpi_datatype_null;
    __attribute__((visibility("default"))) extern struct ompi_predefined_datatype_t ompi_mpi_lb;
    __attribute__((visibility("default"))) extern struct ompi_predefined_datatype_t ompi_mpi_ub;
    __attribute__((visibility("default"))) extern struct ompi_predefined_datatype_t ompi_mpi_char;
    __attribute__((visibility("default"))) extern struct ompi_predefined_datatype_t ompi_mpi_signed_char;
    __attribute__((visibility("default"))) extern struct ompi_predefined_datatype_t ompi_mpi_unsigned_char;
    __attribute__((visibility("default"))) extern struct ompi_predefined_datatype_t ompi_mpi_byte;
    __attribute__((visibility("default"))) extern struct ompi_predefined_datatype_t ompi_mpi_short;
    __attribute__((visibility("default"))) extern struct ompi_predefined_datatype_t ompi_mpi_unsigned_short;
    __attribute__((visibility("default"))) extern struct ompi_predefined_datatype_t ompi_mpi_int;
    __attribute__((visibility("default"))) extern struct ompi_predefined_datatype_t ompi_mpi_unsigned;
    __attribute__((visibility("default"))) extern struct ompi_predefined_datatype_t ompi_mpi_long;
    __attribute__((visibility("default"))) extern struct ompi_predefined_datatype_t ompi_mpi_unsigned_long;
    __attribute__((visibility("default"))) extern struct ompi_predefined_datatype_t ompi_mpi_long_long_int;
    __attribute__((visibility("default"))) extern struct ompi_predefined_datatype_t ompi_mpi_unsigned_long_long;
    __attribute__((visibility("default"))) extern struct ompi_predefined_datatype_t ompi_mpi_float;
    __attribute__((visibility("default"))) extern struct ompi_predefined_datatype_t ompi_mpi_double;
    __attribute__((visibility("default"))) extern struct ompi_predefined_datatype_t ompi_mpi_long_double;
    __attribute__((visibility("default"))) extern struct ompi_predefined_datatype_t ompi_mpi_wchar;
    __attribute__((visibility("default"))) extern struct ompi_predefined_datatype_t ompi_mpi_packed;
    __attribute__((visibility("default"))) extern struct ompi_predefined_datatype_t ompi_mpi_cxx_bool;
    __attribute__((visibility("default"))) extern struct ompi_predefined_datatype_t ompi_mpi_cxx_cplex;
    __attribute__((visibility("default"))) extern struct ompi_predefined_datatype_t ompi_mpi_cxx_dblcplex;
    __attribute__((visibility("default"))) extern struct ompi_predefined_datatype_t ompi_mpi_cxx_ldblcplex;
    __attribute__((visibility("default"))) extern struct ompi_predefined_datatype_t ompi_mpi_logical;
    __attribute__((visibility("default"))) extern struct ompi_predefined_datatype_t ompi_mpi_character;
    __attribute__((visibility("default"))) extern struct ompi_predefined_datatype_t ompi_mpi_integer;
    __attribute__((visibility("default"))) extern struct ompi_predefined_datatype_t ompi_mpi_real;
    __attribute__((visibility("default"))) extern struct ompi_predefined_datatype_t ompi_mpi_dblprec;
    __attribute__((visibility("default"))) extern struct ompi_predefined_datatype_t ompi_mpi_cplex;
    __attribute__((visibility("default"))) extern struct ompi_predefined_datatype_t ompi_mpi_dblcplex;
    __attribute__((visibility("default"))) extern struct ompi_predefined_datatype_t ompi_mpi_ldblcplex;
    __attribute__((visibility("default"))) extern struct ompi_predefined_datatype_t ompi_mpi_2int;
    __attribute__((visibility("default"))) extern struct ompi_predefined_datatype_t ompi_mpi_2integer;
    __attribute__((visibility("default"))) extern struct ompi_predefined_datatype_t ompi_mpi_2real;
    __attribute__((visibility("default"))) extern struct ompi_predefined_datatype_t ompi_mpi_2dblprec;
    __attribute__((visibility("default"))) extern struct ompi_predefined_datatype_t ompi_mpi_2cplex;
    __attribute__((visibility("default"))) extern struct ompi_predefined_datatype_t ompi_mpi_2dblcplex;
    __attribute__((visibility("default"))) extern struct ompi_predefined_datatype_t ompi_mpi_float_int;
    __attribute__((visibility("default"))) extern struct ompi_predefined_datatype_t ompi_mpi_double_int;
    __attribute__((visibility("default"))) extern struct ompi_predefined_datatype_t ompi_mpi_longdbl_int;
    __attribute__((visibility("default"))) extern struct ompi_predefined_datatype_t ompi_mpi_short_int;
    __attribute__((visibility("default"))) extern struct ompi_predefined_datatype_t ompi_mpi_long_int;
    __attribute__((visibility("default"))) extern struct ompi_predefined_datatype_t ompi_mpi_logical1;
    __attribute__((visibility("default"))) extern struct ompi_predefined_datatype_t ompi_mpi_logical2;
    __attribute__((visibility("default"))) extern struct ompi_predefined_datatype_t ompi_mpi_logical4;
    __attribute__((visibility("default"))) extern struct ompi_predefined_datatype_t ompi_mpi_logical8;
    __attribute__((visibility("default"))) extern struct ompi_predefined_datatype_t ompi_mpi_integer1;
    __attribute__((visibility("default"))) extern struct ompi_predefined_datatype_t ompi_mpi_integer2;
    __attribute__((visibility("default"))) extern struct ompi_predefined_datatype_t ompi_mpi_integer4;
    __attribute__((visibility("default"))) extern struct ompi_predefined_datatype_t ompi_mpi_integer8;
    __attribute__((visibility("default"))) extern struct ompi_predefined_datatype_t ompi_mpi_integer16;
    __attribute__((visibility("default"))) extern struct ompi_predefined_datatype_t ompi_mpi_real2;
    __attribute__((visibility("default"))) extern struct ompi_predefined_datatype_t ompi_mpi_real4;
    __attribute__((visibility("default"))) extern struct ompi_predefined_datatype_t ompi_mpi_real8;
    __attribute__((visibility("default"))) extern struct ompi_predefined_datatype_t ompi_mpi_real16;
    __attribute__((visibility("default"))) extern struct ompi_predefined_datatype_t ompi_mpi_complex8;
    __attribute__((visibility("default"))) extern struct ompi_predefined_datatype_t ompi_mpi_complex16;
    __attribute__((visibility("default"))) extern struct ompi_predefined_datatype_t ompi_mpi_complex32;
    __attribute__((visibility("default"))) extern struct ompi_predefined_errhandler_t ompi_mpi_errhandler_null;
    __attribute__((visibility("default"))) extern struct ompi_predefined_errhandler_t ompi_mpi_errors_are_fatal;
    __attribute__((visibility("default"))) extern struct ompi_predefined_errhandler_t ompi_mpi_errors_return;
    __attribute__((visibility("default"))) extern struct ompi_predefined_win_t ompi_mpi_win_null;
    __attribute__((visibility("default"))) extern struct ompi_predefined_file_t ompi_mpi_file_null;
    __attribute__((visibility("default"))) extern struct ompi_predefined_info_t ompi_mpi_info_null;
    __attribute__((visibility("default"))) extern int *MPI_F_STATUS_IGNORE;
    __attribute__((visibility("default"))) extern int *MPI_F_STATUSES_IGNORE;
    __attribute__((visibility("default"))) int MPI_Abort(MPI_Comm comm, int errorcode);
    __attribute__((visibility("default"))) int MPI_Accumulate(void *origin_addr, int origin_count, MPI_Datatype origin_datatype, int target_rank, MPI_Aint target_disp, int target_count, MPI_Datatype target_datatype, MPI_Op op, MPI_Win win);
    __attribute__((visibility("default"))) int MPI_Add_error_class(int *errorclass);
    __attribute__((visibility("default"))) int MPI_Add_error_code(int errorclass, int *errorcode);
    __attribute__((visibility("default"))) int MPI_Add_error_string(int errorcode, char *string);
    __attribute__((visibility("default"))) int MPI_Address(void *location, MPI_Aint *address);
    __attribute__((visibility("default"))) int MPI_Allgather(void *sendbuf, int sendcount, MPI_Datatype sendtype, void *recvbuf, int recvcount, MPI_Datatype recvtype, MPI_Comm comm);
    __attribute__((visibility("default"))) int MPI_Allgatherv(void *sendbuf, int sendcount, MPI_Datatype sendtype, void *recvbuf, int *recvcounts, int *displs, MPI_Datatype recvtype, MPI_Comm comm);
    __attribute__((visibility("default"))) int MPI_Alloc_mem(MPI_Aint size, MPI_Info info, void *baseptr);
    __attribute__((visibility("default"))) int MPI_Allreduce(void *sendbuf, void *recvbuf, int count, MPI_Datatype datatype, MPI_Op op, MPI_Comm comm);
    __attribute__((visibility("default"))) int MPI_Alltoall(void *sendbuf, int sendcount, MPI_Datatype sendtype, void *recvbuf, int recvcount, MPI_Datatype recvtype, MPI_Comm comm);
    __attribute__((visibility("default"))) int MPI_Alltoallv(void *sendbuf, int *sendcounts, int *sdispls, MPI_Datatype sendtype, void *recvbuf, int *recvcounts, int *rdispls, MPI_Datatype recvtype, MPI_Comm comm);
    __attribute__((visibility("default"))) int MPI_Alltoallw(void *sendbuf, int *sendcounts, int *sdispls, MPI_Datatype *sendtypes, void *recvbuf, int *recvcounts, int *rdispls, MPI_Datatype *recvtypes, MPI_Comm comm);
    __attribute__((visibility("default"))) int MPI_Attr_delete(MPI_Comm comm, int keyval);
    __attribute__((visibility("default"))) int MPI_Attr_get(MPI_Comm comm, int keyval, void *attribute_val, int *flag);
    __attribute__((visibility("default"))) int MPI_Attr_put(MPI_Comm comm, int keyval, void *attribute_val);
    __attribute__((visibility("default"))) int MPI_Barrier(MPI_Comm comm);
    __attribute__((visibility("default"))) int MPI_Bcast(void *buffer, int count, MPI_Datatype datatype, int root, MPI_Comm comm);
    __attribute__((visibility("default"))) int MPI_Bsend(void *buf, int count, MPI_Datatype datatype, int dest, int tag, MPI_Comm comm);
    __attribute__((visibility("default"))) int MPI_Bsend_init(void *buf, int count, MPI_Datatype datatype, int dest, int tag, MPI_Comm comm, MPI_Request *request);
    __attribute__((visibility("default"))) int MPI_Buffer_attach(void *buffer, int size);
    __attribute__((visibility("default"))) int MPI_Buffer_detach(void *buffer, int *size);
    __attribute__((visibility("default"))) int MPI_Cancel(MPI_Request *request);
    __attribute__((visibility("default"))) int MPI_Cart_coords(MPI_Comm comm, int rank, int maxdims, int *coords);
    __attribute__((visibility("default"))) int MPI_Cart_create(MPI_Comm old_comm, int ndims, int *dims, int *periods, int reorder, MPI_Comm *comm_cart);
    __attribute__((visibility("default"))) int MPI_Cart_get(MPI_Comm comm, int maxdims, int *dims, int *periods, int *coords);
    __attribute__((visibility("default"))) int MPI_Cart_map(MPI_Comm comm, int ndims, int *dims, int *periods, int *newrank);
    __attribute__((visibility("default"))) int MPI_Cart_rank(MPI_Comm comm, int *coords, int *rank);
    __attribute__((visibility("default"))) int MPI_Cart_shift(MPI_Comm comm, int direction, int disp, int *rank_source, int *rank_dest);
    __attribute__((visibility("default"))) int MPI_Cart_sub(MPI_Comm comm, int *remain_dims, MPI_Comm *new_comm);
    __attribute__((visibility("default"))) int MPI_Cartdim_get(MPI_Comm comm, int *ndims);
    __attribute__((visibility("default"))) int MPI_Close_port(char *port_name);
    __attribute__((visibility("default"))) int MPI_Comm_accept(char *port_name, MPI_Info info, int root, MPI_Comm comm, MPI_Comm *newcomm);
    __attribute__((visibility("default"))) int MPI_Comm_c2f(MPI_Comm comm);
    __attribute__((visibility("default"))) int MPI_Comm_call_errhandler(MPI_Comm comm, int errorcode);
    __attribute__((visibility("default"))) int MPI_Comm_compare(MPI_Comm comm1, MPI_Comm comm2, int *result);
    __attribute__((visibility("default"))) int MPI_Comm_connect(char *port_name, MPI_Info info, int root, MPI_Comm comm, MPI_Comm *newcomm);
    __attribute__((visibility("default"))) int MPI_Comm_create_errhandler(MPI_Comm_errhandler_function *function, MPI_Errhandler *errhandler);
    __attribute__((visibility("default"))) int MPI_Comm_create_keyval(MPI_Comm_copy_attr_function *comm_copy_attr_fn, MPI_Comm_delete_attr_function *comm_delete_attr_fn, int *comm_keyval, void *extra_state);
    __attribute__((visibility("default"))) int MPI_Comm_create(MPI_Comm comm, MPI_Group group, MPI_Comm *newcomm);
    __attribute__((visibility("default"))) int MPI_Comm_delete_attr(MPI_Comm comm, int comm_keyval);
    __attribute__((visibility("default"))) int MPI_Comm_disconnect(MPI_Comm *comm);
    __attribute__((visibility("default"))) int MPI_Comm_dup(MPI_Comm comm, MPI_Comm *newcomm);
    __attribute__((visibility("default"))) MPI_Comm MPI_Comm_f2c(int comm);
    __attribute__((visibility("default"))) int MPI_Comm_free_keyval(int *comm_keyval);
    __attribute__((visibility("default"))) int MPI_Comm_free(MPI_Comm *comm);
    __attribute__((visibility("default"))) int MPI_Comm_get_attr(MPI_Comm comm, int comm_keyval, void *attribute_val, int *flag);
    __attribute__((visibility("default"))) int MPI_Comm_get_errhandler(MPI_Comm comm, MPI_Errhandler *erhandler);
    __attribute__((visibility("default"))) int MPI_Comm_get_name(MPI_Comm comm, char *comm_name, int *resultlen);
    __attribute__((visibility("default"))) int MPI_Comm_get_parent(MPI_Comm *parent);
    __attribute__((visibility("default"))) int MPI_Comm_group(MPI_Comm comm, MPI_Group *group);
    __attribute__((visibility("default"))) int MPI_Comm_join(int fd, MPI_Comm *intercomm);
    __attribute__((visibility("default"))) int MPI_Comm_rank(MPI_Comm comm, int *rank);
    __attribute__((visibility("default"))) int MPI_Comm_remote_group(MPI_Comm comm, MPI_Group *group);
    __attribute__((visibility("default"))) int MPI_Comm_remote_size(MPI_Comm comm, int *size);
    __attribute__((visibility("default"))) int MPI_Comm_set_attr(MPI_Comm comm, int comm_keyval, void *attribute_val);
    __attribute__((visibility("default"))) int MPI_Comm_set_errhandler(MPI_Comm comm, MPI_Errhandler errhandler);
    __attribute__((visibility("default"))) int MPI_Comm_set_name(MPI_Comm comm, char *comm_name);
    __attribute__((visibility("default"))) int MPI_Comm_size(MPI_Comm comm, int *size);
    __attribute__((visibility("default"))) int MPI_Comm_spawn(char *command, char **argv, int maxprocs, MPI_Info info, int root, MPI_Comm comm, MPI_Comm *intercomm, int *array_of_errcodes);
    __attribute__((visibility("default"))) int MPI_Comm_spawn_multiple(int count, char **array_of_commands, char ***array_of_argv, int *array_of_maxprocs, MPI_Info *array_of_info, int root, MPI_Comm comm, MPI_Comm *intercomm, int *array_of_errcodes);
    __attribute__((visibility("default"))) int MPI_Comm_split(MPI_Comm comm, int color, int key, MPI_Comm *newcomm);
    __attribute__((visibility("default"))) int MPI_Comm_test_inter(MPI_Comm comm, int *flag);
    __attribute__((visibility("default"))) int MPI_Dims_create(int nnodes, int ndims, int *dims);
    __attribute__((visibility("default"))) int MPI_Errhandler_c2f(MPI_Errhandler errhandler);
    __attribute__((visibility("default"))) int MPI_Errhandler_create(MPI_Handler_function *function, MPI_Errhandler *errhandler);
    __attribute__((visibility("default"))) MPI_Errhandler MPI_Errhandler_f2c(int errhandler);
    __attribute__((visibility("default"))) int MPI_Errhandler_free(MPI_Errhandler *errhandler);
    __attribute__((visibility("default"))) int MPI_Errhandler_get(MPI_Comm comm, MPI_Errhandler *errhandler);
    __attribute__((visibility("default"))) int MPI_Errhandler_set(MPI_Comm comm, MPI_Errhandler errhandler);
    __attribute__((visibility("default"))) int MPI_Error_class(int errorcode, int *errorclass);
    __attribute__((visibility("default"))) int MPI_Error_string(int errorcode, char *string, int *resultlen);
    __attribute__((visibility("default"))) int MPI_Exscan(void *sendbuf, void *recvbuf, int count, MPI_Datatype datatype, MPI_Op op, MPI_Comm comm);
    __attribute__((visibility("default"))) int MPI_File_c2f(MPI_File file);
    __attribute__((visibility("default"))) MPI_File MPI_File_f2c(int file);
    __attribute__((visibility("default"))) int MPI_File_call_errhandler(MPI_File fh, int errorcode);
    __attribute__((visibility("default"))) int MPI_File_create_errhandler(MPI_File_errhandler_function *function, MPI_Errhandler *errhandler);
    __attribute__((visibility("default"))) int MPI_File_set_errhandler(MPI_File file, MPI_Errhandler errhandler);
    __attribute__((visibility("default"))) int MPI_File_get_errhandler(MPI_File file, MPI_Errhandler *errhandler);
    __attribute__((visibility("default"))) int MPI_File_open(MPI_Comm comm, char *filename, int amode, MPI_Info info, MPI_File *fh);
    __attribute__((visibility("default"))) int MPI_File_close(MPI_File *fh);
    __attribute__((visibility("default"))) int MPI_File_delete(char *filename, MPI_Info info);
    __attribute__((visibility("default"))) int MPI_File_set_size(MPI_File fh, MPI_Offset size);
    __attribute__((visibility("default"))) int MPI_File_preallocate(MPI_File fh, MPI_Offset size);
    __attribute__((visibility("default"))) int MPI_File_get_size(MPI_File fh, MPI_Offset *size);
    __attribute__((visibility("default"))) int MPI_File_get_group(MPI_File fh, MPI_Group *group);
    __attribute__((visibility("default"))) int MPI_File_get_amode(MPI_File fh, int *amode);
    __attribute__((visibility("default"))) int MPI_File_set_info(MPI_File fh, MPI_Info info);
    __attribute__((visibility("default"))) int MPI_File_get_info(MPI_File fh, MPI_Info *info_used);
    __attribute__((visibility("default"))) int MPI_File_set_view(MPI_File fh, MPI_Offset disp, MPI_Datatype etype, MPI_Datatype filetype, char *datarep, MPI_Info info);
    __attribute__((visibility("default"))) int MPI_File_get_view(MPI_File fh, MPI_Offset *disp, MPI_Datatype *etype, MPI_Datatype *filetype, char *datarep);
    __attribute__((visibility("default"))) int MPI_File_read_at(MPI_File fh, MPI_Offset offset, void *buf, int count, MPI_Datatype datatype, MPI_Status *status);
    __attribute__((visibility("default"))) int MPI_File_read_at_all(MPI_File fh, MPI_Offset offset, void *buf, int count, MPI_Datatype datatype, MPI_Status *status);
    __attribute__((visibility("default"))) int MPI_File_write_at(MPI_File fh, MPI_Offset offset, void *buf, int count, MPI_Datatype datatype, MPI_Status *status);
    __attribute__((visibility("default"))) int MPI_File_write_at_all(MPI_File fh, MPI_Offset offset, void *buf, int count, MPI_Datatype datatype, MPI_Status *status);
    __attribute__((visibility("default"))) int MPI_File_iread_at(MPI_File fh, MPI_Offset offset, void *buf, int count, MPI_Datatype datatype, MPI_Request *request);
    __attribute__((visibility("default"))) int MPI_File_iwrite_at(MPI_File fh, MPI_Offset offset, void *buf, int count, MPI_Datatype datatype, MPI_Request *request);
    __attribute__((visibility("default"))) int MPI_File_read(MPI_File fh, void *buf, int count, MPI_Datatype datatype, MPI_Status *status);
    __attribute__((visibility("default"))) int MPI_File_read_all(MPI_File fh, void *buf, int count, MPI_Datatype datatype, MPI_Status *status);
    __attribute__((visibility("default"))) int MPI_File_write(MPI_File fh, void *buf, int count, MPI_Datatype datatype, MPI_Status *status);
    __attribute__((visibility("default"))) int MPI_File_write_all(MPI_File fh, void *buf, int count, MPI_Datatype datatype, MPI_Status *status);
    __attribute__((visibility("default"))) int MPI_File_iread(MPI_File fh, void *buf, int count, MPI_Datatype datatype, MPI_Request *request);
    __attribute__((visibility("default"))) int MPI_File_iwrite(MPI_File fh, void *buf, int count, MPI_Datatype datatype, MPI_Request *request);
    __attribute__((visibility("default"))) int MPI_File_seek(MPI_File fh, MPI_Offset offset, int whence);
    __attribute__((visibility("default"))) int MPI_File_get_position(MPI_File fh, MPI_Offset *offset);
    __attribute__((visibility("default"))) int MPI_File_get_byte_offset(MPI_File fh, MPI_Offset offset, MPI_Offset *disp);
    __attribute__((visibility("default"))) int MPI_File_read_shared(MPI_File fh, void *buf, int count, MPI_Datatype datatype, MPI_Status *status);
    __attribute__((visibility("default"))) int MPI_File_write_shared(MPI_File fh, void *buf, int count, MPI_Datatype datatype, MPI_Status *status);
    __attribute__((visibility("default"))) int MPI_File_iread_shared(MPI_File fh, void *buf, int count, MPI_Datatype datatype, MPI_Request *request);
    __attribute__((visibility("default"))) int MPI_File_iwrite_shared(MPI_File fh, void *buf, int count, MPI_Datatype datatype, MPI_Request *request);
    __attribute__((visibility("default"))) int MPI_File_read_ordered(MPI_File fh, void *buf, int count, MPI_Datatype datatype, MPI_Status *status);
    __attribute__((visibility("default"))) int MPI_File_write_ordered(MPI_File fh, void *buf, int count, MPI_Datatype datatype, MPI_Status *status);
    __attribute__((visibility("default"))) int MPI_File_seek_shared(MPI_File fh, MPI_Offset offset, int whence);
    __attribute__((visibility("default"))) int MPI_File_get_position_shared(MPI_File fh, MPI_Offset *offset);
    __attribute__((visibility("default"))) int MPI_File_read_at_all_begin(MPI_File fh, MPI_Offset offset, void *buf, int count, MPI_Datatype datatype);
    __attribute__((visibility("default"))) int MPI_File_read_at_all_end(MPI_File fh, void *buf, MPI_Status *status);
    __attribute__((visibility("default"))) int MPI_File_write_at_all_begin(MPI_File fh, MPI_Offset offset, void *buf, int count, MPI_Datatype datatype);
    __attribute__((visibility("default"))) int MPI_File_write_at_all_end(MPI_File fh, void *buf, MPI_Status *status);
    __attribute__((visibility("default"))) int MPI_File_read_all_begin(MPI_File fh, void *buf, int count, MPI_Datatype datatype);
    __attribute__((visibility("default"))) int MPI_File_read_all_end(MPI_File fh, void *buf, MPI_Status *status);
    __attribute__((visibility("default"))) int MPI_File_write_all_begin(MPI_File fh, void *buf, int count, MPI_Datatype datatype);
    __attribute__((visibility("default"))) int MPI_File_write_all_end(MPI_File fh, void *buf, MPI_Status *status);
    __attribute__((visibility("default"))) int MPI_File_read_ordered_begin(MPI_File fh, void *buf, int count, MPI_Datatype datatype);
    __attribute__((visibility("default"))) int MPI_File_read_ordered_end(MPI_File fh, void *buf, MPI_Status *status);
    __attribute__((visibility("default"))) int MPI_File_write_ordered_begin(MPI_File fh, void *buf, int count, MPI_Datatype datatype);
    __attribute__((visibility("default"))) int MPI_File_write_ordered_end(MPI_File fh, void *buf, MPI_Status *status);
    __attribute__((visibility("default"))) int MPI_File_get_type_extent(MPI_File fh, MPI_Datatype datatype, MPI_Aint *extent);
    __attribute__((visibility("default"))) int MPI_File_set_atomicity(MPI_File fh, int flag);
    __attribute__((visibility("default"))) int MPI_File_get_atomicity(MPI_File fh, int *flag);
    __attribute__((visibility("default"))) int MPI_File_sync(MPI_File fh);
    __attribute__((visibility("default"))) int MPI_Finalize(void);
    __attribute__((visibility("default"))) int MPI_Finalized(int *flag);
    __attribute__((visibility("default"))) int MPI_Free_mem(void *base);
    __attribute__((visibility("default"))) int MPI_Gather(void *sendbuf, int sendcount, MPI_Datatype sendtype, void *recvbuf, int recvcount, MPI_Datatype recvtype, int root, MPI_Comm comm);
    __attribute__((visibility("default"))) int MPI_Gatherv(void *sendbuf, int sendcount, MPI_Datatype sendtype, void *recvbuf, int *recvcounts, int *displs, MPI_Datatype recvtype, int root, MPI_Comm comm);
    __attribute__((visibility("default"))) int MPI_Get_address(void *location, MPI_Aint *address);
    __attribute__((visibility("default"))) int MPI_Get_count(MPI_Status *status, MPI_Datatype datatype, int *count);
    __attribute__((visibility("default"))) int MPI_Get_elements(MPI_Status *status, MPI_Datatype datatype, int *count);
    __attribute__((visibility("default"))) int MPI_Get(void *origin_addr, int origin_count, MPI_Datatype origin_datatype, int target_rank, MPI_Aint target_disp, int target_count, MPI_Datatype target_datatype, MPI_Win win);
    __attribute__((visibility("default"))) int MPI_Get_processor_name(char *name, int *resultlen);
    __attribute__((visibility("default"))) int MPI_Get_version(int *version, int *subversion);
    __attribute__((visibility("default"))) int MPI_Graph_create(MPI_Comm comm_old, int nnodes, int *index, int *edges, int reorder, MPI_Comm *comm_graph);
    __attribute__((visibility("default"))) int MPI_Graph_get(MPI_Comm comm, int maxindex, int maxedges, int *index, int *edges);
    __attribute__((visibility("default"))) int MPI_Graph_map(MPI_Comm comm, int nnodes, int *index, int *edges, int *newrank);
    __attribute__((visibility("default"))) int MPI_Graph_neighbors_count(MPI_Comm comm, int rank, int *nneighbors);
    __attribute__((visibility("default"))) int MPI_Graph_neighbors(MPI_Comm comm, int rank, int maxneighbors, int *neighbors);
    __attribute__((visibility("default"))) int MPI_Graphdims_get(MPI_Comm comm, int *nnodes, int *nedges);
    __attribute__((visibility("default"))) int MPI_Grequest_complete(MPI_Request request);
    __attribute__((visibility("default"))) int MPI_Grequest_start(MPI_Grequest_query_function *query_fn, MPI_Grequest_free_function *free_fn, MPI_Grequest_cancel_function *cancel_fn, void *extra_state, MPI_Request *request);
    __attribute__((visibility("default"))) int MPI_Group_c2f(MPI_Group group);
    __attribute__((visibility("default"))) int MPI_Group_compare(MPI_Group group1, MPI_Group group2, int *result);
    __attribute__((visibility("default"))) int MPI_Group_difference(MPI_Group group1, MPI_Group group2, MPI_Group *newgroup);
    __attribute__((visibility("default"))) int MPI_Group_excl(MPI_Group group, int n, int *ranks, MPI_Group *newgroup);
    __attribute__((visibility("default"))) MPI_Group MPI_Group_f2c(int group);
    __attribute__((visibility("default"))) int MPI_Group_free(MPI_Group *group);
    __attribute__((visibility("default"))) int MPI_Group_incl(MPI_Group group, int n, int *ranks, MPI_Group *newgroup);
    __attribute__((visibility("default"))) int MPI_Group_intersection(MPI_Group group1, MPI_Group group2, MPI_Group *newgroup);
    __attribute__((visibility("default"))) int MPI_Group_range_excl(MPI_Group group, int n, int ranges[][3], MPI_Group *newgroup);
    __attribute__((visibility("default"))) int MPI_Group_range_incl(MPI_Group group, int n, int ranges[][3], MPI_Group *newgroup);
    __attribute__((visibility("default"))) int MPI_Group_rank(MPI_Group group, int *rank);
    __attribute__((visibility("default"))) int MPI_Group_size(MPI_Group group, int *size);
    __attribute__((visibility("default"))) int MPI_Group_translate_ranks(MPI_Group group1, int n, int *ranks1, MPI_Group group2, int *ranks2);
    __attribute__((visibility("default"))) int MPI_Group_union(MPI_Group group1, MPI_Group group2, MPI_Group *newgroup);
    __attribute__((visibility("default"))) int MPI_Ibsend(void *buf, int count, MPI_Datatype datatype, int dest, int tag, MPI_Comm comm, MPI_Request *request);
    __attribute__((visibility("default"))) int MPI_Info_c2f(MPI_Info info);
    __attribute__((visibility("default"))) int MPI_Info_create(MPI_Info *info);
    __attribute__((visibility("default"))) int MPI_Info_delete(MPI_Info info, char *key);
    __attribute__((visibility("default"))) int MPI_Info_dup(MPI_Info info, MPI_Info *newinfo);
    __attribute__((visibility("default"))) MPI_Info MPI_Info_f2c(int info);
    __attribute__((visibility("default"))) int MPI_Info_free(MPI_Info *info);
    __attribute__((visibility("default"))) int MPI_Info_get(MPI_Info info, char *key, int valuelen, char *value, int *flag);
    __attribute__((visibility("default"))) int MPI_Info_get_nkeys(MPI_Info info, int *nkeys);
    __attribute__((visibility("default"))) int MPI_Info_get_nthkey(MPI_Info info, int n, char *key);
    __attribute__((visibility("default"))) int MPI_Info_get_valuelen(MPI_Info info, char *key, int *valuelen, int *flag);
    __attribute__((visibility("default"))) int MPI_Info_set(MPI_Info info, char *key, char *value);
    __attribute__((visibility("default"))) int MPI_Init(int *argc, char ***argv);
    __attribute__((visibility("default"))) int MPI_Initialized(int *flag);
    __attribute__((visibility("default"))) int MPI_Init_thread(int *argc, char ***argv, int required, int *provided);
    __attribute__((visibility("default"))) int MPI_Intercomm_create(MPI_Comm local_comm, int local_leader, MPI_Comm bridge_comm, int remote_leader, int tag, MPI_Comm *newintercomm);
    __attribute__((visibility("default"))) int MPI_Intercomm_merge(MPI_Comm intercomm, int high, MPI_Comm *newintercomm);
    __attribute__((visibility("default"))) int MPI_Iprobe(int source, int tag, MPI_Comm comm, int *flag, MPI_Status *status);
    __attribute__((visibility("default"))) int MPI_Irecv(void *buf, int count, MPI_Datatype datatype, int source, int tag, MPI_Comm comm, MPI_Request *request);
    __attribute__((visibility("default"))) int MPI_Irsend(void *buf, int count, MPI_Datatype datatype, int dest, int tag, MPI_Comm comm, MPI_Request *request);
    __attribute__((visibility("default"))) int MPI_Isend(void *buf, int count, MPI_Datatype datatype, int dest, int tag, MPI_Comm comm, MPI_Request *request);
    __attribute__((visibility("default"))) int MPI_Issend(void *buf, int count, MPI_Datatype datatype, int dest, int tag, MPI_Comm comm, MPI_Request *request);
    __attribute__((visibility("default"))) int MPI_Is_thread_main(int *flag);
    __attribute__((visibility("default"))) int MPI_Keyval_create(MPI_Copy_function *copy_fn, MPI_Delete_function *delete_fn, int *keyval, void *extra_state);
    __attribute__((visibility("default"))) int MPI_Keyval_free(int *keyval);
    __attribute__((visibility("default"))) int MPI_Lookup_name(char *service_name, MPI_Info info, char *port_name);
    __attribute__((visibility("default"))) int MPI_Op_c2f(MPI_Op op);
    __attribute__((visibility("default"))) int MPI_Op_commutative(MPI_Op op, int *commute);
    __attribute__((visibility("default"))) int MPI_Op_create(MPI_User_function *function, int commute, MPI_Op *op);
    __attribute__((visibility("default"))) int MPI_Open_port(MPI_Info info, char *port_name);
    __attribute__((visibility("default"))) MPI_Op MPI_Op_f2c(int op);
    __attribute__((visibility("default"))) int MPI_Op_free(MPI_Op *op);
    __attribute__((visibility("default"))) int MPI_Pack_external(char *datarep, void *inbuf, int incount, MPI_Datatype datatype, void *outbuf, MPI_Aint outsize, MPI_Aint *position);
    __attribute__((visibility("default"))) int MPI_Pack_external_size(char *datarep, int incount, MPI_Datatype datatype, MPI_Aint *size);
    __attribute__((visibility("default"))) int MPI_Pack(void *inbuf, int incount, MPI_Datatype datatype, void *outbuf, int outsize, int *position, MPI_Comm comm);
    __attribute__((visibility("default"))) int MPI_Pack_size(int incount, MPI_Datatype datatype, MPI_Comm comm, int *size);
    __attribute__((visibility("default"))) int MPI_Pcontrol(const int level, ...);
    __attribute__((visibility("default"))) int MPI_Probe(int source, int tag, MPI_Comm comm, MPI_Status *status);
    __attribute__((visibility("default"))) int MPI_Publish_name(char *service_name, MPI_Info info, char *port_name);
    __attribute__((visibility("default"))) int MPI_Put(void *origin_addr, int origin_count, MPI_Datatype origin_datatype, int target_rank, MPI_Aint target_disp, int target_count, MPI_Datatype target_datatype, MPI_Win win);
    __attribute__((visibility("default"))) int MPI_Query_thread(int *provided);
    __attribute__((visibility("default"))) int MPI_Recv_init(void *buf, int count, MPI_Datatype datatype, int source, int tag, MPI_Comm comm, MPI_Request *request);
    __attribute__((visibility("default"))) int MPI_Recv(void *buf, int count, MPI_Datatype datatype, int source, int tag, MPI_Comm comm, MPI_Status *status);
    __attribute__((visibility("default"))) int MPI_Reduce(void *sendbuf, void *recvbuf, int count, MPI_Datatype datatype, MPI_Op op, int root, MPI_Comm comm);
    __attribute__((visibility("default"))) int MPI_Reduce_local(void *inbuf, void *inoutbuf, int count, MPI_Datatype datatype, MPI_Op op);
    __attribute__((visibility("default"))) int MPI_Reduce_scatter(void *sendbuf, void *recvbuf, int *recvcounts, MPI_Datatype datatype, MPI_Op op, MPI_Comm comm);
    __attribute__((visibility("default"))) int MPI_Register_datarep(char *datarep, MPI_Datarep_conversion_function *read_conversion_fn, MPI_Datarep_conversion_function *write_conversion_fn, MPI_Datarep_extent_function *dtype_file_extent_fn, void *extra_state);
    __attribute__((visibility("default"))) int MPI_Request_c2f(MPI_Request request);
    __attribute__((visibility("default"))) MPI_Request MPI_Request_f2c(int request);
    __attribute__((visibility("default"))) int MPI_Request_free(MPI_Request *request);
    __attribute__((visibility("default"))) int MPI_Request_get_status(MPI_Request request, int *flag, MPI_Status *status);
    __attribute__((visibility("default"))) int MPI_Rsend(void *ibuf, int count, MPI_Datatype datatype, int dest, int tag, MPI_Comm comm);
    __attribute__((visibility("default"))) int MPI_Rsend_init(void *buf, int count, MPI_Datatype datatype, int dest, int tag, MPI_Comm comm, MPI_Request *request);
    __attribute__((visibility("default"))) int MPI_Scan(void *sendbuf, void *recvbuf, int count, MPI_Datatype datatype, MPI_Op op, MPI_Comm comm);
    __attribute__((visibility("default"))) int MPI_Scatter(void *sendbuf, int sendcount, MPI_Datatype sendtype, void *recvbuf, int recvcount, MPI_Datatype recvtype, int root, MPI_Comm comm);
    __attribute__((visibility("default"))) int MPI_Scatterv(void *sendbuf, int *sendcounts, int *displs, MPI_Datatype sendtype, void *recvbuf, int recvcount, MPI_Datatype recvtype, int root, MPI_Comm comm);
    __attribute__((visibility("default"))) int MPI_Send_init(void *buf, int count, MPI_Datatype datatype, int dest, int tag, MPI_Comm comm, MPI_Request *request);
    __attribute__((visibility("default"))) int MPI_Send(void *buf, int count, MPI_Datatype datatype, int dest, int tag, MPI_Comm comm);
    __attribute__((visibility("default"))) int MPI_Sendrecv(void *sendbuf, int sendcount, MPI_Datatype sendtype, int dest, int sendtag, void *recvbuf, int recvcount, MPI_Datatype recvtype, int source, int recvtag, MPI_Comm comm, MPI_Status *status);
    __attribute__((visibility("default"))) int MPI_Sendrecv_replace(void *buf, int count, MPI_Datatype datatype, int dest, int sendtag, int source, int recvtag, MPI_Comm comm, MPI_Status *status);
    __attribute__((visibility("default"))) int MPI_Ssend_init(void *buf, int count, MPI_Datatype datatype, int dest, int tag, MPI_Comm comm, MPI_Request *request);
    __attribute__((visibility("default"))) int MPI_Ssend(void *buf, int count, MPI_Datatype datatype, int dest, int tag, MPI_Comm comm);
    __attribute__((visibility("default"))) int MPI_Start(MPI_Request *request);
    __attribute__((visibility("default"))) int MPI_Startall(int count, MPI_Request *array_of_requests);
    __attribute__((visibility("default"))) int MPI_Status_c2f(MPI_Status *c_status, int *f_status);
    __attribute__((visibility("default"))) int MPI_Status_f2c(int *f_status, MPI_Status *c_status);
    __attribute__((visibility("default"))) int MPI_Status_set_cancelled(MPI_Status *status, int flag);
    __attribute__((visibility("default"))) int MPI_Status_set_elements(MPI_Status *status, MPI_Datatype datatype, int count);
    __attribute__((visibility("default"))) int MPI_Testall(int count, MPI_Request array_of_requests[], int *flag, MPI_Status array_of_statuses[]);
    __attribute__((visibility("default"))) int MPI_Testany(int count, MPI_Request array_of_requests[], int *index, int *flag, MPI_Status *status);
    __attribute__((visibility("default"))) int MPI_Test(MPI_Request *request, int *flag, MPI_Status *status);
    __attribute__((visibility("default"))) int MPI_Test_cancelled(MPI_Status *status, int *flag);
    __attribute__((visibility("default"))) int MPI_Testsome(int incount, MPI_Request array_of_requests[], int *outcount, int array_of_indices[], MPI_Status array_of_statuses[]);
    __attribute__((visibility("default"))) int MPI_Topo_test(MPI_Comm comm, int *status);
    __attribute__((visibility("default"))) int MPI_Type_c2f(MPI_Datatype datatype);
    __attribute__((visibility("default"))) int MPI_Type_commit(MPI_Datatype *type);
    __attribute__((visibility("default"))) int MPI_Type_contiguous(int count, MPI_Datatype oldtype, MPI_Datatype *newtype);
    __attribute__((visibility("default"))) int MPI_Type_create_darray(int size, int rank, int ndims, int gsize_array[], int distrib_array[], int darg_array[], int psize_array[], int order, MPI_Datatype oldtype, MPI_Datatype *newtype);
    __attribute__((visibility("default"))) int MPI_Type_create_f90_complex(int p, int r, MPI_Datatype *newtype);
    __attribute__((visibility("default"))) int MPI_Type_create_f90_integer(int r, MPI_Datatype *newtype);
    __attribute__((visibility("default"))) int MPI_Type_create_f90_real(int p, int r, MPI_Datatype *newtype);
    __attribute__((visibility("default"))) int MPI_Type_create_hindexed(int count, int array_of_blocklengths[], MPI_Aint array_of_displacements[], MPI_Datatype oldtype, MPI_Datatype *newtype);
    __attribute__((visibility("default"))) int MPI_Type_create_hvector(int count, int blocklength, MPI_Aint stride, MPI_Datatype oldtype, MPI_Datatype *newtype);
    __attribute__((visibility("default"))) int MPI_Type_create_keyval(MPI_Type_copy_attr_function *type_copy_attr_fn, MPI_Type_delete_attr_function *type_delete_attr_fn, int *type_keyval, void *extra_state);
    __attribute__((visibility("default"))) int MPI_Type_create_indexed_block(int count, int blocklength, int array_of_displacements[], MPI_Datatype oldtype, MPI_Datatype *newtype);
    __attribute__((visibility("default"))) int MPI_Type_create_struct(int count, int array_of_block_lengths[], MPI_Aint array_of_displacements[], MPI_Datatype array_of_types[], MPI_Datatype *newtype);
    __attribute__((visibility("default"))) int MPI_Type_create_subarray(int ndims, int size_array[], int subsize_array[], int start_array[], int order, MPI_Datatype oldtype, MPI_Datatype *newtype);
    __attribute__((visibility("default"))) int MPI_Type_create_resized(MPI_Datatype oldtype, MPI_Aint lb, MPI_Aint extent, MPI_Datatype *newtype);
    __attribute__((visibility("default"))) int MPI_Type_delete_attr(MPI_Datatype type, int type_keyval);
    __attribute__((visibility("default"))) int MPI_Type_dup(MPI_Datatype type, MPI_Datatype *newtype);
    __attribute__((visibility("default"))) int MPI_Type_extent(MPI_Datatype type, MPI_Aint *extent);
    __attribute__((visibility("default"))) int MPI_Type_free(MPI_Datatype *type);
    __attribute__((visibility("default"))) int MPI_Type_free_keyval(int *type_keyval);
    __attribute__((visibility("default"))) MPI_Datatype MPI_Type_f2c(int datatype);
    __attribute__((visibility("default"))) int MPI_Type_get_attr(MPI_Datatype type, int type_keyval, void *attribute_val, int *flag);
    __attribute__((visibility("default"))) int MPI_Type_get_contents(MPI_Datatype mtype, int max_integers, int max_addresses, int max_datatypes, int array_of_integers[], MPI_Aint array_of_addresses[], MPI_Datatype array_of_datatypes[]);
    __attribute__((visibility("default"))) int MPI_Type_get_envelope(MPI_Datatype type, int *num_integers, int *num_addresses, int *num_datatypes, int *combiner);
    __attribute__((visibility("default"))) int MPI_Type_get_extent(MPI_Datatype type, MPI_Aint *lb, MPI_Aint *extent);
    __attribute__((visibility("default"))) int MPI_Type_get_name(MPI_Datatype type, char *type_name, int *resultlen);
    __attribute__((visibility("default"))) int MPI_Type_get_true_extent(MPI_Datatype datatype, MPI_Aint *true_lb, MPI_Aint *true_extent);
    __attribute__((visibility("default"))) int MPI_Type_hindexed(int count, int array_of_blocklengths[], MPI_Aint array_of_displacements[], MPI_Datatype oldtype, MPI_Datatype *newtype);
    __attribute__((visibility("default"))) int MPI_Type_hvector(int count, int blocklength, MPI_Aint stride, MPI_Datatype oldtype, MPI_Datatype *newtype);
    __attribute__((visibility("default"))) int MPI_Type_indexed(int count, int array_of_blocklengths[], int array_of_displacements[], MPI_Datatype oldtype, MPI_Datatype *newtype);
    __attribute__((visibility("default"))) int MPI_Type_lb(MPI_Datatype type, MPI_Aint *lb);
    __attribute__((visibility("default"))) int MPI_Type_match_size(int typeclass, int size, MPI_Datatype *type);
    __attribute__((visibility("default"))) int MPI_Type_set_attr(MPI_Datatype type, int type_keyval, void *attr_val);
    __attribute__((visibility("default"))) int MPI_Type_set_name(MPI_Datatype type, char *type_name);
    __attribute__((visibility("default"))) int MPI_Type_size(MPI_Datatype type, int *size);
    __attribute__((visibility("default"))) int MPI_Type_struct(int count, int array_of_blocklengths[], MPI_Aint array_of_displacements[], MPI_Datatype array_of_types[], MPI_Datatype *newtype);
    __attribute__((visibility("default"))) int MPI_Type_ub(MPI_Datatype mtype, MPI_Aint *ub);
    __attribute__((visibility("default"))) int MPI_Type_vector(int count, int blocklength, int stride, MPI_Datatype oldtype, MPI_Datatype *newtype);
    __attribute__((visibility("default"))) int MPI_Unpack(void *inbuf, int insize, int *position, void *outbuf, int outcount, MPI_Datatype datatype, MPI_Comm comm);
    __attribute__((visibility("default"))) int MPI_Unpublish_name(char *service_name, MPI_Info info, char *port_name);
    __attribute__((visibility("default"))) int MPI_Unpack_external(char *datarep, void *inbuf, MPI_Aint insize, MPI_Aint *position, void *outbuf, int outcount, MPI_Datatype datatype);
    __attribute__((visibility("default"))) int MPI_Waitall(int count, MPI_Request *array_of_requests, MPI_Status *array_of_statuses);
    __attribute__((visibility("default"))) int MPI_Waitany(int count, MPI_Request *array_of_requests, int *index, MPI_Status *status);
    __attribute__((visibility("default"))) int MPI_Wait(MPI_Request *request, MPI_Status *status);
    __attribute__((visibility("default"))) int MPI_Waitsome(int incount, MPI_Request *array_of_requests, int *outcount, int *array_of_indices, MPI_Status *array_of_statuses);
    __attribute__((visibility("default"))) int MPI_Win_c2f(MPI_Win win);
    __attribute__((visibility("default"))) int MPI_Win_call_errhandler(MPI_Win win, int errorcode);
    __attribute__((visibility("default"))) int MPI_Win_complete(MPI_Win win);
    __attribute__((visibility("default"))) int MPI_Win_create(void *base, MPI_Aint size, int disp_unit, MPI_Info info, MPI_Comm comm, MPI_Win *win);
    __attribute__((visibility("default"))) int MPI_Win_create_errhandler(MPI_Win_errhandler_function *function, MPI_Errhandler *errhandler);
    __attribute__((visibility("default"))) int MPI_Win_create_keyval(MPI_Win_copy_attr_function *win_copy_attr_fn, MPI_Win_delete_attr_function *win_delete_attr_fn, int *win_keyval, void *extra_state);
    __attribute__((visibility("default"))) int MPI_Win_delete_attr(MPI_Win win, int win_keyval);
    __attribute__((visibility("default"))) MPI_Win MPI_Win_f2c(int win);
    __attribute__((visibility("default"))) int MPI_Win_fence(int assert, MPI_Win win);
    __attribute__((visibility("default"))) int MPI_Win_free(MPI_Win *win);
    __attribute__((visibility("default"))) int MPI_Win_free_keyval(int *win_keyval);
    __attribute__((visibility("default"))) int MPI_Win_get_attr(MPI_Win win, int win_keyval, void *attribute_val, int *flag);
    __attribute__((visibility("default"))) int MPI_Win_get_errhandler(MPI_Win win, MPI_Errhandler *errhandler);
    __attribute__((visibility("default"))) int MPI_Win_get_group(MPI_Win win, MPI_Group *group);
    __attribute__((visibility("default"))) int MPI_Win_get_name(MPI_Win win, char *win_name, int *resultlen);
    __attribute__((visibility("default"))) int MPI_Win_lock(int lock_type, int rank, int assert, MPI_Win win);
    __attribute__((visibility("default"))) int MPI_Win_post(MPI_Group group, int assert, MPI_Win win);
    __attribute__((visibility("default"))) int MPI_Win_set_attr(MPI_Win win, int win_keyval, void *attribute_val);
    __attribute__((visibility("default"))) int MPI_Win_set_errhandler(MPI_Win win, MPI_Errhandler errhandler);
    __attribute__((visibility("default"))) int MPI_Win_set_name(MPI_Win win, char *win_name);
    __attribute__((visibility("default"))) int MPI_Win_start(MPI_Group group, int assert, MPI_Win win);
    __attribute__((visibility("default"))) int MPI_Win_test(MPI_Win win, int *flag);
    __attribute__((visibility("default"))) int MPI_Win_unlock(int rank, MPI_Win win);
    __attribute__((visibility("default"))) int MPI_Win_wait(MPI_Win win);
    __attribute__((visibility("default"))) double MPI_Wtick(void);
    __attribute__((visibility("default"))) double MPI_Wtime(void);
    __attribute__((visibility("default"))) int PMPI_Abort(MPI_Comm comm, int errorcode);
    __attribute__((visibility("default"))) int PMPI_Accumulate(void *origin_addr, int origin_count, MPI_Datatype origin_datatype, int target_rank, MPI_Aint target_disp, int target_count, MPI_Datatype target_datatype, MPI_Op op, MPI_Win win);
    __attribute__((visibility("default"))) int PMPI_Add_error_class(int *errorclass);
    __attribute__((visibility("default"))) int PMPI_Add_error_code(int errorclass, int *errorcode);
    __attribute__((visibility("default"))) int PMPI_Add_error_string(int errorcode, char *string);
    __attribute__((visibility("default"))) int PMPI_Address(void *location, MPI_Aint *address);
    __attribute__((visibility("default"))) int PMPI_Allgather(void *sendbuf, int sendcount, MPI_Datatype sendtype, void *recvbuf, int recvcount, MPI_Datatype recvtype, MPI_Comm comm);
    __attribute__((visibility("default"))) int PMPI_Allgatherv(void *sendbuf, int sendcount, MPI_Datatype sendtype, void *recvbuf, int *recvcounts, int *displs, MPI_Datatype recvtype, MPI_Comm comm);
    __attribute__((visibility("default"))) int PMPI_Alloc_mem(MPI_Aint size, MPI_Info info, void *baseptr);
    __attribute__((visibility("default"))) int PMPI_Allreduce(void *sendbuf, void *recvbuf, int count, MPI_Datatype datatype, MPI_Op op, MPI_Comm comm);
    __attribute__((visibility("default"))) int PMPI_Alltoall(void *sendbuf, int sendcount, MPI_Datatype sendtype, void *recvbuf, int recvcount, MPI_Datatype recvtype, MPI_Comm comm);
    __attribute__((visibility("default"))) int PMPI_Alltoallv(void *sendbuf, int *sendcounts, int *sdispls, MPI_Datatype sendtype, void *recvbuf, int *recvcounts, int *rdispls, MPI_Datatype recvtype, MPI_Comm comm);
    __attribute__((visibility("default"))) int PMPI_Alltoallw(void *sendbuf, int *sendcounts, int *sdispls, MPI_Datatype *sendtypes, void *recvbuf, int *recvcounts, int *rdispls, MPI_Datatype *recvtypes, MPI_Comm comm);
    __attribute__((visibility("default"))) int PMPI_Attr_delete(MPI_Comm comm, int keyval);
    __attribute__((visibility("default"))) int PMPI_Attr_get(MPI_Comm comm, int keyval, void *attribute_val, int *flag);
    __attribute__((visibility("default"))) int PMPI_Attr_put(MPI_Comm comm, int keyval, void *attribute_val);
    __attribute__((visibility("default"))) int PMPI_Barrier(MPI_Comm comm);
    __attribute__((visibility("default"))) int PMPI_Bcast(void *buffer, int count, MPI_Datatype datatype, int root, MPI_Comm comm);
    __attribute__((visibility("default"))) int PMPI_Bsend(void *buf, int count, MPI_Datatype datatype, int dest, int tag, MPI_Comm comm);
    __attribute__((visibility("default"))) int PMPI_Bsend_init(void *buf, int count, MPI_Datatype datatype, int dest, int tag, MPI_Comm comm, MPI_Request *request);
    __attribute__((visibility("default"))) int PMPI_Buffer_attach(void *buffer, int size);
    __attribute__((visibility("default"))) int PMPI_Buffer_detach(void *buffer, int *size);
    __attribute__((visibility("default"))) int PMPI_Cancel(MPI_Request *request);
    __attribute__((visibility("default"))) int PMPI_Cart_coords(MPI_Comm comm, int rank, int maxdims, int *coords);
    __attribute__((visibility("default"))) int PMPI_Cart_create(MPI_Comm old_comm, int ndims, int *dims, int *periods, int reorder, MPI_Comm *comm_cart);
    __attribute__((visibility("default"))) int PMPI_Cart_get(MPI_Comm comm, int maxdims, int *dims, int *periods, int *coords);
    __attribute__((visibility("default"))) int PMPI_Cart_map(MPI_Comm comm, int ndims, int *dims, int *periods, int *newrank);
    __attribute__((visibility("default"))) int PMPI_Cart_rank(MPI_Comm comm, int *coords, int *rank);
    __attribute__((visibility("default"))) int PMPI_Cart_shift(MPI_Comm comm, int direction, int disp, int *rank_source, int *rank_dest);
    __attribute__((visibility("default"))) int PMPI_Cart_sub(MPI_Comm comm, int *remain_dims, MPI_Comm *new_comm);
    __attribute__((visibility("default"))) int PMPI_Cartdim_get(MPI_Comm comm, int *ndims);
    __attribute__((visibility("default"))) int PMPI_Close_port(char *port_name);
    __attribute__((visibility("default"))) int PMPI_Comm_accept(char *port_name, MPI_Info info, int root, MPI_Comm comm, MPI_Comm *newcomm);
    __attribute__((visibility("default"))) int PMPI_Comm_c2f(MPI_Comm comm);
    __attribute__((visibility("default"))) int PMPI_Comm_call_errhandler(MPI_Comm comm, int errorcode);
    __attribute__((visibility("default"))) int PMPI_Comm_compare(MPI_Comm comm1, MPI_Comm comm2, int *result);
    __attribute__((visibility("default"))) int PMPI_Comm_connect(char *port_name, MPI_Info info, int root, MPI_Comm comm, MPI_Comm *newcomm);
    __attribute__((visibility("default"))) int PMPI_Comm_create_errhandler(MPI_Comm_errhandler_function *function, MPI_Errhandler *errhandler);
    __attribute__((visibility("default"))) int PMPI_Comm_create_keyval(MPI_Comm_copy_attr_function *comm_copy_attr_fn, MPI_Comm_delete_attr_function *comm_delete_attr_fn, int *comm_keyval, void *extra_state);
    __attribute__((visibility("default"))) int PMPI_Comm_create(MPI_Comm comm, MPI_Group group, MPI_Comm *newcomm);
    __attribute__((visibility("default"))) int PMPI_Comm_delete_attr(MPI_Comm comm, int comm_keyval);
    __attribute__((visibility("default"))) int PMPI_Comm_disconnect(MPI_Comm *comm);
    __attribute__((visibility("default"))) int PMPI_Comm_dup(MPI_Comm comm, MPI_Comm *newcomm);
    __attribute__((visibility("default"))) MPI_Comm PMPI_Comm_f2c(int comm);
    __attribute__((visibility("default"))) int PMPI_Comm_free_keyval(int *comm_keyval);
    __attribute__((visibility("default"))) int PMPI_Comm_free(MPI_Comm *comm);
    __attribute__((visibility("default"))) int PMPI_Comm_get_attr(MPI_Comm comm, int comm_keyval, void *attribute_val, int *flag);
    __attribute__((visibility("default"))) int PMPI_Comm_get_errhandler(MPI_Comm comm, MPI_Errhandler *erhandler);
    __attribute__((visibility("default"))) int PMPI_Comm_get_name(MPI_Comm comm, char *comm_name, int *resultlen);
    __attribute__((visibility("default"))) int PMPI_Comm_get_parent(MPI_Comm *parent);
    __attribute__((visibility("default"))) int PMPI_Comm_group(MPI_Comm comm, MPI_Group *group);
    __attribute__((visibility("default"))) int PMPI_Comm_join(int fd, MPI_Comm *intercomm);
    __attribute__((visibility("default"))) int PMPI_Comm_rank(MPI_Comm comm, int *rank);
    __attribute__((visibility("default"))) int PMPI_Comm_remote_group(MPI_Comm comm, MPI_Group *group);
    __attribute__((visibility("default"))) int PMPI_Comm_remote_size(MPI_Comm comm, int *size);
    __attribute__((visibility("default"))) int PMPI_Comm_set_attr(MPI_Comm comm, int comm_keyval, void *attribute_val);
    __attribute__((visibility("default"))) int PMPI_Comm_set_errhandler(MPI_Comm comm, MPI_Errhandler errhandler);
    __attribute__((visibility("default"))) int PMPI_Comm_set_name(MPI_Comm comm, char *comm_name);
    __attribute__((visibility("default"))) int PMPI_Comm_size(MPI_Comm comm, int *size);
    __attribute__((visibility("default"))) int PMPI_Comm_spawn(char *command, char **argv, int maxprocs, MPI_Info info, int root, MPI_Comm comm, MPI_Comm *intercomm, int *array_of_errcodes);
    __attribute__((visibility("default"))) int PMPI_Comm_spawn_multiple(int count, char **array_of_commands, char ***array_of_argv, int *array_of_maxprocs, MPI_Info *array_of_info, int root, MPI_Comm comm, MPI_Comm *intercomm, int *array_of_errcodes);
    __attribute__((visibility("default"))) int PMPI_Comm_split(MPI_Comm comm, int color, int key, MPI_Comm *newcomm);
    __attribute__((visibility("default"))) int PMPI_Comm_test_inter(MPI_Comm comm, int *flag);
    __attribute__((visibility("default"))) int PMPI_Dims_create(int nnodes, int ndims, int *dims);
    __attribute__((visibility("default"))) int PMPI_Errhandler_c2f(MPI_Errhandler errhandler);
    __attribute__((visibility("default"))) int PMPI_Errhandler_create(MPI_Handler_function *function, MPI_Errhandler *errhandler);
    __attribute__((visibility("default"))) MPI_Errhandler PMPI_Errhandler_f2c(int errhandler);
    __attribute__((visibility("default"))) int PMPI_Errhandler_free(MPI_Errhandler *errhandler);
    __attribute__((visibility("default"))) int PMPI_Errhandler_get(MPI_Comm comm, MPI_Errhandler *errhandler);
    __attribute__((visibility("default"))) int PMPI_Errhandler_set(MPI_Comm comm, MPI_Errhandler errhandler);
    __attribute__((visibility("default"))) int PMPI_Error_class(int errorcode, int *errorclass);
    __attribute__((visibility("default"))) int PMPI_Error_string(int errorcode, char *string, int *resultlen);
    __attribute__((visibility("default"))) int PMPI_Exscan(void *sendbuf, void *recvbuf, int count, MPI_Datatype datatype, MPI_Op op, MPI_Comm comm);
    __attribute__((visibility("default"))) int PMPI_File_c2f(MPI_File file);
    __attribute__((visibility("default"))) MPI_File PMPI_File_f2c(int file);
    __attribute__((visibility("default"))) int PMPI_File_call_errhandler(MPI_File fh, int errorcode);
    __attribute__((visibility("default"))) int PMPI_File_create_errhandler(MPI_File_errhandler_function *function, MPI_Errhandler *errhandler);
    __attribute__((visibility("default"))) int PMPI_File_set_errhandler(MPI_File file, MPI_Errhandler errhandler);
    __attribute__((visibility("default"))) int PMPI_File_get_errhandler(MPI_File file, MPI_Errhandler *errhandler);
    __attribute__((visibility("default"))) int PMPI_File_open(MPI_Comm comm, char *filename, int amode, MPI_Info info, MPI_File *fh);
    __attribute__((visibility("default"))) int PMPI_File_close(MPI_File *fh);
    __attribute__((visibility("default"))) int PMPI_File_delete(char *filename, MPI_Info info);
    __attribute__((visibility("default"))) int PMPI_File_set_size(MPI_File fh, MPI_Offset size);
    __attribute__((visibility("default"))) int PMPI_File_preallocate(MPI_File fh, MPI_Offset size);
    __attribute__((visibility("default"))) int PMPI_File_get_size(MPI_File fh, MPI_Offset *size);
    __attribute__((visibility("default"))) int PMPI_File_get_group(MPI_File fh, MPI_Group *group);
    __attribute__((visibility("default"))) int PMPI_File_get_amode(MPI_File fh, int *amode);
    __attribute__((visibility("default"))) int PMPI_File_set_info(MPI_File fh, MPI_Info info);
    __attribute__((visibility("default"))) int PMPI_File_get_info(MPI_File fh, MPI_Info *info_used);
    __attribute__((visibility("default"))) int PMPI_File_set_view(MPI_File fh, MPI_Offset disp, MPI_Datatype etype, MPI_Datatype filetype, char *datarep, MPI_Info info);
    __attribute__((visibility("default"))) int PMPI_File_get_view(MPI_File fh, MPI_Offset *disp, MPI_Datatype *etype, MPI_Datatype *filetype, char *datarep);
    __attribute__((visibility("default"))) int PMPI_File_read_at(MPI_File fh, MPI_Offset offset, void *buf, int count, MPI_Datatype datatype, MPI_Status *status);
    __attribute__((visibility("default"))) int PMPI_File_read_at_all(MPI_File fh, MPI_Offset offset, void *buf, int count, MPI_Datatype datatype, MPI_Status *status);
    __attribute__((visibility("default"))) int PMPI_File_write_at(MPI_File fh, MPI_Offset offset, void *buf, int count, MPI_Datatype datatype, MPI_Status *status);
    __attribute__((visibility("default"))) int PMPI_File_write_at_all(MPI_File fh, MPI_Offset offset, void *buf, int count, MPI_Datatype datatype, MPI_Status *status);
    __attribute__((visibility("default"))) int PMPI_File_iread_at(MPI_File fh, MPI_Offset offset, void *buf, int count, MPI_Datatype datatype, MPI_Request *request);
    __attribute__((visibility("default"))) int PMPI_File_iwrite_at(MPI_File fh, MPI_Offset offset, void *buf, int count, MPI_Datatype datatype, MPI_Request *request);
    __attribute__((visibility("default"))) int PMPI_File_read(MPI_File fh, void *buf, int count, MPI_Datatype datatype, MPI_Status *status);
    __attribute__((visibility("default"))) int PMPI_File_read_all(MPI_File fh, void *buf, int count, MPI_Datatype datatype, MPI_Status *status);
    __attribute__((visibility("default"))) int PMPI_File_write(MPI_File fh, void *buf, int count, MPI_Datatype datatype, MPI_Status *status);
    __attribute__((visibility("default"))) int PMPI_File_write_all(MPI_File fh, void *buf, int count, MPI_Datatype datatype, MPI_Status *status);
    __attribute__((visibility("default"))) int PMPI_File_iread(MPI_File fh, void *buf, int count, MPI_Datatype datatype, MPI_Request *request);
    __attribute__((visibility("default"))) int PMPI_File_iwrite(MPI_File fh, void *buf, int count, MPI_Datatype datatype, MPI_Request *request);
    __attribute__((visibility("default"))) int PMPI_File_seek(MPI_File fh, MPI_Offset offset, int whence);
    __attribute__((visibility("default"))) int PMPI_File_get_position(MPI_File fh, MPI_Offset *offset);
    __attribute__((visibility("default"))) int PMPI_File_get_byte_offset(MPI_File fh, MPI_Offset offset, MPI_Offset *disp);
    __attribute__((visibility("default"))) int PMPI_File_read_shared(MPI_File fh, void *buf, int count, MPI_Datatype datatype, MPI_Status *status);
    __attribute__((visibility("default"))) int PMPI_File_write_shared(MPI_File fh, void *buf, int count, MPI_Datatype datatype, MPI_Status *status);
    __attribute__((visibility("default"))) int PMPI_File_iread_shared(MPI_File fh, void *buf, int count, MPI_Datatype datatype, MPI_Request *request);
    __attribute__((visibility("default"))) int PMPI_File_iwrite_shared(MPI_File fh, void *buf, int count, MPI_Datatype datatype, MPI_Request *request);
    __attribute__((visibility("default"))) int PMPI_File_read_ordered(MPI_File fh, void *buf, int count, MPI_Datatype datatype, MPI_Status *status);
    __attribute__((visibility("default"))) int PMPI_File_write_ordered(MPI_File fh, void *buf, int count, MPI_Datatype datatype, MPI_Status *status);
    __attribute__((visibility("default"))) int PMPI_File_seek_shared(MPI_File fh, MPI_Offset offset, int whence);
    __attribute__((visibility("default"))) int PMPI_File_get_position_shared(MPI_File fh, MPI_Offset *offset);
    __attribute__((visibility("default"))) int PMPI_File_read_at_all_begin(MPI_File fh, MPI_Offset offset, void *buf, int count, MPI_Datatype datatype);
    __attribute__((visibility("default"))) int PMPI_File_read_at_all_end(MPI_File fh, void *buf, MPI_Status *status);
    __attribute__((visibility("default"))) int PMPI_File_write_at_all_begin(MPI_File fh, MPI_Offset offset, void *buf, int count, MPI_Datatype datatype);
    __attribute__((visibility("default"))) int PMPI_File_write_at_all_end(MPI_File fh, void *buf, MPI_Status *status);
    __attribute__((visibility("default"))) int PMPI_File_read_all_begin(MPI_File fh, void *buf, int count, MPI_Datatype datatype);
    __attribute__((visibility("default"))) int PMPI_File_read_all_end(MPI_File fh, void *buf, MPI_Status *status);
    __attribute__((visibility("default"))) int PMPI_File_write_all_begin(MPI_File fh, void *buf, int count, MPI_Datatype datatype);
    __attribute__((visibility("default"))) int PMPI_File_write_all_end(MPI_File fh, void *buf, MPI_Status *status);
    __attribute__((visibility("default"))) int PMPI_File_read_ordered_begin(MPI_File fh, void *buf, int count, MPI_Datatype datatype);
    __attribute__((visibility("default"))) int PMPI_File_read_ordered_end(MPI_File fh, void *buf, MPI_Status *status);
    __attribute__((visibility("default"))) int PMPI_File_write_ordered_begin(MPI_File fh, void *buf, int count, MPI_Datatype datatype);
    __attribute__((visibility("default"))) int PMPI_File_write_ordered_end(MPI_File fh, void *buf, MPI_Status *status);
    __attribute__((visibility("default"))) int PMPI_File_get_type_extent(MPI_File fh, MPI_Datatype datatype, MPI_Aint *extent);
    __attribute__((visibility("default"))) int PMPI_File_set_atomicity(MPI_File fh, int flag);
    __attribute__((visibility("default"))) int PMPI_File_get_atomicity(MPI_File fh, int *flag);
    __attribute__((visibility("default"))) int PMPI_File_sync(MPI_File fh);
    __attribute__((visibility("default"))) int PMPI_Finalize(void);
    __attribute__((visibility("default"))) int PMPI_Finalized(int *flag);
    __attribute__((visibility("default"))) int PMPI_Free_mem(void *base);
    __attribute__((visibility("default"))) int PMPI_Gather(void *sendbuf, int sendcount, MPI_Datatype sendtype, void *recvbuf, int recvcount, MPI_Datatype recvtype, int root, MPI_Comm comm);
    __attribute__((visibility("default"))) int PMPI_Gatherv(void *sendbuf, int sendcount, MPI_Datatype sendtype, void *recvbuf, int *recvcounts, int *displs, MPI_Datatype recvtype, int root, MPI_Comm comm);
    __attribute__((visibility("default"))) int PMPI_Get_address(void *location, MPI_Aint *address);
    __attribute__((visibility("default"))) int PMPI_Get_count(MPI_Status *status, MPI_Datatype datatype, int *count);
    __attribute__((visibility("default"))) int PMPI_Get_elements(MPI_Status *status, MPI_Datatype datatype, int *count);
    __attribute__((visibility("default"))) int PMPI_Get(void *origin_addr, int origin_count, MPI_Datatype origin_datatype, int target_rank, MPI_Aint target_disp, int target_count, MPI_Datatype target_datatype, MPI_Win win);
    __attribute__((visibility("default"))) int PMPI_Get_processor_name(char *name, int *resultlen);
    __attribute__((visibility("default"))) int PMPI_Get_version(int *version, int *subversion);
    __attribute__((visibility("default"))) int PMPI_Graph_create(MPI_Comm comm_old, int nnodes, int *index, int *edges, int reorder, MPI_Comm *comm_graph);
    __attribute__((visibility("default"))) int PMPI_Graph_get(MPI_Comm comm, int maxindex, int maxedges, int *index, int *edges);
    __attribute__((visibility("default"))) int PMPI_Graph_map(MPI_Comm comm, int nnodes, int *index, int *edges, int *newrank);
    __attribute__((visibility("default"))) int PMPI_Graph_neighbors_count(MPI_Comm comm, int rank, int *nneighbors);
    __attribute__((visibility("default"))) int PMPI_Graph_neighbors(MPI_Comm comm, int rank, int maxneighbors, int *neighbors);
    __attribute__((visibility("default"))) int PMPI_Graphdims_get(MPI_Comm comm, int *nnodes, int *nedges);
    __attribute__((visibility("default"))) int PMPI_Grequest_complete(MPI_Request request);
    __attribute__((visibility("default"))) int PMPI_Grequest_start(MPI_Grequest_query_function *query_fn, MPI_Grequest_free_function *free_fn, MPI_Grequest_cancel_function *cancel_fn, void *extra_state, MPI_Request *request);
    __attribute__((visibility("default"))) int PMPI_Group_c2f(MPI_Group group);
    __attribute__((visibility("default"))) int PMPI_Group_compare(MPI_Group group1, MPI_Group group2, int *result);
    __attribute__((visibility("default"))) int PMPI_Group_difference(MPI_Group group1, MPI_Group group2, MPI_Group *newgroup);
    __attribute__((visibility("default"))) int PMPI_Group_excl(MPI_Group group, int n, int *ranks, MPI_Group *newgroup);
    __attribute__((visibility("default"))) MPI_Group PMPI_Group_f2c(int group);
    __attribute__((visibility("default"))) int PMPI_Group_free(MPI_Group *group);
    __attribute__((visibility("default"))) int PMPI_Group_incl(MPI_Group group, int n, int *ranks, MPI_Group *newgroup);
    __attribute__((visibility("default"))) int PMPI_Group_intersection(MPI_Group group1, MPI_Group group2, MPI_Group *newgroup);
    __attribute__((visibility("default"))) int PMPI_Group_range_excl(MPI_Group group, int n, int ranges[][3], MPI_Group *newgroup);
    __attribute__((visibility("default"))) int PMPI_Group_range_incl(MPI_Group group, int n, int ranges[][3], MPI_Group *newgroup);
    __attribute__((visibility("default"))) int PMPI_Group_rank(MPI_Group group, int *rank);
    __attribute__((visibility("default"))) int PMPI_Group_size(MPI_Group group, int *size);
    __attribute__((visibility("default"))) int PMPI_Group_translate_ranks(MPI_Group group1, int n, int *ranks1, MPI_Group group2, int *ranks2);
    __attribute__((visibility("default"))) int PMPI_Group_union(MPI_Group group1, MPI_Group group2, MPI_Group *newgroup);
    __attribute__((visibility("default"))) int PMPI_Ibsend(void *buf, int count, MPI_Datatype datatype, int dest, int tag, MPI_Comm comm, MPI_Request *request);
    __attribute__((visibility("default"))) int PMPI_Info_c2f(MPI_Info info);
    __attribute__((visibility("default"))) int PMPI_Info_create(MPI_Info *info);
    __attribute__((visibility("default"))) int PMPI_Info_delete(MPI_Info info, char *key);
    __attribute__((visibility("default"))) int PMPI_Info_dup(MPI_Info info, MPI_Info *newinfo);
    __attribute__((visibility("default"))) MPI_Info PMPI_Info_f2c(int info);
    __attribute__((visibility("default"))) int PMPI_Info_free(MPI_Info *info);
    __attribute__((visibility("default"))) int PMPI_Info_get(MPI_Info info, char *key, int valuelen, char *value, int *flag);
    __attribute__((visibility("default"))) int PMPI_Info_get_nkeys(MPI_Info info, int *nkeys);
    __attribute__((visibility("default"))) int PMPI_Info_get_nthkey(MPI_Info info, int n, char *key);
    __attribute__((visibility("default"))) int PMPI_Info_get_valuelen(MPI_Info info, char *key, int *valuelen, int *flag);
    __attribute__((visibility("default"))) int PMPI_Info_set(MPI_Info info, char *key, char *value);
    __attribute__((visibility("default"))) int PMPI_Init(int *argc, char ***argv);
    __attribute__((visibility("default"))) int PMPI_Initialized(int *flag);
    __attribute__((visibility("default"))) int PMPI_Init_thread(int *argc, char ***argv, int required, int *provided);
    __attribute__((visibility("default"))) int PMPI_Intercomm_create(MPI_Comm local_comm, int local_leader, MPI_Comm bridge_comm, int remote_leader, int tag, MPI_Comm *newintercomm);
    __attribute__((visibility("default"))) int PMPI_Intercomm_merge(MPI_Comm intercomm, int high, MPI_Comm *newintercomm);
    __attribute__((visibility("default"))) int PMPI_Iprobe(int source, int tag, MPI_Comm comm, int *flag, MPI_Status *status);
    __attribute__((visibility("default"))) int PMPI_Irecv(void *buf, int count, MPI_Datatype datatype, int source, int tag, MPI_Comm comm, MPI_Request *request);
    __attribute__((visibility("default"))) int PMPI_Irsend(void *buf, int count, MPI_Datatype datatype, int dest, int tag, MPI_Comm comm, MPI_Request *request);
    __attribute__((visibility("default"))) int PMPI_Isend(void *buf, int count, MPI_Datatype datatype, int dest, int tag, MPI_Comm comm, MPI_Request *request);
    __attribute__((visibility("default"))) int PMPI_Issend(void *buf, int count, MPI_Datatype datatype, int dest, int tag, MPI_Comm comm, MPI_Request *request);
    __attribute__((visibility("default"))) int PMPI_Is_thread_main(int *flag);
    __attribute__((visibility("default"))) int PMPI_Keyval_create(MPI_Copy_function *copy_fn, MPI_Delete_function *delete_fn, int *keyval, void *extra_state);
    __attribute__((visibility("default"))) int PMPI_Keyval_free(int *keyval);
    __attribute__((visibility("default"))) int PMPI_Lookup_name(char *service_name, MPI_Info info, char *port_name);
    __attribute__((visibility("default"))) int PMPI_Op_c2f(MPI_Op op);
    __attribute__((visibility("default"))) int PMPI_Op_commutative(MPI_Op op, int *commute);
    __attribute__((visibility("default"))) int PMPI_Op_create(MPI_User_function *function, int commute, MPI_Op *op);
    __attribute__((visibility("default"))) int PMPI_Open_port(MPI_Info info, char *port_name);
    __attribute__((visibility("default"))) MPI_Op PMPI_Op_f2c(int op);
    __attribute__((visibility("default"))) int PMPI_Op_free(MPI_Op *op);
    __attribute__((visibility("default"))) int PMPI_Pack_external(char *datarep, void *inbuf, int incount, MPI_Datatype datatype, void *outbuf, MPI_Aint outsize, MPI_Aint *position);
    __attribute__((visibility("default"))) int PMPI_Pack_external_size(char *datarep, int incount, MPI_Datatype datatype, MPI_Aint *size);
    __attribute__((visibility("default"))) int PMPI_Pack(void *inbuf, int incount, MPI_Datatype datatype, void *outbuf, int outsize, int *position, MPI_Comm comm);
    __attribute__((visibility("default"))) int PMPI_Pack_size(int incount, MPI_Datatype datatype, MPI_Comm comm, int *size);
    __attribute__((visibility("default"))) int PMPI_Pcontrol(const int level, ...);
    __attribute__((visibility("default"))) int PMPI_Probe(int source, int tag, MPI_Comm comm, MPI_Status *status);
    __attribute__((visibility("default"))) int PMPI_Publish_name(char *service_name, MPI_Info info, char *port_name);
    __attribute__((visibility("default"))) int PMPI_Put(void *origin_addr, int origin_count, MPI_Datatype origin_datatype, int target_rank, MPI_Aint target_disp, int target_count, MPI_Datatype target_datatype, MPI_Win win);
    __attribute__((visibility("default"))) int PMPI_Query_thread(int *provided);
    __attribute__((visibility("default"))) int PMPI_Recv_init(void *buf, int count, MPI_Datatype datatype, int source, int tag, MPI_Comm comm, MPI_Request *request);
    __attribute__((visibility("default"))) int PMPI_Recv(void *buf, int count, MPI_Datatype datatype, int source, int tag, MPI_Comm comm, MPI_Status *status);
    __attribute__((visibility("default"))) int PMPI_Reduce(void *sendbuf, void *recvbuf, int count, MPI_Datatype datatype, MPI_Op op, int root, MPI_Comm comm);
    __attribute__((visibility("default"))) int PMPI_Reduce_local(void *inbuf, void *inoutbuf, int count, MPI_Datatype datatype, MPI_Op);
    __attribute__((visibility("default"))) int PMPI_Reduce_scatter(void *sendbuf, void *recvbuf, int *recvcounts, MPI_Datatype datatype, MPI_Op op, MPI_Comm comm);
    __attribute__((visibility("default"))) int PMPI_Register_datarep(char *datarep, MPI_Datarep_conversion_function *read_conversion_fn, MPI_Datarep_conversion_function *write_conversion_fn, MPI_Datarep_extent_function *dtype_file_extent_fn, void *extra_state);
    __attribute__((visibility("default"))) int PMPI_Request_c2f(MPI_Request request);
    __attribute__((visibility("default"))) MPI_Request PMPI_Request_f2c(int request);
    __attribute__((visibility("default"))) int PMPI_Request_free(MPI_Request *request);
    __attribute__((visibility("default"))) int PMPI_Request_get_status(MPI_Request request, int *flag, MPI_Status *status);
    __attribute__((visibility("default"))) int PMPI_Rsend(void *ibuf, int count, MPI_Datatype datatype, int dest, int tag, MPI_Comm comm);
    __attribute__((visibility("default"))) int PMPI_Rsend_init(void *buf, int count, MPI_Datatype datatype, int dest, int tag, MPI_Comm comm, MPI_Request *request);
    __attribute__((visibility("default"))) int PMPI_Scan(void *sendbuf, void *recvbuf, int count, MPI_Datatype datatype, MPI_Op op, MPI_Comm comm);
    __attribute__((visibility("default"))) int PMPI_Scatter(void *sendbuf, int sendcount, MPI_Datatype sendtype, void *recvbuf, int recvcount, MPI_Datatype recvtype, int root, MPI_Comm comm);
    __attribute__((visibility("default"))) int PMPI_Scatterv(void *sendbuf, int *sendcounts, int *displs, MPI_Datatype sendtype, void *recvbuf, int recvcount, MPI_Datatype recvtype, int root, MPI_Comm comm);
    __attribute__((visibility("default"))) int PMPI_Send_init(void *buf, int count, MPI_Datatype datatype, int dest, int tag, MPI_Comm comm, MPI_Request *request);
    __attribute__((visibility("default"))) int PMPI_Send(void *buf, int count, MPI_Datatype datatype, int dest, int tag, MPI_Comm comm);
    __attribute__((visibility("default"))) int PMPI_Sendrecv(void *sendbuf, int sendcount, MPI_Datatype sendtype, int dest, int sendtag, void *recvbuf, int recvcount, MPI_Datatype recvtype, int source, int recvtag, MPI_Comm comm, MPI_Status *status);
    __attribute__((visibility("default"))) int PMPI_Sendrecv_replace(void *buf, int count, MPI_Datatype datatype, int dest, int sendtag, int source, int recvtag, MPI_Comm comm, MPI_Status *status);
    __attribute__((visibility("default"))) int PMPI_Ssend_init(void *buf, int count, MPI_Datatype datatype, int dest, int tag, MPI_Comm comm, MPI_Request *request);
    __attribute__((visibility("default"))) int PMPI_Ssend(void *buf, int count, MPI_Datatype datatype, int dest, int tag, MPI_Comm comm);
    __attribute__((visibility("default"))) int PMPI_Start(MPI_Request *request);
    __attribute__((visibility("default"))) int PMPI_Startall(int count, MPI_Request *array_of_requests);
    __attribute__((visibility("default"))) int PMPI_Status_c2f(MPI_Status *c_status, int *f_status);
    __attribute__((visibility("default"))) int PMPI_Status_f2c(int *f_status, MPI_Status *c_status);
    __attribute__((visibility("default"))) int PMPI_Status_set_cancelled(MPI_Status *status, int flag);
    __attribute__((visibility("default"))) int PMPI_Status_set_elements(MPI_Status *status, MPI_Datatype datatype, int count);
    __attribute__((visibility("default"))) int PMPI_Testall(int count, MPI_Request array_of_requests[], int *flag, MPI_Status array_of_statuses[]);
    __attribute__((visibility("default"))) int PMPI_Testany(int count, MPI_Request array_of_requests[], int *index, int *flag, MPI_Status *status);
    __attribute__((visibility("default"))) int PMPI_Test(MPI_Request *request, int *flag, MPI_Status *status);
    __attribute__((visibility("default"))) int PMPI_Test_cancelled(MPI_Status *status, int *flag);
    __attribute__((visibility("default"))) int PMPI_Testsome(int incount, MPI_Request array_of_requests[], int *outcount, int array_of_indices[], MPI_Status array_of_statuses[]);
    __attribute__((visibility("default"))) int PMPI_Topo_test(MPI_Comm comm, int *status);
    __attribute__((visibility("default"))) int PMPI_Type_c2f(MPI_Datatype datatype);
    __attribute__((visibility("default"))) int PMPI_Type_commit(MPI_Datatype *type);
    __attribute__((visibility("default"))) int PMPI_Type_contiguous(int count, MPI_Datatype oldtype, MPI_Datatype *newtype);
    __attribute__((visibility("default"))) int PMPI_Type_create_darray(int size, int rank, int ndims, int gsize_array[], int distrib_array[], int darg_array[], int psize_array[], int order, MPI_Datatype oldtype, MPI_Datatype *newtype);
    __attribute__((visibility("default"))) int PMPI_Type_create_f90_complex(int p, int r, MPI_Datatype *newtype);
    __attribute__((visibility("default"))) int PMPI_Type_create_f90_integer(int r, MPI_Datatype *newtype);
    __attribute__((visibility("default"))) int PMPI_Type_create_f90_real(int p, int r, MPI_Datatype *newtype);
    __attribute__((visibility("default"))) int PMPI_Type_create_hindexed(int count, int array_of_blocklengths[], MPI_Aint array_of_displacements[], MPI_Datatype oldtype, MPI_Datatype *newtype);
    __attribute__((visibility("default"))) int PMPI_Type_create_hvector(int count, int blocklength, MPI_Aint stride, MPI_Datatype oldtype, MPI_Datatype *newtype);
    __attribute__((visibility("default"))) int PMPI_Type_create_keyval(MPI_Type_copy_attr_function *type_copy_attr_fn, MPI_Type_delete_attr_function *type_delete_attr_fn, int *type_keyval, void *extra_state);
    __attribute__((visibility("default"))) int PMPI_Type_create_indexed_block(int count, int blocklength, int array_of_displacements[], MPI_Datatype oldtype, MPI_Datatype *newtype);
    __attribute__((visibility("default"))) int PMPI_Type_create_struct(int count, int array_of_block_lengths[], MPI_Aint array_of_displacements[], MPI_Datatype array_of_types[], MPI_Datatype *newtype);
    __attribute__((visibility("default"))) int PMPI_Type_create_subarray(int ndims, int size_array[], int subsize_array[], int start_array[], int order, MPI_Datatype oldtype, MPI_Datatype *newtype);
    __attribute__((visibility("default"))) int PMPI_Type_create_resized(MPI_Datatype oldtype, MPI_Aint lb, MPI_Aint extent, MPI_Datatype *newtype);
    __attribute__((visibility("default"))) int PMPI_Type_delete_attr(MPI_Datatype type, int type_keyval);
    __attribute__((visibility("default"))) int PMPI_Type_dup(MPI_Datatype type, MPI_Datatype *newtype);
    __attribute__((visibility("default"))) int PMPI_Type_extent(MPI_Datatype type, MPI_Aint *extent);
    __attribute__((visibility("default"))) int PMPI_Type_free(MPI_Datatype *type);
    __attribute__((visibility("default"))) int PMPI_Type_free_keyval(int *type_keyval);
    __attribute__((visibility("default"))) MPI_Datatype PMPI_Type_f2c(int datatype);
    __attribute__((visibility("default"))) int PMPI_Type_get_attr(MPI_Datatype type, int type_keyval, void *attribute_val, int *flag);
    __attribute__((visibility("default"))) int PMPI_Type_get_contents(MPI_Datatype mtype, int max_integers, int max_addresses, int max_datatypes, int array_of_integers[], MPI_Aint array_of_addresses[], MPI_Datatype array_of_datatypes[]);
    __attribute__((visibility("default"))) int PMPI_Type_get_envelope(MPI_Datatype type, int *num_integers, int *num_addresses, int *num_datatypes, int *combiner);
    __attribute__((visibility("default"))) int PMPI_Type_get_extent(MPI_Datatype type, MPI_Aint *lb, MPI_Aint *extent);
    __attribute__((visibility("default"))) int PMPI_Type_get_name(MPI_Datatype type, char *type_name, int *resultlen);
    __attribute__((visibility("default"))) int PMPI_Type_get_true_extent(MPI_Datatype datatype, MPI_Aint *true_lb, MPI_Aint *true_extent);
    __attribute__((visibility("default"))) int PMPI_Type_hindexed(int count, int array_of_blocklengths[], MPI_Aint array_of_displacements[], MPI_Datatype oldtype, MPI_Datatype *newtype);
    __attribute__((visibility("default"))) int PMPI_Type_hvector(int count, int blocklength, MPI_Aint stride, MPI_Datatype oldtype, MPI_Datatype *newtype);
    __attribute__((visibility("default"))) int PMPI_Type_indexed(int count, int array_of_blocklengths[], int array_of_displacements[], MPI_Datatype oldtype, MPI_Datatype *newtype);
    __attribute__((visibility("default"))) int PMPI_Type_lb(MPI_Datatype type, MPI_Aint *lb);
    __attribute__((visibility("default"))) int PMPI_Type_match_size(int typeclass, int size, MPI_Datatype *type);
    __attribute__((visibility("default"))) int PMPI_Type_set_attr(MPI_Datatype type, int type_keyval, void *attr_val);
    __attribute__((visibility("default"))) int PMPI_Type_set_name(MPI_Datatype type, char *type_name);
    __attribute__((visibility("default"))) int PMPI_Type_size(MPI_Datatype type, int *size);
    __attribute__((visibility("default"))) int PMPI_Type_struct(int count, int array_of_blocklengths[], MPI_Aint array_of_displacements[], MPI_Datatype array_of_types[], MPI_Datatype *newtype);
    __attribute__((visibility("default"))) int PMPI_Type_ub(MPI_Datatype mtype, MPI_Aint *ub);
    __attribute__((visibility("default"))) int PMPI_Type_vector(int count, int blocklength, int stride, MPI_Datatype oldtype, MPI_Datatype *newtype);
    __attribute__((visibility("default"))) int PMPI_Unpack(void *inbuf, int insize, int *position, void *outbuf, int outcount, MPI_Datatype datatype, MPI_Comm comm);
    __attribute__((visibility("default"))) int PMPI_Unpublish_name(char *service_name, MPI_Info info, char *port_name);
    __attribute__((visibility("default"))) int PMPI_Unpack_external(char *datarep, void *inbuf, MPI_Aint insize, MPI_Aint *position, void *outbuf, int outcount, MPI_Datatype datatype);
    __attribute__((visibility("default"))) int PMPI_Waitall(int count, MPI_Request *array_of_requests, MPI_Status *array_of_statuses);
    __attribute__((visibility("default"))) int PMPI_Waitany(int count, MPI_Request *array_of_requests, int *index, MPI_Status *status);
    __attribute__((visibility("default"))) int PMPI_Wait(MPI_Request *request, MPI_Status *status);
    __attribute__((visibility("default"))) int PMPI_Waitsome(int incount, MPI_Request *array_of_requests, int *outcount, int *array_of_indices, MPI_Status *array_of_statuses);
    __attribute__((visibility("default"))) int PMPI_Win_c2f(MPI_Win win);
    __attribute__((visibility("default"))) int PMPI_Win_call_errhandler(MPI_Win win, int errorcode);
    __attribute__((visibility("default"))) int PMPI_Win_complete(MPI_Win win);
    __attribute__((visibility("default"))) int PMPI_Win_create(void *base, MPI_Aint size, int disp_unit, MPI_Info info, MPI_Comm comm, MPI_Win *win);
    __attribute__((visibility("default"))) int PMPI_Win_create_errhandler(MPI_Win_errhandler_function *function, MPI_Errhandler *errhandler);
    __attribute__((visibility("default"))) int PMPI_Win_create_keyval(MPI_Win_copy_attr_function *win_copy_attr_fn, MPI_Win_delete_attr_function *win_delete_attr_fn, int *win_keyval, void *extra_state);
    __attribute__((visibility("default"))) int PMPI_Win_delete_attr(MPI_Win win, int win_keyval);
    __attribute__((visibility("default"))) MPI_Win PMPI_Win_f2c(int win);
    __attribute__((visibility("default"))) int PMPI_Win_fence(int assert, MPI_Win win);
    __attribute__((visibility("default"))) int PMPI_Win_free(MPI_Win *win);
    __attribute__((visibility("default"))) int PMPI_Win_free_keyval(int *win_keyval);
    __attribute__((visibility("default"))) int PMPI_Win_get_attr(MPI_Win win, int win_keyval, void *attribute_val, int *flag);
    __attribute__((visibility("default"))) int PMPI_Win_get_errhandler(MPI_Win win, MPI_Errhandler *errhandler);
    __attribute__((visibility("default"))) int PMPI_Win_get_group(MPI_Win win, MPI_Group *group);
    __attribute__((visibility("default"))) int PMPI_Win_get_name(MPI_Win win, char *win_name, int *resultlen);
    __attribute__((visibility("default"))) int PMPI_Win_lock(int lock_type, int rank, int assert, MPI_Win win);
    __attribute__((visibility("default"))) int PMPI_Win_post(MPI_Group group, int assert, MPI_Win win);
    __attribute__((visibility("default"))) int PMPI_Win_set_attr(MPI_Win win, int win_keyval, void *attribute_val);
    __attribute__((visibility("default"))) int PMPI_Win_set_errhandler(MPI_Win win, MPI_Errhandler errhandler);
    __attribute__((visibility("default"))) int PMPI_Win_set_name(MPI_Win win, char *win_name);
    __attribute__((visibility("default"))) int PMPI_Win_start(MPI_Group group, int assert, MPI_Win win);
    __attribute__((visibility("default"))) int PMPI_Win_test(MPI_Win win, int *flag);
    __attribute__((visibility("default"))) int PMPI_Win_unlock(int rank, MPI_Win win);
    __attribute__((visibility("default"))) int PMPI_Win_wait(MPI_Win win);
    __attribute__((visibility("default"))) double PMPI_Wtick(void);
    __attribute__((visibility("default"))) double PMPI_Wtime(void);
}
namespace std __attribute__((__visibility__("default"))) {
    enum _Rb_tree_color
    {
        _S_red = false, 
        _S_black = true
    };
    struct _Rb_tree_node_base
    {
            typedef _Rb_tree_node_base *_Base_ptr;
            typedef const _Rb_tree_node_base *_Const_Base_ptr;
            _Rb_tree_color _M_color;
            _Base_ptr _M_parent;
            _Base_ptr _M_left;
            _Base_ptr _M_right;
            static _Base_ptr _S_minimum(_Base_ptr __x)
            {
                while (__x->_M_left != 0)
                    __x = __x->_M_left;
                return __x;
            }
            static _Const_Base_ptr _S_minimum(_Const_Base_ptr __x)
            {
                while (__x->_M_left != 0)
                    __x = __x->_M_left;
                return __x;
            }
            static _Base_ptr _S_maximum(_Base_ptr __x)
            {
                while (__x->_M_right != 0)
                    __x = __x->_M_right;
                return __x;
            }
            static _Const_Base_ptr _S_maximum(_Const_Base_ptr __x)
            {
                while (__x->_M_right != 0)
                    __x = __x->_M_right;
                return __x;
            }
    };
    template<typename _Val >
    struct _Rb_tree_node : public _Rb_tree_node_base
    {
            typedef _Rb_tree_node<_Val> *_Link_type;
            _Val _M_value_field;
    };
    _Rb_tree_node_base *_Rb_tree_increment(_Rb_tree_node_base *__x);
    const _Rb_tree_node_base *_Rb_tree_increment(const _Rb_tree_node_base *__x);
    _Rb_tree_node_base *_Rb_tree_decrement(_Rb_tree_node_base *__x);
    const _Rb_tree_node_base *_Rb_tree_decrement(const _Rb_tree_node_base *__x);
    template<typename _Tp >
    struct _Rb_tree_iterator
    {
            typedef _Tp value_type;
            typedef _Tp &reference;
            typedef _Tp *pointer;
            typedef bidirectional_iterator_tag iterator_category;
            typedef ptrdiff_t difference_type;
            typedef _Rb_tree_iterator<_Tp> _Self;
            typedef _Rb_tree_node_base::_Base_ptr _Base_ptr;
            typedef _Rb_tree_node<_Tp> *_Link_type;
            _Rb_tree_iterator()
                : _M_node() 
            {
            }
            explicit _Rb_tree_iterator(_Link_type __x)
                : _M_node(__x) 
            {
            }
            reference operator *() const
            {
                return static_cast<_Link_type >(_M_node)->_M_value_field;
            }
            pointer operator ->() const
            {
                return &static_cast<_Link_type >(_M_node)->_M_value_field;
            }
            _Self &operator ++()
            {
                _M_node = _Rb_tree_increment(_M_node);
                return *this;
            }
            _Self operator ++(int)
            {
                _Self __tmp = *this;
                _M_node = _Rb_tree_increment(_M_node);
                return __tmp;
            }
            _Self &operator --()
            {
                _M_node = _Rb_tree_decrement(_M_node);
                return *this;
            }
            _Self operator --(int)
            {
                _Self __tmp = *this;
                _M_node = _Rb_tree_decrement(_M_node);
                return __tmp;
            }
            bool operator ==(const _Self &__x) const
            {
                return _M_node == __x._M_node;
            }
            bool operator !=(const _Self &__x) const
            {
                return _M_node != __x._M_node;
            }
            _Base_ptr _M_node;
    };
    template<typename _Tp >
    struct _Rb_tree_const_iterator
    {
            typedef _Tp value_type;
            typedef const _Tp &reference;
            typedef const _Tp *pointer;
            typedef _Rb_tree_iterator<_Tp> iterator;
            typedef bidirectional_iterator_tag iterator_category;
            typedef ptrdiff_t difference_type;
            typedef _Rb_tree_const_iterator<_Tp> _Self;
            typedef _Rb_tree_node_base::_Const_Base_ptr _Base_ptr;
            typedef const _Rb_tree_node<_Tp> *_Link_type;
            _Rb_tree_const_iterator()
                : _M_node() 
            {
            }
            explicit _Rb_tree_const_iterator(_Link_type __x)
                : _M_node(__x) 
            {
            }
            _Rb_tree_const_iterator(const iterator &__it)
                : _M_node(__it._M_node) 
            {
            }
            reference operator *() const
            {
                return static_cast<_Link_type >(_M_node)->_M_value_field;
            }
            pointer operator ->() const
            {
                return &static_cast<_Link_type >(_M_node)->_M_value_field;
            }
            _Self &operator ++()
            {
                _M_node = _Rb_tree_increment(_M_node);
                return *this;
            }
            _Self operator ++(int)
            {
                _Self __tmp = *this;
                _M_node = _Rb_tree_increment(_M_node);
                return __tmp;
            }
            _Self &operator --()
            {
                _M_node = _Rb_tree_decrement(_M_node);
                return *this;
            }
            _Self operator --(int)
            {
                _Self __tmp = *this;
                _M_node = _Rb_tree_decrement(_M_node);
                return __tmp;
            }
            bool operator ==(const _Self &__x) const
            {
                return _M_node == __x._M_node;
            }
            bool operator !=(const _Self &__x) const
            {
                return _M_node != __x._M_node;
            }
            _Base_ptr _M_node;
    };
    template<typename _Val >
    inline bool operator ==(const _Rb_tree_iterator<_Val> &__x, const _Rb_tree_const_iterator<_Val> &__y)
    {
        return __x._M_node == __y._M_node;
    }
    template<typename _Val >
    inline bool operator !=(const _Rb_tree_iterator<_Val> &__x, const _Rb_tree_const_iterator<_Val> &__y)
    {
        return __x._M_node != __y._M_node;
    }
    void _Rb_tree_insert_and_rebalance(const bool __insert_left, _Rb_tree_node_base *__x, _Rb_tree_node_base *__p, _Rb_tree_node_base &__header);
    _Rb_tree_node_base *_Rb_tree_rebalance_for_erase(_Rb_tree_node_base *const __z, _Rb_tree_node_base &__header);
    template<typename _Key, typename _Val, typename _KeyOfValue, typename _Compare, typename _Alloc = allocator<_Val> >
    class _Rb_tree
    {
            typedef typename _Alloc::template rebind<_Rb_tree_node<_Val> >::other _Node_allocator;
        protected :
            typedef _Rb_tree_node_base *_Base_ptr;
            typedef const _Rb_tree_node_base *_Const_Base_ptr;
        public :
            typedef _Key key_type;
            typedef _Val value_type;
            typedef value_type *pointer;
            typedef const value_type *const_pointer;
            typedef value_type &reference;
            typedef const value_type &const_reference;
            typedef _Rb_tree_node<_Val> *_Link_type;
            typedef const _Rb_tree_node<_Val> *_Const_Link_type;
            typedef size_t size_type;
            typedef ptrdiff_t difference_type;
            typedef _Alloc allocator_type;
            _Node_allocator &_M_get_Node_allocator()
            {
                return *static_cast<_Node_allocator * >(&this->_M_impl);
            }
            const _Node_allocator &_M_get_Node_allocator() const
            {
                return *static_cast<const _Node_allocator * >(&this->_M_impl);
            }
            allocator_type get_allocator() const
            {
                return allocator_type(_M_get_Node_allocator());
            }
        protected :
            _Link_type _M_get_node()
            {
                return _M_impl._Node_allocator::allocate(1);
            }
            void _M_put_node(_Link_type __p)
            {
                _M_impl._Node_allocator::deallocate(__p, 1);
            }
            _Link_type _M_create_node(const value_type &__x)
            {
                _Link_type __tmp = _M_get_node();
                try
                {
                    get_allocator().construct(&__tmp->_M_value_field, __x);
                }
                catch (...)
                {
                    _M_put_node(__tmp);
                    throw;
                }
                return __tmp;
            }
            void _M_destroy_node(_Link_type __p)
            {
                get_allocator().destroy(&__p->_M_value_field);
                _M_put_node(__p);
            }
            _Link_type _M_clone_node(_Const_Link_type __x)
            {
                _Link_type __tmp = _M_create_node(__x->_M_value_field);
                __tmp->_M_color = __x->_M_color;
                __tmp->_M_left = 0;
                __tmp->_M_right = 0;
                return __tmp;
            }
        protected :
            template<typename _Key_compare, bool _Is_pod_comparator = __is_pod(_Key_compare) >
            struct _Rb_tree_impl : public _Node_allocator
            {
                    _Key_compare _M_key_compare;
                    _Rb_tree_node_base _M_header;
                    size_type _M_node_count;
                    _Rb_tree_impl()
                        : _Node_allocator(), _M_key_compare(), _M_header(), _M_node_count(0) 
                    {
                        _M_initialize();
                    }
                    _Rb_tree_impl(const _Key_compare &__comp, const _Node_allocator &__a)
                        : _Node_allocator(__a), _M_key_compare(__comp), _M_header(), _M_node_count(0) 
                    {
                        _M_initialize();
                    }
                private :
                    void _M_initialize()
                    {
                        this->_M_header._M_color = _S_red;
                        this->_M_header._M_parent = 0;
                        this->_M_header._M_left = &this->_M_header;
                        this->_M_header._M_right = &this->_M_header;
                    }
            };
            _Rb_tree_impl<_Compare> _M_impl;
        protected :
            _Base_ptr &_M_root()
            {
                return this->_M_impl._M_header._M_parent;
            }
            _Const_Base_ptr _M_root() const
            {
                return this->_M_impl._M_header._M_parent;
            }
            _Base_ptr &_M_leftmost()
            {
                return this->_M_impl._M_header._M_left;
            }
            _Const_Base_ptr _M_leftmost() const
            {
                return this->_M_impl._M_header._M_left;
            }
            _Base_ptr &_M_rightmost()
            {
                return this->_M_impl._M_header._M_right;
            }
            _Const_Base_ptr _M_rightmost() const
            {
                return this->_M_impl._M_header._M_right;
            }
            _Link_type _M_begin()
            {
                return static_cast<_Link_type >(this->_M_impl._M_header._M_parent);
            }
            _Const_Link_type _M_begin() const
            {
                return static_cast<_Const_Link_type >(this->_M_impl._M_header._M_parent);
            }
            _Link_type _M_end()
            {
                return static_cast<_Link_type >(&this->_M_impl._M_header);
            }
            _Const_Link_type _M_end() const
            {
                return static_cast<_Const_Link_type >(&this->_M_impl._M_header);
            }
            static const_reference _S_value(_Const_Link_type __x)
            {
                return __x->_M_value_field;
            }
            static const _Key &_S_key(_Const_Link_type __x)
            {
                return _KeyOfValue()(_S_value(__x));
            }
            static _Link_type _S_left(_Base_ptr __x)
            {
                return static_cast<_Link_type >(__x->_M_left);
            }
            static _Const_Link_type _S_left(_Const_Base_ptr __x)
            {
                return static_cast<_Const_Link_type >(__x->_M_left);
            }
            static _Link_type _S_right(_Base_ptr __x)
            {
                return static_cast<_Link_type >(__x->_M_right);
            }
            static _Const_Link_type _S_right(_Const_Base_ptr __x)
            {
                return static_cast<_Const_Link_type >(__x->_M_right);
            }
            static const_reference _S_value(_Const_Base_ptr __x)
            {
                return static_cast<_Const_Link_type >(__x)->_M_value_field;
            }
            static const _Key &_S_key(_Const_Base_ptr __x)
            {
                return _KeyOfValue()(_S_value(__x));
            }
            static _Base_ptr _S_minimum(_Base_ptr __x)
            {
                return _Rb_tree_node_base::_S_minimum(__x);
            }
            static _Const_Base_ptr _S_minimum(_Const_Base_ptr __x)
            {
                return _Rb_tree_node_base::_S_minimum(__x);
            }
            static _Base_ptr _S_maximum(_Base_ptr __x)
            {
                return _Rb_tree_node_base::_S_maximum(__x);
            }
            static _Const_Base_ptr _S_maximum(_Const_Base_ptr __x)
            {
                return _Rb_tree_node_base::_S_maximum(__x);
            }
        public :
            typedef _Rb_tree_iterator<value_type> iterator;
            typedef _Rb_tree_const_iterator<value_type> const_iterator;
            typedef std::reverse_iterator<iterator> reverse_iterator;
            typedef std::reverse_iterator<const_iterator> const_reverse_iterator;
        private :
            iterator _M_insert_(_Const_Base_ptr __x, _Const_Base_ptr __y, const value_type &__v);
            iterator _M_insert_lower(_Base_ptr __x, _Base_ptr __y, const value_type &__v);
            iterator _M_insert_equal_lower(const value_type &__x);
            _Link_type _M_copy(_Const_Link_type __x, _Link_type __p);
            void _M_erase(_Link_type __x);
            iterator _M_lower_bound(_Link_type __x, _Link_type __y, const _Key &__k);
            const_iterator _M_lower_bound(_Const_Link_type __x, _Const_Link_type __y, const _Key &__k) const;
            iterator _M_upper_bound(_Link_type __x, _Link_type __y, const _Key &__k);
            const_iterator _M_upper_bound(_Const_Link_type __x, _Const_Link_type __y, const _Key &__k) const;
        public :
            _Rb_tree()
            {
            }
            _Rb_tree(const _Compare &__comp, const allocator_type &__a = allocator_type())
                : _M_impl(__comp, __a) 
            {
            }
            _Rb_tree(const _Rb_tree &__x)
                : _M_impl(__x._M_impl._M_key_compare, __x._M_get_Node_allocator()) 
            {
                if (__x._M_root() != 0)
                {
                    _M_root() = _M_copy(__x._M_begin(), _M_end());
                    _M_leftmost() = _S_minimum(_M_root());
                    _M_rightmost() = _S_maximum(_M_root());
                    _M_impl._M_node_count = __x._M_impl._M_node_count;
                }
            }
            ~_Rb_tree()
            {
                _M_erase(_M_begin());
            }
            _Rb_tree &operator =(const _Rb_tree &__x);
            _Compare key_comp() const
            {
                return _M_impl._M_key_compare;
            }
            iterator begin()
            {
                return iterator(static_cast<_Link_type >(this->_M_impl._M_header._M_left));
            }
            const_iterator begin() const
            {
                return const_iterator(static_cast<_Const_Link_type >(this->_M_impl._M_header._M_left));
            }
            iterator end()
            {
                return iterator(static_cast<_Link_type >(&this->_M_impl._M_header));
            }
            const_iterator end() const
            {
                return const_iterator(static_cast<_Const_Link_type >(&this->_M_impl._M_header));
            }
            reverse_iterator rbegin()
            {
                return reverse_iterator(end());
            }
            const_reverse_iterator rbegin() const
            {
                return const_reverse_iterator(end());
            }
            reverse_iterator rend()
            {
                return reverse_iterator(begin());
            }
            const_reverse_iterator rend() const
            {
                return const_reverse_iterator(begin());
            }
            bool empty() const
            {
                return _M_impl._M_node_count == 0;
            }
            size_type size() const
            {
                return _M_impl._M_node_count;
            }
            size_type max_size() const
            {
                return _M_get_Node_allocator().max_size();
            }
            void swap(_Rb_tree &__t);
            pair<iterator, bool> _M_insert_unique(const value_type &__x);
            iterator _M_insert_equal(const value_type &__x);
            iterator _M_insert_unique_(const_iterator __position, const value_type &__x);
            iterator _M_insert_equal_(const_iterator __position, const value_type &__x);
            template<typename _InputIterator >
            void _M_insert_unique(_InputIterator __first, _InputIterator __last);
            template<typename _InputIterator >
            void _M_insert_equal(_InputIterator __first, _InputIterator __last);
            void erase(iterator __position);
            void erase(const_iterator __position);
            size_type erase(const key_type &__x);
            void erase(iterator __first, iterator __last);
            void erase(const_iterator __first, const_iterator __last);
            void erase(const key_type *__first, const key_type *__last);
            void clear()
            {
                _M_erase(_M_begin());
                _M_leftmost() = _M_end();
                _M_root() = 0;
                _M_rightmost() = _M_end();
                _M_impl._M_node_count = 0;
            }
            iterator find(const key_type &__k);
            const_iterator find(const key_type &__k) const;
            size_type count(const key_type &__k) const;
            iterator lower_bound(const key_type &__k)
            {
                return _M_lower_bound(_M_begin(), _M_end(), __k);
            }
            const_iterator lower_bound(const key_type &__k) const
            {
                return _M_lower_bound(_M_begin(), _M_end(), __k);
            }
            iterator upper_bound(const key_type &__k)
            {
                return _M_upper_bound(_M_begin(), _M_end(), __k);
            }
            const_iterator upper_bound(const key_type &__k) const
            {
                return _M_upper_bound(_M_begin(), _M_end(), __k);
            }
            pair<iterator, iterator> equal_range(const key_type &__k);
            pair<const_iterator, const_iterator> equal_range(const key_type &__k) const;
            bool __rb_verify() const;
    };
    template<typename _Key, typename _Val, typename _KeyOfValue, typename _Compare, typename _Alloc >
    inline bool operator ==(const _Rb_tree<_Key, _Val, _KeyOfValue, _Compare, _Alloc> &__x, const _Rb_tree<_Key, _Val, _KeyOfValue, _Compare, _Alloc> &__y)
    {
        return __x.size() == __y.size() && std::equal(__x.begin(), __x.end(), __y.begin());
    }
    template<typename _Key, typename _Val, typename _KeyOfValue, typename _Compare, typename _Alloc >
    inline bool operator <(const _Rb_tree<_Key, _Val, _KeyOfValue, _Compare, _Alloc> &__x, const _Rb_tree<_Key, _Val, _KeyOfValue, _Compare, _Alloc> &__y)
    {
        return std::lexicographical_compare(__x.begin(), __x.end(), __y.begin(), __y.end());
    }
    template<typename _Key, typename _Val, typename _KeyOfValue, typename _Compare, typename _Alloc >
    inline bool operator !=(const _Rb_tree<_Key, _Val, _KeyOfValue, _Compare, _Alloc> &__x, const _Rb_tree<_Key, _Val, _KeyOfValue, _Compare, _Alloc> &__y)
    {
        return !(__x == __y);
    }
    template<typename _Key, typename _Val, typename _KeyOfValue, typename _Compare, typename _Alloc >
    inline bool operator >(const _Rb_tree<_Key, _Val, _KeyOfValue, _Compare, _Alloc> &__x, const _Rb_tree<_Key, _Val, _KeyOfValue, _Compare, _Alloc> &__y)
    {
        return __y < __x;
    }
    template<typename _Key, typename _Val, typename _KeyOfValue, typename _Compare, typename _Alloc >
    inline bool operator <=(const _Rb_tree<_Key, _Val, _KeyOfValue, _Compare, _Alloc> &__x, const _Rb_tree<_Key, _Val, _KeyOfValue, _Compare, _Alloc> &__y)
    {
        return !(__y < __x);
    }
    template<typename _Key, typename _Val, typename _KeyOfValue, typename _Compare, typename _Alloc >
    inline bool operator >=(const _Rb_tree<_Key, _Val, _KeyOfValue, _Compare, _Alloc> &__x, const _Rb_tree<_Key, _Val, _KeyOfValue, _Compare, _Alloc> &__y)
    {
        return !(__x < __y);
    }
    template<typename _Key, typename _Val, typename _KeyOfValue, typename _Compare, typename _Alloc >
    inline void swap(_Rb_tree<_Key, _Val, _KeyOfValue, _Compare, _Alloc> &__x, _Rb_tree<_Key, _Val, _KeyOfValue, _Compare, _Alloc> &__y)
    {
        __x.swap(__y);
    }
    template<typename _Key, typename _Val, typename _KeyOfValue, typename _Compare, typename _Alloc >
    _Rb_tree<_Key, _Val, _KeyOfValue, _Compare, _Alloc> &_Rb_tree<_Key, _Val, _KeyOfValue, _Compare, _Alloc>::operator =(const _Rb_tree<_Key, _Val, _KeyOfValue, _Compare, _Alloc> &__x)
    {
        if (this != &__x)
        {
            clear();
            _M_impl._M_key_compare = __x._M_impl._M_key_compare;
            if (__x._M_root() != 0)
            {
                _M_root() = _M_copy(__x._M_begin(), _M_end());
                _M_leftmost() = _S_minimum(_M_root());
                _M_rightmost() = _S_maximum(_M_root());
                _M_impl._M_node_count = __x._M_impl._M_node_count;
            }
        }
        return *this;
    }
    template<typename _Key, typename _Val, typename _KeyOfValue, typename _Compare, typename _Alloc >
    typename _Rb_tree<_Key, _Val, _KeyOfValue, _Compare, _Alloc>::iterator _Rb_tree<_Key, _Val, _KeyOfValue, _Compare, _Alloc>::_M_insert_(_Const_Base_ptr __x, _Const_Base_ptr __p, const _Val &__v)
    {
        bool __insert_left = (__x != 0 || __p == _M_end() || _M_impl._M_key_compare(_KeyOfValue()(__v), _S_key(__p)));
        _Link_type __z = _M_create_node(__v);
        _Rb_tree_insert_and_rebalance(__insert_left, __z, const_cast<_Base_ptr >(__p), this->_M_impl._M_header);
        ++_M_impl._M_node_count;
        return iterator(__z);
    }
    template<typename _Key, typename _Val, typename _KeyOfValue, typename _Compare, typename _Alloc >
    typename _Rb_tree<_Key, _Val, _KeyOfValue, _Compare, _Alloc>::iterator _Rb_tree<_Key, _Val, _KeyOfValue, _Compare, _Alloc>::_M_insert_lower(_Base_ptr __x, _Base_ptr __p, const _Val &__v)
    {
        bool __insert_left = (__x != 0 || __p == _M_end() || !_M_impl._M_key_compare(_S_key(__p), _KeyOfValue()(__v)));
        _Link_type __z = _M_create_node(__v);
        _Rb_tree_insert_and_rebalance(__insert_left, __z, __p, this->_M_impl._M_header);
        ++_M_impl._M_node_count;
        return iterator(__z);
    }
    template<typename _Key, typename _Val, typename _KeyOfValue, typename _Compare, typename _Alloc >
    typename _Rb_tree<_Key, _Val, _KeyOfValue, _Compare, _Alloc>::iterator _Rb_tree<_Key, _Val, _KeyOfValue, _Compare, _Alloc>::_M_insert_equal_lower(const _Val &__v)
    {
        _Link_type __x = _M_begin();
        _Link_type __y = _M_end();
        while (__x != 0)
        {
            __y = __x;
            __x = !_M_impl._M_key_compare(_S_key(__x), _KeyOfValue()(__v)) ? _S_left(__x) : _S_right(__x);
        }
        return _M_insert_lower(__x, __y, __v);
    }
    template<typename _Key, typename _Val, typename _KoV, typename _Compare, typename _Alloc >
    typename _Rb_tree<_Key, _Val, _KoV, _Compare, _Alloc>::_Link_type _Rb_tree<_Key, _Val, _KoV, _Compare, _Alloc>::_M_copy(_Const_Link_type __x, _Link_type __p)
    {
        _Link_type __top = _M_clone_node(__x);
        __top->_M_parent = __p;
        try
        {
            if (__x->_M_right)
                __top->_M_right = _M_copy(_S_right(__x), __top);
            __p = __top;
            __x = _S_left(__x);
            while (__x != 0)
            {
                _Link_type __y = _M_clone_node(__x);
                __p->_M_left = __y;
                __y->_M_parent = __p;
                if (__x->_M_right)
                    __y->_M_right = _M_copy(_S_right(__x), __y);
                __p = __y;
                __x = _S_left(__x);
            }
        }
        catch (...)
        {
            _M_erase(__top);
            throw;
        }
        return __top;
    }
    template<typename _Key, typename _Val, typename _KeyOfValue, typename _Compare, typename _Alloc >
    void _Rb_tree<_Key, _Val, _KeyOfValue, _Compare, _Alloc>::_M_erase(_Link_type __x)
    {
        while (__x != 0)
        {
            _M_erase(_S_right(__x));
            _Link_type __y = _S_left(__x);
            _M_destroy_node(__x);
            __x = __y;
        }
    }
    template<typename _Key, typename _Val, typename _KeyOfValue, typename _Compare, typename _Alloc >
    typename _Rb_tree<_Key, _Val, _KeyOfValue, _Compare, _Alloc>::iterator _Rb_tree<_Key, _Val, _KeyOfValue, _Compare, _Alloc>::_M_lower_bound(_Link_type __x, _Link_type __y, const _Key &__k)
    {
        while (__x != 0)
            if (!_M_impl._M_key_compare(_S_key(__x), __k))
                __y = __x , __x = _S_left(__x);
            else
                __x = _S_right(__x);
        return iterator(__y);
    }
    template<typename _Key, typename _Val, typename _KeyOfValue, typename _Compare, typename _Alloc >
    typename _Rb_tree<_Key, _Val, _KeyOfValue, _Compare, _Alloc>::const_iterator _Rb_tree<_Key, _Val, _KeyOfValue, _Compare, _Alloc>::_M_lower_bound(_Const_Link_type __x, _Const_Link_type __y, const _Key &__k) const
    {
        while (__x != 0)
            if (!_M_impl._M_key_compare(_S_key(__x), __k))
                __y = __x , __x = _S_left(__x);
            else
                __x = _S_right(__x);
        return const_iterator(__y);
    }
    template<typename _Key, typename _Val, typename _KeyOfValue, typename _Compare, typename _Alloc >
    typename _Rb_tree<_Key, _Val, _KeyOfValue, _Compare, _Alloc>::iterator _Rb_tree<_Key, _Val, _KeyOfValue, _Compare, _Alloc>::_M_upper_bound(_Link_type __x, _Link_type __y, const _Key &__k)
    {
        while (__x != 0)
            if (_M_impl._M_key_compare(__k, _S_key(__x)))
                __y = __x , __x = _S_left(__x);
            else
                __x = _S_right(__x);
        return iterator(__y);
    }
    template<typename _Key, typename _Val, typename _KeyOfValue, typename _Compare, typename _Alloc >
    typename _Rb_tree<_Key, _Val, _KeyOfValue, _Compare, _Alloc>::const_iterator _Rb_tree<_Key, _Val, _KeyOfValue, _Compare, _Alloc>::_M_upper_bound(_Const_Link_type __x, _Const_Link_type __y, const _Key &__k) const
    {
        while (__x != 0)
            if (_M_impl._M_key_compare(__k, _S_key(__x)))
                __y = __x , __x = _S_left(__x);
            else
                __x = _S_right(__x);
        return const_iterator(__y);
    }
    template<typename _Key, typename _Val, typename _KeyOfValue, typename _Compare, typename _Alloc >
    pair<typename _Rb_tree<_Key, _Val, _KeyOfValue, _Compare, _Alloc>::iterator, typename _Rb_tree<_Key, _Val, _KeyOfValue, _Compare, _Alloc>::iterator> _Rb_tree<_Key, _Val, _KeyOfValue, _Compare, _Alloc>::equal_range(const _Key &__k)
    {
        _Link_type __x = _M_begin();
        _Link_type __y = _M_end();
        while (__x != 0)
        {
            if (_M_impl._M_key_compare(_S_key(__x), __k))
                __x = _S_right(__x);
            else
                if (_M_impl._M_key_compare(__k, _S_key(__x)))
                    __y = __x , __x = _S_left(__x);
                else
                {
                    _Link_type __xu(__x), __yu(__y);
                    __y = __x , __x = _S_left(__x);
                    __xu = _S_right(__xu);
                    return pair<iterator, iterator>(_M_lower_bound(__x, __y, __k), _M_upper_bound(__xu, __yu, __k));
                }
        }
        return pair<iterator, iterator>(iterator(__y), iterator(__y));
    }
    template<typename _Key, typename _Val, typename _KeyOfValue, typename _Compare, typename _Alloc >
    pair<typename _Rb_tree<_Key, _Val, _KeyOfValue, _Compare, _Alloc>::const_iterator, typename _Rb_tree<_Key, _Val, _KeyOfValue, _Compare, _Alloc>::const_iterator> _Rb_tree<_Key, _Val, _KeyOfValue, _Compare, _Alloc>::equal_range(const _Key &__k) const
    {
        _Const_Link_type __x = _M_begin();
        _Const_Link_type __y = _M_end();
        while (__x != 0)
        {
            if (_M_impl._M_key_compare(_S_key(__x), __k))
                __x = _S_right(__x);
            else
                if (_M_impl._M_key_compare(__k, _S_key(__x)))
                    __y = __x , __x = _S_left(__x);
                else
                {
                    _Const_Link_type __xu(__x), __yu(__y);
                    __y = __x , __x = _S_left(__x);
                    __xu = _S_right(__xu);
                    return pair<const_iterator, const_iterator>(_M_lower_bound(__x, __y, __k), _M_upper_bound(__xu, __yu, __k));
                }
        }
        return pair<const_iterator, const_iterator>(const_iterator(__y), const_iterator(__y));
    }
    template<typename _Key, typename _Val, typename _KeyOfValue, typename _Compare, typename _Alloc >
    void _Rb_tree<_Key, _Val, _KeyOfValue, _Compare, _Alloc>::swap(_Rb_tree<_Key, _Val, _KeyOfValue, _Compare, _Alloc> &__t)
    {
        if (_M_root() == 0)
        {
            if (__t._M_root() != 0)
            {
                _M_root() = __t._M_root();
                _M_leftmost() = __t._M_leftmost();
                _M_rightmost() = __t._M_rightmost();
                _M_root()->_M_parent = _M_end();
                __t._M_root() = 0;
                __t._M_leftmost() = __t._M_end();
                __t._M_rightmost() = __t._M_end();
            }
        }
        else
            if (__t._M_root() == 0)
            {
                __t._M_root() = _M_root();
                __t._M_leftmost() = _M_leftmost();
                __t._M_rightmost() = _M_rightmost();
                __t._M_root()->_M_parent = __t._M_end();
                _M_root() = 0;
                _M_leftmost() = _M_end();
                _M_rightmost() = _M_end();
            }
            else
            {
                std::swap(_M_root(), __t._M_root());
                std::swap(_M_leftmost(), __t._M_leftmost());
                std::swap(_M_rightmost(), __t._M_rightmost());
                _M_root()->_M_parent = _M_end();
                __t._M_root()->_M_parent = __t._M_end();
            }
        std::swap(this->_M_impl._M_node_count, __t._M_impl._M_node_count);
        std::swap(this->_M_impl._M_key_compare, __t._M_impl._M_key_compare);
        std::__alloc_swap<_Node_allocator>::_S_do_it(_M_get_Node_allocator(), __t._M_get_Node_allocator());
    }
    template<typename _Key, typename _Val, typename _KeyOfValue, typename _Compare, typename _Alloc >
    pair<typename _Rb_tree<_Key, _Val, _KeyOfValue, _Compare, _Alloc>::iterator, bool> _Rb_tree<_Key, _Val, _KeyOfValue, _Compare, _Alloc>::_M_insert_unique(const _Val &__v)
    {
        _Link_type __x = _M_begin();
        _Link_type __y = _M_end();
        bool __comp = true;
        while (__x != 0)
        {
            __y = __x;
            __comp = _M_impl._M_key_compare(_KeyOfValue()(__v), _S_key(__x));
            __x = __comp ? _S_left(__x) : _S_right(__x);
        }
        iterator __j = iterator(__y);
        if (__comp)
        {
            if (__j == begin())
                return pair<iterator, bool>(_M_insert_(__x, __y, __v), true);
            else
                --__j;
        }
        if (_M_impl._M_key_compare(_S_key(__j._M_node), _KeyOfValue()(__v)))
            return pair<iterator, bool>(_M_insert_(__x, __y, __v), true);
        return pair<iterator, bool>(__j, false);
    }
    template<typename _Key, typename _Val, typename _KeyOfValue, typename _Compare, typename _Alloc >
    typename _Rb_tree<_Key, _Val, _KeyOfValue, _Compare, _Alloc>::iterator _Rb_tree<_Key, _Val, _KeyOfValue, _Compare, _Alloc>::_M_insert_equal(const _Val &__v)
    {
        _Link_type __x = _M_begin();
        _Link_type __y = _M_end();
        while (__x != 0)
        {
            __y = __x;
            __x = _M_impl._M_key_compare(_KeyOfValue()(__v), _S_key(__x)) ? _S_left(__x) : _S_right(__x);
        }
        return _M_insert_(__x, __y, __v);
    }
    template<typename _Key, typename _Val, typename _KeyOfValue, typename _Compare, typename _Alloc >
    typename _Rb_tree<_Key, _Val, _KeyOfValue, _Compare, _Alloc>::iterator _Rb_tree<_Key, _Val, _KeyOfValue, _Compare, _Alloc>::_M_insert_unique_(const_iterator __position, const _Val &__v)
    {
        if (__position._M_node == _M_end())
        {
            if (size() > 0 && _M_impl._M_key_compare(_S_key(_M_rightmost()), _KeyOfValue()(__v)))
                return _M_insert_(0, _M_rightmost(), __v);
            else
                return _M_insert_unique(__v).first;
        }
        else
            if (_M_impl._M_key_compare(_KeyOfValue()(__v), _S_key(__position._M_node)))
            {
                const_iterator __before = __position;
                if (__position._M_node == _M_leftmost())
                    return _M_insert_(_M_leftmost(), _M_leftmost(), __v);
                else
                    if (_M_impl._M_key_compare(_S_key((--__before)._M_node), _KeyOfValue()(__v)))
                    {
                        if (_S_right(__before._M_node) == 0)
                            return _M_insert_(0, __before._M_node, __v);
                        else
                            return _M_insert_(__position._M_node, __position._M_node, __v);
                    }
                    else
                        return _M_insert_unique(__v).first;
            }
            else
                if (_M_impl._M_key_compare(_S_key(__position._M_node), _KeyOfValue()(__v)))
                {
                    const_iterator __after = __position;
                    if (__position._M_node == _M_rightmost())
                        return _M_insert_(0, _M_rightmost(), __v);
                    else
                        if (_M_impl._M_key_compare(_KeyOfValue()(__v), _S_key((++__after)._M_node)))
                        {
                            if (_S_right(__position._M_node) == 0)
                                return _M_insert_(0, __position._M_node, __v);
                            else
                                return _M_insert_(__after._M_node, __after._M_node, __v);
                        }
                        else
                            return _M_insert_unique(__v).first;
                }
                else
                    return iterator(static_cast<_Link_type >(const_cast<_Base_ptr >(__position._M_node)));
    }
    template<typename _Key, typename _Val, typename _KeyOfValue, typename _Compare, typename _Alloc >
    typename _Rb_tree<_Key, _Val, _KeyOfValue, _Compare, _Alloc>::iterator _Rb_tree<_Key, _Val, _KeyOfValue, _Compare, _Alloc>::_M_insert_equal_(const_iterator __position, const _Val &__v)
    {
        if (__position._M_node == _M_end())
        {
            if (size() > 0 && !_M_impl._M_key_compare(_KeyOfValue()(__v), _S_key(_M_rightmost())))
                return _M_insert_(0, _M_rightmost(), __v);
            else
                return _M_insert_equal(__v);
        }
        else
            if (!_M_impl._M_key_compare(_S_key(__position._M_node), _KeyOfValue()(__v)))
            {
                const_iterator __before = __position;
                if (__position._M_node == _M_leftmost())
                    return _M_insert_(_M_leftmost(), _M_leftmost(), __v);
                else
                    if (!_M_impl._M_key_compare(_KeyOfValue()(__v), _S_key((--__before)._M_node)))
                    {
                        if (_S_right(__before._M_node) == 0)
                            return _M_insert_(0, __before._M_node, __v);
                        else
                            return _M_insert_(__position._M_node, __position._M_node, __v);
                    }
                    else
                        return _M_insert_equal(__v);
            }
            else
            {
                const_iterator __after = __position;
                if (__position._M_node == _M_rightmost())
                    return _M_insert_(0, _M_rightmost(), __v);
                else
                    if (!_M_impl._M_key_compare(_S_key((++__after)._M_node), _KeyOfValue()(__v)))
                    {
                        if (_S_right(__position._M_node) == 0)
                            return _M_insert_(0, __position._M_node, __v);
                        else
                            return _M_insert_(__after._M_node, __after._M_node, __v);
                    }
                    else
                        return _M_insert_equal_lower(__v);
            }
    }
    template<typename _Key, typename _Val, typename _KoV, typename _Cmp, typename _Alloc >
    template<class _II >
    void _Rb_tree<_Key, _Val, _KoV, _Cmp, _Alloc>::_M_insert_unique(_II __first, _II __last)
    {
        for (;
            __first != __last;
            ++__first)
            _M_insert_unique_(end(), *__first);
    }
    template<typename _Key, typename _Val, typename _KoV, typename _Cmp, typename _Alloc >
    template<class _II >
    void _Rb_tree<_Key, _Val, _KoV, _Cmp, _Alloc>::_M_insert_equal(_II __first, _II __last)
    {
        for (;
            __first != __last;
            ++__first)
            _M_insert_equal_(end(), *__first);
    }
    template<typename _Key, typename _Val, typename _KeyOfValue, typename _Compare, typename _Alloc >
    inline void _Rb_tree<_Key, _Val, _KeyOfValue, _Compare, _Alloc>::erase(iterator __position)
    {
        _Link_type __y = static_cast<_Link_type >(_Rb_tree_rebalance_for_erase(__position._M_node, this->_M_impl._M_header));
        _M_destroy_node(__y);
        --_M_impl._M_node_count;
    }
    template<typename _Key, typename _Val, typename _KeyOfValue, typename _Compare, typename _Alloc >
    inline void _Rb_tree<_Key, _Val, _KeyOfValue, _Compare, _Alloc>::erase(const_iterator __position)
    {
        _Link_type __y = static_cast<_Link_type >(_Rb_tree_rebalance_for_erase(const_cast<_Base_ptr >(__position._M_node), this->_M_impl._M_header));
        _M_destroy_node(__y);
        --_M_impl._M_node_count;
    }
    template<typename _Key, typename _Val, typename _KeyOfValue, typename _Compare, typename _Alloc >
    typename _Rb_tree<_Key, _Val, _KeyOfValue, _Compare, _Alloc>::size_type _Rb_tree<_Key, _Val, _KeyOfValue, _Compare, _Alloc>::erase(const _Key &__x)
    {
        pair<iterator, iterator> __p = equal_range(__x);
        const size_type __old_size = size();
        erase(__p.first, __p.second);
        return __old_size - size();
    }
    template<typename _Key, typename _Val, typename _KeyOfValue, typename _Compare, typename _Alloc >
    void _Rb_tree<_Key, _Val, _KeyOfValue, _Compare, _Alloc>::erase(iterator __first, iterator __last)
    {
        if (__first == begin() && __last == end())
            clear();
        else
            while (__first != __last)
                erase(__first++);
    }
    template<typename _Key, typename _Val, typename _KeyOfValue, typename _Compare, typename _Alloc >
    void _Rb_tree<_Key, _Val, _KeyOfValue, _Compare, _Alloc>::erase(const_iterator __first, const_iterator __last)
    {
        if (__first == begin() && __last == end())
            clear();
        else
            while (__first != __last)
                erase(__first++);
    }
    template<typename _Key, typename _Val, typename _KeyOfValue, typename _Compare, typename _Alloc >
    void _Rb_tree<_Key, _Val, _KeyOfValue, _Compare, _Alloc>::erase(const _Key *__first, const _Key *__last)
    {
        while (__first != __last)
            erase(*__first++);
    }
    template<typename _Key, typename _Val, typename _KeyOfValue, typename _Compare, typename _Alloc >
    typename _Rb_tree<_Key, _Val, _KeyOfValue, _Compare, _Alloc>::iterator _Rb_tree<_Key, _Val, _KeyOfValue, _Compare, _Alloc>::find(const _Key &__k)
    {
        iterator __j = _M_lower_bound(_M_begin(), _M_end(), __k);
        return (__j == end() || _M_impl._M_key_compare(__k, _S_key(__j._M_node))) ? end() : __j;
    }
    template<typename _Key, typename _Val, typename _KeyOfValue, typename _Compare, typename _Alloc >
    typename _Rb_tree<_Key, _Val, _KeyOfValue, _Compare, _Alloc>::const_iterator _Rb_tree<_Key, _Val, _KeyOfValue, _Compare, _Alloc>::find(const _Key &__k) const
    {
        const_iterator __j = _M_lower_bound(_M_begin(), _M_end(), __k);
        return (__j == end() || _M_impl._M_key_compare(__k, _S_key(__j._M_node))) ? end() : __j;
    }
    template<typename _Key, typename _Val, typename _KeyOfValue, typename _Compare, typename _Alloc >
    typename _Rb_tree<_Key, _Val, _KeyOfValue, _Compare, _Alloc>::size_type _Rb_tree<_Key, _Val, _KeyOfValue, _Compare, _Alloc>::count(const _Key &__k) const
    {
        pair<const_iterator, const_iterator> __p = equal_range(__k);
        const size_type __n = std::distance(__p.first, __p.second);
        return __n;
    }
    unsigned int _Rb_tree_black_count(const _Rb_tree_node_base *__node, const _Rb_tree_node_base *__root);
    template<typename _Key, typename _Val, typename _KeyOfValue, typename _Compare, typename _Alloc >
    bool _Rb_tree<_Key, _Val, _KeyOfValue, _Compare, _Alloc>::__rb_verify() const
    {
        if (_M_impl._M_node_count == 0 || begin() == end())
            return _M_impl._M_node_count == 0 && begin() == end() && this->_M_impl._M_header._M_left == _M_end() && this->_M_impl._M_header._M_right == _M_end();
        unsigned int __len = _Rb_tree_black_count(_M_leftmost(), _M_root());
        for (const_iterator __it = begin();
            __it != end();
            ++__it)
        {
            _Const_Link_type __x = static_cast<_Const_Link_type >(__it._M_node);
            _Const_Link_type __L = _S_left(__x);
            _Const_Link_type __R = _S_right(__x);
            if (__x->_M_color == _S_red)
                if ((__L && __L->_M_color == _S_red) || (__R && __R->_M_color == _S_red))
                    return false;
            if (__L && _M_impl._M_key_compare(_S_key(__x), _S_key(__L)))
                return false;
            if (__R && _M_impl._M_key_compare(_S_key(__R), _S_key(__x)))
                return false;
            if (!__L && !__R && _Rb_tree_black_count(__x, _M_root()) != __len)
                return false;
        }
        if (_M_leftmost() != _Rb_tree_node_base::_S_minimum(_M_root()))
            return false;
        if (_M_rightmost() != _Rb_tree_node_base::_S_maximum(_M_root()))
            return false;
        return true;
    }
}
namespace std __attribute__((__visibility__("default"))) {
    template<typename _Key, typename _Tp, typename _Compare = std::less<_Key>, typename _Alloc = std::allocator<std::pair<const _Key, _Tp> > >
    class map
    {
        public :
            typedef _Key key_type;
            typedef _Tp mapped_type;
            typedef std::pair<const _Key, _Tp> value_type;
            typedef _Compare key_compare;
            typedef _Alloc allocator_type;
        private :
            typedef typename _Alloc::value_type _Alloc_value_type;
        public :
            class value_compare : public std::binary_function<value_type, value_type, bool>
            {
                    friend class map<_Key, _Tp, _Compare, _Alloc>;
                protected :
                    _Compare comp;
                    value_compare(_Compare __c)
                        : comp(__c) 
                    {
                    }
                public :
                    bool operator ()(const value_type &__x, const value_type &__y) const
                    {
                        return comp(__x.first, __y.first);
                    }
            };
        private :
            typedef typename _Alloc::template rebind<value_type>::other _Pair_alloc_type;
            typedef _Rb_tree<key_type, value_type, _Select1st<value_type>, key_compare, _Pair_alloc_type> _Rep_type;
            _Rep_type _M_t;
        public :
            typedef typename _Pair_alloc_type::pointer pointer;
            typedef typename _Pair_alloc_type::const_pointer const_pointer;
            typedef typename _Pair_alloc_type::reference reference;
            typedef typename _Pair_alloc_type::const_reference const_reference;
            typedef typename _Rep_type::iterator iterator;
            typedef typename _Rep_type::const_iterator const_iterator;
            typedef typename _Rep_type::size_type size_type;
            typedef typename _Rep_type::difference_type difference_type;
            typedef typename _Rep_type::reverse_iterator reverse_iterator;
            typedef typename _Rep_type::const_reverse_iterator const_reverse_iterator;
            map()
                : _M_t() 
            {
            }
            explicit map(const _Compare &__comp, const allocator_type &__a = allocator_type())
                : _M_t(__comp, __a) 
            {
            }
            map(const map &__x)
                : _M_t(__x._M_t) 
            {
            }
            template<typename _InputIterator >
            map(_InputIterator __first, _InputIterator __last)
                : _M_t() 
            {
                _M_t._M_insert_unique(__first, __last);
            }
            template<typename _InputIterator >
            map(_InputIterator __first, _InputIterator __last, const _Compare &__comp, const allocator_type &__a = allocator_type())
                : _M_t(__comp, __a) 
            {
                _M_t._M_insert_unique(__first, __last);
            }
            map &operator =(const map &__x)
            {
                _M_t = __x._M_t;
                return *this;
            }
            allocator_type get_allocator() const
            {
                return _M_t.get_allocator();
            }
            iterator begin()
            {
                return _M_t.begin();
            }
            const_iterator begin() const
            {
                return _M_t.begin();
            }
            iterator end()
            {
                return _M_t.end();
            }
            const_iterator end() const
            {
                return _M_t.end();
            }
            reverse_iterator rbegin()
            {
                return _M_t.rbegin();
            }
            const_reverse_iterator rbegin() const
            {
                return _M_t.rbegin();
            }
            reverse_iterator rend()
            {
                return _M_t.rend();
            }
            const_reverse_iterator rend() const
            {
                return _M_t.rend();
            }
            bool empty() const
            {
                return _M_t.empty();
            }
            size_type size() const
            {
                return _M_t.size();
            }
            size_type max_size() const
            {
                return _M_t.max_size();
            }
            mapped_type &operator [](const key_type &__k)
            {
                iterator __i = lower_bound(__k);
                if (__i == end() || key_comp()(__k, (*__i).first))
                    __i = insert(__i, value_type(__k, mapped_type()));
                return (*__i).second;
            }
            mapped_type &at(const key_type &__k)
            {
                iterator __i = lower_bound(__k);
                if (__i == end() || key_comp()(__k, (*__i).first))
                    __throw_out_of_range(("map::at"));
                return (*__i).second;
            }
            const mapped_type &at(const key_type &__k) const
            {
                const_iterator __i = lower_bound(__k);
                if (__i == end() || key_comp()(__k, (*__i).first))
                    __throw_out_of_range(("map::at"));
                return (*__i).second;
            }
            std::pair<iterator, bool> insert(const value_type &__x)
            {
                return _M_t._M_insert_unique(__x);
            }
            iterator insert(iterator __position, const value_type &__x)
            {
                return _M_t._M_insert_unique_(__position, __x);
            }
            template<typename _InputIterator >
            void insert(_InputIterator __first, _InputIterator __last)
            {
                _M_t._M_insert_unique(__first, __last);
            }
            void erase(iterator __position)
            {
                _M_t.erase(__position);
            }
            size_type erase(const key_type &__x)
            {
                return _M_t.erase(__x);
            }
            void erase(iterator __first, iterator __last)
            {
                _M_t.erase(__first, __last);
            }
            void swap(map &__x)
            {
                _M_t.swap(__x._M_t);
            }
            void clear()
            {
                _M_t.clear();
            }
            key_compare key_comp() const
            {
                return _M_t.key_comp();
            }
            value_compare value_comp() const
            {
                return value_compare(_M_t.key_comp());
            }
            iterator find(const key_type &__x)
            {
                return _M_t.find(__x);
            }
            const_iterator find(const key_type &__x) const
            {
                return _M_t.find(__x);
            }
            size_type count(const key_type &__x) const
            {
                return _M_t.find(__x) == _M_t.end() ? 0 : 1;
            }
            iterator lower_bound(const key_type &__x)
            {
                return _M_t.lower_bound(__x);
            }
            const_iterator lower_bound(const key_type &__x) const
            {
                return _M_t.lower_bound(__x);
            }
            iterator upper_bound(const key_type &__x)
            {
                return _M_t.upper_bound(__x);
            }
            const_iterator upper_bound(const key_type &__x) const
            {
                return _M_t.upper_bound(__x);
            }
            std::pair<iterator, iterator> equal_range(const key_type &__x)
            {
                return _M_t.equal_range(__x);
            }
            std::pair<const_iterator, const_iterator> equal_range(const key_type &__x) const
            {
                return _M_t.equal_range(__x);
            }
            template<typename _K1, typename _T1, typename _C1, typename _A1 >
            friend bool operator ==(const map<_K1, _T1, _C1, _A1> &, const map<_K1, _T1, _C1, _A1> &);
            template<typename _K1, typename _T1, typename _C1, typename _A1 >
            friend bool operator <(const map<_K1, _T1, _C1, _A1> &, const map<_K1, _T1, _C1, _A1> &);
    };
    template<typename _Key, typename _Tp, typename _Compare, typename _Alloc >
    inline bool operator ==(const map<_Key, _Tp, _Compare, _Alloc> &__x, const map<_Key, _Tp, _Compare, _Alloc> &__y)
    {
        return __x._M_t == __y._M_t;
    }
    template<typename _Key, typename _Tp, typename _Compare, typename _Alloc >
    inline bool operator <(const map<_Key, _Tp, _Compare, _Alloc> &__x, const map<_Key, _Tp, _Compare, _Alloc> &__y)
    {
        return __x._M_t < __y._M_t;
    }
    template<typename _Key, typename _Tp, typename _Compare, typename _Alloc >
    inline bool operator !=(const map<_Key, _Tp, _Compare, _Alloc> &__x, const map<_Key, _Tp, _Compare, _Alloc> &__y)
    {
        return !(__x == __y);
    }
    template<typename _Key, typename _Tp, typename _Compare, typename _Alloc >
    inline bool operator >(const map<_Key, _Tp, _Compare, _Alloc> &__x, const map<_Key, _Tp, _Compare, _Alloc> &__y)
    {
        return __y < __x;
    }
    template<typename _Key, typename _Tp, typename _Compare, typename _Alloc >
    inline bool operator <=(const map<_Key, _Tp, _Compare, _Alloc> &__x, const map<_Key, _Tp, _Compare, _Alloc> &__y)
    {
        return !(__y < __x);
    }
    template<typename _Key, typename _Tp, typename _Compare, typename _Alloc >
    inline bool operator >=(const map<_Key, _Tp, _Compare, _Alloc> &__x, const map<_Key, _Tp, _Compare, _Alloc> &__y)
    {
        return !(__x < __y);
    }
    template<typename _Key, typename _Tp, typename _Compare, typename _Alloc >
    inline void swap(map<_Key, _Tp, _Compare, _Alloc> &__x, map<_Key, _Tp, _Compare, _Alloc> &__y)
    {
        __x.swap(__y);
    }
}
namespace std __attribute__((__visibility__("default"))) {
    template<typename _Key, typename _Tp, typename _Compare = std::less<_Key>, typename _Alloc = std::allocator<std::pair<const _Key, _Tp> > >
    class multimap
    {
        public :
            typedef _Key key_type;
            typedef _Tp mapped_type;
            typedef std::pair<const _Key, _Tp> value_type;
            typedef _Compare key_compare;
            typedef _Alloc allocator_type;
        private :
            typedef typename _Alloc::value_type _Alloc_value_type;
        public :
            class value_compare : public std::binary_function<value_type, value_type, bool>
            {
                    friend class multimap<_Key, _Tp, _Compare, _Alloc>;
                protected :
                    _Compare comp;
                    value_compare(_Compare __c)
                        : comp(__c) 
                    {
                    }
                public :
                    bool operator ()(const value_type &__x, const value_type &__y) const
                    {
                        return comp(__x.first, __y.first);
                    }
            };
        private :
            typedef typename _Alloc::template rebind<value_type>::other _Pair_alloc_type;
            typedef _Rb_tree<key_type, value_type, _Select1st<value_type>, key_compare, _Pair_alloc_type> _Rep_type;
            _Rep_type _M_t;
        public :
            typedef typename _Pair_alloc_type::pointer pointer;
            typedef typename _Pair_alloc_type::const_pointer const_pointer;
            typedef typename _Pair_alloc_type::reference reference;
            typedef typename _Pair_alloc_type::const_reference const_reference;
            typedef typename _Rep_type::iterator iterator;
            typedef typename _Rep_type::const_iterator const_iterator;
            typedef typename _Rep_type::size_type size_type;
            typedef typename _Rep_type::difference_type difference_type;
            typedef typename _Rep_type::reverse_iterator reverse_iterator;
            typedef typename _Rep_type::const_reverse_iterator const_reverse_iterator;
            multimap()
                : _M_t() 
            {
            }
            explicit multimap(const _Compare &__comp, const allocator_type &__a = allocator_type())
                : _M_t(__comp, __a) 
            {
            }
            multimap(const multimap &__x)
                : _M_t(__x._M_t) 
            {
            }
            template<typename _InputIterator >
            multimap(_InputIterator __first, _InputIterator __last)
                : _M_t() 
            {
                _M_t._M_insert_equal(__first, __last);
            }
            template<typename _InputIterator >
            multimap(_InputIterator __first, _InputIterator __last, const _Compare &__comp, const allocator_type &__a = allocator_type())
                : _M_t(__comp, __a) 
            {
                _M_t._M_insert_equal(__first, __last);
            }
            multimap &operator =(const multimap &__x)
            {
                _M_t = __x._M_t;
                return *this;
            }
            allocator_type get_allocator() const
            {
                return _M_t.get_allocator();
            }
            iterator begin()
            {
                return _M_t.begin();
            }
            const_iterator begin() const
            {
                return _M_t.begin();
            }
            iterator end()
            {
                return _M_t.end();
            }
            const_iterator end() const
            {
                return _M_t.end();
            }
            reverse_iterator rbegin()
            {
                return _M_t.rbegin();
            }
            const_reverse_iterator rbegin() const
            {
                return _M_t.rbegin();
            }
            reverse_iterator rend()
            {
                return _M_t.rend();
            }
            const_reverse_iterator rend() const
            {
                return _M_t.rend();
            }
            bool empty() const
            {
                return _M_t.empty();
            }
            size_type size() const
            {
                return _M_t.size();
            }
            size_type max_size() const
            {
                return _M_t.max_size();
            }
            iterator insert(const value_type &__x)
            {
                return _M_t._M_insert_equal(__x);
            }
            iterator insert(iterator __position, const value_type &__x)
            {
                return _M_t._M_insert_equal_(__position, __x);
            }
            template<typename _InputIterator >
            void insert(_InputIterator __first, _InputIterator __last)
            {
                _M_t._M_insert_equal(__first, __last);
            }
            void erase(iterator __position)
            {
                _M_t.erase(__position);
            }
            size_type erase(const key_type &__x)
            {
                return _M_t.erase(__x);
            }
            void erase(iterator __first, iterator __last)
            {
                _M_t.erase(__first, __last);
            }
            void swap(multimap &__x)
            {
                _M_t.swap(__x._M_t);
            }
            void clear()
            {
                _M_t.clear();
            }
            key_compare key_comp() const
            {
                return _M_t.key_comp();
            }
            value_compare value_comp() const
            {
                return value_compare(_M_t.key_comp());
            }
            iterator find(const key_type &__x)
            {
                return _M_t.find(__x);
            }
            const_iterator find(const key_type &__x) const
            {
                return _M_t.find(__x);
            }
            size_type count(const key_type &__x) const
            {
                return _M_t.count(__x);
            }
            iterator lower_bound(const key_type &__x)
            {
                return _M_t.lower_bound(__x);
            }
            const_iterator lower_bound(const key_type &__x) const
            {
                return _M_t.lower_bound(__x);
            }
            iterator upper_bound(const key_type &__x)
            {
                return _M_t.upper_bound(__x);
            }
            const_iterator upper_bound(const key_type &__x) const
            {
                return _M_t.upper_bound(__x);
            }
            std::pair<iterator, iterator> equal_range(const key_type &__x)
            {
                return _M_t.equal_range(__x);
            }
            std::pair<const_iterator, const_iterator> equal_range(const key_type &__x) const
            {
                return _M_t.equal_range(__x);
            }
            template<typename _K1, typename _T1, typename _C1, typename _A1 >
            friend bool operator ==(const multimap<_K1, _T1, _C1, _A1> &, const multimap<_K1, _T1, _C1, _A1> &);
            template<typename _K1, typename _T1, typename _C1, typename _A1 >
            friend bool operator <(const multimap<_K1, _T1, _C1, _A1> &, const multimap<_K1, _T1, _C1, _A1> &);
    };
    template<typename _Key, typename _Tp, typename _Compare, typename _Alloc >
    inline bool operator ==(const multimap<_Key, _Tp, _Compare, _Alloc> &__x, const multimap<_Key, _Tp, _Compare, _Alloc> &__y)
    {
        return __x._M_t == __y._M_t;
    }
    template<typename _Key, typename _Tp, typename _Compare, typename _Alloc >
    inline bool operator <(const multimap<_Key, _Tp, _Compare, _Alloc> &__x, const multimap<_Key, _Tp, _Compare, _Alloc> &__y)
    {
        return __x._M_t < __y._M_t;
    }
    template<typename _Key, typename _Tp, typename _Compare, typename _Alloc >
    inline bool operator !=(const multimap<_Key, _Tp, _Compare, _Alloc> &__x, const multimap<_Key, _Tp, _Compare, _Alloc> &__y)
    {
        return !(__x == __y);
    }
    template<typename _Key, typename _Tp, typename _Compare, typename _Alloc >
    inline bool operator >(const multimap<_Key, _Tp, _Compare, _Alloc> &__x, const multimap<_Key, _Tp, _Compare, _Alloc> &__y)
    {
        return __y < __x;
    }
    template<typename _Key, typename _Tp, typename _Compare, typename _Alloc >
    inline bool operator <=(const multimap<_Key, _Tp, _Compare, _Alloc> &__x, const multimap<_Key, _Tp, _Compare, _Alloc> &__y)
    {
        return !(__y < __x);
    }
    template<typename _Key, typename _Tp, typename _Compare, typename _Alloc >
    inline bool operator >=(const multimap<_Key, _Tp, _Compare, _Alloc> &__x, const multimap<_Key, _Tp, _Compare, _Alloc> &__y)
    {
        return !(__x < __y);
    }
    template<typename _Key, typename _Tp, typename _Compare, typename _Alloc >
    inline void swap(multimap<_Key, _Tp, _Compare, _Alloc> &__x, multimap<_Key, _Tp, _Compare, _Alloc> &__y)
    {
        __x.swap(__y);
    }
}
namespace std __attribute__((__visibility__("default"))) {
    namespace rel_ops {
        template<class _Tp >
        inline bool operator !=(const _Tp &__x, const _Tp &__y)
        {
            return !(__x == __y);
        }
        template<class _Tp >
        inline bool operator >(const _Tp &__x, const _Tp &__y)
        {
            return __y < __x;
        }
        template<class _Tp >
        inline bool operator <=(const _Tp &__x, const _Tp &__y)
        {
            return !(__y < __x);
        }
        template<class _Tp >
        inline bool operator >=(const _Tp &__x, const _Tp &__y)
        {
            return !(__x < __y);
        }
    }
}
namespace std __attribute__((__visibility__("default"))) {
    extern istream cin;
    extern ostream cout;
    extern ostream cerr;
    extern ostream clog;
    extern wistream wcin;
    extern wostream wcout;
    extern wostream wcerr;
    extern wostream wclog;
    static ios_base::Init __ioinit;
}
static const int ompi_stdio_seek_set = 0;
static const int ompi_stdio_seek_cur = 1;
static const int ompi_stdio_seek_end = 2;
static const int SEEK_SET = ompi_stdio_seek_set;
static const int SEEK_CUR = ompi_stdio_seek_cur;
static const int SEEK_END = ompi_stdio_seek_end;
struct opal_mutex_t;
extern "C"
void ompi_mpi_cxx_op_intercept(void *invec, void *outvec, int *len, MPI_Datatype *datatype, MPI_User_function *fn);
extern "C"
void ompi_mpi_cxx_comm_errhandler_invoke(ompi_errhandler_t *c_errhandler, MPI_Comm *mpi_comm, int *err, const char *message);
extern "C"
void ompi_mpi_cxx_win_errhandler_invoke(ompi_errhandler_t *c_errhandler, MPI_Win *mpi_comm, int *err, const char *message);
extern "C"
void ompi_mpi_cxx_file_errhandler_invoke(ompi_errhandler_t *c_errhandler, MPI_File *mpi_comm, int *err, const char *message);
enum CommType
{
    eIntracomm, 
    eIntercomm, 
    eCartcomm, 
    eGraphcomm
};
extern "C"
int ompi_mpi_cxx_comm_copy_attr_intercept(MPI_Comm oldcomm, int keyval, void *extra_state, void *attribute_val_in, void *attribute_val_out, int *flag, MPI_Comm newcomm);
extern "C"
int ompi_mpi_cxx_comm_delete_attr_intercept(MPI_Comm comm, int keyval, void *attribute_val, void *extra_state);
extern "C"
int ompi_mpi_cxx_type_copy_attr_intercept(MPI_Datatype oldtype, int keyval, void *extra_state, void *attribute_val_in, void *attribute_val_out, int *flag);
extern "C"
int ompi_mpi_cxx_type_delete_attr_intercept(MPI_Datatype type, int keyval, void *attribute_val, void *extra_state);
extern "C"
int ompi_mpi_cxx_win_copy_attr_intercept(MPI_Win oldwin, int keyval, void *extra_state, void *attribute_val_in, void *attribute_val_out, int *flag);
extern "C"
int ompi_mpi_cxx_win_delete_attr_intercept(MPI_Win win, int keyval, void *attribute_val, void *extra_state);
extern "C"
int ompi_mpi_cxx_grequest_query_fn_intercept(void *state, MPI_Status *status);
extern "C"
int ompi_mpi_cxx_grequest_free_fn_intercept(void *state);
extern "C"
int ompi_mpi_cxx_grequest_cancel_fn_intercept(void *state, int canceled);
namespace MPI {
    extern int mpi_errno;
    class Comm_Null;
    class Comm;
    class Intracomm;
    class Intercomm;
    class Graphcomm;
    class Cartcomm;
    class Datatype;
    class Errhandler;
    class Group;
    class Op;
    class Request;
    class Grequest;
    class Status;
    class Info;
    class Win;
    class File;
    typedef MPI_Aint Aint;
    typedef MPI_Offset Offset;
    static const int SUCCESS = 0;
    static const int ERR_BUFFER = 1;
    static const int ERR_COUNT = 2;
    static const int ERR_TYPE = 3;
    static const int ERR_TAG = 4;
    static const int ERR_COMM = 5;
    static const int ERR_RANK = 6;
    static const int ERR_REQUEST = 7;
    static const int ERR_ROOT = 8;
    static const int ERR_GROUP = 9;
    static const int ERR_OP = 10;
    static const int ERR_TOPOLOGY = 11;
    static const int ERR_DIMS = 12;
    static const int ERR_ARG = 13;
    static const int ERR_UNKNOWN = 14;
    static const int ERR_TRUNCATE = 15;
    static const int ERR_OTHER = 16;
    static const int ERR_INTERN = 17;
    static const int ERR_PENDING = 19;
    static const int ERR_IN_STATUS = 18;
    static const int ERR_LASTCODE = 54;
    static const int ERR_BASE = 24;
    static const int ERR_INFO_VALUE = 33;
    static const int ERR_INFO_KEY = 31;
    static const int ERR_INFO_NOKEY = 32;
    static const int ERR_KEYVAL = 36;
    static const int ERR_NAME = 38;
    static const int ERR_NO_MEM = 39;
    static const int ERR_SERVICE = 48;
    static const int ERR_SPAWN = 50;
    static const int ERR_WIN = 53;
    __attribute__((visibility("default"))) extern void *const BOTTOM;
    __attribute__((visibility("default"))) extern void *const IN_PLACE;
    static const int PROC_NULL = - 2;
    static const int ANY_SOURCE = - 1;
    static const int ROOT = - 4;
    static const int ANY_TAG = - 1;
    static const int UNDEFINED = - 32766;
    static const int BSEND_OVERHEAD = 128;
    static const int KEYVAL_INVALID = - 1;
    static const int ORDER_C = 0;
    static const int ORDER_FORTRAN = 1;
    static const int DISTRIBUTE_BLOCK = 0;
    static const int DISTRIBUTE_CYCLIC = 1;
    static const int DISTRIBUTE_NONE = 2;
    static const int DISTRIBUTE_DFLT_DARG = (- 1);
    __attribute__((visibility("default"))) extern const Errhandler ERRORS_ARE_FATAL;
    __attribute__((visibility("default"))) extern const Errhandler ERRORS_RETURN;
    __attribute__((visibility("default"))) extern const Errhandler ERRORS_THROW_EXCEPTIONS;
    static const int TYPECLASS_INTEGER = 1;
    static const int TYPECLASS_REAL = 2;
    static const int TYPECLASS_COMPLEX = 3;
    static const int MAX_PROCESSOR_NAME = 256;
    static const int MAX_ERROR_STRING = 256;
    static const int MAX_INFO_KEY = 36;
    static const int MAX_INFO_VAL = 256;
    static const int MAX_PORT_NAME = 1024;
    static const int MAX_OBJECT_NAME = 64;
    __attribute__((visibility("default"))) extern const Datatype CHAR;
    __attribute__((visibility("default"))) extern const Datatype SHORT;
    __attribute__((visibility("default"))) extern const Datatype INT;
    __attribute__((visibility("default"))) extern const Datatype LONG;
    __attribute__((visibility("default"))) extern const Datatype SIGNED_CHAR;
    __attribute__((visibility("default"))) extern const Datatype UNSIGNED_CHAR;
    __attribute__((visibility("default"))) extern const Datatype UNSIGNED_SHORT;
    __attribute__((visibility("default"))) extern const Datatype UNSIGNED;
    __attribute__((visibility("default"))) extern const Datatype UNSIGNED_LONG;
    __attribute__((visibility("default"))) extern const Datatype FLOAT;
    __attribute__((visibility("default"))) extern const Datatype DOUBLE;
    __attribute__((visibility("default"))) extern const Datatype LONG_DOUBLE;
    __attribute__((visibility("default"))) extern const Datatype BYTE;
    __attribute__((visibility("default"))) extern const Datatype PACKED;
    __attribute__((visibility("default"))) extern const Datatype WCHAR;
    __attribute__((visibility("default"))) extern const Datatype FLOAT_INT;
    __attribute__((visibility("default"))) extern const Datatype DOUBLE_INT;
    __attribute__((visibility("default"))) extern const Datatype LONG_INT;
    __attribute__((visibility("default"))) extern const Datatype TWOINT;
    __attribute__((visibility("default"))) extern const Datatype SHORT_INT;
    __attribute__((visibility("default"))) extern const Datatype LONG_DOUBLE_INT;
    __attribute__((visibility("default"))) extern const Datatype INTEGER;
    __attribute__((visibility("default"))) extern const Datatype REAL;
    __attribute__((visibility("default"))) extern const Datatype DOUBLE_PRECISION;
    __attribute__((visibility("default"))) extern const Datatype F_COMPLEX;
    __attribute__((visibility("default"))) extern const Datatype LOGICAL;
    __attribute__((visibility("default"))) extern const Datatype CHARACTER;
    __attribute__((visibility("default"))) extern const Datatype TWOREAL;
    __attribute__((visibility("default"))) extern const Datatype TWODOUBLE_PRECISION;
    __attribute__((visibility("default"))) extern const Datatype TWOINTEGER;
    __attribute__((visibility("default"))) extern const Datatype INTEGER1;
    __attribute__((visibility("default"))) extern const Datatype INTEGER2;
    __attribute__((visibility("default"))) extern const Datatype INTEGER4;
    __attribute__((visibility("default"))) extern const Datatype REAL2;
    __attribute__((visibility("default"))) extern const Datatype REAL4;
    __attribute__((visibility("default"))) extern const Datatype REAL8;
    __attribute__((visibility("default"))) extern const Datatype LONG_LONG;
    __attribute__((visibility("default"))) extern const Datatype UNSIGNED_LONG_LONG;
    __attribute__((visibility("default"))) extern const Datatype BOOL;
    __attribute__((visibility("default"))) extern const Datatype COMPLEX;
    __attribute__((visibility("default"))) extern const Datatype DOUBLE_COMPLEX;
    __attribute__((visibility("default"))) extern const Datatype LONG_DOUBLE_COMPLEX;
    __attribute__((visibility("default"))) extern const Datatype UB;
    __attribute__((visibility("default"))) extern const Datatype LB;
    static const int COMBINER_NAMED = MPI_COMBINER_NAMED;
    static const int COMBINER_DUP = MPI_COMBINER_DUP;
    static const int COMBINER_CONTIGUOUS = MPI_COMBINER_CONTIGUOUS;
    static const int COMBINER_VECTOR = MPI_COMBINER_VECTOR;
    static const int COMBINER_HVECTOR_INTEGER = MPI_COMBINER_HVECTOR_INTEGER;
    static const int COMBINER_HVECTOR = MPI_COMBINER_HVECTOR;
    static const int COMBINER_INDEXED = MPI_COMBINER_INDEXED;
    static const int COMBINER_HINDEXED_INTEGER = MPI_COMBINER_HINDEXED_INTEGER;
    static const int COMBINER_HINDEXED = MPI_COMBINER_HINDEXED;
    static const int COMBINER_INDEXED_BLOCK = MPI_COMBINER_INDEXED_BLOCK;
    static const int COMBINER_STRUCT_INTEGER = MPI_COMBINER_STRUCT_INTEGER;
    static const int COMBINER_STRUCT = MPI_COMBINER_STRUCT;
    static const int COMBINER_SUBARRAY = MPI_COMBINER_SUBARRAY;
    static const int COMBINER_DARRAY = MPI_COMBINER_DARRAY;
    static const int COMBINER_F90_REAL = MPI_COMBINER_F90_REAL;
    static const int COMBINER_F90_COMPLEX = MPI_COMBINER_F90_COMPLEX;
    static const int COMBINER_F90_INTEGER = MPI_COMBINER_F90_INTEGER;
    static const int COMBINER_RESIZED = MPI_COMBINER_RESIZED;
    static const int THREAD_SINGLE = MPI_THREAD_SINGLE;
    static const int THREAD_FUNNELED = MPI_THREAD_FUNNELED;
    static const int THREAD_SERIALIZED = MPI_THREAD_SERIALIZED;
    static const int THREAD_MULTIPLE = MPI_THREAD_MULTIPLE;
    __attribute__((visibility("default"))) extern Intracomm COMM_WORLD;
    __attribute__((visibility("default"))) extern Intracomm COMM_SELF;
    static const int IDENT = MPI_IDENT;
    static const int CONGRUENT = MPI_CONGRUENT;
    static const int SIMILAR = MPI_SIMILAR;
    static const int UNEQUAL = MPI_UNEQUAL;
    static const int TAG_UB = MPI_TAG_UB;
    static const int HOST = MPI_HOST;
    static const int IO = MPI_IO;
    static const int WTIME_IS_GLOBAL = MPI_WTIME_IS_GLOBAL;
    static const int APPNUM = MPI_APPNUM;
    static const int LASTUSEDCODE = MPI_LASTUSEDCODE;
    static const int UNIVERSE_SIZE = MPI_UNIVERSE_SIZE;
    static const int WIN_BASE = MPI_WIN_BASE;
    static const int WIN_SIZE = MPI_WIN_SIZE;
    static const int WIN_DISP_UNIT = MPI_WIN_DISP_UNIT;
    __attribute__((visibility("default"))) extern const Op MAX;
    __attribute__((visibility("default"))) extern const Op MIN;
    __attribute__((visibility("default"))) extern const Op SUM;
    __attribute__((visibility("default"))) extern const Op PROD;
    __attribute__((visibility("default"))) extern const Op MAXLOC;
    __attribute__((visibility("default"))) extern const Op MINLOC;
    __attribute__((visibility("default"))) extern const Op BAND;
    __attribute__((visibility("default"))) extern const Op BOR;
    __attribute__((visibility("default"))) extern const Op BXOR;
    __attribute__((visibility("default"))) extern const Op LAND;
    __attribute__((visibility("default"))) extern const Op LOR;
    __attribute__((visibility("default"))) extern const Op LXOR;
    __attribute__((visibility("default"))) extern const Op REPLACE;
    __attribute__((visibility("default"))) extern const Group GROUP_NULL;
    __attribute__((visibility("default"))) extern const Win WIN_NULL;
    __attribute__((visibility("default"))) extern const Info INFO_NULL;
    __attribute__((visibility("default"))) extern Comm_Null COMM_NULL;
    __attribute__((visibility("default"))) extern const Datatype DATATYPE_NULL;
    __attribute__((visibility("default"))) extern Request REQUEST_NULL;
    __attribute__((visibility("default"))) extern const Op OP_NULL;
    __attribute__((visibility("default"))) extern const Errhandler ERRHANDLER_NULL;
    __attribute__((visibility("default"))) extern const File FILE_NULL;
    __attribute__((visibility("default"))) extern const char **ARGV_NULL;
    __attribute__((visibility("default"))) extern const char ***ARGVS_NULL;
    __attribute__((visibility("default"))) extern const Group GROUP_EMPTY;
    static const int GRAPH = 2;
    static const int CART = 1;
    static const int MODE_CREATE = 1;
    static const int MODE_RDONLY = 2;
    static const int MODE_WRONLY = 4;
    static const int MODE_RDWR = 8;
    static const int MODE_DELETE_ON_CLOSE = 16;
    static const int MODE_UNIQUE_OPEN = 32;
    static const int MODE_EXCL = 64;
    static const int MODE_APPEND = 128;
    static const int MODE_SEQUENTIAL = 256;
    static const int DISPLACEMENT_CURRENT = - 54278278;
    static const int SEEK_SET = ::SEEK_SET;
    static const int SEEK_CUR = ::SEEK_CUR;
    static const int SEEK_END = ::SEEK_END;
    static const int MAX_DATAREP_STRING = 128;
    static const int MODE_NOCHECK = 1;
    static const int MODE_NOPRECEDE = 2;
    static const int MODE_NOPUT = 4;
    static const int MODE_NOSTORE = 8;
    static const int MODE_NOSUCCEED = 16;
    static const int LOCK_EXCLUSIVE = 1;
    static const int LOCK_SHARED = 2;
    void Attach_buffer(void *buffer, int size);
    int Detach_buffer(void *&buffer);
    void Compute_dims(int nnodes, int ndims, int dims[]);
    void Get_processor_name(char *name, int &resultlen);
    void Get_error_string(int errorcode, char *string, int &resultlen);
    int Get_error_class(int errorcode);
    double Wtime();
    double Wtick();
    void Init(int &argc, char **&argv);
    void Init();
    __attribute__((visibility("default"))) void InitializeIntercepts();
    void Real_init();
    void Finalize();
    bool Is_initialized();
    bool Is_finalized();
    int Init_thread(int &argc, char **&argv, int required);
    int Init_thread(int required);
    bool Is_thread_main();
    int Query_thread();
    void *Alloc_mem(Aint size, const Info &info);
    void Free_mem(void *base);
    void Close_port(const char *port_name);
    void Lookup_name(const char *service_name, const Info &info, char *port_name);
    void Open_port(const Info &info, char *port_name);
    void Publish_name(const char *service_name, const Info &info, const char *port_name);
    void Unpublish_name(const char *service_name, const Info &info, const char *port_name);
    void Pcontrol(const int level, ...);
    void Get_version(int &version, int &subversion);
    MPI::Aint Get_address(void *location);
    class Datatype
    {
        public :
            inline Datatype()
                : mpi_datatype(((MPI_Datatype) ((void *) &(ompi_mpi_datatype_null)))) 
            {
            }
            inline virtual ~Datatype()
            {
            }
            inline Datatype(MPI_Datatype i)
                : mpi_datatype(i) 
            {
            }
            inline Datatype(const Datatype &dt)
                : mpi_datatype(dt.mpi_datatype) 
            {
            }
            inline Datatype &operator =(const Datatype &dt)
            {
                mpi_datatype = dt.mpi_datatype;
                return *this;
            }
            inline bool operator ==(const Datatype &a) const
            {
                return (bool) (mpi_datatype == a.mpi_datatype);
            }
            inline bool operator !=(const Datatype &a) const
            {
                return (bool) !(*this == a);
            }
            inline Datatype &operator =(const MPI_Datatype &i)
            {
                mpi_datatype = i;
                return *this;
            }
            inline operator MPI_Datatype() const
            {
                return mpi_datatype;
            }
            typedef int Copy_attr_function(const Datatype &oldtype, int type_keyval, void *extra_state, const void *attribute_val_in, void *attribute_val_out, bool &flag);
            typedef int Delete_attr_function(Datatype &type, int type_keyval, void *attribute_val, void *extra_state);
            virtual Datatype Create_contiguous(int count) const;
            virtual Datatype Create_vector(int count, int blocklength, int stride) const;
            virtual Datatype Create_indexed(int count, const int array_of_blocklengths[], const int array_of_displacements[]) const;
            static Datatype Create_struct(int count, const int array_of_blocklengths[], const Aint array_of_displacements[], const Datatype array_if_types[]);
            virtual Datatype Create_hindexed(int count, const int array_of_blocklengths[], const Aint array_of_displacements[]) const;
            virtual Datatype Create_hvector(int count, int blocklength, Aint stride) const;
            virtual Datatype Create_indexed_block(int count, int blocklength, const int array_of_blocklengths[]) const;
            virtual Datatype Create_resized(const Aint lb, const Aint extent) const;
            virtual int Get_size() const;
            virtual void Get_extent(Aint &lb, Aint &extent) const;
            virtual void Get_true_extent(Aint &, Aint &) const;
            virtual void Commit();
            virtual void Free();
            virtual void Pack(const void *inbuf, int incount, void *outbuf, int outsize, int &position, const Comm &comm) const;
            virtual void Unpack(const void *inbuf, int insize, void *outbuf, int outcount, int &position, const Comm &comm) const;
            virtual int Pack_size(int incount, const Comm &comm) const;
            virtual Datatype Create_subarray(int ndims, const int array_of_sizes[], const int array_of_subsizes[], const int array_of_starts[], int order) const;
            virtual Datatype Dup() const;
            static int Create_keyval(Copy_attr_function *type_copy_attr_fn, Delete_attr_function *type_delete_attr_fn, void *extra_state);
            static int Create_keyval(MPI_Type_copy_attr_function *type_copy_attr_fn, MPI_Type_delete_attr_function *type_delete_attr_fn, void *extra_state);
            static int Create_keyval(Copy_attr_function *type_copy_attr_fn, MPI_Type_delete_attr_function *type_delete_attr_fn, void *extra_state);
            static int Create_keyval(MPI_Type_copy_attr_function *type_copy_attr_fn, Delete_attr_function *type_delete_attr_fn, void *extra_state);
        protected :
            static int do_create_keyval(MPI_Type_copy_attr_function *c_copy_fn, MPI_Type_delete_attr_function *c_delete_fn, Copy_attr_function *cxx_copy_fn, Delete_attr_function *cxx_delete_fn, void *extra_state, int &keyval);
        public :
            virtual void Delete_attr(int type_keyval);
            static void Free_keyval(int &type_keyval);
            virtual bool Get_attr(int type_keyval, void *attribute_val) const;
            virtual void Get_contents(int max_integers, int max_addresses, int max_datatypes, int array_of_integers[], Aint array_of_addresses[], Datatype array_of_datatypes[]) const;
            virtual void Get_envelope(int &num_integers, int &num_addresses, int &num_datatypes, int &combiner) const;
            virtual void Get_name(char *type_name, int &resultlen) const;
            virtual void Set_attr(int type_keyval, const void *attribute_val);
            virtual void Set_name(const char *type_name);
        protected :
            MPI_Datatype mpi_datatype;
        public :
            struct keyval_intercept_data_t
            {
                    MPI_Type_copy_attr_function *c_copy_fn;
                    MPI_Type_delete_attr_function *c_delete_fn;
                    Copy_attr_function *cxx_copy_fn;
                    Delete_attr_function *cxx_delete_fn;
                    void *extra_state;
            };
            static opal_mutex_t cxx_extra_states_lock;
    };
    typedef void User_function(const void *invec, void *inoutvec, int len, const Datatype &datatype);
    class Exception
    {
        public :
            inline Exception(int ec)
                : error_code(ec), error_string(0), error_class(- 1) 
            {
                (void) MPI_Error_class(error_code, &error_class);
                int resultlen;
                error_string = new char [MAX_ERROR_STRING];
                (void) MPI_Error_string(error_code, error_string, &resultlen);
            }
            inline ~Exception()
            {
                delete[] error_string;
            }
            inline Exception(const Exception &a)
                : error_code(a.error_code), error_class(a.error_class) 
            {
                error_string = new char [MAX_ERROR_STRING];
                for (int i = 0;
                    i < MAX_ERROR_STRING;
                    i++)
                    error_string[i] = a.error_string[i];
            }
            inline int Get_error_code() const
            {
                return error_code;
            }
            inline int Get_error_class() const
            {
                return error_class;
            }
            inline const char *Get_error_string() const
            {
                return error_string;
            }
        protected :
            int error_code;
            char *error_string;
            int error_class;
    };
    class Op
    {
        public :
            Op();
            Op(MPI_Op i);
            Op(const Op &op);
            virtual ~Op();
            Op &operator =(const Op &op);
            Op &operator =(const MPI_Op &i);
            inline bool operator ==(const Op &a);
            inline bool operator !=(const Op &a);
            inline operator MPI_Op() const;
            virtual void Init(User_function *func, bool commute);
            virtual void Free();
            virtual void Reduce_local(const void *inbuf, void *inoutbuf, int count, const MPI::Datatype &datatype) const;
            virtual bool Is_commutative(void) const;
        protected :
            MPI_Op mpi_op;
    };
    class Status
    {
            friend class MPI::Comm;
            friend class MPI::Request;
            friend class MPI::File;
        public :
            Status()
                : mpi_status() 
            {
            }
            Status(const Status &data)
                : mpi_status(data.mpi_status) 
            {
            }
            Status(const MPI_Status &i)
                : mpi_status(i) 
            {
            }
            virtual ~Status()
            {
            }
            Status &operator =(const Status &data)
            {
                mpi_status = data.mpi_status;
                return *this;
            }
            Status &operator =(const MPI_Status &i)
            {
                mpi_status = i;
                return *this;
            }
            operator MPI_Status() const
            {
                return mpi_status;
            }
            virtual int Get_count(const Datatype &datatype) const;
            virtual bool Is_cancelled() const;
            virtual int Get_elements(const Datatype &datatype) const;
            virtual int Get_source() const;
            virtual void Set_source(int source);
            virtual int Get_tag() const;
            virtual void Set_tag(int tag);
            virtual int Get_error() const;
            virtual void Set_error(int error);
            virtual void Set_elements(const MPI::Datatype &datatype, int count);
            virtual void Set_cancelled(bool flag);
        protected :
            MPI_Status mpi_status;
    };
    class Request
    {
        public :
            Request()
                : mpi_request(((MPI_Request) ((void *) &(ompi_request_null)))) 
            {
            }
            virtual ~Request()
            {
            }
            Request(MPI_Request i)
                : mpi_request(i) 
            {
            }
            Request(const Request &r)
                : mpi_request(r.mpi_request) 
            {
            }
            Request &operator =(const Request &r)
            {
                mpi_request = r.mpi_request;
                return *this;
            }
            bool operator ==(const Request &a)
            {
                return (bool) (mpi_request == a.mpi_request);
            }
            bool operator !=(const Request &a)
            {
                return (bool) !(*this == a);
            }
            Request &operator =(const MPI_Request &i)
            {
                mpi_request = i;
                return *this;
            }
            operator MPI_Request() const
            {
                return mpi_request;
            }
            virtual void Wait(Status &status);
            virtual void Wait();
            virtual bool Test(Status &status);
            virtual bool Test();
            virtual void Free(void);
            static int Waitany(int count, Request array[], Status &status);
            static int Waitany(int count, Request array[]);
            static bool Testany(int count, Request array[], int &index, Status &status);
            static bool Testany(int count, Request array[], int &index);
            static void Waitall(int count, Request req_array[], Status stat_array[]);
            static void Waitall(int count, Request req_array[]);
            static bool Testall(int count, Request req_array[], Status stat_array[]);
            static bool Testall(int count, Request req_array[]);
            static int Waitsome(int incount, Request req_array[], int array_of_indices[], Status stat_array[]);
            static int Waitsome(int incount, Request req_array[], int array_of_indices[]);
            static int Testsome(int incount, Request req_array[], int array_of_indices[], Status stat_array[]);
            static int Testsome(int incount, Request req_array[], int array_of_indices[]);
            virtual void Cancel(void) const;
            virtual bool Get_status(Status &status) const;
            virtual bool Get_status() const;
        protected :
            MPI_Request mpi_request;
        private :
    };
    class Prequest : public Request
    {
        public :
            Prequest()
            {
            }
            Prequest(const Request &p)
                : Request(p) 
            {
            }
            Prequest(const MPI_Request &i)
                : Request(i) 
            {
            }
            virtual ~Prequest()
            {
            }
            Prequest &operator =(const Request &r)
            {
                mpi_request = r;
                return *this;
            }
            Prequest &operator =(const Prequest &r)
            {
                mpi_request = r.mpi_request;
                return *this;
            }
            virtual void Start();
            static void Startall(int count, Prequest array_of_requests[]);
    };
    class Grequest : public MPI::Request
    {
        public :
            typedef int Query_function(void *, Status &);
            typedef int Free_function(void *);
            typedef int Cancel_function(void *, bool);
            Grequest()
            {
            }
            Grequest(const Request &req)
                : Request(req) 
            {
            }
            Grequest(const MPI_Request &req)
                : Request(req) 
            {
            }
            virtual ~Grequest()
            {
            }
            Grequest &operator =(const Request &req)
            {
                mpi_request = req;
                return (*this);
            }
            Grequest &operator =(const Grequest &req)
            {
                mpi_request = req.mpi_request;
                return (*this);
            }
            static Grequest Start(Query_function *, Free_function *, Cancel_function *, void *);
            virtual void Complete();
            struct Intercept_data_t
            {
                    void *id_extra;
                    Grequest::Query_function *id_cxx_query_fn;
                    Grequest::Free_function *id_cxx_free_fn;
                    Grequest::Cancel_function *id_cxx_cancel_fn;
            };
    };
    class Group
    {
        public :
            inline Group()
                : mpi_group(((MPI_Group) ((void *) &(ompi_mpi_group_null)))) 
            {
            }
            inline Group(MPI_Group i)
                : mpi_group(i) 
            {
            }
            inline Group(const Group &g)
                : mpi_group(g.mpi_group) 
            {
            }
            inline virtual ~Group()
            {
            }
            inline Group &operator =(const Group &g)
            {
                mpi_group = g.mpi_group;
                return *this;
            }
            inline bool operator ==(const Group &a)
            {
                return (bool) (mpi_group == a.mpi_group);
            }
            inline bool operator !=(const Group &a)
            {
                return (bool) !(*this == a);
            }
            inline Group &operator =(const MPI_Group &i)
            {
                mpi_group = i;
                return *this;
            }
            inline operator MPI_Group() const
            {
                return mpi_group;
            }
            inline MPI_Group mpi() const
            {
                return mpi_group;
            }
            virtual int Get_size() const;
            virtual int Get_rank() const;
            static void Translate_ranks(const Group &group1, int n, const int ranks1[], const Group &group2, int ranks2[]);
            static int Compare(const Group &group1, const Group &group2);
            static Group Union(const Group &group1, const Group &group2);
            static Group Intersect(const Group &group1, const Group &group2);
            static Group Difference(const Group &group1, const Group &group2);
            virtual Group Incl(int n, const int ranks[]) const;
            virtual Group Excl(int n, const int ranks[]) const;
            virtual Group Range_incl(int n, const int ranges[][3]) const;
            virtual Group Range_excl(int n, const int ranges[][3]) const;
            virtual void Free();
        protected :
            MPI_Group mpi_group;
    };
    class Comm_Null
    {
        public :
            inline Comm_Null()
                : mpi_comm(((MPI_Comm) ((void *) &(ompi_mpi_comm_null)))) 
            {
            }
            inline Comm_Null(const Comm_Null &data)
                : mpi_comm(data.mpi_comm) 
            {
            }
            inline Comm_Null(MPI_Comm data)
                : mpi_comm(data) 
            {
            }
            virtual inline ~Comm_Null()
            {
            }
            inline bool operator ==(const Comm_Null &data) const
            {
                return (bool) (mpi_comm == data.mpi_comm);
            }
            inline bool operator !=(const Comm_Null &data) const
            {
                return (bool) !(*this == data);
            }
            inline operator MPI_Comm() const
            {
                return mpi_comm;
            }
        protected :
            MPI_Comm mpi_comm;
    };
    class Comm : public Comm_Null
    {
        public :
            typedef void Errhandler_function(Comm &, int *, ...);
            typedef Errhandler_function Errhandler_fn;
            typedef int Copy_attr_function(const Comm &oldcomm, int comm_keyval, void *extra_state, void *attribute_val_in, void *attribute_val_out, bool &flag);
            typedef int Delete_attr_function(Comm &comm, int comm_keyval, void *attribute_val, void *extra_state);
            Comm();
            Comm(const Comm_Null &data);
            Comm(const Comm &data)
                : Comm_Null(data.mpi_comm) 
            {
            }
            Comm(MPI_Comm data)
                : Comm_Null(data) 
            {
            }
            virtual void Send(const void *buf, int count, const Datatype &datatype, int dest, int tag) const;
            virtual void Recv(void *buf, int count, const Datatype &datatype, int source, int tag, Status &status) const;
            virtual void Recv(void *buf, int count, const Datatype &datatype, int source, int tag) const;
            virtual void Bsend(const void *buf, int count, const Datatype &datatype, int dest, int tag) const;
            virtual void Ssend(const void *buf, int count, const Datatype &datatype, int dest, int tag) const;
            virtual void Rsend(const void *buf, int count, const Datatype &datatype, int dest, int tag) const;
            virtual Request Isend(const void *buf, int count, const Datatype &datatype, int dest, int tag) const;
            virtual Request Ibsend(const void *buf, int count, const Datatype &datatype, int dest, int tag) const;
            virtual Request Issend(const void *buf, int count, const Datatype &datatype, int dest, int tag) const;
            virtual Request Irsend(const void *buf, int count, const Datatype &datatype, int dest, int tag) const;
            virtual Request Irecv(void *buf, int count, const Datatype &datatype, int source, int tag) const;
            virtual bool Iprobe(int source, int tag, Status &status) const;
            virtual bool Iprobe(int source, int tag) const;
            virtual void Probe(int source, int tag, Status &status) const;
            virtual void Probe(int source, int tag) const;
            virtual Prequest Send_init(const void *buf, int count, const Datatype &datatype, int dest, int tag) const;
            virtual Prequest Bsend_init(const void *buf, int count, const Datatype &datatype, int dest, int tag) const;
            virtual Prequest Ssend_init(const void *buf, int count, const Datatype &datatype, int dest, int tag) const;
            virtual Prequest Rsend_init(const void *buf, int count, const Datatype &datatype, int dest, int tag) const;
            virtual Prequest Recv_init(void *buf, int count, const Datatype &datatype, int source, int tag) const;
            virtual void Sendrecv(const void *sendbuf, int sendcount, const Datatype &sendtype, int dest, int sendtag, void *recvbuf, int recvcount, const Datatype &recvtype, int source, int recvtag, Status &status) const;
            virtual void Sendrecv(const void *sendbuf, int sendcount, const Datatype &sendtype, int dest, int sendtag, void *recvbuf, int recvcount, const Datatype &recvtype, int source, int recvtag) const;
            virtual void Sendrecv_replace(void *buf, int count, const Datatype &datatype, int dest, int sendtag, int source, int recvtag, Status &status) const;
            virtual void Sendrecv_replace(void *buf, int count, const Datatype &datatype, int dest, int sendtag, int source, int recvtag) const;
            virtual Group Get_group() const;
            virtual int Get_size() const;
            virtual int Get_rank() const;
            static int Compare(const Comm &comm1, const Comm &comm2);
            virtual Comm &Clone() const  = 0;
            virtual void Free(void);
            virtual bool Is_inter() const;
            virtual void Barrier() const;
            virtual void Bcast(void *buffer, int count, const Datatype &datatype, int root) const;
            virtual void Gather(const void *sendbuf, int sendcount, const Datatype &sendtype, void *recvbuf, int recvcount, const Datatype &recvtype, int root) const;
            virtual void Gatherv(const void *sendbuf, int sendcount, const Datatype &sendtype, void *recvbuf, const int recvcounts[], const int displs[], const Datatype &recvtype, int root) const;
            virtual void Scatter(const void *sendbuf, int sendcount, const Datatype &sendtype, void *recvbuf, int recvcount, const Datatype &recvtype, int root) const;
            virtual void Scatterv(const void *sendbuf, const int sendcounts[], const int displs[], const Datatype &sendtype, void *recvbuf, int recvcount, const Datatype &recvtype, int root) const;
            virtual void Allgather(const void *sendbuf, int sendcount, const Datatype &sendtype, void *recvbuf, int recvcount, const Datatype &recvtype) const;
            virtual void Allgatherv(const void *sendbuf, int sendcount, const Datatype &sendtype, void *recvbuf, const int recvcounts[], const int displs[], const Datatype &recvtype) const;
            virtual void Alltoall(const void *sendbuf, int sendcount, const Datatype &sendtype, void *recvbuf, int recvcount, const Datatype &recvtype) const;
            virtual void Alltoallv(const void *sendbuf, const int sendcounts[], const int sdispls[], const Datatype &sendtype, void *recvbuf, const int recvcounts[], const int rdispls[], const Datatype &recvtype) const;
            virtual void Alltoallw(const void *sendbuf, const int sendcounts[], const int sdispls[], const Datatype sendtypes[], void *recvbuf, const int recvcounts[], const int rdispls[], const Datatype recvtypes[]) const;
            virtual void Reduce(const void *sendbuf, void *recvbuf, int count, const Datatype &datatype, const Op &op, int root) const;
            virtual void Allreduce(const void *sendbuf, void *recvbuf, int count, const Datatype &datatype, const Op &op) const;
            virtual void Reduce_scatter(const void *sendbuf, void *recvbuf, int recvcounts[], const Datatype &datatype, const Op &op) const;
            virtual void Disconnect();
            static Intercomm Get_parent();
            static Intercomm Join(const int fd);
            virtual void Get_name(char *comm_name, int &resultlen) const;
            virtual void Set_name(const char *comm_name);
            virtual int Get_topology() const;
            virtual void Abort(int errorcode);
            static Errhandler Create_errhandler(Comm::Errhandler_function *function);
            virtual void Set_errhandler(const Errhandler &errhandler) const;
            virtual Errhandler Get_errhandler() const;
            void Call_errhandler(int errorcode) const;
            static int Create_keyval(Copy_attr_function *comm_copy_attr_fn, Delete_attr_function *comm_delete_attr_fn, void *extra_state);
            static int Create_keyval(MPI_Comm_copy_attr_function *comm_copy_attr_fn, MPI_Comm_delete_attr_function *comm_delete_attr_fn, void *extra_state);
            static int Create_keyval(Copy_attr_function *comm_copy_attr_fn, MPI_Comm_delete_attr_function *comm_delete_attr_fn, void *extra_state);
            static int Create_keyval(MPI_Comm_copy_attr_function *comm_copy_attr_fn, Delete_attr_function *comm_delete_attr_fn, void *extra_state);
        protected :
            static int do_create_keyval(MPI_Comm_copy_attr_function *c_copy_fn, MPI_Comm_delete_attr_function *c_delete_fn, Copy_attr_function *cxx_copy_fn, Delete_attr_function *cxx_delete_fn, void *extra_state, int &keyval);
        public :
            static void Free_keyval(int &comm_keyval);
            virtual void Set_attr(int comm_keyval, const void *attribute_val) const;
            virtual bool Get_attr(int comm_keyval, void *attribute_val) const;
            virtual void Delete_attr(int comm_keyval);
            static int NULL_COPY_FN(const Comm &oldcomm, int comm_keyval, void *extra_state, void *attribute_val_in, void *attribute_val_out, bool &flag);
            static int DUP_FN(const Comm &oldcomm, int comm_keyval, void *extra_state, void *attribute_val_in, void *attribute_val_out, bool &flag);
            static int NULL_DELETE_FN(Comm &comm, int comm_keyval, void *attribute_val, void *extra_state);
        private :
        public :
            struct keyval_intercept_data_t
            {
                    MPI_Comm_copy_attr_function *c_copy_fn;
                    MPI_Comm_delete_attr_function *c_delete_fn;
                    Copy_attr_function *cxx_copy_fn;
                    Delete_attr_function *cxx_delete_fn;
                    void *extra_state;
            };
            static opal_mutex_t cxx_extra_states_lock;
    };
    class Win
    {
            friend class MPI::Comm;
            friend class MPI::Request;
        public :
            Win()
                : mpi_win(((MPI_Win) ((void *) &(ompi_mpi_win_null)))) 
            {
            }
            Win(const Win &data)
                : mpi_win(data.mpi_win) 
            {
            }
            Win(MPI_Win i)
                : mpi_win(i) 
            {
            }
            virtual ~Win()
            {
            }
            Win &operator =(const Win &data)
            {
                mpi_win = data.mpi_win;
                return *this;
            }
            Win &operator =(const MPI_Win &i)
            {
                mpi_win = i;
                return *this;
            }
            operator MPI_Win() const
            {
                return mpi_win;
            }
            typedef int Copy_attr_function(const Win &oldwin, int win_keyval, void *extra_state, void *attribute_val_in, void *attribute_val_out, bool &flag);
            typedef int Delete_attr_function(Win &win, int win_keyval, void *attribute_val, void *extra_state);
            typedef void Errhandler_function(Win &, int *, ...);
            typedef Errhandler_function Errhandler_fn;
            static MPI::Errhandler Create_errhandler(Errhandler_function *function);
            virtual void Set_errhandler(const MPI::Errhandler &errhandler) const;
            virtual MPI::Errhandler Get_errhandler() const;
            virtual void Accumulate(const void *origin_addr, int origin_count, const MPI::Datatype &origin_datatype, int target_rank, MPI::Aint target_disp, int target_count, const MPI::Datatype &target_datatype, const MPI::Op &op) const;
            virtual void Complete() const;
            static Win Create(const void *base, MPI::Aint size, int disp_unit, const MPI::Info &info, const MPI::Intracomm &comm);
            virtual void Fence(int assert) const;
            virtual void Free();
            virtual void Get(const void *origin_addr, int origin_count, const MPI::Datatype &origin_datatype, int target_rank, MPI::Aint target_disp, int target_count, const MPI::Datatype &target_datatype) const;
            virtual MPI::Group Get_group() const;
            virtual void Lock(int lock_type, int rank, int assert) const;
            virtual void Post(const MPI::Group &group, int assert) const;
            virtual void Put(const void *origin_addr, int origin_count, const MPI::Datatype &origin_datatype, int target_rank, MPI::Aint target_disp, int target_count, const MPI::Datatype &target_datatype) const;
            virtual void Start(const MPI::Group &group, int assert) const;
            virtual bool Test() const;
            virtual void Unlock(int rank) const;
            virtual void Wait() const;
            virtual void Call_errhandler(int errorcode) const;
            static int Create_keyval(Copy_attr_function *win_copy_attr_fn, Delete_attr_function *win_delete_attr_fn, void *extra_state);
            static int Create_keyval(MPI_Win_copy_attr_function *win_copy_attr_fn, MPI_Win_delete_attr_function *win_delete_attr_fn, void *extra_state);
            static int Create_keyval(Copy_attr_function *win_copy_attr_fn, MPI_Win_delete_attr_function *win_delete_attr_fn, void *extra_state);
            static int Create_keyval(MPI_Win_copy_attr_function *win_copy_attr_fn, Delete_attr_function *win_delete_attr_fn, void *extra_state);
        protected :
            static int do_create_keyval(MPI_Win_copy_attr_function *c_copy_fn, MPI_Win_delete_attr_function *c_delete_fn, Copy_attr_function *cxx_copy_fn, Delete_attr_function *cxx_delete_fn, void *extra_state, int &keyval);
        public :
            virtual void Delete_attr(int win_keyval);
            static void Free_keyval(int &win_keyval);
            bool Get_attr(const Win &win, int win_keyval, void *attribute_val) const;
            bool Get_attr(int win_keyval, void *attribute_val) const;
            virtual void Get_name(char *win_name, int &resultlen) const;
            virtual void Set_attr(int win_keyval, const void *attribute_val);
            virtual void Set_name(const char *win_name);
            struct keyval_intercept_data_t
            {
                    MPI_Win_copy_attr_function *c_copy_fn;
                    MPI_Win_delete_attr_function *c_delete_fn;
                    Copy_attr_function *cxx_copy_fn;
                    Delete_attr_function *cxx_delete_fn;
                    void *extra_state;
            };
            static opal_mutex_t cxx_extra_states_lock;
        protected :
            MPI_Win mpi_win;
    };
    typedef void Datarep_extent_function(const Datatype &datatype, Aint &file_extent, void *extra_state);
    typedef void Datarep_conversion_function(void *userbuf, Datatype &datatype, int count, void *filebuf, Offset position, void *extra_state);
    void Register_datarep(const char *datarep, Datarep_conversion_function *read_conversion_fn, Datarep_conversion_function *write_conversion_fn, Datarep_extent_function *dtype_file_extent_fn, void *extra_state);
    void Register_datarep(const char *datarep, MPI_Datarep_conversion_function *read_conversion_fn, Datarep_conversion_function *write_conversion_fn, Datarep_extent_function *dtype_file_extent_fn, void *extra_state);
    void Register_datarep(const char *datarep, Datarep_conversion_function *read_conversion_fn, MPI_Datarep_conversion_function *write_conversion_fn, Datarep_extent_function *dtype_file_extent_fn, void *extra_state);
    void Register_datarep(const char *datarep, MPI_Datarep_conversion_function *read_conversion_fn, MPI_Datarep_conversion_function *write_conversion_fn, Datarep_extent_function *dtype_file_extent_fn, void *extra_state);
    class File
    {
            friend class MPI::Comm;
            friend class MPI::Request;
        public :
            File()
                : mpi_file(((MPI_File) ((void *) &(ompi_mpi_file_null)))) 
            {
            }
            File(const File &data)
                : mpi_file(data.mpi_file) 
            {
            }
            File(MPI_File i)
                : mpi_file(i) 
            {
            }
            virtual ~File()
            {
            }
            File &operator =(const File &data)
            {
                mpi_file = data.mpi_file;
                return *this;
            }
            File &operator =(const MPI_File &i)
            {
                mpi_file = i;
                return *this;
            }
            operator MPI_File() const
            {
                return mpi_file;
            }
            void Close();
            static void Delete(const char *filename, const MPI::Info &info);
            int Get_amode() const;
            bool Get_atomicity() const;
            MPI::Offset Get_byte_offset(const MPI::Offset disp) const;
            MPI::Group Get_group() const;
            MPI::Info Get_info() const;
            MPI::Offset Get_position() const;
            MPI::Offset Get_position_shared() const;
            MPI::Offset Get_size() const;
            MPI::Aint Get_type_extent(const MPI::Datatype &datatype) const;
            void Get_view(MPI::Offset &disp, MPI::Datatype &etype, MPI::Datatype &filetype, char *datarep) const;
            MPI::Request Iread(void *buf, int count, const MPI::Datatype &datatype);
            MPI::Request Iread_at(MPI::Offset offset, void *buf, int count, const MPI::Datatype &datatype);
            MPI::Request Iread_shared(void *buf, int count, const MPI::Datatype &datatype);
            MPI::Request Iwrite(const void *buf, int count, const MPI::Datatype &datatype);
            MPI::Request Iwrite_at(MPI::Offset offset, const void *buf, int count, const MPI::Datatype &datatype);
            MPI::Request Iwrite_shared(const void *buf, int count, const MPI::Datatype &datatype);
            static MPI::File Open(const MPI::Intracomm &comm, const char *filename, int amode, const MPI::Info &info);
            void Preallocate(MPI::Offset size);
            void Read(void *buf, int count, const MPI::Datatype &datatype);
            void Read(void *buf, int count, const MPI::Datatype &datatype, MPI::Status &status);
            void Read_all(void *buf, int count, const MPI::Datatype &datatype);
            void Read_all(void *buf, int count, const MPI::Datatype &datatype, MPI::Status &status);
            void Read_all_begin(void *buf, int count, const MPI::Datatype &datatype);
            void Read_all_end(void *buf);
            void Read_all_end(void *buf, MPI::Status &status);
            void Read_at(MPI::Offset offset, void *buf, int count, const MPI::Datatype &datatype);
            void Read_at(MPI::Offset offset, void *buf, int count, const MPI::Datatype &datatype, MPI::Status &status);
            void Read_at_all(MPI::Offset offset, void *buf, int count, const MPI::Datatype &datatype);
            void Read_at_all(MPI::Offset offset, void *buf, int count, const MPI::Datatype &datatype, MPI::Status &status);
            void Read_at_all_begin(MPI::Offset offset, void *buf, int count, const MPI::Datatype &datatype);
            void Read_at_all_end(void *buf);
            void Read_at_all_end(void *buf, MPI::Status &status);
            void Read_ordered(void *buf, int count, const MPI::Datatype &datatype);
            void Read_ordered(void *buf, int count, const MPI::Datatype &datatype, MPI::Status &status);
            void Read_ordered_begin(void *buf, int count, const MPI::Datatype &datatype);
            void Read_ordered_end(void *buf);
            void Read_ordered_end(void *buf, MPI::Status &status);
            void Read_shared(void *buf, int count, const MPI::Datatype &datatype);
            void Read_shared(void *buf, int count, const MPI::Datatype &datatype, MPI::Status &status);
            void Seek(MPI::Offset offset, int whence);
            void Seek_shared(MPI::Offset offset, int whence);
            void Set_atomicity(bool flag);
            void Set_info(const MPI::Info &info);
            void Set_size(MPI::Offset size);
            void Set_view(MPI::Offset disp, const MPI::Datatype &etype, const MPI::Datatype &filetype, const char *datarep, const MPI::Info &info);
            void Sync();
            void Write(const void *buf, int count, const MPI::Datatype &datatype);
            void Write(const void *buf, int count, const MPI::Datatype &datatype, MPI::Status &status);
            void Write_all(const void *buf, int count, const MPI::Datatype &datatype);
            void Write_all(const void *buf, int count, const MPI::Datatype &datatype, MPI::Status &status);
            void Write_all_begin(const void *buf, int count, const MPI::Datatype &datatype);
            void Write_all_end(const void *buf);
            void Write_all_end(const void *buf, MPI::Status &status);
            void Write_at(MPI::Offset offset, const void *buf, int count, const MPI::Datatype &datatype);
            void Write_at(MPI::Offset offset, const void *buf, int count, const MPI::Datatype &datatype, MPI::Status &status);
            void Write_at_all(MPI::Offset offset, const void *buf, int count, const MPI::Datatype &datatype);
            void Write_at_all(MPI::Offset offset, const void *buf, int count, const MPI::Datatype &datatype, MPI::Status &status);
            void Write_at_all_begin(MPI::Offset offset, const void *buf, int count, const MPI::Datatype &datatype);
            void Write_at_all_end(const void *buf);
            void Write_at_all_end(const void *buf, MPI::Status &status);
            void Write_ordered(const void *buf, int count, const MPI::Datatype &datatype);
            void Write_ordered(const void *buf, int count, const MPI::Datatype &datatype, MPI::Status &status);
            void Write_ordered_begin(const void *buf, int count, const MPI::Datatype &datatype);
            void Write_ordered_end(const void *buf);
            void Write_ordered_end(const void *buf, MPI::Status &status);
            void Write_shared(const void *buf, int count, const MPI::Datatype &datatype);
            void Write_shared(const void *buf, int count, const MPI::Datatype &datatype, MPI::Status &status);
            typedef void Errhandler_function(MPI::File &, int *, ...);
            typedef Errhandler_function Errhandler_fn;
            static MPI::Errhandler Create_errhandler(Errhandler_function *function);
            MPI::Errhandler Get_errhandler() const;
            void Set_errhandler(const MPI::Errhandler &errhandler) const;
            void Call_errhandler(int errorcode) const;
        protected :
            MPI_File mpi_file;
    };
    class Errhandler
    {
        public :
            inline Errhandler()
                : mpi_errhandler(((MPI_Errhandler) ((void *) &(ompi_mpi_errhandler_null)))) 
            {
            }
            inline virtual ~Errhandler()
            {
            }
            inline Errhandler(MPI_Errhandler i)
                : mpi_errhandler(i) 
            {
            }
            inline Errhandler(const Errhandler &e)
                : mpi_errhandler(e.mpi_errhandler) 
            {
            }
            inline Errhandler &operator =(const Errhandler &e)
            {
                mpi_errhandler = e.mpi_errhandler;
                return *this;
            }
            inline bool operator ==(const Errhandler &a)
            {
                return (bool) (mpi_errhandler == a.mpi_errhandler);
            }
            inline bool operator !=(const Errhandler &a)
            {
                return (bool) !(*this == a);
            }
            inline Errhandler &operator =(const MPI_Errhandler &i)
            {
                mpi_errhandler = i;
                return *this;
            }
            inline operator MPI_Errhandler() const
            {
                return mpi_errhandler;
            }
            virtual void Free();
        private :
            MPI_Errhandler mpi_errhandler;
    };
    class Intracomm : public Comm
    {
        public :
            Intracomm()
            {
            }
            Intracomm(const Comm_Null &data)
                : Comm(data) 
            {
            }
            Intracomm(const Intracomm &data)
                : Comm(data.mpi_comm) 
            {
            }
            inline Intracomm(MPI_Comm data);
            Intracomm &operator =(const Intracomm &data)
            {
                mpi_comm = data.mpi_comm;
                return *this;
            }
            Intracomm &operator =(const Comm_Null &data)
            {
                mpi_comm = data;
                return *this;
            }
            Intracomm &operator =(const MPI_Comm &data)
            {
                mpi_comm = data;
                return *this;
            }
            virtual void Scan(const void *sendbuf, void *recvbuf, int count, const Datatype &datatype, const Op &op) const;
            virtual void Exscan(const void *sendbuf, void *recvbuf, int count, const Datatype &datatype, const Op &op) const;
            Intracomm Dup() const;
            virtual Intracomm &Clone() const;
            virtual Intracomm Create(const Group &group) const;
            virtual Intracomm Split(int color, int key) const;
            virtual Intercomm Create_intercomm(int local_leader, const Comm &peer_comm, int remote_leader, int tag) const;
            virtual Cartcomm Create_cart(int ndims, const int dims[], const bool periods[], bool reorder) const;
            virtual Graphcomm Create_graph(int nnodes, const int index[], const int edges[], bool reorder) const;
            virtual Intercomm Accept(const char *port_name, const Info &info, int root) const;
            virtual Intercomm Connect(const char *port_name, const Info &info, int root) const;
            virtual Intercomm Spawn(const char *command, const char *argv[], int maxprocs, const Info &info, int root) const;
            virtual Intercomm Spawn(const char *command, const char *argv[], int maxprocs, const Info &info, int root, int array_of_errcodes[]) const;
            virtual Intercomm Spawn_multiple(int count, const char *array_of_commands[], const char **array_of_argv[], const int array_of_maxprocs[], const Info array_of_info[], int root);
            virtual Intercomm Spawn_multiple(int count, const char *array_of_commands[], const char **array_of_argv[], const int array_of_maxprocs[], const Info array_of_info[], int root, int array_of_errcodes[]);
        protected :
            static inline MPI_Info *convert_info_to_mpi_info(int p_nbr, const Info p_info_tbl[]);
    };
    class Cartcomm : public Intracomm
    {
        public :
            Cartcomm()
            {
            }
            Cartcomm(const Comm_Null &data)
                : Intracomm(data) 
            {
            }
            inline Cartcomm(const MPI_Comm &data);
            Cartcomm(const Cartcomm &data)
                : Intracomm(data.mpi_comm) 
            {
            }
            Cartcomm &operator =(const Cartcomm &data)
            {
                mpi_comm = data.mpi_comm;
                return *this;
            }
            Cartcomm &operator =(const Comm_Null &data)
            {
                mpi_comm = data;
                return *this;
            }
            Cartcomm &operator =(const MPI_Comm &data)
            {
                mpi_comm = data;
                return *this;
            }
            Cartcomm Dup() const;
            virtual Cartcomm &Clone() const;
            virtual int Get_dim() const;
            virtual void Get_topo(int maxdims, int dims[], bool periods[], int coords[]) const;
            virtual int Get_cart_rank(const int coords[]) const;
            virtual void Get_coords(int rank, int maxdims, int coords[]) const;
            virtual void Shift(int direction, int disp, int &rank_source, int &rank_dest) const;
            virtual Cartcomm Sub(const bool remain_dims[]);
            virtual int Map(int ndims, const int dims[], const bool periods[]) const;
    };
    class Graphcomm : public Intracomm
    {
        public :
            Graphcomm()
            {
            }
            Graphcomm(const Comm_Null &data)
                : Intracomm(data) 
            {
            }
            inline Graphcomm(const MPI_Comm &data);
            Graphcomm(const Graphcomm &data)
                : Intracomm(data.mpi_comm) 
            {
            }
            Graphcomm &operator =(const Graphcomm &data)
            {
                mpi_comm = data.mpi_comm;
                return *this;
            }
            Graphcomm &operator =(const Comm_Null &data)
            {
                mpi_comm = data;
                return *this;
            }
            Graphcomm &operator =(const MPI_Comm &data)
            {
                mpi_comm = data;
                return *this;
            }
            Graphcomm Dup() const;
            virtual Graphcomm &Clone() const;
            virtual void Get_dims(int nnodes[], int nedges[]) const;
            virtual void Get_topo(int maxindex, int maxedges, int index[], int edges[]) const;
            virtual int Get_neighbors_count(int rank) const;
            virtual void Get_neighbors(int rank, int maxneighbors, int neighbors[]) const;
            virtual int Map(int nnodes, const int index[], const int edges[]) const;
    };
    class Intercomm : public Comm
    {
        public :
            Intercomm()
                : Comm(((MPI_Comm) ((void *) &(ompi_mpi_comm_null)))) 
            {
            }
            Intercomm(const Comm_Null &data)
                : Comm(data) 
            {
            }
            Intercomm(MPI_Comm data)
                : Comm(data) 
            {
            }
            Intercomm(const Intercomm &data)
                : Comm(data.mpi_comm) 
            {
            }
            Intercomm &operator =(const Intercomm &data)
            {
                mpi_comm = data.mpi_comm;
                return *this;
            }
            Intercomm &operator =(const Comm_Null &data)
            {
                mpi_comm = data;
                return *this;
            }
            Intercomm &operator =(const MPI_Comm &data)
            {
                mpi_comm = data;
                return *this;
            }
            Intercomm Dup() const;
            virtual Intercomm &Clone() const;
            virtual int Get_remote_size() const;
            virtual Group Get_remote_group() const;
            virtual Intracomm Merge(bool high);
            virtual Intercomm Create(const Group &group) const;
            virtual Intercomm Split(int color, int key) const;
    };
    class Info
    {
            friend class MPI::Comm;
            friend class MPI::Request;
        public :
            Info()
                : mpi_info(((MPI_Info) ((void *) &(ompi_mpi_info_null)))) 
            {
            }
            Info(const Info &data)
                : mpi_info(data.mpi_info) 
            {
            }
            Info(MPI_Info i)
                : mpi_info(i) 
            {
            }
            virtual ~Info()
            {
            }
            Info &operator =(const Info &data)
            {
                mpi_info = data.mpi_info;
                return *this;
            }
            Info &operator =(const MPI_Info &i)
            {
                mpi_info = i;
                return *this;
            }
            operator MPI_Info() const
            {
                return mpi_info;
            }
            static Info Create();
            virtual void Delete(const char *key);
            virtual Info Dup() const;
            virtual void Free();
            virtual bool Get(const char *key, int valuelen, char *value) const;
            virtual int Get_nkeys() const;
            virtual void Get_nthkey(int n, char *key) const;
            virtual bool Get_valuelen(const char *key, int &valuelen) const;
            virtual void Set(const char *key, const char *value);
        protected :
            MPI_Info mpi_info;
    };
    extern const char ompi_libcxx_version_string[];
}
inline MPI::Datatype MPI::Datatype::Create_contiguous(int count) const
{
    MPI_Datatype newtype;
    (void) MPI_Type_contiguous(count, mpi_datatype, &newtype);
    return newtype;
}
inline MPI::Datatype MPI::Datatype::Create_vector(int count, int blocklength, int stride) const
{
    MPI_Datatype newtype;
    (void) MPI_Type_vector(count, blocklength, stride, mpi_datatype, &newtype);
    return newtype;
}
inline MPI::Datatype MPI::Datatype::Create_indexed(int count, const int array_of_blocklengths[], const int array_of_displacements[]) const
{
    MPI_Datatype newtype;
    (void) MPI_Type_indexed(count, const_cast<int * >(array_of_blocklengths), const_cast<int * >(array_of_displacements), mpi_datatype, &newtype);
    return newtype;
}
inline MPI::Datatype MPI::Datatype::Create_struct(int count, const int array_of_blocklengths[], const MPI::Aint array_of_displacements[], const MPI::Datatype array_of_types[])
{
    MPI_Datatype newtype;
    int i;
    MPI_Datatype *type_array = new MPI_Datatype [count];
    for (i = 0;
        i < count;
        i++)
        type_array[i] = array_of_types[i];
    (void) MPI_Type_create_struct(count, const_cast<int * >(array_of_blocklengths), const_cast<MPI_Aint * >(array_of_displacements), type_array, &newtype);
    delete[] type_array;
    return newtype;
}
inline MPI::Datatype MPI::Datatype::Create_hindexed(int count, const int array_of_blocklengths[], const MPI::Aint array_of_displacements[]) const
{
    MPI_Datatype newtype;
    (void) MPI_Type_create_hindexed(count, const_cast<int * >(array_of_blocklengths), const_cast<MPI_Aint * >(array_of_displacements), mpi_datatype, &newtype);
    return newtype;
}
inline MPI::Datatype MPI::Datatype::Create_hvector(int count, int blocklength, MPI::Aint stride) const
{
    MPI_Datatype newtype;
    (void) MPI_Type_create_hvector(count, blocklength, (MPI_Aint) stride, mpi_datatype, &newtype);
    return newtype;
}
inline MPI::Datatype MPI::Datatype::Create_indexed_block(int count, int blocklength, const int array_of_displacements[]) const
{
    MPI_Datatype newtype;
    (void) MPI_Type_create_indexed_block(count, blocklength, const_cast<int * >(array_of_displacements), mpi_datatype, &newtype);
    return newtype;
}
inline MPI::Datatype MPI::Datatype::Create_resized(const MPI::Aint lb, const MPI::Aint extent) const
{
    MPI_Datatype newtype;
    (void) MPI_Type_create_resized(mpi_datatype, lb, extent, &newtype);
    return (newtype);
}
inline int MPI::Datatype::Get_size() const
{
    int size;
    (void) MPI_Type_size(mpi_datatype, &size);
    return size;
}
inline void MPI::Datatype::Get_extent(MPI::Aint &lb, MPI::Aint &extent) const
{
    (void) MPI_Type_get_extent(mpi_datatype, &lb, &extent);
}
inline void MPI::Datatype::Get_true_extent(MPI::Aint &lb, MPI::Aint &extent) const
{
    (void) MPI_Type_get_true_extent(mpi_datatype, &lb, &extent);
}
inline void MPI::Datatype::Commit()
{
    (void) MPI_Type_commit(&mpi_datatype);
}
inline void MPI::Datatype::Pack(const void *inbuf, int incount, void *outbuf, int outsize, int &position, const MPI::Comm &comm) const
{
    (void) MPI_Pack(const_cast<void * >(inbuf), incount, mpi_datatype, outbuf, outsize, &position, comm);
}
inline void MPI::Datatype::Unpack(const void *inbuf, int insize, void *outbuf, int outcount, int &position, const MPI::Comm &comm) const
{
    (void) MPI_Unpack(const_cast<void * >(inbuf), insize, &position, outbuf, outcount, mpi_datatype, comm);
}
inline int MPI::Datatype::Pack_size(int incount, const MPI::Comm &comm) const
{
    int size;
    (void) MPI_Pack_size(incount, mpi_datatype, comm, &size);
    return size;
}
inline MPI::Datatype MPI::Datatype::Create_subarray(int ndims, const int array_of_sizes[], const int array_of_subsizes[], const int array_of_starts[], int order) const
{
    MPI_Datatype type;
    (void) MPI_Type_create_subarray(ndims, const_cast<int * >(array_of_sizes), const_cast<int * >(array_of_subsizes), const_cast<int * >(array_of_starts), order, mpi_datatype, &type);
    return type;
}
inline MPI::Datatype MPI::Datatype::Dup() const
{
    MPI_Datatype type;
    (void) MPI_Type_dup(mpi_datatype, &type);
    return type;
}
inline int MPI::Datatype::Create_keyval(MPI::Datatype::Copy_attr_function *type_copy_attr_fn, MPI::Datatype::Delete_attr_function *type_delete_attr_fn, void *extra_state)
{
    int ret, keyval;
    ret = do_create_keyval(__null, __null, type_copy_attr_fn, type_delete_attr_fn, extra_state, keyval);
    return (0 == ret) ? keyval : ret;
}
inline int MPI::Datatype::Create_keyval(MPI_Type_copy_attr_function *type_copy_attr_fn, MPI_Type_delete_attr_function *type_delete_attr_fn, void *extra_state)
{
    int ret, keyval;
    ret = do_create_keyval(type_copy_attr_fn, type_delete_attr_fn, __null, __null, extra_state, keyval);
    return (0 == ret) ? keyval : ret;
}
inline int MPI::Datatype::Create_keyval(MPI::Datatype::Copy_attr_function *type_copy_attr_fn, MPI_Type_delete_attr_function *type_delete_attr_fn, void *extra_state)
{
    int ret, keyval;
    ret = do_create_keyval(__null, type_delete_attr_fn, type_copy_attr_fn, __null, extra_state, keyval);
    return (0 == ret) ? keyval : ret;
}
inline int MPI::Datatype::Create_keyval(MPI_Type_copy_attr_function *type_copy_attr_fn, MPI::Datatype::Delete_attr_function *type_delete_attr_fn, void *extra_state)
{
    int ret, keyval;
    ret = do_create_keyval(type_copy_attr_fn, __null, __null, type_delete_attr_fn, extra_state, keyval);
    return (0 == ret) ? keyval : ret;
}
inline void MPI::Datatype::Delete_attr(int type_keyval)
{
    (void) MPI_Type_delete_attr(mpi_datatype, type_keyval);
}
inline void MPI::Datatype::Free_keyval(int &type_keyval)
{
    (void) MPI_Type_free_keyval(&type_keyval);
}
inline bool MPI::Datatype::Get_attr(int type_keyval, void *attribute_val) const
{
    int ret;
    (void) MPI_Type_get_attr(mpi_datatype, type_keyval, attribute_val, &ret);
    return ((bool) (ret));
}
inline void MPI::Datatype::Get_contents(int max_integers, int max_addresses, int max_datatypes, int array_of_integers[], MPI::Aint array_of_addresses[], MPI::Datatype array_of_datatypes[]) const
{
    int i;
    MPI_Datatype *c_datatypes = new MPI_Datatype [max_datatypes];
    (void) MPI_Type_get_contents(mpi_datatype, max_integers, max_addresses, max_datatypes, const_cast<int * >(array_of_integers), const_cast<MPI_Aint * >(array_of_addresses), c_datatypes);
    for (i = 0;
        i < max_datatypes;
        ++i)
    {
        array_of_datatypes[i] = c_datatypes[i];
    }
    delete[] c_datatypes;
}
inline void MPI::Datatype::Get_envelope(int &num_integers, int &num_addresses, int &num_datatypes, int &combiner) const
{
    (void) MPI_Type_get_envelope(mpi_datatype, &num_integers, &num_addresses, &num_datatypes, &combiner);
}
inline void MPI::Datatype::Get_name(char *type_name, int &resultlen) const
{
    (void) MPI_Type_get_name(mpi_datatype, type_name, &resultlen);
}
inline void MPI::Datatype::Set_attr(int type_keyval, const void *attribute_val)
{
    (void) MPI_Type_set_attr(mpi_datatype, type_keyval, const_cast<void * >(attribute_val));
}
inline void MPI::Datatype::Set_name(const char *type_name)
{
    (void) MPI_Type_set_name(mpi_datatype, const_cast<char * >(type_name));
}
extern "C"
{
    extern void *memcpy(void *__restrict __dest, __const void *__restrict __src, size_t __n) throw () __attribute__((__nonnull__(1, 2)));
    extern void *memmove(void *__dest, __const void *__src, size_t __n) throw () __attribute__((__nonnull__(1, 2)));
    extern void *memccpy(void *__restrict __dest, __const void *__restrict __src, int __c, size_t __n) throw () __attribute__((__nonnull__(1, 2)));
    extern void *memset(void *__s, int __c, size_t __n) throw () __attribute__((__nonnull__(1)));
    extern int memcmp(__const void *__s1, __const void *__s2, size_t __n) throw () __attribute__((__pure__)) __attribute__((__nonnull__(1, 2)));
    extern "C++"
    {
        extern void *memchr(void *__s, int __c, size_t __n) throw () __asm ("memchr") __attribute__((__pure__)) __attribute__((__nonnull__(1)));
        extern __const void *memchr(__const void *__s, int __c, size_t __n) throw () __asm ("memchr") __attribute__((__pure__)) __attribute__((__nonnull__(1)));
        extern __inline __attribute__((__always_inline__)) __attribute__((__gnu_inline__, __artificial__)) void *memchr(void *__s, int __c, size_t __n) throw ()
        {
            return __builtin_memchr(__s, __c, __n);
        }
        extern __inline __attribute__((__always_inline__)) __attribute__((__gnu_inline__, __artificial__)) __const void *memchr(__const void *__s, int __c, size_t __n) throw ()
        {
            return __builtin_memchr(__s, __c, __n);
        }
    }
    extern "C++"
    void *rawmemchr(void *__s, int __c) throw () __asm ("rawmemchr") __attribute__((__pure__)) __attribute__((__nonnull__(1)));
    extern "C++"
    __const void *rawmemchr(__const void *__s, int __c) throw () __asm ("rawmemchr") __attribute__((__pure__)) __attribute__((__nonnull__(1)));
    extern "C++"
    void *memrchr(void *__s, int __c, size_t __n) throw () __asm ("memrchr") __attribute__((__pure__)) __attribute__((__nonnull__(1)));
    extern "C++"
    __const void *memrchr(__const void *__s, int __c, size_t __n) throw () __asm ("memrchr") __attribute__((__pure__)) __attribute__((__nonnull__(1)));
    extern char *strcpy(char *__restrict __dest, __const char *__restrict __src) throw () __attribute__((__nonnull__(1, 2)));
    extern char *strncpy(char *__restrict __dest, __const char *__restrict __src, size_t __n) throw () __attribute__((__nonnull__(1, 2)));
    extern char *strcat(char *__restrict __dest, __const char *__restrict __src) throw () __attribute__((__nonnull__(1, 2)));
    extern char *strncat(char *__restrict __dest, __const char *__restrict __src, size_t __n) throw () __attribute__((__nonnull__(1, 2)));
    extern int strcmp(__const char *__s1, __const char *__s2) throw () __attribute__((__pure__)) __attribute__((__nonnull__(1, 2)));
    extern int strncmp(__const char *__s1, __const char *__s2, size_t __n) throw () __attribute__((__pure__)) __attribute__((__nonnull__(1, 2)));
    extern int strcoll(__const char *__s1, __const char *__s2) throw () __attribute__((__pure__)) __attribute__((__nonnull__(1, 2)));
    extern size_t strxfrm(char *__restrict __dest, __const char *__restrict __src, size_t __n) throw () __attribute__((__nonnull__(2)));
    extern int strcoll_l(__const char *__s1, __const char *__s2, __locale_t __l) throw () __attribute__((__pure__)) __attribute__((__nonnull__(1, 2, 3)));
    extern size_t strxfrm_l(char *__dest, __const char *__src, size_t __n, __locale_t __l) throw () __attribute__((__nonnull__(2, 4)));
    extern char *strdup(__const char *__s) throw () __attribute__((__malloc__)) __attribute__((__nonnull__(1)));
    extern char *strndup(__const char *__string, size_t __n) throw () __attribute__((__malloc__)) __attribute__((__nonnull__(1)));
    extern "C++"
    {
        extern char *strchr(char *__s, int __c) throw () __asm ("strchr") __attribute__((__pure__)) __attribute__((__nonnull__(1)));
        extern __const char *strchr(__const char *__s, int __c) throw () __asm ("strchr") __attribute__((__pure__)) __attribute__((__nonnull__(1)));
        extern __inline __attribute__((__always_inline__)) __attribute__((__gnu_inline__, __artificial__)) char *strchr(char *__s, int __c) throw ()
        {
            return __builtin_strchr(__s, __c);
        }
        extern __inline __attribute__((__always_inline__)) __attribute__((__gnu_inline__, __artificial__)) __const char *strchr(__const char *__s, int __c) throw ()
        {
            return __builtin_strchr(__s, __c);
        }
    }
    extern "C++"
    {
        extern char *strrchr(char *__s, int __c) throw () __asm ("strrchr") __attribute__((__pure__)) __attribute__((__nonnull__(1)));
        extern __const char *strrchr(__const char *__s, int __c) throw () __asm ("strrchr") __attribute__((__pure__)) __attribute__((__nonnull__(1)));
        extern __inline __attribute__((__always_inline__)) __attribute__((__gnu_inline__, __artificial__)) char *strrchr(char *__s, int __c) throw ()
        {
            return __builtin_strrchr(__s, __c);
        }
        extern __inline __attribute__((__always_inline__)) __attribute__((__gnu_inline__, __artificial__)) __const char *strrchr(__const char *__s, int __c) throw ()
        {
            return __builtin_strrchr(__s, __c);
        }
    }
    extern "C++"
    char *strchrnul(char *__s, int __c) throw () __asm ("strchrnul") __attribute__((__pure__)) __attribute__((__nonnull__(1)));
    extern "C++"
    __const char *strchrnul(__const char *__s, int __c) throw () __asm ("strchrnul") __attribute__((__pure__)) __attribute__((__nonnull__(1)));
    extern size_t strcspn(__const char *__s, __const char *__reject) throw () __attribute__((__pure__)) __attribute__((__nonnull__(1, 2)));
    extern size_t strspn(__const char *__s, __const char *__accept) throw () __attribute__((__pure__)) __attribute__((__nonnull__(1, 2)));
    extern "C++"
    {
        extern char *strpbrk(char *__s, __const char *__accept) throw () __asm ("strpbrk") __attribute__((__pure__)) __attribute__((__nonnull__(1, 2)));
        extern __const char *strpbrk(__const char *__s, __const char *__accept) throw () __asm ("strpbrk") __attribute__((__pure__)) __attribute__((__nonnull__(1, 2)));
        extern __inline __attribute__((__always_inline__)) __attribute__((__gnu_inline__, __artificial__)) char *strpbrk(char *__s, __const char *__accept) throw ()
        {
            return __builtin_strpbrk(__s, __accept);
        }
        extern __inline __attribute__((__always_inline__)) __attribute__((__gnu_inline__, __artificial__)) __const char *strpbrk(__const char *__s, __const char *__accept) throw ()
        {
            return __builtin_strpbrk(__s, __accept);
        }
    }
    extern "C++"
    {
        extern char *strstr(char *__haystack, __const char *__needle) throw () __asm ("strstr") __attribute__((__pure__)) __attribute__((__nonnull__(1, 2)));
        extern __const char *strstr(__const char *__haystack, __const char *__needle) throw () __asm ("strstr") __attribute__((__pure__)) __attribute__((__nonnull__(1, 2)));
        extern __inline __attribute__((__always_inline__)) __attribute__((__gnu_inline__, __artificial__)) char *strstr(char *__haystack, __const char *__needle) throw ()
        {
            return __builtin_strstr(__haystack, __needle);
        }
        extern __inline __attribute__((__always_inline__)) __attribute__((__gnu_inline__, __artificial__)) __const char *strstr(__const char *__haystack, __const char *__needle) throw ()
        {
            return __builtin_strstr(__haystack, __needle);
        }
    }
    extern char *strtok(char *__restrict __s, __const char *__restrict __delim) throw () __attribute__((__nonnull__(2)));
    extern char *__strtok_r(char *__restrict __s, __const char *__restrict __delim, char **__restrict __save_ptr) throw () __attribute__((__nonnull__(2, 3)));
    extern char *strtok_r(char *__restrict __s, __const char *__restrict __delim, char **__restrict __save_ptr) throw () __attribute__((__nonnull__(2, 3)));
    extern "C++"
    char *strcasestr(char *__haystack, __const char *__needle) throw () __asm ("strcasestr") __attribute__((__pure__)) __attribute__((__nonnull__(1, 2)));
    extern "C++"
    __const char *strcasestr(__const char *__haystack, __const char *__needle) throw () __asm ("strcasestr") __attribute__((__pure__)) __attribute__((__nonnull__(1, 2)));
    extern void *memmem(__const void *__haystack, size_t __haystacklen, __const void *__needle, size_t __needlelen) throw () __attribute__((__pure__)) __attribute__((__nonnull__(1, 3)));
    extern void *__mempcpy(void *__restrict __dest, __const void *__restrict __src, size_t __n) throw () __attribute__((__nonnull__(1, 2)));
    extern void *mempcpy(void *__restrict __dest, __const void *__restrict __src, size_t __n) throw () __attribute__((__nonnull__(1, 2)));
    extern size_t strlen(__const char *__s) throw () __attribute__((__pure__)) __attribute__((__nonnull__(1)));
    extern size_t strnlen(__const char *__string, size_t __maxlen) throw () __attribute__((__pure__)) __attribute__((__nonnull__(1)));
    extern char *strerror(int __errnum) throw ();
    extern char *strerror_r(int __errnum, char *__buf, size_t __buflen) throw () __attribute__((__nonnull__(2)));
    extern char *strerror_l(int __errnum, __locale_t __l) throw ();
    extern void __bzero(void *__s, size_t __n) throw () __attribute__((__nonnull__(1)));
    extern void bcopy(__const void *__src, void *__dest, size_t __n) throw () __attribute__((__nonnull__(1, 2)));
    extern void bzero(void *__s, size_t __n) throw () __attribute__((__nonnull__(1)));
    extern int bcmp(__const void *__s1, __const void *__s2, size_t __n) throw () __attribute__((__pure__)) __attribute__((__nonnull__(1, 2)));
    extern "C++"
    {
        extern char *index(char *__s, int __c) throw () __asm ("index") __attribute__((__pure__)) __attribute__((__nonnull__(1)));
        extern __const char *index(__const char *__s, int __c) throw () __asm ("index") __attribute__((__pure__)) __attribute__((__nonnull__(1)));
        extern __inline __attribute__((__always_inline__)) __attribute__((__gnu_inline__, __artificial__)) char *index(char *__s, int __c) throw ()
        {
            return __builtin_index(__s, __c);
        }
        extern __inline __attribute__((__always_inline__)) __attribute__((__gnu_inline__, __artificial__)) __const char *index(__const char *__s, int __c) throw ()
        {
            return __builtin_index(__s, __c);
        }
    }
    extern "C++"
    {
        extern char *rindex(char *__s, int __c) throw () __asm ("rindex") __attribute__((__pure__)) __attribute__((__nonnull__(1)));
        extern __const char *rindex(__const char *__s, int __c) throw () __asm ("rindex") __attribute__((__pure__)) __attribute__((__nonnull__(1)));
        extern __inline __attribute__((__always_inline__)) __attribute__((__gnu_inline__, __artificial__)) char *rindex(char *__s, int __c) throw ()
        {
            return __builtin_rindex(__s, __c);
        }
        extern __inline __attribute__((__always_inline__)) __attribute__((__gnu_inline__, __artificial__)) __const char *rindex(__const char *__s, int __c) throw ()
        {
            return __builtin_rindex(__s, __c);
        }
    }
    extern int ffs(int __i) throw () __attribute__((__const__));
    extern int ffsl(long int __l) throw () __attribute__((__const__));
    __extension__
    extern int ffsll(long long int __ll) throw () __attribute__((__const__));
    extern int strcasecmp(__const char *__s1, __const char *__s2) throw () __attribute__((__pure__)) __attribute__((__nonnull__(1, 2)));
    extern int strncasecmp(__const char *__s1, __const char *__s2, size_t __n) throw () __attribute__((__pure__)) __attribute__((__nonnull__(1, 2)));
    extern int strcasecmp_l(__const char *__s1, __const char *__s2, __locale_t __loc) throw () __attribute__((__pure__)) __attribute__((__nonnull__(1, 2, 3)));
    extern int strncasecmp_l(__const char *__s1, __const char *__s2, size_t __n, __locale_t __loc) throw () __attribute__((__pure__)) __attribute__((__nonnull__(1, 2, 4)));
    extern char *strsep(char **__restrict __stringp, __const char *__restrict __delim) throw () __attribute__((__nonnull__(1, 2)));
    extern char *strsignal(int __sig) throw ();
    extern char *__stpcpy(char *__restrict __dest, __const char *__restrict __src) throw () __attribute__((__nonnull__(1, 2)));
    extern char *stpcpy(char *__restrict __dest, __const char *__restrict __src) throw () __attribute__((__nonnull__(1, 2)));
    extern char *__stpncpy(char *__restrict __dest, __const char *__restrict __src, size_t __n) throw () __attribute__((__nonnull__(1, 2)));
    extern char *stpncpy(char *__restrict __dest, __const char *__restrict __src, size_t __n) throw () __attribute__((__nonnull__(1, 2)));
    extern int strverscmp(__const char *__s1, __const char *__s2) throw () __attribute__((__pure__)) __attribute__((__nonnull__(1, 2)));
    extern char *strfry(char *__string) throw () __attribute__((__nonnull__(1)));
    extern void *memfrob(void *__s, size_t __n) throw () __attribute__((__nonnull__(1)));
    extern "C++"
    char *basename(char *__filename) throw () __asm ("basename") __attribute__((__nonnull__(1)));
    extern "C++"
    __const char *basename(__const char *__filename) throw () __asm ("basename") __attribute__((__nonnull__(1)));
}
inline void MPI::Attach_buffer(void *buffer, int size)
{
    (void) MPI_Buffer_attach(buffer, size);
}
inline int MPI::Detach_buffer(void *&buffer)
{
    int size;
    (void) MPI_Buffer_detach(&buffer, &size);
    return size;
}
inline void MPI::Compute_dims(int nnodes, int ndims, int dims[])
{
    (void) MPI_Dims_create(nnodes, ndims, dims);
}
inline void MPI::Get_processor_name(char *name, int &resultlen)
{
    (void) MPI_Get_processor_name(name, &resultlen);
}
inline void MPI::Get_error_string(int errorcode, char *string, int &resultlen)
{
    (void) MPI_Error_string(errorcode, string, &resultlen);
}
inline int MPI::Get_error_class(int errorcode)
{
    int errorclass;
    (void) MPI_Error_class(errorcode, &errorclass);
    return errorclass;
}
inline double MPI::Wtime()
{
    return (MPI_Wtime());
}
inline double MPI::Wtick()
{
    return (MPI_Wtick());
}
inline void MPI::Real_init()
{
    MPI::InitializeIntercepts();
}
inline void MPI::Init(int &argc, char **&argv)
{
    (void) MPI_Init(&argc, &argv);
    Real_init();
}
inline void MPI::Init()
{
    (void) MPI_Init(0, 0);
    Real_init();
}
inline void MPI::Finalize()
{
    (void) MPI_Finalize();
}
inline bool MPI::Is_initialized()
{
    int t;
    (void) MPI_Initialized(&t);
    return ((bool) (t));
}
inline bool MPI::Is_finalized()
{
    int t;
    (void) MPI_Finalized(&t);
    return ((bool) (t));
}
inline int MPI::Init_thread(int required)
{
    int provided;
    (void) MPI_Init_thread(0, __null, required, &provided);
    Real_init();
    return provided;
}
inline int MPI::Init_thread(int &argc, char **&argv, int required)
{
    int provided;
    (void) MPI_Init_thread(&argc, &argv, required, &provided);
    Real_init();
    return provided;
}
inline bool MPI::Is_thread_main()
{
    int flag;
    (void) MPI_Is_thread_main(&flag);
    return ((bool) (flag == 1));
}
inline int MPI::Query_thread()
{
    int provided;
    (void) MPI_Query_thread(&provided);
    return provided;
}
inline void *MPI::Alloc_mem(MPI::Aint size, const MPI::Info &info)
{
    void *baseptr;
    (void) MPI_Alloc_mem(size, info, &baseptr);
    return baseptr;
}
inline void MPI::Free_mem(void *base)
{
    (void) MPI_Free_mem(base);
}
inline void MPI::Close_port(const char *port_name)
{
    (void) MPI_Close_port(const_cast<char * >(port_name));
}
inline void MPI::Lookup_name(const char *service_name, const MPI::Info &info, char *port_name)
{
    (void) MPI_Lookup_name(const_cast<char * >(service_name), info, port_name);
}
inline void MPI::Open_port(const MPI::Info &info, char *port_name)
{
    (void) MPI_Open_port(info, port_name);
}
inline void MPI::Publish_name(const char *service_name, const MPI::Info &info, const char *port_name)
{
    (void) MPI_Publish_name(const_cast<char * >(service_name), info, const_cast<char * >(port_name));
}
inline void MPI::Unpublish_name(const char *service_name, const MPI::Info &info, const char *port_name)
{
    (void) MPI_Unpublish_name(const_cast<char * >(service_name), info, const_cast<char * >(port_name));
}
inline void MPI::Pcontrol(const int level, ...)
{
    va_list ap;
    __builtin_va_start(ap, level);
    (void) MPI_Pcontrol(level, ap);
    __builtin_va_end(ap);
}
inline void MPI::Get_version(int &version, int &subversion)
{
    (void) MPI_Get_version(&version, &subversion);
}
inline MPI::Aint MPI::Get_address(void *location)
{
    MPI::Aint ret;
    MPI_Get_address(location, &ret);
    return ret;
}
inline void MPI::Request::Wait(MPI::Status &status)
{
    (void) MPI_Wait(&mpi_request, &status.mpi_status);
}
inline void MPI::Request::Wait()
{
    (void) MPI_Wait(&mpi_request, ((MPI_Status *) 0));
}
inline void MPI::Request::Free()
{
    (void) MPI_Request_free(&mpi_request);
}
inline bool MPI::Request::Test(MPI::Status &status)
{
    int t;
    (void) MPI_Test(&mpi_request, &t, &status.mpi_status);
    return ((bool) (t));
}
inline bool MPI::Request::Test()
{
    int t;
    (void) MPI_Test(&mpi_request, &t, ((MPI_Status *) 0));
    return ((bool) (t));
}
inline int MPI::Request::Waitany(int count, MPI::Request array[], MPI::Status &status)
{
    int index, i;
    MPI_Request *array_of_requests = new MPI_Request [count];
    for (i = 0;
        i < count;
        i++)
    {
        array_of_requests[i] = array[i];
    }
    (void) MPI_Waitany(count, array_of_requests, &index, &status.mpi_status);
    for (i = 0;
        i < count;
        i++)
    {
        array[i] = array_of_requests[i];
    }
    delete[] array_of_requests;
    return index;
}
inline int MPI::Request::Waitany(int count, MPI::Request array[])
{
    int index, i;
    MPI_Request *array_of_requests = new MPI_Request [count];
    for (i = 0;
        i < count;
        i++)
    {
        array_of_requests[i] = array[i];
    }
    (void) MPI_Waitany(count, array_of_requests, &index, ((MPI_Status *) 0));
    for (i = 0;
        i < count;
        i++)
    {
        array[i] = array_of_requests[i];
    }
    delete[] array_of_requests;
    return index;
}
inline bool MPI::Request::Testany(int count, MPI::Request array[], int &index, MPI::Status &status)
{
    int i, flag;
    MPI_Request *array_of_requests = new MPI_Request [count];
    for (i = 0;
        i < count;
        i++)
    {
        array_of_requests[i] = array[i];
    }
    (void) MPI_Testany(count, array_of_requests, &index, &flag, &status.mpi_status);
    for (i = 0;
        i < count;
        i++)
    {
        array[i] = array_of_requests[i];
    }
    delete[] array_of_requests;
    return (bool) (flag != 0 ? true : false);
}
inline bool MPI::Request::Testany(int count, MPI::Request array[], int &index)
{
    int i, flag;
    MPI_Request *array_of_requests = new MPI_Request [count];
    for (i = 0;
        i < count;
        i++)
    {
        array_of_requests[i] = array[i];
    }
    (void) MPI_Testany(count, array_of_requests, &index, &flag, ((MPI_Status *) 0));
    for (i = 0;
        i < count;
        i++)
    {
        array[i] = array_of_requests[i];
    }
    delete[] array_of_requests;
    return ((bool) (flag));
}
inline void MPI::Request::Waitall(int count, MPI::Request req_array[], MPI::Status stat_array[])
{
    int i;
    MPI_Request *array_of_requests = new MPI_Request [count];
    MPI_Status *array_of_statuses = new MPI_Status [count];
    for (i = 0;
        i < count;
        i++)
    {
        array_of_requests[i] = req_array[i];
    }
    (void) MPI_Waitall(count, array_of_requests, array_of_statuses);
    for (i = 0;
        i < count;
        i++)
    {
        req_array[i] = array_of_requests[i];
        stat_array[i] = array_of_statuses[i];
    }
    delete[] array_of_requests;
    delete[] array_of_statuses;
}
inline void MPI::Request::Waitall(int count, MPI::Request req_array[])
{
    int i;
    MPI_Request *array_of_requests = new MPI_Request [count];
    for (i = 0;
        i < count;
        i++)
    {
        array_of_requests[i] = req_array[i];
    }
    (void) MPI_Waitall(count, array_of_requests, ((MPI_Status *) 0));
    for (i = 0;
        i < count;
        i++)
    {
        req_array[i] = array_of_requests[i];
    }
    delete[] array_of_requests;
}
inline bool MPI::Request::Testall(int count, MPI::Request req_array[], MPI::Status stat_array[])
{
    int i, flag;
    MPI_Request *array_of_requests = new MPI_Request [count];
    MPI_Status *array_of_statuses = new MPI_Status [count];
    for (i = 0;
        i < count;
        i++)
    {
        array_of_requests[i] = req_array[i];
    }
    (void) MPI_Testall(count, array_of_requests, &flag, array_of_statuses);
    for (i = 0;
        i < count;
        i++)
    {
        req_array[i] = array_of_requests[i];
        stat_array[i] = array_of_statuses[i];
    }
    delete[] array_of_requests;
    delete[] array_of_statuses;
    return ((bool) (flag));
}
inline bool MPI::Request::Testall(int count, MPI::Request req_array[])
{
    int i, flag;
    MPI_Request *array_of_requests = new MPI_Request [count];
    for (i = 0;
        i < count;
        i++)
    {
        array_of_requests[i] = req_array[i];
    }
    (void) MPI_Testall(count, array_of_requests, &flag, ((MPI_Status *) 0));
    for (i = 0;
        i < count;
        i++)
    {
        req_array[i] = array_of_requests[i];
    }
    delete[] array_of_requests;
    return ((bool) (flag));
}
inline int MPI::Request::Waitsome(int incount, MPI::Request req_array[], int array_of_indices[], MPI::Status stat_array[])
{
    int i, outcount;
    MPI_Request *array_of_requests = new MPI_Request [incount];
    MPI_Status *array_of_statuses = new MPI_Status [incount];
    for (i = 0;
        i < incount;
        i++)
    {
        array_of_requests[i] = req_array[i];
    }
    (void) MPI_Waitsome(incount, array_of_requests, &outcount, array_of_indices, array_of_statuses);
    for (i = 0;
        i < incount;
        i++)
    {
        req_array[i] = array_of_requests[i];
        stat_array[i] = array_of_statuses[i];
    }
    delete[] array_of_requests;
    delete[] array_of_statuses;
    return outcount;
}
inline int MPI::Request::Waitsome(int incount, MPI::Request req_array[], int array_of_indices[])
{
    int i, outcount;
    MPI_Request *array_of_requests = new MPI_Request [incount];
    for (i = 0;
        i < incount;
        i++)
    {
        array_of_requests[i] = req_array[i];
    }
    (void) MPI_Waitsome(incount, array_of_requests, &outcount, array_of_indices, ((MPI_Status *) 0));
    for (i = 0;
        i < incount;
        i++)
    {
        req_array[i] = array_of_requests[i];
    }
    delete[] array_of_requests;
    return outcount;
}
inline int MPI::Request::Testsome(int incount, MPI::Request req_array[], int array_of_indices[], MPI::Status stat_array[])
{
    int i, outcount;
    MPI_Request *array_of_requests = new MPI_Request [incount];
    MPI_Status *array_of_statuses = new MPI_Status [incount];
    for (i = 0;
        i < incount;
        i++)
    {
        array_of_requests[i] = req_array[i];
    }
    (void) MPI_Testsome(incount, array_of_requests, &outcount, array_of_indices, array_of_statuses);
    for (i = 0;
        i < incount;
        i++)
    {
        req_array[i] = array_of_requests[i];
        stat_array[i] = array_of_statuses[i];
    }
    delete[] array_of_requests;
    delete[] array_of_statuses;
    return outcount;
}
inline int MPI::Request::Testsome(int incount, MPI::Request req_array[], int array_of_indices[])
{
    int i, outcount;
    MPI_Request *array_of_requests = new MPI_Request [incount];
    for (i = 0;
        i < incount;
        i++)
    {
        array_of_requests[i] = req_array[i];
    }
    (void) MPI_Testsome(incount, array_of_requests, &outcount, array_of_indices, ((MPI_Status *) 0));
    for (i = 0;
        i < incount;
        i++)
    {
        req_array[i] = array_of_requests[i];
    }
    delete[] array_of_requests;
    return outcount;
}
inline void MPI::Request::Cancel(void) const
{
    (void) MPI_Cancel(const_cast<MPI_Request * >(&mpi_request));
}
inline void MPI::Prequest::Start()
{
    (void) MPI_Start(&mpi_request);
}
inline void MPI::Prequest::Startall(int count, MPI::Prequest array_of_requests[])
{
    MPI_Request *mpi_requests = new MPI_Request [count];
    int i;
    for (i = 0;
        i < count;
        i++)
    {
        mpi_requests[i] = array_of_requests[i];
    }
    (void) MPI_Startall(count, mpi_requests);
    for (i = 0;
        i < count;
        i++)
    {
        array_of_requests[i].mpi_request = mpi_requests[i];
    }
    delete[] mpi_requests;
}
inline bool MPI::Request::Get_status(MPI::Status &status) const
{
    int flag = 0;
    MPI_Status c_status;
    (void) MPI_Request_get_status(mpi_request, &flag, &c_status);
    if (flag)
    {
        status = c_status;
    }
    return ((bool) (flag));
}
inline bool MPI::Request::Get_status() const
{
    int flag;
    (void) MPI_Request_get_status(mpi_request, &flag, ((MPI_Status *) 0));
    return ((bool) (flag));
}
inline MPI::Grequest MPI::Grequest::Start(Query_function *query_fn, Free_function *free_fn, Cancel_function *cancel_fn, void *extra)
{
    MPI_Request grequest = 0;
    Intercept_data_t *new_extra = new MPI::Grequest::Intercept_data_t;
    new_extra->id_extra = extra;
    new_extra->id_cxx_query_fn = query_fn;
    new_extra->id_cxx_free_fn = free_fn;
    new_extra->id_cxx_cancel_fn = cancel_fn;
    (void) MPI_Grequest_start(ompi_mpi_cxx_grequest_query_fn_intercept, ompi_mpi_cxx_grequest_free_fn_intercept, ompi_mpi_cxx_grequest_cancel_fn_intercept, new_extra, &grequest);
    return (grequest);
}
inline void MPI::Grequest::Complete()
{
    (void) MPI_Grequest_complete(mpi_request);
}
inline void MPI::Comm::Send(const void *buf, int count, const MPI::Datatype &datatype, int dest, int tag) const
{
    (void) MPI_Send(const_cast<void * >(buf), count, datatype, dest, tag, mpi_comm);
}
inline void MPI::Comm::Recv(void *buf, int count, const MPI::Datatype &datatype, int source, int tag, MPI::Status &status) const
{
    (void) MPI_Recv(buf, count, datatype, source, tag, mpi_comm, &status.mpi_status);
}
inline void MPI::Comm::Recv(void *buf, int count, const MPI::Datatype &datatype, int source, int tag) const
{
    (void) MPI_Recv(buf, count, datatype, source, tag, mpi_comm, ((MPI_Status *) 0));
}
inline void MPI::Comm::Bsend(const void *buf, int count, const MPI::Datatype &datatype, int dest, int tag) const
{
    (void) MPI_Bsend(const_cast<void * >(buf), count, datatype, dest, tag, mpi_comm);
}
inline void MPI::Comm::Ssend(const void *buf, int count, const MPI::Datatype &datatype, int dest, int tag) const
{
    (void) MPI_Ssend(const_cast<void * >(buf), count, datatype, dest, tag, mpi_comm);
}
inline void MPI::Comm::Rsend(const void *buf, int count, const MPI::Datatype &datatype, int dest, int tag) const
{
    (void) MPI_Rsend(const_cast<void * >(buf), count, datatype, dest, tag, mpi_comm);
}
inline MPI::Request MPI::Comm::Isend(const void *buf, int count, const MPI::Datatype &datatype, int dest, int tag) const
{
    MPI_Request request;
    (void) MPI_Isend(const_cast<void * >(buf), count, datatype, dest, tag, mpi_comm, &request);
    return request;
}
inline MPI::Request MPI::Comm::Ibsend(const void *buf, int count, const MPI::Datatype &datatype, int dest, int tag) const
{
    MPI_Request request;
    (void) MPI_Ibsend(const_cast<void * >(buf), count, datatype, dest, tag, mpi_comm, &request);
    return request;
}
inline MPI::Request MPI::Comm::Issend(const void *buf, int count, const MPI::Datatype &datatype, int dest, int tag) const
{
    MPI_Request request;
    (void) MPI_Issend(const_cast<void * >(buf), count, datatype, dest, tag, mpi_comm, &request);
    return request;
}
inline MPI::Request MPI::Comm::Irsend(const void *buf, int count, const MPI::Datatype &datatype, int dest, int tag) const
{
    MPI_Request request;
    (void) MPI_Irsend(const_cast<void * >(buf), count, datatype, dest, tag, mpi_comm, &request);
    return request;
}
inline MPI::Request MPI::Comm::Irecv(void *buf, int count, const MPI::Datatype &datatype, int source, int tag) const
{
    MPI_Request request;
    (void) MPI_Irecv(buf, count, datatype, source, tag, mpi_comm, &request);
    return request;
}
inline bool MPI::Comm::Iprobe(int source, int tag, MPI::Status &status) const
{
    int t;
    (void) MPI_Iprobe(source, tag, mpi_comm, &t, &status.mpi_status);
    return ((bool) (t));
}
inline bool MPI::Comm::Iprobe(int source, int tag) const
{
    int t;
    (void) MPI_Iprobe(source, tag, mpi_comm, &t, ((MPI_Status *) 0));
    return ((bool) (t));
}
inline void MPI::Comm::Probe(int source, int tag, MPI::Status &status) const
{
    (void) MPI_Probe(source, tag, mpi_comm, &status.mpi_status);
}
inline void MPI::Comm::Probe(int source, int tag) const
{
    (void) MPI_Probe(source, tag, mpi_comm, ((MPI_Status *) 0));
}
inline MPI::Prequest MPI::Comm::Send_init(const void *buf, int count, const MPI::Datatype &datatype, int dest, int tag) const
{
    MPI_Request request;
    (void) MPI_Send_init(const_cast<void * >(buf), count, datatype, dest, tag, mpi_comm, &request);
    return request;
}
inline MPI::Prequest MPI::Comm::Bsend_init(const void *buf, int count, const MPI::Datatype &datatype, int dest, int tag) const
{
    MPI_Request request;
    (void) MPI_Bsend_init(const_cast<void * >(buf), count, datatype, dest, tag, mpi_comm, &request);
    return request;
}
inline MPI::Prequest MPI::Comm::Ssend_init(const void *buf, int count, const MPI::Datatype &datatype, int dest, int tag) const
{
    MPI_Request request;
    (void) MPI_Ssend_init(const_cast<void * >(buf), count, datatype, dest, tag, mpi_comm, &request);
    return request;
}
inline MPI::Prequest MPI::Comm::Rsend_init(const void *buf, int count, const MPI::Datatype &datatype, int dest, int tag) const
{
    MPI_Request request;
    (void) MPI_Rsend_init(const_cast<void * >(buf), count, datatype, dest, tag, mpi_comm, &request);
    return request;
}
inline MPI::Prequest MPI::Comm::Recv_init(void *buf, int count, const MPI::Datatype &datatype, int source, int tag) const
{
    MPI_Request request;
    (void) MPI_Recv_init(buf, count, datatype, source, tag, mpi_comm, &request);
    return request;
}
inline void MPI::Comm::Sendrecv(const void *sendbuf, int sendcount, const MPI::Datatype &sendtype, int dest, int sendtag, void *recvbuf, int recvcount, const MPI::Datatype &recvtype, int source, int recvtag, MPI::Status &status) const
{
    (void) MPI_Sendrecv(const_cast<void * >(sendbuf), sendcount, sendtype, dest, sendtag, recvbuf, recvcount, recvtype, source, recvtag, mpi_comm, &status.mpi_status);
}
inline void MPI::Comm::Sendrecv(const void *sendbuf, int sendcount, const MPI::Datatype &sendtype, int dest, int sendtag, void *recvbuf, int recvcount, const MPI::Datatype &recvtype, int source, int recvtag) const
{
    (void) MPI_Sendrecv(const_cast<void * >(sendbuf), sendcount, sendtype, dest, sendtag, recvbuf, recvcount, recvtype, source, recvtag, mpi_comm, ((MPI_Status *) 0));
}
inline void MPI::Comm::Sendrecv_replace(void *buf, int count, const MPI::Datatype &datatype, int dest, int sendtag, int source, int recvtag, MPI::Status &status) const
{
    (void) MPI_Sendrecv_replace(buf, count, datatype, dest, sendtag, source, recvtag, mpi_comm, &status.mpi_status);
}
inline void MPI::Comm::Sendrecv_replace(void *buf, int count, const MPI::Datatype &datatype, int dest, int sendtag, int source, int recvtag) const
{
    (void) MPI_Sendrecv_replace(buf, count, datatype, dest, sendtag, source, recvtag, mpi_comm, ((MPI_Status *) 0));
}
inline MPI::Group MPI::Comm::Get_group() const
{
    MPI_Group group;
    (void) MPI_Comm_group(mpi_comm, &group);
    return group;
}
inline int MPI::Comm::Get_size() const
{
    int size;
    (void) MPI_Comm_size(mpi_comm, &size);
    return size;
}
inline int MPI::Comm::Get_rank() const
{
    int rank;
    (void) MPI_Comm_rank(mpi_comm, &rank);
    return rank;
}
inline int MPI::Comm::Compare(const MPI::Comm &comm1, const MPI::Comm &comm2)
{
    int result;
    (void) MPI_Comm_compare(comm1, comm2, &result);
    return result;
}
inline void MPI::Comm::Free(void)
{
    (void) MPI_Comm_free(&mpi_comm);
}
inline bool MPI::Comm::Is_inter() const
{
    int t;
    (void) MPI_Comm_test_inter(mpi_comm, &t);
    return ((bool) (t));
}
inline void MPI::Comm::Barrier() const
{
    (void) MPI_Barrier(mpi_comm);
}
inline void MPI::Comm::Bcast(void *buffer, int count, const MPI::Datatype &datatype, int root) const
{
    (void) MPI_Bcast(buffer, count, datatype, root, mpi_comm);
}
inline void MPI::Comm::Gather(const void *sendbuf, int sendcount, const MPI::Datatype &sendtype, void *recvbuf, int recvcount, const MPI::Datatype &recvtype, int root) const
{
    (void) MPI_Gather(const_cast<void * >(sendbuf), sendcount, sendtype, recvbuf, recvcount, recvtype, root, mpi_comm);
}
inline void MPI::Comm::Gatherv(const void *sendbuf, int sendcount, const MPI::Datatype &sendtype, void *recvbuf, const int recvcounts[], const int displs[], const MPI::Datatype &recvtype, int root) const
{
    (void) MPI_Gatherv(const_cast<void * >(sendbuf), sendcount, sendtype, recvbuf, const_cast<int * >(recvcounts), const_cast<int * >(displs), recvtype, root, mpi_comm);
}
inline void MPI::Comm::Scatter(const void *sendbuf, int sendcount, const MPI::Datatype &sendtype, void *recvbuf, int recvcount, const MPI::Datatype &recvtype, int root) const
{
    (void) MPI_Scatter(const_cast<void * >(sendbuf), sendcount, sendtype, recvbuf, recvcount, recvtype, root, mpi_comm);
}
inline void MPI::Comm::Scatterv(const void *sendbuf, const int sendcounts[], const int displs[], const MPI::Datatype &sendtype, void *recvbuf, int recvcount, const MPI::Datatype &recvtype, int root) const
{
    (void) MPI_Scatterv(const_cast<void * >(sendbuf), const_cast<int * >(sendcounts), const_cast<int * >(displs), sendtype, recvbuf, recvcount, recvtype, root, mpi_comm);
}
inline void MPI::Comm::Allgather(const void *sendbuf, int sendcount, const MPI::Datatype &sendtype, void *recvbuf, int recvcount, const MPI::Datatype &recvtype) const
{
    (void) MPI_Allgather(const_cast<void * >(sendbuf), sendcount, sendtype, recvbuf, recvcount, recvtype, mpi_comm);
}
inline void MPI::Comm::Allgatherv(const void *sendbuf, int sendcount, const MPI::Datatype &sendtype, void *recvbuf, const int recvcounts[], const int displs[], const MPI::Datatype &recvtype) const
{
    (void) MPI_Allgatherv(const_cast<void * >(sendbuf), sendcount, sendtype, recvbuf, const_cast<int * >(recvcounts), const_cast<int * >(displs), recvtype, mpi_comm);
}
inline void MPI::Comm::Alltoall(const void *sendbuf, int sendcount, const MPI::Datatype &sendtype, void *recvbuf, int recvcount, const MPI::Datatype &recvtype) const
{
    (void) MPI_Alltoall(const_cast<void * >(sendbuf), sendcount, sendtype, recvbuf, recvcount, recvtype, mpi_comm);
}
inline void MPI::Comm::Alltoallv(const void *sendbuf, const int sendcounts[], const int sdispls[], const MPI::Datatype &sendtype, void *recvbuf, const int recvcounts[], const int rdispls[], const MPI::Datatype &recvtype) const
{
    (void) MPI_Alltoallv(const_cast<void * >(sendbuf), const_cast<int * >(sendcounts), const_cast<int * >(sdispls), sendtype, recvbuf, const_cast<int * >(recvcounts), const_cast<int * >(rdispls), recvtype, mpi_comm);
}
inline void MPI::Comm::Alltoallw(const void *sendbuf, const int sendcounts[], const int sdispls[], const MPI::Datatype sendtypes[], void *recvbuf, const int recvcounts[], const int rdispls[], const MPI::Datatype recvtypes[]) const
{
    const int comm_size = Get_size();
    MPI_Datatype *const data_type_tbl = new MPI_Datatype [2 * comm_size];
    for (int i_rank = 0;
        i_rank < comm_size;
        i_rank++)
    {
        data_type_tbl[i_rank] = sendtypes[i_rank];
        data_type_tbl[i_rank + comm_size] = recvtypes[i_rank];
    }
    (void) MPI_Alltoallw(const_cast<void * >(sendbuf), const_cast<int * >(sendcounts), const_cast<int * >(sdispls), data_type_tbl, recvbuf, const_cast<int * >(recvcounts), const_cast<int * >(rdispls), &data_type_tbl[comm_size], mpi_comm);
    delete[] data_type_tbl;
}
inline void MPI::Comm::Reduce(const void *sendbuf, void *recvbuf, int count, const MPI::Datatype &datatype, const MPI::Op &op, int root) const
{
    (void) MPI_Reduce(const_cast<void * >(sendbuf), recvbuf, count, datatype, op, root, mpi_comm);
}
inline void MPI::Comm::Allreduce(const void *sendbuf, void *recvbuf, int count, const MPI::Datatype &datatype, const MPI::Op &op) const
{
    (void) MPI_Allreduce(const_cast<void * >(sendbuf), recvbuf, count, datatype, op, mpi_comm);
}
inline void MPI::Comm::Reduce_scatter(const void *sendbuf, void *recvbuf, int recvcounts[], const MPI::Datatype &datatype, const MPI::Op &op) const
{
    (void) MPI_Reduce_scatter(const_cast<void * >(sendbuf), recvbuf, recvcounts, datatype, op, mpi_comm);
}
inline void MPI::Comm::Disconnect()
{
    (void) MPI_Comm_disconnect(&mpi_comm);
}
inline MPI::Intercomm MPI::Comm::Get_parent()
{
    MPI_Comm parent;
    MPI_Comm_get_parent(&parent);
    return parent;
}
inline MPI::Intercomm MPI::Comm::Join(const int fd)
{
    MPI_Comm newcomm;
    (void) MPI_Comm_join((int) fd, &newcomm);
    return newcomm;
}
inline void MPI::Comm::Get_name(char *comm_name, int &resultlen) const
{
    (void) MPI_Comm_get_name(mpi_comm, comm_name, &resultlen);
}
inline void MPI::Comm::Set_name(const char *comm_name)
{
    (void) MPI_Comm_set_name(mpi_comm, const_cast<char * >(comm_name));
}
inline int MPI::Comm::Get_topology() const
{
    int status;
    (void) MPI_Topo_test(mpi_comm, &status);
    return status;
}
inline void MPI::Comm::Abort(int errorcode)
{
    (void) MPI_Abort(mpi_comm, errorcode);
}
inline MPI::Errhandler MPI::Comm::Get_errhandler() const
{
    MPI_Errhandler errhandler;
    MPI_Comm_get_errhandler(mpi_comm, &errhandler);
    return errhandler;
}
inline void MPI::Comm::Set_errhandler(const MPI::Errhandler &errhandler) const
{
    (void) MPI_Comm_set_errhandler(mpi_comm, errhandler);
}
inline void MPI::Comm::Call_errhandler(int errorcode) const
{
    (void) MPI_Comm_call_errhandler(mpi_comm, errorcode);
}
inline int MPI::Comm::Create_keyval(MPI::Comm::Copy_attr_function *comm_copy_attr_fn, MPI::Comm::Delete_attr_function *comm_delete_attr_fn, void *extra_state)
{
    int ret, keyval;
    ret = do_create_keyval(__null, __null, comm_copy_attr_fn, comm_delete_attr_fn, extra_state, keyval);
    return (0 == ret) ? keyval : ret;
}
inline int MPI::Comm::Create_keyval(MPI_Comm_copy_attr_function *comm_copy_attr_fn, MPI_Comm_delete_attr_function *comm_delete_attr_fn, void *extra_state)
{
    int ret, keyval;
    ret = do_create_keyval(comm_copy_attr_fn, comm_delete_attr_fn, __null, __null, extra_state, keyval);
    return (0 == ret) ? keyval : ret;
}
inline int MPI::Comm::Create_keyval(MPI::Comm::Copy_attr_function *comm_copy_attr_fn, MPI_Comm_delete_attr_function *comm_delete_attr_fn, void *extra_state)
{
    int ret, keyval;
    ret = do_create_keyval(__null, comm_delete_attr_fn, comm_copy_attr_fn, __null, extra_state, keyval);
    return (0 == ret) ? keyval : ret;
}
inline int MPI::Comm::Create_keyval(MPI_Comm_copy_attr_function *comm_copy_attr_fn, MPI::Comm::Delete_attr_function *comm_delete_attr_fn, void *extra_state)
{
    int ret, keyval;
    ret = do_create_keyval(comm_copy_attr_fn, __null, __null, comm_delete_attr_fn, extra_state, keyval);
    return (0 == ret) ? keyval : ret;
}
inline void MPI::Comm::Free_keyval(int &comm_keyval)
{
    (void) MPI_Keyval_free(&comm_keyval);
}
inline void MPI::Comm::Set_attr(int comm_keyval, const void *attribute_val) const
{
    (void) MPI_Attr_put(mpi_comm, comm_keyval, const_cast<void * >(attribute_val));
}
inline bool MPI::Comm::Get_attr(int comm_keyval, void *attribute_val) const
{
    int flag;
    (void) MPI_Attr_get(mpi_comm, comm_keyval, attribute_val, &flag);
    return ((bool) (flag));
}
inline void MPI::Comm::Delete_attr(int comm_keyval)
{
    (void) MPI_Attr_delete(mpi_comm, comm_keyval);
}
inline int MPI::Comm::NULL_COPY_FN(const MPI::Comm &oldcomm, int comm_keyval, void *extra_state, void *attribute_val_in, void *attribute_val_out, bool &flag)
{
    flag = false;
    return 0;
}
inline int MPI::Comm::DUP_FN(const MPI::Comm &oldcomm, int comm_keyval, void *extra_state, void *attribute_val_in, void *attribute_val_out, bool &flag)
{
    if (sizeof(bool) != sizeof(int))
    {
        int f = (int) flag;
        int ret;
        ret = OMPI_C_MPI_DUP_FN(oldcomm, comm_keyval, extra_state, attribute_val_in, attribute_val_out, &f);
        flag = ((bool) (f));
        return ret;
    }
    else
    {
        return OMPI_C_MPI_DUP_FN(oldcomm, comm_keyval, extra_state, attribute_val_in, attribute_val_out, (int *) &flag);
    }
}
inline int MPI::Comm::NULL_DELETE_FN(MPI::Comm &comm, int comm_keyval, void *attribute_val, void *extra_state)
{
    return 0;
}
inline MPI::Intracomm::Intracomm(MPI_Comm data)
{
    int flag = 0;
    if (MPI::Is_initialized() && (data != ((MPI_Comm) ((void *) &(ompi_mpi_comm_null)))))
    {
        (void) MPI_Comm_test_inter(data, &flag);
        if (flag)
        {
            mpi_comm = ((MPI_Comm) ((void *) &(ompi_mpi_comm_null)));
        }
        else
        {
            mpi_comm = data;
        }
    }
    else
    {
        mpi_comm = data;
    }
}
inline void MPI::Intracomm::Scan(const void *sendbuf, void *recvbuf, int count, const MPI::Datatype &datatype, const MPI::Op &op) const
{
    (void) MPI_Scan(const_cast<void * >(sendbuf), recvbuf, count, datatype, op, mpi_comm);
}
inline void MPI::Intracomm::Exscan(const void *sendbuf, void *recvbuf, int count, const MPI::Datatype &datatype, const MPI::Op &op) const
{
    (void) MPI_Exscan(const_cast<void * >(sendbuf), recvbuf, count, datatype, op, mpi_comm);
}
inline MPI::Intracomm MPI::Intracomm::Dup() const
{
    MPI_Comm newcomm;
    (void) MPI_Comm_dup(mpi_comm, &newcomm);
    return newcomm;
}
inline MPI::Intracomm &MPI::Intracomm::Clone() const
{
    MPI_Comm newcomm;
    (void) MPI_Comm_dup(mpi_comm, &newcomm);
    MPI::Intracomm *dup = new MPI::Intracomm (newcomm);
    return *dup;
}
inline MPI::Intracomm MPI::Intracomm::Create(const MPI::Group &group) const
{
    MPI_Comm newcomm;
    (void) MPI_Comm_create(mpi_comm, group, &newcomm);
    return newcomm;
}
inline MPI::Intracomm MPI::Intracomm::Split(int color, int key) const
{
    MPI_Comm newcomm;
    (void) MPI_Comm_split(mpi_comm, color, key, &newcomm);
    return newcomm;
}
inline MPI::Intercomm MPI::Intracomm::Create_intercomm(int local_leader, const MPI::Comm &peer_comm, int remote_leader, int tag) const
{
    MPI_Comm newintercomm;
    (void) MPI_Intercomm_create(mpi_comm, local_leader, peer_comm, remote_leader, tag, &newintercomm);
    return newintercomm;
}
inline MPI::Cartcomm MPI::Intracomm::Create_cart(int ndims, const int dims[], const bool periods[], bool reorder) const
{
    int *int_periods = new int [ndims];
    for (int i = 0;
        i < ndims;
        i++)
        int_periods[i] = (int) periods[i];
    MPI_Comm newcomm;
    (void) MPI_Cart_create(mpi_comm, ndims, const_cast<int * >(dims), int_periods, (int) reorder, &newcomm);
    delete[] int_periods;
    return newcomm;
}
inline MPI::Graphcomm MPI::Intracomm::Create_graph(int nnodes, const int index[], const int edges[], bool reorder) const
{
    MPI_Comm newcomm;
    (void) MPI_Graph_create(mpi_comm, nnodes, const_cast<int * >(index), const_cast<int * >(edges), (int) reorder, &newcomm);
    return newcomm;
}
inline MPI::Intercomm MPI::Intracomm::Accept(const char *port_name, const MPI::Info &info, int root) const
{
    MPI_Comm newcomm;
    (void) MPI_Comm_accept(const_cast<char * >(port_name), info, root, mpi_comm, &newcomm);
    return newcomm;
}
inline MPI::Intercomm MPI::Intracomm::Connect(const char *port_name, const MPI::Info &info, int root) const
{
    MPI_Comm newcomm;
    (void) MPI_Comm_connect(const_cast<char * >(port_name), info, root, mpi_comm, &newcomm);
    return newcomm;
}
inline MPI::Intercomm MPI::Intracomm::Spawn(const char *command, const char *argv[], int maxprocs, const MPI::Info &info, int root) const
{
    MPI_Comm newcomm;
    (void) MPI_Comm_spawn(const_cast<char * >(command), const_cast<char ** >(argv), maxprocs, info, root, mpi_comm, &newcomm, (int *) ((int *) 0));
    return newcomm;
}
inline MPI::Intercomm MPI::Intracomm::Spawn(const char *command, const char *argv[], int maxprocs, const MPI::Info &info, int root, int array_of_errcodes[]) const
{
    MPI_Comm newcomm;
    (void) MPI_Comm_spawn(const_cast<char * >(command), const_cast<char ** >(argv), maxprocs, info, root, mpi_comm, &newcomm, array_of_errcodes);
    return newcomm;
}
inline MPI::Intercomm MPI::Intracomm::Spawn_multiple(int count, const char *array_of_commands[], const char **array_of_argv[], const int array_of_maxprocs[], const Info array_of_info[], int root)
{
    MPI_Comm newcomm;
    MPI_Info *const array_of_mpi_info = convert_info_to_mpi_info(count, array_of_info);
    MPI_Comm_spawn_multiple(count, const_cast<char ** >(array_of_commands), const_cast<char *** >(array_of_argv), const_cast<int * >(array_of_maxprocs), array_of_mpi_info, root, mpi_comm, &newcomm, (int *) ((int *) 0));
    delete[] array_of_mpi_info;
    return newcomm;
}
inline MPI_Info *MPI::Intracomm::convert_info_to_mpi_info(int p_nbr, const Info p_info_tbl[])
{
    MPI_Info *const mpi_info_tbl = new MPI_Info [p_nbr];
    for (int i_tbl = 0;
        i_tbl < p_nbr;
        i_tbl++)
    {
        mpi_info_tbl[i_tbl] = p_info_tbl[i_tbl];
    }
    return mpi_info_tbl;
}
inline MPI::Intercomm MPI::Intracomm::Spawn_multiple(int count, const char *array_of_commands[], const char **array_of_argv[], const int array_of_maxprocs[], const Info array_of_info[], int root, int array_of_errcodes[])
{
    MPI_Comm newcomm;
    MPI_Info *const array_of_mpi_info = convert_info_to_mpi_info(count, array_of_info);
    MPI_Comm_spawn_multiple(count, const_cast<char ** >(array_of_commands), const_cast<char *** >(array_of_argv), const_cast<int * >(array_of_maxprocs), array_of_mpi_info, root, mpi_comm, &newcomm, array_of_errcodes);
    delete[] array_of_mpi_info;
    return newcomm;
}
inline MPI::Cartcomm::Cartcomm(const MPI_Comm &data)
{
    int status = 0;
    if (MPI::Is_initialized() && (data != ((MPI_Comm) ((void *) &(ompi_mpi_comm_null)))))
    {
        (void) MPI_Topo_test(data, &status);
        if (status == 1)
            mpi_comm = data;
        else
            mpi_comm = ((MPI_Comm) ((void *) &(ompi_mpi_comm_null)));
    }
    else
    {
        mpi_comm = data;
    }
}
inline MPI::Cartcomm MPI::Cartcomm::Dup() const
{
    MPI_Comm newcomm;
    (void) MPI_Comm_dup(mpi_comm, &newcomm);
    return newcomm;
}
inline int MPI::Cartcomm::Get_dim() const
{
    int ndims;
    (void) MPI_Cartdim_get(mpi_comm, &ndims);
    return ndims;
}
inline void MPI::Cartcomm::Get_topo(int maxdims, int dims[], bool periods[], int coords[]) const
{
    int *int_periods = new int [maxdims];
    int i;
    for (i = 0;
        i < maxdims;
        i++)
    {
        int_periods[i] = (int) periods[i];
    }
    (void) MPI_Cart_get(mpi_comm, maxdims, dims, int_periods, coords);
    for (i = 0;
        i < maxdims;
        i++)
    {
        periods[i] = ((bool) (int_periods[i]));
    }
    delete[] int_periods;
}
inline int MPI::Cartcomm::Get_cart_rank(const int coords[]) const
{
    int rank;
    (void) MPI_Cart_rank(mpi_comm, const_cast<int * >(coords), &rank);
    return rank;
}
inline void MPI::Cartcomm::Get_coords(int rank, int maxdims, int coords[]) const
{
    (void) MPI_Cart_coords(mpi_comm, rank, maxdims, coords);
}
inline void MPI::Cartcomm::Shift(int direction, int disp, int &rank_source, int &rank_dest) const
{
    (void) MPI_Cart_shift(mpi_comm, direction, disp, &rank_source, &rank_dest);
}
inline MPI::Cartcomm MPI::Cartcomm::Sub(const bool remain_dims[])
{
    int ndims;
    MPI_Cartdim_get(mpi_comm, &ndims);
    int *int_remain_dims = new int [ndims];
    for (int i = 0;
        i < ndims;
        i++)
    {
        int_remain_dims[i] = (int) remain_dims[i];
    }
    MPI_Comm newcomm;
    (void) MPI_Cart_sub(mpi_comm, int_remain_dims, &newcomm);
    delete[] int_remain_dims;
    return newcomm;
}
inline int MPI::Cartcomm::Map(int ndims, const int dims[], const bool periods[]) const
{
    int *int_periods = new int [ndims];
    for (int i = 0;
        i < ndims;
        i++)
    {
        int_periods[i] = (int) periods[i];
    }
    int newrank;
    (void) MPI_Cart_map(mpi_comm, ndims, const_cast<int * >(dims), int_periods, &newrank);
    delete[] int_periods;
    return newrank;
}
inline MPI::Cartcomm &MPI::Cartcomm::Clone() const
{
    MPI_Comm newcomm;
    (void) MPI_Comm_dup(mpi_comm, &newcomm);
    MPI::Cartcomm *dup = new MPI::Cartcomm (newcomm);
    return *dup;
}
inline MPI::Graphcomm::Graphcomm(const MPI_Comm &data)
{
    int status = 0;
    if (MPI::Is_initialized() && (data != ((MPI_Comm) ((void *) &(ompi_mpi_comm_null)))))
    {
        (void) MPI_Topo_test(data, &status);
        if (status == 2)
            mpi_comm = data;
        else
            mpi_comm = ((MPI_Comm) ((void *) &(ompi_mpi_comm_null)));
    }
    else
    {
        mpi_comm = data;
    }
}
inline MPI::Graphcomm MPI::Graphcomm::Dup() const
{
    MPI_Comm newcomm;
    (void) MPI_Comm_dup(mpi_comm, &newcomm);
    return newcomm;
}
inline void MPI::Graphcomm::Get_dims(int nnodes[], int nedges[]) const
{
    (void) MPI_Graphdims_get(mpi_comm, nnodes, nedges);
}
inline void MPI::Graphcomm::Get_topo(int maxindex, int maxedges, int index[], int edges[]) const
{
    (void) MPI_Graph_get(mpi_comm, maxindex, maxedges, index, edges);
}
inline int MPI::Graphcomm::Get_neighbors_count(int rank) const
{
    int nneighbors;
    (void) MPI_Graph_neighbors_count(mpi_comm, rank, &nneighbors);
    return nneighbors;
}
inline void MPI::Graphcomm::Get_neighbors(int rank, int maxneighbors, int neighbors[]) const
{
    (void) MPI_Graph_neighbors(mpi_comm, rank, maxneighbors, neighbors);
}
inline int MPI::Graphcomm::Map(int nnodes, const int index[], const int edges[]) const
{
    int newrank;
    (void) MPI_Graph_map(mpi_comm, nnodes, const_cast<int * >(index), const_cast<int * >(edges), &newrank);
    return newrank;
}
inline MPI::Graphcomm &MPI::Graphcomm::Clone() const
{
    MPI_Comm newcomm;
    (void) MPI_Comm_dup(mpi_comm, &newcomm);
    MPI::Graphcomm *dup = new MPI::Graphcomm (newcomm);
    return *dup;
}
inline MPI::Intercomm MPI::Intercomm::Dup() const
{
    MPI_Comm newcomm;
    (void) MPI_Comm_dup(mpi_comm, &newcomm);
    return newcomm;
}
inline MPI::Intercomm &MPI::Intercomm::Clone() const
{
    MPI_Comm newcomm;
    (void) MPI_Comm_dup(mpi_comm, &newcomm);
    MPI::Intercomm *dup = new MPI::Intercomm (newcomm);
    return *dup;
}
inline int MPI::Intercomm::Get_remote_size() const
{
    int size;
    (void) MPI_Comm_remote_size(mpi_comm, &size);
    return size;
}
inline MPI::Group MPI::Intercomm::Get_remote_group() const
{
    MPI_Group group;
    (void) MPI_Comm_remote_group(mpi_comm, &group);
    return group;
}
inline MPI::Intracomm MPI::Intercomm::Merge(bool high)
{
    MPI_Comm newcomm;
    (void) MPI_Intercomm_merge(mpi_comm, (int) high, &newcomm);
    return newcomm;
}
inline MPI::Intercomm MPI::Intercomm::Create(const Group &group) const
{
    MPI_Comm newcomm;
    (void) MPI_Comm_create(mpi_comm, (MPI_Group) group, &newcomm);
    return newcomm;
}
inline MPI::Intercomm MPI::Intercomm::Split(int color, int key) const
{
    MPI_Comm newcomm;
    (void) MPI_Comm_split(mpi_comm, color, key, &newcomm);
    return newcomm;
}
inline int MPI::Group::Get_size() const
{
    int size;
    (void) MPI_Group_size(mpi_group, &size);
    return size;
}
inline int MPI::Group::Get_rank() const
{
    int rank;
    (void) MPI_Group_rank(mpi_group, &rank);
    return rank;
}
inline void MPI::Group::Translate_ranks(const MPI::Group &group1, int n, const int ranks1[], const MPI::Group &group2, int ranks2[])
{
    (void) MPI_Group_translate_ranks(group1, n, const_cast<int * >(ranks1), group2, const_cast<int * >(ranks2));
}
inline int MPI::Group::Compare(const MPI::Group &group1, const MPI::Group &group2)
{
    int result;
    (void) MPI_Group_compare(group1, group2, &result);
    return result;
}
inline MPI::Group MPI::Group::Union(const MPI::Group &group1, const MPI::Group &group2)
{
    MPI_Group newgroup;
    (void) MPI_Group_union(group1, group2, &newgroup);
    return newgroup;
}
inline MPI::Group MPI::Group::Intersect(const MPI::Group &group1, const MPI::Group &group2)
{
    MPI_Group newgroup;
    (void) MPI_Group_intersection(group1, group2, &newgroup);
    return newgroup;
}
inline MPI::Group MPI::Group::Difference(const MPI::Group &group1, const MPI::Group &group2)
{
    MPI_Group newgroup;
    (void) MPI_Group_difference(group1, group2, &newgroup);
    return newgroup;
}
inline MPI::Group MPI::Group::Incl(int n, const int ranks[]) const
{
    MPI_Group newgroup;
    (void) MPI_Group_incl(mpi_group, n, const_cast<int * >(ranks), &newgroup);
    return newgroup;
}
inline MPI::Group MPI::Group::Excl(int n, const int ranks[]) const
{
    MPI_Group newgroup;
    (void) MPI_Group_excl(mpi_group, n, const_cast<int * >(ranks), &newgroup);
    return newgroup;
}
inline MPI::Group MPI::Group::Range_incl(int n, const int ranges[][3]) const
{
    MPI_Group newgroup;
    (void) MPI_Group_range_incl(mpi_group, n, const_cast<int (*)[3] >(ranges), &newgroup);
    return newgroup;
}
inline MPI::Group MPI::Group::Range_excl(int n, const int ranges[][3]) const
{
    MPI_Group newgroup;
    (void) MPI_Group_range_excl(mpi_group, n, const_cast<int (*)[3] >(ranges), &newgroup);
    return newgroup;
}
inline void MPI::Group::Free()
{
    (void) MPI_Group_free(&mpi_group);
}
inline MPI::Op::Op()
    : mpi_op(((MPI_Op) ((void *) &(ompi_mpi_op_null)))) 
{
}
inline MPI::Op::Op(MPI_Op i)
    : mpi_op(i) 
{
}
inline MPI::Op::Op(const MPI::Op &op)
    : mpi_op(op.mpi_op) 
{
}
inline MPI::Op::~Op()
{
}
inline MPI::Op &MPI::Op::operator =(const MPI::Op &op)
{
    mpi_op = op.mpi_op;
    return *this;
}
inline bool MPI::Op::operator ==(const MPI::Op &a)
{
    return (bool) (mpi_op == a.mpi_op);
}
inline bool MPI::Op::operator !=(const MPI::Op &a)
{
    return (bool) !(*this == a);
}
inline MPI::Op &MPI::Op::operator =(const MPI_Op &i)
{
    mpi_op = i;
    return *this;
}
inline MPI::Op::operator MPI_Op() const
{
    return mpi_op;
}
extern "C"
void ompi_op_set_cxx_callback(MPI_Op op, MPI_User_function *);
inline void MPI::Op::Init(MPI::User_function *func, bool commute)
{
    (void) MPI_Op_create((MPI_User_function *) ompi_mpi_cxx_op_intercept, (int) commute, &mpi_op);
    ompi_op_set_cxx_callback(mpi_op, (MPI_User_function *) func);
}
inline void MPI::Op::Free()
{
    (void) MPI_Op_free(&mpi_op);
}
inline void MPI::Op::Reduce_local(const void *inbuf, void *inoutbuf, int count, const MPI::Datatype &datatype) const
{
    (void) MPI_Reduce_local(const_cast<void * >(inbuf), inoutbuf, count, datatype, mpi_op);
}
inline bool MPI::Op::Is_commutative(void) const
{
    int commute;
    (void) MPI_Op_commutative(mpi_op, &commute);
    return (bool) commute;
}
inline void MPI::Errhandler::Free()
{
    (void) MPI_Errhandler_free(&mpi_errhandler);
}
inline int MPI::Status::Get_count(const MPI::Datatype &datatype) const
{
    int count;
    (void) MPI_Get_count(const_cast<MPI_Status * >(&mpi_status), datatype, &count);
    return count;
}
inline bool MPI::Status::Is_cancelled() const
{
    int t;
    (void) MPI_Test_cancelled(const_cast<MPI_Status * >(&mpi_status), &t);
    return ((bool) (t));
}
inline int MPI::Status::Get_elements(const MPI::Datatype &datatype) const
{
    int count;
    (void) MPI_Get_elements(const_cast<MPI_Status * >(&mpi_status), datatype, &count);
    return count;
}
inline int MPI::Status::Get_source() const
{
    int source;
    source = mpi_status.MPI_SOURCE;
    return source;
}
inline void MPI::Status::Set_source(int source)
{
    mpi_status.MPI_SOURCE = source;
}
inline int MPI::Status::Get_tag() const
{
    int tag;
    tag = mpi_status.MPI_TAG;
    return tag;
}
inline void MPI::Status::Set_tag(int tag)
{
    mpi_status.MPI_TAG = tag;
}
inline int MPI::Status::Get_error() const
{
    int error;
    error = mpi_status.MPI_ERROR;
    return error;
}
inline void MPI::Status::Set_error(int error)
{
    mpi_status.MPI_ERROR = error;
}
inline void MPI::Status::Set_elements(const MPI::Datatype &datatype, int count)
{
    MPI_Status_set_elements(&mpi_status, datatype, count);
}
inline void MPI::Status::Set_cancelled(bool flag)
{
    MPI_Status_set_cancelled(&mpi_status, (int) flag);
}
inline MPI::Info MPI::Info::Create()
{
    MPI_Info newinfo;
    (void) MPI_Info_create(&newinfo);
    return newinfo;
}
inline void MPI::Info::Delete(const char *key)
{
    (void) MPI_Info_delete(mpi_info, const_cast<char * >(key));
}
inline MPI::Info MPI::Info::Dup() const
{
    MPI_Info newinfo;
    (void) MPI_Info_dup(mpi_info, &newinfo);
    return newinfo;
}
inline void MPI::Info::Free()
{
    (void) MPI_Info_free(&mpi_info);
}
inline bool MPI::Info::Get(const char *key, int valuelen, char *value) const
{
    int flag;
    (void) MPI_Info_get(mpi_info, const_cast<char * >(key), valuelen, value, &flag);
    return ((bool) (flag));
}
inline int MPI::Info::Get_nkeys() const
{
    int nkeys;
    MPI_Info_get_nkeys(mpi_info, &nkeys);
    return nkeys;
}
inline void MPI::Info::Get_nthkey(int n, char *key) const
{
    (void) MPI_Info_get_nthkey(mpi_info, n, key);
}
inline bool MPI::Info::Get_valuelen(const char *key, int &valuelen) const
{
    int flag;
    (void) MPI_Info_get_valuelen(mpi_info, const_cast<char * >(key), &valuelen, &flag);
    return ((bool) (flag));
}
inline void MPI::Info::Set(const char *key, const char *value)
{
    (void) MPI_Info_set(mpi_info, const_cast<char * >(key), const_cast<char * >(value));
}
inline MPI::Errhandler MPI::Win::Get_errhandler() const
{
    MPI_Errhandler errhandler;
    MPI_Win_get_errhandler(mpi_win, &errhandler);
    return errhandler;
}
inline void MPI::Win::Set_errhandler(const MPI::Errhandler &errhandler) const
{
    (void) MPI_Win_set_errhandler(mpi_win, errhandler);
}
inline void MPI::Win::Accumulate(const void *origin_addr, int origin_count, const MPI::Datatype &origin_datatype, int target_rank, MPI::Aint target_disp, int target_count, const MPI::Datatype &target_datatype, const MPI::Op &op) const
{
    (void) MPI_Accumulate(const_cast<void * >(origin_addr), origin_count, origin_datatype, target_rank, target_disp, target_count, target_datatype, op, mpi_win);
}
inline void MPI::Win::Complete() const
{
    (void) MPI_Win_complete(mpi_win);
}
inline MPI::Win MPI::Win::Create(const void *base, MPI::Aint size, int disp_unit, const MPI::Info &info, const MPI::Intracomm &comm)
{
    MPI_Win newwin;
    (void) MPI_Win_create(const_cast<void * >(base), size, disp_unit, info, comm, &newwin);
    return newwin;
}
inline void MPI::Win::Fence(int assert) const
{
    (void) MPI_Win_fence(assert, mpi_win);
}
inline void MPI::Win::Get(const void *origin_addr, int origin_count, const MPI::Datatype &origin_datatype, int target_rank, MPI::Aint target_disp, int target_count, const MPI::Datatype &target_datatype) const
{
    (void) MPI_Get(const_cast<void * >(origin_addr), origin_count, origin_datatype, target_rank, target_disp, target_count, target_datatype, mpi_win);
}
inline MPI::Group MPI::Win::Get_group() const
{
    MPI_Group mpi_group;
    (void) MPI_Win_get_group(mpi_win, &mpi_group);
    return mpi_group;
}
inline void MPI::Win::Lock(int lock_type, int rank, int assert) const
{
    (void) MPI_Win_lock(lock_type, rank, assert, mpi_win);
}
inline void MPI::Win::Post(const MPI::Group &group, int assert) const
{
    (void) MPI_Win_post(group, assert, mpi_win);
}
inline void MPI::Win::Put(const void *origin_addr, int origin_count, const MPI::Datatype &origin_datatype, int target_rank, MPI::Aint target_disp, int target_count, const MPI::Datatype &target_datatype) const
{
    (void) MPI_Put(const_cast<void * >(origin_addr), origin_count, origin_datatype, target_rank, target_disp, target_count, target_datatype, mpi_win);
}
inline void MPI::Win::Start(const MPI::Group &group, int assert) const
{
    (void) MPI_Win_start(group, assert, mpi_win);
}
inline bool MPI::Win::Test() const
{
    int flag;
    MPI_Win_test(mpi_win, &flag);
    return ((bool) (flag));
}
inline void MPI::Win::Unlock(int rank) const
{
    (void) MPI_Win_unlock(rank, mpi_win);
}
inline void MPI::Win::Wait() const
{
    (void) MPI_Win_wait(mpi_win);
}
inline void MPI::Win::Call_errhandler(int errorcode) const
{
    (void) MPI_Win_call_errhandler(mpi_win, errorcode);
}
inline int MPI::Win::Create_keyval(MPI::Win::Copy_attr_function *win_copy_attr_fn, MPI::Win::Delete_attr_function *win_delete_attr_fn, void *extra_state)
{
    int ret, keyval;
    ret = do_create_keyval(__null, __null, win_copy_attr_fn, win_delete_attr_fn, extra_state, keyval);
    return (0 == ret) ? keyval : ret;
}
inline int MPI::Win::Create_keyval(MPI_Win_copy_attr_function *win_copy_attr_fn, MPI_Win_delete_attr_function *win_delete_attr_fn, void *extra_state)
{
    int ret, keyval;
    ret = do_create_keyval(win_copy_attr_fn, win_delete_attr_fn, __null, __null, extra_state, keyval);
    return (0 == ret) ? keyval : ret;
}
inline int MPI::Win::Create_keyval(MPI::Win::Copy_attr_function *win_copy_attr_fn, MPI_Win_delete_attr_function *win_delete_attr_fn, void *extra_state)
{
    int ret, keyval;
    ret = do_create_keyval(__null, win_delete_attr_fn, win_copy_attr_fn, __null, extra_state, keyval);
    return (0 == ret) ? keyval : ret;
}
inline int MPI::Win::Create_keyval(MPI_Win_copy_attr_function *win_copy_attr_fn, MPI::Win::Delete_attr_function *win_delete_attr_fn, void *extra_state)
{
    int ret, keyval;
    ret = do_create_keyval(win_copy_attr_fn, __null, __null, win_delete_attr_fn, extra_state, keyval);
    return (0 == ret) ? keyval : ret;
}
inline void MPI::Win::Delete_attr(int win_keyval)
{
    (void) MPI_Win_delete_attr(mpi_win, win_keyval);
}
inline void MPI::Win::Free_keyval(int &win_keyval)
{
    (void) MPI_Win_free_keyval(&win_keyval);
}
inline bool MPI::Win::Get_attr(const Win &win, int win_keyval, void *attribute_val) const
{
    int ret;
    (void) MPI_Win_get_attr(win, win_keyval, attribute_val, &ret);
    return ((bool) (ret));
}
inline bool MPI::Win::Get_attr(int win_keyval, void *attribute_val) const
{
    int ret;
    (void) MPI_Win_get_attr(mpi_win, win_keyval, attribute_val, &ret);
    return ((bool) (ret));
}
inline void MPI::Win::Get_name(char *win_name, int &resultlen) const
{
    (void) MPI_Win_get_name(mpi_win, win_name, &resultlen);
}
inline void MPI::Win::Set_attr(int win_keyval, const void *attribute_val)
{
    (void) MPI_Win_set_attr(mpi_win, win_keyval, const_cast<void * >(attribute_val));
}
inline void MPI::Win::Set_name(const char *win_name)
{
    (void) MPI_Win_set_name(mpi_win, const_cast<char * >(win_name));
}
inline void MPI::File::Delete(const char *filename, const MPI::Info &info)
{
    (void) MPI_File_delete(const_cast<char * >(filename), info);
}
inline int MPI::File::Get_amode() const
{
    int amode;
    (void) MPI_File_get_amode(mpi_file, &amode);
    return amode;
}
inline bool MPI::File::Get_atomicity() const
{
    int flag;
    (void) MPI_File_get_atomicity(mpi_file, &flag);
    return ((bool) (flag));
}
inline MPI::Offset MPI::File::Get_byte_offset(const MPI::Offset disp) const
{
    MPI_Offset offset, ldisp;
    ldisp = disp;
    (void) MPI_File_get_byte_offset(mpi_file, ldisp, &offset);
    return offset;
}
inline MPI::Group MPI::File::Get_group() const
{
    MPI_Group group;
    (void) MPI_File_get_group(mpi_file, &group);
    return group;
}
inline MPI::Info MPI::File::Get_info() const
{
    MPI_Info info_used;
    (void) MPI_File_get_info(mpi_file, &info_used);
    return info_used;
}
inline MPI::Offset MPI::File::Get_position() const
{
    MPI_Offset offset;
    (void) MPI_File_get_position(mpi_file, &offset);
    return offset;
}
inline MPI::Offset MPI::File::Get_position_shared() const
{
    MPI_Offset offset;
    (void) MPI_File_get_position_shared(mpi_file, &offset);
    return offset;
}
inline MPI::Offset MPI::File::Get_size() const
{
    MPI_Offset offset;
    (void) MPI_File_get_size(mpi_file, &offset);
    return offset;
}
inline MPI::Aint MPI::File::Get_type_extent(const MPI::Datatype &datatype) const
{
    MPI_Aint extent;
    (void) MPI_File_get_type_extent(mpi_file, datatype, &extent);
    return extent;
}
inline void MPI::File::Get_view(MPI::Offset &disp, MPI::Datatype &etype, MPI::Datatype &filetype, char *datarep) const
{
    MPI_Datatype type, ftype;
    type = etype;
    ftype = filetype;
    MPI::Offset odisp = disp;
    (void) MPI_File_get_view(mpi_file, &odisp, &type, &ftype, datarep);
}
inline MPI::Request MPI::File::Iread(void *buf, int count, const MPI::Datatype &datatype)
{
    MPI_Request req;
    (void) MPI_File_iread(mpi_file, buf, count, datatype, &req);
    return req;
}
inline MPI::Request MPI::File::Iread_at(MPI::Offset offset, void *buf, int count, const MPI::Datatype &datatype)
{
    MPI_Request req;
    (void) MPI_File_iread_at(mpi_file, offset, buf, count, datatype, &req);
    return req;
}
inline MPI::Request MPI::File::Iread_shared(void *buf, int count, const MPI::Datatype &datatype)
{
    MPI_Request req;
    (void) MPI_File_iread_shared(mpi_file, buf, count, datatype, &req);
    return req;
}
inline MPI::Request MPI::File::Iwrite(const void *buf, int count, const MPI::Datatype &datatype)
{
    MPI_Request req;
    (void) MPI_File_iwrite(mpi_file, const_cast<void * >(buf), count, datatype, &req);
    return req;
}
inline MPI::Request MPI::File::Iwrite_at(MPI::Offset offset, const void *buf, int count, const MPI::Datatype &datatype)
{
    MPI_Request req;
    (void) MPI_File_iwrite_at(mpi_file, offset, const_cast<void * >(buf), count, datatype, &req);
    return req;
}
inline MPI::Request MPI::File::Iwrite_shared(const void *buf, int count, const MPI::Datatype &datatype)
{
    MPI_Request req;
    (void) MPI_File_iwrite_shared(mpi_file, const_cast<void * >(buf), count, datatype, &req);
    return req;
}
inline MPI::File MPI::File::Open(const MPI::Intracomm &comm, const char *filename, int amode, const MPI::Info &info)
{
    MPI_File fh;
    (void) MPI_File_open(comm, const_cast<char * >(filename), amode, info, &fh);
    return fh;
}
inline void MPI::File::Preallocate(MPI::Offset size)
{
    (void) MPI_File_preallocate(mpi_file, size);
}
inline void MPI::File::Read(void *buf, int count, const MPI::Datatype &datatype)
{
    MPI_Status status;
    (void) MPI_File_read(mpi_file, buf, count, datatype, &status);
}
inline void MPI::File::Read(void *buf, int count, const MPI::Datatype &datatype, MPI::Status &status)
{
    (void) MPI_File_read(mpi_file, buf, count, datatype, &status.mpi_status);
}
inline void MPI::File::Read_all(void *buf, int count, const MPI::Datatype &datatype)
{
    MPI_Status status;
    (void) MPI_File_read_all(mpi_file, buf, count, datatype, &status);
}
inline void MPI::File::Read_all(void *buf, int count, const MPI::Datatype &datatype, MPI::Status &status)
{
    (void) MPI_File_read_all(mpi_file, buf, count, datatype, &status.mpi_status);
}
inline void MPI::File::Read_all_begin(void *buf, int count, const MPI::Datatype &datatype)
{
    (void) MPI_File_read_all_begin(mpi_file, buf, count, datatype);
}
inline void MPI::File::Read_all_end(void *buf)
{
    MPI_Status status;
    (void) MPI_File_read_all_end(mpi_file, buf, &status);
}
inline void MPI::File::Read_all_end(void *buf, MPI::Status &status)
{
    (void) MPI_File_read_all_end(mpi_file, buf, &status.mpi_status);
}
inline void MPI::File::Read_at(MPI::Offset offset, void *buf, int count, const MPI::Datatype &datatype)
{
    MPI_Status status;
    (void) MPI_File_read_at(mpi_file, offset, buf, count, datatype, &status);
}
inline void MPI::File::Read_at(MPI::Offset offset, void *buf, int count, const MPI::Datatype &datatype, MPI::Status &status)
{
    (void) MPI_File_read_at(mpi_file, offset, buf, count, datatype, &status.mpi_status);
}
inline void MPI::File::Read_at_all(MPI::Offset offset, void *buf, int count, const MPI::Datatype &datatype)
{
    MPI_Status status;
    (void) MPI_File_read_at_all(mpi_file, offset, buf, count, datatype, &status);
}
inline void MPI::File::Read_at_all(MPI::Offset offset, void *buf, int count, const MPI::Datatype &datatype, MPI::Status &status)
{
    (void) MPI_File_read_at_all(mpi_file, offset, buf, count, datatype, &status.mpi_status);
}
inline void MPI::File::Read_at_all_begin(MPI::Offset offset, void *buf, int count, const MPI::Datatype &datatype)
{
    (void) MPI_File_read_at_all_begin(mpi_file, offset, buf, count, datatype);
}
inline void MPI::File::Read_at_all_end(void *buf)
{
    MPI_Status status;
    (void) MPI_File_read_at_all_end(mpi_file, buf, &status);
}
inline void MPI::File::Read_at_all_end(void *buf, MPI::Status &status)
{
    (void) MPI_File_read_at_all_end(mpi_file, buf, &status.mpi_status);
}
inline void MPI::File::Read_ordered(void *buf, int count, const MPI::Datatype &datatype)
{
    MPI_Status status;
    (void) MPI_File_read_ordered(mpi_file, buf, count, datatype, &status);
}
inline void MPI::File::Read_ordered(void *buf, int count, const MPI::Datatype &datatype, MPI::Status &status)
{
    (void) MPI_File_read_ordered(mpi_file, buf, count, datatype, &status.mpi_status);
}
inline void MPI::File::Read_ordered_begin(void *buf, int count, const MPI::Datatype &datatype)
{
    (void) MPI_File_read_ordered_begin(mpi_file, buf, count, datatype);
}
inline void MPI::File::Read_ordered_end(void *buf)
{
    MPI_Status status;
    (void) MPI_File_read_ordered_end(mpi_file, buf, &status);
}
inline void MPI::File::Read_ordered_end(void *buf, MPI::Status &status)
{
    (void) MPI_File_read_ordered_end(mpi_file, buf, &status.mpi_status);
}
inline void MPI::File::Read_shared(void *buf, int count, const MPI::Datatype &datatype)
{
    MPI_Status status;
    (void) MPI_File_read_shared(mpi_file, buf, count, datatype, &status);
}
inline void MPI::File::Read_shared(void *buf, int count, const MPI::Datatype &datatype, MPI::Status &status)
{
    (void) MPI_File_read_shared(mpi_file, buf, count, datatype, &status.mpi_status);
}
inline void MPI::File::Seek(MPI::Offset offset, int whence)
{
    (void) MPI_File_seek(mpi_file, offset, whence);
}
inline void MPI::File::Seek_shared(MPI::Offset offset, int whence)
{
    (void) MPI_File_seek_shared(mpi_file, offset, whence);
}
inline void MPI::File::Set_atomicity(bool flag)
{
    (void) MPI_File_set_atomicity(mpi_file, flag);
}
inline void MPI::File::Set_info(const MPI::Info &info)
{
    (void) MPI_File_set_info(mpi_file, info);
}
inline void MPI::File::Set_size(MPI::Offset size)
{
    (void) MPI_File_set_size(mpi_file, size);
}
inline void MPI::File::Set_view(MPI::Offset disp, const MPI::Datatype &etype, const MPI::Datatype &filetype, const char *datarep, const MPI::Info &info)
{
    (void) MPI_File_set_view(mpi_file, disp, etype, filetype, const_cast<char * >(datarep), info);
}
inline void MPI::File::Sync()
{
    (void) MPI_File_sync(mpi_file);
}
inline void MPI::File::Write(const void *buf, int count, const MPI::Datatype &datatype)
{
    MPI_Status status;
    (void) MPI_File_write(mpi_file, const_cast<void * >(buf), count, datatype, &status);
}
inline void MPI::File::Write(const void *buf, int count, const MPI::Datatype &datatype, MPI::Status &status)
{
    (void) MPI_File_write(mpi_file, const_cast<void * >(buf), count, datatype, &status.mpi_status);
}
inline void MPI::File::Write_all(const void *buf, int count, const MPI::Datatype &datatype)
{
    MPI_Status status;
    (void) MPI_File_write_all(mpi_file, const_cast<void * >(buf), count, datatype, &status);
}
inline void MPI::File::Write_all(const void *buf, int count, const MPI::Datatype &datatype, MPI::Status &status)
{
    (void) MPI_File_write_all(mpi_file, const_cast<void * >(buf), count, datatype, &status.mpi_status);
}
inline void MPI::File::Write_all_begin(const void *buf, int count, const MPI::Datatype &datatype)
{
    (void) MPI_File_write_all_begin(mpi_file, const_cast<void * >(buf), count, datatype);
}
inline void MPI::File::Write_all_end(const void *buf)
{
    MPI_Status status;
    (void) MPI_File_write_all_end(mpi_file, const_cast<void * >(buf), &status);
}
inline void MPI::File::Write_all_end(const void *buf, MPI::Status &status)
{
    (void) MPI_File_write_all_end(mpi_file, const_cast<void * >(buf), &status.mpi_status);
}
inline void MPI::File::Write_at(MPI::Offset offset, const void *buf, int count, const MPI::Datatype &datatype)
{
    MPI_Status status;
    (void) MPI_File_write_at(mpi_file, offset, const_cast<void * >(buf), count, datatype, &status);
}
inline void MPI::File::Write_at(MPI::Offset offset, const void *buf, int count, const MPI::Datatype &datatype, MPI::Status &status)
{
    (void) MPI_File_write_at(mpi_file, offset, const_cast<void * >(buf), count, datatype, &status.mpi_status);
}
inline void MPI::File::Write_at_all(MPI::Offset offset, const void *buf, int count, const MPI::Datatype &datatype)
{
    MPI_Status status;
    (void) MPI_File_write_at_all(mpi_file, offset, const_cast<void * >(buf), count, datatype, &status);
}
inline void MPI::File::Write_at_all(MPI::Offset offset, const void *buf, int count, const MPI::Datatype &datatype, MPI::Status &status)
{
    (void) MPI_File_write_at_all(mpi_file, offset, const_cast<void * >(buf), count, datatype, &status.mpi_status);
}
inline void MPI::File::Write_at_all_begin(MPI::Offset offset, const void *buf, int count, const MPI::Datatype &datatype)
{
    (void) MPI_File_write_at_all_begin(mpi_file, offset, const_cast<void * >(buf), count, datatype);
}
inline void MPI::File::Write_at_all_end(const void *buf)
{
    MPI_Status status;
    (void) MPI_File_write_at_all_end(mpi_file, const_cast<void * >(buf), &status);
}
inline void MPI::File::Write_at_all_end(const void *buf, MPI::Status &status)
{
    (void) MPI_File_write_at_all_end(mpi_file, const_cast<void * >(buf), &status.mpi_status);
}
inline void MPI::File::Write_ordered(const void *buf, int count, const MPI::Datatype &datatype)
{
    MPI_Status status;
    (void) MPI_File_write_ordered(mpi_file, const_cast<void * >(buf), count, datatype, &status);
}
inline void MPI::File::Write_ordered(const void *buf, int count, const MPI::Datatype &datatype, MPI::Status &status)
{
    (void) MPI_File_write_ordered(mpi_file, const_cast<void * >(buf), count, datatype, &status.mpi_status);
}
inline void MPI::File::Write_ordered_begin(const void *buf, int count, const MPI::Datatype &datatype)
{
    (void) MPI_File_write_ordered_begin(mpi_file, const_cast<void * >(buf), count, datatype);
}
inline void MPI::File::Write_ordered_end(const void *buf)
{
    MPI_Status status;
    (void) MPI_File_write_ordered_end(mpi_file, const_cast<void * >(buf), &status);
}
inline void MPI::File::Write_ordered_end(const void *buf, MPI::Status &status)
{
    (void) MPI_File_write_ordered_end(mpi_file, const_cast<void * >(buf), &status.mpi_status);
}
inline void MPI::File::Write_shared(const void *buf, int count, const MPI::Datatype &datatype)
{
    MPI_Status status;
    (void) MPI_File_write_shared(mpi_file, const_cast<void * >(buf), count, datatype, &status);
}
inline void MPI::File::Write_shared(const void *buf, int count, const MPI::Datatype &datatype, MPI::Status &status)
{
    (void) MPI_File_write_shared(mpi_file, const_cast<void * >(buf), count, datatype, &status.mpi_status);
}
inline void MPI::File::Set_errhandler(const MPI::Errhandler &errhandler) const
{
    (void) MPI_File_set_errhandler(mpi_file, errhandler);
}
inline MPI::Errhandler MPI::File::Get_errhandler() const
{
    MPI_Errhandler errhandler;
    MPI_File_get_errhandler(mpi_file, &errhandler);
    return errhandler;
}
inline void MPI::File::Call_errhandler(int errorcode) const
{
    (void) MPI_File_call_errhandler(mpi_file, errorcode);
}
extern "C"
{
    typedef float float_t;
    typedef double double_t;
    extern double acos(double __x) throw ();
    extern double __acos(double __x) throw ();
    extern double asin(double __x) throw ();
    extern double __asin(double __x) throw ();
    extern double atan(double __x) throw ();
    extern double __atan(double __x) throw ();
    extern double atan2(double __y, double __x) throw ();
    extern double __atan2(double __y, double __x) throw ();
    extern double cos(double __x) throw ();
    extern double __cos(double __x) throw ();
    extern double sin(double __x) throw ();
    extern double __sin(double __x) throw ();
    extern double tan(double __x) throw ();
    extern double __tan(double __x) throw ();
    extern double cosh(double __x) throw ();
    extern double __cosh(double __x) throw ();
    extern double sinh(double __x) throw ();
    extern double __sinh(double __x) throw ();
    extern double tanh(double __x) throw ();
    extern double __tanh(double __x) throw ();
    extern void sincos(double __x, double *__sinx, double *__cosx) throw ();
    extern void __sincos(double __x, double *__sinx, double *__cosx) throw ();
    extern double acosh(double __x) throw ();
    extern double __acosh(double __x) throw ();
    extern double asinh(double __x) throw ();
    extern double __asinh(double __x) throw ();
    extern double atanh(double __x) throw ();
    extern double __atanh(double __x) throw ();
    extern double exp(double __x) throw ();
    extern double __exp(double __x) throw ();
    extern double frexp(double __x, int *__exponent) throw ();
    extern double __frexp(double __x, int *__exponent) throw ();
    extern double ldexp(double __x, int __exponent) throw ();
    extern double __ldexp(double __x, int __exponent) throw ();
    extern double log(double __x) throw ();
    extern double __log(double __x) throw ();
    extern double log10(double __x) throw ();
    extern double __log10(double __x) throw ();
    extern double modf(double __x, double *__iptr) throw ();
    extern double __modf(double __x, double *__iptr) throw ();
    extern double exp10(double __x) throw ();
    extern double __exp10(double __x) throw ();
    extern double pow10(double __x) throw ();
    extern double __pow10(double __x) throw ();
    extern double expm1(double __x) throw ();
    extern double __expm1(double __x) throw ();
    extern double log1p(double __x) throw ();
    extern double __log1p(double __x) throw ();
    extern double logb(double __x) throw ();
    extern double __logb(double __x) throw ();
    extern double exp2(double __x) throw ();
    extern double __exp2(double __x) throw ();
    extern double log2(double __x) throw ();
    extern double __log2(double __x) throw ();
    extern double pow(double __x, double __y) throw ();
    extern double __pow(double __x, double __y) throw ();
    extern double sqrt(double __x) throw ();
    extern double __sqrt(double __x) throw ();
    extern double hypot(double __x, double __y) throw ();
    extern double __hypot(double __x, double __y) throw ();
    extern double cbrt(double __x) throw ();
    extern double __cbrt(double __x) throw ();
    extern double ceil(double __x) throw () __attribute__((__const__));
    extern double __ceil(double __x) throw () __attribute__((__const__));
    extern double fabs(double __x) throw () __attribute__((__const__));
    extern double __fabs(double __x) throw () __attribute__((__const__));
    extern double floor(double __x) throw () __attribute__((__const__));
    extern double __floor(double __x) throw () __attribute__((__const__));
    extern double fmod(double __x, double __y) throw ();
    extern double __fmod(double __x, double __y) throw ();
    extern int __isinf(double __value) throw () __attribute__((__const__));
    extern int __finite(double __value) throw () __attribute__((__const__));
    extern int isinf(double __value) throw () __attribute__((__const__));
    extern int finite(double __value) throw () __attribute__((__const__));
    extern double drem(double __x, double __y) throw ();
    extern double __drem(double __x, double __y) throw ();
    extern double significand(double __x) throw ();
    extern double __significand(double __x) throw ();
    extern double copysign(double __x, double __y) throw () __attribute__((__const__));
    extern double __copysign(double __x, double __y) throw () __attribute__((__const__));
    extern double nan(__const char *__tagb) throw () __attribute__((__const__));
    extern double __nan(__const char *__tagb) throw () __attribute__((__const__));
    extern int __isnan(double __value) throw () __attribute__((__const__));
    extern int isnan(double __value) throw () __attribute__((__const__));
    extern double j0(double) throw ();
    extern double __j0(double) throw ();
    extern double j1(double) throw ();
    extern double __j1(double) throw ();
    extern double jn(int, double) throw ();
    extern double __jn(int, double) throw ();
    extern double y0(double) throw ();
    extern double __y0(double) throw ();
    extern double y1(double) throw ();
    extern double __y1(double) throw ();
    extern double yn(int, double) throw ();
    extern double __yn(int, double) throw ();
    extern double erf(double) throw ();
    extern double __erf(double) throw ();
    extern double erfc(double) throw ();
    extern double __erfc(double) throw ();
    extern double lgamma(double) throw ();
    extern double __lgamma(double) throw ();
    extern double tgamma(double) throw ();
    extern double __tgamma(double) throw ();
    extern double gamma(double) throw ();
    extern double __gamma(double) throw ();
    extern double lgamma_r(double, int *__signgamp) throw ();
    extern double __lgamma_r(double, int *__signgamp) throw ();
    extern double rint(double __x) throw ();
    extern double __rint(double __x) throw ();
    extern double nextafter(double __x, double __y) throw () __attribute__((__const__));
    extern double __nextafter(double __x, double __y) throw () __attribute__((__const__));
    extern double nexttoward(double __x, long double __y) throw () __attribute__((__const__));
    extern double __nexttoward(double __x, long double __y) throw () __attribute__((__const__));
    extern double remainder(double __x, double __y) throw ();
    extern double __remainder(double __x, double __y) throw ();
    extern double scalbn(double __x, int __n) throw ();
    extern double __scalbn(double __x, int __n) throw ();
    extern int ilogb(double __x) throw ();
    extern int __ilogb(double __x) throw ();
    extern double scalbln(double __x, long int __n) throw ();
    extern double __scalbln(double __x, long int __n) throw ();
    extern double nearbyint(double __x) throw ();
    extern double __nearbyint(double __x) throw ();
    extern double round(double __x) throw () __attribute__((__const__));
    extern double __round(double __x) throw () __attribute__((__const__));
    extern double trunc(double __x) throw () __attribute__((__const__));
    extern double __trunc(double __x) throw () __attribute__((__const__));
    extern double remquo(double __x, double __y, int *__quo) throw ();
    extern double __remquo(double __x, double __y, int *__quo) throw ();
    extern long int lrint(double __x) throw ();
    extern long int __lrint(double __x) throw ();
    extern long long int llrint(double __x) throw ();
    extern long long int __llrint(double __x) throw ();
    extern long int lround(double __x) throw ();
    extern long int __lround(double __x) throw ();
    extern long long int llround(double __x) throw ();
    extern long long int __llround(double __x) throw ();
    extern double fdim(double __x, double __y) throw ();
    extern double __fdim(double __x, double __y) throw ();
    extern double fmax(double __x, double __y) throw ();
    extern double __fmax(double __x, double __y) throw ();
    extern double fmin(double __x, double __y) throw ();
    extern double __fmin(double __x, double __y) throw ();
    extern int __fpclassify(double __value) throw () __attribute__((__const__));
    extern int __signbit(double __value) throw () __attribute__((__const__));
    extern double fma(double __x, double __y, double __z) throw ();
    extern double __fma(double __x, double __y, double __z) throw ();
    extern double scalb(double __x, double __n) throw ();
    extern double __scalb(double __x, double __n) throw ();
    extern float acosf(float __x) throw ();
    extern float __acosf(float __x) throw ();
    extern float asinf(float __x) throw ();
    extern float __asinf(float __x) throw ();
    extern float atanf(float __x) throw ();
    extern float __atanf(float __x) throw ();
    extern float atan2f(float __y, float __x) throw ();
    extern float __atan2f(float __y, float __x) throw ();
    extern float cosf(float __x) throw ();
    extern float __cosf(float __x) throw ();
    extern float sinf(float __x) throw ();
    extern float __sinf(float __x) throw ();
    extern float tanf(float __x) throw ();
    extern float __tanf(float __x) throw ();
    extern float coshf(float __x) throw ();
    extern float __coshf(float __x) throw ();
    extern float sinhf(float __x) throw ();
    extern float __sinhf(float __x) throw ();
    extern float tanhf(float __x) throw ();
    extern float __tanhf(float __x) throw ();
    extern void sincosf(float __x, float *__sinx, float *__cosx) throw ();
    extern void __sincosf(float __x, float *__sinx, float *__cosx) throw ();
    extern float acoshf(float __x) throw ();
    extern float __acoshf(float __x) throw ();
    extern float asinhf(float __x) throw ();
    extern float __asinhf(float __x) throw ();
    extern float atanhf(float __x) throw ();
    extern float __atanhf(float __x) throw ();
    extern float expf(float __x) throw ();
    extern float __expf(float __x) throw ();
    extern float frexpf(float __x, int *__exponent) throw ();
    extern float __frexpf(float __x, int *__exponent) throw ();
    extern float ldexpf(float __x, int __exponent) throw ();
    extern float __ldexpf(float __x, int __exponent) throw ();
    extern float logf(float __x) throw ();
    extern float __logf(float __x) throw ();
    extern float log10f(float __x) throw ();
    extern float __log10f(float __x) throw ();
    extern float modff(float __x, float *__iptr) throw ();
    extern float __modff(float __x, float *__iptr) throw ();
    extern float exp10f(float __x) throw ();
    extern float __exp10f(float __x) throw ();
    extern float pow10f(float __x) throw ();
    extern float __pow10f(float __x) throw ();
    extern float expm1f(float __x) throw ();
    extern float __expm1f(float __x) throw ();
    extern float log1pf(float __x) throw ();
    extern float __log1pf(float __x) throw ();
    extern float logbf(float __x) throw ();
    extern float __logbf(float __x) throw ();
    extern float exp2f(float __x) throw ();
    extern float __exp2f(float __x) throw ();
    extern float log2f(float __x) throw ();
    extern float __log2f(float __x) throw ();
    extern float powf(float __x, float __y) throw ();
    extern float __powf(float __x, float __y) throw ();
    extern float sqrtf(float __x) throw ();
    extern float __sqrtf(float __x) throw ();
    extern float hypotf(float __x, float __y) throw ();
    extern float __hypotf(float __x, float __y) throw ();
    extern float cbrtf(float __x) throw ();
    extern float __cbrtf(float __x) throw ();
    extern float ceilf(float __x) throw () __attribute__((__const__));
    extern float __ceilf(float __x) throw () __attribute__((__const__));
    extern float fabsf(float __x) throw () __attribute__((__const__));
    extern float __fabsf(float __x) throw () __attribute__((__const__));
    extern float floorf(float __x) throw () __attribute__((__const__));
    extern float __floorf(float __x) throw () __attribute__((__const__));
    extern float fmodf(float __x, float __y) throw ();
    extern float __fmodf(float __x, float __y) throw ();
    extern int __isinff(float __value) throw () __attribute__((__const__));
    extern int __finitef(float __value) throw () __attribute__((__const__));
    extern int isinff(float __value) throw () __attribute__((__const__));
    extern int finitef(float __value) throw () __attribute__((__const__));
    extern float dremf(float __x, float __y) throw ();
    extern float __dremf(float __x, float __y) throw ();
    extern float significandf(float __x) throw ();
    extern float __significandf(float __x) throw ();
    extern float copysignf(float __x, float __y) throw () __attribute__((__const__));
    extern float __copysignf(float __x, float __y) throw () __attribute__((__const__));
    extern float nanf(__const char *__tagb) throw () __attribute__((__const__));
    extern float __nanf(__const char *__tagb) throw () __attribute__((__const__));
    extern int __isnanf(float __value) throw () __attribute__((__const__));
    extern int isnanf(float __value) throw () __attribute__((__const__));
    extern float j0f(float) throw ();
    extern float __j0f(float) throw ();
    extern float j1f(float) throw ();
    extern float __j1f(float) throw ();
    extern float jnf(int, float) throw ();
    extern float __jnf(int, float) throw ();
    extern float y0f(float) throw ();
    extern float __y0f(float) throw ();
    extern float y1f(float) throw ();
    extern float __y1f(float) throw ();
    extern float ynf(int, float) throw ();
    extern float __ynf(int, float) throw ();
    extern float erff(float) throw ();
    extern float __erff(float) throw ();
    extern float erfcf(float) throw ();
    extern float __erfcf(float) throw ();
    extern float lgammaf(float) throw ();
    extern float __lgammaf(float) throw ();
    extern float tgammaf(float) throw ();
    extern float __tgammaf(float) throw ();
    extern float gammaf(float) throw ();
    extern float __gammaf(float) throw ();
    extern float lgammaf_r(float, int *__signgamp) throw ();
    extern float __lgammaf_r(float, int *__signgamp) throw ();
    extern float rintf(float __x) throw ();
    extern float __rintf(float __x) throw ();
    extern float nextafterf(float __x, float __y) throw () __attribute__((__const__));
    extern float __nextafterf(float __x, float __y) throw () __attribute__((__const__));
    extern float nexttowardf(float __x, long double __y) throw () __attribute__((__const__));
    extern float __nexttowardf(float __x, long double __y) throw () __attribute__((__const__));
    extern float remainderf(float __x, float __y) throw ();
    extern float __remainderf(float __x, float __y) throw ();
    extern float scalbnf(float __x, int __n) throw ();
    extern float __scalbnf(float __x, int __n) throw ();
    extern int ilogbf(float __x) throw ();
    extern int __ilogbf(float __x) throw ();
    extern float scalblnf(float __x, long int __n) throw ();
    extern float __scalblnf(float __x, long int __n) throw ();
    extern float nearbyintf(float __x) throw ();
    extern float __nearbyintf(float __x) throw ();
    extern float roundf(float __x) throw () __attribute__((__const__));
    extern float __roundf(float __x) throw () __attribute__((__const__));
    extern float truncf(float __x) throw () __attribute__((__const__));
    extern float __truncf(float __x) throw () __attribute__((__const__));
    extern float remquof(float __x, float __y, int *__quo) throw ();
    extern float __remquof(float __x, float __y, int *__quo) throw ();
    extern long int lrintf(float __x) throw ();
    extern long int __lrintf(float __x) throw ();
    extern long long int llrintf(float __x) throw ();
    extern long long int __llrintf(float __x) throw ();
    extern long int lroundf(float __x) throw ();
    extern long int __lroundf(float __x) throw ();
    extern long long int llroundf(float __x) throw ();
    extern long long int __llroundf(float __x) throw ();
    extern float fdimf(float __x, float __y) throw ();
    extern float __fdimf(float __x, float __y) throw ();
    extern float fmaxf(float __x, float __y) throw ();
    extern float __fmaxf(float __x, float __y) throw ();
    extern float fminf(float __x, float __y) throw ();
    extern float __fminf(float __x, float __y) throw ();
    extern int __fpclassifyf(float __value) throw () __attribute__((__const__));
    extern int __signbitf(float __value) throw () __attribute__((__const__));
    extern float fmaf(float __x, float __y, float __z) throw ();
    extern float __fmaf(float __x, float __y, float __z) throw ();
    extern float scalbf(float __x, float __n) throw ();
    extern float __scalbf(float __x, float __n) throw ();
    extern long double acosl(long double __x) throw ();
    extern long double __acosl(long double __x) throw ();
    extern long double asinl(long double __x) throw ();
    extern long double __asinl(long double __x) throw ();
    extern long double atanl(long double __x) throw ();
    extern long double __atanl(long double __x) throw ();
    extern long double atan2l(long double __y, long double __x) throw ();
    extern long double __atan2l(long double __y, long double __x) throw ();
    extern long double cosl(long double __x) throw ();
    extern long double __cosl(long double __x) throw ();
    extern long double sinl(long double __x) throw ();
    extern long double __sinl(long double __x) throw ();
    extern long double tanl(long double __x) throw ();
    extern long double __tanl(long double __x) throw ();
    extern long double coshl(long double __x) throw ();
    extern long double __coshl(long double __x) throw ();
    extern long double sinhl(long double __x) throw ();
    extern long double __sinhl(long double __x) throw ();
    extern long double tanhl(long double __x) throw ();
    extern long double __tanhl(long double __x) throw ();
    extern void sincosl(long double __x, long double *__sinx, long double *__cosx) throw ();
    extern void __sincosl(long double __x, long double *__sinx, long double *__cosx) throw ();
    extern long double acoshl(long double __x) throw ();
    extern long double __acoshl(long double __x) throw ();
    extern long double asinhl(long double __x) throw ();
    extern long double __asinhl(long double __x) throw ();
    extern long double atanhl(long double __x) throw ();
    extern long double __atanhl(long double __x) throw ();
    extern long double expl(long double __x) throw ();
    extern long double __expl(long double __x) throw ();
    extern long double frexpl(long double __x, int *__exponent) throw ();
    extern long double __frexpl(long double __x, int *__exponent) throw ();
    extern long double ldexpl(long double __x, int __exponent) throw ();
    extern long double __ldexpl(long double __x, int __exponent) throw ();
    extern long double logl(long double __x) throw ();
    extern long double __logl(long double __x) throw ();
    extern long double log10l(long double __x) throw ();
    extern long double __log10l(long double __x) throw ();
    extern long double modfl(long double __x, long double *__iptr) throw ();
    extern long double __modfl(long double __x, long double *__iptr) throw ();
    extern long double exp10l(long double __x) throw ();
    extern long double __exp10l(long double __x) throw ();
    extern long double pow10l(long double __x) throw ();
    extern long double __pow10l(long double __x) throw ();
    extern long double expm1l(long double __x) throw ();
    extern long double __expm1l(long double __x) throw ();
    extern long double log1pl(long double __x) throw ();
    extern long double __log1pl(long double __x) throw ();
    extern long double logbl(long double __x) throw ();
    extern long double __logbl(long double __x) throw ();
    extern long double exp2l(long double __x) throw ();
    extern long double __exp2l(long double __x) throw ();
    extern long double log2l(long double __x) throw ();
    extern long double __log2l(long double __x) throw ();
    extern long double powl(long double __x, long double __y) throw ();
    extern long double __powl(long double __x, long double __y) throw ();
    extern long double sqrtl(long double __x) throw ();
    extern long double __sqrtl(long double __x) throw ();
    extern long double hypotl(long double __x, long double __y) throw ();
    extern long double __hypotl(long double __x, long double __y) throw ();
    extern long double cbrtl(long double __x) throw ();
    extern long double __cbrtl(long double __x) throw ();
    extern long double ceill(long double __x) throw () __attribute__((__const__));
    extern long double __ceill(long double __x) throw () __attribute__((__const__));
    extern long double fabsl(long double __x) throw () __attribute__((__const__));
    extern long double __fabsl(long double __x) throw () __attribute__((__const__));
    extern long double floorl(long double __x) throw () __attribute__((__const__));
    extern long double __floorl(long double __x) throw () __attribute__((__const__));
    extern long double fmodl(long double __x, long double __y) throw ();
    extern long double __fmodl(long double __x, long double __y) throw ();
    extern int __isinfl(long double __value) throw () __attribute__((__const__));
    extern int __finitel(long double __value) throw () __attribute__((__const__));
    extern int isinfl(long double __value) throw () __attribute__((__const__));
    extern int finitel(long double __value) throw () __attribute__((__const__));
    extern long double dreml(long double __x, long double __y) throw ();
    extern long double __dreml(long double __x, long double __y) throw ();
    extern long double significandl(long double __x) throw ();
    extern long double __significandl(long double __x) throw ();
    extern long double copysignl(long double __x, long double __y) throw () __attribute__((__const__));
    extern long double __copysignl(long double __x, long double __y) throw () __attribute__((__const__));
    extern long double nanl(__const char *__tagb) throw () __attribute__((__const__));
    extern long double __nanl(__const char *__tagb) throw () __attribute__((__const__));
    extern int __isnanl(long double __value) throw () __attribute__((__const__));
    extern int isnanl(long double __value) throw () __attribute__((__const__));
    extern long double j0l(long double) throw ();
    extern long double __j0l(long double) throw ();
    extern long double j1l(long double) throw ();
    extern long double __j1l(long double) throw ();
    extern long double jnl(int, long double) throw ();
    extern long double __jnl(int, long double) throw ();
    extern long double y0l(long double) throw ();
    extern long double __y0l(long double) throw ();
    extern long double y1l(long double) throw ();
    extern long double __y1l(long double) throw ();
    extern long double ynl(int, long double) throw ();
    extern long double __ynl(int, long double) throw ();
    extern long double erfl(long double) throw ();
    extern long double __erfl(long double) throw ();
    extern long double erfcl(long double) throw ();
    extern long double __erfcl(long double) throw ();
    extern long double lgammal(long double) throw ();
    extern long double __lgammal(long double) throw ();
    extern long double tgammal(long double) throw ();
    extern long double __tgammal(long double) throw ();
    extern long double gammal(long double) throw ();
    extern long double __gammal(long double) throw ();
    extern long double lgammal_r(long double, int *__signgamp) throw ();
    extern long double __lgammal_r(long double, int *__signgamp) throw ();
    extern long double rintl(long double __x) throw ();
    extern long double __rintl(long double __x) throw ();
    extern long double nextafterl(long double __x, long double __y) throw () __attribute__((__const__));
    extern long double __nextafterl(long double __x, long double __y) throw () __attribute__((__const__));
    extern long double nexttowardl(long double __x, long double __y) throw () __attribute__((__const__));
    extern long double __nexttowardl(long double __x, long double __y) throw () __attribute__((__const__));
    extern long double remainderl(long double __x, long double __y) throw ();
    extern long double __remainderl(long double __x, long double __y) throw ();
    extern long double scalbnl(long double __x, int __n) throw ();
    extern long double __scalbnl(long double __x, int __n) throw ();
    extern int ilogbl(long double __x) throw ();
    extern int __ilogbl(long double __x) throw ();
    extern long double scalblnl(long double __x, long int __n) throw ();
    extern long double __scalblnl(long double __x, long int __n) throw ();
    extern long double nearbyintl(long double __x) throw ();
    extern long double __nearbyintl(long double __x) throw ();
    extern long double roundl(long double __x) throw () __attribute__((__const__));
    extern long double __roundl(long double __x) throw () __attribute__((__const__));
    extern long double truncl(long double __x) throw () __attribute__((__const__));
    extern long double __truncl(long double __x) throw () __attribute__((__const__));
    extern long double remquol(long double __x, long double __y, int *__quo) throw ();
    extern long double __remquol(long double __x, long double __y, int *__quo) throw ();
    extern long int lrintl(long double __x) throw ();
    extern long int __lrintl(long double __x) throw ();
    extern long long int llrintl(long double __x) throw ();
    extern long long int __llrintl(long double __x) throw ();
    extern long int lroundl(long double __x) throw ();
    extern long int __lroundl(long double __x) throw ();
    extern long long int llroundl(long double __x) throw ();
    extern long long int __llroundl(long double __x) throw ();
    extern long double fdiml(long double __x, long double __y) throw ();
    extern long double __fdiml(long double __x, long double __y) throw ();
    extern long double fmaxl(long double __x, long double __y) throw ();
    extern long double __fmaxl(long double __x, long double __y) throw ();
    extern long double fminl(long double __x, long double __y) throw ();
    extern long double __fminl(long double __x, long double __y) throw ();
    extern int __fpclassifyl(long double __value) throw () __attribute__((__const__));
    extern int __signbitl(long double __value) throw () __attribute__((__const__));
    extern long double fmal(long double __x, long double __y, long double __z) throw ();
    extern long double __fmal(long double __x, long double __y, long double __z) throw ();
    extern long double scalbl(long double __x, long double __n) throw ();
    extern long double __scalbl(long double __x, long double __n) throw ();
    extern int signgam;
    enum 
    {
        FP_NAN, 
        FP_INFINITE, 
        FP_ZERO, 
        FP_SUBNORMAL, 
        FP_NORMAL
    };
    typedef enum 
    {
        _IEEE_ = - 1, 
        _SVID_, 
        _XOPEN_, 
        _POSIX_, 
        _ISOC_
    } _LIB_VERSION_TYPE;
    extern _LIB_VERSION_TYPE _LIB_VERSION;
    struct __exception
    {
            int type;
            char *name;
            double arg1;
            double arg2;
            double retval;
    };
    extern int matherr(struct __exception *__exc) throw ();
    extern __inline __attribute__((__gnu_inline__)) int __signbitf(float __x) throw ()
    {
        int __m;
        __asm ("pmovmskb %1, %0": "=r" (__m): "x" (__x));
        return __m & 0x8;
    }
    extern __inline __attribute__((__gnu_inline__)) int __signbit(double __x) throw ()
    {
        int __m;
        __asm ("pmovmskb %1, %0": "=r" (__m): "x" (__x));
        return __m & 0x80;
    }
    extern __inline __attribute__((__gnu_inline__)) int __signbitl(long double __x) throw ()
    {
        __extension__
        union 
        {
                long double __l;
                int __i[3];
        } __u = {__l:__x};
        return (__u.__i[2] & 0x8000) != 0;
    }
}
namespace std __attribute__((__visibility__("default"))) {
    template<typename _Tp >
    _Tp __cmath_power(_Tp, unsigned int);
    template<typename _Tp >
    inline _Tp __pow_helper(_Tp __x, int __n)
    {
        return __n < 0 ? _Tp(1) / __cmath_power(__x, - __n) : __cmath_power(__x, __n);
    }
    inline double abs(double __x)
    {
        return __builtin_fabs(__x);
    }
    inline float abs(float __x)
    {
        return __builtin_fabsf(__x);
    }
    inline long double abs(long double __x)
    {
        return __builtin_fabsl(__x);
    }
    using ::acos;
    inline float acos(float __x)
    {
        return __builtin_acosf(__x);
    }
    inline long double acos(long double __x)
    {
        return __builtin_acosl(__x);
    }
    template<typename _Tp >
    inline typename __gnu_cxx::__enable_if<__is_integer<_Tp>::__value, double>::__type acos(_Tp __x)
    {
        return __builtin_acos(__x);
    }
    using ::asin;
    inline float asin(float __x)
    {
        return __builtin_asinf(__x);
    }
    inline long double asin(long double __x)
    {
        return __builtin_asinl(__x);
    }
    template<typename _Tp >
    inline typename __gnu_cxx::__enable_if<__is_integer<_Tp>::__value, double>::__type asin(_Tp __x)
    {
        return __builtin_asin(__x);
    }
    using ::atan;
    inline float atan(float __x)
    {
        return __builtin_atanf(__x);
    }
    inline long double atan(long double __x)
    {
        return __builtin_atanl(__x);
    }
    template<typename _Tp >
    inline typename __gnu_cxx::__enable_if<__is_integer<_Tp>::__value, double>::__type atan(_Tp __x)
    {
        return __builtin_atan(__x);
    }
    using ::atan2;
    inline float atan2(float __y, float __x)
    {
        return __builtin_atan2f(__y, __x);
    }
    inline long double atan2(long double __y, long double __x)
    {
        return __builtin_atan2l(__y, __x);
    }
    template<typename _Tp, typename _Up >
    inline typename __gnu_cxx::__promote_2<typename __gnu_cxx::__enable_if<__is_arithmetic<_Tp>::__value && __is_arithmetic<_Up>::__value, _Tp>::__type, _Up>::__type atan2(_Tp __y, _Up __x)
    {
        typedef typename __gnu_cxx::__promote_2<_Tp, _Up>::__type __type;
        return atan2(__type(__y), __type(__x));
    }
    using ::ceil;
    inline float ceil(float __x)
    {
        return __builtin_ceilf(__x);
    }
    inline long double ceil(long double __x)
    {
        return __builtin_ceill(__x);
    }
    template<typename _Tp >
    inline typename __gnu_cxx::__enable_if<__is_integer<_Tp>::__value, double>::__type ceil(_Tp __x)
    {
        return __builtin_ceil(__x);
    }
    using ::cos;
    inline float cos(float __x)
    {
        return __builtin_cosf(__x);
    }
    inline long double cos(long double __x)
    {
        return __builtin_cosl(__x);
    }
    template<typename _Tp >
    inline typename __gnu_cxx::__enable_if<__is_integer<_Tp>::__value, double>::__type cos(_Tp __x)
    {
        return __builtin_cos(__x);
    }
    using ::cosh;
    inline float cosh(float __x)
    {
        return __builtin_coshf(__x);
    }
    inline long double cosh(long double __x)
    {
        return __builtin_coshl(__x);
    }
    template<typename _Tp >
    inline typename __gnu_cxx::__enable_if<__is_integer<_Tp>::__value, double>::__type cosh(_Tp __x)
    {
        return __builtin_cosh(__x);
    }
    using ::exp;
    inline float exp(float __x)
    {
        return __builtin_expf(__x);
    }
    inline long double exp(long double __x)
    {
        return __builtin_expl(__x);
    }
    template<typename _Tp >
    inline typename __gnu_cxx::__enable_if<__is_integer<_Tp>::__value, double>::__type exp(_Tp __x)
    {
        return __builtin_exp(__x);
    }
    using ::fabs;
    inline float fabs(float __x)
    {
        return __builtin_fabsf(__x);
    }
    inline long double fabs(long double __x)
    {
        return __builtin_fabsl(__x);
    }
    template<typename _Tp >
    inline typename __gnu_cxx::__enable_if<__is_integer<_Tp>::__value, double>::__type fabs(_Tp __x)
    {
        return __builtin_fabs(__x);
    }
    using ::floor;
    inline float floor(float __x)
    {
        return __builtin_floorf(__x);
    }
    inline long double floor(long double __x)
    {
        return __builtin_floorl(__x);
    }
    template<typename _Tp >
    inline typename __gnu_cxx::__enable_if<__is_integer<_Tp>::__value, double>::__type floor(_Tp __x)
    {
        return __builtin_floor(__x);
    }
    using ::fmod;
    inline float fmod(float __x, float __y)
    {
        return __builtin_fmodf(__x, __y);
    }
    inline long double fmod(long double __x, long double __y)
    {
        return __builtin_fmodl(__x, __y);
    }
    using ::frexp;
    inline float frexp(float __x, int *__exp)
    {
        return __builtin_frexpf(__x, __exp);
    }
    inline long double frexp(long double __x, int *__exp)
    {
        return __builtin_frexpl(__x, __exp);
    }
    template<typename _Tp >
    inline typename __gnu_cxx::__enable_if<__is_integer<_Tp>::__value, double>::__type frexp(_Tp __x, int *__exp)
    {
        return __builtin_frexp(__x, __exp);
    }
    using ::ldexp;
    inline float ldexp(float __x, int __exp)
    {
        return __builtin_ldexpf(__x, __exp);
    }
    inline long double ldexp(long double __x, int __exp)
    {
        return __builtin_ldexpl(__x, __exp);
    }
    template<typename _Tp >
    inline typename __gnu_cxx::__enable_if<__is_integer<_Tp>::__value, double>::__type ldexp(_Tp __x, int __exp)
    {
        return __builtin_ldexp(__x, __exp);
    }
    using ::log;
    inline float log(float __x)
    {
        return __builtin_logf(__x);
    }
    inline long double log(long double __x)
    {
        return __builtin_logl(__x);
    }
    template<typename _Tp >
    inline typename __gnu_cxx::__enable_if<__is_integer<_Tp>::__value, double>::__type log(_Tp __x)
    {
        return __builtin_log(__x);
    }
    using ::log10;
    inline float log10(float __x)
    {
        return __builtin_log10f(__x);
    }
    inline long double log10(long double __x)
    {
        return __builtin_log10l(__x);
    }
    template<typename _Tp >
    inline typename __gnu_cxx::__enable_if<__is_integer<_Tp>::__value, double>::__type log10(_Tp __x)
    {
        return __builtin_log10(__x);
    }
    using ::modf;
    inline float modf(float __x, float *__iptr)
    {
        return __builtin_modff(__x, __iptr);
    }
    inline long double modf(long double __x, long double *__iptr)
    {
        return __builtin_modfl(__x, __iptr);
    }
    using ::pow;
    inline float pow(float __x, float __y)
    {
        return __builtin_powf(__x, __y);
    }
    inline long double pow(long double __x, long double __y)
    {
        return __builtin_powl(__x, __y);
    }
    inline double pow(double __x, int __i)
    {
        return __builtin_powi(__x, __i);
    }
    inline float pow(float __x, int __n)
    {
        return __builtin_powif(__x, __n);
    }
    inline long double pow(long double __x, int __n)
    {
        return __builtin_powil(__x, __n);
    }
    template<typename _Tp, typename _Up >
    inline typename __gnu_cxx::__promote_2<typename __gnu_cxx::__enable_if<__is_arithmetic<_Tp>::__value && __is_arithmetic<_Up>::__value, _Tp>::__type, _Up>::__type pow(_Tp __x, _Up __y)
    {
        typedef typename __gnu_cxx::__promote_2<_Tp, _Up>::__type __type;
        return pow(__type(__x), __type(__y));
    }
    using ::sin;
    inline float sin(float __x)
    {
        return __builtin_sinf(__x);
    }
    inline long double sin(long double __x)
    {
        return __builtin_sinl(__x);
    }
    template<typename _Tp >
    inline typename __gnu_cxx::__enable_if<__is_integer<_Tp>::__value, double>::__type sin(_Tp __x)
    {
        return __builtin_sin(__x);
    }
    using ::sinh;
    inline float sinh(float __x)
    {
        return __builtin_sinhf(__x);
    }
    inline long double sinh(long double __x)
    {
        return __builtin_sinhl(__x);
    }
    template<typename _Tp >
    inline typename __gnu_cxx::__enable_if<__is_integer<_Tp>::__value, double>::__type sinh(_Tp __x)
    {
        return __builtin_sinh(__x);
    }
    using ::sqrt;
    inline float sqrt(float __x)
    {
        return __builtin_sqrtf(__x);
    }
    inline long double sqrt(long double __x)
    {
        return __builtin_sqrtl(__x);
    }
    template<typename _Tp >
    inline typename __gnu_cxx::__enable_if<__is_integer<_Tp>::__value, double>::__type sqrt(_Tp __x)
    {
        return __builtin_sqrt(__x);
    }
    using ::tan;
    inline float tan(float __x)
    {
        return __builtin_tanf(__x);
    }
    inline long double tan(long double __x)
    {
        return __builtin_tanl(__x);
    }
    template<typename _Tp >
    inline typename __gnu_cxx::__enable_if<__is_integer<_Tp>::__value, double>::__type tan(_Tp __x)
    {
        return __builtin_tan(__x);
    }
    using ::tanh;
    inline float tanh(float __x)
    {
        return __builtin_tanhf(__x);
    }
    inline long double tanh(long double __x)
    {
        return __builtin_tanhl(__x);
    }
    template<typename _Tp >
    inline typename __gnu_cxx::__enable_if<__is_integer<_Tp>::__value, double>::__type tanh(_Tp __x)
    {
        return __builtin_tanh(__x);
    }
}
namespace std __attribute__((__visibility__("default"))) {
    template<typename _Tp >
    inline typename __gnu_cxx::__enable_if<__is_arithmetic<_Tp>::__value, int>::__type fpclassify(_Tp __f)
    {
        typedef typename __gnu_cxx::__promote<_Tp>::__type __type;
        return __builtin_fpclassify(FP_NAN, FP_INFINITE, FP_NORMAL, FP_SUBNORMAL, FP_ZERO, __type(__f));
    }
    template<typename _Tp >
    inline typename __gnu_cxx::__enable_if<__is_arithmetic<_Tp>::__value, int>::__type isfinite(_Tp __f)
    {
        typedef typename __gnu_cxx::__promote<_Tp>::__type __type;
        return __builtin_isfinite(__type(__f));
    }
    template<typename _Tp >
    inline typename __gnu_cxx::__enable_if<__is_arithmetic<_Tp>::__value, int>::__type isinf(_Tp __f)
    {
        typedef typename __gnu_cxx::__promote<_Tp>::__type __type;
        return __builtin_isinf(__type(__f));
    }
    template<typename _Tp >
    inline typename __gnu_cxx::__enable_if<__is_arithmetic<_Tp>::__value, int>::__type isnan(_Tp __f)
    {
        typedef typename __gnu_cxx::__promote<_Tp>::__type __type;
        return __builtin_isnan(__type(__f));
    }
    template<typename _Tp >
    inline typename __gnu_cxx::__enable_if<__is_arithmetic<_Tp>::__value, int>::__type isnormal(_Tp __f)
    {
        typedef typename __gnu_cxx::__promote<_Tp>::__type __type;
        return __builtin_isnormal(__type(__f));
    }
    template<typename _Tp >
    inline typename __gnu_cxx::__enable_if<__is_arithmetic<_Tp>::__value, int>::__type signbit(_Tp __f)
    {
        typedef typename __gnu_cxx::__promote<_Tp>::__type __type;
        return __builtin_signbit(__type(__f));
    }
    template<typename _Tp >
    inline typename __gnu_cxx::__enable_if<__is_arithmetic<_Tp>::__value, int>::__type isgreater(_Tp __f1, _Tp __f2)
    {
        typedef typename __gnu_cxx::__promote<_Tp>::__type __type;
        return __builtin_isgreater(__type(__f1), __type(__f2));
    }
    template<typename _Tp >
    inline typename __gnu_cxx::__enable_if<__is_arithmetic<_Tp>::__value, int>::__type isgreaterequal(_Tp __f1, _Tp __f2)
    {
        typedef typename __gnu_cxx::__promote<_Tp>::__type __type;
        return __builtin_isgreaterequal(__type(__f1), __type(__f2));
    }
    template<typename _Tp >
    inline typename __gnu_cxx::__enable_if<__is_arithmetic<_Tp>::__value, int>::__type isless(_Tp __f1, _Tp __f2)
    {
        typedef typename __gnu_cxx::__promote<_Tp>::__type __type;
        return __builtin_isless(__type(__f1), __type(__f2));
    }
    template<typename _Tp >
    inline typename __gnu_cxx::__enable_if<__is_arithmetic<_Tp>::__value, int>::__type islessequal(_Tp __f1, _Tp __f2)
    {
        typedef typename __gnu_cxx::__promote<_Tp>::__type __type;
        return __builtin_islessequal(__type(__f1), __type(__f2));
    }
    template<typename _Tp >
    inline typename __gnu_cxx::__enable_if<__is_arithmetic<_Tp>::__value, int>::__type islessgreater(_Tp __f1, _Tp __f2)
    {
        typedef typename __gnu_cxx::__promote<_Tp>::__type __type;
        return __builtin_islessgreater(__type(__f1), __type(__f2));
    }
    template<typename _Tp >
    inline typename __gnu_cxx::__enable_if<__is_arithmetic<_Tp>::__value, int>::__type isunordered(_Tp __f1, _Tp __f2)
    {
        typedef typename __gnu_cxx::__promote<_Tp>::__type __type;
        return __builtin_isunordered(__type(__f1), __type(__f2));
    }
}
namespace std __attribute__((__visibility__("default"))) {
    template<typename _Tp >
    inline _Tp __cmath_power(_Tp __x, unsigned int __n)
    {
        _Tp __y = __n % 2 ? __x : _Tp(1);
        while (__n >>= 1)
        {
            __x = __x * __x;
            if (__n % 2)
                __y = __y * __x;
        }
        return __y;
    }
}
namespace std __attribute__((__visibility__("default"))) {
    template<typename _Tp >
    class complex;
    template<>
    class complex<float>;
    template<>
    class complex<double>;
    template<>
    class complex<long double>;
    template<typename _Tp >
    _Tp abs(const complex<_Tp> &);
    template<typename _Tp >
    _Tp arg(const complex<_Tp> &);
    template<typename _Tp >
    _Tp norm(const complex<_Tp> &);
    template<typename _Tp >
    complex<_Tp> conj(const complex<_Tp> &);
    template<typename _Tp >
    complex<_Tp> polar(const _Tp &, const _Tp & = 0);
    template<typename _Tp >
    complex<_Tp> cos(const complex<_Tp> &);
    template<typename _Tp >
    complex<_Tp> cosh(const complex<_Tp> &);
    template<typename _Tp >
    complex<_Tp> exp(const complex<_Tp> &);
    template<typename _Tp >
    complex<_Tp> log(const complex<_Tp> &);
    template<typename _Tp >
    complex<_Tp> log10(const complex<_Tp> &);
    template<typename _Tp >
    complex<_Tp> pow(const complex<_Tp> &, int);
    template<typename _Tp >
    complex<_Tp> pow(const complex<_Tp> &, const _Tp &);
    template<typename _Tp >
    complex<_Tp> pow(const complex<_Tp> &, const complex<_Tp> &);
    template<typename _Tp >
    complex<_Tp> pow(const _Tp &, const complex<_Tp> &);
    template<typename _Tp >
    complex<_Tp> sin(const complex<_Tp> &);
    template<typename _Tp >
    complex<_Tp> sinh(const complex<_Tp> &);
    template<typename _Tp >
    complex<_Tp> sqrt(const complex<_Tp> &);
    template<typename _Tp >
    complex<_Tp> tan(const complex<_Tp> &);
    template<typename _Tp >
    complex<_Tp> tanh(const complex<_Tp> &);
    template<typename _Tp >
    struct complex
    {
            typedef _Tp value_type;
            complex(const _Tp &__r = _Tp(), const _Tp &__i = _Tp())
                : _M_real(__r), _M_imag(__i) 
            {
            }
            template<typename _Up >
            complex(const complex<_Up> &__z)
                : _M_real(__z.real()), _M_imag(__z.imag()) 
            {
            }
            _Tp &real()
            {
                return _M_real;
            }
            const _Tp &real() const
            {
                return _M_real;
            }
            _Tp &imag()
            {
                return _M_imag;
            }
            const _Tp &imag() const
            {
                return _M_imag;
            }
            void real(_Tp __val)
            {
                _M_real = __val;
            }
            void imag(_Tp __val)
            {
                _M_imag = __val;
            }
            complex<_Tp> &operator =(const _Tp &);
            complex<_Tp> &operator +=(const _Tp &__t)
            {
                _M_real += __t;
                return *this;
            }
            complex<_Tp> &operator -=(const _Tp &__t)
            {
                _M_real -= __t;
                return *this;
            }
            complex<_Tp> &operator *=(const _Tp &);
            complex<_Tp> &operator /=(const _Tp &);
            template<typename _Up >
            complex<_Tp> &operator =(const complex<_Up> &);
            template<typename _Up >
            complex<_Tp> &operator +=(const complex<_Up> &);
            template<typename _Up >
            complex<_Tp> &operator -=(const complex<_Up> &);
            template<typename _Up >
            complex<_Tp> &operator *=(const complex<_Up> &);
            template<typename _Up >
            complex<_Tp> &operator /=(const complex<_Up> &);
            const complex &__rep() const
            {
                return *this;
            }
        private :
            _Tp _M_real;
            _Tp _M_imag;
    };
    template<typename _Tp >
    complex<_Tp> &complex<_Tp>::operator =(const _Tp &__t)
    {
        _M_real = __t;
        _M_imag = _Tp();
        return *this;
    }
    template<typename _Tp >
    complex<_Tp> &complex<_Tp>::operator *=(const _Tp &__t)
    {
        _M_real *= __t;
        _M_imag *= __t;
        return *this;
    }
    template<typename _Tp >
    complex<_Tp> &complex<_Tp>::operator /=(const _Tp &__t)
    {
        _M_real /= __t;
        _M_imag /= __t;
        return *this;
    }
    template<typename _Tp >
    template<typename _Up >
    complex<_Tp> &complex<_Tp>::operator =(const complex<_Up> &__z)
    {
        _M_real = __z.real();
        _M_imag = __z.imag();
        return *this;
    }
    template<typename _Tp >
    template<typename _Up >
    complex<_Tp> &complex<_Tp>::operator +=(const complex<_Up> &__z)
    {
        _M_real += __z.real();
        _M_imag += __z.imag();
        return *this;
    }
    template<typename _Tp >
    template<typename _Up >
    complex<_Tp> &complex<_Tp>::operator -=(const complex<_Up> &__z)
    {
        _M_real -= __z.real();
        _M_imag -= __z.imag();
        return *this;
    }
    template<typename _Tp >
    template<typename _Up >
    complex<_Tp> &complex<_Tp>::operator *=(const complex<_Up> &__z)
    {
        const _Tp __r = _M_real * __z.real() - _M_imag * __z.imag();
        _M_imag = _M_real * __z.imag() + _M_imag * __z.real();
        _M_real = __r;
        return *this;
    }
    template<typename _Tp >
    template<typename _Up >
    complex<_Tp> &complex<_Tp>::operator /=(const complex<_Up> &__z)
    {
        const _Tp __r = _M_real * __z.real() + _M_imag * __z.imag();
        const _Tp __n = std::norm(__z);
        _M_imag = (_M_imag * __z.real() - _M_real * __z.imag()) / __n;
        _M_real = __r / __n;
        return *this;
    }
    template<typename _Tp >
    inline complex<_Tp> operator +(const complex<_Tp> &__x, const complex<_Tp> &__y)
    {
        complex<_Tp> __r = __x;
        __r += __y;
        return __r;
    }
    template<typename _Tp >
    inline complex<_Tp> operator +(const complex<_Tp> &__x, const _Tp &__y)
    {
        complex<_Tp> __r = __x;
        __r += __y;
        return __r;
    }
    template<typename _Tp >
    inline complex<_Tp> operator +(const _Tp &__x, const complex<_Tp> &__y)
    {
        complex<_Tp> __r = __y;
        __r += __x;
        return __r;
    }
    template<typename _Tp >
    inline complex<_Tp> operator -(const complex<_Tp> &__x, const complex<_Tp> &__y)
    {
        complex<_Tp> __r = __x;
        __r -= __y;
        return __r;
    }
    template<typename _Tp >
    inline complex<_Tp> operator -(const complex<_Tp> &__x, const _Tp &__y)
    {
        complex<_Tp> __r = __x;
        __r -= __y;
        return __r;
    }
    template<typename _Tp >
    inline complex<_Tp> operator -(const _Tp &__x, const complex<_Tp> &__y)
    {
        complex<_Tp> __r(__x, - __y.imag());
        __r -= __y.real();
        return __r;
    }
    template<typename _Tp >
    inline complex<_Tp> operator *(const complex<_Tp> &__x, const complex<_Tp> &__y)
    {
        complex<_Tp> __r = __x;
        __r *= __y;
        return __r;
    }
    template<typename _Tp >
    inline complex<_Tp> operator *(const complex<_Tp> &__x, const _Tp &__y)
    {
        complex<_Tp> __r = __x;
        __r *= __y;
        return __r;
    }
    template<typename _Tp >
    inline complex<_Tp> operator *(const _Tp &__x, const complex<_Tp> &__y)
    {
        complex<_Tp> __r = __y;
        __r *= __x;
        return __r;
    }
    template<typename _Tp >
    inline complex<_Tp> operator /(const complex<_Tp> &__x, const complex<_Tp> &__y)
    {
        complex<_Tp> __r = __x;
        __r /= __y;
        return __r;
    }
    template<typename _Tp >
    inline complex<_Tp> operator /(const complex<_Tp> &__x, const _Tp &__y)
    {
        complex<_Tp> __r = __x;
        __r /= __y;
        return __r;
    }
    template<typename _Tp >
    inline complex<_Tp> operator /(const _Tp &__x, const complex<_Tp> &__y)
    {
        complex<_Tp> __r = __x;
        __r /= __y;
        return __r;
    }
    template<typename _Tp >
    inline complex<_Tp> operator +(const complex<_Tp> &__x)
    {
        return __x;
    }
    template<typename _Tp >
    inline complex<_Tp> operator -(const complex<_Tp> &__x)
    {
        return complex<_Tp>(- __x.real(), - __x.imag());
    }
    template<typename _Tp >
    inline bool operator ==(const complex<_Tp> &__x, const complex<_Tp> &__y)
    {
        return __x.real() == __y.real() && __x.imag() == __y.imag();
    }
    template<typename _Tp >
    inline bool operator ==(const complex<_Tp> &__x, const _Tp &__y)
    {
        return __x.real() == __y && __x.imag() == _Tp();
    }
    template<typename _Tp >
    inline bool operator ==(const _Tp &__x, const complex<_Tp> &__y)
    {
        return __x == __y.real() && _Tp() == __y.imag();
    }
    template<typename _Tp >
    inline bool operator !=(const complex<_Tp> &__x, const complex<_Tp> &__y)
    {
        return __x.real() != __y.real() || __x.imag() != __y.imag();
    }
    template<typename _Tp >
    inline bool operator !=(const complex<_Tp> &__x, const _Tp &__y)
    {
        return __x.real() != __y || __x.imag() != _Tp();
    }
    template<typename _Tp >
    inline bool operator !=(const _Tp &__x, const complex<_Tp> &__y)
    {
        return __x != __y.real() || _Tp() != __y.imag();
    }
    template<typename _Tp, typename _CharT, class _Traits >
    basic_istream<_CharT, _Traits> &operator >>(basic_istream<_CharT, _Traits> &__is, complex<_Tp> &__x)
    {
        _Tp __re_x, __im_x;
        _CharT __ch;
        __is >> __ch;
        if (__ch == '(')
        {
            __is >> __re_x >> __ch;
            if (__ch == ',')
            {
                __is >> __im_x >> __ch;
                if (__ch == ')')
                    __x = complex<_Tp>(__re_x, __im_x);
                else
                    __is.setstate(ios_base::failbit);
            }
            else
                if (__ch == ')')
                    __x = __re_x;
                else
                    __is.setstate(ios_base::failbit);
        }
        else
        {
            __is.putback(__ch);
            __is >> __re_x;
            __x = __re_x;
        }
        return __is;
    }
    template<typename _Tp, typename _CharT, class _Traits >
    basic_ostream<_CharT, _Traits> &operator <<(basic_ostream<_CharT, _Traits> &__os, const complex<_Tp> &__x)
    {
        basic_ostringstream<_CharT, _Traits> __s;
        __s.flags(__os.flags());
        __s.imbue(__os.getloc());
        __s.precision(__os.precision());
        __s << '(' << __x.real() << ',' << __x.imag() << ')';
        return __os << __s.str();
    }
    template<typename _Tp >
    inline _Tp &real(complex<_Tp> &__z)
    {
        return __z.real();
    }
    template<typename _Tp >
    inline const _Tp &real(const complex<_Tp> &__z)
    {
        return __z.real();
    }
    template<typename _Tp >
    inline _Tp &imag(complex<_Tp> &__z)
    {
        return __z.imag();
    }
    template<typename _Tp >
    inline const _Tp &imag(const complex<_Tp> &__z)
    {
        return __z.imag();
    }
    template<typename _Tp >
    inline _Tp __complex_abs(const complex<_Tp> &__z)
    {
        _Tp __x = __z.real();
        _Tp __y = __z.imag();
        const _Tp __s = std::max(abs(__x), abs(__y));
        if (__s == _Tp())
            return __s;
        __x /= __s;
        __y /= __s;
        return __s * sqrt(__x * __x + __y * __y);
    }
    inline float __complex_abs(__complex__ float __z)
    {
        return __builtin_cabsf(__z);
    }
    inline double __complex_abs(__complex__ double __z)
    {
        return __builtin_cabs(__z);
    }
    inline long double __complex_abs(const __complex__ long double &__z)
    {
        return __builtin_cabsl(__z);
    }
    template<typename _Tp >
    inline _Tp abs(const complex<_Tp> &__z)
    {
        return __complex_abs(__z.__rep());
    }
    template<typename _Tp >
    inline _Tp __complex_arg(const complex<_Tp> &__z)
    {
        return atan2(__z.imag(), __z.real());
    }
    inline float __complex_arg(__complex__ float __z)
    {
        return __builtin_cargf(__z);
    }
    inline double __complex_arg(__complex__ double __z)
    {
        return __builtin_carg(__z);
    }
    inline long double __complex_arg(const __complex__ long double &__z)
    {
        return __builtin_cargl(__z);
    }
    template<typename _Tp >
    inline _Tp arg(const complex<_Tp> &__z)
    {
        return __complex_arg(__z.__rep());
    }
    template<bool >
    struct _Norm_helper
    {
            template<typename _Tp >
            static inline _Tp _S_do_it(const complex<_Tp> &__z)
            {
                const _Tp __x = __z.real();
                const _Tp __y = __z.imag();
                return __x * __x + __y * __y;
            }
    };
    template<>
    struct _Norm_helper<true>
    {
            template<typename _Tp >
            static inline _Tp _S_do_it(const complex<_Tp> &__z)
            {
                _Tp __res = std::abs(__z);
                return __res * __res;
            }
    };
    template<typename _Tp >
    inline _Tp norm(const complex<_Tp> &__z)
    {
        return _Norm_helper<__is_floating<_Tp>::__value && !1>::_S_do_it(__z);
    }
    template<typename _Tp >
    inline complex<_Tp> polar(const _Tp &__rho, const _Tp &__theta)
    {
        return complex<_Tp>(__rho * cos(__theta), __rho * sin(__theta));
    }
    template<typename _Tp >
    inline complex<_Tp> conj(const complex<_Tp> &__z)
    {
        return complex<_Tp>(__z.real(), - __z.imag());
    }
    template<typename _Tp >
    inline complex<_Tp> __complex_cos(const complex<_Tp> &__z)
    {
        const _Tp __x = __z.real();
        const _Tp __y = __z.imag();
        return complex<_Tp>(cos(__x) * cosh(__y), - sin(__x) * sinh(__y));
    }
    inline __complex__ float __complex_cos(__complex__ float __z)
    {
        return __builtin_ccosf(__z);
    }
    inline __complex__ double __complex_cos(__complex__ double __z)
    {
        return __builtin_ccos(__z);
    }
    inline __complex__ long double __complex_cos(const __complex__ long double &__z)
    {
        return __builtin_ccosl(__z);
    }
    template<typename _Tp >
    inline complex<_Tp> cos(const complex<_Tp> &__z)
    {
        return __complex_cos(__z.__rep());
    }
    template<typename _Tp >
    inline complex<_Tp> __complex_cosh(const complex<_Tp> &__z)
    {
        const _Tp __x = __z.real();
        const _Tp __y = __z.imag();
        return complex<_Tp>(cosh(__x) * cos(__y), sinh(__x) * sin(__y));
    }
    inline __complex__ float __complex_cosh(__complex__ float __z)
    {
        return __builtin_ccoshf(__z);
    }
    inline __complex__ double __complex_cosh(__complex__ double __z)
    {
        return __builtin_ccosh(__z);
    }
    inline __complex__ long double __complex_cosh(const __complex__ long double &__z)
    {
        return __builtin_ccoshl(__z);
    }
    template<typename _Tp >
    inline complex<_Tp> cosh(const complex<_Tp> &__z)
    {
        return __complex_cosh(__z.__rep());
    }
    template<typename _Tp >
    inline complex<_Tp> __complex_exp(const complex<_Tp> &__z)
    {
        return std::polar(exp(__z.real()), __z.imag());
    }
    inline __complex__ float __complex_exp(__complex__ float __z)
    {
        return __builtin_cexpf(__z);
    }
    inline __complex__ double __complex_exp(__complex__ double __z)
    {
        return __builtin_cexp(__z);
    }
    inline __complex__ long double __complex_exp(const __complex__ long double &__z)
    {
        return __builtin_cexpl(__z);
    }
    template<typename _Tp >
    inline complex<_Tp> exp(const complex<_Tp> &__z)
    {
        return __complex_exp(__z.__rep());
    }
    template<typename _Tp >
    inline complex<_Tp> __complex_log(const complex<_Tp> &__z)
    {
        return complex<_Tp>(log(std::abs(__z)), std::arg(__z));
    }
    inline __complex__ float __complex_log(__complex__ float __z)
    {
        return __builtin_clogf(__z);
    }
    inline __complex__ double __complex_log(__complex__ double __z)
    {
        return __builtin_clog(__z);
    }
    inline __complex__ long double __complex_log(const __complex__ long double &__z)
    {
        return __builtin_clogl(__z);
    }
    template<typename _Tp >
    inline complex<_Tp> log(const complex<_Tp> &__z)
    {
        return __complex_log(__z.__rep());
    }
    template<typename _Tp >
    inline complex<_Tp> log10(const complex<_Tp> &__z)
    {
        return std::log(__z) / log(_Tp(10.0));
    }
    template<typename _Tp >
    inline complex<_Tp> __complex_sin(const complex<_Tp> &__z)
    {
        const _Tp __x = __z.real();
        const _Tp __y = __z.imag();
        return complex<_Tp>(sin(__x) * cosh(__y), cos(__x) * sinh(__y));
    }
    inline __complex__ float __complex_sin(__complex__ float __z)
    {
        return __builtin_csinf(__z);
    }
    inline __complex__ double __complex_sin(__complex__ double __z)
    {
        return __builtin_csin(__z);
    }
    inline __complex__ long double __complex_sin(const __complex__ long double &__z)
    {
        return __builtin_csinl(__z);
    }
    template<typename _Tp >
    inline complex<_Tp> sin(const complex<_Tp> &__z)
    {
        return __complex_sin(__z.__rep());
    }
    template<typename _Tp >
    inline complex<_Tp> __complex_sinh(const complex<_Tp> &__z)
    {
        const _Tp __x = __z.real();
        const _Tp __y = __z.imag();
        return complex<_Tp>(sinh(__x) * cos(__y), cosh(__x) * sin(__y));
    }
    inline __complex__ float __complex_sinh(__complex__ float __z)
    {
        return __builtin_csinhf(__z);
    }
    inline __complex__ double __complex_sinh(__complex__ double __z)
    {
        return __builtin_csinh(__z);
    }
    inline __complex__ long double __complex_sinh(const __complex__ long double &__z)
    {
        return __builtin_csinhl(__z);
    }
    template<typename _Tp >
    inline complex<_Tp> sinh(const complex<_Tp> &__z)
    {
        return __complex_sinh(__z.__rep());
    }
    template<typename _Tp >
    complex<_Tp> __complex_sqrt(const complex<_Tp> &__z)
    {
        _Tp __x = __z.real();
        _Tp __y = __z.imag();
        if (__x == _Tp())
        {
            _Tp __t = sqrt(abs(__y) / 2);
            return complex<_Tp>(__t, __y < _Tp() ? - __t : __t);
        }
        else
        {
            _Tp __t = sqrt(2 * (std::abs(__z) + abs(__x)));
            _Tp __u = __t / 2;
            return __x > _Tp() ? complex<_Tp>(__u, __y / __t) : complex<_Tp>(abs(__y) / __t, __y < _Tp() ? - __u : __u);
        }
    }
    inline __complex__ float __complex_sqrt(__complex__ float __z)
    {
        return __builtin_csqrtf(__z);
    }
    inline __complex__ double __complex_sqrt(__complex__ double __z)
    {
        return __builtin_csqrt(__z);
    }
    inline __complex__ long double __complex_sqrt(const __complex__ long double &__z)
    {
        return __builtin_csqrtl(__z);
    }
    template<typename _Tp >
    inline complex<_Tp> sqrt(const complex<_Tp> &__z)
    {
        return __complex_sqrt(__z.__rep());
    }
    template<typename _Tp >
    inline complex<_Tp> __complex_tan(const complex<_Tp> &__z)
    {
        return std::sin(__z) / std::cos(__z);
    }
    inline __complex__ float __complex_tan(__complex__ float __z)
    {
        return __builtin_ctanf(__z);
    }
    inline __complex__ double __complex_tan(__complex__ double __z)
    {
        return __builtin_ctan(__z);
    }
    inline __complex__ long double __complex_tan(const __complex__ long double &__z)
    {
        return __builtin_ctanl(__z);
    }
    template<typename _Tp >
    inline complex<_Tp> tan(const complex<_Tp> &__z)
    {
        return __complex_tan(__z.__rep());
    }
    template<typename _Tp >
    inline complex<_Tp> __complex_tanh(const complex<_Tp> &__z)
    {
        return std::sinh(__z) / std::cosh(__z);
    }
    inline __complex__ float __complex_tanh(__complex__ float __z)
    {
        return __builtin_ctanhf(__z);
    }
    inline __complex__ double __complex_tanh(__complex__ double __z)
    {
        return __builtin_ctanh(__z);
    }
    inline __complex__ long double __complex_tanh(const __complex__ long double &__z)
    {
        return __builtin_ctanhl(__z);
    }
    template<typename _Tp >
    inline complex<_Tp> tanh(const complex<_Tp> &__z)
    {
        return __complex_tanh(__z.__rep());
    }
    template<typename _Tp >
    inline complex<_Tp> pow(const complex<_Tp> &__z, int __n)
    {
        return std::__pow_helper(__z, __n);
    }
    template<typename _Tp >
    complex<_Tp> pow(const complex<_Tp> &__x, const _Tp &__y)
    {
        if (__x.imag() == _Tp() && __x.real() > _Tp())
            return pow(__x.real(), __y);
        complex<_Tp> __t = std::log(__x);
        return std::polar(exp(__y * __t.real()), __y * __t.imag());
    }
    template<typename _Tp >
    inline complex<_Tp> __complex_pow(const complex<_Tp> &__x, const complex<_Tp> &__y)
    {
        return __x == _Tp() ? _Tp() : std::exp(__y * std::log(__x));
    }
    inline __complex__ float __complex_pow(__complex__ float __x, __complex__ float __y)
    {
        return __builtin_cpowf(__x, __y);
    }
    inline __complex__ double __complex_pow(__complex__ double __x, __complex__ double __y)
    {
        return __builtin_cpow(__x, __y);
    }
    inline __complex__ long double __complex_pow(const __complex__ long double &__x, const __complex__ long double &__y)
    {
        return __builtin_cpowl(__x, __y);
    }
    template<typename _Tp >
    inline complex<_Tp> pow(const complex<_Tp> &__x, const complex<_Tp> &__y)
    {
        return __complex_pow(__x.__rep(), __y.__rep());
    }
    template<typename _Tp >
    inline complex<_Tp> pow(const _Tp &__x, const complex<_Tp> &__y)
    {
        return __x > _Tp() ? std::polar(pow(__x, __y.real()), __y.imag() * log(__x)) : std::pow(complex<_Tp>(__x), __y);
    }
    template<>
    struct complex<float>
    {
            typedef float value_type;
            typedef __complex__ float _ComplexT;
            complex(_ComplexT __z)
                : _M_value(__z) 
            {
            }
            complex(float __r = 0.0f, float __i = 0.0f)
            {
                __real__ _M_value = __r;
                __imag__ _M_value = __i;
            }
            explicit complex(const complex<double> &);
            explicit complex(const complex<long double> &);
            float &real()
            {
                return __real__ _M_value;
            }
            const float &real() const
            {
                return __real__ _M_value;
            }
            float &imag()
            {
                return __imag__ _M_value;
            }
            const float &imag() const
            {
                return __imag__ _M_value;
            }
            void real(float __val)
            {
                __real__ _M_value = __val;
            }
            void imag(float __val)
            {
                __imag__ _M_value = __val;
            }
            complex<float> &operator =(float __f)
            {
                __real__ _M_value = __f;
                __imag__ _M_value = 0.0f;
                return *this;
            }
            complex<float> &operator +=(float __f)
            {
                __real__ _M_value += __f;
                return *this;
            }
            complex<float> &operator -=(float __f)
            {
                __real__ _M_value -= __f;
                return *this;
            }
            complex<float> &operator *=(float __f)
            {
                _M_value *= __f;
                return *this;
            }
            complex<float> &operator /=(float __f)
            {
                _M_value /= __f;
                return *this;
            }
            template<typename _Tp >
            complex<float> &operator =(const complex<_Tp> &__z)
            {
                __real__ _M_value = __z.real();
                __imag__ _M_value = __z.imag();
                return *this;
            }
            template<typename _Tp >
            complex<float> &operator +=(const complex<_Tp> &__z)
            {
                __real__ _M_value += __z.real();
                __imag__ _M_value += __z.imag();
                return *this;
            }
            template<class _Tp >
            complex<float> &operator -=(const complex<_Tp> &__z)
            {
                __real__ _M_value -= __z.real();
                __imag__ _M_value -= __z.imag();
                return *this;
            }
            template<class _Tp >
            complex<float> &operator *=(const complex<_Tp> &__z)
            {
                _ComplexT __t;
                __real__ __t = __z.real();
                __imag__ __t = __z.imag();
                _M_value *= __t;
                return *this;
            }
            template<class _Tp >
            complex<float> &operator /=(const complex<_Tp> &__z)
            {
                _ComplexT __t;
                __real__ __t = __z.real();
                __imag__ __t = __z.imag();
                _M_value /= __t;
                return *this;
            }
            const _ComplexT &__rep() const
            {
                return _M_value;
            }
        private :
            _ComplexT _M_value;
    };
    template<>
    struct complex<double>
    {
            typedef double value_type;
            typedef __complex__ double _ComplexT;
            complex(_ComplexT __z)
                : _M_value(__z) 
            {
            }
            complex(double __r = 0.0, double __i = 0.0)
            {
                __real__ _M_value = __r;
                __imag__ _M_value = __i;
            }
            complex(const complex<float> &__z)
                : _M_value(__z.__rep()) 
            {
            }
            explicit complex(const complex<long double> &);
            double &real()
            {
                return __real__ _M_value;
            }
            const double &real() const
            {
                return __real__ _M_value;
            }
            double &imag()
            {
                return __imag__ _M_value;
            }
            const double &imag() const
            {
                return __imag__ _M_value;
            }
            void real(double __val)
            {
                __real__ _M_value = __val;
            }
            void imag(double __val)
            {
                __imag__ _M_value = __val;
            }
            complex<double> &operator =(double __d)
            {
                __real__ _M_value = __d;
                __imag__ _M_value = 0.0;
                return *this;
            }
            complex<double> &operator +=(double __d)
            {
                __real__ _M_value += __d;
                return *this;
            }
            complex<double> &operator -=(double __d)
            {
                __real__ _M_value -= __d;
                return *this;
            }
            complex<double> &operator *=(double __d)
            {
                _M_value *= __d;
                return *this;
            }
            complex<double> &operator /=(double __d)
            {
                _M_value /= __d;
                return *this;
            }
            template<typename _Tp >
            complex<double> &operator =(const complex<_Tp> &__z)
            {
                __real__ _M_value = __z.real();
                __imag__ _M_value = __z.imag();
                return *this;
            }
            template<typename _Tp >
            complex<double> &operator +=(const complex<_Tp> &__z)
            {
                __real__ _M_value += __z.real();
                __imag__ _M_value += __z.imag();
                return *this;
            }
            template<typename _Tp >
            complex<double> &operator -=(const complex<_Tp> &__z)
            {
                __real__ _M_value -= __z.real();
                __imag__ _M_value -= __z.imag();
                return *this;
            }
            template<typename _Tp >
            complex<double> &operator *=(const complex<_Tp> &__z)
            {
                _ComplexT __t;
                __real__ __t = __z.real();
                __imag__ __t = __z.imag();
                _M_value *= __t;
                return *this;
            }
            template<typename _Tp >
            complex<double> &operator /=(const complex<_Tp> &__z)
            {
                _ComplexT __t;
                __real__ __t = __z.real();
                __imag__ __t = __z.imag();
                _M_value /= __t;
                return *this;
            }
            const _ComplexT &__rep() const
            {
                return _M_value;
            }
        private :
            _ComplexT _M_value;
    };
    template<>
    struct complex<long double>
    {
            typedef long double value_type;
            typedef __complex__ long double _ComplexT;
            complex(_ComplexT __z)
                : _M_value(__z) 
            {
            }
            complex(long double __r = 0.0L, long double __i = 0.0L)
            {
                __real__ _M_value = __r;
                __imag__ _M_value = __i;
            }
            complex(const complex<float> &__z)
                : _M_value(__z.__rep()) 
            {
            }
            complex(const complex<double> &__z)
                : _M_value(__z.__rep()) 
            {
            }
            long double &real()
            {
                return __real__ _M_value;
            }
            const long double &real() const
            {
                return __real__ _M_value;
            }
            long double &imag()
            {
                return __imag__ _M_value;
            }
            const long double &imag() const
            {
                return __imag__ _M_value;
            }
            void real(long double __val)
            {
                __real__ _M_value = __val;
            }
            void imag(long double __val)
            {
                __imag__ _M_value = __val;
            }
            complex<long double> &operator =(long double __r)
            {
                __real__ _M_value = __r;
                __imag__ _M_value = 0.0L;
                return *this;
            }
            complex<long double> &operator +=(long double __r)
            {
                __real__ _M_value += __r;
                return *this;
            }
            complex<long double> &operator -=(long double __r)
            {
                __real__ _M_value -= __r;
                return *this;
            }
            complex<long double> &operator *=(long double __r)
            {
                _M_value *= __r;
                return *this;
            }
            complex<long double> &operator /=(long double __r)
            {
                _M_value /= __r;
                return *this;
            }
            template<typename _Tp >
            complex<long double> &operator =(const complex<_Tp> &__z)
            {
                __real__ _M_value = __z.real();
                __imag__ _M_value = __z.imag();
                return *this;
            }
            template<typename _Tp >
            complex<long double> &operator +=(const complex<_Tp> &__z)
            {
                __real__ _M_value += __z.real();
                __imag__ _M_value += __z.imag();
                return *this;
            }
            template<typename _Tp >
            complex<long double> &operator -=(const complex<_Tp> &__z)
            {
                __real__ _M_value -= __z.real();
                __imag__ _M_value -= __z.imag();
                return *this;
            }
            template<typename _Tp >
            complex<long double> &operator *=(const complex<_Tp> &__z)
            {
                _ComplexT __t;
                __real__ __t = __z.real();
                __imag__ __t = __z.imag();
                _M_value *= __t;
                return *this;
            }
            template<typename _Tp >
            complex<long double> &operator /=(const complex<_Tp> &__z)
            {
                _ComplexT __t;
                __real__ __t = __z.real();
                __imag__ __t = __z.imag();
                _M_value /= __t;
                return *this;
            }
            const _ComplexT &__rep() const
            {
                return _M_value;
            }
        private :
            _ComplexT _M_value;
    };
    inline complex<float>::complex(const complex<double> &__z)
        : _M_value(__z.__rep()) 
    {
    }
    inline complex<float>::complex(const complex<long double> &__z)
        : _M_value(__z.__rep()) 
    {
    }
    inline complex<double>::complex(const complex<long double> &__z)
        : _M_value(__z.__rep()) 
    {
    }
    extern template istream &operator >>(istream &, complex<float> &);
    extern template ostream &operator <<(ostream &, const complex<float> &);
    extern template istream &operator >>(istream &, complex<double> &);
    extern template ostream &operator <<(ostream &, const complex<double> &);
    extern template istream &operator >>(istream &, complex<long double> &);
    extern template ostream &operator <<(ostream &, const complex<long double> &);
    extern template wistream &operator >>(wistream &, complex<float> &);
    extern template wostream &operator <<(wostream &, const complex<float> &);
    extern template wistream &operator >>(wistream &, complex<double> &);
    extern template wostream &operator <<(wostream &, const complex<double> &);
    extern template wistream &operator >>(wistream &, complex<long double> &);
    extern template wostream &operator <<(wostream &, const complex<long double> &);
}
namespace __gnu_cxx __attribute__((__visibility__("default"))) {
    template<typename _Tp, typename _Up >
    struct __promote_2<std::complex<_Tp>, _Up>
    {
        public :
            typedef std::complex<typename __promote_2<_Tp, _Up>::__type> __type;
    };
    template<typename _Tp, typename _Up >
    struct __promote_2<_Tp, std::complex<_Up> >
    {
        public :
            typedef std::complex<typename __promote_2<_Tp, _Up>::__type> __type;
    };
    template<typename _Tp, typename _Up >
    struct __promote_2<std::complex<_Tp>, std::complex<_Up> >
    {
        public :
            typedef std::complex<typename __promote_2<_Tp, _Up>::__type> __type;
    };
}
static const double h_a = cos(0.02);
static const double h_b = sin(0.02);
void calculate_borders(int coord, int dim, int *start, int *end, int *inner_start, int *inner_end, int length, int halo);
void print_complex_matrix(std::string filename, float *matrix_real, float *matrix_imag, size_t stride, size_t width, size_t height);
void print_matrix(std::string filename, float *matrix, size_t stride, size_t width, size_t height);
void init_p(float *p_real, float *p_imag, int start_x, int end_x, int start_y, int end_y);
void memcpy2D(void *dst, size_t dstride, const void *src, size_t sstride, size_t width, size_t height);
void get_quadrant_sample(const float *r00, const float *r01, const float *r10, const float *r11, const float *i00, const float *i01, const float *i10, const float *i11, size_t src_stride, size_t dest_stride, size_t x, size_t y, size_t width, size_t height, float *dest_real, float *dest_imag);
class ITrotterKernel
{
    public :
        virtual void run_kernel()  = 0;
        virtual void run_kernel_on_halo()  = 0;
        virtual void wait_for_completion()  = 0;
        virtual void get_sample(size_t dest_stride, size_t x, size_t y, size_t width, size_t height, float *dest_real, float *dest_imag) const  = 0;
        virtual bool runs_in_place() const  = 0;
        virtual std::string get_name() const  = 0;
        virtual void initialize_MPI(MPI_Comm cartcomm, int _start_x, int _inner_end_x, int _start_y, int _inner_start_y, int _inner_end_y)  = 0;
        virtual void start_halo_exchange()  = 0;
        virtual void finish_halo_exchange()  = 0;
};
class CPUBlock : public ITrotterKernel
{
    public :
        CPUBlock(float *p_real, float *p_imag, float a, float b, size_t tile_width, size_t tile_height, int halo_x, int halo_y);
        ~CPUBlock();
        void run_kernel();
        void run_kernel_on_halo();
        void wait_for_completion();
        void copy_results();
        void get_sample(size_t dest_stride, size_t x, size_t y, size_t width, size_t height, float *dest_real, float *dest_imag) const;
        bool runs_in_place() const
        {
            return false;
        }
        std::string get_name() const
        {
            std::stringstream name;
            name << "OpenMP block kernel (" << omp_get_max_threads() << " threads)";
            return name.str();
        }
        ;
        void initialize_MPI(MPI_Comm cartcomm, int _start_x, int _inner_end_x, int _start_y, int _inner_start_y, int _inner_end_y);
        void start_halo_exchange();
        void finish_halo_exchange();
    private :
        void kernel8(const float *p_real, const float *p_imag, float *next_real, float *next_imag);
        void process_band(size_t, size_t, size_t, size_t, float, float, const float *, const float *, float *, float *, int, int);
        void process_sides(size_t read_y, size_t read_height, size_t write_offset, size_t write_height, float a, float b, const float *p_real, const float *p_imag, float *next_real, float *next_imag, float *block_real, float *block_imag);
        float *orig_real;
        float *orig_imag;
        float *p_real[2];
        float *p_imag[2];
        float a;
        float b;
        int sense;
        int halo_x, halo_y, tile_width, tile_height;
        static const size_t block_width  = 128u;
        static const size_t block_height  = 128u;
        MPI_Comm cartcomm;
        int neighbors[4];
        int start_x, inner_end_x, start_y, inner_start_y, inner_end_y;
        MPI_Request req[8];
        MPI_Status statuses[8];
        MPI_Datatype horizontalBorder, verticalBorder;
};
void trotter(const int matrix_width, const int matrix_height, const int iterations, const int snapshots, const int kernel_type)
{
    float *p_real, *p_imag;
    std::stringstream filename;
    int start_x, end_x, inner_start_x, inner_end_x, start_y, end_y, inner_start_y, inner_end_y;
    MPI_Comm cartcomm;
    int coords[2], dims[2] = {
        0,
        0
    };
    int periods[2] = {
        0,
        0
    };
    int rank;
    int nProcs;
    MPI_Comm_size(((MPI_Comm) ((void *) &(ompi_mpi_comm_world))), &nProcs);
    MPI_Dims_create(nProcs, 2, dims);
    MPI_Cart_create(((MPI_Comm) ((void *) &(ompi_mpi_comm_world))), 2, dims, periods, 0, &cartcomm);
    MPI_Comm_rank(cartcomm, &rank);
    MPI_Cart_coords(cartcomm, rank, 2, coords);
    int halo_x = (kernel_type == 2 ? 3 : 4);
    int halo_y = 4;
    calculate_borders(coords[1], dims[1], &start_x, &end_x, &inner_start_x, &inner_end_x, matrix_width, halo_x);
    calculate_borders(coords[0], dims[0], &start_y, &end_y, &inner_start_y, &inner_end_y, matrix_height, halo_y);
    int width = end_x - start_x;
    int height = end_y - start_y;
    p_real = new float [width * height];
    p_imag = new float [width * height];
    init_p(p_real, p_imag, start_x, end_x, start_y, end_y);
    ITrotterKernel *kernel;
    switch (kernel_type)
    {
        case 0 : 
        kernel = new CPUBlock (p_real, p_imag, h_a, h_b, width, height, halo_x, halo_y);
        break;
        case 1 : 
        break;
        case 2 : 
        if (coords[0] == 0 && coords[1] == 0)
        {
            std::cerr << "Compiled without CUDA\n";
        }
        MPI_Abort(((MPI_Comm) ((void *) &(ompi_mpi_comm_world))), 2);
        break;
        default : 
        kernel = new CPUBlock (p_real, p_imag, h_a, h_b, width, height, halo_x, halo_y);
    }
    kernel->initialize_MPI(cartcomm, start_x, inner_end_x, start_y, inner_start_y, inner_end_y);
    struct timeval start, end;
    gettimeofday(&start, __null);
    for (int i = 0;
        i < iterations;
        i++)
    {
        if ((snapshots > 0) && (i % snapshots == 0))
        {
            kernel->get_sample(width, 0, 0, width, height, p_real, p_imag);
            filename.str("");
            filename << i << "-iter-" << coords[1] << "-" << coords[0] << "-real.dat";
            print_matrix(filename.str(), p_real + ((inner_start_y - start_y) * width + inner_start_x - start_x), width, inner_end_x - inner_start_x, inner_end_y - inner_start_y);
        }
        kernel->run_kernel_on_halo();
        if (i != iterations - 1)
        {
            kernel->start_halo_exchange();
        }
        kernel->run_kernel();
        if (i != iterations - 1)
        {
            kernel->finish_halo_exchange();
        }
        kernel->wait_for_completion();
    }
    gettimeofday(&end, __null);
    if (coords[0] == 0 && coords[1] == 0)
    {
        long time = (end.tv_sec - start.tv_sec) * 1000000 + (end.tv_usec - start.tv_usec);
        std::cout << matrix_width << "x" << matrix_height << " " << kernel->get_name() << " " << matrix_width * matrix_height << " " << time << std::endl;
    }
    delete[] p_real;
    delete[] p_imag;
    delete kernel;
}
int main(int argc, char **argv)
{
    int dim = 640, iterations = 1000, snapshots = 0, kernel_type = 0;
    int rank;
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(((MPI_Comm) ((void *) &(ompi_mpi_comm_world))), &rank);
    trotter(dim, dim, iterations, snapshots, kernel_type);
    MPI_Finalize();
    return 0;
}
__attribute__((weak, section("nanos_init"))) nanos_init_desc_t __section__nanos_init = {
    nanos_omp_set_interface,
    (void *) 0
};
