#!/usr/bin/env python3
"""The style guards (the blank-line, naming, macro and raw-array rules), run by
.githooks/pre-commit over the STAGED blobs of first-party C++ files:

  1. Blank-line rule (G1): the check_control_blank_lines.py violations, so
     the pre-commit hook mechanically enforces the blank line before a
     multi-line control statement and after its closing brace.
  2. snake_case scan (G6): any project identifier with an underscore beyond
     the sanctioned `_camelCase` member prefix. Library/vendor identifiers
     (std, CUDA, Boost, Eigen, amgcl, spdlog, stdexec, MKL, ...) live in
     ALLOWED_TOKENS below - the allowlist is the curated union of what the
     tree legitimately uses, grown deliberately when a new library token
     appears.
  3. Macro audit (G6): only the sanctioned `Qcx*` macros may be `#define`d
     in source - plus include guards and the two QCX_ERI_CUDA_* inclusion
     markers (the eri_cuda.cu dual-inclusion seam: the .cu is included as a
     header by eri_cuda.cpp and by eri_dispatch_gen.hpp, so its guard and
     header-only switch are file-inclusion mechanics, not policy macros).
     CMake-side Qcx* compile definitions (QcxBasisDataDir, QcxIntegralsFp16)
     are sanctioned by review in the CMakeLists, outside this source-level
     audit.
  4. Raw-array-in-public-headers scan (G6): no raw `type name[N]`
     declarations in `include/qcx/` headers (the G2 policy: std::array /
     std::span; raw arrays only where ABIs mandate them - CPUID, CUDA
     kernels).
  5. Internal-linkage guard (check_internal_linkage.py): a free function
     DECLARED in a public header must not be DEFINED with internal linkage
     (anonymous namespace, or `static`). A static library does not
     link-check, so this defect compiles green in the author's build and
     surfaces as LNK2019 in whoever links next - see the module docstring
     there for what it flags and what it deliberately declines.

Guards 1 and 2 scan the STAGED blobs of the files this commit touches. Guard
5 pairs a declaration with a definition, so it reads the COMMITTED VIEW of
every tracked public header and source (the index, never the working tree) -
and it does so only when the commit touches that surface at all. The staged
(index) copy is verified, not the working tree, matching the clang-format
check in the hook. Not part of the build.
"""

import re
import subprocess
import sys

from check_control_blank_lines import violations as blank_line_violations
from check_internal_linkage import linkage_violations

CPP_EXTENSIONS = (".cpp", ".hpp", ".h", ".cu", ".cc", ".cxx", ".inl")

# The sanctioned Qcx* macros (limits.hpp QcxIntegralsLMax, eri_cuda.hpp
# QcxIntegralsCudaLMax, boys_impl.hpp QcxBoysForceInline - the last is the
# compiler-abstraction inline hint: __forceinline is MSVC-only, GCC/Clang
# spell the same intent always_inline). BoysForceInline is the same hint as
# it is spelled in a standalone copy of that code, which uses its own macro
# prefix and carries no Qcx tokens of its own.
SANCTIONED_MACROS = frozenset({
    "QcxIntegralsLMax", "QcxIntegralsCudaLMax", "QcxBoysForceInline",
    "BoysForceInline",
    # The architecture test (backend/include/qcx/backend/cpu_features.hpp):
    # 1 on an x86-64 target, 0 elsewhere, read from the compiler's own macros
    # (__x86_64__ / _M_X64). It is a public header's define rather than a
    # CMake one on purpose - a private target-level definition does not reach
    # a consumer of that header - and the CMake probe that decides which
    # per-file target features are legal derives its answer from the same two
    # macros, so the flag set and these guards cannot drift apart.
    "QcxArchX86_64",
    # The slice C shim's public status contract (boys_c.h): C consumers
    # expect all-caps status macros, so the four BOYS_* tokens are the
    # header's own naming convention, not project snake_case.
    "BOYS_SUCCESS", "BOYS_ERROR_INVALID_ARGUMENT",
    "BOYS_ERROR_UNSUPPORTED_MULTIPLIER", "BOYS_MAX_ORDER",
})

# The eri_cuda.cu dual-inclusion seam (see the module docstring).
TU_INCLUSION_MACROS = frozenset({"QCX_ERI_CUDA_HEADER_ONLY", "QCX_ERI_CUDA_FILE_INCLUDED"})

# The snake_case scan allowlist: std / CUDA / Boost / Eigen / amgcl / spdlog /
# stdexec / MKL / SYCL / OpenMP identifiers the tree legitimately uses (library
# names are not project naming). Grows deliberately when a new library token
# appears.
ALLOWED_TOKENS = frozenset({
    "HOST_VM_INFO64",
    "HOST_VM_INFO64_COUNT",
    # macOS/POSIX API names in the platform memory probe (Apple's, not ours)
    "KERN_SUCCESS",
    "cuda_backend",
    "free_count",
    "host_info64_t",
    "host_statistics64",
    "mach_host",
    "mach_host_self",
    "mach_msg_type_number_t",
    "memory_topology",
    "vm_statistics",
    "vm_statistics64_data_t",
    # macOS vm_statistics64 field names (Apple's, not ours)
    "inactive_count",
    "speculative_count",
    "ADD_FAILURE", "ASSERT_EQ", "ASSERT_FALSE", "ASSERT_GE", "ASSERT_GT",
    "ASSERT_LE", "ASSERT_LT", "ASSERT_NE", "ASSERT_NEAR", "ASSERT_TRUE", "BENCHMARK_MAIN", "EXPECT_DOUBLE_EQ",
    "EXPECT_EQ", "EXPECT_FALSE", "EXPECT_FLOAT_EQ", "EXPECT_GE",
    "EXPECT_GT", "EXPECT_LE", "EXPECT_LT", "EXPECT_NE", "EXPECT_NEAR",
    "EXPECT_STREQ", "EXPECT_TRUE", "GTEST_SKIP", "SCOPED_TRACE", "and_then", "as_array", "initializer_list",
    "is_table", "lock_guard", "node_view", "none_of",
    # The OpenMP runtime's entry points (omp.h). ABI names, not project
    # naming: the backend pins the runtime's thread maximum through
    # omp_set_num_threads and reads it back through omp_get_max_threads.
    "omp_get_max_threads", "omp_set_num_threads",
    "parse_error",
    "BOYS_MAX_ORDER", "BOYS_SUCCESS", "BOYS_ERROR_INVALID_ARGUMENT",
    "BOYS_ERROR_UNSUPPORTED_MULTIPLIER",
    "ALL_PROCESSOR_GROUPS", "add_edge", "adjacency_list", "align_val_t",
    "all_of", "any_of",
    "bad_alloc", "bfloat16_t", "binary_search", "bit_cast",
    "c_str", "cblas_dgemm", "cblas_dgemm_batch", "CBLAS_TRANSPOSE", "coarse_enough",
    "const_cast", "CP_UTF8",
    "CREATE_NO_WINDOW", "ERROR_ACCESS_DENIED", "ERROR_NOT_SUPPORTED", "WAIT_TIMEOUT",
    # The POSIX half of the same harness: fork and waitpid return a pid_t
    # (sys/types.h). A library ABI type, like int32_t or hsize_t above.
    "pid_t",
    "istreambuf_iterator",
    "JOBOBJECT_EXTENDED_LIMIT_INFORMATION", "JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE",
    "JOB_OBJECT_LIMIT_PROCESS_MEMORY", "MAX_PATH",
    "PROCESS_INFORMATION", "PROCESS_MEMORY_COUNTERS", "PROCESS_MEMORY_COUNTERS_EX",
    "read_symlink", "replace_extension", "to_wstring", "wchar_t",
    "compare_exchange_weak", "condition_variable", "conditional_t",
    "convertible_to", "count_if",
    "cublasHandle_t",
    "cublasStatus_t",
    "CUBLAS_OP_N", "CUBLAS_STATUS_EXECUTION_FAILED", "CUBLAS_STATUS_SUCCESS",
    "cudaError_t", "cudaEvent_t", "cudaStream_t", "default_selector_v",
    "directory_iterator",
    "duration_cast", "emplace_back", "error_code", "fetch_add", "fetch_sub",
    "file_size",
    "filesystem_error",
    "find_first_not_of", "find_first_of", "find_last_not_of",
    "FmEval_Chebyshev7",
    "find_if", "float16_t", "format_string_t", "from_chars", "get_scheduler",
    "hardware_concurrency", "has_value", "hsize_t", "int32_t", "int64_t", "int8_t", "INT_MAX",
    "is_array", "is_object",
    "is_boolean", "is_copy_assignable_v",
    "is_copy_constructible_v", "is_directory", "is_integer",
    "is_move_assignable_v",
    "is_move_constructible_v", "is_nothrow_move_constructible_v",
    "is_number", "is_number_integer", "is_open", "is_regular_file", "is_same_v", "is_sorted",
    "is_string", "is_trivially_copyable_v",
    "libint2_static_cleanup", "libint2_static_init",
    "lower_bound", "sregex_iterator", "stable_sort", "starts_with",
    "make_move_iterator", "make_pair", "make_shared", "make_solver",
    "make_unique", "max_align_t", "max_element", "memory_order_acquire", "memory_order_relaxed",
    "memory_order_release", "min_element", "MKL_COL_MAJOR", "MKL_INT",
    # The C++ standard attribute: a language token, not a project identifier,
    # and the generated-kernel prune emits it (the `[[maybe_unused]]` on the
    # skeletons' unused constants). Added deliberately, per the docstring's
    # rule that the allowlist grows when a legitimate token appears.
    "maybe_unused",
    "MKL_JIT_SUCCESS", "MKL_NOTRANS", "mkl_jit_create_dgemm",
    "_putenv_s",
    "tv_sec", "tv_usec", "ULARGE_INTEGER",
    "mkl_jit_destroy", "mkl_jit_get_dgemm_ptr", "mkl_jit_status_t",
    "mt19937_64", "notify_all", "num_edges", "numeric_limits", "parent_path",
    "perform_op", "pop_back",
    "ptrdiff_t", "push_back", "quiet_NaN", "read_raw", "reference_wrapper",
    # The <regex> matcher entry points, beside directory_iterator and
    # istreambuf_iterator above: std library names, not project naming. The
    # schema-help test matches the parser's own refusal shape and scans its
    # read sites, so it calls both.
    "regex_match",
    "reinterpret_cast", "resize_file",
    "RUSAGE_SELF", "ru_stime", "ru_utime",
    "same_as", "set_pattern", "shared_ptr", "size_t", "sleep_for",
    "smoothed_aggregation",
    "static_assert", "static_cast", "static_thread_pool", "steady_clock",
    "string_view", "sync_wait", "system_error", "temp_directory_path",
    "this_thread", "thread_local",
    "SYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX", "time_point", "to_array",
    "to_string", "uint16_t", "uint32_t", "uint64_t",
    "uint8_t", "uintmax_t", "uintptr_t", "uniform_int_distribution",
    "uniform_real_distribution", "unique_lock", "unique_ptr", "unordered_map",
    "unordered_set",
    "value_or", "value_type", "write_raw",
})

# The violation classes: classic snake_case identifiers, underscore-initial
# members (a second underscore after the leading one - sanctioned _camelCase
# has exactly one), and kConstant-class tokens with a snake underscore.
SNAKE_TOKEN = re.compile(
    r"\b[a-z][a-zA-Z0-9]*_[a-zA-Z0-9_]*\b"
    r"|\b_[a-z][a-zA-Z0-9]*_[a-zA-Z0-9_]*\b"
    r"|\b[A-Z][a-zA-Z0-9]*_[a-zA-Z0-9_]*\b"
)
DEFINE = re.compile(r"^\s*#\s*define\s+([A-Za-z_]\w*)")
IFNDEF = re.compile(r"^\s*#\s*ifndef\s+([A-Za-z_]\w*)")
RAW_ARRAY_DECL = re.compile(r"\b(?:const\s+)?[A-Za-z_][A-Za-z0-9_:<>,]*\s+"
                            r"[A-Za-z_]\w*\s*(?:\[\s*[^\[\]]+\s*\])+\s*[=;,)&]")
COMMENT = re.compile(r"//[^\n]*")
BLOCK_COMMENT = re.compile(r"/\*.*?\*/", re.S)
# C++ raw string literals (R"(...)", R"delim(...)delim"): must be stripped
# before STRING_LIT, whose quote-pair pattern cannot represent them - it
# would span quote to quote across newlines and leave dangling identifier
# fragments of the string's content (false snake_case positives).
RAW_STRING_LIT = re.compile(r'[rR]"([A-Za-z0-9_/]{0,16})\(.*?\)\1"', re.S)
STRING_LIT = re.compile(r'"(?:[^"\\]|\\.)*"')
CHAR_LIT = re.compile(r"'(?:[^'\\]|\\.)*'")
PREPROC_LINE = re.compile(r"^[ \t]*#[^\n]*$", re.M)


def staged_cpp_files() -> list:
    out = subprocess.run(
        ["git", "diff", "--cached", "--name-only", "--diff-filter=ACMR", "-z"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    return [name for name in out.split("\0") if name and name.endswith(CPP_EXTENSIONS)]


def staged_blob(path: str) -> str:
    out = subprocess.run(
        ["git", "cat-file", "blob", f":{path}"], check=True, capture_output=True
    )
    return out.stdout.decode("utf-8", errors="replace")


def code_only(text: str) -> str:
    text = COMMENT.sub("", text)
    text = BLOCK_COMMENT.sub("", text)
    text = RAW_STRING_LIT.sub("", text)
    text = STRING_LIT.sub("", text)
    text = CHAR_LIT.sub("", text)
    text = PREPROC_LINE.sub("", text)
    return text


def snake_case_violations(text: str) -> list:
    found = []
    for i, line in enumerate(code_only(text).split("\n"), 1):
        for token in SNAKE_TOKEN.findall(line):
            if token.startswith("_mm"):
                continue  # Intel intrinsics: library tokens, never project
                # naming (leading underscores in project names are forbidden)
            if token not in ALLOWED_TOKENS:
                found.append((i, f"snake_case identifier: {token}"))
    return found


def macro_violations(text: str) -> list:
    found = []
    pending_guards = []
    for i, line in enumerate(text.split("\n"), 1):
        ifndef = IFNDEF.match(line)
        if ifndef:
            pending_guards.append(ifndef.group(1))
            continue
        define = DEFINE.match(line)
        if not define:
            continue
        name = define.group(1)
        if name in SANCTIONED_MACROS or name in TU_INCLUSION_MACROS:
            continue
        if pending_guards and pending_guards[-1] == name:
            pending_guards.pop()  # include guard
            continue
        found.append((i, f"unsanctioned macro definition: {name}"))
    return found


def raw_array_violations(text: str) -> list:
    found = []
    for i, line in enumerate(code_only(text).split("\n"), 1):
        if RAW_ARRAY_DECL.search(line):
            found.append((i, f"raw array declaration: {line.strip()}"))
    return found


def main() -> int:
    bad = 0
    for path in staged_cpp_files():
        text = staged_blob(path)
        for line_number, description in blank_line_violations(text):
            print(f"{path}:{line_number}: {description}")
            bad += 1
        for line_number, description in snake_case_violations(text):
            print(f"{path}:{line_number}: {description}")
            bad += 1
        for line_number, description in macro_violations(text):
            print(f"{path}:{line_number}: {description}")
            bad += 1
        if "/include/qcx/" in path:
            for line_number, description in raw_array_violations(text):
                print(f"{path}:{line_number}: {description}")
                bad += 1
    # Guard 5 is tree-wide rather than per-staged-file: it pairs a header
    # declaration with a source definition, and either side may be untouched by
    # this commit. It does its own index read and its own skip.
    for message in linkage_violations():
        print(message)
        bad += 1
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
