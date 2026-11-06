# The Fock contraction's baseline copy, compiled in a directory of its own.
#
# It is the one first-party unit that must NOT inherit the build's
# instruction-set tier: a processor without AVX2 has to run the fallback the
# dispatcher selects for it. The tier is stated at the root as directory-wide
# compile options so that no target can miss it, and CMake offers no
# per-source way to remove one - a later /arch flag on the command line
# changes the codegen but, measured on MSVC, leaves __AVX2__ defined, so the
# unit cannot even report its own tier honestly. A directory, however, starts
# from a copy of its parent's COMPILE_OPTIONS list and may replace it: that
# is the one mechanism that subtracts for a subtree, and it is why this unit
# lives here. Everything the unit needs is restated below, and the baseline
# flags are added by the parent's tier block for a baseline configure and by
# the else() branch of the same block (none) - in an avx2 configure this
# directory adds nothing, because nothing was inherited.
#
# The objects join qcx-integrals through $<TARGET_OBJECTS:...>, and in the
# shared configuration they are registered for the composed qcx target, the
# same way the MD class shards are (the parent object library's
# $<TARGET_OBJECTS> does not recurse into a nested one, measured on CMake
# 4.4).
set_property(DIRECTORY PROPERTY COMPILE_OPTIONS "")

add_library(qcx-fock-scalar OBJECT fock_contract_scalar.cpp)

if (QCX_BUILD_SHARED)
    set_property(GLOBAL APPEND PROPERTY QCX_SHARED_NESTED_OBJECT_LIBRARIES
        qcx-fock-scalar)
endif()

target_compile_features(qcx-fock-scalar PRIVATE cxx_std_23)
target_include_directories(qcx-fock-scalar PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/../include
    ${CMAKE_CURRENT_SOURCE_DIR}/../src)
# The usage requirements of the linked targets reach the object TU's compile
# (include dirs, definitions); nothing links from the object library itself.
target_link_libraries(qcx-fock-scalar PRIVATE
    qcx-core qcx-molecule qcx-basisset qcx-memory eigen)

# The reset above removes the tier, and with it every other directory-wide
# option the root had stated by the time it added this directory. Those are
# restated here, and only they: the ASan pair and /FS from the root's early
# blocks, its /W4, and the tier block's contraction pin, which the unit needs
# at baseline for the same reason every other unit does (the source
# arithmetic is the two-rounding one at either tier; MSVC never contracts
# under its default /fp:precise, so only the GNU spelling appears). A flag
# added to the root's directory options later must be considered here too;
# the tree does not add any between those blocks and this directory.
if (QCX_ASAN)
    target_compile_options(qcx-fock-scalar PRIVATE /fsanitize=address)
    target_compile_options(qcx-fock-scalar PRIVATE $<$<COMPILE_LANGUAGE:CXX>:/Zi>)
endif()
if (MSVC)
    target_compile_options(qcx-fock-scalar PRIVATE $<$<COMPILE_LANGUAGE:CXX>:/FS>)
    target_compile_options(qcx-fock-scalar PRIVATE $<$<COMPILE_LANGUAGE:CXX>:/W4>)
else()
    target_compile_options(qcx-fock-scalar PRIVATE -ffp-contract=off)
endif()

# No precompiled header: the target is one translation unit, and a PCH would
# compile the same headers a second time for it.
