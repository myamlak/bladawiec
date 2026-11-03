# Generate the composed qcx shared library's export list (MSVC only).
#
# Inputs: QCX_LINKER (link.exe, used in its dumpbin role), QCX_OBJECT_LIST (a
# file with one object path per line), QCX_DEF_OUT (the .def to write).
#
# WHY NOT AUTO-EXPORT. CMake's WINDOWS_EXPORT_ALL_SYMBOLS emits every external
# symbol of the target's objects. On this tree that is 68,096 names, of which
# 36,453 are vendored Eigen inline-template instantiations and most of the rest
# are the same kind of instantiation from the other vendored headers; the
# import library then exceeds MSVC's 65,535-member limit and the link dies with
# LNK1189. Those instantiations are not a consumer's API - a consumer compiles
# its own copy from the headers - so the export list carries the first-party
# `qcx::` symbols only. The list is regenerated from the target's own objects
# on every build, so it cannot drift from the code.
#
# The symbol dump is written to a file and filtered natively by file(STRINGS):
# the per-class MD objects carry tens of thousands of sections each, and a
# CMake loop over every dump line is minutes of scripting where the native
# filter is seconds.
if (NOT EXISTS "${QCX_OBJECT_LIST}")
  message(FATAL_ERROR "gen_qcx_def: object list missing: ${QCX_OBJECT_LIST}")
endif()

file(STRINGS "${QCX_OBJECT_LIST}" _qcxObjects)
set(_qcxNames "")
set(_qcxDumpFile "${QCX_DEF_OUT}.dump")

foreach(_qcxObject IN LISTS _qcxObjects)
  if (NOT _qcxObject)
    continue()
  endif()

  execute_process(
    COMMAND "${QCX_LINKER}" -dump -symbols "${_qcxObject}"
    OUTPUT_FILE "${_qcxDumpFile}"
    ERROR_QUIET
    RESULT_VARIABLE _qcxDumpResult)

  if (NOT _qcxDumpResult EQUAL 0)
    message(FATAL_ERROR "gen_qcx_def: link -dump failed on ${_qcxObject}: ${_qcxDumpResult}")
  endif()

  # Only the external symbols can be exported; file(STRINGS) does the filter
  # before the script sees a line.
  file(STRINGS "${_qcxDumpFile}" _qcxLines REGEX "External")

  foreach(_qcxLine IN LISTS _qcxLines)
    # An undefined external has no definition to export.
    if (_qcxLine MATCHES "UNDEF")
      continue()
    endif()

    if (NOT _qcxLine MATCHES "\\| +([^ ]+)")
      continue()
    endif()

    set(_qcxName "${CMAKE_MATCH_1}")

    # First-party symbols only: the mangled form carries the namespace token.
    if (NOT _qcxName MATCHES "qcx@")
      continue()
    endif()

    # A symbol line with an argument list is a function; otherwise it is data,
    # which the .def must mark so the linker exports the storage and not a
    # code thunk.
    if (_qcxLine MATCHES "notype +\\(")
      list(APPEND _qcxNames "${_qcxName}")
    else()
      list(APPEND _qcxNames "${_qcxName} DATA")
    endif()
  endforeach()
endforeach()

file(REMOVE "${_qcxDumpFile}")

list(REMOVE_DUPLICATES _qcxNames)
list(SORT _qcxNames)

file(WRITE "${QCX_DEF_OUT}" "EXPORTS\n")
foreach(_qcxName IN LISTS _qcxNames)
  file(APPEND "${QCX_DEF_OUT}" "\t${_qcxName}\n")
endforeach()

list(LENGTH _qcxNames _qcxNameCount)
message(STATUS "gen_qcx_def: ${_qcxNameCount} first-party exports -> ${QCX_DEF_OUT}")
