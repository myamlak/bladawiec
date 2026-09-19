add_library(qcx-memory
  src/allocation_instrument.cpp
  src/device_buffer.cpp
  src/first_touch.cpp
)
target_include_directories(qcx-memory PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_compile_features(qcx-memory PUBLIC cxx_std_23)
target_link_libraries(qcx-memory PUBLIC qcx-core qcx-backend)
target_precompile_headers(qcx-memory PRIVATE
    <qcx/memory/device_buffer.hpp>)

if (QCX_ENABLE_CUDA)
    target_link_libraries(qcx-memory PUBLIC qcx-backend-cuda)
endif()

if (QCX_BUILD_TESTS AND TARGET gtest)
    add_executable(qcx-memory-tests
      tests/allocation_instrument_test.cpp
      tests/device_buffer_test.cpp
      tests/tensor_test.cpp
      tests/e2e_overlap_test.cpp
      tests/workspace_budget_test.cpp
      tests/device_workspace_budget_test.cpp
      tests/first_touch_test.cpp
    )
    target_link_libraries(qcx-memory-tests PRIVATE qcx-memory qcx-backend gtest gtest_main)
    target_precompile_headers(qcx-memory-tests PRIVATE
        <gtest/gtest.h>
        <qcx/memory/tensor.hpp>
        <qcx/memory/workspace_budget.hpp>
        <qcx/memory/device_workspace_budget.hpp>)

    qcx_apply_asan_runtime(qcx-memory-tests)

    include(GoogleTest)
    gtest_discover_tests(qcx-memory-tests PROPERTIES TIMEOUT ${QCX_TEST_TIMEOUT_SECONDS})

    if (QCX_ENABLE_CUDA)
        # Host-side C++23 tests (kernels live in backend's .cu); need the
        # CUDA toolkit headers and cudart, which CUDA targets get implicitly.
        add_executable(qcx-memory-cuda-tests
          tests/device_buffer_cuda_test.cpp
          tests/tensor_cuda_test.cpp
        )
        target_include_directories(qcx-memory-cuda-tests PRIVATE ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIR})
        target_link_libraries(qcx-memory-cuda-tests PRIVATE qcx-memory qcx-backend-cuda CUDA::cudart_static gtest gtest_main)
        target_precompile_headers(qcx-memory-cuda-tests PRIVATE
            <gtest/gtest.h>
            <cuda_runtime.h>)
        qcx_apply_asan_runtime(qcx-memory-cuda-tests)
        gtest_discover_tests(qcx-memory-cuda-tests PROPERTIES TIMEOUT ${QCX_TEST_TIMEOUT_SECONDS})
    endif()
endif()
