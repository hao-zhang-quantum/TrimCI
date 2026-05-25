# TrimCI/cpp/trimci_core/CMakeLists.txt
cmake_minimum_required(VERSION 3.18)
project(trimci_core LANGUAGES CXX)
# Enforce C++17 for features like if constexpr and std::is_same_v
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Use LLVM's libc++ on macOS when compiling with LLVM clang
# This is needed because LLVM clang may generate symbols not present in Apple's libc++
if(APPLE AND CMAKE_CXX_COMPILER_ID MATCHES "Clang")
    # Check if we're using Homebrew LLVM
    execute_process(
        COMMAND ${CMAKE_CXX_COMPILER} --version
        OUTPUT_VARIABLE CXX_COMPILER_VERSION
        OUTPUT_STRIP_TRAILING_WHITESPACE
    )
    if(CXX_COMPILER_VERSION MATCHES "Homebrew")
        message(STATUS "Detected Homebrew LLVM, using LLVM libc++")
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -stdlib=libc++")
        # Link to Homebrew LLVM's libc++
        link_directories(/opt/homebrew/opt/llvm/lib/c++)
        set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -L/opt/homebrew/opt/llvm/lib/c++ -Wl,-rpath,/opt/homebrew/opt/llvm/lib/c++")
        set(CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS} -L/opt/homebrew/opt/llvm/lib/c++ -Wl,-rpath,/opt/homebrew/opt/llvm/lib/c++")
    endif()
endif()

find_package(pybind11 CONFIG REQUIRED)

# 1) Find Eigen3: prefer vendored third_party/eigen; then system; finally FetchContent
# Try local vendored copy first
set(EIGEN_LOCAL_DIR "${CMAKE_SOURCE_DIR}/../../third_party/eigen")
if(EXISTS "${EIGEN_LOCAL_DIR}/Eigen/Core")
    message(STATUS "Using vendored Eigen at ${EIGEN_LOCAL_DIR}")
    set(EIGEN3_INCLUDE_DIR "${EIGEN_LOCAL_DIR}")
    if(NOT TARGET Eigen3::Eigen)
        add_library(Eigen3::Eigen INTERFACE IMPORTED)
        set_target_properties(Eigen3::Eigen PROPERTIES
            INTERFACE_INCLUDE_DIRECTORIES "${EIGEN3_INCLUDE_DIR}"
        )
    endif()
else()
    # Fall back to system package
    find_package(Eigen3 3.3 QUIET NO_MODULE)
    if(NOT Eigen3_FOUND)
        include(FetchContent)
        FetchContent_Declare(eigen
            GIT_REPOSITORY https://gitlab.com/libeigen/eigen.git
            GIT_TAG 3.4.0
            GIT_SHALLOW TRUE
        )
        FetchContent_MakeAvailable(eigen)
        # Set include dir and provide a compatible imported target only if not already present
        set(EIGEN3_INCLUDE_DIR "${eigen_SOURCE_DIR}")
        if(NOT TARGET Eigen3::Eigen)
            add_library(Eigen3::Eigen INTERFACE IMPORTED)
            set_target_properties(Eigen3::Eigen PROPERTIES
                INTERFACE_INCLUDE_DIRECTORIES "${EIGEN3_INCLUDE_DIR}"
            )
        endif()
        message(STATUS "Fetched Eigen3 headers from ${EIGEN3_INCLUDE_DIR}")
    else()
        # Ensure EIGEN3_INCLUDE_DIR is defined when found via config
        if(NOT EIGEN3_INCLUDE_DIR)
            get_target_property(_eigen_includes Eigen3::Eigen INTERFACE_INCLUDE_DIRECTORIES)
            if(_eigen_includes)
                set(EIGEN3_INCLUDE_DIR "${_eigen_includes}")
            endif()
        endif()
    endif()
endif()

# 2) Optional OpenMP
find_package(OpenMP)

# 3) High-performance hash table options
option(USE_ABSL "Use Abseil flat_hash_map" OFF)
option(USE_ROBIN_HOOD "Use robin_hood unordered_map" ON)

if(USE_ABSL)
    find_package(absl REQUIRED)
    message(STATUS "Using Abseil flat_hash_map for high performance")
elseif(USE_ROBIN_HOOD)
    # Default include dir for robin_hood if none provided
    if(NOT DEFINED ROBIN_HOOD_INCLUDE_DIR AND NOT DEFINED ROBIN_HOOD_INCLUDE_DIRS)
        set(ROBIN_HOOD_INCLUDE_DIR "${CMAKE_SOURCE_DIR}/../../third_party/robin_hood")
    endif()
    if(DEFINED ROBIN_HOOD_INCLUDE_DIR)
        if(EXISTS "${ROBIN_HOOD_INCLUDE_DIR}/robin_hood.h")
            message(STATUS "Using robin_hood unordered_map from: ${ROBIN_HOOD_INCLUDE_DIR}")
        else()
            message(FATAL_ERROR "robin_hood.h not found in: ${ROBIN_HOOD_INCLUDE_DIR}")
        endif()
    else()
        find_package(PkgConfig REQUIRED)
        pkg_check_modules(ROBIN_HOOD REQUIRED robin_hood)
        message(STATUS "Using robin_hood unordered_map for high performance")
    endif()
else()
    message(STATUS "Using std::unordered_map (default)")
endif()

# 4) Build Python extension
pybind11_add_module(trimci_core
    # common/
    common/determinant.cpp
    common/hamiltonian.cpp
    common/spin_operator.cpp
    # trimci/
    trimci/screening.cpp
    trimci/trim.cpp
    trimci/iterative_workflow.cpp
    # solvers/
    solvers/davidson_gep.cpp
    solvers/matfree_davidson.cpp
    solvers/direct_s_builder.cpp
    solvers/direct_h_builder.cpp
    solvers/direct_h_detspace.cpp
    solvers/excitation_generator.cpp
    # rdm/
    rdm/rdm.cpp
    rdm/rdm_low_order.cpp
    rdm/rdm_3body.cpp
    rdm/rdm_4body.cpp
    rdm/rdm_4body_cumulant.cpp
    rdm/rdm_5body.cpp
    rdm/rdm_6body.cpp
    rdm/rdm_transition.cpp
    rdm/rdm_cumulant_element.cpp
    # orbital_opt/
    orbital_opt/orbital_opt.cpp
    # fast_expansion/
    fast_expansion/ab_index.cpp
    fast_expansion/morton_sort.cpp
    fast_expansion/ab_group_matvec.cpp
    fast_expansion/detspace_matvec.cpp
    fast_expansion/detspace_davidson.cpp
    fast_expansion/sparse_update_davidson.cpp
    fast_expansion/expansion_loop.cpp
    fast_expansion/streaming_pt2.cpp
    fast_expansion/enpt3.cpp
    fast_expansion/dressed_ci.cpp
    fast_expansion/dressed_ci_hybrid.cpp
    fast_expansion/ab_group_matvec.cpp
    fast_expansion/detspace_matvec_gpu.cpp  # GPU Davidson matvec (guarded by TRIMCI_HAS_GPU)
    # davidson_utils/
    davidson_utils/davidson_utils.cpp
    # bindings (pybind11)
    bindings/bind_common.cpp
    bindings/bind_trimci.cpp
    bindings/bind_rdm.cpp
    bindings/bind_orbital_opt.cpp
    bindings/bind_workflow.cpp
    bindings/bind_solvers.cpp
    bindings/bind_fast_expansion.cpp
    bindings/bind_davidson_utils.cpp
    module.cpp
)

# Optional: Build scalable determinant example (for testing)
option(BUILD_DETERMINANT_EXAMPLE "Build determinant scalability example" OFF)
if(BUILD_DETERMINANT_EXAMPLE)
    add_executable(determinant_example determinant_usage_example.cpp)
    target_include_directories(determinant_example PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
endif()

# 5) Include directories
target_include_directories(trimci_core PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/common
    ${CMAKE_CURRENT_SOURCE_DIR}/trimci
    ${CMAKE_CURRENT_SOURCE_DIR}/solvers
    ${CMAKE_CURRENT_SOURCE_DIR}/rdm
    ${CMAKE_CURRENT_SOURCE_DIR}/orbital_opt
    ${CMAKE_CURRENT_SOURCE_DIR}/fast_expansion
    ${CMAKE_CURRENT_SOURCE_DIR}/davidson_utils
    ${CMAKE_CURRENT_SOURCE_DIR}/bindings
    ${EIGEN3_INCLUDE_DIR}
)

# 6) Hash table compile definitions and linking
if(USE_ABSL)
    target_compile_definitions(trimci_core PRIVATE USE_ABSL)
    target_link_libraries(trimci_core PRIVATE absl::flat_hash_map)
elseif(USE_ROBIN_HOOD)
    target_compile_definitions(trimci_core PRIVATE USE_ROBIN_HOOD)
    if(DEFINED ROBIN_HOOD_INCLUDE_DIR)
        target_include_directories(trimci_core PRIVATE ${ROBIN_HOOD_INCLUDE_DIR})
    else()
        target_include_directories(trimci_core PRIVATE ${ROBIN_HOOD_INCLUDE_DIRS})
    endif()
endif()

# 7) OpenMP support
if(OpenMP_CXX_FOUND)
    message(STATUS "OpenMP found: enabling parallel screening")
    target_compile_options(trimci_core PRIVATE ${OpenMP_CXX_FLAGS})
    target_link_libraries(trimci_core PRIVATE OpenMP::OpenMP_CXX)
endif()

# 7.5) Architecture-specific optimizations
# Use TRIMCI_MARCH to override -march (e.g. -DTRIMCI_MARCH=x86-64-v3 for portability).
# Default: "native" (best performance on build host, but NOT portable across CPU types).
set(TRIMCI_MARCH "native" CACHE STRING "Target -march for trimci_core (native, x86-64-v3, etc.)")
include(CheckCXXCompilerFlag)
check_cxx_compiler_flag("-march=${TRIMCI_MARCH}" COMPILER_SUPPORTS_MARCH)
if(COMPILER_SUPPORTS_MARCH)
    target_compile_options(trimci_core PRIVATE -march=${TRIMCI_MARCH})
    message(STATUS "Enabled -march=${TRIMCI_MARCH} for architecture optimizations")
endif()

# 8) Link third-party libraries
target_link_libraries(trimci_core PRIVATE
    Eigen3::Eigen
)

# 8.5) Enable BLAS backend for Eigen on macOS (Apple Accelerate)
if(APPLE)
    find_library(ACCELERATE_FRAMEWORK Accelerate)
    if(ACCELERATE_FRAMEWORK)
        message(STATUS "Enabling Eigen BLAS backend with Apple Accelerate")
        target_compile_definitions(trimci_core PRIVATE EIGEN_USE_BLAS)
        target_link_libraries(trimci_core PRIVATE ${ACCELERATE_FRAMEWORK})
    endif()
endif()

# 9) Install .so/.pyd into package directory
install(TARGETS trimci_core DESTINATION trimci)
