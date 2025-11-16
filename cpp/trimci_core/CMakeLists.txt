# TrimCI/cpp/trimci_core/CMakeLists.txt
cmake_minimum_required(VERSION 3.18)
project(trimci_core LANGUAGES CXX)
# Enforce C++17 for features like if constexpr and std::is_same_v
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
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
    determinant.cpp
    hamiltonian.cpp
    screening.cpp
    trim.cpp
    pybind_wrapper.cpp
    hamiltonian_scalable.cpp
    screening_scalable.cpp
    trim_scalable.cpp
    pybind_wrapper_scalable.cpp
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

# 8) Link third-party libraries
target_link_libraries(trimci_core PRIVATE
    Eigen3::Eigen
)

# 9) Install .so/.pyd into package directory
install(TARGETS trimci_core DESTINATION trimci)
