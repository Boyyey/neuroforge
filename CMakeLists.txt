cmake_minimum_required(VERSION 3.10)
project(neuroforge VERSION 1.0.0 LANGUAGES C CUDA)

# Set C standard
set(CMAKE_C_STANDARD 99)
set(CMAKE_C_STANDARD_REQUIRED ON)

# Options
option(USE_CUDA "Enable CUDA support" ON)
option(BUILD_EXAMPLES "Build example programs" ON)
option(BUILD_TESTS "Build test programs" ON)
option(BUILD_SHARED "Build shared library" OFF)

# Find required packages
find_package(OpenMP)
find_package(BLAS)

# Set compiler flags
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -Wall -Wextra -O3")
if(OpenMP_C_FOUND)
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
endif()

# Include directories
include_directories(src)

# Source files
file(GLOB_RECURSE SOURCES 
    "src/*.c"
    "src/layers/*.c"
    "src/optimizers/*.c"
    "src/activations/*.c"
)

# CUDA sources
if(USE_CUDA AND CMAKE_CUDA_COMPILER)
    file(GLOB_RECURSE CUDA_SOURCES "src/cuda/*.cu")
    set(SOURCES ${SOURCES} ${CUDA_SOURCES})
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -arch=sm_70 -O3")
    enable_language(CUDA)
endif()

# Create library
if(BUILD_SHARED)
    add_library(neuroforge SHARED ${SOURCES})
    set_target_properties(neuroforge PROPERTIES
        VERSION ${PROJECT_VERSION}
        SOVERSION ${PROJECT_VERSION_MAJOR}
        OUTPUT_NAME "neuroforge"
    )
else()
    add_library(neuroforge STATIC ${SOURCES})
    set_target_properties(neuroforge PROPERTIES
        OUTPUT_NAME "neuroforge"
    )
endif()

# Link libraries
target_link_libraries(neuroforge m)
if(OpenMP_C_FOUND)
    target_link_libraries(neuroforge OpenMP::OpenMP_C)
endif()
if(BLAS_FOUND)
    target_link_libraries(neuroforge ${BLAS_LIBRARIES})
endif()

# Examples
if(BUILD_EXAMPLES)
    file(GLOB EXAMPLES "examples/*.c")
    foreach(EXAMPLE ${EXAMPLES})
        get_filename_component(EXAMPLE_NAME ${EXAMPLE} NAME_WE)
        add_executable(${EXAMPLE_NAME} ${EXAMPLE})
        target_link_libraries(${EXAMPLE_NAME} neuroforge)
    endforeach()
endif()

# Tests
if(BUILD_TESTS)
    file(GLOB TESTS "tests/*.c")
    foreach(TEST ${TESTS})
        get_filename_component(TEST_NAME ${TEST} NAME_WE)
        add_executable(${TEST_NAME} ${TEST})
        target_link_libraries(${TEST_NAME} neuroforge)
    endforeach()
    
    # Enable testing
    enable_testing()
    foreach(TEST ${TESTS})
        get_filename_component(TEST_NAME ${TEST} NAME_WE)
        add_test(NAME ${TEST_NAME} COMMAND ${TEST_NAME})
    endforeach()
endif()

# Installation
install(TARGETS neuroforge
    EXPORT neuroforgeTargets
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib
    RUNTIME DESTINATION bin
)

install(DIRECTORY src/
    DESTINATION include/neuroforge
    FILES_MATCHING PATTERN "*.h"
)

install(EXPORT neuroforgeTargets
    FILE neuroforgeTargets.cmake
    NAMESPACE neuroforge::
    DESTINATION lib/cmake/neuroforge
)

# Create config file
include(CMakePackageConfigHelpers)
write_basic_package_version_file(
    "neuroforgeConfigVersion.cmake"
    VERSION ${PROJECT_VERSION}
    COMPATIBILITY AnyNewerVersion
)

configure_package_config_file(
    "cmake/neuroforgeConfig.cmake.in"
    "neuroforgeConfig.cmake"
    INSTALL_DESTINATION lib/cmake/neuroforge
)

install(FILES
    "${CMAKE_CURRENT_BINARY_DIR}/neuroforgeConfig.cmake"
    "${CMAKE_CURRENT_BINARY_DIR}/neuroforgeConfigVersion.cmake"
    DESTINATION lib/cmake/neuroforge
)

# Create pkg-config file
configure_file(
    "neuroforge.pc.in"
    "neuroforge.pc"
    @ONLY
)
install(FILES "${CMAKE_CURRENT_BINARY_DIR}/neuroforge.pc"
    DESTINATION lib/pkgconfig
)
