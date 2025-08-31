cmake_minimum_required(VERSION 3.16)
project(tesseract_decoder LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

include(FetchContent)
find_package(Threads REQUIRED)

# === External dependencies ===
# Stim
FetchContent_Declare(
  stim
  GIT_REPOSITORY https://github.com/quantumlib/stim.git
  GIT_TAG bd60b73525fd5a9b30839020eb7554ad369e4337
)
FetchContent_MakeAvailable(stim)

# HiGHS
FetchContent_Declare(
  highs
  URL https://github.com/ERGO-Code/HiGHS/archive/refs/tags/v1.9.0.tar.gz
  URL_HASH SHA256=dff575df08d88583c109702c7c5c75ff6e51611e6eacca8b5b3fdfba8ecc2cb4
)
FetchContent_MakeAvailable(highs)

# argparse (header only)
FetchContent_Declare(
  argparse
  URL https://github.com/p-ranav/argparse/archive/refs/tags/v3.1.zip
  URL_HASH SHA256=3e5a59ab7688dcd1f918bc92051a10564113d4f36c3bbed3ef596c25e519a062
)
FetchContent_MakeAvailable(argparse)

# nlohmann_json (header only)
FetchContent_Declare(
  nlohmann_json
  URL https://github.com/nlohmann/json/archive/9cca280a4d0ccf0c08f47a99aa71d1b0e52f8d03.zip
)
FetchContent_MakeAvailable(nlohmann_json)

# Boost headers
FetchContent_Declare(
  boost
  URL https://archives.boost.io/release/1.83.0/source/boost_1_83_0.tar.gz
  URL_HASH SHA256=c0685b68dd44cc46574cce86c4e17c0f611b15e195be9848dfd0769a0a207628
)
FetchContent_MakeAvailable(boost)
add_library(boost_headers INTERFACE)
target_include_directories(boost_headers INTERFACE ${boost_SOURCE_DIR})

# pybind11
FetchContent_Declare(
  pybind11
  GIT_REPOSITORY https://github.com/pybind/pybind11.git
  GIT_TAG v2.11.1
)
set(PYBIND11_TEST OFF CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(pybind11)

# Google Test
FetchContent_Declare(
  googletest
  GIT_REPOSITORY https://github.com/google/googletest.git
  GIT_TAG release-1.12.1
)
FetchContent_MakeAvailable(googletest)


set(OPT_COPTS -Ofast -fno-fast-math -march=native)

set(TESSERACT_SRC_DIR ${CMAKE_CURRENT_SOURCE_DIR}/src)

# === Libraries ===
add_library(common ${TESSERACT_SRC_DIR}/common.cc ${TESSERACT_SRC_DIR}/common.h)
target_include_directories(common PUBLIC ${TESSERACT_SRC_DIR})
target_compile_options(common PRIVATE ${OPT_COPTS})
target_link_libraries(common PUBLIC libstim Threads::Threads)

add_library(utils ${TESSERACT_SRC_DIR}/utils.cc ${TESSERACT_SRC_DIR}/utils.h)
target_include_directories(utils PUBLIC ${TESSERACT_SRC_DIR})
target_compile_options(utils PRIVATE ${OPT_COPTS})
target_link_libraries(utils PUBLIC common libstim Threads::Threads)

add_library(visualization ${TESSERACT_SRC_DIR}/visualization.cc ${TESSERACT_SRC_DIR}/visualization.h)
target_include_directories(visualization PUBLIC ${TESSERACT_SRC_DIR})
target_compile_options(visualization PRIVATE ${OPT_COPTS})
target_link_libraries(visualization PUBLIC common boost_headers)

add_library(tesseract_lib ${TESSERACT_SRC_DIR}/tesseract.cc ${TESSERACT_SRC_DIR}/tesseract.h)
target_include_directories(tesseract_lib PUBLIC ${TESSERACT_SRC_DIR})
target_compile_options(tesseract_lib PRIVATE ${OPT_COPTS})
target_link_libraries(tesseract_lib PUBLIC utils boost_headers visualization)

add_library(simplex ${TESSERACT_SRC_DIR}/simplex.cc ${TESSERACT_SRC_DIR}/simplex.h)
target_include_directories(simplex PUBLIC ${TESSERACT_SRC_DIR})
target_compile_options(simplex PRIVATE ${OPT_COPTS})
target_link_libraries(simplex PUBLIC common utils tesseract_lib highs libstim Threads::Threads)

# === Executables ===
add_executable(tesseract ${TESSERACT_SRC_DIR}/tesseract_main.cc)
target_compile_options(tesseract PRIVATE ${OPT_COPTS})
target_link_libraries(tesseract PRIVATE tesseract_lib argparse::argparse nlohmann_json::nlohmann_json)

add_executable(simplex_bin ${TESSERACT_SRC_DIR}/simplex_main.cc)
set_target_properties(simplex_bin PROPERTIES OUTPUT_NAME simplex)
target_compile_options(simplex_bin PRIVATE ${OPT_COPTS})
target_link_libraries(simplex_bin PRIVATE common simplex argparse::argparse nlohmann_json::nlohmann_json)

# === Python module ===
pybind11_add_module(tesseract_decoder MODULE ${TESSERACT_SRC_DIR}/tesseract.pybind.cc)
target_compile_options(tesseract_decoder PRIVATE ${OPT_COPTS})
target_include_directories(tesseract_decoder PRIVATE ${TESSERACT_SRC_DIR})
target_link_libraries(tesseract_decoder PRIVATE common utils simplex tesseract_lib)
set_target_properties(tesseract_decoder PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/src
    LIBRARY_OUTPUT_DIRECTORY_DEBUG ${PROJECT_SOURCE_DIR}/src
    LIBRARY_OUTPUT_DIRECTORY_RELEASE ${PROJECT_SOURCE_DIR}/src
    LIBRARY_OUTPUT_DIRECTORY_MINSIZEREL ${PROJECT_SOURCE_DIR}/src
    LIBRARY_OUTPUT_DIRECTORY_RELWITHDEBINFO ${PROJECT_SOURCE_DIR}/src
)

# === Tests ===
enable_testing()

add_executable(common_test ${TESSERACT_SRC_DIR}/common.test.cc)
target_link_libraries(common_test PRIVATE common GTest::gtest_main)
add_test(NAME common_test COMMAND common_test)

add_executable(tesseract_test ${TESSERACT_SRC_DIR}/tesseract.test.cc)
target_link_libraries(tesseract_test PRIVATE tesseract_lib simplex GTest::gtest_main)
add_test(NAME tesseract_test COMMAND tesseract_test)

