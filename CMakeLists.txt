cmake_minimum_required(VERSION 3.10)
project(lab2 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

include_directories(${CMAKE_SOURCE_DIR}/include)

set(SRC
    src/main.cpp
    src/kobbelt.cpp
    src/long_accumulator.cpp
    src/sorting.cpp
    src/pichat.cpp
    src/fma.cpp
    
    src/merge.cpp
)

add_executable(main ${SRC})

add_executable(dot_product_tests
    tests/tests.cpp
    src/kobbelt.cpp
    src/long_accumulator.cpp
    src/sorting.cpp
    src/pichat.cpp
    src/fma.cpp

    src/merge.cpp
)

target_include_directories(dot_product_tests PRIVATE ${CMAKE_SOURCE_DIR}/include)
target_link_libraries(dot_product_tests PRIVATE gmp gmpxx)

enable_testing()
add_test(NAME run_dot_product_tests COMMAND dot_product_tests)

add_custom_target(run_tests
    COMMAND ${CMAKE_CTEST_COMMAND} --output-on-failure
    DEPENDS dot_product_tests
)

# Альтернативная цель для запуска тестового бинаря напрямую
add_custom_target(tests
    COMMAND dot_product_tests
    DEPENDS dot_product_tests
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
)
