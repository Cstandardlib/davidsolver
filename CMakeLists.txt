cmake_minimum_required(VERSION 3.14)

project(davidsolver VERSION 1.0
                  DESCRIPTION "Very primitive project"
                  LANGUAGES CXX)

set(default_build_type "Debug")

# 设置源文件列表
set(SOURCES main.cpp)

# 添加子目录
add_subdirectory(davidsolver)
include_directories(davidsolver)
add_subdirectory(fast_matrix_market)
include_directories(fast_matrix_market/include)

# 查找 OpenBLAS 库
find_package(BLAS REQUIRED)

# 查找 LAPACK 库
find_package(LAPACK REQUIRED)


# 添加可执行文件
add_executable(main main.cpp)

# 启用 AddressSanitizer
target_compile_options(main PRIVATE -fsanitize=address -fno-omit-frame-pointer)
target_link_libraries(main PRIVATE -fsanitize=address -fno-omit-frame-pointer)

target_link_libraries(main PRIVATE davidsolver ${OpenBLAS_LIBRARIES} ${LAPACK_LIBRARIES})

