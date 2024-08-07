cmake_minimum_required(VERSION 3.12)

project(davidsolver VERSION 1.0
                  DESCRIPTION "Very primitive project"
                  LANGUAGES CXX)

# 设置源文件列表
set(SOURCES main.cpp)

# 添加子目录
add_subdirectory(davidsolver)

# 查找 OpenBLAS 库
find_package(BLAS REQUIRED)

# 查找 LAPACK 库
find_package(LAPACK REQUIRED)

# 添加可执行文件
add_executable(main main.cpp)

target_link_libraries(main davidsolver ${OpenBLAS_LIBRARIES} ${LAPACK_LIBRARIES})