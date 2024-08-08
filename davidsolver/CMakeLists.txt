# 添加包含目录
include_directories(${CMAKE_SOURCE_DIR}/davidsolver)

# 设置源文件列表
set(DAVID_SOURCES
    diag_const_nums.cpp
    diago_david.cpp

    module_base/parallel_reduce.cpp
    
    module_base/module_device/device.cpp
    module_base/module_device/memory_op.cpp
    
    # module_hsolver/kernels/dngvd_op.cpp
    module_hsolver/kernels/math_kernel_op.cpp
    module_hsolver/kernels/dngvd_op.cpp
)

# 添加库
add_library(davidsolver ${DAVID_SOURCES})

# 添加子目录
add_subdirectory(module_base)
add_subdirectory(module_hsolver)