#include "davidsolver/module_hsolver/kernels/math_kernel_op.h" // gemm_op
#include "davidsolver/module_base/module_device/memory_op.h" // synchronize_memory_op
#include "davidsolver/diago_david.h" // DiagoDavid

#include "davidsolver/utils.h" // print vector and matrix

#include <fast_matrix_market/fast_matrix_market.hpp>

#include <fstream>

#include <complex>
#include <iostream>
#include <random> // for random init of eigenvectors
#include <memory> // smart pointers

// #define _DEBUG_SMALL_MATRIX

using T = std::complex<double>;// double;
using Real = double;

struct array_matrix {
    int64_t nrows = 0, ncols = 0;
    std::vector<T> vals;       // or int64_t, float, std::complex<double>, etc.
} mat;

int main(int argc, char **argv) {

#ifdef __MPI
    const hsolver::diag_comm_info comm_info = {MPI_COMM_WORLD, 0, 1};
#else
    const hsolver::diag_comm_info comm_info = {0, 1};
#endif

    // 构造H矩阵
#ifndef _DEBUG_SMALL_MATRIX
    // read matrix
    // 打开文件
    // std::ifstream file("matrix/Na5/Na5.mtx", std::ios::binary);
    std::ifstream file("matrix/Si2/Si2.mtx", std::ios::binary);
    if (!file) {
        std::cerr << "Unable to open file" << std::endl;
        return 1;
    }
    fast_matrix_market::read_matrix_market_array(
                file,
                mat.nrows, mat.ncols,
                mat.vals,
                fast_matrix_market::row_major);
    std::cout << "size of matrix: " << mat.nrows << " x " << mat.ncols << std::endl;
    std::cout << "number of total elements: " << mat.vals.size() << std::endl;
    if(mat.nrows != mat.ncols) {
        std::cerr << "Matrix is not square" << std::endl;
        return 1;
    }
    std::vector<T>& h_mat = mat.vals;
#else
    mat.nrows=5;
#endif

    const int dim = mat.nrows;//mat.nrows; //25;
    const int nband = 10; //5;
    const int david_ndim = 4; //4;
    const bool use_paw = false;
#ifdef _DEBUG_SMALL_MATRIX
    std::vector<T> h_mat(dim * dim, T(0.0));
    // 构造 H 矩阵
// std::vector<std::complex<double>> h_mat(25 * 25, std::complex<double>(0.0, 0.0));
// 填充对角线元素
// for (int i = 0; i < dim; ++i) {
//     h_mat[i * dim + i] = T(1.0, 0.0); // 一个示例值
// }
// 填充上三角部分
// for (int i = 1; i < dim; ++i) {
//     for (int j = 0; j < i; ++j) {
//         T value = T(i, j);
//         h_mat[i * dim + j] = value;
//         h_mat[j * dim + i] = std::conj(value);
//     }
// }
    // 填充对角线元素为1+i
// #ifdef _DEBUG_SMALL_MATRIX
//     for (int i = 0; i < dim; ++i) {
//         h_mat[i * dim + i] = T(i+1.0);
//     }
// #endif
    // for (int i = 1; i < dim; ++i) {
    //     for (int j = 0; j < i; ++j) {
    //         T random_value = T(
    //             static_cast<double>(rand() % dim), // 随机实部
    //             static_cast<double>(rand() % dim)  // 随机虚部
    //         );
    //         h_mat[i * dim + j] = random_value;
    //         h_mat[j * dim + i] = std::conj(random_value);
    //     }
    // }
    std::cout << "h_mat = "<< std::endl; printVector(h_mat, dim);
#endif

    std::vector<Real> precondition(dim, 1.0);
#ifdef _DEBUG_SMALL_MATRIX
    std::cout << "precondition = "<< std::endl; printVector(precondition, dim);
#endif
    hsolver::DiagoDavid<T, base_device::DEVICE_CPU> dav(
        precondition.data(),
        nband, dim,
        david_ndim,
        use_paw, comm_info);


    auto hpsi_func = [h_mat](T *hpsi_out, T *psi_in,
                             const int nband_in, const int nbasis_in,
                             const int band_index1, const int band_index2) {
        auto one = std::make_unique<T>(1.0);
        auto zero = std::make_unique<T>(0.0);

        // 智能指针自动转换为原始指针，所以可以直接传递给函数
        const T *one_ = one.get();
        const T *zero_ = zero.get();

        base_device::DEVICE_CPU *ctx = {};

        hsolver::gemm_op<T, base_device::DEVICE_CPU>()(
            ctx, 'N', 'N', nbasis_in, band_index2 - band_index1 + 1, nbasis_in, one_,
            h_mat.data(),
            nbasis_in, psi_in + band_index1 * nbasis_in, nbasis_in,
            zero_, hpsi_out + band_index1 * nbasis_in, nbasis_in);
    };

    auto spsi_func = [](T* X, T* SX, const int nrow, const int npw, const int nbands) {
        // copy X to SX of size nbands * nrow
        // for (int i = 0; i < nbands; ++i) {
        //     for (int j = 0; j < nrow; ++j) {
        //         SX[i * nrow + j] = X[i * nrow + j];
        //     }
        // }
        base_device::DEVICE_CPU *ctx = {};
        base_device::memory::synchronize_memory_op<T, base_device::DEVICE_CPU, base_device::DEVICE_CPU>()(ctx, ctx, SX, X, nbands * nrow);
    };

    std::vector<T> psi(dim * nband, 0.0);
    // 创建一个随机数生成器
    std::random_device rd;  // 非确定性随机数种子
    std::mt19937 gen(rd()); // 以 rd() 为种子初始化 Mersenne Twister 引擎
    // 定义一个均匀分布的随机数分布范围
    std::uniform_real_distribution<> dis(0.0, 1.0); // [0.0, 1.0) 范围内的随机浮点数

    // 使用随机数填充 vector
    for (auto& elem : psi) {
        elem = dis(gen); // 生成随机数并赋值给 vector 中的元素
    }
    for (int i = 0; i < dim; ++i) {
        for (int j = 0; j < nband; ++j) {
            psi[j * dim + i] = T(j==i);
        }
    }

    // 输出初始化的向量
    // for (int i = 0; i < psi.size(); ++i) {
    //     std::cout << psi[i].real() << " ";
    //     if ((i + 1) % dim == 0) std::cout << std::endl; // 每 dim 个元素后换行
    // }
#ifdef _DEBUG_SMALL_MATRIX
    std::cout << "initial eigenvectors:" << std::endl; printVector(psi, dim);
#endif
    
    std::vector<Real> eigenvalue(nband, 1.0);

    const Real david_diag_thr = 1e-2;
    const int david_maxiter = 100;

    int sum_iter = dav.diag(hpsi_func, spsi_func,
                        dim, psi.data(), 
                        eigenvalue.data(),
                        david_diag_thr, david_maxiter);

    std::cout << "sum_iter: " << sum_iter << std::endl;
    std::cout << "eigenvalues:" << std::endl;
    // printArray(eigenvalue.data(), nband); //printVector(eigenvalue, nband);
    for (int i = 0; i < nband; ++i) {
        std::cout << eigenvalue[i] << " ";
    }
    std::cout << std::endl;
#ifdef _DEBUG_SMALL_MATRIX
    std::cout << "eigenvectors:" << std::endl; printVector(psi, dim);
#endif
    return 0;
}