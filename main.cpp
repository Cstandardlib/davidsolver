#include "davidsolver/module_hsolver/kernels/math_kernel_op.h" // gemm_op
#include "davidsolver/module_base/module_device/memory_op.h" // synchronize_memory_op
#include "davidsolver/diago_david.h" // DiagoDavid

#include "davidsolver/utils.h" // print vector and matrix

#include <complex>
#include <iostream>
#include <random> // for random init of eigenvectors
#include <memory> // smart pointers

using T = std::complex<double>;
using Real = double;

int main(int argc, char **argv) {

#ifdef __MPI
    const hsolver::diag_comm_info comm_info = {MPI_COMM_WORLD, 0, 1};
#else
    const hsolver::diag_comm_info comm_info = {0, 1};
#endif

    const int dim = 8; //25;
    const int nband = 2; //5;
    const int david_ndim = 2; //4;
    const bool use_paw = false;

    std::vector<double> precondition(dim, 1.0);
    std::cout << "precondition = "<< std::endl;
    printVector(precondition, dim);

    hsolver::DiagoDavid<T, base_device::DEVICE_CPU> dav(
        precondition.data(),
        nband, dim,
        david_ndim,
        use_paw, comm_info);

    // 构造H矩阵
    std::vector<T> h_mat(dim * dim, T(0.0, 0.0));
    // 填充对角线元素为1+i
    for (int i = 0; i < dim; ++i) {
        h_mat[i * dim + i] = T(i+1.0, 0.0);
    }
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
    // for (int i = 0; i < h_mat.size(); ++i) {
    //     std::cout << h_mat[i].real() << " ";
    //     if ((i + 1) % dim == 0) std::cout << std::endl; // 每 dim 个元素后换行
    // }

    auto hpsi_func = [h_mat](T *hpsi_out, T *psi_in,
                             const int nband_in, const int nbasis_in,
                             const int band_index1, const int band_index2) {
        auto one = std::make_unique<T>(1.0, 0.0);
        auto zero = std::make_unique<T>(0.0, 0.0);

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

    std::vector<std::complex<double>> psi(dim * nband, 0.0);
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
            psi[j * dim + i] = T(j==i, 0.0);
        }
    }

    // 输出初始化的向量
    // for (int i = 0; i < psi.size(); ++i) {
    //     std::cout << psi[i].real() << " ";
    //     if ((i + 1) % dim == 0) std::cout << std::endl; // 每 dim 个元素后换行
    // }
    std::cout << "initial eigenvectors:" << std::endl; printVector(psi, dim);

    
    std::vector<Real> eigenvalue(nband, 1.0);

    const Real david_diag_thr = 1e-2;
    const int david_maxiter = 100;

    int sum_iter = dav.diag(hpsi_func, spsi_func,
                        dim, psi.data(), 
                        eigenvalue.data(),
                        david_diag_thr, david_maxiter);

    std::cout << "sum_iter: " << sum_iter << std::endl;
    std::cout << "eigenvalues:" << std::endl; printVector(eigenvalue, nband);
    std::cout << "eigenvectors:" << std::endl; printVector(psi, dim);

    return 0;
}