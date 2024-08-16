#ifndef UTILS_H
#define UTILS_H

#include <iostream>
#include <vector>
#include <complex>

#define __LARGE_MATRIX

// #include <Eigen/Dense>

// template<typename T>
// void printVector(const std::vector<T>& vec) {
//     std::cout << "Vector: ";
//     for (const auto& element : vec) {
//         std::cout << element << " ";
//     }
//     std::cout << std::endl;
// }

// template<typename T>
// void printVector(const std::vector<std::complex<T>>& vec) {
//     std::cout << "Vector: ";
//     for (const auto& element : vec) {
//         std::cout << element.real() << " ";
//     }
//     std::cout << std::endl;
// }

 
/**
 * @brief Prints the elements of a vector in a formatted manner.
 * 
 * This function prints each element followed by a space,
 * and after every 'dim' elements, it inserts a new line.
 * It can be used to print a matrix stored in std::vector.
 * 
 * @tparam T The type of elements in the vector.
 * @param vec The vector to be printed.
 * @param dim The number of elements after which a new line should be inserted.
 */
template <typename T>
void printVector(const std::vector<T>& vec, const int dim){
#ifdef __LARGE_MATRIX
    return;
#endif
    for (int i = 0; i < vec.size(); ++i) {
        std::cout << vec[i] << " ";
        if ((i + 1) % dim == 0) std::cout << std::endl; // 每 dim 个元素后换行
    }
}

template <typename T>
void printVector(const std::vector<std::complex<T>>& vec, const int dim){
#ifdef __LARGE_MATRIX
    return;
#endif
    for (int i = 0; i < vec.size(); ++i) {
        std::cout << vec[i].real() << " ";
        if ((i + 1) % dim == 0) std::cout << std::endl; // 每 dim 个元素后换行
    }
}

template <typename T>
void printArray(const T *array, const int dim){
#ifdef __LARGE_MATRIX
    return;
#endif
    for (int i = 0; i < dim; ++i) {
        std::cout << array[i] << " ";
    }
    std::cout << std::endl;
}

template <typename T>
void printArray(const std::complex<T> *array, const int dim){
#ifdef __LARGE_MATRIX
    return;
#endif
    for (int i = 0; i < dim; ++i) {
        std::cout << array[i].real() << " ";
    }
    std::cout << std::endl;
}

// template<typename T>
// void printMatrix(const Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>& matrix) {
//     std::cout << "Matrix:" << std::endl;
//     std::cout << matrix << std::endl;
// }

#endif // UTILS_H