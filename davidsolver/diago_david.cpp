#include "diago_david.h"

#include "utils.h"

#include "module_base/module_device/device.h"

#include "module_hsolver/kernels/dngvd_op.h"
#include "module_hsolver/kernels/math_kernel_op.h"

#ifdef USE_PAW
#include "module_cell/module_paw/paw_cell.h"
#endif

using namespace hsolver;


/**
 * @brief Constructor for the DiagoDavid class.
 * 
 * @param[in] precondition_in Pointer to the preconditioning matrix.
 * @param[in] nband_in Number of eigenpairs required(i.e. bands).
 * @param[in] dim_in Dimension of the matrix.
 * @param[in] david_ndim_in Dimension of the reduced basis set of Davidson.
 *                      `david_ndim_in` * `nband_in` is the maximum allowed size of
 *                      the reduced basis set before \b restart of Davidson.
 * @param[in] use_paw_in Flag indicating whether to use PAW.
 * @param[in] diag_comm_in Communication information for diagonalization.
 * 
 * @tparam T The data type of the matrices and arrays.
 * @tparam Device The device type (base_device::DEVICE_CPU or DEVICE_GPU).
 * 
 * @note Auxiliary memory is allocated in the constructor and deallocated in the destructor.
 */
template <typename T, typename Device>
DiagoDavid<T, Device>::DiagoDavid(const Real* precondition_in, 
                                  const int nband_in,
                                  const int dim_in,
                                  const int david_ndim_in,
                                  const bool use_paw_in,
                                  const diag_comm_info& diag_comm_in)
    : nband(nband_in), dim(dim_in), nbase_x(david_ndim_in * nband_in), david_ndim(david_ndim_in), use_paw(use_paw_in), diag_comm(diag_comm_in)
{
    this->device = base_device::get_device_type<Device>(this->ctx);
    this->precondition = precondition_in;

    this->one = &one_;
    this->zero = &zero_;
    this->neg_one = &neg_one_;

    test_david = 2;
    // 1: check which function is called and which step is executed
    // 2: check the eigenvalues of the result of each iteration
    // 3: check the eigenvalues and errors of the last result
    // default: no check


    // set auxiliary memory
    // !!!
    // allocate operaions MUST be paired with deallocate operations
    // in the destructor
    assert(this->david_ndim > 1);
    assert(this->david_ndim * nband < dim * diag_comm.nproc);

    // qianrui change it 2021-7-25.
    // In strictly speaking, it shoule be PW_DIAG_NDIM*nband < npw sum of all pools. We roughly estimate it here.
    // However, in most cases, total number of plane waves should be much larger than nband*PW_DIAG_NDIM

    /// initialize variables
    /// k_first = 0 means that nks is more like a dimension of "basis" to be contracted in "HPsi".In LR-TDDFT the formula writes :
    /// $$\sum_{ jb\mathbf{k}'}A^I_{ia\mathbf{k}, jb\mathbf{k}' }X ^ I_{ jb\mathbf{k}'}$$
    /// In the code :
    /// - "H" means A
    /// - "Psi" means X
    /// - "band" means the superscript I : the number of excited states to be solved
    /// - k : k-points, the same meaning as the ground state
    /// - "basis" : number of occupied ks-orbitals(subscripts i,j) * number of unoccupied ks-orbitals(subscripts a,b), corresponding to "bands" of the ground state

    // the lowest N eigenvalues
    base_device::memory::resize_memory_op<Real, base_device::DEVICE_CPU>()(
                        this->cpu_ctx, this->eigenvalue, nbase_x, "DAV::eig");
    base_device::memory::set_memory_op<Real, base_device::DEVICE_CPU>()(
                        this->cpu_ctx, this->eigenvalue, 0, nbase_x);

    // basis(dim, nbase_x), leading dimension = dim
    resmem_complex_op()(this->ctx, basis, nbase_x * dim, "DAV::basis");
    setmem_complex_op()(this->ctx, basis, 0, nbase_x * dim);

    //<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
    // hpsi(nbase_x, dim); // the product of H and psi in the reduced basis set
    resmem_complex_op()(this->ctx, this->hpsi, nbase_x * dim, "DAV::hpsi");
    setmem_complex_op()(this->ctx, this->hpsi, 0, nbase_x * dim);

    // spsi(nbase_x, dim); // the Product of S and psi in the reduced basis set
    resmem_complex_op()(this->ctx, this->spsi, nbase_x * dim, "DAV::spsi");
    setmem_complex_op()(this->ctx, this->spsi, 0, nbase_x * dim);

    // hcc(nbase_x, nbase_x); // Hamiltonian on the reduced basis
    resmem_complex_op()(this->ctx, this->hcc, nbase_x * nbase_x, "DAV::hcc");
    setmem_complex_op()(this->ctx, this->hcc, 0, nbase_x * nbase_x);

    // scc(nbase_x, nbase_x); // Overlap on the reduced basis
    resmem_complex_op()(this->ctx, this->scc, nbase_x * nbase_x, "DAV::scc");
    setmem_complex_op()(this->ctx, this->scc, 0, nbase_x * nbase_x);

    // vcc(nbase_x, nbase_x); // Eigenvectors of hcc
    resmem_complex_op()(this->ctx, this->vcc, nbase_x * nbase_x, "DAV::vcc");
    setmem_complex_op()(this->ctx, this->vcc, 0, nbase_x * nbase_x);
    //<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
    
    // lagrange_matrix(nband, nband); // for orthogonalization
    resmem_complex_op()(this->ctx, this->lagrange_matrix, nband * nband);
    setmem_complex_op()(this->ctx, this->lagrange_matrix, 0, nband * nband);
}

/**
 * @brief Destructor for the DiagoDavid class.
 * 
 * This destructor releases the dynamically allocated memory used by the class members.
 * It deletes the basis, hpsi, spsi, hcc, scc, vcc, lagrange_matrix, and eigenvalue arrays.
 * If the device is a GPU device, it also deletes the d_precondition array.
 * 
 */
template <typename T, typename Device>
DiagoDavid<T, Device>::~DiagoDavid()
{
    delmem_complex_op()(this->ctx, this->basis);
    delmem_complex_op()(this->ctx, this->hpsi);
    delmem_complex_op()(this->ctx, this->spsi);
    delmem_complex_op()(this->ctx, this->hcc);
    delmem_complex_op()(this->ctx, this->scc);
    delmem_complex_op()(this->ctx, this->vcc);
    delmem_complex_op()(this->ctx, this->lagrange_matrix);
    base_device::memory::delete_memory_op<Real, base_device::DEVICE_CPU>()(this->cpu_ctx, this->eigenvalue);

#if defined(__CUDA) || defined(__ROCM)
    if (this->device == base_device::GpuDevice)
    {
        delmem_var_op()(this->ctx, this->d_precondition);
    }
#endif
}

template <typename T, typename Device>
int DiagoDavid<T, Device>::diag_once(const HPsiFunc& hpsi_func,
                                     const SPsiFunc& spsi_func,
                                     const int dim,
                                     const int nband,
                                     const int ldPsi,
                                     T *psi_in,
                                     Real* eigenvalue_in,
                                     const Real david_diag_thr,
                                     const int david_maxiter)
{
    if (test_david == 1)
    {
        // ModuleBase::TITLE("DiagoDavid", "diag_once");
    }
    // ModuleBase::timer::tick("DiagoDavid", "diag_once");

    // convflag[m] = true if the m th band is converged
    std::vector<bool> convflag(nband, false);
    // unconv[m] store the number of the m th unconverged band
    std::vector<int> unconv(nband);

    int nbase = 0; // the dimension of the reduced basis set

    this->notconv = nband; // the number of unconverged eigenvalues

    for (int m = 0; m < nband; m++) {
        unconv[m] = m;
    }

    // ModuleBase::timer::tick("DiagoDavid", "first");
#ifdef DEBUG
    std::cout << "first start" << std::endl;
#endif

    // orthogonalise the initial trial psi(0~nband-1)

    // plan for SchmidtOrth
    std::vector<int> pre_matrix_mm_m(nband, 0);
    std::vector<int> pre_matrix_mv_m(nband, 1);
    this->planSchmidtOrth(nband, pre_matrix_mm_m, pre_matrix_mv_m);

    for (int m = 0; m < nband; m++)
    {
        if(this->use_paw)
        {
#ifdef USE_PAW
            GlobalC::paw_cell.paw_nl_psi(1, reinterpret_cast<const std::complex<double>*> (psi_in + m*ldPsi),
                reinterpret_cast<std::complex<double>*>(&this->spsi[m * dim]));
#endif
        }
        else
        {
            // phm_in->sPsi(psi_in + m*ldPsi, &this->spsi[m * dim], dim, dim, 1);
#ifdef DEBUG
            // std::cout << "psi_in[" << m << "] = " << std::endl; printArray(psi_in + m*ldPsi, ldPsi);
#endif
            spsi_func(psi_in + m*ldPsi,&this->spsi[m*dim],dim,dim,1);
#ifdef DEBUG
            // std::cout << "spsi[" << m << "] = " << std::endl; printArray(&spsi[m*dim], ldPsi);
#endif
        }
    }
    // begin SchmidtOrth
    for (int m = 0; m < nband; m++)
    {
        syncmem_complex_op()(this->ctx, this->ctx, basis + dim*m, psi_in + m*ldPsi, dim);

        this->SchmidtOrth(dim,
                         nband,
                         m,
                         this->spsi,
                         &this->lagrange_matrix[m * nband],
                         pre_matrix_mm_m[m],
                         pre_matrix_mv_m[m]);
        if(this->use_paw)
        {
#ifdef USE_PAW
            GlobalC::paw_cell.paw_nl_psi(1,reinterpret_cast<const std::complex<double>*> (basis + dim*m),
                reinterpret_cast<std::complex<double>*>(&this->spsi[m * dim]));
#endif
        }
        else
        {
            // phm_in->sPsi(basis + dim*m, &this->spsi[m * dim], dim, dim, 1);
            spsi_func(basis + dim*m, &this->spsi[m * dim], dim, dim, 1);
        }
    }

    // end of SchmidtOrth and calculate H|psi>
    // hpsi_info dav_hpsi_in(&basis, psi::Range(true, 0, 0, nband - 1), this->hpsi);
    // phm_in->ops->hPsi(dav_hpsi_in);
#ifdef DEBUG
    std::cout << "hpsi_func start" << std::endl;
    std::cout << "where: " << __LINE__ << std::endl;
    std::cout << "hpsi = " << std::endl;
    std::vector<T> hpsi_tmp(hpsi, hpsi + nbase_x*dim);
    printVector(hpsi_tmp, dim);
    std::cout << "basis = " << std::endl;
    std::vector<T> basis_tmp(basis, basis + nbase_x*dim);
    printVector(basis_tmp, dim);
#endif
    hpsi_func(this->hpsi, basis, nbase_x, dim, 0, nband - 1);
#ifdef DEBUG
    std::cout << "hpsi_func over" << std::endl;
    std::cout << "where: " << __LINE__ << std::endl;
    std::cout << "hpsi = " << std::endl;
    hpsi_tmp.assign(hpsi, hpsi + nbase_x*dim);
    printVector(hpsi_tmp, dim);
#endif
    this->cal_elem(dim, nbase, nbase_x, this->notconv, this->hpsi, this->spsi, this->hcc, this->scc);

    this->diag_zhegvx(nbase, nband, this->hcc, this->scc, nbase_x, this->eigenvalue, this->vcc);

    for (int m = 0; m < nband; m++)
    {
        eigenvalue_in[m] = this->eigenvalue[m];
    }

    // ModuleBase::timer::tick("DiagoDavid", "first");
#ifdef DEBUG
    std::cout << "first over" << std::endl;
    std::cout << "eigenvalue = " << std::endl;
    printArray(eigenvalue_in, nband);
    // print hcc
    std::cout << "hcc = " << std::endl;
    std::vector<T> hcc_tmp(hcc, hcc + nbase_x*nbase_x);
    printVector(hcc_tmp, nbase_x);
    // print vcc
    std::cout << "vcc = " << std::endl;
    std::vector<T> vcc_tmp(vcc, vcc + nbase_x*nbase_x);
    printVector(vcc_tmp, nbase_x);
#endif

    int dav_iter = 0;
    do
    {
        dav_iter++;
#ifdef DEBUG
        std::cout << "---" << dav_iter << "th iter" << "---" << std::endl;
#endif
#ifdef DEBUG
        std::cout << "<" <<"cal_grad start" << "-" << std::endl;
        std::cout << "where: " << __LINE__ << std::endl;
        std::cout << "hpsi = " << std::endl;
        std::vector<T> hpsi_tmp(hpsi, hpsi + nbase_x*dim);
        printVector(hpsi_tmp, dim);
        std::cout << "spsi = " << std::endl;
        std::vector<T> spsi_tmp(spsi, spsi + nbase_x*dim);
        printVector(spsi_tmp, dim);
#endif
        this->cal_grad(hpsi_func,
                       spsi_func,
                       dim,
                       nbase,
                       nbase_x,
                       this->notconv,
                       this->hpsi,
                       this->spsi,
                       this->vcc,
                       unconv.data(),
                       this->eigenvalue);
#ifdef DEBUG
        std::cout << "-" <<"cal_grad over" << ">" << std::endl;
        std::cout << "where: " << __LINE__ << std::endl;
        std::cout << "<" <<"cal_elem start" << "-" << std::endl;
#endif
        this->cal_elem(dim, nbase, nbase_x, this->notconv, this->hpsi, this->spsi, this->hcc, this->scc);
#ifdef DEBUG
        std::cout << "-" <<"cal_elem over" << ">" << std::endl;
        std::cout << "where: " << __LINE__ << std::endl;
        std::cout << "<" <<"diag_zhegvx start" << "-" << std::endl;
#endif
        this->diag_zhegvx(nbase, nband, this->hcc, this->scc, nbase_x, this->eigenvalue, this->vcc);
#ifdef DEBUG
        std::cout << "-" <<"diag_zhegvx over" << ">" << std::endl;
        std::cout << "where: " << __LINE__ << std::endl;
        std::cout << "<" <<"check_conv start" << "-" << std::endl;
#endif
        // check convergence and update eigenvalues
        // ModuleBase::timer::tick("DiagoDavid", "check_update");

        this->notconv = 0;
        for (int m = 0; m < nband; m++)
        {
            convflag[m] = (std::abs(this->eigenvalue[m] - eigenvalue_in[m]) < david_diag_thr);
            if (!convflag[m])
            {
                unconv[this->notconv] = m;
                this->notconv++;
            }
            eigenvalue_in[m] = this->eigenvalue[m];
        }

        // ModuleBase::timer::tick("DiagoDavid", "check_update");
        if (!this->notconv || (nbase + this->notconv > nbase_x)
            || (dav_iter == david_maxiter))
        {
            // ModuleBase::timer::tick("DiagoDavid", "last");
#ifdef DEBUG
        std::cout << "-" <<"converged or restart or maxiter" << ">" << std::endl;
        std::cout << "where: " << __LINE__ << std::endl;
        std::cout << "---" <<"last" << "---" << std::endl;
#endif
            // update eigenvectors of Hamiltonian

            setmem_complex_op()(this->ctx, psi_in, 0, nband * ldPsi);
            //<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
            gemm_op<T, Device>()(this->ctx,
                                      'N',
                                      'N',
                                      dim,          // m: row of A,C
                                      nband,        // n: col of B,C
                                      nbase,        // k: col of A, row of B
                                      this->one,
                                      basis,       // A dim * nbase
                                      dim,
                                      this->vcc,    // B nbase * nband
                                      nbase_x,
                                      this->zero,
                                      psi_in,       // C dim * nband
                                      ldPsi
            );

            if (!this->notconv || (dav_iter == david_maxiter))
            {
                // overall convergence or last iteration: exit the iteration
                // ModuleBase::timer::tick("DiagoDavid", "last");
                break;
            }
            else
            {
                // if the dimension of the reduced basis set is becoming too large,
                // then replace the first N (=nband) basis vectors with the current
                // estimate of the eigenvectors and set the basis dimension to N;
                this->refresh(dim,
                              nband,
                              nbase,
                              nbase_x,
                              eigenvalue_in,
                              psi_in,
                              ldPsi,
                              this->hpsi,
                              this->spsi,
                              this->hcc,
                              this->scc,
                              this->vcc);
                // ModuleBase::timer::tick("DiagoDavid", "last");
            }

        } // end of if

    } while (true);

    // ModuleBase::timer::tick("DiagoDavid", "diag_once");

    return dav_iter;
}

/**
 * Calculates the preconditioned gradient of the eigenvectors in Davidson method.
 *
 * @param hpsi_func The function to calculate the matrix-blockvector product H * psi.
 * @param spsi_func The function to calculate the matrix-blockvector product overlap S * psi.
 * @param dim The dimension of the blockvector.
 * @param nbase The current dimension of the reduced basis.
 * @param nbase_x The maximum dimension of the reduced basis set.
 * @param notconv The number of unconverged eigenpairs.
 * @param hpsi The output array for the Hamiltonian H times blockvector psi.
 * @param spsi The output array for the overlap matrix S times blockvector psi.
 * @param vcc The input array for the eigenvector coefficients.
 * @param unconv The array of indices for the unconverged eigenpairs.
 * @param eigenvalue The array of eigenvalues.
 */
template <typename T, typename Device>
void DiagoDavid<T, Device>::cal_grad(const HPsiFunc& hpsi_func,
                                        const SPsiFunc& spsi_func,
                                        const int& dim,
                                        const int& nbase,   // current dimension of the reduced basis
                                        const int nbase_x,  // maximum dimension of the reduced basis set
                                        const int& notconv,
                                        T* hpsi,
                                        T* spsi,
                                        const T* vcc,
                                        const int* unconv,
                                        const Real* eigenvalue)
{
    if (test_david == 1) {
        // ModuleBase::TITLE("DiagoDavid", "cal_grad");
    }
    // if all eigenpairs have converged, return
    if (notconv == 0) {
        return;
    }
    // ModuleBase::timer::tick("DiagoDavid", "cal_grad");

    // use template pointer for accelerate
    // std::complex<double>* spsi;
    // std::complex<double>* ppsi;

    // expand the reduced basis set with the new basis vectors P|Real(psi)>...
    // in which psi are the last eigenvectors
    // we define |Real(psi)> as (H-ES)*|Psi>, E = <psi|H|psi>/<psi|S|psi>

    // vc_ev_vector(notconv, nbase);
    // eigenvectors of unconverged index extracted from vcc
    T* vc_ev_vector = nullptr;
    resmem_complex_op()(this->ctx, vc_ev_vector, notconv * nbase);
    setmem_complex_op()(this->ctx, vc_ev_vector, 0, notconv * nbase);

    //>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
    // for (int m = 0; m < notconv; m++)
    // {
    //     for (int i = 0; i < nbase; i++)
    //     {
    //         // vc_ev_vector(m, i) = vc(i, unconv[m]);
    //         vc_ev_vector[m * nbase + i] = vcc[i * nbase_x + unconv[m]];
    //     }
    // }
    //>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

    // set vc_ev_vector = vcc (unconverged index, leading dimension = nbase_x)
    // vc_ev_vector(m, i) = vc(i, unconv[m]);
    //         vc_ev_vector[m * nbase + i] = vcc[i * nbase_x + unconv[m]];
    for (int m = 0; m < notconv; m++)
    {
        syncmem_complex_op()(this->ctx,
                             this->ctx,
                             vc_ev_vector + m * nbase,
                             vcc + unconv[m] * nbase_x,
                             nbase);
    }

    // basis[nbase] = hpsi * vc_ev_vector = hpsi*vcc
    // basis'        =   vc_ev_vector' * hpsi'
    // (dim, notconv)  (dim, nbase) (nbase, notconv)
    gemm_op<T, Device>()(this->ctx,
                              'N',
                              'N',
                              dim, // m: row of A,C
                              notconv, // n: col of B,C
                              nbase, // k: col of A, row of B
                              this->one, // alpha
                              hpsi, // A dim * nbase
                              dim, // LDA: if(N) max(1,m) if(T) max(1,k)
                              vc_ev_vector, // B nbase * notconv
                              nbase, // LDB: if(N) max(1,k) if(T) max(1,n)
                              this->zero, // belta
                              basis + dim*nbase, // C dim * notconv
                              dim // LDC: if(N) max(1, m)
    );

    // for (int m = 0; m < notconv; m++)
    // {
    //     for (int i = 0; i < nbase; i++)
    //     {
    //         vc_ev_vector[m * nbase + i] *= -1 * this->eigenvalue[unconv[m]];
    //     }
    // }

    // e_temp_cpu = {-lambda}
    // vc_ev_vector[nbase] = vc_ev_vector[nbase] * e_temp_cpu
    // now vc_ev_vector[nbase] = - lambda * ev = -lambda * vcc
    for (int m = 0; m < notconv; m++)
    {
        std::vector<Real> e_temp_cpu(nbase, (-1.0 * this->eigenvalue[unconv[m]]));

        if (this->device == base_device::GpuDevice)
        {
#if defined(__CUDA) || defined(__ROCM)
            Real* e_temp_gpu = nullptr;
            resmem_var_op()(this->ctx, e_temp_gpu, nbase);
            syncmem_var_h2d_op()(this->ctx, this->cpu_ctx, e_temp_gpu, e_temp_cpu.data(), nbase);
            vector_mul_vector_op<T, Device>()(this->ctx,
                                                   nbase,
                                                   vc_ev_vector + m * nbase,
                                                   vc_ev_vector + m * nbase,
                                                   e_temp_gpu);
            delmem_var_op()(this->ctx, e_temp_gpu);
#endif
        }
        else
        {
            vector_mul_vector_op<T, Device>()(this->ctx,
                                                   nbase,
                                                   vc_ev_vector + m * nbase,
                                                   vc_ev_vector + m * nbase,
                                                   e_temp_cpu.data());
        }
    }
#ifdef DEBUG
    std::cout << "->" <<"before basis[nbase] = (H - lambda * S) * psi_new in cal_grad" << "<" << std::endl;
    std::cout << "where: " << __LINE__ << std::endl;
    std::cout << "basis = " << std::endl;
    std::vector<T> basis_tmp(basis, basis + nbase_x * dim);
    // basis_tmp.assign(basis, basis + nbase_x * dim);
    printVector(basis_tmp, dim);
    std::cout << "basis[nbase] = " << std::endl;
    basis_tmp.assign(basis + dim*nbase, basis + nbase_x * dim);
    printVector(basis_tmp, dim);
    std::cout << "spsi = " << std::endl;
    std::vector<T> spsi_tmp(spsi, spsi + nbase_x * dim);
    // spsi_tmp.assign(spsi, spsi + nbase_x * dim);
    printVector(spsi_tmp, dim);
    std::cout << "vc_ev_vector = " << std::endl;
    std::vector<T> vc_ev_vector_tmp(vc_ev_vector, vc_ev_vector + nbase * notconv);
    // vc_ev_vector_tmp.assign(vc_ev_vector, vc_ev_vector + nbase * notconv);
    printVector(vc_ev_vector_tmp, nbase);
#endif
    // basis[nbase] = basis[nbase] - spsi * vc_ev_vector
    //              = hpsi - spsi * lambda * vcc
    //              = (H - lambda * S) * psi * vcc
    //              = (H - lambda * S) * psi_new 
    gemm_op<T, Device>()(this->ctx,
                              'N',
                              'N',
                              dim, // m: row of A,C
                              notconv, // n: col of B,C
                              nbase, // k: col of A, row of B
                              this->one, // alpha
                              spsi, // A
                              dim, // LDA: if(N) max(1,m) if(T) max(1,k)
                              vc_ev_vector, // B
                              nbase, // LDB: if(N) max(1,k) if(T) max(1,n)
                              this->one, // belta
                              basis + dim*nbase, // C dim * notconv
                              dim // LDC: if(N) max(1, m)
    );
#ifdef DEBUG
    std::cout << "->" <<"after basis[nbase] = (H - lambda * S) * psi_new in cal_grad" << "<" << std::endl;
    std::cout << "where: " << __LINE__ << std::endl;
    std::cout << "basis = " << std::endl;
    // std::vector<T> basis_tmp(basis, basis + nbase_x * dim);
    basis_tmp.assign(basis, basis + nbase_x * dim);
    printVector(basis_tmp, dim);
    std::cout << "spsi = " << std::endl;
    // std::vector<T> spsi_tmp(spsi, spsi + nbase_x * dim);
    spsi_tmp.assign(spsi, spsi + nbase_x * dim);
    printVector(spsi_tmp, dim);
    std::cout << "vc_ev_vector = " << std::endl;
    // std::vector<T> vc_ev_vector_tmp(vc_ev_vector, vc_ev_vector + nbase * notconv);
    vc_ev_vector_tmp.assign(vc_ev_vector, vc_ev_vector + nbase * notconv);
    printVector(vc_ev_vector_tmp, nbase);
#endif
    // Preconditioning
    // basis[nbase] = T * basis[nbase] = T * (H - lambda * S) * psi
    // where T, the preconditioner, is an approximate inverse of H
    //          T is a diagonal stored in array `precondition`
    // to do preconditioning, divide each column of basis by the corresponding element of precondition
    for (int m = 0; m < notconv; m++)
    {
        if (this->device == base_device::GpuDevice)
        {
#if defined(__CUDA) || defined(__ROCM)
            vector_div_vector_op<T, Device>()(this->ctx,
                                                   dim,
                                                   basis + dim*(nbase + m),
                                                   basis + dim*(nbase + m),
                                                   this->d_precondition);
#endif
        }
        else
        {
            vector_div_vector_op<T, Device>()(this->ctx,
                                                   dim,
                                                   basis + dim*(nbase + m),
                                                   basis + dim*(nbase + m),
                                                   this->precondition);
        }
        // for (int ig = 0; ig < dim; ig++)
        // {
        //     ppsi[ig] /= this->precondition[ig];
        // }
    }

    // there is a nbase to nbase + notconv band orthogonalise
    // plan for SchmidtOrth
    T* lagrange = nullptr;
    resmem_complex_op()(this->ctx, lagrange, notconv * (nbase + notconv));
    setmem_complex_op()(this->ctx, lagrange, 0, notconv * (nbase + notconv));
#ifdef DEBUG
    std::cout << "->" <<"planSchmidtOrth in cal_grad" << "<" << std::endl;
    std::cout << "where: " << __LINE__ << std::endl;
    std::cout << "basis = " << std::endl;
    // std::vector<T> basis_tmp(basis, basis + nbase_x * dim);
    basis_tmp.assign(basis, basis + nbase_x * dim);
    printVector(basis_tmp, dim);
    std::cout << "spsi = " << std::endl;
    // std::vector<T> spsi_tmp(spsi, spsi + nbase_x * dim);
    spsi_tmp.assign(spsi, spsi + nbase_x * dim);
    printVector(spsi_tmp, dim);
#endif
    std::vector<int> pre_matrix_mm_m(notconv, 0);
    std::vector<int> pre_matrix_mv_m(notconv, 1);
    this->planSchmidtOrth(notconv, pre_matrix_mm_m, pre_matrix_mv_m);
    for (int m = 0; m < notconv; m++)
    {
        if(this->use_paw)
        {
#ifdef USE_PAW
            GlobalC::paw_cell.paw_nl_psi(1,reinterpret_cast<const std::complex<double>*> (basis + dim*(nbase + m)),
                reinterpret_cast<std::complex<double>*>(&spsi[(nbase + m) * dim]));
#endif
        }
        else
        {
            // phm_in->sPsi(basis + dim*(nbase + m), &spsi[(nbase + m) * dim], dim, dim, 1);
            spsi_func(basis + dim*(nbase + m), &spsi[(nbase + m) * dim], dim, dim, 1);
        }
    }
    // first nbase bands psi* dot notconv bands spsi to prepare lagrange_matrix

    // calculate the square matrix for future lagranges
    gemm_op<T, Device>()(this->ctx,
                              'C',
                              'N',
                              nbase, // m: row of A,C
                              notconv, // n: col of B,C
                              dim, // k: col of A, row of B
                              this->one, // alpha
                              basis, // A
                              dim, // LDA: if(N) max(1,m) if(T) max(1,k)
                              &spsi[nbase * dim], // B
                              dim, // LDB: if(N) max(1,k) if(T) max(1,n)
                              this->zero, // belta
                              lagrange, // C
                              nbase + notconv // LDC: if(N) max(1, m)
    );
#ifdef DEBUG
    std::cout << "->" <<"just before SchmidtOrth in cal_grad" << "<" << std::endl;
    std::cout << "where: " << __LINE__ << std::endl;
    std::cout << "basis = " << std::endl;
    basis_tmp.assign(basis, basis + nbase_x * dim);
    printVector(basis_tmp, dim);
    std::cout << "spsi = " << std::endl;
    spsi_tmp.assign(spsi, spsi + nbase_x * dim);
    printVector(spsi_tmp, dim);
#endif
    for (int m = 0; m < notconv; m++)
    {
        this->SchmidtOrth(dim,
                         nbase + notconv,
                         nbase + m,
                         spsi,
                         &lagrange[m * (nbase + notconv)],
                         pre_matrix_mm_m[m],
                         pre_matrix_mv_m[m]);
        if(this->use_paw)
        {
#ifdef USE_PAW
            GlobalC::paw_cell.paw_nl_psi(1,reinterpret_cast<const std::complex<double>*> (basis + dim*(nbase + m)),
                reinterpret_cast<std::complex<double>*>(&spsi[(nbase + m) * dim]));
#endif
        }
        else
        {
            // phm_in->sPsi(basis + dim*(nbase + m), &spsi[(nbase + m) * dim], dim, dim, 1);
            spsi_func(basis + dim*(nbase + m), &spsi[(nbase + m) * dim], dim, dim, 1);
        }
    }
#ifdef DEBUG
    std::cout << "->" <<"after SchmidtOrth in cal_elem" << "<" << std::endl;
    std::cout << "where: " << __LINE__ << std::endl;
    std::cout << "basis = " << std::endl;
    basis_tmp.assign(basis, basis + nbase_x * dim);
    printVector(basis_tmp, dim);
    std::cout << "spsi = " << std::endl;
    spsi_tmp.assign(spsi, spsi + nbase_x * dim);
    printVector(spsi_tmp, dim);
#endif
    // calculate H|psi> for not convergence bands
    // hpsi_info dav_hpsi_in(&basis,
    //                       psi::Range(true, 0, nbase, nbase + notconv - 1),
    //                       &hpsi[nbase * dim]); // &hp(nbase, 0)
    // phm_in->ops->hPsi(dav_hpsi_in);
    // hpsi_func(&hpsi[nbase * dim], basis, nbase_x, dim, nbase, nbase + notconv - 1);
    hpsi_func(hpsi, basis, nbase_x, dim, nbase, nbase + notconv - 1);

    delmem_complex_op()(this->ctx, lagrange);
    delmem_complex_op()(this->ctx, vc_ev_vector);

    // ModuleBase::timer::tick("DiagoDavid", "cal_grad");
    return;
}

/**
 * Calculates the elements of the diagonalization matrix for the DiagoDavid class.
 * 
 * @param dim The dimension of the problem.
 * @param nbase The current dimension of the reduced basis.
 * @param nbase_x The maximum dimension of the reduced basis set.
 * @param notconv The number of newly added basis vectors.
 * @param hpsi The output array for the Hamiltonian H times blockvector psi.
 * @param spsi The output array for the overlap matrix S times blockvector psi.
 * @param hcc Pointer to the array where the calculated Hamiltonian matrix elements will be stored.
 * @param scc Pointer to the array where the calculated overlap matrix elements will be stored.
 */
template <typename T, typename Device>
void DiagoDavid<T, Device>::cal_elem(const int& dim,
                                          int& nbase,           // current dimension of the reduced basis
                                          const int nbase_x,    // maximum dimension of the reduced basis set
                                          const int& notconv,   // number of newly added basis vectors
                                          const T* hpsi,
                                          const T* spsi,
                                          T* hcc,
                                          T* scc)
{
    if (test_david == 1) {
        // ModuleBase::TITLE("DiagoDavid", "cal_elem");
    }

    if (notconv == 0) {
        return;
    }
    // ModuleBase::timer::tick("DiagoDavid", "cal_elem");

    // hcc[nbase](notconv, nbase + notconv)= basis[nbase]' * hpsi
    gemm_op<T, Device>()(this->ctx,
                              'C',
                              'N',
                              notconv,
                              nbase + notconv,
                              dim,
                              this->one,
                              basis + dim*nbase, // basis(:,nbase:)  dim * notconv
                              dim,
                              hpsi,               // dim * (nbase + notconv)
                              dim,
                              this->zero,
                              hcc + nbase,        // notconv * (nbase + notconv)
                              nbase_x);
    // scc[nbase] = basis[nbase]' * spsi
    gemm_op<T, Device>()(this->ctx,
                              'C',
                              'N',
                              notconv,
                              nbase + notconv,
                              dim,
                              this->one,
                              basis + dim*nbase, // dim * notconv
                              dim,
                              spsi,               // dim * (nbase + notconv)
                              dim,
                              this->zero,
                              scc + nbase,        // notconv * (nbase + notconv)
                              nbase_x);


#ifdef __MPI
    if (diag_comm.nproc > 1)
    {
        matrixTranspose_op<T, Device>()(this->ctx, nbase_x, nbase_x, hcc, hcc);
        matrixTranspose_op<T, Device>()(this->ctx, nbase_x, nbase_x, scc, scc);

        auto* swap = new T[notconv * nbase_x];
        syncmem_complex_op()(this->ctx, this->ctx, swap, hcc + nbase * nbase_x, notconv * nbase_x);
        if (std::is_same<T, double>::value)
        {
            Parallel_Reduce::reduce_pool(hcc + nbase * nbase_x, notconv * nbase_x);
        }
        else
        {
            if (base_device::get_current_precision(swap) == "single") {
                MPI_Reduce(swap, hcc + nbase * nbase_x, notconv * nbase_x, MPI_COMPLEX, MPI_SUM, 0, diag_comm.comm);
            }
            else {
                MPI_Reduce(swap, hcc + nbase * nbase_x, notconv * nbase_x, MPI_DOUBLE_COMPLEX, MPI_SUM, 0, diag_comm.comm);
            }
            syncmem_complex_op()(this->ctx, this->ctx, swap, scc + nbase * nbase_x, notconv * nbase_x);
            if (base_device::get_current_precision(swap) == "single") {
                MPI_Reduce(swap, scc + nbase * nbase_x, notconv * nbase_x, MPI_COMPLEX, MPI_SUM, 0, diag_comm.comm);
            }
            else {
                MPI_Reduce(swap, scc + nbase * nbase_x, notconv * nbase_x, MPI_DOUBLE_COMPLEX, MPI_SUM, 0, diag_comm.comm);
            }
        }
        delete[] swap;

        // Parallel_Reduce::reduce_complex_double_pool( hcc + nbase * nbase_x, notconv * nbase_x );
        // Parallel_Reduce::reduce_complex_double_pool( scc + nbase * nbase_x, notconv * nbase_x );

        matrixTranspose_op<T, Device>()(this->ctx, nbase_x, nbase_x, hcc, hcc);
        matrixTranspose_op<T, Device>()(this->ctx, nbase_x, nbase_x, scc, scc);
    }
#endif

    nbase += notconv;
    // ModuleBase::timer::tick("DiagoDavid", "cal_elem");
    return;
}

//==============================================================================
// optimize diag_zhegvx().
// 09-05-09 wangjp
// fixed a bug in diag_zhegvx().
// modify the dimension of h and s as (n,n) and copy the leading N*N
// part of hc & sc into h & s
// 09-05-10 wangjp
// As the complexmatrixs will be copied again in the subroutine ZHEGVX(...  ),
// i.e ZHEGVX(...) will not destroy the input complexmatrixs,
// we needn't creat another two complexmatrixs in diag_zhegvx().
//==============================================================================
template <typename T, typename Device>
void DiagoDavid<T, Device>::diag_zhegvx(const int& nbase,
                                             const int& nband,
                                             const T* hcc,
                                             const T* /*scc*/,
                                             const int& nbase_x,
                                             Real* eigenvalue, // in CPU
                                             T* vcc)
{
    // ModuleBase::timer::tick("DiagoDavid", "diag_zhegvx");
    if (diag_comm.rank == 0)
    {
        assert(nbase_x >= std::max(1, nbase));

        if (this->device == base_device::GpuDevice)
        {
#if defined(__CUDA) || defined(__ROCM)
            Real* eigenvalue_gpu = nullptr;
            resmem_var_op()(this->ctx, eigenvalue_gpu, nbase_x);
            syncmem_var_h2d_op()(this->ctx, this->cpu_ctx, eigenvalue_gpu, this->eigenvalue, nbase_x);

            dnevx_op<T, Device>()(this->ctx, nbase, nbase_x, this->hcc, nband, eigenvalue_gpu, this->vcc);

            syncmem_var_d2h_op()(this->cpu_ctx, this->ctx, this->eigenvalue, eigenvalue_gpu, nbase_x);
            delmem_var_op()(this->ctx, eigenvalue_gpu);
#endif
        }
        else
        {
            dnevx_op<T, Device>()(this->ctx, nbase, nbase_x, this->hcc, nband, this->eigenvalue, this->vcc);
        }
    }

#ifdef __MPI
    if (diag_comm.nproc > 1)
    {
        // vcc: nbase * nband
        for (int i = 0; i < nband; i++)
        {
            MPI_Bcast(&vcc[i * nbase_x], nbase, MPI_DOUBLE_COMPLEX, 0, diag_comm.comm);
        }
        MPI_Bcast(this->eigenvalue, nband, MPI_DOUBLE, 0, diag_comm.comm);
    }
#endif

    // ModuleBase::timer::tick("DiagoDavid", "diag_zhegvx");
    return;
}

/**
 * Refreshes the diagonalization solver by updating the basis and the reduced Hamiltonian.
 * 
 * @param dim The dimension of the problem.
 * @param nband The number of bands.
 * @param nbase The number of basis states.
 * @param nbase_x The maximum dimension of the reduced basis set.
 * @param eigenvalue_in Pointer to the array of eigenvalues.
 * @param psi_in Pointer to the array of wavefunctions.
 * @param ldPsi The leading dimension of the wavefunction array.
 * @param hpsi Pointer to the output array for the updated basis set.
 * @param spsi Pointer to the output array for the updated basis set (nband-th column).
 * @param hcc Pointer to the output array for the updated reduced Hamiltonian.
 * @param scc Pointer to the output array for the updated overlap matrix.
 * @param vcc Pointer to the output array for the updated eigenvector matrix.
 * 
 */
template <typename T, typename Device>
void DiagoDavid<T, Device>::refresh(const int& dim,
                                         const int& nband,
                                         int& nbase,
                                         const int nbase_x, // maximum dimension of the reduced basis set
                                         const Real* eigenvalue_in,
                                         const T *psi_in,
                                         const int ldPsi,
                                         T* hpsi,
                                         T* spsi,
                                         T* hcc,
                                         T* scc,
                                         T* vcc)
{
    if (test_david == 1) {
        // ModuleBase::TITLE("DiagoDavid", "refresh");
    }
    // ModuleBase::timer::tick("DiagoDavid", "refresh");

    // update hp,sp
    setmem_complex_op()(this->ctx, basis , 0, nbase_x * dim);

    // basis(dim, nband) = hpsi(dim, nbase) * vcc(nbase, nband)
    gemm_op<T, Device>()(this->ctx,
                              'N',
                              'N',
                              dim,              // m: row of A,C
                              nband,            // n: col of B,C
                              nbase,            // k: col of A, row of B
                              this->one,
                              hpsi,       // A dim * nbase
                              dim,
                              vcc,        // B nbase * nband
                              nbase_x,
                              zero,
                              basis,           // C dim * nband
                              dim
    );

    //<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
    // basis[nband] = spsi * vcc
    gemm_op<T, Device>()(this->ctx,
                              'N',
                              'N',
                              dim,                // m: row of A,C
                              nband,              // n: col of B,C
                              nbase,              // k: col of A, row of B
                              this->one,
                              spsi,         // A dim * nbase
                              dim,
                              vcc,          // B nbase * nband
                              nbase_x,
                              this->zero,
                              basis + dim*nband,  // C dim * nband
                              dim
    );

    // hpsi = basis, spsi = basis[nband]
    syncmem_complex_op()(this->ctx, this->ctx, hpsi, basis, dim * nband);
    syncmem_complex_op()(this->ctx, this->ctx, spsi, basis + dim*nband, dim * nband);
    /*for (int m = 0; m < nband; m++) {
        for (int ig = 0; ig < dim; ig++)
        {
            hp(m, ig) = basis(m, ig);
            sp(m, ig) = basis(m + nband, ig);
        }
    }*/

    // update basis
    setmem_complex_op()(this->ctx, basis , 0, nbase_x * dim);

    for (int m = 0; m < nband; m++)
    {
        syncmem_complex_op()(this->ctx, this->ctx, basis + dim*m,psi_in + m*ldPsi, dim);
        /*for (int ig = 0; ig < npw; ig++)
            basis(m, ig) = psi(m, ig);*/
    }

    // update the reduced Hamiltonian
    // basis set size reset to nband
    nbase = nband;

    setmem_complex_op()(this->ctx, hcc, 0, nbase_x * nbase_x);

    setmem_complex_op()(this->ctx, scc, 0, nbase_x * nbase_x);

    if (this->device == base_device::GpuDevice)
    {
#if defined(__CUDA) || defined(__ROCM)
        T* hcc_cpu = nullptr;
        T* scc_cpu = nullptr;
        T* vcc_cpu = nullptr;
        base_device::memory::resize_memory_op<T, base_device::DEVICE_CPU>()(this->cpu_ctx,
                                                                            hcc_cpu,
                                                                            nbase_x * nbase_x,
                                                                            "DAV::hcc");
        base_device::memory::resize_memory_op<T, base_device::DEVICE_CPU>()(this->cpu_ctx,
                                                                            scc_cpu,
                                                                            nbase_x * nbase_x,
                                                                            "DAV::scc");
        base_device::memory::resize_memory_op<T, base_device::DEVICE_CPU>()(this->cpu_ctx,
                                                                            vcc_cpu,
                                                                            nbase_x * nbase_x,
                                                                            "DAV::vcc");

        syncmem_d2h_op()(this->cpu_ctx, this->ctx, hcc_cpu, hcc, nbase_x * nbase_x);
        syncmem_d2h_op()(this->cpu_ctx, this->ctx, scc_cpu, scc, nbase_x * nbase_x);
        syncmem_d2h_op()(this->cpu_ctx, this->ctx, vcc_cpu, vcc, nbase_x * nbase_x);

        for (int i = 0; i < nbase; i++)
        {
            hcc_cpu[i * nbase_x + i] = eigenvalue_in[i];
            scc_cpu[i * nbase_x + i] = this->one[0];
            vcc_cpu[i * nbase_x + i] = this->one[0];
        }

        syncmem_h2d_op()(this->ctx, this->cpu_ctx, hcc, hcc_cpu, nbase_x * nbase_x);
        syncmem_h2d_op()(this->ctx, this->cpu_ctx, scc, scc_cpu, nbase_x * nbase_x);
        syncmem_h2d_op()(this->ctx, this->cpu_ctx, vcc, vcc_cpu, nbase_x * nbase_x);

        base_device::memory::delete_memory_op<T, base_device::DEVICE_CPU>()(this->cpu_ctx, hcc_cpu);
        base_device::memory::delete_memory_op<T, base_device::DEVICE_CPU>()(this->cpu_ctx, scc_cpu);
        base_device::memory::delete_memory_op<T, base_device::DEVICE_CPU>()(this->cpu_ctx, vcc_cpu);
#endif
    }
    else
    {
        for (int i = 0; i < nbase; i++)
        {
            hcc[i * nbase_x + i] = eigenvalue_in[i];
            // sc(i, i) = this->one;
            scc[i * nbase_x + i] = this->one[0];
            // vc(i, i) = this->one;
            vcc[i * nbase_x + i] = this->one[0];
        }
    }
    // ModuleBase::timer::tick("DiagoDavid", "refresh");
    return;
}

template <typename T, typename Device>
void DiagoDavid<T, Device>::SchmidtOrth(const int& dim,
                                            const int nband,
                                            const int m,
                                            const T* spsi,
                                            T* lagrange_m,
                                            const int mm_size,
                                            const int mv_size)
{
    //	if(test_david == 1) ModuleBase::TITLE("DiagoDavid","SchmidtOrth");
    // ModuleBase::timer::tick("DiagoDavid", "SchmidtOrth");

    // orthogonalize starting eigenfunction to those already calculated
    // psi_m orthogonalize to psi(0) ~ psi(m-1)
    // Attention, the orthogonalize here read as
    // psi(m) -> psi(m) - \sum_{i < m} \langle psi(i)|S|psi(m) \rangle psi(i)
    // so the orthogonalize is performed about S.

    // assert(basis.get_nbands() >= nband);
    assert(m >= 0);
    assert(m < nband);

    T* psi_m = basis + dim*m;

    // std::complex<double> *lagrange = new std::complex<double>[m + 1];
    // ModuleBase::GlobalFunc::ZEROS(lagrange, m + 1);

    // calculate the square matrix for future lagranges
    if (mm_size != 0)
    {
        //<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
        gemm_op<T, Device>()(this->ctx,
                                  'C',
                                  'N',
                                  mm_size, // m: row of A,C
                                  mm_size, // n: col of B,C
                                  dim, // k: col of A, row of B
                                  this->one, // alpha
                                  basis + dim*(m - mv_size + 1 - mm_size), // A
                                  dim, // LDA: if(N) max(1,m) if(T) max(1,k)
                                  &spsi[m * dim], // B
                                  dim, // LDB: if(N) max(1,k) if(T) max(1,n)
                                  this->zero, // belta
                                  &lagrange_m[m - mv_size + 1 - mm_size], // C
                                  nband // LDC: if(N) max(1, m)
        );
    }
    // calculate other lagranges for this band
    gemv_op<T, Device>()(this->ctx,
                              'C',
                              dim,
                              mv_size,
                              this->one,
                              basis + dim*(m - mv_size + 1),
                              dim,
                              &spsi[m * dim],
                              1,
                              this->zero,
                              &lagrange_m[m - mv_size + 1],
                              1);

    Parallel_Reduce::reduce_pool(lagrange_m, m + 1);

    T var = *this->zero;
    syncmem_d2h_op()(this->cpu_ctx, this->ctx, &var, lagrange_m + m, 1);
    double psi_norm = get_real(var);

    // !!test
    // assert(psi_norm > 0.0);

    gemv_op<T, Device>()(this->ctx,
                              'N',
                              dim,
                              m,
                              this->neg_one,
                              basis,
                              dim,
                              lagrange_m,
                              1,
                              this->one,
                              psi_m,
                              1);

    psi_norm -= dot_real_op<T, Device>()(this->ctx, m, lagrange_m, lagrange_m, false);

    // for (int j = 0; j < m; j++)
    // {
    //     const std::complex<double> alpha = std::complex<double>(-1, 0) * lagrange_m[j];
    //     zaxpy_(&npw, &alpha, &psi(j,0), &inc, psi_m, &inc);
    //     /*for (int ig = 0; ig < npw; ig++)
    //     {
    //         psi_m[ig] -= lagrange[j] * psi(j, ig);
    //     }*/
    //     psi_norm -= (conj(lagrange_m[j]) * lagrange_m[j]).real();
    // }

    // !!test
    // assert(psi_norm > 0.0);

    psi_norm = sqrt(psi_norm);

    if (psi_norm < 1.0e-12)
    {
        std::cout << "DiagoDavid::SchmidtOrth:aborted for psi_norm <1.0e-12" << std::endl;
#ifdef DEBUG
        std::cout << "psi_norm = " << psi_norm << std::endl;
#endif
        std::cout << "nband = " << nband << std::endl;
        std::cout << "m = " << m << std::endl;
        // !!test
        // exit(0);
        return;
    }
    else
    {
        vector_div_constant_op<T, Device>()(this->ctx, dim, psi_m, psi_m, psi_norm);
        // for (int i = 0; i < npw; i++)
        // {
        //     psi_m[i] /= psi_norm;
        // }
    }

    // delete[] lagrange;
    // ModuleBase::timer::tick("DiagoDavid", "SchmidtOrth");
    return;
}

template <typename T, typename Device>
void DiagoDavid<T, Device>::planSchmidtOrth(const int nband, std::vector<int>& pre_matrix_mm_m, std::vector<int>& pre_matrix_mv_m)
{
    if (nband <= 0) {
        return;
}
    std::fill(pre_matrix_mm_m.begin(), pre_matrix_mm_m.end(), 0);
    std::fill(pre_matrix_mv_m.begin(), pre_matrix_mv_m.end(), 0);
    int last_matrix_size = nband;
    int matrix_size = int(nband / 2);
    int divide_times = 0;
    std::vector<int> divide_points(nband);
    int res_nband = nband - matrix_size;
    while (matrix_size > 1)
    {
        int index = nband - matrix_size;
        if (divide_times == 0)
        {
            divide_points[0] = index;
            pre_matrix_mm_m[index] = matrix_size;
            if (res_nband == matrix_size) {
                pre_matrix_mv_m[index] = 1;
            } else {
                pre_matrix_mv_m[index] = 2;
}
            divide_times = 1;
        }
        else
        {
            for (int i = divide_times - 1; i >= 0; i--)
            {
                divide_points[i * 2] = divide_points[i] - matrix_size;
                divide_points[i * 2 + 1] = divide_points[i * 2] + last_matrix_size;
                pre_matrix_mm_m[divide_points[i * 2]] = matrix_size;
                pre_matrix_mm_m[divide_points[i * 2 + 1]] = matrix_size;
                if (res_nband == matrix_size)
                {
                    pre_matrix_mv_m[divide_points[i * 2]] = 1;
                    pre_matrix_mv_m[divide_points[i * 2 + 1]] = 1;
                }
                else
                {
                    pre_matrix_mv_m[divide_points[i * 2]] = 2;
                    pre_matrix_mv_m[divide_points[i * 2 + 1]] = 2;
                }
            }
            divide_times *= 2;
        }
        last_matrix_size = matrix_size;
        matrix_size = int(res_nband / 2);
        res_nband -= matrix_size;
    }
    // fill the pre_matrix_mv_m array
    pre_matrix_mv_m[0] = 1;
    for (int m = 1; m < nband; m++)
    {
        if (pre_matrix_mv_m[m] == 0)
        {
            pre_matrix_mv_m[m] = pre_matrix_mv_m[m - 1] + 1;
        }
    }
}

/**
 * @brief Performs iterative diagonalization using the David algorithm.
 * 
 * @tparam T The type of the elements in the matrix.
 * @tparam Device The device type (CPU or GPU).
 * @param hpsi_func The function object that computes the matrix-blockvector product H * psi.
 * @param spsi_func The function object that computes the matrix-blockvector product overlap S * psi.
 * @param ldPsi The leading dimension of the psi_in array.
 * @param psi_in The input wavefunction.
 * @param eigenvalue_in The array to store the eigenvalues.
 * @param david_diag_thr The convergence threshold for the diagonalization.
 * @param david_maxiter The maximum number of iterations for the diagonalization.
 * @param ntry_max The maximum number of attempts for the diagonalization restart.
 * @param notconv_max The maximum number of bands unconverged allowed.
 * @return The total number of iterations performed during the diagonalization.
 * 
 * @note ntry_max is an empirical parameter that should be specified in external routine, default 5
 *       notconv_max is determined by the accuracy required for the calculation, default 0
 */
template <typename T, typename Device>
int DiagoDavid<T, Device>::diag(const HPsiFunc& hpsi_func,
                                const SPsiFunc& spsi_func,
                                const int ldPsi,
                                T *psi_in,
                                Real* eigenvalue_in,
                                const Real david_diag_thr,
                                const int david_maxiter,
                                const int ntry_max,
                                const int notconv_max)
{
    /// record the times of trying iterative diagonalization
    int ntry = 0;
    this->notconv = 0;

#if defined(__CUDA) || defined(__ROCM)
    if (this->device == base_device::GpuDevice)
    {
        resmem_var_op()(this->ctx, this->d_precondition, ldPsi);
        syncmem_var_h2d_op()(this->ctx, this->cpu_ctx, this->d_precondition, this->precondition, ldPsi);
    }
#endif

    int sum_dav_iter = 0;
    do
    {
        sum_dav_iter += this->diag_once(hpsi_func, spsi_func, dim, nband, ldPsi, psi_in, eigenvalue_in, david_diag_thr, david_maxiter);
        ++ntry;
    } while (!check_block_conv(ntry, this->notconv, ntry_max, notconv_max));

    if (notconv > std::max(5, nband / 4))
    {
        std::cout << "\n notconv = " << this->notconv;
        std::cout << "\n DiagoDavid::diag', too many bands are not converged! \n";
    }
    return sum_dav_iter;
}

/**
 * @brief Check the convergence of block eigenvectors in the Davidson iteration.
 *
 * This function determines whether the block eigenvectors have reached convergence
 * during the iterative diagonalization process. Convergence is judged based on
 * the number of eigenvectors that have not converged and the maximum allowed
 * number of such eigenvectors.
 *
 * @tparam T The data type for the eigenvalues and eigenvectors (e.g., float, double).
 * @tparam Device The device type (e.g., base_device::DEVICE_CPU).
 * @param ntry The current number of tries for diagonalization.
 * @param notconv The current number of eigenvectors that have not converged.
 * @param ntry_max The maximum allowed number of tries for diagonalization.
 * @param notconv_max The maximum allowed number of eigenvectors that can fail to converge.
 * @return true if the eigenvectors are considered converged or the maximum number
 *         of tries has been reached, false otherwise.
 *
 * @note Exits the diagonalization loop if either the convergence criteria
 *       are met or the maximum number of tries is exceeded.
 */
template <typename T, typename Device>
inline bool DiagoDavid<T, Device>::check_block_conv(const int& ntry,
                                                    const int& notconv,
                                                    const int& ntry_max,
                                                    const int& notconv_max)
{
    // Allow at most 5 tries at diag. If more than 5 then exit loop.
    if(ntry > ntry_max)
    {
        return true;
    }
    // If notconv <= notconv_max allowed, set convergence to true and exit loop.
    if(notconv <= notconv_max)
    {
        return true;
    }
    // else return false, continue loop until either condition above is met.
    return false;
}

namespace hsolver {
template class DiagoDavid<std::complex<float>, base_device::DEVICE_CPU>;
template class DiagoDavid<std::complex<double>, base_device::DEVICE_CPU>;
#if ((defined __CUDA) || (defined __ROCM))
template class DiagoDavid<std::complex<float>, base_device::DEVICE_GPU>;
template class DiagoDavid<std::complex<double>, base_device::DEVICE_GPU>;
#endif
#ifdef __LCAO
template class DiagoDavid<double, base_device::DEVICE_CPU>;
#if ((defined __CUDA) || (defined __ROCM))
template class DiagoDavid<double, base_device::DEVICE_GPU>;
#endif
#endif
} // namespace hsolver
