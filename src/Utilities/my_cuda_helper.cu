
#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <sstream>
#include <iomanip>
#include <chrono>
#include <assert.h>

#include <cuda_runtime.h>
#include "device_launch_parameters.h"
#include "cusparse.h"
#include "cublas_v2.h"
#include "thrust/host_vector.h"
#include "thrust/device_vector.h"
#include "thrust/copy.h"
#include "npp.h"

#include "my_cuda_helper.cuh"


/* ------------------------------------Basic binary file IOs------------------------------------ */
//Read binary files from disk, with Number of elements.
template<typename T> 
void readFromBin(T *Output, int Num_Elements, const std::string FILENAME) {
	std::ifstream InputStream;
	InputStream.open(FILENAME, std::ios::in | std::ios::binary);

	if (!InputStream.good()) {
		std::cout << "Failed to open " << FILENAME << std::endl;
		exit(0);
	}

	InputStream.read(reinterpret_cast<char*>(Output), sizeof(T) * Num_Elements);
	Output = reinterpret_cast<T*>(Output);

	InputStream.close();
}
template void readFromBin<float>(float *Output, int Num_Elements, const std::string FILENAME);
template void readFromBin<int>(int *Output, int Num_Elements, const std::string FILENAME);

//Write binary files to disk, with Number of elements.
template<typename T>
void writeToBin(T *Output, int Num_Elements, const std::string FILENAME) {
	std::ofstream OutputStream;
	OutputStream.open(FILENAME, std::ios::app | std::ios::binary);

	if (!OutputStream.good()) {
		std::cout << "Failed to open " << FILENAME << std::endl;
		exit(0);
	}

	OutputStream.write(reinterpret_cast<char*>(Output), sizeof(T) * Num_Elements);

	OutputStream.close();
}
template void writeToBin<float>(float *Output, int Num_Elements, const std::string FILENAME);
template void writeToBin<int>(int *Output, int Num_Elements, const std::string FILENAME);
template void writeToBin<uint8_t>(uint8_t *Output, int Num_Elements, const std::string FILENAME);


/* ------------------------------------ cuSPARSE functions ------------------------------------ */
void convertDenseToSparseCSR( 
    float* InputMatrix, 
    std::vector<float>& csrVal, 
    std::vector<int>& csrRowPtr, 
    std::vector<int>& csrColInd, 
    const int num_rows, 
    const int num_cols, 
    int& nnz_output)
{
    //handle and describer initialization:
    cusparseHandle_t sparse_handle = NULL;
    cusparseMatDescr_t Mat_Descr = NULL;
    cusparseStatus_t status = CUSPARSE_STATUS_SUCCESS;

    const int ld = num_rows;

    //Sparse output arrays:
    int* csrRowPtr_array = NULL;
    int* csrColInd_array = NULL;
    float* csrVal_array = NULL;

    //declare device mem-pointers:
    float* d_InputMatrix = NULL;
    int* d_csrRowPtr = NULL;
    int* d_csrColInd = NULL;
    float* d_csrVal = NULL;

    //store the counter for nnz, finally convey to nnz_output;
    int nnz = 0;

    float threshold = 0;

    //Create cusparse_handle
    status = cusparseCreate(&sparse_handle);
    assert(CUSPARSE_STATUS_SUCCESS == status);

    status = cusparseCreateMatDescr(&Mat_Descr);
    assert(CUSPARSE_STATUS_SUCCESS == status);

    cusparseSetMatIndexBase(Mat_Descr, CUSPARSE_INDEX_BASE_ZERO);
    cusparseSetMatType(Mat_Descr, CUSPARSE_MATRIX_TYPE_GENERAL);

    //mem-alloc:
    cudaMalloc((void**)&d_InputMatrix, sizeof(float) * ld * num_cols);
    cudaMalloc((void**)&d_csrRowPtr, sizeof(int) * (num_rows + 1));

    cudaMemcpy(d_InputMatrix, InputMatrix, sizeof(float) * ld * num_cols, cudaMemcpyHostToDevice);
    
    //declare and allocate temp memspace
    std::size_t lworkInBytes = 0;
    char *d_work = NULL;
    status = cusparseSpruneDense2csr_bufferSizeExt(sparse_handle, num_rows, num_cols, d_InputMatrix, ld, &threshold, Mat_Descr, d_csrVal, d_csrRowPtr, d_csrColInd, &lworkInBytes);
    assert(CUSPARSE_STATUS_SUCCESS == status);
    cudaMalloc((void**)&d_work, lworkInBytes);

    //Calculate number of non-zero elements:
    status = cusparseSpruneDense2csrNnz(sparse_handle, num_rows, num_cols, d_InputMatrix, ld, &threshold, Mat_Descr, d_csrRowPtr, &nnz, d_work);
    cudaDeviceSynchronize();
    assert(CUSPARSE_STATUS_SUCCESS == status);
    nnz_output = nnz;

    //allocate output CSR device mem:
    cudaMalloc((void**)&d_csrColInd, sizeof(int) * nnz);
    cudaMalloc((void**)&d_csrVal, sizeof(float) * nnz);

    //Calculate CSR outputs:
    status = cusparseSpruneDense2csr(sparse_handle, num_rows, num_cols, d_InputMatrix, ld, &threshold, Mat_Descr, d_csrVal, d_csrRowPtr, d_csrColInd, d_work);
    cudaDeviceSynchronize();
    assert(CUSPARSE_STATUS_SUCCESS == status);

    csrVal_array = new float[nnz];
    csrRowPtr_array = new int[num_rows + 1];    
    csrColInd_array = new int[nnz];


    cudaMemcpy(csrVal_array, d_csrVal, sizeof(float) * nnz, cudaMemcpyDeviceToHost);
    csrVal = std::vector<float>(csrVal_array, csrVal_array + nnz);

    cudaMemcpy(csrRowPtr_array, d_csrRowPtr, sizeof(int) * (num_rows + 1), cudaMemcpyDeviceToHost);
    csrRowPtr = std::vector<int>(csrRowPtr_array, csrRowPtr_array + (num_rows + 1));

    cudaMemcpy(csrColInd_array, d_csrColInd, sizeof(int) * nnz, cudaMemcpyDeviceToHost);
    csrColInd = std::vector<int>(csrColInd_array, csrColInd_array + nnz);

    cudaFree(d_csrRowPtr);
    cudaFree(d_csrVal);
    cudaFree(d_csrColInd);
    cudaFree(d_work);

    cusparseDestroy(sparse_handle);
    cusparseDestroyMatDescr(Mat_Descr);

    delete[] csrVal_array;
    delete[] csrRowPtr_array;
    delete[] csrColInd_array;

    cudaDeviceReset();
}

/* ------------------------------------ cuBLAS functions ------------------------------------ */
/*
    Solving linear system function, by 
        
        minimizing: || Carray[i] - Aarray[i]*Xarray[i] || , with i = 0, ...,batchSize-1
    
    *Aarray has dimension of [m] rows, [3] columns.
    *Carray has dimension of [m] rows, [1] columns.
    *Xarray, the solution array, has dimension of [3] rows, [1] columns. 

    **All the inputs are in Column-Major format. 
*/
// Overload - 1: Inputs are in host memory
void lsq_solver_basedQR_batched( 
    std::vector<float>& Carray, //Input, dimension m x 1 * batchSize
    std::vector<float>& Aarray, //Input, dimension m x 3 * batchSize
    std::vector<float>& Solution_Array, //Output, dimension 3 x 1 * batchSize
    // std::vector<float>& Residual, //Output, dimension m x 1 * batchSize
    const int batchSize)
{
    cudaError_t cudaStat = cudaSuccess;

    //cuBLAS initialization:
    cublasStatus_t cublas_stat;
    cublasHandle_t handle;

    cublas_stat = cublasCreate(&handle);
    assert(cublas_stat == CUBLAS_STATUS_SUCCESS);

    //Verify if the dimensions are matching:
    if(Carray.size() % batchSize == 0){
        std::cout << "Carray leading dimension is: " << Carray.size() / batchSize << std::endl;
    }
    else{
        std::cout << "Carray is not matching. " << std::endl;
        return;
    }
    if(Aarray.size() % Carray.size() == 0){
        std::cout << "Solution has number of parameters of: " << Aarray.size() / Carray.size() << std::endl;
    }
    else{
        std::cout << "Solution is not matching. " << std::endl;
        return;
    }
    //TODO...

    //Size variables declarations:
    //Carray:
    int num_rows_Carray         =       Carray.size() / batchSize;
    int num_columns_Carray      =       1;
    //Aarray: 
    int num_rows_Aarray         =       Aarray.size() / Solution_Array.size();
    int num_columns_Aarray      =       Aarray.size() / Carray.size();
    //Solution: 
    int num_solParams           =       Aarray.size() / Carray.size();

    std::cout << "Array: " << num_rows_Aarray << " x " << num_columns_Aarray << std::endl;
    std::cout << "Crray: " << num_rows_Carray << " x " << num_columns_Carray << std::endl;
    std::cout << "Solutionarray: " << num_solParams << " x " << num_columns_Carray << std::endl;

    //declare device raw data variables: 
    //copy data from [HOST] to [DEVICE]
    float *Carray_d = NULL, 
          *Aarray_d = NULL;
    cudaStat = cudaMalloc((void**)&Carray_d, sizeof(float) * num_rows_Carray * num_columns_Carray * batchSize);
    assert(cudaSuccess == cudaStat);
    cudaStat = cudaMalloc((void**)&Aarray_d, sizeof(float) * num_rows_Aarray * num_columns_Aarray * batchSize);
    assert(cudaSuccess == cudaStat);

    cudaStat = cudaMemcpy(Carray_d, Carray.data(), sizeof(float) * num_rows_Carray * num_columns_Carray * batchSize, cudaMemcpyHostToDevice);
    assert(cudaSuccess == cudaStat);
    cudaStat = cudaMemcpy(Aarray_d, Aarray.data(), sizeof(float) * num_rows_Aarray * num_columns_Aarray * batchSize, cudaMemcpyHostToDevice);
    assert(cudaSuccess == cudaStat);

    //batches pointers declarations:
    float **Carray_ptr = new float*[batchSize], 
          **Aarray_ptr = new float*[batchSize];
    for(int it_batch = 0; it_batch < batchSize; ++it_batch){
        Carray_ptr[it_batch] = Carray_d + it_batch * num_rows_Carray * num_columns_Carray;
        Aarray_ptr[it_batch] = Aarray_d + it_batch * num_rows_Aarray * num_columns_Aarray;
    }

    float **Carray_ptr_d = NULL, **Aarray_ptr_d = NULL;
    cudaStat = cudaMalloc((void ***)&Carray_ptr_d, sizeof(float*) * batchSize);
    assert(cudaSuccess == cudaStat);
    cudaStat = cudaMalloc((void ***)&Aarray_ptr_d, sizeof(float*) * batchSize);
    assert(cudaSuccess == cudaStat);

    cudaStat = cudaMemcpy(Carray_ptr_d, Carray_ptr, sizeof(float*) * batchSize, cudaMemcpyHostToDevice);
    assert(cudaSuccess == cudaStat);
    cudaStat = cudaMemcpy(Aarray_ptr_d, Aarray_ptr, sizeof(float*) * batchSize, cudaMemcpyHostToDevice);
    assert(cudaSuccess == cudaStat);

    //perform QR-based-least-square solver:
    int OutputInfo = 0;
    int *devInfo_d = NULL;
    cudaStat = cudaMalloc((void**)&devInfo_d, sizeof(int) * batchSize);
    assert(cudaSuccess == cudaStat);
    cublas_stat = cublasSgelsBatched( 
        handle, 
        CUBLAS_OP_N, 
        num_rows_Aarray, 
        num_columns_Aarray, 
        num_columns_Carray, 
        Aarray_ptr_d, 
        num_rows_Aarray, 
        Carray_ptr_d, 
        num_rows_Carray, 
        &OutputInfo, 
        devInfo_d, 
        batchSize);
    cudaDeviceSynchronize();
    assert(cublas_stat == CUBLAS_STATUS_SUCCESS);
    
    //Transfer results back:
    for(int it_batch = 0; it_batch < batchSize; ++it_batch){
        cudaStat = cudaMemcpy(Solution_Array.data() + it_batch * num_solParams, Carray_d + it_batch * num_rows_Carray * num_columns_Carray, sizeof(float) * num_solParams, cudaMemcpyDeviceToHost);
        assert(cudaSuccess == cudaStat);
    }

    //Memories free
    cublasDestroy(handle);

    cudaFree(Carray_ptr_d);
    cudaFree(Aarray_ptr_d);
    cudaFree(Carray_d);
    cudaFree(Aarray_d);
    cudaFree(devInfo_d);

    free(Carray_ptr);
    free(Aarray_ptr);
}

// Overload - 2: Inputs are in device memory
void lsq_solver_basedQR_batched( 
    float *Carray, //Input, dimension Num_rows x 1 * batchSize
    float *Aarray, //Input, dimension Num_rows x 3 * batchSize
    const int Num_rows, 
    const int batchSize, 
    cublasStatus_t* cublas_stat, 
    cublasHandle_t* handle)
{
    //Array batche-pointers definition:
    //batches pointers declarations:
    float **Carray_ptr = new float*[batchSize], 
          **Aarray_ptr = new float*[batchSize];
    for(int it_batch = 0; it_batch < batchSize; ++it_batch){
        Carray_ptr[it_batch] = Carray + it_batch * Num_rows * 1;
        Aarray_ptr[it_batch] = Aarray + it_batch * Num_rows * 3;
    }

    float **Carray_ptr_d = NULL, **Aarray_ptr_d = NULL;
    cudaMalloc((void ***)&Carray_ptr_d, sizeof(float*) * batchSize);
    cudaMalloc((void ***)&Aarray_ptr_d, sizeof(float*) * batchSize);

    cudaMemcpy(Carray_ptr_d, Carray_ptr, sizeof(float*) * batchSize, cudaMemcpyHostToDevice);
    cudaMemcpy(Aarray_ptr_d, Aarray_ptr, sizeof(float*) * batchSize, cudaMemcpyHostToDevice);

    //perform QR-based-least-square solver:
    int OutputInfo = 0;
    int *devInfo_d = NULL;
    cudaMalloc((void**)&devInfo_d, sizeof(int) * batchSize);
    *cublas_stat = cublasSgelsBatched( 
        *handle, 
        CUBLAS_OP_N, 
        Num_rows, 
        3, 
        1, 
        Aarray_ptr_d, 
        Num_rows, 
        Carray_ptr_d, 
        Num_rows, 
        &OutputInfo, 
        devInfo_d, 
        batchSize);
    cudaDeviceSynchronize();
    assert(*cublas_stat == CUBLAS_STATUS_SUCCESS);

    // int *devInfo = new int[batchSize];
    // cudaMemcpy(devInfo, devInfo_d, sizeof(int) * batchSize, cudaMemcpyDeviceToHost);

    cudaFree(Carray_ptr_d);
    cudaFree(Aarray_ptr_d);
    cudaFree(devInfo_d);

    free(Carray_ptr);
    free(Aarray_ptr);
    // free(devInfo);
}

void histogramCalculator
( 
    const thrust::device_vector<float>& _srcVector, 
    thrust::device_vector<int>& _hist, 
    const int _numberOfBins, 
    float _minIntensity, 
    float _maxIntensity
)
{
    NppiSize srcSizeROI; 
    srcSizeROI.height = 1; 
    srcSizeROI.width = (int)_srcVector.size(); 
    
    _hist.resize(_numberOfBins - 1, 0); 

    //get buffer size and allocate: 
    int nDeviceBufferSize;
    nppiHistogramEvenGetBufferSize_8u_C1R(srcSizeROI, _numberOfBins ,&nDeviceBufferSize);
    thrust::device_vector<unsigned char> scratchBuffer_d(nDeviceBufferSize, 0); 

    //initialize pLevels: 
    thrust::host_vector<float> pLevel_h(_numberOfBins, 0); 
    float unitSpacing = (_maxIntensity - _minIntensity) / (_numberOfBins - 1); 
    for(int i = 0; i < _numberOfBins; ++i){
        pLevel_h[i] = _minIntensity + i * unitSpacing; 
    }
    thrust::device_vector<float> pLevel_d = pLevel_h; 

    NppStatus results = 
    nppiHistogramRange_32f_C1R(
        thrust::raw_pointer_cast(_srcVector.data()), 
        srcSizeROI.width * sizeof(float), 
        srcSizeROI, 
        thrust::raw_pointer_cast(_hist.data()), 
        // pLevel_const, 
        thrust::raw_pointer_cast(pLevel_d.data()), 
        _numberOfBins, 
        thrust::raw_pointer_cast(scratchBuffer_d.data())
    ); 
}

/* ------------------------------------ basic Math functions ------------------------------------ */
bool InvertS4X4Matrix(const float m[16], float invOut[16])
{
    float inv[16], det;
    int i;

    inv[0] = m[5]  * m[10] * m[15] - 
             m[5]  * m[11] * m[14] - 
             m[9]  * m[6]  * m[15] + 
             m[9]  * m[7]  * m[14] +
             m[13] * m[6]  * m[11] - 
             m[13] * m[7]  * m[10];

    inv[4] = -m[4]  * m[10] * m[15] + 
              m[4]  * m[11] * m[14] + 
              m[8]  * m[6]  * m[15] - 
              m[8]  * m[7]  * m[14] - 
              m[12] * m[6]  * m[11] + 
              m[12] * m[7]  * m[10];

    inv[8] = m[4]  * m[9] * m[15] - 
             m[4]  * m[11] * m[13] - 
             m[8]  * m[5] * m[15] + 
             m[8]  * m[7] * m[13] + 
             m[12] * m[5] * m[11] - 
             m[12] * m[7] * m[9];

    inv[12] = -m[4]  * m[9] * m[14] + 
               m[4]  * m[10] * m[13] +
               m[8]  * m[5] * m[14] - 
               m[8]  * m[6] * m[13] - 
               m[12] * m[5] * m[10] + 
               m[12] * m[6] * m[9];

    inv[1] = -m[1]  * m[10] * m[15] + 
              m[1]  * m[11] * m[14] + 
              m[9]  * m[2] * m[15] - 
              m[9]  * m[3] * m[14] - 
              m[13] * m[2] * m[11] + 
              m[13] * m[3] * m[10];

    inv[5] = m[0]  * m[10] * m[15] - 
             m[0]  * m[11] * m[14] - 
             m[8]  * m[2] * m[15] + 
             m[8]  * m[3] * m[14] + 
             m[12] * m[2] * m[11] - 
             m[12] * m[3] * m[10];

    inv[9] = -m[0]  * m[9] * m[15] + 
              m[0]  * m[11] * m[13] + 
              m[8]  * m[1] * m[15] - 
              m[8]  * m[3] * m[13] - 
              m[12] * m[1] * m[11] + 
              m[12] * m[3] * m[9];

    inv[13] = m[0]  * m[9] * m[14] - 
              m[0]  * m[10] * m[13] - 
              m[8]  * m[1] * m[14] + 
              m[8]  * m[2] * m[13] + 
              m[12] * m[1] * m[10] - 
              m[12] * m[2] * m[9];

    inv[2] = m[1]  * m[6] * m[15] - 
             m[1]  * m[7] * m[14] - 
             m[5]  * m[2] * m[15] + 
             m[5]  * m[3] * m[14] + 
             m[13] * m[2] * m[7] - 
             m[13] * m[3] * m[6];

    inv[6] = -m[0]  * m[6] * m[15] + 
              m[0]  * m[7] * m[14] + 
              m[4]  * m[2] * m[15] - 
              m[4]  * m[3] * m[14] - 
              m[12] * m[2] * m[7] + 
              m[12] * m[3] * m[6];

    inv[10] = m[0]  * m[5] * m[15] - 
              m[0]  * m[7] * m[13] - 
              m[4]  * m[1] * m[15] + 
              m[4]  * m[3] * m[13] + 
              m[12] * m[1] * m[7] - 
              m[12] * m[3] * m[5];

    inv[14] = -m[0]  * m[5] * m[14] + 
               m[0]  * m[6] * m[13] + 
               m[4]  * m[1] * m[14] - 
               m[4]  * m[2] * m[13] - 
               m[12] * m[1] * m[6] + 
               m[12] * m[2] * m[5];

    inv[3] = -m[1] * m[6] * m[11] + 
              m[1] * m[7] * m[10] + 
              m[5] * m[2] * m[11] - 
              m[5] * m[3] * m[10] - 
              m[9] * m[2] * m[7] + 
              m[9] * m[3] * m[6];

    inv[7] = m[0] * m[6] * m[11] - 
             m[0] * m[7] * m[10] - 
             m[4] * m[2] * m[11] + 
             m[4] * m[3] * m[10] + 
             m[8] * m[2] * m[7] - 
             m[8] * m[3] * m[6];

    inv[11] = -m[0] * m[5] * m[11] + 
               m[0] * m[7] * m[9] + 
               m[4] * m[1] * m[11] - 
               m[4] * m[3] * m[9] - 
               m[8] * m[1] * m[7] + 
               m[8] * m[3] * m[5];

    inv[15] = m[0] * m[5] * m[10] - 
              m[0] * m[6] * m[9] - 
              m[4] * m[1] * m[10] + 
              m[4] * m[2] * m[9] + 
              m[8] * m[1] * m[6] - 
              m[8] * m[2] * m[5];

    det = m[0] * inv[0] + m[1] * inv[4] + m[2] * inv[8] + m[3] * inv[12];

    if (det == 0)
        return false;

    det = 1.0 / det;

    for (i = 0; i < 16; i++)
        invOut[i] = inv[i] * det;

    return true;
}

void MatrixPointS4X4Multiply(const float m_i[16], const float p_i[4], float p_o[4]){
    float accumulator = 0.0f;
    for(int row_it = 0; row_it < 4; ++row_it){
        for(int ele_it = 0; ele_it < 4; ++ele_it){
            accumulator += m_i[ele_it + row_it * 4] * p_i[ele_it];
        }
        p_o[row_it] = accumulator;
        accumulator = 0.0f;
    }
}

//Display 2d matrix:
void MatrixS4X4Print(const std::vector<float> InputMatrix){
    for(int row_it = 0; row_it < 4; ++row_it){
        for(int col_it = 0; col_it < 4; ++col_it){
            std::cout << std::setw(15) << InputMatrix[col_it + row_it * 4] << std::setw(15);
        }
        std::cout << std::endl;
    }
}

void MatrixS4X4Print(const float *InputMatrix){
    for(int row_it = 0; row_it < 4; ++row_it){
        for(int col_it = 0; col_it < 4; ++col_it){
            std::cout << std::setw(15) << InputMatrix[col_it + row_it * 4] << std::setw(15);
        }
        std::cout << std::endl;
    }
}

void MatrixS3X3Print(const std::vector<float> InputMatrix){
    for(int row_it = 0; row_it < 3; ++row_it){
        for(int col_it = 0; col_it < 3; ++col_it){
            std::cout << std::setw(15) << InputMatrix[col_it + row_it * 3] << std::setw(15);
        }
        std::cout << std::endl;
    }
}

void MatrixS3X3Print(const float *InputMatrix){
    for(int row_it = 0; row_it < 3; ++row_it){
        for(int col_it = 0; col_it < 3; ++col_it){
            std::cout << std::setw(15) << InputMatrix[col_it + row_it * 3] << std::setw(15);
        }
        std::cout << std::endl;
    }
}

/* ------------------------------------ Utilities functions ------------------------------------ */
MyTimer::MyTimer(){

    BeginSet = false;
    EndSet = false;

    duration = 0.0;
}
void MyTimer::tic(){
    BeginSet = true;
    begin = std::chrono::steady_clock::now();
}
void MyTimer::toc(){
    end = std::chrono::steady_clock::now();
    EndSet = true;
}
double MyTimer::Duration(std::ostream& os){
    if(BeginSet && EndSet){
        duration = std::chrono::duration_cast<std::chrono::nanoseconds>(end - begin).count() / 1.0e6;
        os << "Running time: " << duration << "[ms]. \n";
        BeginSet = false;
        EndSet = false;

        return duration;
    }
    else{
        os << "Please set timestamp! \n";
        return 0.0;
    }
}
double MyTimer::Duration(){
    if(BeginSet && EndSet){
        duration = std::chrono::duration_cast<std::chrono::nanoseconds>(end - begin).count() / 1.0e6;
        BeginSet = false;
        EndSet = false;

        return duration;
    }
    else{
        std::cout << "Please set timestamp! \n";
        return 0.0;
    }
}
