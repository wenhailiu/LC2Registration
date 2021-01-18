#ifndef MYCUDAHELPER
#define MYCUDAHELPER

#include <iostream>
#include <vector>
#include <string>
#include <chrono>

#include "cusparse.h"
#include "cublas_v2.h"

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

template<typename T> 
extern void readFromBin(T *Output, int Num_Elements, const std::string FILENAME);

template<typename T>
extern void writeToBin(T *Output, int Num_Elements, const std::string FILENAME);


extern void lsq_solver_basedQR_batched(std::vector<float>& Carray, std::vector<float>& Aarray, std::vector<float>& Solution_Array, const int batchSize);
extern void lsq_solver_basedQR_batched(float *Carray, float *Aarray, const int Num_rows, const int batchSize, cublasStatus_t* cublas_stat, cublasHandle_t* handle);

void histogramCalculator
( 
    const thrust::device_vector<float>& _srcVector, 
    thrust::device_vector<int>& _hist, 
    const int _numberOfBins = 256, 
    float _minIntensity = 0, 
    float _maxIntensity = 255
); 

extern bool InvertS4X4Matrix(const float m[16], float invOut[16]);
extern void MatrixPointS4X4Multiply(const float m_i[16], const float p_i[4], float p_o[4]);
extern void MatrixS4X4Print(const std::vector<float> InputMatrix);
extern void MatrixS4X4Print(const float *InputMatrix);
extern void MatrixS3X3Print(const std::vector<float> InputMatrix);
extern void MatrixS3X3Print(const float *InputMatrix);

class MyTimer{
public:
    MyTimer();
    void tic();
    void toc();
    double Duration(std::ostream& os);
    double Duration();

private:
    std::chrono::steady_clock::time_point begin;
    std::chrono::steady_clock::time_point end;

    bool BeginSet;
    bool EndSet;
    double duration;
};
#endif