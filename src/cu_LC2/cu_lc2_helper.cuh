#ifndef CU_LC2_HELPER
#define CU_LC2_HELPER

#define EPS12 1e-12
#define ZERO_THRESHOLD 1e-6

//Constant memory declarations:
__constant__ float transformMatrix_dConstant[16];
__constant__ float Sobel_X_dConstant[27];
__constant__ float Sobel_Y_dConstant[27];
__constant__ float Sobel_Z_dConstant[27];

//Shared memory declarations:
extern __shared__ float dynSharedMem[];

struct IsNotZero{    
    __host__ __device__ bool operator()(const float x){ return x != 0.0f; }
};

struct divide_pdf_Functor {
    float denominator;
    divide_pdf_Functor(float _denominator){denominator = _denominator;}
    __host__ __device__
    float operator()(float i)
    {
        if(i == 0){
            return 1; 
        }
        else{
            return i / denominator;
        }
    }
};

struct divide_entropy_Functor {
    __host__ __device__
    float operator()(float i)
    {
        return i * logf(i); 
    }
};

#endif