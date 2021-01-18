#include <iostream>
#include <vector>
#include <fstream>
#include <chrono>
#include <cmath>
#include <thread>
#include <assert.h>

#include <cuda_runtime.h>
#include "device_launch_parameters.h"
#include <thrust/execution_policy.h>
#include "thrust/device_vector.h"
#include "thrust/count.h"
#include <thrust/replace.h>
#include "nppi_geometry_transforms.h"
#include "cublas_v2.h"

#include "src/Utilities/my_cuda_helper.cuh"
#include "src/cu_LC2/cu_lc2_2d.cuh"
#include "src/cu_LC2/cu_lc2_helper.cuh"

bool CULC2_2D_ONETIME_SHOW = true; 

// -----------------------------------------------Device Functions---------------------------------------------------------
__global__ void gradient2DSobelKernel(
    //Inputs:
    float *HighResolution, 
    int2 ImageDim, 
    //Output:
    float *HighResolutionGradient)
{
    int col_idx = threadIdx.x + blockIdx.x * blockDim.x;
    int row_idx = threadIdx.y + blockIdx.y * blockDim.y;
    
    //Declare shared memory pointers:
    extern __shared__ float dynSharedMem[];
    float *HighResolution_shared = dynSharedMem;

    //Restrict threads within the valid volume dimension:
    //blocks iterate within shared memory to Copy:
    for(int moving_blk_row_it = 0; moving_blk_row_it < (int)ceil((2 * 1 + blockDim.y) / (float)blockDim.y); ++moving_blk_row_it){
        for(int moving_blk_col_it = 0; moving_blk_col_it < (int)ceil((2 * 1 + blockDim.x) / (float)blockDim.x); ++moving_blk_col_it){
            int shared_col_idx = threadIdx.x + moving_blk_col_it * blockDim.x;
            int shared_row_idx = threadIdx.y + moving_blk_row_it * blockDim.y;

            //Ensure sharedMem iterators are within valid range:
            if(shared_col_idx < (2 * 1 + blockDim.x) && shared_row_idx < (2 * 1 + blockDim.y)){
                int shared_col_idx_vol = (col_idx - threadIdx.x - 1) + shared_col_idx;
                int shared_row_idx_vol = (row_idx - threadIdx.y - 1) + shared_row_idx;
                //Ensure threads access the valid image:
                if(shared_col_idx_vol >= 0 && shared_row_idx_vol >= 0 && shared_col_idx_vol < ImageDim.x && shared_row_idx_vol < ImageDim.y){
                    HighResolution_shared[shared_col_idx + shared_row_idx * (2 * 1 + blockDim.x)] = 
                    HighResolution[shared_col_idx_vol + shared_row_idx_vol * ImageDim.x];
                }
                else{
                    HighResolution_shared[shared_col_idx + shared_row_idx * (2 * 1 + blockDim.x)] = 0.0f;
                }
            }
        }
    }

    __syncthreads(); //wait all threads finish copying...

    //Perform convoluting: 
    if(col_idx < ImageDim.x && row_idx < ImageDim.y){
        float conv_x = 0.0f, conv_y = 0.0f;
        for(int filter_row_idx = 0; filter_row_idx < 3; ++filter_row_idx){
            for(int filter_col_idx = 0; filter_col_idx < 3; ++filter_col_idx){
                conv_x += 
                Sobel_X_dConstant[filter_col_idx + filter_row_idx * 3] * 
                HighResolution_shared[ 
                    (threadIdx.x + filter_col_idx) + 
                    (threadIdx.y + filter_row_idx) * (blockDim.x + 2 * 1)];

                conv_y += 
                Sobel_Y_dConstant[filter_col_idx + filter_row_idx * 3] * 
                HighResolution_shared[ 
                    (threadIdx.x + filter_col_idx) + 
                    (threadIdx.y + filter_row_idx) * (blockDim.x + 2 * 1)];
            }
        }
        //Calculate gradient magnitude: 
        HighResolutionGradient[col_idx + row_idx * ImageDim.x] = sqrt(conv_x * conv_x + conv_y * conv_y);
    }
    __syncthreads(); //wait all threads finish calculating...
}

__global__ void calculate2DWeightingFactorOnSharedMemory( 
    //Inputs:
    float* Ultrasound, 
    int2 ImageDim, 
    const int patchSize, 
    //Output:
    float* WeightingFactor)
{
    int col_idx = threadIdx.x + blockIdx.x * blockDim.x;
    int row_idx = threadIdx.y + blockIdx.y * blockDim.y;

    //Declare shared memory pointers:
    extern __shared__ float dynSharedMem[];
    float *UltrasoundPatches_shared = dynSharedMem;

    //blocks iterate within shared memory to Copy:
    for(int moving_blk_row_it = 0; moving_blk_row_it < (int)ceil((2 * patchSize + blockDim.y) / (float)blockDim.y); ++moving_blk_row_it){
        for(int moving_blk_col_it = 0; moving_blk_col_it < (int)ceil((2 * patchSize + blockDim.x) / (float)blockDim.x); ++moving_blk_col_it){
            int shared_col_idx = threadIdx.x + moving_blk_col_it * blockDim.x;
            int shared_row_idx = threadIdx.y + moving_blk_row_it * blockDim.y;

            //Ensure sharedMem iterators are within valid range:
            if(shared_col_idx < (2 * patchSize + blockDim.x) && shared_row_idx < (2 * patchSize + blockDim.y)){
                int shared_col_idx_vol = (col_idx - threadIdx.x - patchSize) + shared_col_idx;
                int shared_row_idx_vol = (row_idx - threadIdx.y - patchSize) + shared_row_idx;
                //Ensure threads access the valid image:
                if(shared_col_idx_vol >= 0 && shared_row_idx_vol >= 0 && shared_col_idx_vol < ImageDim.x && shared_row_idx_vol < ImageDim.y){
                    UltrasoundPatches_shared[shared_col_idx + shared_row_idx * (2 * patchSize + blockDim.x)] = 
                    Ultrasound[shared_col_idx_vol + shared_row_idx_vol * ImageDim.x];
                }
                else{
                    UltrasoundPatches_shared[shared_col_idx + shared_row_idx * (2 * patchSize + blockDim.x)] = 0.0f;
                }
            }
        }
    }

    __syncthreads(); //wait all threads finish copying...

    //Iterate all voxel elements:
    //Limite for valid threads within volume:
    if(col_idx < ImageDim.x && row_idx < ImageDim.y){
        int patchDiameter = patchSize * 2 + 1;
        int Num_NNZ = 0; 
        float Sum_NNZ = 0.0f;
        for(int patch_row_idx = 0; patch_row_idx < patchDiameter; ++patch_row_idx){
            for(int patch_col_idx = 0; patch_col_idx < patchDiameter; ++patch_col_idx){
                if(UltrasoundPatches_shared[ 
                    (threadIdx.x + patch_col_idx) + 
                    (threadIdx.y + patch_row_idx) * (blockDim.x + 2 * patchSize)] > ZERO_THRESHOLD){
                    
                    ++Num_NNZ;
                    Sum_NNZ += UltrasoundPatches_shared[ 
                        (threadIdx.x + patch_col_idx) + 
                        (threadIdx.y + patch_row_idx) * (blockDim.x + 2 * patchSize)];
                }
            }
        }
        //Select the FullSized patches:
        if(Num_NNZ == patchDiameter * patchDiameter){
            float Mean_NNZ = Sum_NNZ / Num_NNZ;
            float Var_NNZ = 0.0f;
            for(int patch_row_idx = 0; patch_row_idx < patchDiameter; ++patch_row_idx){
                for(int patch_col_idx = 0; patch_col_idx < patchDiameter; ++patch_col_idx){
                    if(UltrasoundPatches_shared[ 
                        (threadIdx.x + patch_col_idx) + 
                        (threadIdx.y + patch_row_idx) * (blockDim.x + 2 * patchSize)] > ZERO_THRESHOLD){
                        
                        Var_NNZ += 
                        (UltrasoundPatches_shared[ 
                            (threadIdx.x + patch_col_idx) + 
                            (threadIdx.y + patch_row_idx) * (blockDim.x + 2 * patchSize)] - Mean_NNZ) * 
                        (UltrasoundPatches_shared[ 
                            (threadIdx.x + patch_col_idx) + 
                            (threadIdx.y + patch_row_idx) * (blockDim.x + 2 * patchSize)] - Mean_NNZ);
                    }
                }
            }
            //Calculate standard deviation: 
            Var_NNZ /= (Num_NNZ - 1);
            if(Var_NNZ > ZERO_THRESHOLD){
                WeightingFactor[col_idx + row_idx * ImageDim.x] = sqrt(Var_NNZ);
            }
            else{
                WeightingFactor[col_idx + row_idx * ImageDim.x] = 0.0f;
            }
        }
        else{
            WeightingFactor[col_idx + row_idx * ImageDim.x] = 0.0f;
        }
    }
    __syncthreads(); //wait all threads finish calculating...
}

__device__ float conjugateGradient_2Dkernel( 
    float *y, float *C_1, float *C_2, 
    float *xmin, const int patchDiameter, const int patchSize, const int maxit)
{
    //give a start guess of x:
    xmin[0] = 0.0f; xmin[1] = 0.0f; xmin[2] = 0.0f; 

    //Calculate initial residual:
    float g0[6] = {0.0f};
    for(int i = 0; i < patchDiameter * patchDiameter; ++i){
        int row_idx = i / patchDiameter;
        int column_idx = i % patchDiameter;

        float temp_res = 
        //Residuals:
        (
            y[ 
            (threadIdx.x + column_idx) + 
            (threadIdx.y + row_idx) * (blockDim.x + 2 * patchSize)] - 
            (
                xmin[0] * 
                C_1[ 
                (threadIdx.x + column_idx) + 
                (threadIdx.y + row_idx) * (blockDim.x + 2 * patchSize)] + 
                xmin[1] * 
                C_2[ 
                (threadIdx.x + column_idx) + 
                (threadIdx.y + row_idx) * (blockDim.x + 2 * patchSize)] + 
                xmin[2] * 1.0f
            )
        );

        g0[0] += 
        C_1[ 
        (threadIdx.x + column_idx) + 
        (threadIdx.y + row_idx) * (blockDim.x + 2 * patchSize)] * temp_res;
        
        g0[1] += 
        C_2[ 
        (threadIdx.x + column_idx) + 
        (threadIdx.y + row_idx) * (blockDim.x + 2 * patchSize)] * temp_res;

        g0[2] += 1.0f * temp_res;
    }

    //Main loop:
    float beta = 0.0f;
    float pi[3] = {0.0f};
    float alpha = 0.0f;

    for(int loop_it = 1; loop_it <= maxit; ++ loop_it){
        if( sqrt(g0[0] * g0[0] + g0[1] * g0[1] + g0[2] * g0[2]) < 1e-6 ){
            break; 
        }
        if(loop_it > 1){
            beta = -(g0[0] * g0[0] + g0[1] * g0[1] + g0[2] * g0[2]) / (g0[3] * g0[3] + g0[4] * g0[4] + g0[5] * g0[5]);
        }
        if(loop_it == 1){
            pi[0] = g0[0]; pi[1] = g0[1]; pi[2] = g0[2]; 
        }
        else{
            pi[0] = g0[0] - beta * pi[0]; pi[1] = g0[1] - beta * pi[1]; pi[2] = g0[2] - beta * pi[2]; 
        }

        //calculate Cpi:
        float normSqCpi = 0.0f;
        for(int i = 0; i < patchDiameter * patchDiameter; ++i){
            int row_idx = i / patchDiameter;
            int column_idx = i % patchDiameter;
            
            normSqCpi += 
            ( 
                C_1[ 
                (threadIdx.x + column_idx) + 
                (threadIdx.y + row_idx) * (blockDim.x + 2 * patchSize)] * pi[0] + 
                C_2[ 
                (threadIdx.x + column_idx) + 
                (threadIdx.y + row_idx) * (blockDim.x + 2 * patchSize)] * pi[1] + 
                1.0f * pi[2]
            ) * 
            (
                C_1[ 
                (threadIdx.x + column_idx) + 
                (threadIdx.y + row_idx) * (blockDim.x + 2 * patchSize)] * pi[0] + 
                C_2[ 
                (threadIdx.x + column_idx) + 
                (threadIdx.y + row_idx) * (blockDim.x + 2 * patchSize)] * pi[1] + 
                1.0f * pi[2]
            );
        }
        alpha = (g0[0] * g0[0] + g0[1] * g0[1] + g0[2] * g0[2]) / normSqCpi;

        //shift g0:
        g0[3] = g0[0]; g0[4] = g0[1]; g0[5] = g0[2]; 
        g0[0] = 0.0f; g0[1] = 0.0f; g0[2] = 0.0f; 
        
        //Evaluate g0 again:
        for(int i = 0; i < patchDiameter * patchDiameter; ++i){
            int row_idx = i / patchDiameter;
            int column_idx = i % patchDiameter;
            
            //Residuals:
            float temp_res = 
            (
                y[ 
                (threadIdx.x + column_idx) + 
                (threadIdx.y + row_idx) * (blockDim.x + 2 * patchSize)] - 
                (
                    xmin[0] * 
                    C_1[ 
                    (threadIdx.x + column_idx) + 
                    (threadIdx.y + row_idx) * (blockDim.x + 2 * patchSize)] + 
                    xmin[1] * 
                    C_2[ 
                    (threadIdx.x + column_idx) + 
                    (threadIdx.y + row_idx) * (blockDim.x + 2 * patchSize)] + 
                    xmin[2] * 1.0f
                )
            );

            //Cpi:
            float Cpi = 
            (
                C_1[ 
                (threadIdx.x + column_idx) + 
                (threadIdx.y + row_idx) * (blockDim.x + 2 * patchSize)] * pi[0] + 
                C_2[ 
                (threadIdx.x + column_idx) + 
                (threadIdx.y + row_idx) * (blockDim.x + 2 * patchSize)] * pi[1] + 
                1.0f * pi[2]
            );

            g0[0] += 
            C_1[ 
            (threadIdx.x + column_idx) + 
            (threadIdx.y + row_idx) * (blockDim.x + 2 * patchSize)] * 
            (temp_res - alpha * Cpi);
            
            g0[1] += 
            C_2[ 
            (threadIdx.x + column_idx) + 
            (threadIdx.y + row_idx) * (blockDim.x + 2 * patchSize)] * 
            (temp_res - alpha * Cpi);

            g0[2] += 1.0f * (temp_res - alpha * Cpi);
        }

        //Update xmin[]:
        xmin[0] = xmin[0] + alpha * pi[0];
        xmin[1] = xmin[1] + alpha * pi[1];
        xmin[2] = xmin[2] + alpha * pi[2];
    }

    //Found xmin, then calculate the mean residuals:
    float mean_res = 0.0f;
    for(int i = 0; i < patchDiameter * patchDiameter; ++i){
        int row_idx = i / patchDiameter;
        int column_idx = i % patchDiameter;

        mean_res += 
        (
            y[ 
            (threadIdx.x + column_idx) + 
            (threadIdx.y + row_idx) * (blockDim.x + 2 * patchSize)] - 
            (
                xmin[0] * 
                C_1[ 
                (threadIdx.x + column_idx) + 
                (threadIdx.y + row_idx) * (blockDim.x + 2 * patchSize)] + 
                xmin[1] * 
                C_2[ 
                (threadIdx.x + column_idx) + 
                (threadIdx.y + row_idx) * (blockDim.x + 2 * patchSize)] + 
                xmin[2] * 1.0f
            )
        );
    }
    mean_res /= (patchDiameter * patchDiameter);

    //then the variance:
    float variance = 0.0f;
    for(int i = 0; i < patchDiameter * patchDiameter; ++i){
        int row_idx = i / patchDiameter;
        int column_idx = i % patchDiameter;

        float temp_res = 
        (
            y[ 
            (threadIdx.x + column_idx) + 
            (threadIdx.y + row_idx) * (blockDim.x + 2 * patchSize)] - 
            (
                xmin[0] * 
                C_1[ 
                (threadIdx.x + column_idx) + 
                (threadIdx.y + row_idx) * (blockDim.x + 2 * patchSize)] + 
                xmin[1] * 
                C_2[ 
                (threadIdx.x + column_idx) + 
                (threadIdx.y + row_idx) * (blockDim.x + 2 * patchSize)] + 
                xmin[2] * 1.0f
            )
        );

        variance += (mean_res - temp_res) * (mean_res - temp_res);
    }
    variance = variance / ((patchDiameter * patchDiameter) - 1);

    return variance;
}

__global__ void calculate2DWeightedSimilarityOnSharedMemory( 
    //Inputs:
    float* Ultrasound, 
    float* HighResolution, 
    float* HighResolutionGradient, 
    float* WeightingFactor, 
    int2 ImageDim, 
    const int patchSize, 
    //Output:
    float* WeightedSimilarity)
{
    int col_idx = threadIdx.x + blockIdx.x * blockDim.x;
    int row_idx = threadIdx.y + blockIdx.y * blockDim.y;

    //Declare shared memory pointers:
    extern __shared__ float dynSharedMem[];
    float *UltrasoundPatches_shared = dynSharedMem; 
    float *HighResolutionPatches_shared = dynSharedMem + (2 * patchSize + blockDim.x) * (2 * patchSize + blockDim.y);
    float *HighResolutionGradientPatches_shared = HighResolutionPatches_shared + (2 * patchSize + blockDim.x) * (2 * patchSize + blockDim.y);

    //blocks iterate within shared memory to Copy:
    for(int moving_blk_row_it = 0; moving_blk_row_it < (int)ceil((2 * patchSize + blockDim.y) / (float)blockDim.y); ++moving_blk_row_it){
        for(int moving_blk_col_it = 0; moving_blk_col_it < (int)ceil((2 * patchSize + blockDim.x) / (float)blockDim.x); ++moving_blk_col_it){
            int shared_col_idx = threadIdx.x + moving_blk_col_it * blockDim.x;
            int shared_row_idx = threadIdx.y + moving_blk_row_it * blockDim.y;

            //Ensure sharedMem iterators are within valid range:
            if(shared_col_idx < (2 * patchSize + blockDim.x) && shared_row_idx < (2 * patchSize + blockDim.y)){
                int shared_col_idx_vol = (col_idx - threadIdx.x - patchSize) + shared_col_idx;
                int shared_row_idx_vol = (row_idx - threadIdx.y - patchSize) + shared_row_idx;
                //Ensure threads access the valid image:
                if(shared_col_idx_vol >= 0 && shared_row_idx_vol >= 0 && shared_col_idx_vol < ImageDim.x && shared_row_idx_vol < ImageDim.y){
                    //Loading Ultrasound: 
                    UltrasoundPatches_shared[shared_col_idx + shared_row_idx * (2 * patchSize + blockDim.x)] = 
                    Ultrasound[shared_col_idx_vol + shared_row_idx_vol * ImageDim.x];

                    //Loading HighResolutionVolume:
                    HighResolutionPatches_shared[shared_col_idx + shared_row_idx * (2 * patchSize + blockDim.x)] = 
                    HighResolution[shared_col_idx_vol + shared_row_idx_vol * ImageDim.x];

                    //Loading HighResolutionGradient:
                    HighResolutionGradientPatches_shared[shared_col_idx + shared_row_idx * (2 * patchSize + blockDim.x)] = 
                    HighResolutionGradient[shared_col_idx_vol + shared_row_idx_vol * ImageDim.x];
                }
                else{
                    UltrasoundPatches_shared[shared_col_idx + shared_row_idx * (2 * patchSize + blockDim.x)] = 0.0f;
                    HighResolutionPatches_shared[shared_col_idx + shared_row_idx * (2 * patchSize + blockDim.x)] = 0.0f;
                    HighResolutionGradientPatches_shared[shared_col_idx + shared_row_idx * (2 * patchSize + blockDim.x)] = 0.0f;
                }
            }
        }
    }
    __syncthreads(); //wait all threads finish copying...

    //Iterate every non-zero weightingFactors, seen as the fullSized patches to calculate weightedSimilarity:
    //Limite for valid threads within volume:
    if(col_idx < ImageDim.x && row_idx < ImageDim.y){
        //Select the NON-ZERO weightingFactors:
        if(WeightingFactor[col_idx + row_idx * ImageDim.x] > ZERO_THRESHOLD){
            
            //Calculate ls-fitting using conjugate gradient method, then compute the variance of the residuals:
            float xmin[3] = {0.0f};
            float patchVariance = 0.0f;
            patchVariance = conjugateGradient_2Dkernel( 
                UltrasoundPatches_shared, 
                HighResolutionPatches_shared, 
                HighResolutionGradientPatches_shared, 
                xmin, 
                (2 * patchSize + 1), 
                patchSize, 
                20);
            
            //Calculate measures:
            float weighting = WeightingFactor[col_idx + row_idx * ImageDim.x];
            float measure = weighting - patchVariance / weighting;
            // printf("%f, %f, %f\n", weighting, patchVariance, measure);
            if(!isnan(measure) && !isinf(measure) && (measure > 0)){
                WeightedSimilarity[col_idx + row_idx * ImageDim.x] = measure;
            }
            else{
                WeightedSimilarity[col_idx + row_idx * ImageDim.x] = 0.0f;
            }
        }
        else{
            WeightedSimilarity[col_idx + row_idx * ImageDim.x] = 0.0f;
        }
    }
    __syncthreads(); //wait all threads finish calculating...
}

// -----------------------------------------------Host Functions---------------------------------------------------------

LC2_2D_class::LC2_2D_class(int x, int y, std::vector<float>& Ultrasound_Extern, std::vector<float>& HighReso_Extern){
    img_dim.x = x;
    img_dim.y = y;

    Ultrasound_static = Ultrasound_Extern;
    Ultrasound = Ultrasound_static;

    HighResolution_static = HighReso_Extern;

    HighResolutionGradient_static.resize(x * y, 0.0f);
    weightingFactor.resize(x * y, 0.0f);
    weightedSimilarity.resize(x * y, 0.0f);

    blendMask_RU.resize(x * y, 0.0f);
    blendMask_LU.resize(x * y, 1.0f);
    for(int it_row = 0; it_row < y / 2; ++it_row){
        for(int it_col = 0; it_col < x / 2; ++it_col){
            blendMask_RU[it_col + it_row * x] = 1.0f;
            blendMask_LU[it_col + it_row * x] = 0.0f;
        }
    }

    for(int it_row = y / 2; it_row < y; ++it_row){
        for(int it_col = x / 2; it_col < x; ++it_col){
            blendMask_RU[it_col + it_row * x] = 1.0f;
            blendMask_LU[it_col + it_row * x] = 0.0f;
        }
    }

}

void LC2_2D_class::PortedFromInterpolator(int x, int y){
    img_dim.x = x; 
    img_dim.y = y; 

    //Only USE: Ultrasound, HighResolution_static, HighResolutionGradient_static, blendMask_RU, blendMask_LU; 
    Ultrasound.resize(x * y, 0.0f); 
    HighResolution_static.resize(x * y, 0.0f); 
    HighResolutionGradient_static.resize(x * y, 0.0f); 
    weightingFactor.resize(x * y, 0.0f);
    weightedSimilarity.resize(x * y, 0.0f);

    blendMask_RU.resize(x * y, 0.0f);
    blendMask_LU.resize(x * y, 1.0f);
    for(int it_row = 0; it_row < y / 2; ++it_row){
        for(int it_col = 0; it_col < x / 2; ++it_col){
            blendMask_RU[it_col + it_row * x] = 1.0f;
            blendMask_LU[it_col + it_row * x] = 0.0f;
        }
    }

    for(int it_row = y / 2; it_row < y; ++it_row){
        for(int it_col = x / 2; it_col < x; ++it_col){
            blendMask_RU[it_col + it_row * x] = 1.0f;
            blendMask_LU[it_col + it_row * x] = 0.0f;
        }
    }

    image_us = cimg_library::CImg<float>(x, y, 1, 1); 
    image_high = cimg_library::CImg<float>(x, y, 1, 1); 
    mainDisp = cimg_library::CImgDisplay(image_us, "Ultrasound Image Fusion", 3); 
    mainDisp.move(100, 100); 
}

void LC2_2D_class::PrepareGradientFilter(){
    //Calculate Highresolution gradient using Sobel operator: 
    //Prepare Sobel operator:
    float hx[3] = {1.0f, 2.0f, 1.0f};
    float hy[3] = {1.0f, 2.0f, 1.0f};
    float hpx[3] = {-1.0f, 0.0f, 1.0f};
    float hpy[3] = {-1.0f, 0.0f, 1.0f};

    //Filter kernel Definition:
    //Along every dimension:
    float Sobel_X_h[9] = {0.0f}, Sobel_Y_h[9] = {0.0f};
    for(int it_y = 0; it_y < 3; ++it_y){
        for(int it_x = 0; it_x < 3; ++it_x){
            Sobel_X_h[it_x + it_y * 3] = hpx[it_x] * hy[it_y];
            Sobel_Y_h[it_x + it_y * 3] = hx[it_x] * hpy[it_y];
        }
    }

    // //Try Scharr operator:
    // Sobel_X_h[0] = -3.0f; Sobel_X_h[1] = 0.0f; Sobel_X_h[2] = 3.0f; 
    // Sobel_X_h[3] = -10.0f; Sobel_X_h[4] = 0.0f; Sobel_X_h[5] = 10.0f; 
    // Sobel_X_h[6] = -3.0f; Sobel_X_h[7] = 0.0f; Sobel_X_h[8] = 3.0f; 

    // Sobel_Y_h[0] = -3.0f; Sobel_Y_h[1] = -10.0f; Sobel_Y_h[2] = -3.0f; 
    // Sobel_Y_h[3] = 0.0f; Sobel_Y_h[4] = 0.0f; Sobel_Y_h[5] = 0.0f; 
    // Sobel_Y_h[6] = 3.0f; Sobel_Y_h[7] = 10.0f; Sobel_Y_h[8] = 3.0f; 

    //Copy filter kernel to Device Constant memory: 
    cudaMemcpyToSymbol(Sobel_X_dConstant, Sobel_X_h, 9 * sizeof(float));
    cudaMemcpyToSymbol(Sobel_Y_dConstant, Sobel_Y_h, 9 * sizeof(float));
}

void LC2_2D_class::CalculateGradient(const int Blk_Dim_x, const int Blk_Dim_y){
    //Prepare threads: 512 in total per block: 8 x 8 x 8
    dim3 BlockDim_ComputeGradient( Blk_Dim_x, Blk_Dim_y, 1);
    dim3 GridDim_ComputeGradient( 
        (int)ceil((float)(img_dim.x) / Blk_Dim_x), 
        (int)ceil((float)(img_dim.y) / Blk_Dim_y), 
        1
    );

    //Calculate shared memory size:
    int sharedMemSize = (Blk_Dim_x + 2 * 1) * (Blk_Dim_y + 2 * 1) * sizeof(float);

    gradient2DSobelKernel<<<GridDim_ComputeGradient, BlockDim_ComputeGradient, sharedMemSize>>>( 
        thrust::raw_pointer_cast(HighResolution_static.data()), 
        img_dim, 
        thrust::raw_pointer_cast(HighResolutionGradient_static.data())
    );
}

double LC2_2D_class::GetSimilarityMetric(const int Blk_Dim_x, const int Blk_Dim_y, const int patchSize){
    //Prepare threads: 512 in total per block: 8 x 8 x 8
    dim3 BlockDim_Analyse( Blk_Dim_x, Blk_Dim_y, 1);
    dim3 GridDim_Analyse( 
        (int)ceil((float)(img_dim.x) / Blk_Dim_x), 
        (int)ceil((float)(img_dim.y) / Blk_Dim_y), 
        1
    );
    
    //Calculate shared memory Unit size:
    int sharedMemSize = (Blk_Dim_x + 2 * patchSize) * (Blk_Dim_y + 2 * patchSize) * sizeof(float);
    //Memory check:
    if(sharedMemSize * 3 < 48 * 1024){
        //Implement using shared memory to reduce global memory access:
        //Step 1: 
        calculate2DWeightingFactorOnSharedMemory<<<GridDim_Analyse, BlockDim_Analyse, sharedMemSize * 1>>>( 
            thrust::raw_pointer_cast(Ultrasound.data()), 
            img_dim, 
            patchSize, 
            thrust::raw_pointer_cast(weightingFactor.data())
        );

        //Step 2:
        calculate2DWeightedSimilarityOnSharedMemory<<<GridDim_Analyse, BlockDim_Analyse, sharedMemSize * 3>>>(
            thrust::raw_pointer_cast(Ultrasound.data()), 
            thrust::raw_pointer_cast(HighResolution_static.data()), 
            thrust::raw_pointer_cast(HighResolutionGradient_static.data()), 
            thrust::raw_pointer_cast(weightingFactor.data()), 
            img_dim, 
            patchSize, 
            thrust::raw_pointer_cast(weightedSimilarity.data())
        );

    }
    else{
        std::cout << "sharedMemory is insufficient, use global memory instead. " << std::endl;
        //Implement using global memory:
        //Step 1: 
        //Step 2: 
    }
    
    //Step 3: 
    // cudaDeviceSynchronize();
    double sumWeightedSimilarity = (double)thrust::reduce(weightedSimilarity.begin(), weightedSimilarity.end(), 0.0f, thrust::plus<float>()); 
    double sumWeightingFactor = (double)thrust::reduce(weightingFactor.begin(), weightingFactor.end(), 0.0f, thrust::plus<float>());

    double Similarity = sumWeightedSimilarity / sumWeightingFactor; 
    if(isnan(Similarity) || isinf(Similarity)){
        Similarity = 0.0f;
    }

    // std::cout << "Similarity: " << Similarity << std::endl;

    return Similarity;
}

// void showImagePairs_2D(cv::cuda::GpuMat &img1, cv::cuda::GpuMat &img2, cv::cuda::GpuMat &Mask_LU, cv::cuda::GpuMat &Mask_RU) {
//     std::vector<cv::cuda::GpuMat> channels;
//     cv::cuda::GpuMat imgPair;

//     cv::cuda::GpuMat img1_normalized;
//     cv::cuda::normalize(img1, img1_normalized, 1, 0, cv::NORM_MINMAX, -1);

//     float window = 256.0f;
//     float level = 128.0f;
//     cv::cuda::GpuMat img2_normalized;
//     cv::cuda::threshold(img2, img2_normalized, (level - 0.5 * window), 0.0f, cv::THRESH_TOZERO);
//     cv::cuda::threshold(img2_normalized, img2_normalized, (level + 0.5 * window), 0.0f, cv::THRESH_TOZERO_INV);

//     cv::cuda::normalize(img2_normalized, img2_normalized, 1, 0, cv::NORM_MINMAX, -1);

//     // cv::cuda::GpuMat blendedImage;
//     // cv::cuda::multiply(img1_normalized, Mask_LU, img1_normalized);
//     // cv::cuda::multiply(img2_normalized, Mask_RU, img2_normalized);
//     // cv::cuda::add(img1_normalized, img2_normalized, blendedImage);

//     channels.push_back(img2_normalized);
//     channels.push_back(img1_normalized);
//     channels.push_back(img2_normalized);

//     cv::cuda::merge(channels, imgPair);

//     cv::Mat local_imgPair;
//     imgPair.download(local_imgPair);

//     cv::imshow("Registration Monitor", local_imgPair);
//     if(CULC2_2D_ONETIME_SHOW){
//         cv::waitKey();
//         CULC2_2D_ONETIME_SHOW = false; 
//     }
//     cv::waitKey(10);
// }

// void LC2_2D_class::ShowImages(){

//     cv::cuda::GpuMat Fixed(img_dim.y, img_dim.x, CV_32F, thrust::raw_pointer_cast(Ultrasound.data()));
//     cv::cuda::GpuMat Moving(img_dim.y, img_dim.x, CV_32F, thrust::raw_pointer_cast(HighResolution_static.data()));

//     cv::cuda::GpuMat Mask_LU(img_dim.y, img_dim.x, CV_32F, thrust::raw_pointer_cast(blendMask_LU.data()));
//     cv::cuda::GpuMat Mask_RU(img_dim.y, img_dim.x, CV_32F, thrust::raw_pointer_cast(blendMask_RU.data()));

//     showImagePairs_2D(Fixed, Moving, Mask_LU, Mask_RU);
// }

// void LC2_2D_class::ShowImages(float *ImgToShow_d){
//     cv::cuda::GpuMat CVImgToShow(img_dim.y, img_dim.x, CV_32F, ImgToShow_d);

//     cv::cuda::GpuMat CVImgToShow_normalized;
//     cv::cuda::normalize(CVImgToShow, CVImgToShow_normalized, 1, 0, cv::NORM_MINMAX, -1);

//     cv::Mat local_img;
//     CVImgToShow.download(local_img);

//     cv::imshow("Image", local_img);
//     cv::waitKey();
// }

#ifdef OPENCV_DISPLAY
void showImagePairs_2D(cv::Mat &img1, cv::Mat &img2) {
    std::vector<cv::Mat> channels;
    cv::Mat imgPair;

    cv::Mat img1_normalized;
    cv::normalize(img1, img1_normalized, 1, 0, cv::NORM_MINMAX, -1);

    cv::Mat img2_normalized;

    float window = 256.0f;
    float level = 128.0f;
    cv::threshold(img2, img2_normalized, (level - 0.5 * window), 0.0f, cv::THRESH_TOZERO);
    cv::threshold(img2_normalized, img2_normalized, (level + 0.5 * window), 0.0f, cv::THRESH_TOZERO_INV);

    cv::normalize(img2_normalized, img2_normalized, 1, 0, cv::NORM_MINMAX, -1);
    
    channels.push_back(img2_normalized);
    channels.push_back(img1_normalized);
    channels.push_back(img2_normalized);

    cv::merge(channels, imgPair);

    cv::imshow("Registration Monitor", imgPair);
    if(CULC2_2D_ONETIME_SHOW){
        cv::waitKey();
        CULC2_2D_ONETIME_SHOW = false; 
    }
    cv::waitKey(10);
}

void showImagePairs_2D(cv::Mat &img1, cv::Mat &img2, cv::Mat &Mask_LU, cv::Mat &Mask_RU) {
    std::vector<cv::Mat> channels;
    cv::Mat imgPair;

    cv::Mat img1_normalized;
    cv::normalize(img1, img1_normalized, 1, 0, cv::NORM_MINMAX, -1);

    cv::Mat img2_normalized;

    float window = 400.0f;
    float level = 64.0f;
    cv::threshold(img2, img2_normalized, (level - 0.5 * window), 0.0f, cv::THRESH_TOZERO);
    cv::threshold(img2_normalized, img2_normalized, (level + 0.5 * window), 0.0f, cv::THRESH_TOZERO_INV);

    cv::normalize(img2_normalized, img2_normalized, 1, 0, cv::NORM_MINMAX, -1);

    cv::Mat blendedImage;
    cv::multiply(img1_normalized, Mask_LU, img1_normalized);
    cv::multiply(img2_normalized, Mask_RU, img2_normalized);
    cv::add(img1_normalized, img2_normalized, blendedImage);

    cv::imshow("Registration Monitor", blendedImage);
    if(CULC2_2D_ONETIME_SHOW){
        cv::waitKey();
        CULC2_2D_ONETIME_SHOW = false; 
    }
    cv::waitKey(10);
}
#endif

void LC2_2D_class::ShowImages(int display_pattern){
    if(display_pattern == 0){
        return; 
    }
    else{
        thrust::host_vector<float> h_Ultrasound = Ultrasound; 
        thrust::host_vector<float> h_HighResolution = HighResolution_static; 

        thrust::host_vector<float> h_Mask_RU = blendMask_RU; 
        thrust::host_vector<float> h_Mask_LU = blendMask_LU; 

        image_us.assign(h_Ultrasound.data(), img_dim.x, img_dim.y, 1, 1); 
        image_us = image_us.get_equalize(256); 
        // image_us = image_us.get_norm(); 
        image_us = image_us.normalize(0, 255); 
        image_high.assign(h_HighResolution.data(), img_dim.x, img_dim.y, 1, 1); 
        image_high = image_high.cut(-136, 264); 
        // image_high = image_high.get_equalize(256); 
        // image_high = image_high.get_norm(); 
        image_high = image_high.normalize(0, 255); 

        cimg_library::CImg<float> blendedImage(img_dim.x, img_dim.y, 1, 3); 
        for(int i = 0; i < img_dim.y; ++i){
            for(int j = 0; j < img_dim.x; ++j){
                *blendedImage.data(j, i, 0, 0) = image_us[j + i * img_dim.x]; 
                *blendedImage.data(j, i, 0, 1) = image_high[j + i * img_dim.x]; 
                *blendedImage.data(j, i, 0, 2) = image_us[j + i * img_dim.x]; 
            }
        }

        if(mainDisp.is_closed()){
            mainDisp.show(); 
        }

        mainDisp.render(blendedImage);
        mainDisp.paint();

        if(CULC2_2D_ONETIME_SHOW){
            std::this_thread::sleep_for(std::chrono::seconds(5)); 
            CULC2_2D_ONETIME_SHOW = false; 
        }
    }    
}

float* LC2_2D_class::GetUltrasoundRawPointer(){
    return thrust::raw_pointer_cast(Ultrasound.data());
}

float* LC2_2D_class::GetHighResolutionRawPointer(){
    return thrust::raw_pointer_cast(HighResolution_static.data());
}

float* LC2_2D_class::GetHighResolutionGradientRawPointer(){
    return thrust::raw_pointer_cast(HighResolutionGradient_static.data());
}

void LC2_2D_class::UpdateImageAffine(double T_x, double T_y, double Rotation_ratio){

    double R = M_PI * Rotation_ratio;
    //Compute transform matrix:
    double sinR = std::sin(R);
    double cosR = std::cos(R);

    // hAffineTransform[0][0] = cosR;
    // hAffineTransform[0][1] = -sinR;
    // hAffineTransform[0][2] = - (((img_dimension.x - 1) / 2.0) *  cosR) + (((img_dimension.y - 1) / 2.0) *  sinR) + ((img_dimension.x - 1) / 2.0) + T_x * (img_dimension.x - 1);
    // hAffineTransform[1][0] = sinR;
    // hAffineTransform[1][1] = cosR;
    // hAffineTransform[1][2] = - (((img_dimension.x - 1) / 2.0) *  sinR) - (((img_dimension.y - 1) / 2.0) *  cosR) + ((img_dimension.y - 1) / 2.0) + T_y * (img_dimension.y - 1);


    hAffineTransform[0][0] = cosR;
    hAffineTransform[0][1] = -sinR;
    hAffineTransform[0][2] = - (((100 - 1) / 2.0) *  cosR) + (((100 - 1) / 2.0) *  sinR) + ((100 - 1) / 2.0) + T_x * (100 - 1);
    hAffineTransform[1][0] = sinR;
    hAffineTransform[1][1] = cosR;
    hAffineTransform[1][2] = - (((100 - 1) / 2.0) *  sinR) - (((100 - 1) / 2.0) *  cosR) + ((100 - 1) / 2.0) + T_y * (100 - 1);


    // //Copy transform to device constant memory:
    // cudaMemcpyToSymbol(dAffineTransform[0], hAffineTransform[0], 3 * sizeof(double));
    // cudaMemcpyToSymbol(dAffineTransform[1], hAffineTransform[1], 3 * sizeof(double));

    NppiSize ImageSize;
    ImageSize.width = img_dim.x; 
    ImageSize.height = img_dim.y;

    NppiRect ROISize;
    ROISize.x = 0;
    ROISize.y = 0;
    ROISize.width = img_dim.x; 
    ROISize.height = img_dim.y;

    // Moving Highresolution image:
    // thrust::device_vector<float> Moving_Dst(high_resolution_img_orig.size(), 0);

    // //perform transformation:
    // lc2_npp_stat = nppiWarpAffine_32f_C1R( 
    //     thrust::raw_pointer_cast(high_resolution_img_orig.data()), 
    //     ImageSize, 
    //     img_dimension.x * sizeof(float),
    //     ROISize, 
    //     thrust::raw_pointer_cast(Moving_Dst.data()), 
    //     img_dimension.x * sizeof(float), 
    //     ROISize, 
    //     hAffineTransform, 
    //     NPPI_INTER_LINEAR
    // );
    // assert(lc2_npp_stat == NPP_NO_ERROR);

    // //Overwrite to the current moving image:
    // high_resolution_img = Moving_Dst;

    // Moving Ultrasound image:
    thrust::device_vector<float> Moving_Dst(Ultrasound_static.size(), 0);

    //perform transformation:
    nppiWarpAffine_32f_C1R( 
        thrust::raw_pointer_cast(Ultrasound_static.data()), 
        ImageSize, 
        img_dim.x * sizeof(float),
        ROISize, 
        thrust::raw_pointer_cast(Moving_Dst.data()), 
        img_dim.x * sizeof(float), 
        ROISize, 
        hAffineTransform, 
        NPPI_INTER_LINEAR
    );

    //Overwrite to the current moving image:
    Ultrasound = Moving_Dst;
}

void LC2_2D_class::UpdateOrigImageAffine(double T_x, double T_y, double Rotation_ratio){
    double R = M_PI * Rotation_ratio;
    //Compute transform matrix:
    double sinR = std::sin(R);
    double cosR = std::cos(R);

    // hAffineTransform[0][0] = cosR;
    // hAffineTransform[0][1] = -sinR;
    // hAffineTransform[0][2] = - (((img_dimension.x - 1) / 2.0) *  cosR) + (((img_dimension.y - 1) / 2.0) *  sinR) + ((img_dimension.x - 1) / 2.0) + T_x * (img_dimension.x - 1);
    // hAffineTransform[1][0] = sinR;
    // hAffineTransform[1][1] = cosR;
    // hAffineTransform[1][2] = - (((img_dimension.x - 1) / 2.0) *  sinR) - (((img_dimension.y - 1) / 2.0) *  cosR) + ((img_dimension.y - 1) / 2.0) + T_y * (img_dimension.y - 1);

    hAffineTransform[0][0] = cosR;
    hAffineTransform[0][1] = -sinR;
    hAffineTransform[0][2] = - (((100 - 1) / 2.0) *  cosR) + (((100 - 1) / 2.0) *  sinR) + ((100 - 1) / 2.0) + T_x * (100 - 1);
    hAffineTransform[1][0] = sinR;
    hAffineTransform[1][1] = cosR;
    hAffineTransform[1][2] = - (((100 - 1) / 2.0) *  sinR) - (((100 - 1) / 2.0) *  cosR) + ((100 - 1) / 2.0) + T_y * (100 - 1);

    // //Copy transform to device constant memory:
    // cudaMemcpyToSymbol(dAffineTransform[0], hAffineTransform[0], 3 * sizeof(double));
    // cudaMemcpyToSymbol(dAffineTransform[1], hAffineTransform[1], 3 * sizeof(double));

    NppiSize ImageSize;
    ImageSize.width = img_dim.x; 
    ImageSize.height = img_dim.y;

    NppiRect ROISize;
    ROISize.x = 0;
    ROISize.y = 0;
    ROISize.width = img_dim.x; 
    ROISize.height = img_dim.y;

    // thrust::device_vector<float> Moving_Dst(high_resolution_img_orig.size(), 0);

    // //perform transformation:
    // lc2_npp_stat = nppiWarpAffine_32f_C1R( 
    //     thrust::raw_pointer_cast(high_resolution_img_orig.data()), 
    //     ImageSize, 
    //     img_dimension.x * sizeof(float),
    //     ROISize, 
    //     thrust::raw_pointer_cast(Moving_Dst.data()), 
    //     img_dimension.x * sizeof(float), 
    //     ROISize, 
    //     hAffineTransform, 
    //     NPPI_INTER_LINEAR
    // );
    // assert(lc2_npp_stat == NPP_NO_ERROR);

    // //Overwrite to the Original image:
    // high_resolution_img_orig = Moving_Dst;

    thrust::device_vector<float> Moving_Dst(Ultrasound_static.size(), 0);

    //perform transformation:
    nppiWarpAffine_32f_C1R( 
        thrust::raw_pointer_cast(Ultrasound_static.data()), 
        ImageSize, 
        img_dim.x * sizeof(float),
        ROISize, 
        thrust::raw_pointer_cast(Moving_Dst.data()), 
        img_dim.x * sizeof(float), 
        ROISize, 
        hAffineTransform, 
        NPPI_INTER_LINEAR
    );

    //Overwrite to the Original image:
    Ultrasound_static = Moving_Dst;
    Ultrasound = Moving_Dst;
}

void LC2_2D_class::SaveResults(){
    thrust::host_vector<float> OutputImage = Ultrasound;

    writeToBin(thrust::raw_pointer_cast(OutputImage.data()), img_dim.x * img_dim.y, "/home/wenhai/vsc_workspace/cu_lc2/registration.bin");
}