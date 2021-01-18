
#include <iostream>
#include <vector>
#include <fstream>
#include <chrono>
#include <cmath>
#include <assert.h>
#include <thread>

#include <cuda_runtime.h>
#include "device_launch_parameters.h"
#include <thrust/execution_policy.h>
#include "thrust/device_vector.h"
#include "thrust/count.h"
#include "thrust/extrema.h"
#include <thrust/replace.h>
#include "nppi_geometry_transforms.h"
#include "cublas_v2.h"
#include "cuda.h"

#include "src/Utilities/my_cuda_helper.cuh"
#include "src/cu_LC2/cu_lc2_3d.cuh"
#include "src/cu_LC2/cu_lc2_helper.cuh"

bool CULC2_3D_ONETIME_SHOW = true; 

// -----------------------------------------------Device Functions---------------------------------------------------------

__device__ void multiply4x4Matrix(float *outputPoint, float *inputPoint){
    float Acc_Row = 0.0f;
    for(int row_it = 0; row_it < 3; ++row_it){
        for(int col_it = 0; col_it < 4; ++col_it){
            Acc_Row = Acc_Row + transformMatrix_dConstant[col_it + row_it * 4] * inputPoint[col_it];
        }
        outputPoint[row_it] = Acc_Row;
        Acc_Row = 0.0f;
    }
}
__device__ float bi_linear_interp(float *valuesContainer, int3 vol_dim, float *inputPoint){
    float4 interp_x;
    {
        interp_x.x = 
        valuesContainer[(int)floor(inputPoint[0]) + (int)floor(inputPoint[1]) * vol_dim.x + (int)floor(inputPoint[2]) * vol_dim.x * vol_dim.y] + 
        (inputPoint[0] - floor(inputPoint[0])) * 
        (valuesContainer[(int)ceil(inputPoint[0]) + (int)floor(inputPoint[1]) * vol_dim.x + (int)floor(inputPoint[2]) * vol_dim.x * vol_dim.y] - 
        valuesContainer[(int)floor(inputPoint[0]) + (int)floor(inputPoint[1]) * vol_dim.x + (int)floor(inputPoint[2]) * vol_dim.x * vol_dim.y]);

        interp_x.y = 
        valuesContainer[(int)floor(inputPoint[0]) + (int)floor(inputPoint[1]) * vol_dim.x + (int)ceil(inputPoint[2]) * vol_dim.x * vol_dim.y] + 
        (inputPoint[0] - floor(inputPoint[0])) * 
        (valuesContainer[(int)ceil(inputPoint[0]) + (int)floor(inputPoint[1]) * vol_dim.x + (int)ceil(inputPoint[2]) * vol_dim.x * vol_dim.y] - 
        valuesContainer[(int)floor(inputPoint[0]) + (int)floor(inputPoint[1]) * vol_dim.x + (int)ceil(inputPoint[2]) * vol_dim.x * vol_dim.y]);

        interp_x.z = 
        valuesContainer[(int)floor(inputPoint[0]) + (int)ceil(inputPoint[1]) * vol_dim.x + (int)floor(inputPoint[2]) * vol_dim.x * vol_dim.y] + 
        (inputPoint[0] - floor(inputPoint[0])) * 
        (valuesContainer[(int)ceil(inputPoint[0]) + (int)ceil(inputPoint[1]) * vol_dim.x + (int)floor(inputPoint[2]) * vol_dim.x * vol_dim.y] - 
        valuesContainer[(int)floor(inputPoint[0]) + (int)ceil(inputPoint[1]) * vol_dim.x + (int)floor(inputPoint[2]) * vol_dim.x * vol_dim.y]);

        interp_x.w = 
        valuesContainer[(int)floor(inputPoint[0]) + (int)ceil(inputPoint[1]) * vol_dim.x + (int)ceil(inputPoint[2]) * vol_dim.x * vol_dim.y] + 
        (inputPoint[0] - floor(inputPoint[0])) * 
        (valuesContainer[(int)ceil(inputPoint[0]) + (int)ceil(inputPoint[1]) * vol_dim.x + (int)ceil(inputPoint[2]) * vol_dim.x * vol_dim.y] - 
        valuesContainer[(int)floor(inputPoint[0]) + (int)ceil(inputPoint[1]) * vol_dim.x + (int)ceil(inputPoint[2]) * vol_dim.x * vol_dim.y]);
    }

    float2 interp_y;
    {
        interp_y.x = interp_x.x + (inputPoint[1] - floor(inputPoint[1])) * (interp_x.z - interp_x.x);
        interp_y.y = interp_x.y + (inputPoint[1] - floor(inputPoint[1])) * (interp_x.w - interp_x.y);
    }

    return interp_y.x + (inputPoint[2] - floor(inputPoint[2])) * (interp_y.y - interp_y.x);
}
__global__ void warpKernel3D(int3 Volume_dim, float *SrcVolume, float *DstVolume){
    int col_idx = threadIdx.x + blockIdx.x * blockDim.x;
    int row_idx = threadIdx.y + blockIdx.y * blockDim.y;
    int lay_idx = threadIdx.z + blockIdx.z * blockDim.z;

    if(col_idx < Volume_dim.x && row_idx < Volume_dim.y && lay_idx < Volume_dim.z){
        //reverse-mapping the distination voxel to the source voxel:
        float dstPoint[4] = {(float)col_idx + 1.0f, (float)row_idx + 1.0f, (float)lay_idx + 1.0f, 1.0f};
        float srcPoint[4] = {0.0f, 0.0f, 0.0f, 1.0f};
        multiply4x4Matrix(srcPoint, dstPoint);
        srcPoint[0] -= 1.0f; srcPoint[1] -= 1.0f; srcPoint[2] -= 1.0f; 

        //Extract the reversed Voxel in source volume, then apply it to distination volume:
        //bi-linear:
        if(srcPoint[0] >= 0.0f && srcPoint[1] >= 0.0f & srcPoint[2] >= 0.0f && 
           srcPoint[0] <= Volume_dim.x - 1 && srcPoint[1] <= Volume_dim.y - 1 & srcPoint[2] <= Volume_dim.z - 1){
            DstVolume[col_idx + row_idx * Volume_dim.x + lay_idx * Volume_dim.x * Volume_dim.y] 
            = bi_linear_interp(SrcVolume, Volume_dim, srcPoint);
        }
        else{
            DstVolume[col_idx + row_idx * Volume_dim.x + lay_idx * Volume_dim.x * Volume_dim.y] = 0.0f;
        }
    }
}

__global__ void gradient3DSobelKernel(
    //Inputs:
    float *HighResolutionVolume, 
    int3 VolumeDim, 
    //Output:
    float *HighResolutionGradient)
{
    int col_idx = threadIdx.x + blockIdx.x * blockDim.x;
    int row_idx = threadIdx.y + blockIdx.y * blockDim.y;
    int pag_idx = threadIdx.z + blockIdx.z * blockDim.z;
    
    //Declare shared memory pointers:
    extern __shared__ float dynSharedMem[];
    float *HighResolution_shared = dynSharedMem;

    //Restrict threads within the valid volume dimension:
    // if(col_idx <= VolumeDim.x && row_idx <= VolumeDim.y && pag_idx <= VolumeDim.z){
        //blocks iterate within shared memory to Copy:
        for(int moving_blk_pag_it = 0; moving_blk_pag_it < (int)ceil((2 * 1 + blockDim.z) / (float)blockDim.z); ++moving_blk_pag_it){
            for(int moving_blk_row_it = 0; moving_blk_row_it < (int)ceil((2 * 1 + blockDim.y) / (float)blockDim.y); ++moving_blk_row_it){
                for(int moving_blk_col_it = 0; moving_blk_col_it < (int)ceil((2 * 1 + blockDim.x) / (float)blockDim.x); ++moving_blk_col_it){
                    int shared_col_idx = threadIdx.x + moving_blk_col_it * blockDim.x;
                    int shared_row_idx = threadIdx.y + moving_blk_row_it * blockDim.y;
                    int shared_pag_idx = threadIdx.z + moving_blk_pag_it * blockDim.z;

                    //Ensure sharedMem iterators are within valid range:
                    if(shared_col_idx < (2 * 1 + blockDim.x) && shared_row_idx < (2 * 1 + blockDim.y) && shared_pag_idx < (2 * 1 + blockDim.z)){
                        int shared_col_idx_vol = (col_idx - threadIdx.x - 1) + shared_col_idx;
                        int shared_row_idx_vol = (row_idx - threadIdx.y - 1) + shared_row_idx;
                        int shared_pag_idx_vol = (pag_idx - threadIdx.z - 1) + shared_pag_idx;
                        //Ensure threads access the valid image:
                        if(shared_col_idx_vol >= 0 && shared_row_idx_vol >= 0 && shared_pag_idx_vol >= 0 && shared_col_idx_vol < VolumeDim.x && shared_row_idx_vol < VolumeDim.y && shared_pag_idx_vol < VolumeDim.z){
                            HighResolution_shared[shared_col_idx + shared_row_idx * (2 * 1 + blockDim.x) + shared_pag_idx * (2 * 1 + blockDim.x) * (2 * 1 + blockDim.y)] = 
                            HighResolutionVolume[shared_col_idx_vol + shared_row_idx_vol * VolumeDim.x + shared_pag_idx_vol * VolumeDim.x * VolumeDim.y];
                        }
                        else{
                            HighResolution_shared[shared_col_idx + shared_row_idx * (2 * 1 + blockDim.x) + shared_pag_idx * (2 * 1 + blockDim.x) * (2 * 1 + blockDim.y)] = 0.0f;
                        }
                    }
                }
            }
        }

        __syncthreads(); //wait all threads finish copying...

        //Perform convoluting: 
        float conv_x = 0.0f, conv_y = 0.0f, conv_z = 0.0f;
        for(int filter_pag_idx = 0; filter_pag_idx < 3; ++filter_pag_idx){
            for(int filter_row_idx = 0; filter_row_idx < 3; ++filter_row_idx){
                for(int filter_col_idx = 0; filter_col_idx < 3; ++filter_col_idx){
                    conv_x += 
                    Sobel_X_dConstant[filter_col_idx + filter_row_idx * 3 + filter_pag_idx * 3 * 3] * 
                    HighResolution_shared[ 
                        (threadIdx.x + filter_col_idx) + 
                        (threadIdx.y + filter_row_idx) * (blockDim.x + 2 * 1) + 
                        (threadIdx.z + filter_pag_idx) * (blockDim.x + 2 * 1) * (blockDim.y + 2 * 1)];

                    conv_y += 
                    Sobel_Y_dConstant[filter_col_idx + filter_row_idx * 3 + filter_pag_idx * 3 * 3] * 
                    HighResolution_shared[ 
                        (threadIdx.x + filter_col_idx) + 
                        (threadIdx.y + filter_row_idx) * (blockDim.x + 2 * 1) + 
                        (threadIdx.z + filter_pag_idx) * (blockDim.x + 2 * 1) * (blockDim.y + 2 * 1)];

                    conv_z += 
                    Sobel_Z_dConstant[filter_col_idx + filter_row_idx * 3 + filter_pag_idx * 3 * 3] * 
                    HighResolution_shared[ 
                        (threadIdx.x + filter_col_idx) + 
                        (threadIdx.y + filter_row_idx) * (blockDim.x + 2 * 1) + 
                        (threadIdx.z + filter_pag_idx) * (blockDim.x + 2 * 1) * (blockDim.y + 2 * 1)];
                }
            }
        }
        //Calculate gradient magnitude: 
        if(col_idx < VolumeDim.x && row_idx < VolumeDim.y && pag_idx < VolumeDim.z){
            HighResolutionGradient[col_idx + row_idx * VolumeDim.x + pag_idx * VolumeDim.x * VolumeDim.y] = sqrt(conv_x * conv_x + conv_y * conv_y + conv_z * conv_z);
        }
        __syncthreads(); //wait all threads finish calculating...
    // }
}

__global__ void calculate3DWeightingFactorOnSharedMemory( 
    //Inputs:
    float* UltrasoundVolume, 
    int3 VolumeDim, 
    const int patchSize, 
    //Output:
    float* WeightingFactor)
{
    int col_idx = threadIdx.x + blockIdx.x * blockDim.x;
    int row_idx = threadIdx.y + blockIdx.y * blockDim.y;
    int pag_idx = threadIdx.z + blockIdx.z * blockDim.z;

    //Declare shared memory pointers:
    extern __shared__ float dynSharedMem[];
    float *UltrasoundPatches_shared = dynSharedMem;

    //blocks iterate within shared memory to Copy:
    for(int moving_blk_pag_it = 0; moving_blk_pag_it < (int)ceil((2 * patchSize + blockDim.z) / (float)blockDim.z); ++moving_blk_pag_it){
        for(int moving_blk_row_it = 0; moving_blk_row_it < (int)ceil((2 * patchSize + blockDim.y) / (float)blockDim.y); ++moving_blk_row_it){
            for(int moving_blk_col_it = 0; moving_blk_col_it < (int)ceil((2 * patchSize + blockDim.x) / (float)blockDim.x); ++moving_blk_col_it){
                int shared_col_idx = threadIdx.x + moving_blk_col_it * blockDim.x;
                int shared_row_idx = threadIdx.y + moving_blk_row_it * blockDim.y;
                int shared_pag_idx = threadIdx.z + moving_blk_pag_it * blockDim.z;

                //Ensure sharedMem iterators are within valid range:
                if(shared_col_idx < (2 * patchSize + blockDim.x) && shared_row_idx < (2 * patchSize + blockDim.y) && shared_pag_idx < (2 * patchSize + blockDim.z)){
                    int shared_col_idx_vol = (col_idx - threadIdx.x - patchSize) + shared_col_idx;
                    int shared_row_idx_vol = (row_idx - threadIdx.y - patchSize) + shared_row_idx;
                    int shared_pag_idx_vol = (pag_idx - threadIdx.z - patchSize) + shared_pag_idx;
                    //Ensure threads access the valid image:
                    if(shared_col_idx_vol >= 0 && shared_row_idx_vol >= 0 && shared_pag_idx_vol >= 0 && shared_col_idx_vol < VolumeDim.x && shared_row_idx_vol < VolumeDim.y && shared_pag_idx_vol < VolumeDim.z){
                        UltrasoundPatches_shared[shared_col_idx + shared_row_idx * (2 * patchSize + blockDim.x) + shared_pag_idx * (2 * patchSize + blockDim.x) * (2 * patchSize + blockDim.y)] = 
                        UltrasoundVolume[shared_col_idx_vol + shared_row_idx_vol * VolumeDim.x + shared_pag_idx_vol * VolumeDim.x * VolumeDim.y];
                    }
                    else{
                        UltrasoundPatches_shared[shared_col_idx + shared_row_idx * (2 * patchSize + blockDim.x) + shared_pag_idx * (2 * patchSize + blockDim.x) * (2 * patchSize + blockDim.y)] = 0.0f;
                    }
                }
            }
        }
    }
    __syncthreads(); //wait all threads finish copying...

    //Iterate all voxel elements:
    //Limite for valid threads within volume:
    if(col_idx < VolumeDim.x && row_idx < VolumeDim.y && pag_idx < VolumeDim.z){
        int patchDiameter = patchSize * 2 + 1;
        int Num_NNZ = 0; 
        float Sum_NNZ = 0.0f;
        for(int patch_pag_idx = 0; patch_pag_idx < patchDiameter; ++patch_pag_idx){
            for(int patch_row_idx = 0; patch_row_idx < patchDiameter; ++patch_row_idx){
                for(int patch_col_idx = 0; patch_col_idx < patchDiameter; ++patch_col_idx){
                    if(UltrasoundPatches_shared[ 
                        (threadIdx.x + patch_col_idx) + 
                        (threadIdx.y + patch_row_idx) * (blockDim.x + 2 * patchSize) + 
                        (threadIdx.z + patch_pag_idx) * (blockDim.x + 2 * patchSize) * (blockDim.y + 2 * patchSize)] > ZERO_THRESHOLD){
                        
                        ++Num_NNZ;
                        Sum_NNZ += UltrasoundPatches_shared[ 
                            (threadIdx.x + patch_col_idx) + 
                            (threadIdx.y + patch_row_idx) * (blockDim.x + 2 * patchSize) + 
                            (threadIdx.z + patch_pag_idx) * (blockDim.x + 2 * patchSize) * (blockDim.y + 2 * patchSize)];
                    }
                }
            }
        }

        //Select the FullSized patches:
        if(Num_NNZ == patchDiameter * patchDiameter * patchDiameter){
            float Mean_NNZ = Sum_NNZ / Num_NNZ;
            float Var_NNZ = 0.0f;
            for(int patch_pag_idx = 0; patch_pag_idx < patchDiameter; ++patch_pag_idx){
                for(int patch_row_idx = 0; patch_row_idx < patchDiameter; ++patch_row_idx){
                    for(int patch_col_idx = 0; patch_col_idx < patchDiameter; ++patch_col_idx){
                        if(UltrasoundPatches_shared[ 
                            (threadIdx.x + patch_col_idx) + 
                            (threadIdx.y + patch_row_idx) * (blockDim.x + 2 * patchSize) + 
                            (threadIdx.z + patch_pag_idx) * (blockDim.x + 2 * patchSize) * (blockDim.y + 2 * patchSize)] > ZERO_THRESHOLD){
                            
                            Var_NNZ += 
                            (UltrasoundPatches_shared[ 
                                (threadIdx.x + patch_col_idx) + 
                                (threadIdx.y + patch_row_idx) * (blockDim.x + 2 * patchSize) + 
                                (threadIdx.z + patch_pag_idx) * (blockDim.x + 2 * patchSize) * (blockDim.y + 2 * patchSize)] - Mean_NNZ) * 
                            (UltrasoundPatches_shared[ 
                                (threadIdx.x + patch_col_idx) + 
                                (threadIdx.y + patch_row_idx) * (blockDim.x + 2 * patchSize) + 
                                (threadIdx.z + patch_pag_idx) * (blockDim.x + 2 * patchSize) * (blockDim.y + 2 * patchSize)] - Mean_NNZ);
                        }
                    }
                }
            }
            //Calculate standard deviation: 
            Var_NNZ /= (Num_NNZ - 1);
            if(Var_NNZ > ZERO_THRESHOLD){
                WeightingFactor[col_idx + row_idx * VolumeDim.x + pag_idx * VolumeDim.x * VolumeDim.y] = sqrt(Var_NNZ);
            }
            else{
                WeightingFactor[col_idx + row_idx * VolumeDim.x + pag_idx * VolumeDim.x * VolumeDim.y] = 0.0f;
            }
        }
        else{
            WeightingFactor[col_idx + row_idx * VolumeDim.x + pag_idx * VolumeDim.x * VolumeDim.y] = 0.0f;
        }
    }
    __syncthreads(); //wait all threads finish calculating...
}

__device__ float conjugateGradient_3Dkernel( 
    float *y, float *C_1, float *C_2, 
    float *xmin, const int patchDiameter, const int patchSize, const int maxit)
{
    //give a start guess of x:
    xmin[0] = 0.0f; xmin[1] = 0.0f; xmin[2] = 0.0f; 

    //Calculate initial residual:
    float g0[6] = {0.0f};
    for(int i = 0; i < patchDiameter * patchDiameter * patchDiameter; ++i){
        int page_idx = i / (patchDiameter * patchDiameter);
        int row_idx = (i % (patchDiameter * patchDiameter)) / patchDiameter;
        int column_idx = (i % (patchDiameter * patchDiameter)) % patchDiameter;

        float temp_res = 
        //Residuals:
        (
            y[ 
            (threadIdx.x + column_idx) + 
            (threadIdx.y + row_idx) * (blockDim.x + 2 * patchSize) + 
            (threadIdx.z + page_idx) * (blockDim.x + 2 * patchSize) * (blockDim.y + 2 * patchSize)] - 
            (
                xmin[0] * 
                C_1[ 
                (threadIdx.x + column_idx) + 
                (threadIdx.y + row_idx) * (blockDim.x + 2 * patchSize) + 
                (threadIdx.z + page_idx) * (blockDim.x + 2 * patchSize) * (blockDim.y + 2 * patchSize)] + 
                xmin[1] * 
                C_2[ 
                (threadIdx.x + column_idx) + 
                (threadIdx.y + row_idx) * (blockDim.x + 2 * patchSize) + 
                (threadIdx.z + page_idx) * (blockDim.x + 2 * patchSize) * (blockDim.y + 2 * patchSize)] + 
                xmin[2] * 1.0f
            )
        );

        g0[0] += 
        C_1[ 
        (threadIdx.x + column_idx) + 
        (threadIdx.y + row_idx) * (blockDim.x + 2 * patchSize) + 
        (threadIdx.z + page_idx) * (blockDim.x + 2 * patchSize) * (blockDim.y + 2 * patchSize)] * temp_res;
        
        g0[1] += 
        C_2[ 
        (threadIdx.x + column_idx) + 
        (threadIdx.y + row_idx) * (blockDim.x + 2 * patchSize) + 
        (threadIdx.z + page_idx) * (blockDim.x + 2 * patchSize) * (blockDim.y + 2 * patchSize)] * temp_res;

        g0[2] += 1.0f * temp_res;
    }

    //Main loop:
    float beta = 0.0f;
    float pi[3] = {0.0f};
    float alpha = 0.0f;

    for(int loop_it = 1; loop_it <= maxit; ++ loop_it){
        if( sqrt(g0[0] * g0[0] + g0[1] * g0[1] + g0[2] * g0[2]) < 1e-4 ){
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
        for(int i = 0; i < patchDiameter * patchDiameter * patchDiameter; ++i){
            int page_idx = i / (patchDiameter * patchDiameter);
            int row_idx = (i % (patchDiameter * patchDiameter)) / patchDiameter;
            int column_idx = (i % (patchDiameter * patchDiameter)) % patchDiameter;
            
            normSqCpi += 
            ( 
                C_1[ 
                (threadIdx.x + column_idx) + 
                (threadIdx.y + row_idx) * (blockDim.x + 2 * patchSize) + 
                (threadIdx.z + page_idx) * (blockDim.x + 2 * patchSize) * (blockDim.y + 2 * patchSize)] * pi[0] + 
                C_2[ 
                (threadIdx.x + column_idx) + 
                (threadIdx.y + row_idx) * (blockDim.x + 2 * patchSize) + 
                (threadIdx.z + page_idx) * (blockDim.x + 2 * patchSize) * (blockDim.y + 2 * patchSize)] * pi[1] + 
                1.0f * pi[2]
            ) * 
            (
                C_1[ 
                (threadIdx.x + column_idx) + 
                (threadIdx.y + row_idx) * (blockDim.x + 2 * patchSize) + 
                (threadIdx.z + page_idx) * (blockDim.x + 2 * patchSize) * (blockDim.y + 2 * patchSize)] * pi[0] + 
                C_2[ 
                (threadIdx.x + column_idx) + 
                (threadIdx.y + row_idx) * (blockDim.x + 2 * patchSize) + 
                (threadIdx.z + page_idx) * (blockDim.x + 2 * patchSize) * (blockDim.y + 2 * patchSize)] * pi[1] + 
                1.0f * pi[2]
            );
        }
        alpha = (g0[0] * g0[0] + g0[1] * g0[1] + g0[2] * g0[2]) / normSqCpi;

        //shift g0:
        g0[3] = g0[0]; g0[4] = g0[1]; g0[5] = g0[2]; 
        g0[0] = 0.0f; g0[1] = 0.0f; g0[2] = 0.0f; 
        
        //Evaluate g0 again:
        for(int i = 0; i < patchDiameter * patchDiameter * patchDiameter; ++i){
            int page_idx = i / (patchDiameter * patchDiameter);
            int row_idx = (i % (patchDiameter * patchDiameter)) / patchDiameter;
            int column_idx = (i % (patchDiameter * patchDiameter)) % patchDiameter;
            
            //Residuals:
            float temp_res = 
            (
                y[ 
                (threadIdx.x + column_idx) + 
                (threadIdx.y + row_idx) * (blockDim.x + 2 * patchSize) + 
                (threadIdx.z + page_idx) * (blockDim.x + 2 * patchSize) * (blockDim.y + 2 * patchSize)] - 
                (
                    xmin[0] * 
                    C_1[ 
                    (threadIdx.x + column_idx) + 
                    (threadIdx.y + row_idx) * (blockDim.x + 2 * patchSize) + 
                    (threadIdx.z + page_idx) * (blockDim.x + 2 * patchSize) * (blockDim.y + 2 * patchSize)] + 
                    xmin[1] * 
                    C_2[ 
                    (threadIdx.x + column_idx) + 
                    (threadIdx.y + row_idx) * (blockDim.x + 2 * patchSize) + 
                    (threadIdx.z + page_idx) * (blockDim.x + 2 * patchSize) * (blockDim.y + 2 * patchSize)] + 
                    xmin[2] * 1.0f
                )
            );

            //Cpi:
            float Cpi = 
            (
                C_1[ 
                (threadIdx.x + column_idx) + 
                (threadIdx.y + row_idx) * (blockDim.x + 2 * patchSize) + 
                (threadIdx.z + page_idx) * (blockDim.x + 2 * patchSize) * (blockDim.y + 2 * patchSize)] * pi[0] + 
                C_2[ 
                (threadIdx.x + column_idx) + 
                (threadIdx.y + row_idx) * (blockDim.x + 2 * patchSize) + 
                (threadIdx.z + page_idx) * (blockDim.x + 2 * patchSize) * (blockDim.y + 2 * patchSize)] * pi[1] + 
                1.0f * pi[2]
            );

            g0[0] += 
            C_1[ 
            (threadIdx.x + column_idx) + 
            (threadIdx.y + row_idx) * (blockDim.x + 2 * patchSize) + 
            (threadIdx.z + page_idx) * (blockDim.x + 2 * patchSize) * (blockDim.y + 2 * patchSize)] * 
            (temp_res - alpha * Cpi);
            
            g0[1] += 
            C_2[ 
            (threadIdx.x + column_idx) + 
            (threadIdx.y + row_idx) * (blockDim.x + 2 * patchSize) + 
            (threadIdx.z + page_idx) * (blockDim.x + 2 * patchSize) * (blockDim.y + 2 * patchSize)] * 
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
    for(int i = 0; i < patchDiameter * patchDiameter * patchDiameter; ++i){
        int page_idx = i / (patchDiameter * patchDiameter);
        int row_idx = (i % (patchDiameter * patchDiameter)) / patchDiameter;
        int column_idx = (i % (patchDiameter * patchDiameter)) % patchDiameter;

        mean_res += 
        (
            y[ 
            (threadIdx.x + column_idx) + 
            (threadIdx.y + row_idx) * (blockDim.x + 2 * patchSize) + 
            (threadIdx.z + page_idx) * (blockDim.x + 2 * patchSize) * (blockDim.y + 2 * patchSize)] - 
            (
                xmin[0] * 
                C_1[ 
                (threadIdx.x + column_idx) + 
                (threadIdx.y + row_idx) * (blockDim.x + 2 * patchSize) + 
                (threadIdx.z + page_idx) * (blockDim.x + 2 * patchSize) * (blockDim.y + 2 * patchSize)] + 
                xmin[1] * 
                C_2[ 
                (threadIdx.x + column_idx) + 
                (threadIdx.y + row_idx) * (blockDim.x + 2 * patchSize) + 
                (threadIdx.z + page_idx) * (blockDim.x + 2 * patchSize) * (blockDim.y + 2 * patchSize)] + 
                xmin[2] * 1.0f
            )
        );
    }
    mean_res /= (patchDiameter * patchDiameter * patchDiameter);

    //then the variance:
    float variance = 0.0f;
    for(int i = 0; i < patchDiameter * patchDiameter * patchDiameter; ++i){
        int page_idx = i / (patchDiameter * patchDiameter);
        int row_idx = (i % (patchDiameter * patchDiameter)) / patchDiameter;
        int column_idx = (i % (patchDiameter * patchDiameter)) % patchDiameter;

        float temp_res = 
        (
            y[ 
            (threadIdx.x + column_idx) + 
            (threadIdx.y + row_idx) * (blockDim.x + 2 * patchSize) + 
            (threadIdx.z + page_idx) * (blockDim.x + 2 * patchSize) * (blockDim.y + 2 * patchSize)] - 
            (
                xmin[0] * 
                C_1[ 
                (threadIdx.x + column_idx) + 
                (threadIdx.y + row_idx) * (blockDim.x + 2 * patchSize) + 
                (threadIdx.z + page_idx) * (blockDim.x + 2 * patchSize) * (blockDim.y + 2 * patchSize)] + 
                xmin[1] * 
                C_2[ 
                (threadIdx.x + column_idx) + 
                (threadIdx.y + row_idx) * (blockDim.x + 2 * patchSize) + 
                (threadIdx.z + page_idx) * (blockDim.x + 2 * patchSize) * (blockDim.y + 2 * patchSize)] + 
                xmin[2] * 1.0f
            )
        );

        variance += (mean_res - temp_res) * (mean_res - temp_res);
    }
    variance = variance / ((patchDiameter * patchDiameter * patchDiameter) - 1);

    return variance;
}

__global__ void calculate3DWeightedSimilarityOnSharedMemory( 
    //Inputs:
    float* UltrasoundVolume, 
    float* HighResolutionVolume, 
    float* HighResolutionGradientVolume, 
    float* WeightingFactorVolume, 
    int3 VolumeDim, 
    const int patchSize, 
    //Output:
    float* WeightedSimilarity)
{
    int col_idx = threadIdx.x + blockIdx.x * blockDim.x;
    int row_idx = threadIdx.y + blockIdx.y * blockDim.y;
    int pag_idx = threadIdx.z + blockIdx.z * blockDim.z;

    //Declare shared memory pointers:
    extern __shared__ float dynSharedMem[];
    float *UltrasoundPatches_shared = dynSharedMem; 
    float *HighResolutionPatches_shared = dynSharedMem + (2 * patchSize + blockDim.x) * (2 * patchSize + blockDim.y) * (2 * patchSize + blockDim.z);
    float *HighResolutionGradientPatches_shared = HighResolutionPatches_shared + (2 * patchSize + blockDim.x) * (2 * patchSize + blockDim.y) * (2 * patchSize + blockDim.z);

    //blocks iterate within shared memory to Copy:
    for(int moving_blk_pag_it = 0; moving_blk_pag_it < (int)ceil((2 * patchSize + blockDim.z) / (float)blockDim.z); ++moving_blk_pag_it){
        for(int moving_blk_row_it = 0; moving_blk_row_it < (int)ceil((2 * patchSize + blockDim.y) / (float)blockDim.y); ++moving_blk_row_it){
            for(int moving_blk_col_it = 0; moving_blk_col_it < (int)ceil((2 * patchSize + blockDim.x) / (float)blockDim.x); ++moving_blk_col_it){
                int shared_col_idx = threadIdx.x + moving_blk_col_it * blockDim.x;
                int shared_row_idx = threadIdx.y + moving_blk_row_it * blockDim.y;
                int shared_pag_idx = threadIdx.z + moving_blk_pag_it * blockDim.z;

                //Ensure sharedMem iterators are within valid range:
                if(shared_col_idx < (2 * patchSize + blockDim.x) && shared_row_idx < (2 * patchSize + blockDim.y) && shared_pag_idx < (2 * patchSize + blockDim.z)){
                    int shared_col_idx_vol = (col_idx - threadIdx.x - patchSize) + shared_col_idx;
                    int shared_row_idx_vol = (row_idx - threadIdx.y - patchSize) + shared_row_idx;
                    int shared_pag_idx_vol = (pag_idx - threadIdx.z - patchSize) + shared_pag_idx;
                    //Ensure threads access the valid image:
                    if(shared_col_idx_vol >= 0 && shared_row_idx_vol >= 0 && shared_pag_idx_vol >= 0 && shared_col_idx_vol < VolumeDim.x && shared_row_idx_vol < VolumeDim.y && shared_pag_idx_vol < VolumeDim.z){
                        //Loading Ultrasound: 
                        UltrasoundPatches_shared[shared_col_idx + shared_row_idx * (2 * patchSize + blockDim.x) + shared_pag_idx * (2 * patchSize + blockDim.x) * (2 * patchSize + blockDim.y)] = 
                        UltrasoundVolume[shared_col_idx_vol + shared_row_idx_vol * VolumeDim.x + shared_pag_idx_vol * VolumeDim.x * VolumeDim.y];

                        //Loading HighResolutionVolume:
                        HighResolutionPatches_shared[shared_col_idx + shared_row_idx * (2 * patchSize + blockDim.x) + shared_pag_idx * (2 * patchSize + blockDim.x) * (2 * patchSize + blockDim.y)] = 
                        HighResolutionVolume[shared_col_idx_vol + shared_row_idx_vol * VolumeDim.x + shared_pag_idx_vol * VolumeDim.x * VolumeDim.y];

                        //Loading HighResolutionGradient:
                        HighResolutionGradientPatches_shared[shared_col_idx + shared_row_idx * (2 * patchSize + blockDim.x) + shared_pag_idx * (2 * patchSize + blockDim.x) * (2 * patchSize + blockDim.y)] = 
                        HighResolutionGradientVolume[shared_col_idx_vol + shared_row_idx_vol * VolumeDim.x + shared_pag_idx_vol * VolumeDim.x * VolumeDim.y];
                    }
                    else{
                        UltrasoundPatches_shared[shared_col_idx + shared_row_idx * (2 * patchSize + blockDim.x) + shared_pag_idx * (2 * patchSize + blockDim.x) * (2 * patchSize + blockDim.y)] = 0.0f;
                        HighResolutionPatches_shared[shared_col_idx + shared_row_idx * (2 * patchSize + blockDim.x) + shared_pag_idx * (2 * patchSize + blockDim.x) * (2 * patchSize + blockDim.y)] = 0.0f;
                        HighResolutionGradientPatches_shared[shared_col_idx + shared_row_idx * (2 * patchSize + blockDim.x) + shared_pag_idx * (2 * patchSize + blockDim.x) * (2 * patchSize + blockDim.y)] = 0.0f;
                    }
                }
            }
        }
    }
    __syncthreads(); //wait all threads finish copying...

    //Iterate every non-zero weightingFactors, seen as the fullSized patches to calculate weightedSimilarity:
    //Limite for valid threads within volume:
    if(col_idx < VolumeDim.x && row_idx < VolumeDim.y && pag_idx < VolumeDim.z){
        //Select the NON-ZERO weightingFactors:
        if(WeightingFactorVolume[col_idx + row_idx * VolumeDim.x + pag_idx * VolumeDim.x * VolumeDim.y] > ZERO_THRESHOLD){
            
            //Calculate ls-fitting using conjugate gradient method, then compute the variance of the residuals:
            float xmin[3] = {0.0f};
            float patchVariance = 0.0f;
            patchVariance = conjugateGradient_3Dkernel( 
                UltrasoundPatches_shared, 
                HighResolutionPatches_shared, 
                HighResolutionGradientPatches_shared, 
                xmin, 
                (2 * patchSize + 1), 
                patchSize, 
                20);
            
            //Calculate measures:
            float weighting = WeightingFactorVolume[col_idx + row_idx * VolumeDim.x + pag_idx * VolumeDim.x * VolumeDim.y];
            float measure = weighting - patchVariance / weighting;
            if(!isnan(measure) && !isinf(measure) && (measure > 0)){
                WeightedSimilarity[col_idx + row_idx * VolumeDim.x + pag_idx * VolumeDim.x * VolumeDim.y] = measure;
            }
            else{
                WeightedSimilarity[col_idx + row_idx * VolumeDim.x + pag_idx * VolumeDim.x * VolumeDim.y] = 0.0f;
            }
        }
        else{
            WeightedSimilarity[col_idx + row_idx * VolumeDim.x + pag_idx * VolumeDim.x * VolumeDim.y] = 0.0f;
        }
    }
    __syncthreads(); //wait all threads finish calculating...
}

//TODO: 
__global__ void calculate3DWeightingFactorOnGlobalMemory(){

}

__device__ float conjugateGradient_3Dkernel_ForGlobalMemory(    
    float *y, float *C_1, float *C_2, 
    float *xmin, const int patchDiameter, const int patchSize, const int maxit, int3 Vol_Dim)
{
    //give a start guess of x:
    xmin[0] = 0.0f; xmin[1] = 0.0f; xmin[2] = 0.0f; 

    //Calculate initial residual:
    float g0[6] = {0.0f};
    for(int i = 0; i < patchDiameter * patchDiameter * patchDiameter; ++i){
        //All start from -P to +P: 
        int page_idx = (i / (patchDiameter * patchDiameter) - patchSize) + threadIdx.z + blockIdx.z * blockDim.z;
        int row_idx = ((i % (patchDiameter * patchDiameter)) / patchDiameter - patchSize) + threadIdx.y + blockIdx.y * blockDim.y;
        int column_idx = ((i % (patchDiameter * patchDiameter)) % patchDiameter - patchSize) + threadIdx.x + blockIdx.x * blockDim.x;

        float temp_res = y[column_idx + row_idx * Vol_Dim.x + page_idx * Vol_Dim.x * Vol_Dim.y] - (
            xmin[0] * C_1[column_idx + row_idx * Vol_Dim.x + page_idx * Vol_Dim.x * Vol_Dim.y] + 
            xmin[1] * C_2[column_idx + row_idx * Vol_Dim.x + page_idx * Vol_Dim.x * Vol_Dim.y] + 
            xmin[2] * 1.0f
        ); 

        g0[0] += C_1[column_idx + row_idx * Vol_Dim.x + page_idx * Vol_Dim.x * Vol_Dim.y] * temp_res;
        
        g0[1] += C_2[column_idx + row_idx * Vol_Dim.x + page_idx * Vol_Dim.x * Vol_Dim.y] * temp_res;

        g0[2] += 1.0f * temp_res;
    }

    //Main loop:
    float beta = 0.0f;
    float pi[3] = {0.0f};
    float alpha = 0.0f;

    for(int loop_it = 1; loop_it <= maxit; ++ loop_it){
        if( sqrt(g0[0] * g0[0] + g0[1] * g0[1] + g0[2] * g0[2]) < 1e-4 ){
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
        for(int i = 0; i < patchDiameter * patchDiameter * patchDiameter; ++i){
            //All start from -P to +P: 
            int page_idx = (i / (patchDiameter * patchDiameter) - patchSize) + threadIdx.z + blockIdx.z * blockDim.z;
            int row_idx = ((i % (patchDiameter * patchDiameter)) / patchDiameter - patchSize) + threadIdx.y + blockIdx.y * blockDim.y;
            int column_idx = ((i % (patchDiameter * patchDiameter)) % patchDiameter - patchSize) + threadIdx.x + blockIdx.x * blockDim.x;
            
            normSqCpi += 
            (
                pi[0] * C_1[column_idx + row_idx * Vol_Dim.x + page_idx * Vol_Dim.x * Vol_Dim.y] + 
                pi[1] * C_2[column_idx + row_idx * Vol_Dim.x + page_idx * Vol_Dim.x * Vol_Dim.y] + 
                pi[2] * 1.0f
            ) * 
            (
                pi[0] * C_1[column_idx + row_idx * Vol_Dim.x + page_idx * Vol_Dim.x * Vol_Dim.y] + 
                pi[1] * C_2[column_idx + row_idx * Vol_Dim.x + page_idx * Vol_Dim.x * Vol_Dim.y] + 
                pi[2] * 1.0f
            ); 
        }
        alpha = (g0[0] * g0[0] + g0[1] * g0[1] + g0[2] * g0[2]) / normSqCpi;

        //shift g0:
        g0[3] = g0[0]; g0[4] = g0[1]; g0[5] = g0[2]; 
        g0[0] = 0.0f; g0[1] = 0.0f; g0[2] = 0.0f; 

        //Evaluate g0 again:
        for(int i = 0; i < patchDiameter * patchDiameter * patchDiameter; ++i){
            //All start from -P to +P: 
            int page_idx = (i / (patchDiameter * patchDiameter) - patchSize) + threadIdx.z + blockIdx.z * blockDim.z;
            int row_idx = ((i % (patchDiameter * patchDiameter)) / patchDiameter - patchSize) + threadIdx.y + blockIdx.y * blockDim.y;
            int column_idx = ((i % (patchDiameter * patchDiameter)) % patchDiameter - patchSize) + threadIdx.x + blockIdx.x * blockDim.x;

            //Residuals:
            float temp_res = y[column_idx + row_idx * Vol_Dim.x + page_idx * Vol_Dim.x * Vol_Dim.y] - (
                xmin[0] * C_1[column_idx + row_idx * Vol_Dim.x + page_idx * Vol_Dim.x * Vol_Dim.y] + 
                xmin[1] * C_2[column_idx + row_idx * Vol_Dim.x + page_idx * Vol_Dim.x * Vol_Dim.y] + 
                xmin[2] * 1.0f
            ); 

            //Cpi:
            float Cpi = 
            (
                pi[0] * C_1[column_idx + row_idx * Vol_Dim.x + page_idx * Vol_Dim.x * Vol_Dim.y] + 
                pi[1] * C_2[column_idx + row_idx * Vol_Dim.x + page_idx * Vol_Dim.x * Vol_Dim.y] + 
                pi[2] * 1.0f
            ); 

            g0[0] += C_1[column_idx + row_idx * Vol_Dim.x + page_idx * Vol_Dim.x * Vol_Dim.y] * (temp_res - alpha * Cpi); 
            g0[1] += C_2[column_idx + row_idx * Vol_Dim.x + page_idx * Vol_Dim.x * Vol_Dim.y] * (temp_res - alpha * Cpi); 
            g0[2] += 1.0f * (temp_res - alpha * Cpi);
        }

        //Update xmin[]:
        xmin[0] = xmin[0] + alpha * pi[0];
        xmin[1] = xmin[1] + alpha * pi[1];
        xmin[2] = xmin[2] + alpha * pi[2];
    }

    //Found xmin, then calculate the mean residuals:
    float mean_res = 0.0f;
    for(int i = 0; i < patchDiameter * patchDiameter * patchDiameter; ++i){
        //All start from -P to +P: 
        int page_idx = (i / (patchDiameter * patchDiameter) - patchSize) + threadIdx.z + blockIdx.z * blockDim.z;
        int row_idx = ((i % (patchDiameter * patchDiameter)) / patchDiameter - patchSize) + threadIdx.y + blockIdx.y * blockDim.y;
        int column_idx = ((i % (patchDiameter * patchDiameter)) % patchDiameter - patchSize) + threadIdx.x + blockIdx.x * blockDim.x;
        
        mean_res += y[column_idx + row_idx * Vol_Dim.x + page_idx * Vol_Dim.x * Vol_Dim.y] - 
        (
            xmin[0] * C_1[column_idx + row_idx * Vol_Dim.x + page_idx * Vol_Dim.x * Vol_Dim.y] + 
            xmin[1] * C_2[column_idx + row_idx * Vol_Dim.x + page_idx * Vol_Dim.x * Vol_Dim.y] + 
            xmin[2] * 1.0f
        ); 
    }
    mean_res /= (patchDiameter * patchDiameter * patchDiameter);

    //then the variance:
    float variance = 0.0f;
    for(int i = 0; i < patchDiameter * patchDiameter * patchDiameter; ++i){
        //All start from -P to +P: 
        int page_idx = (i / (patchDiameter * patchDiameter) - patchSize) + threadIdx.z + blockIdx.z * blockDim.z;
        int row_idx = ((i % (patchDiameter * patchDiameter)) / patchDiameter - patchSize) + threadIdx.y + blockIdx.y * blockDim.y;
        int column_idx = ((i % (patchDiameter * patchDiameter)) % patchDiameter - patchSize) + threadIdx.x + blockIdx.x * blockDim.x;
        
        float temp_res = y[column_idx + row_idx * Vol_Dim.x + page_idx * Vol_Dim.x * Vol_Dim.y] - (
            xmin[0] * C_1[column_idx + row_idx * Vol_Dim.x + page_idx * Vol_Dim.x * Vol_Dim.y] + 
            xmin[1] * C_2[column_idx + row_idx * Vol_Dim.x + page_idx * Vol_Dim.x * Vol_Dim.y] + 
            xmin[2] * 1.0f
        );

        variance += (mean_res - temp_res) * (mean_res - temp_res);
    }

    variance = variance / ((patchDiameter * patchDiameter * patchDiameter) - 1);

    return variance; 
}
__global__ void calculate3DWeightedSimilarityOnGlobalMemory(    
    //Inputs:
    float* UltrasoundVolume, 
    float* HighResolutionVolume, 
    float* HighResolutionGradientVolume, 
    float* WeightingFactorVolume, 
    int3 VolumeDim, 
    const int patchSize, 
    //Output:
    float* WeightedSimilarity)
{
    int col_idx = threadIdx.x + blockIdx.x * blockDim.x;
    int row_idx = threadIdx.y + blockIdx.y * blockDim.y;
    int pag_idx = threadIdx.z + blockIdx.z * blockDim.z;

    //within the valid volume dimension: 
    if(col_idx < VolumeDim.x && row_idx < VolumeDim.y && pag_idx < VolumeDim.z){
        //within the valid patch-region: 
        if( col_idx >= patchSize && col_idx <= VolumeDim.x - 1 - patchSize && 
            row_idx >= patchSize && row_idx <= VolumeDim.y - 1 - patchSize && 
            pag_idx >= patchSize && pag_idx <= VolumeDim.z - 1 - patchSize){

            //Fire the main calculation: 
            if(WeightingFactorVolume[col_idx + row_idx * VolumeDim.x + pag_idx * VolumeDim.x * VolumeDim.y] > ZERO_THRESHOLD){
                //Calculate the weighted similarity measure: 
                //Step 1: using conjugate gradient method, to calculate the fitting variance. 
                float xmin[3] = {0.0f};
                float patchVariance = 0.0f;
                patchVariance = conjugateGradient_3Dkernel_ForGlobalMemory( 
                    UltrasoundVolume, 
                    HighResolutionVolume, 
                    HighResolutionGradientVolume, 
                    xmin, 
                    (2 * patchSize + 1), 
                    patchSize, 
                    20, 
                    VolumeDim);

                //Step 2: Calculate measures:
                float weighting = WeightingFactorVolume[col_idx + row_idx * VolumeDim.x + pag_idx * VolumeDim.x * VolumeDim.y];
                float measure = weighting - patchVariance / weighting;
                if(!isnan(measure) && !isinf(measure) && (measure > 0)){
                    WeightedSimilarity[col_idx + row_idx * VolumeDim.x + pag_idx * VolumeDim.x * VolumeDim.y] = measure;
                }
                else{
                    WeightedSimilarity[col_idx + row_idx * VolumeDim.x + pag_idx * VolumeDim.x * VolumeDim.y] = 0.0f;
                }
            }
            else{
                //Low local variance, then set the similarity to ZERO. 
                WeightedSimilarity[col_idx + row_idx * VolumeDim.x + pag_idx * VolumeDim.x * VolumeDim.y] = 0.0f;
            }
        }
        else{
            //Out of valid range, then set the similarity to ZERO. 
            WeightedSimilarity[col_idx + row_idx * VolumeDim.x + pag_idx * VolumeDim.x * VolumeDim.y] = 0.0f;
        }
    }
}


__global__ void TestKernel(){
    for(int it_z = 0; it_z < 3; ++it_z){
        for(int it_y = 0; it_y < 3; ++it_y){
            for(int it_x = 0; it_x < 3; ++it_x){
                printf("%f, ", Sobel_X_dConstant[it_x + it_y * 3 + it_z * 3 * 3]);
            }
            printf("\n");
        }
        printf("\n\n");
    }
    
}

__global__ void CalculateWeightingMap( 
    float* WeightingFactorVolume, 
    float* WeightedSimilarity, 
    int3 VolumeDim, 
    //Output:
    float* WeightingMap
)
{
    int col_idx = threadIdx.x + blockIdx.x * blockDim.x;
    int row_idx = threadIdx.y + blockIdx.y * blockDim.y;
    int pag_idx = threadIdx.z + blockIdx.z * blockDim.z;

    if(col_idx >= 0 && col_idx < VolumeDim.x && row_idx >= 0 && row_idx < VolumeDim.y && pag_idx >= 0 && pag_idx < VolumeDim.z){
        float weightingFactor = WeightingFactorVolume[col_idx + row_idx * VolumeDim.x + pag_idx * VolumeDim.x * VolumeDim.y]; 
        float weightedSim = WeightedSimilarity[col_idx + row_idx * VolumeDim.x + pag_idx * VolumeDim.x * VolumeDim.y]; 

        if(weightingFactor > ZERO_THRESHOLD){ // => not equal to zero: 
            WeightingMap[col_idx + row_idx * VolumeDim.x + pag_idx * VolumeDim.x * VolumeDim.y] = weightedSim / weightingFactor; 
        }
        else{
            WeightingMap[col_idx + row_idx * VolumeDim.x + pag_idx * VolumeDim.x * VolumeDim.y] = 0; 
        }
    }
}

__global__ void computeJointHistogram(
    float* _vol0, float* _vol1, 
    int3 _VolumeDim, 
    int* _histJoint, 
    const int _numV0Bins, 
    float _maxV0, float _minV0, 
    const int _numV1Bins, 
    float _maxV1, float _minV1
)
{
    int col_idx = threadIdx.x + blockIdx.x * blockDim.x;
    int row_idx = threadIdx.y + blockIdx.y * blockDim.y;
    int pag_idx = threadIdx.z + blockIdx.z * blockDim.z;

    if(col_idx >= 0 && col_idx < _VolumeDim.x && row_idx >= 0 && row_idx < _VolumeDim.y && pag_idx >= 0 && pag_idx < _VolumeDim.z){
        //get value from v0, and map: 
        float v0 = _vol0[col_idx + row_idx * _VolumeDim.x + pag_idx * _VolumeDim.x * _VolumeDim.y]; 
        int idxHistMapppedFromV0 = (int)roundf((v0 - _minV0) * (_numV0Bins - 1.0) / (_maxV0 - _minV0)); 

        //get value from v0, and map: 
        float v1 = _vol1[col_idx + row_idx * _VolumeDim.x + pag_idx * _VolumeDim.x * _VolumeDim.y]; 
        int idxHistMapppedFromV1 = (int)roundf((v1 - _minV1) * (_numV1Bins - 1.0) / (_maxV1 - _minV1)); 

        //increment: 
        if(idxHistMapppedFromV0 >= 0 && idxHistMapppedFromV0 <= (_numV0Bins - 1) && 
            idxHistMapppedFromV1 >= 0 && idxHistMapppedFromV1 <= (_numV1Bins - 1)
        )
        {
            atomicAdd(&_histJoint[idxHistMapppedFromV0 + _numV0Bins * idxHistMapppedFromV1], 1); 
        }
    }
}

__global__ void computeHistogram(
    float* _vol, int3 _VolumeDim, 
    int* _hist, 
    const int _numBins, 
    float _maxV, float _minV
)
{
    int col_idx = threadIdx.x + blockIdx.x * blockDim.x;
    int row_idx = threadIdx.y + blockIdx.y * blockDim.y;
    int pag_idx = threadIdx.z + blockIdx.z * blockDim.z;

    if(col_idx >= 0 && col_idx < _VolumeDim.x && row_idx >= 0 && row_idx < _VolumeDim.y && pag_idx >= 0 && pag_idx < _VolumeDim.z){
        //get value from v, and map: 
        float v = _vol[col_idx + row_idx * _VolumeDim.x + pag_idx * _VolumeDim.x * _VolumeDim.y]; 
        int idxHistMapppedFromV = (int)roundf((v - _minV) * (_numBins - 1.0) / (_maxV - _minV)); 

        //increment: 
        if(idxHistMapppedFromV >= 0 && idxHistMapppedFromV <= (_numBins - 1))
        {
            atomicAdd(&_hist[idxHistMapppedFromV], 1); 
        }
    }
}

// -----------------------------------------------Host Functions---------------------------------------------------------
LC2_3D_class::LC2_3D_class(int x, int y, int z, std::vector<float>& UltrasoundVolume_Extern, std::vector<float>& HighResoVolume_Extern){
    vol_dim.x = x;
    vol_dim.y = y;
    vol_dim.z = z;

    Ultrasound_volume_static = UltrasoundVolume_Extern ;
    Ultrasound_volume = Ultrasound_volume_static;
    HighResolution_volume_static = HighResoVolume_Extern;
    HighResolutionGradient_volume_static.resize(x * y * z, 0.0f);
    weightingFactor_volume.resize(x * y * z, 0.0f);
    weightedSimilarity_volume.resize(x * y * z, 0.0f);
    WeightingMap.resize(x * y * z, 0.0f);

    IsGradientReady = false;
}

void LC2_3D_class::PortedFromInterpolator(int x, int y, int z){
    vol_dim.x = x;
    vol_dim.y = y;
    vol_dim.z = z;

    Ultrasound_volume.resize(x * y * z, 0.0f); 
    HighResolution_volume_static.resize(x * y * z, 0.0f); 
    HighResolutionGradient_volume_static.resize(x * y * z, 0.0f); 
    
    weightingFactor_volume.resize(x * y * z, 0.0f); 
    weightedSimilarity_volume.resize(x * y * z, 0.0f); 
    WeightingMap.resize(x * y * z, 0.0f);

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
    mainDisp = cimg_library::CImgDisplay(image_us, "Registration Moniror", 3); 
    mainDisp.move(100, 100); 
}

void LC2_3D_class::PrepareGradientFilter(){
    //Calculate Highresolution gradient using Sobel operator: 

    //Prepare Sobel operator:
    //FilterKernel Definition:
    float hx[3] = {1.0f, 2.0f, 1.0f};
    float hy[3] = {1.0f, 2.0f, 1.0f};
    float hz[3] = {1.0f, 2.0f, 1.0f};
    float hpx[3] = {-1.0f, 0.0f, 1.0f};
    float hpy[3] = {-1.0f, 0.0f, 1.0f};
    float hpz[3] = {-1.0f, 0.0f, 1.0f};

    //Along every dimension:
    float Sobel_X_h[27] = {0.0f}, Sobel_Y_h[27] = {0.0f}, Sobel_Z_h[27] = {0.0f};
    for(int it_z = 0; it_z < 3; ++it_z){
        for(int it_y = 0; it_y < 3; ++it_y){
            for(int it_x = 0; it_x < 3; ++it_x){
                Sobel_X_h[it_x + it_y * 3 + it_z * 3 * 3] = hpx[it_x] * hy[it_y] * hz[it_z];
                Sobel_Y_h[it_x + it_y * 3 + it_z * 3 * 3] = hx[it_x] * hpy[it_y] * hz[it_z];
                Sobel_Z_h[it_x + it_y * 3 + it_z * 3 * 3] = hx[it_x] * hy[it_y] * hpz[it_z];
            }
        }
    }

    //Copy filter kernel to Device Constant memory: 
    cudaMemcpyToSymbol(Sobel_X_dConstant, Sobel_X_h, 27 * sizeof(float));
    cudaMemcpyToSymbol(Sobel_Y_dConstant, Sobel_Y_h, 27 * sizeof(float));
    cudaMemcpyToSymbol(Sobel_Z_dConstant, Sobel_Z_h, 27 * sizeof(float));
}

void LC2_3D_class::CalculateGradient(const int Blk_Dim_x, const int Blk_Dim_y, const int Blk_Dim_z){
    //Prepare threads: 512 in total per block: 8 x 8 x 8
    dim3 BlockDim_ComputeGradient( Blk_Dim_x, Blk_Dim_y, Blk_Dim_z);
    dim3 GridDim_ComputeGradient( 
        (int)ceil((float)(vol_dim.x) / Blk_Dim_x), 
        (int)ceil((float)(vol_dim.y) / Blk_Dim_y), 
        (int)ceil((float)(vol_dim.z) / Blk_Dim_z)
    );
    
    //Calculate shared memory size:
    int sharedMemSize = (Blk_Dim_x + 2 * 1) * (Blk_Dim_y + 2 * 1) * (Blk_Dim_z + 2 * 1) * sizeof(float);
    // //Memory check:
    // if(sharedMemSize * 4 / 1024 > 24){
        
    // }
    gradient3DSobelKernel<<<GridDim_ComputeGradient, BlockDim_ComputeGradient, sharedMemSize>>>( 
        thrust::raw_pointer_cast(HighResolution_volume_static.data()), 
        vol_dim, 
        thrust::raw_pointer_cast(HighResolutionGradient_volume_static.data())
    );

    //for debug: 
    // thrust::host_vector<float> test_4dbg = HighResolutionGradient_volume_static; 
    // writeToBin(test_4dbg.data(), vol_dim.x * vol_dim.y * vol_dim.z, "/home/wenhai/img_registration_ws/fixed.bin"); 
    
    // cudaDeviceSynchronize();
}

void LC2_3D_class::warpVolume(float Rx, float Ry, float Rz, float Tx, float Ty, float Tz, float Scaling, const int Blk_Dim_x, const int Blk_Dim_y, const int Blk_Dim_z){
    float cosRx = (float)std::cos(Rx * M_PI / 180.0f); float sinRx = (float)std::sin(Rx * M_PI / 180.0f); 
    float cosRy = (float)std::cos(Ry * M_PI / 180.0f); float sinRy = (float)std::sin(Ry * M_PI / 180.0f); 
    float cosRz = (float)std::cos(Rz * M_PI / 180.0f); float sinRz = (float)std::sin(Rz * M_PI / 180.0f); 
    
    //transform order: Rotation X->Y->Z around volume center; Translation Tx, Ty, Tz, after rotation. 
    float transformMatrix[16] = {
        //Rotation - Row 1
        cosRz * cosRy, cosRz * sinRy * sinRx - sinRz * cosRx, cosRz * sinRy * cosRx + sinRz * sinRx, 
        //Translation - along X
        (cosRz * cosRy) * (- vol_dim.x / 2.0f) + 
        (cosRz * sinRy * sinRx - sinRz * cosRx) * (- vol_dim.y / 2.0f) + 
        (cosRz * sinRy * cosRx + sinRz * sinRx) * (- vol_dim.z / 2.0f) + 
        vol_dim.x / 2.0f + 
        Tx, 
        //Rotation - Row 2
        sinRz * cosRy, sinRz * sinRy * sinRx + cosRz * cosRx, sinRz * sinRy * cosRx - cosRz * sinRx, 
        //Translation - along Y
        (sinRz * cosRy) * (- vol_dim.x / 2.0f) + 
        (sinRz * sinRy * sinRx + cosRz * cosRx) * (- vol_dim.y / 2.0f) + 
        (sinRz * sinRy * cosRx - cosRz * sinRx) * (- vol_dim.z / 2.0f) + 
        vol_dim.y / 2.0f + 
        Ty, 
        //Rotation - Row 3
        -sinRy, cosRy * sinRx, cosRy * cosRx, 
        //Translation - along Z
        (-sinRy) * (- vol_dim.x / 2.0f) + 
        (cosRy * sinRx) * (- vol_dim.y / 2.0f) + 
        (cosRy * cosRx) * (- vol_dim.z / 2.0f) + 
        vol_dim.z / 2.0f + 
        Tz, 
        0.0f, 0.0f, 0.0f, 1.0f
    };

    transformMatrix[0] *= Scaling;
    // transformMatrix[1] *= Scaling;
    // transformMatrix[2] *= Scaling;
    // transformMatrix[3] *= Scaling;

    // transformMatrix[4] *= Scaling;
    transformMatrix[5] *= Scaling;
    // transformMatrix[6] *= Scaling;
    // transformMatrix[7] *= Scaling;

    // transformMatrix[8] *= Scaling;
    // transformMatrix[9] *= Scaling;
    transformMatrix[10] *= Scaling;
    // transformMatrix[11] *= Scaling;

    float Inv_transformMatrix[16] = {0.0};
    if(InvertS4X4Matrix(transformMatrix, Inv_transformMatrix) == false){
        std::cout << "Transform matrix is invalid. " << std::endl;
        return;
    }
    Inv_transformMatrix[12] = 0.0f;
    Inv_transformMatrix[13] = 0.0f;
    Inv_transformMatrix[14] = 0.0f;
    Inv_transformMatrix[15] = 1.0f;

    cudaMemcpyToSymbol(transformMatrix_dConstant, Inv_transformMatrix, sizeof(float) * 16);

    //warp static volume, then store and overwrite into current volume:
    // dim3 BlockDim_warpVolume( BLOCK_X8_3D, BLOCK_Y8_3D, BLOCK_Z4_3D);
    // dim3 GridDim_warpVolume( 
    //     (int)ceil((float)(vol_dim.x) / BLOCK_X8_3D), 
    //     (int)ceil((float)(vol_dim.y) / BLOCK_Y8_3D), 
    //     (int)ceil((float)(vol_dim.z) / BLOCK_Z4_3D)
    // );
    dim3 BlockDim_warpVolume( Blk_Dim_x, Blk_Dim_y, Blk_Dim_z);
    dim3 GridDim_warpVolume( 
        (int)ceil((float)(vol_dim.x) / Blk_Dim_x), 
        (int)ceil((float)(vol_dim.y) / Blk_Dim_y), 
        (int)ceil((float)(vol_dim.z) / Blk_Dim_z)
    );


    warpKernel3D<<<GridDim_warpVolume, BlockDim_warpVolume>>>( 
        vol_dim, 
        thrust::raw_pointer_cast(Ultrasound_volume_static.data()), 
        thrust::raw_pointer_cast(Ultrasound_volume.data())
    );
    cudaDeviceSynchronize();
}

double LC2_3D_class::GetSimilarityMetric(const int Blk_Dim_x, const int Blk_Dim_y, const int Blk_Dim_z, const int patchSize){
    //Step 1: Calculate patched weightingFactors[vol_dim].
    //Step 2: Calculate patched weightedSimilatiry[vol_dim].
    //Step 3: Merge the results to get final Similarity.

    //Prepare threads: 512 in total per block: 8 x 8 x 8
    dim3 BlockDim_AnalyseVolume( Blk_Dim_x, Blk_Dim_y, Blk_Dim_z);
    dim3 GridDim_AnalyseVolume( 
        (int)ceil((float)(vol_dim.x) / Blk_Dim_x), 
        (int)ceil((float)(vol_dim.y) / Blk_Dim_y), 
        (int)ceil((float)(vol_dim.z) / Blk_Dim_z)
    );
    
    //Calculate shared memory Unit size:
    int sharedMemSize = (Blk_Dim_x + 2 * patchSize) * (Blk_Dim_y + 2 * patchSize) * (Blk_Dim_z + 2 * patchSize) * sizeof(float);
    //Memory check:
    if(sharedMemSize * 3 / 1024 < 48){
    // if(false){
        //Implement using shared memory to reduce global memory access:
        //Step 1: 
        calculate3DWeightingFactorOnSharedMemory<<<GridDim_AnalyseVolume, BlockDim_AnalyseVolume, sharedMemSize * 1>>>( 
            thrust::raw_pointer_cast(Ultrasound_volume.data()), 
            vol_dim, 
            patchSize, 
            thrust::raw_pointer_cast(weightingFactor_volume.data())
        );

        //Step 2:
        calculate3DWeightedSimilarityOnSharedMemory<<<GridDim_AnalyseVolume, BlockDim_AnalyseVolume, sharedMemSize * 3>>>(
            thrust::raw_pointer_cast(Ultrasound_volume.data()), 
            thrust::raw_pointer_cast(HighResolution_volume_static.data()), 
            thrust::raw_pointer_cast(HighResolutionGradient_volume_static.data()), 
            thrust::raw_pointer_cast(weightingFactor_volume.data()), 
            vol_dim, 
            patchSize, 
            thrust::raw_pointer_cast(weightedSimilarity_volume.data())
        );
    }
    else{
        // std::cout << "sharedMemory is insufficient, use global memory instead. " << std::endl;
        //Implement using global memory:
        //Step 1: 
        calculate3DWeightingFactorOnSharedMemory<<<GridDim_AnalyseVolume, BlockDim_AnalyseVolume, sharedMemSize * 1>>>( 
            thrust::raw_pointer_cast(Ultrasound_volume.data()), 
            vol_dim, 
            patchSize, 
            thrust::raw_pointer_cast(weightingFactor_volume.data())
        );
        //Step 2: 
        calculate3DWeightedSimilarityOnGlobalMemory<<<GridDim_AnalyseVolume, BlockDim_AnalyseVolume>>>(
            thrust::raw_pointer_cast(Ultrasound_volume.data()), 
            thrust::raw_pointer_cast(HighResolution_volume_static.data()), 
            thrust::raw_pointer_cast(HighResolutionGradient_volume_static.data()), 
            thrust::raw_pointer_cast(weightingFactor_volume.data()), 
            vol_dim, 
            patchSize, 
            thrust::raw_pointer_cast(weightedSimilarity_volume.data())
        );
    }
    
    //Step 3: 
    // cudaDeviceSynchronize();
    double sumWeightedSimilarity = (double)thrust::reduce(weightedSimilarity_volume.begin(), weightedSimilarity_volume.end(), 0.0f, thrust::plus<float>()); 
    double sumWeightingFactor = (double)thrust::reduce(weightingFactor_volume.begin(), weightingFactor_volume.end(), 0.0f, thrust::plus<float>());

    double Similarity = sumWeightedSimilarity / sumWeightingFactor; 
    // thrust::reduce(weightedSimilarity_volume.begin(), weightedSimilarity_volume.end(), 0.0f, thrust::plus<float>()) 
    // / 
    // thrust::reduce(weightingFactor_volume.begin(), weightingFactor_volume.end(), 0.0f, thrust::plus<float>());

    if(isnan(Similarity) || isinf(Similarity)){
        Similarity = 0.0;
    }
    // std::cout << "Similarity: " << Similarity << std::endl; 
    return Similarity;
}

double LC2_3D_class::GetMutualInformationSimilarityMetric(const int Blk_Dim_x, const int Blk_Dim_y, const int Blk_Dim_z, const int bins_usVolume, const int bins_highResoVolume){
    float maxUS = *thrust::max_element(Ultrasound_volume.begin(), Ultrasound_volume.end()); 
    float minUS = *thrust::min_element(Ultrasound_volume.begin(), Ultrasound_volume.end()); 
    float maxH = *thrust::max_element(HighResolution_volume_static.begin(), HighResolution_volume_static.end()); 
    float minH = *thrust::min_element(HighResolution_volume_static.begin(), HighResolution_volume_static.end()); 

    dim3 BlockDim_AnalyseVolume(Blk_Dim_x, Blk_Dim_y, Blk_Dim_z);
    dim3 GridDim_AnalyseVolume( 
        (int)ceil((float)(vol_dim.x) / Blk_Dim_x), 
        (int)ceil((float)(vol_dim.y) / Blk_Dim_y), 
        (int)ceil((float)(vol_dim.z) / Blk_Dim_z)
    );

    //calculate pdf for ultrasound volume: 
    float us_entropy = 0; 
    {
        // std::cout << "US: max " << maxUS << ", min "<< minUS << std::endl; 

        //histogram: 
        thrust::device_vector<int> hist_us_d(bins_usVolume, 0); 
        computeHistogram<<<GridDim_AnalyseVolume, BlockDim_AnalyseVolume>>>(
            thrust::raw_pointer_cast(Ultrasound_volume.data()), 
            vol_dim, 
            thrust::raw_pointer_cast(hist_us_d.data()), 
            bins_usVolume, 
            maxUS, minUS
        ); 
        // histogramCalculator(
        //     Ultrasound_volume, 
        //     hist_us_d, 
        //     bins_usVolume, 
        //     minUS, 
        //     maxUS
        // ); 

        //pdf: 
        thrust::device_vector<float> pdf_us_d(hist_us_d.size()); 
        thrust::copy(hist_us_d.begin(), hist_us_d.end(), pdf_us_d.begin()); 
        thrust::transform(pdf_us_d.begin(), pdf_us_d.end(), pdf_us_d.begin(), divide_pdf_Functor(float(vol_dim.x * vol_dim.y * vol_dim.z))); 

        //Entropy: 
        thrust::transform(pdf_us_d.begin(), pdf_us_d.end(), pdf_us_d.begin(), divide_entropy_Functor()); 
        us_entropy = -thrust::reduce(pdf_us_d.begin(), pdf_us_d.end(), 0.0f, thrust::plus<float>()); 
    }
    
    //calculate pdf for high resolution volume: 
    float high_reso_entropy = 0; 
    {
        // std::cout << "H: max " << maxH << ", min "<< minH << std::endl; 

        //histogram: 
        thrust::device_vector<int> hist_high_d(bins_highResoVolume, 0); 
        computeHistogram<<<GridDim_AnalyseVolume, BlockDim_AnalyseVolume>>>(
            thrust::raw_pointer_cast(HighResolution_volume_static.data()), 
            vol_dim, 
            thrust::raw_pointer_cast(hist_high_d.data()), 
            bins_highResoVolume, 
            maxH, minH
        ); 
        // histogramCalculator(
        //     HighResolution_volume_static, 
        //     hist_high_d, 
        //     bins_highResoVolume, 
        //     minH, 
        //     maxH
        // ); 

        //pdf: 
        thrust::device_vector<float> pdf_high_d(hist_high_d.size()); 
        thrust::copy(hist_high_d.begin(), hist_high_d.end(), pdf_high_d.begin()); 
        thrust::transform(pdf_high_d.begin(), pdf_high_d.end(), pdf_high_d.begin(), divide_pdf_Functor(float(vol_dim.x * vol_dim.y * vol_dim.z))); 

        //Entropy: 
        thrust::transform(pdf_high_d.begin(), pdf_high_d.end(), pdf_high_d.begin(), divide_entropy_Functor()); 
        high_reso_entropy = -thrust::reduce(pdf_high_d.begin(), pdf_high_d.end(), 0.0f, thrust::plus<float>()); 
    }

    //cross antropy: 
    float cross_entropy = 0; 
    {
        //create space for increment: 
        thrust::device_vector<int> histJoint(bins_usVolume * bins_highResoVolume, 0); 
        computeJointHistogram<<<GridDim_AnalyseVolume, BlockDim_AnalyseVolume>>>(
            thrust::raw_pointer_cast(Ultrasound_volume.data()), 
            thrust::raw_pointer_cast(HighResolution_volume_static.data()), 
            vol_dim, 
            thrust::raw_pointer_cast(histJoint.data()), 
            bins_usVolume, 
            maxUS, minUS, 
            bins_highResoVolume, 
            maxH, minH
        ); 

        //pdf: 
        thrust::device_vector<float> pdf_joint_d(histJoint.size()); 
        thrust::copy(histJoint.begin(), histJoint.end(), pdf_joint_d.begin()); 
        thrust::transform(pdf_joint_d.begin(), pdf_joint_d.end(), pdf_joint_d.begin(), divide_pdf_Functor(float(vol_dim.x * vol_dim.y * vol_dim.z))); 

        //Entropy: 
        thrust::transform(pdf_joint_d.begin(), pdf_joint_d.end(), pdf_joint_d.begin(), divide_entropy_Functor()); 
        cross_entropy = -thrust::reduce(pdf_joint_d.begin(), pdf_joint_d.end(), 0.0f, thrust::plus<float>()); 
    }
    
    // return us_entropy + high_reso_entropy - cross_entropy; 
    return (us_entropy + high_reso_entropy) / cross_entropy; 
}

void LC2_3D_class::GenerateWeightingMap(){
    //construct weighting map: 
    const int Blk_Dim_x = 8;
    const int Blk_Dim_y = 8;
    const int Blk_Dim_z = 8;

    dim3 BlockDim_AnalyseVolume( Blk_Dim_x, Blk_Dim_y, Blk_Dim_z);
    dim3 GridDim_AnalyseVolume( 
        (int)ceil((float)(vol_dim.x) / Blk_Dim_x), 
        (int)ceil((float)(vol_dim.y) / Blk_Dim_y), 
        (int)ceil((float)(vol_dim.z) / Blk_Dim_z)
    );

    CalculateWeightingMap<<<GridDim_AnalyseVolume, BlockDim_AnalyseVolume>>>( 
        thrust::raw_pointer_cast(weightingFactor_volume.data()), 
        thrust::raw_pointer_cast(weightedSimilarity_volume.data()), 
        vol_dim, 
        thrust::raw_pointer_cast(WeightingMap.data())
    ); 

    // thrust::host_vector<float> WeightingMap_h = WeightingMap; 
    // writeToBin(thrust::raw_pointer_cast(WeightingMap_h.data()), vol_dim.x * vol_dim.y * vol_dim.z, _path + ".raw"); 
    // thrust::host_vector<float> ultrasound_h = Ultrasound_volume; 
    // writeToBin(thrust::raw_pointer_cast(ultrasound_h.data()), vol_dim.x * vol_dim.y * vol_dim.z, _path + "_f.raw"); 
    // thrust::host_vector<float> highReso_h = HighResolution_volume_static; 
    // writeToBin(thrust::raw_pointer_cast(highReso_h.data()), vol_dim.x * vol_dim.y * vol_dim.z, _path + "_m.raw"); 
}

thrust::device_vector<float> LC2_3D_class::GetWeightedMapBuffer(){
    return WeightingMap; 
}

thrust::device_vector<float> LC2_3D_class::GetUltrasoundBuffer(){
    return Ultrasound_volume; 
}

thrust::device_vector<float> LC2_3D_class::GetHighResolutionBuffer(){
    return HighResolution_volume_static; 
}

thrust::device_vector<float> LC2_3D_class::GetHighResolutionGradientBuffer(){
    return HighResolutionGradient_volume_static; 
}

// void showImagePairs_3D(cv::cuda::GpuMat &img1, cv::cuda::GpuMat &img2, cv::cuda::GpuMat &Mask_LU, cv::cuda::GpuMat &Mask_RU) {
//     std::vector<cv::cuda::GpuMat> channels;
//     cv::cuda::GpuMat imgPair;

//     cv::cuda::GpuMat img1_normalized;
//     cv::cuda::normalize(img1, img1_normalized, 1, 0, cv::NORM_MINMAX, -1);

//     cv::cuda::GpuMat img2_normalized;
//     cv::cuda::normalize(img2, img2_normalized, 1, 0, cv::NORM_MINMAX, -1);

//     // cv::cuda::GpuMat blendedImage;
//     // cv::cuda::multiply(img1_normalized, Mask_LU, img1_normalized);
//     // cv::cuda::multiply(img2_normalized, Mask_RU, img2_normalized);
//     // cv::cuda::add(img1_normalized, img2_normalized, blendedImage);

//     channels.push_back(img2_normalized);
//     channels.push_back(img1_normalized);
//     channels.push_back(img2_normalized);

//     cv::cuda::merge(channels, imgPair);

//     cv::Mat local_imgPair;
//     // blendedImage.download(local_imgPair);
//     imgPair.download(local_imgPair);

//     cv::imshow("Registration Monitor", local_imgPair);
//     // if(CULC2_3D_ONETIME_SHOW){
//     //     cv::waitKey();
//     //     CULC2_3D_ONETIME_SHOW = false; 
//     // }
//     cv::waitKey(10);
// }

// void LC2_3D_class::ShowImages(){

//     int MiddleSlice_idx = (int)floor(vol_dim.z / 2.0) + 10; 
//     cv::cuda::GpuMat Fixed(vol_dim.y, vol_dim.x, CV_32F, thrust::raw_pointer_cast(HighResolution_volume_static.data() + MiddleSlice_idx * vol_dim.y * vol_dim.x));
//     cv::cuda::GpuMat Moving(vol_dim.y, vol_dim.x, CV_32F, thrust::raw_pointer_cast(Ultrasound_volume.data() + MiddleSlice_idx * vol_dim.y * vol_dim.x));

//     cv::cuda::GpuMat Mask_LU(vol_dim.y, vol_dim.x, CV_32F, thrust::raw_pointer_cast(blendMask_LU.data()));
//     cv::cuda::GpuMat Mask_RU(vol_dim.y, vol_dim.x, CV_32F, thrust::raw_pointer_cast(blendMask_RU.data()));

//     showImagePairs_3D(Fixed, Moving, Mask_RU, Mask_LU);
// }

#ifdef OPENCV_DISPLAY
void showImagePairs_3D(cv::Mat &img1, cv::Mat &img2) {
    std::vector<cv::Mat> channels;
    cv::Mat imgPair;

    cv::Mat img1_normalized;
    cv::normalize(img1, img1_normalized, 1, 0, cv::NORM_MINMAX, -1);

    cv::Mat img2_normalized;
    cv::normalize(img2, img2_normalized, 1, 0, cv::NORM_MINMAX, -1);

    channels.push_back(img2_normalized);
    channels.push_back(img1_normalized);
    channels.push_back(img2_normalized);

    cv::merge(channels, imgPair);

    cv::imshow("Registration Monitor", imgPair);
    if(CULC2_3D_ONETIME_SHOW){
        cv::waitKey();
        CULC2_3D_ONETIME_SHOW = false; 
    }
    cv::waitKey(10);
}

void showImagePairs_3D(cv::Mat &img1, cv::Mat &img2, cv::Mat &Mask_LU, cv::Mat &Mask_RU) {
    std::vector<cv::Mat> channels;
    cv::Mat imgPair;

    cv::Mat img1_normalized;
    cv::normalize(img1, img1_normalized, 1, 0, cv::NORM_MINMAX, -1);

    cv::Mat img2_normalized;
    cv::normalize(img2, img2_normalized, 1, 0, cv::NORM_MINMAX, -1);

    cv::Mat blendedImage;
    cv::multiply(img1_normalized, Mask_LU, img1_normalized);
    cv::multiply(img2_normalized, Mask_RU, img2_normalized);
    cv::add(img1_normalized, img2_normalized, blendedImage);

    cv::imshow("Registration Monitor", blendedImage);
    if(CULC2_3D_ONETIME_SHOW){
        cv::waitKey();
        CULC2_3D_ONETIME_SHOW = false; 
    }
    cv::waitKey(10);
}
#endif
void LC2_3D_class::ShowImages(int display_pattern){

    if(display_pattern == 0){
        return; 
    }
    else{
        int MiddleSlice_idx = (int)floor(vol_dim.z / 2.0) + 10; 

        thrust::host_vector<float> h_Ultrasound( 
            Ultrasound_volume.begin() + MiddleSlice_idx * vol_dim.y * vol_dim.x, 
            Ultrasound_volume.begin() + (MiddleSlice_idx + 1) * vol_dim.y * vol_dim.x
        ); 

        thrust::host_vector<float> h_HighResolution( 
            HighResolution_volume_static.begin() + MiddleSlice_idx * vol_dim.y * vol_dim.x, 
            HighResolution_volume_static.begin() + (MiddleSlice_idx + 1) * vol_dim.y * vol_dim.x
        ); 

        thrust::host_vector<float> h_Mask_RU( 
            blendMask_RU.begin(), 
            blendMask_RU.begin() + vol_dim.y * vol_dim.x
        ); 
        thrust::host_vector<float> h_Mask_LU( 
            blendMask_LU.begin(), 
            blendMask_LU.begin() + vol_dim.y * vol_dim.x
        ); 

        image_us.assign(h_Ultrasound.data(), vol_dim.x, vol_dim.y, 1, 1); 
        image_us = image_us.get_equalize(256); 
        image_us = image_us.get_norm(); 
        image_us = image_us.normalize(0, 255); 
        image_high.assign(h_HighResolution.data(), vol_dim.x, vol_dim.y, 1, 1); 
        image_high = image_high.get_equalize(256); 
        image_high = image_high.get_norm(); 
        image_high = image_high.normalize(0, 255); 

        cimg_library::CImg<float> blendedImage(vol_dim.x, vol_dim.y, 1, 3); 
        for(int i = 0; i < vol_dim.y; ++i){
            for(int j = 0; j < vol_dim.x; ++j){
                *blendedImage.data(j, i, 0, 0) = image_us[j + i * vol_dim.x]; 
                *blendedImage.data(j, i, 0, 1) = image_high[j + i * vol_dim.x]; 
                *blendedImage.data(j, i, 0, 2) = image_us[j + i * vol_dim.x]; 
            }
        }

        if(mainDisp.is_closed()){
            mainDisp.show(); 
        }

        mainDisp.render(blendedImage);
        mainDisp.paint();

        if(CULC2_3D_ONETIME_SHOW){
            std::this_thread::sleep_for(std::chrono::seconds(5)); 
            CULC2_3D_ONETIME_SHOW = false; 
        }
    }
}
