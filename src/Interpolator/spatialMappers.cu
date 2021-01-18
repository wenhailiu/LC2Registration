#include <iostream>
#include <vector>
#include <fstream>

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include "src/Interpolator/spatialMappers.cuh"

//operator definitions: 
__host__ __device__ float2 operator-(float2 lPoint, float2 rPoint){
    float2 resultPoint; 
    resultPoint.x = lPoint.x - rPoint.x; 
    resultPoint.y = lPoint.y - rPoint.y; 
    return resultPoint; 
}

__host__ __device__ float3 operator-(float3 lPoint, float3 rPoint){
    float3 resultPoint; 
    resultPoint.x = lPoint.x - rPoint.x; 
    resultPoint.y = lPoint.y - rPoint.y; 
    resultPoint.z = lPoint.z - rPoint.z; 
    return resultPoint; 
}

__host__ __device__ float2 operator+(float2 lPoint, float2 rPoint){
    float2 resultPoint; 
    resultPoint.x = lPoint.x + rPoint.x; 
    resultPoint.y = lPoint.y + rPoint.y; 
    return resultPoint; 
}

__host__ __device__ float3 operator+(float3 lPoint, float3 rPoint){
    float3 resultPoint; 
    resultPoint.x = lPoint.x + rPoint.x; 
    resultPoint.y = lPoint.y + rPoint.y; 
    resultPoint.z = lPoint.z + rPoint.z; 
    return resultPoint; 
}

__host__ __device__ float3 operator*(float *rMatrix, float3 lPoint){ 
    float3 retPoint;
        
    retPoint.x = 0;
    retPoint.x += (rMatrix[0 + 0 * 4] * lPoint.x);
    retPoint.x += (rMatrix[1 + 0 * 4] * lPoint.y);
    retPoint.x += (rMatrix[2 + 0 * 4] * lPoint.z);
    retPoint.x += (rMatrix[3 + 0 * 4] * 1.0f);

    retPoint.y = 0;
    retPoint.y += (rMatrix[0 + 1 * 4] * lPoint.x);
    retPoint.y += (rMatrix[1 + 1 * 4] * lPoint.y);
    retPoint.y += (rMatrix[2 + 1 * 4] * lPoint.z);
    retPoint.y += (rMatrix[3 + 1 * 4] * 1.0f);

    retPoint.z = 0;
    retPoint.z += (rMatrix[0 + 2 * 4] * lPoint.x);
    retPoint.z += (rMatrix[1 + 2 * 4] * lPoint.y);
    retPoint.z += (rMatrix[2 + 2 * 4] * lPoint.z);
    retPoint.z += (rMatrix[3 + 2 * 4] * 1.0f);

    return retPoint;
}

__host__ __device__ float2 operator*(float *rMatrix, float2 lPoint){ 
    float2 retPoint;
        
    retPoint.x = 0;
    retPoint.x += (rMatrix[0 + 0 * 3] * lPoint.x);
    retPoint.x += (rMatrix[1 + 0 * 3] * lPoint.y);
    retPoint.x += (rMatrix[2 + 0 * 4] * 1.0f);

    retPoint.y = 0;
    retPoint.y += (rMatrix[0 + 1 * 3] * lPoint.x);
    retPoint.y += (rMatrix[1 + 1 * 3] * lPoint.y);
    retPoint.y += (rMatrix[2 + 1 * 3] * 1.0f);

    return retPoint;
}