#ifndef INTERPOLATOR_LC2
#define INTERPOLATOR_LC2

#include <iostream>
#include <string>
#include <vector>
#include <fstream>
#include <memory>

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "thrust/device_vector.h"
#include <thrust/copy.h>

#include "yaml-cpp/yaml.h"

#include "RegistrationParametersConfig.h"

#include "src/Interpolator/spatialMappers.cuh"
#include "src/cu_LC2/cu_lc2_2d.cuh"
#include "src/cu_LC2/cu_lc2_3d.cuh"

#include "igtlOSUtil.h"
#include "igtlTransformMessage.h"
#include "igtlServerSocket.h"

//3D: 
__constant__ PhysioSpatial3DMapper Fixed3DMapper_d;
__constant__ PhysioSpatial3DMapper Moving3DMapper_d;
__constant__ PhysioSpatial3DMapper Virtual3DMapper_d;

__constant__ Spatial3DMapper VirtualToFixed3D_d;
__constant__ Spatial3DMapper VirtualToMoving3D_d;

//2D: 
__constant__ PhysioSpatial2DMapper Fixed2DMapper_d;
__constant__ PhysioSpatial2DMapper Moving2DMapper_d;
__constant__ PhysioSpatial2DMapper Virtual2DMapper_d;

__constant__ Spatial2DMapper VirtualToFixed2D_d;
__constant__ Spatial2DMapper VirtualToMoving2D_d;

class Interpolator3D{

public:
    Interpolator3D(const std::string yaml_filePath, bool _useIGTL = true);
    Interpolator3D(const LC2Configuration3D* ExternalConfig);
    ~Interpolator3D(); 
    void InitiateMappers(); //Copy PhysioSpatialMappers, 
    void InitiateRegistrationHandle(); 
    void GenerateMappers(float3 RotationAngle, float3 Translation); //Copy/Update SpatialMappers
    void GenerateMappers(float3 RotationAngle, float3 Translation, float3 scale, float3 shear);
    void GetCurrentTransform(float _t[16]); 
    void Interpolate(const int Blk_Dim_x = 8, const int Blk_Dim_y = 8, const int Blk_Dim_z = 8); //Filling Interpolation space
    double GetSimilarityMeasure(const int patchSize = 3); 
    void GetMovingToFixedTransform(float3 RotationAngle, float3 Translation, float* outMovingToFixed); 
    void GetMovingToFixedTransform(float3 RotationAngle, float3 Translation, float3 scale, float3 shear, float* outMovingToFixed); 
    void WriteOut(); 
    void WriteOut(float* OptimizedMatrix); 
    std::string GetOutputPath() { return FinalExportPath; }

private:
    std::unique_ptr<LC2_3D_class> LC2SimilarityMeasure; 

    //Raw data:
    thrust::device_vector<float> FixedVolume_d; //stay fixed
    thrust::host_vector<float> FixedVolume_h; 
    thrust::device_vector<float> MovingVolume_d; //stay fixed
    thrust::host_vector<float> MovingVolume_h; 

    thrust::device_vector<float> InterpFixedVolume_d; //updated
    thrust::host_vector<float> InterpFixedVolume_h; //updated
    thrust::device_vector<float> InterpMovingVolume_d; //updated
    thrust::host_vector<float> InterpMovingVolume_h; //updated

    //Parameters for fixed volume:
    float3 spacingFixed; 
    float3 originFixed; 
    int3 dimensionFixed; 
    std::string FixedFilePath; 

    //Parameters for moving volume:
    float3 spacingMoving; 
    float3 originMoving; 
    int3 dimensionMoving; 
    std::string MovingFilePath; 

    //Parameters for virtual volume:
    float3 spacingVirtual; 
    float3 originVirtual; 
    int3 dimensionVirtual; 

    //Host mappers:
    Spatial3DMapper VirtualToFixed3D_h;
    Spatial3DMapper VirtualToMoving3D_h;
    PhysioSpatial3DMapper Fixed3DMapper_h;
    PhysioSpatial3DMapper Moving3DMapper_h;
    PhysioSpatial3DMapper Virtual3DMapper_h;

    //Resampling Factor:
    int ResamplingFactor; 

    //Kernel parameters:
    dim3 GridDim_Interpolator;
    dim3 BlockDim_Interpolator;

    //flags:
    bool Initialized;
    bool Centered; 
    int DisplayPattern; 

    std::string HighResolutionModality; 

    //counter:
    int counter;

    //path:
    std::string InputParameterPath;
    std::string ImageFileFormat;
    std::string FinalExportPath; 

    //yaml handle:
    YAML::Node interpolator_yaml_handle;

    //OpenIgtLink: 
    bool UseIGTL; 
    igtl::TransformMessage::Pointer TransformMsg;
    igtl::ServerSocket::Pointer ServerSocket;
    igtl::Socket::Pointer CommunicationSocket; 
};

class Interpolator2D{

public: 
    Interpolator2D(const std::string yaml_filePath); 
    Interpolator2D(const LC2Configuration2D* ExternalConfig);
    void InitiateMappers();
    void InitiateRegistrationHandle(); 
    void GenerateMappers(float RotationAngle, float2 Translation); 
    void GenerateMappers(float RotationAngle, float2 Translation, float2 scale, float shear); 
    void Interpolate(const int Blk_Dim_x = 8, const int Blk_Dim_y = 8); // Host only
    double GetSimilarityMeasure(const int patchSize = 9); 
    void GetMovingToFixedTransform(float RotationAngle, float2 Translation, float* outMovingToFixed); 
    void GetMovingToFixedTransform(float RotationAngle, float2 Translation, float2 scale, float shear, float* outMovingToFixed); 
    void WriteOut(); 
    void WriteOut(float* OptimizedMatrix); 

private:

    std::unique_ptr<LC2_2D_class> LC2SimilarityMeasure; 

    //Raw data: always stay fixed
    thrust::host_vector<float> FixedImage_h;
    thrust::device_vector<float> FixedImage_d;
    thrust::host_vector<float> MovingImage_h;
    thrust::device_vector<float> MovingImage_d;
    thrust::host_vector<float> Moving_Reg_Image_h;
    thrust::device_vector<float> Moving_Reg_Image_d;

    thrust::host_vector<float> InterpFixedImage_h; //updated
    thrust::host_vector<float> InterpMovingImage_h; //updated
    thrust::device_vector<float> InterpFixedImage_d; //updated
    thrust::device_vector<float> InterpMovingImage_d; //updated

    //Parameters for fixed Image:
    float2 spacingFixed; 
    float2 originFixed; 
    int2 dimensionFixed; 
    std::string FixedImagePath; 

    //Parameters for moving Image:
    float2 spacingMoving; 
    float2 originMoving; 
    int2 dimensionMoving; 
    std::string MovingImagePath; 

    //Parameters for virtual volume:
    float2 spacingVirtual; 
    float2 originVirtual; 
    int2 dimensionVirtual; 

    //Host mappers:
    Spatial2DMapper VirtualToFixed2D_h;
    Spatial2DMapper VirtualToMoving2D_h;
    PhysioSpatial2DMapper Fixed2DMapper_h;
    PhysioSpatial2DMapper Moving2DMapper_h;
    PhysioSpatial2DMapper Virtual2DMapper_h;

    //Resampling Factor:
    int ResamplingFactor; 

    //Kernel parameters:
    dim3 GridDim_Interpolator;
    dim3 BlockDim_Interpolator;

    //flags:
    bool Initialized;
    bool Centered; 
    int DisplayPattern; 

    std::string HighResolutionModality; 

    //counter:
    int counter;

    //path:
    std::string InputParameterPath;
    std::string ImageFileFormat;
    std::string FinalExportPath; 

    //yaml handle:
    YAML::Node interpolator_yaml_handle;
};

class Interpolator2DTo3D{

public:
    Interpolator2DTo3D(const std::string yaml_filePath);
    Interpolator2DTo3D(const LC2Configuration2DTo3D* ExternalConfig);
    void InitiateMappers(); //Copy PhysioSpatialMappers, 
    void InitiateRegistrationHandle(); 
    void GenerateMappers(float3 RotationAngle, float3 Translation); //Copy/Update SpatialMappers
    void GenerateMappers(float3 RotationAngle, float3 Translation, float3 scale, float3 shear);
    void Interpolate(const int Blk_Dim_x = 8, const int Blk_Dim_y = 8, const int Blk_Dim_z = 8); //Filling Interpolation space
    double GetSimilarityMeasure(const int patchSize = 9); 
    void GetMovingToFixedTransform(float3 RotationAngle, float3 Translation, float* outMovingToFixed); 
    void GetMovingToFixedTransform(float3 RotationAngle, float3 Translation, float3 scale, float3 shear, float* outMovingToFixed); 
    void WriteOut(); 

private:
    std::unique_ptr<LC2_2D_class> LC2SimilarityMeasure; 

    //Raw data:
    thrust::device_vector<float> FixedVolume_d; //stay fixed
    thrust::host_vector<float> FixedVolume_h; 
    thrust::device_vector<float> MovingVolume_d; //stay fixed
    thrust::host_vector<float> MovingVolume_h; 

    thrust::device_vector<float> InterpFixedVolume_d; //updated
    thrust::host_vector<float> InterpFixedVolume_h; //updated
    thrust::device_vector<float> InterpMovingVolume_d; //updated
    thrust::host_vector<float> InterpMovingVolume_h; //updated

    //Parameters for fixed volume:
    float3 spacingFixed; 
    float3 originFixed; 
    int3 dimensionFixed; 
    std::string FixedFilePath; 

    //Parameters for moving volume:
    float3 spacingMoving; 
    float3 originMoving; 
    int3 dimensionMoving; 
    std::string MovingFilePath; 

    //Parameters for virtual volume:
    float3 spacingVirtual; 
    float3 originVirtual; 
    int3 dimensionVirtual; 

    //Host mappers:
    Spatial3DMapper VirtualToFixed3D_h;
    Spatial3DMapper VirtualToMoving3D_h;
    PhysioSpatial3DMapper Fixed3DMapper_h;
    PhysioSpatial3DMapper Moving3DMapper_h;
    PhysioSpatial3DMapper Virtual3DMapper_h;

    //Resampling Factor:
    int ResamplingFactor; 

    //Kernel parameters:
    dim3 GridDim_Interpolator;
    dim3 BlockDim_Interpolator;

    //flags:
    bool Initialized;
    bool Centered; 
    int DisplayPattern; 

    std::string HighResolutionModality; 

    //counter:
    int counter;

    //path:
    std::string InputParameterPath;
    std::string ImageFileFormat;
    std::string FinalExportPath; 

    //yaml handle:
    YAML::Node interpolator_yaml_handle;
};

#endif