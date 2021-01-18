#ifndef CULC2_3D
#define CULC2_3D

#include "thrust/device_vector.h"

#include "CImg.h"

#include "src/cu_LC2/cu_lc2_helper.cuh"

class LC2_3D_class{

public:
    //Initializer:
    LC2_3D_class(int x, int y, int z, std::vector<float>& UltrasoundVolume_Extern, std::vector<float>& HighResoVolume_Extern);
    LC2_3D_class(){}; 

    //Link from interpolator: 
    void PortedFromInterpolator(int x, int y, int z); 

    //Member functions:
    //Update current image:
    void warpVolume(float Rx, float Ry, float Rz, float Tx, float Ty, float Tz, float Scaling, const int Blk_Dim_x = 8, const int Blk_Dim_y = 8, const int Blk_Dim_z = 8);


    /*

    Following member functions only deal with the volumes with the same dimension:
    Every dimension should be less than 200 vxl.
    Maximum: 200 x 200 x 200.
    
    */
    //Calculate gradient for high resolution volume, using Sobel Operator:
    void PrepareGradientFilter(); 
    void CalculateGradient(const int Blk_Dim_x = 8, const int Blk_Dim_y = 8, const int Blk_Dim_z = 8);

    //Analyse the input volumes, counting the meaningful US voxels.
    //Simply: check if the current US voxel is zero. If it is, start counting the surrounding patch. 
    //If fullSized patch observed, then marked 1s on the countingVector, 
    //if not, then fill with zeros. 
    double GetSimilarityMetric(const int Blk_Dim_x = 8, const int Blk_Dim_y = 8, const int Blk_Dim_z = 8, const int patchSize = 3);

    double GetMutualInformationSimilarityMetric(const int Blk_Dim_x = 8, const int Blk_Dim_y = 8, const int Blk_Dim_z = 8, const int bins_usVolume = 256, const int bins_highResoVolume = 256);

    //write out the weighting map: 
    void GenerateWeightingMap(); 
    thrust::device_vector<float> GetWeightedMapBuffer(); 
    thrust::device_vector<float> GetUltrasoundBuffer(); 
    thrust::device_vector<float> GetHighResolutionBuffer(); 
    thrust::device_vector<float> GetHighResolutionGradientBuffer(); 

    //show volume slice:
    // static void showImagePairs(cv::cuda::GpuMat &img1, cv::cuda::GpuMat &img2);
    // virtual void ShowImages();

    virtual void ShowImages(int display_pattern = 0);

    //Extract pitches:
    

protected:
    //bridge volumes: 
    thrust::device_vector<float> Ultrasound_volume;
    thrust::device_vector<float> HighResolution_volume_static;

private: 
    //Volume parameters:
    int3 vol_dim;

    //Volume data:
    thrust::device_vector<float> Ultrasound_volume_static;

    //needed for external similarity calculate: 
    thrust::device_vector<float> HighResolutionGradient_volume_static;

    thrust::device_vector<float> weightingFactor_volume;
    thrust::device_vector<float> weightedSimilarity_volume;

    thrust::device_vector<float> blendMask_RU;
    thrust::device_vector<float> blendMask_LU;

    thrust::device_vector<float> WeightingMap; 

    //flags:
    bool IsGradientReady;

    //functors:
    IsNotZero isNotZero;

    cimg_library::CImgDisplay mainDisp; 
    cimg_library::CImg<float> image_us; 
    cimg_library::CImg<float> image_high; 

};

#endif