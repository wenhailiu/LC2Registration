#ifndef CULC2_2D
#define CULC2_2D

#include "thrust/device_vector.h"

#include "CImg.h"

#include "src/cu_LC2/cu_lc2_helper.cuh"

class LC2_2D_class{

public:
    //Initializer:
    LC2_2D_class(int x, int y, std::vector<float>& Ultrasound_Extern, std::vector<float>& HighReso_Extern);
    LC2_2D_class(){}; 

    void PortedFromInterpolator(int x, int y); 

    //Member functions:
    //Calculate the gradient for Highresolition image:
    void PrepareGradientFilter(); 
    void CalculateGradient(const int Blk_Dim_x = 16, const int Blk_Dim_y = 16);

    //Analyse the input images, counting the meaningful US pixels.
    //Simply: check if the current US pixel is zero. If it is, start counting the surrounding patch. 
    //If fullSized patch observed, then marked 1s on the countingVector, 
    //if not, then fill with zeros. 
    double GetSimilarityMetric(const int Blk_Dim_x = 16, const int Blk_Dim_y = 16, const int patchSize = 9);

    //show image:
    // static void showImagePairs(cv::cuda::GpuMat &img1, cv::cuda::GpuMat &img2, cv::cuda::GpuMat &Mask_LU, cv::cuda::GpuMat &Mask_RU);
    // virtual void ShowImages();
    // virtual void ShowImages(float *ImgToShow_d);

    virtual void ShowImages(int display_pattern = 0); 

    //Get raw pointers:
    float* GetUltrasoundRawPointer();
    float* GetHighResolutionRawPointer();
    float* GetHighResolutionGradientRawPointer();

    //Update high resolution image: based on transform parameters:
    void UpdateImageAffine(double T_x, double T_y, double R);

    void UpdateOrigImageAffine(double T_x, double T_y, double R);

    void SaveResults();

protected: 
    //May get values from outside
    thrust::device_vector<float> Ultrasound;
    thrust::device_vector<float> HighResolution_static;

private:
    //Volume parameters:
    int2 img_dim;

    //Volume data:
    thrust::device_vector<float> Ultrasound_static;
    thrust::device_vector<float> HighResolutionGradient_static;

    thrust::device_vector<float> weightingFactor;
    thrust::device_vector<float> weightedSimilarity;

    thrust::device_vector<float> blendMask_RU;
    thrust::device_vector<float> blendMask_LU;

    //flags:
    // bool IsGradientReady;

    //Transform:
    double hAffineTransform[2][3];

    //functors:
    IsNotZero isNotZero;

    cimg_library::CImgDisplay mainDisp; 
    cimg_library::CImg<float> image_us; 
    cimg_library::CImg<float> image_high; 

};


#endif