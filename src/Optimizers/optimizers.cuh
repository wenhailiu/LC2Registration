#ifndef LC2OPIMIZERS
#define LC2OPIMIZERS

#include <iostream>
#include <ctime>
#include <ratio>
#include <chrono>
#include <utility>

#include <cuda_runtime.h>
#include "device_launch_parameters.h"

#include "src/Interpolator/interpolator.cuh"

#define RIGID2D_NUMBER 3
#define AFFINE2D_NUMBER 6

#define RIGID3D_NUMBER 6
#define AFFINE3D_NUMBER 12

enum OptimizationEntry{
    Translation, 
    Rotation, 
    Scale, 
    Shear
}; 

class Optimizers2D{

public: 
    Optimizers2D(std::string FilePath); 

    void LinkToInterpolator(Interpolator2D* external_Interpolator); 

    void ShowCurrentResults(); 
    double operator()(const std::vector<double> &x); 
    static double wrapper(const std::vector<double> &x, std::vector<double> &grad, void *data); 
    void Optimize(); 

    void GetOptimalTransform(float* outMatrix);
    void GetOptimalTransform();
    void GetInternalMatrix(float* OutMatrix); 

    void ExportResults(); 

private: 

    //Internal interpolator pointer: 
    Interpolator2D* interpolation_handle; 

    //Initial starters: 
    std::vector<double> InitialValues; 

    //Bounds: 
    std::vector<double> LowerBounds; 
    std::vector<double> UpperBounds; 

    //DIRECT: 
    bool DIRECT_Enabled; 
    std::vector<std::pair<OptimizationEntry, bool>> DIRECT_OptEntryPairs; 
    int DIRECT_MaxEvals; 

    //BOBYQA: 
    int BOBYQA_Rounds; 
    bool BOBYQA_Enabled; 
    std::vector<std::pair<OptimizationEntry, bool>> BOBYQA_OptEntryPairs; 
    double BOBYQA_AbsTol; 

    //Current Entry
    std::vector<OptimizationEntry> CurrentHolding_OptEntries; 

    //Optimized results: 
    std::vector<double> OptimalSolution; 

    //Patch size: 
    int patchSize; 

    //Similarity measure: 
    float SimilarityMeasure; 

    //optimal transformMatrix: 
    float MovingToFixed_Optimal[9]; 

    //test:
    float testTarget[6];

    //counter: 
    int EvalCounter; 

    //yaml_handle: 
    YAML::Node YAML_ParameterParser; 
    std::string YAML_ParameterPath; 

    //timePoint: 
    std::time_t currentTime; 
};

class Optimizers3D{

public: 
    Optimizers3D(std::string FilePath); 
    Optimizers3D(const LC2Configuration3D* ExternalConfig); 

    void LinkToInterpolator(Interpolator3D* external_Interpolator); 

    void ShowCurrentResults(); 
    double operator()(const std::vector<double> &x); 
    static double wrapper(const std::vector<double> &x, std::vector<double> &grad, void *data); 
    void Optimize(); 

    void GetOptimalTransform(float* outMatrix);
    void GetOptimalTransform();
    void GetInternalMatrix(float* OutMatrix); 

    void ExportResults(); 

private: 

    //Internal interpolator pointer: 
    Interpolator3D* interpolation_handle; 

    //Initial starters: 
    std::vector<double> InitialValues; 

    //Bounds: 
    std::vector<double> LowerBounds; 
    std::vector<double> UpperBounds; 

    //DIRECT: 
    bool DIRECT_Enabled; 
    std::vector<std::pair<OptimizationEntry, bool>> DIRECT_OptEntryPairs; 
    int DIRECT_MaxEvals; 

    //BOBYQA: 
    int BOBYQA_Rounds; 
    bool BOBYQA_Enabled; 
    std::vector<std::pair<OptimizationEntry, bool>> BOBYQA_OptEntryPairs; 
    double BOBYQA_AbsTol; 

    //Current Entry
    std::vector<OptimizationEntry> CurrentHolding_OptEntries; 

    //Optimized results: 
    std::vector<double> OptimalSolution; 

    //Patch size: 
    int patchSize; 

    //parameters to Optimizer: 
    int EvalCounter; 

    //Similarity measure: 
    double SimilarityMeasure; 

    //optimal transformMatrix: 
    float MovingToFixed_Optimal[16]; 

    //test:
    float testTarget[12];

    //yaml_handle: 
    YAML::Node YAML_ParameterParser; 
    std::string YAML_ParameterPath; 

    //timePoint: 
    std::time_t currentTime; 

    //tmp log: 
    std::ofstream m_logFileId; 

    int m_stepMarker; 
};

class Optimizers2DTo3D{

public: 
    Optimizers2DTo3D(std::string FilePath); 
    Optimizers2DTo3D(const LC2Configuration3D* ExternalConfig); 

    void LinkToInterpolator(Interpolator2DTo3D* external_Interpolator); 

    void GlobalIterate(); 
    void ShowCurrentResults(); 
    double operator()(const std::vector<double> &x); 
    static double wrapper(const std::vector<double> &x, std::vector<double> &grad, void *data); 
    void Optimize(); 

    void GetOptimalTransform(float* outMatrix);
    void GetOptimalTransform();
    void GetInternalMatrix(float* OutMatrix); 

    void ExportResults(); 

private: 

    //Internal interpolator pointer: 
    Interpolator2DTo3D* interpolation_handle; 

    //Initial starters: 
    std::vector<double> InitialValues; 

    //Bounds: 
    std::vector<double> LowerBounds; 
    std::vector<double> UpperBounds; 

    //DIRECT: 
    bool DIRECT_Enabled; 
    std::vector<std::pair<OptimizationEntry, bool>> DIRECT_OptEntryPairs; 
    int DIRECT_MaxEvals; 

    //BOBYQA: 
    int BOBYQA_Rounds; 
    bool BOBYQA_Enabled; 
    std::vector<std::pair<OptimizationEntry, bool>> BOBYQA_OptEntryPairs; 
    double BOBYQA_AbsTol; 

    //Current Entry
    std::vector<OptimizationEntry> CurrentHolding_OptEntries; 

    //Optimized results: 
    std::vector<double> OptimalSolution; 

    //Patch size: 
    int patchSize; 

    //parameters to Optimizer: 
    int EvalCounter; 

    //Similarity measure: 
    double SimilarityMeasure; 

    //optimal transformMatrix: 
    float MovingToFixed_Optimal[16]; 

    //test:
    float testTarget[12];

    //yaml_handle: 
    YAML::Node YAML_ParameterParser; 
    std::string YAML_ParameterPath; 

    //timePoint: 
    std::time_t currentTime; 

    //tmp log: 
    std::ofstream m_logFileId; 
};

#endif