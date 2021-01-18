#ifndef LC2PARAMETERS
#define LC2PARAMETERS

#include <iostream>
#include <string>
#include <vector>

struct regIntPoints3D{
    int x;
    int y;
    int z; 
};

struct regFloatingPoints3D{
    float x;
    float y;
    float z;
};

struct regIntPoints2D{
    int x;
    int y;
};

struct regFloatingPoints2D{
    float x;
    float y;
};

struct LC2Configuration2D{

    //Constructor default: 
    LC2Configuration2D(){
        //FixedImage: 
        FixedDimension.x = 0; FixedDimension.y = 0;
        FixedSpacing.x = 1.0f; FixedSpacing.y = 1.0f; 
        FixedOrigin.x = 0.0f; FixedOrigin.y = 0.0f; 
        FixedImageFilePath = ""; 

        //MovingImage: 
        MovingDimension.x = 0; MovingDimension.y = 0;
        MovingSpacing.x = 1.0f; MovingSpacing.y = 1.0f; 
        MovingOrigin.x = 0.0f; MovingOrigin.y = 0.0f; 
        MovingImageFilePath = ""; 

        //Registration parameters; 
        IsFixedHighResolution = true; 
        IsMovingHighResolution = false; 
        SamplerSpacing.x = 1.0f; SamplerSpacing.y = 1.0f; 
        PatchSize = 9; 
        ExportPath = ""; 
        IsCenterOverlaid = true; 
        IsAffine = false; 

        //For optimizers: 
        initFRE = 50.0f; 
        initTranslation.x = 0.0f; initTranslation.y = 0.0f; 
        initRotation = 0.0f; 
        initScale.x = 1.0f; initScale.y = 1.0f; 
        initShear = 0.0f; 

        //DIRECT: 
        boundTranslation_DIRECT.x = 0.8f; boundTranslation_DIRECT.y = 0.5f; 
        MaxEval = 25; 

        //BOBYQA: 
        boundTranslation_BOBYQA.x = 1.0f; boundTranslation_BOBYQA.y = 1.0f; 
        boundRotation_BOBYQA = 1.0f; 
        boundScale_BOBYQA.x = 1.0f; boundScale_BOBYQA.y = 1.0f; 
        boundShear_BOBYQA = 1.0f; 
        RhoBegin = 0.1f; 
        Tol = 1e-4f;
        MaxIterations = 1000; 

        //flags: 
        IsConfigured = false; 

        //Buffers: 
        MovingBuffer = NULL; 
        FixedBuffer = NULL; 

        //Results: 
        IsConverged = false; 
        LastSimilarity = 0.0f; 
        NumberOfEvaluations = 0; 
        //Identity: 
        MovingToFixedMatrix.resize(9, 0.0f); 
        MovingToFixedMatrix[0] = 1.0f; MovingToFixedMatrix[4] = 1.0f; MovingToFixedMatrix[8] = 1.0f; 
    }

    void DumpRegistrtationResults( 
        const bool Ext_IsConverged, 
        const float Ext_LastSimilarity, 
        const int Ext_NumberOfEvaluations, 
        const float* Ext_FinalTransformMatrix)
    {
        IsConverged = Ext_IsConverged; 
        LastSimilarity = Ext_LastSimilarity; 
        NumberOfEvaluations = Ext_NumberOfEvaluations; 

        for(int i = 0; i < 9; ++i){
            MovingToFixedMatrix[i] = Ext_FinalTransformMatrix[i]; 
        }
    }

    //FixedImage: 
    regIntPoints2D FixedDimension; 
    regFloatingPoints2D FixedSpacing; 
    regFloatingPoints2D FixedOrigin; 
    std::string FixedImageFilePath; 

    //MovingImage: 
    regIntPoints2D MovingDimension; 
    regFloatingPoints2D MovingSpacing; 
    regFloatingPoints2D MovingOrigin; 
    std::string MovingImageFilePath; 

    //Registration parameters; 
    bool IsFixedHighResolution; 
    bool IsMovingHighResolution; 
    regFloatingPoints2D SamplerSpacing; 
    int PatchSize; 
    std::string ExportPath; 
    bool IsCenterOverlaid; 
    bool IsAffine; 

    //For optimizers: 
    float initFRE; 
    regFloatingPoints2D initTranslation; 
    float initRotation; 
    regFloatingPoints2D initScale; 
    float initShear; 

    //DIRECT: 
    regFloatingPoints2D boundTranslation_DIRECT; 
    // float boundRotation_DIRECT; 
    // regFloatingPoints2D boundScale_DIRECT; 
    // float boundShear_DIRECT;
    int MaxEval; 

    //BOBYQA: 
    regFloatingPoints2D boundTranslation_BOBYQA; 
    float boundRotation_BOBYQA; 
    regFloatingPoints2D boundScale_BOBYQA; 
    float boundShear_BOBYQA;
    float RhoBegin; 
    float Tol; 
    float MaxIterations; 

    //flags: 
    bool IsConfigured; 

    //Raw data pointers: 
    float* MovingBuffer; 
    float* FixedBuffer; 

    //Results: 
    bool IsConverged; 
    float LastSimilarity; 
    int NumberOfEvaluations; 
    std::vector<float> MovingToFixedMatrix; 
}; 

struct LC2Configuration3D{

    //Constructor: 
    LC2Configuration3D(){
        //FixedImage: 
        FixedDimension.x = 0; FixedDimension.y = 0; FixedDimension.z = 0;
        FixedSpacing.x = 1.0f; FixedSpacing.y = 1.0f; FixedSpacing.z = 1.0f; 
        FixedOrigin.x = 0.0f; FixedOrigin.y = 0.0f; FixedOrigin.z = 0.0f; 
        FixedImageFilePath = ""; 

        //MovingImage: 
        MovingDimension.x = 0; MovingDimension.y = 0; MovingDimension.z = 0; 
        MovingSpacing.x = 1.0f; MovingSpacing.y = 1.0f; MovingSpacing.z = 1.0f; 
        MovingOrigin.x = 0.0f; MovingOrigin.y = 0.0f; MovingOrigin.z = 0.0f; 
        MovingImageFilePath = ""; 

        //Registration parameters; 
        IsFixedHighResolution = false; 
        IsMovingHighResolution = false; 
        SamplerSpacing.x = 1.0f; SamplerSpacing.y = 1.0f; SamplerSpacing.z = 1.0f; 
        PatchSize = 3; 
        ExportPath = ""; 
        IsCenterOverlaid = true; 
        IsAffine = false; 

        //For optimizers: 
        initFRE = 30.0f; 
        initTranslation.x = 0.0f; initTranslation.y = 0.0f; initTranslation.z = 0.0f; 
        initRotation.x = 0.0f; initRotation.y = 0.0f; initRotation.z = 0.0f; 
        initScale.x = 1.0f; initScale.y = 1.0f; initScale.z = 1.0f; 
        initShear.x = 0.0f; initShear.y = 0.0f; initShear.z = 0.0f; 

        //DIRECT: 
        boundTranslation_DIRECT.x = 0.5f; boundTranslation_DIRECT.y = 0.5f; boundTranslation_DIRECT.z = 0.5f; 
        MaxEval = 100; 

        //BOBYQA: 
        boundTranslation_BOBYQA.x = 1.0f; boundTranslation_BOBYQA.y = 1.0f; boundTranslation_BOBYQA.z = 1.0f; 
        boundRotation_BOBYQA.x = 1.0f; boundRotation_BOBYQA.y = 1.0f; boundRotation_BOBYQA.z = 1.0f;
        boundScale_BOBYQA.x = 1.0f; boundScale_BOBYQA.y = 1.0f; boundScale_BOBYQA.z = 1.0f; 
        boundShear_BOBYQA.x = 1.0f; boundShear_BOBYQA.y = 1.0f; boundShear_BOBYQA.z = 1.0f; 
        RhoBegin = 0.01f; 
        Tol = 1e-6f;
        MaxIterations = 1000; 

        //flags: 
        IsConfigured = false; 

        //Buffers: 
        MovingBuffer = NULL; 
        FixedBuffer = NULL; 

        //Results: 
        IsConverged = false; 
        LastSimilarity = 0.0f; 
        NumberOfEvaluations = 0; 
        //Identity: 
        MovingToFixedMatrix.resize(16, 0.0f); 
        MovingToFixedMatrix[0] = 1.0f; MovingToFixedMatrix[5] = 1.0f; MovingToFixedMatrix[10] = 1.0f; MovingToFixedMatrix[15] = 1.0f; 
    }

    //Dump results from registration handle: 
    void DumpRegistrtationResults( 
        const bool Ext_IsConverged, 
        const float Ext_LastSimilarity, 
        const int Ext_NumberOfEvaluations, 
        const float* Ext_FinalTransformMatrix)
    {
        IsConverged = Ext_IsConverged; 
        LastSimilarity = Ext_LastSimilarity; 
        NumberOfEvaluations = Ext_NumberOfEvaluations; 

        for(int i = 0; i < 16; ++i){
            MovingToFixedMatrix[i] = Ext_FinalTransformMatrix[i]; 
        }
    }

    //FixedImage: 
    regIntPoints3D FixedDimension; 
    regFloatingPoints3D FixedSpacing; 
    regFloatingPoints3D FixedOrigin; 
    std::string FixedImageFilePath; 

    //MovingImage: 
    regIntPoints3D MovingDimension; 
    regFloatingPoints3D MovingSpacing; 
    regFloatingPoints3D MovingOrigin; 
    std::string MovingImageFilePath; 

    //Registration parameters; 
    bool IsFixedHighResolution; 
    bool IsMovingHighResolution; 
    regFloatingPoints3D SamplerSpacing; 
    int PatchSize; 
    std::string ExportPath; 
    bool IsCenterOverlaid; 
    bool IsAffine; 

    //For optimizers: 
    float initFRE; 
    regFloatingPoints3D initTranslation; 
    regFloatingPoints3D initRotation; 
    regFloatingPoints3D initScale; 
    regFloatingPoints3D initShear; 

    //DIRECT: 
    regFloatingPoints3D boundTranslation_DIRECT; 
    // regFloatingPoints3D boundRotation_DIRECT; 
    // regFloatingPoints3D boundScale_DIRECT; 
    // regFloatingPoints3D boundShear_DIRECT;
    int MaxEval; 

    //BOBYQA: 
    regFloatingPoints3D boundTranslation_BOBYQA; 
    regFloatingPoints3D boundRotation_BOBYQA; 
    regFloatingPoints3D boundScale_BOBYQA; 
    regFloatingPoints3D boundShear_BOBYQA;
    float RhoBegin; 
    float Tol; 
    float MaxIterations; 

    //flags: 
    bool IsConfigured; 

    //Raw data pointers: 
    float* MovingBuffer; 
    float* FixedBuffer; 

    //Results: 
    bool IsConverged; 
    float LastSimilarity; 
    int NumberOfEvaluations; 
    std::vector<float> MovingToFixedMatrix; 
}; 

struct LC2Configuration2DTo3D{

    //Constructor: 
    LC2Configuration2DTo3D(){
        //FixedImage: 
        FixedDimension.x = 0; FixedDimension.y = 0; FixedDimension.z = 0;
        FixedSpacing.x = 1.0f; FixedSpacing.y = 1.0f; FixedSpacing.z = 1.0f; 
        FixedOrigin.x = 0.0f; FixedOrigin.y = 0.0f; FixedOrigin.z = 0.0f; 
        FixedImageFilePath = ""; 

        //MovingImage: 
        MovingDimension.x = 0; MovingDimension.y = 0; MovingDimension.z = 0; 
        MovingSpacing.x = 1.0f; MovingSpacing.y = 1.0f; MovingSpacing.z = 1.0f; 
        MovingOrigin.x = 0.0f; MovingOrigin.y = 0.0f; MovingOrigin.z = 0.0f; 
        MovingImageFilePath = ""; 

        //Registration parameters; 
        IsFixedHighResolution = false; 
        IsMovingHighResolution = false; 
        SamplerSpacing.x = 1.0f; SamplerSpacing.y = 1.0f; SamplerSpacing.z = 1.0f; 
        PatchSize = 3; 
        ExportPath = ""; 
        IsCenterOverlaid = true; 
        IsAffine = false; 

        //For optimizers: 
        initFRE = 30.0f; 
        initTranslation.x = 0.0f; initTranslation.y = 0.0f; initTranslation.z = 0.0f; 
        initRotation.x = 0.0f; initRotation.y = 0.0f; initRotation.z = 0.0f; 
        initScale.x = 1.0f; initScale.y = 1.0f; initScale.z = 1.0f; 
        initShear.x = 0.0f; initShear.y = 0.0f; initShear.z = 0.0f; 

        //DIRECT: 
        boundTranslation_DIRECT.x = 0.5f; boundTranslation_DIRECT.y = 0.5f; boundTranslation_DIRECT.z = 0.5f; 
        MaxEval = 100; 

        //BOBYQA: 
        boundTranslation_BOBYQA.x = 1.0f; boundTranslation_BOBYQA.y = 1.0f; boundTranslation_BOBYQA.z = 1.0f; 
        boundRotation_BOBYQA.x = 1.0f; boundRotation_BOBYQA.y = 1.0f; boundRotation_BOBYQA.z = 1.0f;
        boundScale_BOBYQA.x = 1.0f; boundScale_BOBYQA.y = 1.0f; boundScale_BOBYQA.z = 1.0f; 
        boundShear_BOBYQA.x = 1.0f; boundShear_BOBYQA.y = 1.0f; boundShear_BOBYQA.z = 1.0f; 
        RhoBegin = 0.01f; 
        Tol = 1e-6f;
        MaxIterations = 1000; 

        //flags: 
        IsConfigured = false; 

        //Buffers: 
        MovingBuffer = NULL; 
        FixedBuffer = NULL; 

        //Results: 
        IsConverged = false; 
        LastSimilarity = 0.0f; 
        NumberOfEvaluations = 0; 
        //Identity: 
        MovingToFixedMatrix.resize(16, 0.0f); 
        MovingToFixedMatrix[0] = 1.0f; MovingToFixedMatrix[5] = 1.0f; MovingToFixedMatrix[10] = 1.0f; MovingToFixedMatrix[15] = 1.0f; 
    }

    void DumpRegistrtationResults( 
        const bool Ext_IsConverged, 
        const float Ext_LastSimilarity, 
        const int Ext_NumberOfEvaluations, 
        const float* Ext_FinalTransformMatrix)
    {
        IsConverged = Ext_IsConverged; 
        LastSimilarity = Ext_LastSimilarity; 
        NumberOfEvaluations = Ext_NumberOfEvaluations; 

        for(int i = 0; i < 16; ++i){
            MovingToFixedMatrix[i] = Ext_FinalTransformMatrix[i]; 
        }
    }

    //FixedImage: 
    regIntPoints3D FixedDimension; 
    regFloatingPoints3D FixedSpacing; 
    regFloatingPoints3D FixedOrigin; 
    std::string FixedImageFilePath; 

    //MovingImage: 
    regIntPoints3D MovingDimension; 
    regFloatingPoints3D MovingSpacing; 
    regFloatingPoints3D MovingOrigin; 
    std::string MovingImageFilePath; 

    //Registration parameters; 
    bool IsFixedHighResolution; 
    bool IsMovingHighResolution; 
    regFloatingPoints3D SamplerSpacing; 
    int PatchSize; 
    std::string ExportPath; 
    bool IsCenterOverlaid; 
    bool IsAffine; 

    //For optimizers: 
    float initFRE; 
    regFloatingPoints3D initTranslation; 
    regFloatingPoints3D initRotation; 
    regFloatingPoints3D initScale; 
    regFloatingPoints3D initShear; 

    //DIRECT: 
    regFloatingPoints3D boundTranslation_DIRECT; 
    // regFloatingPoints3D boundRotation_DIRECT; 
    // regFloatingPoints3D boundScale_DIRECT; 
    // regFloatingPoints3D boundShear_DIRECT;
    int MaxEval; 

    //BOBYQA: 
    regFloatingPoints3D boundTranslation_BOBYQA; 
    regFloatingPoints3D boundRotation_BOBYQA; 
    regFloatingPoints3D boundScale_BOBYQA; 
    regFloatingPoints3D boundShear_BOBYQA;
    float RhoBegin; 
    float Tol; 
    float MaxIterations; 

    //flags: 
    bool IsConfigured; 

    //Raw data pointers: 
    float* MovingBuffer; 
    float* FixedBuffer; 

    //Results: 
    bool IsConverged; 
    float LastSimilarity; 
    int NumberOfEvaluations; 
    std::vector<float> MovingToFixedMatrix; 
}; 

#endif 