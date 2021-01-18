#include <iostream>
#include <iomanip>
#include <thread>
#include <vector>
#include <string>
#include <algorithm>
#include <stdexcept>
#include <fstream>
#include <map>
#include "md5.h"

#include "yaml-cpp/yaml.h"

#include "src/Interpolator/interpolator.cuh"
#include "src/Optimizers/optimizers.cuh"

#include "src/Utilities/my_cuda_helper.cuh"

//Arguments parser:
class InputParser{
public:
    InputParser (int &argc, char **argv){
        for (int i=1; i < argc; ++i)
            this->tokens.push_back(std::string(argv[i]));
    }
    /// @author iain
    const std::string& getCmdOption(const std::string &option) const{
        std::vector<std::string>::const_iterator itr;
        itr =  std::find(this->tokens.begin(), this->tokens.end(), option);
        if (itr != this->tokens.end() && ++itr != this->tokens.end()){
            return *itr;
        }
        static const std::string empty_string("");
        return empty_string;
    }
    /// @author iain
    bool cmdOptionExists(const std::string &option) const{
        return std::find(this->tokens.begin(), this->tokens.end(), option)
                != this->tokens.end();
    }
private:
    std::vector <std::string> tokens;
};

void PrintUsage(std::ostream& os){
    os << "Usage: " << std::endl;
    os << std::setw(15) << "-h:    Print usages. " << std::endl;
    os << std::setw(15) << "-m:    Registration modes, 2D or 3D. " << std::endl;
    os << std::setw(15) << "-c:    Load configuration file, which is necessary for image registration. " << std::endl;
    os << std::setw(15) << "-t:    Generate configuration file template. " << std::endl;
}

void Generate2DTemplate(std::string filename){

}

void Generate3DTemplate(std::string filename){
    std::map<std::string, int> dim;
    dim["x"] = 128;
    dim["y"] = 128;
    dim["z"] = 128;
    std::map<std::string, float> spacing;
    spacing["x"] = 0.5f;
    spacing["y"] = 0.5f;
    spacing["z"] = 0.5f;
    std::map<std::string, float> origin;
    origin["x"] = -10.5f;
    origin["y"] = -15.4f;
    origin["z"] = -21.6f;
    std::map<std::string, float> initialOffsets;
    initialOffsets["translationX"] = 0.0f;
    initialOffsets["translationY"] = 0.0f;
    initialOffsets["translationZ"] = 0.0f;
    initialOffsets["rotationX"] = 0.0f;
    initialOffsets["rotationY"] = 0.0f;
    initialOffsets["rotationZ"] = 0.0f;
    initialOffsets["scaleX"] = 1.0f;
    initialOffsets["scaleY"] = 1.0f;
    initialOffsets["scaleZ"] = 1.0f;
    initialOffsets["shearXY"] = 0.0f;
    initialOffsets["shearXZ"] = 0.0f;
    initialOffsets["shearYZ"] = 0.0f;
    std::map<std::string, float> DIRECTBoundRange;
    DIRECTBoundRange["translationX"] = 0.3f;
    DIRECTBoundRange["translationY"] = 0.3f;
    DIRECTBoundRange["translationZ"] = 0.3f;
    DIRECTBoundRange["rotationX"] = 0.3f;
    DIRECTBoundRange["rotationY"] = 0.3f;
    DIRECTBoundRange["rotationZ"] = 0.3f;
    DIRECTBoundRange["scaleX"] = 1.0f;
    DIRECTBoundRange["scaleY"] = 1.0f;
    DIRECTBoundRange["scaleZ"] = 1.0f;
    DIRECTBoundRange["shearXY"] = 0.0f;
    DIRECTBoundRange["shearXZ"] = 0.0f;
    DIRECTBoundRange["shearYZ"] = 0.0f;

    std::map<std::string, float> BOBYQABoundRange;
    BOBYQABoundRange["translationX"] = 1.0f;
    BOBYQABoundRange["translationY"] = 1.0f;
    BOBYQABoundRange["translationZ"] = 1.0f;
    BOBYQABoundRange["rotationX"] = 1.0f;
    BOBYQABoundRange["rotationY"] = 1.0f;
    BOBYQABoundRange["rotationZ"] = 1.0f;
    BOBYQABoundRange["scaleX"] = 1.0f;
    BOBYQABoundRange["scaleY"] = 1.0f;
    BOBYQABoundRange["scaleZ"] = 1.0f;
    BOBYQABoundRange["shearXY"] = 1.0f;
    BOBYQABoundRange["shearXZ"] = 1.0f;
    BOBYQABoundRange["shearYZ"] = 1.0f;

    std::ofstream fileHandle;
    fileHandle.open(filename, std::ofstream::out | std::ofstream::trunc);
    YAML::Emitter TemplateEmitter(fileHandle);

    //start emit: 

    //Start Layer - 0: InputParameters
    TemplateEmitter << YAML::BeginMap;
    TemplateEmitter << YAML::Key << "InputParameters";

    //Start sublayer - 0: FixedVolume
    TemplateEmitter << YAML::Value << YAML::BeginMap;
    TemplateEmitter << YAML::Key << "FixedVolume"; 

    //Start inner layer: 
    TemplateEmitter << YAML::Value << YAML::BeginMap;

    TemplateEmitter << YAML::Key << "Dimension";
    TemplateEmitter << YAML::Value << YAML::BeginMap;
    TemplateEmitter << YAML::Flow << dim;
    TemplateEmitter << YAML::EndMap;

    TemplateEmitter << YAML::Key << "Spacing";
    TemplateEmitter << YAML::Value << YAML::BeginMap;
    TemplateEmitter << YAML::Flow << spacing;
    TemplateEmitter << YAML::EndMap;

    TemplateEmitter << YAML::Key << "Origin";
    TemplateEmitter << YAML::Value << YAML::BeginMap;
    TemplateEmitter << YAML::Flow << origin;
    TemplateEmitter << YAML::EndMap;

    TemplateEmitter << YAML::Key << "FilePath";
    TemplateEmitter << YAML::Value << "/path/to/volume.format";

    //End inner layer: 
    TemplateEmitter << YAML::EndMap;

    //End sublayer - 0: FixedVolume
    // TemplateEmitter << YAML::EndMap;

    //Start sublayer - 1: MovingVolume
    // TemplateEmitter << YAML::Value << YAML::BeginMap;
    TemplateEmitter << YAML::Key << "MovingVolume"; 

    //Start inner layer: 
    TemplateEmitter << YAML::Value << YAML::BeginMap;

    TemplateEmitter << YAML::Key << "Dimension";
    TemplateEmitter << YAML::Value << YAML::BeginMap;
    TemplateEmitter << YAML::Flow << dim;
    TemplateEmitter << YAML::EndMap;

    TemplateEmitter << YAML::Key << "Spacing";
    TemplateEmitter << YAML::Value << YAML::BeginMap;
    TemplateEmitter << YAML::Flow << spacing;
    TemplateEmitter << YAML::EndMap;

    TemplateEmitter << YAML::Key << "Origin";
    TemplateEmitter << YAML::Value << YAML::BeginMap;
    TemplateEmitter << YAML::Flow << origin;
    TemplateEmitter << YAML::EndMap;

    TemplateEmitter << YAML::Key << "FilePath";
    TemplateEmitter << YAML::Value << "/path/to/volume.format";

    //End inner layer: 
    TemplateEmitter << YAML::EndMap;

    //End sublayer - 1: MovingVolume
    // TemplateEmitter << YAML::EndMap;

    //Start sublayer - 2: RegistrationParameters
    // TemplateEmitter << YAML::Value << YAML::BeginMap;
    TemplateEmitter << YAML::Key << "RegistrationParameters"; 

    //Start inner layer: 
    TemplateEmitter << YAML::Value << YAML::BeginMap;

    TemplateEmitter << YAML::Key << "HighResolutionModality";
    TemplateEmitter << YAML::Value << "Moving";

    TemplateEmitter << YAML::Key << "Type";
    TemplateEmitter << YAML::Value << "Affine";

    TemplateEmitter << YAML::Key << "DisplayPattern";
    TemplateEmitter << YAML::Value << 2;

    TemplateEmitter << YAML::Key << "ResampledSpacing";
    TemplateEmitter << YAML::Value << YAML::BeginMap;
    TemplateEmitter << YAML::Flow << spacing;
    TemplateEmitter << YAML::EndMap;

    TemplateEmitter << YAML::Key << "PatchSize";
    TemplateEmitter << YAML::Value << 3;

    TemplateEmitter << YAML::Key << "FinalRoundExportRootPath";
    TemplateEmitter << YAML::Value << "/path/to/output";

    TemplateEmitter << YAML::Key << "CenterOverlaid";
    TemplateEmitter << YAML::Value << false;

    //End inner layer: 
    TemplateEmitter << YAML::EndMap;

    //End sublayer - 2: RegistrationParameters
    // TemplateEmitter << YAML::EndMap;

    //Start sublayer - 3: OptimizerSettings
    // TemplateEmitter << YAML::Value << YAML::BeginMap;
    TemplateEmitter << YAML::Key << "OptimizerSettings"; 

    //Start inner layer: 
    TemplateEmitter << YAML::Value << YAML::BeginMap;

    TemplateEmitter << YAML::Key << "InitialFRE";
    TemplateEmitter << YAML::Value << 30.0;

    TemplateEmitter << YAML::Key << "InitialOffset";
    TemplateEmitter << YAML::Value << YAML::BeginMap;
    TemplateEmitter << YAML::Flow << initialOffsets;
    TemplateEmitter << YAML::EndMap;

    TemplateEmitter << YAML::Key << "DIRECT";
    TemplateEmitter << YAML::Value << YAML::BeginMap;
    TemplateEmitter << YAML::Key << "BoundRange";
    TemplateEmitter << YAML::Value << YAML::BeginMap;
    TemplateEmitter << YAML::Flow << DIRECTBoundRange;
    TemplateEmitter << YAML::EndMap;
    TemplateEmitter << YAML::Key << "MaxEvaluations";
    TemplateEmitter << YAML::Value << 100;
    TemplateEmitter << YAML::EndMap;

    TemplateEmitter << YAML::Key << "BOBYQA";
    TemplateEmitter << YAML::Value << YAML::BeginMap;
    TemplateEmitter << YAML::Key << "BoundRange";
    TemplateEmitter << YAML::Value << YAML::BeginMap;
    TemplateEmitter << YAML::Flow << BOBYQABoundRange;
    TemplateEmitter << YAML::EndMap;
    TemplateEmitter << YAML::Key << "Rho_begin";
    TemplateEmitter << YAML::Value << 0.1;
    TemplateEmitter << YAML::Key << "Tol";
    TemplateEmitter << YAML::Value << 1.0e-6;
    TemplateEmitter << YAML::Key << "MaxIterations";
    TemplateEmitter << YAML::Value << 1000;
    TemplateEmitter << YAML::EndMap;

    //End inner layer: 
    TemplateEmitter << YAML::EndMap;

    //End sublayer - 3: OptimizerSettings
    // TemplateEmitter << YAML::EndMap;

    //End Layer - 0: InputParameters
    TemplateEmitter << YAML::EndMap;

    //Start Layer - 1: Outputs
    // TemplateEmitter << YAML::BeginMap;
    TemplateEmitter << YAML::Key << "Outputs";

    //Start inner layer: 
    TemplateEmitter << YAML::Value << YAML::BeginMap;

    TemplateEmitter << YAML::Key << "IsRegistered";
    TemplateEmitter << YAML::Value << false;

    std::vector<std::vector<float>> DummyMatrix;
    DummyMatrix.push_back( std::vector<float>{1.0f, 0.0f, 0.0f, 0.0f} );
    DummyMatrix.push_back( std::vector<float>{0.0f, 1.0f, 0.0f, 0.0f} );
    DummyMatrix.push_back( std::vector<float>{0.0f, 0.0f, 1.0f, 0.0f} );
    DummyMatrix.push_back( std::vector<float>{0, 0, 0, 1} );
    TemplateEmitter << YAML::Key << "TransformMatrix" << YAML::Flow << YAML::Value << DummyMatrix; 

    TemplateEmitter << YAML::Key << "LastSimilarity";
    TemplateEmitter << YAML::Value << 0.0;

    TemplateEmitter << YAML::Key << "NumEvaluations";
    TemplateEmitter << YAML::Value << 0.0;

    TemplateEmitter << YAML::Key << "RegistrationDateTime";
    TemplateEmitter << YAML::Value << "Fri Mar  6 12:05:47 2020";

    //End inner layer: 
    TemplateEmitter << YAML::EndMap;

    //End Layer - 1: Outputs
    TemplateEmitter << YAML::EndMap;

    //close file: 
    fileHandle.close(); 
}

std::string exec(const char* cmd) {
    char buffer[128];
    std::string result = "";
    FILE* pipe = _popen(cmd, "r");
    if (!pipe) throw std::runtime_error("popen() failed!");
    try {
        while (fgets(buffer, sizeof buffer, pipe) != NULL) {
            result += buffer;
        }
    } catch (...) {
        _pclose(pipe);
        throw;
    }
    _pclose(pipe);
    return result;
}

int main(int argc, char* argv[]){
    InputParser input(argc, argv);

    if(input.cmdOptionExists("-h") || argc == 1){
        PrintUsage(std::cout);
        return 0;
    }

    std::string RegistrationMode; 
    if(input.cmdOptionExists("-m")){
        RegistrationMode = input.getCmdOption("-m");

        if(RegistrationMode.empty()){
            PrintUsage(std::cout);
            return 0;
        }
    }
    else{
        PrintUsage(std::cout);
        return 0;
    }

    std::string FixedImagePath = ""; 
    if(input.cmdOptionExists("-fixedPath")){
        FixedImagePath = input.getCmdOption("-fixedPath");
    }

    std::string MovingImagePath = ""; 
    if(input.cmdOptionExists("-movingPath")){
        MovingImagePath = input.getCmdOption("-movingPath");
    }

    std::string OutputImagePath = ""; 
    if(input.cmdOptionExists("-outputPath")){
        OutputImagePath = input.getCmdOption("-outputPath");
    }

    float tX = 0;
    if(input.cmdOptionExists("-tX")){
        tX = std::stof(input.getCmdOption("-tX"));
    }
    float tY = 0; 
    if(input.cmdOptionExists("-tY")){
        tY = std::stof(input.getCmdOption("-tY"));
    }
    float tZ = 0; 
    if(input.cmdOptionExists("-tZ")){
        tZ = std::stof(input.getCmdOption("-tZ"));
    }
    float rcX = 0;
    if(input.cmdOptionExists("-rcX")){
        rcX = std::stof(input.getCmdOption("-rcX"));
    }
    float rcY = 0; 
    if(input.cmdOptionExists("-rcY")){
        rcY = std::stof(input.getCmdOption("-rcY"));
    }
    float rcZ = 0; 
    if(input.cmdOptionExists("-rcZ")){
        rcZ = std::stof(input.getCmdOption("-rcZ"));
    }
    float rZ = 0; 
    if(input.cmdOptionExists("-rZ")){
        rZ = std::stof(input.getCmdOption("-rZ"));
    }

    /* ------------------------------------ Lambda start ----------------------------------------- */ 
    auto Generate3DTemplateFromInputs = [&](std::string filename){
        std::map<std::string, float> spacing;
        spacing["x"] = 1.0f;
        spacing["y"] = 1.0f;
        spacing["z"] = 1.0f;
        std::map<std::string, float> rotCenter;
        rotCenter["x"] = rcX;
        rotCenter["y"] = rcY;
        rotCenter["z"] = rcZ;
        std::map<std::string, float> initialOffsets;
        initialOffsets["TX"] = tX;
        initialOffsets["TY"] = tY;
        initialOffsets["TZ"] = tZ;
        initialOffsets["RX"] = 0.0f;
        initialOffsets["RY"] = 0.0f;
        initialOffsets["RZ"] = rZ;
        initialOffsets["ScaleX"] = 1.0f;
        initialOffsets["ScaleY"] = 1.0f;
        initialOffsets["ScaleZ"] = 1.0f;
        initialOffsets["ShearX"] = 0.0f;
        initialOffsets["ShearY"] = 0.0f;
        initialOffsets["ShearZ"] = 0.0f;
        std::map<std::string, float> lowerBounds;
        lowerBounds["TX"] = -10;
        lowerBounds["TY"] = -10;
        lowerBounds["TZ"] = -10;
        lowerBounds["RX"] = -10;
        lowerBounds["RY"] = -10;
        lowerBounds["RZ"] = -10;
        lowerBounds["ScaleX"] = 0.8;
        lowerBounds["ScaleY"] = 0.8;
        lowerBounds["ScaleZ"] = 0.8;
        lowerBounds["ShearX"] = -0.2;
        lowerBounds["ShearY"] = -0.2;
        lowerBounds["ShearZ"] = -0.2;
        std::map<std::string, float> upperBounds;
        upperBounds["TX"] = 10;
        upperBounds["TY"] = 10;
        upperBounds["TZ"] = 10;
        upperBounds["RX"] = 10;
        upperBounds["RY"] = 10;
        upperBounds["RZ"] = 10;
        upperBounds["ScaleX"] = 1.2;
        upperBounds["ScaleY"] = 1.2;
        upperBounds["ScaleZ"] = 1.2;
        upperBounds["ShearX"] = 0.2;
        upperBounds["ShearY"] = 0.2;
        upperBounds["ShearZ"] = 0.2;
        std::map<std::string, bool> DIRECT_Entries;
        DIRECT_Entries["Translation"] = true; 
        DIRECT_Entries["Rotation"] = false; 
        DIRECT_Entries["Scale"] = false; 
        DIRECT_Entries["Shear"] = false; 
        std::map<std::string, bool> BOBYQA_Entries0;
        BOBYQA_Entries0["Translation"] = true; 
        BOBYQA_Entries0["Rotation"] = true; 
        BOBYQA_Entries0["Scale"] = false; 
        BOBYQA_Entries0["Shear"] = false; 
        std::map<std::string, bool> BOBYQA_Entries1;
        BOBYQA_Entries1["Translation"] = true; 
        BOBYQA_Entries1["Rotation"] = true; 
        BOBYQA_Entries1["Scale"] = true; 
        BOBYQA_Entries1["Shear"] = true; 

        std::ofstream fileHandle;
        fileHandle.open(filename, std::ofstream::out | std::ofstream::trunc);
        YAML::Emitter TemplateEmitter(fileHandle);

        //start emit: 

        //Start Layer - 0: InputParameters
        TemplateEmitter << YAML::BeginMap;
        TemplateEmitter << YAML::Key << "InputParameters";

        //Start sublayer - 0: FixedVolume
        TemplateEmitter << YAML::Value << YAML::BeginMap;
        TemplateEmitter << YAML::Key << "FixedVolume"; 

        //Start inner layer: 
        TemplateEmitter << YAML::Value << YAML::BeginMap;
        TemplateEmitter << YAML::Key << "FilePath";
        TemplateEmitter << YAML::Value << FixedImagePath; 
        TemplateEmitter << YAML::Key << "Label";
        TemplateEmitter << YAML::Value << "Ultrasoud";

        //End inner layer: 
        TemplateEmitter << YAML::EndMap;

        //End sublayer - 0: FixedVolume
        // TemplateEmitter << YAML::EndMap;

        //Start sublayer - 1: MovingVolume
        // TemplateEmitter << YAML::Value << YAML::BeginMap;
        TemplateEmitter << YAML::Key << "MovingVolume"; 

        //Start inner layer: 
        TemplateEmitter << YAML::Value << YAML::BeginMap;
        TemplateEmitter << YAML::Key << "FilePath";
        TemplateEmitter << YAML::Value << MovingImagePath;
        TemplateEmitter << YAML::Key << "Label";
        TemplateEmitter << YAML::Value << "MRI";

        //End inner layer: 
        TemplateEmitter << YAML::EndMap;

        //End sublayer - 1: MovingVolume
        // TemplateEmitter << YAML::EndMap;

        //Start sublayer - 2: RegistrationParameters
        // TemplateEmitter << YAML::Value << YAML::BeginMap;
        TemplateEmitter << YAML::Key << "RegistrationParameters"; 

        //Start inner layer: 
        TemplateEmitter << YAML::Value << YAML::BeginMap;

        TemplateEmitter << YAML::Key << "HighResolutionModality";
        TemplateEmitter << YAML::Value << "Moving";

        TemplateEmitter << YAML::Key << "Type";
        TemplateEmitter << YAML::Value << "Affine";

        TemplateEmitter << YAML::Key << "DisplayPattern";
        TemplateEmitter << YAML::Value << 0;

        TemplateEmitter << YAML::Key << "ResampledSpacing";
        TemplateEmitter << YAML::Value << YAML::BeginMap;
        TemplateEmitter << YAML::Flow << spacing;
        TemplateEmitter << YAML::EndMap;

        TemplateEmitter << YAML::Key << "PatchSize";
        TemplateEmitter << YAML::Value << 3;

        TemplateEmitter << YAML::Key << "FinalRoundExportRootPath";
        TemplateEmitter << YAML::Value << OutputImagePath;

        TemplateEmitter << YAML::Key << "CenterOverlaid";
        TemplateEmitter << YAML::Value << false;

        TemplateEmitter << YAML::Key << "RotationCenter";
        TemplateEmitter << YAML::Value << YAML::BeginMap;
        TemplateEmitter << YAML::Flow << rotCenter;
        TemplateEmitter << YAML::EndMap;

        //End inner layer: 
        TemplateEmitter << YAML::EndMap;

        //End sublayer - 2: RegistrationParameters
        // TemplateEmitter << YAML::EndMap;

        //Start sublayer - 3: OptimizerSettings
        // TemplateEmitter << YAML::Value << YAML::BeginMap;
        TemplateEmitter << YAML::Key << "OptimizerSettings"; 

        //Start inner layer: 
        TemplateEmitter << YAML::Value << YAML::BeginMap;

        TemplateEmitter << YAML::Key << "InitialValues";
        TemplateEmitter << YAML::Value << YAML::BeginMap;
        TemplateEmitter << YAML::Flow << initialOffsets;
        TemplateEmitter << YAML::EndMap;

        TemplateEmitter << YAML::Key << "LowerBounds";
        TemplateEmitter << YAML::Value << YAML::BeginMap;
        TemplateEmitter << YAML::Flow << lowerBounds;
        TemplateEmitter << YAML::EndMap;

        TemplateEmitter << YAML::Key << "UpperBounds";
        TemplateEmitter << YAML::Value << YAML::BeginMap;
        TemplateEmitter << YAML::Flow << upperBounds;
        TemplateEmitter << YAML::EndMap;

        TemplateEmitter << YAML::Key << "DIRECT";
        TemplateEmitter << YAML::Value << YAML::BeginMap;

        TemplateEmitter << YAML::Key << "Enabled";
        TemplateEmitter << YAML::Value << false;

        TemplateEmitter << YAML::Key << "OptimizedEntries";
        TemplateEmitter << YAML::Value << YAML::BeginMap;
        TemplateEmitter << YAML::Flow << DIRECT_Entries;
        TemplateEmitter << YAML::EndMap;

        TemplateEmitter << YAML::Key << "MaxEvaluations";
        TemplateEmitter << YAML::Value << 100;
        TemplateEmitter << YAML::EndMap;

        TemplateEmitter << YAML::Key << "BOBYQA";
        TemplateEmitter << YAML::Value << YAML::BeginMap;

        TemplateEmitter << YAML::Key << 0; 
        TemplateEmitter << YAML::Value << YAML::BeginMap;

        TemplateEmitter << YAML::Key << "Enabled";
        TemplateEmitter << YAML::Value << true;

        TemplateEmitter << YAML::Key << "OptimizedEntries";
        TemplateEmitter << YAML::Value << YAML::BeginMap;
        TemplateEmitter << YAML::Flow << BOBYQA_Entries0;
        TemplateEmitter << YAML::EndMap;

        TemplateEmitter << YAML::Key << "AbsTol";
        TemplateEmitter << YAML::Value << 1e-4;
        TemplateEmitter << YAML::EndMap;

        TemplateEmitter << YAML::Key << 1; 
        TemplateEmitter << YAML::Value << YAML::BeginMap;

        TemplateEmitter << YAML::Key << "Enabled";
        TemplateEmitter << YAML::Value << true;

        TemplateEmitter << YAML::Key << "OptimizedEntries";
        TemplateEmitter << YAML::Value << YAML::BeginMap;
        TemplateEmitter << YAML::Flow << BOBYQA_Entries1;
        TemplateEmitter << YAML::EndMap;

        TemplateEmitter << YAML::Key << "AbsTol";
        TemplateEmitter << YAML::Value << 1e-4;
        TemplateEmitter << YAML::EndMap;

        TemplateEmitter << YAML::EndMap;

        //End inner layer: 
        TemplateEmitter << YAML::EndMap;

        //End sublayer - 3: OptimizerSettings
        // TemplateEmitter << YAML::EndMap;

        //End Layer - 0: InputParameters
        TemplateEmitter << YAML::EndMap;

        //Start Layer - 1: Outputs
        // TemplateEmitter << YAML::BeginMap;
        TemplateEmitter << YAML::Key << "Outputs";

        //Start inner layer: 
        TemplateEmitter << YAML::Value << YAML::BeginMap;

        TemplateEmitter << YAML::Key << "IsRegistered";
        TemplateEmitter << YAML::Value << false;

        std::vector<std::vector<float>> DummyMatrix;
        DummyMatrix.push_back( std::vector<float>{1.0f, 0.0f, 0.0f, 0.0f} );
        DummyMatrix.push_back( std::vector<float>{0.0f, 1.0f, 0.0f, 0.0f} );
        DummyMatrix.push_back( std::vector<float>{0.0f, 0.0f, 1.0f, 0.0f} );
        DummyMatrix.push_back( std::vector<float>{0, 0, 0, 1} );
        TemplateEmitter << YAML::Key << "TransformMatrix" << YAML::Flow << YAML::Value << DummyMatrix; 

        TemplateEmitter << YAML::Key << "LastSimilarity";
        TemplateEmitter << YAML::Value << 0.0;

        TemplateEmitter << YAML::Key << "NumEvaluations";
        TemplateEmitter << YAML::Value << 0.0;

        TemplateEmitter << YAML::Key << "RegistrationDateTime";
        TemplateEmitter << YAML::Value << "Fri Mar  6 12:05:47 2020";

        //End inner layer: 
        TemplateEmitter << YAML::EndMap;

        //End Layer - 1: Outputs
        TemplateEmitter << YAML::EndMap;

        //close file: 
        fileHandle.close(); 
    }; 
    /* ------------------------------------ Lambda start ----------------------------------------- */ 

    //To generate templates: 
    std::string templateFileName;
    if(!RegistrationMode.empty()){
        if(input.cmdOptionExists("-t")){
            templateFileName = input.getCmdOption("-t");

            if(templateFileName.empty()){
                if( 
                    input.cmdOptionExists("-tX") && 
                    input.cmdOptionExists("-tY") && 
                    input.cmdOptionExists("-tZ") && 
                    input.cmdOptionExists("-fixedPath") && 
                    input.cmdOptionExists("-movingPath") && 
                    input.cmdOptionExists("-outputPath"))
                {
                    templateFileName = "./targetTemplate3D.yaml";
                    Generate3DTemplateFromInputs(templateFileName); 
                }
                else{
                    if(RegistrationMode == "2D"){
                        templateFileName = "./template2D.yaml";
                        Generate2DTemplate(templateFileName); 
                    }
                    else if(RegistrationMode == "3D"){
                        templateFileName = "./template3D.yaml";
                        Generate3DTemplate(templateFileName); 
                    }
                }
            }
            else{
                if( 
                    input.cmdOptionExists("-tX") && 
                    input.cmdOptionExists("-tY") && 
                    input.cmdOptionExists("-tZ") && 
                    input.cmdOptionExists("-fixedPath") && 
                    input.cmdOptionExists("-movingPath") && 
                    input.cmdOptionExists("-outputPath"))
                {
                    Generate3DTemplateFromInputs(templateFileName); 
                }
                else{
                    if(RegistrationMode == "2D"){
                        Generate2DTemplate(templateFileName); 
                    }
                    else if(RegistrationMode == "3D"){
                        Generate3DTemplate(templateFileName); 
                    }
                }
            }

            std::cout << "Template was generated at: " << templateFileName << std::endl;
        }
        
    }
    else{
        std::cout << "Please specify: Registration modes, 2D or 3D." << std::endl; 
    }

    std::string fileName;
    if(input.cmdOptionExists("-c")){
        fileName = input.getCmdOption("-c");

        if(fileName.empty()){
            PrintUsage(std::cout);
            return 0;
        }
    }
    else{
        PrintUsage(std::cout);
        return 0;
    }
    
    MyTimer timer; 
    int stat = 0; 
    if(RegistrationMode == "2D"){
        std::cout << "2D Image Registration is launching..." << std::endl;

        Interpolator2D interpolateHandle(fileName);
        interpolateHandle.InitiateMappers(); 
        interpolateHandle.InitiateRegistrationHandle(); 

        Optimizers2D optmizer_handle(fileName);
        optmizer_handle.LinkToInterpolator(&interpolateHandle); 
        
        timer.tic(); 
        optmizer_handle.Optimize(); 
        timer.toc();

        if(stat != 0){
            std::cout << "Optimizer errors raised!" << std::endl; 
            return 1; 
        }

        optmizer_handle.GetOptimalTransform(); 
        optmizer_handle.ExportResults(); 

        float OptimalMatrix[9] = {0.0f}; 
        optmizer_handle.GetInternalMatrix(OptimalMatrix); 

        interpolateHandle.WriteOut(OptimalMatrix); 
        // interpolateHandle.WriteOut(); 
        timer.Duration(std::cout); 
    }
    else if(RegistrationMode == "3D"){
        std::cout << "3D Image Registration is launching..." << std::endl;

        //3D registration routine: 
        bool useIGTL = true; 
        if(input.cmdOptionExists("--avoid-igtl")){
            useIGTL = false; 
        }
        else{
            useIGTL = true; 
        }
        timer.tic(); 
        Interpolator3D interpolateHandle(fileName, useIGTL);
        interpolateHandle.InitiateMappers();
        timer.toc(); 
        double durTransData = timer.Duration(); 
        interpolateHandle.InitiateRegistrationHandle(); 

        Optimizers3D optmizer_handle(fileName);
        optmizer_handle.LinkToInterpolator(&interpolateHandle); 

        // tune initial offset: 
        // double transform[6] = {0.0}; 
        // optmizer_handle(transform); 
        timer.tic(); 
        optmizer_handle.Optimize(); 
        timer.toc(); 
        double durOpt = timer.Duration(); 

        if(stat != 0){
            std::cout << "Optimizer errors raised!" << std::endl; 
            return 1; 
        }

        optmizer_handle.GetOptimalTransform(); 
        optmizer_handle.ExportResults(); 

        float OptimalMatrix[16] = {0.0f}; 
        optmizer_handle.GetInternalMatrix(OptimalMatrix); 

        interpolateHandle.WriteOut(OptimalMatrix); 
        // timer.Duration(std::cout); 
        std::cout << "Elapsed time on: data load and transfer from H to D, " << durTransData << "[ms]. " << std::endl; 
        std::cout << "Elapsed time on: computation, " << durOpt << "[ms]. " << std::endl; 
    }
    else if(RegistrationMode == "2DTo3D"){
        Interpolator2DTo3D interpolateHandle(fileName);
        interpolateHandle.InitiateMappers();
        interpolateHandle.InitiateRegistrationHandle(); 
        
        Optimizers2DTo3D optmizer_handle(fileName);
        optmizer_handle.LinkToInterpolator(&interpolateHandle); 

        // tune initial offset: 
        // double transform[6] = {0.0}; 
        // optmizer_handle(transform); 
        timer.tic(); 
        optmizer_handle.Optimize(); 
        timer.toc(); 

        if(stat != 0){
            std::cout << "Optimizer errors raised!" << std::endl; 
            return 1; 
        }

        optmizer_handle.GetOptimalTransform(); 
        optmizer_handle.ExportResults(); 

        interpolateHandle.WriteOut(); 
        timer.Duration(std::cout); 
    }

//Test for 3D: 
    //Load Parameter file:
    // Interpolator3D interpolateHandle(fileName);
    // interpolateHandle.InitiateMappers();
    // interpolateHandle.InitiateRegistrationHandle(); 

    // Optimizers3D optmizer_handle(fileName);
    // optmizer_handle.LinkToInterpolator(&interpolateHandle); 

    // // tune initial offset: 
    // // double transform[6] = {0.0}; 
    // // optmizer_handle(transform); 

    // int stat = 0; 
    // stat = optmizer_handle.OptimizeWithDIRECT(); 
    // stat = optmizer_handle.OptimizeWithBOBYQA(); 

    // optmizer_handle.GetOptimalTransform(); 
    // optmizer_handle.ExportResults(); 

    // interpolateHandle.WriteOut(); 

    // for(float i = 0.0; i < 100; i = i + 0.1){
        // float3 rot; rot.x = 0.0f; rot.y = 0.0f; rot.z = 0.0f; 
        // float3 tsl; tsl.x = 0.0f; tsl.y = 0.0f; tsl.z = 0.0f; 
        // interpolateHandle.GenerateMappers(rot, tsl);
        // interpolateHandle.Interpolate();
        // float sim = interpolateHandle.GetSimilarityMeasure(); 
        // std::cout << sim; 
    // }

// Test for 2D: 
    // Interpolator2D interpolateHandle(fileName);
    // interpolateHandle.InitiateMappers(); 
    // interpolateHandle.InitiateRegistrationHandle(); 

    // Optimizers2D optmizer_handle(fileName);
    // optmizer_handle.LinkToInterpolator(&interpolateHandle); 
    // int stat = 0; 
    // stat = optmizer_handle.OptimizeWithDIRECT(); 
    // stat = optmizer_handle.OptimizeWithBOBYQA(); 

    // optmizer_handle.GetOptimalTransform(); 
    // optmizer_handle.ExportResults(); 

    // interpolateHandle.WriteOut(); 
    // float2 tsl; tsl.x = 0; tsl.y = 0; 

    // MyTimer timer; 
    // for(float i = 0.0f; i < 100.0f; i = i + 10.0f){
    //     interpolateHandle.GenerateMappers(i, tsl);
    //     timer.tic();
    //     interpolateHandle.Interpolate();
    //     timer.toc(); 
    //     timer.Duration(std::cout);
    // }
    

    return 0;
}