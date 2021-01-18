#include "src/Optimizers/optimizers.cuh"

#include "src/Utilities/my_cuda_helper.cuh"

#include <iostream>
#include <string>
#include <ctime>
#include <ratio>
#include <chrono>
#include <experimental/filesystem>

#include "nlopt.hpp"

template<typename T>
std::ostream& operator<<(std::ostream& os, const std::vector<T>& _vectorToPrint){
    for(auto element: _vectorToPrint){
        os << element << ", "; 
    }
    return os;
}

/* --------------------------------------- 2D Volume registration optimizer -------------------------------------- */
Optimizers2D::Optimizers2D(std::string FilePath){
    //Load parameter: 
    YAML_ParameterParser = YAML::LoadFile(FilePath); 
    YAML_ParameterPath = FilePath; 

    auto ExtractParam = [&](const std::string& _tagName, std::vector<double>& _targetVector){
        if(_targetVector.size() != 6){
            _targetVector.resize(6, 0); 
        }
        _targetVector[0] = YAML_ParameterParser["InputParameters"]["OptimizerSettings"][_tagName]["TX"].as<double>(); 
        _targetVector[1] = YAML_ParameterParser["InputParameters"]["OptimizerSettings"][_tagName]["TY"].as<double>(); 
        _targetVector[2] = YAML_ParameterParser["InputParameters"]["OptimizerSettings"][_tagName]["R"].as<double>(); 
        _targetVector[3] = YAML_ParameterParser["InputParameters"]["OptimizerSettings"][_tagName]["ScaleX"].as<double>(); 
        _targetVector[4] = YAML_ParameterParser["InputParameters"]["OptimizerSettings"][_tagName]["ScaleY"].as<double>(); 
        _targetVector[5] = YAML_ParameterParser["InputParameters"]["OptimizerSettings"][_tagName]["Shear"].as<double>(); 
    }; 

    //Initials: 
    InitialValues.resize(6, 0); 
    ExtractParam("InitialValues", InitialValues); 

    OptimalSolution = InitialValues; 

    //Bounds: 
    LowerBounds.resize(6, 0); 
    ExtractParam("LowerBounds", LowerBounds); 
    for(int i = 0; i < 3; ++i){
        LowerBounds[i] += InitialValues[i]; 
    }

    UpperBounds.resize(6, 0); 
    ExtractParam("UpperBounds", UpperBounds); 
    for(int i = 0; i < 3; ++i){
        UpperBounds[i] += InitialValues[i]; 
    }

    //DIRECT: 
    DIRECT_Enabled = YAML_ParameterParser["InputParameters"]["OptimizerSettings"]["DIRECT"]["Enabled"].as<bool>(); 
    DIRECT_OptEntryPairs.clear(); 
    DIRECT_OptEntryPairs.push_back(
        std::pair<OptimizationEntry, bool>( 
            OptimizationEntry::Translation, 
            YAML_ParameterParser["InputParameters"]["OptimizerSettings"]["DIRECT"]["OptimizedEntries"]["Translation"].as<bool>()
        )
    ); 
    DIRECT_OptEntryPairs.push_back(
        std::pair<OptimizationEntry, bool>( 
            OptimizationEntry::Rotation, 
            YAML_ParameterParser["InputParameters"]["OptimizerSettings"]["DIRECT"]["OptimizedEntries"]["Rotation"].as<bool>()
        )
    ); 
    DIRECT_OptEntryPairs.push_back(
        std::pair<OptimizationEntry, bool>( 
            OptimizationEntry::Scale, 
            YAML_ParameterParser["InputParameters"]["OptimizerSettings"]["DIRECT"]["OptimizedEntries"]["Scale"].as<bool>()
        )
    ); 
    DIRECT_OptEntryPairs.push_back(
        std::pair<OptimizationEntry, bool>( 
            OptimizationEntry::Shear, 
            YAML_ParameterParser["InputParameters"]["OptimizerSettings"]["DIRECT"]["OptimizedEntries"]["Shear"].as<bool>()
        )
    ); 

    DIRECT_MaxEvals = YAML_ParameterParser["InputParameters"]["OptimizerSettings"]["DIRECT"]["MaxEvaluations"].as<int>(); 

    //BOBYQA: 
    BOBYQA_Enabled = false; 
    BOBYQA_Rounds = (int)YAML_ParameterParser["InputParameters"]["OptimizerSettings"]["BOBYQA"].size(); 
    for(int i = 0; i < BOBYQA_Rounds; ++i){
        if(!YAML_ParameterParser["InputParameters"]["OptimizerSettings"]["BOBYQA"][std::to_string(i)]){
            std::cout << "[Error]: BOBYQA rounds are not matching and consistent. " << std::endl; 
            std::exit(EXIT_FAILURE); 
        }
        else{
            BOBYQA_Enabled = BOBYQA_Enabled || YAML_ParameterParser["InputParameters"]["OptimizerSettings"]["BOBYQA"][std::to_string(i)]["Enabled"].as<bool>(); 
        }
    }

    patchSize = YAML_ParameterParser["InputParameters"]["RegistrationParameters"]["PatchSize"].as<int>(); 

    SimilarityMeasure = 0.0f; 
    EvalCounter = 0; 
}

void Optimizers2D::LinkToInterpolator(Interpolator2D* external_Interpolator){
    interpolation_handle = external_Interpolator; 
}

void Optimizers2D::ShowCurrentResults(){
    float2 Translation, Scale; 
    float Rotation, Shear; 
    Translation.x = (float)OptimalSolution[0]; 
    Translation.y = (float)OptimalSolution[1]; 

    Rotation = (float)OptimalSolution[2]; 

    Scale.x = (float)OptimalSolution[3]; 
    Scale.y = (float)OptimalSolution[4]; 

    Shear = (float)OptimalSolution[5]; 

    interpolation_handle->GenerateMappers(Rotation, Translation, Scale, Shear); 
    interpolation_handle->Interpolate(); 
    SimilarityMeasure = interpolation_handle->GetSimilarityMeasure(patchSize); 
}

double Optimizers2D::operator()(const std::vector<double> &x){ //x: size of 6
    float2 Translation, Scale; 
    float Rotation, Shear; 
    Translation.x = (float)OptimalSolution[0]; 
    Translation.y = (float)OptimalSolution[1]; 

    Rotation = (float)OptimalSolution[2]; 

    Scale.x = (float)OptimalSolution[3]; 
    Scale.y = (float)OptimalSolution[4]; 

    Shear = (float)OptimalSolution[5]; 

    int ParameterIdxOffset = 0; 
    for(int i = 0; i < CurrentHolding_OptEntries.size(); ++i){
        switch (CurrentHolding_OptEntries[i])
        {
        case OptimizationEntry::Translation:
            Translation.x = (float)x[ParameterIdxOffset]; 
            Translation.y = (float)x[ParameterIdxOffset + 1]; 
            ParameterIdxOffset += 2; 
            break;
        case OptimizationEntry::Rotation:
            Rotation = (float)x[ParameterIdxOffset]; 
            ParameterIdxOffset += 1; 
            break;
        case OptimizationEntry::Scale:
            Scale.x = (float)x[ParameterIdxOffset]; 
            Scale.y = (float)x[ParameterIdxOffset + 1]; 
            ParameterIdxOffset += 2; 
            break;
        case OptimizationEntry::Shear:
            Shear = (float)x[ParameterIdxOffset]; 
            ParameterIdxOffset += 1; 
            break;
        default:
            break;
        }
    }

    //Evaluate similarityMeasure: 
    interpolation_handle->GenerateMappers(Rotation, Translation, Scale, Shear); 
    interpolation_handle->Interpolate(); 
    SimilarityMeasure = interpolation_handle->GetSimilarityMeasure(patchSize); 

    ++EvalCounter; 
    return -std::log(SimilarityMeasure); 
}

double Optimizers2D::wrapper(const std::vector<double> &x, std::vector<double> &grad, void *data){
    return reinterpret_cast<Optimizers2D*>(data)->operator()(x); 
}

void Optimizers2D::Optimize(){
    this->ShowCurrentResults(); 

    //functors: 
    auto ExtractBounds = [&](std::vector<double>& _l, std::vector<double>& _u, std::vector<double>& _x){
        int ParameterIdxOffset = 0; 
        for(int idxEntry = 0; idxEntry < CurrentHolding_OptEntries.size(); ++idxEntry){
            switch (CurrentHolding_OptEntries[idxEntry])
            {
            case OptimizationEntry::Translation:
                _l[0 + ParameterIdxOffset] = LowerBounds[0]; 
                _l[1 + ParameterIdxOffset] = LowerBounds[1]; 

                _u[0 + ParameterIdxOffset] = UpperBounds[0]; 
                _u[1 + ParameterIdxOffset] = UpperBounds[1]; 

                _x[0 + ParameterIdxOffset] = OptimalSolution[0]; 
                _x[1 + ParameterIdxOffset] = OptimalSolution[1]; 
                ParameterIdxOffset += 2; 
                break;
            case OptimizationEntry::Rotation:
                _l[ParameterIdxOffset] = LowerBounds[2]; 

                _u[ParameterIdxOffset] = UpperBounds[2]; 

                _x[ParameterIdxOffset] = OptimalSolution[2]; 
                ++ParameterIdxOffset; 
                break;
            case OptimizationEntry::Scale:
                _l[0 + ParameterIdxOffset] = LowerBounds[3]; 
                _l[1 + ParameterIdxOffset] = LowerBounds[4]; 

                _u[0 + ParameterIdxOffset] = UpperBounds[3]; 
                _u[1 + ParameterIdxOffset] = UpperBounds[4]; 

                _x[0 + ParameterIdxOffset] = OptimalSolution[3]; 
                _x[1 + ParameterIdxOffset] = OptimalSolution[4]; 
                ParameterIdxOffset += 2; 
                break;
            case OptimizationEntry::Shear:
                _l[ParameterIdxOffset] = LowerBounds[5]; 

                _u[ParameterIdxOffset] = UpperBounds[5]; 

                _x[ParameterIdxOffset] = OptimalSolution[5]; 
                ++ParameterIdxOffset; 
                break;
            default:
                break;
            }
        }
    }; 
    auto AsignBackToOptimal = [&](std::vector<double>& _x){
        int ParameterIdxOffset = 0; 
        for(int idxEntry = 0; idxEntry < CurrentHolding_OptEntries.size(); ++idxEntry){
            switch (CurrentHolding_OptEntries[idxEntry])
            {
            case OptimizationEntry::Translation:
                OptimalSolution[0] = _x[0 + ParameterIdxOffset]; 
                OptimalSolution[1] = _x[1 + ParameterIdxOffset]; 
                ParameterIdxOffset += 2; 
                break;
            case OptimizationEntry::Rotation:
                OptimalSolution[2] = _x[ParameterIdxOffset]; 
                ++ParameterIdxOffset; 
                break;
            case OptimizationEntry::Scale:
                OptimalSolution[3] = _x[0 + ParameterIdxOffset]; 
                OptimalSolution[4] = _x[1 + ParameterIdxOffset]; 
                ParameterIdxOffset += 2; 
                break;
            case OptimizationEntry::Shear:
                OptimalSolution[5] = _x[ParameterIdxOffset]; 
                ++ParameterIdxOffset; 
                break;
            default:
                break;
            }
        }
    }; 
    
    //DIRECT: 
    if(DIRECT_Enabled){
        CurrentHolding_OptEntries.clear(); 
        unsigned int NumberOfParameters = 0; 
        for(int idxDIRECT_Entry = 0; idxDIRECT_Entry < DIRECT_OptEntryPairs.size(); ++idxDIRECT_Entry){
            if(DIRECT_OptEntryPairs[idxDIRECT_Entry].second){
                CurrentHolding_OptEntries.push_back(DIRECT_OptEntryPairs[idxDIRECT_Entry].first); 
                switch (DIRECT_OptEntryPairs[idxDIRECT_Entry].first)
                {
                case OptimizationEntry::Translation:
                    NumberOfParameters += 2; 
                    break;
                case OptimizationEntry::Rotation:
                    NumberOfParameters += 1; 
                    break;
                case OptimizationEntry::Scale:
                    NumberOfParameters += 2; 
                    break;
                case OptimizationEntry::Shear:
                    NumberOfParameters += 1; 
                    break;
                default:
                    break;
                }
            }
        }
        
        if(!CurrentHolding_OptEntries.empty()){
            std::vector<double> l_LowerBounds(NumberOfParameters); 
            std::vector<double> l_UpperBounds(NumberOfParameters); 
            std::vector<double> l_X(NumberOfParameters); 
            ExtractBounds(l_LowerBounds, l_UpperBounds, l_X); 

            //start DIRECT opt: 
            nlopt::opt MyOptimizer(nlopt::GN_DIRECT_L, NumberOfParameters); 
            MyOptimizer.set_min_objective(Optimizers2D::wrapper, this); 
            MyOptimizer.set_lower_bounds(l_LowerBounds); 
            MyOptimizer.set_upper_bounds(l_UpperBounds); 
            MyOptimizer.set_maxeval(DIRECT_MaxEvals); 

            double minf = 0; 
            nlopt::result result; 
            try{
                result = MyOptimizer.optimize(l_X, minf);
                std::cout << "Optimization stage-DIRECT finished: similarity -> "  << std::exp(-minf) << ", accu evaluations -> " << EvalCounter << std::endl; 
            }
            catch(std::exception &e) {
                std::cout << "nlopt failed: " << e.what() << std::endl;
            }
            AsignBackToOptimal(l_X); 
            SimilarityMeasure = std::exp(-minf); 
        }
    }

    if(BOBYQA_Enabled){
        for(int idxBOBYQA_Round = 0; idxBOBYQA_Round < BOBYQA_Rounds; ++idxBOBYQA_Round){
            //Parse the corresponsing round parameters: 
            BOBYQA_Enabled = YAML_ParameterParser["InputParameters"]["OptimizerSettings"]["BOBYQA"][std::to_string(idxBOBYQA_Round)]["Enabled"].as<bool>(); 
            if(!BOBYQA_Enabled){
                continue; 
            }
            BOBYQA_OptEntryPairs.clear(); 
            BOBYQA_OptEntryPairs.push_back(
                std::pair<OptimizationEntry, bool>( 
                    OptimizationEntry::Translation, 
                    YAML_ParameterParser["InputParameters"]["OptimizerSettings"]["BOBYQA"][std::to_string(idxBOBYQA_Round)]["OptimizedEntries"]["Translation"].as<bool>()
                )
            ); 
            BOBYQA_OptEntryPairs.push_back(
                std::pair<OptimizationEntry, bool>( 
                    OptimizationEntry::Rotation, 
                    YAML_ParameterParser["InputParameters"]["OptimizerSettings"]["BOBYQA"][std::to_string(idxBOBYQA_Round)]["OptimizedEntries"]["Rotation"].as<bool>()
                )
            ); 
            BOBYQA_OptEntryPairs.push_back(
                std::pair<OptimizationEntry, bool>( 
                    OptimizationEntry::Scale, 
                    YAML_ParameterParser["InputParameters"]["OptimizerSettings"]["BOBYQA"][std::to_string(idxBOBYQA_Round)]["OptimizedEntries"]["Scale"].as<bool>()
                )
            ); 
            BOBYQA_OptEntryPairs.push_back(
                std::pair<OptimizationEntry, bool>( 
                    OptimizationEntry::Shear, 
                    YAML_ParameterParser["InputParameters"]["OptimizerSettings"]["BOBYQA"][std::to_string(idxBOBYQA_Round)]["OptimizedEntries"]["Shear"].as<bool>()
                )
            ); 
            BOBYQA_AbsTol = YAML_ParameterParser["InputParameters"]["OptimizerSettings"]["BOBYQA"][std::to_string(idxBOBYQA_Round)]["AbsTol"].as<double>(); 

            //optimazation launches: 
            CurrentHolding_OptEntries.clear(); 
            unsigned int NumberOfParameters = 0; 
            for(int idxBOBYQA_Entry = 0; idxBOBYQA_Entry < BOBYQA_OptEntryPairs.size(); ++idxBOBYQA_Entry){
                if(BOBYQA_OptEntryPairs[idxBOBYQA_Entry].second){
                    CurrentHolding_OptEntries.push_back(BOBYQA_OptEntryPairs[idxBOBYQA_Entry].first); 
                    switch (BOBYQA_OptEntryPairs[idxBOBYQA_Entry].first)
                    {
                    case OptimizationEntry::Translation:
                        NumberOfParameters += 2; 
                        break;
                    case OptimizationEntry::Rotation:
                        NumberOfParameters += 1; 
                        break;
                    case OptimizationEntry::Scale:
                        NumberOfParameters += 2; 
                        break;
                    case OptimizationEntry::Shear:
                        NumberOfParameters += 1; 
                        break;
                    default:
                        break;
                    }
                }
            }

            if(!CurrentHolding_OptEntries.empty()){
                std::vector<double> l_LowerBounds(NumberOfParameters); 
                std::vector<double> l_UpperBounds(NumberOfParameters); 
                std::vector<double> l_X(NumberOfParameters); 
                ExtractBounds(l_LowerBounds, l_UpperBounds, l_X); 

                //start BOBYQA opt: 
                nlopt::opt MyOptimizer(nlopt::LN_BOBYQA, NumberOfParameters); 
                MyOptimizer.set_min_objective(Optimizers2D::wrapper, this); 
                MyOptimizer.set_lower_bounds(l_LowerBounds); 
                MyOptimizer.set_upper_bounds(l_UpperBounds); 
                // if(idxBOBYQA_Round == 0){
                    MyOptimizer.set_xtol_rel(BOBYQA_AbsTol); 
                // }
                // else{
                //     MyOptimizer.set_ftol_rel(BOBYQA_AbsTol); 
                // }

                double minf = 0;
                nlopt::result result; 
                try{
                    result = MyOptimizer.optimize(l_X, minf);
                    std::cout << "Optimization stage-BOBYQA finished: similarity -> "  << std::exp(-minf) << ", accu evaluations -> " << EvalCounter << std::endl; 
                }
                catch(std::exception &e) {
                    std::cout << "nlopt failed: " << e.what() << std::endl;
                }
                AsignBackToOptimal(l_X); 
                SimilarityMeasure = std::exp(-minf); 
            }
        }
    }
}

void Optimizers2D::GetOptimalTransform(float* outMatrix){
    float2 Translation, Scale; 
    float Rotation, Shear; 
    Translation.x = (float)OptimalSolution[0]; 
    Translation.y = (float)OptimalSolution[1]; 

    Rotation = (float)OptimalSolution[2]; 

    Scale.x = (float)OptimalSolution[3]; 
    Scale.y = (float)OptimalSolution[4]; 

    Shear = (float)OptimalSolution[5]; 

    interpolation_handle->GetMovingToFixedTransform(Rotation, Translation, Scale, Shear, outMatrix);
}

void Optimizers2D::GetOptimalTransform(){
    float2 Translation, Scale; 
    float Rotation, Shear; 
    Translation.x = (float)OptimalSolution[0]; 
    Translation.y = (float)OptimalSolution[1]; 

    Rotation = (float)OptimalSolution[2]; 

    Scale.x = (float)OptimalSolution[3]; 
    Scale.y = (float)OptimalSolution[4]; 

    Shear = (float)OptimalSolution[5]; 

    interpolation_handle->GetMovingToFixedTransform(Rotation, Translation, Scale, Shear, MovingToFixed_Optimal);
}

void Optimizers2D::GetInternalMatrix(float* OutMatrix){
    for(int i = 0; i < 9; ++i){
        OutMatrix[i] = MovingToFixed_Optimal[i]; 
    }
}

void Optimizers2D::ExportResults(){
    std::vector<std::vector<float>> resultMatrix; 
    for(int row_it = 0; row_it < 3; ++row_it){
        std::vector<float> tmpVector(3, 0.0f); 
        for(int col_it = 0; col_it < 3; ++col_it){
            tmpVector[col_it] = MovingToFixed_Optimal[col_it + 3 * row_it]; 
        }
        resultMatrix.push_back(tmpVector); 
    }

    std::string ParameterExportPath = YAML_ParameterPath; 
    ParameterExportPath.insert(YAML_ParameterPath.find('.'), "_result"); 
    std::ofstream outputFileHandle(ParameterExportPath); 

    YAML_ParameterParser["Outputs"]["IsRegistered"] = true; 
    YAML_ParameterParser["Outputs"]["TransformMatrix"] = resultMatrix; 
    YAML_ParameterParser["Outputs"]["LastSimilarity"] = SimilarityMeasure; 
    YAML_ParameterParser["Outputs"]["NumEvaluations"] = EvalCounter;

    //time stamp: 
    currentTime = std::chrono::system_clock::to_time_t(std::chrono::system_clock::now()); 
    std::string currentTimeParsed_str(ctime(&currentTime)); 
    YAML_ParameterParser["Outputs"]["RegistrationDateTime"] = currentTimeParsed_str; 
    
    outputFileHandle << YAML_ParameterParser; 

    std::cout << "Optimal transform matrix found: " << std::endl; 
    MatrixS3X3Print(MovingToFixed_Optimal); 

    std::cout << "Last Similarity Estimate: " << SimilarityMeasure << ", with " << EvalCounter << " evaluations. " << std::endl; 
}

/* --------------------------------------- 3D Volume registration optimizer -------------------------------------- */
//member function definitions: 
Optimizers3D::Optimizers3D(std::string FilePath){
    //Load parameter: 
    YAML_ParameterParser = YAML::LoadFile(FilePath); 
    YAML_ParameterPath = FilePath; 

    auto ExtractParam = [&](const YAML::Node& _node, const std::string& _tagName, std::vector<double>& _targetVector){
        if(_targetVector.size() != 12){
            _targetVector.resize(12, 0); 
        }
        _targetVector[0] = _node["InputParameters"]["OptimizerSettings"][_tagName]["TX"].as<double>(); 
        _targetVector[1] = _node["InputParameters"]["OptimizerSettings"][_tagName]["TY"].as<double>(); 
        _targetVector[2] = _node["InputParameters"]["OptimizerSettings"][_tagName]["TZ"].as<double>(); 
        _targetVector[3] = _node["InputParameters"]["OptimizerSettings"][_tagName]["RX"].as<double>(); 
        _targetVector[4] = _node["InputParameters"]["OptimizerSettings"][_tagName]["RY"].as<double>(); 
        _targetVector[5] = _node["InputParameters"]["OptimizerSettings"][_tagName]["RZ"].as<double>(); 
        _targetVector[6] = _node["InputParameters"]["OptimizerSettings"][_tagName]["ScaleX"].as<double>(); 
        _targetVector[7] = _node["InputParameters"]["OptimizerSettings"][_tagName]["ScaleY"].as<double>(); 
        _targetVector[8] = _node["InputParameters"]["OptimizerSettings"][_tagName]["ScaleZ"].as<double>(); 
        _targetVector[9] = _node["InputParameters"]["OptimizerSettings"][_tagName]["ShearX"].as<double>(); 
        _targetVector[10] = _node["InputParameters"]["OptimizerSettings"][_tagName]["ShearY"].as<double>(); 
        _targetVector[11] = _node["InputParameters"]["OptimizerSettings"][_tagName]["ShearZ"].as<double>(); 
    }; 

    //Initial starting values: 
    InitialValues.resize(12, 0); 
    ExtractParam(YAML_ParameterParser, "InitialValues", InitialValues); 
    
    OptimalSolution = InitialValues; 

    //Bounds: 
    LowerBounds.resize(12, 0); 
    ExtractParam(YAML_ParameterParser, "LowerBounds", LowerBounds); 
    for(int i = 0; i < 6; ++i){
        LowerBounds[i] += InitialValues[i]; 
    }

    UpperBounds.resize(12, 0); 
    ExtractParam(YAML_ParameterParser, "UpperBounds", UpperBounds); 
    for(int i = 0; i < 6; ++i){
        UpperBounds[i] += InitialValues[i]; 
    }

    //DIRECT: 
    DIRECT_Enabled = YAML_ParameterParser["InputParameters"]["OptimizerSettings"]["DIRECT"]["Enabled"].as<bool>(); 
    DIRECT_OptEntryPairs.clear(); 
    DIRECT_OptEntryPairs.push_back(
        std::pair<OptimizationEntry, bool>( 
            OptimizationEntry::Translation, 
            YAML_ParameterParser["InputParameters"]["OptimizerSettings"]["DIRECT"]["OptimizedEntries"]["Translation"].as<bool>()
        )
    ); 
    DIRECT_OptEntryPairs.push_back(
        std::pair<OptimizationEntry, bool>( 
            OptimizationEntry::Rotation, 
            YAML_ParameterParser["InputParameters"]["OptimizerSettings"]["DIRECT"]["OptimizedEntries"]["Rotation"].as<bool>()
        )
    ); 
    DIRECT_OptEntryPairs.push_back(
        std::pair<OptimizationEntry, bool>( 
            OptimizationEntry::Scale, 
            YAML_ParameterParser["InputParameters"]["OptimizerSettings"]["DIRECT"]["OptimizedEntries"]["Scale"].as<bool>()
        )
    ); 
    DIRECT_OptEntryPairs.push_back(
        std::pair<OptimizationEntry, bool>( 
            OptimizationEntry::Shear, 
            YAML_ParameterParser["InputParameters"]["OptimizerSettings"]["DIRECT"]["OptimizedEntries"]["Shear"].as<bool>()
        )
    ); 

    DIRECT_MaxEvals = YAML_ParameterParser["InputParameters"]["OptimizerSettings"]["DIRECT"]["MaxEvaluations"].as<int>(); 

    if(DIRECT_Enabled){
        std::cout << "Global translation search enabled. " << std::endl; 
    }
    else{
        std::cout << "Local search only. " << std::endl; 
    }

    //BOBYQA: 
    BOBYQA_Enabled = false; 
    BOBYQA_Rounds = (int)YAML_ParameterParser["InputParameters"]["OptimizerSettings"]["BOBYQA"].size(); 
    for(int i = 0; i < BOBYQA_Rounds; ++i){
        if(!YAML_ParameterParser["InputParameters"]["OptimizerSettings"]["BOBYQA"][std::to_string(i)]){
            std::cout << "[Error]: BOBYQA rounds are not matching and consistent. " << std::endl; 
            std::exit(EXIT_FAILURE); 
        }
        else{
            BOBYQA_Enabled = BOBYQA_Enabled || YAML_ParameterParser["InputParameters"]["OptimizerSettings"]["BOBYQA"][std::to_string(i)]["Enabled"].as<bool>(); 
        }
        
    }

    patchSize = YAML_ParameterParser["InputParameters"]["RegistrationParameters"]["PatchSize"].as<int>(); 

    SimilarityMeasure = 0.0f; 
    EvalCounter = 0; 
    m_stepMarker = -1; 
}

void Optimizers3D::LinkToInterpolator(Interpolator3D* external_Interpolator){
    interpolation_handle = external_Interpolator; 
}

void Optimizers3D::ShowCurrentResults(){
    float3 Translation, Rotation, Scale, Shear; 
    Translation.x = (float)OptimalSolution[0]; 
    Translation.y = (float)OptimalSolution[1]; 
    Translation.z = (float)OptimalSolution[2]; 

    Rotation.x = (float)OptimalSolution[3]; 
    Rotation.y = (float)OptimalSolution[4]; 
    Rotation.z = (float)OptimalSolution[5]; 

    Scale.x = (float)OptimalSolution[6]; 
    Scale.y = (float)OptimalSolution[7]; 
    Scale.z = (float)OptimalSolution[8]; 

    Shear.x = (float)OptimalSolution[9]; 
    Shear.y = (float)OptimalSolution[10]; 
    Shear.z = (float)OptimalSolution[11]; 

    interpolation_handle->GenerateMappers(Rotation, Translation, Scale, Shear); 
    interpolation_handle->Interpolate(); 
    SimilarityMeasure = interpolation_handle->GetSimilarityMeasure(patchSize); 

    m_logFileId << "Step,Transform[16],Translation_x,Translation_y,Translation_z,Rotation_x,Rotation_y,Rotation_z,Scale_x,Scale_y,Scale_z,Shear_x,Shear_y,Shear_z,Similarity,-log(sim),IteratorCounter" << std::endl; 
    
    //Dump results: 
    m_logFileId << m_stepMarker << ","; 
    float matrix[16] = {0}; 
    interpolation_handle->GetCurrentTransform(matrix); 
    for(int i = 0; i < 16; ++i){
        m_logFileId << matrix[i] << ","; 
    }
    m_logFileId << Translation.x << "," << Translation.y << "," << Translation.z << "," << 
    Rotation.x << "," << Rotation.y << "," << Rotation.z << "," << 
    Scale.x << "," << Scale.y << "," << Scale.z << "," << 
    Shear.x << "," << Shear.y << "," << Shear.z << "," << 
    SimilarityMeasure << "," << EvalCounter << std::endl; 

    //Randomize test: 
    // {
    //     m_logFileId << "Tx,Ty,Tz,Sim" << std::endl; 
    //     // for(double translationShiftZ = -50.0; translationShiftZ <= 50.0; translationShiftZ += 0.1){
    //         for(double translationShiftY = -50.0; translationShiftY <= 50.0; translationShiftY += 0.1){
    //             for(double translationShiftX = -50.0; translationShiftX <= 50.0; translationShiftX += 0.1){
    //                 Translation.x = (float)OptimalSolution[0] + translationShiftX; 
    //                 Translation.y = (float)OptimalSolution[1] + translationShiftY; 
    //                 Translation.z = (float)OptimalSolution[2]; 

    //                 interpolation_handle->GenerateMappers(Rotation, Translation, Scale, Shear); 
    //                 interpolation_handle->Interpolate(); 
    //                 SimilarityMeasure = interpolation_handle->GetSimilarityMeasure(patchSize); 

    //                 m_logFileId << Translation.x << "," << Translation.y << "," << Translation.z << "," << SimilarityMeasure << std::endl; 
    //                 std::cout << translationShiftX << ", " << translationShiftY << std::endl; 
    //             }
    //         }
    //     // }

    //     m_logFileId.close(); 

    // }
}

double Optimizers3D::operator()(const std::vector<double> &x){
    float3 Translation, Rotation, Scale, Shear; 
    Translation.x = (float)OptimalSolution[0]; 
    Translation.y = (float)OptimalSolution[1]; 
    Translation.z = (float)OptimalSolution[2]; 

    Rotation.x = (float)OptimalSolution[3]; 
    Rotation.y = (float)OptimalSolution[4]; 
    Rotation.z = (float)OptimalSolution[5]; 

    Scale.x = (float)OptimalSolution[6]; 
    Scale.y = (float)OptimalSolution[7]; 
    Scale.z = (float)OptimalSolution[8]; 

    Shear.x = (float)OptimalSolution[9]; 
    Shear.y = (float)OptimalSolution[10]; 
    Shear.z = (float)OptimalSolution[11]; 

    if(x.size() != (CurrentHolding_OptEntries.size() * 3)){
        std::cout << "[Error]: opt parameters do not match with entries! " << std::endl; 
        return 0; 
    }
    else{
        for(int i = 0; i < CurrentHolding_OptEntries.size(); ++i){
            switch (CurrentHolding_OptEntries[i])
            {
            case OptimizationEntry::Translation:
                Translation.x = (float)x[0 + i * 3]; 
                Translation.y = (float)x[1 + i * 3]; 
                Translation.z = (float)x[2 + i * 3]; 
                break;
            case OptimizationEntry::Rotation:
                Rotation.x = (float)x[0 + i * 3]; 
                Rotation.y = (float)x[1 + i * 3]; 
                Rotation.z = (float)x[2 + i * 3]; 
                break;
            case OptimizationEntry::Scale:
                Scale.x = (float)x[0 + i * 3]; 
                Scale.y = (float)x[1 + i * 3]; 
                Scale.z = (float)x[2 + i * 3]; 
                break;
            case OptimizationEntry::Shear:
                Shear.x = (float)x[0 + i * 3]; 
                Shear.y = (float)x[1 + i * 3]; 
                Shear.z = (float)x[2 + i * 3]; 
                break;
            default:
                break;
            }
        }
    }

    //Evaluate similarityMeasure: 
    interpolation_handle->GenerateMappers(Rotation, Translation, Scale, Shear); 
    interpolation_handle->Interpolate(); 
    SimilarityMeasure = interpolation_handle->GetSimilarityMeasure(patchSize); 

    ++EvalCounter; 

    //Dump results: 
    m_logFileId << m_stepMarker << ","; 
    float matrix[16] = {0}; 
    interpolation_handle->GetCurrentTransform(matrix); 
    for(int i = 0; i < 16; ++i){
        m_logFileId << matrix[i] << ","; 
    }
    m_logFileId << Translation.x << "," << Translation.y << "," << Translation.z << "," << 
    Rotation.x << "," << Rotation.y << "," << Rotation.z << "," << 
    Scale.x << "," << Scale.y << "," << Scale.z << "," << 
    Shear.x << "," << Shear.y << "," << Shear.z << "," << 
    SimilarityMeasure << "," << -std::log(SimilarityMeasure) << "," << EvalCounter << std::endl; 

    if(patchSize == 0){
        return SimilarityMeasure; 
    }
    else{
        return -std::log(SimilarityMeasure); 
    }
}

double Optimizers3D::wrapper(const std::vector<double> &x, std::vector<double> &grad, void *data){
    return reinterpret_cast<Optimizers3D*>(data)->operator()(x); 
}

void Optimizers3D::Optimize(){
    //Initialize log: 
    m_logFileId.open(interpolation_handle->GetOutputPath() + "/RegistrationInfo.log", std::ofstream::out); 

    this->ShowCurrentResults(); 

    //functors: 
    auto ExtractBounds = [&](std::vector<double>& _l, std::vector<double>& _u, std::vector<double>& _x){
        for(int idxEntry = 0; idxEntry < CurrentHolding_OptEntries.size(); ++idxEntry){
            switch (CurrentHolding_OptEntries[idxEntry])
            {
            case OptimizationEntry::Translation:
                _l[0 + idxEntry * 3] = LowerBounds[0]; 
                _l[1 + idxEntry * 3] = LowerBounds[1]; 
                _l[2 + idxEntry * 3] = LowerBounds[2]; 

                _u[0 + idxEntry * 3] = UpperBounds[0]; 
                _u[1 + idxEntry * 3] = UpperBounds[1]; 
                _u[2 + idxEntry * 3] = UpperBounds[2]; 

                _x[0 + idxEntry * 3] = OptimalSolution[0]; 
                _x[1 + idxEntry * 3] = OptimalSolution[1]; 
                _x[2 + idxEntry * 3] = OptimalSolution[2]; 
                
                break;
            case OptimizationEntry::Rotation:
                _l[0 + idxEntry * 3] = LowerBounds[3]; 
                _l[1 + idxEntry * 3] = LowerBounds[4]; 
                _l[2 + idxEntry * 3] = LowerBounds[5]; 

                _u[0 + idxEntry * 3] = UpperBounds[3]; 
                _u[1 + idxEntry * 3] = UpperBounds[4]; 
                _u[2 + idxEntry * 3] = UpperBounds[5]; 

                _x[0 + idxEntry * 3] = OptimalSolution[3]; 
                _x[1 + idxEntry * 3] = OptimalSolution[4]; 
                _x[2 + idxEntry * 3] = OptimalSolution[5]; 

                break;
            case OptimizationEntry::Scale:
                _l[0 + idxEntry * 3] = LowerBounds[6]; 
                _l[1 + idxEntry * 3] = LowerBounds[7]; 
                _l[2 + idxEntry * 3] = LowerBounds[8]; 

                _u[0 + idxEntry * 3] = UpperBounds[6]; 
                _u[1 + idxEntry * 3] = UpperBounds[7]; 
                _u[2 + idxEntry * 3] = UpperBounds[8]; 

                _x[0 + idxEntry * 3] = OptimalSolution[6]; 
                _x[1 + idxEntry * 3] = OptimalSolution[7]; 
                _x[2 + idxEntry * 3] = OptimalSolution[8]; 
                
                break;
            case OptimizationEntry::Shear:
                _l[0 + idxEntry * 3] = LowerBounds[9]; 
                _l[1 + idxEntry * 3] = LowerBounds[10]; 
                _l[2 + idxEntry * 3] = LowerBounds[11]; 

                _u[0 + idxEntry * 3] = UpperBounds[9]; 
                _u[1 + idxEntry * 3] = UpperBounds[10]; 
                _u[2 + idxEntry * 3] = UpperBounds[11]; 

                _x[0 + idxEntry * 3] = OptimalSolution[9]; 
                _x[1 + idxEntry * 3] = OptimalSolution[10]; 
                _x[2 + idxEntry * 3] = OptimalSolution[11]; 

                break;
            default:
                break;
            }
        }
    }; 
    auto AsignBackToOptimal = [&](std::vector<double>& _x){
        for(int idxEntry = 0; idxEntry < CurrentHolding_OptEntries.size(); ++idxEntry){
            switch (CurrentHolding_OptEntries[idxEntry])
            {
            case OptimizationEntry::Translation:
                OptimalSolution[0] = _x[0 + idxEntry * 3]; 
                OptimalSolution[1] = _x[1 + idxEntry * 3]; 
                OptimalSolution[2] = _x[2 + idxEntry * 3]; 
                break;
            case OptimizationEntry::Rotation:
                OptimalSolution[3] = _x[0 + idxEntry * 3]; 
                OptimalSolution[4] = _x[1 + idxEntry * 3]; 
                OptimalSolution[5] = _x[2 + idxEntry * 3]; 
                break;
            case OptimizationEntry::Scale:
                OptimalSolution[6] = _x[0 + idxEntry * 3]; 
                OptimalSolution[7] = _x[1 + idxEntry * 3]; 
                OptimalSolution[8] = _x[2 + idxEntry * 3]; 
                break;
            case OptimizationEntry::Shear:
                OptimalSolution[9] = _x[0 + idxEntry * 3]; 
                OptimalSolution[10] = _x[1 + idxEntry * 3]; 
                OptimalSolution[11] = _x[2 + idxEntry * 3]; 
                break;
            default:
                break;
            }
        }
    }; 
    
    //DIRECT: 
    if(DIRECT_Enabled){
        m_stepMarker = 0; 
        CurrentHolding_OptEntries.clear(); 
        for(int idxDIRECT_Entry = 0; idxDIRECT_Entry < DIRECT_OptEntryPairs.size(); ++idxDIRECT_Entry){
            if(DIRECT_OptEntryPairs[idxDIRECT_Entry].second){
                CurrentHolding_OptEntries.push_back(DIRECT_OptEntryPairs[idxDIRECT_Entry].first); 
            }
        }
        
        if(!CurrentHolding_OptEntries.empty()){
            unsigned int NumberOfParameters = (unsigned int)CurrentHolding_OptEntries.size() * 3; 
            std::vector<double> l_LowerBounds(NumberOfParameters); 
            std::vector<double> l_UpperBounds(NumberOfParameters); 
            std::vector<double> l_X(NumberOfParameters); 
            ExtractBounds(l_LowerBounds, l_UpperBounds, l_X); 

            //start DIRECT opt: 
            nlopt::opt MyOptimizer(nlopt::GN_ORIG_DIRECT_L, NumberOfParameters); 
            if(patchSize == 0){
                MyOptimizer.set_max_objective(Optimizers3D::wrapper, this); 
            }
            else{
                MyOptimizer.set_min_objective(Optimizers3D::wrapper, this); 
            }
            MyOptimizer.set_lower_bounds(l_LowerBounds); 
            MyOptimizer.set_upper_bounds(l_UpperBounds); 
            MyOptimizer.set_maxeval(DIRECT_MaxEvals); 

            double minf = 0;
            nlopt::result result; 
            try{
                result = MyOptimizer.optimize(l_X, minf);
                if(patchSize == 0){
                    std::cout << "Optimization stage-DIRECT finished: similarity -> "  << minf << ", accu evaluations -> " << EvalCounter << std::endl; 
                }
                else{
                    std::cout << "Optimization stage-DIRECT finished: similarity -> "  << std::exp(-minf) << ", accu evaluations -> " << EvalCounter << std::endl; 
                }
            }
            catch(std::exception &e) {
                std::cout << "nlopt failed: " << e.what() << std::endl;
            }
            std::cout << "Result code: " << result << std::endl; 
            AsignBackToOptimal(l_X); 
            SimilarityMeasure = std::exp(-minf); 
        }
    }

    if(BOBYQA_Enabled){
        for(int idxBOBYQA_Round = 0; idxBOBYQA_Round < BOBYQA_Rounds; ++idxBOBYQA_Round){
            m_stepMarker = idxBOBYQA_Round + 1; 
            //Parse the corresponsing round parameters: 
            BOBYQA_Enabled = YAML_ParameterParser["InputParameters"]["OptimizerSettings"]["BOBYQA"][std::to_string(idxBOBYQA_Round)]["Enabled"].as<bool>(); 
            if(!BOBYQA_Enabled){
                continue; 
            }
            BOBYQA_OptEntryPairs.clear(); 
            BOBYQA_OptEntryPairs.push_back(
                std::pair<OptimizationEntry, bool>( 
                    OptimizationEntry::Translation, 
                    YAML_ParameterParser["InputParameters"]["OptimizerSettings"]["BOBYQA"][std::to_string(idxBOBYQA_Round)]["OptimizedEntries"]["Translation"].as<bool>()
                )
            ); 
            BOBYQA_OptEntryPairs.push_back(
                std::pair<OptimizationEntry, bool>( 
                    OptimizationEntry::Rotation, 
                    YAML_ParameterParser["InputParameters"]["OptimizerSettings"]["BOBYQA"][std::to_string(idxBOBYQA_Round)]["OptimizedEntries"]["Rotation"].as<bool>()
                )
            ); 
            BOBYQA_OptEntryPairs.push_back(
                std::pair<OptimizationEntry, bool>( 
                    OptimizationEntry::Scale, 
                    YAML_ParameterParser["InputParameters"]["OptimizerSettings"]["BOBYQA"][std::to_string(idxBOBYQA_Round)]["OptimizedEntries"]["Scale"].as<bool>()
                )
            ); 
            BOBYQA_OptEntryPairs.push_back(
                std::pair<OptimizationEntry, bool>( 
                    OptimizationEntry::Shear, 
                    YAML_ParameterParser["InputParameters"]["OptimizerSettings"]["BOBYQA"][std::to_string(idxBOBYQA_Round)]["OptimizedEntries"]["Shear"].as<bool>()
                )
            ); 
            BOBYQA_AbsTol = YAML_ParameterParser["InputParameters"]["OptimizerSettings"]["BOBYQA"][std::to_string(idxBOBYQA_Round)]["AbsTol"].as<double>(); 

            //optimazation launches: 
            CurrentHolding_OptEntries.clear(); 
            for(int idxBOBYQA_Entry = 0; idxBOBYQA_Entry < BOBYQA_OptEntryPairs.size(); ++idxBOBYQA_Entry){
                if(BOBYQA_OptEntryPairs[idxBOBYQA_Entry].second){
                    CurrentHolding_OptEntries.push_back(BOBYQA_OptEntryPairs[idxBOBYQA_Entry].first); 
                }
            }

            if(!CurrentHolding_OptEntries.empty()){
                unsigned int NumberOfParameters = (unsigned int)CurrentHolding_OptEntries.size() * 3; 
                std::vector<double> l_LowerBounds(NumberOfParameters); 
                std::vector<double> l_UpperBounds(NumberOfParameters); 
                std::vector<double> l_X(NumberOfParameters); 
                std::vector<double> l_AbsTolX(NumberOfParameters); 
                ExtractBounds(l_LowerBounds, l_UpperBounds, l_X); 

                //start DIRECT opt: 
                nlopt::opt MyOptimizer(nlopt::LN_BOBYQA, NumberOfParameters); 
                if(patchSize == 0){
                    MyOptimizer.set_max_objective(Optimizers3D::wrapper, this); 
                }
                else{
                    MyOptimizer.set_min_objective(Optimizers3D::wrapper, this); 
                }
                MyOptimizer.set_lower_bounds(l_LowerBounds); 
                MyOptimizer.set_upper_bounds(l_UpperBounds); 
                MyOptimizer.set_xtol_rel(BOBYQA_AbsTol); 

                double minf = 0;
                nlopt::result result; 
                try{
                    result = MyOptimizer.optimize(l_X, minf);
                    if(patchSize == 0){
                        std::cout << "Optimization stage-BOBYQA finished: similarity -> "  << minf << ", accu evaluations -> " << EvalCounter << std::endl; 
                    }
                    else{
                        std::cout << "Optimization stage-BOBYQA finished: similarity -> "  << std::exp(-minf) << ", accu evaluations -> " << EvalCounter << std::endl; 
                    }
                }
                catch(std::exception &e) {
                    std::cout << "nlopt failed: " << e.what() << std::endl;
                }
                std::cout << "Result code: " << result << std::endl; 
                AsignBackToOptimal(l_X); 
                SimilarityMeasure = std::exp(-minf); 
            }
        }
    }

    m_logFileId.close(); 
}

void Optimizers3D::GetOptimalTransform(float* outMatrix){
    float3 Translation, Rotation, Scale, Shear; 
    Translation.x = (float)OptimalSolution[0]; 
    Translation.y = (float)OptimalSolution[1]; 
    Translation.z = (float)OptimalSolution[2]; 

    Rotation.x = (float)OptimalSolution[3]; 
    Rotation.y = (float)OptimalSolution[4]; 
    Rotation.z = (float)OptimalSolution[5]; 

    Scale.x = (float)OptimalSolution[6]; 
    Scale.y = (float)OptimalSolution[7]; 
    Scale.z = (float)OptimalSolution[8]; 

    Shear.x = (float)OptimalSolution[9]; 
    Shear.y = (float)OptimalSolution[10]; 
    Shear.z = (float)OptimalSolution[11]; 
    //Write matrix out: under physical space, Moving To Fixed. 
    interpolation_handle->GetMovingToFixedTransform(Rotation, Translation, Scale, Shear, outMatrix);

    std::cout << "Initial parameter: " << std::endl; 
    std::cout << "        Translation: " << InitialValues[0] << ", " << InitialValues[1] << ", " << InitialValues[2] << std::endl; 
    std::cout << "        Translation_rel (opt - init): " << Translation.x - InitialValues[0] << ", " << Translation.y - InitialValues[1] << ", " << Translation.z - InitialValues[2] << std::endl; 

    std::cout << "Optimized parameters: " << std::endl; 
    std::cout << "        Translation: " << Translation.x << ", " << Translation.y << ", " << Translation.z << std::endl; 
    std::cout << "        Rotation: " << Rotation.x << ", " << Rotation.y << ", " << Rotation.z << std::endl; 
    std::cout << "        Scaling: " << Scale.x << ", " << Scale.y << ", " << Scale.z << std::endl; 
    std::cout << "        Shear: " << Shear.x << ", " << Shear.y << ", " << Shear.z << std::endl; 
}

void Optimizers3D::GetOptimalTransform(){
    float3 Translation, Rotation, Scale, Shear; 
    Translation.x = (float)OptimalSolution[0]; 
    Translation.y = (float)OptimalSolution[1]; 
    Translation.z = (float)OptimalSolution[2]; 

    Rotation.x = (float)OptimalSolution[3]; 
    Rotation.y = (float)OptimalSolution[4]; 
    Rotation.z = (float)OptimalSolution[5]; 

    Scale.x = (float)OptimalSolution[6]; 
    Scale.y = (float)OptimalSolution[7]; 
    Scale.z = (float)OptimalSolution[8]; 

    Shear.x = (float)OptimalSolution[9]; 
    Shear.y = (float)OptimalSolution[10]; 
    Shear.z = (float)OptimalSolution[11]; 
    //Write matrix out: under physical space, Moving To Fixed. 
    interpolation_handle->GetMovingToFixedTransform(Rotation, Translation, Scale, Shear, MovingToFixed_Optimal);

    std::cout << "Initial parameter: " << std::endl; 
    std::cout << "        Translation: " << InitialValues[0] << ", " << InitialValues[1] << ", " << InitialValues[2] << std::endl; 
    std::cout << "        Translation_rel (opt - init): " << Translation.x - InitialValues[0] << ", " << Translation.y - InitialValues[1] << ", " << Translation.z - InitialValues[2] << std::endl; 

    std::cout << "Optimized parameters: " << std::endl; 
    std::cout << "        Translation: " << Translation.x << ", " << Translation.y << ", " << Translation.z << std::endl; 
    std::cout << "        Rotation: " << Rotation.x << ", " << Rotation.y << ", " << Rotation.z << std::endl; 
    std::cout << "        Scaling: " << Scale.x << ", " << Scale.y << ", " << Scale.z << std::endl; 
    std::cout << "        Shear: " << Shear.x << ", " << Shear.y << ", " << Shear.z << std::endl; 
}

void Optimizers3D::GetInternalMatrix(float* OutMatrix){
    for(int i = 0; i < 16; ++i){
        OutMatrix[i] = MovingToFixed_Optimal[i]; 
    }
}

void Optimizers3D::ExportResults(){
    std::vector<std::vector<float>> resultMatrix; 
    for(int row_it = 0; row_it < 4; ++row_it){
        std::vector<float> tmpVector(4, 0.0f); 
        for(int col_it = 0; col_it < 4; ++col_it){
            tmpVector[col_it] = MovingToFixed_Optimal[col_it + 4 * row_it]; 
        }
        resultMatrix.push_back(tmpVector); 
    }

    std::string ParameterExportPath = YAML_ParameterPath; 
    ParameterExportPath.insert(YAML_ParameterPath.find('.'), "_result"); 
    std::ofstream outputFileHandle(ParameterExportPath); 

    //Write out the results: 
    {
        float3 Translation, Rotation, Scale, Shear; 
        Translation.x = (float)OptimalSolution[0]; 
        Translation.y = (float)OptimalSolution[1]; 
        Translation.z = (float)OptimalSolution[2]; 

        Rotation.x = (float)OptimalSolution[3]; 
        Rotation.y = (float)OptimalSolution[4]; 
        Rotation.z = (float)OptimalSolution[5]; 

        Scale.x = (float)OptimalSolution[6]; 
        Scale.y = (float)OptimalSolution[7]; 
        Scale.z = (float)OptimalSolution[8]; 

        Shear.x = (float)OptimalSolution[9]; 
        Shear.y = (float)OptimalSolution[10]; 
        Shear.z = (float)OptimalSolution[11]; 

        YAML_ParameterParser["InputParameters"]["OptimizerSettings"]["InitialValues"]["TX"] = Translation.x; 
        YAML_ParameterParser["InputParameters"]["OptimizerSettings"]["InitialValues"]["TY"] = Translation.y; 
        YAML_ParameterParser["InputParameters"]["OptimizerSettings"]["InitialValues"]["TZ"] = Translation.z; 
        YAML_ParameterParser["InputParameters"]["OptimizerSettings"]["InitialValues"]["RX"] = Rotation.x; 
        YAML_ParameterParser["InputParameters"]["OptimizerSettings"]["InitialValues"]["RY"] = Rotation.y; 
        YAML_ParameterParser["InputParameters"]["OptimizerSettings"]["InitialValues"]["RZ"] = Rotation.z; 
        YAML_ParameterParser["InputParameters"]["OptimizerSettings"]["InitialValues"]["ScaleX"] = Scale.x; 
        YAML_ParameterParser["InputParameters"]["OptimizerSettings"]["InitialValues"]["ScaleY"] = Scale.y; 
        YAML_ParameterParser["InputParameters"]["OptimizerSettings"]["InitialValues"]["ScaleZ"] = Scale.z; 
        YAML_ParameterParser["InputParameters"]["OptimizerSettings"]["InitialValues"]["ShearX"] = Shear.x; 
        YAML_ParameterParser["InputParameters"]["OptimizerSettings"]["InitialValues"]["ShearY"] = Shear.y; 
        YAML_ParameterParser["InputParameters"]["OptimizerSettings"]["InitialValues"]["ShearZ"] = Shear.z; 

        YAML_ParameterParser["InputParameters"]["OptimizerSettings"]["DIRECT"]["Enabled"] = false; 
        YAML_ParameterParser["InputParameters"]["OptimizerSettings"]["BOBYQA"]["0"]["Enabled"] = false; 
    }

    YAML_ParameterParser["Outputs"]["IsRegistered"] = true; 
    YAML_ParameterParser["Outputs"]["TransformMatrix"] = resultMatrix; 
    YAML_ParameterParser["Outputs"]["LastSimilarity"] = SimilarityMeasure; 
    YAML_ParameterParser["Outputs"]["NumEvaluations"] = EvalCounter;

    //time stamp: 
    currentTime = std::chrono::system_clock::to_time_t(std::chrono::system_clock::now()); 
    std::string currentTimeParsed_str(ctime(&currentTime)); 
    YAML_ParameterParser["Outputs"]["RegistrationDateTime"] = currentTimeParsed_str; 
    
    outputFileHandle << YAML_ParameterParser; 

    std::cout << "Optimal transform matrix found: " << std::endl; 
    MatrixS4X4Print(MovingToFixed_Optimal); 

    std::cout << "Last Similarity Estimate: " << SimilarityMeasure << ", with " << EvalCounter << " evaluations. " << std::endl; 
}

/* --------------------------------------- 2DTo3D Volume registration optimizer -------------------------------------- */
Optimizers2DTo3D::Optimizers2DTo3D(std::string FilePath){
    //Load parameter: 
    YAML_ParameterParser = YAML::LoadFile(FilePath); 
    YAML_ParameterPath = FilePath; 

    auto ExtractParam = [&](const std::string& _tagName, std::vector<double>& _targetVector){
        if(_targetVector.size() != 12){
            _targetVector.resize(12, 0); 
        }
        _targetVector[0] = YAML_ParameterParser["InputParameters"]["OptimizerSettings"][_tagName]["TX"].as<double>(); 
        _targetVector[1] = YAML_ParameterParser["InputParameters"]["OptimizerSettings"][_tagName]["TY"].as<double>(); 
        _targetVector[2] = YAML_ParameterParser["InputParameters"]["OptimizerSettings"][_tagName]["TZ"].as<double>(); 
        _targetVector[3] = YAML_ParameterParser["InputParameters"]["OptimizerSettings"][_tagName]["RX"].as<double>(); 
        _targetVector[4] = YAML_ParameterParser["InputParameters"]["OptimizerSettings"][_tagName]["RY"].as<double>(); 
        _targetVector[5] = YAML_ParameterParser["InputParameters"]["OptimizerSettings"][_tagName]["RZ"].as<double>(); 
        _targetVector[6] = YAML_ParameterParser["InputParameters"]["OptimizerSettings"][_tagName]["ScaleX"].as<double>(); 
        _targetVector[7] = YAML_ParameterParser["InputParameters"]["OptimizerSettings"][_tagName]["ScaleY"].as<double>(); 
        _targetVector[8] = YAML_ParameterParser["InputParameters"]["OptimizerSettings"][_tagName]["ScaleZ"].as<double>(); 
        _targetVector[9] = YAML_ParameterParser["InputParameters"]["OptimizerSettings"][_tagName]["ShearX"].as<double>(); 
        _targetVector[10] = YAML_ParameterParser["InputParameters"]["OptimizerSettings"][_tagName]["ShearY"].as<double>(); 
        _targetVector[11] = YAML_ParameterParser["InputParameters"]["OptimizerSettings"][_tagName]["ShearZ"].as<double>(); 
    }; 

    //Initial starting values: 
    InitialValues.resize(12, 0); 
    ExtractParam("InitialValues", InitialValues); 
    
    OptimalSolution = InitialValues; 

    //Bounds: 
    LowerBounds.resize(12, 0); 
    ExtractParam("LowerBounds", LowerBounds); 
    for(int i = 0; i < 6; ++i){
        LowerBounds[i] += InitialValues[i]; 
    }

    UpperBounds.resize(12, 0); 
    ExtractParam("UpperBounds", UpperBounds); 
    for(int i = 0; i < 6; ++i){
        UpperBounds[i] += InitialValues[i]; 
    }

    //DIRECT: 
    DIRECT_Enabled = YAML_ParameterParser["InputParameters"]["OptimizerSettings"]["DIRECT"]["Enabled"].as<bool>(); 
    DIRECT_OptEntryPairs.clear(); 
    DIRECT_OptEntryPairs.push_back(
        std::pair<OptimizationEntry, bool>( 
            OptimizationEntry::Translation, 
            YAML_ParameterParser["InputParameters"]["OptimizerSettings"]["DIRECT"]["OptimizedEntries"]["Translation"].as<bool>()
        )
    ); 
    DIRECT_OptEntryPairs.push_back(
        std::pair<OptimizationEntry, bool>( 
            OptimizationEntry::Rotation, 
            YAML_ParameterParser["InputParameters"]["OptimizerSettings"]["DIRECT"]["OptimizedEntries"]["Rotation"].as<bool>()
        )
    ); 
    DIRECT_OptEntryPairs.push_back(
        std::pair<OptimizationEntry, bool>( 
            OptimizationEntry::Scale, 
            YAML_ParameterParser["InputParameters"]["OptimizerSettings"]["DIRECT"]["OptimizedEntries"]["Scale"].as<bool>()
        )
    ); 
    DIRECT_OptEntryPairs.push_back(
        std::pair<OptimizationEntry, bool>( 
            OptimizationEntry::Shear, 
            YAML_ParameterParser["InputParameters"]["OptimizerSettings"]["DIRECT"]["OptimizedEntries"]["Shear"].as<bool>()
        )
    ); 

    DIRECT_MaxEvals = YAML_ParameterParser["InputParameters"]["OptimizerSettings"]["DIRECT"]["MaxEvaluations"].as<int>(); 

    //BOBYQA: 
    BOBYQA_Enabled = false; 
    BOBYQA_Rounds = (int)YAML_ParameterParser["InputParameters"]["OptimizerSettings"]["BOBYQA"].size(); 
    for(int i = 0; i < BOBYQA_Rounds; ++i){
        if(!YAML_ParameterParser["InputParameters"]["OptimizerSettings"]["BOBYQA"][std::to_string(i)]){
            std::cout << "[Error]: BOBYQA rounds are not matching and consistent. " << std::endl; 
            std::exit(EXIT_FAILURE); 
        }
        else{
            BOBYQA_Enabled = BOBYQA_Enabled || YAML_ParameterParser["InputParameters"]["OptimizerSettings"]["BOBYQA"][std::to_string(i)]["Enabled"].as<bool>(); 
        }
        
    }

    patchSize = YAML_ParameterParser["InputParameters"]["RegistrationParameters"]["PatchSize"].as<int>(); 

    SimilarityMeasure = 0.0f; 
    EvalCounter = 0; 
}

void Optimizers2DTo3D::LinkToInterpolator(Interpolator2DTo3D* external_Interpolator){
    interpolation_handle = external_Interpolator; 
}

void Optimizers2DTo3D::GlobalIterate(){
    float3 Translation, Rotation, Scale, Shear; 
    Translation.x = (float)OptimalSolution[0]; 
    Translation.y = (float)OptimalSolution[1]; 
    Translation.z = (float)OptimalSolution[2]; 

    Rotation.x = (float)OptimalSolution[3]; 
    Rotation.y = (float)OptimalSolution[4]; 
    Rotation.z = (float)OptimalSolution[5]; 

    Scale.x = (float)OptimalSolution[6]; 
    Scale.y = (float)OptimalSolution[7]; 
    Scale.z = (float)OptimalSolution[8]; 

    Shear.x = (float)OptimalSolution[9]; 
    Shear.y = (float)OptimalSolution[10]; 
    Shear.z = (float)OptimalSolution[11]; 

    m_logFileId << "Translation_x,Translation_y,Translation_z,Similarity" << std::endl; 
    float3 Translation_it; 
    for(float it_z = -5.0 + Translation.z; it_z <= 5.0 + Translation.z; it_z += 0.229762){
        for(float it_y = -5.0 + Translation.y; it_y <= 5.0 + Translation.y; it_y += 0.229762){
            for(float it_x = -5.0 + Translation.x; it_x <= 5.0 + Translation.x; it_x += 0.229762){
                Translation_it.x = it_x; 
                Translation_it.y = it_y; 
                Translation_it.z = it_z; 
                interpolation_handle->GenerateMappers(Rotation, Translation_it, Scale, Shear); 
                interpolation_handle->Interpolate(); 
                SimilarityMeasure = interpolation_handle->GetSimilarityMeasure(patchSize); 
                m_logFileId << Translation_it.x << "," << Translation_it.y << "," << Translation_it.z << "," << SimilarityMeasure << std::endl; 
                std::cout << Translation_it.x << "," << Translation_it.y << "," << Translation_it.z << "," << SimilarityMeasure << std::endl; 
            }
        }
    }
}

void Optimizers2DTo3D::ShowCurrentResults(){
    float3 Translation, Rotation, Scale, Shear; 
    Translation.x = (float)OptimalSolution[0]; 
    Translation.y = (float)OptimalSolution[1]; 
    Translation.z = (float)OptimalSolution[2]; 

    Rotation.x = (float)OptimalSolution[3]; 
    Rotation.y = (float)OptimalSolution[4]; 
    Rotation.z = (float)OptimalSolution[5]; 

    Scale.x = (float)OptimalSolution[6]; 
    Scale.y = (float)OptimalSolution[7]; 
    Scale.z = (float)OptimalSolution[8]; 

    Shear.x = (float)OptimalSolution[9]; 
    Shear.y = (float)OptimalSolution[10]; 
    Shear.z = (float)OptimalSolution[11]; 

    interpolation_handle->GenerateMappers(Rotation, Translation, Scale, Shear); 
    interpolation_handle->Interpolate(); 
    SimilarityMeasure = interpolation_handle->GetSimilarityMeasure(patchSize); 

    m_logFileId << "Translation_x,Translation_y,Translation_z,Rotation_x,Rotation_y,Rotation_z,Scale_x,Scale_y,Scale_z,Shear_x,Shear_y,Shear_z,Similarity,IteratorCounter" << std::endl; 
    m_logFileId << "Initial step" << std::endl; 
    
    //Dump results: 
    m_logFileId << 
    Translation.x << "," << Translation.y << "," << Translation.z << "," << 
    Rotation.x << "," << Rotation.y << "," << Rotation.z << "," << 
    Scale.x << "," << Scale.y << "," << Scale.z << "," << 
    Shear.x << "," << Shear.y << "," << Shear.z << "," << 
    SimilarityMeasure << "," << EvalCounter << std::endl; 
}

double Optimizers2DTo3D::operator()(const std::vector<double> &x){
    float3 Translation, Rotation, Scale, Shear; 
    Translation.x = (float)OptimalSolution[0]; 
    Translation.y = (float)OptimalSolution[1]; 
    Translation.z = (float)OptimalSolution[2]; 

    Rotation.x = (float)OptimalSolution[3]; 
    Rotation.y = (float)OptimalSolution[4]; 
    Rotation.z = (float)OptimalSolution[5]; 

    Scale.x = (float)OptimalSolution[6]; 
    Scale.y = (float)OptimalSolution[7]; 
    Scale.z = (float)OptimalSolution[8]; 

    Shear.x = (float)OptimalSolution[9]; 
    Shear.y = (float)OptimalSolution[10]; 
    Shear.z = (float)OptimalSolution[11]; 

    if(x.size() != (CurrentHolding_OptEntries.size() * 3)){
        std::cout << "[Error]: opt parameters do not match with entries! " << std::endl; 
        return 0; 
    }
    else{
        for(int i = 0; i < CurrentHolding_OptEntries.size(); ++i){
            switch (CurrentHolding_OptEntries[i])
            {
            case OptimizationEntry::Translation:
                Translation.x = (float)x[0 + i * 3]; 
                Translation.y = (float)x[1 + i * 3]; 
                Translation.z = (float)x[2 + i * 3]; 
                break;
            case OptimizationEntry::Rotation:
                Rotation.x = (float)x[0 + i * 3]; 
                Rotation.y = (float)x[1 + i * 3]; 
                Rotation.z = (float)x[2 + i * 3]; 
                break;
            case OptimizationEntry::Scale:
                Scale.x = (float)x[0 + i * 3]; 
                Scale.y = (float)x[1 + i * 3]; 
                Scale.z = (float)x[2 + i * 3]; 
                break;
            case OptimizationEntry::Shear:
                Shear.x = (float)x[0 + i * 3]; 
                Shear.y = (float)x[1 + i * 3]; 
                Shear.z = (float)x[2 + i * 3]; 
                break;
            default:
                break;
            }
        }
    }

    //Evaluate similarityMeasure: 
    interpolation_handle->GenerateMappers(Rotation, Translation, Scale, Shear); 
    interpolation_handle->Interpolate(); 
    SimilarityMeasure = interpolation_handle->GetSimilarityMeasure(patchSize); 

    ++EvalCounter; 

    //Dump results: 
    m_logFileId << 
    Translation.x << "," << Translation.y << "," << Translation.z << "," << 
    Rotation.x << "," << Rotation.y << "," << Rotation.z << "," << 
    Scale.x << "," << Scale.y << "," << Scale.z << "," << 
    Shear.x << "," << Shear.y << "," << Shear.z << "," << 
    SimilarityMeasure << "," << EvalCounter << std::endl; 

    return -std::log(SimilarityMeasure); 
}

double Optimizers2DTo3D::wrapper(const std::vector<double> &x, std::vector<double> &grad, void *data){
    return reinterpret_cast<Optimizers2DTo3D*>(data)->operator()(x); 
}

void Optimizers2DTo3D::Optimize(){
    //Initialize log: 
    m_logFileId.open("RegistrationInfo.log", std::ofstream::out); 

    // this->GlobalIterate(); 
    // m_logFileId.close(); 
    // return; 

    this->ShowCurrentResults(); 

    //functors: 
    auto ExtractBounds = [&](std::vector<double>& _l, std::vector<double>& _u, std::vector<double>& _x){
        for(int idxEntry = 0; idxEntry < CurrentHolding_OptEntries.size(); ++idxEntry){
            switch (CurrentHolding_OptEntries[idxEntry])
            {
            case OptimizationEntry::Translation:
                _l[0 + idxEntry * 3] = LowerBounds[0]; 
                _l[1 + idxEntry * 3] = LowerBounds[1]; 
                _l[2 + idxEntry * 3] = LowerBounds[2]; 

                _u[0 + idxEntry * 3] = UpperBounds[0]; 
                _u[1 + idxEntry * 3] = UpperBounds[1]; 
                _u[2 + idxEntry * 3] = UpperBounds[2]; 

                _x[0 + idxEntry * 3] = OptimalSolution[0]; 
                _x[1 + idxEntry * 3] = OptimalSolution[1]; 
                _x[2 + idxEntry * 3] = OptimalSolution[2]; 
                break;
            case OptimizationEntry::Rotation:
                _l[0 + idxEntry * 3] = LowerBounds[3]; 
                _l[1 + idxEntry * 3] = LowerBounds[4]; 
                _l[2 + idxEntry * 3] = LowerBounds[5]; 

                _u[0 + idxEntry * 3] = UpperBounds[3]; 
                _u[1 + idxEntry * 3] = UpperBounds[4]; 
                _u[2 + idxEntry * 3] = UpperBounds[5]; 

                _x[0 + idxEntry * 3] = OptimalSolution[3]; 
                _x[1 + idxEntry * 3] = OptimalSolution[4]; 
                _x[2 + idxEntry * 3] = OptimalSolution[5]; 
                break;
            case OptimizationEntry::Scale:
                _l[0 + idxEntry * 3] = LowerBounds[6]; 
                _l[1 + idxEntry * 3] = LowerBounds[7]; 
                _l[2 + idxEntry * 3] = LowerBounds[8]; 

                _u[0 + idxEntry * 3] = UpperBounds[6]; 
                _u[1 + idxEntry * 3] = UpperBounds[7]; 
                _u[2 + idxEntry * 3] = UpperBounds[8]; 

                _x[0 + idxEntry * 3] = OptimalSolution[6]; 
                _x[1 + idxEntry * 3] = OptimalSolution[7]; 
                _x[2 + idxEntry * 3] = OptimalSolution[8]; 
                break;
            case OptimizationEntry::Shear:
                _l[0 + idxEntry * 3] = LowerBounds[9]; 
                _l[1 + idxEntry * 3] = LowerBounds[10]; 
                _l[2 + idxEntry * 3] = LowerBounds[11]; 

                _u[0 + idxEntry * 3] = UpperBounds[9]; 
                _u[1 + idxEntry * 3] = UpperBounds[10]; 
                _u[2 + idxEntry * 3] = UpperBounds[11]; 

                _x[0 + idxEntry * 3] = OptimalSolution[9]; 
                _x[1 + idxEntry * 3] = OptimalSolution[10]; 
                _x[2 + idxEntry * 3] = OptimalSolution[11]; 
                break;
            default:
                break;
            }
        }
    }; 
    auto AsignBackToOptimal = [&](std::vector<double>& _x){
        for(int idxEntry = 0; idxEntry < CurrentHolding_OptEntries.size(); ++idxEntry){
            switch (CurrentHolding_OptEntries[idxEntry])
            {
            case OptimizationEntry::Translation:
                OptimalSolution[0] = _x[0 + idxEntry * 3]; 
                OptimalSolution[1] = _x[1 + idxEntry * 3]; 
                OptimalSolution[2] = _x[2 + idxEntry * 3]; 
                break;
            case OptimizationEntry::Rotation:
                OptimalSolution[3] = _x[0 + idxEntry * 3]; 
                OptimalSolution[4] = _x[1 + idxEntry * 3]; 
                OptimalSolution[5] = _x[2 + idxEntry * 3]; 
                break;
            case OptimizationEntry::Scale:
                OptimalSolution[6] = _x[0 + idxEntry * 3]; 
                OptimalSolution[7] = _x[1 + idxEntry * 3]; 
                OptimalSolution[8] = _x[2 + idxEntry * 3]; 
                break;
            case OptimizationEntry::Shear:
                OptimalSolution[9] = _x[0 + idxEntry * 3]; 
                OptimalSolution[10] = _x[1 + idxEntry * 3]; 
                OptimalSolution[11] = _x[2 + idxEntry * 3]; 
                break;
            default:
                break;
            }
        }
    }; 
    
    //DIRECT: 
    m_logFileId << "DIRECT step" << std::endl; 
    if(DIRECT_Enabled){
        CurrentHolding_OptEntries.clear(); 
        for(int idxDIRECT_Entry = 0; idxDIRECT_Entry < DIRECT_OptEntryPairs.size(); ++idxDIRECT_Entry){
            if(DIRECT_OptEntryPairs[idxDIRECT_Entry].second){
                CurrentHolding_OptEntries.push_back(DIRECT_OptEntryPairs[idxDIRECT_Entry].first); 
            }
        }
        
        if(!CurrentHolding_OptEntries.empty()){
            unsigned int NumberOfParameters = (unsigned int)CurrentHolding_OptEntries.size() * 3; 
            std::vector<double> l_LowerBounds(NumberOfParameters); 
            std::vector<double> l_UpperBounds(NumberOfParameters); 
            std::vector<double> l_X(NumberOfParameters); 
            ExtractBounds(l_LowerBounds, l_UpperBounds, l_X); 

            std::cout << "L: " << l_LowerBounds << std::endl; 
            std::cout << "U: " << l_UpperBounds << std::endl; 
            std::cout << "x: " << l_X << std::endl; 

            //start DIRECT opt: 
            nlopt::opt MyOptimizer(nlopt::GN_ORIG_DIRECT_L, NumberOfParameters); 
            MyOptimizer.set_min_objective(Optimizers2DTo3D::wrapper, this); 
            MyOptimizer.set_lower_bounds(l_LowerBounds); 
            MyOptimizer.set_upper_bounds(l_UpperBounds); 
            MyOptimizer.set_maxeval(DIRECT_MaxEvals); 

            double minf = 0;
            nlopt::result result; 
            try{
                result = MyOptimizer.optimize(l_X, minf);
                std::cout << "Optimization stage-DIRECT finished: similarity -> "  << std::exp(-minf) << ", accu evaluations -> " << EvalCounter << std::endl; 
            }
            catch(std::exception &e) {
                std::cout << "nlopt failed: " << e.what() << std::endl;
            }
            AsignBackToOptimal(l_X); 
            
        }
    }

    if(BOBYQA_Enabled){
        for(int idxBOBYQA_Round = 0; idxBOBYQA_Round < BOBYQA_Rounds; ++idxBOBYQA_Round){
            m_logFileId << "BOBYQA step - " << idxBOBYQA_Round << std::endl; 
            //Parse the corresponsing round parameters: 
            BOBYQA_Enabled = YAML_ParameterParser["InputParameters"]["OptimizerSettings"]["BOBYQA"][std::to_string(idxBOBYQA_Round)]["Enabled"].as<bool>(); 
            if(!BOBYQA_Enabled){
                continue; 
            }
            BOBYQA_OptEntryPairs.clear(); 
            BOBYQA_OptEntryPairs.push_back(
                std::pair<OptimizationEntry, bool>( 
                    OptimizationEntry::Translation, 
                    YAML_ParameterParser["InputParameters"]["OptimizerSettings"]["BOBYQA"][std::to_string(idxBOBYQA_Round)]["OptimizedEntries"]["Translation"].as<bool>()
                )
            ); 
            BOBYQA_OptEntryPairs.push_back(
                std::pair<OptimizationEntry, bool>( 
                    OptimizationEntry::Rotation, 
                    YAML_ParameterParser["InputParameters"]["OptimizerSettings"]["BOBYQA"][std::to_string(idxBOBYQA_Round)]["OptimizedEntries"]["Rotation"].as<bool>()
                )
            ); 
            BOBYQA_OptEntryPairs.push_back(
                std::pair<OptimizationEntry, bool>( 
                    OptimizationEntry::Scale, 
                    YAML_ParameterParser["InputParameters"]["OptimizerSettings"]["BOBYQA"][std::to_string(idxBOBYQA_Round)]["OptimizedEntries"]["Scale"].as<bool>()
                )
            ); 
            BOBYQA_OptEntryPairs.push_back(
                std::pair<OptimizationEntry, bool>( 
                    OptimizationEntry::Shear, 
                    YAML_ParameterParser["InputParameters"]["OptimizerSettings"]["BOBYQA"][std::to_string(idxBOBYQA_Round)]["OptimizedEntries"]["Shear"].as<bool>()
                )
            ); 
            BOBYQA_AbsTol = YAML_ParameterParser["InputParameters"]["OptimizerSettings"]["BOBYQA"][std::to_string(idxBOBYQA_Round)]["AbsTol"].as<double>(); 

            //optimazation launches: 
            CurrentHolding_OptEntries.clear(); 
            for(int idxBOBYQA_Entry = 0; idxBOBYQA_Entry < BOBYQA_OptEntryPairs.size(); ++idxBOBYQA_Entry){
                if(BOBYQA_OptEntryPairs[idxBOBYQA_Entry].second){
                    CurrentHolding_OptEntries.push_back(BOBYQA_OptEntryPairs[idxBOBYQA_Entry].first); 
                }
            }

            if(!CurrentHolding_OptEntries.empty()){
                unsigned int NumberOfParameters = (unsigned int)CurrentHolding_OptEntries.size() * 3; 
                std::vector<double> l_LowerBounds(NumberOfParameters); 
                std::vector<double> l_UpperBounds(NumberOfParameters); 
                std::vector<double> l_X(NumberOfParameters); 
                ExtractBounds(l_LowerBounds, l_UpperBounds, l_X); 

                std::cout << "L: " << l_LowerBounds << std::endl; 
                std::cout << "U: " << l_UpperBounds << std::endl; 
                std::cout << "x: " << l_X << std::endl; 

                //start DIRECT opt: 
                nlopt::opt MyOptimizer(nlopt::LN_BOBYQA, NumberOfParameters); 
                MyOptimizer.set_min_objective(Optimizers2DTo3D::wrapper, this); 
                MyOptimizer.set_lower_bounds(l_LowerBounds); 
                MyOptimizer.set_upper_bounds(l_UpperBounds); 
                // if(idxBOBYQA_Round == 0){
                    MyOptimizer.set_xtol_rel(BOBYQA_AbsTol); 
                // }
                // else{
                //     MyOptimizer.set_ftol_rel(BOBYQA_AbsTol); 
                // }

                double minf = 0;
                nlopt::result result; 
                try{
                    result = MyOptimizer.optimize(l_X, minf);
                    std::cout << "Optimization stage-BOBYQA finished: similarity -> "  << std::exp(-minf) << ", accu evaluations -> " << EvalCounter << std::endl; 
                }
                catch(std::exception &e) {
                    std::cout << "nlopt failed: " << e.what() << std::endl;
                }
                AsignBackToOptimal(l_X); 
            }
        }
    }

    m_logFileId.close(); 
}

void Optimizers2DTo3D::GetOptimalTransform(float* outMatrix){
    float3 Translation, Rotation, Scale, Shear; 
    Translation.x = (float)OptimalSolution[0]; 
    Translation.y = (float)OptimalSolution[1]; 
    Translation.z = (float)OptimalSolution[2]; 

    Rotation.x = (float)OptimalSolution[3]; 
    Rotation.y = (float)OptimalSolution[4]; 
    Rotation.z = (float)OptimalSolution[5]; 

    Scale.x = (float)OptimalSolution[6]; 
    Scale.y = (float)OptimalSolution[7]; 
    Scale.z = (float)OptimalSolution[8]; 

    Shear.x = (float)OptimalSolution[9]; 
    Shear.y = (float)OptimalSolution[10]; 
    Shear.z = (float)OptimalSolution[11]; 
    //Write matrix out: under physical space, Moving To Fixed. 
    interpolation_handle->GetMovingToFixedTransform(Rotation, Translation, Scale, Shear, outMatrix);

    std::cout << "Optimized parameters: " << std::endl; 
    std::cout << "        Translation: " << Translation.x << ", " << Translation.y << ", " << Translation.z << std::endl; 
    std::cout << "        Rotation: " << Rotation.x << ", " << Rotation.y << ", " << Rotation.z << std::endl; 
    std::cout << "        Scaling: " << Scale.x << ", " << Scale.y << ", " << Scale.z << std::endl; 
    std::cout << "        Shear: " << Shear.x << ", " << Shear.y << ", " << Shear.z << std::endl; 
}

void Optimizers2DTo3D::GetOptimalTransform(){
    float3 Translation, Rotation, Scale, Shear; 
    Translation.x = (float)OptimalSolution[0]; 
    Translation.y = (float)OptimalSolution[1]; 
    Translation.z = (float)OptimalSolution[2]; 

    Rotation.x = (float)OptimalSolution[3]; 
    Rotation.y = (float)OptimalSolution[4]; 
    Rotation.z = (float)OptimalSolution[5]; 

    Scale.x = (float)OptimalSolution[6]; 
    Scale.y = (float)OptimalSolution[7]; 
    Scale.z = (float)OptimalSolution[8]; 

    Shear.x = (float)OptimalSolution[9]; 
    Shear.y = (float)OptimalSolution[10]; 
    Shear.z = (float)OptimalSolution[11]; 
    //Write matrix out: under physical space, Moving To Fixed. 
    interpolation_handle->GetMovingToFixedTransform(Rotation, Translation, Scale, Shear, MovingToFixed_Optimal);

    std::cout << "Optimized parameters: " << std::endl; 
    std::cout << "        Translation: " << Translation.x << ", " << Translation.y << ", " << Translation.z << std::endl; 
    std::cout << "        Rotation: " << Rotation.x << ", " << Rotation.y << ", " << Rotation.z << std::endl; 
    std::cout << "        Scaling: " << Scale.x << ", " << Scale.y << ", " << Scale.z << std::endl; 
    std::cout << "        Shear: " << Shear.x << ", " << Shear.y << ", " << Shear.z << std::endl; 
}

void Optimizers2DTo3D::GetInternalMatrix(float* OutMatrix){
    for(int i = 0; i < 16; ++i){
        OutMatrix[i] = MovingToFixed_Optimal[i]; 
    }
}

void Optimizers2DTo3D::ExportResults(){
    std::vector<std::vector<float>> resultMatrix; 
    for(int row_it = 0; row_it < 4; ++row_it){
        std::vector<float> tmpVector(4, 0.0f); 
        for(int col_it = 0; col_it < 4; ++col_it){
            tmpVector[col_it] = MovingToFixed_Optimal[col_it + 4 * row_it]; 
        }
        resultMatrix.push_back(tmpVector); 
    }

    std::string ParameterExportPath = YAML_ParameterPath; 
    ParameterExportPath.insert(YAML_ParameterPath.find('.'), "_result"); 
    std::ofstream outputFileHandle(ParameterExportPath); 

    //Write out the results: 
    {
        float3 Translation, Rotation, Scale, Shear; 
        Translation.x = (float)OptimalSolution[0]; 
        Translation.y = (float)OptimalSolution[1]; 
        Translation.z = (float)OptimalSolution[2]; 

        Rotation.x = (float)OptimalSolution[3]; 
        Rotation.y = (float)OptimalSolution[4]; 
        Rotation.z = (float)OptimalSolution[5]; 

        Scale.x = (float)OptimalSolution[6]; 
        Scale.y = (float)OptimalSolution[7]; 
        Scale.z = (float)OptimalSolution[8]; 

        Shear.x = (float)OptimalSolution[9]; 
        Shear.y = (float)OptimalSolution[10]; 
        Shear.z = (float)OptimalSolution[11]; 

        YAML_ParameterParser["InputParameters"]["OptimizerSettings"]["InitialValues"]["TX"] = Translation.x; 
        YAML_ParameterParser["InputParameters"]["OptimizerSettings"]["InitialValues"]["TY"] = Translation.y; 
        YAML_ParameterParser["InputParameters"]["OptimizerSettings"]["InitialValues"]["TZ"] = Translation.z; 
        YAML_ParameterParser["InputParameters"]["OptimizerSettings"]["InitialValues"]["RX"] = Rotation.x; 
        YAML_ParameterParser["InputParameters"]["OptimizerSettings"]["InitialValues"]["RY"] = Rotation.y; 
        YAML_ParameterParser["InputParameters"]["OptimizerSettings"]["InitialValues"]["RZ"] = Rotation.z; 
        YAML_ParameterParser["InputParameters"]["OptimizerSettings"]["InitialValues"]["ScaleX"] = Scale.x; 
        YAML_ParameterParser["InputParameters"]["OptimizerSettings"]["InitialValues"]["ScaleY"] = Scale.y; 
        YAML_ParameterParser["InputParameters"]["OptimizerSettings"]["InitialValues"]["ScaleZ"] = Scale.z; 
        YAML_ParameterParser["InputParameters"]["OptimizerSettings"]["InitialValues"]["ShearX"] = Shear.x; 
        YAML_ParameterParser["InputParameters"]["OptimizerSettings"]["InitialValues"]["ShearY"] = Shear.y; 
        YAML_ParameterParser["InputParameters"]["OptimizerSettings"]["InitialValues"]["ShearZ"] = Shear.z; 

        YAML_ParameterParser["InputParameters"]["OptimizerSettings"]["DIRECT"]["Enabled"] = false; 
        YAML_ParameterParser["InputParameters"]["OptimizerSettings"]["BOBYQA"]["0"]["Enabled"] = false; 
    }

    YAML_ParameterParser["Outputs"]["IsRegistered"] = true; 
    YAML_ParameterParser["Outputs"]["TransformMatrix"] = resultMatrix; 
    YAML_ParameterParser["Outputs"]["LastSimilarity"] = SimilarityMeasure; 
    YAML_ParameterParser["Outputs"]["NumEvaluations"] = EvalCounter;

    //time stamp: 
    currentTime = std::chrono::system_clock::to_time_t(std::chrono::system_clock::now()); 
    std::string currentTimeParsed_str(ctime(&currentTime)); 
    YAML_ParameterParser["Outputs"]["RegistrationDateTime"] = currentTimeParsed_str; 
    
    outputFileHandle << YAML_ParameterParser; 

    std::cout << "Optimal transform matrix found: " << std::endl; 
    MatrixS4X4Print(MovingToFixed_Optimal); 

    std::cout << "Last Similarity Estimate: " << SimilarityMeasure << ", with " << EvalCounter << " evaluations. " << std::endl; 
}