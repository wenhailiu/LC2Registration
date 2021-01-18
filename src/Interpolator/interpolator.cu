#include <iostream>
#include <vector>
#include <fstream>
#include <thread>
#include <future>
#include <algorithm>

#include <cuda_runtime.h>
#include "device_launch_parameters.h"
#include "nppi_geometry_transforms.h"
#include "thrust/extrema.h"

#include "src/Interpolator/interpolator.cuh"
#include "src/Utilities/my_cuda_helper.cuh"
#include "src/cu_LC2/cu_lc2_2d.cuh"
#include "src/cu_LC2/cu_lc2_3d.cuh"

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

#include "volumeImporter.h"

namespace YAML {
template<>
struct convert<float3> {
    static Node encode(const float3& rhs) {
        Node node;
        node.push_back(rhs.x);
        node.push_back(rhs.y);
        node.push_back(rhs.z);
        return node;
    }

    static bool decode(const Node& node, float3& rhs) {
        if(!node.IsSequence() || node.size() != 3) {
            return false;
        }

        rhs.x = node[0].as<float>();
        rhs.y = node[1].as<float>();
        rhs.z = node[2].as<float>();
        return true;
    }
};

template<>
struct convert<int3> {
    static Node encode(const int3& rhs) {
        Node node;
        node.push_back(rhs.x);
        node.push_back(rhs.y);
        node.push_back(rhs.z);
        return node;
    }

    static bool decode(const Node& node, int3& rhs) {
        if(!node.IsSequence() || node.size() != 3) {
            return false;
        }

        rhs.x = node[0].as<int>();
        rhs.y = node[1].as<int>();
        rhs.z = node[2].as<int>();
        return true;
    }
};
}

__global__ void Interpolate_3Dkernel(float *VirtualSpace_Fixed, float *VirtualSpace_Moving, const float *FixedSpace, const float *MovingSpace){
    
    //Iterate all elements within virtual space: 
    int col_idx = threadIdx.x + blockIdx.x * blockDim.x;
    int row_idx = threadIdx.y + blockIdx.y * blockDim.y;
    int lay_idx = threadIdx.z + blockIdx.z * blockDim.z;

    if(col_idx < Virtual3DMapper_d.GetDimension().x && row_idx < Virtual3DMapper_d.GetDimension().y && lay_idx < Virtual3DMapper_d.GetDimension().z){
        float3 location_Virtual;
        location_Virtual = Virtual3DMapper_d.ToLoc((float)col_idx, (float)row_idx, (float)lay_idx);
        //Mapping physical virtual point to Fixed, and Moving spaces: 
        float3 location_Fixed, location_Moving;
        location_Fixed = VirtualToFixed3D_d * location_Virtual;
        location_Moving = VirtualToMoving3D_d * location_Virtual;

        float3 voxel_Fixed, voxel_Moving;
        voxel_Fixed = Fixed3DMapper_d.ToPxl(location_Fixed);
        voxel_Moving = Moving3DMapper_d.ToPxl(location_Moving);
        
        //Interpolation: to be re-used
        float interpRatio_x = 0.0f, interpRatio_y = 0.0f, interpRatio_z = 0.0f; 

        //Filling Fixed space: 
        float InterpolatedFixedIntensity = 0.0f; 
        if( voxel_Fixed.x >= 0 && voxel_Fixed.x <= (Fixed3DMapper_d.GetDimension().x - 1) && 
            voxel_Fixed.y >= 0 && voxel_Fixed.y <= (Fixed3DMapper_d.GetDimension().y - 1) && 
            voxel_Fixed.z >= 0 && voxel_Fixed.z <= (Fixed3DMapper_d.GetDimension().z - 1)){
            //Perform interpolation on Fixed space: 
            {
                interpRatio_x = voxel_Fixed.x - (int)floor(voxel_Fixed.x);
                interpRatio_y = voxel_Fixed.y - (int)floor(voxel_Fixed.y);
                interpRatio_z = voxel_Fixed.z - (int)floor(voxel_Fixed.z);

                //Step 1: interp. along z direction: 
                float interp_z_LU, interp_z_RU, interp_z_RB, interp_z_LB;

                interp_z_LU = FixedSpace[(int)floor(voxel_Fixed.x) + (int)floor(voxel_Fixed.y) * Fixed3DMapper_d.GetDimension().x + (int)floor(voxel_Fixed.z) * Fixed3DMapper_d.GetDimension().x * Fixed3DMapper_d.GetDimension().y] + 
                    interpRatio_z * (FixedSpace[(int)floor(voxel_Fixed.x) + (int)floor(voxel_Fixed.y) * Fixed3DMapper_d.GetDimension().x + (int)ceil(voxel_Fixed.z) * Fixed3DMapper_d.GetDimension().x * Fixed3DMapper_d.GetDimension().y] - 
                    FixedSpace[(int)floor(voxel_Fixed.x) + (int)floor(voxel_Fixed.y) * Fixed3DMapper_d.GetDimension().x + (int)floor(voxel_Fixed.z) * Fixed3DMapper_d.GetDimension().x * Fixed3DMapper_d.GetDimension().y]);

                interp_z_RU = FixedSpace[(int)ceil(voxel_Fixed.x) + (int)floor(voxel_Fixed.y) * Fixed3DMapper_d.GetDimension().x + (int)floor(voxel_Fixed.z) * Fixed3DMapper_d.GetDimension().x * Fixed3DMapper_d.GetDimension().y] + 
                    interpRatio_z * (FixedSpace[(int)ceil(voxel_Fixed.x) + (int)floor(voxel_Fixed.y) * Fixed3DMapper_d.GetDimension().x + (int)ceil(voxel_Fixed.z) * Fixed3DMapper_d.GetDimension().x * Fixed3DMapper_d.GetDimension().y] - 
                    FixedSpace[(int)ceil(voxel_Fixed.x) + (int)floor(voxel_Fixed.y) * Fixed3DMapper_d.GetDimension().x + (int)floor(voxel_Fixed.z) * Fixed3DMapper_d.GetDimension().x * Fixed3DMapper_d.GetDimension().y]);

                interp_z_RB = FixedSpace[(int)ceil(voxel_Fixed.x) + (int)ceil(voxel_Fixed.y) * Fixed3DMapper_d.GetDimension().x + (int)floor(voxel_Fixed.z) * Fixed3DMapper_d.GetDimension().x * Fixed3DMapper_d.GetDimension().y] + 
                    interpRatio_z * (FixedSpace[(int)ceil(voxel_Fixed.x) + (int)ceil(voxel_Fixed.y) * Fixed3DMapper_d.GetDimension().x + (int)ceil(voxel_Fixed.z) * Fixed3DMapper_d.GetDimension().x * Fixed3DMapper_d.GetDimension().y] - 
                    FixedSpace[(int)ceil(voxel_Fixed.x) + (int)ceil(voxel_Fixed.y) * Fixed3DMapper_d.GetDimension().x + (int)floor(voxel_Fixed.z) * Fixed3DMapper_d.GetDimension().x * Fixed3DMapper_d.GetDimension().y]);

                interp_z_LB = FixedSpace[(int)floor(voxel_Fixed.x) + (int)ceil(voxel_Fixed.y) * Fixed3DMapper_d.GetDimension().x + (int)floor(voxel_Fixed.z) * Fixed3DMapper_d.GetDimension().x * Fixed3DMapper_d.GetDimension().y] + 
                    interpRatio_z * (FixedSpace[(int)floor(voxel_Fixed.x) + (int)ceil(voxel_Fixed.y) * Fixed3DMapper_d.GetDimension().x + (int)ceil(voxel_Fixed.z) * Fixed3DMapper_d.GetDimension().x * Fixed3DMapper_d.GetDimension().y] - 
                    FixedSpace[(int)floor(voxel_Fixed.x) + (int)ceil(voxel_Fixed.y) * Fixed3DMapper_d.GetDimension().x + (int)floor(voxel_Fixed.z) * Fixed3DMapper_d.GetDimension().x * Fixed3DMapper_d.GetDimension().y]);

                //Step 2: interp. along y direction: 
                float interp_y_L, interp_y_R;
                interp_y_L = interp_z_LU + interpRatio_y * (interp_z_LB - interp_z_LU);
                interp_y_R = interp_z_RU + interpRatio_y * (interp_z_RB - interp_z_RU);

                //Step 3: interp. along x direction: 
                InterpolatedFixedIntensity = interp_y_L + interpRatio_x * (interp_y_R - interp_y_L);
            }
            VirtualSpace_Fixed[col_idx + row_idx * Virtual3DMapper_d.GetDimension().x + lay_idx * Virtual3DMapper_d.GetDimension().x * Virtual3DMapper_d.GetDimension().y] = InterpolatedFixedIntensity;
        }
        else{
            VirtualSpace_Fixed[col_idx + row_idx * Virtual3DMapper_d.GetDimension().x + lay_idx * Virtual3DMapper_d.GetDimension().x * Virtual3DMapper_d.GetDimension().y] = 0.0f;
        }

        //Filling Moving space: 
        float InterpolatedMovingIntensity = 0.0f; 
        if( voxel_Moving.x >= 0 && voxel_Moving.x <= (Moving3DMapper_d.GetDimension().x - 1) && 
            voxel_Moving.y >= 0 && voxel_Moving.y <= (Moving3DMapper_d.GetDimension().y - 1) && 
            voxel_Moving.z >= 0 && voxel_Moving.z <= (Moving3DMapper_d.GetDimension().z - 1)){
            //Perform interpolation on Fixed space: 
            {
                interpRatio_x = voxel_Moving.x - (int)floor(voxel_Moving.x);
                interpRatio_y = voxel_Moving.y - (int)floor(voxel_Moving.y);
                interpRatio_z = voxel_Moving.z - (int)floor(voxel_Moving.z);

                //Step 1: interp. along z direction: 
                float interp_z_LU, interp_z_RU, interp_z_RB, interp_z_LB;

                interp_z_LU = MovingSpace[(int)floor(voxel_Moving.x) + (int)floor(voxel_Moving.y) * Moving3DMapper_d.GetDimension().x + (int)floor(voxel_Moving.z) * Moving3DMapper_d.GetDimension().x * Moving3DMapper_d.GetDimension().y] + 
                    interpRatio_z * (MovingSpace[(int)floor(voxel_Moving.x) + (int)floor(voxel_Moving.y) * Moving3DMapper_d.GetDimension().x + (int)ceil(voxel_Moving.z) * Moving3DMapper_d.GetDimension().x * Moving3DMapper_d.GetDimension().y] - 
                    MovingSpace[(int)floor(voxel_Moving.x) + (int)floor(voxel_Moving.y) * Moving3DMapper_d.GetDimension().x + (int)floor(voxel_Moving.z) * Moving3DMapper_d.GetDimension().x * Moving3DMapper_d.GetDimension().y]);

                interp_z_RU = MovingSpace[(int)ceil(voxel_Moving.x) + (int)floor(voxel_Moving.y) * Moving3DMapper_d.GetDimension().x + (int)floor(voxel_Moving.z) * Moving3DMapper_d.GetDimension().x * Moving3DMapper_d.GetDimension().y] + 
                    interpRatio_z * (MovingSpace[(int)ceil(voxel_Moving.x) + (int)floor(voxel_Moving.y) * Moving3DMapper_d.GetDimension().x + (int)ceil(voxel_Moving.z) * Moving3DMapper_d.GetDimension().x * Moving3DMapper_d.GetDimension().y] - 
                    MovingSpace[(int)ceil(voxel_Moving.x) + (int)floor(voxel_Moving.y) * Moving3DMapper_d.GetDimension().x + (int)floor(voxel_Moving.z) * Moving3DMapper_d.GetDimension().x * Moving3DMapper_d.GetDimension().y]);

                interp_z_RB = MovingSpace[(int)ceil(voxel_Moving.x) + (int)ceil(voxel_Moving.y) * Moving3DMapper_d.GetDimension().x + (int)floor(voxel_Moving.z) * Moving3DMapper_d.GetDimension().x * Moving3DMapper_d.GetDimension().y] + 
                    interpRatio_z * (MovingSpace[(int)ceil(voxel_Moving.x) + (int)ceil(voxel_Moving.y) * Moving3DMapper_d.GetDimension().x + (int)ceil(voxel_Moving.z) * Moving3DMapper_d.GetDimension().x * Moving3DMapper_d.GetDimension().y] - 
                    MovingSpace[(int)ceil(voxel_Moving.x) + (int)ceil(voxel_Moving.y) * Moving3DMapper_d.GetDimension().x + (int)floor(voxel_Moving.z) * Moving3DMapper_d.GetDimension().x * Moving3DMapper_d.GetDimension().y]);

                interp_z_LB = MovingSpace[(int)floor(voxel_Moving.x) + (int)ceil(voxel_Moving.y) * Moving3DMapper_d.GetDimension().x + (int)floor(voxel_Moving.z) * Moving3DMapper_d.GetDimension().x * Moving3DMapper_d.GetDimension().y] + 
                    interpRatio_z * (MovingSpace[(int)floor(voxel_Moving.x) + (int)ceil(voxel_Moving.y) * Moving3DMapper_d.GetDimension().x + (int)ceil(voxel_Moving.z) * Moving3DMapper_d.GetDimension().x * Moving3DMapper_d.GetDimension().y] - 
                    MovingSpace[(int)floor(voxel_Moving.x) + (int)ceil(voxel_Moving.y) * Moving3DMapper_d.GetDimension().x + (int)floor(voxel_Moving.z) * Moving3DMapper_d.GetDimension().x * Moving3DMapper_d.GetDimension().y]);

                //Step 2: interp. along y direction: 
                float interp_y_L, interp_y_R;
                interp_y_L = interp_z_LU + interpRatio_y * (interp_z_LB - interp_z_LU);
                interp_y_R = interp_z_RU + interpRatio_y * (interp_z_RB - interp_z_RU);

                //Step 3: interp. along x direction: 
                InterpolatedMovingIntensity = interp_y_L + interpRatio_x * (interp_y_R - interp_y_L);
            }
            VirtualSpace_Moving[col_idx + row_idx * Virtual3DMapper_d.GetDimension().x + lay_idx * Virtual3DMapper_d.GetDimension().x * Virtual3DMapper_d.GetDimension().y] = InterpolatedMovingIntensity;
        }
        else{
            VirtualSpace_Moving[col_idx + row_idx * Virtual3DMapper_d.GetDimension().x + lay_idx * Virtual3DMapper_d.GetDimension().x * Virtual3DMapper_d.GetDimension().y] = 0.0f;
        }
    }
}

__global__ void Interpolate_2Dkernel(float *VirtualSpace_Fixed, float *VirtualSpace_Moving, const float *FixedSpace, const float *MovingSpace){
    //Iterate all elements within virtual space: 
    int col_idx = threadIdx.x + blockIdx.x * blockDim.x;
    int row_idx = threadIdx.y + blockIdx.y * blockDim.y;

    if(col_idx < Virtual2DMapper_d.GetDimension().x && row_idx < Virtual2DMapper_d.GetDimension().y){
        //Map virtual pixel to virtual location: 
        float2 VirtualLocation = Virtual2DMapper_d.ToLoc((float)col_idx, (float)row_idx);

        //Transform from virtual space to fixed and moving space, then map to pixel space: 
        float2 Mapped_FixedPixel = Fixed2DMapper_d.ToPxl(VirtualToFixed2D_d * VirtualLocation);
        float2 Mapper_MovingPixel = Moving2DMapper_d.ToPxl(VirtualToMoving2D_d * VirtualLocation);

        //Start interpolate: 
        float InterpPixel = 0.0f, ratio_x = 0.0f, ratio_y = 0.0f;
        // For Fixed Image: 
        if(Mapped_FixedPixel.x >= 0 && Mapped_FixedPixel.x <= (Fixed2DMapper_d.GetDimension().x - 1) && Mapped_FixedPixel.y >= 0 && Mapped_FixedPixel.y <= (Fixed2DMapper_d.GetDimension().y - 1)){
            //Along y direction: 
            ratio_y = Mapped_FixedPixel.y - floor(Mapped_FixedPixel.y);
            float interp_xL = FixedSpace[(int)floor(Mapped_FixedPixel.x) + (int)floor(Mapped_FixedPixel.y) * Fixed2DMapper_d.GetDimension().x] + ratio_y * (FixedSpace[(int)floor(Mapped_FixedPixel.x) + (int)ceil(Mapped_FixedPixel.y) * Fixed2DMapper_d.GetDimension().x] - FixedSpace[(int)floor(Mapped_FixedPixel.x) + (int)floor(Mapped_FixedPixel.y) * Fixed2DMapper_d.GetDimension().x]); 
            float interp_xR = FixedSpace[(int)ceil(Mapped_FixedPixel.x) + (int)floor(Mapped_FixedPixel.y) * Fixed2DMapper_d.GetDimension().x] + ratio_y * (FixedSpace[(int)ceil(Mapped_FixedPixel.x) + (int)ceil(Mapped_FixedPixel.y) * Fixed2DMapper_d.GetDimension().x] - FixedSpace[(int)ceil(Mapped_FixedPixel.x) + (int)floor(Mapped_FixedPixel.y) * Fixed2DMapper_d.GetDimension().x]); 
            
            //Along x direction: 
            ratio_x = Mapped_FixedPixel.x - floor(Mapped_FixedPixel.x);
            InterpPixel = interp_xL + ratio_x * (interp_xR - interp_xL);

            //Assign: 
            VirtualSpace_Fixed[col_idx + row_idx * Virtual2DMapper_d.GetDimension().x] = InterpPixel;
        }
        else{
            VirtualSpace_Fixed[col_idx + row_idx * Virtual2DMapper_d.GetDimension().x] = 0.0f;;
        }

        // For Moving Image: 
        if(Mapper_MovingPixel.x >= 0 && Mapper_MovingPixel.x <= (Moving2DMapper_d.GetDimension().x - 1) && Mapper_MovingPixel.y >= 0 && Mapper_MovingPixel.y <= (Moving2DMapper_d.GetDimension().y - 1)){
            //Along y direction: 
            ratio_y = Mapper_MovingPixel.y - floor(Mapper_MovingPixel.y);
            float interp_xL = MovingSpace[(int)floor(Mapper_MovingPixel.x) + (int)floor(Mapper_MovingPixel.y) * Moving2DMapper_d.GetDimension().x] + ratio_y * (MovingSpace[(int)floor(Mapper_MovingPixel.x) + (int)ceil(Mapper_MovingPixel.y) * Moving2DMapper_d.GetDimension().x] - MovingSpace[(int)floor(Mapper_MovingPixel.x) + (int)floor(Mapper_MovingPixel.y) * Moving2DMapper_d.GetDimension().x]); 
            float interp_xR = MovingSpace[(int)ceil(Mapper_MovingPixel.x) + (int)floor(Mapper_MovingPixel.y) * Moving2DMapper_d.GetDimension().x] + ratio_y * (MovingSpace[(int)ceil(Mapper_MovingPixel.x) + (int)ceil(Mapper_MovingPixel.y) * Moving2DMapper_d.GetDimension().x] - MovingSpace[(int)ceil(Mapper_MovingPixel.x) + (int)floor(Mapper_MovingPixel.y) * Moving2DMapper_d.GetDimension().x]); 
            
            //Along x direction: 
            ratio_x = Mapper_MovingPixel.x - floor(Mapper_MovingPixel.x);
            InterpPixel = interp_xL + ratio_x * (interp_xR - interp_xL);

            //Assign: 
            VirtualSpace_Moving[col_idx + row_idx * Virtual2DMapper_d.GetDimension().x] = InterpPixel;
        }
        else{
            VirtualSpace_Moving[col_idx + row_idx * Virtual2DMapper_d.GetDimension().x] = 0.0f;;
        }
    }
}

Interpolator3D::Interpolator3D(const std::string yaml_filePath, bool _useIGTL){
    InputParameterPath = yaml_filePath;

    counter = 0;

    //Load parameter file:
    interpolator_yaml_handle = YAML::LoadFile(yaml_filePath);
    
    // parse image data by ITK: 
    VolumeImporter importer; 

    //Fixed volume: 
    FixedFilePath = interpolator_yaml_handle["InputParameters"]["FixedVolume"]["FilePath"].as<std::string>();
    {
        importer.setFilePath(FixedFilePath); 
        if(importer.read()){
            //extract contents: 
            importer.getDimension( 
                dimensionFixed.x, 
                dimensionFixed.y, 
                dimensionFixed.z
            ); 

            importer.getSpacing( 
                spacingFixed.x, 
                spacingFixed.y, 
                spacingFixed.z
            ); 
            
            importer.getOrigin( 
                originFixed.x, 
                originFixed.y, 
                originFixed.z
            ); 

            //Initialize fixed volume:
            FixedVolume_h.resize(importer.getBufferSize(), 0.0f);
            thrust::copy( 
                importer.getBufferPtr(), 
                importer.getBufferPtr() + importer.getBufferSize(), 
                FixedVolume_h.data()
            ); 
            FixedVolume_d = FixedVolume_h;
        }
        else{
            std::cout << "ERROR reading the volume: " << FixedFilePath << std::endl; 
            std::exit(1); 
        }
    }

    //Moving volume: 
    MovingFilePath = interpolator_yaml_handle["InputParameters"]["MovingVolume"]["FilePath"].as<std::string>();
    {
        importer.setFilePath(MovingFilePath); 
        if(importer.read()){
            //extract contents: 
            importer.getDimension( 
                dimensionMoving.x, 
                dimensionMoving.y, 
                dimensionMoving.z
            ); 

            importer.getSpacing( 
                spacingMoving.x, 
                spacingMoving.y, 
                spacingMoving.z
            ); 
            
            importer.getOrigin( 
                originMoving.x, 
                originMoving.y, 
                originMoving.z
            ); 
            
            //Initialize moving volume:
            MovingVolume_h.resize(importer.getBufferSize(), 0.0f);
            thrust::copy( 
                importer.getBufferPtr(), 
                importer.getBufferPtr() + importer.getBufferSize(), 
                MovingVolume_h.data()
            ); 
            MovingVolume_d = MovingVolume_h;
        }
    }

    //Initialize virtual plane: 
    originVirtual = originFixed;
    HighResolutionModality = interpolator_yaml_handle["InputParameters"]["RegistrationParameters"]["HighResolutionModality"].as<std::string>();
    spacingVirtual.x = interpolator_yaml_handle["InputParameters"]["RegistrationParameters"]["ResampledSpacing"]["x"].as<float>();
    spacingVirtual.y = interpolator_yaml_handle["InputParameters"]["RegistrationParameters"]["ResampledSpacing"]["y"].as<float>();
    spacingVirtual.z = interpolator_yaml_handle["InputParameters"]["RegistrationParameters"]["ResampledSpacing"]["z"].as<float>();

    FinalExportPath = interpolator_yaml_handle["InputParameters"]["RegistrationParameters"]["FinalRoundExportRootPath"].as<std::string>();

    Centered = interpolator_yaml_handle["InputParameters"]["RegistrationParameters"]["CenterOverlaid"].as<bool>();

    if(Centered){
        std::cout << "Fixed and Moving spaces are centered. " << std::endl; 
    }
    else{
        std::cout << "Fixed and Moving spaces are overlaid with their origins. " << std::endl; 
    }

    dimensionVirtual.x = (int)ceil( (dimensionFixed.x * spacingFixed.x) / spacingVirtual.x );
    dimensionVirtual.y = (int)ceil( (dimensionFixed.y * spacingFixed.y) / spacingVirtual.y );
    dimensionVirtual.z = (int)ceil( (dimensionFixed.z * spacingFixed.z) / spacingVirtual.z );

    //Print virtual info:
    {
        std::cout << "Interpolation is performed on an intermediate virtual space: " << std::endl;
        std::cout << "Virtual Spacing: [x: " << spacingVirtual.x << "], " << "[y: " << spacingVirtual.y << "], " << "[z: " << spacingVirtual.z << "]. " << std::endl;
        std::cout << "Virtual Dimension: [x: " << dimensionVirtual.x << "], " << "[y: " << dimensionVirtual.y << "], " << "[z: " << dimensionVirtual.z << "]. " << std::endl;
    }

    //Initialize Interpolated volume:
    InterpFixedVolume_d.resize(dimensionVirtual.x * dimensionVirtual.y * dimensionVirtual.z, 0.0f);
    InterpMovingVolume_d.resize(dimensionVirtual.x * dimensionVirtual.y * dimensionVirtual.z, 0.0f);

    if(interpolator_yaml_handle["InputParameters"]["RegistrationParameters"]["DisplayPattern"]){
        DisplayPattern = interpolator_yaml_handle["InputParameters"]["RegistrationParameters"]["DisplayPattern"].as<int>(); 
    }
    else{
        DisplayPattern = 0; 
    }

    //Initialize OpenIgtLink: 
    UseIGTL = _useIGTL; 
    if(UseIGTL){
        TransformMsg = igtl::TransformMessage::New();
        TransformMsg->SetDeviceName("RegistrationServer");

        ServerSocket = igtl::ServerSocket::New();
        int _r = ServerSocket->CreateServer(23333);
        if (_r < 0){
            std::cerr << "Cannot create a server socket." << std::endl;
            exit(0);
        }

        std::cout << "Registration Server: server created, localhost:23333" << std::endl; 

        //wait client to connect: 
        auto ConnectToClient = [&](){
            while(CommunicationSocket.IsNull()){
                CommunicationSocket = ServerSocket->WaitForConnection(200); 
            }
            return 0;
        }; 
        std::future<int> LaunchConnectionWaiter = std::async(ConnectToClient); 
        int x = LaunchConnectionWaiter.get();
        
        std::string _Client_Address; 
        int _Client_Port; 
        CommunicationSocket->GetSocketAddressAndPort(_Client_Address, _Client_Port); 
        std::cout << "Registration Server: client connected, from " << _Client_Address << ":" << _Client_Port << std::endl; 
    }

    LC2SimilarityMeasure = std::make_unique<LC2_3D_class>(); 
}

Interpolator3D::Interpolator3D(const LC2Configuration3D* ExternalConfig){
    if(!ExternalConfig->IsConfigured){
        std::cout << "Please configure the Parameter file! " << std::endl; 
        return; 
    }
    //Port in the configuration parameters: 
    counter = 0;

    dimensionFixed.x = ExternalConfig->FixedDimension.x;
    dimensionFixed.y = ExternalConfig->FixedDimension.y;
    dimensionFixed.z = ExternalConfig->FixedDimension.z;

    spacingFixed.x = ExternalConfig->FixedSpacing.x;
    spacingFixed.y = ExternalConfig->FixedSpacing.y;
    spacingFixed.z = ExternalConfig->FixedSpacing.z;

    originFixed.x = ExternalConfig->FixedOrigin.x;
    originFixed.y = ExternalConfig->FixedOrigin.y;
    originFixed.z = ExternalConfig->FixedOrigin.z;

    FixedFilePath = ExternalConfig->FixedImageFilePath;

    dimensionMoving.x = ExternalConfig->MovingDimension.x;
    dimensionMoving.y = ExternalConfig->MovingDimension.y;
    dimensionMoving.z = ExternalConfig->MovingDimension.z;

    spacingMoving.x = ExternalConfig->MovingSpacing.x;
    spacingMoving.y = ExternalConfig->MovingSpacing.y;
    spacingMoving.z = ExternalConfig->MovingSpacing.z;

    originMoving.x = ExternalConfig->MovingOrigin.x;
    originMoving.y = ExternalConfig->MovingOrigin.y; 
    originMoving.z = ExternalConfig->MovingOrigin.z;

    MovingFilePath = ExternalConfig->MovingImageFilePath;
    
    originVirtual = originFixed;
    if(ExternalConfig->IsFixedHighResolution){
        HighResolutionModality = "Fixed"; 
    }
    else if(ExternalConfig->IsMovingHighResolution){
        HighResolutionModality = "Moving"; 
    }
    
    spacingVirtual.x = ExternalConfig->SamplerSpacing.x;
    spacingVirtual.y = ExternalConfig->SamplerSpacing.y;
    spacingVirtual.z = ExternalConfig->SamplerSpacing.z;

    FinalExportPath = ExternalConfig->ExportPath;

    Centered = ExternalConfig->IsCenterOverlaid;

    if(Centered){
        std::cout << "Fixed and Moving spaces are centered. " << std::endl; 
    }
    else{
        std::cout << "Fixed and Moving spaces are overlaid with their origins. " << std::endl; 
    }

    dimensionVirtual.x = (int)ceil( (dimensionFixed.x * spacingFixed.x) / spacingVirtual.x );
    dimensionVirtual.y = (int)ceil( (dimensionFixed.y * spacingFixed.y) / spacingVirtual.y );
    dimensionVirtual.z = (int)ceil( (dimensionFixed.z * spacingFixed.z) / spacingVirtual.z );

    //Print virtual info:
    {
        std::cout << "Interpolation is performed on an intermediate virtual space: " << std::endl;
        std::cout << "Virtual Spacing: [x: " << spacingVirtual.x << "], " << "[y: " << spacingVirtual.y << "], " << "[z: " << spacingVirtual.z << "]. " << std::endl;
        std::cout << "Virtual Dimension: [x: " << dimensionVirtual.x << "], " << "[y: " << dimensionVirtual.y << "], " << "[z: " << dimensionVirtual.z << "]. " << std::endl;
    }

    //Initialize fixed volume:
    FixedVolume_h.clear(); 
    FixedVolume_d.resize(ExternalConfig->FixedDimension.x * ExternalConfig->FixedDimension.y * ExternalConfig->FixedDimension.z, 0.0f); 
    thrust::copy( 
        ExternalConfig->FixedBuffer, 
        ExternalConfig->FixedBuffer + ExternalConfig->FixedDimension.x * ExternalConfig->FixedDimension.y * ExternalConfig->FixedDimension.z, 
        FixedVolume_d.begin()
    ); 

    //Initialize moving volume: 
    MovingVolume_h.clear(); 
    MovingVolume_d.resize(ExternalConfig->MovingDimension.x * ExternalConfig->MovingDimension.y * ExternalConfig->MovingDimension.z, 0.0f); 
    thrust::copy( 
        ExternalConfig->MovingBuffer, 
        ExternalConfig->MovingBuffer + ExternalConfig->MovingDimension.x * ExternalConfig->MovingDimension.y * ExternalConfig->MovingDimension.z, 
        MovingVolume_d.begin()
    ); 

    //Initialize Interpolated volume:
    InterpFixedVolume_d.resize(dimensionVirtual.x * dimensionVirtual.y * dimensionVirtual.z, 0.0f);
    InterpMovingVolume_d.resize(dimensionVirtual.x * dimensionVirtual.y * dimensionVirtual.z, 0.0f);

    DisplayPattern = 0; 
    LC2SimilarityMeasure = std::make_unique<LC2_3D_class>(); 
}

Interpolator3D::~Interpolator3D(){
    if(CommunicationSocket.IsNotNull()){
        CommunicationSocket->CloseSocket();
    }
}

void Interpolator3D::InitiateMappers(){
    Fixed3DMapper_h.SetSpacing(spacingFixed);
    Fixed3DMapper_h.SetOrigin(originFixed);
    Fixed3DMapper_h.SetDimension(dimensionFixed);
    Fixed3DMapper_h.SetCenter();

    Moving3DMapper_h.SetSpacing(spacingMoving);
    Moving3DMapper_h.SetOrigin(originMoving);
    Moving3DMapper_h.SetDimension(dimensionMoving);
    Moving3DMapper_h.SetCenter();

    Virtual3DMapper_h.SetSpacing(spacingVirtual);
    Virtual3DMapper_h.SetOrigin(originVirtual);
    Virtual3DMapper_h.SetDimension(dimensionVirtual);
    Virtual3DMapper_h.SetCenter();

    //Copy to constant memory:
    cudaMemcpyToSymbol(Fixed3DMapper_d, &Fixed3DMapper_h, sizeof(PhysioSpatial3DMapper));
    cudaMemcpyToSymbol(Moving3DMapper_d, &Moving3DMapper_h, sizeof(PhysioSpatial3DMapper));
    cudaMemcpyToSymbol(Virtual3DMapper_d, &Virtual3DMapper_h, sizeof(PhysioSpatial3DMapper));

    VirtualToMoving3D_h.Identity();
    VirtualToMoving3D_h.SetCenter(Fixed3DMapper_h.GetCenter());
    if(Centered){
        float3 shift_movingToVirtual = Moving3DMapper_h.GetCenter() - Virtual3DMapper_h.GetCenter();
        VirtualToMoving3D_h.SetShift(shift_movingToVirtual); 
    }
    else{
        VirtualToMoving3D_h.SetShift();
    }

    VirtualToFixed3D_h.Identity();
    VirtualToFixed3D_h.SetCenter(Fixed3DMapper_h.GetCenter());
    VirtualToFixed3D_h.SetShift();
}

void Interpolator3D::InitiateRegistrationHandle(){
    LC2SimilarityMeasure->PortedFromInterpolator(dimensionVirtual.x, dimensionVirtual.y, dimensionVirtual.z); 
}

void Interpolator3D::GenerateMappers(float3 RotationAngle, float3 Translation){
    VirtualToMoving3D_h.RigidInit(RotationAngle, Translation);
    cudaMemcpyToSymbol(VirtualToMoving3D_d, &VirtualToMoving3D_h, sizeof(Spatial3DMapper));

    cudaMemcpyToSymbol(VirtualToFixed3D_d, &VirtualToFixed3D_h, sizeof(Spatial3DMapper));
}

void Interpolator3D::GenerateMappers(float3 RotationAngle, float3 Translation, float3 scale, float3 shear){
    VirtualToMoving3D_h.AffineInit(RotationAngle, Translation, scale, shear);

    if(UseIGTL){
        float tmpMatrix[16]; 
        VirtualToMoving3D_h.GetMyselfInversed(tmpMatrix); 

        //Send from server: 
        igtl::Matrix4x4 _matrix; 
        _matrix[0][0] = tmpMatrix[0]; 
        _matrix[0][1] = tmpMatrix[1]; 
        _matrix[0][2] = tmpMatrix[2]; 
        _matrix[0][3] = tmpMatrix[3]; 
        _matrix[1][0] = tmpMatrix[4]; 
        _matrix[1][1] = tmpMatrix[5]; 
        _matrix[1][2] = tmpMatrix[6]; 
        _matrix[1][3] = tmpMatrix[7]; 
        _matrix[2][0] = tmpMatrix[8]; 
        _matrix[2][1] = tmpMatrix[9]; 
        _matrix[2][2] = tmpMatrix[10]; 
        _matrix[2][3] = tmpMatrix[11]; 
        _matrix[3][0] = tmpMatrix[12]; 
        _matrix[3][1] = tmpMatrix[13]; 
        _matrix[3][2] = tmpMatrix[14]; 
        _matrix[3][3] = tmpMatrix[15]; 

        TransformMsg->SetDeviceName("RegistrationServer");
        TransformMsg->SetHeaderVersion(2); 
        TransformMsg->SetMatrix(_matrix);
        TransformMsg->Pack();
        CommunicationSocket->Send(TransformMsg->GetPackPointer(), TransformMsg->GetPackSize());
    }

    cudaMemcpyToSymbol(VirtualToMoving3D_d, &VirtualToMoving3D_h, sizeof(Spatial3DMapper));
    cudaMemcpyToSymbol(VirtualToFixed3D_d, &VirtualToFixed3D_h, sizeof(Spatial3DMapper));
}

void Interpolator3D::GetCurrentTransform(float _t[16]){
    VirtualToMoving3D_h.GetMyselfInversed(_t); 
}

void Interpolator3D::Interpolate(const int Blk_Dim_x, const int Blk_Dim_y, const int Blk_Dim_z){
    //kernel threads resources initialization: 
    BlockDim_Interpolator.x = Blk_Dim_x; BlockDim_Interpolator.y = Blk_Dim_y; BlockDim_Interpolator.z = Blk_Dim_z; 
    GridDim_Interpolator.x = (int)ceil(dimensionVirtual.x / (float)Blk_Dim_x); 
    GridDim_Interpolator.y = (int)ceil(dimensionVirtual.y / (float)Blk_Dim_y); 
    GridDim_Interpolator.z = (int)ceil(dimensionVirtual.z / (float)Blk_Dim_z); 

    if(HighResolutionModality == "Fixed"){
        Interpolate_3Dkernel<<<GridDim_Interpolator, BlockDim_Interpolator>>>( 
            thrust::raw_pointer_cast(LC2SimilarityMeasure->GetHighResolutionBuffer().data()), 
            thrust::raw_pointer_cast(LC2SimilarityMeasure->GetUltrasoundBuffer().data()), 
            thrust::raw_pointer_cast(FixedVolume_d.data()), 
            thrust::raw_pointer_cast(MovingVolume_d.data())
        );
    }
    else if(HighResolutionModality == "Moving"){
        Interpolate_3Dkernel<<<GridDim_Interpolator, BlockDim_Interpolator>>>( 
            thrust::raw_pointer_cast(LC2SimilarityMeasure->GetUltrasoundBuffer().data()), 
            thrust::raw_pointer_cast(LC2SimilarityMeasure->GetHighResolutionBuffer().data()), 
            thrust::raw_pointer_cast(FixedVolume_d.data()), 
            thrust::raw_pointer_cast(MovingVolume_d.data())
        );
    }
    
    ++counter;

    // thrust::host_vector<float> test_fixed = LC2_3D_class::HighResolution_volume_static; 
    // thrust::host_vector<float> test_moving = LC2_3D_class::Ultrasound_volume; 

    // writeToBin(test_fixed.data(), dimensionVirtual.x * dimensionVirtual.y * dimensionVirtual.z, "/home/wenhai/img_registration_ws/fixed.bin"); 
    // writeToBin(test_moving.data(), dimensionVirtual.x * dimensionVirtual.y * dimensionVirtual.z, "/home/wenhai/img_registration_ws/moving.bin"); 
    // std::cout << counter << std::endl;
}

double Interpolator3D::GetSimilarityMeasure(const int patchSize){
    LC2SimilarityMeasure->ShowImages(DisplayPattern);

    if(patchSize == 0){
        return LC2SimilarityMeasure->GetMutualInformationSimilarityMetric(); 
    }

    LC2SimilarityMeasure->PrepareGradientFilter(); 
    LC2SimilarityMeasure->CalculateGradient(); 
    // std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now(); 
    // float _sim = LC2_3D_class::GetSimilarityMetric(8, 8, 8, patchSize); 
    // std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now(); 
    // std::cout << "Time for each iteration = " << std::chrono::duration_cast<std::chrono::milliseconds> (end - begin).count() << "[ms]" << std::endl;
    return LC2SimilarityMeasure->GetSimilarityMetric(8, 8, 8, patchSize); 
}

void Interpolator3D::GetMovingToFixedTransform(float3 RotationAngle, float3 Translation, float* outMovingToFixed){
    VirtualToMoving3D_h.RigidInit(RotationAngle, Translation);

    float MovingToVirtualMatrix[16] = {0.0f}, VirtualToFixedMatrix[16] = {0.0f}; 
    VirtualToMoving3D_h.GetMyselfInversed(); 
    VirtualToMoving3D_h.GetMatrix(MovingToVirtualMatrix); 
    VirtualToFixed3D_h.GetMatrix(VirtualToFixedMatrix); 

    for(int row_it = 0; row_it < 4; ++row_it){
        for(int col_it = 0; col_it < 4; ++col_it){
            outMovingToFixed[col_it + row_it * 4] = 0.0f; 
            for(int element_it = 0; element_it < 4; ++element_it){
                outMovingToFixed[col_it + row_it * 4] += VirtualToFixedMatrix[element_it + row_it * 4] * MovingToVirtualMatrix[col_it + (element_it) * 4];
            }
        }
    }
}

void Interpolator3D::GetMovingToFixedTransform(float3 RotationAngle, float3 Translation, float3 scale, float3 shear, float* outMovingToFixed){
    VirtualToMoving3D_h.AffineInit(RotationAngle, Translation, scale, shear);

    // float affineMatrix[16] = {0.0f}; 
    // affineMatrix[15] = 1.0f; 
    // affineMatrix[0] = scale.x; affineMatrix[5] = scale.y; affineMatrix[10] = scale.z; //apply scaling 
    // affineMatrix[1] = shear.x; 
    // affineMatrix[4] = shear.y; 
    // affineMatrix[9] = shear.z; 
    // //Regularization on rotation center: 
    // affineMatrix[3] = VirtualToMoving3D_h.GetCenter().x - scale.x * VirtualToMoving3D_h.GetCenter().x - VirtualToMoving3D_h.GetCenter().y * shear.x; 
    // affineMatrix[7] = VirtualToMoving3D_h.GetCenter().y - scale.y * VirtualToMoving3D_h.GetCenter().y - VirtualToMoving3D_h.GetCenter().x * shear.y; 
    // affineMatrix[11] = VirtualToMoving3D_h.GetCenter().z - scale.z * VirtualToMoving3D_h.GetCenter().z - VirtualToMoving3D_h.GetCenter().y * shear.z; 
    // VirtualToMoving3D_h.ApplyAdditionalMatrix(affineMatrix); 

    float MovingToVirtualMatrix[16] = {0.0f}, VirtualToFixedMatrix[16] = {0.0f}; 
    VirtualToMoving3D_h.GetMyselfInversed(); 
    VirtualToMoving3D_h.GetMatrix(MovingToVirtualMatrix); 
    VirtualToFixed3D_h.GetMatrix(VirtualToFixedMatrix); 

    for(int row_it = 0; row_it < 4; ++row_it){
        for(int col_it = 0; col_it < 4; ++col_it){
            outMovingToFixed[col_it + row_it * 4] = 0.0f; 
            for(int element_it = 0; element_it < 4; ++element_it){
                outMovingToFixed[col_it + row_it * 4] += VirtualToFixedMatrix[element_it + row_it * 4] * MovingToVirtualMatrix[col_it + (element_it) * 4];
            }
        }
    }
}

void Interpolator3D::WriteOut(){
    std::string FixedImageName( 
        FixedFilePath.begin() + FixedFilePath.find_last_of("/"), 
        FixedFilePath.begin() + FixedFilePath.find_last_of(".")
    ); 
    
    std::string MovingImageName( 
        MovingFilePath.begin() + MovingFilePath.find_last_of("/"), 
        MovingFilePath.begin() + MovingFilePath.find_last_of(".")
    ); 

    MovingImageName = MovingImageName + "_Registered.nrrd"; 
    FixedImageName = FixedImageName + ".nrrd"; 

    //EXPORT: NRRD
    std::ofstream Export_NRRD;
    Export_NRRD.open(FinalExportPath + FixedImageName, std::ofstream::out);
    Export_NRRD << "NRRD0004" << std::endl;
    Export_NRRD << "type: float" << std::endl;
    Export_NRRD << "dimension: 3" << std::endl;
    Export_NRRD << "space: scanner-xyz" << std::endl;
    Export_NRRD << "sizes: " << dimensionVirtual.x << " " << dimensionVirtual.y << " " << dimensionVirtual.z << std::endl;
    Export_NRRD << "space directions: (" << spacingVirtual.x << ",0,0) (0," << spacingVirtual.y << ",0) (0,0," << spacingVirtual.z << ")" << std::endl;
    Export_NRRD << "kinds: domain domain domain" << std::endl;
    Export_NRRD << "endian: little" << std::endl;
    Export_NRRD << "encoding: raw" << std::endl;
    Export_NRRD << "space origin: (0,0,0)" << std::endl << std::endl;
    Export_NRRD.close();
    
    InterpFixedVolume_h = LC2SimilarityMeasure->GetHighResolutionBuffer(); 
    writeToBin(thrust::raw_pointer_cast(InterpFixedVolume_h.data()), dimensionVirtual.x * dimensionVirtual.y * dimensionVirtual.z, FinalExportPath + FixedImageName); 

    Export_NRRD.open(FinalExportPath + MovingImageName, std::ofstream::out);
    Export_NRRD << "NRRD0004" << std::endl;
    Export_NRRD << "type: float" << std::endl;
    Export_NRRD << "dimension: 3" << std::endl;
    Export_NRRD << "space: scanner-xyz" << std::endl;
    Export_NRRD << "sizes: " << dimensionVirtual.x << " " << dimensionVirtual.y << " " << dimensionVirtual.z << std::endl;
    Export_NRRD << "space directions: (" << spacingVirtual.x << ",0,0) (0," << spacingVirtual.y << ",0) (0,0," << spacingVirtual.z << ")" << std::endl;
    Export_NRRD << "kinds: domain domain domain" << std::endl;
    Export_NRRD << "endian: little" << std::endl;
    Export_NRRD << "encoding: raw" << std::endl;
    Export_NRRD << "space origin: (0,0,0)" << std::endl << std::endl;
    Export_NRRD.close();
    InterpMovingVolume_h = LC2SimilarityMeasure->GetUltrasoundBuffer(); 
    writeToBin(thrust::raw_pointer_cast(InterpMovingVolume_h.data()), dimensionVirtual.x * dimensionVirtual.y * dimensionVirtual.z, FinalExportPath + MovingImageName); 

    std::cout << "Final round results in virtual space are saved at: " << std::endl; 
    std::cout << FinalExportPath << ", " << FixedImageName << ", " << MovingImageName << std::endl;
    
    // EXPORT: raw
    // InterpFixedVolume_h = LC2_3D_class::HighResolution_volume_static; 
    // InterpMovingVolume_h = LC2_3D_class::Ultrasound_volume; 
    // writeToBin(thrust::raw_pointer_cast(InterpFixedVolume_h.data()), dimensionVirtual.x * dimensionVirtual.y * dimensionVirtual.z, FinalExportPath + "/fixed.raw"); 
    // writeToBin(thrust::raw_pointer_cast(InterpMovingVolume_h.data()), dimensionVirtual.x * dimensionVirtual.y * dimensionVirtual.z, FinalExportPath + "/moving.raw"); 
}

void Interpolator3D::WriteOut(float* OptimizedMatrix){
    std::string FixedImageNameBase( 
        FixedFilePath.begin() + FixedFilePath.find_last_of("/"), 
        FixedFilePath.begin() + FixedFilePath.find_last_of(".")
    ); 
    
    std::string MovingImageNameBase( 
        MovingFilePath.begin() + MovingFilePath.find_last_of("/"), 
        MovingFilePath.begin() + MovingFilePath.find_last_of(".")
    ); 

    std::string MovingImageName = MovingImageNameBase + ".nrrd"; 
    std::string MovingImageName_registered = MovingImageNameBase + "_Registered.nrrd"; 
    std::string FixedImageName = FixedImageNameBase + ".nrrd"; 

    //EXPORT: NRRD
    std::ofstream Export_NRRD;
    // Export_NRRD.open(FinalExportPath + FixedImageName, std::ofstream::out);
    // Export_NRRD << "NRRD0004" << std::endl;
    // Export_NRRD << "type: float" << std::endl;
    // Export_NRRD << "dimension: 3" << std::endl;
    // Export_NRRD << "space: scanner-xyz" << std::endl;
    // Export_NRRD << "sizes: " << dimensionFixed.x << " " << dimensionFixed.y << " " << dimensionFixed.z << std::endl;
    // Export_NRRD << "space directions: (" << spacingFixed.x << ",0,0) (0," << spacingFixed.y << ",0) (0,0," << spacingFixed.z << ")" << std::endl;
    // Export_NRRD << "kinds: domain domain domain" << std::endl;
    // Export_NRRD << "endian: little" << std::endl;
    // Export_NRRD << "encoding: raw" << std::endl;
    // Export_NRRD << "space origin: (" << originFixed.x << "," << originFixed.y << "," << originFixed.z << ")" << std::endl << std::endl;
    // Export_NRRD.close();
    
    // writeToBin(thrust::raw_pointer_cast(FixedVolume_h.data()), dimensionFixed.x * dimensionFixed.y * dimensionFixed.z, FinalExportPath + FixedImageName); 

    // Export_NRRD.open(FinalExportPath + MovingImageName, std::ofstream::out);
    // Export_NRRD << "NRRD0004" << std::endl;
    // Export_NRRD << "type: float" << std::endl;
    // Export_NRRD << "dimension: 3" << std::endl;
    // Export_NRRD << "space: scanner-xyz" << std::endl;
    // Export_NRRD << "sizes: " << dimensionMoving.x << " " << dimensionMoving.y << " " << dimensionMoving.z << std::endl;
    // Export_NRRD << "space directions: (" << spacingMoving.x << ",0,0) (0," << spacingMoving.y << ",0) (0,0," << spacingMoving.z << ")" << std::endl;
    // Export_NRRD << "kinds: domain domain domain" << std::endl;
    // Export_NRRD << "endian: little" << std::endl;
    // Export_NRRD << "encoding: raw" << std::endl;
    // Export_NRRD << "space origin: (" << originMoving.x << "," << originMoving.y << "," << originMoving.z << ")" << std::endl << std::endl;
    // Export_NRRD.close();
    
    // writeToBin(thrust::raw_pointer_cast(MovingVolume_h.data()), dimensionMoving.x * dimensionMoving.y * dimensionMoving.z, FinalExportPath + MovingImageName); 

    Export_NRRD.open(FinalExportPath + MovingImageName_registered, std::ofstream::out);
    Export_NRRD << "NRRD0004" << std::endl;
    Export_NRRD << "type: float" << std::endl;
    Export_NRRD << "dimension: 3" << std::endl;
    Export_NRRD << "space: scanner-xyz" << std::endl;
    Export_NRRD << "sizes: " << dimensionMoving.x << " " << dimensionMoving.y << " " << dimensionMoving.z << std::endl;
    
    float directionMovingTransformed_X[4] = {0.0f}; 
    float directionMovingTransformed_Y[4] = {0.0f}; 
    float directionMovingTransformed_Z[4] = {0.0f}; 
    
    {//Calculate spacing direction: 
        float directionMoving_X[4] = {spacingMoving.x, 0.0f, 0.0f, 1.0f}; 
        float directionMoving_Y[4] = {0.0f, spacingMoving.y, 0.0f, 1.0f}; 
        float directionMoving_Z[4] = {0.0f, 0.0f, spacingMoving.z, 1.0f}; 

        float accu_X = 0.0f, accu_Y = 0.0f, accu_Z = 0.0f; 
        for(int row_idx = 0; row_idx < 3; ++ row_idx){
            for(int ele_idx = 0; ele_idx < 3; ++ele_idx){
                accu_X += OptimizedMatrix[ele_idx + row_idx * 4] * directionMoving_X[ele_idx]; 
                accu_Y += OptimizedMatrix[ele_idx + row_idx * 4] * directionMoving_Y[ele_idx]; 
                accu_Z += OptimizedMatrix[ele_idx + row_idx * 4] * directionMoving_Z[ele_idx]; 
            }
            directionMovingTransformed_X[row_idx] = accu_X; accu_X = 0.0f; 
            directionMovingTransformed_Y[row_idx] = accu_Y; accu_Y = 0.0f; 
            directionMovingTransformed_Z[row_idx] = accu_Z; accu_Z = 0.0f; 
        }
    }
    Export_NRRD << "space directions: " << 
    "(" << directionMovingTransformed_X[0] << "," << directionMovingTransformed_X[1] << "," << directionMovingTransformed_X[2] << ") " << 
    "(" << directionMovingTransformed_Y[0] << "," << directionMovingTransformed_Y[1] << "," << directionMovingTransformed_Y[2] << ") " << 
    "(" << directionMovingTransformed_Z[0] << "," << directionMovingTransformed_Z[1] << "," << directionMovingTransformed_Z[2] << ") " << std::endl;

    Export_NRRD << "kinds: domain domain domain" << std::endl;
    Export_NRRD << "endian: little" << std::endl;
    Export_NRRD << "encoding: raw" << std::endl;

    float originMovingTransformed[4] = {0.0f}; 
    float originMoving_array[4] = {originMoving.x, originMoving.y, originMoving.z, 1.0f}; 
    float accu_origin = 0.0f; 
    for(int row_idx = 0; row_idx < 4; ++ row_idx){
        for(int ele_idx = 0; ele_idx < 4; ++ele_idx){
            accu_origin += OptimizedMatrix[ele_idx + row_idx * 4] * originMoving_array[ele_idx]; 
        }
        originMovingTransformed[row_idx] = accu_origin; accu_origin = 0.0f; 
    }

    Export_NRRD << "space origin: (" << originMovingTransformed[0] << "," << originMovingTransformed[1] << "," << originMovingTransformed[2] << ")" << std::endl << std::endl;
    Export_NRRD.close();

    writeToBin(thrust::raw_pointer_cast(MovingVolume_h.data()), dimensionMoving.x * dimensionMoving.y * dimensionMoving.z, FinalExportPath + MovingImageName_registered); 

    std::cout << "Final round results in virtual space are saved at: " << std::endl; 
    std::cout << FinalExportPath << ", " << FixedImageName << ", " << MovingImageName_registered << std::endl;

    //Write out mtx: 
    std::ofstream MtxFileHandle; 
    MtxFileHandle.open(FinalExportPath + "/MovingToFixed.mat", std::ofstream::out); 
    for(int rowIdx = 0; rowIdx < 4; ++ rowIdx){
        for(int colIdx = 0; colIdx < 4; ++colIdx){
            MtxFileHandle << OptimizedMatrix[rowIdx * 4 + colIdx] << ","; 
        }
        MtxFileHandle << std::endl; 
    }
    MtxFileHandle.close(); 

    // //export weighting map: 
    // LC2_3D_class::GenerateWeightingMap(); 

    // //export maps: 
    // Export_NRRD.open(FinalExportPath + "/WeightingMap.nrrd", std::ofstream::out);
    // Export_NRRD << "NRRD0004" << std::endl;
    // Export_NRRD << "type: float" << std::endl;
    // Export_NRRD << "dimension: 3" << std::endl;
    // Export_NRRD << "space: scanner-xyz" << std::endl;
    // Export_NRRD << "sizes: " << dimensionVirtual.x << " " << dimensionVirtual.y << " " << dimensionVirtual.z << std::endl;
    // Export_NRRD << "space directions: (" << spacingVirtual.x << ",0,0) (0," << spacingVirtual.y << ",0) (0,0," << spacingVirtual.z << ")" << std::endl;
    // Export_NRRD << "kinds: domain domain domain" << std::endl;
    // Export_NRRD << "endian: little" << std::endl;
    // Export_NRRD << "encoding: raw" << std::endl;
    // Export_NRRD << "space origin: (" << originVirtual.x << "," << originVirtual.y << "," << originVirtual.z << ")" << std::endl << std::endl;
    // Export_NRRD.close();
    // thrust::host_vector<float> weightingMap = LC2_3D_class::GetWeightedMapBuffer(); 
    // writeToBin(thrust::raw_pointer_cast(weightingMap.data()), dimensionVirtual.x * dimensionVirtual.y * dimensionVirtual.z, FinalExportPath + "/WeightingMap.nrrd"); 

    // Export_NRRD.open(FinalExportPath + "/WeightingMap_f.nrrd", std::ofstream::out);
    // Export_NRRD << "NRRD0004" << std::endl;
    // Export_NRRD << "type: float" << std::endl;
    // Export_NRRD << "dimension: 3" << std::endl;
    // Export_NRRD << "space: scanner-xyz" << std::endl;
    // Export_NRRD << "sizes: " << dimensionVirtual.x << " " << dimensionVirtual.y << " " << dimensionVirtual.z << std::endl;
    // Export_NRRD << "space directions: (" << spacingVirtual.x << ",0,0) (0," << spacingVirtual.y << ",0) (0,0," << spacingVirtual.z << ")" << std::endl;
    // Export_NRRD << "kinds: domain domain domain" << std::endl;
    // Export_NRRD << "endian: little" << std::endl;
    // Export_NRRD << "encoding: raw" << std::endl;
    // Export_NRRD << "space origin: (" << originVirtual.x << "," << originVirtual.y << "," << originVirtual.z << ")" << std::endl << std::endl;
    // Export_NRRD.close();
    // thrust::host_vector<float> weightingMap_f = LC2_3D_class::GetUltrasoundBuffer(); 
    // writeToBin(thrust::raw_pointer_cast(weightingMap_f.data()), dimensionVirtual.x * dimensionVirtual.y * dimensionVirtual.z, FinalExportPath + "/WeightingMap_f.nrrd"); 

    // Export_NRRD.open(FinalExportPath + "/WeightingMap_m.nrrd", std::ofstream::out);
    // Export_NRRD << "NRRD0004" << std::endl;
    // Export_NRRD << "type: float" << std::endl;
    // Export_NRRD << "dimension: 3" << std::endl;
    // Export_NRRD << "space: scanner-xyz" << std::endl;
    // Export_NRRD << "sizes: " << dimensionVirtual.x << " " << dimensionVirtual.y << " " << dimensionVirtual.z << std::endl;
    // Export_NRRD << "space directions: (" << spacingVirtual.x << ",0,0) (0," << spacingVirtual.y << ",0) (0,0," << spacingVirtual.z << ")" << std::endl;
    // Export_NRRD << "kinds: domain domain domain" << std::endl;
    // Export_NRRD << "endian: little" << std::endl;
    // Export_NRRD << "encoding: raw" << std::endl;
    // Export_NRRD << "space origin: (" << originVirtual.x << "," << originVirtual.y << "," << originVirtual.z << ")" << std::endl << std::endl;
    // Export_NRRD.close();
    // thrust::host_vector<float> weightingMap_m = LC2_3D_class::GetHighResolutionBuffer(); 
    // writeToBin(thrust::raw_pointer_cast(weightingMap_m.data()), dimensionVirtual.x * dimensionVirtual.y * dimensionVirtual.z, FinalExportPath + "/WeightingMap_m.nrrd"); 

    // Export_NRRD.open(FinalExportPath + "/WeightingMap_m_gradient.nrrd", std::ofstream::out);
    // Export_NRRD << "NRRD0004" << std::endl;
    // Export_NRRD << "type: float" << std::endl;
    // Export_NRRD << "dimension: 3" << std::endl;
    // Export_NRRD << "space: scanner-xyz" << std::endl;
    // Export_NRRD << "sizes: " << dimensionVirtual.x << " " << dimensionVirtual.y << " " << dimensionVirtual.z << std::endl;
    // Export_NRRD << "space directions: (" << spacingVirtual.x << ",0,0) (0," << spacingVirtual.y << ",0) (0,0," << spacingVirtual.z << ")" << std::endl;
    // Export_NRRD << "kinds: domain domain domain" << std::endl;
    // Export_NRRD << "endian: little" << std::endl;
    // Export_NRRD << "encoding: raw" << std::endl;
    // Export_NRRD << "space origin: (" << originVirtual.x << "," << originVirtual.y << "," << originVirtual.z << ")" << std::endl << std::endl;
    // Export_NRRD.close();
    // thrust::host_vector<float> weightingMap_m_gradient = LC2_3D_class::GetHighResolutionGradientBuffer(); 
    // writeToBin(thrust::raw_pointer_cast(weightingMap_m_gradient.data()), dimensionVirtual.x * dimensionVirtual.y * dimensionVirtual.z, FinalExportPath + "/WeightingMap_m_gradient.nrrd"); 

    // EXPORT: raw
    // InterpFixedVolume_h = LC2_3D_class::HighResolution_volume_static; 
    // InterpMovingVolume_h = LC2_3D_class::Ultrasound_volume; 
    // writeToBin(thrust::raw_pointer_cast(InterpFixedVolume_h.data()), dimensionVirtual.x * dimensionVirtual.y * dimensionVirtual.z, FinalExportPath + "/fixed.raw"); 
    // writeToBin(thrust::raw_pointer_cast(InterpMovingVolume_h.data()), dimensionVirtual.x * dimensionVirtual.y * dimensionVirtual.z, FinalExportPath + "/moving.raw"); 
}

/* --------------------------------------------------------- 2D ----------------------------------------------------------------- */
Interpolator2D::Interpolator2D(const std::string yaml_filePath){
    InputParameterPath = yaml_filePath;

    counter = 0;

    //Load parameter file:
    interpolator_yaml_handle = YAML::LoadFile(yaml_filePath);

    // parse image data by ITK: 
    VolumeImporter importer; 

    //Fixed volume: 
    FixedImagePath = interpolator_yaml_handle["InputParameters"]["FixedImage"]["FilePath"].as<std::string>();
    {
        importer.setFilePath(FixedImagePath); 
        if(importer.read()){
            //extract contents: 
            int _placeHolderInt; 
            float _placeHolderFloat; 
            importer.getDimension( 
                dimensionFixed.x, 
                dimensionFixed.y, 
                _placeHolderInt
            ); 

            importer.getSpacing( 
                spacingFixed.x, 
                spacingFixed.y, 
                _placeHolderFloat
            ); 
            
            importer.getOrigin( 
                originFixed.x, 
                originFixed.y, 
                _placeHolderFloat
            ); 

            //Initialize fixed volume:
            FixedImage_h.resize(importer.getBufferSize(), 0.0f);
            thrust::copy( 
                importer.getBufferPtr(), 
                importer.getBufferPtr() + importer.getBufferSize(), 
                FixedImage_h.data()
            ); 
            FixedImage_d = FixedImage_h;
        }
        else{
            std::cout << "ERROR reading the volume: " << FixedImagePath << std::endl; 
            std::exit(1); 
        }
    }

    //Moving volume: 
    MovingImagePath = interpolator_yaml_handle["InputParameters"]["MovingImage"]["FilePath"].as<std::string>();
    {
        importer.setFilePath(MovingImagePath); 
        if(importer.read()){
            //extract contents: 
            int _placeHolderInt; 
            float _placeHolderFloat; 
            importer.getDimension( 
                dimensionMoving.x, 
                dimensionMoving.y, 
                _placeHolderInt
            ); 

            importer.getSpacing( 
                spacingMoving.x, 
                spacingMoving.y, 
                _placeHolderFloat
            ); 
            
            importer.getOrigin( 
                originMoving.x, 
                originMoving.y, 
                _placeHolderFloat
            ); 
            
            //Initialize moving volume:
            MovingImage_h.resize(importer.getBufferSize(), 0.0f);
            thrust::copy( 
                importer.getBufferPtr(), 
                importer.getBufferPtr() + importer.getBufferSize(), 
                MovingImage_h.data()
            ); 
            MovingImage_d = MovingImage_h;
        }
    }

    //extract geometry info if supplied: 
    if(interpolator_yaml_handle["InputParameters"]["FixedImage"]["Dimension"]["x"]){
        dimensionFixed.x = interpolator_yaml_handle["InputParameters"]["FixedImage"]["Dimension"]["x"].as<int>();
    }

    if(interpolator_yaml_handle["InputParameters"]["FixedImage"]["Dimension"]["y"]){
        dimensionFixed.y = interpolator_yaml_handle["InputParameters"]["FixedImage"]["Dimension"]["y"].as<int>();
    }

    if(interpolator_yaml_handle["InputParameters"]["FixedImage"]["Spacing"]["x"]){
        spacingFixed.x = interpolator_yaml_handle["InputParameters"]["FixedImage"]["Spacing"]["x"].as<float>();
    }

    if(interpolator_yaml_handle["InputParameters"]["FixedImage"]["Spacing"]["y"]){
        spacingFixed.y = interpolator_yaml_handle["InputParameters"]["FixedImage"]["Spacing"]["y"].as<float>();
    }

    if(interpolator_yaml_handle["InputParameters"]["FixedImage"]["Origin"]["x"]){
        originFixed.x = interpolator_yaml_handle["InputParameters"]["FixedImage"]["Origin"]["x"].as<float>();
    }

    if(interpolator_yaml_handle["InputParameters"]["FixedImage"]["Origin"]["y"]){
        originFixed.y = interpolator_yaml_handle["InputParameters"]["FixedImage"]["Origin"]["y"].as<float>();
    }

    //extract geometry info if supplied: 
    if(interpolator_yaml_handle["InputParameters"]["MovingImage"]["Dimension"]["x"]){
        dimensionMoving.x = interpolator_yaml_handle["InputParameters"]["MovingImage"]["Dimension"]["x"].as<int>();
    }

    if(interpolator_yaml_handle["InputParameters"]["MovingImage"]["Dimension"]["y"]){
        dimensionMoving.y = interpolator_yaml_handle["InputParameters"]["MovingImage"]["Dimension"]["y"].as<int>();
    }

    if(interpolator_yaml_handle["InputParameters"]["MovingImage"]["Spacing"]["x"]){
        spacingMoving.x = interpolator_yaml_handle["InputParameters"]["MovingImage"]["Spacing"]["x"].as<float>();
    }

    if(interpolator_yaml_handle["InputParameters"]["MovingImage"]["Spacing"]["y"]){
        spacingMoving.y = interpolator_yaml_handle["InputParameters"]["MovingImage"]["Spacing"]["y"].as<float>();
    }

    if(interpolator_yaml_handle["InputParameters"]["MovingImage"]["Origin"]["x"]){
        originMoving.x = interpolator_yaml_handle["InputParameters"]["MovingImage"]["Origin"]["x"].as<float>();
    }

    if(interpolator_yaml_handle["InputParameters"]["MovingImage"]["Origin"]["y"]){
        originMoving.y = interpolator_yaml_handle["InputParameters"]["MovingImage"]["Origin"]["y"].as<float>();
    }

    //Initialize virtual space: 
    originVirtual = originFixed;
    HighResolutionModality = interpolator_yaml_handle["InputParameters"]["RegistrationParameters"]["HighResolutionModality"].as<std::string>();
    spacingVirtual.x = interpolator_yaml_handle["InputParameters"]["RegistrationParameters"]["ResampledSpacing"]["x"].as<float>();
    spacingVirtual.y = interpolator_yaml_handle["InputParameters"]["RegistrationParameters"]["ResampledSpacing"]["y"].as<float>();

    FinalExportPath = interpolator_yaml_handle["InputParameters"]["RegistrationParameters"]["FinalRoundExportRootPath"].as<std::string>();

    Centered = interpolator_yaml_handle["InputParameters"]["RegistrationParameters"]["CenterOverlaid"].as<bool>();

    if(Centered){
        std::cout << "Fixed and Moving spaces are centered. " << std::endl; 
    }
    else{
        std::cout << "Fixed and Moving spaces are overlaid with their origins. " << std::endl; 
    }
    
    dimensionVirtual.x = (int)ceil( (dimensionFixed.x * spacingFixed.x) / spacingVirtual.x );
    dimensionVirtual.y = (int)ceil( (dimensionFixed.y * spacingFixed.y) / spacingVirtual.y );

    //Print virtual info:
    {
        std::cout << "Interpolation is performed on an intermediate virtual space: " << std::endl;
        std::cout << "Virtual Spacing: [x: " << spacingVirtual.x << "], " << "[y: " << spacingVirtual.y << "]. " << std::endl;
        std::cout << "Virtual Dimension: [x: " << dimensionVirtual.x << "], " << "[y: " << dimensionVirtual.y << "]. " << std::endl;
    }

    //Initialize Interpolated volume:
    InterpFixedImage_h.resize(dimensionVirtual.x * dimensionVirtual.y, 0.0f);
    InterpFixedImage_d.resize(dimensionVirtual.x * dimensionVirtual.y, 0.0f);
    InterpMovingImage_h.resize(dimensionVirtual.x * dimensionVirtual.y, 0.0f);
    InterpMovingImage_d.resize(dimensionVirtual.x * dimensionVirtual.y, 0.0f);

    if(interpolator_yaml_handle["InputParameters"]["RegistrationParameters"]["DisplayPattern"]){
        DisplayPattern = interpolator_yaml_handle["InputParameters"]["RegistrationParameters"]["DisplayPattern"].as<int>(); 
    }
    else{
        DisplayPattern = 0; 
    }

    LC2SimilarityMeasure = std::make_unique<LC2_2D_class>(); 
}

Interpolator2D::Interpolator2D(const LC2Configuration2D* ExternalConfig){
    if(!ExternalConfig->IsConfigured){
        std::cout << "Please configure the Parameter file! " << std::endl; 
        return; 
    }
    //Port in the configuration parameters: 
    counter = 0;

    dimensionFixed.x = ExternalConfig->FixedDimension.x;
    dimensionFixed.y = ExternalConfig->FixedDimension.y;

    spacingFixed.x = ExternalConfig->FixedSpacing.x;
    spacingFixed.y = ExternalConfig->FixedSpacing.y;

    originFixed.x = ExternalConfig->FixedOrigin.x;
    originFixed.y = ExternalConfig->FixedOrigin.y;

    FixedImagePath = ExternalConfig->FixedImageFilePath;

    dimensionMoving.x = ExternalConfig->MovingDimension.x;
    dimensionMoving.y = ExternalConfig->MovingDimension.y;

    spacingMoving.x = ExternalConfig->MovingSpacing.x;
    spacingMoving.y = ExternalConfig->MovingSpacing.y;

    originMoving.x = ExternalConfig->MovingOrigin.x;
    originMoving.y = ExternalConfig->MovingOrigin.y; 

    MovingImagePath = ExternalConfig->MovingImageFilePath;
    
    originVirtual = originFixed;
    if(ExternalConfig->IsFixedHighResolution){
        HighResolutionModality = "Fixed"; 
    }
    else if(ExternalConfig->IsMovingHighResolution){
        HighResolutionModality = "Moving"; 
    }
    
    spacingVirtual.x = ExternalConfig->SamplerSpacing.x;
    spacingVirtual.y = ExternalConfig->SamplerSpacing.y;

    FinalExportPath = ExternalConfig->ExportPath;

    Centered = ExternalConfig->IsCenterOverlaid;

    if(Centered){
        std::cout << "Fixed and Moving spaces are centered. " << std::endl; 
    }
    else{
        std::cout << "Fixed and Moving spaces are overlaid with their origins. " << std::endl; 
    }

    dimensionVirtual.x = (int)ceil( (dimensionFixed.x * spacingFixed.x) / spacingVirtual.x );
    dimensionVirtual.y = (int)ceil( (dimensionFixed.y * spacingFixed.y) / spacingVirtual.y );

    //Print virtual info:
    {
        std::cout << "Interpolation is performed on an intermediate virtual space: " << std::endl;
        std::cout << "Virtual Spacing: [x: " << spacingVirtual.x << "], " << "[y: " << spacingVirtual.y << "]. " << std::endl;
        std::cout << "Virtual Dimension: [x: " << dimensionVirtual.x << "], " << "[y: " << dimensionVirtual.y << "]. " << std::endl;
    }

    //Initialize fixed volume:
    FixedImage_h.clear(); 
    FixedImage_d.resize(ExternalConfig->FixedDimension.x * ExternalConfig->FixedDimension.y, 0.0f); 
    thrust::copy( 
        ExternalConfig->FixedBuffer, 
        ExternalConfig->FixedBuffer + ExternalConfig->FixedDimension.x * ExternalConfig->FixedDimension.y, 
        FixedImage_d.begin()
    ); 
    FixedImage_h = FixedImage_d; 

    //Initialize moving volume: 
    MovingImage_h.clear(); 
    MovingImage_d.resize(ExternalConfig->MovingDimension.x * ExternalConfig->MovingDimension.y, 0.0f); 
    thrust::copy( 
        ExternalConfig->MovingBuffer, 
        ExternalConfig->MovingBuffer + ExternalConfig->MovingDimension.x * ExternalConfig->MovingDimension.y, 
        MovingImage_d.begin()
    ); 
    MovingImage_h = MovingImage_d; 

    //Initialize Interpolated volume:
    InterpFixedImage_d.resize(dimensionVirtual.x * dimensionVirtual.y, 0.0f);
    InterpFixedImage_h.resize(dimensionVirtual.x * dimensionVirtual.y, 0.0f);
    InterpMovingImage_d.resize(dimensionVirtual.x * dimensionVirtual.y, 0.0f);
    InterpMovingImage_h.resize(dimensionVirtual.x * dimensionVirtual.y, 0.0f);

    DisplayPattern = 0; 

    LC2SimilarityMeasure = std::make_unique<LC2_2D_class>(); 
}

void Interpolator2D::InitiateMappers(){
    Fixed2DMapper_h.SetSpacing(spacingFixed);
    Fixed2DMapper_h.SetOrigin(originFixed);
    Fixed2DMapper_h.SetDimension(dimensionFixed);
    Fixed2DMapper_h.SetCenter();

    Moving2DMapper_h.SetSpacing(spacingMoving);
    Moving2DMapper_h.SetOrigin(originMoving);
    Moving2DMapper_h.SetDimension(dimensionMoving);
    Moving2DMapper_h.SetCenter();

    Virtual2DMapper_h.SetSpacing(spacingVirtual);
    Virtual2DMapper_h.SetOrigin(originVirtual);
    Virtual2DMapper_h.SetDimension(dimensionVirtual);
    Virtual2DMapper_h.SetCenter();

    //Copy to constant memory:
    cudaMemcpyToSymbol(Fixed2DMapper_d, &Fixed2DMapper_h, sizeof(PhysioSpatial2DMapper));
    cudaMemcpyToSymbol(Moving2DMapper_d, &Moving2DMapper_h, sizeof(PhysioSpatial2DMapper));
    cudaMemcpyToSymbol(Virtual2DMapper_d, &Virtual2DMapper_h, sizeof(PhysioSpatial2DMapper));

    VirtualToMoving2D_h.Identity();
    // VirtualToMoving2D_h.SetCenter(Moving2DMapper_h.GetCenter());
    VirtualToMoving2D_h.SetCenter(Fixed2DMapper_h.GetCenter());
    // VirtualToMoving2D_h.SetCenter(Virtual2DMapper_h.GetCenter());
    if(Centered){
        float2 shift_movingToVirtual = Moving2DMapper_h.GetCenter() - Virtual2DMapper_h.GetCenter();
        VirtualToMoving2D_h.SetShift(shift_movingToVirtual);
    }
    else{
        VirtualToMoving2D_h.SetShift(); 
    }

    VirtualToFixed2D_h.Identity();
    VirtualToFixed2D_h.SetCenter(Fixed2DMapper_h.GetOrigin());
    VirtualToFixed2D_h.SetShift();
}

void Interpolator2D::InitiateRegistrationHandle(){
    LC2SimilarityMeasure->PortedFromInterpolator(dimensionVirtual.x, dimensionVirtual.y); 
}

void Interpolator2D::GenerateMappers(float RotationAngle, float2 Translation){
    
    VirtualToMoving2D_h.RigidInit(RotationAngle, Translation);
    cudaMemcpyToSymbol(VirtualToMoving2D_d, &VirtualToMoving2D_h, sizeof(Spatial2DMapper));

    cudaMemcpyToSymbol(VirtualToFixed2D_d, &VirtualToFixed2D_h, sizeof(Spatial2DMapper));
}

void Interpolator2D::GenerateMappers(float RotationAngle, float2 Translation, float2 scale, float shear){
    VirtualToMoving2D_h.AffineInit(RotationAngle, Translation, scale, shear);

    // float affineMatrix[9] = {0.0f}; 
    // affineMatrix[8] = 1.0f; 
    // affineMatrix[0] = scale.x; affineMatrix[4] = scale.y; //apply scaling 
    // affineMatrix[1] = shear; affineMatrix[3] = 0.0f;
    // //Regularization on rotation center: 
    // affineMatrix[2] = VirtualToMoving2D_h.GetCenter().x - scale.x * VirtualToMoving2D_h.GetCenter().x - VirtualToMoving2D_h.GetCenter().y * shear; 
    // affineMatrix[5] = VirtualToMoving2D_h.GetCenter().y - scale.y * VirtualToMoving2D_h.GetCenter().y; 

    // VirtualToMoving2D_h.ApplyAdditionalMatrix(affineMatrix); 

    cudaMemcpyToSymbol(VirtualToMoving2D_d, &VirtualToMoving2D_h, sizeof(Spatial2DMapper));

    cudaMemcpyToSymbol(VirtualToFixed2D_d, &VirtualToFixed2D_h, sizeof(Spatial2DMapper));
}

void Interpolator2D::Interpolate(const int Blk_Dim_x, const int Blk_Dim_y){
    //kernel threads resources initialization: 
    BlockDim_Interpolator.x = Blk_Dim_x; BlockDim_Interpolator.y = Blk_Dim_y; BlockDim_Interpolator.z = 1; 
    GridDim_Interpolator.x = (int)ceil(dimensionVirtual.x / (float)Blk_Dim_x); 
    GridDim_Interpolator.y = (int)ceil(dimensionVirtual.y / (float)Blk_Dim_y); 
    GridDim_Interpolator.z = 1; 

 /* CPU implementation: 

    //Start to fill in the virtual plane: 
    //Iterate every pixel in virtual space:
    for(int row_it = 0; row_it < dimensionVirtual.y; ++row_it){
        for(int col_it = 0; col_it < dimensionVirtual.x; ++col_it){
            //Map virtual pixel to virtual location: 
            float2 virtualPixel; 
            virtualPixel.x = (float)col_it; virtualPixel.y = (float)row_it; 
            float2 virtualLocation = Virtual2DMapper_h.ToLoc(virtualPixel);

            //Extract location from fixed and moving space: 
            float2 MappedFixedLocation, MappedMovingLocation; 
            MappedFixedLocation = VirtualToFixed2D_h * virtualLocation;
            MappedMovingLocation = VirtualToMoving2D_h * virtualLocation; 

            //Map fixed, and moving location to Pixel: 
            float2 MappedFixedPixel, MappedMovingPixel; 
            MappedFixedPixel = Fixed2DMapper_h.ToPxl(MappedFixedLocation); 
            MappedMovingPixel = Moving2DMapper_h.ToPxl(MappedMovingLocation); 

            float InterpPixel = 0.0f, ratio_x = 0.0f, ratio_y = 0.0f;
            //Interpolate Fixed image pixel: 
            {
                if(MappedFixedPixel.x >= 0 && MappedFixedPixel.x <= (dimensionFixed.x - 1) && MappedFixedPixel.y >= 0 && MappedFixedPixel.y <= (dimensionFixed.y - 1)){
                    int c_x = (int)ceil(MappedFixedPixel.x); int f_x = (int)floor(MappedFixedPixel.x); 
                    int c_y = (int)ceil(MappedFixedPixel.y); int f_y = (int)floor(MappedFixedPixel.y); 
                    //Along y direction: 
                    ratio_y = MappedFixedPixel.y - floor(MappedFixedPixel.y);
                    float interp_xL = FixedImage_h[f_x + f_y * dimensionFixed.x] + ratio_y * (FixedImage_h[f_x + c_y * dimensionFixed.x] - FixedImage_h[f_x + f_y * dimensionFixed.x]); 
                    float interp_xR = FixedImage_h[c_x + f_y * dimensionFixed.x] + ratio_y * (FixedImage_h[c_x + c_y * dimensionFixed.x] - FixedImage_h[c_x + f_y * dimensionFixed.x]); 
                    
                    //Along x direction: 
                    ratio_x = MappedFixedPixel.x - floor(MappedFixedPixel.x);
                    InterpPixel = interp_xL + ratio_x * (interp_xR - interp_xL);

                    //Assign: 
                    InterpFixedImage_h[col_it + row_it * dimensionVirtual.x] = InterpPixel;
                }
                else{
                    InterpFixedImage_h[col_it + row_it * dimensionVirtual.x] = 0.0f;
                }
            }
            //Interpolate Moving image pixel: 
            {
                if(MappedMovingPixel.x >= 0 && MappedMovingPixel.x <= (dimensionMoving.x - 1) && MappedMovingPixel.y >= 0 && MappedMovingPixel.y <= (dimensionMoving.y - 1)){
                    int c_x = (int)ceil(MappedMovingPixel.x); int f_x = (int)floor(MappedMovingPixel.x); 
                    int c_y = (int)ceil(MappedMovingPixel.y); int f_y = (int)floor(MappedMovingPixel.y); 
                    //Along y direction: 
                    ratio_y = MappedMovingPixel.y - floor(MappedMovingPixel.y);
                    float interp_xL = MovingImage_h[f_x + f_y * dimensionMoving.x] + ratio_y * (MovingImage_h[f_x + c_y * dimensionMoving.x] - MovingImage_h[f_x + f_y * dimensionMoving.x]); 
                    float interp_xR = MovingImage_h[c_x + f_y * dimensionMoving.x] + ratio_y * (MovingImage_h[c_x + c_y * dimensionMoving.x] - MovingImage_h[c_x + f_y * dimensionMoving.x]); 
                    
                    //Along x direction: 
                    ratio_x = MappedMovingPixel.x - floor(MappedMovingPixel.x);
                    InterpPixel = interp_xL + ratio_x * (interp_xR - interp_xL);

                    //Assign: 
                    InterpMovingImage_h[col_it + row_it * dimensionVirtual.x] = InterpPixel;
                }
                else{
                    InterpMovingImage_h[col_it + row_it * dimensionVirtual.x] = 0.0f;
                }
            }
        }
    }

    InterpFixedImage_d = InterpFixedImage_h; 
    InterpMovingImage_d = InterpMovingImage_h;
 */ 

 ///* GPU implementation: 
    if(HighResolutionModality == "Fixed"){
        Interpolate_2Dkernel<<<GridDim_Interpolator, BlockDim_Interpolator>>>( 
            thrust::raw_pointer_cast(LC2SimilarityMeasure->GetHighResolutionRawPointer()), //interpolated fixed, 
            thrust::raw_pointer_cast(LC2SimilarityMeasure->GetUltrasoundRawPointer()), //interpolated moving, 
            thrust::raw_pointer_cast(FixedImage_d.data()), 
            thrust::raw_pointer_cast(MovingImage_d.data())
        );
    }
    else if(HighResolutionModality == "Moving"){
        Interpolate_2Dkernel<<<GridDim_Interpolator, BlockDim_Interpolator>>>( 
            thrust::raw_pointer_cast(LC2SimilarityMeasure->GetUltrasoundRawPointer()), //interpolated moving, 
            thrust::raw_pointer_cast(LC2SimilarityMeasure->GetHighResolutionRawPointer()), //interpolated fixed, 
            thrust::raw_pointer_cast(FixedImage_d.data()), 
            thrust::raw_pointer_cast(MovingImage_d.data())
        );
    }
    
    ++counter;

    // // //Copy out for debug. 
    // InterpFixedImage_h = InterpFixedImage_d; 
    // InterpMovingImage_h = InterpMovingImage_d; 
    // writeToBin(thrust::raw_pointer_cast(InterpFixedImage_h.data()), dimensionVirtual.x * dimensionVirtual.y, "/home/wenhai/vsc_workspace/RegistrationTools/config/Data/CT_KidneyPhantom_virtual.bin"); 
    // writeToBin(thrust::raw_pointer_cast(InterpMovingImage_h.data()), dimensionVirtual.x * dimensionVirtual.y, "/home/wenhai/vsc_workspace/RegistrationTools/config/Data/US_KidneyPhantom_virtual.bin"); 

 //*/

}

double Interpolator2D::GetSimilarityMeasure(const int patchSize){
    LC2SimilarityMeasure->ShowImages(DisplayPattern);

    LC2SimilarityMeasure->PrepareGradientFilter(); 
    LC2SimilarityMeasure->CalculateGradient(); 
    return LC2SimilarityMeasure->GetSimilarityMetric(16, 16, patchSize); 
}

void Interpolator2D::GetMovingToFixedTransform(float RotationAngle, float2 Translation, float* outMovingToFixed){
    VirtualToMoving2D_h.RigidInit(RotationAngle, Translation);

    float MovingToVirtualMatrix[9] = {0.0f}, VirtualToFixedMatrix[9] = {0.0f}; 
    VirtualToMoving2D_h.GetMyselfInversed(); 
    VirtualToMoving2D_h.GetMatrix(MovingToVirtualMatrix); 
    VirtualToFixed2D_h.GetMatrix(VirtualToFixedMatrix); 

    for(int row_it = 0; row_it < 3; ++row_it){
        for(int col_it = 0; col_it < 3; ++col_it){
            outMovingToFixed[col_it + row_it * 3] = 0.0f; 
            for(int element_it = 0; element_it < 3; ++element_it){
                outMovingToFixed[col_it + row_it * 3] += VirtualToFixedMatrix[element_it + row_it * 3] * MovingToVirtualMatrix[col_it + (element_it) * 3];
            }
        }
    }
}

void Interpolator2D::GetMovingToFixedTransform(float RotationAngle, float2 Translation, float2 scale, float shear, float* outMovingToFixed){
    VirtualToMoving2D_h.AffineInit(RotationAngle, Translation, scale, shear);

    // float affineMatrix[9] = {0.0f}; 
    // affineMatrix[8] = 1.0f; 
    // affineMatrix[0] = scale.x; affineMatrix[4] = scale.y; //apply scaling 
    // affineMatrix[1] = shear; affineMatrix[3] = 0.0f;
    // //Regularization on rotation center: 
    // affineMatrix[2] = VirtualToMoving2D_h.GetCenter().x - scale.x * VirtualToMoving2D_h.GetCenter().x - VirtualToMoving2D_h.GetCenter().y * shear; 
    // affineMatrix[5] = VirtualToMoving2D_h.GetCenter().y - scale.y * VirtualToMoving2D_h.GetCenter().y; 
    // VirtualToMoving2D_h.ApplyAdditionalMatrix(affineMatrix); 

    float MovingToVirtualMatrix[9] = {0.0f}, VirtualToFixedMatrix[9] = {0.0f}; 
    VirtualToMoving2D_h.GetMyselfInversed(); 
    VirtualToMoving2D_h.GetMatrix(MovingToVirtualMatrix); 
    VirtualToFixed2D_h.GetMatrix(VirtualToFixedMatrix); 

    for(int row_it = 0; row_it < 3; ++row_it){
        for(int col_it = 0; col_it < 3; ++col_it){
            outMovingToFixed[col_it + row_it * 3] = 0.0f; 
            for(int element_it = 0; element_it < 3; ++element_it){
                outMovingToFixed[col_it + row_it * 3] += VirtualToFixedMatrix[element_it + row_it * 3] * MovingToVirtualMatrix[col_it + (element_it) * 3];
            }
        }
    }
}

void Interpolator2D::WriteOut(){
    // // //Copy out for debug. 
    // InterpFixedImage_h = LC2_2D_class::HighResolution_static; 
    // InterpMovingImage_h = LC2_2D_class::Ultrasound; 
    // writeToBin(thrust::raw_pointer_cast(InterpFixedImage_h.data()), dimensionVirtual.x * dimensionVirtual.y, FinalExportPath + "/fixed.raw"); 
    // writeToBin(thrust::raw_pointer_cast(InterpMovingImage_h.data()), dimensionVirtual.x * dimensionVirtual.y, FinalExportPath + "/moving.raw"); 

    // std::cout << "Final round results in virtual space are saved at: " << std::endl; 
    // std::cout << "          " << FinalExportPath << ", fixed.raw, moving.raw" << std::endl; 
}

void Interpolator2D::WriteOut(float* OptimizedMatrix){
    std::string FixedImageName( 
        FixedImagePath.begin() + FixedImagePath.find_last_of("/"), 
        FixedImagePath.begin() + FixedImagePath.find_last_of(".")
    ); 
    
    std::string MovingImageName( 
        MovingImagePath.begin() + MovingImagePath.find_last_of("/"), 
        MovingImagePath.begin() + MovingImagePath.find_last_of(".")
    ); 

    std::cout << MovingImageName << std::endl; 

    //EXPORT: RAW
    //Transform current moving image: 
    Moving_Reg_Image_d.resize(FixedImage_d.size(), 0.0f); 

    //Get transform at pixel level: 
    double AffineTransform[2][3]; 
    for(int i = 0; i < 2; ++i){
        for(int j = 0; j < 3; ++j){
            AffineTransform[i][j] = OptimizedMatrix[j + i * 3]; 
        }
    }
    AffineTransform[0][2] /= spacingFixed.x; 
    AffineTransform[1][2] /= spacingFixed.y; //mm to pixel

    AffineTransform[0][0] *= (spacingMoving.x / spacingFixed.x); 
    AffineTransform[1][0] *= (spacingMoving.x / spacingFixed.x); 

    AffineTransform[0][1] *= (spacingMoving.y / spacingFixed.y); 
    AffineTransform[1][1] *= (spacingMoving.y / spacingFixed.y); 

    NppiSize ImageSize;
    ImageSize.width = dimensionMoving.x; 
    ImageSize.height = dimensionMoving.y;

    NppiRect ROISize_src;
    ROISize_src.x = 0;
    ROISize_src.y = 0;
    ROISize_src.width = dimensionMoving.x; 
    ROISize_src.height = dimensionMoving.y;

    NppiRect ROISize_dst;
    ROISize_dst.x = 0;
    ROISize_dst.y = 0;
    ROISize_dst.width = dimensionFixed.x; 
    ROISize_dst.height = dimensionFixed.y;

    //perform transformation:
    nppiWarpAffine_32f_C1R( 
        thrust::raw_pointer_cast(MovingImage_d.data()), 
        ImageSize, 
        dimensionMoving.x * sizeof(float),
        ROISize_src, 
        thrust::raw_pointer_cast(Moving_Reg_Image_d.data()), 
        dimensionFixed.x * sizeof(float), 
        ROISize_dst, 
        AffineTransform, 
        NPPI_INTER_LINEAR
    );

    //Copy registered moving image back to host mem: 
    Moving_Reg_Image_h = Moving_Reg_Image_d; 

    //check if the data are out of bound: 
    std::string MovingImageNameOut; // = MovingImageName + "_Registered.raw"; 
    float maxPixel = *thrust::max_element(Moving_Reg_Image_d.begin(), Moving_Reg_Image_d.end()); 
    if(maxPixel > 255){
        MovingImageNameOut = MovingImageName + "_Registered.raw"; 
        std::ofstream oFid; 
        oFid.open(FinalExportPath + MovingImageNameOut, std::ios::out | std::ios::binary); 
        oFid.write((char*)thrust::raw_pointer_cast(Moving_Reg_Image_h.data()), dimensionFixed.x * dimensionFixed.y * sizeof(float)); 
        oFid.close(); 
    }
    else{
        MovingImageNameOut = MovingImageName + "_Registered.png"; 

        std::vector<float> hostLocalMovingRegImage(thrust::raw_pointer_cast(Moving_Reg_Image_h.data()), thrust::raw_pointer_cast(Moving_Reg_Image_h.data()) + dimensionFixed.x * dimensionFixed.y); 
        for(auto it = hostLocalMovingRegImage.begin(); it < hostLocalMovingRegImage.end(); ++it){
            if(*it > 255 || *it < 0){
                *it = 0; 
            }
            else{
                *it = std::roundf(*it); 
            }
        }

        std::vector<uint8_t> targetImageToExport(dimensionFixed.x * dimensionFixed.y, 0); 
        std::copy(hostLocalMovingRegImage.begin(), hostLocalMovingRegImage.end(), targetImageToExport.begin()); 

        stbi_write_png(std::string(FinalExportPath + MovingImageNameOut).c_str(), dimensionFixed.x, dimensionFixed.y, 1, targetImageToExport.data(), 0); 
    }
    

    std::cout << "Moving image was registered, and result is saved at: " << std::endl; 
    std::cout << FinalExportPath << MovingImageNameOut << std::endl;
}

/* --------------------------------------------------------- 2DTo3D ----------------------------------------------------------------- */
Interpolator2DTo3D::Interpolator2DTo3D(const std::string yaml_filePath){
    InputParameterPath = yaml_filePath;

    counter = 0;

    //Load parameter file:
    interpolator_yaml_handle = YAML::LoadFile(yaml_filePath);

    //Start parse:
    ImageFileFormat = interpolator_yaml_handle["InputParameters"]["FixedVolume"]["Format"].as<std::string>();

    //Format: raw
    if(ImageFileFormat == "raw"){
        dimensionFixed.x = interpolator_yaml_handle["InputParameters"]["FixedVolume"]["Dimension"]["x"].as<int>();
        dimensionFixed.y = interpolator_yaml_handle["InputParameters"]["FixedVolume"]["Dimension"]["y"].as<int>();
        dimensionFixed.z = interpolator_yaml_handle["InputParameters"]["FixedVolume"]["Dimension"]["z"].as<int>();

        spacingFixed.x = interpolator_yaml_handle["InputParameters"]["FixedVolume"]["Spacing"]["x"].as<float>();
        spacingFixed.y = interpolator_yaml_handle["InputParameters"]["FixedVolume"]["Spacing"]["y"].as<float>();
        spacingFixed.z = interpolator_yaml_handle["InputParameters"]["FixedVolume"]["Spacing"]["z"].as<float>();

        originFixed.x = interpolator_yaml_handle["InputParameters"]["FixedVolume"]["Origin"]["x"].as<float>();
        originFixed.y = interpolator_yaml_handle["InputParameters"]["FixedVolume"]["Origin"]["y"].as<float>();
        originFixed.z = interpolator_yaml_handle["InputParameters"]["FixedVolume"]["Origin"]["z"].as<float>();

        FixedFilePath = interpolator_yaml_handle["InputParameters"]["FixedVolume"]["FilePath"].as<std::string>();

        dimensionMoving.x = interpolator_yaml_handle["InputParameters"]["MovingVolume"]["Dimension"]["x"].as<int>();
        dimensionMoving.y = interpolator_yaml_handle["InputParameters"]["MovingVolume"]["Dimension"]["y"].as<int>();
        dimensionMoving.z = interpolator_yaml_handle["InputParameters"]["MovingVolume"]["Dimension"]["z"].as<int>();

        spacingMoving.x = interpolator_yaml_handle["InputParameters"]["MovingVolume"]["Spacing"]["x"].as<float>();
        spacingMoving.y = interpolator_yaml_handle["InputParameters"]["MovingVolume"]["Spacing"]["y"].as<float>();
        spacingMoving.z = interpolator_yaml_handle["InputParameters"]["MovingVolume"]["Spacing"]["z"].as<float>();

        originMoving.x = interpolator_yaml_handle["InputParameters"]["MovingVolume"]["Origin"]["x"].as<float>();
        originMoving.y = interpolator_yaml_handle["InputParameters"]["MovingVolume"]["Origin"]["y"].as<float>();
        originMoving.z = interpolator_yaml_handle["InputParameters"]["MovingVolume"]["Origin"]["z"].as<float>();

        MovingFilePath = interpolator_yaml_handle["InputParameters"]["MovingVolume"]["FilePath"].as<std::string>();
        
        originVirtual = originFixed;
        HighResolutionModality = interpolator_yaml_handle["InputParameters"]["RegistrationParameters"]["HighResolutionModality"].as<std::string>();
        spacingVirtual.x = interpolator_yaml_handle["InputParameters"]["RegistrationParameters"]["ResampledSpacing"]["x"].as<float>();
        spacingVirtual.y = interpolator_yaml_handle["InputParameters"]["RegistrationParameters"]["ResampledSpacing"]["y"].as<float>();
        spacingVirtual.z = interpolator_yaml_handle["InputParameters"]["RegistrationParameters"]["ResampledSpacing"]["z"].as<float>();

        FinalExportPath = interpolator_yaml_handle["InputParameters"]["RegistrationParameters"]["FinalRoundExportRootPath"].as<std::string>();

        Centered = interpolator_yaml_handle["InputParameters"]["RegistrationParameters"]["CenterOverlaid"].as<bool>();

        if(Centered){
            std::cout << "Fixed and Moving spaces are centered. " << std::endl; 
        }
        else{
            std::cout << "Fixed and Moving spaces are overlaid with their origins. " << std::endl; 
        }

        if(interpolator_yaml_handle["InputParameters"]["RegistrationParameters"]["DisplayPattern"]){
            DisplayPattern = interpolator_yaml_handle["InputParameters"]["RegistrationParameters"]["DisplayPattern"].as<int>(); 
        }
        else{
            DisplayPattern = 0; 
        }

        dimensionVirtual.x = (int)ceil( (dimensionFixed.x * spacingFixed.x) / spacingVirtual.x );
        dimensionVirtual.y = (int)ceil( (dimensionFixed.y * spacingFixed.y) / spacingVirtual.y );
        dimensionVirtual.z = (int)ceil( (dimensionFixed.z * spacingFixed.z) / spacingVirtual.z );

        //Print virtual info:
        {
            std::cout << "Interpolation is performed on an intermediate virtual space: " << std::endl;
            std::cout << "Virtual Spacing: [x: " << spacingVirtual.x << "], " << "[y: " << spacingVirtual.y << "], " << "[z: " << spacingVirtual.z << "]. " << std::endl;
            std::cout << "Virtual Dimension: [x: " << dimensionVirtual.x << "], " << "[y: " << dimensionVirtual.y << "], " << "[z: " << dimensionVirtual.z << "]. " << std::endl;
        }

        //Initialize fixed volume:
        FixedVolume_h.resize(dimensionFixed.x * dimensionFixed.y * dimensionFixed.z, 0.0f);
        readFromBin(FixedVolume_h.data(), dimensionFixed.x * dimensionFixed.y * dimensionFixed.z, FixedFilePath);
        FixedVolume_d = FixedVolume_h;

        //Initialize moving volume: 
        MovingVolume_h.resize(dimensionMoving.x * dimensionMoving.y * dimensionMoving.z, 0.0f);
        readFromBin(MovingVolume_h.data(), dimensionMoving.x * dimensionMoving.y * dimensionMoving.z, MovingFilePath);
        MovingVolume_d = MovingVolume_h;

        //Initialize Interpolated volume:
        InterpFixedVolume_d.resize(dimensionVirtual.x * dimensionVirtual.y * dimensionVirtual.z, 0.0f);
        InterpMovingVolume_d.resize(dimensionVirtual.x * dimensionVirtual.y * dimensionVirtual.z, 0.0f);
    }
    else{
        //Other format:
    }

    LC2SimilarityMeasure = std::make_unique<LC2_2D_class>(); 
}

void Interpolator2DTo3D::InitiateMappers(){
    Fixed3DMapper_h.SetSpacing(spacingFixed);
    Fixed3DMapper_h.SetOrigin(originFixed);
    Fixed3DMapper_h.SetDimension(dimensionFixed);
    Fixed3DMapper_h.SetCenter();

    Moving3DMapper_h.SetSpacing(spacingMoving);
    Moving3DMapper_h.SetOrigin(originMoving);
    Moving3DMapper_h.SetDimension(dimensionMoving);
    Moving3DMapper_h.SetCenter();

    Virtual3DMapper_h.SetSpacing(spacingVirtual);
    Virtual3DMapper_h.SetOrigin(originVirtual);
    Virtual3DMapper_h.SetDimension(dimensionVirtual);
    Virtual3DMapper_h.SetCenter();

    //Copy to constant memory:
    cudaMemcpyToSymbol(Fixed3DMapper_d, &Fixed3DMapper_h, sizeof(PhysioSpatial3DMapper));
    cudaMemcpyToSymbol(Moving3DMapper_d, &Moving3DMapper_h, sizeof(PhysioSpatial3DMapper));
    cudaMemcpyToSymbol(Virtual3DMapper_d, &Virtual3DMapper_h, sizeof(PhysioSpatial3DMapper));

    VirtualToMoving3D_h.Identity();
    VirtualToMoving3D_h.SetCenter(Fixed3DMapper_h.GetCenter());
    if(Centered){
        float3 shift_movingToVirtual = Moving3DMapper_h.GetCenter() - Virtual3DMapper_h.GetCenter();
        VirtualToMoving3D_h.SetShift(shift_movingToVirtual); 
    }
    else{
        VirtualToMoving3D_h.SetShift();
    }

    VirtualToFixed3D_h.Identity();
    VirtualToFixed3D_h.SetCenter(Fixed3DMapper_h.GetOrigin());
    VirtualToFixed3D_h.SetShift();
}

void Interpolator2DTo3D::InitiateRegistrationHandle(){
    LC2SimilarityMeasure->PortedFromInterpolator(dimensionVirtual.x, dimensionVirtual.y); 
}

void Interpolator2DTo3D::GenerateMappers(float3 RotationAngle, float3 Translation){
    VirtualToMoving3D_h.RigidInit(RotationAngle, Translation);
    cudaMemcpyToSymbol(VirtualToMoving3D_d, &VirtualToMoving3D_h, sizeof(Spatial3DMapper));

    cudaMemcpyToSymbol(VirtualToFixed3D_d, &VirtualToFixed3D_h, sizeof(Spatial3DMapper));
}
void Interpolator2DTo3D::GenerateMappers(float3 RotationAngle, float3 Translation, float3 scale, float3 shear){
    VirtualToMoving3D_h.AffineInit(RotationAngle, Translation, scale, shear);
    cudaMemcpyToSymbol(VirtualToMoving3D_d, &VirtualToMoving3D_h, sizeof(Spatial3DMapper));

    cudaMemcpyToSymbol(VirtualToFixed3D_d, &VirtualToFixed3D_h, sizeof(Spatial3DMapper));
}

void Interpolator2DTo3D::Interpolate(const int Blk_Dim_x, const int Blk_Dim_y, const int Blk_Dim_z){
    //kernel threads resources initialization: 
    BlockDim_Interpolator.x = Blk_Dim_x; BlockDim_Interpolator.y = Blk_Dim_y; BlockDim_Interpolator.z = Blk_Dim_z; 
    GridDim_Interpolator.x = (int)ceil(dimensionVirtual.x / (float)Blk_Dim_x); 
    GridDim_Interpolator.y = (int)ceil(dimensionVirtual.y / (float)Blk_Dim_y); 
    GridDim_Interpolator.z = (int)ceil(dimensionVirtual.z / (float)Blk_Dim_z); 

    if(HighResolutionModality == "Fixed"){
        Interpolate_3Dkernel<<<GridDim_Interpolator, BlockDim_Interpolator>>>( 
            thrust::raw_pointer_cast(LC2SimilarityMeasure->GetHighResolutionRawPointer()), 
            thrust::raw_pointer_cast(LC2SimilarityMeasure->GetUltrasoundRawPointer()), 
            thrust::raw_pointer_cast(FixedVolume_d.data()), 
            thrust::raw_pointer_cast(MovingVolume_d.data())
        );
    }
    else if(HighResolutionModality == "Moving"){
        Interpolate_3Dkernel<<<GridDim_Interpolator, BlockDim_Interpolator>>>( 
            thrust::raw_pointer_cast(LC2SimilarityMeasure->GetUltrasoundRawPointer()), 
            thrust::raw_pointer_cast(LC2SimilarityMeasure->GetHighResolutionRawPointer()), 
            thrust::raw_pointer_cast(FixedVolume_d.data()), 
            thrust::raw_pointer_cast(MovingVolume_d.data())
        );
    }

    ++counter;

    // thrust::host_vector<float> test_fixed = LC2_3D_class::HighResolution_volume_static; 
    // thrust::host_vector<float> test_moving = LC2_3D_class::Ultrasound_volume; 

    // writeToBin(test_fixed.data(), dimensionVirtual.x * dimensionVirtual.y * dimensionVirtual.z, "/home/wenhai/img_registration_ws/fixed.bin"); 
    // writeToBin(test_moving.data(), dimensionVirtual.x * dimensionVirtual.y * dimensionVirtual.z, "/home/wenhai/img_registration_ws/moving.bin"); 
    // std::cout << counter << std::endl;
}

double Interpolator2DTo3D::GetSimilarityMeasure(const int patchSize){
    LC2SimilarityMeasure->ShowImages(DisplayPattern);
    LC2SimilarityMeasure->PrepareGradientFilter(); 
    LC2SimilarityMeasure->CalculateGradient(); 
    return LC2SimilarityMeasure->GetSimilarityMetric(16, 16, patchSize); 
}

void Interpolator2DTo3D::GetMovingToFixedTransform(float3 RotationAngle, float3 Translation, float* outMovingToFixed){
    VirtualToMoving3D_h.RigidInit(RotationAngle, Translation);

    float MovingToVirtualMatrix[16] = {0.0f}, VirtualToFixedMatrix[16] = {0.0f}; 
    VirtualToMoving3D_h.GetMyselfInversed(); 
    VirtualToMoving3D_h.GetMatrix(MovingToVirtualMatrix); 
    VirtualToFixed3D_h.GetMatrix(VirtualToFixedMatrix); 

    for(int row_it = 0; row_it < 4; ++row_it){
        for(int col_it = 0; col_it < 4; ++col_it){
            outMovingToFixed[col_it + row_it * 4] = 0.0f; 
            for(int element_it = 0; element_it < 4; ++element_it){
                outMovingToFixed[col_it + row_it * 4] += VirtualToFixedMatrix[element_it + row_it * 4] * MovingToVirtualMatrix[col_it + (element_it) * 4];
            }
        }
    }
}

void Interpolator2DTo3D::GetMovingToFixedTransform(float3 RotationAngle, float3 Translation, float3 scale, float3 shear, float* outMovingToFixed){
    VirtualToMoving3D_h.AffineInit(RotationAngle, Translation, scale, shear);

    // float affineMatrix[16] = {0.0f}; 
    // affineMatrix[15] = 1.0f; 
    // affineMatrix[0] = scale.x; affineMatrix[5] = scale.y; affineMatrix[10] = scale.z; //apply scaling 
    // affineMatrix[1] = shear.x; 
    // affineMatrix[4] = shear.y; 
    // affineMatrix[9] = shear.z; 
    // //Regularization on rotation center: 
    // affineMatrix[3] = VirtualToMoving3D_h.GetCenter().x - scale.x * VirtualToMoving3D_h.GetCenter().x - VirtualToMoving3D_h.GetCenter().y * shear.x; 
    // affineMatrix[7] = VirtualToMoving3D_h.GetCenter().y - scale.y * VirtualToMoving3D_h.GetCenter().y - VirtualToMoving3D_h.GetCenter().x * shear.y; 
    // affineMatrix[11] = VirtualToMoving3D_h.GetCenter().z - scale.z * VirtualToMoving3D_h.GetCenter().z - VirtualToMoving3D_h.GetCenter().y * shear.z; 
    // VirtualToMoving3D_h.ApplyAdditionalMatrix(affineMatrix); 

    float MovingToVirtualMatrix[16] = {0.0f}, VirtualToFixedMatrix[16] = {0.0f}; 
    VirtualToMoving3D_h.GetMyselfInversed(); 
    VirtualToMoving3D_h.GetMatrix(MovingToVirtualMatrix); 
    VirtualToFixed3D_h.GetMatrix(VirtualToFixedMatrix); 

    for(int row_it = 0; row_it < 4; ++row_it){
        for(int col_it = 0; col_it < 4; ++col_it){
            outMovingToFixed[col_it + row_it * 4] = 0.0f; 
            for(int element_it = 0; element_it < 4; ++element_it){
                outMovingToFixed[col_it + row_it * 4] += VirtualToFixedMatrix[element_it + row_it * 4] * MovingToVirtualMatrix[col_it + (element_it) * 4];
            }
        }
    }
}

void Interpolator2DTo3D::WriteOut(){
    // // //Copy out for debug. 
    // InterpFixedVolume_h = LC2_2D_class::Ultrasound; 
    // InterpMovingVolume_h = LC2_2D_class::HighResolution_static; 
    // writeToBin(thrust::raw_pointer_cast(InterpFixedVolume_h.data()), dimensionVirtual.x * dimensionVirtual.y * dimensionVirtual.z, FinalExportPath + "/fixed.raw"); 
    // writeToBin(thrust::raw_pointer_cast(InterpMovingVolume_h.data()), dimensionVirtual.x * dimensionVirtual.y * dimensionVirtual.z, FinalExportPath + "/moving.raw"); 

    // std::cout << "Final round results in virtual space are saved at: " << std::endl; 
    // std::cout << "          " << FinalExportPath << ", fixed.raw, moving.raw" << std::endl; 
}