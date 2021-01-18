#ifndef SPATIALMAPPER_LC2
#define SPATIALMAPPER_LC2

#include <iostream>
#include <vector>
#include <fstream>

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

/*

Documentation: 
This header defines all the transformation between different spaces. 
Note that the core registration is happening in PHYSICAL space, the real world, with the unit of milimeter. 
First the origin image or volume have to be transformed from their pixel/voxel space to physical space. 

Secondly, pure physical space transform is performed, from one Physical space to another Physical space. 

*/ 

//Operators:
__host__ __device__ float2 operator-(float2 lPoint, float2 rPoint);
__host__ __device__ float3 operator-(float3 lPoint, float3 rPoint);
__host__ __device__ float2 operator+(float2 lPoint, float2 rPoint);
__host__ __device__ float3 operator+(float3 lPoint, float3 rPoint);
__host__ __device__ float3 operator*(float *rMatrix, float3 lPoint);
__host__ __device__ float2 operator*(float *rMatrix, float2 lPoint);


/*  
 ---------------------------------- class PhysioSpatial?DMapper: ---------------------------------------
Transform between the pixel/voxel space to physical space, unit transform: 
                        
                        **from pixel/voxel -> mm.** 

Inside of this class, spacing, origin and dimension have to be pre-defined, or read from external patameter file. 
Those parameters are necessary to construct the mapper. 

Center defines the geometricall center of the object image or volume in physical space in mm, 
and is calculated by member function SetCenter(); . 
Or, can be manually set by overloaded functions: SetCenter(*);.  

Spacing, origin, dimension are not allowed to be modified, but the center, can be defined arbitraily. 
Usually, the center is defined to the physical center. 

*/ 

class PhysioSpatial3DMapper{

public:
    __host__ __device__ PhysioSpatial3DMapper(){}
    __host__ PhysioSpatial3DMapper(float3 ext_spacing, float3 ext_origin, int3 ext_dimension){
        spacing = ext_spacing;
        origin = ext_origin;
        dimension = ext_dimension;

        //Initialize center: 
        center.x = origin.x + spacing.x * (float)dimension.x / 2.0f;
        center.y = origin.y + spacing.y * (float)dimension.y / 2.0f;
        center.z = origin.z + spacing.z * (float)dimension.z / 2.0f;
    }
    __host__ __device__ float3 ToLoc(float3 pxl) const { 
        float3 loc;  
        loc.x = pxl.x * spacing.x + origin.x;
        loc.y = pxl.y * spacing.y + origin.y;
        loc.z = pxl.z * spacing.z + origin.z;
        return loc;  
    }

    __host__ __device__ float3 ToLoc(float pxl_x, float pxl_y, float pxl_z) const { 
        float3 loc;  
        loc.x = pxl_x * spacing.x + origin.x;
        loc.y = pxl_y * spacing.y + origin.y;
        loc.z = pxl_z * spacing.z + origin.z;
        return loc;  
    }

    __host__ __device__ float3 ToPxl(float3 loc) const { 
        float3 pxl;  
        pxl.x = (loc.x - origin.x) / spacing.x;
        pxl.y = (loc.y - origin.y) / spacing.y;
        pxl.z = (loc.z - origin.z) / spacing.z;
        return pxl;  
    }

    __host__ __device__ float3 ToPxl(float loc_x, float loc_y, float loc_z) const { 
        float3 pxl;  
        pxl.x = (loc_x - origin.x) / spacing.x;
        pxl.y = (loc_y - origin.y) / spacing.y;
        pxl.z = (loc_z - origin.z) / spacing.z;
        return pxl;  
    }

    __host__ __device__ float3 GetSpacing() const { return spacing; }
    __host__ __device__ float3 GetOrigin() const { return origin; }
    __host__ __device__ float3 GetCenter() const { return center; }
    __host__ __device__ int3 GetDimension() const { return dimension; }

    __host__ void SetSpacing(float3 Spacing_ext){ spacing = Spacing_ext; }
    __host__ void SetOrigin(float3 Origin_ext){ origin = Origin_ext; }
    __host__ void SetDimension(int3 Dimension_ext){ dimension = Dimension_ext; }
    __host__ void SetCenter(){
        center.x = origin.x + spacing.x * (float)dimension.x / 2.0f;
        center.y = origin.y + spacing.y * (float)dimension.y / 2.0f;
        center.z = origin.z + spacing.z * (float)dimension.z / 2.0f;
    }

    __host__ void SetCenter(float3 center_extern){
        center = center_extern;
    }

    __host__ void SetCenter(float center_x, float center_y, float center_z){
        center.x = center_x;
        center.y = center_y;
        center.z = center_z;
    }

private:
    float3 spacing; 
    float3 origin;
    mutable float3 center;
    int3 dimension;
};

class PhysioSpatial2DMapper{

public:
    __host__ __device__ PhysioSpatial2DMapper(){}
    __host__ PhysioSpatial2DMapper(float2 ext_spacing, float2 ext_origin, int2 ext_dimension){
        spacing = ext_spacing;
        origin = ext_origin;
        dimension = ext_dimension;

        //Initialize center: 
        center.x = origin.x + spacing.x * (float)dimension.x / 2.0f;
        center.y = origin.y + spacing.y * (float)dimension.y / 2.0f;
    }
    __host__ __device__ float2 ToLoc(float2 pxl) const { 
        float2 loc;  
        loc.x = pxl.x * spacing.x + origin.x;
        loc.y = pxl.y * spacing.y + origin.y;
        return loc;  
    }

    __host__ __device__ float2 ToLoc(float pxl_x, float pxl_y) const { 
        float2 loc;  
        loc.x = pxl_x * spacing.x + origin.x;
        loc.y = pxl_y * spacing.y + origin.y;
        return loc;  
    }

    __host__ __device__ float2 ToPxl(float2 loc) const { 
        float2 pxl;  
        pxl.x = (loc.x - origin.x) / spacing.x;
        pxl.y = (loc.y - origin.y) / spacing.y;
        return pxl;  
    }

    __host__ __device__ float2 ToPxl(float loc_x, float loc_y) const { 
        float2 pxl;  
        pxl.x = (loc_x - origin.x) / spacing.x;
        pxl.y = (loc_y - origin.y) / spacing.y;
        return pxl;  
    }

    __host__ __device__ float2 GetSpacing() const { return spacing; }
    __host__ __device__ float2 GetOrigin() const { return origin; }
    __host__ __device__ float2 GetCenter() const { return center; }
    __host__ __device__ int2 GetDimension() const { return dimension; }

    __host__ void SetSpacing(float2 Spacing_ext){ spacing = Spacing_ext; }
    __host__ void SetOrigin(float2 Origin_ext){ origin = Origin_ext; }
    __host__ void SetDimension(int2 Dimension_ext){ dimension = Dimension_ext; }
    __host__ void SetCenter(){
        center.x = origin.x + spacing.x * (float)dimension.x / 2.0f;
        center.y = origin.y + spacing.y * (float)dimension.y / 2.0f;
    }

    __host__ void SetCenter(float2 center_extern){
        center = center_extern;
    }

    __host__ void SetCenter(float center_x, float center_y){
        center.x = center_x;
        center.y = center_y;
    }

private:
    float2 spacing; 
    float2 origin;
    mutable float2 center;
    int2 dimension;
};

/*  
 ---------------------------------- class Spatial?DMapper: ---------------------------------------
Transform from One physical space, to another physical space. Transform in physical spaces. 

It internally defines the transform matrix, initialized as identity matrix. 

    + RotCenter defines the rotational center. Transforma matrix, perform rigid, or affine transformation. 
    Rigid transformation translates and or rotates between two spaces. 
    Translations are relatively to the original position. 
    NOTE that the rotations are performed around the pre-defined RotCenter. 
    By default, RotCenter is set to the origin, zero usually. 

    + Shift defines additional translation between two space. 
    By default, shift is set to ZERO. 

Transformation can be seen as we perform the affine or rigid transformation on the Moving image, then map it to virtual space. 

Clockwise rotation, seen as anti-clockwise on the transform VirtualToMovingMapper. 

Virtual space now is seen as "Fixed" space. The only transformed space is the moving space. 

Moving image now is moving for the reference of the pre-set RotCenter, with additional shift. 

By using the pre-set shift, we force the moving space center to overlap with the virtual space center. 

Basic routine:
    1. construct: Spatial3DMapper(); 
    2. set Identity: Spatial3DMapper::Identity(); 
    3. set rotational center: Spatial3DMapper::SetCenter( MovingMapper.GetCenter() ); To rotate around itself physical center. 
    4. calculate the shift between Moving-itself-center and Reference center(normally the virtual space center). 
    5. generate rigid transform matrix: Spatial3DMapper::RigidInit(rotationAngle, translation); 
        Transform order: T_translation * T_backToOrigin * T_rotZ * T_rotY * T_rotX * T_ToCenter * T_shiftM_CenterToR_Center. 
    6. Copy from Host to Device Constant. 
*/ 

class Spatial3DMapper{

public:
    __host__ __device__ Spatial3DMapper(){ }

    __host__ __device__ void Identity(){
        for(int i = 0; i < 16; ++i){
            TransformMatrix[i] = 0.0f;
            TransformMatrix_init[i] = 0.0f; 
        }
        
        TransformMatrix[0] = 1.0f;
        TransformMatrix[5] = 1.0f;
        TransformMatrix[10] = 1.0f;
        TransformMatrix[15] = 1.0f;

        TransformMatrix_init[0] = 1.0f;
        TransformMatrix_init[5] = 1.0f;
        TransformMatrix_init[10] = 1.0f;
        TransformMatrix_init[15] = 1.0f;
    }

    __host__ void SetCenter(float3 ext_center){ RotCenter = ext_center; }
    __host__ void SetCenter(){ RotCenter.x = 0.0f; RotCenter.y = 0.0f; RotCenter.z = 0.0f; }
    __host__ float3 GetCenter(){ return RotCenter; }

    __host__ void SetShift(float3 ext_shift){ shift = ext_shift; }
    __host__ void SetShift(float ext_shift_x, float ext_shift_y, float ext_shift_z){ shift.x = ext_shift_x; shift.y = ext_shift_y; shift.z = ext_shift_z; }
    __host__ void SetShift(){ shift.x = 0.0f; shift.y = 0.0f; shift.z = 0.0f; }

    __host__ __device__ void SetMatrix(const float *ext_Matrix){ 
        for(int i = 0; i < 16; ++i){
            TransformMatrix[i] = ext_Matrix[i];
        }
    }

    __host__ __device__ void GetMatrix(float *ext_Matrix_out){
        for(int i = 0; i < 16; ++i){
            ext_Matrix_out[i] = TransformMatrix[i];
        }
    }

    __host__ __device__ void SetInitMatrix(const float *ext_Matrix){ 
        for(int i = 0; i < 16; ++i){
            TransformMatrix_init[i] = ext_Matrix[i];
        }
    }

    __host__ __device__ void GetInitMatrix(float *ext_Matrix_out){
        for(int i = 0; i < 16; ++i){
            ext_Matrix_out[i] = TransformMatrix_init[i];
        }
    }

    __host__ __device__ float* GetRawMatrix(){ return &TransformMatrix[0]; }

    __host__ __device__ float3 operator*(float3 lPoint){
        float3 retPoint;
        
        retPoint.x = 0;
        retPoint.x += (TransformMatrix[0 + 0 * 4] * lPoint.x);
        retPoint.x += (TransformMatrix[1 + 0 * 4] * lPoint.y);
        retPoint.x += (TransformMatrix[2 + 0 * 4] * lPoint.z);
        retPoint.x += (TransformMatrix[3 + 0 * 4] * 1.0f);

        retPoint.y = 0;
        retPoint.y += (TransformMatrix[0 + 1 * 4] * lPoint.x);
        retPoint.y += (TransformMatrix[1 + 1 * 4] * lPoint.y);
        retPoint.y += (TransformMatrix[2 + 1 * 4] * lPoint.z);
        retPoint.y += (TransformMatrix[3 + 1 * 4] * 1.0f);

        retPoint.z = 0;
        retPoint.z += (TransformMatrix[0 + 2 * 4] * lPoint.x);
        retPoint.z += (TransformMatrix[1 + 2 * 4] * lPoint.y);
        retPoint.z += (TransformMatrix[2 + 2 * 4] * lPoint.z);
        retPoint.z += (TransformMatrix[3 + 2 * 4] * 1.0f);

        return retPoint;
    }

    __host__ bool GetMyselfInversed(){
        float inv[16] = {0.0f}, det;
        inv[0] = TransformMatrix[5] * TransformMatrix[10] * TransformMatrix[15] - 
                TransformMatrix[5]  * TransformMatrix[11] * TransformMatrix[14] - 
                TransformMatrix[9]  * TransformMatrix[6]  * TransformMatrix[15] + 
                TransformMatrix[9]  * TransformMatrix[7]  * TransformMatrix[14] +
                TransformMatrix[13] * TransformMatrix[6]  * TransformMatrix[11] - 
                TransformMatrix[13] * TransformMatrix[7]  * TransformMatrix[10];

        inv[4] = -TransformMatrix[4]  * TransformMatrix[10] * TransformMatrix[15] + 
                TransformMatrix[4]  * TransformMatrix[11] * TransformMatrix[14] + 
                TransformMatrix[8]  * TransformMatrix[6]  * TransformMatrix[15] - 
                TransformMatrix[8]  * TransformMatrix[7]  * TransformMatrix[14] - 
                TransformMatrix[12] * TransformMatrix[6]  * TransformMatrix[11] + 
                TransformMatrix[12] * TransformMatrix[7]  * TransformMatrix[10];

        inv[8] = TransformMatrix[4]  * TransformMatrix[9] * TransformMatrix[15] - 
                TransformMatrix[4]  * TransformMatrix[11] * TransformMatrix[13] - 
                TransformMatrix[8]  * TransformMatrix[5] * TransformMatrix[15] + 
                TransformMatrix[8]  * TransformMatrix[7] * TransformMatrix[13] + 
                TransformMatrix[12] * TransformMatrix[5] * TransformMatrix[11] - 
                TransformMatrix[12] * TransformMatrix[7] * TransformMatrix[9];

        inv[12] = -TransformMatrix[4]  * TransformMatrix[9] * TransformMatrix[14] + 
                TransformMatrix[4]  * TransformMatrix[10] * TransformMatrix[13] +
                TransformMatrix[8]  * TransformMatrix[5] * TransformMatrix[14] - 
                TransformMatrix[8]  * TransformMatrix[6] * TransformMatrix[13] - 
                TransformMatrix[12] * TransformMatrix[5] * TransformMatrix[10] + 
                TransformMatrix[12] * TransformMatrix[6] * TransformMatrix[9];

        inv[1] = -TransformMatrix[1]  * TransformMatrix[10] * TransformMatrix[15] + 
                TransformMatrix[1]  * TransformMatrix[11] * TransformMatrix[14] + 
                TransformMatrix[9]  * TransformMatrix[2] * TransformMatrix[15] - 
                TransformMatrix[9]  * TransformMatrix[3] * TransformMatrix[14] - 
                TransformMatrix[13] * TransformMatrix[2] * TransformMatrix[11] + 
                TransformMatrix[13] * TransformMatrix[3] * TransformMatrix[10];

        inv[5] = TransformMatrix[0]  * TransformMatrix[10] * TransformMatrix[15] - 
                TransformMatrix[0]  * TransformMatrix[11] * TransformMatrix[14] - 
                TransformMatrix[8]  * TransformMatrix[2] * TransformMatrix[15] + 
                TransformMatrix[8]  * TransformMatrix[3] * TransformMatrix[14] + 
                TransformMatrix[12] * TransformMatrix[2] * TransformMatrix[11] - 
                TransformMatrix[12] * TransformMatrix[3] * TransformMatrix[10];

        inv[9] = -TransformMatrix[0]  * TransformMatrix[9] * TransformMatrix[15] + 
                TransformMatrix[0]  * TransformMatrix[11] * TransformMatrix[13] + 
                TransformMatrix[8]  * TransformMatrix[1] * TransformMatrix[15] - 
                TransformMatrix[8]  * TransformMatrix[3] * TransformMatrix[13] - 
                TransformMatrix[12] * TransformMatrix[1] * TransformMatrix[11] + 
                TransformMatrix[12] * TransformMatrix[3] * TransformMatrix[9];

        inv[13] = TransformMatrix[0]  * TransformMatrix[9] * TransformMatrix[14] - 
                TransformMatrix[0]  * TransformMatrix[10] * TransformMatrix[13] - 
                TransformMatrix[8]  * TransformMatrix[1] * TransformMatrix[14] + 
                TransformMatrix[8]  * TransformMatrix[2] * TransformMatrix[13] + 
                TransformMatrix[12] * TransformMatrix[1] * TransformMatrix[10] - 
                TransformMatrix[12] * TransformMatrix[2] * TransformMatrix[9];

        inv[2] = TransformMatrix[1]  * TransformMatrix[6] * TransformMatrix[15] - 
                TransformMatrix[1]  * TransformMatrix[7] * TransformMatrix[14] - 
                TransformMatrix[5]  * TransformMatrix[2] * TransformMatrix[15] + 
                TransformMatrix[5]  * TransformMatrix[3] * TransformMatrix[14] + 
                TransformMatrix[13] * TransformMatrix[2] * TransformMatrix[7] - 
                TransformMatrix[13] * TransformMatrix[3] * TransformMatrix[6];

        inv[6] = -TransformMatrix[0]  * TransformMatrix[6] * TransformMatrix[15] + 
                TransformMatrix[0]  * TransformMatrix[7] * TransformMatrix[14] + 
                TransformMatrix[4]  * TransformMatrix[2] * TransformMatrix[15] - 
                TransformMatrix[4]  * TransformMatrix[3] * TransformMatrix[14] - 
                TransformMatrix[12] * TransformMatrix[2] * TransformMatrix[7] + 
                TransformMatrix[12] * TransformMatrix[3] * TransformMatrix[6];

        inv[10] = TransformMatrix[0]  * TransformMatrix[5] * TransformMatrix[15] - 
                TransformMatrix[0]  * TransformMatrix[7] * TransformMatrix[13] - 
                TransformMatrix[4]  * TransformMatrix[1] * TransformMatrix[15] + 
                TransformMatrix[4]  * TransformMatrix[3] * TransformMatrix[13] + 
                TransformMatrix[12] * TransformMatrix[1] * TransformMatrix[7] - 
                TransformMatrix[12] * TransformMatrix[3] * TransformMatrix[5];

        inv[14] = -TransformMatrix[0]  * TransformMatrix[5] * TransformMatrix[14] + 
                TransformMatrix[0]  * TransformMatrix[6] * TransformMatrix[13] + 
                TransformMatrix[4]  * TransformMatrix[1] * TransformMatrix[14] - 
                TransformMatrix[4]  * TransformMatrix[2] * TransformMatrix[13] - 
                TransformMatrix[12] * TransformMatrix[1] * TransformMatrix[6] + 
                TransformMatrix[12] * TransformMatrix[2] * TransformMatrix[5];

        inv[3] = -TransformMatrix[1] * TransformMatrix[6] * TransformMatrix[11] + 
                TransformMatrix[1] * TransformMatrix[7] * TransformMatrix[10] + 
                TransformMatrix[5] * TransformMatrix[2] * TransformMatrix[11] - 
                TransformMatrix[5] * TransformMatrix[3] * TransformMatrix[10] - 
                TransformMatrix[9] * TransformMatrix[2] * TransformMatrix[7] + 
                TransformMatrix[9] * TransformMatrix[3] * TransformMatrix[6];

        inv[7] = TransformMatrix[0] * TransformMatrix[6] * TransformMatrix[11] - 
                TransformMatrix[0] * TransformMatrix[7] * TransformMatrix[10] - 
                TransformMatrix[4] * TransformMatrix[2] * TransformMatrix[11] + 
                TransformMatrix[4] * TransformMatrix[3] * TransformMatrix[10] + 
                TransformMatrix[8] * TransformMatrix[2] * TransformMatrix[7] - 
                TransformMatrix[8] * TransformMatrix[3] * TransformMatrix[6];

        inv[11] = -TransformMatrix[0] * TransformMatrix[5] * TransformMatrix[11] + 
                TransformMatrix[0] * TransformMatrix[7] * TransformMatrix[9] + 
                TransformMatrix[4] * TransformMatrix[1] * TransformMatrix[11] - 
                TransformMatrix[4] * TransformMatrix[3] * TransformMatrix[9] - 
                TransformMatrix[8] * TransformMatrix[1] * TransformMatrix[7] + 
                TransformMatrix[8] * TransformMatrix[3] * TransformMatrix[5];

        inv[15] = TransformMatrix[0] * TransformMatrix[5] * TransformMatrix[10] - 
                TransformMatrix[0] * TransformMatrix[6] * TransformMatrix[9] - 
                TransformMatrix[4] * TransformMatrix[1] * TransformMatrix[10] + 
                TransformMatrix[4] * TransformMatrix[2] * TransformMatrix[9] + 
                TransformMatrix[8] * TransformMatrix[1] * TransformMatrix[6] - 
                TransformMatrix[8] * TransformMatrix[2] * TransformMatrix[5];

        det = TransformMatrix[0] * inv[0] + TransformMatrix[1] * inv[4] + TransformMatrix[2] * inv[8] + TransformMatrix[3] * inv[12];

        if (det == 0)
            return false;

        det = 1.0f / det;

        for (int i = 0; i < 16; i++)
            TransformMatrix[i] = inv[i] * det;

        TransformMatrix[12] = 0.0f; 
        TransformMatrix[13] = 0.0f; 
        TransformMatrix[14] = 0.0f; 
        TransformMatrix[15] = 1.0f; 

        return true;
    }

    __host__ bool GetMyselfInversed(float _oMatrix[16]){
        float inv[16] = {0.0f}, det;
        inv[0] = TransformMatrix[5] * TransformMatrix[10] * TransformMatrix[15] - 
                TransformMatrix[5]  * TransformMatrix[11] * TransformMatrix[14] - 
                TransformMatrix[9]  * TransformMatrix[6]  * TransformMatrix[15] + 
                TransformMatrix[9]  * TransformMatrix[7]  * TransformMatrix[14] +
                TransformMatrix[13] * TransformMatrix[6]  * TransformMatrix[11] - 
                TransformMatrix[13] * TransformMatrix[7]  * TransformMatrix[10];

        inv[4] = -TransformMatrix[4]  * TransformMatrix[10] * TransformMatrix[15] + 
                TransformMatrix[4]  * TransformMatrix[11] * TransformMatrix[14] + 
                TransformMatrix[8]  * TransformMatrix[6]  * TransformMatrix[15] - 
                TransformMatrix[8]  * TransformMatrix[7]  * TransformMatrix[14] - 
                TransformMatrix[12] * TransformMatrix[6]  * TransformMatrix[11] + 
                TransformMatrix[12] * TransformMatrix[7]  * TransformMatrix[10];

        inv[8] = TransformMatrix[4]  * TransformMatrix[9] * TransformMatrix[15] - 
                TransformMatrix[4]  * TransformMatrix[11] * TransformMatrix[13] - 
                TransformMatrix[8]  * TransformMatrix[5] * TransformMatrix[15] + 
                TransformMatrix[8]  * TransformMatrix[7] * TransformMatrix[13] + 
                TransformMatrix[12] * TransformMatrix[5] * TransformMatrix[11] - 
                TransformMatrix[12] * TransformMatrix[7] * TransformMatrix[9];

        inv[12] = -TransformMatrix[4]  * TransformMatrix[9] * TransformMatrix[14] + 
                TransformMatrix[4]  * TransformMatrix[10] * TransformMatrix[13] +
                TransformMatrix[8]  * TransformMatrix[5] * TransformMatrix[14] - 
                TransformMatrix[8]  * TransformMatrix[6] * TransformMatrix[13] - 
                TransformMatrix[12] * TransformMatrix[5] * TransformMatrix[10] + 
                TransformMatrix[12] * TransformMatrix[6] * TransformMatrix[9];

        inv[1] = -TransformMatrix[1]  * TransformMatrix[10] * TransformMatrix[15] + 
                TransformMatrix[1]  * TransformMatrix[11] * TransformMatrix[14] + 
                TransformMatrix[9]  * TransformMatrix[2] * TransformMatrix[15] - 
                TransformMatrix[9]  * TransformMatrix[3] * TransformMatrix[14] - 
                TransformMatrix[13] * TransformMatrix[2] * TransformMatrix[11] + 
                TransformMatrix[13] * TransformMatrix[3] * TransformMatrix[10];

        inv[5] = TransformMatrix[0]  * TransformMatrix[10] * TransformMatrix[15] - 
                TransformMatrix[0]  * TransformMatrix[11] * TransformMatrix[14] - 
                TransformMatrix[8]  * TransformMatrix[2] * TransformMatrix[15] + 
                TransformMatrix[8]  * TransformMatrix[3] * TransformMatrix[14] + 
                TransformMatrix[12] * TransformMatrix[2] * TransformMatrix[11] - 
                TransformMatrix[12] * TransformMatrix[3] * TransformMatrix[10];

        inv[9] = -TransformMatrix[0]  * TransformMatrix[9] * TransformMatrix[15] + 
                TransformMatrix[0]  * TransformMatrix[11] * TransformMatrix[13] + 
                TransformMatrix[8]  * TransformMatrix[1] * TransformMatrix[15] - 
                TransformMatrix[8]  * TransformMatrix[3] * TransformMatrix[13] - 
                TransformMatrix[12] * TransformMatrix[1] * TransformMatrix[11] + 
                TransformMatrix[12] * TransformMatrix[3] * TransformMatrix[9];

        inv[13] = TransformMatrix[0]  * TransformMatrix[9] * TransformMatrix[14] - 
                TransformMatrix[0]  * TransformMatrix[10] * TransformMatrix[13] - 
                TransformMatrix[8]  * TransformMatrix[1] * TransformMatrix[14] + 
                TransformMatrix[8]  * TransformMatrix[2] * TransformMatrix[13] + 
                TransformMatrix[12] * TransformMatrix[1] * TransformMatrix[10] - 
                TransformMatrix[12] * TransformMatrix[2] * TransformMatrix[9];

        inv[2] = TransformMatrix[1]  * TransformMatrix[6] * TransformMatrix[15] - 
                TransformMatrix[1]  * TransformMatrix[7] * TransformMatrix[14] - 
                TransformMatrix[5]  * TransformMatrix[2] * TransformMatrix[15] + 
                TransformMatrix[5]  * TransformMatrix[3] * TransformMatrix[14] + 
                TransformMatrix[13] * TransformMatrix[2] * TransformMatrix[7] - 
                TransformMatrix[13] * TransformMatrix[3] * TransformMatrix[6];

        inv[6] = -TransformMatrix[0]  * TransformMatrix[6] * TransformMatrix[15] + 
                TransformMatrix[0]  * TransformMatrix[7] * TransformMatrix[14] + 
                TransformMatrix[4]  * TransformMatrix[2] * TransformMatrix[15] - 
                TransformMatrix[4]  * TransformMatrix[3] * TransformMatrix[14] - 
                TransformMatrix[12] * TransformMatrix[2] * TransformMatrix[7] + 
                TransformMatrix[12] * TransformMatrix[3] * TransformMatrix[6];

        inv[10] = TransformMatrix[0]  * TransformMatrix[5] * TransformMatrix[15] - 
                TransformMatrix[0]  * TransformMatrix[7] * TransformMatrix[13] - 
                TransformMatrix[4]  * TransformMatrix[1] * TransformMatrix[15] + 
                TransformMatrix[4]  * TransformMatrix[3] * TransformMatrix[13] + 
                TransformMatrix[12] * TransformMatrix[1] * TransformMatrix[7] - 
                TransformMatrix[12] * TransformMatrix[3] * TransformMatrix[5];

        inv[14] = -TransformMatrix[0]  * TransformMatrix[5] * TransformMatrix[14] + 
                TransformMatrix[0]  * TransformMatrix[6] * TransformMatrix[13] + 
                TransformMatrix[4]  * TransformMatrix[1] * TransformMatrix[14] - 
                TransformMatrix[4]  * TransformMatrix[2] * TransformMatrix[13] - 
                TransformMatrix[12] * TransformMatrix[1] * TransformMatrix[6] + 
                TransformMatrix[12] * TransformMatrix[2] * TransformMatrix[5];

        inv[3] = -TransformMatrix[1] * TransformMatrix[6] * TransformMatrix[11] + 
                TransformMatrix[1] * TransformMatrix[7] * TransformMatrix[10] + 
                TransformMatrix[5] * TransformMatrix[2] * TransformMatrix[11] - 
                TransformMatrix[5] * TransformMatrix[3] * TransformMatrix[10] - 
                TransformMatrix[9] * TransformMatrix[2] * TransformMatrix[7] + 
                TransformMatrix[9] * TransformMatrix[3] * TransformMatrix[6];

        inv[7] = TransformMatrix[0] * TransformMatrix[6] * TransformMatrix[11] - 
                TransformMatrix[0] * TransformMatrix[7] * TransformMatrix[10] - 
                TransformMatrix[4] * TransformMatrix[2] * TransformMatrix[11] + 
                TransformMatrix[4] * TransformMatrix[3] * TransformMatrix[10] + 
                TransformMatrix[8] * TransformMatrix[2] * TransformMatrix[7] - 
                TransformMatrix[8] * TransformMatrix[3] * TransformMatrix[6];

        inv[11] = -TransformMatrix[0] * TransformMatrix[5] * TransformMatrix[11] + 
                TransformMatrix[0] * TransformMatrix[7] * TransformMatrix[9] + 
                TransformMatrix[4] * TransformMatrix[1] * TransformMatrix[11] - 
                TransformMatrix[4] * TransformMatrix[3] * TransformMatrix[9] - 
                TransformMatrix[8] * TransformMatrix[1] * TransformMatrix[7] + 
                TransformMatrix[8] * TransformMatrix[3] * TransformMatrix[5];

        inv[15] = TransformMatrix[0] * TransformMatrix[5] * TransformMatrix[10] - 
                TransformMatrix[0] * TransformMatrix[6] * TransformMatrix[9] - 
                TransformMatrix[4] * TransformMatrix[1] * TransformMatrix[10] + 
                TransformMatrix[4] * TransformMatrix[2] * TransformMatrix[9] + 
                TransformMatrix[8] * TransformMatrix[1] * TransformMatrix[6] - 
                TransformMatrix[8] * TransformMatrix[2] * TransformMatrix[5];

        det = TransformMatrix[0] * inv[0] + TransformMatrix[1] * inv[4] + TransformMatrix[2] * inv[8] + TransformMatrix[3] * inv[12];

        if (det == 0)
            return false;

        det = 1.0f / det;

        for (int i = 0; i < 16; i++)
            _oMatrix[i] = inv[i] * det;

        _oMatrix[12] = 0.0f; 
        _oMatrix[13] = 0.0f; 
        _oMatrix[14] = 0.0f; 
        _oMatrix[15] = 1.0f; 

        return true;
    }

    __host__ void RigidInit(float3 RotationAngle, float3 Translation){
        float cosRx = (float)std::cos(RotationAngle.x * M_PI / 180.0f); float sinRx = (float)std::sin(RotationAngle.x * M_PI / 180.0f); 
        float cosRy = (float)std::cos(RotationAngle.y * M_PI / 180.0f); float sinRy = (float)std::sin(RotationAngle.y * M_PI / 180.0f); 
        float cosRz = (float)std::cos(RotationAngle.z * M_PI / 180.0f); float sinRz = (float)std::sin(RotationAngle.z * M_PI / 180.0f); 

        //transform order: Rotation X->Y->Z around volume center; Translation Tx, Ty, Tz, after rotation. 
        float transformMatrix[16] = {
            //Rotation - Row 1
            cosRz * cosRy, cosRz * sinRy * sinRx - sinRz * cosRx, cosRz * sinRy * cosRx + sinRz * sinRx, 
            //Translation - along X
            (cosRz * cosRy) * (- RotCenter.x) + 
            (cosRz * sinRy * sinRx - sinRz * cosRx) * (- RotCenter.y) + 
            (cosRz * sinRy * cosRx + sinRz * sinRx) * (- RotCenter.z) + 
            RotCenter.x + 
            Translation.x + 
            shift.x * cosRy * cosRz - shift.y * (cosRx*sinRz - cosRz*sinRx*sinRy) + shift.z * (sinRx*sinRz + cosRx*cosRz*sinRy), 
            //Rotation - Row 2
            sinRz * cosRy, sinRz * sinRy * sinRx + cosRz * cosRx, sinRz * sinRy * cosRx - cosRz * sinRx, 
            //Translation - along Y
            (sinRz * cosRy) * (- RotCenter.x) + 
            (sinRz * sinRy * sinRx + cosRz * cosRx) * (- RotCenter.y) + 
            (sinRz * sinRy * cosRx - cosRz * sinRx) * (- RotCenter.z) + 
            RotCenter.y + 
            Translation.y + 
            shift.x * cosRy * sinRz + shift.y * (cosRx * cosRz + sinRx * sinRy * sinRz) - shift.z * (cosRz * sinRx - cosRx * sinRy * sinRz), 
            //Rotation - Row 3
            -sinRy, cosRy * sinRx, cosRy * cosRx, 
            //Translation - along Z
            (-sinRy) * (- RotCenter.x) + 
            (cosRy * sinRx) * (- RotCenter.y) + 
            (cosRy * cosRx) * (- RotCenter.z) + 
            RotCenter.z + 
            Translation.z + 
            - shift.x * sinRy + shift.y * cosRy * sinRx + shift.z * cosRx * cosRy, 
            0.0f, 0.0f, 0.0f, 1.0f
        };

        for(int row_it = 0; row_it < 4; ++row_it){
            for(int col_it = 0; col_it < 4; ++col_it){
                TransformMatrix[col_it + row_it * 4] = 0.0f; 
                for(int element_it = 0; element_it < 4; ++element_it){
                    TransformMatrix[col_it + row_it * 4] += transformMatrix[element_it + row_it * 4] * TransformMatrix_init[col_it + (element_it) * 4];
                }
            }
        }
    }

    static void MatrixMultiplication(const float _a[16], const float _b[16], float _c[16]){
        for(int row_it = 0; row_it < 4; ++row_it){
            for(int col_it = 0; col_it < 4; ++col_it){
                _c[col_it + row_it * 4] = 0.0f; 
                for(int element_it = 0; element_it < 4; ++element_it){
                    _c[col_it + row_it * 4] += _a[element_it + row_it * 4] * _b[col_it + (element_it) * 4];
                }
            }
        }
    }

    __host__ void AffineInit(float3 RotationAngle, float3 Translation, float3 Scale, float3 Shear){
        float cosRx = (float)std::cos(RotationAngle.x * M_PI / 180.0f); float sinRx = (float)std::sin(RotationAngle.x * M_PI / 180.0f); 
        float cosRy = (float)std::cos(RotationAngle.y * M_PI / 180.0f); float sinRy = (float)std::sin(RotationAngle.y * M_PI / 180.0f); 
        float cosRz = (float)std::cos(RotationAngle.z * M_PI / 180.0f); float sinRz = (float)std::sin(RotationAngle.z * M_PI / 180.0f); 

        // //Final result: 
        // float transformMatrix[16] = { 
        //     1, 0, 0, 0, 
        //     0, 1, 0, 0, 
        //     0, 0, 1, 0, 
        //     0, 0, 0, 1
        // }; 
        // //shift: 
        // {
        //     float _shiftMatrix[16] = {
        //         1, 0, 0, shift.x, 
        //         0, 1, 0, shift.y, 
        //         0, 0, 1, shift.z, 
        //         0, 0, 0, 1
        //     };
            
        //     for (int i = 0; i < 16; ++i){
        //         transformMatrix[i] = _shiftMatrix[i]; 
        //     }
        // }

        // //Corner to centre: 
        // {
        //     float _corner2centre[16] = {
        //         1, 0, 0, -RotCenter.x, 
        //         0, 1, 0, -RotCenter.y, 
        //         0, 0, 1, -RotCenter.z, 
        //         0, 0, 0, 1
        //     };

        //     float _resultsMatrix[16] = {0}; 
        //     MatrixMultiplication(_corner2centre, transformMatrix, _resultsMatrix); 

        //     for (int i = 0; i < 16; ++i){
        //         transformMatrix[i] = _resultsMatrix[i]; 
        //     }
        // }

        // //scale: 
        // {
        //     float _scaleMatrix[16] = {
        //         Scale.x, 0, 0, 0, 
        //         0, Scale.y, 0, 0, 
        //         0, 0, Scale.z, 0, 
        //         0, 0, 0, 1
        //     };

        //     float _resultsMatrix[16] = {0}; 
        //     MatrixMultiplication(_scaleMatrix, transformMatrix, _resultsMatrix); 

        //     for (int i = 0; i < 16; ++i){
        //         transformMatrix[i] = _resultsMatrix[i]; 
        //     }
        // }

        // //shear: 
        // {
        //     float _shearMatrix[16] = {
        //         1, Shear.x, Shear.y, 0, 
        //         0, 1, Shear.z, 0, 
        //         0, 0, 1, 0, 
        //         0, 0, 0, 1
        //     };

        //     float _resultsMatrix[16] = {0}; 
        //     MatrixMultiplication(_shearMatrix, transformMatrix, _resultsMatrix); 

        //     for (int i = 0; i < 16; ++i){
        //         transformMatrix[i] = _resultsMatrix[i]; 
        //     }
        // }

        // //rotX: 
        // {
        //     float _rotXMatrix[16] = {
        //         1, 0, 0, 0, 
        //         0, cosRx, -sinRx, 0, 
        //         0, sinRx, cosRx, 0, 
        //         0, 0, 0, 1
        //     };

        //     float _resultsMatrix[16] = {0}; 
        //     MatrixMultiplication(_rotXMatrix, transformMatrix, _resultsMatrix); 

        //     for (int i = 0; i < 16; ++i){
        //         transformMatrix[i] = _resultsMatrix[i]; 
        //     }
        // }

        // //rotY: 
        // {
        //     float _rotYMatrix[16] = {
        //         cosRy, 0, sinRy, 0, 
        //         0, 1, 0, 0, 
        //         -sinRy, 0, cosRy, 0, 
        //         0, 0, 0, 1
        //     };

        //     float _resultsMatrix[16] = {0}; 
        //     MatrixMultiplication(_rotYMatrix, transformMatrix, _resultsMatrix); 

        //     for (int i = 0; i < 16; ++i){
        //         transformMatrix[i] = _resultsMatrix[i]; 
        //     }
        // }

        // //rotY: 
        // {
        //     float _rotZMatrix[16] = {
        //         cosRz, -sinRz, 0, 0, 
        //         sinRz, cosRz, 0, 0, 
        //         0, 0, 1, 0, 
        //         0, 0, 0, 1
        //     };

        //     float _resultsMatrix[16] = {0}; 
        //     MatrixMultiplication(_rotZMatrix, transformMatrix, _resultsMatrix); 

        //     for (int i = 0; i < 16; ++i){
        //         transformMatrix[i] = _resultsMatrix[i]; 
        //     }
        // }

        // //centre to corner: 
        // {
        //     float _centre2corner[16] = {
        //         1, 0, 0, RotCenter.x, 
        //         0, 1, 0, RotCenter.y, 
        //         0, 0, 1, RotCenter.z, 
        //         0, 0, 0, 1
        //     };

        //     float _resultsMatrix[16] = {0}; 
        //     MatrixMultiplication(_centre2corner, transformMatrix, _resultsMatrix); 

        //     for (int i = 0; i < 16; ++i){
        //         transformMatrix[i] = _resultsMatrix[i]; 
        //     }
        // }

        // //translation: 
        // {
        //     float _translation[16] = {
        //         1, 0, 0, Translation.x, 
        //         0, 1, 0, Translation.y, 
        //         0, 0, 1, Translation.z, 
        //         0, 0, 0, 1
        //     };

        //     float _resultsMatrix[16] = {0}; 
        //     MatrixMultiplication(_translation, transformMatrix, _resultsMatrix); 

        //     for (int i = 0; i < 16; ++i){
        //         transformMatrix[i] = _resultsMatrix[i]; 
        //     }
        // }

        //transform order: Rotation X->Y->Z around volume center; Translation Tx, Ty, Tz, after rotation. 
        float transformMatrix[16] = {
            //Rotation - Row 1
            Scale.x * cosRz * cosRy, 
            Scale.y * (Shear.x * cosRy * cosRz - cosRx * sinRz + cosRz * sinRx * sinRy), 
            Scale.z * (Shear.y * cosRy * cosRz + Shear.z * (-cosRx * sinRz + cosRz * sinRx * sinRy) + cosRx * cosRz * sinRy + sinRx * sinRz), 
            //Translation - along X
            -RotCenter.x * Scale.x * cosRy * cosRz + RotCenter.x - 
            RotCenter.y * Scale.y * (Shear.x * cosRy * cosRz - cosRx * sinRz + cosRz * sinRx * sinRy) - 
            RotCenter.z * Scale.z * (Shear.y * cosRy * cosRz + Shear.z * (-cosRx * sinRz + cosRz * sinRx * sinRy) + cosRx * cosRz * sinRy + sinRx * sinRz) + 
            Scale.x * shift.x * cosRy * cosRz + 
            Scale.y * shift.y * (Shear.x * cosRy * cosRz - cosRx * sinRz + cosRz * sinRx * sinRy) + 
            Scale.z * shift.z*(Shear.y * cosRy * cosRz + Shear.z * (-cosRx * sinRz + cosRz * sinRx * sinRy) + cosRx * cosRz * sinRy + sinRx * sinRz) + 
            Translation.x, 
            //Rotation - Row 2
            Scale.x * cosRy * sinRz, 
            Scale.y * (Shear.x * cosRy * sinRz + cosRx*cosRz + sinRx*sinRy*sinRz), 
            Scale.z * (Shear.y * cosRy * sinRz + Shear.z * (cosRx * cosRz + sinRx * sinRy * sinRz) + cosRx * sinRy * sinRz - cosRz * sinRx),
            //Translation - along Y
            -RotCenter.x * Scale.x * cosRy * sinRz - 
            RotCenter.y * Scale.y * (Shear.x * cosRy * sinRz + cosRx * cosRz + sinRx * sinRy * sinRz) + 
            RotCenter.y - 
            RotCenter.z * Scale.z * (Shear.y * cosRy * sinRz + Shear.z * (cosRx * cosRz + sinRx * sinRy * sinRz) + cosRx * sinRy * sinRz - cosRz * sinRx) + 
            Scale.x * shift.x * cosRy * sinRz + 
            Scale.y * shift.y * (Shear.x * cosRy * sinRz + cosRx * cosRz + sinRx * sinRy * sinRz) + 
            Scale.z * shift.z * (Shear.y * cosRy * sinRz + Shear.z * (cosRx * cosRz + sinRx * sinRy * sinRz) + cosRx * sinRy * sinRz - cosRz * sinRx) + 
            Translation.y, 
            //Rotation - Row 3
            -Scale.x * sinRy, 
            Scale.y * (-Shear.x * sinRy + cosRy * sinRx), 
            Scale.z * (-Shear.y * sinRy + Shear.z * cosRy * sinRx + cosRx * cosRy), 
            //Translation - along Z
            RotCenter.x * Scale.x * sinRy - 
            RotCenter.y * Scale.y * (-Shear.x * sinRy + cosRy * sinRx) - 
            RotCenter.z * Scale.z * (-Shear.y * sinRy + Shear.z * cosRy * sinRx + cosRx * cosRy) + 
            RotCenter.z - 
            Scale.x * shift.x * sinRy + 
            Scale.y * shift.y * (-Shear.x * sinRy + cosRy * sinRx) + 
            Scale.z * shift.z * (-Shear.y * sinRy + Shear.z * cosRy * sinRx + cosRx * cosRy) + 
            Translation.z, 
            0.0f, 0.0f, 0.0f, 1.0f
        };

        for(int row_it = 0; row_it < 4; ++row_it){
            for(int col_it = 0; col_it < 4; ++col_it){
                TransformMatrix[col_it + row_it * 4] = 0.0f; 
                for(int element_it = 0; element_it < 4; ++element_it){
                    TransformMatrix[col_it + row_it * 4] += transformMatrix[element_it + row_it * 4] * TransformMatrix_init[col_it + (element_it) * 4];
                }
            }
        }
    }

    __host__ void ApplyAdditionalMatrix(float* externMatrix){
        float tmp_results[16]; 
        for(int row_it = 0; row_it < 4; ++row_it){
            for(int col_it = 0; col_it < 4; ++col_it){
                tmp_results[col_it + row_it * 4] = 0.0f; 
                for(int element_it = 0; element_it < 4; ++element_it){
                    tmp_results[col_it + row_it * 4] += externMatrix[element_it + row_it * 4] * TransformMatrix[col_it + (element_it) * 4];
                }
            }
        }

        //overwrite: 
        for(int i = 0; i < 16; ++i){
            TransformMatrix[i] = tmp_results[i]; 
        }
    }

private: 
    float TransformMatrix[16];
    float3 RotCenter; // in physical space. 
    float3 shift; // shift: difference between the center of Moving image, and center of Virtual image. 
    float TransformMatrix_init[16];
};

class Spatial2DMapper{

public:
    __host__ __device__ Spatial2DMapper(){ }

    __host__ __device__ void Identity(){
        for(int i = 0; i < 9; ++i){
            TransformMatrix[i] = 0.0f;
            TransformMatrix_init[i] = 0.0f;
        }
        TransformMatrix[0] = 1.0f;
        TransformMatrix[4] = 1.0f;
        TransformMatrix[8] = 1.0f;

        TransformMatrix_init[0] = 1.0f;
        TransformMatrix_init[4] = 1.0f;
        TransformMatrix_init[8] = 1.0f;
    }

    __host__ void SetCenter(float2 ext_center){ RotCenter = ext_center; }
    __host__ void SetCenter(float ext_center_x, float ext_center_y){ RotCenter.x = ext_center_x; RotCenter.y = ext_center_y; }
    __host__ void SetCenter(){ RotCenter.x = 0.0f; RotCenter.y = 0.0f; }
    __host__ float2 GetCenter(){ return RotCenter; }

    __host__ void SetShift(float2 ext_shift){ shift = ext_shift; }
    __host__ void SetShift(float ext_shift_x, float ext_shift_y){ shift.x = ext_shift_x; shift.y = ext_shift_y; }
    __host__ void SetShift(){ shift.x = 0.0f; shift.y = 0.0f; }

    __host__ __device__ void SetMatrix(const float *ext_Matrix){ 
        for(int i = 0; i < 9; ++i){
            TransformMatrix[i] = ext_Matrix[i];
        }
    }

    __host__ __device__ void GetMatrix(float *ext_Matrix_out){
        for(int i = 0; i < 9; ++i){
            ext_Matrix_out[i] = TransformMatrix[i];
        }
    }

    __host__ __device__ void SetInitMatrix(const float *ext_Matrix){ 
        for(int i = 0; i < 9; ++i){
            TransformMatrix_init[i] = ext_Matrix[i];
        }
    }

    __host__ __device__ void GetInitMatrix(float *ext_Matrix_out){
        for(int i = 0; i < 9; ++i){
            ext_Matrix_out[i] = TransformMatrix_init[i];
        }
    }

    __host__ __device__ float* GetRawMatrix(){ return &TransformMatrix[0]; }

    __host__ __device__ float2 operator*(float2 lPoint){
        float2 retPoint;
        
        retPoint.x = 0;
        retPoint.x += (TransformMatrix[0 + 0 * 3] * lPoint.x);
        retPoint.x += (TransformMatrix[1 + 0 * 3] * lPoint.y);
        retPoint.x += (TransformMatrix[2 + 0 * 3] * 1.0f);

        retPoint.y = 0;
        retPoint.y += (TransformMatrix[0 + 1 * 3] * lPoint.x);
        retPoint.y += (TransformMatrix[1 + 1 * 3] * lPoint.y);
        retPoint.y += (TransformMatrix[2 + 1 * 3] * 1.0f);

        return retPoint;
    }

    __host__ bool GetMyselfInversed(){
        float inv[16] = {0.0f}, det;
        inv[0] = TransformMatrix[4] * TransformMatrix[8] - TransformMatrix[7] * TransformMatrix[5];
        inv[1] = TransformMatrix[2] * TransformMatrix[7] - TransformMatrix[8] * TransformMatrix[1];
        inv[2] = TransformMatrix[1] * TransformMatrix[5] - TransformMatrix[2] * TransformMatrix[4];

        inv[3] = TransformMatrix[5] * TransformMatrix[6] - TransformMatrix[3] * TransformMatrix[8];
        inv[4] = TransformMatrix[0] * TransformMatrix[8] - TransformMatrix[2] * TransformMatrix[6];
        inv[5] = TransformMatrix[2] * TransformMatrix[3] - TransformMatrix[0] * TransformMatrix[5];

        inv[6] = TransformMatrix[3] * TransformMatrix[7] - TransformMatrix[6] * TransformMatrix[4];
        inv[7] = TransformMatrix[1] * TransformMatrix[6] - TransformMatrix[0] * TransformMatrix[7];
        inv[8] = TransformMatrix[0] * TransformMatrix[4] - TransformMatrix[1] * TransformMatrix[3];

        det = TransformMatrix[0] * inv[0] + TransformMatrix[1] * inv[3] + TransformMatrix[2] * inv[6];

        if (det == 0)
            return false;

        det = 1.0f / det;

        for (int i = 0; i < 9; i++)
            TransformMatrix[i] = inv[i] * det;

        TransformMatrix[6] = 0.0f; 
        TransformMatrix[7] = 0.0f; 
        TransformMatrix[8] = 1.0f; 

        return true;
    }

    __host__ void RigidInit(float RotationAngle, float2 Translation){
        float cosR = (float)std::cos(RotationAngle * M_PI / 180.0f); float sinR = (float)std::sin(RotationAngle * M_PI / 180.0f); 
        
        //transform order: Rotation X->Y around image center; Translation Tx, Ty, after rotation. 
        float transformMatrix[9] = {
            cosR, -sinR, 
            - (RotCenter.x *  cosR) + (RotCenter.y *  sinR) + RotCenter.x + Translation.x + cosR * shift.x - shift.y * sinR, 
            sinR, cosR, 
            - (RotCenter.x *  sinR) - (RotCenter.y *  cosR) + RotCenter.y + Translation.y + cosR * shift.y + shift.x * sinR, 
            0.0f, 0.0f, 1.0f
        };

        for(int row_it = 0; row_it < 3; ++row_it){
            for(int col_it = 0; col_it < 3; ++col_it){
                TransformMatrix[col_it + row_it * 3] = 0.0f; 
                for(int element_it = 0; element_it < 3; ++element_it){
                    TransformMatrix[col_it + row_it * 3] += transformMatrix[element_it + row_it * 3] * TransformMatrix_init[col_it + (element_it) * 3];
                }
            }
        }
    }

    __host__ void AffineInit(float RotationAngle, float2 Translation, float2 Scale, float Shear){
        float cosR = (float)std::cos(RotationAngle * M_PI / 180.0f); float sinR = (float)std::sin(RotationAngle * M_PI / 180.0f); 
        
        //transform order: Rotation X->Y around image center; Translation Tx, Ty, after rotation. 
        float transformMatrix[9] = {
            //Rotation: 
            Scale.x * cosR, 
            -Scale.y * (sinR - Shear * cosR), 
            //Translation: 
            RotCenter.x + Translation.x - 
            RotCenter.x * Scale.x * cosR + 
            RotCenter.y * Scale.y * (sinR - Shear * cosR) + 
            Scale.x * shift.x * cosR - 
            Scale.y * shift.y * (sinR - Shear * cosR), 
            //Rotation: 
            Scale.x * sinR, 
            Scale.y * (cosR + Shear * sinR), 
            //Translation: 
            RotCenter.y + Translation.y - 
            RotCenter.x * Scale.x * sinR - 
            RotCenter.y * Scale.y * (cosR + Shear * sinR) + 
            Scale.x * shift.x * sinR + 
            Scale.y * shift.y * (cosR + Shear * sinR), 
            0.0f, 0.0f, 1.0f
        };

        for(int row_it = 0; row_it < 3; ++row_it){
            for(int col_it = 0; col_it < 3; ++col_it){
                TransformMatrix[col_it + row_it * 3] = 0.0f; 
                for(int element_it = 0; element_it < 3; ++element_it){
                    TransformMatrix[col_it + row_it * 3] += transformMatrix[element_it + row_it * 3] * TransformMatrix_init[col_it + (element_it) * 3];
                }
            }
        }
    }

    __host__ void ApplyAdditionalMatrix(float* externMatrix){
        float tmp_results[9]; 
        for(int row_it = 0; row_it < 3; ++row_it){
            for(int col_it = 0; col_it < 3; ++col_it){
                tmp_results[col_it + row_it * 3] = 0.0f; 
                for(int element_it = 0; element_it < 3; ++element_it){
                    tmp_results[col_it + row_it * 3] += externMatrix[element_it + row_it * 3] * TransformMatrix[col_it + (element_it) * 3];
                }
            }
        }

        //overwrite: 
        for(int i = 0; i < 9; ++i){
            TransformMatrix[i] = tmp_results[i]; 
        }
    }

private: 
    float TransformMatrix[9];
    float2 RotCenter; // In physical space
    float2 shift;
    float TransformMatrix_init[9];
};

#endif