#include "volumeImporter.h"

#include <algorithm>

#include "itkImageFileReader.h"
#include "itkImageIOBase.h"
#include "itkObject.h"
#include "itkNrrdImageIOFactory.h"
#include "itkNiftiImageIOFactory.h"
#include "itkMINCImageIOFactory.h"
#include "itkMetaImageIOFactory.h"
#include "itkBMPImageIOFactory.h"
#include "itkPNGImageIOFactory.h"
#include "itkJPEGImageIOFactory.h"
#include "itkGDCMImageIO.h"
#include "itkGDCMImageIOFactory.h"
#include "itkGDCMSeriesFileNames.h"
#include "itkImageSeriesReader.h"
#include "itkOrientImageFilter.h"
#include "itkMetaDataObject.h"
#include "itkImageToVTKImageFilter.h"

VolumeImporter::VolumeImporter(){
    //ITK factories register: 
    itk::NrrdImageIOFactory::RegisterOneFactory(); 
    itk::NiftiImageIOFactory::RegisterOneFactory(); 
    itk::MINCImageIOFactory::RegisterOneFactory(); 
    itk::MetaImageIOFactory::RegisterOneFactory(); 
    itk::GDCMImageIOFactory::RegisterOneFactory(); 
    itk::BMPImageIOFactory::RegisterOneFactory(); 
    itk::PNGImageIOFactory::RegisterOneFactory(); 
    itk::JPEGImageIOFactory::RegisterOneFactory(); 

    itk::Object::SetGlobalWarningDisplay(false); 
}

void VolumeImporter::setFilePath(std::string _filePath){
    m_volumePath = _filePath; 
    m_mainBuffer.clear(); 
}

bool VolumeImporter::read(){
    using PixelType = float;
    using ImageType = itk::Image<PixelType, 3>; 

    itk::ImageFileReader<ImageType>::Pointer reader = itk::ImageFileReader<ImageType>::New(); 
    reader->SetFileName(m_volumePath); 
    try{
        reader->Update(); 
    }
    catch(itk::ExceptionObject & err){
        std::cout << "ERROR while parsing fixed volume: " << err.what() << std::endl; 
        return false; 
    }

    itk::OrientImageFilter<ImageType, ImageType>::Pointer orientationFilter = itk::OrientImageFilter<ImageType, ImageType>::New(); 
    orientationFilter->UseImageDirectionOn(); 
    orientationFilter->SetDesiredCoordinateOrientation(itk::SpatialOrientation::ITK_COORDINATE_ORIENTATION_RAI); 
    orientationFilter->SetInput(reader->GetOutput()); 
    try{
        orientationFilter->Update();
    }
    catch (itk::ExceptionObject & err){
        std::cout << "ERROR while correcting fixed volume orientation: " << err.what() << std::endl; 
        return false; 
    }

    //set parameter inside: 
    auto volumePtr = orientationFilter->GetOutput(); 
    
    m_dimension[0] = volumePtr->GetLargestPossibleRegion().GetSize()[0]; 
    m_dimension[1] = volumePtr->GetLargestPossibleRegion().GetSize()[1]; 
    m_dimension[2] = volumePtr->GetLargestPossibleRegion().GetSize()[2]; 

    m_spacing[0] = volumePtr->GetSpacing()[0]; 
    m_spacing[1] = volumePtr->GetSpacing()[1]; 
    m_spacing[2] = volumePtr->GetSpacing()[2]; 

    m_origin[0] = volumePtr->GetOrigin()[0]; 
    m_origin[1] = volumePtr->GetOrigin()[1]; 
    m_origin[2] = volumePtr->GetOrigin()[2]; 

    int bufferSize = volumePtr->GetPixelContainer()->Size(); 

    m_mainBuffer.resize(bufferSize, 0); 
    std::copy( 
        volumePtr->GetPixelContainer()->GetBufferPointer(), 
        volumePtr->GetPixelContainer()->GetBufferPointer() + bufferSize, 
        m_mainBuffer.data()
    ); 

    return true; 
}

void VolumeImporter::getDimension(int& _x, int& _y, int& _z) const{
    _x = m_dimension[0]; 
    _y = m_dimension[1]; 
    _z = m_dimension[2]; 
}

void VolumeImporter::getSpacing(float& _x, float& _y, float& _z) const{
    _x = m_spacing[0]; 
    _y = m_spacing[1]; 
    _z = m_spacing[2]; 
}

void VolumeImporter::getOrigin(float& _x, float& _y, float& _z) const{
    _x = m_origin[0]; 
    _y = m_origin[1]; 
    _z = m_origin[2]; 
}

