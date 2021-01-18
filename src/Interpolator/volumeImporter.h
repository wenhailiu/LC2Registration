#ifndef VOLUME_IMPORTER
#define VOLUME_IMPORTER

#include <iostream>
#include <array>
#include <string>
#include <vector>

class VolumeImporter{

public: 
    VolumeImporter(); 

    void setFilePath(std::string _filePath); 
    bool read(); 

    //getters: 
    const int getNumberOfDimensions() const { return m_numberOfDimensions; }
    void getDimension(int& _x, int& _y, int& _z) const; 
    void getSpacing(float& _x, float& _y, float& _z) const; 
    void getOrigin(float& _x, float& _y, float& _z) const; 
    const size_t getBufferSize() const { return m_mainBuffer.size(); }
    const float* getBufferPtr() const { return m_mainBuffer.data(); }

private: 

    //file info: 
    std::string m_volumePath; 

    //geometry information: 
    int m_numberOfDimensions; 
    std::array<int, 3> m_dimension; 
    std::array<float, 3> m_spacing; 
    std::array<float, 3> m_origin; 

    //buffer: 
    std::vector<float> m_mainBuffer; 

}; 

#endif