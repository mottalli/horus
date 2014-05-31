#pragma once
#include "basedriver.hpp"

// STD
#include <array>

// uEye API
#include <ueye.h>

namespace horus
{

class UEyeVideoDriver : public BaseVideoDriver
{
public:
    UEyeVideoDriver();
    virtual ~UEyeVideoDriver();

    virtual std::vector<VideoCamera> queryCameras();

protected:
    virtual ColorImage _captureFrame();
    virtual void       _doInitialization();
    virtual void       _doDestroy();
    void               _unlockCurrentBuffer();

    cv::VideoCapture   _capture;

    HIDS               _ueyeCamid;
    INT                _linesize;
    int                _currBufferId;
    int                _frameWidth, _frameHeight;

    struct ImageBuffer
    {
        char* data;
        INT   imageId;
    };
    std::array<ImageBuffer, 5> _imageBuffers;
};

}
