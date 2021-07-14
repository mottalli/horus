#pragma once

#include "basedriver.hpp"

namespace horus
{

class V4LVideoDriver : public BaseVideoDriver
{
public:
    V4LVideoDriver();
    virtual ~V4LVideoDriver();

    virtual std::vector<VideoCamera> queryCameras();

protected:
    virtual ColorImage _captureFrame();
    virtual void       _doInitialization();
    virtual void       _doDestroy();

    cv::VideoCapture   _capture;
};

}
