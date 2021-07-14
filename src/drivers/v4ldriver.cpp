#include "v4ldriver.hpp"

namespace horus
{

V4LVideoDriver::V4LVideoDriver() :
    BaseVideoDriver()
{

}

V4LVideoDriver::~V4LVideoDriver()
{

}

std::vector<BaseVideoDriver::VideoCamera> V4LVideoDriver::queryCameras()
{
    std::vector<VideoCamera> res = { {0, "/dev/video0"} };
    return res;
}

void V4LVideoDriver::_doInitialization()
{
    _capture.open(this->_cameraID);
}

ColorImage V4LVideoDriver::_captureFrame()
{
    Mat frame;
    _capture >> frame;
    return frame;
}

void V4LVideoDriver::_doDestroy()
{
    _capture.release();
}

}   // namespace horus
