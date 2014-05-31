#include "basedriver.hpp"

namespace horus
{

BaseVideoDriver::BaseVideoDriver() :
    _cameraID(-1), _initialized(false)
{

}

BaseVideoDriver::~BaseVideoDriver()
{
}

void BaseVideoDriver::initializeCamera(CameraID cameraID)
{
    this->_cameraID = cameraID;
    this->_doInitialization();
    this->_initialized = true;
}

std::thread BaseVideoDriver::startCaptureThread(FrameCallback frameCallback)
{
    this->_frameCallback = frameCallback;

    auto captureThread = [this]() {
        this->_stopCapture = false;

        while (!this->_stopCapture) {
            ColorImage frame = this->_captureFrame();
            frame.copyTo(this->_lastFrame);
            this->_frameCallback(this->_lastFrame);
        }
    };

    return std::thread(captureThread);
}

void BaseVideoDriver::stopCaptureThread()
{
    this->_stopCapture = true;
}

void BaseVideoDriver::release()
{
    this->_doDestroy();
}

} // namespace horus
