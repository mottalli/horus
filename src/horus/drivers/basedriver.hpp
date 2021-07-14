#pragma once

// STD
#include <vector>
#include <string>
#include <functional>
#include <thread>

// OpenCV
#include <opencv2/opencv.hpp>

// Boost
#include <boost/noncopyable.hpp>

#include "../types.h"

namespace horus
{

class BaseVideoDriver : boost::noncopyable
{
public:
    typedef int CameraID;
    typedef std::function<void(ColorImage&)> FrameCallback;

    BaseVideoDriver();
    virtual ~BaseVideoDriver();

    struct VideoCamera
    {
        CameraID    cameraID;
        std::string description;
    };

    virtual std::vector<VideoCamera> queryCameras() = 0;
    void        initializeCamera(CameraID cameraID);
    std::thread startCaptureThread(FrameCallback frameCallback);
    void        stopCaptureThread();
    void        release();

protected:

    virtual ColorImage _captureFrame() = 0;
    virtual void       _doInitialization() = 0;

    FrameCallback _frameCallback;
    ColorImage    _lastFrame;
    bool          _initialized;
    CameraID      _cameraID;
    bool          _stopCapture;
};

}  /* namespace horus */
