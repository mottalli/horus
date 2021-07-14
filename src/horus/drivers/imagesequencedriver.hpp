#pragma once

#include "basedriver.hpp"

namespace horus
{


class ImageSequenceDriver : public BaseVideoDriver
{
public:
    ImageSequenceDriver(const std::string pathFormat, int minNumber, int maxNumber);
    virtual ~ImageSequenceDriver();

    virtual std::vector<VideoCamera> queryCameras();

protected:
    virtual ColorImage _captureFrame();
    virtual void _doInitialization();
    virtual void _doDestroy();

    std::string _pathFormat;
    int _minNumber, _maxNumber;
    int _lastNumber;
};

} // namespace horus
