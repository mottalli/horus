#include "imagesequencedriver.hpp"

#include <boost/format.hpp>

namespace horus
{

ImageSequenceDriver::ImageSequenceDriver(const std::string pathFormat, int minNumber, int maxNumber) :
    _pathFormat(pathFormat), _minNumber(minNumber), _maxNumber(maxNumber), _lastNumber(minNumber-1)
{
}

ImageSequenceDriver::~ImageSequenceDriver()
{
}

std::vector<BaseVideoDriver::VideoCamera> ImageSequenceDriver::queryCameras()
{
    VideoCamera defaultCamera{0, "default"};
    return { defaultCamera };
}

ColorImage ImageSequenceDriver::_captureFrame()
{
    _lastNumber++;
    std::string filename = boost::str(boost::format(_pathFormat) % _lastNumber);
    return cv::imread(filename);
}

void ImageSequenceDriver::_doInitialization()
{
}

void ImageSequenceDriver::_doDestroy()
{
}

}   // namespace horus
