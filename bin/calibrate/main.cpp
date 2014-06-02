#include <iostream>
#include <functional>
#include <array>
#include <strstream>

//#include "../../src/drivers/v4ldriver.hpp"
#include "../../src/drivers/ueyedriver.hpp"
#include "../../src/eyedetect.h"
#include "../../src/horus.h"

struct VideoProcessor
{
    horus::BaseVideoDriver& driver;
    std::vector<horus::GrayscaleImage> _rgbChannels;

    horus::Decorator decorator;
    horus::VideoProcessor videoProcessor;

    VideoProcessor(horus::BaseVideoDriver& driver_) :
        driver(driver_), _rgbChannels(3)
    {
        //videoProcessor.parameters.doEyeDetect = true;
    }

    void processFrame(horus::ColorImage& frame)
    {
        assert(this->_rgbChannels.size() == 3);
        for (horus::GrayscaleImage& channel : this->_rgbChannels)
            channel.create(frame.size());

        // Extract only the RED channel from the RGB frame
        horus::GrayscaleImage out[] = { this->_rgbChannels[0], this->_rgbChannels[1], this->_rgbChannels[2] };
        int from_to[] = { 0,0, 1,1, 2,2 };
        cv::mixChannels(&frame, 1, out, 3, from_to, 3);

        horus::GrayscaleImage& image = this->_rgbChannels[0];
        horus::VideoProcessor::VideoStatus status = videoProcessor.processFrame(image);

        std::ostringstream strstatus;

        switch (status)
        {
        case horus::VideoProcessor::UNKNOWN_ERROR:
            strstatus << "UNKNOWN_ERROR"; break;
        case horus::VideoProcessor::UNPROCESSED:
            strstatus << "UNPROCESSED"; break;
        case horus::VideoProcessor::DEFOCUSED:
            strstatus << "DEFOCUSED"; break;
        case horus::VideoProcessor::INTERLACED:
            strstatus << "INTERLACED"; break;
        case horus::VideoProcessor::NO_EYE:
            strstatus << "NO_EYE"; break;
        case horus::VideoProcessor::FOCUSED_NO_IRIS:
            strstatus << "FOCUSED_NO_IRIS"; break;
        case horus::VideoProcessor::IRIS_LOW_QUALITY:
            strstatus << "IRIS_LOW_QUALITY"; break;
        case horus::VideoProcessor::IRIS_TOO_CLOSE:
            strstatus << "IRIS_TOO_CLOSE"; break;
        case horus::VideoProcessor::IRIS_TOO_FAR:
            strstatus << "IRIS_TOO_FAR"; break;
        case horus::VideoProcessor::FOCUSED_IRIS:
            strstatus << "FOCUSED_IRIS"; break;
        case horus::VideoProcessor::BAD_TEMPLATE:
            strstatus << "BAD_TEMPLATE"; break;
        case horus::VideoProcessor::FINISHED_CAPTURE:
            strstatus << "FINISHED_CAPTURE"; break;
        case horus::VideoProcessor::GOT_TEMPLATE:
            strstatus << "GOT_TEMPLATE"; break;
        }

        cv::putText(image, strstatus.str(), cv::Point(30, 30), cv::FONT_HERSHEY_DUPLEX, 1.0, CV_RGB(255,255,255));

        if (status >= horus::VideoProcessor::FOCUSED_NO_IRIS) {
            cv::rectangle(image, videoProcessor.eyeROI, CV_RGB(255,255,255), 1);
            decorator.drawSegmentationResult(image, videoProcessor.lastSegmentationResult);

            cv::imshow("equalized", videoProcessor.segmentator.pupilSegmentator.equalizedImage);
            cv::imshow("similarity", videoProcessor.segmentator.pupilSegmentator.similarityImage);
            cv::imshow("adjustmentRing", horus::tools::normalizeImage(videoProcessor.segmentator.pupilSegmentator.adjustmentRing));

        }

        cv::imshow("frame", image);

        char k = (char) cvWaitKey(20);
        switch (k) {
        case 's':
            videoProcessor.segmentator.pupilSegmentator.parameters.muPupil += 1;
            std::clog << "Mu pupil: " << videoProcessor.segmentator.pupilSegmentator.parameters.muPupil << std::endl;
            break;
        case 'a':
            videoProcessor.segmentator.pupilSegmentator.parameters.muPupil -= 1;
            std::clog << "Mu pupil: " << videoProcessor.segmentator.pupilSegmentator.parameters.muPupil << std::endl;
            break;
        case 'x':
            videoProcessor.segmentator.pupilSegmentator.parameters.sigmaPupil += 0.5;
            std::clog << "Sigma pupil: " << videoProcessor.segmentator.pupilSegmentator.parameters.sigmaPupil << std::endl;
            break;
        case 'z':
            videoProcessor.segmentator.pupilSegmentator.parameters.sigmaPupil -= 0.5;
            std::clog << "Sigma pupil: " << videoProcessor.segmentator.pupilSegmentator.parameters.sigmaPupil << std::endl;
            break;
        case 'q':
            driver.stopCaptureThread();
            break;
        default:
            break;
        }
    }
};


int main()
{
    //horus::V4LVideoDriver driver;
    horus::UEyeVideoDriver driver;
    VideoProcessor processor(driver);

    vector<horus::BaseVideoDriver::VideoCamera> cameras = driver.queryCameras();
    for (auto camera: cameras) {
        std::cout << camera.description << std::endl;
    }

    driver.initializeCamera(0);

    auto frameCallback = std::bind(&VideoProcessor::processFrame, &processor, std::placeholders::_1);

    std::thread captureThread = driver.startCaptureThread(frameCallback);
    captureThread.join();
}
