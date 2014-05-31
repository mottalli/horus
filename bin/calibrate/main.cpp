#include <iostream>
#include <functional>

//#include "../../src/drivers/v4ldriver.hpp"
#include "../../src/drivers/ueyedriver.hpp"

struct VideoProcessor
{
    horus::BaseVideoDriver& driver;

    VideoProcessor(horus::BaseVideoDriver& driver_) :
        driver(driver_)
    {
    }

    void processFrame(horus::ColorImage& frame)
    {
        cv::imshow("frame", frame);
        if ('q' == (char)cvWaitKey(40))
            driver.stopCaptureThread();
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

    driver.release();
}
