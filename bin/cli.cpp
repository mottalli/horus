// STD
#include <iostream>
#include <stdlib.h>
#include <sstream>
#include <functional>

// OpenCV
#include <opencv2/opencv.hpp>

// Horus
#include <horus/horus.h>

struct ROIGetter
{
    cv::Mat preview;
    cv::Mat tmp;
    cv::Point pdown, pup;
    const cv::Point NO_POINT;
    bool isDown;

    ROIGetter() :
        NO_POINT(-1, -1)
    {
        pdown = pup = NO_POINT;
        isDown = false;
    }

    cv::Mat getROI(const cv::Mat& img, float previewFactor)
    {
        cv::resize(img, preview, cv::Size(), previewFactor, previewFactor);
        cv::imshow("preview", preview);
        cv::setMouseCallback("preview", &mouseCallback, this);

        bool skip = false;
        while (true) {
            char k = char(cv::waitKey(20));
            if (k == 'x') {
                skip = true;
                break;
            }

            if (pdown != NO_POINT && pup != NO_POINT)
                break;
        }

        cv::destroyWindow("preview");

        if (skip)
            return cv::Mat();

        pdown = cv::Point( float(pdown.x)/previewFactor, float(pdown.y)/previewFactor );
        pup = cv::Point( float(pup.x)/previewFactor, float(pup.y)/previewFactor );

        cv::Rect rect(pdown, pup);
        return img(rect).clone();

    }

    static void mouseCallback(int event, int x, int y, int flags, void* userdata)
    {
        ROIGetter* getter = (ROIGetter*) userdata;
        if (event == cv::EVENT_LBUTTONDOWN) {
            getter->pdown = cv::Point(x, y);
            getter->isDown = true;
        } else if (event == cv::EVENT_LBUTTONUP) {
            getter->isDown = false;
            getter->pup = cv::Point(x, y);
        } else if (getter->pdown != getter->NO_POINT && getter->isDown) {
            cv::Mat tmp = getter->preview.clone();
            cv::rectangle(tmp, getter->pdown, cv::Point(x, y), cv::Scalar(0,0,0));
            cv::imshow("preview", tmp);
        }
    }
};

int main(int argc, char** argv)
{
    if (argc == 1) {
        std::cerr << "Usage: " << argv[0] << " [file1] [file2] ... [fileN]" << std::endl;
        exit(1);
    }

    std::vector<cv::Mat> images;

    for (int i = 1; i < argc; i++) {
        const char* filename = argv[i];
        std::cerr << "Reading " << filename << "..." << std::endl;

        cv::Mat img = cv::imread(filename);
        float resizeFactor = 0.3f;

        ROIGetter roiGetter;
        cv::Mat roi = roiGetter.getROI(img, resizeFactor).clone();
        if (roi.empty())
            continue;

        images.push_back(roi);
    }

    horus::Decorator decorator;
    horus::Segmentator segmentator;

    horus::PupilSegmentatorParameters pupilParameters;
    pupilParameters.muPupil = 0.0;
    pupilParameters.sigmaPupil = 7.0;
    segmentator.pupilSegmentator.parameters = pupilParameters;

    horus::LogGaborEncoder encoder;

    std::vector<horus::IrisTemplate> templates;
    std::vector<horus::SegmentationResult> segmentationResults;
    //for (cv::Mat image : images) {
    for (unsigned i = 0; i < images.size(); i++) {
        cv::Mat image = images[i];
        horus::SegmentationResult segmentationResult = segmentator.segmentImage(image);
        segmentationResults.push_back(segmentationResult);

        templates.push_back(encoder.generateTemplate(image, segmentationResult));

        /*std::ostringstream winname;
        winname << "seg" << (i+1);
        cv::imshow(winname.str(), segmentator.pupilSegmentator.similarityImage);*/

    }

    int width = 7;
    int precision = 2;
    std::cout << std::setw(width) << ' ';
    for (unsigned i = 0; i < templates.size(); i++) {
        std::cout << std::setw(width) << i+1;
    }
    std::cout << std::endl;

    const std::string COLOR_RED = "\033[0;31m";
    const std::string COLOR_GREEN = "\033[0;32m";
    const std::string COLOR_NORMAL = "\033[0m";

    for (unsigned i = 0; i < templates.size(); i++) {
        std::cout << std::setw(width) << (i+1);

        horus::TemplateComparator comparator(templates[i]);
        for ( unsigned j = 0; j <= i; j++ ) {
            std::cout << std::setw(width) << ' ';
        }

        for ( unsigned j = i+1; j < templates.size(); j++ ) {
            double comp = comparator.compare(templates[j]);
            std::string color = (comp < 0.32f) ? COLOR_GREEN : COLOR_RED;

            std::cout << color << std::setprecision(precision) << std::setw(width) << comp << COLOR_NORMAL;
        }
        std::cout << std::endl;

        cv::Mat image = images[i];
        decorator.drawSegmentationResult(image, segmentationResults[i]);
        decorator.drawEncodingZone(image, segmentationResults[i]);

        std::ostringstream winname;
        winname << "image" << (i+1);
        cv::imshow(winname.str(), image);
    }

    while (char(cv::waitKey(0)) != 'q') {}
}
