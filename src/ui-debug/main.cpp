// STD
#include <iostream>

// OpenCV
#include <opencv2/opencv.hpp>

// QT
#include <qt5/QtWidgets/QApplication>

#include "ui-common/imagewidget.h"

int main(int argc, char** argv)
{
    QApplication app(argc, argv);

    cv::Mat image = cv::imread("/home/marcelo/Pictures/Varias/Iris/IMG-20140529-WA0002.jpg");

    ImageWidget imwidget;
    imwidget.showImage(image);
    imwidget.show();

    return app.exec();

}
