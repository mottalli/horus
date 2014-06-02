/*
 * File:   segmentator.h
 * Author: marcelo
 *
 * Created on January 21, 2009, 8:37 PM
 */

#pragma once

#include "common.h"
#include "pupilsegmentator.h"
#include "irissegmentator.h"
#include "eyelidsegmentator.h"
#include "clock.h"

namespace horus {

class SegmentationResult {
public:
    Contour irisContour;
    Contour pupilContour;
    Circle pupilCircle;
    Circle irisCircle;
    Parabola upperEyelid;
    Parabola lowerEyelid;

    double pupilContourQuality;

    bool eyelidsSegmented;

    /**
     * Class used for iterating the iris texture according to the segmentation result
     */
    class iterator {
    public:
        Point imagePoint;
        Point texturePoint;
        bool isOccluded;

        iterator() {}

        iterator(const SegmentationResult* sr_, Size size, double theta0_, double theta1_, double radiusMin_, double radiusMax_) :
            theta0(theta0_), theta1(theta1_), width(size.width), height(size.height), radiusMin(radiusMin_), radiusMax(radiusMax_), sr(sr_)
        {
            assert(height > 1);
            assert(radiusMin < radiusMax);

            this->theta = this->theta0;
            this->radius = this->radiusMin;

            this->calculateFromTo(theta);
            this->texturePoint = Point(0,0);
            this->imagePoint = this->from;
            this->isOccluded = this->checkOccluded(this->imagePoint);
        }

        inline bool next() {
            this->texturePoint.y++;

            if (this->texturePoint.y >= this->height) {
                // Process next column
                this->texturePoint.y = 0;
                this->texturePoint.x++;
                if (this->texturePoint.x >= this->width) {
                    // Finished processing the image
                    return false;
                }

                double prop = double(this->texturePoint.x) / double(this->width-1);
                this->theta = this->theta0 + ((this->theta1-this->theta0) * prop);
                this->calculateFromTo(theta);
            }

            double prop = double(this->texturePoint.y) / double(this->height-1);
            this->radius = this->radiusMin + ((this->radiusMax-this->radiusMin) * prop);
            this->imagePoint = this->from + ((this->to-this->from)*this->radius);
            this->isOccluded = this->checkOccluded(this->imagePoint);

            return true;
        }

    protected:
        double theta, radius;
        double theta0, theta1;
        int width, height;
        double radiusMin, radiusMax;
        const SegmentationResult* sr;
        Point2d to, from;

        inline void calculateFromTo(double theta_) {
            if (theta_ < 0) theta_ = 2.0*M_PI + theta;

            double prop = (theta_/(2.0*M_PI));
            this->from = sr->pupilContour.getPoint(prop);
            this->to = sr->irisContour.getPoint(prop);
        }

        inline bool checkOccluded(const Point& p) {
            if (!sr->eyelidsSegmented) return false;
            else return (p.y < sr->upperEyelid.value(p.x) || p.y > sr->lowerEyelid.value(p.x));
        }

    };

    inline iterator iterateIris(Size size, double theta0=0.0, double theta1=2.0*M_PI, double radiusMin=0.0, double radiusMax=1.0) const {
        return iterator(this, size , theta0, theta1, radiusMin, radiusMax);
    }


protected:
};

class Segmentator {
public:
    Segmentator();
    virtual ~Segmentator();

    SegmentationResult segmentImage(const Mat& image, cv::Rect ROI=cv::Rect());
    void segmentEyelids(const Mat& image, SegmentationResult& result);

    PupilSegmentator pupilSegmentator;
    IrisSegmentator irisSegmentator;
    EyelidSegmentator eyelidSegmentator;

    double segmentationTime;

private:
    float resizeFactor;
    Timer timer;

    GrayscaleImage blurredImage;			// Used as a buffer to calculate the ROI
};

}
