/*
 * File:   parameters.cpp
 * Author: marcelo
 *
 * Created on January 22, 2009, 2:15 PM
 */

#include "parameters.h"

Parameters* Parameters::instance = (Parameters*)0;

Parameters::Parameters() {
    bufferWidth = 320;
    muPupil = 5.0;
    sigmaPupil = 5.0;

    pupilAdjustmentRingWidth = 100;
    pupilAdjustmentRingHeight = 40;

    irisAdjustmentRingWidth = 100;
    irisAdjustmentRingHeight = 90;

    parabolicDetectorStep = 4;

}

Parameters::~Parameters() {
}

Parameters* Parameters::getParameters() {
    if (Parameters::instance == 0) {
        Parameters::instance = new Parameters();
    }

    return Parameters::instance;
}

