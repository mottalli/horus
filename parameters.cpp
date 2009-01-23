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
    muPupil = 2.0;
    sigmaPupil = 10.0;
}

Parameters::~Parameters() {
}

Parameters* Parameters::getParameters() {
    if (Parameters::instance == 0) {
        Parameters::instance = new Parameters();
    }

    return Parameters::instance;
}

