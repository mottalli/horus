/*
 * File:   parameters.h
 * Author: marcelo
 *
 * Created on January 22, 2009, 2:15 PM
 */

#ifndef _PARAMETERS_H
#define	_PARAMETERS_H

/**
 * Singleton class
 */
class Parameters {
public:
    virtual ~Parameters();
    static Parameters* getParameters();

    unsigned bufferWidth;
    double muPupil;
    double sigmaPupil;

    int pupilAdjustmentRingWidth, pupilAdjustmentRingHeight;
    int irisAdjustmentRingWidth, irisAdjustmentRingHeight;

private:
    Parameters();
    static Parameters* instance;

};

#endif	/* _PARAMETERS_H */

