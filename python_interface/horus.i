%module "horus"

%{
#include "../src/common.h"
#include "../src/clock.h"
#include "../src/decorator.h"
#include "../src/eyelidsegmentator.h"
#include "../src/helperfunctions.h"
#include "../src/irisencoder.h"
#include "../src/loggaborencoder.h"
#include "../src/irissegmentator.h"
#include "../src/iristemplate.h"
#include "../src/irisdatabase.h"
#include "../src/parameters.h"
#include "../src/pupilsegmentator.h"
#include "../src/segmentationresult.h"
#include "../src/segmentator.h"
#include "../src/serializer.h"
#include "../src/templatecomparator.h"
#include "../src/tools.h"
#include "../src/types.h"
#include "../src/videoprocessor.h"
//#include "../src/irisdctencoder.h"
#include "../src/gaborencoder.h"
#ifdef USE_CUDA
#include "../src/irisdatabasecuda.h"
#endif
%}


%include "std_vector.i"
%include "std_string.i"

// Without this Swig complains about knowing nothing about the namespace "cv"
namespace cv {
};


// This is just done to instantiante CvMat as a pointer in Swig
%{
CvMat __horus_swig_unused;
%}
CvMat __horus_swig_unused;


%typemap(in) Mat const & {
	// From Python comes a CvMat*
	CvMat* arg = 0;
	SWIG_ConvertPtr($input, (void**)&arg, SWIGTYPE_p_CvMat, 0);
	Mat* inarg = new Mat(arg);
	$1 = inarg;
}

%typemap(freearg) Mat const & {
	delete $1;
}

%typemap(in) Mat & {
	// From Python comes a CvMat*
	CvMat* arg = 0;
	SWIG_ConvertPtr($input, (void**)&arg, SWIGTYPE_p_CvMat, 0);
	Mat* inarg = new Mat(arg);
	$1 = inarg;
}

%typemap(freearg) Mat & {
	delete $1;
}


namespace std
{
	%template(vectord) vector<double>;
	%template(vectori) vector<int>;
	%template(vectorvectord) vector< vector<double> >;
}

%include "exception.i"

%exception {
  try {
    $action
  } catch (const std::exception& e) {
    SWIG_exception(SWIG_RuntimeError, e.what());
  }
}

%include "../src/common.h"
%include "../src/clock.h"
%include "../src/decorator.h"
%include "../src/eyelidsegmentator.h"
%include "../src/helperfunctions.h"
%include "../src/irisencoder.h"
%include "../src/loggaborencoder.h"
%include "../src/irissegmentator.h"
%include "../src/iristemplate.h"
%include "../src/irisdatabase.h"
%include "../src/parameters.h"
%include "../src/pupilsegmentator.h"
%include "../src/segmentationresult.h"
%include "../src/segmentator.h"
%include "../src/serializer.h"
%include "../src/templatecomparator.h"
%include "../src/tools.h"
%include "../src/types.h"
%include "../src/videoprocessor.h"
//%include "../src/irisdctencoder.h"
%include "../src/gaborencoder.h"
#ifdef USE_CUDA
%include "../src/irisdatabasecuda.h"
#endif
