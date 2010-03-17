%module "horus"

%include "std_vector.i"
%include "std_string.i"

%typemap(in) IplImage* {
	void* argp = 0;
	CvMat* arg = 0;
	SWIG_ConvertPtr($input, &argp, SWIGTYPE_p_CvMat, 0);
	arg = reinterpret_cast<CvMat*>(argp);
	IplImage *tmp = new IplImage;
	$1 = cvGetImage(arg, tmp);
}

%typemap(freearg) IplImage* {
	delete $1;
}

%include "../src/common.h"

%{
#include "../src/clock.h"
#include "../src/common.h"
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
#include "../src/irisdctencoder.h"
#ifdef USE_CUDA
#include "../src/irisdatabasecuda.h"
#endif
%}

%include "../src/clock.h"
%include "../src/common.h"
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
%include "../src/irisdctencoder.h"
#ifdef USE_CUDA
%include "../src/irisdatabasecuda.h"
#endif



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
