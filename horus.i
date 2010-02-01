%module "horus"

%include "std_vector.i"
%include "std_string.i"

%typemap(in) Image* {
	void* argp = 0;
	CvMat* arg = 0;
	SWIG_ConvertPtr($input, &argp, SWIGTYPE_p_CvMat, 0);
	arg = reinterpret_cast<CvMat*>(argp);
	Image *tmp = new Image;
	$1 = cvGetImage(arg, tmp);
}

%typemap(freearg) Image* {
	delete $1;
}


%{
#include "clock.h"
#include "common.h"
#include "decorator.h"
#include "eyelidsegmentator.h"
#include "helperfunctions.h"
#include "irisencoder.h"
#include "irissegmentator.h"
#include "iristemplate.h"
#include "irisdatabase.h"
#include "parameters.h"
#include "pupilsegmentator.h"
#include "segmentationresult.h"
#include "segmentator.h"
#include "serializer.h"
#include "templatecomparator.h"
#include "tools.h"
#include "types.h"
#include "videoprocessor.h"
#include "irisdctencoder.h"
%}

%include "clock.h"
%include "common.h"
%include "decorator.h"
%include "eyelidsegmentator.h"
%include "helperfunctions.h"
%include "irisencoder.h"
%include "irissegmentator.h"
%include "iristemplate.h"
%include "irisdatabase.h"
%include "parameters.h"
%include "pupilsegmentator.h"
%include "segmentationresult.h"
%include "segmentator.h"
%include "serializer.h"
%include "templatecomparator.h"
%include "tools.h"
%include "types.h"
%include "videoprocessor.h"
%include "irisdctencoder.h"


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
