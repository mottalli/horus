%module "pyhorus"

%{
#include "../src/horus.h"

using namespace horus;

// From the OpenCV 2.2 Python interface
struct cvmat_t {
	PyObject_HEAD
	CvMat *a;
	PyObject *data;
	size_t offset;
};

// Convert a OpenCV Python wrapper matrix to a Mat object
inline Mat* convertParam(PyObject* input)
{
	// Depending on the OpenCV wrappers, $input could be 
	// either a CvMat pointer (SWIG interface) or a cv.cvmat object (new Python interface)
	const string python_typename = string(input->ob_type->tp_name);
	CvMat* arg = 0;
	
	if (python_typename == "cv.cvmat") {
		arg = ((cvmat_t*)input)->a;
	} else if (python_typename == "CvMat") {
		SWIG_ConvertPtr(input, (void**)&arg, SWIGTYPE_p_CvMat, 0);
	} else {
		string error = string("SWIG - pyhorus::ConvertParam: Don't know how to convert ") + python_typename + " to Mat object";
		throw std::runtime_error(error);
	}
	
	return new Mat(arg);
}

%}

// This is done just to instantiate the CvMat type in SWIG (will provide the SWIGTYPE_p_CvMat type)
CvMat __cvmat_unused;
%{
CvMat __cvmat_unused;
%}


%include "exception.i"

%exception {
	try {
		$action
	} catch (const std::exception& e) {
		SWIG_exception(SWIG_RuntimeError, e.what());
	}
}


%include "std_vector.i"
%include "std_string.i"

// Without this Swig complains about knowing nothing about the namespace "cv"
namespace cv {
};

%typemap(in) Mat const & {
	$1 = convertParam($input);
}

%typemap(freearg) Mat const & {
	delete $1;
}

%typemap(in) Mat & {
	$1 = convertParam($input);
}

%typemap(freearg) Mat & {
	delete $1;
}

/*%typemap(out) Mat_<uint8_t> {
	// TODO: Fix this memory leak (typemap(freearg) does not work)
	CvMat* m = new CvMat($1);
	$result = SWIG_NewPointerObj(m, SWIGTYPE_p_CvMat, 0);
}*/

// Initialize used vector types
namespace std
{
	%template(vectord) vector<double>;
	%template(vectori) vector<int>;
	%template(vectorvectord) vector< vector<double> >;
}

%include "../src/common.h"
%include "../src/clock.h"
%include "../src/decorator.h"
%include "../src/eyelidsegmentator.h"
%include "../src/irissegmentator.h"
%include "../src/iristemplate.h"
%include "../src/irisencoder.h"
%include "../src/loggaborencoder.h"
%include "../src/irisdatabase.h"
%include "../src/pupilsegmentator.h"
%include "../src/qualitychecker.h"
%include "../src/segmentator.h"
%include "../src/serializer.h"
%include "../src/templatecomparator.h"
%include "../src/tools.h"
%include "../src/types.h"
%include "../src/videoprocessor.h"
%include "../src/eyedetect.h"
//%include "../src/irisdctencoder.h"
%include "../src/gaborencoder.h"
#ifdef USE_CUDA
%include "../src/irisdatabasecuda.h"
#endif


