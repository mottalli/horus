%module "pyhorus"

%{
#include "../src/horus/horus.h"

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
// Initialize used vector types
namespace std
{
	%template(vectord) vector<double>;
	%template(vectori) vector<int>;
	%template(vectorvectord) vector< vector<double> >;
	%template(vectorpoint) vector< Point >;
}

%include "std_string.i"

// Without this Swig complains about knowing nothing about the namespace "cv"
namespace cv {
};

// --------------
%typemap(in) Mat const & {
	$1 = convertParam($input);
}
%typemap(freearg) Mat const & {
	delete $1;
}

// --------------
%typemap(in) Mat & {
	$1 = convertParam($input);
}
%typemap(freearg) Mat & {
	delete $1;
}

%include "../src/horus/types.h"
%include "../src/horus/common.h"
%include "../src/horus/clock.h"
%include "../src/horus/segmentator.h"
%include "../src/horus/decorator.h"
%include "../src/horus/eyelidsegmentator.h"
%include "../src/horus/irissegmentator.h"
%include "../src/horus/iristemplate.h"
%include "../src/horus/irisencoder.h"
%include "../src/horus/loggaborencoder.h"
%include "../src/horus/irisdatabase.h"
%include "../src/horus/pupilsegmentator.h"
%include "../src/horus/qualitychecker.h"
%include "../src/horus/segmentator.h"
%include "../src/horus/serializer.h"
%include "../src/horus/templatecomparator.h"
%include "../src/horus/tools.h"
%include "../src/horus/types.h"
%include "../src/horus/videoprocessor.h"
%include "../src/horus/eyedetect.h"
//%include "../src/irisdctencoder.h"
%include "../src/horus/gaborencoder.h"
#ifdef USE_CUDA
%include "../src/horus/irisdatabasecuda.h"
#endif

// ---------------- Wrappers for functions with default arguments -----------------

void normalizeIrisWRAP(const Mat& image, Mat& dest, Mat& destMask, const horus::SegmentationResult& segmentationResult, double theta0, double theta1, double radiusMin, double radiusMax);
%{
void normalizeIrisWRAP(const Mat& image_, Mat& dest_, Mat& destMask_, const SegmentationResult& segmentationResult, double theta0, double theta1, double radiusMin, double radiusMax)
{
	GrayscaleImage image(image_), dest(dest_), destMask(destMask_);
	horus::IrisEncoder::normalizeIris(image,dest,destMask,segmentationResult,theta0,theta1,radiusMin,radiusMax);
}
%}

void superimposeTextureWRAP(Mat& image, const Mat& texture, const horus::SegmentationResult& segmentation, double theta0, double theta1, double minRadius, double maxRadius, bool blend, double blendStart);
%{
void superimposeTextureWRAP(Mat& image_, const Mat& texture_, const horus::SegmentationResult& segmentation, double theta0, double theta1, double minRadius, double maxRadius, bool blend, double blendStart)
{
	GrayscaleImage image(image_), texture(texture_);
	horus::tools::superimposeTexture(image,texture,segmentation,theta0,theta1,minRadius,maxRadius,blend,blendStart);
}
%}
