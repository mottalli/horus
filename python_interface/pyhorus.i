%module "pyhorus"

%{
#include "../src/horus.h"
%}

%include "std_vector.i"
%include "std_string.i"

// Without this Swig complains about knowing nothing about the namespace "cv"
namespace cv {
};


%{
// From OpenCV 2.2 Python interface
struct cvmat_t {
  PyObject_HEAD
  CvMat *a;
  PyObject *data;
  size_t offset;
};
%}

%typemap(in) Mat const & {
	// Depending on the OpenCV wrappers, $input could be 
	// either a CvMat pointer (SWIG interface) or a cv.cvmat object (new Python interface)
	
	CvMat* arg = 0;
	if (strcmp($input->ob_type->tp_name, "cv.cvmat") == 0) {		// New python interface object
		arg = ((cvmat_t*)$input)->a;
	} else {	// Old SWIG interface (CvMat pointer)
		SWIG_ConvertPtr($input, (void**)&arg, SWIGTYPE_p_CvMat, 0);
	}
	$1 = new Mat(arg);
}

%typemap(freearg) Mat const & {
	delete $1;
}

%typemap(in) Mat & {
	// Depending on the OpenCV wrappers, $input could be 
	// either a CvMat pointer (SWIG interface) or a cv.cvmat object (new Python interface)

	CvMat* arg = 0;
	if (strcmp($input->ob_type->tp_name, "cv.cvmat") == 0) {		// New python interface object
		arg = ((cvmat_t*)$input)->a;
	} else {	// Old SWIG interface (CvMat pointer)
		SWIG_ConvertPtr($input, (void**)&arg, SWIGTYPE_p_CvMat, 0);
	}
	$1 = new Mat(arg);
}

%typemap(freearg) Mat & {
	delete $1;
}

%typemap(out) Mat_<uint8_t> {
	// TODO: Fix this memory leak (typemap(freearg) does not work)
	CvMat* m = new CvMat($1);
	$result = SWIG_NewPointerObj(m, SWIGTYPE_p_CvMat, 0);
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

%typemap(in) Mat* matin {
	void* arg = 0;
	SWIG_ConvertPtr($input, (void**)&arg, SWIGTYPE_p_Mat, 0);
	$1 = reinterpret_cast<Mat*>(arg);
}

%newobject pyutilCloneImage;
%inline %{
	CvMat* pyutilCloneFromHorus(Mat* matin) {
		CvMat* a = new CvMat;
		*a = (CvMat)*matin;
		return a;
	}
%}

%extend IrisEncoder {
	static void normalizeIrisWRAP(const GrayscaleImage& image, GrayscaleImage& dest, GrayscaleImage& destMask, const SegmentationResult& segmentationResult, double theta0, double theta1, double radius)
	{
		IrisEncoder::normalizeIris(image, dest, destMask, segmentationResult, theta0, theta1, radius);
	};
}

void superimposeTextureWRAP(GrayscaleImage& image, const GrayscaleImage& texture, const SegmentationResult& segmentation, double theta0, double theta1, double radius, bool blend, double blendStart);
%{
void superimposeTextureWRAP(GrayscaleImage& image, const GrayscaleImage& texture, const SegmentationResult& segmentation, double theta0, double theta1, double radius, bool blend, double blendStart)
{
	Tools::superimposeTexture(image, texture, segmentation, theta0, theta1, radius, blend, blendStart);
}
%}



%include "../src/common.h"
%include "../src/clock.h"
%include "../src/decorator.h"
%include "../src/eyelidsegmentator.h"
%include "../src/irisencoder.h"
%include "../src/loggaborencoder.h"
%include "../src/irissegmentator.h"
%include "../src/iristemplate.h"
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

%extend Decorator {
	void Decorator::drawTemplateWRAP(Image& image, const IrisTemplate& irisTemplate)
	{
		$self->drawTemplate(image, irisTemplate);
	};
}

