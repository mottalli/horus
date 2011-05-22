#include <sstream>
#include "serializer.h"
#include "tools.h"


string Serializer::serializeSegmentationResult(const SegmentationResult& sr)
{
	ostringstream stream;

	stream << serializeContour(sr.pupilContour) << ',' << serializeContour(sr.irisContour);
	stream << ',' << (sr.eyelidsSegmented ? '1' : '0');

	if (sr.eyelidsSegmented) {
		stream << ',' << serializeParabola(sr.upperEyelid);
		stream << ',' << serializeParabola(sr.lowerEyelid);
	}

	return stream.str();
}

SegmentationResult Serializer::unserializeSegmentationResult(const string& s)
{
	SegmentationResult res;
	istringstream stream(s);
	char c;

	res.pupilContour = unserializeContour(stream);
	stream >> c; assert(c == ',');
	res.irisContour = unserializeContour(stream);

	stream >> c; assert(c == ',');
	stream >> c;		// Either '1' or '0'

	res.eyelidsSegmented = (c == '1');

	if (res.eyelidsSegmented) {
		stream >> c; assert(c == ',');
		res.upperEyelid = unserializeParabola(stream);
		stream >> c; assert(c == ',');
		res.lowerEyelid = unserializeParabola(stream);
	}

	res.irisCircle = Tools::approximateCircle(res.irisContour);
	res.pupilCircle = Tools::approximateCircle(res.pupilContour);

	return res;
}

string Serializer::serializeContour(const Contour& contour)
{
	ostringstream stream;
	// Generate a matrix from the contour
	Mat_<int16_t> mat(2, contour.size());
	for (size_t i = 0; i < contour.size(); i++) {
		const Point& p = contour[i];
		mat(0, i) = p.x;
		mat(1, i) = p.y;
	}

	string serializedMat = Tools::base64EncodeMat<int16_t>(mat);
	stream << serializedMat.size() << ',' << serializedMat;

	return stream.str();
}

Contour Serializer::unserializeContour(istringstream& stream)
{
	size_t size;
	char comma;

	stream >> size >> comma;
	assert(comma == ',');

	char* buffer = new char[size+1];				// +1 for \0
	stream.read(buffer, size);

	Mat_<int16_t> mat = Tools::base64DecodeMat<int16_t>(buffer);
	Contour res(mat.cols);

	for (int i = 0; i < mat.cols; i++) {
		res[i].x = mat(0, i);
		res[i].y = mat(1, i);
	}

	delete[] buffer;
	return res;
}

string Serializer::serializeParabola(const Parabola& parabola)
{
	ostringstream stream;
	stream << parabola.x0 << ',' << parabola.y0 << ',' << parabola.p;
	return stream.str();
}

Parabola Serializer::unserializeParabola(istringstream& stream)
{
	char c;
	int x0, y0, p;

	stream >> x0;
	stream >> c; assert(c == ',');
	stream >> y0;
	stream >> c; assert(c == ',');
	stream >> p;

	return Parabola(x0, y0, p);
}

string Serializer::serializeIrisTemplate(const IrisTemplate& irisTemplate)
{
	ostringstream stream;
	string serializedTemplate = Tools::base64EncodeMat<uint8_t>(irisTemplate.getPackedTemplate());
	string serializedMask = Tools::base64EncodeMat<uint8_t>(irisTemplate.getPackedMask());
	string signature = irisTemplate.encoderSignature;

	assert(find(signature.begin(), signature.end(), ',') == signature.end());	// Signature must not contain a comma

	stream << irisTemplate.encoderSignature << ',' << serializedTemplate.length() << "," << serializedTemplate << serializedMask;

	return stream.str();
}

IrisTemplate Serializer::unserializeIrisTemplate(const string& serializedTemplate)
{
	istringstream stream(serializedTemplate);
	char comma;
	string encodedMat, signature;
	char buffer[serializedTemplate.length()];
	int encodedTemplateLength;

	// Read the signature (all the text until the first comma)
	stream.get(buffer, serializedTemplate.length(), ',');
	signature = string(buffer);
	stream >> comma;
	assert(comma == ',');

	// Now read the encoded template
	stream >> encodedTemplateLength >> comma;
	stream.get(buffer, encodedTemplateLength+1);		// According to documentation, it reads up to (encodedTemplateLength+1)-1 characters
	encodedMat = string(buffer);
	Mat_<uint8_t> packedTemplate = Tools::base64DecodeMat<uint8_t>(encodedMat);

	stream >> encodedMat;
	Mat_<uint8_t> packedMask = Tools::base64DecodeMat<uint8_t>(encodedMat);

	IrisTemplate res;
	res.setPackedData(packedTemplate, packedMask, signature);
	return res;
}

Contour Serializer::unserializeContourOLD(istringstream& stream)
{
	int size;
	char c;
	stream >> size;
	Contour res(size);

	stream >> c; assert(c == ',');
	for (int i = 0; i < size; i++) {
		stream >> c; assert(c == '(');
		stream >> res[i].x;
		stream >> c; assert(c == ',');
		stream >> res[i].y;
		stream >> c; assert(c == ')');
	}

	return res;
}
