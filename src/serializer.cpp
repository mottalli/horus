#include <sstream>
#include "serializer.h"
#include "tools.h"


string Serializer::serializeParabola(const Parabola& parabola)
{
	ostringstream stream;
	stream << parabola.x0 << ',' << parabola.y0 << ',' << parabola.p;
	return stream.str();
}

string Serializer::serializeContour(const Contour& contour)
{
	ostringstream stream;
	stream << contour.size() << ',';
	for (Contour::const_iterator it = contour.begin(); it != contour.end(); it++) {
		stream << '(' << (*it).x << ',' << (*it).y << ')';
	}
	return stream.str();
}

SegmentationResult Serializer::unserializeSegmentationResult(const string& s)
{
	SegmentationResult res;
	istringstream stream(s);
	char c;
	int foo;

	res.pupilContour = unserializeContour(stream);
	res.irisContour = unserializeContour(stream);

	stream >> c; assert(c == ',');
	stream >> foo;		// UNUSED
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

Contour Serializer::unserializeContour(istringstream& stream)
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

string Serializer::serializeSegmentationResult(const SegmentationResult& sr)
{
	ostringstream stream;

	stream << serializeContour(sr.pupilContour);
	stream << serializeContour(sr.irisContour);

	stream << ",1";		// UNUSED - used to be quality of the contour
	stream << ',' << (sr.eyelidsSegmented ? '1' : '0');

	if (sr.eyelidsSegmented) {
		stream << ',' << serializeParabola(sr.upperEyelid);
		stream << ',' << serializeParabola(sr.lowerEyelid);
	}

	return stream.str();
}

string Serializer::serializeIrisTemplate(const IrisTemplate& irisTemplate)
{
	ostringstream stream;
	string serializedTemplate = Tools::base64EncodeMat(irisTemplate.irisTemplate);
	string serializedMask = Tools::base64EncodeMat(irisTemplate.mask);
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
	Mat packedTemplate = Tools::base64DecodeMat(encodedMat);

	stream >> encodedMat;
	Mat packedMask = Tools::base64DecodeMat(encodedMat);

	IrisTemplate res;
	// Note that by doing this, irisTemplate takes posession of the template and the mask
	res.irisTemplate = packedTemplate;
	res.mask = packedMask;
	res.encoderSignature = signature;

	return res;
}
