#include "serializer.h"
#include <sstream>


std::string Serializer::serializeParabola(const Parabola& parabola)
{
	std::ostringstream stream;
	stream << parabola.x0 << ',' << parabola.y0 << ',' << parabola.p;
	return stream.str();
}

std::string Serializer::serializeContour(const Contour& contour)
{
	std::ostringstream stream;
	stream << contour.size() << ',';
	for (Contour::const_iterator it = contour.begin(); it != contour.end(); it++) {
		stream << '(' << (*it).x << ',' << (*it).y << ')';
	}
	return stream.str();
}

SegmentationResult Serializer::unserializeSegmentationResult(const std::string& s)
{
	SegmentationResult res;
	std::istringstream stream(s);
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

	return res;
}

Contour Serializer::unserializeContour(std::istringstream& stream)
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

Parabola Serializer::unserializeParabola(std::istringstream& stream)
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

std::string Serializer::serializeSegmentationResult(const SegmentationResult& sr)
{
	std::ostringstream stream;

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

