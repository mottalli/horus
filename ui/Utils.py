import opencv

def aColor(imagen, dest=None):
	if not dest or opencv.cvGetElemType(dest) != opencv.CV_8UC3 or opencv.cvGetSize(dest).width != cvGetSize(imagen).width or opencv.cvGetSize(dest).height != opencv.cvGetSize(imagen).height:
		dest = opencv.cvCreateImage(opencv.cvGetSize(imagen), opencv.IPL_DEPTH_8U, 3)

	elemType = opencv.cvGetElemType(imagen)
	
	if elemType == opencv.CV_8UC1:
		opencv.cvMerge(imagen, None, None, None, dest)
		opencv.cvMerge(None, imagen, None, None, dest)
		opencv.cvMerge(None, None, imagen, None, dest)
	
	elif elemType == opencv.CV_8UC3:
		opencv.cvCopy(imagen, dest)

	return dest
