from opencv import *

def aColor(imagen, dest=None):
	if not dest or cvGetElemType(dest) != CV_8UC3 or cvGetSize(dest).width != cvGetSize(imagen).width or cvGetSize(dest).height != cvGetSize(imagen).height:
		dest = cvCreateImage(cvGetSize(imagen), IPL_DEPTH_8U, 3)

	elemType = cvGetElemType(imagen)
	
	if elemType == CV_8UC1:
		cvMerge(imagen, None, None, None, dest)
		cvMerge(None, imagen, None, None, dest)
		cvMerge(None, None, imagen, None, dest)
	
	elif elemType == CV_8UC3:
		cvCopy(imagen, dest)

	return dest
