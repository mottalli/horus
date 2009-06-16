import horus
from opencv import *
from opencv.highgui import *

segmentator = horus.Segmentator()
decorator = horus.Decorator()

imagen = cvLoadImage('/home/marcelo/Mis_Documentos/Facu/Tesis/Bases de datos/UBA/norberto_der_1.jpg', 0)

results = segmentator.segmentImage(imagen)
decorator.drawSegmentationResult(imagen, results)

cvNamedWindow("imagen")
cvShowImage("imagen", imagen)
cvWaitKey(0)
cvReleaseImage(imagen)