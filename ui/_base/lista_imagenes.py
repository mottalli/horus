#!/usr/bin/python
import os
import sys
import re

if len(sys.argv) < 2:
	path = '.'
else:
	path = sys.argv[1]

if not os.path.isdir(path):
	raise Exception("No existe el directorio " + path)

print "DELETE FROM base_iris;"

idClase = 0
for root, dirs, files in os.walk(path):
	if root == '.': continue
	
	# Me quedo solo con las jpg:
	files = filter(lambda s: re.match('.*.JPG$', s, re.IGNORECASE), files)
	if len(files) == 0: continue
	
	idClase = idClase+1
	for imagen in files:
		print "INSERT INTO base_iris(imagen,id_clase) VALUES('%s', '%s');" % (str(os.path.join(root, imagen)), idClase)
