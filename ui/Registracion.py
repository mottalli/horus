from PyQt4 import uic

def registrar(irisDatabase, template, imagen, segmentacion):
	form = uic.loadUi('forms/Registracion.ui')
	form.lblError.setText('')

	while True:
		form.show()
		if not form.exec_():
			return

		try:
			irisDatabase.agregarTemplate(form.txtNombreUsuario.text(), imagen, template, segmentacion)
			return
		except Exception as e:
			form.lblError.setText('<font color="red">' + str(e) + '</font>')
