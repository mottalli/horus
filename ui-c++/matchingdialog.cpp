#include <QMessageBox>
#include "matchingdialog.h"
#include "ui_matchingdialog.h"

extern SQLite3IrisDatabase DB;

MatchingDialog::MatchingDialog(QWidget *parent) :
    QDialog(parent),
    ui(new Ui::MatchingDialog)
{
    ui->setupUi(this);
}

MatchingDialog::~MatchingDialog()
{
    delete ui;
}

void MatchingDialog::doMatch(IrisTemplate irisTemplate, const GrayscaleImage& image, SegmentationResult segmentationResult)
{
	// Hace la identificación
	DB.doMatch(irisTemplate);
	double hdTime = DB.getMatchingTime();
	DB.doAContrarioMatch(irisTemplate);
	double aContrarioTime = DB.getMatchingTime();

	int hdMatch = DB.getMinDistanceId();
	int aContrarioMatch = DB.getMinNFAId();

	double matchingNFA = DB.getMinNFA();
	double matchingHD = DB.getMinDistance();
	double aContrarioHD = DB.getDistanceFor(aContrarioMatch);

	//SQLite3IrisDatabase::IrisData irisData = DB.getIrisData(hdMatch);
	SQLite3IrisDatabase::IrisData irisData = DB.getIrisData(aContrarioMatch);

	this->lastMatch = irisData;


	// Mostrar las imágenes
	ColorImage decorated, decoratedSmall;
	Decorator decorator;

	this->lastQueryImage = image.clone();

	if (!image.empty()) {
		cvtColor(image, decorated, CV_GRAY2RGB);
		decorator.drawSegmentationResult(decorated, segmentationResult);
		cv::resize(decorated, decoratedSmall, Size(480, 360));
		decorator.drawTemplate(decoratedSmall, irisTemplate);
		this->ui->capturedImage->showImage(decoratedSmall);
	}


	if (!irisData.image.empty()) {
		decorated = irisData.image.clone();
		decorator.drawSegmentationResult(decorated, irisData.segmentation);
		cv::resize(decorated, decoratedSmall, Size(480, 360));
		decorator.drawTemplate(decoratedSmall, irisData.irisTemplate);
		this->ui->dbImage->showImage(decoratedSmall);
	}

	if (hdMatch != aContrarioMatch) {
		// ???
	}

	QString textoCant = (boost::format("Sobre un total de %1% iris - Tiempo de busqueda: %2% miliseg.") % DB.databaseSize() % (hdTime+aContrarioTime)).str().c_str();

	this->ui->lblHammingDistance->setText( (boost::format("%.5f") % aContrarioHD).str().c_str() );
	this->ui->lblNFA->setText( (boost::format("%.5f") % matchingNFA).str().c_str() );
	this->ui->lblCantidadImagenes->setText(textoCant);
	this->ui->lblUsername->setText(irisData.userName.c_str());

	QString identificacion = (aContrarioHD < 0.35 ? "<font color='green'>Positiva</font>" : "<font color='red'>Negativa</font>");

	this->ui->lblIdentification->setText(identificacion);

	this->show();

	/*BOOST_FOREACH(double d, DB.resultDistances) {
		qDebug() << d;
	}*/
}

void MatchingDialog::on_btnConfirmarIdentificacion_clicked()
{
	int add = QMessageBox::question(this, "Agregar imagen", "Agregar la imagen a la base de datos?", QMessageBox::Yes, QMessageBox::No);
	if (add == QMessageBox::Yes) {
		DB.addImage(this->lastMatch.userId, this->lastQueryImage);
	}
	this->accept();
}
