#include <QMessageBox>
#include "matchingdialog.h"
#include "ui_matchingdialog.h"

#define A_CONTRARIO_MATCH false

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
	// Guarda la información de la query
	this->lastQueryImage = image.clone();
	this->lastTemplate = irisTemplate;
	this->lastSegmentationResult = segmentationResult;

	// Hace la identificación
	DB.doMatch(irisTemplate);
	double hdTime = DB.getMatchingTime();
	int hdMatch = DB.getMinDistanceId();
	double matchingHD = DB.getMinDistance();

	double totalTime;

#if A_CONTRARIO_MATCH
	DB.doAContrarioMatch(irisTemplate);
	double aContrarioTime = DB.getMatchingTime();
	int aContrarioMatch = DB.getMinNFAId();
	double matchingNFA = DB.getMinNFA();
	matchingHD = DB.getDistanceFor(aContrarioMatch);
	SQLite3IrisDatabase::IrisData irisData = DB.getIrisData(aContrarioMatch);
	totalTime = hdTime + aContrarioTime;

	this->ui->lblNFA->setText( (format("%.5f") % matchingNFA).str().c_str() );
#else
	SQLite3IrisDatabase::IrisData irisData = DB.getIrisData(hdMatch);
	totalTime = hdTime;
	this->ui->lblNFA->setText("No disponible");
#endif

	this->lastMatch = irisData;

	// Mostrar las imágenes
	ColorImage decorated, decoratedSmall;
	Decorator decorator;

	bool hasMatch = (matchingHD < 0.34);

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

	QString textoCant = (boost::format("Sobre un total de %i iris - Tiempo de busqueda: %.2f miliseg.") % DB.databaseSize() % totalTime).str().c_str();

	this->ui->lblHammingDistance->setText( (boost::format("%.5f") % matchingHD).str().c_str() );
	this->ui->lblCantidadImagenes->setText(textoCant);
	this->ui->lblUsername->setText(irisData.userName.c_str());

	QString identificacion = (hasMatch ? "<font color='green'>Positiva</font>" : "<font color='red'>Negativa</font>");
	this->ui->lblIdentification->setText(identificacion);

	this->show();
}

void MatchingDialog::on_btnConfirmarIdentificacion_clicked()
{
	int add = QMessageBox::question(this, "Agregar imagen", "Agregar la imagen a la base de datos?", QMessageBox::Yes, QMessageBox::No);
	if (add == QMessageBox::Yes) {
		DB.addImage(this->lastMatch.userId, this->lastQueryImage, this->lastSegmentationResult, this->lastTemplate);
	}
	this->accept();
}
