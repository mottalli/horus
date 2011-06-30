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

void MatchingDialog::doMatch(IrisTemplate irisTemplate, const GrayscaleImage& image, SegmentationResult segmentationResult, horus::VideoProcessor::CaptureBurst captureBurst)
{
	// Guarda la información de la query
	this->lastQueryImage = image.clone();
	this->lastTemplate = irisTemplate;
	this->lastSegmentationResult = segmentationResult;
	this->lastBurst = captureBurst;

	// Hace la identificación
	DB.doMatch(irisTemplate);
	double hdTime = DB.getMatchingTime();
	int hdMatch = DB.getMinDistanceId();
	double matchingHD = DB.getMinDistance();
	const GrayscaleImage& comparationImage = DB.getComparationImage();

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

	bool hasMatch = (matchingHD < 0.30);

	QString textoCant = (boost::format("Sobre un total de %i iris - Tiempo de busqueda: %.2f miliseg.") % DB.databaseSize() % totalTime).str().c_str();
	this->ui->lblCantidadImagenes->setText(textoCant);

	if (!image.empty()) {
		cvtColor(image, decorated, CV_GRAY2RGB);
		decorator.drawSegmentationResult(decorated, segmentationResult);
		cv::resize(decorated, decoratedSmall, Size(480, 360));
		decorator.drawTemplate(decoratedSmall, irisTemplate);
		this->ui->capturedImage->showImage(decoratedSmall);
	}

	if (hasMatch) {
		if (!irisData.image.empty()) {
			decorated = irisData.image.clone();
			decorator.drawSegmentationResult(decorated, irisData.segmentation);
			cv::resize(decorated, decoratedSmall, Size(480, 360));
			decorator.drawTemplate(decoratedSmall, irisData.irisTemplate);

			const GrayscaleImage& comparationImage = DB.getComparationImage();
			Point p0(25, decoratedSmall.rows-comparationImage.rows-20);
			horus::tools::superimposeImage(comparationImage, decoratedSmall, p0, true);

			this->ui->dbImage->showImage(decoratedSmall);
		}

		this->ui->lblHammingDistance->setText( (boost::format("%.5f") % matchingHD).str().c_str() );
		this->ui->lblUsername->setText(irisData.userName.c_str());
		this->ui->lblIdentification->setText("<font color='green'>Positiva</font>");
	} else {
		Image decoratedSmall = MatchingDialog::getNoMatchImage();
		const GrayscaleImage& comparationImage = DB.getComparationImage();
		Point p0(25, decoratedSmall.rows-comparationImage.rows-20);
		horus::tools::superimposeImage(comparationImage, decoratedSmall, p0, true);

		this->ui->dbImage->showImage(decoratedSmall);
		this->ui->lblHammingDistance->setText( (boost::format("%.5f (mas cercana)") % matchingHD).str().c_str() );
		this->ui->lblUsername->setText("(Ninguna)");
		this->ui->lblIdentification->setText("<font color='red'>Negativa</font>");
	}

	this->show();
}

void MatchingDialog::on_btnConfirmarIdentificacion_clicked()
{
	DB.addImage(this->lastMatch.userId, this->lastQueryImage, this->lastSegmentationResult, this->lastTemplate, this->lastBurst);
	this->accept();
}

Image MatchingDialog::getNoMatchImage(Size size)
{
	ColorImage image(size);
	image.setTo(Scalar(0,0,0));

	line(image, Point(0,0), Point(image.cols-1,image.rows-1), CV_RGB(255,255,255), 1);
	line(image, Point(image.cols-1,0), Point(0,image.rows-1), CV_RGB(255,255,255), 1);

	return image;
}

void MatchingDialog::on_btnVerSimilares_clicked()
{
	//TODO
	BOOST_FOREACH(MatchDistance d, DB.getMatchingDistances()) {
		qDebug() << d.first << d.second;
	}
}
