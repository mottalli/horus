#include "registerdialog.h"
#include "ui_registerdialog.h"

extern SQLite3IrisDatabase DB;

RegisterDialog::RegisterDialog(QWidget *parent) :
    QDialog(parent),
    ui(new Ui::RegisterDialog)
{
    ui->setupUi(this);
}

RegisterDialog::~RegisterDialog()
{
    delete ui;
}

void RegisterDialog::doRegister(IrisTemplate irisTemplate, const GrayscaleImage& image_, SegmentationResult segmentationResult, horus::VideoProcessor::CaptureBurst captureBurst)
{
	Decorator decorator;

	this->image = image_.clone();
	cvtColor(this->image, this->decoratedImage, CV_GRAY2BGR);
	decorator.drawSegmentationResult(this->decoratedImage, segmentationResult);
	decorator.drawTemplate(this->decoratedImage, irisTemplate);

	this->ui->image->showImage(this->decoratedImage);
	this->ui->lblError->setText("");
	this->ui->txtUserName->setText("");
	this->ui->txtUserName->setFocus();

	//this->show();

	while (true) {
		int res = this->exec();
		if (res == Rejected) break;

		qDebug() << "Insertando entrada en la base de datos...";

		try {
			DB.addUser(this->ui->txtUserName->text().toStdString(), irisTemplate, segmentationResult, this->image, captureBurst);
			break;
		} catch (std::runtime_error ex) {
			string msgError = string("<font color=\"#ff0000\">") + ex.what() + string("</font>");
			this->ui->lblError->setText(msgError.c_str());
			this->ui->txtUserName->setFocus();			// Lo más probable es que haya que corregir el nombre de usuario
		}
	}
}
