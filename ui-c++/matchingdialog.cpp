#include "matchingdialog.h"
#include "ui_matchingdialog.h"

#include "sqlite3irisdatabase.h"

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

void MatchingDialog::doMatch(const IrisTemplate& irisTemplate, const Mat_<uint8_t>& image, const SegmentationResult& segmentationResult)
{
	DB.doMatch(irisTemplate);

	Mat decorated, decoratedSmall;
	Decorator decorator;

	if (!image.empty()) {
		cvtColor(image, decorated, CV_GRAY2RGB);
		decorator.drawSegmentationResult(decorated, segmentationResult);
		cv::resize(decorated, decoratedSmall, Size(480, 360));
		decorator.drawTemplate(decoratedSmall, irisTemplate);
		this->ui->capturedImage->showImage(decoratedSmall);
	}

	SQLite3IrisDatabase::IrisData irisData = DB.getIrisData(DB.getMinDistanceId());
	if (!irisData.image.empty()) {
		decorated = irisData.image.clone();
		decorator.drawSegmentationResult(decorated, irisData.segmentation);
		cv::resize(decorated, decoratedSmall, Size(480, 360));
		decorator.drawTemplate(decoratedSmall, irisData.irisTemplate);
		this->ui->dbImage->showImage(decoratedSmall);
	}

	this->ui->lblUsername->setText(irisData.userName.c_str());

	ostringstream oss;
	oss << DB.getMinDistance();
	this->ui->lblHammingDistance->setText( oss.str().c_str() );

	this->show();
}
