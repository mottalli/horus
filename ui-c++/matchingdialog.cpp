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

void MatchingDialog::doMatch(const IrisTemplate& irisTemplate, Mat imagen, SegmentationResult segmentationResult)
{
	DB.doMatch(irisTemplate);

	cout << DB.getMinDistance() << ", " << DB.getMinDistanceId() << endl;
	this->show();
}
