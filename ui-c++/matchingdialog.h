#ifndef MATCHINGDIALOG_H
#define MATCHINGDIALOG_H

#include <QDialog>
#include "common.h"

namespace Ui {
    class MatchingDialog;
}

class MatchingDialog : public QDialog
{
    Q_OBJECT

public:
    explicit MatchingDialog(QWidget *parent = 0);
    ~MatchingDialog();

	void doMatch(const IrisTemplate& irisTemplate, Mat imagen=Mat(), SegmentationResult segmentationResult=SegmentationResult());

private:
    Ui::MatchingDialog *ui;
};

#endif // MATCHINGDIALOG_H
