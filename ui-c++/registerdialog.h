#ifndef REGISTERDIALOG_H
#define REGISTERDIALOG_H

#include <QDialog>
#include "common.h"
#include "sqlite3irisdatabase.h"

namespace Ui {
    class RegisterDialog;
}

class RegisterDialog : public QDialog
{
    Q_OBJECT

public:
    explicit RegisterDialog(QWidget *parent = 0);
    ~RegisterDialog();

	void doRegister(IrisTemplate irisTemplate, const GrayscaleImage& image, SegmentationResult segmentationResult, horus::VideoProcessor::CaptureBurst captureBurst=horus::VideoProcessor::CaptureBurst());

private:
    Ui::RegisterDialog *ui;

	GrayscaleImage image;
	ColorImage decoratedImage;
};

#endif // REGISTERDIALOG_H
