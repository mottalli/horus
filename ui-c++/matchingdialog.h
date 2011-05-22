#ifndef MATCHINGDIALOG_H
#define MATCHINGDIALOG_H

#include <QDialog>
#include "common.h"
#include "sqlite3irisdatabase.h"

namespace Ui {
    class MatchingDialog;
}

class MatchingDialog : public QDialog
{
    Q_OBJECT

public:
    explicit MatchingDialog(QWidget *parent = 0);
    ~MatchingDialog();

	void doMatch(IrisTemplate irisTemplate, const GrayscaleImage& image, SegmentationResult segmentationResult);

private slots:
	void on_btnConfirmarIdentificacion_clicked();

private:
    Ui::MatchingDialog *ui;

	GrayscaleImage lastQueryImage;
	SQLite3IrisDatabase::IrisData lastMatch;
	IrisTemplate lastTemplate;
	SegmentationResult lastSegmentationResult;
};

#endif // MATCHINGDIALOG_H
