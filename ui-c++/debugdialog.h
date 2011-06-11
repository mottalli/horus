#ifndef DEBUGDIALOG_H
#define DEBUGDIALOG_H

#include "common.h"
#include <QDialog>

namespace Ui {
    class DebugDialog;
}

class DebugDialog : public QDialog
{
    Q_OBJECT

public:
    explicit DebugDialog(QWidget *parent = 0);
    ~DebugDialog();

public slots:
	void open();
	void done(int r);
	void slotFrameProcessed(const VideoProcessor& videoProcessor);

private:
    Ui::DebugDialog *ui;
	Decorator decorator;
};

#endif // DEBUGDIALOG_H
