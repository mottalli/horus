#ifndef MATCHINGDIALOG_H
#define MATCHINGDIALOG_H

#include <QDialog>

namespace Ui {
    class MatchingDialog;
}

class MatchingDialog : public QDialog
{
    Q_OBJECT

public:
    explicit MatchingDialog(QWidget *parent = 0);
    ~MatchingDialog();

private:
    Ui::MatchingDialog *ui;
};

#endif // MATCHINGDIALOG_H
