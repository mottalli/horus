#ifndef VIDEOTHREAD_H
#define VIDEOTHREAD_H

#include <QObject>
#include <QThread>

#include "common.h"

class VideoThread : public QThread
{
    Q_OBJECT
public:
	explicit VideoThread(int cam = 0);
	void run();
	void stop() { _stop = true; }

signals:
	void signalFrameAvailable(const Mat& frame);

public slots:
private:
	VideoCapture _cap;
	bool _stop;
	Mat _frame;
};

#endif // VIDEOTHREAD_H
