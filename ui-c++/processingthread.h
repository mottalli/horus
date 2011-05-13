#ifndef PROCESSINGTHREAD_H
#define PROCESSINGTHREAD_H

#include <QThread>

#include "common.h"

class ProcessingThread : public QThread
{
    Q_OBJECT
public:
    explicit ProcessingThread(QObject *parent = 0);
	void run();

	VideoProcessor videoProcessor;

signals:
	void signalFrameProcessed(const VideoProcessor& vp);
	void signalGotTemplate(const VideoProcessor& vp);

public slots:
	void slotProcessFrame(const ColorImage& frame);

protected:
	ColorImage _frame;

};

#endif // PROCESSINGTHREAD_H
