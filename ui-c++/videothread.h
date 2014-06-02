#ifndef VIDEOTHREAD_H
#define VIDEOTHREAD_H

#include <QObject>
#include <QThread>

#include "drivers/basedriver.hpp"

#include "common.h"

class VideoThread : public QThread
{
    Q_OBJECT
public:
    explicit VideoThread(int cam = 0);
    ~VideoThread();

    void run();
    void stop() { qDebug() << "VideoThread::stop"; _stop = true; }

    inline void setCapture(int cam) { _cam = cam; }

signals:
    void signalFrameAvailable(const ColorImage& frame);

public slots:
private:
    horus::BaseVideoDriver* _driver;
    bool _stop;
    Mat _frame;
    int _cam;
};

#endif // VIDEOTHREAD_H
