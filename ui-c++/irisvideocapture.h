#ifndef IRISVIDEOCAPTURE_H
#define IRISVIDEOCAPTURE_H

#include <QObject>
#include "common.h"

class IrisVideoCapture : public QObject
{
    Q_OBJECT
public:
    explicit IrisVideoCapture(QObject *parent = 0);
	IrisVideoCapture(const string path = "/tmp");

	static const int VIDEO_FORMAT = CV_FOURCC('D','I','V','X');
	static const int FPS = 15;

	bool isPaused() { return _paused; }

signals:

public slots:
	void slotFrameProcessed(const VideoProcessor& videoProcessor);
	void setPause(int p);

private:
	string _path;
	VideoWriter* _writer;
	unsigned _imageNumber;
	unsigned _framesLeft;

	string getNextFilename();
	string getCurrentFilename() const;
	bool _capturing;
	bool _gotTemplate;
	bool _paused;

};

#endif // IRISVIDEOCAPTURE_H
