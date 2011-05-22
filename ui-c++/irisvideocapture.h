#ifndef IRISVIDEOCAPTURE_H
#define IRISVIDEOCAPTURE_H

#include <QObject>
#include "common.h"

class IrisVideoCapture : public QObject
{
    Q_OBJECT
public:
	IrisVideoCapture(const string path = "/tmp");

	static const int VIDEO_FORMAT;
	static const int FPS = 15;

	bool isPaused() { return this->paused; }

signals:

public slots:
	void slotFrameProcessed(const VideoProcessor& videoProcessor);
	void setPause(int p);

private:
	string path;
	VideoWriter* writer;
	unsigned framesLeft;

	string getNextFilename();

	string currentFilename;

	bool capturing;
	bool gotTemplate;
	bool paused;

};

#endif // IRISVIDEOCAPTURE_H
