#include "irisvideocapture.h"

IrisVideoCapture::IrisVideoCapture(QObject *parent) :
    QObject(parent)
{
	_path = "/tmp";
}

IrisVideoCapture::IrisVideoCapture(const string path)
{
	_path = path;
	_imageNumber = 1;
	_capturing = false;
	_writer = NULL;
	_paused = true;
}

void IrisVideoCapture::slotFrameProcessed(const VideoProcessor& videoProcessor)
{
	if (_paused) return;


	VideoProcessor::VideoStatus status = videoProcessor.lastStatus;

	if (status >= VideoProcessor::IRIS_LOW_QUALITY && !_capturing) {
		// Inicializo la captura
		_writer = new VideoWriter(this->getNextFilename(), VIDEO_FORMAT, FPS, videoProcessor.lastFrame.size(), true);
		if (!_writer->isOpened()) {
			qDebug() << "No se pudo inicializar capturador de video";
			delete _writer;
			_writer = NULL;
			return;
		}

		_capturing = true;
		_framesLeft = 8*FPS;
		_gotTemplate = false;
	}

	if (status == VideoProcessor::GOT_TEMPLATE) {
		_framesLeft = 1.5*FPS;
		_gotTemplate = true;
	}


	if (_capturing) {
		assert(_writer != NULL);
		(*_writer) << videoProcessor.lastFrame;

		_framesLeft--;
		if (!_framesLeft) {
			delete _writer;
			_writer = NULL;
			_capturing = false;

			if (!_gotTemplate) {
				// En la ráfaga de video no hubo un iris - borro el video
				qDebug() << "Iris no detectado -- borrando" << this->getCurrentFilename().c_str();
				boost::filesystem::remove(this->getCurrentFilename());
			}
		}
	}
}

string IrisVideoCapture::getNextFilename()
{
	while (true) {
		string fileName = (boost::format("%1%/irisvideo%2%.avi") % _path % _imageNumber).str();

		if (!boost::filesystem::is_regular_file(fileName)) {
			return fileName;
		}

		_imageNumber++;
	}
}

string IrisVideoCapture::getCurrentFilename() const
{
	return (boost::format("%1%/irisvideo%2%.avi") % _path % _imageNumber).str();
}

void IrisVideoCapture::setPause(int p)
{
	if (p == 0) {
		_paused = true;
		if (_writer != NULL) {
			delete _writer;
			_writer = NULL;
		}
	} else {
		_paused = false;
	}
}
