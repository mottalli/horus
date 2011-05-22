#include "irisvideocapture.h"

#include <boost/date_time/posix_time/posix_time.hpp>

const int IrisVideoCapture::VIDEO_FORMAT = CV_FOURCC('D','I','V','X');

IrisVideoCapture::IrisVideoCapture(const string path_)
{
	this->path = path_;
	this->capturing = false;
	this->writer = NULL;
	this->paused = true;
}

void IrisVideoCapture::slotFrameProcessed(const VideoProcessor& videoProcessor)
{
	if (this->paused) return;


	VideoProcessor::VideoStatus status = videoProcessor.lastStatus;

	if (status >= VideoProcessor::IRIS_LOW_QUALITY && !this->capturing) {
		// Inicializo la captura
		this->writer = new VideoWriter(this->getNextFilename(), VIDEO_FORMAT, FPS, videoProcessor.lastFrame.size(), true);
		if (!this->writer->isOpened()) {
			qDebug() << "No se pudo inicializar capturador de video";
			delete this->writer;
			this->writer = NULL;
			return;
		}

		this->capturing = true;
		this->framesLeft = 8*FPS;
		this->gotTemplate = false;
	}

	if (status == VideoProcessor::GOT_TEMPLATE) {
		this->framesLeft = 1.5*FPS;
		this->gotTemplate = true;
	}


	if (this->capturing) {
		assert(this->writer != NULL);
		(*this->writer) << videoProcessor.lastFrame;

		this->framesLeft--;
		if (!this->framesLeft) {
			delete this->writer;
			this->writer = NULL;
			this->capturing = false;

			if (!this->gotTemplate) {
				// En la ráfaga de video no hubo un iris - borro el video
				qDebug() << "Iris no detectado -- borrando" << this->currentFilename.c_str();
				filesystem::remove(this->currentFilename);
			} else {
				qDebug() << "Video guardado.";
			}
		}
	}
}

string IrisVideoCapture::getNextFilename()
{
	// Mierda que esto es complicado...
	using namespace posix_time;
	ptime now = second_clock::local_time();
	gregorian::date_facet* df = new gregorian::date_facet("%Y-%m-%d");

	ostringstream filename;
	filename.imbue(locale(filename.getloc(), df));
	filename << this->path << '/' << now.date() << ' ' << now.time_of_day() << ".avi";

	this->currentFilename = filename.str();
	return this->currentFilename;
}


void IrisVideoCapture::setPause(int p)
{
	if (p == 0) {
		this->paused = true;
		if (this->writer != NULL) {
			delete this->writer;
			this->writer = NULL;
		}
	} else {
		this->paused = false;
	}
}
