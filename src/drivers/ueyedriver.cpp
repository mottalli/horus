#include "ueyedriver.hpp"

// STD
#include <cassert>

IDSEXP CHECK_CALL(IDSEXP code)
{
    if (code != IS_SUCCESS) {
        throw std::runtime_error("uEye: Return wasn't success");
    }
    return code;
}

namespace horus
{

UEyeVideoDriver::UEyeVideoDriver() :
    BaseVideoDriver()
{
    this->_currBufferId = -1;
}

UEyeVideoDriver::~UEyeVideoDriver()
{
    this->_release();
}

std::vector<BaseVideoDriver::VideoCamera> UEyeVideoDriver::queryCameras()
{
    std::vector<VideoCamera> res;

    INT numcam;
    CHECK_CALL(is_GetNumberOfCameras(&numcam));

    UEYE_CAMERA_LIST* clist = (UEYE_CAMERA_LIST*) new char[sizeof(ULONG) + numcam*sizeof(UEYE_CAMERA_INFO)];
    clist->dwCount = numcam;
    CHECK_CALL(is_GetCameraList(clist));
    for (int i = 0; i < numcam; i++) {
        const char* model = clist->uci[i].Model;
        const char* serNo = clist->uci[i].SerNo;
        string description = string(model) + "-" + string(serNo);
        VideoCamera cam{ (CameraID) clist->uci[i].dwCameraID, description };
        res.push_back(cam);
    }

    delete[] clist;
    return res;
}

void UEyeVideoDriver::_doInitialization()
{
    this->_ueyeCamid = (HIDS) this->_cameraID;

    std::clog << "Initializing camera " << this->_ueyeCamid << "..." << std::endl;
    CHECK_CALL(is_InitCamera(&this->_ueyeCamid, NULL));
    CHECK_CALL(is_EnableAutoExit (this->_ueyeCamid, IS_ENABLE_AUTO_EXIT));

    std::clog << "Setting color mode..." << std::endl;
    CHECK_CALL(is_SetColorMode(this->_ueyeCamid, IS_CM_RGB8_PACKED));

    std::clog << "Enabling subsampling..." << std::endl;
    CHECK_CALL(is_SetSubSampling(this->_ueyeCamid, IS_SUBSAMPLING_2X_HORIZONTAL | IS_SUBSAMPLING_2X_VERTICAL));

    // Get frame size
    std::clog << "Querying frame size..." << std::endl;
    IS_RECT rect;
    CHECK_CALL(is_AOI(this->_ueyeCamid, IS_AOI_IMAGE_GET_AOI, &rect, sizeof(rect)));
    this->_frameWidth = rect.s32Width;
    this->_frameHeight = rect.s32Height;

    // Initialize image buffers
    std::clog << "Initializing image buffers..." << std::endl;
    int bitspp = 24;
    for (ImageBuffer& imageBuffer : this->_imageBuffers) {
        CHECK_CALL(is_AllocImageMem(this->_ueyeCamid, this->_frameWidth, this->_frameHeight, bitspp, &imageBuffer.data, &imageBuffer.imageId));
        CHECK_CALL(is_AddToSequence(this->_ueyeCamid, imageBuffer.data, imageBuffer.imageId));
    }

    // Initialize non-triggered, to-memory capture
    std::clog << "Initializing capture mode..." << std::endl;
    CHECK_CALL(is_EnableEvent(this->_ueyeCamid, IS_SET_EVENT_FRAME));
    CHECK_CALL(is_SetDisplayMode(this->_ueyeCamid, IS_SET_DM_DIB));
    CHECK_CALL(is_CaptureVideo(this->_ueyeCamid, IS_DONT_WAIT));

    CHECK_CALL(is_GetImageMemPitch(this->_ueyeCamid, &this->_linesize));

}

void UEyeVideoDriver::_unlockCurrentBuffer()
{
    if (this->_currBufferId >= 0) {
        ImageBuffer& currBuffer = this->_imageBuffers[this->_currBufferId];
        CHECK_CALL(is_UnlockSeqBuf(this->_ueyeCamid, currBuffer.imageId, currBuffer.data));
    }
}

ColorImage UEyeVideoDriver::_captureFrame()
{
    this->_unlockCurrentBuffer();

    CHECK_CALL(is_WaitEvent(this->_ueyeCamid, IS_SET_EVENT_FRAME, 1000));

    char *data, *last;
    INT dummy = 0;
    CHECK_CALL(is_GetActSeqBuf(this->_ueyeCamid, &dummy, &last, &data));
    for (int j = 0; j < this->_imageBuffers.size(); j++) {
        if (data == this->_imageBuffers[j].data) {
            this->_currBufferId = j;
            break;
        }
    }
    assert(this->_currBufferId >= 0);

    ImageBuffer& currBuffer = this->_imageBuffers[this->_currBufferId];
    CHECK_CALL(is_LockSeqBuf(this->_ueyeCamid, currBuffer.imageId, currBuffer.data));
    ColorImage im(this->_frameHeight, this->_frameWidth, (cv::Vec3b*)currBuffer.data, this->_linesize);
    return im;
}

void UEyeVideoDriver::_release()
{
    this->_unlockCurrentBuffer();
    CHECK_CALL(is_ExitCamera(this->_ueyeCamid));
    std::clog << "Camera handler released" << std::endl;
}

}   // namespace horus
