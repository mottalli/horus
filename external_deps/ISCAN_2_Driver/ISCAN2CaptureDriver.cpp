#include "ISCAN2CaptureDriver.h"

using namespace cv;

HMODULE CrossMatchIrisCap::dllbba_iscan = 0;
HMODULE CrossMatchIrisCap::dllCmtIris = 0;
CrossMatchIrisCap::CM_API CrossMatchIrisCap::API;

CrossMatchIrisCap::CrossMatchIrisCap()
{
	if (!CrossMatchIrisCap::dllbba_iscan) {
		CrossMatchIrisCap::dllbba_iscan = ::LoadLibraryA("bba_iscan.dll");
		if (!CrossMatchIrisCap::dllbba_iscan) {
			throw runtime_error("No se pudo cargar bba_iscan.dll");
		}

		CrossMatchIrisCap::dllCmtIris = ::LoadLibraryA("CmtIris.dll");
		if (!CrossMatchIrisCap::dllCmtIris) {
			throw runtime_error("No se pudo cargar CmtIris.dll");
		}

		loadAPI();
		API.BioB_Open();
	}
}

void CrossMatchIrisCap::loadAPI()
{
	CrossMatchIrisCap::API.BioB_Open = CrossMatchIrisCap::loadFromDLL<BIOB_OPEN*>(CrossMatchIrisCap::dllbba_iscan, "BioB_Open");
	CrossMatchIrisCap::API.BioB_Close = CrossMatchIrisCap::loadFromDLL<BIOB_CLOSE*>(dllbba_iscan, "BioB_Close");
	CrossMatchIrisCap::API.BioB_GetAPIProperties = CrossMatchIrisCap::loadFromDLL<BIOB_GETAPIPROPERTIES*>(dllbba_iscan, "BioB_GetAPIProperties");
	CrossMatchIrisCap::API.BioB_GetDeviceCount = CrossMatchIrisCap::loadFromDLL<BIOB_GETDEVICECOUNT*>(dllbba_iscan, "BioB_GetDeviceCount");
	CrossMatchIrisCap::API.BioB_GetDevicesInfo = CrossMatchIrisCap::loadFromDLL<BIOB_GETDEVICESINFO*>(dllbba_iscan, "BioB_GetDevicesInfo");
	CrossMatchIrisCap::API.BioB_OpenDevice = CrossMatchIrisCap::loadFromDLL<BIOB_OPENDEVICE*>(dllbba_iscan, "BioB_OpenDevice");
	CrossMatchIrisCap::API.BioB_CloseDevice = CrossMatchIrisCap::loadFromDLL<BIOB_CLOSEDEVICE*>(dllbba_iscan, "BioB_CloseDevice");
	CrossMatchIrisCap::API.BioB_RegisterDeviceCallback = CrossMatchIrisCap::loadFromDLL<BIOB_REGISTERDEVCALLBCK*>(dllbba_iscan, "BioB_RegisterDeviceCallback");
	CrossMatchIrisCap::API.BioB_GetProperties = CrossMatchIrisCap::loadFromDLL<BIOB_GETPROPERTIES*>(dllbba_iscan, "BioB_GetProperties");
	CrossMatchIrisCap::API.BioB_GetProperty = CrossMatchIrisCap::loadFromDLL<BIOB_GETPROPERTY*>(dllbba_iscan, "BioB_GetProperty");
	CrossMatchIrisCap::API.BioB_SetProperty = CrossMatchIrisCap::loadFromDLL<BIOB_SETPROPERTY*>(dllbba_iscan, "BioB_SetProperty");
	CrossMatchIrisCap::API.BioB_CancelAcquisition = CrossMatchIrisCap::loadFromDLL<BIOB_CANCELACQUISITION*>(dllbba_iscan, "BioB_CancelAcquisition");
	CrossMatchIrisCap::API.BioB_BeginAcquisitionProcess = CrossMatchIrisCap::loadFromDLL<BIOB_BEGINACQUISITION*>(dllbba_iscan, "BioB_BeginAcquisitionProcess");
	CrossMatchIrisCap::API.BioB_RequestAcquisitionOverride = CrossMatchIrisCap::loadFromDLL<BIOB_REQUESTACQUISITIONOVERRIDE*>(dllbba_iscan, "BioB_RequestAcquisitionOverride");
	CrossMatchIrisCap::API.BioB_SetVisualizationWindow = CrossMatchIrisCap::loadFromDLL<BIOB_SETVISUALIZATIONWINDOW*>(dllbba_iscan, "BioB_SetVisualizationWindow");
	CrossMatchIrisCap::API.BioB_Free = CrossMatchIrisCap::loadFromDLL<BIOB_FREE*>(dllbba_iscan, "BioB_Free");
	CrossMatchIrisCap::API.BioB_IsDeviceAcquiring = CrossMatchIrisCap::loadFromDLL<BIOB_ISDEVICEACQUIRING*>(dllbba_iscan, "BioB_IsDeviceAcquiring");
	CrossMatchIrisCap::API.BioB_IsDeviceOpened = CrossMatchIrisCap::loadFromDLL<BIOB_ISDEVICEOPENED*>(dllbba_iscan, "BioB_IsDeviceOpened");
	CrossMatchIrisCap::API.BioB_IsDeviceReady = CrossMatchIrisCap::loadFromDLL<BIOB_ISDEVICEREADY*>(dllbba_iscan, "BioB_IsDeviceReady");

	CrossMatchIrisCap::API.CmtIris_create = CrossMatchIrisCap::loadFromDLL<CMTIRIS_CREATE*>(CrossMatchIrisCap::dllCmtIris, "cmtiris_create");
	CrossMatchIrisCap::API.CmtIris_decode = CrossMatchIrisCap::loadFromDLL<CMTIRIS_DECODE*>(CrossMatchIrisCap::dllCmtIris, "cmtiris_decode");
	CrossMatchIrisCap::API.CmtIris_encode = CrossMatchIrisCap::loadFromDLL<CMTIRIS_ENCODE*>(CrossMatchIrisCap::dllCmtIris, "cmtiris_encode");
	CrossMatchIrisCap::API.CmtIris_encode_to_bmp = CrossMatchIrisCap::loadFromDLL<CMTIRIS_ENCODE_TO_BMP*>(CrossMatchIrisCap::dllCmtIris, "cmtiris_encode_to_bmp");
	CrossMatchIrisCap::API.CmtIris_set_raster = CrossMatchIrisCap::loadFromDLL<CMTIRIS_SET_RASTER*>(CrossMatchIrisCap::dllCmtIris, "cmtiris_set_raster");
	CrossMatchIrisCap::API.CmtIris_get_property = CrossMatchIrisCap::loadFromDLL<CMTIRIS_GET_PROPERTY*>(CrossMatchIrisCap::dllCmtIris, "cmtiris_get_property");
	CrossMatchIrisCap::API.CmtIris_free = CrossMatchIrisCap::loadFromDLL<CMTIRIS_FREE*>(CrossMatchIrisCap::dllCmtIris, "cmtiris_free");

}

vector<wstring> CrossMatchIrisCap::getDevices() const
{
	//TODO: parse each element returned in the XML separately
	wchar_t* buffer = NULL;
	BioBErrorCode errCode = API.BioB_GetDevicesInfo(&buffer);

	vector<wstring> res(1);
	res[0] = buffer;
	return res;
}

CrossMatchIrisCap::CaptureDevice CrossMatchIrisCap::openDevice(const wstring& deviceID)
{
	BioBErrorCode errCode = API.BioB_OpenDevice(deviceID.c_str(), false);
	return CaptureDevice(deviceID);
}

Mat1b CrossMatchIrisCap::CaptureDevice::decodeBMP(const unsigned char* buffer)
{
	LPBITMAPFILEHEADER lpbmfh = (LPBITMAPFILEHEADER) buffer;
	LPBITMAPINFO       lpBMI = (LPBITMAPINFO)((LPBYTE)lpbmfh + sizeof(BITMAPFILEHEADER));
	LPBYTE             lpCaptureBits = (LPBYTE)((LPBYTE)lpbmfh + lpbmfh->bfOffBits);

	int width = lpBMI->bmiHeader.biWidth;
	int height = lpBMI->bmiHeader.biHeight;
	unsigned step = width + (width % 4);

	Mat1b res = Mat1b(height, width, lpCaptureBits, step).clone();
	flip(res, res, 0);
	return res;
}

int CrossMatchIrisCap::CaptureDevice::previewCallbackWrapper(const wchar_t* deviceID_, const BioBData* cimageData)
{
	assert(deviceID.compare(deviceID_) == 0);

	BioBData* imageData = const_cast<BioBData*>(cimageData);		// Removes const cast
	if (imageData->FormatType != BIOB_IIR) {
		return 0;			// Not iris data
	}

	cmtiris_error irisError;
	CMT_IRIS_RECORD obj;
	irisError = (cmtiris_error) API.CmtIris_create(&obj);
	irisError = (cmtiris_error) API.CmtIris_decode(obj, (unsigned char*)imageData->Buffer, imageData->BufferSize);

	int length;
	// Attempt to get right eye
	cmtiris_position eyepos = CMTIIR_POSITION_RIGHT;
	Mat1b previewImage;
	unsigned char* buffer;
	irisError = (cmtiris_error) API.CmtIris_encode_to_bmp(obj, eyepos, NULL, &length);
	if (irisError == CMTIIR_ERROR_OK && length) {			// Got right eye
		buffer = new unsigned char[length];
		irisError = (cmtiris_error) API.CmtIris_encode_to_bmp(obj, eyepos, buffer, &length);

		previewImage = decodeBMP(buffer);
		delete[] buffer;

		if (!previewCallback.empty()) {
			previewCallback(previewImage);
		}
	}

	// Attempt to get left eye
	eyepos = CMTIIR_POSITION_LEFT;
	irisError = (cmtiris_error) API.CmtIris_encode_to_bmp(obj, eyepos, NULL, &length);
	if (irisError == CMTIIR_ERROR_OK && length) {			// Got left eye
		buffer = new unsigned char[length];
		irisError = (cmtiris_error) API.CmtIris_encode_to_bmp(obj, eyepos, buffer, &length);

		previewImage = decodeBMP(buffer);
		delete[] buffer;

		if (!previewCallback.empty()) {
			previewCallback(previewImage);
		}
	}

	return 0;
}

int CrossMatchIrisCap::CaptureDevice::dataAvailableCallbackWrapper(const wchar_t* deviceID_,  const BioBErrorCode dataStatus, const BioBData* cimageData, const int detectedObjects)
{
	assert(deviceID.compare(deviceID_) == 0);

	BioBData* imageData = const_cast<BioBData*>(cimageData);		// Removes const cast
	if (imageData->FormatType != BIOB_IIR) {
		return 0;			// Not iris data
	}

	cmtiris_error irisError;
	CMT_IRIS_RECORD obj;
	
	irisError = (cmtiris_error) API.CmtIris_create(&obj);
	irisError = (cmtiris_error) API.CmtIris_decode(obj, (unsigned char*)imageData->Buffer, imageData->BufferSize);	

	int length;
	// Attempt to get right eye
	cmtiris_position eyepos = CMTIIR_POSITION_RIGHT;
	Mat1b previewImage;
	unsigned char* buffer;
	irisError = (cmtiris_error) API.CmtIris_encode_to_bmp(obj, eyepos, NULL, &length);
	if (irisError == CMTIIR_ERROR_OK && length) {			// Got right eye
		buffer = new unsigned char[length];
		irisError = (cmtiris_error) API.CmtIris_encode_to_bmp(obj, eyepos, buffer, &length);

		previewImage = decodeBMP(buffer);
		delete[] buffer;

		if (!dataAvailableCallback.empty()) {
			dataAvailableCallback(previewImage);
		}
	}

	// Attempt to get left eye
	eyepos = CMTIIR_POSITION_LEFT;
	irisError = (cmtiris_error) API.CmtIris_encode_to_bmp(obj, eyepos, NULL, &length);
	if (irisError == CMTIIR_ERROR_OK && length) {			// Got left eye
		buffer = new unsigned char[length];
		irisError = (cmtiris_error) API.CmtIris_encode_to_bmp(obj, eyepos, buffer, &length);

		previewImage = decodeBMP(buffer);
		delete[] buffer;

		if (!dataAvailableCallback.empty()) {
			dataAvailableCallback(previewImage);
		}
	}

	return 0;
}

// Callback wrappers
int __stdcall __previewCallbackWrapper(const wchar_t* deviceID, const void* context, const BioBData* imageData)
{
	CrossMatchIrisCap::CaptureDevice* dev = const_cast<CrossMatchIrisCap::CaptureDevice*>( (CrossMatchIrisCap::CaptureDevice*)context);
	return dev->previewCallbackWrapper(deviceID, imageData);
}

int __stdcall __dataAvailableCallbackWrapper(const wchar_t* deviceID, const void* context, const BioBErrorCode dataStatus, const BioBData* imageData, const int detectedObjects)
{
	CrossMatchIrisCap::CaptureDevice* dev = const_cast<CrossMatchIrisCap::CaptureDevice*>( (CrossMatchIrisCap::CaptureDevice*)context);
	return dev->dataAvailableCallbackWrapper(deviceID, dataStatus, imageData, detectedObjects);
}

int __stdcall __imageQualityCallbackWrapper(const wchar_t* deviceID, const void *context, const BioBObjectQualityState *qualityStateArray, const int qualityCnt)
{
	return 0;
}

void CrossMatchIrisCap::CaptureDevice::beginAcquisition(BioBPositionTypes posType, BioBImpressionTypes impType)
{
	BioBErrorCode errCode;
	errCode = API.BioB_RegisterDeviceCallback(deviceID.c_str(), this, BIOB_PREVIEW, ::__previewCallbackWrapper);
	errCode = API.BioB_RegisterDeviceCallback(deviceID.c_str(), this, BIOB_DATA_AVAILABLE, ::__dataAvailableCallbackWrapper);
	errCode = API.BioB_RegisterDeviceCallback(deviceID.c_str(), this, BIOB_OBJECT_QUALITY, __imageQualityCallbackWrapper);
	API.BioB_BeginAcquisitionProcess(deviceID.c_str(), posType, impType);
}