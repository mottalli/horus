#pragma once
#include <iostream>
#include <opencv2/opencv.hpp>
#include <Windows.h>
#include <boost/function.hpp>

#include "biob_defs.h"
#include "CmtIris.h"

typedef BioBErrorCode (BIOB_BEGINACQUISITION)(const wchar_t *pwDevID, const BioBPositionTypes PosType, const BioBImpressionTypes ImpType);
typedef BioBErrorCode (BIOB_CANCELACQUISITION)(const wchar_t *pwDevID);
typedef BioBErrorCode (BIOB_CLOSE)(void);
typedef BioBErrorCode (BIOB_CLOSEDEVICE)(const wchar_t *pwDevID);
typedef BioBErrorCode (BIOB_GETAPIPROPERTIES)(wchar_t **WcXml);
typedef BioBErrorCode (BIOB_GETDEVICECOUNT)(int *pdevices);
typedef BioBErrorCode (BIOB_GETPROPERTIES)(const wchar_t *pwDevID, wchar_t **WcXml);
typedef BioBErrorCode (BIOB_GETDEVICESINFO)(wchar_t **WcXml);
typedef BioBErrorCode (BIOB_GETPROPERTY)(const wchar_t *pwDevID, const wchar_t *pwPropertyName, wchar_t **pptrWval);
typedef BioBErrorCode (BIOB_ISDEVICEACQUIRING)(const wchar_t *pwDevID, BOOL *bLive);
typedef BioBErrorCode (BIOB_ISDEVICEOPENED)(const wchar_t *pwDevID, BOOL *bOpen);
typedef BioBErrorCode (BIOB_ISDEVICEREADY)(const wchar_t *pwDevID, BOOL *bReady);
typedef BioBErrorCode (BIOB_OPEN)(void);
typedef BioBErrorCode (BIOB_OPENDEVICE)(const wchar_t *pwDevID, const BOOL reset);
typedef BioBErrorCode (BIOB_REGISTERDEVCALLBCK)(const wchar_t *ptrDevID,void *ptrContext,BioBEvents Event,void *callback);
typedef BioBErrorCode (BIOB_REQUESTACQUISITIONOVERRIDE)(const wchar_t *pwDevID);
typedef BioBErrorCode (BIOB_SETPROPERTY)(const wchar_t *pwDevID, const wchar_t *pwPropertyName, const wchar_t *ptrWval);
typedef BioBErrorCode (BIOB_SETVISUALIZATIONWINDOW)(const wchar_t *pwDevID, const BB_WND_HND window, const wchar_t *pwBioBWindowModalType, const BioBOsType os);
typedef BioBErrorCode (BIOB_FREE)(const wchar_t *ptrWval);
typedef int (_stdcall CMTIRIS_CREATE)(CMT_IRIS_RECORD *);
typedef int (_stdcall CMTIRIS_DECODE)(CMT_IRIS_RECORD, unsigned char *, int);
typedef int (_stdcall CMTIRIS_ENCODE)(CMT_IRIS_RECORD, cmtiris_encoding_format, unsigned char *, int *);
typedef void (_stdcall CMTIRIS_FREE)(CMT_IRIS_RECORD);
typedef int (_stdcall CMTIRIS_ENCODE_TO_BMP)(CMT_IRIS_RECORD, cmtiris_position position, unsigned char *, int *);
typedef int (_stdcall CMTIRIS_SET_RASTER)(CMT_IRIS_RECORD, cmtiris_position position, int, int, unsigned char *);
typedef int (_stdcall CMTIRIS_GET_PROPERTY)(CMT_IRIS_RECORD, cmtiris_position position, cmtiris_property, int *);

class CrossMatchIrisCap
{
public:
	CrossMatchIrisCap();
	//~CrossMatchIrisCap() { API.BioB_Close(); }

	typedef std::runtime_error CaptureException;

	class CaptureDevice {
	public:
		CaptureDevice(wstring deviceID_) : deviceID(deviceID_) {
			//previewCallback = NULL;
			//dataAvailableCallback = NULL;
		}
		//~CaptureDevice() { API.BioB_CloseDevice(deviceID.c_str()); }

		//typedef void (*PREVIEW_CALLBACK_TYPE)(const cv::Mat1b&);
		//typedef void (*DATA_AVAILABLE_CALLBACK_TYPE)(const cv::Mat1b&);
		typedef boost::function<int (const cv::Mat1b&)> PREVIEW_CALLBACK_TYPE;
		typedef boost::function<int (const cv::Mat1b&)> DATA_AVAILABLE_CALLBACK_TYPE;
		
		
		// Callbacks
		void setPreviewCallback(PREVIEW_CALLBACK_TYPE callback) { previewCallback = callback; }
		void setDataAvailableCallback(DATA_AVAILABLE_CALLBACK_TYPE callback) { dataAvailableCallback = callback; }
		void beginAcquisition(BioBPositionTypes posType = BIOB_IRIS_BOTH, BioBImpressionTypes impType = BIOB_IRIS_IMP_REGULAR);

		int previewCallbackWrapper(const wchar_t* deviceID, const BioBData* imageData);
		int dataAvailableCallbackWrapper(const wchar_t* deviceID, const BioBErrorCode dataStatus, const BioBData* imageData, const int detectedObjects);
	protected:
		wstring deviceID;
		cv::Mat1b decodeBMP(const unsigned char* buffer);
		PREVIEW_CALLBACK_TYPE previewCallback;
		DATA_AVAILABLE_CALLBACK_TYPE dataAvailableCallback;
	};

	vector<wstring> getDevices() const;
	wstring getDeviceProperties(const wstring& deviceID);
	CaptureDevice openDevice(const wstring& deviceID);

protected:
	static HMODULE dllbba_iscan, dllCmtIris;
	
	template<class T> static inline T loadFromDLL(HMODULE dllInstance, const char* functionName) {
		T res = (T) ::GetProcAddress(dllInstance, functionName);
		if (!res) {
			throw runtime_error(string("Couldn't load function ") + string(functionName));
		}
		return res;
	}

	typedef struct {
		BIOB_OPEN* BioB_Open;
		BIOB_CLOSE* BioB_Close;
		BIOB_GETAPIPROPERTIES* BioB_GetAPIProperties;
		BIOB_GETDEVICECOUNT* BioB_GetDeviceCount;
		BIOB_GETDEVICESINFO* BioB_GetDevicesInfo;
		BIOB_OPENDEVICE* BioB_OpenDevice;
		BIOB_CLOSEDEVICE* BioB_CloseDevice;
		BIOB_REGISTERDEVCALLBCK* BioB_RegisterDeviceCallback;
		BIOB_GETPROPERTIES* BioB_GetProperties;
		BIOB_GETPROPERTY* BioB_GetProperty;
		BIOB_SETPROPERTY* BioB_SetProperty;
		BIOB_CANCELACQUISITION* BioB_CancelAcquisition;
		BIOB_BEGINACQUISITION* BioB_BeginAcquisitionProcess;
		BIOB_REQUESTACQUISITIONOVERRIDE* BioB_RequestAcquisitionOverride;
		BIOB_SETVISUALIZATIONWINDOW* BioB_SetVisualizationWindow;
		BIOB_FREE* BioB_Free;
		BIOB_ISDEVICEACQUIRING* BioB_IsDeviceAcquiring;
		BIOB_ISDEVICEOPENED* BioB_IsDeviceOpened;
		BIOB_ISDEVICEREADY* BioB_IsDeviceReady;
		CMTIRIS_CREATE* CmtIris_create;
		CMTIRIS_DECODE* CmtIris_decode;
		CMTIRIS_ENCODE* CmtIris_encode;
		CMTIRIS_ENCODE_TO_BMP* CmtIris_encode_to_bmp;
		CMTIRIS_SET_RASTER* CmtIris_set_raster;
		CMTIRIS_GET_PROPERTY* CmtIris_get_property;
		CMTIRIS_FREE* CmtIris_free;
	} CM_API;

	static CM_API API;

	static void loadAPI();
};
