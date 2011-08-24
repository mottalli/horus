package com.mm.horus;

import org.opencv.android.Utils;
import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.imgproc.Imgproc;

import android.content.Context;
import android.graphics.Bitmap;
import android.util.Log;
import android.view.SurfaceHolder;

class HorusCameraView extends CameraViewBase {
	private static final String TAG = "Horus::HorusCameraView";
	private Mat matYUV, matRGBA, matGray;

	public HorusCameraView(Context context) {
		super(context);
		Log.d(TAG, "Instantiated new " + this.getClass());
	}
	
	@Override
	public void surfaceChanged(SurfaceHolder holder, int format, int width, int height) {
		super.surfaceChanged(holder, format, width, height);
		
		synchronized (this) {
			matYUV = new Mat(frameHeight + frameHeight/2, frameWidth, CvType.CV_8UC1);		// Wrapper for the YUV matrix
			matGray = matYUV.submat(0, frameHeight, 0, frameWidth);			// Reference to the gray part of the YUV matrix
			matRGBA = new Mat();
		}
	}
	
    static {
        System.loadLibrary("horus_wrapper");
    }
	
	public native void doNativeProcess(long matInputGray, long matOutputRGB);
	
	@Override
	protected Bitmap processFrame(byte[] data) {
		Log.d(TAG, "Processing frame...");

		matYUV.put(0, 0, data);
		Imgproc.cvtColor(matYUV, matRGBA, Imgproc.COLOR_YUV420sp2RGB, 4);
		
			doNativeProcess(matGray.getNativeObjAddr(), matRGBA.getNativeObjAddr());
		
		Bitmap bmp = Bitmap.createBitmap(frameWidth, frameHeight, Bitmap.Config.ARGB_8888);
		if (!Utils.MatToBitmap(matRGBA, bmp)) {
			bmp.recycle();
			return null;
		}
		
		return bmp;
	}
	
    @Override
    public void run() {
        super.run();

        synchronized (this) {
            // Explicitly deallocate Mats
            if (matYUV != null)
            	matYUV.release();
            if (matRGBA != null)
            	matRGBA.release();
            if (matGray != null)
            	matGray.release();

            matYUV = null;
            matRGBA = null;
            matGray = null;
        }
    }
}