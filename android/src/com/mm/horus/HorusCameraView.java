package com.mm.horus;

import org.opencv.android.Utils;
import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.imgproc.Imgproc;

import android.content.Context;
import android.graphics.Bitmap;
import android.util.Log;
import android.view.SurfaceHolder;

final class HorusCameraView extends CameraViewBase {
	private static final String TAG = "Horus::HorusCameraView";

	public HorusCameraView(Context context) {
		super(context);
		Log.d(TAG, "Instantiated new " + this.getClass());
	}
	
	@Override
	public void surfaceChanged(SurfaceHolder holder, int format, int width, int height) {
		super.surfaceChanged(holder, format, width, height);
	}
	
	@Override
	protected Bitmap processFrame(byte[] data) {
		Log.d(TAG, "Processing frame...");

		Mat mYUV = new Mat(frameHeight + frameHeight/2, frameWidth, CvType.CV_8UC1);
		Mat mRGBA = new Mat();
		
		mYUV.put(0, 0, data);
		Imgproc.cvtColor(mYUV, mRGBA, Imgproc.COLOR_YUV420sp2RGB, 4);
		
		Bitmap bmp = Bitmap.createBitmap(frameWidth, frameHeight, Bitmap.Config.ARGB_8888);
		if (Utils.MatToBitmap(mRGBA, bmp)) {
			return bmp;
		}
		
		bmp.recycle();
		return null;
	}
}