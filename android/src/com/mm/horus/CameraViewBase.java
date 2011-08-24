package com.mm.horus;

import java.io.IOException;
import java.util.List;

import android.content.Context;
import android.graphics.Bitmap;
import android.graphics.Canvas;
import android.hardware.Camera;
import android.hardware.Camera.PreviewCallback;
import android.util.Log;
import android.view.SurfaceHolder;
import android.view.SurfaceView;

public abstract class CameraViewBase extends SurfaceView implements SurfaceHolder.Callback, Runnable {
	private static final String TAG = "Horus::CameraViewBase";
	
	private Camera camera;
	private SurfaceHolder surfaceHolder;
	private byte[] frameData;
	private boolean runThread = false;

	protected int frameWidth = 0;
	protected int frameHeight = 0;

	public CameraViewBase(Context context) {
		super(context);
		
		surfaceHolder = this.getHolder();
		surfaceHolder.addCallback(this);
		
		Log.d(TAG, "Instantiated new " + this.getClass());
	}
	
	public void surfaceChanged(SurfaceHolder holder, int format, int width, int height) {
		Log.d(TAG, "surfaceChanged");
		
		if (camera == null) {
			return;
		}
		
		Camera.Parameters parameters = camera.getParameters();
		
		// Find out the size that best fits the surface
		int minDiff = Integer.MAX_VALUE;
		List<Camera.Size> sizes = parameters.getSupportedPreviewSizes();
		for (Camera.Size size : sizes) {
			if (size.height < height && size.width < width && (height-size.height) < minDiff) {
				minDiff = height-size.height;
				frameWidth = size.width;
				frameHeight = size.height;
			}
		}
		
		assert (frameWidth > 0 && frameHeight > 0);
		
		parameters.setPreviewSize(frameWidth, frameHeight);
		try {
			camera.setParameters(parameters);
			camera.setPreviewDisplay(null);
			//camera.setPreviewDisplay(holder);
			camera.startPreview();
		} catch (IOException e) {
			Log.e(TAG, "Error setting camera preview: " + e);
		}
		
	}
	
	public void surfaceCreated(SurfaceHolder holder) {
		Log.d(TAG, "surfaceCreated");
		
		camera = Camera.open();
		if (camera == null) {
			Log.e(TAG, "Couldn't open camera");
			return;
		}
		
		camera.setPreviewCallback(new PreviewCallback() {
			public void onPreviewFrame(byte[] data, Camera camera) {
				synchronized (CameraViewBase.this) {
					frameData = data;
					CameraViewBase.this.notify();
				}
			}
		});
		
		Thread thread = new Thread(this);
		thread.start();
		
	}

	public void surfaceDestroyed(SurfaceHolder holder) {
		Log.d(TAG, "surfaceDestroyed");

		// Cleanup
		runThread = false;
		
		if (camera == null) {
			return;
		}
		
		synchronized (this) {
			camera.stopPreview();
			camera.setPreviewCallback(null);
			camera.release();
			camera = null;
		}
	}
	
	protected abstract Bitmap processFrame(byte[] data);
	
	public void run() {
		// Process the frame
		runThread = true;
		Log.d(TAG, "Processing frame thread");
		
		while (runThread) {
			Bitmap bmp = null;
			
			synchronized (this) {
				try {
					Log.d(TAG, "Waiting frame data...");
					this.wait();
					Log.d(TAG, "Got frame data.");
					bmp = processFrame(frameData);

					// Draw the result
					Canvas canvas = surfaceHolder.lockCanvas();
					if (canvas != null) { 
						canvas.drawBitmap(bmp, (canvas.getWidth()-frameWidth)/2, (canvas.getHeight()-frameHeight)/2, null);
						surfaceHolder.unlockCanvasAndPost(canvas);
					}
					
					bmp.recycle();
				} catch (InterruptedException e) {
					e.printStackTrace();
				}
			}
		}
	}
}