package com.example.SignTracker;

import android.content.res.AssetFileDescriptor;
import android.content.res.AssetManager;
import android.graphics.Bitmap;
import android.util.Log;
import android.view.View;
import android.widget.Button;
import android.widget.TextView;

import org.opencv.android.Utils;
import org.opencv.core.Core;
import org.opencv.core.Mat;
import org.opencv.core.Point;
import org.opencv.core.Rect;
import org.opencv.core.Scalar;
import org.opencv.imgproc.Imgproc;
import org.tensorflow.lite.Interpreter;
import org.tensorflow.lite.gpu.GpuDelegate;

import java.io.BufferedReader;
import java.io.FileInputStream;
import java.io.IOException;
import java.io.InputStreamReader;
import java.lang.reflect.Array;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.channels.FileChannel;
import java.util.ArrayList;
import java.util.Map;
import java.util.TreeMap;

public class SignDetectorClass {
    private Interpreter handInterpreter, signInterpreter;
    private int INPUT_SIZE;
    private int SIGN_INPUT_SIZE;
    private GpuDelegate gpuDelegate;
    private int height=0;
    private  int width=0;
    String signPath="";
    private String finalWord="";
    private String currentWord="";
    private AssetManager assetManager;

    SignDetectorClass(Button addBtn, Button clearBtn, TextView word, AssetManager assetManager, String modelPath, int inputSize, String signTrackerPath, String signLabelPath, int signInputSize ) throws IOException{
        INPUT_SIZE=inputSize;
        SIGN_INPUT_SIZE=signInputSize;

        signPath = signLabelPath;
        this.assetManager = assetManager;

        Interpreter.Options handOptions=new Interpreter.Options();
        handOptions.setNumThreads(4);
        gpuDelegate=new GpuDelegate();
        handOptions.addDelegate(gpuDelegate);
        handInterpreter=new Interpreter(loadModelFile(assetManager,modelPath),handOptions);

        Interpreter.Options signOptions=new Interpreter.Options();
        signOptions.setNumThreads(2);
        signInterpreter=new Interpreter(loadModelFile(assetManager,signTrackerPath),signOptions);

        clearBtn.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View view) {
                finalWord = "";
                word.setText(finalWord);
            }
        });

        addBtn.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View view) {
                finalWord = finalWord + currentWord;
                word.setText(finalWord);
            }
        });

    }

    private ByteBuffer loadModelFile(AssetManager assetManager, String modelPath) throws IOException {
        // use to get description of file
        AssetFileDescriptor fileDescriptor=assetManager.openFd(modelPath);
        FileInputStream inputStream=new FileInputStream(fileDescriptor.getFileDescriptor());
        FileChannel fileChannel=inputStream.getChannel();
        long startOffset =fileDescriptor.getStartOffset();
        long declaredLength=fileDescriptor.getDeclaredLength();

        return fileChannel.map(FileChannel.MapMode.READ_ONLY,startOffset,declaredLength);
    }

    public Mat recognizeImage(Mat mat_image){
        Mat rotated_mat_image=new Mat();

        Mat a=mat_image.t();
        Core.flip(a,rotated_mat_image,1);
        a.release();


        Bitmap bitmap=null;
        bitmap=Bitmap.createBitmap(rotated_mat_image.cols(),rotated_mat_image.rows(),Bitmap.Config.ARGB_8888);
        Utils.matToBitmap(rotated_mat_image,bitmap);

        height=bitmap.getHeight();
        width=bitmap.getWidth();

        Bitmap scaledBitmap=Bitmap.createScaledBitmap(bitmap,INPUT_SIZE,INPUT_SIZE,false);
        ByteBuffer byteBuffer=convertBitmapToByteBuffer(scaledBitmap, INPUT_SIZE);


        Object[] input=new Object[1];
        input[0]=byteBuffer;

        Map<Integer,Object> output_map=new TreeMap<>();

        float[][][]boxes =new float[1][10][4];
        // 10: top 10 object detected
        // 4: there coordinate in image
        float[][] scores=new float[1][10];
        // stores scores of 10 object
        float[][] classes=new float[1][10];
        // stores class of object
        output_map.put(0,boxes);
        output_map.put(1,classes);
        output_map.put(2,scores);

        handInterpreter.runForMultipleInputsOutputs(input,output_map);

        Object value=output_map.get(0);
        //Object Object_class=output_map.get(1); --> perché ho solo hand come label della mano
        Object score=output_map.get(2);

        for (int i=0;i<10;i++){
            float score_value=(float) Array.get(Array.get(score,0),i);

            if(score_value>0.5){
                Object box1=Array.get(Array.get(value,0),i);

                float y1=(float) Array.get(box1,0)*height;
                float x1=(float) Array.get(box1,1)*width;
                float y2=(float) Array.get(box1,2)*height;
                float x2=(float) Array.get(box1,3)*width;

                if(y1 < 0) {
                    y1 = 0;
                }
                if(x1 < 0) {
                    x1 = 0;
                }
                if(y2 > height) {
                    y2 = height;
                }
                if(x2 > width) {
                    x2 = width;
                }
                float w1 = x2 - x1;
                float h1 = y2 - y1;

                Rect cropped_roi = new Rect((int)x1, (int)y1, (int)w1, (int)h1);
                Mat cropped = new Mat(rotated_mat_image, cropped_roi).clone();
                Bitmap signBitmap = null;
                signBitmap = Bitmap.createBitmap(cropped.cols(), cropped.rows(), Bitmap.Config.ARGB_8888);
                Utils.matToBitmap(cropped, signBitmap);
                Bitmap signScaledBitmap = Bitmap.createScaledBitmap(signBitmap, SIGN_INPUT_SIZE, SIGN_INPUT_SIZE, false);
                ByteBuffer signByteBuffer = convertBitmapToByteBuffer(signScaledBitmap, SIGN_INPUT_SIZE);

                float[][] output_class_value = new float[1][1];
                signInterpreter.run(signByteBuffer, output_class_value);
                Log.d("SignDetectorClass", "output_class_value; "+output_class_value[0][0]);

                String letter = getLetter(assetManager, output_class_value[0][0]);

                currentWord = letter;

                Imgproc.putText(rotated_mat_image, ""+letter, new Point(x1+10,y1+40),2,1.5,new Scalar(255, 255, 255, 255),3);
                Imgproc.rectangle(rotated_mat_image,new Point(x1,y1),new Point(x2,y2),new Scalar(255, 0, 0, 0),3);
            }
        }

        Mat b=rotated_mat_image.t();
        Core.flip(b,mat_image,0);
        b.release();
        return mat_image;
    }

    private String getLetter(AssetManager assetManager, float v) {
        ArrayList<String> alphabet = new ArrayList<String>();
        String letter = "";

        try {
            BufferedReader reader=new BufferedReader(new InputStreamReader(assetManager.open(signPath)));
            String line;
            // loop through each line and store it to labelList
            while ((line=reader.readLine())!=null){
                alphabet.add(line);
            }

            reader.close();
        } catch (IOException e) {
            e.printStackTrace();
        }

        float startValue = -0.5f;
        for(int i = 0; i < alphabet.size(); i++) {
            if (v >= startValue+i & v < startValue+1+i) {
                return alphabet.get(i);
            }
        }
        return letter;
    }

    private ByteBuffer convertBitmapToByteBuffer(Bitmap bitmap, Integer size_images) {
        ByteBuffer byteBuffer;

        byteBuffer=ByteBuffer.allocateDirect(4*1*size_images*size_images*3);
        byteBuffer.order(ByteOrder.nativeOrder());
        int[] intValues=new int[size_images*size_images];
        bitmap.getPixels(intValues,0,bitmap.getWidth(),0,0,bitmap.getWidth(),bitmap.getHeight());
        int pixel=0;
        for (int i=0;i<size_images;++i){
            for (int j=0;j<size_images;++j){
                final  int val=intValues[pixel++];
                if(size_images==SIGN_INPUT_SIZE){
                    byteBuffer.putFloat(((val >> 16) & 0xFF));
                    byteBuffer.putFloat(((val >> 8) & 0xFF));
                    byteBuffer.putFloat(((val) & 0xFF));
                }
                else {
                    byteBuffer.putFloat((((val >> 16) & 0xFF))/255.0f);
                    byteBuffer.putFloat((((val >> 8) & 0xFF))/255.0f);
                    byteBuffer.putFloat((((val) & 0xFF))/255.0f);
                }
            }
        }
        return byteBuffer;
    }
}
