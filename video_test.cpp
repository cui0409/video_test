#include "stdafx.h"
#include "video_test.h"
#include "E:\opencv\opencv\build\include\opencv2/opencv.hpp"
#include "E:\opencv\opencv\build\include\opencv2/video/video.hpp"
#include "E:\opencv\opencv\build\include\opencv2\core\core.hpp"
#include "E:\opencv\opencv\build\include\opencv2\highgui\highgui.hpp"
#include "E:\opencv\opencv\build\include\opencv2\imgproc\imgproc.hpp"
#include "E:\opencv\opencv\build\include\opencv2/imgproc\imgproc_c.h"
#include "E:\opencv\opencv\build\include\opencv2\ml.hpp"
#include <windows.h>
#include <iostream>

using namespace cv;
using namespace std;

void save_videoFile()
{
	string outputVideoPath = "F:\\testvideo\\testvideo.mpeg"; // 文件的保存位置

	VideoCapture capture(1);//0打开的是自带的摄像头，1 打开外接的相机

	double rate = 25.0;//视频的帧率

	int width = capture.get(CAP_PROP_FRAME_WIDTH);
	int height = capture.get(CAP_PROP_FRAME_HEIGHT);
	Size videoSize(width, height);

	VideoWriter writer;
	writer.open(outputVideoPath, CAP_OPENCV_MJPEG, rate, videoSize);

	Mat frame;

	//if (!writer.isOpened())
	//	return -1;
	int delay = 30;
	while (capture.isOpened())
	{
		capture >> frame;
		writer << frame;

		imshow("video", frame);
		//waitKey(30);
		
		namedWindow("video", WINDOW_AUTOSIZE);

		if (waitKey(25) == 27)//键盘摁下esc退出播放
			break;

		//if (delay >= 0 && waitKey(delay) >= 0)//按一下ESC就暂停或播放
		//	waitKey(0);

	}


	writer.release();

	//system("pause");//Press   any   key   to   exit
}

void Video_To_Image(string filename)
{
	cv::VideoCapture capture(filename);
	if (!capture.isOpened())
	{
		cout << "open video error";
	}
	/*CV_CAP_PROP_POS_MSEC – 视频的当前位置（毫秒）
	CV_CAP_PROP_POS_FRAMES – 视频的当前位置（帧）
	CV_CAP_PROP_FRAME_WIDTH – 视频流的宽度
	CV_CAP_PROP_FRAME_HEIGHT – 视频流的高度
	CV_CAP_PROP_FPS – 帧速率（帧 / 秒）*/
	int frame_width = (int)capture.get(CAP_PROP_FRAME_WIDTH);
	int frame_height = (int)capture.get(CAP_PROP_FRAME_HEIGHT);
	float frame_fps = capture.get(CAP_PROP_FPS);
	int frame_number = capture.get(CAP_PROP_FRAME_COUNT);//总帧数  
	frame_number = 10;//暂时先只取前10帧图片
	cout << "frame_width is " << frame_width << endl;
	cout << "frame_height is " << frame_height << endl;
	cout << "frame_fps is " << frame_fps << endl;

	int num = 0;//统计帧数  
	cv::Mat img;
	string img_name;
	char image_name[40];
	//cv::namedWindow("MyVideo", WINDOW_AUTOSIZE);
	while (true)
	{
		cv::Mat frame;
		//从视频中读取一帧  
		bool bSuccess = capture.read(frame);
		if (!bSuccess)
		{
			break;
		}
		//在MyVideo窗口上显示当前帧  
		//imshow("MyVideo", frame);
		//保存的图片名  
		sprintf_s(image_name, "%s%d%s", "F:\\testpics\\image", ++num, ".jpg");
		img_name = image_name;
		imwrite(img_name, frame);//保存一帧图片  

		if (cv::waitKey(30) == 27 || num == frame_number)
		{
			break;
		}
	}
}


int APIENTRY wWinMain(_In_ HINSTANCE hInstance,  _In_opt_ HINSTANCE hPrevInstance, _In_ LPWSTR    lpCmdLine, _In_ int       nCmdShow)
{
	string folder_pics;//图片文件夹路径
	folder_pics = "F:\\testpics/*.jpg";
	vector<String> image_files;//所有图片

	string folder_videos;//视频文件夹路径
	folder_videos = "F:\\testvideo/*.mpeg";
	vector<String> video_files;//所有视频


	//每次保存之前，清空视频和图片文件，遍历文件夹删除所有
	cv::glob(folder_pics, image_files);
	for (int i = 0; i < image_files.size(); i++)
	/*	remove(image_files[i]);*/


	cv::glob(folder_videos, video_files);
	for (int i = 0; i<video_files.size(); i++)
		/*remove(video_files[i]);*/









	save_videoFile();//保存视频文件

	string local_FileName = "F:\\testvideo\\testvideo.mpeg";

	Video_To_Image(local_FileName); //视频转图片  
	 
	// 获取图片以处理
	cv::glob(folder_pics, image_files);
	for (int ii = 0; ii<image_files.size(); ii++)
	{
		cout << image_files[ii] << endl;
		//namedWindow("image");
		Mat dd = imread(image_files[ii]);
		//imshow("image", dd);
	}

	//根据提取的视频帧，判断是否正常（1.正确性比对：看内容每一帧比较  2.错误性比对：黑屏，雪花）
	bool bNormal = true;

	if (image_files.empty())
		bNormal = false;//没获取到图片肯定是视频不正常的

	if (bNormal)
		MessageBox(NULL, TEXT("视频正常"), TEXT("结果"), MB_DEFBUTTON1 | MB_DEFBUTTON2);
	else
		MessageBox(NULL, TEXT("视频不正常"), TEXT("结果"), MB_DEFBUTTON1 | MB_DEFBUTTON2);

	return 0;
}


