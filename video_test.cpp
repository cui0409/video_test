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
	string outputVideoPath = "F:\\testvideo\\testvideo.mpeg"; // �ļ��ı���λ��

	VideoCapture capture(1);//0�򿪵����Դ�������ͷ��1 ����ӵ����

	double rate = 25.0;//��Ƶ��֡��

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

		if (waitKey(25) == 27)//��������esc�˳�����
			break;

		//if (delay >= 0 && waitKey(delay) >= 0)//��һ��ESC����ͣ�򲥷�
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
	/*CV_CAP_PROP_POS_MSEC �C ��Ƶ�ĵ�ǰλ�ã����룩
	CV_CAP_PROP_POS_FRAMES �C ��Ƶ�ĵ�ǰλ�ã�֡��
	CV_CAP_PROP_FRAME_WIDTH �C ��Ƶ���Ŀ��
	CV_CAP_PROP_FRAME_HEIGHT �C ��Ƶ���ĸ߶�
	CV_CAP_PROP_FPS �C ֡���ʣ�֡ / �룩*/
	int frame_width = (int)capture.get(CAP_PROP_FRAME_WIDTH);
	int frame_height = (int)capture.get(CAP_PROP_FRAME_HEIGHT);
	float frame_fps = capture.get(CAP_PROP_FPS);
	int frame_number = capture.get(CAP_PROP_FRAME_COUNT);//��֡��  
	frame_number = 10;//��ʱ��ֻȡǰ10֡ͼƬ
	cout << "frame_width is " << frame_width << endl;
	cout << "frame_height is " << frame_height << endl;
	cout << "frame_fps is " << frame_fps << endl;

	int num = 0;//ͳ��֡��  
	cv::Mat img;
	string img_name;
	char image_name[40];
	//cv::namedWindow("MyVideo", WINDOW_AUTOSIZE);
	while (true)
	{
		cv::Mat frame;
		//����Ƶ�ж�ȡһ֡  
		bool bSuccess = capture.read(frame);
		if (!bSuccess)
		{
			break;
		}
		//��MyVideo��������ʾ��ǰ֡  
		//imshow("MyVideo", frame);
		//�����ͼƬ��  
		sprintf_s(image_name, "%s%d%s", "F:\\testpics\\image", ++num, ".jpg");
		img_name = image_name;
		imwrite(img_name, frame);//����һ֡ͼƬ  

		if (cv::waitKey(30) == 27 || num == frame_number)
		{
			break;
		}
	}
}


int APIENTRY wWinMain(_In_ HINSTANCE hInstance,  _In_opt_ HINSTANCE hPrevInstance, _In_ LPWSTR    lpCmdLine, _In_ int       nCmdShow)
{
	string folder_pics;//ͼƬ�ļ���·��
	folder_pics = "F:\\testpics/*.jpg";
	vector<String> image_files;//����ͼƬ

	string folder_videos;//��Ƶ�ļ���·��
	folder_videos = "F:\\testvideo/*.mpeg";
	vector<String> video_files;//������Ƶ


	//ÿ�α���֮ǰ�������Ƶ��ͼƬ�ļ��������ļ���ɾ������
	cv::glob(folder_pics, image_files);
	for (int i = 0; i < image_files.size(); i++)
	/*	remove(image_files[i]);*/


	cv::glob(folder_videos, video_files);
	for (int i = 0; i<video_files.size(); i++)
		/*remove(video_files[i]);*/









	save_videoFile();//������Ƶ�ļ�

	string local_FileName = "F:\\testvideo\\testvideo.mpeg";

	Video_To_Image(local_FileName); //��ƵתͼƬ  
	 
	// ��ȡͼƬ�Դ���
	cv::glob(folder_pics, image_files);
	for (int ii = 0; ii<image_files.size(); ii++)
	{
		cout << image_files[ii] << endl;
		//namedWindow("image");
		Mat dd = imread(image_files[ii]);
		//imshow("image", dd);
	}

	//������ȡ����Ƶ֡���ж��Ƿ�������1.��ȷ�Աȶԣ�������ÿһ֡�Ƚ�  2.�����Աȶԣ�������ѩ����
	bool bNormal = true;

	if (image_files.empty())
		bNormal = false;//û��ȡ��ͼƬ�϶�����Ƶ��������

	if (bNormal)
		MessageBox(NULL, TEXT("��Ƶ����"), TEXT("���"), MB_DEFBUTTON1 | MB_DEFBUTTON2);
	else
		MessageBox(NULL, TEXT("��Ƶ������"), TEXT("���"), MB_DEFBUTTON1 | MB_DEFBUTTON2);

	return 0;
}


