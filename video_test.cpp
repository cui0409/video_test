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


//int APIENTRY wWinMain(_In_ HINSTANCE hInstance,  _In_opt_ HINSTANCE hPrevInstance, _In_ LPWSTR    lpCmdLine, _In_ int       nCmdShow)
//{
//	string folder_pics = "F:\\testpics/*.jpg";//图片文件夹路径
//	vector<String> image_files;//所有图片
//
//	string folder_videos = "F:\\testvideo/*.mpeg";//视频文件夹路径
//	vector<String> video_files;//所有视频
//
//	//每次保存之前，清空视频和图片文件
//	system("del F:\\testpics/*.jpg");
//	system("del F:\\testvideo/*.mpeg");
//
//	//保存视频文件
//	string outputVideoPath = "F:\\testvideo\\testvideo.mpeg"; // 文件的保存位置
//
//	VideoCapture capture(1);//0打开的是自带的摄像头，1 打开外接的相机
//	if (!capture.isOpened())
//	{
//		cout << "open video error";
//	}
//
//
//	double rate = capture.get(CAP_PROP_FPS);//获取视频帧率
//
//	int width = capture.get(CAP_PROP_FRAME_WIDTH);
//	int height = capture.get(CAP_PROP_FRAME_HEIGHT);
//	Size videoSize(width, height);
//
//	VideoWriter writer;
//	writer.open(outputVideoPath, CAP_OPENCV_MJPEG, rate, videoSize);
//
//	Mat frame;
//
//	int delay = 30;
//	while (capture.isOpened())
//	{
//		capture >> frame;
//		writer << frame;
//
//		imshow("video", frame);
//
//		namedWindow("video", WINDOW_AUTOSIZE);
//
//		if (waitKey(25) == 27)//键盘摁下esc退出播放
//			break;
//
//	}
//	writer.release();
//
//
//	//视频转图片  
//	int frame_width = (int)capture.get(CAP_PROP_FRAME_WIDTH);
//	int frame_height = (int)capture.get(CAP_PROP_FRAME_HEIGHT);
//	float frame_fps = capture.get(CAP_PROP_FPS);
//	int frame_number = capture.get(CAP_PROP_FRAME_COUNT);//总帧数  
//	frame_number = 10;//暂时先只取前10帧图片
//	cout << "frame_width is " << frame_width << endl;
//	cout << "frame_height is " << frame_height << endl;
//	cout << "frame_fps is " << frame_fps << endl;
//
//	int num = 0;//统计帧数  
//	cv::Mat img;
//	string img_name;
//	char image_name[40];
//	//cv::namedWindow("MyVideo", WINDOW_AUTOSIZE);
//	while (true)
//	{
//		cv::Mat frame;
//		//从视频中读取一帧  
//		bool bSuccess = capture.read(frame);
//		if (!bSuccess)
//		{
//			break;
//		}
//		//在MyVideo窗口上显示当前帧  
//		//imshow("MyVideo", frame);
//		//保存的图片名  
//		sprintf_s(image_name, "%s%d%s", "F:\\testpics\\image", ++num, ".jpg");
//		img_name = image_name;
//		imwrite(img_name, frame);//保存一帧图片  
//
//	}
//
//
//	// 获取其中一张图片以处理
//	Mat pic;
//	cv::glob(folder_pics, image_files);
//	if (image_files.empty())
//		return -1;
//	pic = imread(image_files[image_files.size() / 2]);//中间图片
//	imshow("image",pic);
//
//
//
//	//根据提取的视频帧，判断是否正常（1.正确性比对：看内容每一帧比较  2.错误性比对：黑屏，雪花）
//
//
//	//1.黑屏
//	//即全黑像素值都为0，Mat矩阵最大值为0
//
//	double minv = 0.0, maxv = 0.0;
//	double* minp = &minv;
//	double* maxp = &maxv;
//	minMaxIdx(pic, minp, maxp);
//	cout << "Mat minv = " << minv << endl;
//	if (maxv == 0.0)
//	{
//		MessageBox(NULL, TEXT("黑屏，视频不正常"), TEXT("结果"), MB_DEFBUTTON1 | MB_DEFBUTTON2);
//		return -1;
//	}
//
//
//	//2.雪花噪声
//
//
//
//
//	//其余为正常
//	MessageBox(NULL, TEXT("视频正常"), TEXT("结果"), MB_DEFBUTTON1 | MB_DEFBUTTON2);
//
//	return 0;
//}
//

double getPSNR(const Mat& I1, const Mat& I2);
Scalar getMSSIM(const Mat& I1, const Mat& I2);

double getPSNR(const Mat& I1, const Mat& I2)
{
	Mat s1;
	absdiff(I1, I2, s1);
	s1.convertTo(s1, CV_32F);
	s1 = s1.mul(s1);
	Scalar s = sum(s1);
	double sse = s.val[0] + s.val[1] + s.val[2];
	if (sse <= 1e-10)
		return 0;
	else
	{
		double mse = sse / (double)(I1.channels() * I1.total());
		double psnr = 10.0 * log10((255 * 255) / mse);
		return psnr;
	}
}
Scalar getMSSIM(const Mat& i1, const Mat& i2)
{
	const double C1 = 6.5025, C2 = 58.5225;

	int d = CV_32F;
	Mat I1, I2;
	i1.convertTo(I1, d);
	i2.convertTo(I2, d);
	Mat I2_2 = I2.mul(I2);
	Mat I1_2 = I1.mul(I1);
	Mat I1_I2 = I1.mul(I2);

	Mat mu1, mu2;
	GaussianBlur(I1, mu1, Size(11, 11), 1.5);
	GaussianBlur(I2, mu2, Size(11, 11), 1.5);
	Mat mu1_2 = mu1.mul(mu1);
	Mat mu2_2 = mu2.mul(mu2);
	Mat mu1_mu2 = mu1.mul(mu2);
	Mat sigma1_2, sigma2_2, sigma12;
	GaussianBlur(I1_2, sigma1_2, Size(11, 11), 1.5);
	sigma1_2 -= mu1_2;
	GaussianBlur(I2_2, sigma2_2, Size(11, 11), 1.5);
	sigma2_2 -= mu2_2;
	GaussianBlur(I1_I2, sigma12, Size(11, 11), 1.5);
	sigma12 -= mu1_mu2;
	Mat t1, t2, t3;
	t1 = 2 * mu1_mu2 + C1;
	t2 = 2 * sigma12 + C2;
	t3 = t1.mul(t2);
	t1 = mu1_2 + mu2_2 + C1;
	t2 = sigma1_2 + sigma2_2 + C2;
	t1 = t1.mul(t2);
	Mat ssim_map;
	divide(t3, t1, ssim_map);
	Scalar mssim = mean(ssim_map);
	return mssim;
}

int APIENTRY wWinMain(_In_ HINSTANCE hInstance, _In_opt_ HINSTANCE hPrevInstance, _In_ LPWSTR    lpCmdLine, _In_ int       nCmdShow)
{
	//string folder_videos = "F:\\testvideo/*.mpeg";//视频文件夹路径
	//vector<String> video_files;//所有视频
	
	//每次保存之前，清空视频文件
	//system("del F:\\testvideo/*.mpeg");
	
	//保存视频文件
	string outputVideoPath = "F:\\testvideo\\testvideo.mpeg"; // 文件的保存位置
	
	VideoCapture capture(1);
	if (!capture.isOpened())
	{
		cout << "open video error";
	}
	
	double rate = capture.get(CAP_PROP_FPS);//获取视频帧率
	
	int width = capture.get(CAP_PROP_FRAME_WIDTH);
	int height = capture.get(CAP_PROP_FRAME_HEIGHT);
	Size videoSize(width, height);
	
	VideoWriter writer;
	writer.open(outputVideoPath, CAP_OPENCV_MJPEG, rate, videoSize);//保存当前test视频

	
																	
	//开始逐帧比较2个的峰值信噪比



	stringstream conv;
	const string src_video = "F:\\Megamind.avi", test_video = "F:\\Megamind_bugy.avi";
	int psnrTriggerValue, delay = 30;
	conv >> psnrTriggerValue >> delay;

	int frameNum = -1;        
	VideoCapture capture_src(src_video), capture_test(test_video);
	if (!capture_src.isOpened())
	{
		cout << "can not open src video " << src_video << endl;
		return -1;
	}
	if (!capture_test.isOpened())
	{
		cout << "can not open test video " << test_video << endl;
		return -1;
	}

	Size refS = Size((int)capture_src.get(CAP_PROP_FRAME_WIDTH), (int)capture_src.get(CAP_PROP_FRAME_HEIGHT)),
		uTSi = Size((int)capture_test.get(CAP_PROP_FRAME_WIDTH), (int)capture_test.get(CAP_PROP_FRAME_HEIGHT));

	if (refS != uTSi)
	{
		cout << "Inputs have different size!!! Closing." << endl;
		return -1;
	}
	const char* win_test = "Test video";
	const char* win_src = "Src video";
	// Windows
	namedWindow(win_src, WINDOW_AUTOSIZE);
	namedWindow(win_test, WINDOW_AUTOSIZE);
	moveWindow(win_src, 400, 0);         
	moveWindow(win_test, refS.width, 0);        

	Mat frame_src, frame_test;
	double psnrV;
	vector<double> vec_psnrv;
	Scalar mssimV;
	for (;;) 
	{
		capture_src >> frame_src;
		capture_test >> frame_test;
		if (frame_src.empty() || frame_test.empty())
		{
			break;
		}
		++frameNum;
		cout << "Frame: " << frameNum << "# ";
		psnrV = getPSNR(frame_src, frame_test);
		vec_psnrv.push_back(psnrV);
		cout << setiosflags(ios::fixed) << setprecision(3) << psnrV << "dB";
		if (psnrV < psnrTriggerValue && psnrV)
		{
			mssimV = getMSSIM(frame_src, frame_test);
			cout << " MSSIM: "
				<< " R " << setiosflags(ios::fixed) << setprecision(2) << mssimV.val[2] * 100 << "%"
				<< " G " << setiosflags(ios::fixed) << setprecision(2) << mssimV.val[1] * 100 << "%"
				<< " B " << setiosflags(ios::fixed) << setprecision(2) << mssimV.val[0] * 100 << "%";
		}
		cout << endl;
		imshow(win_src, frame_src);
		imshow(win_test, frame_test);
		char c = (char)waitKey(delay);
		if (c == 27) break;
	}

	//判断峰值信噪比，所有值都为0，即视频源与测试视频的每帧图像都是相同的
	int totalNum = 0;//统计元素为0的个数
	for (size_t i = 0; i < vec_psnrv.size(); ++i)
	{
		if (vec_psnrv[i] == 0)
			totalNum++;
	}
	if(totalNum == vec_psnrv.size())//元素全部为0
		MessageBox(NULL, TEXT("视频正常，视频源与测试视频每帧图像比对都是相同"), TEXT("结果"), MB_DEFBUTTON1 | MB_DEFBUTTON2);
	else
		MessageBox(NULL, TEXT("视频不正常，视频源与测试视频每帧图像比对存在不同"), TEXT("结果"), MB_DEFBUTTON1 | MB_DEFBUTTON2);

	return 0;
}
