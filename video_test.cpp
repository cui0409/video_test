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

string get_time()
{
	SYSTEMTIME  st, lt;
	//GetSystemTime(&lt);
	GetLocalTime(&lt);

	char szResult[30] = "\0";

	sprintf_s(szResult, 30, "%d-%d-%d-%d-%d-%d-%d", lt.wYear, lt.wMonth, lt.wDay, lt.wHour, lt.wMinute, lt.wSecond, lt.wMilliseconds);

	return szResult;
}


//ģ����⣬���ԭͼ����ģ��ͼ�񣬷���0�����򷵻�1
//10������Խ�������һ��Ϊ5
int VideoBlurDetect(const cv::Mat &srcimg)
{
	cv::Mat img;
	cv::cvtColor(srcimg, img, CV_BGR2GRAY); // �������ͼƬתΪ�Ҷ�ͼ��ʹ�ûҶ�ͼ���ģ����

											//ͼƬÿ���ֽ�������  
	int width = img.cols;
	int height = img.rows;
	ushort* sobelTable = new ushort[width*height];
	memset(sobelTable, 0, width*height * sizeof(ushort));

	int i, j, mul;
	//ָ��ͼ���׵�ַ  
	uchar* udata = img.data;
	for (i = 1, mul = i*width; i < height - 1; i++, mul += width)
		for (j = 1; j < width - 1; j++)

			sobelTable[mul + j] = abs(udata[mul + j - width - 1] + 2 * udata[mul + j - 1] + udata[mul + j - 1 + width] - \
				udata[mul + j + 1 - width] - 2 * udata[mul + j + 1] - udata[mul + j + width + 1]);

	for (i = 1, mul = i*width; i < height - 1; i++, mul += width)
		for (j = 1; j < width - 1; j++)
			if (sobelTable[mul + j] < 50 || sobelTable[mul + j] <= sobelTable[mul + j - 1] || \
				sobelTable[mul + j] <= sobelTable[mul + j + 1]) sobelTable[mul + j] = 0;

	int totLen = 0;
	int totCount = 1;

	uchar suddenThre = 50;
	uchar sameThre = 3;
	//����ͼƬ  
	for (i = 1, mul = i*width; i < height - 1; i++, mul += width)
	{
		for (j = 1; j < width - 1; j++)
		{
			if (sobelTable[mul + j])
			{
				int   count = 0;
				uchar tmpThre = 5;
				uchar max = udata[mul + j] > udata[mul + j - 1] ? 0 : 1;

				for (int t = j; t > 0; t--)
				{
					count++;
					if (abs(udata[mul + t] - udata[mul + t - 1]) > suddenThre)
						break;

					if (max && udata[mul + t] > udata[mul + t - 1])
						break;

					if (!max && udata[mul + t] < udata[mul + t - 1])
						break;

					int tmp = 0;
					for (int s = t; s > 0; s--)
					{
						if (abs(udata[mul + t] - udata[mul + s]) < sameThre)
						{
							tmp++;
							if (tmp > tmpThre) break;
						}
						else break;
					}

					if (tmp > tmpThre) break;
				}

				max = udata[mul + j] > udata[mul + j + 1] ? 0 : 1;

				for (int t = j; t < width; t++)
				{
					count++;
					if (abs(udata[mul + t] - udata[mul + t + 1]) > suddenThre)
						break;

					if (max && udata[mul + t] > udata[mul + t + 1])
						break;

					if (!max && udata[mul + t] < udata[mul + t + 1])
						break;

					int tmp = 0;
					for (int s = t; s < width; s++)
					{
						if (abs(udata[mul + t] - udata[mul + s]) < sameThre)
						{
							tmp++;
							if (tmp > tmpThre) break;
						}
						else break;
					}

					if (tmp > tmpThre) break;
				}
				count--;

				totCount++;
				totLen += count;
			}
		}
	}
	//ģ����
	float result = (float)totLen / totCount;
	delete[] sobelTable;
	sobelTable = NULL;

	return result;
}

//ģ���жϣ������������ڣ��趨��ֵ10000��С�����ֵ������,���㷽��
bool isImageBlurry(cv::Mat& img)
{
	cv::Mat matImageGray;
	// converting image's color space (RGB) to grayscale
	cv::cvtColor(img, matImageGray, CV_BGR2GRAY);
	cv::Mat dst, abs_dst;
	cv::Laplacian(matImageGray, dst, CV_16S, 3);//������˹�任
	cv::convertScaleAbs(dst, abs_dst);
	cv::Mat tmp_m, tmp_sd;
	double m = 0, sd = 0;
	int threshold = 10000;//�Լ����õ���ֵ
	cv::meanStdDev(dst, tmp_m, tmp_sd);
	m = tmp_m.at<double>(0, 0);
	sd = tmp_sd.at<double>(0, 0);
	std::cout << "StdDev: " << sd * sd << std::endl;
	return ((sd * sd) <= threshold);
}

int APIENTRY wWinMain(_In_ HINSTANCE hInstance, _In_opt_ HINSTANCE hPrevInstance, _In_ LPWSTR    lpCmdLine, _In_ int       nCmdShow)
{
	char video_name[100];

	string time;
	time = get_time();
	sprintf_s(video_name, "%s%s%s", "F:\\testvideo\\testvideo", time.c_str(), ".mpeg");
	 
	VideoCapture capture(1);
	if (!capture.isOpened())
	{
		cout << "open video error";
		MessageBox(NULL, TEXT("�ɼ���Ƶ����"), TEXT("���"), MB_DEFBUTTON1 | MB_DEFBUTTON2);
		return -1;
	}

	double rate = capture.get(CAP_PROP_FPS);//��ȡ��Ƶ֡��

	int width = capture.get(CAP_PROP_FRAME_WIDTH);
	int height = capture.get(CAP_PROP_FRAME_HEIGHT);
	Size videoSize(width, height);

	VideoWriter writer;
	writer.open(video_name, CAP_OPENCV_MJPEG, rate, videoSize);

	//���������ʼ֡
	long frameToStart = 0;
	capture.set(CAP_PROP_POS_FRAMES, frameToStart);

	//�����������֡  //�ɼ�5��
	int frameToStop = 300;

	Mat frame;
	//ָ��ÿ�ζ���ȡ�̶�֡������Ƶ
	while (frameToStart < frameToStop)
	{
		capture >> frame;
		writer << frame;

		//imshow("video", frame);
		//namedWindow("video", WINDOW_AUTOSIZE);

		frameToStart++;
		//waitKey(1);
	}
	writer.release();
	//.... �����ǲɼ���Ƶ





	stringstream conv;
	//const string src_video = "F:\\testvideo\\src_video.mpeg";

	const string test_video = video_name;
	//const string test_video = "F:\\zhengchang.mpeg";
	//const string test_video = "F:\\kadun.mpeg";
	//const string test_video = "F:\\heiping.mpeg";
	//const string test_video = "F:\\huaping.mpeg";

	int psnrTriggerValue, delay = 30;
	conv >> psnrTriggerValue >> delay;

	VideoCapture /*capture_src(src_video),*/ capture_test(test_video);
	//if (!capture_src.isOpened())
	//{
	//	cout << "can not open src video " << src_video << endl;
	//	return -1;
	//}
	if (!capture_test.isOpened())
	{
		cout << "can not open test video " << test_video << endl;
		return -1;
	}

	//double rate_src = capture_src.get(CAP_PROP_FPS);
	double rate_test = capture_test.get(CAP_PROP_FPS);

	//int num_src = capture_src.get(CAP_PROP_FRAME_COUNT);//src��֡��
	int num_test = capture_test.get(CAP_PROP_FRAME_COUNT);//test��֡��

	//��ʾһ�±�����Ƶ  test_video
	int i = 0;
	Mat frame_show;
	while (i < num_test)
	{
		capture_test >> frame_show;

		imshow("video", frame_show);
		namedWindow("video", WINDOW_AUTOSIZE);

		i++;
		waitKey(1);
	}

	//Size refS = Size((int)capture_src.get(CAP_PROP_FRAME_WIDTH), (int)capture_src.get(CAP_PROP_FRAME_HEIGHT));
	Size uTSi = Size((int)capture_test.get(CAP_PROP_FRAME_WIDTH), (int)capture_test.get(CAP_PROP_FRAME_HEIGHT));

	//if (refS != uTSi)
	//{
	//	cout << "Inputs have different size!!! Closing." << endl;
	//	return -1;
	//}
	const char* win_test = "Test video";
	const char* win_src = "Src video";
	// Windows
	//namedWindow(win_src, WINDOW_AUTOSIZE);
	//namedWindow(win_test, WINDOW_AUTOSIZE);
	//moveWindow(win_src, 400, 0);
	//moveWindow(win_test, refS.width, 0);

	Mat frame_src, frame_test;
	double psnrV;
	vector<double> vec_psnrv;

	Mat mat_test, mat_src;



	int index_src = 0; //src��ǰ֡
	int totalNum = 0;//ͳ�Ʋ������ĸ���
	double psnrv_mat;
	double psnrv_mat2;

	//for (size_t i = 0; i < num_src / rate_src / 6; ++i)                      
	//{
	//	capture_src.set(CAP_PROP_POS_FRAMES, i * rate_src);//0��60��120��...
	//	capture_src >> mat_src;

	//	for (size_t j = 0; j < num_test / rate_test; ++j)                
	//	{
	//		capture_test.set(CAP_PROP_POS_FRAMES, j * rate_test);//0��60��120��...
	//		capture_test >> mat_test;


	//		psnrv_mat = getPSNR(mat_src, mat_test);//ֵԽ��ʧ��ԽС��20��40��
	//		vec_psnrv.push_back(psnrv_mat);

	//		//Scalar sc = getMSSIM(mat_src, mat_test);//ֵԽ��ʧ��ԽС����0��1��

	//		//if(sc[0] > 0.7 && sc[1] > 0.7 && sc[2] > 0.7)//����Ƚ�
	//		//	totalNum++;
	//	}

	//}
	////����Ԫ�أ�����Ԫ�ز�ֵ����ֵ����10��Ϊ������
	//for (size_t k = 0; k < vec_psnrv.size() - 1 ; ++k)
	//{
	//	if(abs(vec_psnrv[k + 1] - vec_psnrv[k]) > 10 || vec_psnrv[k] < 10)
	//		totalNum++;
	//}

	//���ȡtest��Ƶ3֡,ȡ��test��ǰ��֡
	Mat first_test;
	capture_test.set(CAP_PROP_POS_FRAMES, rate_test/2);
	capture_test >> first_test;

	Mat second_test;
	capture_test.set(CAP_PROP_POS_FRAMES, rate_test);
	capture_test >> second_test;

	Mat third_test;
	capture_test.set(CAP_PROP_POS_FRAMES, rate_test * 3 / 2);
	capture_test >> third_test;


	// 1.����ʾ������ ����ȫ������ֵ��Ϊ0��Mat�������ֵΪ0��
	double minv1 = 0.0, maxv1 = 0.0;
	double* minp1 = &minv1;
	double* maxp1 = &maxv1;
	minMaxIdx(first_test, minp1, maxp1);

	double minv2 = 0.0, maxv2 = 0.0;
	double* minp2 = &minv2;
	double* maxp2 = &maxv2;
	minMaxIdx(second_test, minp2, maxp2);

	double minv3 = 0.0, maxv3 = 0.0;
	double* minp3 = &minv3;
	double* maxp3 = &maxv3;
	minMaxIdx(third_test, minp3, maxp3);

	if (maxv1 == 0.0 && maxv2 == 0.0)
	{
		MessageBox(NULL, TEXT("������������"), TEXT("��Ƶ�����"), MB_DEFBUTTON1 | MB_DEFBUTTON2);
		return -1;
	}
	
	//2.���٣�1.һֱ����һ�����棬���⼸֡����С��ע����Խ��滹�����ڱ仯 2.�ü��뿨����ѭ����

	psnrv_mat = getPSNR(first_test, third_test);//����һ�����棬��ǰ�����2֮֡�����ƶȺܸ�
	if (psnrv_mat > 25 && psnrv_mat < 60 || psnrv_mat == 0)
	{
		MessageBox(NULL, TEXT("���٣�������"), TEXT("��Ƶ�����"), MB_DEFBUTTON1 | MB_DEFBUTTON2);
		return -1;
	}
	//else if(psnrv_mat2 > 25 && psnrv_mat2 < 40 || psnrv_mat2 == 0)
	//	MessageBox(NULL, TEXT("���٣�������"), TEXT("��Ƶ�����"), MB_DEFBUTTON1 | MB_DEFBUTTON2);



	//3.ѩ�����߻���(˺�ѻ��λ)         
	//����ͼ�����صĻҶ�ֵ�仯һ�㶼ƽ���������С����ѩ���ġ���˸�㡱���ػҶ�ֵ���ұ仯��
	//�Ҷ�ֵ��Ծ�Դ󣬼��㷽��Ҳƫ�󡣼��ѩ����˼·��С���ڷ����

	bool mohu = isImageBlurry(first_test);
	if(mohu)
	{
		MessageBox(NULL, TEXT("ģ����������"), TEXT("��Ƶ�����"), MB_DEFBUTTON1 | MB_DEFBUTTON2);
		return -1;
	}


	//����Ϊ����
	MessageBox(NULL, TEXT("����"), TEXT("��Ƶ�����"), MB_DEFBUTTON1 | MB_DEFBUTTON2);

	exit(0);

	return 0;
}
