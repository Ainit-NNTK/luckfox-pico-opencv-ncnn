#include <assert.h>
#include <errno.h>
#include <fcntl.h>
#include <getopt.h>
#include <pthread.h>
#include <signal.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/poll.h>
#include <time.h>
#include <unistd.h>
#include <vector>

#include "rtsp_demo.h"
#include "luckfox_mpi.h"

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include "net.h"
#include "benchmark.h"
#include "cpu.h"

cv::Mat preprocess_digit_region(const cv::Mat region);
cv::Rect find_digit_contour(const cv::Mat &image);
int ncnn_run_inf(ncnn::Net &net,cv::Mat frame_resize);


int main(int argc, char *argv[])
{
	if (argc != 1)
	{
		printf("ERROR\n");
	}
	RK_S32 s32Ret = 0;

	int sX, sY, eX, eY;
	int width = 640;
	int height = 480;

	char fps_text[16];
	float fps = 0;
	memset(fps_text, 0, 16);

	// h264_frame
	VENC_STREAM_S stFrame;
	stFrame.pstPack = (VENC_PACK_S *)malloc(sizeof(VENC_PACK_S));
	VIDEO_FRAME_INFO_S h264_frame;
	VIDEO_FRAME_INFO_S stVpssFrame;

	// rkaiq init
	RK_BOOL multi_sensor = RK_FALSE;
	const char *iq_dir = "/etc/iqfiles";
	rk_aiq_working_mode_t hdr_mode = RK_AIQ_WORKING_MODE_NORMAL;
	// hdr_mode = RK_AIQ_WORKING_MODE_ISP_HDR2;
	SAMPLE_COMM_ISP_Init(0, hdr_mode, multi_sensor, iq_dir);
	SAMPLE_COMM_ISP_Run(0);

	// rkmpi init
	if (RK_MPI_SYS_Init() != RK_SUCCESS)
	{
		RK_LOGE("rk mpi sys init fail!");
		return -1;
	}

	// rtsp init
	rtsp_demo_handle g_rtsplive = NULL;
	rtsp_session_handle g_rtsp_session;
	g_rtsplive = create_rtsp_demo(554);
	g_rtsp_session = rtsp_new_session(g_rtsplive, "/live/0");
	rtsp_set_video(g_rtsp_session, RTSP_CODEC_ID_VIDEO_H264, NULL, 0);
	rtsp_sync_video_ts(g_rtsp_session, rtsp_get_reltime(), rtsp_get_ntptime());

	// vi init
	vi_dev_init();
	vi_chn_init(0, width, height);

	// vpss init
	vpss_init(0, width, height);

	// bind vi to vpss
	MPP_CHN_S stSrcChn, stvpssChn;
	stSrcChn.enModId = RK_ID_VI;
	stSrcChn.s32DevId = 0;
	stSrcChn.s32ChnId = 0;

	stvpssChn.enModId = RK_ID_VPSS;
	stvpssChn.s32DevId = 0;
	stvpssChn.s32ChnId = 0;
	printf("====RK_MPI_SYS_Bind vi0 to vpss0====\n");
	s32Ret = RK_MPI_SYS_Bind(&stSrcChn, &stvpssChn);
	if (s32Ret != RK_SUCCESS)
	{
		RK_LOGE("bind 0 ch venc failed");
		return -1;
	}

	// venc init
	RK_CODEC_ID_E enCodecType = RK_VIDEO_ID_AVC;
	venc_init(0, width, height, enCodecType);

	ncnn::Net net;
	printf("Loading ncnn mnist model...");
	if (net.load_param("./ncnn-mnist.param"))
		exit(-1);
	if (net.load_model("./ncnn-mnist.bin"))
		exit(-1);
	printf("Done.\n");

	cv::Mat frame_resize;
	cv::Mat frame_resize_3ch;
	cv::Mat gray_3ch;
	cv::Mat gray_1ch;
	cv::Mat inv;

	while (1)
	{
		// get vpss frame
		s32Ret = RK_MPI_VPSS_GetChnFrame(0, 0, &stVpssFrame, -1);
		if (s32Ret == RK_SUCCESS)
		{
			void *data = RK_MPI_MB_Handle2VirAddr(stVpssFrame.stVFrame.pMbBlk);

			cv::Mat frame(height, width, CV_8UC3, data);
			printf("cv frame:cols:%d rows:%d c:%d\n", frame.cols, frame.rows, frame.channels());

			cv::Rect digit_rect = find_digit_contour(frame);
			digit_rect.x = std::max(0, digit_rect.x - 10);
			digit_rect.y = std::max(0, digit_rect.y - 50);
			digit_rect.width = std::min(frame.cols - digit_rect.x, digit_rect.width + 20);
			digit_rect.height = std::min(frame.rows - digit_rect.y, digit_rect.height + 100);

			if (digit_rect.area() > 0)
			{
				cv::Mat digit_region = frame(digit_rect);
				// printf("cv digit_region-cols:%d rows:%d c:%d\n", digit_region.cols, digit_region.rows, digit_region.channels());
				////imwrite("./img/digit_region.png",digit_region);
				// cv::Mat preprocessed = preprocess_digit_region(digit_region);
				// printf("cv preprocessed-cols:%d rows:%d c:%d\n", preprocessed.cols, preprocessed.rows, preprocessed.channels());
				// imwrite("./img/preprocessed.png",preprocessed);

				////**************************************************//

				cv::cvtColor(digit_region, gray_1ch, cv::COLOR_BGR2GRAY);
				threshold(gray_1ch, gray_1ch, atoi(argv[1]), 255, cv::THRESH_BINARY_INV);
				cv::cvtColor(gray_1ch, gray_3ch, cv::COLOR_GRAY2BGR);
				printf("cv gray_3ch-cols:%d rows:%d c:%d\n", gray_3ch.cols, gray_3ch.rows, gray_3ch.channels());
				////imwrite("./img/gray_3ch.png",gray_3ch);

				//cv::resize(gray_1ch, frame_resize, cv::Size(28, 28));
				cv::resize(gray_1ch, frame_resize, cv::Size(28, 28), 0, 0, cv::INTER_AREA);
				cv::cvtColor(frame_resize, frame_resize_3ch, cv::COLOR_GRAY2BGR);
				imwrite("./img/frame_resize_3ch.png",frame_resize_3ch);
//*****//
				ncnn::Mat in = ncnn::Mat::from_pixels(frame_resize.data, ncnn::Mat::PIXEL_GRAY, frame_resize.cols, frame_resize.rows); // PIXEL_BGR2GRAY
				// const float mean_vals[3] = {0, 0, 0};
				// const float norm_vals[3] = {1 / 255.f, 1 / 255.f, 1 / 255.f};
				// in.substract_mean_normalize(mean_vals, norm_vals);

				ncnn::Mat out;
				printf("ncnn mat in:%d %d %d\n", in.w, in.h, in.c);
				printf("Start Mesuring!\n");
				double total_latency = 0;
				// double start = ncnn::get_current_time();
				ncnn::Extractor ex = net.create_extractor();
				ex.input("flatten_input", in);
				ex.extract("dense_2", out);

				printf("output shape (w, h) = (%d %d) \n", out.w, out.h);
				const float *ptr = out.channel(0);
				int gussed = -1;
				float guss_exp = -10000000;
				for (int i = 0; i < out.w * out.h; i++)
				{
					printf("%d: %.2f\n", i, ptr[i]);
					if (guss_exp < ptr[i])
					{
						gussed = i;
						guss_exp = ptr[i];
					}
				}
				printf("I think it is number %d!\n", gussed);
//*****//
				// sprintf(fps_text,"fps:%.2f",fps);
				// cv::putText(frame,fps_text,
				// 				cv::Point(40, 40),
				// 				cv::FONT_HERSHEY_SIMPLEX,1,
				// 				cv::Scalar(0,255,0),2);
				// 在图像上显示预测结果
				cv::rectangle(frame, digit_rect, cv::Scalar(0, 255, 0), 2);
				sprintf(fps_text, "number:%d", gussed);
				//sprintf(fps_text, "number:%d", ncnn_run_inf(net, gray_1ch));

				cv::putText(frame, fps_text,
							cv::Point(40, 40),
							cv::FONT_HERSHEY_SIMPLEX, 1,
							cv::Scalar(0, 255, 0), 2);

				memcpy(data, frame.data, width * height * 3);
				// memcpy(data, frame_resize_3ch.data, 28 * 28*3);

				// memcpy(data + width * height, inv.data, width * height);
			}
		}

		// send stream
		// encode H264
		RK_MPI_VENC_SendFrame(0, &stVpssFrame, -1);
		// rtsp
		s32Ret = RK_MPI_VENC_GetStream(0, &stFrame, -1);
		if (s32Ret == RK_SUCCESS)
		{
			if (g_rtsplive && g_rtsp_session)
			{
				// printf("len = %d PTS = %d \n",stFrame.pstPack->u32Len, stFrame.pstPack->u64PTS);
				void *pData = RK_MPI_MB_Handle2VirAddr(stFrame.pstPack->pMbBlk);
				rtsp_tx_video(g_rtsp_session, (uint8_t *)pData, stFrame.pstPack->u32Len,
							  stFrame.pstPack->u64PTS);
				rtsp_do_event(g_rtsplive);
			}
			RK_U64 nowUs = TEST_COMM_GetNowUs();
			fps = (float)1000000 / (float)(nowUs - stVpssFrame.stVFrame.u64PTS);
		}

		// release frame
		s32Ret = RK_MPI_VPSS_ReleaseChnFrame(0, 0, &stVpssFrame);
		if (s32Ret != RK_SUCCESS)
		{
			RK_LOGE("RK_MPI_VI_ReleaseChnFrame fail %x", s32Ret);
		}
		s32Ret = RK_MPI_VENC_ReleaseStream(0, &stFrame);
		if (s32Ret != RK_SUCCESS)
		{
			RK_LOGE("RK_MPI_VENC_ReleaseStream fail %x", s32Ret);
		}
	}

	RK_MPI_SYS_UnBind(&stSrcChn, &stvpssChn);

	RK_MPI_VI_DisableChn(0, 0);
	RK_MPI_VI_DisableDev(0);

	RK_MPI_VPSS_StopGrp(0);
	RK_MPI_VPSS_DestroyGrp(0);

	SAMPLE_COMM_ISP_Stop(0);

	RK_MPI_VENC_StopRecvFrame(0);
	RK_MPI_VENC_DestroyChn(0);

	free(stFrame.pstPack);

	if (g_rtsplive)
		rtsp_del_demo(g_rtsplive);

	RK_MPI_SYS_Exit();

	return 0;
}

cv::Rect find_digit_contour(const cv::Mat &image)
{
	cv::Mat gray, blurred, edged;
	cv::cvtColor(image, gray, cv::COLOR_BGR2GRAY);
	cv::GaussianBlur(gray, blurred, cv::Size(5, 5), 0);
	cv::Canny(blurred, edged, 50, 150);

	std::vector<std::vector<cv::Point>> contours;
	cv::findContours(edged, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

	if (contours.empty())
	{
		return cv::Rect();
	}

	// 找到最大的轮廓
	auto largest_contour = std::max_element(contours.begin(), contours.end(),
											[](const std::vector<cv::Point> &a, const std::vector<cv::Point> &b)
											{
												return cv::contourArea(a) < cv::contourArea(b);
											});

	return cv::boundingRect(*largest_contour);
}

cv::Mat preprocess_digit_region(const cv::Mat region)
{
	cv::Mat gray, resized, normalized;
	cv::cvtColor(region, gray, cv::COLOR_BGR2GRAY);
	cv::resize(gray, resized, cv::Size(28, 28), 0, 0, cv::INTER_LINEAR);
	resized.convertTo(normalized, CV_32F, 1.0 / 255.0);
	return normalized;
}

int ncnn_run_inf(ncnn::Net &net,cv::Mat frame_resize)
{
	////

	// Input
	// ncnn::Mat in = ncnn::Mat::from_pixels_resize(frame_resize.data,
	// 		ncnn::Mat::PIXEL_BGR2GRAY,
	// 		28, 28);
	ncnn::Mat in = ncnn::Mat::from_pixels(frame_resize.data, ncnn::Mat::PIXEL_GRAY, frame_resize.cols, frame_resize.rows); // PIXEL_BGR2GRAY
	// const float mean_vals[3] = {0, 0, 0};
    // const float norm_vals[3] = {1 / 255.f, 1 / 255.f, 1 / 255.f};
    // in.substract_mean_normalize(mean_vals, norm_vals);

	ncnn::Mat out;
	printf("ncnn mat in:%d %d %d\n", in.w, in.h, in.c);

	printf("Start Mesuring!\n");

	double total_latency = 0;
	//double start = ncnn::get_current_time();
	ncnn::Extractor ex = net.create_extractor();
	ex.input("flatten_input", in);
	ex.extract("dense_2", out);

	printf("output shape (w, h, c) = (%d %d) \n", out.w, out.h,out.c);
	const float *ptr = out.channel(0);
	int gussed = -1;
	float guss_exp = -10000000;
	for (int i = 0; i < out.w * out.h; i++)
	{
		printf("%d: %.2f\n", i, ptr[i]);
		if (guss_exp < ptr[i])
		{
			gussed = i;
			guss_exp = ptr[i];
		}
	}
	printf("I think it is number %d!\n", gussed);
	//printf("Latency, avg: %.2fms, max: %.2f, min: %.2f. Avg Flops: %.2fMFlops\n", total_latency / 10.0, max, min, 0.78 / (total_latency / 10.0 / 1000.0));
	return gussed;
}