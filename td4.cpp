#include <opencv2/opencv.hpp>
#include <iostream>

using namespace std;
using namespace cv;

VideoCapture capture, overlayVideo;
Mat frame, image_train;

Ptr<AKAZE> akaze;
Ptr<DescriptorMatcher> matcher;
vector<KeyPoint> kp_train, kp_query;
Mat desc_train, desc_query;

Mat akazeTrackinghomography;
int num_good_matches;
int num_inliers;

vector<Point2f> sceneCorners;
vector<Point2f> objectCorners;

vector<Point2f> previousSceneCorners;
Mat previousFrame;
vector<uchar> status;
vector<float> err;
bool MOTION_TRACKING = false;

void overlay() 
{
	Mat overlayFrame, warpedOverlayFrame;

	overlayVideo >> overlayFrame;
	resize(overlayFrame, overlayFrame, image_train.size());

	if (overlayVideo.get(CAP_PROP_POS_FRAMES) == overlayVideo.get(CAP_PROP_FRAME_COUNT))
		overlayVideo.set(CAP_PROP_POS_FRAMES, 0);

	Mat motionTrackingHomography = findHomography(objectCorners, sceneCorners);

	warpPerspective(overlayFrame, warpedOverlayFrame, motionTrackingHomography, frame.size());

	vector<Point> vec_SceneCorners(sceneCorners.begin(), sceneCorners.end());

	Mat mask_image(frame.size(), CV_8U, Scalar(0));
	vector<vector<Point>> polyRecObject;
	polyRecObject.push_back(vec_SceneCorners);
	fillPoly(mask_image, polyRecObject, Scalar(255));

	warpedOverlayFrame.copyTo(frame, mask_image);
}

void akazeTracker()
{
	double elapsedTime = getTickCount();

	Mat corres_frame;

	kp_query.clear();

	akaze->detectAndCompute(frame, noArray(), kp_query, desc_query);

	vector<vector<DMatch>> matches;
	matcher->knnMatch(desc_query, desc_train, matches, 2);

	vector<KeyPoint> matchedQuery_kp, matchedTrain_kp;
	for (int i = 0; i < matches.size(); i++)
	{
		if (matches[i][0].distance < 0.8 * matches[i][1].distance)
		{
			matchedQuery_kp.push_back(kp_query[matches[i][0].queryIdx]);
			matchedTrain_kp.push_back(kp_train[matches[i][0].trainIdx]);
		}
	}
	num_good_matches = matchedQuery_kp.size();

	vector<Point2f> inliers_query_points, inliers_train_points;
	for (int i = 0; i < num_good_matches; i++)
	{
		inliers_query_points.push_back(matchedQuery_kp[i].pt);
		inliers_train_points.push_back(matchedTrain_kp[i].pt);
	}

	vector<KeyPoint> inliers_train_kp, inliers_query_kp;
	vector<DMatch> inlier_matches;
	Mat inlier_mask;

	if (num_good_matches >= 4)
	{
		akazeTrackinghomography = findHomography(inliers_train_points, inliers_query_points, RANSAC, 2, inlier_mask, 2000);

		for (int i = 0, j = 0; i < num_good_matches; i++)
		{
			if (inlier_mask.at<uchar>(i))
			{
				inliers_query_kp.push_back(matchedQuery_kp[i]);
				inliers_train_kp.push_back(matchedTrain_kp[i]);
				inlier_matches.push_back(DMatch(j, j, 0));
				j++;
			}
		}

		drawMatches(image_train, inliers_train_kp, frame, inliers_query_kp, inlier_matches, corres_frame,
			Scalar(Scalar::all(-1)), Scalar(Scalar::all(-1)));

		num_inliers = inlier_matches.size();

		elapsedTime = getTickCount() - elapsedTime;

		putText(corres_frame, format("Good Matches : %d", num_good_matches), Point(5, corres_frame.rows - 80),
			FONT_HERSHEY_COMPLEX, 0.5, Scalar(80, 220, 80), 1, LINE_AA);
		putText(corres_frame, format("Inliers : %d", num_inliers), Point(5, corres_frame.rows - 50),
			FONT_HERSHEY_COMPLEX, 0.5, Scalar(80, 220, 80), 1, LINE_AA);
		putText(corres_frame, format("FPS : %.1f", getTickFrequency() / elapsedTime), Point(5, corres_frame.rows - 20),
			FONT_HERSHEY_COMPLEX, 0.5, Scalar(80, 220, 80), 1, LINE_AA);

		imshow("AKAZE Recognition", corres_frame);
	}
}

int main()
{
	capture.open("data/videoTuto4.avi");

	if (!capture.isOpened())
	{
		cerr << "Couldn't open the source video ..." << endl;
		return 1;
	}

	overlayVideo.open("data/Mercedes-C-Class.mp4");
	if (!overlayVideo.isOpened())
	{
		cerr << "Couldn't open the video ..." << endl;
		return 1;
	}

	akaze = AKAZE::create();
	matcher = DescriptorMatcher::create("BruteForce-Hamming");

	image_train = imread("data/Mercedes-C-Class.png", IMREAD_UNCHANGED);
	if (image_train.empty())
	{
		cerr << "Couldn't load the image ..." << endl;
		return 1;
	}
	resize(image_train, image_train, image_train.size() / 2);
	objectCorners = { Point2f(0,0), Point2f(image_train.cols, 0),
				  Point2f(image_train.cols, image_train.rows), Point2f(0, image_train.rows) };

	akaze->detectAndCompute(image_train, noArray(), kp_train, desc_train);

	while (true)
	{
		double elapsedTime = getTickCount();

		capture >> frame;
		if (frame.empty()) break;

		resize(frame, frame, Size(640, 480));
		Mat grayFrame;
		cvtColor(frame, grayFrame, COLOR_BGR2GRAY);

		if (MOTION_TRACKING == true)
		{
			calcOpticalFlowPyrLK(previousFrame, grayFrame, previousSceneCorners, sceneCorners, status, err);

			overlay();

			if (norm(previousSceneCorners, sceneCorners, NORM_L2) > 3) MOTION_TRACKING = false;

			swap(previousSceneCorners, sceneCorners);
			swap(previousFrame, grayFrame);
		}
		else
		{
			akazeTracker();

			if (num_inliers > 20)
			{
				perspectiveTransform(objectCorners, sceneCorners, akazeTrackinghomography);

				overlay();

				previousSceneCorners = sceneCorners;
				previousFrame = grayFrame;

				MOTION_TRACKING = true;
			}
		}

		elapsedTime = getTickCount() - elapsedTime;
		putText(frame, format("FPS %.1f", getTickFrequency() / elapsedTime), Point(10, 20),
			FONT_HERSHEY_COMPLEX, 0.7, Scalar(0, 255, 0), 1, LINE_AA);
		imshow("AR 2D Overlay", frame);

		if (waitKey(1) == 27) break;
	}

	return 0;
}