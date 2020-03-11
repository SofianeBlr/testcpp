#include <opencv2/opencv.hpp>
#include <iostream>

using namespace cv;
using namespace cv::ml;
using namespace std;

// Declare global variables
int slider_pos = 0, slider_pos_old = 0;
VideoCapture capture;

int menu()
{
	int choice;
	system("cls"); 

	cout << endl << "--- Computer Vision Basic Processings ---" << endl << endl;
	cout << "-----------------------------------------" << endl;
	cout << " 1 : Read and display an image" << endl;
	cout << " 2 : Linear filtering" << endl;
	cout << " 3 : Smoothing" << endl;
	cout << " 4 : Morphology" << endl;
	cout << " 5 : Thresholding" << endl;
	cout << " 6 : Edge detection" << endl;
	cout << " 7 : Histogram equalization" << endl;
	cout << " 8 : Template matching" << endl;
	cout << " 9 : Find contours" << endl;
	cout << " 10: Convex hull" << endl;
	cout << " 11: Matching descriptors" << endl;
	cout << " 12: SVM classification" << endl;
	cout << " 13: GUI video" << endl;
	cout << " 14: Quit the program..." << endl;
	cout << "-----------------------------------------" << endl << endl;
	cout << "Please enter the number of the function to run : ";
	
	cin >> choice;
	if ( choice > 0 && choice < 14 )
		cout << "Press any key to quit the function" << endl;

	return choice;
}

int readDisplayImage()
{
	// Declare the window name variable
	string windowName = "Read/Display Image";

	// Load the image
	Mat frame = imread( "images/landscape.jpg", IMREAD_UNCHANGED );

	// Check if the image is empty
	if( frame.empty() )
	{
		cerr << endl << "Couldn't read the image..." << endl;
		return 1;
	}

	// Create a window
	namedWindow( windowName, WINDOW_AUTOSIZE );

	// Display the image
	imshow( windowName, frame );

	// Wait for pressed key
	waitKey(0) ;

	// Destroy the window
	destroyWindow(windowName);

	return 0;
}

int linearFiltering()
{
	// Declare variables
	Mat src, dst;
	Mat kernel;
	Point anchor;
	double delta;
	int ddepth;
	int kernel_size;
	string windowName_src = "Source Image";
	string windowName_dst = "Linear Filtering";

	// Initialize filter arguments
	anchor = Point( -1, -1 );
	delta = 0;
	ddepth = -1;
	kernel_size = 3;
	kernel = Mat::ones( kernel_size, kernel_size, CV_32F )/ (float)(kernel_size*kernel_size);

	// Load the image
	src = imread( "images/plane.jpg", IMREAD_UNCHANGED );

	// Check if the image is empty
	if( src.empty() )
    {
		cerr << endl <<"Couldn't read the image..." << endl;
        return 1;
    }

	// Create windows
	namedWindow( windowName_src, WINDOW_AUTOSIZE );
	namedWindow( windowName_dst, WINDOW_AUTOSIZE );

	// Apply the filter
	filter2D( src, dst, ddepth , kernel, anchor, delta, BORDER_DEFAULT );

	// Display the images
	imshow( windowName_src, src );
	imshow( windowName_dst, dst );

	// Wait for a pressed key
	waitKey(0) ;
	
	// Destroy windows
	destroyWindow(windowName_src);
	destroyWindow(windowName_dst);

	return 0;
}

int smoothing()
{
	// Declare variables
	int MAX_KERNEL_LENGTH = 31;
	Mat src; Mat dst1, dst2, dst3;
	string windowName_src = "Source Image";
	string windowName_dst1 = "Gaussian blur";
	string windowName_dst2 = "Median blur";
	string windowName_dst3 = "Bilateral Filter";

    // Load the source image
    src = imread( "images/monalisa.jpg", IMREAD_UNCHANGED );

	// Check if the image is empty
	if( src.empty() )
    {
		cerr << endl <<"Couldn't read the image..." << endl;
        return 1;
    }

	// Create windows
	namedWindow( windowName_src, WINDOW_AUTOSIZE );
	namedWindow( windowName_dst1, WINDOW_AUTOSIZE );
	namedWindow( windowName_dst2, WINDOW_AUTOSIZE );
	namedWindow( windowName_dst3, WINDOW_AUTOSIZE );

	// Apply the Gaussian blur
    GaussianBlur( src, dst1, Size( 5, 5 ), 0, 0 );
	
    // Apply the Median blur
    medianBlur( src, dst2, 5 );

	// Apply the Bilateral Filter
    bilateralFilter( src, dst3, 15, 15*2, 15/2 );

	// Display images
	imshow( windowName_src, src );
	imshow( windowName_dst1, dst1 );
	imshow( windowName_dst2, dst2 );
	imshow( windowName_dst3, dst3 );

	// Wait for pressed key
	waitKey(0) ;

	// Destroy windows
	destroyWindow(windowName_src);
	destroyWindow(windowName_dst1);
	destroyWindow(windowName_dst2);
	destroyWindow(windowName_dst3);

	return 0;
}

int morphology()
{
	// Declare variables
	string windowName_src = "Source Image";
	string windowName_dst1 = "Dilatation";
	string windowName_dst2 = "Erosion";
	Mat src, erosion_dst, dilation_dst;

	// Load the image
	src = imread( "images/apple.png", IMREAD_UNCHANGED );

	// Check if the image is empty
	if( src.empty() )
	{
		cerr << endl <<"Couldn't read the image..." << endl;
		return 1;
	}

	// Create windows
	namedWindow( windowName_src, WINDOW_AUTOSIZE );
	namedWindow( windowName_dst1, WINDOW_AUTOSIZE );
	namedWindow( windowName_dst2, WINDOW_AUTOSIZE );

	// Apply the dilation
	int dilation_size = 3;
	int dilation_type = MORPH_RECT;

	Mat dilatation_element = getStructuringElement( dilation_type,
                       Size( 2*dilation_size + 1, 2*dilation_size+1 ),
                       Point( dilation_size, dilation_size ) );

    dilate( src, dilation_dst, dilatation_element );

	// Apply the erosion
	int erosion_size = 3;
	int erosion_type = MORPH_RECT;

	Mat erosion_element = getStructuringElement( erosion_type,
                       Size( 2*erosion_size + 1, 2*erosion_size+1 ),
                       Point( erosion_size, erosion_size ) );

    erode( src, erosion_dst, erosion_element );

	// Display images
	imshow( windowName_src, src );
	imshow( windowName_dst1, dilation_dst );
	imshow( windowName_dst2, erosion_dst );

	// Wait for pressed key
	waitKey(0) ;

	// Destroy windows
	destroyWindow(windowName_src);
	destroyWindow(windowName_dst1);
	destroyWindow(windowName_dst2);

	return 0;
}

int thresholding()
{
	// Declare variables
	Mat src, src_gray, dst;
	string windowName_src = "Source Image";
	string windowName_dst = "Threshold";

	int threshold_value = 100;
	int threshold_type = THRESH_BINARY;
	int max_BINARY_value = 255;

	// Load the image
	src = imread( "images/tiger.jpg", IMREAD_UNCHANGED );

	// Check if the image is empty
	if( src.empty() )
    {
		cerr << endl <<"Couldn't read the image..." << endl;
        return 1;
    }

	// Create windows
	namedWindow( windowName_src, WINDOW_AUTOSIZE );
	namedWindow( windowName_dst, WINDOW_AUTOSIZE );

	// Convert the image to gray
	cvtColor( src, src_gray, COLOR_BGR2GRAY );

	// Threshold the image
	threshold( src_gray, dst, threshold_value, max_BINARY_value, threshold_type );

	// Display images
	imshow( windowName_src, src );
	imshow( windowName_dst, dst );

	// Wait for pressed key
	waitKey(0) ;

	// Destroy windows
	destroyWindow(windowName_src);
	destroyWindow(windowName_dst);	
	
	return 0;
}

int edgeDetection()
{
	// Declare variables
	Mat src, src_gray, dst1, dst2;
	string windowName_src = "Source Image";
	string windowName_dst1 = "Laplace";
	string windowName_dst2 = "Canny";

	// Load the image
	src = imread( "images/building.jpg", IMREAD_UNCHANGED );

	// Check if the image is empty
	if( src.empty() )
	{
		cerr << endl <<"Couldn't read the image..." << endl;
		return 1;
	}

	// Create windows
	namedWindow( windowName_src, WINDOW_AUTOSIZE );
	namedWindow( windowName_dst1, WINDOW_AUTOSIZE );
	namedWindow( windowName_dst2, WINDOW_AUTOSIZE );

	// Convert the image to gray
	cvtColor( src, src_gray, COLOR_BGR2GRAY );

	// Apply Laplace
	int kernel_size = 3;
	int ddepth = CV_16S;

	Laplacian( src_gray, dst1, ddepth, kernel_size );

	convertScaleAbs( dst1, dst1 );
	
	// Apply Canny
	int lowThreshold = 20;
	int highThreshold = 110;

	Canny( src_gray, dst2, lowThreshold, highThreshold );

	// Display the images
	imshow( windowName_src, src );
	imshow( windowName_dst1, dst1 );
	imshow( windowName_dst2, dst2 );

	// Wait for pressed key
	waitKey(0) ;

	// Destroy windows
	destroyWindow(windowName_src);
	destroyWindow(windowName_dst1);
	destroyWindow(windowName_dst2);

	return 0;
}

int histogramEqualization()
{
	// Declare variables
	Mat src, src_gray, dst;
	string windowName_src = "Source Image";
	string windowName_dst = "Histogram Equalization";

	// Load the image
	src = imread( "images/bird.jpg", IMREAD_UNCHANGED );
	
	// Check if the image is empty
	if( src.empty() )
    {
		cerr << endl <<"Couldn't read the image..." << endl;
        return 1;
    }

	// Create windows
	namedWindow( windowName_src, WINDOW_AUTOSIZE );
	namedWindow( windowName_dst, WINDOW_AUTOSIZE );

	// Convert the image to gray
	cvtColor( src, src_gray, COLOR_BGR2GRAY );

	// Equalize the histogram
	equalizeHist( src_gray, dst );

	// Display the images
	imshow( windowName_src, src );
	imshow( windowName_dst, dst );

	// Wait for pressed key
	waitKey(0) ;

	// Destroy windows
	destroyWindow( windowName_src );
	destroyWindow( windowName_dst );

	return 0;
}

int templateMatching()
{
	// Declare variables
	string windowName_src = "Source Image";
	string windowName_templ = "Template Image";
	string windowName_dst = "Template Matching";
	Mat src, templ, dst;
	int match_method = TM_SQDIFF_NORMED;

	// Load the reference image and the template
	src = imread( "images/bus.jpg", IMREAD_UNCHANGED );
	templ = imread( "images/bus_template.png", IMREAD_UNCHANGED );

	// Check if images are empty
	if( src.empty() || templ.empty() )
	{
		cerr << endl << "Couldn't read one of the images..." << endl;
		return 1;
	}

	// Create windows
	namedWindow( windowName_src, WINDOW_AUTOSIZE );
	namedWindow( windowName_templ, WINDOW_AUTOSIZE );
	namedWindow( windowName_dst, WINDOW_AUTOSIZE );

	// Match the template with the reference image
	matchTemplate( src, templ, dst, match_method );

	// Localize the best match with minMaxLoc
	double minVal; double maxVal; Point minLoc; Point maxLoc;
	Point matchLoc;
	minMaxLoc( dst, &minVal, &maxVal, &minLoc, &maxLoc, Mat() );

	matchLoc = minLoc;
	dst = src.clone();
	
	// Draw the rectangle
	rectangle( dst, matchLoc, Point( matchLoc.x + templ.cols , matchLoc.y + templ.rows ), 
							CV_RGB(0,255,0), 5, 8, 0 );
		
	// Display the images
	imshow( windowName_src, src );
	imshow( windowName_templ, templ );
	imshow( windowName_dst, dst );

	// Wait for pressed key
	waitKey(0) ;

	// Destroy the windows
	destroyWindow( windowName_src );
	destroyWindow( windowName_templ );
	destroyWindow( windowName_dst );

	return 0;
}

int findContours()
{
	// Declare variables
	string windowName_src = "Source Image";
	string windowName_dst = "Contours";
	Mat src, src_gray, dst;

	// Load the image
	src = imread( "images/porsche.jpg", IMREAD_UNCHANGED );

	// Check if the image is empty
	if( src.empty() )
	{
		cerr << endl <<"Couldn't read the image..." << endl;
		return 1;
	}

	// Create windows
	namedWindow( windowName_src, WINDOW_AUTOSIZE );
	namedWindow( windowName_dst, WINDOW_AUTOSIZE );

	// Convert the image to gray
	cvtColor( src, src_gray, COLOR_BGR2GRAY );

	// Find contours
	int thresh = 100;
	RNG rng(12345);
	vector<vector<Point>> contours;

	// Detect edges using canny
	Canny( src_gray, src_gray, thresh, thresh*2 );

	// Find contours
	findContours( src_gray, contours, RETR_TREE, CHAIN_APPROX_SIMPLE, Point(0, 0) );

	// Draw contours
	dst = Mat::zeros( src_gray.size(), CV_8UC3 );
	for( size_t i = 0; i< contours.size(); i++ )
	{
		Scalar color = Scalar( rng.uniform(0, 255), rng.uniform(0,255), rng.uniform(0,255) );
		drawContours( dst, contours, (int)i, color, 2, 8 );
	}

	// Display images
	imshow( windowName_src, src );
	imshow( windowName_dst, dst );

	// Wait for pressed key
	waitKey(0) ;

	// Destroy the windows
	destroyWindow( windowName_src );
	destroyWindow( windowName_dst );

return 0;
}

void convexHull()
{
	// Declare an image and a random generator
    Mat img(500, 500, CV_8UC3);

	// Generate a random number of points
	RNG& rng = theRNG();    
	int count = rng.uniform(1, 100);
	
	vector<Point> points;

	// Generate random points
    for( int i = 0; i < count; i++ )
    {
        Point pt;
		pt.x = rng.uniform(img.cols/4, img.cols*3/4);
		pt.y = rng.uniform(img.rows/4, img.rows*3/4);

        points.push_back(pt);
    }

	// Compute the convex hull
    vector<int> hull;
    convexHull( Mat(points), hull, true );

	// Draw points
    img = Scalar::all(0);
    for( int i = 0; i < count; i++ )
        circle( img, points[i], 3, Scalar(0, 0, 255), FILLED, LINE_AA );

	// Draw the convex hull
    int hullcount = (int)hull.size();
    Point pt0 = points[hull[hullcount-1]];

    for( int i = 0; i < hullcount; i++ )
    {
        Point pt = points[hull[i]];
        line( img, pt0, pt, Scalar(0, 255, 0), 1, LINE_AA );
        pt0 = pt;
    }

	// Display the image
    imshow( "Convex hull", img );

	// Press escape to quit the loop
	waitKey(0);

	// Destroy the window
	destroyWindow( "Convex hull" );
}

int matchingDescriptors()
{
	// Load reference and query images
    Mat img1 = imread( "images/box.png", IMREAD_GRAYSCALE );	
    Mat img2 = imread( "images/box_in_scene.png", IMREAD_GRAYSCALE );

	// Check if images are empty
    if( img1.empty() || img2.empty() )
    {
		cerr << endl << "Couldn't read one of the images..." << endl;
		return 1;
    }

    // Detect keypoints
	Ptr<FeatureDetector> orb = ORB::create(100);
    vector<KeyPoint> keypoints1, keypoints2;
    orb->detect( img1, keypoints1 );
    orb->detect( img2, keypoints2 );

    // Compute descriptors    
    Mat descriptors1, descriptors2;
	orb->compute( img1, keypoints1, descriptors1 );
	orb->compute( img2, keypoints2, descriptors2 );

    // Match descriptors
    BFMatcher matcher( NORM_L2 );
    vector<DMatch> matches;
    matcher.match( descriptors1, descriptors2, matches );

    // Draw results
    string windowName = "Matches";
    namedWindow( windowName, WINDOW_AUTOSIZE );
    Mat img_matches;
    drawMatches( img1, keypoints1, img2, keypoints2, matches, img_matches );
   	imshow( windowName, img_matches );

	// Wait for pressed key
	waitKey(0);

	// Destroy the window
	destroyWindow( windowName );

	return 0;
}

void classificationSVM()
{
    // Data for visual representation
    int width = 512, height = 512;
    Mat image = Mat::zeros( height, width, CV_8UC3 );

    // Set up training data
    int labels[4] = { 1, 2, 3, 4 };
    Mat labelsMat( 4, 1, CV_32SC1, labels );

	// Generate random training data 
	Point data1 = Point(rand() % 512, rand() % 512);
	Point data2 = Point(rand() % 512, rand() % 512);
	Point data3 = Point(rand() % 512, rand() % 512);
	Point data4 = Point(rand() % 512, rand() % 512);
	float trainingData[4][2] = { {data1.x,data1.y }, {data2.x, data2.y }, {data3.x, data3.y }, {data4.x, data4.y } };
    Mat trainingDataMat( 4, 2, CV_32FC1, trainingData );

    // Set up the SVM parameters
	Ptr<SVM> svm = SVM::create();
	svm->setType(SVM::C_SVC);
	svm->setKernel(SVM::LINEAR);
	svm->setTermCriteria(TermCriteria(TermCriteria::MAX_ITER, 100, 1e-6));

    // Train the SVM
    svm->train( trainingDataMat, ROW_SAMPLE, labelsMat );
	
    Vec3b green(0,255,0), blue (255,0,0), red (0,0,255), yellow (0,255,255);

    // Show the decision regions given by the SVM
    for ( int i = 0; i < image.rows; ++i )
        for ( int j = 0; j < image.cols; ++j )
        {
            Mat sampleMat = ( Mat_<float>(1,2) << i,j );
            float response = svm->predict( sampleMat );

            if (response == 1)
                image.at<Vec3b>(j, i)  = green;
            else if (response == 2)
                 image.at<Vec3b>(j, i) = blue;
			else if (response == 3)
                 image.at<Vec3b>(j, i) = red;
			else if (response == 4)
                 image.at<Vec3b>(j, i) = yellow;
        }
	
    // Show the training data
    int thickness = -1;
    int lineType = 8;
    circle( image, data1, 5, Scalar( 0, 0, 0), thickness, lineType );
    circle( image, data2, 5, Scalar( 0, 0, 0), thickness, lineType );
    circle( image, data3, 5, Scalar( 0, 0, 0), thickness, lineType );
    circle( image, data4, 5, Scalar( 0, 0, 0), thickness, lineType );

	// Draw results
	string windowName = "SVM classification";
    namedWindow( windowName, WINDOW_AUTOSIZE );
    imshow( windowName, image ); // show it to the user

	// Wait for pressed key
	waitKey(0);

	// Destroy the window
	destroyWindow( windowName );
}

void onTrackbarSlide(int, void*)
{
    // Update the slider position only if an event has occured
	if( abs( slider_pos - slider_pos_old ) > 1 )
		capture.set( CAP_PROP_POS_FRAMES, slider_pos );
}

int guiVideo()
{
	// Declare variables
	string windowName = "Video Capture";
	const char *trackbar_name = "Slider";
	Mat frame;

	// Open the video
    capture.open( "videos/video.mp4" );

	// Check if the video is opened
    if( !capture.isOpened() )
    {
		cerr << endl << "Error: Could not initialize capturing..." << endl;
        return 1;
    }

	// Create the window
    namedWindow( windowName, WINDOW_AUTOSIZE );

	// Get the number of frames of the video
	int count_frames = (int) capture.get( CAP_PROP_FRAME_COUNT );

	// Create the trackbar
	if( count_frames )
		createTrackbar( trackbar_name, windowName, &slider_pos, count_frames, onTrackbarSlide );

	// Create the infinite loop
	while(true)
	    {
			// Grab the frame from the video stream
			capture >> frame;

			// Check if the frame is empty
			if( frame.empty() )	break;

			// Display the frame
			imshow( windowName, frame );

			// Press any key to quit
			if( waitKey(40) > 0 ) break;

			// Update the position of trackbar
			setTrackbarPos( trackbar_name, windowName, slider_pos );
			slider_pos_old = slider_pos;
			slider_pos++;

	    }
		// Destroy the window
		destroyWindow( windowName );

	return 0;
}
int main ()
{
	while ( true )
	{ 
		int fct_selected = menu();

		switch ( fct_selected )
		{
			case 1 : readDisplayImage();
					 break;
			case 2 : linearFiltering();
					 break;
			case 3 : smoothing();
					 break;
			case 4 : morphology();
					 break;
			case 5 : thresholding();
					 break;
			case 6 : edgeDetection();
					 break;
			case 7 : histogramEqualization();
					 break;
			case 8 : templateMatching();
					 break;
			case 9 : findContours();
					 break;
			case 10: convexHull();
					 break;
			case 11: matchingDescriptors();
					 break;
			case 12: classificationSVM();
					 break;
			case 13: guiVideo();
					 break;
			case 14: cout << endl << endl << "The program is exited... ";
					 return 0;				
			default: cout << endl << "The selected function is not valid! Please retry..." << endl;		
					 system("pause");
					 break;
		}
	} 

	return 0;
}