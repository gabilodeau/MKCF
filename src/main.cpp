#include <iostream>  
#include <string> 
#include "opencv2/opencv.hpp" 
#include <opencv2/opencv.hpp> 
#include <opencv2/video/video.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/core/utility.hpp>
#include <opencv2/tracking.hpp>
#include <opencv2/videoio.hpp>

#include <cstdio>
#include <fstream>
#include <sstream>
#include <math.h>
#include <time.h>
#include "tracker.hpp"

using namespace cv;
using namespace std;

int main(int argc, char **argv)
{
	int startframe, endframe;
	string videoname, bgsfolder, resfilename, maskfilename;	
	Mat  frame;
	vector <Frameresult> result;
	vector<Rect2d> bboxvec; 
	clock_t start, finish;
	char bgspath[150];

	//Parameters
	const int Merge_threshold = 400;     // square     example, 20^2=400.
	const int Min_blob_size = 50;
	const int UnmatchedDuration = 8;
	const int minTrackLength = 12;
	char bgsfilepattern[50] = "%s/%08ld.png"; 
	

	if (argc >= 6)
	{
		startframe = atoi(argv[1]);
		endframe = atoi(argv[2]);
		videoname = argv[3];
		bgsfolder = argv[4];
		resfilename = argv[5];
		if(argc == 7)
			maskfilename = argv[6];

	}
	else
	{
		cout << "Error: not enough arguments. Usage: MKCF startframe endframe videofile bgsfolder xmloutputfilename" << endl;
		return -1;
	}
	
	
	cv::VideoCapture capture(videoname.c_str());
	if (!capture.isOpened())
	{
		cout << "Error: Problem reading input video!" << endl;
		return -1;
	}
	capture.set(CV_CAP_PROP_POS_FRAMES, startframe);

	MKCFTracker trackeur(startframe, endframe, Merge_threshold, UnmatchedDuration, minTrackLength);	
	Mat mask = imread(maskfilename, CV_8U);
	start = clock();
	for(int ii=startframe; ii<endframe; ii++)
	{
		bboxvec.clear();
		cout << "current_frame=" << ii << endl;
		if (!capture.read(frame))
		{
			cout << "Error: Problem reading frame from video. Aborting..." << ii << endl;
			break;
		}

		// Reading the bgs file.	
		sprintf(bgspath, bgsfilepattern, bgsfolder.c_str(), ii);
		Mat foreground = imread(bgspath, CV_8U);
		if(!foreground.data )                              
		{
    			cout <<  "Could not open or find the bgs image. Check bgsfilepattern parameter!" << endl ;
    			return -1;
		}

		// Todo
		// apply mask if provided
		Mat foreground2;
		if(argc ==7)
		{
			foreground.copyTo(foreground2, mask);
		}
		
		// Getting connected component in bgs frame. Keep only if > Min_blob_size
		vector<vector<Point>> contours;  
		findContours(foreground2, contours, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_NONE);		
		Rect2d bbox;
		for (int i = 0; i< contours.size(); i++){
			bbox = boundingRect(Mat(contours[i]));
			if ((bbox.height*bbox.width) > Min_blob_size)
 				bboxvec.push_back(bbox);
		}
		result = trackeur.track(frame, foreground2, bboxvec, ii);

		for(int i=0; i<result.size();i++)
		{
			rectangle(frame, result[i].bboxes.tl(), result[i].bboxes.br(), Scalar(100, 255, 0), 2, 8, 0);
			putText(frame, to_string(result[i].label), cvPoint(result[i].bboxes.x, result[i].bboxes.y + result[i].bboxes.height*0.5), CV_FONT_HERSHEY_COMPLEX, 0.7, Scalar(255, 255, 255));
		}
				

		imshow("Original video", frame);

		//Esc to quit.
		int c = cv::waitKey(3);

	}//for

	finish = clock();
	double totaltime;
	totaltime = (double)(finish - start) * 1000 / CLOCKS_PER_SEC;    
	cout << "total_time=" << totaltime << "ms" << endl;
	cout << "--------------------------------------------------------------------------------";
	trackeur.saveToXML(resfilename);

	return 1;

}//main
