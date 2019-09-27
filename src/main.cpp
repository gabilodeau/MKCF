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
	string filename, bgsfolder, maskfilename, outputtype, resfilename;	
	Mat  frame;
	vector <Frameresult> result;
	vector<Rect2d> bboxvec; 
	clock_t start, finish;
	char bgspath[300];

	//Parameters
	int Merge_threshold = 2000;     // square     example, 20^2=400.
	int Min_blob_size = 900;
	int UnmatchedDuration = 8;
	int minTrackLength = 12;
	//char bgsfilepattern[50] = "%s%08d.png"; 
	char bgsfilepattern[50] = "%s/img%05d.png"; 
	char filepath[250];

	if(argc==6)
	{
		filename= argv[1];
		bgsfolder= argv[2];
		resfilename= argv[3];
		Merge_threshold = atoi(argv[4]);
		Min_blob_size = atoi(argv[5]);	
	}
		else
	{
		cout << "Error: not enough arguments. Usage: MKCF framefolder bgsfolder xmloutputfilename" << endl;
		return -1;
	}
	
	
	cout << "starting..." << endl;
	
	startframe=1;
	endframe=0;	
	bool okonce=false;
	do
	{
		endframe++;		
		//sprintf(filepath, "%simg%05d.jpg", filename.c_str(), endframe);
		sprintf(bgspath, bgsfilepattern, bgsfolder.c_str(), endframe);
		//cout <<bgspath << endl;	
		frame = imread(bgspath, CV_8U);		
		if(!frame.empty() && okonce==false)
		{	
			okonce = true;
			startframe=endframe;
		}
		if(endframe > 20000)
			break; // Something is wrong... 
		
	}
	while(!frame.empty() || okonce==false);
	endframe--;
	cout << startframe << endl;
	cout << endframe << endl;

	MKCFTracker trackeur(startframe, endframe, Merge_threshold, UnmatchedDuration, minTrackLength);	
	//Mat mask = imread(maskfilename, CV_8U);
	start = clock();
	for(int ii=startframe; ii<=endframe; ii++)
	{
		bboxvec.clear();
		cout << "current_frame=" << ii << endl;
		//sprintf(filepath, "%s%08d.jpg", filename.c_str(), ii);
		sprintf(filepath, "%simg%05d.jpg", filename.c_str(), ii);
		cout << filepath << endl; 
		
		frame = imread(filepath, CV_LOAD_IMAGE_COLOR);
		if( frame.empty() )                      // Check for invalid input
    		{
			cout <<  "Could not open or find the frame." << endl ;
       			break;
    		}

		
		// Reading the bgs file.	
		sprintf(bgspath, bgsfilepattern, bgsfolder.c_str(), ii);
		//cout << bgspath << endl;
		Mat foreground = imread(bgspath, CV_8U);
		if(!foreground.data )                              
		{
    			cout <<  "Could not open or find the bgs image. Check bgsfilepattern parameter!" << endl ;
    			break;
		}

		
		// Getting connected component in bgs frame. Keep only if > Min_blob_size
		vector<vector<Point>> contours;  
		findContours(foreground, contours, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_NONE);		
		Rect2d bbox;
		for (int i = 0; i< contours.size(); i++){
			bbox = boundingRect(Mat(contours[i]));
			if ((bbox.height*bbox.width) > Min_blob_size)
 				bboxvec.push_back(bbox);
		}
		result = trackeur.track(frame, foreground, bboxvec, ii);

		for(int i=0; i<result.size();i++)
		{
			rectangle(frame, result[i].bboxes.tl(), result[i].bboxes.br(), Scalar(100, 255, 0), 2, 8, 0);
			putText(frame, to_string(result[i].label), cvPoint(result[i].bboxes.x, result[i].bboxes.y + result[i].bboxes.height*0.5), CV_FONT_HERSHEY_COMPLEX, 0.7, Scalar(255, 255, 255));
		}
				

		imshow("Original video", frame);

		//Esc to quit.
		int c = cv::waitKey(1);

	}//for

	finish = clock();
	double totaltime;
	totaltime = (double)(finish - start) / CLOCKS_PER_SEC;    
	cout << "total_time=" << totaltime << "s" << endl;
	cout << "--------------------------------------------------------------------------------";
	
	trackeur.saveToXML(resfilename.c_str());
	

	return 1;

}//main
