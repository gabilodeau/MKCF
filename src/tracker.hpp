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

using namespace cv;
using namespace std;

class Frameresult
{
	public:	
	int label;
	Rect2d bboxes; 

	Frameresult(){}
	Frameresult(const Frameresult &traj) {label = traj.label; bboxes = traj.bboxes;} 
	Frameresult& operator=(const Frameresult &traj) {label = traj.label; bboxes = traj.bboxes; return *this;}  
};

class MKCFTrack
{
	public:	
	int label;
	int startframe;	
	vector<Rect2d> bboxes; 
	int KCF_occlusionTime;     
	int unmatchedTime;  
	Ptr<Tracker> tracker;

	MKCFTrack(){startframe=-1;}
	MKCFTrack(const MKCFTrack &atrack) {label = atrack.label; bboxes = atrack.bboxes; startframe = atrack.startframe; KCF_occlusionTime = atrack.KCF_occlusionTime; unmatchedTime = atrack.unmatchedTime; tracker = atrack.tracker;} 
	MKCFTrack &operator=(const MKCFTrack &atrack) {label = atrack.label; bboxes = atrack.bboxes; startframe = atrack.startframe; KCF_occlusionTime = atrack.KCF_occlusionTime; unmatchedTime = atrack.unmatchedTime; tracker = atrack.tracker; return *this;}  
};


class MKCFTracker 
{
	int startframe, endframe, minTrackLengthth, mergeth, unmatchedDurationth;
	int ids;
	vector<MKCFTrack> Result;
	vector<MKCFTrack> activeTracks;

	vector<Rect2d> mergeBlobs(const Mat &foreground, vector<Rect2d> bboxlist);
	void Create_new_obj(Rect2d bbox, const Mat &frame, int currentFrame);
	float bbOverlap(const Rect2d &box1, const Rect2d &box2);
	void saveLastActiveTracks();

	
     public:	
	MKCFTracker(int start=0, int end =0, int merge = 2000, int unmatched = 8, int minTrackLength = 12);
	~MKCFTracker();
	vector <Frameresult> track(const Mat &frame, const Mat &foreground, vector<Rect2d> bboxlist, int currentFrame);
	bool saveToXML(string filename);
	bool saveToUADetrac(string filename);
	bool readUADetrac(string filename);
};
