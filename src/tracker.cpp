#include "tracker.hpp"

MKCFTracker::MKCFTracker(int start, int end, int merge, int unmatched, int minTrackLength)
{
	ids =0; 	
	startframe = start, minTrackLengthth = minTrackLength, mergeth = merge; unmatchedDurationth=unmatched;
	endframe = end;
	
}

void MKCFTracker::saveLastActiveTracks()
{
	for (int i = 0; i < activeTracks.size(); i++)
	{
		Result.push_back(activeTracks[i]);
	}	
}

bool MKCFTracker::saveToXML(string filename)
{
	saveLastActiveTracks();
	outdata.open(filename, ios::out);	
	outdata << "<?xml version="<< char(34) << "1.0"<< char(34) << "?>"<< endl;
	outdata << "<Video fname="<< char(34)<< filename<< char(34) << " start_frame=" << char(34) << startframe << char(34) << " end_frame=" << char(34) << endframe << char(34)<<">" << endl;

	for(int i=0;i<Result.size();i++)
	{
		if(Result[i].bboxes.size()>minTrackLengthth) // Tracks longeur than N
		{
			outdata << "	<Trajectory obj_id=" << char(34) << Result[i].label << char(34) << " obj_type=" << char(34) << "Unknown" << char(34) << " start_frame = " << char(34) << Result[i].startframe << char(34) << " end_frame = " << char(34) << Result[i].startframe + Result[i].bboxes.size() - 1 << char(34) << ">" << endl;
			for (int j = 0; j < Result[i].bboxes.size(); j++) {
					if(Result[i].bboxes[j].x<0) Result[i].bboxes[j].x=0;
					if(Result[i].bboxes[j].y<0) Result[i].bboxes[j].y=0;
					outdata << "		<Frame frame_no=" << char(34) << Result[i].startframe+j
						<< char(34) << " x=" << char(34) << Result[i].bboxes[j].x
						<< char(34) << " y=" << char(34) << Result[i].bboxes[j].y 
						<< char(34) << " width=" << char(34) << Result[i].bboxes[j].width
						<< char(34) << " height=" << char(34) << Result[i].bboxes[j].height
						<< char(34) << " observation=" << char(34) << 0 << char(34) << " annotation=" << char(34)
						<< 0 << char(34) << " contour_pt=" << char(34) << 0 << char(34) << "></Frame>" << endl;
			}//for
			outdata << "	</Trajectory>" << endl;
		}

	}//for
	outdata << "</Video>";
	outdata.close();
	return true;
}

MKCFTracker::~MKCFTracker()
{
}

/*********************************************************************************************************/
vector <Frameresult> MKCFTracker::track(const Mat &frame, const Mat &foreground, vector<Rect2d> bboxlist, int currentFrame)
{
	Rect2d track_roi;
	bool success;	

	// Merging nearby blobs	
	bboxlist = mergeBlobs(foreground, bboxlist);
	
	//identify whether KCF tracker match in this frame or not
	vector<int>Matches(activeTracks.size(), -1); // Suppose no match.
	vector<bool>targetMatches(bboxlist.size(), false); // Suppose no match.
	vector<bool>targetPreferredMatch(bboxlist.size(), false); // Suppose no match.

	vector<Rect2d>Predictedroi(activeTracks.size());
	vector<bool>Predictedsuccess(activeTracks.size());
	
	// calculate multiple matches. 
	vector<int> KCF_Num_Blob(activeTracks.size(), 0);

	/**********************Tracker to candidate objects *********************************/

	// Do predictions
	#pragma omp parallel for
	for (int i = 0; i < activeTracks.size(); i++)
	{ 
		//Predict		
		Predictedroi[i] = activeTracks[i].bboxes.back();
		Predictedsuccess[i] = activeTracks[i].tracker->update(frame, Predictedroi[i]);
		if(Predictedsuccess[i] == false || bbOverlap(Predictedroi[i], activeTracks[i].bboxes.back())<0.1)
			Predictedroi[i] = activeTracks[i].bboxes.back();
	}//for
		
	// match objects by comparing overlapping rates of objects saved in previous frame
	for (int i = 0; i < activeTracks.size(); i++)
	{ 
		float max = 0.0;
		int label = 0;
		for (int j = 0; j < bboxlist.size(); j++){ //Candidates
			// find the most suitable one.
			float overlap = bbOverlap(Predictedroi[i], bboxlist[j]);
			if(overlap > 0.0)
			{
				targetMatches[j] = true;
			}
			if (overlap>max && targetPreferredMatch[j]==false) //Prevent 2 trackers on same object.
			{
				KCF_Num_Blob[i]++;
				max = overlap;
				Matches[i] = j;
			}//if
		}//for
		if(KCF_Num_Blob[i] >= 1)
		{
			
			targetPreferredMatch[Matches[i]] = true;	
		}
	}//for
		
	#pragma omp parallel for
	for (int i = 0; i < activeTracks.size(); i++){
		/**********************  case 1: Occlusion occurs   ***************************/
		if (KCF_Num_Blob[i] >= 2)
		{
			activeTracks[i].KCF_occlusionTime++;
			activeTracks[i].unmatchedTime=0;
			activeTracks[i].bboxes.push_back(Predictedroi[i]); //Occluded, using prediction.
		}		

		/************************ case 2: One to one match  *****************************/
		else if (KCF_Num_Blob[i] == 1)
		{
			Rect2d temp1;
			Rect2d temp2;	

			temp1 = bboxlist[Matches[i]];
			temp2 = activeTracks[i].bboxes.back();
	
			float area_previous = temp2.width*temp2.height;
			float area_bgs = temp1.width*temp1.height;

			if ((area_previous >= 1.4*area_bgs) && (area_previous <= 1.8*area_bgs))
			{
				activeTracks[i].bboxes.push_back(Predictedroi[i]);
			}//if
			else
			{
				/// BGS is more precise
				activeTracks[i].bboxes.push_back(temp1);				
				activeTracks[i].tracker->init(frame, temp1); // Reset tracker.	
			}
			activeTracks[i].KCF_occlusionTime=0;
			activeTracks[i].unmatchedTime=0;
		}
			/******************* case 3: Tracker has no match **********************************/
		else
		{  			
			activeTracks[i].bboxes.push_back(Predictedroi[i]); //no match, using prediction.
			activeTracks[i].unmatchedTime++;
			activeTracks[i].KCF_occlusionTime=0;
		}
	}  


	/**********************Unmatched candidate objects *********************************/

	for (int i = 0; i < bboxlist.size(); i++)
	{
		if(targetMatches[i]==false) 
		{
			Create_new_obj(bboxlist[i], frame, currentFrame);
		}
	}	


	/**************************************Clean up ************************************/
		
	vector<int> toerase;
	for (int i = 0; i < activeTracks.size(); i++)
	{
		if(activeTracks[i].unmatchedTime>=unmatchedDurationth)
		{
			Result.push_back(activeTracks[i]);
			toerase.push_back(i);
		}
	}
		
	for (int i = 0; i< toerase.size(); i++)
	{
		activeTracks.erase(activeTracks.begin()+toerase[i]-i);
	}
		
	/************************************Displaying***************************************/
		
	vector <Frameresult> out;		
	for (int i = 0; i < activeTracks.size(); i++)
	{
		if(activeTracks[i].unmatchedTime<unmatchedDurationth)
		{
			Frameresult fres;
			fres.label = activeTracks[i].label;
			fres.bboxes = activeTracks[i].bboxes.back();
			out.push_back(fres);
		}
	} 
	
	return  out;
}// KCF_tracker

/*********************************************************************************************************/
void MKCFTracker::Create_new_obj(Rect2d bbox, const Mat &frame, int currentFrame)
{	
	//create a new object.
	MKCFTrack at;
	at.label = ids;
	ids++;
	at.startframe = currentFrame;
	at.bboxes.push_back(bbox);	
	at.KCF_occlusionTime=0; 
	at.unmatchedTime=0; 
	
	
	TrackerKCF::Params param;
	param.desc_pca = TrackerKCF::MODE::CN | TrackerKCF::MODE::GRAY;
	Ptr <Tracker> tk = TrackerKCF::create(param);
	tk->init(frame, bbox);
	at.tracker =  tk;
	activeTracks.push_back(at);
}


float MKCFTracker::bbOverlap(const Rect2d &box1, const Rect2d &box2)
{
	if (box1.x > box2.x + box2.width) { return 0.0; }
	if (box1.y > box2.y + box2.height) { return 0.0; }
	if (box1.x + box1.width < box2.x) { return 0.0; }
	if (box1.y + box1.height < box2.y) { return 0.0; }
	float colInt = min(box1.x + box1.width, box2.x + box2.width) - max(box1.x, box2.x);
	float rowInt = min(box1.y + box1.height, box2.y + box2.height) - max(box1.y, box2.y);
	float intersection = colInt * rowInt;
	float area1 = box1.width*box1.height;
	float area2 = box2.width*box2.height;
	return (intersection / area1);
}

vector<Rect2d> MKCFTracker::mergeBlobs(const Mat &foreground, vector<Rect2d> bboxlist)
{
	
		vector<int> flag(bboxlist.size());
		vector<Point2f> centroid(bboxlist.size());
		Mat binImage, ROI;

		threshold(foreground, binImage, 100,255,THRESH_BINARY);

		for (int i = 0; i < bboxlist.size(); i++)
		{
			ROI = binImage(bboxlist[i]);
			Moments m = moments(ROI,true);
			Point p(m.m10/m.m00, m.m01/m.m00);
			centroid[i].x = p.x + bboxlist[i].x;
			centroid[i].y = p.y + bboxlist[i].y;
			flag[i] = 0;
		}
		
		for (int i = 0; i < bboxlist.size(); i++){
			if (flag[i] == 1)
				continue;
			if (bboxlist[i].width*bboxlist[i].height == 0){
				flag[i] = 1; continue;
			}
			for (int j = i + 1; j < bboxlist.size(); j++){
				Point a= centroid[i];
				Point b= centroid[j];
				if (((a.x - b.x)*(a.x - b.x) + (a.y - b.y)*(a.y - b.y)) < mergeth)												{
					bboxlist[i] = bboxlist[i] | bboxlist[j];
					flag[j] = 1; //bboxvec[j] is going to be deleted.
				}
			}//for
		}//for

		for (int i = 0; i < bboxlist.size();){
			if (flag[i] == 1){
				bboxlist.erase(bboxlist.begin() + (i));
				flag.erase(flag.begin() + i);
			}
			else i++;
		}//for

	return bboxlist;
}

