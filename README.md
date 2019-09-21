# MKCF
Multiple object tracker based on the paper: https://128.84.21.199/abs/1611.02364

This is a new code version for our paper: Yang, Y., Bilodeau, G.-A., 
Multiple Object Tracking with Kernelized Correlation Filters in Urban Mixed Traffic, 
14th Conference on Computer and Robot Vision (CRV), Edmonton, Alberta, Canada, May 16-19, 2017, pp. 209-216

This is a re-coded version of our orignal implementation located at : https://github.com/iyybpatrick/M-OBJ-TRK-ALGO
The results may differ slightly from the original.

For computing CLEAR MOT metrics, the following code can be used: https://github.com/beaupreda/clear_mot_metrics

# Dependencies

- OpenCV 3.4.7
- OpenMP (optional)

# Usage: 
./MKCF start_frame_number end_frame_number video_file backgroundsubtraction_folder xmloutput_file mask_file

mask_file is optional. It is to ignore some detections.
Some parameters can be specified at the beginning of main.cpp

# License

See the LICENSE file for more details.
