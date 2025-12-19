# Importing Libraries
import numpy as np
import pandas as pd
import cv2

import os, sys, warnings
warnings.filterwarnings("ignore")
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import subprocess
import functions.IO_functions as IO_functions
import functions.extract_functions as extract_functions
import defaults


# Resize Video
def resize_video(
	original_video:np.array,
	resize_dimensions:any
):
	"""
	Args:
		original_video (np.array): Original numpy video.
		resize_dimensions (any): Resize video to given dimensions (width, height).
	"""
	# Dimension
	width = original_video.shape[2]
	height = original_video.shape[1]

	# Skip if original video has same dimensions as resize dimensions
	if resize_dimensions[0] == width and resize_dimensions[1] == height:
		# Logging
		print ("Original Video Dimensions = Resize Dimensions = {}".format(resize_dimensions))

		return original_video
	
	# Resizing
	video = []
	for i in range(original_video.shape[0]):
		video.append(
			cv2.resize(original_video[i], dsize=(resize_dimensions[0], resize_dimensions[1]), interpolation=cv2.INTER_LANCZOS4)
		)

	video = np.array(video)

	# Assertions
	assert (video.dtype == np.uint8) and (np.min(video) >= 0 and np.max(video) <= 255), "Input Image/Videos should of type uint8 and should have range [0,255]."

	# Logging
	print ("Original Video Dimensions = {}".format((width, height)))
	print ("Resized Video Dimensions = {}".format((video.shape[2], video.shape[1])))
	print ("Data-Type of Video = {}".format(video.dtype))

	return video


# Video Properties
df = pd.read_csv(defaults.source_dataset_info_path)
df = df.set_index("filename").to_dict(orient="index")


# For each video-file
for video_file in defaults.Video_Titles:
	# File and Properties
	video_properties = df[video_file]

	# Logging
	print ("-"*75 + "\n" + video_file + "\n" + "-"*75, flush=True)

	# Reading Video 
	file_path = os.path.join(defaults.source_dataset_path, video_file + ".yuv")
	yuv_reader = IO_functions.YUV_Reader(
		filepath=file_path,
		width=video_properties["width"],
		height=video_properties["height"],
		yuv_type=video_properties["pix_fmt"]
	)
	video = yuv_reader.get_RGB_video()

	# Resize Video
	video = resize_video(
		original_video=np.copy(video),
		resize_dimensions=(3840,2160)
	)

	# Paths
	video_temp_path = "logs/misc/temp_{}.npy".format(video_file)
	video_time_path = "logs/misc/{}.npy".format(video_file)

	# Saving Video
	np.save(video_temp_path, video)


	# Execution Time for Low-Level Features
	cmd = "/usr/bin/time python3 get_execution_time.py --func LLF --video_file {} --video_path {}".format(
		video_file,
		video_temp_path
	)

	output = subprocess.getoutput(cmd)
	LLF_time = extract_functions.extract_execution_time(output)


	# Execution Time for VIF Features
	cmd = "/usr/bin/time python3 get_execution_time.py --func VIF --video_file {} --video_path {}".format(
		video_file,
		video_temp_path
	)

	output = subprocess.getoutput(cmd)
	VIF_time = extract_functions.extract_execution_time(output)


	# Execution Time for Extra-Trees
	cmd = "/usr/bin/time python3 get_execution_time.py --func ExtraTrees --video_file {} --video_path {}".format(
		video_file,
		video_temp_path
	)

	output = subprocess.getoutput(cmd)
	ExtraTrees_time = extract_functions.extract_execution_time(output)


	# Saving Results
	np.save(video_time_path, np.array([LLF_time, VIF_time, ExtraTrees_time]))


	# Delete Temp Video
	os.remove(video_temp_path)