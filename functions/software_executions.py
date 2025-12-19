# Importing Functions
import numpy as np
import re

import os, sys, warnings
warnings.filterwarnings("ignore")
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from typing import Type, Callable, Tuple, Optional, Set, List, Union
import subprocess, shlex, shutil


# Encoders Path
ffmpeg_path = "/home/krishna/Leveraging-Compression-to-Construct-Transferable-Bitrate-Ladders/ffmpeg_static/ffmpeg-6.0-amd64-static"


# Extracting Video Properties using ffprobe
def ffprobe_get_video_properties(
	input_video_path:str
):
	# Extension
	ext = os.path.splitext(os.path.basename(input_video_path))[1]
	assert ext != "yuv" or ext != "YUV", "Input files should not be raw, as they do not contain anuy metadata."

	# Get Output Video Statistics
	print ("Statistics of {}".format(input_video_path))

	# Command
	cmd = "{}/ffprobe -hide_banner -loglevel error -show_entries stream=bits_per_raw_sample,pix_fmt,avg_frame_rate,codec_name,width,height,num_frames,bit_rate -select_streams v -i {}".format(ffmpeg_path, input_video_path)

	# Execute
	output = subprocess.getoutput(cmd)

	# Processing Output
	output = output.split("\n")
	properties = {}
	for line in output:
		if "=" in line:
			key,value = line.split("=", maxsplit=1)

			if value == "N/A":
				properties[key] = value
			elif key == "width" or key == "height":
				properties[key] = int(value)
			elif key == "avg_frame_rate":
				properties[key] = np.round(eval(value), decimals=2)
			else:
				properties[key] = value
	
	return properties, output



# Converting videos to YUV files
def convert_to_yuv(
	input_video_path:str,
	output_yuv_path:str,
	resolution:Tuple,
	frame_rate:float,
	pixel_format:str,
):
	"""
	Converting a video files to raw YUV file.

	Args:
		input_video_path (str): Path of input video file.
		output_yuv_path (str): Path to output YUV file.
	"""
	# Assertions
	assert os.path.exists(input_video_path), "Provide a valid input_yuv_path."
	assert resolution is not None, "For a 'YUV' format video input, resolution is required are inputs."
	assert frame_rate is not None, "For a 'YUV' format video input, frame-rate is required are inputs."
	assert pixel_format is not None,  "For a 'YUV' format video input, pixel_format is required are inputs."
	
	# Input/Output Settings
	cmd_resolution = "-video_size " + str(resolution[0]) + "x" + str(resolution[1])
	cmd_frame_rate = "-framerate " + str(frame_rate)
	cmd_pixel_format = "-pix_fmt " + str(pixel_format)

	cmd_video_settings = " ".join([cmd_resolution, cmd_frame_rate, cmd_pixel_format])

	# Input File
	cmd_input_video_path = "-i " + input_video_path

	# Output Settings
	cmd_output_raw_video = "-f rawvideo -c:v rawvideo"

	# Output File
	cmd_output_video_path =  "-threads 8 {}".format(output_yuv_path)

	# Command
	cmd_ffmpeg = "{}/ffmpeg".format(ffmpeg_path)
	cmd = " ".join([
		cmd_ffmpeg, 
		cmd_input_video_path,
		cmd_output_raw_video, cmd_video_settings, cmd_output_video_path
	])

	# Execute
	subprocess.run(shlex.split(cmd))



# Cropping the raw video to min-frames
def temporal_cropping(
	input_video_path:str,
	output_yuv_path:str,
	resolution:Tuple,
	frame_rate:float,
	pixel_format:str,
	max_num_frames:int,
):
	"""
	Temporal cropping
	
	Args:
		input_video_path (str): Path to input YUV file.
		output_yuv_path (str): Path to input YUV file.
		resolution (tuple): The input resolution needs to provided when video format is yuv as (width, height) as frame size is not stored in the input file. (Default: None)
		frame_rate (float): The frame-rate of the video. (Default: None)
		pixel_format (str): The input video pixel format. Use command `ffmpeg -pix_fmts` to get all the options. (Default: None)
		max_num_frames (int): Maximum no.of frames.
	"""
	# Assertions
	assert os.path.exists(input_video_path), "Provide a valid input_video_path."
	assert resolution is not None, "For a 'YUV' format video input, resolution is required are inputs."
	assert frame_rate is not None, "For a 'YUV' format video input, frame-rate is required are inputs."
	assert pixel_format is not None,  "For a 'YUV' format video input, pixel_format is required are inputs."
	assert max_num_frames is not None and max_num_frames > 0, "Invalid max_num_frames"
	
	# Input/Output Settings
	cmd_resolution = "-video_size " + str(resolution[0]) + "x" + str(resolution[1])
	cmd_frame_rate = "-framerate " + str(frame_rate)
	cmd_pixel_format = "-pix_fmt " + str(pixel_format)

	cmd_video_settings = " ".join([cmd_resolution, cmd_frame_rate, cmd_pixel_format])

	# Input File
	cmd_input_video_path = "-i " + input_video_path

	# Processing
	cmd_crop_frames = "-vframes {}".format(max_num_frames)

	# Output Settings
	cmd_output_raw_video = "-f rawvideo -c:v rawvideo"

	# Output File
	cmd_output_video_path =  "-threads 8 {}".format(output_yuv_path)

	# Command
	cmd_ffmpeg = "{}/ffmpeg".format(ffmpeg_path)
	cmd = " ".join([
		cmd_ffmpeg, 
		cmd_input_video_path,
		cmd_crop_frames,
		cmd_output_raw_video, cmd_video_settings, cmd_output_video_path
	])

	# Execute
	subprocess.run(shlex.split(cmd))