"""
Processing BVI-DVC dataset and AV2 videos
BVI-DVC dataset: https://fan-aaron-zhang.github.io/BVI-DVC/
AV2 Videos: https://media.xiph.org/video/av2/
"""
# Importing Libraries
import pandas as pd

import os, sys, warnings
warnings.filterwarnings("ignore")
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import joblib
from typing import Type, Callable, Tuple, Optional, Set, List, Union
import subprocess, shlex, shutil
import functions.software_commands as software_commands
import functions.software_executions as software_executions
import functions.IO_functions as IO_functions
import defaults


# Saving Video Properties
def save_dataset_properties(
	video_paths:list,
	output_csv_file:str
):
	# Video Properties
	video_properties = []

	for video_path in video_paths:
		properties, log = software_executions.ffprobe_get_video_properties(
			input_video_path=video_path
		)
		if properties == {}:
			print (log)
			assert False, "Properties are not extracted"

		properties["filename"] = os.path.splitext(os.path.basename(video_path))[0]
		video_properties.append(properties)

	# Saving as pandas Dataframe
	df = pd.DataFrame(video_properties, columns=["filename", "codec_name", "width", "height", "pix_fmt", "avg_frame_rate"])
	df.to_csv(output_csv_file, sep=",", index=False)



if __name__ == "__main__":
	# List of Files in our Dataset
	video_paths = []
	for video_file in os.listdir("/home/krishna/Nebula/krishna/Xiph"):
		video_paths.append(
			os.path.join("/home/krishna/Nebula/krishna/Xiph", video_file)
		)
	for video_file in os.listdir("/home/krishna/Nebula/krishna/BVI-DVC"):
		if video_file.__contains__("Harmonics"):
			continue
		video_paths.append(
			os.path.join("/home/krishna/Nebula/krishna/BVI-DVC", video_file)
		)


	# Save videos properties
	save_dataset_properties(
		video_paths=video_paths,
		output_csv_file="/home/krishna/Nebula/krishna/4K_Shots/video_properties.csv"
	)


	# Convert to YUV
	df = pd.read_csv("/home/krishna/Nebula/krishna/4K_Shots/video_properties.csv")
	df = df.set_index("filename").to_dict(orient="index")

	for video_path in video_paths:
		# File and Properties
		filename = os.path.splitext(os.path.basename(video_path))[0]
		video_properties = df[filename]
		
		# Debugging
		print (filename)
		print ()

		# Convert
		software_executions.temporal_cropping(
			input_video_path=video_path,
			output_yuv_path=os.path.join(
				"/home/krishna/Nebula/krishna/4K_Shots/yuv_files", 
				filename + ".yuv"
			),
			resolution=(video_properties["width"], video_properties["height"]),
			frame_rate=video_properties["avg_frame_rate"],
			pixel_format=video_properties["pix_fmt"],
			max_num_frames=64
		)


	# Convert to x265
	df = pd.read_csv("/home/krishna/Nebula/krishna/4K_Shots/video_properties.csv")
	df = df.set_index("filename").to_dict(orient="index")
	yuv_files = os.listdir("/home/krishna/Nebula/krishna/4K_Shots/yuv_files")

	for yuv_file in yuv_files:
		# File and Properties
		filename = os.path.splitext(os.path.basename(yuv_file))[0]
		video_properties = df[filename]
		
		# Debugging
		print (filename)
		print ()

		# Convert
		cmd = software_commands.compression_command(
			raw_video=True,
			input_resolution=(video_properties["width"], video_properties["height"]),
			frame_rate=video_properties["avg_frame_rate"],
			pixel_format=video_properties["pix_fmt"],
			input_video_path=os.path.join(
				"/home/krishna/Nebula/krishna/4K_Shots/yuv_files", 
				filename + ".yuv"
			),
			output_resolution=(video_properties["width"], video_properties["height"]),
			scaling_algo="lanczos",
			video_codec="libx265",
			preset="veryfast",
			QP=None,
			CRF=23,
			bitrate=None,
			output_video_path=os.path.join(
				"/home/krishna/Nebula/krishna/4K_Shots/mp4_files", 
				filename + ".mp4"
			),
			num_threads=8
		)

		subprocess.run(shlex.split(cmd))