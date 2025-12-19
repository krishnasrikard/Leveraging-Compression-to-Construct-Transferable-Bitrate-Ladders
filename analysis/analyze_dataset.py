"""
Analyzing Dataset
"""
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

import os, sys, warnings
warnings.filterwarnings("ignore")
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import functions.IO_functions as IO_functions

# Path to dataset
main_dataset_dir = "/home/krishna/Nebula/krishna/4K_Shots/yuv_files"
df = pd.read_csv("/home/krishna/Nebula/krishna/4K_Shots/video_properties.csv")
df = df.set_index("filename").to_dict(orient="index")

for video_file in os.listdir(main_dataset_dir):
	# File and Properties
	filename = os.path.splitext(os.path.basename(video_file))[0]
	video_properties = df[filename]

	print ("Video-File: {}".format(video_file))
	F = IO_functions.YUV_Reader(
		filepath=os.path.join(main_dataset_dir, video_file),
		width=video_properties["width"],
		height=video_properties["height"],
		yuv_type=video_properties["pix_fmt"]
	)

	print (video_file, F.num_frames)
	print ()