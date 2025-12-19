"""
Analyzing B-Frame in SVT-AV1 video.
"""
# Importing Libraries
import numpy as np
import matplotlib.pyplot as plt

import os, sys, warnings
import pickle
from tqdm import tqdm
warnings.filterwarnings("ignore")
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import functions.IO_functions as IO_functions
import functions.extract_functions as extract_functions
import defaults

for codec in defaults.codec_preset_pairs.keys():
	for preset in defaults.codec_preset_pairs[codec]:
		# Plotting Pareto-Front
		folders_path = os.path.join(defaults.rq_points_dataset_path, codec, preset)

		for video_filename in tqdm(defaults.Video_Titles):
			video_rq_points_info = IO_functions.read_create_jsonfile(os.path.join(folders_path, video_filename, "crfs.json"))

			# Analyzing Frame-Count
			"""
			for resolution in defaults.resolutions:
				for crf in defaults.codec_CRF_ranges[codec]:
					if video_rq_points_info["{}x{}".format(resolution[0], resolution[1])][str(crf)]["I_frame_Count"] > 1:
						print (codec, preset, video_filename, resolution, crf, video_rq_points_info["{}x{}".format(resolution[0], resolution[1])][str(crf)]["I_frame_Count"], video_rq_points_info["{}x{}".format(resolution[0], resolution[1])][str(crf)]["P_frame_Count"], video_rq_points_info["{}x{}".format(resolution[0], resolution[1])][str(crf)]["B_frame_Count"])"
			"""
			
			# Analyzing Bitrate of B-Frames
			RQ_pairs = extract_functions.Extract_RQ_Information(
				video_rq_points_info=video_rq_points_info,
				quality_metric="vmaf",
				resolutions=defaults.resolutions,
				CRFs=defaults.codec_CRF_ranges[codec],
				QPs=None,
				min_quality=defaults.min_quality,
				max_quality=defaults.max_quality,
				min_bitrate=defaults.min_bitrate,
				max_bitrate=defaults.max_bitrate,
				set_bitrate_log_base=2
			)

			for resolution in defaults.resolutions:
				for i in range(len(RQ_pairs[resolution])):
					if RQ_pairs[resolution][i,4] == -1 and RQ_pairs[resolution][i,7] == -1:
						print (codec, preset, video_filename, resolution)