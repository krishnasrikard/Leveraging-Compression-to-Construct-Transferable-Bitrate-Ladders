"""
Extract Exection Time i.e. System + User Times for ML operations.
"""

# Importing Libraries
import numpy as np
import cv2

import os, sys, warnings
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import argparse
import pickle
import features.VIF as VIF
import generate.calculate_low_level_features as calculate_low_level_features
import modules.bitrate_ladder_construction_functions as bitrate_ladder_construction_functions
import defaults


# VIF Feature Extraction on a video from calculate_vif_features.py
def extract_vif_features(video):
	# Initializing VIF
	VIF_Function = VIF.Compute_VIF()

	# Computing Reference Video Features
	Reference_Video_Features = []

	# Iterating for each frame
	for i in range(video.shape[0]):
		# Calculating VIF features
		frame = np.copy(video[i])

		# Luma Component of current frame
		# Converting to int32 to avoid overflow during operations.
		frame = cv2.cvtColor(frame, cv2.COLOR_RGB2YUV)[:,:,0]
		frame = frame.astype(np.int32)

		# Assertion
		assert (frame.dtype == np.int32) and (np.min(frame) >= 0 and np.max(frame) <= 255), "Before calculation frame should of type uint8 and should have range [0,255]."


		# Decomposation
		vif_pyr_ref, vif_subband_keys = VIF_Function.Decomposation(frame)
		vif_subband_keys.sort(reverse=True)

		# GSM Model
		[vif_S_squared_all, vif_EigenValues_all] = VIF_Function.GSM_Model(vif_pyr_ref, vif_subband_keys)

		# Information in each subband along each eigen value
		vif_features_reference = VIF_Function.Reference_Subband_Eigen_Information_Matrix(
			subband_keys=vif_subband_keys, S_squared_all=vif_S_squared_all, EigenValues_all=vif_EigenValues_all
		)


		# Calculating Diff-VIF (T-VIF) Features
		if i == 0:
			current_frame = np.zeros(video[i].shape, dtype=np.uint8)
			previous_frame = np.zeros(video[i].shape, dtype=np.uint8)
		else:
			current_frame = np.copy(video[i])
			previous_frame = np.copy(video[i-1])

		# Luma Component of current frame
		# Converting to int32 to avoid overflow during operations.
		current_frame = cv2.cvtColor(current_frame, cv2.COLOR_RGB2YUV)[:,:,0]
		current_frame = current_frame.astype(np.int32)
			
		# Luma Component of previous frame
		# Converting to int32 to avoid overflow during operations.
		previous_frame = cv2.cvtColor(previous_frame, cv2.COLOR_RGB2YUV)[:,:,0]
		previous_frame = previous_frame.astype(np.int32)

		# Assertions
		assert (current_frame.dtype == np.int32) and (np.min(current_frame) >= 0 and np.max(current_frame) <= 255), "Before calculation frame should of type int32 and should have range [0,255]."
		assert (previous_frame.dtype == np.int32) and (np.min(previous_frame) >= 0 and np.max(previous_frame) <= 255), "Before calculation frame should of type int32 and should have range [0,255]."

		# Frame Difference
		diff_frame = np.copy(current_frame - previous_frame)


		# Decomposation
		diff_vif_pyr_ref, diff_vif_subband_keys = VIF_Function.Decomposation(diff_frame)
		diff_vif_subband_keys.sort(reverse=True)

		# GSM Model
		[diff_vif_S_squared_all, diff_vif_EigenValues_all] = VIF_Function.GSM_Model(diff_vif_pyr_ref, diff_vif_subband_keys)

		# Information in each subband along each eigen value
		diff_vif_features_reference = VIF_Function.Reference_Subband_Eigen_Information_Matrix(
			subband_keys=diff_vif_subband_keys, S_squared_all=diff_vif_S_squared_all, EigenValues_all=diff_vif_EigenValues_all
		)

		# Appending reference video features and all other parameters
		Reference_Video_Features.append({"vif_info":vif_features_reference, "diff_vif_info":diff_vif_features_reference, "mean_abs_frame_diff":np.mean(np.abs(diff_frame))})



# Execution Time Functions
def Execution_Time_LLF(video_path):
	# Load Video
	video = np.load(video_path)

	# Calculating Low-Level Features
	F = calculate_low_level_features.generate_low_level_features(video)
	F.generate_low_level_features()


def Execution_Time_VIF(video_path):
	# Load Video
	video = np.load(video_path)

	# Calculating VIF Features
	extract_vif_features(
		video=video
	)


def Execution_Time_ExtraTrees(video_file):
	## Features
	# Low-Level Features (Custom-Features always at the end so as to match code in 'dataset_evaluation_functions.py')
	features_names = []
	for features_subset in [defaults.glcm_features, defaults.tc_features, defaults.si_features, defaults.ti_features, defaults.cti_features, defaults.cf_features, defaults.ci_features, defaults.dct_features, list(defaults.bitrate_texture_features.keys())]:
		for f in features_subset:
			features_names.append(f)

	# VIF-Approach Number
	vif_approach_number = "9"

	VIF_Approach_Map = {
		"1": [["vif_info"], "per_scale"],
		"2": [["vif_info"], "per_subband"],
		"3": [["vif_info"], "per_eigen_value"],
		"4": [["vif_info", "mean_abs_frame_diff"], "per_scale"],
		"5": [["vif_info", "mean_abs_frame_diff"], "per_subband"],
		"6": [["vif_info", "mean_abs_frame_diff"], "per_eigen_value"],
		"7": [["vif_info", "mean_abs_frame_diff", "diff_vif_info"], "per_scale"],
		"8": [["vif_info", "mean_abs_frame_diff", "diff_vif_info"], "per_subband"],
		"9": [["vif_info", "mean_abs_frame_diff", "diff_vif_info"], "per_eigen_value"],
	}

	
	Inputs, _, _ = bitrate_ladder_construction_functions.Bitrate_Ladder_Construction_with_LowLevelFeatures_VIFFeatures(
		# Files
		video_filenames=[video_file],

		# Method Arguments
		features_names=features_names,
		temporal_low_level_features=False,
		per_frame=False,
		per_frame_features_flatten=False,
		vif_setting=VIF_Approach_Map[vif_approach_number][1],
		vif_features_list=VIF_Approach_Map[vif_approach_number][0],

		# Arguments
		codec="libx265",
		preset="veryfast",
		quality_metric="vmaf",
		compression_statistics=True,
		Resolutions_Considered=defaults.resolutions,
		evaluation_crfs=defaults.codec_evaluation_CRF_ranges["libx265"],
		evaluation_qps=None,
		evaluation_bitrates=defaults.evaluation_bitrates,
		min_quality=defaults.min_quality,
		max_quality=defaults.max_quality,
		min_bitrate=defaults.min_bitrate,
		max_bitrate=defaults.max_bitrate
	)
	
	for i in range(5):
		# Load Model
		Model = pickle.load(open("/home/krishna/Leveraging-Compression-to-Construct-Transferable-Bitrate-Ladders/results/main/libx265/Split-{}/With_Compression/models/low_level_features_vif_features.pkl".format(i), "rb"))

		# Predicting Quality
		Model.predict(Inputs[defaults.Video_Titles[0]]).flatten()



# Main Function
if __name__ == "__main__":
	# Get Arguments
	parser = argparse.ArgumentParser(description='')

	parser.add_argument(
		'--func', 
		help='Function to execute.'
	)
	parser.add_argument(
		'--video_file', 
		help='Video-File'
	)
	parser.add_argument(
		'--video_path', 
		help='Width of Source Video'
	)

	# Parse Arguments
	args = parser.parse_args()

	# Get Time
	if args.func == "LLF":
		print (args.func)
		Execution_Time_LLF(
			video_path=args.video_path
		)
	elif args.func == "VIF":
		print (args.func)
		Execution_Time_VIF(
			video_path=args.video_path
		)
	elif args.func == "ExtraTrees":
		print (args.func)
		Execution_Time_ExtraTrees(
			video_file=args.video_file
		)
	else:
		assert False, "Unknown Func = {}".format(args.func)