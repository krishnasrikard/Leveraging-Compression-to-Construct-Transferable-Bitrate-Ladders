"""
Calculating BD-metrics
"""
# Importing Libraries
import numpy as np
import matplotlib.pyplot as plt

import os, sys, warnings
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import pickle
import modules.bitrate_ladder_evaluation_functions as bitrate_ladder_evaluation_functions
import defaults


def calculate_closeness(
	# Files
	Test_Video_Files:list,

	# Path
	save_path:str,

	# Arguments
	Predicted_BD_Metrics:dict,
	Reference_BD_Metrics:dict
):
	# Assertions
	assert len(Predicted_BD_Metrics.keys()) > 0.1 * len(Test_Video_Files), "No.of video-files in Predicted BD-Metrics is less 90% of no.of Test Video Files"
	assert len(Reference_BD_Metrics.keys()) == len(Test_Video_Files), "No.of video-files in Predicted BD-Metrics is less 100% of no.of Test Video Files"

	# Closeness
	Closeness = {}

	for video_file in Predicted_BD_Metrics.keys():
		Closeness[video_file] = bitrate_ladder_evaluation_functions.Calculate_Closeness(
			Predicted_Metrics=Predicted_BD_Metrics[video_file],
			Reference_Metrics=Reference_BD_Metrics[video_file], 
		)

	# Saving Closeness
	np.save(save_path, Closeness)



def main(
	# Files
	Test_Video_Files:list,

	# Path
	results_dir,
):
	# Calculating Closeness of Cross-Over Bitrate Ladders
	# """
	print ()
	print ("-"*10, "Closeness Cross-Over Bitrate Ladders", "-"*10)
	print ()

	# Creating Directories
	os.makedirs(os.path.join(results_dir, "CrossOver_Bitrates", "closeness"), exist_ok=True)

	for codec in defaults.codec_preset_pairs.keys():
		for preset in defaults.codec_preset_pairs[codec]:
			print ("Codec: {}, Preset: {}".format(codec, preset))
			print ()

			calculate_closeness(
				# Files
				Test_Video_Files=Test_Video_Files,

				# Path
				save_path = os.path.join(
					results_dir, "CrossOver_Bitrates", "closeness", "{}_{}_low_level_features.npy".format(codec, preset)
				),

				# Arguments
				Predicted_BD_Metrics = np.load(
					os.path.join(
						results_dir, "CrossOver_Bitrates", "bd_metrics", "{}_{}_low_level_features.npy".format(codec, preset)
					), allow_pickle=True
				)[()],

				Reference_BD_Metrics =  np.load(
					os.path.join(
						results_dir, "Standard", "bd_metrics", "Convex_Hull", "{}_{}.npy".format(codec, preset)
					), allow_pickle=True
				)[()]
			)
	# """

			
	# Constructing Closeness with and without compression
	for compression_statistics in [False, True]:
		# Sub-Folder
		if compression_statistics:
			subfolder = "With_Compression"
		else:
			subfolder = "Without_Compression"
		
		# Creating Directories
		os.makedirs(os.path.join(results_dir, subfolder, "closeness"), exist_ok=True)


		# Calculating Closeness of Metadata Bitrate Ladders
		# """
		print ()
		print ("-"*10, "Closeness Metadata Bitrate Ladders", "-"*10)
		print ()

		for codec in defaults.codec_preset_pairs.keys():
			for preset in defaults.codec_preset_pairs[codec]:
				print ("Codec: {}, Preset: {}".format(codec, preset))
				print ()

				calculate_closeness(
					# Files
					Test_Video_Files=Test_Video_Files,

					# Path
					save_path = os.path.join(
						results_dir, subfolder, "closeness", "{}_{}_metadata.npy".format(codec, preset)
					),

					# Arguments
					Predicted_BD_Metrics = np.load(
						os.path.join(
							results_dir, subfolder, "bd_metrics", "{}_{}_metadata.npy".format(codec, preset)
						), allow_pickle=True
					)[()],

					Reference_BD_Metrics =  np.load(
						os.path.join(
							results_dir, "Standard", "bd_metrics", "Convex_Hull", "{}_{}.npy".format(codec, preset)
						), allow_pickle=True
					)[()]
				)	
		# """


		# Calculating Closeness of Low-Level Features Bitrate Ladders
		# """
		print ()
		print ("-"*10, "Closeness Low-Level Bitrate Ladders", "-"*10)
		print ()

		for codec in defaults.codec_preset_pairs.keys():
			for preset in defaults.codec_preset_pairs[codec]:
				print ("Codec: {}, Preset: {}".format(codec, preset))
				print ()

				calculate_closeness(
					# Files
					Test_Video_Files=Test_Video_Files,

					# Path
					save_path = os.path.join(
						results_dir, subfolder, "closeness", "{}_{}_low_level_features.npy".format(codec, preset)
					),

					# Arguments
					Predicted_BD_Metrics = np.load(
						os.path.join(
							results_dir, subfolder, "bd_metrics", "{}_{}_low_level_features.npy".format(codec, preset)
						), allow_pickle=True
					)[()],

					Reference_BD_Metrics =  np.load(
						os.path.join(
							results_dir, "Standard", "bd_metrics", "Convex_Hull", "{}_{}.npy".format(codec, preset)
						), allow_pickle=True
					)[()]
				)
		# """


		# Calculating Closeness of VIF Features Bitrate Ladders
		# """
		print ()
		print ("-"*10, "Closeness VIF Bitrate Ladders", "-"*10)
		print ()

		for codec in defaults.codec_preset_pairs.keys():
			for preset in defaults.codec_preset_pairs[codec]:
				print ("Codec: {}, Preset: {}".format(codec, preset))
				print ()

				calculate_closeness(
					# Files
					Test_Video_Files=Test_Video_Files,

					# Path
					save_path = os.path.join(
						results_dir, subfolder, "closeness", "{}_{}_vif_features.npy".format(codec, preset)
					),

					# Arguments
					Predicted_BD_Metrics = np.load(
						os.path.join(
							results_dir, subfolder, "bd_metrics", "{}_{}_vif_features.npy".format(codec, preset)
						), allow_pickle=True
					)[()],

					Reference_BD_Metrics =  np.load(
						os.path.join(
							results_dir, "Standard", "bd_metrics", "Convex_Hull", "{}_{}.npy".format(codec, preset)
						), allow_pickle=True
					)[()]
				)
		# """


		# Calculating Closeness of Low-Level Features and VIF Features Bitrate Ladders
		# """
		print ()
		print ("-"*10, "Closeness Low-Level Features and VIF Features Bitrate Ladders", "-"*10)
		print ()

		for codec in defaults.codec_preset_pairs.keys():
			for preset in defaults.codec_preset_pairs[codec]:
				print ("Codec: {}, Preset: {}".format(codec, preset))
				print ()

				calculate_closeness(
					# Files
					Test_Video_Files=Test_Video_Files,

					# Path
					save_path = os.path.join(
						results_dir, subfolder, "closeness", "{}_{}_low_level_features_vif_features.npy".format(codec, preset)
					),

					# Arguments
					Predicted_BD_Metrics = np.load(
						os.path.join(
							results_dir, subfolder, "bd_metrics", "{}_{}_low_level_features_vif_features.npy".format(codec, preset)
						), allow_pickle=True
					)[()],

					Reference_BD_Metrics =  np.load(
						os.path.join(
							results_dir, "Standard", "bd_metrics", "Convex_Hull", "{}_{}.npy".format(codec, preset)
						), allow_pickle=True
					)[()]
				)
		# """