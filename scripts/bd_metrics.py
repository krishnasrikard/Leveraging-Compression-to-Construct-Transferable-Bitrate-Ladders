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


def calculate_bd_metrics(
	# Files
	Test_Video_Files:list,

	# Path
	save_path:str,

	# Arguments:
	codec:str,
	preset:str,
	bitrate_ladder_path:str,
	fixed_bitrate_ladder_path:str,
	convex_hull_bitrate_ladder_path:str
):
	# Calculating BD-Metrics
	BD_Metrics = {}
	skipped_files = []

	for video_file in Test_Video_Files:
		Metrics = bitrate_ladder_evaluation_functions.Calculate_BD_Metrics(
			# Video and Encoder Settings
			video_file=video_file,
			codec=codec,
			preset=preset,

			# Bitrate Ladders
			bitrate_ladder_path=bitrate_ladder_path,
			fixed_bitrate_ladder_path=fixed_bitrate_ladder_path,
			convex_hull_bitrate_ladder_path=convex_hull_bitrate_ladder_path
		)
				
		# Increasing Skipped Files
		if Metrics is None:
			skipped_files.append(video_file)
		else:
			BD_Metrics[video_file] = Metrics

		# If we skip more than 2.5% of the files, throw an error.
		assert len(skipped_files) <= 0.025*len(Test_Video_Files), "Skipped too many files."
		if len(skipped_files) > 0:
			print ("Skipped files:\n", skipped_files)

	# Saving BD-Metrics
	np.save(save_path, BD_Metrics)



def main(
	# Files
	Test_Video_Files:list,

	# Path
	results_dir,
):
	# Calculating BD Metrics of Cross-Over Bitrate Ladders
	# """
	print ()
	print ("-"*10, "Cross-Over Bitrate Ladders", "-"*10)
	print ()

	# Creating Directories
	os.makedirs(os.path.join(results_dir, "CrossOver_Bitrates", "bd_metrics"), exist_ok=True)

	for codec in defaults.codec_preset_pairs.keys():
		for preset in defaults.codec_preset_pairs[codec]:
			print ("Codec: {}, Preset: {}".format(codec, preset))
			print ()

			calculate_bd_metrics(
				# Files
				Test_Video_Files=Test_Video_Files,

				# Paths
				save_path=os.path.join(
					results_dir, "CrossOver_Bitrates", "bd_metrics", "{}_{}_low_level_features.npy".format(codec, preset)
				),

				# Arguments
				codec=codec,
				preset=preset,
				bitrate_ladder_path = os.path.join(
					results_dir, "CrossOver_Bitrates", "bitrate_ladders", "low_level_features.npy"
				),
				fixed_bitrate_ladder_path = os.path.join(
					results_dir, "Standard", "bitrate_ladders", "fixed_bitrate_ladder.npy"
				),
				convex_hull_bitrate_ladder_path = os.path.join(
					results_dir, "Standard", "bitrate_ladders", "Convex_Hull", "{}_{}.npy".format(codec, preset)
				)
			)
	# """


	# Constructing BD metrics with and without compression
	for compression_statistics in [False, True]:
		# Sub-Folder
		if compression_statistics:
			subfolder = "With_Compression"
		else:
			subfolder = "Without_Compression"
		
		# Creating Directories
		os.makedirs(os.path.join(results_dir, subfolder, "bd_metrics"), exist_ok=True)


		# Calculating BD Metrics of Metadata Bitrate Ladders
		# """
		print ()
		print ("-"*10, "BD Metrics Metadata Bitrate Ladders", "-"*10)
		print ()

		for codec in defaults.codec_preset_pairs.keys():
			for preset in defaults.codec_preset_pairs[codec]:
				print ("Codec: {}, Preset: {}".format(codec, preset))
				print ()

				calculate_bd_metrics(
					# Files
					Test_Video_Files=Test_Video_Files,

					# Paths
					save_path=os.path.join(
						results_dir, subfolder, "bd_metrics", "{}_{}_metadata.npy".format(codec, preset)
					),

					# Arguments
					codec=codec,
					preset=preset,
					bitrate_ladder_path = os.path.join(
						results_dir, subfolder, "bitrate_ladders", "metadata.npy"
					),
					fixed_bitrate_ladder_path = os.path.join(
						results_dir, "Standard", "bitrate_ladders", "fixed_bitrate_ladder.npy"
					),
					convex_hull_bitrate_ladder_path = os.path.join(
						results_dir, "Standard", "bitrate_ladders", "Convex_Hull", "{}_{}.npy".format(codec, preset)
					)
				)
		# """


		# Calculating BD Metrics of Low-Level Features Bitrate Ladders
		# """
		print ()
		print ("-"*10, "BD Metrics Low-Level Bitrate Ladders", "-"*10)
		print ()

		for codec in defaults.codec_preset_pairs.keys():
			for preset in defaults.codec_preset_pairs[codec]:
				print ("Codec: {}, Preset: {}".format(codec, preset))
				print ()

				calculate_bd_metrics(
					# Files
					Test_Video_Files=Test_Video_Files,

					# Paths
					save_path=os.path.join(
						results_dir, subfolder, "bd_metrics", "{}_{}_low_level_features.npy".format(codec, preset)
					),

					# Arguments
					codec=codec,
					preset=preset,
					bitrate_ladder_path = os.path.join(
						results_dir, subfolder, "bitrate_ladders", "low_level_features.npy"
					),
					fixed_bitrate_ladder_path = os.path.join(
						results_dir, "Standard", "bitrate_ladders", "fixed_bitrate_ladder.npy"
					),
					convex_hull_bitrate_ladder_path = os.path.join(
						results_dir, "Standard", "bitrate_ladders", "Convex_Hull", "{}_{}.npy".format(codec, preset)
					)
				)
		# """


		# Calculating BD Metrics of VIF Features Bitrate Ladders
		# """
		print ()
		print ("-"*10, "BD Metrics VIF Bitrate Ladders", "-"*10)
		print ()

		for codec in defaults.codec_preset_pairs.keys():
			for preset in defaults.codec_preset_pairs[codec]:
				print ("Codec: {}, Preset: {}".format(codec, preset))
				print ()

				calculate_bd_metrics(
					# Files
					Test_Video_Files=Test_Video_Files,

					# Paths
					save_path=os.path.join(
						results_dir, subfolder, "bd_metrics", "{}_{}_vif_features.npy".format(codec, preset)
					),

					# Arguments
					codec=codec,
					preset=preset,
					bitrate_ladder_path = os.path.join(
						results_dir, subfolder, "bitrate_ladders", "vif_features.npy"
					),
					fixed_bitrate_ladder_path = os.path.join(
						results_dir, "Standard", "bitrate_ladders", "fixed_bitrate_ladder.npy"
					),
					convex_hull_bitrate_ladder_path = os.path.join(
						results_dir, "Standard", "bitrate_ladders", "Convex_Hull", "{}_{}.npy".format(codec, preset)
					)
				)
		# """


		# Calculating BD Metrics of Low-Level Features and VIF Features Bitrate Ladders
		# """
		print ()
		print ("-"*10, "BD Metrics Low-Level Features and VIF Features Bitrate Ladders", "-"*10)
		print ()

		for codec in defaults.codec_preset_pairs.keys():
			for preset in defaults.codec_preset_pairs[codec]:
				print ("Codec: {}, Preset: {}".format(codec, preset))
				print ()

				calculate_bd_metrics(
					# Files
					Test_Video_Files=Test_Video_Files,

					# Paths
					save_path=os.path.join(
						results_dir, subfolder, "bd_metrics", "{}_{}_low_level_features_vif_features.npy".format(codec, preset)
					),

					# Arguments
					codec=codec,
					preset=preset,
					bitrate_ladder_path = os.path.join(
						results_dir, subfolder, "bitrate_ladders", "low_level_features_vif_features.npy"
					),
					fixed_bitrate_ladder_path = os.path.join(
						results_dir, "Standard", "bitrate_ladders", "fixed_bitrate_ladder.npy"
					),
					convex_hull_bitrate_ladder_path = os.path.join(
						results_dir, "Standard", "bitrate_ladders", "Convex_Hull", "{}_{}.npy".format(codec, preset)
					)
				)
		# """