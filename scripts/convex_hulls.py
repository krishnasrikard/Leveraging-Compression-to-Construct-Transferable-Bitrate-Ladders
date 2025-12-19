"""
Construct Bitrate Ladders for each Convex-Hull
"""
# Importing Libraries
import numpy as np
import matplotlib.pyplot as plt

import os, sys, warnings
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import modules.convex_hull_functions as convex_hull_functions
import modules.bitrate_ladder_evaluation_functions as bitrate_ladder_evaluation_functions
import defaults


def main(
	# Files
	Test_Video_Files:list,

	# Path
	results_dir,
):
	# Creating Directories
	os.makedirs(os.path.join(results_dir, "Standard", "bitrate_ladders"), exist_ok=True)
	os.makedirs(os.path.join(results_dir, "Standard", "bitrate_ladders", "Convex_Hull"), exist_ok=True)
	os.makedirs(os.path.join(results_dir, "Standard", "bd_metrics"), exist_ok=True)
	os.makedirs(os.path.join(results_dir, "Standard", "bd_metrics", "Convex_Hull"), exist_ok=True)
	os.makedirs(os.path.join(results_dir, "Standard", "bd_metrics", "Two_Step_Convex_Hull"), exist_ok=True)


	# Constructing Apple's Fixed Bitrate Ladder
	print ()
	print ("-"*10, "Apple's Fixed Bitrate Ladder", "-"*10)
	print ()

	Fixed_Bitrate_Ladder = convex_hull_functions.Construct_Apple_Fixed_Bitrate_Ladder(
		evaluation_bitrates=np.copy(defaults.evaluation_bitrates),
	)

	np.save(
		os.path.join(
			results_dir, "Standard", "bitrate_ladders", "fixed_bitrate_ladder.npy"
		),
		Fixed_Bitrate_Ladder
	)


	# Constructing Convex-Hull Bitrate Ladders
	print ()
	print ("-"*10, "Convex Hull Bitrate Ladder", "-"*10)
	print ()

	for codec in defaults.codec_preset_pairs.keys():
		for preset in defaults.codec_preset_pairs[codec]:
			print ("Codec: {}, Preset: {}".format(codec, preset))
			print ()

			Convex_Hull_Bitrate_Ladder = {}

			for video_file in Test_Video_Files:
				Convex_Hull_Bitrate_Ladder[video_file] = convex_hull_functions.Construct_Convex_Hull_Bitrate_Ladder(
					# Video and Encoder Settings
					video_file=video_file,
					codec=codec,
					preset=preset,

					# Evaluation Bitrates
					evaluation_bitrates=defaults.evaluation_bitrates
				)

			np.save(
				os.path.join(
					results_dir, "Standard", "bitrate_ladders", "Convex_Hull", "{}_{}.npy".format(codec, preset)
				),
				Convex_Hull_Bitrate_Ladder
			)


	# Calculating BD Metrics of Convex-Hull Bitrate Ladders
	print ()
	print ("-"*10, "BD Metrics Convex Hull", "-"*10)
	print ()

	for codec in defaults.codec_preset_pairs.keys():
		for preset in defaults.codec_preset_pairs[codec]:
			print ("Codec: {}, Preset: {}".format(codec, preset))
			print ()

			BD_Metrics_Convex_Hull = {}

			for video_file in Test_Video_Files:
				Metrics = bitrate_ladder_evaluation_functions.Calculate_BD_Metrics(
					# Video and Encoder Settings
					video_file=video_file,
					codec=codec,
					preset=preset,

					# Bitrate Ladders
					bitrate_ladder_path = os.path.join(
						results_dir, "Standard", "bitrate_ladders", "Convex_Hull", "{}_{}.npy".format(codec, preset)
					),
					fixed_bitrate_ladder_path = os.path.join(
						results_dir, "Standard", "bitrate_ladders", "fixed_bitrate_ladder.npy"
					),
					convex_hull_bitrate_ladder_path = os.path.join(
						results_dir, "Standard", "bitrate_ladders", "Convex_Hull", "{}_{}.npy".format(codec, preset)
					)
				)

				# Assertion
				if Metrics is None:
					assert False, "BD-Metrics of Convex-Hull is None for video-file: {}".format(video_file)
				else:
					BD_Metrics_Convex_Hull[video_file] = Metrics

			np.save(
				os.path.join(
					results_dir, "Standard", "bd_metrics", "Convex_Hull", "{}_{}.npy".format(codec, preset)
				),
				BD_Metrics_Convex_Hull
			)


	# Calculating BD Metrics of libx265-veryfast Convex-Hull Bitrate Ladders
	print ()
	print ("-"*10, "BD Metrics libx265-veryfast Convex Hull", "-"*10)
	print ()

	for codec in defaults.codec_preset_pairs.keys():
		for preset in defaults.codec_preset_pairs[codec]:
			print ("Codec: {}, Preset: {}".format(codec, preset))
			print ()

			BD_Metrics_Convex_Hull = {}

			for video_file in Test_Video_Files:
				Metrics = bitrate_ladder_evaluation_functions.Calculate_BD_Metrics(
					# Video and Encoder Settings
					video_file=video_file,
					codec=codec,
					preset=preset,

					# Bitrate Ladders
					bitrate_ladder_path = os.path.join(
						results_dir, "Standard", "bitrate_ladders", "Convex_Hull", "{}_{}.npy".format("libx265", "veryfast")
					),
					fixed_bitrate_ladder_path = os.path.join(
						results_dir, "Standard", "bitrate_ladders", "fixed_bitrate_ladder.npy"
					),
					convex_hull_bitrate_ladder_path = os.path.join(
						results_dir, "Standard", "bitrate_ladders", "Convex_Hull", "{}_{}.npy".format(codec, preset)
					)
				)

				# Assertion
				if Metrics is None:
					assert False, "BD-Metrics of Convex-Hull constructed using libx265,veryfast Convex-Hull is None for video-file: {}".format(video_file)
				else:
					BD_Metrics_Convex_Hull[video_file] = Metrics

			np.save(
				os.path.join(
					results_dir, "Standard", "bd_metrics", "Two_Step_Convex_Hull", "{}_{}.npy".format(codec, preset)
				),
				BD_Metrics_Convex_Hull
			)