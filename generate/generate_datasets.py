"""
Generating RQ points after downscaling, compression and quality estimation.
Saving compressed videos
"""
# Imporint Libraries
import pandas as pd

import os, sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import argparse
import functions.rate_quality_estimations as rate_quality_estimations
import defaults


def main(args):
	# Assertions of Paths
	assert os.path.exists(args.main_path), "Path to main directory does not exist."
	assert os.path.exists(args.raw_videos_path), "Path to raw-videos does not exist."
	
	if args.rq_points_dataset != "None":
		assert os.path.exists(args.rq_points_dataset), "Path to RQ-points does not exist."
	else:
		args.rq_points_dataset = None
		
	if args.compressed_videos_dataset != "None":
		assert os.path.exists(args.compressed_videos_dataset), "Path to compressed videos dataset does not exist."
	else:
		args.compressed_videos_dataset = None


	# Assertions
	assert args.rate_control in ["crf", "qp"], 'Provide a valid rate-control. Rate-Control: ["crf", "qp"]'


	# Creating a temporary paths
	temp_folders_path = os.path.join(args.rq_points_dataset, "temp_folders")
	if os.path.exists(temp_folders_path) == False:
		os.mkdir(temp_folders_path)
	temp_path = os.path.join(temp_folders_path, "temp_{}_{}_{}".format(args.codec, args.preset, args.rate_control))
	if os.path.exists(temp_path) == False:
		os.mkdir(temp_path)


	# Selected Rate-Control
	if args.rate_control == "qp":
		selected_qps = defaults.QPs
		selected_crfs = None
	elif args.rate_control == "crf":
		selected_qps = None
		selected_crfs = defaults.codec_full_CRF_ranges[args.codec]
	else:
		None


	# Video Properties
	df = pd.read_csv(defaults.source_dataset_info_path)
	df = df.set_index("filename").to_dict(orient="index")


	# Generating RQ-Points for each Video File
	for video_file in sorted(os.listdir(args.raw_videos_path)):
		# File and Properties
		filename = os.path.splitext(os.path.basename(video_file))[0]
		video_properties = df[filename]

		# Logging
		print ("-"*75 + "\n" + filename + "\n" + "-"*75, flush=True)

		# Calculating
		rate_quality_estimations.estimate_rate_quality_points(
			# Input
			input_yuv_path=os.path.join(args.raw_videos_path, video_file),
			input_resolution=(video_properties["width"], video_properties["height"]),
			input_framerate=video_properties["avg_frame_rate"],
			input_pixelformat=video_properties["pix_fmt"],

			# Encoder Settings
			output_resolutions=defaults.resolutions,
			codec=args.codec,
			preset=args.preset,
			QPs=selected_qps,
			CRFs=selected_crfs,

			# Save Paths
			rq_points_output_path=args.rq_points_dataset,
			compressed_videos_output_path=args.compressed_videos_dataset,
			temp_path=temp_path,

			# Quality Estimation
			vmaf_resolution=(3840,2160),
			quality_metrics={"PSNR":True, "SSIM":False, "MS_SSIM":False},

			# Multi-Threading
			num_threads=args.num_threads
		)


# Calling Main function
if __name__ == '__main__':
	root_dir = os.path.dirname(os.path.realpath(__file__))

	# Get Arguments
	parser = argparse.ArgumentParser(description='Estimating compressed video information')

	# Dataset Paths
	parser.add_argument(
		'--raw_videos_path', 
		default=defaults.source_dataset_path, 
		help='Path to dataset.'
	)
	parser.add_argument(
		'--rq_points_dataset', 
		default=defaults.rq_points_dataset_path, 
		help='Path to RQ points dataset.'
	)
	parser.add_argument(
		'--compressed_videos_dataset', 
		default='None', 
		help='Path to compressed videos dataset.'
	)

	# Main Path
	parser.add_argument(
		'--main_path', 
		default=os.path.join(defaults.main_working_dir, 'generate'), 
		type=str, 
		help='Path to main folder'
	)

	# Processing
	parser.add_argument(
		'--num_threads', 
		default=8, 
		type=int, 
		help='No.of threads used to run ffmpeg commands. Generally no.of threads are set between 4-8 for ffmpeg. (Default: 8)'
	)

	# Select Codec and Preset
	parser.add_argument(
		'--codec', 
		default="libx265", 
		type=str, 
		help='Index of the codec that needs to be selected. Options: ["libx265", "libvpx-vp9", "libaom-av1", "libsvtav1"]. (Default: "libx265")'
	)

	parser.add_argument(
		'--preset', 
		default="medium", 
		type=str, 
		help='Index of the preset that needs to be selected. Options: Vary depending on video codec. (Default: "medium")'
	)

	parser.add_argument(
		'--rate_control', 
		default='crf', 
		type=str, 
		help='Index of the rate-control method that needs to be selected. Options: ["crf", "qp"]. (Default: "crf")'
	)


	# Parse Arguments
	args = parser.parse_args()


	# -----------------------------------------------------------------
	# Flushing Output
	import functools
	print = functools.partial(print, flush=True)

	# Saving stdout
	sys.stdout = open('logs/{}_{}_{}.log'.format(args.codec, args.preset, os.path.basename(__file__)[:-3]), 'w')

	# -----------------------------------------------------------------
	

	main(args)