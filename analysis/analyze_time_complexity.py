"""
Analyzing Time-Complexity
"""
# Importing Libraries
import numpy as np
import matplotlib.pyplot as plt

import os, sys, warnings
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from tqdm import tqdm
import functions.IO_functions as IO_functions
import defaults


# Time-Complexity
def calculate_time_complexity(
	codec:str,
	preset:str,
	video_files:list,
):
	"""
	Calculating Time-Complexity

	Args:
		codec (str): Codec
		preset (str): Preset of Codec
		video_files (list): List of video-files to consider.
	"""
	# Mean Compression and VMAF time
	Compression_time = {}
	VMAF_time = {}
	Bitrate = {}
	Quality = {}

	for resolution in defaults.resolutions:
		Compression_time[resolution] = []
		VMAF_time[resolution] = []
		Bitrate[resolution] = []
		Quality[resolution] = []

	# Extract Time
	for video_filename in tqdm(video_files, desc="{}_{}".format(codec, preset)):
		path = os.path.join(defaults.rq_points_dataset_path, codec, preset, video_filename, "crfs.json")
		data = IO_functions.read_create_jsonfile(path)
		
		for resolution in defaults.resolutions:
			for rc in defaults.codec_CRF_ranges[codec]:
				info = data["{}x{}".format(resolution[0], resolution[1])][str(rc)]

				Compression_time[resolution].append(info['downscaling_compression_time'])
				VMAF_time[resolution].append(info["quality_estimation_time"])
				Bitrate[resolution].append(info['bitrate'])
				Quality[resolution].append(info["vmaf"])

	return Compression_time, VMAF_time, Bitrate, Quality


# Plot Box-Plots for Time-Complexity
def save_plot_box_plots(
	codec:str,
	preset:str,
	RQ_details:dict,
	save_dir:str
):
	# Get System + User Times
	Compression_Times = RQ_details["Compression_Times"][(codec, preset)]
	VMAF_Times = RQ_details["VMAF_Times"][(codec, preset)]
	Bitrate = RQ_details["Bitrate"][(codec, preset)]
	Quality = RQ_details["Quality"][(codec, preset)]

	# Plot
	plt.figure(figsize=(10,8))
	plt.grid()
	plt.xlabel("Resolutions")
	plt.ylabel("Upsampling + VMAF Time")
	plt.title("{}, {}".format(codec, preset))
	plt.boxplot(VMAF_Times.values(), labels=defaults.resolutions, patch_artist=True, meanline=True, showmeans=True, showfliers=False)

	plt.savefig(os.path.join(save_dir, "{}_{}.png".format(codec, preset)), dpi=400, bbox_inches='tight', pad_inches=0.1)


# Plot Median Execution Time
def save_plot_median_complexity(
	codec_preset_pairs:list,
	RQ_details:dict,
	save_dir:str
):
	# Get System + User Times
	Compression_Times = RQ_details["Compression_Times"]
	Bitrate = RQ_details["Bitrate"]
	Quality = RQ_details["Quality"]

	# Plot
	plt.figure(figsize=(12,6))
	plt.xlabel("Resolutions")
	plt.ylabel("Median Down-Sampling + Compression Time")
	plt.xticks(np.arange(len(defaults.resolutions)), defaults.resolutions)
	plt.grid()

	# Colors
	colors = [
		'#1f77b4',  # Blue
		'#ff7f0e',  # Orange
		'#2ca02c',  # Green
		'#d62728',  # Red
		'#9467bd',  # Purple
		'#8c564b',  # Brown
		'#e377c2',  # Pink
		'#7f7f7f',  # Gray
		'#bcbd22',  # Olive
		'#17becf',  # Cyan
		'#FFD700',  # Gold
	]

	for i,key in enumerate(codec_preset_pairs):
		# Logging
		"""
		print (key)
		print ()

		print ("Bitrate (in kbps):")
		for resolution in defaults.resolutions:
			print ("{}: {}".format(resolution, np.round(np.array(Bitrate[key][resolution])/1e3, decimals=2)))
		print ()

		print ("Quality (VMAF):")
		for resolution in defaults.resolutions:
			print ("{}: {}".format(resolution, np.round(np.array(Quality[key][resolution]), decimals=2)))
		print ()

		print ("System + User Time:")
		Data = []
		for resolution in defaults.resolutions:
			Data.append(
				np.mean(Compression_Times[key][resolution])
			)
			print ("{}: {}".format(resolution, np.round(Compression_Times[key][resolution], decimals=2)))
		print ()

		print ("-"* 100)
		"""

		Data = []
		for resolution in defaults.resolutions:
			Data.append(
				np.median(Compression_Times[key][resolution])
			)
		
		plt.semilogy(np.arange(len(defaults.resolutions)), Data, label=key, color=colors[i])
		plt.scatter(np.arange(len(defaults.resolutions)), Data, color=colors[i])

	plt.legend(markerscale=3, loc='upper center', bbox_to_anchor=(0.5, 1.165), ncol=5, fancybox=True, shadow=True)
	plt.savefig(os.path.join(save_dir, "Median_Time-Complexity.png"), dpi=400, bbox_inches='tight', pad_inches=0.05)


# Codec Preset Pairs
codec_preset_pairs=[
	("libx265", "veryfast"), ("libx265", "fast"), ("libx265", "medium"), ("libx265", "slow"), 
	("libsvtav1", "8"), ("libsvtav1", "6"), ("libsvtav1", "4"),
	("libvpx-vp9", "4"), ("libvpx-vp9", "3"), 
	("libaom-av1", "7"), ("libaom-av1", "5")
]

# Get RQ Details
"""
RQ_details = {}
RQ_details["Compression_Times"] = {}
RQ_details["VMAF_Times"] = {}
RQ_details["Bitrate"] = {}
RQ_details["Quality"] = {}

for key in codec_preset_pairs:
	CT, VT, B, Q = calculate_time_complexity(codec=key[0], preset=key[1], video_files=defaults.Video_Titles)
	RQ_details["Compression_Times"][key] = CT
	RQ_details["VMAF_Times"][key] = VT
	RQ_details["Bitrate"][key] = B
	RQ_details["Quality"][key] = Q

np.save("plots/time_complexity/RQ_details.npy", RQ_details)
"""


# Plot
RQ_details = np.load("plots/time_complexity/RQ_details.npy", allow_pickle=True)[()]
save_plot_median_complexity(
	codec_preset_pairs=codec_preset_pairs,
	RQ_details=RQ_details,
	save_dir="plots/time_complexity"
)

for codec, preset in codec_preset_pairs:
	save_plot_box_plots(
		codec=codec,
		preset=preset,
		RQ_details=RQ_details,
		save_dir="plots/time_complexity"
	)