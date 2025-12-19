"""
Plotting performance of the constructed bitrate ladder across different codecs and presets
"""
# Importing Libraries
import numpy as np
np.set_printoptions(suppress=True)
import matplotlib.ticker as ticker
import matplotlib.pyplot as plt
import seaborn as sns

import os, sys, warnings
warnings.filterwarnings("ignore")
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from collections import ChainMap
import functions.IO_functions as IO_functions
import modules.bitrate_ladder_evaluation_functions as bitrate_ladder_evaluation_functions
import defaults


# Get Max and Min Value of Y
def get_max_min_y(bars, max_y=None, min_y=None):
	if min_y is None:
		min_y = min([bar.get_height() for bar in bars])
	else:
		min_y = min(min_y, min([bar.get_height() for bar in bars]))
	
	if max_y is None:
		max_y = max([bar.get_height() for bar in bars])
	else:
		max_y = max(max_y, max([bar.get_height() for bar in bars]))

	return max_y, min_y


# Function to Add Text Anotations
def add_text_anotations(bars, target):
	if target == "Fixed_BD_Rate" or target == "Convex_Hull_BD_VMAF":
		sign = -3.25
		fontsize = 8
	else:
		sign = 0.75
		fontsize = 8
	
	for bar in bars:
		height = bar.get_height()
		plt.annotate(
			f'{height:.2f}',
			xy=(bar.get_x() + bar.get_width() / 2, height),
			xytext=(0, 5*sign),
			textcoords="offset points",
			ha='center',
			va='bottom',
			fontsize=fontsize
		)


# Get BD-Metric Target
def get_BD_Metric_target(Data, index, target, presets):
	if target == "Fixed_BD_Rate":
		target_index = 0
	if target == "Fixed_BD_VMAF":
		target_index = 1
	if target == "Convex_Hull_BD_Rate":
		target_index = 2
	if target == "Convex_Hull_BD_VMAF":
		target_index = 3
	if target == "Fixed_BD_Rate_IoU":
		target_index = 4
	if target == "Fixed_BD_VMAF_IoU":
		target_index = 5
	if target == "Convex_Hull_BD_Rate_IoU":
		target_index = 6
	if target == "Convex_Hull_BD_VMAF_IoU":
		target_index = 7

	# Target-Results
	target_results = []
	for preset in presets:
		# Processing BD-Metrics
		Metrics = np.array(list(Data[preset][index].values()))

		if target == "Fixed_BD_Rate" or target == "Fixed_BD_VMAF":
			# Ignoring files BD-Rate IoU against Fixed Bitrate Ladder <= 0
			N = len(Metrics)
			mask = np.where(Metrics[:,4] > 0)[0]
			Metrics = Metrics[mask]

			# Assertions
			assert len(Metrics) > 0.975*N, "More than 2.5% of BD-Metrics have BD-Rate aganst the fixed bitrate ladder more than 100%."
			print ("Preset: {}, Index: {}, Diff: {}".format(preset, index, N - len(Metrics)))
			
			# Logging
			"""
			ignored_files = np.delete(list(Data[preset][index].keys()), mask)
			for i,file in enumerate(ignored_files):
				values = Data[preset][index][file]
				print(f'{file:70}{values[0]:8}{values[4]:8}')
			"""

		# Assertions
		assert len(Metrics) > 0.975*len(defaults.Video_Titles), "More than 97.5% of total files should be considered while calculating mean metrics"

		# Appending
		target_results.append(
			np.mean(Metrics[:, target_index])
		)

	return np.asarray(target_results)


# Get Closeness Target
def get_Closeness_target(Data, index, target, presets):
	if target == "f_25":
		target_index = 0
	if target == "f_50":
		target_index = 1
	if target == "f_75":
		target_index = 2
	if target == "f_85":
		target_index = 3

	# Target-Results
	target_results = []
	for preset in presets:
		# Closeness
		Closeness = np.array(list(Data[preset][index].values()))

		# Assertions
		assert len(Closeness) > 0.975*len(defaults.Video_Titles), "More than 97.5% of total files should be considered while calculating mean metrics"

		# Appending
		target_results.append(
			np.mean(Closeness[:, target_index])
		)

	return np.asarray(target_results)


# Plotting Performance across Codecs and Presets
def Plot(
	codec:str,
	target:str,
	ylabel:str,
	results_dir:str,
	save_dir:str,
):
	"""
	Return Target
	Args:
		codec (str): Codec
		target (str): Target
		ylabel (str): Y-Label
		results_dir (str): Results directory
		save_dir (str): Save directory
	"""
	# --------------------------------------------------

	# Presets
	presets = defaults.codec_preset_pairs[codec]

	# Figure Size
	if target.__contains__("IoU"):
		plt.figure(figsize=(2*len(presets), 3))
	elif target.__contains__("BD_Rate") or target.__contains__("BD_VMAF"):
		plt.figure(figsize=(3*len(presets), 4))
	elif target.__contains__("f_"):
		plt.figure(figsize=(2*len(presets), 3.5))
	else:
		assert False, "Unknown Target: {}".format(target)

	# Labels
	plt.ylabel(ylabel, fontsize=9)
	plt.xlabel("Presets", fontsize=9)
	plt.grid(alpha=0.1)

	# --------------------------------------------------

	# Load Data
	Convex_Hull_Data = {}
	Two_Step_Convex_Hull_Data = {}
	CrossOver_Bitrates_Data = {}
	Without_Compression_Data = {}
	With_Compression_Data = {}

	# BD-Metrics
	if target.__contains__("BD"):
		# -------------------------

		# Paths
		Standard_BD_metrics_paths = [
			os.path.join(
				results_dir, "Split-{}".format(i), "Standard", "bd_metrics"
			) for i in range(len(defaults.Datasets))
		]
		CrossOver_Bitrates_metrics_paths = [
			os.path.join(
				results_dir, "Split-{}".format(i), "CrossOver_Bitrates", "bd_metrics"
			) for i in range(len(defaults.Datasets))
		]
		Without_Compression_BD_metrics_paths = [
			os.path.join(
				results_dir, "Split-{}".format(i), "Without_Compression", "bd_metrics"
				) for i in range(len(defaults.Datasets))
		]
		With_Compression_BD_metrics_paths = [
			os.path.join(
				results_dir, "Split-{}".format(i), "With_Compression", "bd_metrics"
				) for i in range(len(defaults.Datasets))
		]

		# -------------------------

		# Load Data for each Preset
		for preset in defaults.codec_preset_pairs[codec]:
			# -------------------------
			
			# Convex-Hull
			Convex_Hull_Data[preset] = [
				dict(ChainMap(
					*[np.load(
							os.path.join(
								Standard_BD_metrics_paths[i], "Convex_Hull", "{}_{}.npy".format(codec, preset)
							), allow_pickle=True)[()] for i in range(len(defaults.Datasets))
					]
				))
			]

			# -------------------------

			# Two-Step
			Two_Step_Convex_Hull_Data[preset] = [
				dict(ChainMap(
					*[np.load(
							os.path.join(
								Standard_BD_metrics_paths[i], "Two_Step_Convex_Hull", "{}_{}.npy".format(codec, preset)
							), allow_pickle=True)[()] for i in range(len(defaults.Datasets))
					]
				))
			]

			# -------------------------

			# Cross-Over Bitrates
			CrossOver_Bitrates_Data[preset] = [
				dict(ChainMap(
					*[np.load(
							os.path.join(
								CrossOver_Bitrates_metrics_paths[i], "{}_{}_low_level_features.npy".format(codec, preset)
							), allow_pickle=True)[()] for i in range(len(defaults.Datasets))
					]
				))
			]

			# -------------------------

			# Without-Compression
			Without_Compression_Data[preset] = [
				dict(ChainMap(
					*[np.load(
							os.path.join(
								Without_Compression_BD_metrics_paths[i], "{}_{}_metadata.npy".format(codec, preset)
							), allow_pickle=True)[()] for i in range(len(defaults.Datasets))
					]
				)),
				dict(ChainMap(
					*[np.load(
							os.path.join(
								Without_Compression_BD_metrics_paths[i], "{}_{}_low_level_features.npy".format(codec, preset)
							), allow_pickle=True)[()] for i in range(len(defaults.Datasets))
					]
				)),
				dict(ChainMap(
					*[np.load(
							os.path.join(
								Without_Compression_BD_metrics_paths[i], "{}_{}_vif_features.npy".format(codec, preset)
							), allow_pickle=True)[()] for i in range(len(defaults.Datasets))
					]
				)),
				dict(ChainMap(
					*[np.load(
							os.path.join(
								Without_Compression_BD_metrics_paths[i], "{}_{}_low_level_features_vif_features.npy".format(codec, preset)
							), allow_pickle=True)[()] for i in range(len(defaults.Datasets))
					]
				))
			]

			# -------------------------

			# With-Compression
			With_Compression_Data[preset] = [
				dict(ChainMap(
					*[np.load(
							os.path.join(
								With_Compression_BD_metrics_paths[i], "{}_{}_metadata.npy".format(codec, preset)
							), allow_pickle=True)[()] for i in range(len(defaults.Datasets))
					]
				)),
				dict(ChainMap(
					*[np.load(
							os.path.join(
								With_Compression_BD_metrics_paths[i], "{}_{}_low_level_features.npy".format(codec, preset)
							), allow_pickle=True)[()] for i in range(len(defaults.Datasets))
					]
				)),
				dict(ChainMap(
					*[np.load(
							os.path.join(
								With_Compression_BD_metrics_paths[i], "{}_{}_vif_features.npy".format(codec, preset)
							), allow_pickle=True)[()] for i in range(len(defaults.Datasets))
					]
				)),
				dict(ChainMap(
					*[np.load(
							os.path.join(
								With_Compression_BD_metrics_paths[i], "{}_{}_low_level_features_vif_features.npy".format(codec, preset)
							), allow_pickle=True)[()] for i in range(len(defaults.Datasets))
					]
				))
			]

			# -------------------------

	else:
		# -------------------------

		# Paths
		CrossOver_Bitrates_Closeness_paths = [
			os.path.join(
				results_dir, "Split-{}".format(i), "CrossOver_Bitrates", "closeness"
				) for i in range(len(defaults.Datasets))
		]
		Without_Compression_Closeness_paths = [
			os.path.join(
				results_dir, "Split-{}".format(i), "Without_Compression", "closeness"
				) for i in range(len(defaults.Datasets))
		]
		With_Compression_Closeness_paths = [
			os.path.join(
				results_dir, "Split-{}".format(i), "With_Compression", "closeness"
				) for i in range(len(defaults.Datasets))
		]

		# -------------------------

		# Load Data for each Preset
		for preset in defaults.codec_preset_pairs[codec]:
			# -------------------------

			# Cross-Over Bitrates
			CrossOver_Bitrates_Data[preset] = [
				dict(ChainMap(
					*[np.load(
							os.path.join(
								CrossOver_Bitrates_Closeness_paths[i], "{}_{}_low_level_features.npy".format(codec, preset)
							), allow_pickle=True)[()] for i in range(len(defaults.Datasets))
					]
				))
			]

			# -------------------------
			
			# Without-Compression
			Without_Compression_Data[preset] = [
				dict(ChainMap(
					*[np.load(
							os.path.join(
								Without_Compression_Closeness_paths[i], "{}_{}_metadata.npy".format(codec, preset)
							), allow_pickle=True)[()] for i in range(len(defaults.Datasets))
					]
				)),
				dict(ChainMap(
					*[np.load(
							os.path.join(
								Without_Compression_Closeness_paths[i], "{}_{}_low_level_features.npy".format(codec, preset)
							), allow_pickle=True)[()] for i in range(len(defaults.Datasets))
					]
				)),
				dict(ChainMap(
					*[np.load(
							os.path.join(
								Without_Compression_Closeness_paths[i], "{}_{}_vif_features.npy".format(codec, preset)
							), allow_pickle=True)[()] for i in range(len(defaults.Datasets))
					]
				)),
				dict(ChainMap(
					*[np.load(
							os.path.join(
								Without_Compression_Closeness_paths[i], "{}_{}_low_level_features_vif_features.npy".format(codec, preset)
							), allow_pickle=True)[()] for i in range(len(defaults.Datasets))
					]
				))
			]

			# -------------------------

			# With-Compression
			With_Compression_Data[preset] = [
				dict(ChainMap(
					*[np.load(
							os.path.join(
								With_Compression_Closeness_paths[i], "{}_{}_metadata.npy".format(codec, preset)
							), allow_pickle=True)[()] for i in range(len(defaults.Datasets))
					]
				)),
				dict(ChainMap(
					*[np.load(
							os.path.join(
								With_Compression_Closeness_paths[i], "{}_{}_low_level_features.npy".format(codec, preset)
							), allow_pickle=True)[()] for i in range(len(defaults.Datasets))
					]
				)),
				dict(ChainMap(
					*[np.load(
							os.path.join(
								With_Compression_Closeness_paths[i], "{}_{}_vif_features.npy".format(codec, preset)
							), allow_pickle=True)[()] for i in range(len(defaults.Datasets))
					]
				)),
				dict(ChainMap(
					*[np.load(
							os.path.join(
								With_Compression_Closeness_paths[i], "{}_{}_low_level_features_vif_features.npy".format(codec, preset)
							), allow_pickle=True)[()] for i in range(len(defaults.Datasets))
					]
				))
			]

			# -------------------------

	# --------------------------------------------------
	
	# X and Y
	X = np.arange(len(presets))
	Y = None

	# Width and Factor for Bars
	if target.__contains__("IoU"):
		width = 1/5
		f = 0.9
	elif target.__contains__("BD_Rate") or target.__contains__("BD_VMAF"):
		width = 1/7
		f = 0.9
	else:
		width = 1/5
		f = 0.9

	start_point = 4

	# --------------------------------------------------

	# Starting Max and Min Y
	max_y, min_y = None, None

	# -------------------------

	# Non-ML: Convex-Hull and Two-Step Convex-Hull
	if target.__contains__("BD") and target.__contains__("IoU") == False:
		# Non-ML: Convex-Hull
		if target.__contains__("Fixed"):
			Y = get_BD_Metric_target(
				Data=Convex_Hull_Data, 
				index=0, target=target, presets=presets
			)
			bars = plt.bar(
				x=X-(start_point + 2)*width, 
				height=Y, width=f*width, 
				color=["#FF3D00"], label="Convex_Hull"
			)
			add_text_anotations(bars, target)
			max_y, min_y = get_max_min_y(bars, max_y=max_y, min_y=min_y)

		# Non-ML: Two-Step Convex-Hull
		Y = get_BD_Metric_target(
			Data=Two_Step_Convex_Hull_Data, 
			index=0, target=target, presets=presets
		)
		bars = plt.bar(
			x=X-(start_point + 1)*width, 
			height=Y, width=f*width, 
			color=["#FFB700"], label="Two-Step"
		)
		add_text_anotations(bars, target)
		max_y, min_y = get_max_min_y(bars, max_y=max_y, min_y=min_y)

	# -------------------------

	# ML: Cross-Over Bitrates
	if target.__contains__("BD"):
		Y = get_BD_Metric_target(
			Data=CrossOver_Bitrates_Data, 
			index=0, target=target, presets=presets
		)
	else:
		Y = get_Closeness_target(
			Data=CrossOver_Bitrates_Data, 
			index=0, target=target, presets=presets
		)
	
	bars = plt.bar(
		x=X-(start_point)*width, 
		height=Y, width=f*width, 
		color=["#1E90FF"], label="Cross-Over Bitrates"
	)
	add_text_anotations(bars, target)
	max_y, min_y = get_max_min_y(bars, max_y=max_y, min_y=min_y)

	# ML: Without Compression
	"""
	colors = ["#87CEEB", "#4682B4", "#4169E1", "#1E90FF"]
	labels = ["WO_CS_Metadata", "WO_CS_LLF", "WO_CS_VIFF", "WO_CS_LLF_VIFF"]
	for i in range(4):
		if i==0:
			continue
			
		if target.__contains__("BD"):
			Y = get_BD_Metric_target(
				Data=Without_Compression_Data, 
				index=i, target=target, presets=presets
			)
		else:
			Y = get_Closeness_target(
				Data=Without_Compression_Data, 
				index=i, target=target, presets=presets
			)
		
		bars = plt.bar(
			x=X-(start_point)*width+(width*(i-1)), 
			height=Y, width=f*width, 
			color=[colors[i]], label=labels[i]
		)
		add_text_anotations(bars, target)
		max_y, min_y = get_max_min_y(bars, max_y=None, min_y=None)
	"""

	# ML: With Compression
	colors = ["#98FB98", "#3CB371", "#32CD32", "#228B22"]
	labels = ["Metadata", "LLF", "VIFF", "LLF_VIFF"]
	for i in range(4):
		if i==0:
			continue

		if target.__contains__("BD"):
			Y = get_BD_Metric_target(
				Data=With_Compression_Data, 
				index=i, target=target, presets=presets
			)
		else:
			Y = get_Closeness_target(
				Data=With_Compression_Data, 
				index=i, target=target, presets=presets
			)
		
		bars = plt.bar(
			x=X-(start_point-1)*width+(width*(i-1)), 
			height=Y, width=f*width, 
			color=[colors[i]], label=labels[i]
		)
		add_text_anotations(bars, target)
		max_y, min_y = get_max_min_y(bars, max_y=max_y, min_y=min_y)

	# --------------------------------------------------

	# Set Background Color
	if "IoU" in target:
		plt.gca().set_facecolor("#F5F9FF")
	elif "BD_Rate" in target:
		plt.gca().set_facecolor("#FFFFFF")
	elif "BD_VMAF" in target:
		plt.gca().set_facecolor("#F0F0F0")
	else:
		plt.gca().set_facecolor("#FFF8E7")

	# Legend
	if target.__contains__("IoU") or target.__contains__("f_"):
		if codec == "libx265" or codec == "libsvtav1":
			plt.legend(markerscale=1.1, loc='upper center', bbox_to_anchor=(0.5, 1.125), ncol=4, fancybox=True, shadow=True, fontsize=9)
		else:
			plt.legend(markerscale=1.1, loc='upper center', bbox_to_anchor=(0.5, 1.25), ncol=2, fancybox=True, shadow=True, fontsize=9)
	else:
		if codec == "libx265" or codec == "libsvtav1":
			plt.legend(markerscale=1.1, loc='upper center', bbox_to_anchor=(0.5, 1.125), ncol=9, fancybox=True, shadow=True, fontsize=9)
		else:
			plt.legend(markerscale=1.1, loc='upper center', bbox_to_anchor=(0.5, 1.2), ncol=3, fancybox=True, shadow=True, fontsize=9)
	
	# Set X and Y Ticks
	if target.__contains__("IoU") or target.__contains__("f_"):
		plt.xticks(ticks=X-2.5*width, labels=presets, fontsize=9)
	elif target.__contains__("Convex_Hull"):
		plt.xticks(ticks=X-3*width, labels=presets, fontsize=9)
	else:
		plt.xticks(ticks=X-3.5*width, labels=presets, fontsize=9)
	plt.yticks(fontsize=9)

	# Set Y-Axis limits
	if target == "Fixed_BD_Rate":
		plt.ylim(min_y - 0.05*abs(min_y), max_y + 0.05*abs(max_y))
	elif target == "Convex_Hull_BD_VMAF":
		plt.ylim(min_y - 0.1*abs(min_y), max_y + 0.05*abs(max_y))
	elif target == "Convex_Hull_BD_Rate" or target == "Fixed_BD_VMAF":
		plt.ylim(min_y - 0.05*abs(min_y), max_y + 0.075*abs(max_y))
	elif target.__contains__("f_") or target.__contains__("_IoU"):
		plt.ylim(min_y - 0.01*abs(min_y), min(1.01, max_y + 0.03*abs(max_y)))
	else:
		assert False, "Unknown Target: {}".format(target)
	
	plt.tight_layout()
	plt.savefig(os.path.join(save_dir, "{}_{}.png".format(codec, target)), dpi=250, bbox_inches='tight')

	# --------------------------------------------------


# Main Execution
if __name__ == "__main__":
	# Plot for libx265, veryfast as fast encoder
	for codec in ["libx265", "libsvtav1", "libaom-av1", "libvpx-vp9"]:
		Plot(
			codec=codec,
			target="Fixed_BD_Rate",
			ylabel="Mean BD-Rate against Fixed Bitrate Ladder",
			results_dir="../results/main/libx265",
			save_dir="plots/performance/bar_graphs/libx265"
		)
		Plot(
			codec=codec,
			target="Convex_Hull_BD_Rate",
			ylabel="Mean BD-Rate against Convex Hull",
			results_dir="../results/main/libx265",
			save_dir="plots/performance/bar_graphs/libx265"
		)
		Plot(
			codec=codec,
			target="Convex_Hull_BD_VMAF",
			ylabel="Mean BD-VMAF against Convex Hull",
			results_dir="../results/main/libx265",
			save_dir="plots/performance/bar_graphs/libx265"
		)
		Plot(
			codec=codec,
			target="f_75",
			ylabel=r"$f_{75}$ against Convex Hull",
			results_dir="../results/main/libx265",
			save_dir="plots/performance/bar_graphs/libx265"
		)
		Plot(
			codec=codec,
			target="Fixed_BD_Rate_IoU",
			ylabel="Mean IoU against Fixed Bitrate Ladder",
			results_dir="../results/main/libx265",
			save_dir="plots/performance/bar_graphs/libx265"
		)
		Plot(
			codec=codec,
			target="Convex_Hull_BD_Rate_IoU",
			ylabel="Mean IoU against Convex Hull",
			results_dir="../results/main/libx265",
			save_dir="plots/performance/bar_graphs/libx265"
		)


	# Plot for libsvtav1, 8 as fast encoder
	for codec in ["libx265", "libsvtav1", "libaom-av1", "libvpx-vp9"]:
		Plot(
			codec=codec,
			target="Fixed_BD_Rate",
			ylabel="Mean BD-Rate against Fixed Bitrate Ladder",
			results_dir="../results/main/libsvtav1",
			save_dir="plots/performance/bar_graphs/libsvtav1"
		)
		Plot(
			codec=codec,
			target="Convex_Hull_BD_Rate",
			ylabel="Mean BD-Rate against Convex Hull",
			results_dir="../results/main/libsvtav1",
			save_dir="plots/performance/bar_graphs/libsvtav1"
		)
		Plot(
			codec=codec,
			target="Convex_Hull_BD_VMAF",
			ylabel="Mean BD-VMAF against Convex Hull",
			results_dir="../results/main/libsvtav1",
			save_dir="plots/performance/bar_graphs/libsvtav1"
		)
		Plot(
			codec=codec,
			target="f_75",
			ylabel=r"$f_{75}$ against Convex Hull",
			results_dir="../results/main/libsvtav1",
			save_dir="plots/performance/bar_graphs/libsvtav1"
		)
		Plot(
			codec=codec,
			target="Fixed_BD_Rate_IoU",
			ylabel="Mean IoU against Fixed Bitrate Ladder",
			results_dir="../results/main/libsvtav1",
			save_dir="plots/performance/bar_graphs/libsvtav1"
		)
		Plot(
			codec=codec,
			target="Convex_Hull_BD_Rate_IoU",
			ylabel="Mean IoU against Convex Hull",
			results_dir="../results/main/libsvtav1",
			save_dir="plots/performance/bar_graphs/libsvtav1"
		)