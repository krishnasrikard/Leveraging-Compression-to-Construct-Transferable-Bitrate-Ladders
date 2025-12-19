"""
Plotting performance of the constructed bitrate ladder across different codecs and presets
"""
# Importing Libraries
import numpy as np
np.set_printoptions(suppress=True)
import matplotlib
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
	results_dir:str,
	save_dir:str,
):
	"""
	Return Target
	Args:
		codec (str): Codec
		target (str): Target
		results_dir (str): Results directory
		save_dir (str): Save directory
	"""
	# Logging
	print ()
	print ("Codec: {}, Target: {}".format(codec, target))
	print ()

	# Figure
	presets = defaults.codec_preset_pairs[codec]
	plt.figure(figsize=(12,8))
	plt.ylabel(target)
	plt.xlabel("Presets")


	# BD-Metrics
	if target.__contains__("BD"):
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

		# Store
		Convex_Hull_BD_Metrics = {}
		Two_Step_Convex_Hull_BD_Metrics = {}
		CrossOver_Bitrates_BD_Metrics = {}
		Without_Compression_BD_Metrics = {}
		With_Compression_BD_Metrics = {}

		for preset in defaults.codec_preset_pairs[codec]:
			# Convex-Hull
			Convex_Hull_BD_Metrics[preset] = [
				dict(ChainMap(
					*[np.load(
							os.path.join(
								Standard_BD_metrics_paths[i], "Convex_Hull", "{}_{}.npy".format(codec, preset)
							), allow_pickle=True)[()] for i in range(len(defaults.Datasets))
					]
				))
			]

			# Two-Step
			Two_Step_Convex_Hull_BD_Metrics[preset] = [
				dict(ChainMap(
					*[np.load(
							os.path.join(
								Standard_BD_metrics_paths[i], "Two_Step_Convex_Hull", "{}_{}.npy".format(codec, preset)
							), allow_pickle=True)[()] for i in range(len(defaults.Datasets))
					]
				))
			]

			# Cross-Over Bitrates
			CrossOver_Bitrates_BD_Metrics[preset] = [
				dict(ChainMap(
					*[np.load(
							os.path.join(
								CrossOver_Bitrates_metrics_paths[i], "{}_{}_low_level_features.npy".format(codec, preset)
							), allow_pickle=True)[()] for i in range(len(defaults.Datasets))
					]
				))
			]

			# Without-Compression
			Without_Compression_BD_Metrics[preset] = [
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

			# With-Compression
			With_Compression_BD_Metrics[preset] = [
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
		
		# Points
		X = np.arange(len(presets))
		Y = None

		# -----------------------------------------------------------------------------------

		# Non-ML: Convex-Hull
		print ()
		print ("Non-ML: Convex-Hull")
		print ()

		Y = get_BD_Metric_target(
			Data=Convex_Hull_BD_Metrics, 
			index=0, target=target, presets=presets
		)
		plt.plot(X, Y, c="black", linewidth=2.5, marker="o", markersize=8)
		plt.scatter(X, Y, c="black", marker="o", s=8, label="Convex_Hull")

		# -----------------------------------------------------------------------------------

		# Non-ML: Two-Step Convex-Hull
		print ()
		print ("Non-ML: Two-Step Convex-Hull")
		print ()

		Y = get_BD_Metric_target(
			Data=Two_Step_Convex_Hull_BD_Metrics, 
			index=0, target=target, presets=presets
		)
		plt.plot(X, Y, c="red", linewidth=2.5, marker="o", markersize=8)
		plt.scatter(X, Y, c="red", marker="o", s=8, label="Two-Step")

		# -----------------------------------------------------------------------------------

		# ML: Cross-Over Bitrates
		print ()
		print ("ML: Cross-Over Bitrates")
		print ()

		Y = get_BD_Metric_target(
			Data=CrossOver_Bitrates_BD_Metrics, 
			index=0, target=target, presets=presets
		)
		plt.plot(X, Y, c="purple", linewidth=2.5, marker="o", markersize=8)
		plt.scatter(X, Y, c="purple", marker="o", s=8, label="Cross-Over Bitrates")

		# -----------------------------------------------------------------------------------

		# ML: Without Compression
		print ()
		print ("ML: Without Compression")
		print ()

		colors = ["#add8e6", "#4682b4", "#0000ff", "#00008b"]
		labels = ["Metadata", "LLF", "VIFF", "LLF_VIFF"]
		for i in range(len(labels)):
			if i==0:
				continue
			Y = get_BD_Metric_target(
				Data=Without_Compression_BD_Metrics, 
				index=i, target=target, presets=presets
			)
			plt.plot(X, Y, c=colors[i], linewidth=2.5, marker="v", markersize=8)
			plt.scatter(X, Y, c=colors[i], marker="v", s=8, label=labels[i])

		# -----------------------------------------------------------------------------------

		# ML: With Compression
		print ()
		print ("ML: With Compression")
		print ()

		colors = ["#98fb98", "#3cb371", "#228b22", "#006400"]
		labels = ["C_Metadata", "C_LLF", "C_VIFF", "C_LLF_VIFF"]
		for i in range(len(labels)):
			if i==0:
				continue
			Y = get_BD_Metric_target(
				Data=With_Compression_BD_Metrics, 
				index=i, target=target, presets=presets
			)
			plt.plot(X, Y, c=colors[i], linewidth=2.5, marker="*", markersize=8)
			plt.scatter(X, Y, c=colors[i], label=labels[i], marker="*", s=8)

		# -----------------------------------------------------------------------------------


	else:
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

		# Store
		CrossOver_Bitrates_Closeness = {}
		Without_Compression_Closeness = {}
		With_Compression_Closeness = {}

		for preset in defaults.codec_preset_pairs[codec]:
			# Cross-Over Bitrates
			CrossOver_Bitrates_Closeness[preset] = [
				dict(ChainMap(
					*[np.load(
							os.path.join(
								CrossOver_Bitrates_Closeness_paths[i], "{}_{}_low_level_features.npy".format(codec, preset)
							), allow_pickle=True)[()] for i in range(len(defaults.Datasets))
					]
				))
			]
			
			# Without-Compression
			Without_Compression_Closeness[preset] = [
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

			# With-Compression
			With_Compression_Closeness[preset] = [
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

		
		# Points
		X = np.arange(len(presets))
		Y = None

		# -----------------------------------------------------------------------------------

		# ML: CrossOver Bitrates
		print ()
		print ("ML: CrossOver Bitrates")
		print ()

		Y = get_Closeness_target(
			Data=CrossOver_Bitrates_Closeness, 
			index=0, target=target, presets=presets
		)
		plt.plot(X, Y, c="purple", linewidth=2.5, marker="v", markersize=8)
		plt.scatter(X, Y, c="purple", label="Cross-Over Bitrates", marker="v", s=8)

		# -----------------------------------------------------------------------------------

		# ML: Without Compression
		print ()
		print ("ML: Without Compression")
		print ()

		colors = ["#add8e6", "#4682b4", "#0000ff", "#00008b"]
		labels = ["Metadata", "LLF", "VIFF", "LLF_VIFF"]
		for i in range(len(labels)):
			if i==0:
				continue
			Y = get_Closeness_target(
				Data=Without_Compression_Closeness, 
				index=i, target=target, presets=presets
			)
			plt.plot(X, Y, c=colors[i], linewidth=2.5, marker="v", markersize=8)
			plt.scatter(X, Y, c=colors[i], label=labels[i], marker="v", s=8)

		# -----------------------------------------------------------------------------------

		# ML: With Compression
		print ()
		print ("ML: With Compression")
		print ()

		colors = ["#98fb98", "#3cb371", "#228b22", "#006400"]
		labels = ["C_Metadata", "C_LLF", "C_VIFF", "C_LLF_VIFF"]
		for i in range(len(labels)):
			if i==0:
				continue
			Y = get_Closeness_target(
				Data=With_Compression_Closeness, 
				index=i, target=target, presets=presets
			)
			plt.plot(X, Y, c=colors[i], linewidth=2.5, marker="*", markersize=8)
			plt.scatter(X, Y, c=colors[i], label=labels[i], marker="*", s=8)

		# -----------------------------------------------------------------------------------


	# Plot and Save
	plt.grid()
	plt.xticks(ticks=np.arange(len(presets)), labels=presets)
	plt.legend(markerscale=3, loc='upper center', bbox_to_anchor=(0.5, 1.1), ncol=4, fancybox=True, shadow=True)
	plt.savefig(os.path.join(save_dir, "{}_{}.png".format(codec, target)), dpi=500, bbox_inches='tight')


	# Logging
	print ("-"*75)


# Plot for libx265, veryfast as fast encoder
# """
for codec in ["libx265", "libsvtav1", "libaom-av1", "libvpx-vp9"]:
	Plot(
		codec=codec,
		target="Fixed_BD_Rate",
		results_dir="../results/main/libx265",
		save_dir="plots/performance/scatter_plots/libx265"
	)
	Plot(
		codec=codec,
		target="Convex_Hull_BD_Rate",
		results_dir="../results/main/libx265",
		save_dir="plots/performance/scatter_plots/libx265"
	)
	Plot(
		codec=codec,
		target="f_85",
		results_dir="../results/main/libx265",
		save_dir="plots/performance/scatter_plots/libx265"
	)
	Plot(
		codec=codec,
		target="Convex_Hull_BD_Rate_IoU",
		results_dir="../results/main/libx265",
		save_dir="plots/performance/scatter_plots/libx265"
	)
# """


# Plot for libsvtav1, 8 as fast encoder
# """
for codec in ["libx265", "libsvtav1", "libaom-av1", "libvpx-vp9"]:
	Plot(
		codec=codec,
		target="Fixed_BD_Rate",
		results_dir="../results/main/libsvtav1",
		save_dir="plots/performance/scatter_plots/libsvtav1"
	)
	Plot(
		codec=codec,
		target="Convex_Hull_BD_Rate",
		results_dir="../results/main/libsvtav1",
		save_dir="plots/performance/scatter_plots/libsvtav1"
	)
	Plot(
		codec=codec,
		target="f_75",
		results_dir="../results/main/libsvtav1",
		save_dir="plots/performance/scatter_plots/libsvtav1"
	)
	Plot(
		codec=codec,
		target="Convex_Hull_BD_Rate_IoU",
		results_dir="../results/main/libsvtav1",
		save_dir="plots/performance/scatter_plots/libsvtav1"
	)
# """