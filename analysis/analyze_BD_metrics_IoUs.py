"""
Analyze BD-Metrics and IoUs
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


# Get Description Statistics
def get_description_statistics(Data):
	return {
		"Min": np.round(np.min(Data), decimals=3),
		"Max": np.round(np.max(Data), decimals=3),
		"Mean": np.round(np.mean(Data), decimals=3),
		"Std": np.round(np.std(Data), decimals=3),
		"5-Q": np.round(np.percentile(Data, 5), decimals=3),
		"10-Q": np.round(np.percentile(Data, 5), decimals=3),
		"25-Q": np.round(np.percentile(Data, 25), decimals=3),
		"Median": np.round(np.median(Data), decimals=3),
		"75-Q": np.round(np.percentile(Data, 75), decimals=3),
	}


# Get BD-Metric Target
def get_BD_Metric_target(Data, index, target, preset):
	if target == "Fixed_BD_Rate":
		target_index = 0
	if target == "Fixed_BD_VMAF":
		target_index = 1
	if target == "Reference_BD_Rate":
		target_index = 2
	if target == "Reference_BD_VMAF":
		target_index = 3
	if target == "Fixed_BD_Rate_IoU":
		target_index = 4
	if target == "Fixed_BD_VMAF_IoU":
		target_index = 5
	if target == "Reference_BD_Rate_IoU":
		target_index = 6
	if target == "Reference_BD_VMAF_IoU":
		target_index = 7
		

	# Processing BD-Metrics
	Metrics = np.array(list(Data[preset][index].values()))

	# Appending
	target_results = Metrics[:, target_index]

	return np.asarray(target_results)


# Get Closeness Target
def get_Closeness_target(Data, index, target, preset):
	if target == "f_25":
		target_index = 0
	if target == "f_50":
		target_index = 1
	if target == "f_75":
		target_index = 2
	if target == "f_85":
		target_index = 3

	# Processing BD-Metrics
	Metrics = np.array(list(Data[preset][index].values()))

	# Appending
	target_results = Metrics[:, target_index]

	return np.asarray(target_results)


# Get Statistics on Performance across Codecs and Presets
def Get_Statistics(
	codec:str,
	target:str,
	results_dir:str,
):
	"""
	Return Target
	Args:
		codec (str): Codec
		target (str): Target
		results_dir (str): Results directory
	"""
	# Logging
	print ()
	print ("Codec: {}, Target: {}".format(codec, target))
	print ()

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
		presets = defaults.codec_preset_pairs[codec]
		for preset in presets:
			print ("Preset: {}".format(preset))

			# -----------------------------------------------------------------------------------

			# Non-ML: Convex-Hull
			print ()
			print ("Non-ML: Convex-Hull")
			print ()

			Y = get_BD_Metric_target(
				Data=Convex_Hull_BD_Metrics, 
				index=0, target=target, preset=preset
			)
			print (get_description_statistics(Y))

			# -----------------------------------------------------------------------------------

			# Non-ML: Two-Step Convex-Hull
			print ()
			print ("Non-ML: Two-Step Convex-Hull")
			print ()

			Y = get_BD_Metric_target(
				Data=Two_Step_Convex_Hull_BD_Metrics, 
				index=0, target=target, preset=preset
			)
			print (get_description_statistics(Y))

			# -----------------------------------------------------------------------------------

			# ML: Cross-Over Bitrates
			print ()
			print ("ML: Cross-Over Bitrates")
			print ()

			Y = get_BD_Metric_target(
				Data=CrossOver_Bitrates_BD_Metrics, 
				index=0, target=target, preset=preset
			)
			print (get_description_statistics(Y))

			# -----------------------------------------------------------------------------------

			# ML: Without Compression
			print ()
			print ("ML: Without Compression")
			print ()

			labels = ["Metadata", "LLF", "VIFF", "LLF_VIFF"]
			for i in range(4):
				if i==0:
					continue
				Y = get_BD_Metric_target(
					Data=Without_Compression_BD_Metrics, 
					index=i, target=target, preset=preset
				)
				print ("{:<10}{}".format(labels[i], get_description_statistics(Y)))

			# -----------------------------------------------------------------------------------

			# ML: With Compression
			print ()
			print ("ML: With Compression")
			print ()

			labels = ["C_Metadata", "C_LLF", "C_VIFF", "C_LLF_VIFF"]
			for i in range(4):
				if i==0:
					continue
				Y = get_BD_Metric_target(
					Data=With_Compression_BD_Metrics, 
					index=i, target=target, preset=preset
				)
				print ("{:<10}{}".format(labels[i], get_description_statistics(Y)))

			# -----------------------------------------------------------------------------------

			print ()
			print ("-"*100)
			print ()


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
		presets = defaults.codec_preset_pairs[codec]
		for preset in presets:
			print ("Preset: {}".format(preset))

			# -----------------------------------------------------------------------------------

			# ML: CrossOver Bitrates
			print ()
			print ("ML: CrossOver Bitrates")
			print ()

			Y = get_Closeness_target(
				Data=CrossOver_Bitrates_Closeness, 
				index=0, target=target, preset=preset
			)
			print (get_description_statistics(Y))

			# -----------------------------------------------------------------------------------

			# ML: Without Compression
			print ()
			print ("ML: Without Compression")
			print ()

			labels = ["Metadata", "LLF", "VIFF", "LLF_VIFF"]
			for i in range(4):
				if i==0:
					continue
				Y = get_Closeness_target(
					Data=Without_Compression_Closeness, 
					index=i, target=target, preset=preset
				)
				print ("{:<10}{}".format(labels[i], get_description_statistics(Y)))

			# -----------------------------------------------------------------------------------

			# ML: With Compression
			print ()
			print ("ML: With Compression")
			print ()

			labels = ["C_Metadata", "C_LLF", "C_VIFF", "C_LLF_VIFF"]
			for i in range(4):
				if i==0:
					continue
				Y = get_Closeness_target(
					Data=With_Compression_Closeness, 
					index=i, target=target, preset=preset
				)
				print ("{:<10}{}".format(labels[i], get_description_statistics(Y)))

			# -----------------------------------------------------------------------------------

			print ()
			print ("-"*100)
			print ()


	# Logging
	print ("="*100)


# Plot for libx265, veryfast as fast encoder
for codec in ["libx265", "libsvtav1", "libaom-av1", "libvpx-vp9"]:
	Get_Statistics(
		codec=codec,
		target="Reference_BD_Rate",
		results_dir="../results/main/libx265"
	)

	Get_Statistics(
		codec=codec,
		target="Reference_BD_Rate_IoU",
		results_dir="../results/main/libx265"
	)