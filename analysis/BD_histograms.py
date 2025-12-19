"""
Plotting BD-Metrics
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
import functions.plot_functions as plot_functions
import functions.IO_functions as IO_functions
import defaults

# Path
plots_dir = "plots/bd_histograms"
results_dir = "../results"


# Constructing BD-Histograms
for compression_statistics in [False, True]:
	# Sub-Folder
	if compression_statistics:
		subfolder = "With_Compression"
	else:
		subfolder = "Without_Compression"

	# Creating Directories
	os.makedirs(os.path.join(plots_dir, subfolder), exist_ok=True)

	# For each codec and preset
	for codec in defaults.codec_preset_pairs.keys():
		for preset in defaults.codec_preset_pairs[codec]:
			print ("Codec: {}, Preset: {}".format(codec, preset))
			print ()
			

			# Metadata
			"""
			bd_metrics_paths = []
			for i in range(len(defaults.Datasets)):
				bd_metrics_paths.append(
					os.path.join(
						results_dir, "Split-{}".format(i), subfolder, "bd_metrics", "{}_{}_metadata.npy".format(codec, preset)
					)
				)

			plot_functions.Plot_BD_Metrics(
				bd_metrics_paths=bd_metrics_paths,
				save_path=os.path.join(plots_dir, subfolder, "{}_{}_metadata.png".format(codec, preset))
			)
			"""


			# Low-Level Features
			bd_metrics_paths = []
			for i in range(len(defaults.Datasets)):
				bd_metrics_paths.append(
					os.path.join(
						results_dir, "Split-{}".format(i), subfolder, "bd_metrics", "{}_{}_low_level_features.npy".format(codec, preset)
					)
				)

			plot_functions.Plot_BD_Metrics(
				bd_metrics_paths=bd_metrics_paths,
				save_path=os.path.join(plots_dir, subfolder, "{}_{}_low_level_features.png".format(codec, preset))
			)


			# VIF Features
			bd_metrics_paths = []
			for i in range(len(defaults.Datasets)):
				bd_metrics_paths.append(
					os.path.join(
						results_dir, "Split-{}".format(i), subfolder, "bd_metrics", "{}_{}_vif_features.npy".format(codec, preset)
					)
				)

			plot_functions.Plot_BD_Metrics(
				bd_metrics_paths=bd_metrics_paths,
				save_path=os.path.join(plots_dir, subfolder, "{}_{}_vif_features.png".format(codec, preset))
			)


			# Low-Level Features
			bd_metrics_paths = []
			for i in range(len(defaults.Datasets)):
				bd_metrics_paths.append(
					os.path.join(
						results_dir, "Split-{}".format(i), subfolder, "bd_metrics", "{}_{}_low_level_features_vif_features.npy".format(codec, preset)
					)
				)

			plot_functions.Plot_BD_Metrics(
				bd_metrics_paths=bd_metrics_paths,
				save_path=os.path.join(plots_dir, subfolder, "{}_{}_low_level_features_vif_features.png".format(codec, preset))
			)