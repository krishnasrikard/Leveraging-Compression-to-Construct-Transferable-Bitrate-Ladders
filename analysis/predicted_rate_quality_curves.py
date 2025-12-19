"""
Plotting Predicted Rate-Quality Curves
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

# Inputs Settings
results_dir = "../results/main/libx265"
Split_Num = 0
codec = "libsvtav1"
preset = "6"

# Features
# features = "low_level_features"
# features = "vif_features"
features = "low_level_features_vif_features"

# Method
# subfolder = "CrossOver_Bitrates"
# subfolder = "Without_Compression"
# subfolder = "With_Compression"

# Video File
video_file = defaults.Datasets["Split-0"]["Test_Video_Files"][0]


# Directory
# os.makedirs(os.path.join("plots/predicted_rate_quality_curves/{}/{}".format(subfolder, features)), exist_ok=True)
os.makedirs(os.path.join("plots/predicted_rate_quality_curves"), exist_ok=True)


# Plot
plot_functions.Plot_Predicted_RQ_Curve(
	video_file=video_file,
	codec=codec,
	preset=preset,
	ladder_paths=[
		# os.path.join(results_dir, "Split-{}/Standard/bitrate_ladders/fixed_bitrate_ladder.npy".format(Split_Num)),
		os.path.join(results_dir, "Split-{}/Standard/bitrate_ladders/Convex_Hull/{}_{}.npy".format(Split_Num, codec, preset)),
		# os.path.join(results_dir, "Split-{}/{}/bitrate_ladders/{}.npy".format(Split_Num, "CrossOver_Bitrates", "low_level_features")),
		os.path.join(results_dir, "Split-{}/{}/bitrate_ladders/{}.npy".format(Split_Num, "Without_Compression", features)),
		os.path.join(results_dir, "Split-{}/{}/bitrate_ladders/{}.npy".format(Split_Num, "With_Compression", features)),
	],
	ladder_labels=[
		# "Fixed-Ladder",
		"Convex-Hull",
		# "CoB",
		"Old",
		"New"
	],
	save_path="plots/predicted_rate_quality_curves/{}_{}_{}.png".format(codec, preset, video_file)
)