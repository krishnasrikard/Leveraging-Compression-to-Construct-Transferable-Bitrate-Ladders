"""
Box Plots for distribution of bitrate and quality
"""
# Importing Libraries
import numpy as np
import matplotlib.pyplot as plt

import os, sys, warnings
warnings.filterwarnings("ignore")
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import functions.extract_features as extract_features
import defaults


# Parameters
Resolutions = defaults.resolutions


# Function to plot Box-Plots
def Box_Plot(
	Meta_Information:dict,
	resolutions:list,
	Rate_Controls:list,
	x_axis:str,
	y_axis:str,
	x_label:str,
	y_label:str,
	save_path:str
):
	"""
	Box-Plot of compressed videos metadata for different resolutions or Rate_Controls
	Args:
		Meta_Information (dict): Dictionary containing Meta-Information of various video files
		resolutions (list): List of resolutions.
		Rate_Controls (list): List of Rate_Controls.
		x_axis (str): Options: ["resolutions", "crfs"].
		y_axis (str): Options: ["bitrate", "quality"].
		x_label (str): Label on x-axis in the figure.
		y_label (str): Label on y-axis in the figure.
		save_path (str): Path to save plot.
	"""
	# Assertions
	assert x_axis in ["resolutions", "crfs", "qps"], "Invalid x-axis"
	assert y_axis in ["bitrate", "quality"], "Invalid y-axis"

	# Index
	if y_axis == "bitrate":
		index = 0
	else:
		index = 1

	# Box-Plot Data
	Data = []

	# Extract information based on X and Y axis arguments
	if x_axis == "resolutions":
		labels = resolutions

		for _,res in enumerate(resolutions):
			scaled_h = np.round(res[1]/3840, decimals=4)
			data_per_resolution = []

			for _,video_file in enumerate(Meta_Information.keys()):
				mask = [i for i,h in enumerate(Meta_Information[video_file][:,-1]) if np.isclose(np.round(h, decimals=4), scaled_h)]
				data_per_resolution.append(Meta_Information[video_file][mask][:,index])

			data_per_resolution = np.concatenate(data_per_resolution, axis=0)
			Data.append(data_per_resolution)
	else:
		labels = Rate_Controls

		for _,RC in enumerate(Rate_Controls):
			data_per_rc = []

			for _,video_file in enumerate(Meta_Information.keys()):
				mask = [i for i,rc in enumerate(Meta_Information[video_file][:,-3]) if rc == RC]
				data_per_rc.append(Meta_Information[video_file][mask][:,index])

			data_per_rc = np.concatenate(data_per_rc, axis=0)
			Data.append(data_per_rc)


	# Box-Plot
	plt.figure(figsize=(10,8))
	plt.grid()
	plt.title("Box-plot of {} vs {}".format(y_label, x_label))
	plt.xlabel(x_label)
	plt.ylabel(y_label)
	plt.boxplot(Data, labels=labels, patch_artist=True, meanline=True, showmeans=True)
	plt.plot(1+np.arange(len(labels)), np.asarray([np.mean(Data[i]) for i in range(len(Data))]), color="green", linewidth=2, label="Mean")
	plt.plot(1+np.arange(len(labels)), np.asarray([np.median(Data[i]) for i in range(len(Data))]), color="orange", linewidth=2, label="Median")
	plt.legend()
	plt.savefig(save_path, dpi=400, bbox_inches='tight')



# Plotting Box-Plots for each codec and preset
for codec in defaults.codec_preset_pairs.keys():
	for preset in defaults.codec_preset_pairs[codec]:
		# Extracting RQ Information for all video files
		Meta_Information = extract_features.Extract_RQ_Features(
			codec=codec,
			preset=preset,
			quality_metric="vmaf",
			video_filenames=defaults.Video_Titles,
			Resolutions_Considered=Resolutions,
			CRFs_Considered=defaults.codec_CRF_ranges[codec],
			QPs_Considered=None,
			min_quality=-np.inf,
			max_quality=np.inf,
			min_bitrate=-np.inf,
			max_bitrate=np.inf
		)

		# Plotting
		for _,x_info in enumerate([("resolutions", "Resolutions")]):
			for _,y_info in enumerate([("bitrate", "Bitrate"), ("quality", "VMAF")]):
				x_axis,x_label = x_info
				y_axis,y_label = y_info

				Box_Plot(
					Meta_Information=Meta_Information,
					resolutions=Resolutions,
					Rate_Controls=defaults.codec_CRF_ranges[codec],
					x_axis=x_axis,
					y_axis=y_axis,
					x_label=x_label,
					y_label=y_label,
					save_path="plots/box_plots/{}_{}_Box_Plot_{}_{}.png".format(codec, preset,x_label,y_label)
				)