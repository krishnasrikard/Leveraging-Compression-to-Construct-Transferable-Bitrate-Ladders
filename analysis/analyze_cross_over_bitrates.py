"""
Analyzing the Correlation between Cross-Over Bitrate between different encoder settings
"""
# Importing Libraries
import numpy as np
import matplotlib.pyplot as plt
import scipy

import os, sys, warnings
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import modules.dataset_functions as dataset_functions
import defaults


def calculate_correlation(
	codecA:str,
	presetA:str,
	codecB:str,
	presetB:str,
):
	CrossOver_BitratesA = np.array(list(dataset_functions.Extract_CrossOver_Bitrates(
		codec=codecA,
		preset=presetA,
		quality_metric="vmaf",
		video_filenames=defaults.Video_Titles,
		Resolutions_Considered=defaults.resolutions,
		CRFs_Considered=defaults.codec_CRF_ranges[codecA],
		QPs_Considered=None
	).values()))

	CrossOver_BitratesB = np.array(list(dataset_functions.Extract_CrossOver_Bitrates(
		codec=codecB,
		preset=presetB,
		quality_metric="vmaf",
		video_filenames=defaults.Video_Titles,
		Resolutions_Considered=defaults.resolutions,
		CRFs_Considered=defaults.codec_CRF_ranges[codecB],
		QPs_Considered=None
	).values()))

	for i in range(len(defaults.resolutions)-1):
		X1 = CrossOver_BitratesA[:,i]
		X2 = CrossOver_BitratesB[:,i]

		print ("PLCC =", np.round(scipy.stats.pearsonr(np.squeeze(X1), np.squeeze(X2))[0], decimals=3))


calculate_correlation(
	codecA="libx265",
	presetA="veryfast",
	codecB="libsvtav1",
	presetB="9"
)

