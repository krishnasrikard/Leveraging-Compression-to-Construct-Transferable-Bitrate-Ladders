"""
Mapping and Analyzing CRFs between different codecs and presets.
"""
# Importing Libraries
import numpy as np
np.set_printoptions(suppress=True)
import matplotlib.pyplot as plt

import os, sys, warnings
warnings.filterwarnings("ignore")
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import joblib
from tqdm import tqdm
import pprint
import functions.extract_features as extract_features
import defaults


def get_map(
	codecA:str,
	presetA:str,
	codecB:str,
	presetB:str,
	quality_metric:str,
	video_filenames:list,
	Resolutions_Considered:list
):
	"""
	Extracting an Median Rate-Control Map between (CodecA, PresetA) and (CodecB, PresetB).

	Args:
		codecA (str): Reference Codec used to generate RQ points that need to be extracted.
		presetA (str): Reference Preset used to generate RQ points that need to be extracted.
		codecB (str): Target Codec used to generate RQ points that need to be extracted.
		presetB (str): Target Preset used to generate RQ points that need to be extracted.
		quality_metric (str): Quality Metric to consider.
		video_filenames (list): List of video filenames to be considered for feature-extraction.
		Resolutions_Considered (list): Resolutions to be considered.
	"""
	# Mapping: Finding CRFs in Encoder Settings A for each CRF using Encoder Settings B to achieve the criterion.
	Bitrate_Mapping = {}
	Quality_Mapping = {}
	Combined_Mapping = {}

	for _,resolution in tqdm(enumerate(Resolutions_Considered), desc="codec: {}, preset: {}".format(codecB, presetB)):
		Bitrate_Mapping[resolution] = {}
		Quality_Mapping[resolution] = {}
		Combined_Mapping[resolution] = {}
		
		for CRF in defaults.codec_full_CRF_ranges[codecB]:
			Bitrate_Mapping[resolution][CRF] = []
			Quality_Mapping[resolution][CRF] = []
			Combined_Mapping[resolution][CRF] = []

			for _,video_file in enumerate(video_filenames):
				Meta_InformationA = extract_features.Extract_RQ_Features(
					codec=codecA,
					preset=presetA,
					quality_metric=quality_metric,
					video_filenames=[video_file],
					Resolutions_Considered=[resolution],
					CRFs_Considered=defaults.codec_full_CRF_ranges[codecA],
					QPs_Considered=None,
					min_quality=-np.inf,
					max_quality=np.inf,
					min_bitrate=-np.inf,
					max_bitrate=np.inf
				)[video_file]

				Meta_InformationB = extract_features.Extract_RQ_Features(
					codec=codecB,
					preset=presetB,
					quality_metric=quality_metric,
					video_filenames=[video_file],
					Resolutions_Considered=[resolution],
					CRFs_Considered=[CRF],
					QPs_Considered=None,
					min_quality=-np.inf,
					max_quality=np.inf,
					min_bitrate=-np.inf,
					max_bitrate=np.inf
				)[video_file]

				# Closest CRF for Bitrate_Mapping
				error = np.linalg.norm(Meta_InformationA[:,[0]] - Meta_InformationB[:,[0]], axis=1)
				index = np.argmin(error)
				Bitrate_Mapping[resolution][CRF].append(Meta_InformationA[index, -3])

				# Closest CRF for Quality_Mapping
				error = np.linalg.norm(Meta_InformationA[:,[1]] - Meta_InformationB[:,[1]], axis=1)
				index = np.argmin(error)
				Quality_Mapping[resolution][CRF].append(Meta_InformationA[index, -3])

				# Closest CRF for Quality_Mapping
				error = np.linalg.norm(Meta_InformationA[:,[0,1]] - Meta_InformationB[:,[0,1]], axis=1)
				index = np.argmin(error)
				Combined_Mapping[resolution][CRF].append(Meta_InformationA[index, -3])

			# Median CRF
			Bitrate_Mapping[resolution][CRF] = np.array(Bitrate_Mapping[resolution][CRF])
			Quality_Mapping[resolution][CRF] = np.array(Quality_Mapping[resolution][CRF])
			Combined_Mapping[resolution][CRF] = np.array(Combined_Mapping[resolution][CRF])

	# Return
	Mapping = {}
	Mapping["Bitrate"] = Bitrate_Mapping
	Mapping["Quality"] = Quality_Mapping
	Mapping["Both"] = Combined_Mapping

	return Mapping


def save_maps(
	codec:str,
	preset:str,
	fast_codec:str,
	fast_preset:str,
):
	# Logging
	print ("\n")
	print ("Calculating Maps for codec: {} and preset: {}".format(codec, preset))
	print ()

	Mapping = get_map(
		codecA=fast_codec,
		presetA=fast_preset,
		codecB=codec,
		presetB=preset,
		quality_metric="vmaf",
		video_filenames=defaults.Video_Titles,
		Resolutions_Considered=defaults.resolutions
	)

	# Logging
	print ()
	print ("Saving Maps")
	print ()

	np.save("plots/CRF_maps/maps/{}_{}-{}_{}.npy".format(fast_codec, fast_preset, codec, preset), Mapping)


def save_per_resolution_plot(
	codec:str,
	preset:str,
	fast_codec:str,
	fast_preset:str,
	save_dir:str
):
	# Load Mappings
	Mappings = np.load("plots/CRF_maps/maps/{}_{}-{}_{}.npy".format(fast_codec, fast_preset, codec, preset), allow_pickle=True)[()]

	# Plot
	plt.figure(figsize=(30,15))
	
	for i,resolution in enumerate(defaults.resolutions):
		plt.subplot(2,3,i+1)
		plt.title(resolution)
		plt.grid()
		plt.xlabel("{}, {}".format(codec, preset))
		plt.ylabel("{}, {}".format(fast_codec, fast_preset))

		for criterion in ["Bitrate"]:
			X = sorted(list(Mappings[criterion][resolution].keys()))
			Y = []
			for x in X:
				Y.append(Mappings[criterion][resolution][x])
			plt.boxplot(Y, labels=X, patch_artist=True, meanline=True, showmeans=True)

	plt.savefig(os.path.join(save_dir, "{}_{}-{}_{}.png".format(fast_codec, fast_preset, codec, preset)), dpi=400, bbox_inches='tight', pad_inches=0.1)


def save_simple_plot(
	codec:str,
	preset:str,
	fast_codec:str,
	fast_preset:str,
	save_dir:str
):
	# Load Mappings
	Mappings = np.load("plots/CRF_maps/maps/{}_{}-{}_{}.npy".format(fast_codec, fast_preset, codec, preset), allow_pickle=True)[()]

	# Plot
	plt.figure(figsize=(10,6))
	plt.grid()
	plt.xlabel("{}, {}".format(codec, preset))
	plt.ylabel("{}, {}".format(fast_codec, fast_preset))
	
	for criterion in ["Bitrate"]:
		X = sorted(list(Mappings[criterion][defaults.resolutions[0]].keys()))
		Y = []
		for x in X:
			data = []
			for _,resolution in enumerate(defaults.resolutions):	
				data.append(Mappings[criterion][resolution][x])
			Y.append(np.array(data).flatten())

		plt.boxplot(Y, labels=X, patch_artist=True, meanline=True, showmeans=True)

	plt.savefig(os.path.join(save_dir, "{}_{}-{}_{}.png".format(fast_codec, fast_preset, codec, preset)), dpi=400, bbox_inches='tight', pad_inches=0.1)



# libx265, veryfast as fast encoder
# """
Codec_Preset_pairs = []
for codec in defaults.codec_preset_pairs.keys():
	for preset in defaults.codec_preset_pairs[codec]:
		if codec == "libx265" and preset == "veryfast":
			continue

		Codec_Preset_pairs.append((codec, preset))

joblib.Parallel(n_jobs=-1)(
	joblib.delayed(save_maps)(
		codec=codec,
		preset=preset,
		fast_codec="libx265",
		fast_preset="veryfast",
	) for codec, preset in Codec_Preset_pairs
)

for codec, preset in Codec_Preset_pairs:
	print (codec, preset)

	save_simple_plot(
		codec=codec,
		preset=preset,
		fast_codec="libx265",
		fast_preset="veryfast",
		save_dir="plots/CRF_maps"
	)
# """


# libsvtav1, 8 as fast encoder
Codec_Preset_pairs = []
for codec in defaults.codec_preset_pairs.keys():
	for preset in defaults.codec_preset_pairs[codec]:
		if codec == "libsvtav1" and preset == "8":
			continue

		Codec_Preset_pairs.append((codec, preset))


joblib.Parallel(n_jobs=-1)(
	joblib.delayed(save_maps)(
		codec=codec,
		preset=preset,
		fast_codec="libsvtav1",
		fast_preset="8"
	) for codec, preset in Codec_Preset_pairs
)

for codec, preset in Codec_Preset_pairs:
	print (codec, preset)

	save_simple_plot(
		codec=codec,
		preset=preset,
		fast_codec="libsvtav1",
		fast_preset="8",
		save_dir="plots/CRF_maps"
	)