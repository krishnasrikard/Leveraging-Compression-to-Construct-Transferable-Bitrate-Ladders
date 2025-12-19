"""
Functions to Construct Convex-Hull/Pareto-Front Bitrate Ladders
"""
# Importing Libraries
import numpy as np

import os, sys, warnings
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import functions.IO_functions as IO_functions
import functions.extract_functions as extract_functions
import functions.pareto_front_points as pareto_front_points
import defaults


# Apple's Fixed Bitrate Ladder
def Construct_Apple_Fixed_Bitrate_Ladder(
	evaluation_bitrates:list
):
	"""
	Returns Bitrate-Ladder for corresponding bitrates using Apple's Bitrate Ladder
	Args:
		bitrates (list): List of bitrates present to be in the bitrate ladder.
	"""
	# Resolutions
	Resolutions = defaults.resolutions
	evaluation_bitrates = list(evaluation_bitrates)
	evaluation_bitrates.sort(reverse=True)

	# Apple's Bitrate Ladder's Cross-Over Bitrate
	CrossOver_Bitrates = np.asarray([11600, 8100, 4500, 2400, 600, 300, 145])
	CrossOver_Bitrates = np.round(np.log2(1000*CrossOver_Bitrates), decimals=4)

	# Calculating Bitrate-Ladder
	Bitrate_Ladder = {}
	for i in range(len(evaluation_bitrates)):
		# Switching happends to higher resolution when bitrate >= cross-over-bitrate of corresponding higher resolution.
		b = evaluation_bitrates[i]

		Bitrate_Ladder[b] = None

		for j in range(1+len(CrossOver_Bitrates)):
			if (j==0) and (b >= CrossOver_Bitrates[j]):
				Bitrate_Ladder[b] = Resolutions[0]
			elif (j <= len(CrossOver_Bitrates)-1) and (CrossOver_Bitrates[j] <= b < CrossOver_Bitrates[j-1]):
				if j < len(Resolutions):
					Bitrate_Ladder[b] = Resolutions[j]
				else:
					# For evaluation bitrates less that 600, we sill consider 540p videos as 540p is the limit for out experiments
					Bitrate_Ladder[b] = Resolutions[-1]
			elif (j==len(CrossOver_Bitrates)) and (b < CrossOver_Bitrates[j-1]):
				Bitrate_Ladder[b] = Resolutions[-1]
			else:
				None

		if Bitrate_Ladder[b] is None:
			assert False, "Something is Wrong"

	# Debugging
	# print ()
	# print (CrossOver_Bitrates)
	# print (Bitrate_Ladder)
	# print ()

	return Bitrate_Ladder



# Constructing Convex-Hull Bitrate Ladder
def Construct_Convex_Hull_Bitrate_Ladder(
	video_file:str,
	codec:str,
	preset:str,
	evaluation_bitrates:list
):
	"""
	Returns Bitrate-Ladder of Convex-Hull for corresponding evaluation_bitrates using Pareto Front constructed
	Args:
		video_file (str): The video file name.
		codec (str): Codec used to generate RQ points that need to be extracted.
		preset (str): Preset used to generate RQ points that need to be extracted.
		evaluation_bitrates (list): List of bitrates present to be in the bitrate ladder.
	"""
	# Resolutions
	Resolutions = defaults.resolutions
	Resolutions.sort(reverse=True)
	evaluation_bitrates.sort(reverse=True)

	# Reading RQ-Info of video file
	video_rq_points_info = IO_functions.read_create_jsonfile(os.path.join(defaults.rq_points_dataset_path, codec, preset, video_file, "crfs.json"))

	# Rate-Quality Dataset
	RQ_pairs = extract_functions.Extract_RQ_Information(
		video_rq_points_info=video_rq_points_info,
		quality_metric="vmaf",
		resolutions=defaults.resolutions,
		CRFs=defaults.codec_CRF_ranges[codec],
		QPs=None,
		min_quality=defaults.min_quality,
		max_quality=defaults.max_quality,
		min_bitrate=defaults.min_bitrate,
		max_bitrate=defaults.max_bitrate,
		set_bitrate_log_base=2
	)

	# Constructing Pareto-Front
	ParetoFront_pairs = pareto_front_points.Pareto_Front_Points(
		RQ_pairs=RQ_pairs,
		Resolutions=defaults.resolutions,
		use_interpolated_points=False
	)
	ParetoFront_pairs = pareto_front_points.Correct_Pareto_Front(ParetoFront_pairs)

	# Cross-Over Bitrates
	CrossOver_Bitrates = []
	for i in range(len(Resolutions)-1):
		if ParetoFront_pairs[Resolutions[i]].shape[0] > 0:
			# Switching happens to higher resolution from cross-over bitrate
			CrossOver_Bitrates.append(np.min(ParetoFront_pairs[Resolutions[i]][:,0]))
		else:
			if i==0:
				# If Pareto-Front doesn't contain highest resolution, we assuming highest resolution dominates after infinity.
				# Generally, higher resolution can dominate lower resolution after for some smaller CRF value. But consider our quality constraints and CRF values, that CRF value doesn't lie in our experiments.
				CrossOver_Bitrates.append(np.inf)
			else:
				# If Pareto-Front doesn't contain a resolution, cross-over bitrate of previous highest resolution is cross-over bitrate of current resolution.
				CrossOver_Bitrates.append(CrossOver_Bitrates[-1])

	# 540p should be encoded from starting of minimum of evaluation bitrates
	# We are setting it to -np.inf to avoid duplication of keys in Bitrate_Ladder because of "Imposing Monotonicity"
	CrossOver_Bitrates.append(-np.inf)

	# Imposing Monotonicity on estimated cross-over bitrates
	for i in range(1,len(CrossOver_Bitrates)):
		CrossOver_Bitrates[i] = min(CrossOver_Bitrates[i], CrossOver_Bitrates[i-1])
	

	# Bitrate Ladder
	Bitrate_Ladder = {}
	for i,b in enumerate(CrossOver_Bitrates):
		if b in Bitrate_Ladder.keys():
			Bitrate_Ladder[np.round(b - 1e-4, decimals=4)] = Resolutions[i]
		else:
			Bitrate_Ladder[b] = Resolutions[i]

	# Debugging
	# print ()
	# print (CrossOver_Bitrates)
	# print (Bitrate_Ladder)
	# print ()

	return Bitrate_Ladder
