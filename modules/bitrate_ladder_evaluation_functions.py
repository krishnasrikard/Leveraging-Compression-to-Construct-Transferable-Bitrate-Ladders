"""
Calculating BD-Metrics from Bitrate Ladder
"""
# Importing Libraries
import numpy as np
np.set_printoptions(suppress=True)

import os, sys, warnings
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import functions.IO_functions as IO_functions
import functions.extract_functions as extract_functions
import functions.bjontegaard_metric as bd_metrics
import functions.correction_algorithms as correction_algorithms
import defaults



# Create Rate-Quality Curve using Bitrate Ladder
def Rate_Quality_Curve_from_Bitrate_Ladder(
	RQ_pairs:dict,
	Bitrate_Ladder:dict
):
	"""
	Create Pareto-Front using Bitrate Ladder
	Args:
		RQ_pairs (dict): The rate-quality information of a video.
		Bitrate_Ladder (dict): The bitrate-ladder that should be used for construction of pareto-front.
	Returns:
		(dict): Pareto-Front with reoslutions as keys and (rate, quality) as values for each resolution.
		(list): Bitrate-Quality points on the pareto-front without resolution information.
	"""
	# Resolutions
	Resolutions = defaults.resolutions

	# Pareto-Front
	RQ_curve_pairs = {}
	for res in defaults.resolutions:
		RQ_curve_pairs[res] = []

	# Sorting Bitrate Ladder
	Bitrate_Ladder = dict(sorted(Bitrate_Ladder.items(), reverse=True))

	# Global Thresholds
	# Quality Threshold are the same. But we consider Bitrate Thresholds based on the range of evaluation bitrates
	global_min_quality = defaults.min_quality
	global_max_quality = defaults.max_quality

	if defaults.min_bitrate > 0:
		global_min_bitrate = np.log2(defaults.min_bitrate)
	else:
		global_min_bitrate = defaults.min_bitrate
		
	if np.isnan(defaults.max_bitrate):
		global_max_bitrate = defaults.max_bitrate
	else:
		global_max_bitrate = np.log2(defaults.max_bitrate)

	# Thresholds during construction
	temp_min_quality = defaults.min_quality
	temp_max_quality = defaults.max_quality

	if defaults.min_bitrate > 0:
		temp_min_bitrate = np.log2(defaults.min_bitrate)
	else:
		temp_min_bitrate = defaults.min_bitrate

	if np.isnan(defaults.max_bitrate):
		temp_max_bitrate = defaults.max_bitrate
	else:
		temp_max_bitrate = np.log2(defaults.max_bitrate)


	# Changing global_min_bitrate and temp_min_bitrate
	global_min_bitrate = np.min(defaults.evaluation_bitrates)
	temp_min_bitrate = np.min(defaults.evaluation_bitrates)

	# Adding Points to Rate-Quality
	for b_step in Bitrate_Ladder.keys():
		# Updating Thresholds: Min bitrate should be based on the step of bitrate
		temp_min_bitrate = b_step

		# Resolution the video should be encoded according to bitrate ladder
		res = Bitrate_Ladder[b_step]

		for rq_point in RQ_pairs[res]:
			# Debugging
			"""
			print (rq_point)
			print ()
			print (rq_point[0] >= global_min_bitrate, rq_point[0] <= global_max_bitrate, rq_point[1] >= global_min_quality, rq_point[1] <= global_max_quality, rq_point[0] >= temp_min_bitrate, rq_point[0] <= temp_max_bitrate, rq_point[1] >= temp_min_quality, rq_point[1] <= temp_max_quality)
			print ()
			"""

			# Global and Temporary Threshold
			if (rq_point[0] >= global_min_bitrate and rq_point[0] <= global_max_bitrate) and (rq_point[1] >= global_min_quality and rq_point[1] <= global_max_quality) and (rq_point[0] >= temp_min_bitrate and rq_point[0] <= temp_max_bitrate) and (rq_point[1] >= temp_min_quality and rq_point[1] <= temp_max_quality) :
				# Ignoring Duplicates
				Consider = True

				for check_res in Resolutions:
					if any(
						np.isclose(
							np.round(rq_point[0], decimals=2), 
							np.round(row[0], decimals=2)
						) or 
						np.isclose(
							np.round(rq_point[1], decimals=2), 
							np.round(row[1], decimals=2)
						)
						for row in RQ_curve_pairs[check_res]
					):
						Consider = False
				
				# Consider the RQ point if there is no conflict
				if Consider:
					RQ_curve_pairs[res].append([rq_point[0], rq_point[1]])
			else:
				None

		# Updating Thresholds
		data = np.asarray(RQ_curve_pairs[res])
		if len(data) > 0:
			temp_max_quality = np.min(data[:,1])
		temp_max_bitrate = b_step

	# Rate-Quality Points and Pairs
	RQ_Points = []
	for res in Resolutions:
		RQ_curve_pairs[res].sort()
		RQ_Points += RQ_curve_pairs[res]
		RQ_curve_pairs[res] = np.asarray(RQ_curve_pairs[res])

	# RQ Points
	RQ_Points.sort()
	RQ_Points = np.asarray(RQ_Points)

	# Assertions
	if len(RQ_Points) > 0:
		assert (
			np.all(RQ_Points[:,0] >= global_min_bitrate) and 
			np.all(RQ_Points[:,0] <= global_max_bitrate) and 
			np.all(RQ_Points[:,1] >= global_min_quality) and
			np.all(RQ_Points[:,1] <= global_max_quality) 
		), "Points on the Rate-Qaulity curve are not the global bitrate-quality range."
	else:
		None
		# Debugging
		# print ()
		# print (Bitrate_Ladder)
		# print ()
		# print (RQ_pairs)
		# print ()

	# Debugging
	# print ()
	# print (Bitrate_Ladder)
	# print ()
	# print (RQ_pairs)
	# print ()
	# print (RQ_curve_pairs)
	# print ()
	# print (RQ_Points)
	# print ()

	return RQ_curve_pairs, RQ_Points



# Calculating BD-Metrics from Bitrate Ladder
def Calculate_BD_Metrics(
	video_file:str,
	codec:str,
	preset:str,
	bitrate_ladder_path:str,
	fixed_bitrate_ladder_path:str,
	convex_hull_bitrate_ladder_path:str
):
	"""
	Args:
		video_file (str): The video file name.
		codec (str): Codec used to generate RQ points that need to be extracted.
		preset (str): Preset used to generate RQ points that need to be extracted.
		bitrate_ladder_path (str): The path to Bitrate Ladder that needs to be considered.
		fixed_bitrate_ladder_path (str): The path to Fixed Bitrate Ladder that needs to be considered.
		convex_hull_bitrate_ladder_path (str): The path to Reference Bitrate Ladder that needs to be considered.
	Returns:
		(float): BD-rate in percentage wrt Fixed Bitrate-Ladder
		(float): BD-quality wrt Fixed Bitrate-Ladder
		(float): BD-rate in percentage wrt Reference Bitrate-Ladder
		(float): BD-quality wrt Reference Bitrate-Ladder
	"""
	# Rate-Quality points
	RQ_pairs = extract_functions.Extract_RQ_Information(
		video_rq_points_info=IO_functions.read_create_jsonfile(os.path.join(defaults.rq_points_dataset_path, codec, preset, video_file, "crfs.json")),
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


	# Fixed Bitrate-Ladder
	AL = np.load(fixed_bitrate_ladder_path, allow_pickle=True)[()]
	AL = correction_algorithms.Top_Bottom(AL)

	# Reference Bitrate Ladder
	RL = np.load(convex_hull_bitrate_ladder_path, allow_pickle=True)[()][video_file]
	RL = correction_algorithms.Top_Bottom(RL)

	# Predicted Bitrate Ladder
	BL = np.load(bitrate_ladder_path, allow_pickle=True)[()][video_file]
	BL = correction_algorithms.Top_Bottom(BL)


	# Constructing Rate-Quality curves corresponding to Bitrate Ladder
	_, Fixed_RQ_points = Rate_Quality_Curve_from_Bitrate_Ladder(
		RQ_pairs=RQ_pairs,
		Bitrate_Ladder=AL
	)

	_, Convex_Hull_RQ_points = Rate_Quality_Curve_from_Bitrate_Ladder(
		RQ_pairs=RQ_pairs,
		Bitrate_Ladder=RL
	)
	
	_, RQ_points = Rate_Quality_Curve_from_Bitrate_Ladder(
		RQ_pairs=RQ_pairs,
		Bitrate_Ladder=BL
	)

	# Debugging
	"""
	print (Fixed_RQ_points)
	print ()
	print (Convex_Hull_RQ_points)
	print ()
	print (RQ_points)
	print ()
	"""


	# Converting Bitrate from log-scale to linear-scale
	if len(RQ_points) < 4:
		# Check: We have predicted a Bad Bitrate Ladder
		return None
	else:
		# Convertion
		RQ_points[:,0] = np.round(np.power(2, RQ_points[:,0]), decimals=4)
		# Assertion
		assert np.all(RQ_points[:,0] <= 30) == False, "Bitrate Ladder Pareto-Front points are in the wrong range."

	if len(Fixed_RQ_points) < 4:
		# If there are less than four points in Fixed Bitrate Ladder Range for the considered video, then gains against Fixed Bitrate Ladder are considered zero.
		None
	else:
		# Convertion
		Fixed_RQ_points[:,0] = np.round(np.power(2, Fixed_RQ_points[:,0]), decimals=4)
		# Assertion
		assert np.all(Fixed_RQ_points[:,0] <= 30) == False, "Fixed Bitrate Ladder Pareto-Front points are in the wrong range."


	
	# Convex-Hull should have more than four points
	# Assertion
	assert len(Convex_Hull_RQ_points) >= 4, "Convex-Hull Bitrate Ladder Pareto-Front has insufficient points."
	# Convertion
	Convex_Hull_RQ_points[:,0] = np.round(np.power(2, Convex_Hull_RQ_points[:,0]), decimals=4)
	# Assertion
	assert np.all(Convex_Hull_RQ_points[:,0] <= 30) == False, "Convex-Hull Bitrate Ladder Pareto-Front points are in the wrong range."
	
	
	# BD-Metrics wrt Apple Fixed Bitrate Ladder
	if len(Fixed_RQ_points) >= 4:
		f_bd_rate, iou_f_bd_rate = bd_metrics.BD_Rate(
			R1=Fixed_RQ_points[:,0],
			Q1=Fixed_RQ_points[:,1],
			R2=RQ_points[:,0],
			Q2=RQ_points[:,1],
			piecewise=True
		)
		f_bd_quality, iou_f_bd_quality = bd_metrics.BD_Quality(
			R1=Fixed_RQ_points[:,0],
			Q1=Fixed_RQ_points[:,1],
			R2=RQ_points[:,0],
			Q2=RQ_points[:,1],
			piecewise=True
		)
	else:
		# If there are no points in Fixed Bitrate Ladder Range for the considered video, then gains against Fixed Bitrate Ladder are considered zero.
		f_bd_rate = 0
		f_bd_quality = 0
		iou_f_bd_rate = 1
		iou_f_bd_quality = 1

	# BD-Metrics wrt Convex-Hull Bitrate Ladder
	r_bd_rate, iou_r_bd_rate = bd_metrics.BD_Rate(
		R1=Convex_Hull_RQ_points[:,0],
		Q1=Convex_Hull_RQ_points[:,1],
		R2=RQ_points[:,0],
		Q2=RQ_points[:,1],
		piecewise=True
	)
	r_bd_quality, iou_r_bd_quality = bd_metrics.BD_Quality(
		R1=Convex_Hull_RQ_points[:,0],
		Q1=Convex_Hull_RQ_points[:,1],
		R2=RQ_points[:,0],
		Q2=RQ_points[:,1],
		piecewise=True
	)

	return (
		np.round(f_bd_rate, decimals=4), np.round(f_bd_quality, decimals=4), 
		np.round(r_bd_rate, decimals=4), np.round(r_bd_quality, decimals=4), 
		np.round(iou_f_bd_rate, decimals=4), np.round(iou_f_bd_quality, decimals=4),
		np.round(iou_r_bd_rate, decimals=4), np.round(iou_r_bd_quality, decimals=4)
	)



# Correction
def Applying_Correction_BD_metrics(
	BD_metrics:np.array,
	iou_terms:np.array
):
	# Assertions
	assert (len(BD_metrics.shape) == 2) and (BD_metrics.shape[1] == 4), "Invalid BD-metrics shape"
	assert (len(iou_terms.shape) == 2) and (iou_terms.shape[1] == 4), "Invalid correction-terms shape"
	assert BD_metrics.shape[0] == iou_terms.shape[0], "Both BD-metrics and correction-terms don't have the same first dimensions"

	IoU_BD_metrics = []
	for i in range(BD_metrics.shape[0]):
		metrics = []
		
		# Fixed: BD-Rate
		if BD_metrics[i][0] < 0:
			metrics.append(BD_metrics[i][0] * iou_terms[i][0])
		else:
			metrics.append(BD_metrics[i][0] / iou_terms[i][0])

		# Fixed: BD-Quality
		if BD_metrics[i][1] < 0:
			metrics.append(BD_metrics[i][1] / iou_terms[i][1])
		else:
			metrics.append(BD_metrics[i][1] * iou_terms[i][1])

		# Convex-Hull: BD-Rate
		if BD_metrics[i][2] < 0:
			metrics.append(BD_metrics[i][2] * iou_terms[i][2])
		else:
			metrics.append(BD_metrics[i][2] / iou_terms[i][2])

		# Convex-Hull: BD-Quality
		if BD_metrics[i][3] < 0:
			metrics.append(BD_metrics[i][3] / iou_terms[i][3])
		else:
			metrics.append(BD_metrics[i][3] * iou_terms[i][3])

		# Append
		IoU_BD_metrics.append(metrics)

	IoU_BD_metrics = np.array(IoU_BD_metrics)

	return IoU_BD_metrics



# Calculating Closeness to Performance of Convex-Hull Bitrate 
def Calculate_Closeness(
	Predicted_Metrics:any,
	Reference_Metrics:any,
):
	"""
	Args:
		Predicted_Metrics (any): BD-Metrics of Predicted Bitrate Ladder against Fixed Bitrate Ladder.
		Reference_Metrics (any): BD-Metrics of Convex-Hull Bitrate Ladder against Fixed Bitrate Ladder.
	Returns:
		(int): Returns 1, if f_{25} is True
		(int): Returns 1, if f_{50} is True
		(int): Returns 1, if f_{75} is True
		(int): Returns 1, if f_{85} is True 
	"""
	# Closeness before Correction
	# Calculating fraction of samples close to convex-hull bitrate ladder performance
	if Predicted_Metrics[0] < 0.25*Reference_Metrics[0] and Predicted_Metrics[1] > 0.25*Reference_Metrics[1]:
		f_25 = 1
	else:
		f_25 = 0

	if Predicted_Metrics[0] < 0.50*Reference_Metrics[0] and Predicted_Metrics[1] > 0.50*Reference_Metrics[1]:
		f_50 = 1
	else:
		f_50 = 0

	if Predicted_Metrics[0] < 0.75*Reference_Metrics[0] and Predicted_Metrics[1] > 0.75*Reference_Metrics[1]:
		f_75 = 1
	else:
		f_75 = 0

	if Predicted_Metrics[0] < 0.85*Reference_Metrics[0] and Predicted_Metrics[1] > 0.85*Reference_Metrics[1]:
		f_85 = 1
	else:
		f_85 = 0

	# Apply Correction
	IoU_Predicted_Metrics = Applying_Correction_BD_metrics(
		BD_metrics=np.array([Predicted_Metrics[:4]]),
		iou_terms=np.array([Predicted_Metrics[4:]])
	)[0]
	
	IoU_Reference_Metrics = Applying_Correction_BD_metrics(
		BD_metrics=np.array([Reference_Metrics[:4]]),
		iou_terms=np.array([Reference_Metrics[4:]])
	)[0]

	# Closeness after Correction
	# Calculating fraction of samples close to convex-hull bitrate ladder performance
	if IoU_Predicted_Metrics[0] < 0.25*IoU_Reference_Metrics[0] and IoU_Predicted_Metrics[1] > 0.25*IoU_Reference_Metrics[1]:
		iou_f_25 = 1
	else:
		iou_f_25 = 0

	if IoU_Predicted_Metrics[0] < 0.50*IoU_Reference_Metrics[0] and IoU_Predicted_Metrics[1] > 0.50*IoU_Reference_Metrics[1]:
		iou_f_50 = 1
	else:
		iou_f_50 = 0

	if IoU_Predicted_Metrics[0] < 0.75*IoU_Reference_Metrics[0] and IoU_Predicted_Metrics[1] > 0.75*IoU_Reference_Metrics[1]:
		iou_f_75 = 1
	else:
		iou_f_75 = 0

	if IoU_Predicted_Metrics[0] < 0.85*IoU_Reference_Metrics[0] and IoU_Predicted_Metrics[1] > 0.85*IoU_Reference_Metrics[1]:
		iou_f_85 = 1
	else:
		iou_f_85 = 0	

	return f_25, f_50, f_75, f_85, iou_f_25, iou_f_50, iou_f_75, iou_f_85