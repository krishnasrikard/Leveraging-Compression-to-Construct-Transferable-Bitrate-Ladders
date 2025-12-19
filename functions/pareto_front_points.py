"""
Functions for constructing Convex-Hull or Pareto-Front
"""
# Importing Libraries
import numpy as np
from scipy.interpolate import CubicHermiteSpline

import os, sys, warnings
import pickle
from tqdm import tqdm
warnings.filterwarnings("ignore")
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import functions.utils as utils


# Extract points on the Pareto-Front
def Pareto_Front_Points(
	RQ_pairs:dict=None,
	Resolutions:list=None,
	use_interpolated_points=False
):
	"""
	Extracting points from RQ_pairs to generate Pareto-Front. Cubic-Hermite Interpolation functions for various resolutions are used to compare mutiple resolutions at different bitrates.
	Args:
		RQ_pairs (dict): Dictionary with resolution as keys containing (bitrate (in kbps), quality) points extracted from selected rate_control json file. (Default: None)
		Resolutions (list): Resolutions that needs to be considered while plotting RQ points. (Default: None)
		use_interpolated_points (bool): Whether to use interpolated points or not. If True, it doesn't return CRF values of each RQ-point in ParetoFront_pairs and ParetoFront_info. (Default: False)
	Returns:
		ParetoFront_pairs (dict): Dictionary with resolution as keys containing (bitrate (in kbps), quality, rate_control_setting_value) points on pareto-front.
	"""
	# All the bitrates obtained from encoding with various resolution and RQ pairs of a particular video are considered.

	# Cubic-Hermite Interpolation functions for various resolutions.
	all_bitrates = []
	RQ_pairs_Interpolation_Functions = {}
	for res in Resolutions:
		# Need atleast 2 RQ points for a RQ curve of a resolution
		if len(RQ_pairs[res]) > 1:
			Monotonic_RQ_Points = utils.Maintain_Monotonicity(RQ_Points=RQ_pairs[res])
			bitrate = Monotonic_RQ_Points[:,0]
			quality = Monotonic_RQ_Points[:,1]

			RQ_pairs_Interpolation_Functions[res] = CubicHermiteSpline(bitrate, quality, dydx=utils.dydx(bitrate, quality))

		if len(RQ_pairs[res]) > 0:
			all_bitrates.append(np.asarray(RQ_pairs[res])[:,0])

	# All Bitrates
	all_bitrates = np.concatenate(all_bitrates, axis=0)
	all_bitrates = np.sort(all_bitrates)

	# Extracting points on Pareto-Front
	ParetoFront_pairs = {res:[] for res in Resolutions}

	for _,bitrate in enumerate(all_bitrates):
		interpolated_quality = []
		for i,res in enumerate(Resolutions):
			if len(RQ_pairs[res]) > 1:
				if bitrate <= np.max(RQ_pairs[res][:,0]) and bitrate >= np.min(RQ_pairs[res][:,0]):
					# If bitrate lie in range of resolutions's RQ curve, we calculate the Interpolated Quality. (If-Condition avoid's extrapolation)
					quality = RQ_pairs_Interpolation_Functions[res](bitrate)
					interpolated_quality.append(quality)
				else:
					# Else, we assume it's -1
					interpolated_quality.append(-1)
		
		# Optimal Resolution and Optimal RQ pair: Resolution with high quality at a particular bitrate
		optimal_res = Resolutions[np.argmax(interpolated_quality)]
		optimal_rq_pair = (bitrate, np.max(interpolated_quality))

		# Selecting Points on Pareto-Front
		if len(RQ_pairs[optimal_res]) == 0:
			# If there are no RQ points that fit the considered Quality Constraints
			None
		
		elif use_interpolated_points == True:
			# If interpolation points can be used
			ParetoFront_pairs[optimal_res].append(optimal_rq_pair)

		else:
			# If interpolation points cannot be used, the find the closest point
			RQ_data = RQ_pairs[optimal_res][:,0:2]
			Optimal_data = np.asarray(optimal_rq_pair)

			# Closest-Bitrate
			closest_bitrate_index = None

			for i in range(len(RQ_data)):
				if np.isclose(np.round(Optimal_data, decimals=4)[0], np.round(RQ_data[i], decimals=4)[0]) == True:
					if closest_bitrate_index is None:
						closest_bitrate_index = i
					else:
						print (Optimal_data, RQ_data[i], RQ_data[closest_bitrate_index], i, res)
						print (RQ_pairs[optimal_res])
						assert False, "Found Multiple Closest Bitrates"


			if closest_bitrate_index is not None:
				data = list(RQ_pairs[optimal_res][closest_bitrate_index])

				if len(ParetoFront_pairs[optimal_res]) == 0:
					ParetoFront_pairs[optimal_res].append(data)
				else:
					if any(np.isclose(ParetoFront_pairs[optimal_res][i][-1], data[-1]) for i in range(len(ParetoFront_pairs[optimal_res]))) == False:
						ParetoFront_pairs[optimal_res].append(data)
			else:
				# If bitrate is not in RQ_pairs[optimal_res], we skip
				None
				# print (Optimal_data, res)
				# print (RQ_pairs[optimal_res])


	# Sorting (R,Q,rc) pairs by bitrate
	for res in Resolutions:
		ParetoFront_pairs[res].sort()
		ParetoFront_pairs[res] = np.asarray(ParetoFront_pairs[res])

	return ParetoFront_pairs


# Finding the length of Largest Common Increasing Sequence
def findLengthOfLCIS(nums):
	# max-length
	ans = 0
	count = 0
	for i in range(0, len(nums)):
		if nums[i] == 0:
			count = 0
		else:
			if nums[i] >= nums[i - 1]:
				count += 1
				ans = max(ans, count)

	return ans


# Function to reject outliers based on CRF or QP values.
def reject_outlier_rate_control(data):
	"""
	Rejecting outliers based on CRF/QP values.
	- For end RQ-points, we expect the CRF difference between their close neighbour (only one) to be less that 4.
	- For middle RQ-points, we expect the CRF difference betweem their closest neighbours (both) to be less that 4. 
	"""
	mask = []
	if len(data) <= 2:
		mask = np.ones((len(data)))
	elif len(data) == 3:
		for i in range(0,len(data)):
			if i == 0:
				if abs(data[i,-1] - data[i+1,-1]) <= 4:
					mask.append(1)
				else:
					mask.append(0)
			elif i < len(data)-1:
				if abs(data[i,-1] - data[i+1,-1]) <= 4 or abs(data[i,-1] - data[i-1,-1]) <= 4:
					mask.append(1)
				else:
					mask.append(0)
			else:
				if abs(data[i,-1] - data[i-1,-1]) <= 4:
					mask.append(1)
				else:
					mask.append(0)
	else:
		for i in range(0,len(data)):
			if i == 0:
				if abs(data[i,-1] - data[i+1,-1]) <= 4:
					mask.append(1)
				else:
					mask.append(0)
			elif i < len(data)-1:
				if abs(data[i,-1] - data[i+1,-1]) <= 4 and abs(data[i,-1] - data[i-1,-1]) <= 4:
					mask.append(1)
				else:
					mask.append(0)
			else:
				if abs(data[i,-1] - data[i-1,-1]) <= 4:
					mask.append(1)
				else:
					mask.append(0)
	
	mask = np.asarray(mask)
	maxlength = findLengthOfLCIS(mask)

	start_index = 0
	for i in range(0, len(mask)-maxlength+1):
		if np.prod(mask[i:i+maxlength]) == 1:
			start_index = i
			break

	mask = mask * 0
	mask[start_index:start_index+maxlength] = np.ones(maxlength)
	mask = np.where(mask == 1)

	return mask


# Function to correct the points on Pareto-Front
def Correct_Pareto_Front(
	RQ_pairs:dict,
):
	"""
	Correcting the pareto-front so that
	- Resolutions are in ascending order
	- There is no overlap between resolutions. Let R2 be high resolution and R1 be lower resolution. 
		- RQ-Points from R2 start after highest bitrate of R1.
		- If RQ-points of R2 exist below highest bitrate of R1, they are removed from RQ_pairs.
		- This is considered for smooth continuity of pareto-front.
	Args:
		RQ_pairs (dict): Dictionary with resolution as keys containing (bitrate (in kbps), quality, rate_control_setting_value) points extracted from selected rate_control json file.
	Returns:
		Updated_RQ_pairs (dict): Updated RQ_pairs after correcting the Pareto-Front.
	"""
	# Resolutions
	Resolutions = list(RQ_pairs.keys())
	Resolutions.sort(reverse=True)

	# Points on Pareto-Front
	Updated_RQ_pairs = {res:[] for res in Resolutions}

	# Starting Bitrate
	min_bitrate_previous_resolution = np.inf

	for _,res in enumerate(Resolutions):
		Data = RQ_pairs[res]

		# Selecting Points on Pareto-Front
		for _,data in enumerate(Data):
			if len(data) == 0:
				Updated_RQ_pairs[res].append([])
			else:
				if data[0] < min_bitrate_previous_resolution:
					Updated_RQ_pairs[res].append(data.tolist())

		# Converting to a Numpy Array
		if len(Updated_RQ_pairs[res]) > 0:
			Updated_RQ_pairs[res].sort()
			Updated_RQ_pairs[res] = np.asarray(Updated_RQ_pairs[res])
		else:
			Updated_RQ_pairs[res] = np.array([])

		# Removing Outliers
		if len(Updated_RQ_pairs[res]) > 0:
			mask = reject_outlier_rate_control(data=Updated_RQ_pairs[res])
			Updated_RQ_pairs[res] = np.copy(Updated_RQ_pairs[res])[mask]
		else:
			Updated_RQ_pairs[res] = np.copy(Updated_RQ_pairs[res])

		# Setting Minimum bitrate of current resolution i.e Upper Bound of bitrate of next resolution
		if len(Updated_RQ_pairs[res]) > 0:
			min_bitrate_previous_resolution = np.min(Updated_RQ_pairs[res][:,0])

	return Updated_RQ_pairs