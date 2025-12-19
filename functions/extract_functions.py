"""
Functions
"""
# Importing Libraries
import numpy as np
import pandas as pd
import re

import os
import json
import csv
import subprocess


def extract_qualityestimates(
	logfile:str=None
):
	"""
	Extracting quality estimates from log files during dataset generation
	Args:
		logfile (str): Path to logfile containing quality estimations. (Default: None)
	"""
	f = open(logfile)
	I = json.load(f)
	f.close()

	Quality_Estimations = {}

	for metric in I['pooled_metrics']:
		Quality_Estimations[metric] = I['pooled_metrics'][metric]['mean']

	for metric in ["psnr_y", "float_ssim", "float_ms_ssim", "vmaf"]:
		if metric not in Quality_Estimations:
			Quality_Estimations[metric] = -1

	Quality_Estimations["frame_level_info"] = I["frames"]
	
	return Quality_Estimations


def extract_execution_time(O):
	"""
	Args:
		O (str): Output of the the process executed with /usr/bin/time as prefix to the command of execution.
	"""

	time_info = O.splitlines()[-2:]
	time_info = time_info[0].split()[0:2]
	usr_time = np.round(float(time_info[0][:-4]), decimals=4)
	sys_time = np.round(float(time_info[1][:-6]), decimals=4)
	total_execution_time = np.round(usr_time + sys_time, decimals=4)

	return total_execution_time


def extract_bitrate(O):
	"""
	Args:
		O (str): Output of the the process executed with /usr/bin/time as prefix to the command of execution.
	"""
	O = O.split("\n")
	bitrate = int(O[0])

	return bitrate


def process_SVTAV1_frame_info(
	compressed_video_path:str,
	frame_rate:float,
	frame_qp_info_path:str,
	frame_info_csv:str
):
	"""
	Args:
		compressed_video_path (str): Path to Compressed Videos
		frame_rate (float): Frame rate of the video.
		qps_txt (str): Path to QPs info file
		frame_info_csv (str): Path of Frame-Info.
	"""
	# Calculate frame duration
	frame_duration = 1.0 / frame_rate

	# Run ffprobe to extract frame information and save to CSV file
	ffprobe_command = [
		'ffprobe', '-show_frames', '-select_streams', 'v:0',
		'-show_entries', 'frame=pkt_size,pict_type', '-of', 'csv=p=0',
		compressed_video_path
	]
	with open(frame_info_csv, 'w') as csvfile:
		subprocess.run(ffprobe_command, stdout=csvfile)
		

	# Calculating Bitrate based on packet_size and frame_duration
	def calculate_bitrate(packet_size, frame_duration):
		return (packet_size * 8) / frame_duration


	# Read frame information from CSV file
	frame_info = []
	with open(frame_info_csv, 'r') as csvfile:
		reader = csv.reader(csvfile)
		for row in reader:
			pkt_size = int(row[0])
			pict_type = row[1]
			frame_info.append((pkt_size, pict_type))


	# Read QPs from text file
	qps = []
	print (frame_qp_info_path)
	with open(frame_qp_info_path, 'r') as qpfile:
		for line in qpfile:
			qps.append(int(line.strip()))


	# Clearing .txt file
	with open(frame_qp_info_path, 'w') as _:
		None


	# Assertion
	assert len(qps) == len(frame_info), "Both QPs and Frame-Info should have same length"


	# Calculate bitrate for each frame and combine the information
	Picture_Info = {}
	Picture_Info["I"] = []
	Picture_Info["P"] = []
	Picture_Info["B"] = []

	for i, (pkt_size, pict_type) in enumerate(frame_info):
		bitrate = calculate_bitrate(pkt_size, frame_duration)
		qp = qps[i]

		Picture_Info[pict_type].append([qp, bitrate])

	
	# Average
	output = {}
	for frame_type in ["I", "P", "B"]:
		if len(Picture_Info[frame_type]) > 0:
			output[frame_type + "_frame_Count"] = len(Picture_Info[frame_type])
			output[frame_type + "_frame_AvgQP"] = np.mean(np.array(Picture_Info[frame_type])[:,0])
			output[frame_type + "_frame_Bitrate"] = np.round(np.mean(np.array(Picture_Info[frame_type])[:,1])/1000.0, decimals=2)
		else:
			output[frame_type + "_frame_Count"] = -1
			output[frame_type + "_frame_AvgQP"] = -1
			output[frame_type + "_frame_Bitrate"] = -1

	return output



def extract_compression_logs(logs, video_codec):
	"""
	Args:
		logs (str): Output of the the process executed with /usr/bin/time as prefix to the command of execution.
		video_codec (str): The video-codec used for encoding.
	"""
	if video_codec == "libx264":
		pattern = {
			"I": r"frame I:\s*(\d+)\s*Avg QP:(\d+\.\d+)\s*size:\s*(\d+)",
			"P": r"frame P:\s*(\d+)\s*Avg QP:(\d+\.\d+)\s*size:\s*(\d+)",
			"B": r"frame B:\s*(\d+)\s*Avg QP:(\d+\.\d+)\s*size:\s*(\d+)"
		}

		output = {}
		for frame_type in ["I", "P", "B"]:
			match = re.search(pattern[frame_type], logs)
			if match:
				output[frame_type + "_frame_Count"] = int(match.group(1))
				output[frame_type + "_frame_AvgQP"] = float(match.group(2))
				output[frame_type + "_frame_Bitrate"] = -1
			else:
				output[frame_type + "_frame_Count"] = 0
				output[frame_type + "_frame_AvgQP"] = 0
				output[frame_type + "_frame_Bitrate"] = -1

	elif video_codec == "libx265":
		pattern = {
			"I": r"frame I:\s+(\d+), Avg QP:(\d+\.\d+)\s+kb/s:\s+(\d+\.\d+)",
			"P": r"frame P:\s+(\d+), Avg QP:(\d+\.\d+)\s+kb/s:\s+(\d+\.\d+)",
			"B": r"frame B:\s+(\d+), Avg QP:(\d+\.\d+)\s+kb/s:\s+(\d+\.\d+)"
		}
	
		output = {}
		for frame_type in ["I", "P", "B"]:
			match = re.search(pattern[frame_type], logs)
			if match:
				output[frame_type + "_frame_Count"] = int(match.group(1))
				output[frame_type + "_frame_AvgQP"] = float(match.group(2))
				output[frame_type + "_frame_Bitrate"] = float(match.group(3))
			else:
				output[frame_type + "_frame_Count"] = 0
				output[frame_type + "_frame_AvgQP"] = 0
				output[frame_type + "_frame_Bitrate"] = -1

	else:
		output = {}
		for frame_type in ["I", "P", "B"]:
			output[frame_type + "_frame_Count"] = -1
			output[frame_type + "_frame_AvgQP"] = -1
			output[frame_type + "_frame_Bitrate"] = -1
	
		
	return output


def Extract_RQ_Information(
	video_rq_points_info:dict=None,
	quality_metric:str=None,
	resolutions:list=None,
	CRFs:any=None,
	QPs:list=None,
	min_quality=-np.inf,
	max_quality=np.inf,
	min_bitrate=-np.inf,
	max_bitrate=np.inf,
	set_bitrate_log_base=2
):
	"""
	Extracting rate-quality information from the dataset generated by encoding uncompressed videos using multiple compression settings.
	Args:
		video_rq_points_info (dict): Estimations of a uncompressed video under different compression settings extracted from json file. (Default: None)
		quality_metric (str): Selected quality-metric. Options: ["psnr_y", "integer_motion2", "integer_motion","integer_adm2","integer_adm_scale0","integer_adm_scale1","integer_adm_scale2","integer_adm_scale3","float_ssim","integer_vif_scale0","integer_vif_scale1","integer_vif_scale2","integer_vif_scale3","float_ms_ssim","vmaf"] (Default: None)
		resolutions (list): Resolutions that needs to be considered while plotting RQ points. (Default: None)
		CRFs (any): CRFs that needs to be considered while plotting RQ points. (Default: None)
		QPs (list): QPs that needs to be considered while plotting RQ points. (Default: None)
		min_quality (float): Minimum quality to be considered for in output pairs/info. (Default: -np.inf)
		max_quality (float): Maximum quality to be considered for in output pairs/info. (Default: np.inf)
		min_bitrate (float): Minimum bitrate (in kbps) to be considered for in output pairs/info. (Default: -np.inf)
		max_bitrate (float): Maximum bitrate (in kbps) to be considered for in output pairs/info. (Default: np.inf)
		set_bitrate_log_base (float): Base value of logarithm that needs to applied on bitrate. If None, log Operation will not be applied (Default: None)
	Return:
		RQ_pairs (dict): Dictionary with resolution as keys containing (bitrate (in kbps), quality, I_frame_bitrate (in kbps), P_frame_bitrate (in kbps), B_frame_bitrate (in kbps), I_frame_AvgQP, P_frame_AvgQP, B_frame_AvgQP, rate_control_setting_value) points extracted from selected rate_control json file.
	"""

	# Assertions
	valid_rate_settings = ((QPs is not None) and (CRFs is None)) or ((QPs is None) and (CRFs is not None)) or ((QPs is None) and (CRFs is None))
	assert valid_rate_settings, "Provide valid rate-control settings i.e only one list of QPs or CRFs"
	assert quality_metric in ["psnr_y", "integer_motion2", "integer_motion","integer_adm2","integer_adm_scale0","integer_adm_scale1","integer_adm_scale2","integer_adm_scale3","float_ssim","integer_vif_scale0","integer_vif_scale1","integer_vif_scale2","integer_vif_scale3","float_ms_ssim","vmaf"], "Provide valid quality metric"
	
	# Rate-Control
	rate_control = CRFs if CRFs is not None else QPs

	# Output
	RQ_pairs = {}

	for res in resolutions:
		res_string = str(res[0])+"x"+str(res[1])
		RQ_pairs[res] = []

		if isinstance(rate_control, dict):
			rate_control_for_resolution = rate_control[res]
		else:
			rate_control_for_resolution = rate_control

		for rc in rate_control_for_resolution:
			rc_string = str(rc)

			# Bitrate (in bits) and Quality
			R = np.round(video_rq_points_info[res_string][rc_string]["bitrate"], decimals=8)
			Q = np.round(video_rq_points_info[res_string][rc_string][quality_metric], decimals=4)

			# Bitrate of I, P and B frames is in kbps
			I_R = np.round(video_rq_points_info[res_string][rc_string]["I_frame_Bitrate"] * 1000, decimals=8)
			P_R = np.round(video_rq_points_info[res_string][rc_string]["P_frame_Bitrate"] * 1000, decimals=8)
			B_R = np.round(video_rq_points_info[res_string][rc_string]["B_frame_Bitrate"] * 1000, decimals=8)

			# Avg.QPs of I, P and B frames
			I_QP = np.round(video_rq_points_info[res_string][rc_string]["I_frame_AvgQP"], decimals=4)
			P_QP = np.round(video_rq_points_info[res_string][rc_string]["P_frame_AvgQP"], decimals=4)
			B_QP = np.round(video_rq_points_info[res_string][rc_string]["B_frame_AvgQP"], decimals=4)

			if (R >= min_bitrate and R <= max_bitrate) and (Q >= min_quality and Q <= max_quality):
				if set_bitrate_log_base is not None:
					R = np.round(np.log10(R)/np.log10(set_bitrate_log_base), decimals=4)

					if I_R < 0:
						I_R = -1
					else:
						I_R = np.round(np.log10(I_R)/np.log10(set_bitrate_log_base), decimals=4)

					if P_R < 0:
						P_R = -1
					else:
						P_R = np.round(np.log10(P_R)/np.log10(set_bitrate_log_base), decimals=4)

					if B_R < 0:
						B_R = -1
					else:
						B_R = np.round(np.log10(B_R)/np.log10(set_bitrate_log_base), decimals=4)
				else:
					R = np.round(R/1000.0, decimals=4)

					I_R = np.round(I_R/1000.0, decimals=4)
					P_R = np.round(P_R/1000.0, decimals=4)
					B_R = np.round(B_R/1000.0, decimals=4)

				# Assertions
				assert not np.isnan(I_R).any(), "NaN values found in I_R"
				assert not np.isnan(P_R).any(), "NaN values found in P_R"
				assert not np.isnan(B_R).any(), "NaN values found in B_R"

				assert not np.isnan(I_QP).any(), "NaN values found in I_QP"
				assert not np.isnan(P_QP).any(), "NaN values found in P_QP"
				assert not np.isnan(B_QP).any(), "NaN values found in B_QP"

				# Appending to RQ_pairs
				RQ_pairs[res].append([R,Q, I_R,P_R,B_R, I_QP,P_QP,B_QP, rc])

		RQ_pairs[res].sort()
		RQ_pairs[res] = np.array(RQ_pairs[res])

	return RQ_pairs