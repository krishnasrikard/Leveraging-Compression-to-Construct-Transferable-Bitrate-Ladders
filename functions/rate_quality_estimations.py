"""
Functions to estimate RQ points for different compression settings.
"""
# Importing Libraries
import numpy as np
import pandas as pd

import os, sys, warnings
import pickle
import subprocess, shlex
import time
from tqdm import tqdm
warnings.filterwarnings("ignore")
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from typing import Type, Callable, Tuple, Optional, Set, List, Union
import functions.software_commands as software_commands
import functions.extract_functions as extract_functions
import functions.IO_functions as IO_functions


# Encoding and Quality Estimation Pipeline
def quality_estimation_pipeline(
	input_yuv_path:str=None,
	input_resolution:Tuple=None,
	input_framerate:float=None,
	input_pixelformat:str=None,
	output_resolution:str=None,
	codec:str=None,
	preset:str=None,
	QP:int=None,
	CRF:int=None,
	output_video_path:str=None,
	reference_video_path:str=None,
	temp_path:str=None,
	vmaf_resolution:tuple=(3840,2160),
	quality_metrics:dict={"PSNR":True, "SSIM":True, "MS_SSIM":True},
	num_threads:int=8
):
	"""
	Compress a YUV video and estimate its filesize, bitrate and quality
	Args:
		input_yuv_path (str): YUV file as input path. (Default: None)
		input_resolution (Tuple): Resolution of YUV file. (Default: None)
		input_framerate (float): Frame-rate of YUV file. (Default: None)
		input_pixelformat (str): Pixel-Format of YUV file. (Default: None)
		output_resolution (tuple): The resolution of the output. (Default: None)
		codec (str): The encoder/decoder video codec. (Default: None)
		preset (str): A preset of the encoder. (Default: None)
		QP (int): The quantization-paramter (QP) setting. (Default: None)
		CRF (int): The control-rate factor. (Default: None)
		output_video_path (str): The path to the output file. (Default: None)
		reference_video_path (str): The path to the reference video file. (Default: None)
		temp_path (str): A temporary path for managing files. (Default: None)
		vmaf_resolution (tuple): The resolution at which VMAF is calculated. (Default: (3840,2160))
		quality_metrics (dict): A dictionary containing what quality-metrics need to be computed. All the keys i.e ["PSNR", "SSIM", "MS_SSIM"] are mandatory. (Default: {"PSNR":True, "SSIM":True, "MS_SSIM":True})
		num_threads (int): No.of threads used to run ffmpeg commands. Generally no.of threads are set between 4-8 for ffmpeg. (Default: 8)
	Returns:
		(list): List of file_size, bitrate, PSNR, SSIM, MS_SSIM and VMAF values.
	"""

	# Assertions
	assert temp_path is not None, "A temporary video path must be provided to store files during bitrate and quality estimations."

	# Extension
	if (codec == "libx264") or (codec == "libx265") or (codec == "libsvtav1"):
		ext = ".mp4"
	elif (codec == "libaom-av1"):
		ext = ".mkv"
	elif (codec == "libvpx-vp9"):
		ext = ".webm"
	else:
		ext = ".mp4"


	# Set output video path
	if output_video_path is None:
		set_output_video_path = temp_path + "/output_video{}".format(ext)
	else:
		set_output_video_path = output_video_path


	# Downsampling and Compression
	print ("\n" + "-"*25 + "\n" + "Down-Scaling and Compression" + "\n" + "-"*25, flush=True)
	cmd = software_commands.compression_command(
		raw_video=True,
		input_resolution=input_resolution,
		frame_rate=input_framerate,
		pixel_format=input_pixelformat,
		input_video_path=input_yuv_path,
		output_resolution=output_resolution,
		scaling_algo="lanczos",
		video_codec=codec,
		preset=preset,
		QP=QP,
		CRF=CRF,
		output_video_path=set_output_video_path,
		num_threads=num_threads
	)

	compression_logs = subprocess.getoutput(cmd)
	downscaling_compression_time = extract_functions.extract_execution_time(compression_logs)
	print ("Downscaling and Compression Time =", downscaling_compression_time, flush=True)

	# Extracting Output Filesize
	output_filesize = os.path.getsize(set_output_video_path)

	# Extracting Compression Logs
	if codec != "libsvtav1":
		compression_output_logs = extract_functions.extract_compression_logs(compression_logs, codec)
	else:
		compression_output_logs = extract_functions.process_SVTAV1_frame_info(
			compressed_video_path=set_output_video_path,
			frame_rate=input_framerate,

			frame_qp_info_path="/home/krishna/Nebula/krishna/Leveraging-Compression-to-Construct-Transferable-Bitrate-Ladders/rq_points_dataset/temp_files/QPs_{}.txt".format(preset),

			frame_info_csv="/home/krishna/Nebula/krishna/Leveraging-Compression-to-Construct-Transferable-Bitrate-Ladders/rq_points_dataset/temp_files/Logs_{}.csv".format(preset)
		)


	# Calculating bitrate and quality for a compressed video/non-reference video
	if reference_video_path is not None:
		# Bitrate Estimation
		print ("\n" + "-"*25 + "\n" + "Bitrate-Estimation" + "\n" + "-"*25, flush=True)
		cmd = software_commands.bitrate_estimation_command(
			input_video_path=set_output_video_path,
			num_threads=num_threads
		)

		output = subprocess.getoutput(cmd)
		estimated_bitrate = extract_functions.extract_bitrate(output)


		# Logging
		print ("Bitrate (in kbps) =", estimated_bitrate/1000.0)


		# Quality Estimation
		print ("\n" + "-"*25 + "\n" + "Quality-Estimation" + "\n" + "-"*25, flush=True)
		cmd = software_commands.quality_estimation_command(
			raw_video=True,
			input_resolution=input_resolution,
			frame_rate=input_framerate,
			pixel_format=input_pixelformat,
			reference_video_path=reference_video_path,
			distored_video_path=set_output_video_path,
			vmaf_resolution=vmaf_resolution,
			scaling_algo="lanczos",
			PSNR=quality_metrics["PSNR"],
			SSIM=quality_metrics["SSIM"],
			MS_SSIM=quality_metrics["MS_SSIM"],
			logfile_path=temp_path + "/logfile.log",
			num_threads=num_threads
		)

		output = subprocess.getoutput(cmd)
		upscaling_quality_estimation_time = extract_functions.extract_execution_time(output)

		print ("Quality Estimation Time =", upscaling_quality_estimation_time, flush=True)
		Quality_Estimations = extract_functions.extract_qualityestimates(temp_path + "/logfile.log")
	else:
		estimated_bitrate = -1
		Quality_Estimations = {}
		upscaling_quality_estimation_time = -1


	# Remove temporary files
	if output_video_path is None:
		os.remove(set_output_video_path)
	if reference_video_path is not None:
		os.remove(temp_path + "/logfile.log")

	return {**{"downscaling_compression_time": downscaling_compression_time, "quality_estimation_time":upscaling_quality_estimation_time}, **{"filesize":output_filesize, "bitrate":estimated_bitrate}, **compression_output_logs,  **Quality_Estimations}



# Estimation Rate-Quality points for various (resolution,bitrate/crf/qp) pairs.
def estimate_rate_quality_points(
	input_yuv_path:str=None,
	input_resolution:Tuple=None,
	input_framerate:float=None,
	input_pixelformat:str=None,
	output_resolutions:list=None,
	codec:str=None,
	preset:str=None,
	QPs:list=None,
	CRFs:list=None,
	rq_points_output_path:str=None,
	compressed_videos_output_path:str=None,
	temp_path:str=None,
	vmaf_resolution:tuple=(3840,2160),
	quality_metrics:dict={"PSNR":True, "SSIM":True, "MS_SSIM":True},
	num_threads:int=8
):
	"""
	Args:
		input_yuv_path (str): YUV file as input path. (Default: None)
		input_resolution (Tuple): Resolution of YUV file. (Default: None)
		input_framerate (float): Frame-rate of YUV file. (Default: None)
		input_pixelformat (str): Pixel-Format of YUV file. (Default: None)
		output_resolutions (list): List of output resolutions. (Default: None)
		codec (str): The video codec. (Default: None)
		preset (str): The preset setting of the encoder. (Default: None)
		QPs (list): List of QP values. (Default: None)
		CRFs (list): List of CRF values. (Default: None)
		rq_points_output_path (str): Path to store resolution, rate, quality information extracted. (Default: None)
		compressed_videos_output_path (str): Path to store compressed videos. The compressed videos will not be saved if path is set to None. (Default: None)
		temp_path (str): A temporary path for managing files. (Default: None)
		vmaf_resolution (tuple): The resolution at which VMAF is calculated. (Default: (3840,2160))
		quality_metrics (dict): A dictionary containing what quality-metrics need to be computed. All the keys i.e ["PSNR", "SSIM", "MS_SSIM"] are mandatory. (Default: {"PSNR":True, "SSIM":True, "MS_SSIM":True})
		num_threads (int): No.of threads used to run ffmpeg commands. Generally no.of threads are set between 4-8 for ffmpeg. (Default: 8)
	"""

	# Assertions
	assert os.path.exists(rq_points_output_path), "Provide a path to store resolution, rate, quality information extracted from different compression settings."
	assert os.path.exists(temp_path), "Provide path for temporary file management."
	assert sorted(quality_metrics.keys()) == sorted(["PSNR", "SSIM", "MS_SSIM"]), "Quality Metrics are not provided correctly."


	# Filename
	input_filename = os.path.splitext(os.path.basename(input_yuv_path))[0]


	# Extension
	if (codec == "libx264") or (codec == "libx265") or (codec == "libsvtav1"):
		ext = ".mp4"
	elif (codec == "libaom-av1"):
		ext = ".mkv"
	elif (codec == "libvpx-vp9"):
		ext = ".webm"
	else:
		ext = ".mp4"


	# Creating folders for storing rate-quality information encoded using a codec and preset
	os.makedirs(os.path.join(rq_points_output_path, codec, preset), exist_ok=True)


	# Creating folders for storing compressed videos encoded using a codec and preset
	if compressed_videos_output_path is not None:	
		os.makedirs(os.path.join(compressed_videos_output_path, codec, preset), exist_ok=True)


	# Creating a folder to save rate-quality points with different settings
	rq_points_folder_path = os.path.join(rq_points_output_path, codec, preset, input_filename)
	if os.path.exists(rq_points_folder_path) == False:
		os.mkdir(rq_points_folder_path)


	# Creating a folder to save compressed videos with different settings
	if compressed_videos_output_path is not None:
		compressed_videos_folder_path = os.path.join(compressed_videos_output_path, codec, preset, input_filename)
		if os.path.exists(compressed_videos_folder_path) == False:
			os.mkdir(compressed_videos_folder_path)


			
	# Rate-Control: CRF or Constant-Quality Encoding
	if CRFs is not None:
		rq_points_path = os.path.join(rq_points_folder_path, "crfs.json")
		data = IO_functions.read_create_jsonfile(rq_points_path)

		# For each resolution
		for resolution in output_resolutions:
			resolution_string = str(resolution[0]) + "x" + str(resolution[1])
			if resolution_string not in data.keys():
				data[resolution_string] = {}

			# For each CRF
			for crf in CRFs:
				crf_string = str(crf)
				if compressed_videos_output_path is not None:
					compressed_video_path = os.path.join(compressed_videos_folder_path, "_".join(["res=" + resolution_string, "crf=" + crf_string]) + ext)
				else:
					compressed_video_path = None

				if (crf_string not in data[resolution_string].keys()) or (compressed_video_path is not None and os.path.exists(compressed_video_path) == False):
					# Logging
					print ("\nEstimating quality of video compressed with settings:\ncodec={}\npreset={}\nresolution={}\ncrf={}\n".format(codec, preset, resolution, crf), flush=True)

					# Calculate
					Results = quality_estimation_pipeline(
						input_yuv_path=input_yuv_path,
						input_resolution=input_resolution,
						input_framerate=input_framerate,
						input_pixelformat=input_pixelformat,
						output_resolution=resolution,
						codec=codec,
						preset=preset,
						QP=None,
						CRF=crf,
						output_video_path=compressed_video_path,
						reference_video_path=input_yuv_path,
						temp_path=temp_path,
						vmaf_resolution=vmaf_resolution,
						quality_metrics=quality_metrics,
						num_threads=num_threads
					)
					data[resolution_string][crf_string] = Results
				else:
					print ("\nQuality estimations exists for video compressed with settings:\ncodec={}\npreset={}\nresolution={}\ncrf={}\n".format(codec, preset, resolution, crf), flush=True)

			IO_functions.save_jsonfile(rq_points_path, data)
			

	# Rate-Control: QP or Constant-QP Encoding
	elif QPs is not None:
		rq_points_path = os.path.join(rq_points_folder_path, "qps.json")
		data = IO_functions.read_create_jsonfile(rq_points_path)

		# For each resolution
		for resolution in output_resolutions:
			resolution_string = str(resolution[0]) + "x" + str(resolution[1])
			if resolution_string not in data.keys():
				data[resolution_string] = {}

			# For each QP
			for qp in QPs:
				qp_string = str(qp)
				if compressed_videos_output_path is not None:
					compressed_video_path = os.path.join(compressed_videos_folder_path, "_".join(["res=" + resolution_string, "qp=" + qp_string]) + ext)
				else:
					compressed_video_path = None

				if (qp_string not in data[resolution_string].keys()) or (compressed_video_path is not None and os.path.exists(compressed_video_path) == False):
					# Logging
					print ("\nEstimating quality of video compressed with settings:\ncodec={}\npreset={}\nresolution={}\nqp={}\n".format(codec, preset, resolution, qp), flush=True)

					# Calculate
					Results = quality_estimation_pipeline(
						input_yuv_path=input_yuv_path,
						input_resolution=input_resolution,
						input_framerate=input_framerate,
						input_pixelformat=input_pixelformat,
						output_resolution=resolution,
						codec=codec,
						preset=preset,
						QP=qp,
						CRF=None,
						output_video_path=compressed_video_path,
						reference_video_path=input_yuv_path,
						temp_path=temp_path,
						vmaf_resolution=vmaf_resolution,
						quality_metrics=quality_metrics,
						num_threads=num_threads
					)
					data[resolution_string][qp_string] = Results
				else:
					print ("\nQuality estimations exists for video compressed with settings:\ncodec={}\npreset={}\nresolution={}\nqp={}\n".format(codec, preset, resolution, qp), flush=True)

			IO_functions.save_jsonfile(rq_points_path, data)


	else:
		assert False, "Both QPs and CRFs are None"