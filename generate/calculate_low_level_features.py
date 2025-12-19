"""
Calculating Low-Level features of uncompressed videos
"""
# Importing Libraires
import numpy as np
import pandas as pd
import cv2

import os, sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import time
import argparse
import features.GLCM as GLCM
import features.TC as TC
import features.SI as SI
import features.TI as TI
import features.CF as CF
import features.CI as CI
import features.CTI as CTI
import features.Texture_DCT as Texture_DCT
import functions.IO_functions as IO_functions
import defaults


# Fexture Extraction Modules
cf = CF.CF_Features()
ci = CI.CI_Features(rgb=True, WR=5)
cti = CTI.CTI_Features(rgb=True)
glcm = GLCM.GLCM_Features(descriptors=["contrast","correlation","energy","homogeneity"],angles=[0],distance=1,levels=256,block_size=(64,64),rgb=True)
tc = TC.TC_Features(rgb=True)
si = SI.SI_Features(rgb=True)
ti = TI.TI_Features(rgb=True)
texture_dct_features = Texture_DCT.Texture_DCT_Features(block_size=(32,32),rgb=True)



# Resize Video
def resize_video(
	original_video:np.array,
	resize_dimensions:any
):
	"""
	Args:
		original_video (np.array): Original numpy video.
		resize_dimensions (any): Resize video to given dimensions (width, height).
	"""
	# Dimension
	width = original_video.shape[2]
	height = original_video.shape[1]

	# Skip if original video has same dimensions as resize dimensions
	if resize_dimensions[0] == width and resize_dimensions[1] == height:
		# Logging
		print ("Original Video Dimensions = Resize Dimensions = {}".format(resize_dimensions))

		return original_video
	
	# Resizing
	video = []
	for i in range(original_video.shape[0]):
		video.append(
			cv2.resize(original_video[i], dsize=(resize_dimensions[0], resize_dimensions[1]), interpolation=cv2.INTER_LANCZOS4)
		)

	video = np.array(video)

	# Assertions
	assert (video.dtype == np.uint8) and (np.min(video) >= 0 and np.max(video) <= 255), "Input Image/Videos should of type uint8 and should have range [0,255]."

	# Logging
	print ("Original Video Dimensions = {}".format((width, height)))
	print ("Resized Video Dimensions = {}".format((video.shape[2], video.shape[1])))

	return video



# Feature Extraction from Video
class generate_low_level_features():
	def __init__(self,
		video:np.array
	):
		# Video
		self.video = video

		# Features
		self.features = {}
		self.per_frame_features = {}

		# Generate
		self.generate_glcm_features()
		self.generate_tc_features()
		self.generate_si_features()
		self.generate_ti_features()
		self.generate_cti_features()
		self.generate_cf_features()
		self.generate_ci_features()
		self.generate_texture_dct_features()


	def generate_per_frame_low_level_features(self):
		return self.per_frame_features
	

	def generate_low_level_features(self):
		return self.features


	def generate_glcm_features(self):
		# GLCM Features
		print ("GLCM Features:\n",flush=True)

		# Stats
		spatial_stats = np.sort(["mean", "std"])
		temporal_stats = np.sort(["mean", "std", "skew", "kurt"])
		stats = []
		for t in temporal_stats:
			for s in spatial_stats:
				stats.append([t,s])

		# Feature Names
		per_frame_features_names = []
		for f in ["GLCM_contrast", "GLCM_correlation", "GLCM_energy", "GLCM_homogeneity"]:
			for s in spatial_stats:
				per_frame_features_names.append(f+"_"+s)

		features_names = []
		for f in ["GLCM_contrast", "GLCM_correlation", "GLCM_energy", "GLCM_homogeneity"]:
			for t in temporal_stats:
				for s in spatial_stats:
					features_names.append(t+"_"+f+"_"+s)
		
		# Compute
		time_instant = time.time()
		per_frame_features, features = glcm.compute_video_glcm_features(video=np.copy(self.video), stats=stats)
		compute_time = time.time() - time_instant

		# Assertions
		assert len(per_frame_features_names) == per_frame_features.shape[1], "Dimensions of per-frame features do not match."
		assert len(features_names) == features.shape[0], "Dimensions of per-frame features do not match."

		for i,name in enumerate(per_frame_features_names):
			self.per_frame_features[name] = per_frame_features[:,i]

		for i,name in enumerate(features_names):
			self.features[name] = features[i]
		self.features["GLCM_compute_time"] = np.round(compute_time, decimals=6)


	def generate_tc_features(self):
		# TC Features
		print ("TC Features:\n",flush=True)

		# Stats
		spatial_stats = np.sort(["mean", "std", "skew", "kurt"])
		temporal_stats = np.sort(["mean", "std"])
		stats = []
		for t in temporal_stats:
			for s in spatial_stats:
				stats.append([t,s])

		# Feature Names
		per_frame_features_names = []
		for f in ["TC"]:
			for s in spatial_stats:
				per_frame_features_names.append(f+"_"+s)

		features_names = []
		for f in ["TC"]:
			for t in temporal_stats:
				for s in spatial_stats:
					features_names.append(t+"_"+f+"_"+s)

		# Compute
		time_instant = time.time()
		per_frame_features, features = tc.compute_video_tc_features(video=np.copy(self.video), stats=stats)
		compute_time = time.time() - time_instant

		# Assertions
		assert len(per_frame_features_names) == per_frame_features.shape[1], "Dimensions of per-frame features do not match."
		assert len(features_names) == features.shape[0], "Dimensions of per-frame features do not match."

		for i,name in enumerate(per_frame_features_names):
			self.per_frame_features[name] = per_frame_features[:,i]

		for i,name in enumerate(features_names):
			self.features[name] = features[i]
		self.features["TC_compute_time"] = np.round(compute_time, decimals=6)

	
	def generate_si_features(self):
		# SI Features
		print ("SI Features:\n",flush=True)

		# Stats
		spatial_stats = np.sort(["mean", "std"])
		temporal_stats = np.sort(["mean", "std", "skew", "kurt"])
		stats = []
		for t in temporal_stats:
			for s in spatial_stats:
				stats.append([t,s])

		# Feature Names
		per_frame_features_names = []
		for f in ["SI"]:
			for s in spatial_stats:
				per_frame_features_names.append(f+"_"+s)

		features_names = []
		for f in ["SI"]:
			for t in temporal_stats:
				for s in spatial_stats:
					features_names.append(t+"_"+f+"_"+s)

		# Compute
		time_instant = time.time()
		per_frame_features, features = si.compute_video_spatial_information(video=np.copy(self.video), stats=stats)
		compute_time = time.time() - time_instant

		# Assertions
		assert len(per_frame_features_names) == per_frame_features.shape[1], "Dimensions of per-frame features do not match."
		assert len(features_names) == features.shape[0], "Dimensions of per-frame features do not match."

		for i,name in enumerate(per_frame_features_names):
			self.per_frame_features[name] = per_frame_features[:,i]

		for i,name in enumerate(features_names):
			self.features[name] = features[i]
		self.features["SI_compute_time"] = np.round(compute_time, decimals=6)


	def generate_ti_features(self):
		# TI Features
		print ("TI Features:\n",flush=True)

		# Stats
		spatial_stats = np.sort(["mean", "std"])
		temporal_stats = np.sort(["mean", "std", "skew", "kurt"])
		stats = []
		for t in temporal_stats:
			for s in spatial_stats:
				stats.append([t,s])

		# Feature Names
		per_frame_features_names = []
		for f in ["TI"]:
			for s in spatial_stats:
				per_frame_features_names.append(f+"_"+s)

		features_names = []
		for f in ["TI"]:
			for t in temporal_stats:
				for s in spatial_stats:
					features_names.append(t+"_"+f+"_"+s)

		# Compute
		time_instant = time.time()
		per_frame_features, features = ti.compute_video_temporal_information(video=np.copy(self.video), stats=stats)
		compute_time = time.time() - time_instant

		# Assertions
		assert len(per_frame_features_names) == per_frame_features.shape[1], "Dimensions of per-frame features do not match."
		assert len(features_names) == features.shape[0], "Dimensions of per-frame features do not match."

		for i,name in enumerate(per_frame_features_names):
			self.per_frame_features[name] = per_frame_features[:,i]

		for i,name in enumerate(features_names):
			self.features[name] = features[i]
		self.features["TI_compute_time"] = np.round(compute_time, decimals=6)


	def generate_cti_features(self):
		# CTI Features
		print ("CTI Features:\n",flush=True)

		# Stats
		spatial_stats = np.sort(["mean", "std"])
		temporal_stats = np.sort(["mean", "std", "skew", "kurt"])
		stats = []
		for t in temporal_stats:
			for s in spatial_stats:
				stats.append([t,s])

		# Feature Names
		per_frame_features_names = []
		for f in ["CTI"]:
			for s in spatial_stats:
				per_frame_features_names.append(f+"_"+s)

		features_names = []
		for f in ["CTI"]:
			for t in temporal_stats:
				for s in spatial_stats:
					features_names.append(t+"_"+f+"_"+s)

		# Compute
		time_instant = time.time()
		per_frame_features, features = cti.compute_video_contrast_information(video=np.copy(self.video), stats=stats)
		compute_time = time.time() - time_instant

		# Assertions
		assert len(per_frame_features_names) == per_frame_features.shape[1], "Dimensions of per-frame features do not match."
		assert len(features_names) == features.shape[0], "Dimensions of per-frame features do not match."

		for i,name in enumerate(per_frame_features_names):
			self.per_frame_features[name] = per_frame_features[:,i]

		for i,name in enumerate(features_names):
			self.features[name] = features[i]
		self.features["CTI_compute_time"] = np.round(compute_time, decimals=6)


	def generate_cf_features(self):
		# CF Features
		print ("CF Features:\n",flush=True)

		# Stats
		temporal_stats = np.sort(["mean", "std", "skew", "kurt"])
		stats = temporal_stats

		# Feature Names
		per_frame_features_names = ["CF"]

		features_names = []
		for f in ["CF"]:
			for t in temporal_stats:
				features_names.append(t+"_"+f)

		# Compute
		time_instant = time.time()
		per_frame_features, features = cf.compute_video_colorfulness(video=np.copy(self.video), stats=stats)
		compute_time = time.time() - time_instant

		# Assertions
		assert len(features_names) == features.shape[0], "Dimensions of per-frame features do not match."

		self.per_frame_features[per_frame_features_names[0]] = per_frame_features

		for i,name in enumerate(features_names):
			self.features[name] = features[i]
		self.features["CF_compute_time"] = np.round(compute_time, decimals=6)


	def generate_ci_features(self):
		# CI Features
		print ("CI Features:\n",flush=True)

		# Stats
		spatial_stats = np.sort(["mean", "std"])
		temporal_stats = np.sort(["mean", "std", "skew", "kurt"])
		stats = []
		for t in temporal_stats:
			for s in spatial_stats:
				stats.append([t,s])

		# Feature Names
		per_frame_features_names = []
		for f in ["CI_U"]:
			for s in spatial_stats:
				per_frame_features_names.append(f+"_"+s)

		features_names = []
		for f in ["CI_U"]:
			for t in temporal_stats:
				for s in spatial_stats:
					features_names.append(t+"_"+f+"_"+s)

		# Compute
		time_instant = time.time()
		per_frame_features, features = ci.compute_video_chroma_information(video=np.copy(self.video), component="U", stats=stats)
		compute_time = time.time() - time_instant

		# Assertions
		assert len(per_frame_features_names) == per_frame_features.shape[1], "Dimensions of per-frame features do not match."
		assert len(features_names) == features.shape[0], "Dimensions of per-frame features do not match."

		for i,name in enumerate(per_frame_features_names):
			self.per_frame_features[name] = per_frame_features[:,i]

		for i,name in enumerate(features_names):
			self.features[name] = features[i]
		self.features["CI_U_compute_time"] = np.round(compute_time, decimals=6)

		# Feature Names
		per_frame_features_names = []
		for f in ["CI_V"]:
			for s in spatial_stats:
				per_frame_features_names.append(f+"_"+s)

		features_names = []
		for f in ["CI_V"]:
			for t in temporal_stats:
				for s in spatial_stats:
					features_names.append(t+"_"+f+"_"+s)

		# Compute
		time_instant = time.time()
		per_frame_features, features = ci.compute_video_chroma_information(video=np.copy(self.video), component="V", stats=stats)
		compute_time = time.time() - time_instant

		# Assertions
		assert len(per_frame_features_names) == per_frame_features.shape[1], "Dimensions of per-frame features do not match."
		assert len(features_names) == features.shape[0], "Dimensions of per-frame features do not match."

		for i,name in enumerate(per_frame_features_names):
			self.per_frame_features[name] = per_frame_features[:,i]

		for i,name in enumerate(features_names):
			self.features[name] = features[i]
		self.features["CI_V_compute_time"] = np.round(compute_time, decimals=6)
		

	def generate_texture_dct_features(self):
		# Texture-DCT Features
		print ("Texture-DCT Features:\n",flush=True)

		# Feature Names
		per_frame_features_names = ["E_Y", "h_Y", "L_Y"]
		features_names = ["mean_E_Y", "mean_h_Y", "mean_L_Y"]

		# Compute
		time_instant = time.time()
		per_frame_features, features = texture_dct_features.compute_video_features(video=np.copy(self.video), component="Y")
		compute_time = time.time() - time_instant

		# Assertions
		assert len(per_frame_features_names) == per_frame_features.shape[1], "Dimensions of per-frame features do not match."
		assert len(features_names) == features.shape[0], "Dimensions of per-frame features do not match."

		for i,name in enumerate(per_frame_features_names):
			self.per_frame_features[name] = per_frame_features[:,i]

		for i,name in enumerate(features_names):
			self.features[name] = features[i]
		self.features["EhL_Y_compute_time"] = np.round(compute_time, decimals=6)


		# Feature Names
		per_frame_features_names = ["E_U", "h_U", "L_U"]
		features_names = ["mean_E_U", "mean_h_U", "mean_L_U"]

		# Compute
		time_instant = time.time()
		per_frame_features, features = texture_dct_features.compute_video_features(np.copy(self.video), component="U")
		compute_time = time.time() - time_instant

		# Assertions
		assert len(per_frame_features_names) == per_frame_features.shape[1], "Dimensions of per-frame features do not match."
		assert len(features_names) == features.shape[0], "Dimensions of per-frame features do not match."

		for i,name in enumerate(per_frame_features_names):
			self.per_frame_features[name] = per_frame_features[:,i]

		for i,name in enumerate(features_names):
			self.features[name] = features[i]
		self.features["EhL_U_compute_time"] = np.round(compute_time, decimals=6)


		# Feature Names
		per_frame_features_names = ["E_V", "h_V", "L_V"]
		features_names = ["mean_E_V", "mean_h_V", "mean_L_V"]

		# Compute
		time_instant = time.time()
		per_frame_features, features = texture_dct_features.compute_video_features(video=np.copy(self.video), component="V")
		compute_time = time.time() - time_instant

		# Assertions
		assert len(per_frame_features_names) == per_frame_features.shape[1], "Dimensions of per-frame features do not match."
		assert len(features_names) == features.shape[0], "Dimensions of per-frame features do not match."

		for i,name in enumerate(per_frame_features_names):
			self.per_frame_features[name] = per_frame_features[:,i]

		for i,name in enumerate(features_names):
			self.features[name] = features[i]
		self.features["EhL_V_compute_time"] = np.round(compute_time, decimals=6)



def main(args):	
	# Path Assertions
	assert os.path.exists(args.raw_videos_path), "Invalid path to raw videos"
	assert os.path.exists(args.features_information_path), "Invalid path to save features"

	# Filenames
	video_filenames = os.listdir(args.raw_videos_path)

	# Video Properties
	df = pd.read_csv(defaults.source_dataset_info_path)
	df = df.set_index("filename").to_dict(orient="index")
	
	# Save Path
	save_path = os.path.join(args.features_information_path, "uncompressed_low_level_features_info.csv")
	Feature_Info = pd.DataFrame()

	# For each video-file
	for video_file in video_filenames:
		# File and Properties
		filename = os.path.splitext(os.path.basename(video_file))[0]
		video_properties = df[filename]

		# Logging
		print ("-"*75 + "\n" + filename + "\n" + "-"*75, flush=True)

		# Reading Video 
		file_path = os.path.join(args.raw_videos_path, video_file)
		yuv_reader = IO_functions.YUV_Reader(
			filepath=file_path,
			width=video_properties["width"],
			height=video_properties["height"],
			yuv_type=video_properties["pix_fmt"]
		)
		video = yuv_reader.get_RGB_video()

		# Resize Video
		video = resize_video(
			original_video=np.copy(video),
			resize_dimensions=(3840,2160)
		)

		# Calculating Low-Level Features
		F = generate_low_level_features(video)
		per_frame_data = F.generate_per_frame_low_level_features()
		data = F.generate_low_level_features()

		# Saving computed features
		data = pd.DataFrame([{
			**{
				"filename": filename, 
				"resolution":"{}x{}".format(video.shape[2], video.shape[1]), 
				"framerate": video_properties["avg_frame_rate"],
				"pixel_format": video_properties["pix_fmt"],
				"num_frames":video.shape[0]
			}, 
			**data
		}])
		Feature_Info = pd.concat([Feature_Info, data], ignore_index=True)

		np.save(os.path.join(args.features_information_path, filename+".npy"), per_frame_data)
		Feature_Info.to_csv(save_path, index=False)

	Feature_Info.to_csv(save_path, index=False)



# Calling Main function
if __name__ == '__main__':
	root_dir = os.path.dirname(os.path.realpath(__file__))

	# Get Arguments
	parser = argparse.ArgumentParser(description='Calculating Low-Level features on source videos')

	# Dataset Paths
	parser.add_argument(
		'--raw_videos_path', 
		default=defaults.source_dataset_path, 
		help='Path to dataset.'
	)
	parser.add_argument(
		'--features_information_path', 
		default=defaults.low_level_features_path, 
		help='Path to information of features extracted from uncompressed videos.'
	)
	
	# Main Path
	parser.add_argument(
		'--main_path', 
		default=os.path.join(defaults.main_working_dir, 'generate'), 
		type=str, 
		help='Path to main folder'
	)

	# Parse Arguments
	args = parser.parse_args()

	main(args)