# Main Function
# Importing Libraries
import numpy as np

import os, sys, warnings
warnings.filterwarnings('ignore')
from tqdm import tqdm
import joblib
import scripts.train as train
import scripts.test as test
import scripts.convex_hulls as convex_hulls
import scripts.bitrate_ladders as bitrate_ladders
import scripts.bd_metrics as bd_metrics
import scripts.closeness as closeness
import defaults

# Execute Function
def execute_training(
	Train_Video_Files:list,
	Valid_Video_Files:list,
	Test_Video_Files:list,
	save_results_folder:str
):
	"""
	Args:
		Train_Video_Files (list): List of video-files used for training.
		Valid_Video_Files (list): List of video-files used for validation.
		Test_Video_Files (list): List of video-files used for testing.
		save_results_folder (list): Folder name to in results folder to save results.
	"""
	## Assertions
	# """
	print ()
	print ("-"*25, "Assertions", "-"*25)
	print ()

	for video_file in tqdm(Train_Video_Files + Valid_Video_Files + Test_Video_Files):
		# Check for YUV file
		assert os.path.exists(
			os.path.join(defaults.source_dataset_path, video_file + ".yuv")
		), "For video-file: {}, YUV does not exist."

		# Check for RQ Points
		for codec in defaults.codec_preset_pairs.keys():
			for preset in defaults.codec_preset_pairs[codec]:
				assert os.path.exists(
					os.path.join(defaults.rq_points_dataset_path, codec, preset, video_file, "crfs.json")
				)
	# """


	## Features
	# Low-Level Features (Custom-Features always at the end so as to match code in 'dataset_evaluation_functions.py')
	features_names = []
	for features_subset in [defaults.glcm_features, defaults.tc_features, defaults.si_features, defaults.ti_features, defaults.cti_features, defaults.cf_features, defaults.ci_features, defaults.dct_features, list(defaults.bitrate_texture_features.keys())]:
		for f in features_subset:
			features_names.append(f)

	# VIF-Approach Number
	vif_approach_number = "9"


	## Training Regressors
	# """
	print ()
	print ("-"*25, "Training", "-"*25)
	print ()

	train.main(
		Train_Video_Files=Train_Video_Files,
		Valid_Video_Files=Valid_Video_Files,
		Test_Video_Files=Test_Video_Files,
		results_dir=save_results_folder,
		features_names=features_names,
		vif_approach_number=vif_approach_number
	)
	# """
	


# Execute Function
def execute(
	Train_Video_Files:list,
	Valid_Video_Files:list,
	Test_Video_Files:list,
	save_results_folder:str
):
	"""
	Args:
		Train_Video_Files (list): List of video-files used for training.
		Valid_Video_Files (list): List of video-files used for validation.
		Test_Video_Files (list): List of video-files used for testing.
		save_results_folder (list): Folder name to in results folder to save results.
	"""
	## Features
	# Low-Level Features (Custom-Features always at the end so as to match code in 'dataset_evaluation_functions.py')
	features_names = []
	for features_subset in [defaults.glcm_features, defaults.tc_features, defaults.si_features, defaults.ti_features, defaults.cti_features, defaults.cf_features, defaults.ci_features, defaults.dct_features, list(defaults.bitrate_texture_features.keys())]:
		for f in features_subset:
			features_names.append(f)

	# VIF-Approach Number
	vif_approach_number = "9"


	## Convex-Hulls
	# """
	print ()
	print ("-"*25, "Convex-Hulls", "-"*25)
	print ()

	convex_hulls.main(
		Test_Video_Files=Test_Video_Files,
		results_dir=save_results_folder,
	)
	# """


	## Predict Bitrate Ladders
	# """
	print ()
	print ("-"*25, "Bitrate Ladders", "-"*25)
	print ()

	bitrate_ladders.main(
		Test_Video_Files=Test_Video_Files,
		results_dir=save_results_folder,
		features_names=features_names,
		vif_approach_number=vif_approach_number
	)
	# """


	## Calculate BD-Metrics
	# """
	print ()
	print ("-"*25, "BD-Metrics", "-"*25)
	print ()

	bd_metrics.main(
		Test_Video_Files=Test_Video_Files,
		results_dir=save_results_folder,
	)
	# """


	## Calculate Closeness
	# """
	print ()
	print ("-"*25, "Closeness", "-"*25)
	print ()

	closeness.main(
		Test_Video_Files=Test_Video_Files,
		results_dir=save_results_folder,
	)
	# """



# Execute Function
def testing(
	Test_Video_Files:list,
	save_results_folder:str
):
	"""
	Args:
		Test_Video_Files (list): List of video-files used for testing.
		save_results_folder (list): Folder name to in results folder to save results.
	"""
	## Assertions
	# """
	print ()
	print ("-"*25, "Assertions", "-"*25)
	print ()

	for video_file in tqdm(Test_Video_Files):
		# Check for YUV file
		assert os.path.exists(
			os.path.join(defaults.source_dataset_path, video_file + ".yuv")
		), "For video-file: {}, YUV does not exist."

		# Check for RQ Points
		for codec in defaults.codec_preset_pairs.keys():
			for preset in defaults.codec_preset_pairs[codec]:
				assert os.path.exists(
					os.path.join(defaults.rq_points_dataset_path, codec, preset, video_file, "crfs.json")
				)
	# """


	## Features
	# Low-Level Features (Custom-Features always at the end so as to match code in 'dataset_evaluation_functions.py')
	features_names = []
	for features_subset in [defaults.glcm_features, defaults.tc_features, defaults.si_features, defaults.ti_features, defaults.cti_features, defaults.cf_features, defaults.ci_features, defaults.dct_features, list(defaults.bitrate_texture_features.keys())]:
		for f in features_subset:
			features_names.append(f)

	# VIF-Approach Number
	vif_approach_number = "9"


	## Testing Regressors
	# """
	print ()
	print ("-"*25, "Testing", "-"*25)
	print ()

	test.main(
		Test_Video_Files=Test_Video_Files,
		results_dir=save_results_folder,
		features_names=features_names,
		vif_approach_number=vif_approach_number
	)
	# """



if __name__ == "__main__":
	# Sequential
	# """
	for i in range(len(defaults.Datasets)):
		print ("-"*100)
		print ("Split-{}".format(i))
		print ("-"*100)
		execute_training(
			Train_Video_Files=defaults.Datasets["Split-{}".format(i)]["Train_Video_Files"],
			Valid_Video_Files=defaults.Datasets["Split-{}".format(i)]["Valid_Video_Files"],
			Test_Video_Files=defaults.Datasets["Split-{}".format(i)]["Test_Video_Files"],
			save_results_folder="results/main/libx265/Split-{}".format(i)
		)
	# """

	# Parallel
	# """
	joblib.Parallel(n_jobs=5)(
		joblib.delayed(execute)(
			Train_Video_Files=defaults.Datasets["Split-{}".format(i)]["Train_Video_Files"],
			Valid_Video_Files=defaults.Datasets["Split-{}".format(i)]["Valid_Video_Files"],
			Test_Video_Files=defaults.Datasets["Split-{}".format(i)]["Test_Video_Files"],
			save_results_folder="results/main/libx265/Split-{}".format(i)
		) for i in range(len(defaults.Datasets))
	)
	# """

	# Sequential
	"""
	for i in range(len(defaults.Datasets)):
		print ("-"*100)
		print ("Split-{}".format(i))
		print ("-"*100)
		testing(
			Test_Video_Files=defaults.Datasets["Split-{}".format(i)]["Test_Video_Files"],
			save_results_folder="results/main/libx265/Split-{}".format(i)
		)
	"""