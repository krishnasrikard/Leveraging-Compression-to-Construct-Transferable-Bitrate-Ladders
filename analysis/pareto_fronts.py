"""
Constructing Pareto-Fronts
"""
# Importing Libraries
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import CubicSpline

import os, sys, warnings
import pickle
from tqdm import tqdm
warnings.filterwarnings("ignore")
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import functions.IO_functions as IO_functions
import functions.extract_functions as extract_functions
import functions.pareto_front_points as pareto_front_points
import defaults

# Paths and Settings
Resolution_Color_Map = dict(zip(defaults.resolutions, ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f']))


# Plotting Pareto-Fronts for each codec and preset
for codec in defaults.codec_preset_pairs.keys():
	for preset in defaults.codec_preset_pairs[codec]:
		# Plotting Pareto-Front
		folders_path = os.path.join(defaults.rq_points_dataset_path, codec, preset)

		plt.figure(figsize=(10,8))
		plt.grid()
		plt.xlabel(r"$\log_{2}(Bitrate)$")
		plt.ylabel("VMAF")
		plt.title("Pareto-Front")

		for video_filename in tqdm(defaults.Video_Titles):
			video_rq_points_info = IO_functions.read_create_jsonfile(os.path.join(folders_path, video_filename, "crfs.json"))
			
			RQ_pairs = extract_functions.Extract_RQ_Information(
				video_rq_points_info=video_rq_points_info,
				quality_metric="vmaf",
				resolutions=defaults.resolutions,
				CRFs=defaults.codec_CRF_ranges[codec],
				QPs=None,
				min_quality=20,
				max_quality=100,
				min_bitrate=-np.inf,
				max_bitrate=np.inf,
				set_bitrate_log_base=2
			)

			ParetoFront_pairs = pareto_front_points.Pareto_Front_Points(
				RQ_pairs=RQ_pairs,
				Resolutions=defaults.resolutions,
				use_interpolated_points=False
			)
			ParetoFront_pairs = pareto_front_points.Correct_Pareto_Front(ParetoFront_pairs)
			
			for res in defaults.resolutions:
				data = ParetoFront_pairs[res]
				if len(data) > 0:
					plt.plot(data[:,0], data[:,1], color=Resolution_Color_Map[res], label=res)
					plt.scatter(data[:,0], data[:,1], color=Resolution_Color_Map[res], label=res)

		handles, labels = plt.gca().get_legend_handles_labels()
		by_label = dict(zip(labels, handles))
		plt.legend(by_label.values(), by_label.keys())
		plt.savefig('plots/pareto_front_curves/{}_{}_Pareto_Front_Curves.png'.format(codec, preset), dpi=400, bbox_inches='tight')