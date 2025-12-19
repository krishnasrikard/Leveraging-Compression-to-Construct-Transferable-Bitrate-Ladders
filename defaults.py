# Importing Libraries
import numpy as np
from sklearn.model_selection import KFold

# Codec and Preset
codec_preset_pairs = {
	"libx265": ["veryfast", "fast", "medium", "slow"],
	"libsvtav1": ["8", "6", "4"],
	"libvpx-vp9": ["4", "3"],
	"libaom-av1": ["7", "5"]
}

# Resolutions
resolutions = [(3840,2160),(2560,1440),(1920,1080),(1280,720),(960,540)]

## Rate-Control Parameters
# CRFs
codec_full_CRF_ranges = {
	"libx265": [14,16,18,20,22,24,26,28,30,32,34,36,38,40,42,44,46,48,50],
	"libsvtav1": [16,18,20,22,24,26,28,30,32,34,36,38,40,42,44,46,48,50,52,54,56,58,60,62],
	"libvpx-vp9": [16,18,20,22,24,26,28,30,32,34,36,38,40,42,44,46,48,50,52,54,56,58,60,62],
	"libaom-av1": [16,18,20,22,24,26,28,30,32,34,36,38,40,42,44,46,48,50,52,54,56,58,60,62]
}

codec_CRF_ranges = {
	"libx265": [14,16,18,20,22,24,26,28,30,32,34,36,38,40,42,44,46,48,50],
	"libsvtav1": [16,18,20,22,24,26,28,30,32,34,36,38,40,42,44,46,48,50,52,54,56,58,60,62],
	"libvpx-vp9": [16,18,20,22,24,26,28,30,32,34,36,38,40,42,44,46,48,50,52,54,56,58,60,62],
	"libaom-av1": [16,18,20,22,24,26,28,30,32,34,36,38,40,42,44,46,48,50,52,54,56,58,60,62]
}

# QPs
QPs = [18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47,48,49,50]


# Thresholds for RQ-points
min_bitrate = -np.inf
max_bitrate = np.inf
min_quality = 20
max_quality = 99.9


## Main Working Directory
main_working_dir = "/home/krishna/Leveraging-Compression-to-Construct-Transferable-Bitrate-Ladders"


## Paths to Datasets

# Source
source_dataset_path = "/home/krishna/Nebula/krishna/Leveraging-Compression-to-Construct-Transferable-Bitrate-Ladders/Datasets/4K_Shots/yuv_files"
source_dataset_info_path = "/home/krishna/Nebula/krishna/Leveraging-Compression-to-Construct-Transferable-Bitrate-Ladders/Datasets/4K_Shots/video_properties.csv"

# RQ-Points
rq_points_dataset_path = "/home/krishna/Nebula/krishna/Leveraging-Compression-to-Construct-Transferable-Bitrate-Ladders/rq_points_dataset"

# Features
low_level_features_path = "/home/krishna/Nebula/krishna/Leveraging-Compression-to-Construct-Transferable-Bitrate-Ladders/features_dataset/low_level_features"
vif_features_path = "/home/krishna/Nebula/krishna/Leveraging-Compression-to-Construct-Transferable-Bitrate-Ladders/features_dataset/vif_features"


# Evaluation Bitrates i.e Steps in Bitrate Ladder
evaluation_bitrates = [100, 200, 400, 600, 800, 1000, 1500, 2000, 2400, 3000, 3500, 4000, 4500, 5000, 6000, 7000, 8100, 9000, 10000, 11600, 13000, 15000]
evaluation_bitrates = np.log2(1000*np.asarray(evaluation_bitrates))
evaluation_bitrates = list(np.round(evaluation_bitrates, decimals=4))

# Evaluation CRFs
codec_evaluation_CRF_ranges = {
	"libx265": [18,22,26,30,34,38,42],
	"libsvtav1": [20,26,32,38,44,50,56,62]
}


## Video Filenames in 4K_Shots dataset
Video_Titles = ['AAdvertisingMassagesBangkokVidevo_3840x2176_25fps_10bit_420', 'ABangkokMarketVidevo_3840x2176_25fps_10bit_420', 'ABasketballGoalScoredS1Videvo_3840x2176_25fps_10bit_420', 'ABasketballGoalScoredS2Videvo_3840x2176_25fps_10bit_420', 'ABasketballS1YonseiUniversity_3840x2176_30fps_10bit_420', 'ABasketballS2YonseiUniversity_3840x2176_30fps_10bit_420', 'ABasketballS3YonseiUniversity_3840x2176_30fps_10bit_420', 'ABoatsChaoPhrayaRiverVidevo_3840x2176_23fps_10bit_420', 'ABobbleheadBVIHFR_3840x2176_120fps_10bit_420', 'ABookcaseBVITexture_3840x2176_120fps_10bit_420', 'ABricksBushesStaticBVITexture_3840x2176_120fps_10bit_420', 'ABricksLeavesBVITexture_3840x2176_120fps_10bit_420', 'ABricksTiltingBVITexture_3840x2176_120fps_10bit_420', 'ABubblesPitcherS1BVITexture_3840x2176_120fps_10bit_420', 'ABuildingRoofS1IRIS_3840x2176_24fps_10bit_420', 'ABuildingRoofS2IRIS_3840x2176_24fps_10bit_420', 'ABuildingRoofS3IRIS_3840x2176_24fps_10bit_420', 'ABuildingRoofS4IRIS_3840x2176_24fps_10bit_420', 'ABuntingHangingAcrossHongKongVidevo_3840x2176_25fps_10bit_420', 'ABusyHongKongStreetVidevo_3840x2176_25fps_10bit_420', 'ACalmingWaterBVITexture_3840x2176_120fps_10bit_420', 'ACarpetPanAverageBVITexture_3840x2176_120fps_10bit_420', 'ACatchBVIHFR_3840x2176_120fps_10bit_420', 'ACeramicsandSpicesMoroccoVidevo_3840x2176_50fps_10bit_420', 'ACharactersYonseiUniversity_3840x2176_30fps_10bit_420', 'AChristmasPresentsIRIS_3840x2176_24fps_10bit_420', 'AChristmasRoomDareful_3840x2176_29fps_10bit_420', 'AChurchInsideMCLJCV_3840x2176_30fps_10bit_420', 'ACityScapesS1IRIS_3840x2176_24fps_10bit_420', 'ACityScapesS2IRIS_3840x2176_24fps_10bit_420', 'ACityScapesS3IRIS_3840x2176_24fps_10bit_420', 'ACityStreetS1IRIS_3840x2176_24fps_10bit_420', 'ACityStreetS3IRIS_3840x2176_24fps_10bit_420', 'ACityStreetS4IRIS_3840x2176_24fps_10bit_420', 'ACityStreetS5IRIS_3840x2176_24fps_10bit_420', 'ACityStreetS6IRIS_3840x2176_24fps_10bit_420', 'ACityStreetS7IRIS_3840x2176_24fps_10bit_420', 'ACloseUpBasketballSceneVidevo_3840x2176_25fps_10bit_420', 'ACloudsStaticBVITexture_3840x2176_120fps_10bit_420', 'AColourfulDecorationWatPhoVidevo_3840x2176_50fps_10bit_420', 'AColourfulKoreanLanternsVidevo_3840x2176_50fps_10bit_420', 'AColourfulPaperLanternsVidevo_3840x2176_50fps_10bit_420', 'AColourfulRugsMoroccoVidevo_3840x2176_50fps_10bit_420', 'AConstructionS2YonseiUniversity_3840x2176_30fps_10bit_420', 'ACrosswalkHongKong2S1Videvo_3840x2176_25fps_10bit_420', 'ACrosswalkHongKong2S2Videvo_3840x2176_25fps_10bit_420', 'ACrosswalkHongKongVidevo_3840x2176_25fps_10bit_420', 'ACrowdRunMCLV_3840x2176_25fps_10bit_420', 'ACyclistS1BVIHFR_3840x2176_120fps_10bit_420', 'ACyclistVeniceBeachBoardwalkVidevo_3840x2176_25fps_10bit_420', 'ADollsScene1YonseiUniversity_3840x2176_30fps_10bit_420', 'ADollsScene2YonseiUniversity_3840x2176_30fps_10bit_420', 'ADowntownHongKongVidevo_3840x2176_25fps_10bit_420', 'ADropsOnWaterBVITexture_3840x2176_120fps_10bit_420', 'AElFuenteMaskLIVENetFlix_3840x2176_24fps_10bit_420', 'AEnteringHongKongStallS1Videvo_3840x2176_25fps_10bit_420', 'AEnteringHongKongStallS2Videvo_3840x2176_25fps_10bit_420', 'AFerrisWheelTurningVidevo_3840x2176_50fps_10bit_420', 'AFireS18Mitch_3840x2176_24fps_10bit_420', 'AFireS21Mitch_3840x2176_24fps_10bit_420', 'AFireS71Mitch_3840x2176_24fps_10bit_420', 'AFirewoodS1IRIS_3840x2176_24fps_10bit_420', 'AFirewoodS2IRIS_3840x2176_25fps_10bit_420', 'AFitnessIRIS_3840x2176_24fps_10bit_420', 'AFlagShootTUMSVT_3840x2176_50fps_10bit_420', 'AFlowerChapelS1IRIS_3840x2176_24fps_10bit_420', 'AFlowerChapelS2IRIS_3840x2176_24fps_10bit_420', 'AFlyingCountrysideDareful_3840x2176_29fps_10bit_420', 'AFlyingMountainsDareful_3840x2176_29fps_10bit_420', 'AFlyingThroughLAStreetVidevo_3840x2176_23fps_10bit_420', 'AFungusZoomBVITexture_3840x2176_120fps_10bit_420', 'AGrassBVITexture_3840x2176_120fps_10bit_420', 'AGrazTowerIRIS_3840x2176_24fps_10bit_420', 'AHamsterBVIHFR_3840x2176_120fps_10bit_420', 'AHarleyDavidsonIRIS_3840x2176_24fps_10bit_420', 'AHongKongIslandVidevo_3840x2176_25fps_10bit_420', 'AHongKongMarket1Videvo_3840x2176_25fps_10bit_420', 'AHongKongMarket2Videvo_3840x2176_25fps_10bit_420', 'AHongKongMarket3S1Videvo_3840x2176_25fps_10bit_420', 'AHongKongMarket3S2Videvo_3840x2176_25fps_10bit_420', 'AHongKongMarket4S1Videvo_3840x2176_25fps_10bit_420', 'AHongKongMarket4S2Videvo_3840x2176_25fps_10bit_420', 'AHorseDrawnCarriagesVidevo_3840x2176_50fps_10bit_420', 'AHorseStaringS1Videvo_3840x2176_50fps_10bit_420', 'AHorseStaringS2Videvo_3840x2176_50fps_10bit_420', 'AJoggersS1BVIHFR_3840x2176_120fps_10bit_420', 'AJoggersS2BVIHFR_3840x2176_120fps_10bit_420', 'AKartingIRIS_3840x2176_24fps_10bit_420', 'AKoraDrumsVidevo_3840x2176_25fps_10bit_420', 'ALakeYonseiUniversity_3840x2176_30fps_10bit_420', 'ALampLeavesBVITexture_3840x2176_120fps_10bit_420', 'ALaundryHangingOverHongKongVidevo_3840x2176_25fps_10bit_420', 'ALeaves1BVITexture_3840x2176_120fps_10bit_420', 'ALeaves3BVITexture_3840x2176_120fps_10bit_420', 'ALowLevelShotAlongHongKongVidevo_3840x2176_25fps_10bit_420', 'ALungshanTempleS1Videvo_3840x2176_50fps_10bit_420', 'ALungshanTempleS2Videvo_3840x2176_50fps_10bit_420', 'AManMoTempleVidevo_3840x2176_25fps_10bit_420', 'AManStandinginProduceTruckVidevo_3840x2176_25fps_10bit_420', 'AManWalkingThroughBangkokVidevo_3840x2176_25fps_10bit_420', 'AMaplesS1YonseiUniversity_3840x2176_30fps_10bit_420', 'AMaplesS2YonseiUniversity_3840x2176_30fps_10bit_420', 'AMirabellParkS1IRIS_3840x2176_24fps_10bit_420', 'AMirabellParkS2IRIS_3840x2176_24fps_10bit_420', 'AMoroccanCeramicsShopVidevo_3840x2176_50fps_10bit_420', 'AMoroccanSlippersVidevo_3840x2176_50fps_10bit_420', 'AMuralPaintingVidevo_3840x2176_25fps_10bit_420', 'AMyeongDongVidevo_3840x2176_25fps_10bit_420', 'ANewYorkStreetDareful_3840x2176_30fps_10bit_420', 'AOrangeBuntingoverHongKongVidevo_3840x2176_25fps_10bit_420', 'APaintingTiltingBVITexture_3840x2176_120fps_10bit_420', 'AParkViolinMCLJCV_3840x2176_25fps_10bit_420', 'APedestriansSeoulatDawnVidevo_3840x2176_25fps_10bit_420', 'APeopleWalkingS1IRIS_3840x2176_24fps_10bit_420', 'APersonRunningOutsideVidevo_3840x2176_50fps_10bit_420', 'APillowsTransBVITexture_3840x2176_120fps_10bit_420', 'APlasmaFreeBVITexture_3840x2176_120fps_10bit_420', 'APresentsChristmasTreeDareful_3840x2176_29fps_10bit_420', 'AReadySetGoS2TampereUniversity_3840x2176_120fps_10bit_420', 'AResidentialBuildingSJTU_3840x2176_60fps_10bit_420', 'ARollerCoaster2Netflix_3840x2176_60fps_10bit_420', 'ARunnersSJTU_3840x2176_60fps_10bit_420', 'ARuralSetupIRIS_3840x2176_24fps_10bit_420', 'ARuralSetupS2IRIS_3840x2176_24fps_10bit_420', 'AScarfSJTU_3840x2176_60fps_10bit_420', 'ASeasideWalkIRIS_3840x2176_24fps_10bit_420', 'ASeekingMCLV_3840x2176_25fps_10bit_420', 'ASeoulCanalatDawnVidevo_3840x2176_25fps_10bit_420', 'AShoppingCentreVidevo_3840x2176_25fps_10bit_420', 'ASignboardBoatLIVENetFlix_3840x2176_30fps_10bit_420', 'ASkyscraperBangkokVidevo_3840x2176_23fps_10bit_420', 'ASmokeClearBVITexture_3840x2176_120fps_10bit_420', 'ASmokeS45Mitch_3840x2176_24fps_10bit_420', 'ASparklerBVIHFR_3840x2176_120fps_10bit_420', 'ASquareS1IRIS_3840x2176_24fps_10bit_420', 'ASquareS2IRIS_3840x2176_24fps_10bit_420', 'AStreetArtVidevo_3840x2176_30fps_10bit_420', 'AStreetDancerS1IRIS_3840x2176_24fps_10bit_420', 'AStreetDancerS2IRIS_3840x2176_24fps_10bit_420', 'AStreetDancerS3IRIS_3840x2176_24fps_10bit_420', 'AStreetDancerS4IRIS_3840x2176_24fps_10bit_420', 'AStreetDancerS5IRIS_3840x2176_24fps_10bit_420', 'ATaiChiHongKongS1Videvo_3840x2176_25fps_10bit_420', 'ATaiChiHongKongS2Videvo_3840x2176_25fps_10bit_420', 'ATaipeiCityRooftops8Videvo_3840x2176_25fps_10bit_420', 'ATaipeiCityRooftopsS1Videvo_3840x2176_25fps_10bit_420', 'ATaipeiCityRooftopsS2Videvo_3840x2176_25fps_10bit_420', 'ATaksinBridgeVidevo_3840x2176_23fps_10bit_420', 'ATallBuildingsSJTU_3840x2176_60fps_10bit_420', 'ATennisMCLV_3840x2176_24fps_10bit_420', 'AToddlerFountain2Netflix_3840x2176_60fps_10bit_420', 'ATouristsSatOutsideVidevo_3840x2176_25fps_10bit_420', 'ATrackingDownHongKongSideVidevo_3840x2176_25fps_10bit_420', 'ATrackingPastRestaurantVidevo_3840x2176_25fps_10bit_420', 'ATrackingPastStallHongKongVidevo_3840x2176_25fps_10bit_420', 'ATraditionalIndonesianKecakVidevo_3840x2176_25fps_10bit_420', 'ATrafficFlowSJTU_3840x2176_60fps_10bit_420', 'ATrafficandBuildingSJTU_3840x2176_60fps_10bit_420', 'ATrafficonTasksinBridgeVidevo_3840x2176_25fps_10bit_420', 'ATreeWillsBVITexture_3840x2176_120fps_10bit_420', 'ATruckIRIS_3840x2176_24fps_10bit_420', 'AUnloadingVegetablesVidevo_3840x2176_25fps_10bit_420', 'AVegetableMarketS1LIVENetFlix_3840x2176_30fps_10bit_420', 'AVegetableMarketS2LIVENetFlix_3840x2176_30fps_10bit_420', 'AVegetableMarketS3LIVENetFlix_3840x2176_30fps_10bit_420', 'AVegetableMarketS4LIVENetFlix_3840x2176_30fps_10bit_420', 'AVeniceSceneIRIS_3840x2176_24fps_10bit_420', 'AWalkingDownKhaoStreetVidevo_3840x2176_25fps_10bit_420', 'AWalkingDownNorthRodeoVidevo_3840x2176_25fps_10bit_420', 'AWalkingThroughFootbridgeVidevo_3840x2176_25fps_10bit_420', 'AWatPhoTempleVidevo_3840x2176_50fps_10bit_420', 'AWaterS65Mitch_3840x2176_24fps_10bit_420', 'AWaterS81Mitch_3840x2176_24fps_10bit_420', 'AWoodSJTU_3840x2176_60fps_10bit_420', 'AWovenVidevo_3840x2176_25fps_10bit_420', 'Chimera-ep01_3840x2160_2997fps_10bit_422', 'Chimera-ep02_3840x2160_2997fps_10bit_422', 'Chimera-ep08_3840x2160_2997fps_10bit_422', 
# 'Chimera-ep11_3840x2160_2997fps_10bit_422', 
'Chimera-ep12_3840x2160_2997fps_10bit_422', 'Chimera-ep16_3840x2160_2997fps_10bit_422', 'Neon1224_3840x2160_2997fps', 'Netflix_BoxingPractice_4096x2160_60fps_10bit_420', 'Netflix_FoodMarket2_4096x2160_60fps_10bit_420', 'Netflix_Narrator_4096x2160_60fps_10bit_420', 'Netflix_RitualDanceShot_4096x2160_60fps_10bit_420', 'Netflix_Tango_4096x2160_60fps_10bit_420', 'Nightclub113_3840x2160_24fp', 'SmoothSkaterP5_3840x2160_30fps', 'Water1394_3840x2160', 'cosmos_aom_sdr_11589-11752', 'cosmos_aom_sdr_12025-12075', 'cosmos_aom_sdr_12149-12330', 'cosmos_aom_sdr_12916-13078', 'cosmos_aom_sdr_13446-13649', 'cosmos_aom_sdr_1573-1749', 'cosmos_aom_sdr_8686-8826', 'cosmos_aom_sdr_9561-9789', 'meridian_aom_sdr_11872-12263', 'meridian_aom_sdr_12264-12745', 'meridian_aom_sdr_15932-16309', 'meridian_aom_sdr_1782-2163', 'meridian_aom_sdr_20988-21412', 'meridian_aom_sdr_22412-22738', 'meridian_aom_sdr_24058-24550', 'nocturne_aom_sdr_17140-17709', 'nocturne_aom_sdr_18013-18315', 'nocturne_aom_sdr_2370-2539', 'nocturne_aom_sdr_23820-24322', 'nocturne_aom_sdr_27740-28109', 'nocturne_aom_sdr_32660-32799', 'nocturne_aom_sdr_8540-9009', 'nocturne_aom_sdr_9010-9349', 'sol_levante_aom_sdr_2268-2412', 'sol_levante_aom_sdr_289-453', 'sol_levante_aom_sdr_3282-3874', 'sol_levante_aom_sdr_4123-4545', 'sol_levante_aom_sdr_519-649']


# Splitting in 5 different Training and Test datasets
np.random.seed(0)
kfold = KFold(n_splits=5, shuffle=True, random_state=0)
Datasets = {}
for i, (train_indices, test_indices) in enumerate(kfold.split(Video_Titles)):
	# Training, Validation and Testing Set Indices
	num_valid_indices = int(0.1*(len(Video_Titles)))
	np.random.shuffle(train_indices)
	valid_indices = train_indices[0:num_valid_indices]
	train_indices = train_indices[num_valid_indices:]

	# Sorting
	train_indices.sort()
	valid_indices.sort()
	test_indices.sort()

	Datasets["Split-{}".format(i)] = {
		"Train_Video_Files": list(np.array(Video_Titles)[train_indices]),
		"Valid_Video_Files": list(np.array(Video_Titles)[valid_indices]),
		"Test_Video_Files": list(np.array(Video_Titles)[test_indices]),
	}


# Feature Names

# GLCM Features
glcm_features = ['kurt_GLCM_contrast_mean',
'kurt_GLCM_contrast_std',
'mean_GLCM_contrast_mean',
'mean_GLCM_contrast_std',
'skew_GLCM_contrast_mean',
'skew_GLCM_contrast_std',
'std_GLCM_contrast_mean',
'std_GLCM_contrast_std',
'kurt_GLCM_correlation_mean',
'kurt_GLCM_correlation_std',
'mean_GLCM_correlation_mean',
'mean_GLCM_correlation_std',
'skew_GLCM_correlation_mean',
'skew_GLCM_correlation_std',
'std_GLCM_correlation_mean',
'std_GLCM_correlation_std',
'kurt_GLCM_energy_mean',
'kurt_GLCM_energy_std',
'mean_GLCM_energy_mean',
'mean_GLCM_energy_std',
'skew_GLCM_energy_mean',
'skew_GLCM_energy_std',
'std_GLCM_energy_mean',
'std_GLCM_energy_std',
'kurt_GLCM_homogeneity_mean',
'kurt_GLCM_homogeneity_std',
'mean_GLCM_homogeneity_mean',
'mean_GLCM_homogeneity_std',
'skew_GLCM_homogeneity_mean',
'skew_GLCM_homogeneity_std',
'std_GLCM_homogeneity_mean',
'std_GLCM_homogeneity_std'
]

per_frame_glcm_features = ['GLCM_contrast_mean',
'GLCM_contrast_std',
'GLCM_correlation_mean',
'GLCM_correlation_std',
'GLCM_energy_mean',
'GLCM_energy_std',
'GLCM_homogeneity_mean',
'GLCM_homogeneity_std'
]

# Temporal Coherence Features
tc_features = ['mean_TC_kurt',
'mean_TC_mean',
'mean_TC_skew',
'mean_TC_std',
'std_TC_kurt',
'std_TC_mean',
'std_TC_skew',
'std_TC_std'
]

per_frame_tc_features = ['TC_kurt',
'TC_mean',
'TC_skew',
'TC_std'
]

# Spatial Information
si_features = ['kurt_SI_mean',
'kurt_SI_std',
'mean_SI_mean',
'mean_SI_std',
'skew_SI_mean',
'skew_SI_std',
'std_SI_mean',
'std_SI_std'
]

per_frame_si_features = ['SI_mean',
'SI_std'
]

# Temporal Information
ti_features = ['kurt_TI_mean',
'kurt_TI_std',
'mean_TI_mean',
'mean_TI_std',
'skew_TI_mean',
'skew_TI_std',
'std_TI_mean',
'std_TI_std'
]

per_frame_ti_features = ['TI_mean',
'TI_std'
]

# Contrast Information
cti_features = ['kurt_CTI_mean',
'kurt_CTI_std',
'mean_CTI_mean',
'mean_CTI_std',
'skew_CTI_mean',
'skew_CTI_std',
'std_CTI_mean',
'std_CTI_std'
]

per_frame_cti_features = ['CTI_mean',
'CTI_std'
]

# Colofulness
cf_features = ['kurt_CF',
'mean_CF',
'skew_CF',
'std_CF'
]

per_frame_cf_features = ['CF']

# Chromiance Information
ci_features = ['kurt_CI_U_mean',
'kurt_CI_U_std',
'mean_CI_U_mean',
'mean_CI_U_std',
'skew_CI_U_mean',
'skew_CI_U_std',
'std_CI_U_mean',
'std_CI_U_std',
'kurt_CI_V_mean',
'kurt_CI_V_std',
'mean_CI_V_mean',
'mean_CI_V_std',
'skew_CI_V_mean',
'skew_CI_V_std',
'std_CI_V_mean',
'std_CI_V_std'
]

per_frame_ci_features = ['CI_U_mean',
'CI_U_std',
'CI_V_mean',
'CI_V_std'
]

# DCT Texture Energy
dct_features = ['mean_E_Y',
'mean_h_Y',
'mean_L_Y',
'mean_E_U',
'mean_h_U',
'mean_L_U',
'mean_E_V',
'mean_h_V',
'mean_L_V'
]

per_frame_dct_features = ['E_Y',
'h_Y',
'L_Y',
'E_U',
'h_U',
'L_U',
'E_V',
'h_V',
'L_V'
]

# Customized Bitrate dependent Texture Features
bitrate_texture_features = {
"log2(sqrt(mean_h_Y/mean_E_Y)) + 2b": ["mean_E_Y", "mean_h_Y"],
"log2(sqrt(mean_h_U/mean_E_U)) + 2b": ["mean_E_U", "mean_h_U"],
"log2(sqrt(mean_h_V/mean_E_V)) + 2b": ["mean_E_V", "mean_h_V"],
}

per_frame_bitrate_texture_features = {
"log2(sqrt(h_Y/E_Y)) + 2b": ["E_Y", "h_Y"],
"log2(sqrt(h_U/E_U)) + 2b": ["E_U", "h_U"],
"log2(sqrt(h_V/E_V)) + 2b": ["E_V", "h_V"],
}

# Customized Quality dependent Texture Features
quality_texture_features = {
"0.5*(q - log2(sqrt(mean_h_Y/mean_E_Y)))": ["mean_E_Y", "mean_h_Y"],
"0.5*(q - log2(sqrt(mean_h_U/mean_E_U)))": ["mean_E_U", "mean_h_U"],
"0.5*(q - log2(sqrt(mean_h_V/mean_E_V)))": ["mean_E_V", "mean_h_V"],
}

per_frame_quality_texture_features = {
"0.5*(q - log2(sqrt(h_Y/E_Y)))": ["mean_E_Y", "mean_h_Y"],
"0.5*(q - log2(sqrt(h_Y/E_Y)))": ["mean_E_U", "mean_h_U"],
"0.5*(q - log2(sqrt(h_Y/E_Y)))": ["mean_E_V", "mean_h_V"],
}

# Compute Time
compute_time_features = [
'GLCM_compute_time',
'TC_compute_time',
'SI_compute_time',
'TI_compute_time',
'CTI_compute_time',
'CF_compute_time',
'CI_U_compute_time',
'CI_V_compute_time',
'EhL_Y_compute_time',
'EhL_U_compute_time',
'EhL_V_compute_time'
]