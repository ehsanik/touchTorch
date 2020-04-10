import torch
contact_points_on_plane = {"0": [0.05332787712580389, -0.12846163086112117, 0.47724519560618406], "1": [0.27414052537778244, -0.026931013289426535, 0.2893882406584509], "2": [0.27140876126025726, -0.09632579535532362, 0.316487697465908], "3": [0.2796991934346229, -0.16680952173134878, 0.3122024414918917], "4": [0.29097315246397126, -0.271058414211527, 0.3043431598093991]}

DEFAULT_IMAGE_SIZE = torch.Tensor([1920., 1080.])
CAMERA_INTRINS_UNNORM = torch.Tensor([
    [1.0689931481490066e+03, 0., 9.6684897214782484e+02],
    [0., 1.0704204272330430e+03, 5.5191508359978184e+02],
    [0., 0., 1.]
])
DISTORTION = torch.Tensor([7.1227911507083103e-02, -1.1256457262779640e-01, 4.4180521704768683e-03, 2.3432065752824159e-03, 3.9544748403990908e-02])

CONTACT_POINT_MASK_VALUE = -100


ALL_OBJECTS = [
'003_cracker_box', '004_sugar_box', '005_tomato_soup_can', '006_mustard_bottle','007_tuna_fish_can', '008_pudding_box', '009_gelatin_box', '010_potted_meat_can', '019_pitcher_base','021_bleach_cleanser', '024_bowl','025_mug', '027_skillet', '029_plate','030_fork', '031_spoon', '032_knife','033_spatula', '035_power_drill', '037_scissors','042_adjustable_wrench', '048_hammer','055_baseball', '062_dice','072-a_toy_airplane', '077_rubiks_cube'#,'001_chips_can_UCB',
]

ALL_OBJECT_KEYPOINT_PATH = [
'objects_16k/003_cracker_box/google_16k/textured.urdf_2019_09_29_17_01_44_keypoints.json',
'objects_16k/004_sugar_box/google_16k/textured.urdf_2019_09_29_17_04_29_keypoints.json',
'objects_16k/005_tomato_soup_can/google_16k/textured.urdf_2019_09_29_17_07_20_keypoints.json',
'objects_16k/006_mustard_bottle/google_16k/textured.urdf_2019_09_29_17_09_19_keypoints.json',
'objects_16k/007_tuna_fish_can/google_16k/textured.urdf_2019_09_29_17_11_35_keypoints.json',
'objects_16k/008_pudding_box/google_16k/textured.urdf_2019_09_29_17_14_30_keypoints.json',
'objects_16k/009_gelatin_box/google_16k/textured.urdf_2019_09_29_17_19_34_keypoints.json',
'objects_16k/010_potted_meat_can/google_16k/textured.urdf_2019_09_29_17_21_14_keypoints.json',
'objects_16k/019_pitcher_base/google_16k/textured.urdf_2019_09_29_17_23_08_keypoints.json',
'objects_16k/021_bleach_cleanser/google_16k/textured.urdf_2019_09_29_17_24_43_keypoints.json',
'objects_16k/024_bowl/google_16k/textured.urdf_2019_09_29_17_27_03_keypoints.json',
'objects_16k/025_mug/google_16k/textured.urdf_2019_09_29_17_29_17_keypoints.json',
'objects_16k/027_skillet/google_16k/textured.urdf_2019_09_29_17_31_36_keypoints.json',
'objects_16k/029_plate/google_16k/textured.urdf_2019_09_29_17_34_46_keypoints.json',
'objects_16k/030_fork/google_16k/textured.urdf_2019_09_29_17_36_44_keypoints.json',
'objects_16k/031_spoon/google_16k/textured.urdf_2019_09_29_17_38_34_keypoints.json',
'objects_16k/032_knife/google_16k/textured.urdf_2019_09_30_13_47_31_keypoints.json',
'objects_16k/033_spatula/google_16k/textured.urdf_2019_09_29_17_41_21_keypoints.json',
'objects_16k/035_power_drill/google_16k/textured.urdf_2019_09_29_17_43_00_keypoints.json',
'objects_16k/037_scissors/google_16k/textured.urdf_2019_09_29_17_44_32_keypoints.json',
'objects_16k/042_adjustable_wrench/google_16k/textured.urdf_2019_09_29_17_46_28_keypoints.json',
'objects_16k/048_hammer/google_16k/textured.urdf_2019_09_29_17_48_18_keypoints.json',
'objects_16k/055_baseball/google_16k/textured.urdf_2019_09_30_10_15_08_keypoints.json',
'objects_16k/062_dice/google_16k/textured.urdf_2019_09_30_10_16_57_keypoints.json',
'objects_16k/072-a_toy_airplane/google_16k/textured.urdf_2019_09_30_10_18_15_keypoints.json',
'objects_16k/077_rubiks_cube/google_16k/textured.urdf_2019_09_30_10_20_24_keypoints.json',
]
CHOSEN_OBJECTS = [
    '019_pitcher_base','021_bleach_cleanser', '027_skillet', '035_power_drill', '048_hammer','072-a_toy_airplane', '005_tomato_soup_can', '006_mustard_bottle'
]

ALL_OBJECT_KEYPOINT_NAME = {
    obj_path.split('/')[1]:obj_path for obj_path in ALL_OBJECT_KEYPOINT_PATH
}

GRAVITY_VALUE = torch.Tensor([-0.1397,  0.8652, -0.4816])

OBJECT_TO_SCALE = {
'003_cracker_box':5,
'004_sugar_box':5,
'005_tomato_soup_can':10,
'006_mustard_bottle':5,
'007_tuna_fish_can':10,
'008_pudding_box':5,
'009_gelatin_box':5,
'010_potted_meat_can':5,
'019_pitcher_base':5,
'021_bleach_cleanser':5,
'024_bowl':5,
'025_mug':5,
'027_skillet':5,
'029_plate':5,
'030_fork':10,
'031_spoon':10,
'032_knife':10,
'033_spatula':5,
'035_power_drill':5,
'037_scissors':5,
'042_adjustable_wrench':5,
'048_hammer':5,
'055_baseball':10,
'062_dice':20,
'072-a_toy_airplane':5,
'077_rubiks_cube':10,
}

OBJECT_NAME_TO_CENTER_OF_MASS = {
    '003_cracker_box' : torch.Tensor([-0.0743, -0.0710,  0.5110]),
    '004_sugar_box' : torch.Tensor([-0.0385, -0.0854,  0.4301]),
    '005_tomato_soup_can' : torch.Tensor([-0.0934,  0.8422,  0.5003]),
    '006_mustard_bottle' : torch.Tensor([-0.0753, -0.1155,  0.3752]),
    '007_tuna_fish_can' : torch.Tensor([-0.2600, -0.2214,  0.1299]),
    '008_pudding_box' : torch.Tensor([0.0076, 0.0940, 0.0933]),
    '009_gelatin_box' : torch.Tensor([-0.1134, -0.0393,  0.0694]),
    '010_potted_meat_can' : torch.Tensor([-0.1642, -0.1318,  0.1892]),
    '019_pitcher_base' : torch.Tensor([-0.0671,  0.1674,  0.6132]),
    '021_bleach_cleanser' : torch.Tensor([-0.0857,  0.0579,  0.4985]),
    '024_bowl' : torch.Tensor([-0.0714, -0.2286,  0.1339]),
    '025_mug' : torch.Tensor([-0.0943,  0.0829,  0.1345]),
    '027_skillet' : torch.Tensor([-0.0338,  0.1226,  0.1916]),
    '029_plate' : torch.Tensor([-0.0500,  0.0062,  0.0442]),
    '030_fork' : torch.Tensor([ 0.0850, -0.2153,  0.0671]),
    '031_spoon' : torch.Tensor([-0.1806, -0.1016,  0.1005]),
    '032_knife' : torch.Tensor([ 0.0845, -0.2909,  0.0500]),
    '033_spatula' : torch.Tensor([-0.2329, -0.3930,  0.0659]),
    '035_power_drill' : torch.Tensor([-0.1536,  0.1218,  0.1178]),
    '037_scissors' : torch.Tensor([0.0730, 0.0158, 0.0370]),
    '042_adjustable_wrench' : torch.Tensor([ 0.0725, -0.2483,  0.0213]),
    '048_hammer' : torch.Tensor([-0.1429, -0.0634,  0.0783]),
    '055_baseball' : torch.Tensor([-0.0995, -0.4812,  0.3593]),
    '062_dice' : torch.Tensor([-0.2241,  0.1833,  0.1681]),
    '072-a_toy_airplane' : torch.Tensor([0.1786, 0.1540, 0.4546]),
    '077_rubiks_cube' : torch.Tensor([-0.1622, -0.0049,  0.2859]),

}

OBJECT_TO_SCALE = {obj_name: 1. / scale for (obj_name, scale) in OBJECT_TO_SCALE.items()}

