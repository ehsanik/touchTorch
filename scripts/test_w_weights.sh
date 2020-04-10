#!/bin/zsh

#To test with our pretrained models

# Force prediction with gt CP
python3 main.py --title test_plane --sequence_length 10 --gpu-ids 0 --number_of_cp 5 --model ImageAndCPInputKPOutModel --dataset DatasetWAugmentation --loss KeypointProjectionLoss --object_list 072-a_toy_airplane --data DatasetForce --reload DatasetForce/trained_weights/plane_gt_cp.pytar test
python3 main.py --title test_skillet --sequence_length 10 --gpu-ids 0 --number_of_cp 5 --model ImageAndCPInputKPOutModel --dataset DatasetWAugmentation --loss KeypointProjectionLoss --object_list 027_skillet --data DatasetForce --reload DatasetForce/trained_weights/skillet_gt_cp.pytar test
python3 main.py --title test_pitcher --sequence_length 10 --gpu-ids 0 --number_of_cp 5 --model ImageAndCPInputKPOutModel --dataset DatasetWAugmentation --loss KeypointProjectionLoss --object_list 019_pitcher_base --data DatasetForce --reload DatasetForce/trained_weights/pitcher_gt_cp.pytar test
python3 main.py --title test_drill --sequence_length 10 --gpu-ids 0 --number_of_cp 5 --model ImageAndCPInputKPOutModel --dataset DatasetWAugmentation --loss KeypointProjectionLoss --object_list 035_power_drill --data DatasetForce --reload DatasetForce/trained_weights/drill_gt_cp.pytar test
#-------------
python3 main.py --title train_all --sequence_length 10 --gpu-ids 0 --number_of_cp 5 --model ImageAndCPInputKPOutModel --dataset DatasetWAugmentation --loss KeypointProjectionLoss --object_list ALL --data DatasetForce --reload DatasetForce/trained_weights/all_obj_gt_cp.pytar test


# Only CP prediction
python3 main.py --title test_cp_prediction --batch-size 64 --workers 10 --gpu-ids 0 --number_of_cp 5 --model NoForceOnlyCPModel --dataset DatasetWAugmentation --loss CPPredictionLoss --object_list ALL --break-batch 1 --data DatasetForce --reload  DatasetForce/trained_weights/all_obj_only_cp.pytar test


# Optimizing L_KP and L_CP individually
python3 main.py --title test_plane_indiv_training --sequence_length 10 --gpu-ids 0 --number_of_cp 5 --model SeparateTowerModel --dataset DatasetWAugmentation --loss KPProjectionCPPredictionLoss --object_list 072-a_toy_airplane --data DatasetForce --reload DatasetForce/trained_weights/plane_separate.pytar test
python3 main.py --title test_skillet_indiv_training --sequence_length 10 --gpu-ids 0 --number_of_cp 5 --model SeparateTowerModel --dataset DatasetWAugmentation --loss KPProjectionCPPredictionLoss --object_list 027_skillet --data DatasetForce --reload DatasetForce/trained_weights/skillet_separate.pytar test
python3 main.py --title test_pitcher_indiv_training --sequence_length 10 --gpu-ids 0 --number_of_cp 5 --model SeparateTowerModel --dataset DatasetWAugmentation --loss KPProjectionCPPredictionLoss --object_list 019_pitcher_base --data DatasetForce --reload DatasetForce/trained_weights/pitcher_separate.pytar test
python3 main.py --title test_drill_indiv_training --sequence_length 10 --gpu-ids 0 --number_of_cp 5 --model SeparateTowerModel --dataset DatasetWAugmentation --loss KPProjectionCPPredictionLoss --object_list 035_power_drill --data DatasetForce --reload DatasetForce/trained_weights/drill_separate.pytar test
python3 main.py --title test_all_indiv_training --sequence_length 10 --gpu-ids 0 --number_of_cp 5 --model SeparateTowerModel --dataset DatasetWAugmentation --loss KPProjectionCPPredictionLoss --object_list ALL --data DatasetForce --reload DatasetForce/trained_weights/all_obj_separate.pytar test

# Jointly optimizing L_KP and L_CP
python3 main.py --title test_plane_joint_training --sequence_length 10 --gpu-ids 0 --number_of_cp 5 --model SeparateTowerModel --dataset DatasetWAugmentation --loss KPProjectionCPPredictionLoss --object_list 072-a_toy_airplane --data DatasetForce --reload DatasetForce/trained_weights/plane_end2end.pytar test
python3 main.py --title test_skillet_joint_training --sequence_length 10 --gpu-ids 0 --number_of_cp 5 --model SeparateTowerModel --dataset DatasetWAugmentation --loss KPProjectionCPPredictionLoss --object_list 027_skillet --data DatasetForce --reload DatasetForce/trained_weights/skillet_end2end.pytar test
python3 main.py --title test_pitcher_joint_training --sequence_length 10 --gpu-ids 0 --number_of_cp 5 --model SeparateTowerModel --dataset DatasetWAugmentation --loss KPProjectionCPPredictionLoss --object_list 019_pitcher_base --data DatasetForce --reload DatasetForce/trained_weights/pitcher_end2end.pytar test
python3 main.py --title test_drill_joint_training --sequence_length 10 --gpu-ids 0 --number_of_cp 5 --model SeparateTowerModel --dataset DatasetWAugmentation --loss KPProjectionCPPredictionLoss --object_list 035_power_drill --data DatasetForce --reload DatasetForce/trained_weights/drill_end2end.pytar test
python3 main.py --title test_all_joint_training --sequence_length 10 --gpu-ids 0 --number_of_cp 5 --model SeparateTowerModel --dataset DatasetWAugmentation --loss KPProjectionCPPredictionLoss --object_list ALL --data DatasetForce --reload DatasetForce/trained_weights/all_obj_end2end.pytar test