#!/bin/zsh

# Train force prediction with gt CP
python3 main.py --title train_all --sequence_length 10 --gpu-ids 0 --number_of_cp 5 --model ImageAndCPInputKPOutModel --dataset DatasetWAugmentation --loss KeypointProjectionLoss --object_list ALL --data DatasetForce

# Only predicting contact points
python3 main.py --title train_cp_prediction --batch-size 64 --workers 10 --gpu-ids 0 --number_of_cp 5 --model NoForceOnlyCPModel --dataset DatasetWAugmentation --loss CPPredictionLoss --object_list ALL --break-batch 1 --data DatasetForce

# Jointly optimizing
python3 main.py --title joint_training --sequence_length 10 --gpu-ids 0 --number_of_cp 5 --model SeparateTowerModel --dataset DatasetWAugmentation --loss KPProjectionCPPredictionLoss --object_list ALL --data DatasetForce

# Train with pseudo-gt force labels, gtforce_train.json calculated using NoModelGTForceBaseline in "models/no_model_gt_calculator.py"
python3 main.py --title regress_force_baseline --batch-size 64 --workers 5 --sequence_length 10 --gpu-ids 0 --number_of_cp 5 --model BaselineRegressForce --dataset BaselineForceDatasetWAugmentation --loss ForceRegressionLoss --object_list 072-a_toy_airplane --break-batch 1 --data DatasetForce --predicted_cp_adr DatasetForce/trained_weights/gtforce_train.json

# Ablation initial state prediction
python3 main.py --title predict_initial_pose --sequence_length 10 --gpu-ids 0 --number_of_cp 5 --model PredictInitPoseAndForce --dataset DatasetWAugmentation --loss KeypointProjectionLoss --object_list 072-a_toy_airplane --data DatasetForce