SCRIPT=scripts/evaluation/evaluate_uuv_dataset.py
NET=/home/asl/Storage/20241115_SINTEF_Journal/01_Datasets/radar_b_bs4_e21.pth
DATASET=/home/asl/Storage/20241115_SINTEF_Journal/01_Datasets
python3 $SCRIPT --network $NET --dataset $DATASET/case_47/output
# python3 $SCRIPT --network $NET --dataset $DATASET/case_38/output
# python3 $SCRIPT --network $NET --dataset $DATASET/case_37/output
# python3 $SCRIPT --network $NET --dataset $DATASET/case_36/output
# python3 $SCRIPT --network $NET --dataset $DATASET/case_30/output
# python3 $SCRIPT --network $NET --dataset $DATASET/case_29/output
# python3 $SCRIPT --network $NET --dataset $DATASET/case_27/output
# python3 $SCRIPT --network $NET --dataset $DATASET/case_25/output
# python3 $SCRIPT --network $NET --dataset $DATASET/case_4_2/output
# python3 $SCRIPT --network $NET --dataset $DATASET/case_4/output
