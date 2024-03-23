
FILE_PATH=./dataset/few_NERD_transfer_data/5shot1/
FILE_NAME=cluster/models/coarseft_new/kmeans/layer-1_50eachclass_train.json


echo "---------------------------------------Training with file ${FILE_NAME}----------------------------------------"
echo ""

CUDA_VISIBLE_DEVICES=0  python3 run_ner_no_trainer_cluster.py \
  --model_name_or_path ./bert-base-cased \
  --train_file $FILE_NAME \
  --validation_file cluster/models/coarseft_new/kmeans/layer-1_50eachclass_test.json \
  --num_train_epochs 3 \
  --learning_rate 5e-5 \
  --per_device_train_batch_size 32 \
  --return_entity_level_metrics \
    --label_list ./dataset/coarse/labels.txt \
  --output_dir models/coarseft_cluster_2_1 \
  --label_schema IO   \
