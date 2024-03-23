#!/bin/bash

for j in {1..3}

do

FILE_PATH=dataset/fine/5shot
FILE_NAME=dataset/fine/5shot/${j}.json
for i in {1..3}
do

echo "---------------------------------------Training with file ${FILE_NAME}, round $i----------------------------------------"
echo ""

CUDA_VISIBLE_DEVICES=0  python3 run_ner_no_trainer_.py \
  --model_name_or_path models/coarseft_cluster_2_1 \
  --train_file $FILE_NAME \
  --validation_file ./dataset/fine/test.json \
  --num_train_epochs 20 \
  --learning_rate 1e-4 \
  --sample_path $FILE_PATH \
  --per_device_train_batch_size 4 \
  --per_device_eval_batch_size 32\
  --return_entity_level_metrics \
    --label_list ./dataset/fine/labels.txt \
  --output_dir test_models/coarseft_cluster_2_1_5shot_20epoch_${j}_${i} \
  --label_schema IO \
  --eval_last_epoch \
  --sample_id ${j} \
  --no_save \

done
done
