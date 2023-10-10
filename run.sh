BASE_DIR=/root/autodl-tmp/BertClassNews
echo ${BASE_DIR}

# cd ${BASE_DIR}/load_dataset

echo "开始处理数据" 
nohup python3 load_dataset/load_data_cls.py >> log.log 2>&1 &

# cd ..
# cd ${BASE_DIR}/train
echo "开始训练模型"
nohup python3 train/train_eval.py >> log.log 2>&1 &

echo "预测模型"
nohup python3 train/predict.py >> log.log 2>&1 &
