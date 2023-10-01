BASE_DIR = '/root/autodl-tmp/BertClassNews'
ECHO $BASE_DIR

cd {BASE_DIR}/load_dataset

ECHO "开始处理数据" 
nohup python3 load_data_cls.py

cd ..
cd {BASE_DIR}/train
ECHO "开始训练模型"
nohup python3 train_eval.py

ECHO "预测模型"
nohup python3 predict.py
