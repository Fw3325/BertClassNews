import time
import subprocess
import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn import CrossEntropyLoss
from transformers import BertModel
from transformers import BertTokenizer, BertForSequenceClassification, AdamW, get_linear_schedule_with_warmup
from tqdm import tqdm
import sklearn.model_selection
import pandas as pd
import logging


class Config(object):

    """配置参数"""
    def __init__(self):
        self.datPath = '/root/autodl-tmp/BertClassNews/data/'
        self.model_dir = 'BertRetrainNewAllDat_v1.pt'
        self.wtPath = '/root/autodl-tmp/BertClassNews/wt/'
        self.batch_size = 16                                           # mini-batch大小
        self.lblEncode = {'本地旅游': 0,
                             '通报查处': 1,
                             '基建民生': 2,
                             '社会热点': 3,
                             '暖新闻': 4,
                             '人事任免': 5,
                             '政策类型': 6,
                             '产业金融': 7,
                             '人文历史': 8,
                             '数据排名': 9}
        self.reverse_lblEncode= {0: '本地旅游',
                                 1: '通报查处',
                                 2: '基建民生',
                                 3: '社会热点',
                                 4: '暖新闻',
                                 5: '人事任免',
                                 6: '政策类型',
                                 7: '产业金融',
                                 8: '人文历史',
                                 9: '数据排名'}
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.num_epochs = 3
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')
        self.model = BertForSequenceClassification.from_pretrained('bert-base-chinese',num_labels=len(self.lblEncode))

#         self.model_name = 'bert'
#         self.class_list = ['体育', '军事', '娱乐', '政治', '教育', '灾难', '社会', '科技', '财经', '违法']          # 类别名单
#         self.save_path = './THUCNews/saved_dict/bert.ckpt'        # 模型训练结果
#         self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')   # 设备

#         self.require_improvement = 1000                                 # 若超过1000batch效果还没提升，则提前结束训练
#         self.num_classes = len(self.class_list)                         # 类别数
#         self.num_epochs = 3                                             # epoch数
#         self.pad_size = 32                                              # 每句话处理成的长度(短填长切)
#         self.learning_rate = 5e-5                                       # 学习率
#         self.bert_path = './bert_pretrain'
#         self.tokenizer = BertTokenizer.from_pretrained(self.bert_path)
#         self.hidden_size = 768
        
result = subprocess.run('bash -c "source /etc/network_turbo && env | grep proxy"', shell=True, capture_output=True, text=True)
output = result.stdout
for line in output.splitlines():
    if '=' in line:
        var, value = line.split('=', 1)
        os.environ[var] = value
        
class BERTDataset(Dataset):

    def __init__(self, inputs, labels):
        self.inputs = inputs
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: val[idx] for key, val in self.inputs.items()}
        item['labels'] = self.labels[idx]
        return item

    def __len__(self):
        return len(self.labels)
    
def get_token_model(n_labels):
    tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')

    model = BertForSequenceClassification.from_pretrained('bert-base-chinese',num_labels=n_labels)
    return tokenizer, model


def evaluate(model, dataloader, config):
    model.to(config.device)
    model.eval()
    final_pred = []
    total = 0
    correct = 0
    pred = {i:[] for i in range(len(config.lblEncode))}
    # pred = {i:[] for i in [0,2,3]}
    with torch.no_grad():

        for batch in dataloader:
            inputs = torch.stack([t for t in batch['input_ids']]).to(device)
            labels = torch.tensor(batch['labels']).to(device)
            attention_mask =torch.stack([t for t in batch['attention_mask']]).to(device)
            outputs = model(input_ids=inputs, attention_mask=attention_mask,
                      labels=labels)
            predictions = outputs[1].argmax(dim=1)
            for i in range(len(predictions)):
                predRes = predictions[i].item()
                pred[predRes].append((predictions[i] == labels[i]).item())
                final_pred.append((config.reverse_lblEncode[predRes], config.reverse_lblEncode[labels[i].item()]))
                # final_pred.append((predRes, labels[i].item()))

            correct += (predictions == labels).sum().item()
            total += labels.shape[0]
      
    accuracy = correct/total
    res = {config.reverse_lblEncode[i]:sum(pred[i])/len(pred[i]) if len(pred[i]) > 0 else np.nan for i in pred }
    return accuracy, res, final_pred


def generateDataloader(df, lbl, Tokenizer, BATCH_SIZE = 16):
    inputs = Tokenizer(df, padding=True, truncation=True,return_tensors='pt')
    dataset = BERTDataset(inputs, lbl)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE)
    return dataloader

def save_model(model, wtPath, model_name = 'BertOrigmodelAllDat_v0.pt'):
    # datPath = '/root/wt/'
    # model_save_path  = datPath + 'BertOrigmodelAllDat_v0.pt'
    model_save_path  = wtPath + model_name
    if os.path.exists(model_save_path):
        loaded_paras = torch.load(model_save_path)
        model.load_state_dict(loaded_paras)
        logging.info("## 成功载入已有模型，进行追加训练......")
    else:
        torch.save(model.state_dict(), model_save_path)
    return model

def main():
    config = Config()
    test_df = pd.read_json(config.datPath + 'test_cls.json') 
    # model, tokenizer = get_token_model(len(config.lblEncode))
    test_dataloader = generateDataloader(test_df['content'].tolist(), test_df['tag'].tolist(), Tokenizer= config.tokenizer, BATCH_SIZE = config.batch_size)
    model = config.model
    model = save_model(model, config.wtPath, model_name = config.model_dir)
    test_accuracy, test_cat_acc, test_pred = evaluate(model, test_dataloader, config.lblEncode, config.reverse_lblEncode)
    print("Test Accuracy:", test_accuracy, "Test Cat Accuracy:", test_cat_acc)
    
if __name__ == "__main__":
    main()