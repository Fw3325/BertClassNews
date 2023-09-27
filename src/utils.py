import torch
from torch.utils.data import Dataset, DataLoader



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

train_texts, val_texts, test_texts = train_df['content'].tolist(), val_df['content'].tolist(), test_df['content'].tolist()
train_labels, val_labels, test_labels = train_df['tag'].tolist(), val_df['tag'].tolist(), test_df['tag'].tolist()


def generateDataloader(df, lbl, BATCH_SIZE = config.batch_size):
    inputs = tokenizer(df, padding=True, truncation=True,return_tensors='pt')
    dataset = BERTDataset(inputs, lbl)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE)
    return dataloader


def remove_part_fPred_train(train_df, train_pred):
    cond_acc = []
    for i,j in train_pred:
        cond_acc.append((i==j) |((i!=j)&(j not in ['本地旅游','基建民生','社会热点'])))
    # cond_acc = pd.Series(cond_acc)
    new_train = train_df[cond_acc]
    new_train = new_train.reset_index(drop = True)
    return new_train

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