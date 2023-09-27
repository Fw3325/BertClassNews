import re
import pandas as pd
from sklearn.model_selection import train_test_split
import warnings

datPath = '/root/autodl-tmp/BertClassNews/data/'

warnings.filterwarnings("ignore")
#%%


Trpath = datPath + 'train_cls-sample.txt'
Testpath = datPath + 'dev_cls-sample.txt'

def read_process_cls_dat(Readpath):
    try:
        with open(Readpath) as f:
            text = f.read()
    except UnicodeDecodeError:
        with open(Readpath, errors='ignore') as f:
            text = f.read()
    pattern = r'[\t\n]'
    result = re.split(pattern, text)
    contentIndx = 0
    lblIndx = 1
    n = len(result) - 1
    content, label = [], []
    while lblIndx < n:
        content.append(result[contentIndx])
        label.append(result[lblIndx])
        contentIndx += 2
        lblIndx += 2
    df = pd.DataFrame()
    df['tag'] = label
    df['content'] = content
    df['len'] = df['content'].apply(lambda x: sum([i.isalpha() for i in x]))
    df = df[df['len']>27]
    return df

def train_val_test_df(train_val_df, test_df, SavePath, is_save = True):

    
    train_df, val_df = train_test_split(train_val_df, test_size=0.2, stratify=train_val_df['tag'], random_state=42)
    lblEncode = {j:i for i,j in zip(range(len(train_df['tag'].unique())), train_df['tag'].unique())}
    reverse_lblEncode = {lblEncode[i]:i for i in lblEncode}
    if is_save == True:
        save_file(train_df, SavePath + 'train_cls.json')
        save_file(val_df, SavePath + 'val_cls.json')
        save_file(test_df, SavePath + 'test_cls.json')

    train_df['tag'] = train_df['tag'].map(lblEncode)
    val_df['tag'] = val_df['tag'].map(lblEncode)
    test_df['tag'] = test_df['tag'].map(lblEncode)
    return train_df, val_df, test_df, lblEncode, reverse_lblEncode

def save_file(df, path):
    df.to_json(path)

if __name__ == '__main__':
    train_val_df  = read_process_cls_dat(Trpath)
    test_df = read_process_cls_dat(Testpath)
    train_df, val_df, test_df, lblEncode, reverse_lblEncode = train_val_test_df(train_val_df, test_df,datPath, is_save = False)
    print (train_df.shape, test_df.shape, val_df.shape)
    
    