from nlpcda import Randomword
from nlpcda import Similarword
from nlpcda import Homophone
from nlpcda import RandomDeleteChar
from nlpcda import Ner
from nlpcda import CharPositionExchange
from nlpcda import baidu_translate
from nlpcda import EquivalentChar
import logging
import torch
import sys
import os
from torch.utils.data import Dataset, DataLoader
sys.path.insert(0, '/root/autodl-tmp/BertClassNews/config')
from aug_config import *

        
def logger():
    logging.basicConfig(level=logging.INFO)
    return logging.getLogger(__name__)


def save_model(model, config):
    # datPath = '/root/wt/'
    # model_save_path  = datPath + 'BertOrigmodelAllDat_v0.pt'
    model_save_path  = config.wtPath + config.model_dir
    if os.path.exists(model_save_path):
        loaded_paras = torch.load(model_save_path)
        model.load_state_dict(loaded_paras)
        logging.info("## 成功载入已有模型，进行追加训练......")
    else:
        torch.save(model.state_dict(), model_save_path)
    return model

def augment_minority_randomSample(train_df, config):
    oversample = RandomOverSampler(sampling_strategy = config.augment_rate)
    X_over, y_over = oversample.fit_resample(train_df.drop('tag', axis=1), train_df['tag'])
    X_over['tag'] = y_over
    return X_over

def augment_nlpcda(ts, config):
    smw1 = Randomword(create_num=config.rw_cn, change_rate=self.rw_cr)
    smw2 = Similarword(create_num=config.sw_cn, change_rate=self.sw_cr)
    hoe = Homophone(create_num=self.hoe_cn, change_rate=self.hoe_cr)
    smw3 = RandomDeleteChar(create_num=self.rd_cn, change_rate=self.rd_cr)
    s = EquivalentChar(create_num=self.evc_cn, change_rate=self.evc_cr)
    # return s.replace(ts)[0]
    res = s.replace(ts)
    return [smw1.replace(smw2.replace(hoe.replace(smw3.replace(i)[0])[0])[0])[0] for i in res]


def test_Randomword(test_str, create_num=3, change_rate=0.3):
    '''
    随机【（等价）实体】替换，这里是extdata/company.txt ，随机公司实体替换
    :param test_str: 替换文本
    :param create_num: 增强为多少个
    :param change_rate: 文本变化率/最大多少比例会被改变
    :return:
    '''
    smw = Randomword(create_num=create_num, change_rate=change_rate)
    return smw.replace(test_str)


def test_Similarword(test_str, create_num=3, change_rate=0.3):
    '''
    随机【同义词】替换
    :param test_str: 替换文本
    :param create_num: 增强为多少个
    :param change_rate: 文本变化率/最大多少比例会被改变
    :return:
    '''
    smw = Similarword(create_num=create_num, change_rate=change_rate)
    return smw.replace(test_str)


def test_Homophone(test_str, create_num=3, change_rate=0.1):
    '''
    随机【同意/同音字】替换
    :param test_str: 替换文本
    :param create_num: 增强为多少个
    :param change_rate: 文本变化率/最大多少比例会被改变
    :return:
    '''
    hoe = Homophone(create_num=create_num, change_rate=change_rate)
    return hoe.replace(test_str)


def test_RandomDeleteChar(test_str, create_num=3, change_rate=0.1):
    smw = RandomDeleteChar(create_num=create_num, change_rate=change_rate)
    return smw.replace(test_str)



def test_ner():
    ner = Ner(ner_dir_name='../write',
              ignore_tag_list=['O', 'T'],
              data_augument_tag_list=['Cause', 'Effect'],
              augument_size=3, seed=0)
    data_sentence_arrs, data_label_arrs = ner.augment('../write/1.txt')
    print(data_sentence_arrs, data_label_arrs)


def test_CharPositionExchange(test_str, create_num=10, change_rate=0.5):
    smw = CharPositionExchange(create_num=create_num, change_rate=change_rate)
    return smw.replace(test_str)


def test_baidu_translate():
    a = 'Free translation for each platform'
    s = baidu_translate(a, appid='xxx', secretKey='xxx')
    print(s)


def test_EquivalentChar(test_str, create_num=10, change_rate=0.5):
    s = EquivalentChar(create_num=create_num, change_rate=change_rate)
    return s.replace(test_str)

def test():
    ts = '''这是个实体：58同城；今天是2020年3月8日11:40，天气晴朗，天气很不错，空气很好，不差；这个nlpcad包，用于方便一键数据增强，可有效增强NLP模型的泛化性能、减少波动、抵抗对抗攻击'''
    rs1 = test_Randomword(ts)
    rs2 = test_Similarword(ts)
    rs3 = test_Homophone(ts)
    rs4 = test_RandomDeleteChar(ts)
    rs5 = test_EquivalentChar(ts)
    print('随机实体替换>>>>>>')
    for s in rs1:
        print(s)
    print('随机近义词替换>>>>>>')
    for s in rs2:
        print(s)
    print('随机近义字替换>>>>>>')
    for s in rs3:
        print(s)

    print('随机字删除>>>>>>')
    for s in rs4:
        print(s)
    print('等价字替换>>>>>>')
    for s in rs5:
        print(s)
        
if __name__ == '__main__':
    config = aug_Configure()
    ts = 'sssddd jjjhhh'
    print (ts, augment_nlpcda(ts, config))
