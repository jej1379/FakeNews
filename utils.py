# -*- coding:utf-8 -*-
# git add . ; git commit -m "message" ; git push; git push -u --force origin master

import os, json, logging
from nltk.tokenize import word_tokenize
from collections import Counter
import pandas as pd
import numpy as np
import re, pickle, glob
from torch.utils.data import TensorDataset, DataLoader
import torch
try:
    from nltk.corpus import stopwords
except:
    import nltk
    nltk.download('stopwords')
    nltk.download('punkt')
    from nltk.corpus import stopwords

def logger_fn(name, fnm, level=logging.INFO):
    logger = logging.getLogger(name)
    logger.setLevel(level)
    fh = logging.FileHandler(fnm, mode='w')
    #fh.setFormatter(logging.Formatter('%(asctime)s: %(message)s'))
    logger.addHandler(fh)
    return logger

def save_ckpt(print_datetime, epoch, model, model_name, acc, bestK=3):
    ckpt_path='./ckpt/%s/' %print_datetime
    if not os.path.exists(ckpt_path): os.makedirs(ckpt_path)
    path=ckpt_path+'%s_%d_%.2f.pt' %(model_name, epoch, acc*100,)
    torch.save(model.state_dict(), path)
    if len(glob.glob(ckpt_path)) <= bestK: return None

    while len(glob.glob(ckpt_path)) == bestK :
        ckpts=glob.glob(ckpt_path)
        idx = np.argsort([float(ckpt.replace('.pt', '').split('_')[-1]) for ckpt in ckpts])
        os.remove(ckpts[idx[0]])
    print('%d ckpts in %s' %(len(glob.glob(ckpt_path)), ckpt_path))

def make_vocab(corpus, vocab_size=10000):
    if os.path.exists('./data/word2idx.json'):
        word2idx = json.load(open('./data/word2idx.json'))
        return word2idx

    vocab = [w for w, _ in Counter(corpus).most_common(vocab_size-1)]
    print('# total_words = %d, vocab size = %d' %(len(set(corpus)), vocab_size))

    word2idx = {word: i for i, word in enumerate(vocab, 1)}
    word2idx.update({'<unk>': 0})
    json.dump(word2idx, open('./data/word2idx.json', 'w'))
    print('word2idx.json saved')
    return word2idx

def load_glove(only_word2idx=False, embedding_dim=200):
    vocab=np.loadtxt('./data/glove.6B.%dd.txt' %embedding_dim, delimiter=" ", usecols=[0], encoding='utf-8', dtype='str')
    if os.path.exists('./data/word2idx.json'):
        word2idx = json.load(open('./data/word2idx.json'))
    else:
        word2idx = {word: i for i, word in enumerate(vocab, 1)}
        word2idx.update({'<unk>': 0})
        json.dump(word2idx, open('./data/word2idx.json', 'w'))
        print('word2idx.json saved')
    if only_word2idx: return word2idx

    embedding = np.zeros([len(vocab)+1, embedding_dim])
    embedding[0] = np.zeros(embedding_dim)
    for line in open('./data/glove.6B.%dd.txt' %embedding_dim, 'rb'):
        values = line.split()
        embedding[word2idx.get(values[0].decode(encoding="utf-8"), 0)] = np.asarray(values[1:], 'float32')
    return embedding

def split_data():
    data_dir = './data/'
    total_idx = {'train': pickle.load(open(data_dir+'train_idx.pkl','rb')),
                 'valid': pickle.load(open(data_dir+'valid_idx.pkl','rb')),
                 'test': pickle.load(open(data_dir+'test_idx.pkl','rb'))}
    total = {'train': {'id':[], 'sent':[], 'label':[]},
            'valid': {'id':[], 'sent':[], 'label':[]},
            'test': {'id':[], 'sent':[], 'label':[]}}
    stop_words = set(stopwords.words('english'))
    label_set = set()
    if not os.path.exists('./data/word2idx.json'): load_glove(only_word2idx=True)
    word2idx = json.load(open('./data/word2idx.json', 'r'))
    df = pd.read_csv(data_dir+'fake_or_real_news.csv', header=0, names=["id", "title", "text", "label"])
    for idx, row in df.iterrows():
        id = row[0]
        label = row[-1]
        label_set.add(label)
        sent = [word2idx.get(w.lower(), 0) for w in word_tokenize(row[1]) if (w not in stop_words) and (re.search('[a-zA-Z0-9]', w))] + \
               [word2idx.get(w.lower(), 0) for w in word_tokenize(row[2]) if (w not in stop_words) and (re.search('[a-zA-Z0-9]', w))]
        for k, v in total_idx.items():
            if idx in v:
                total[k]['id'].append(id)
                total[k]['label'].append(label)
                total[k]['sent'].append(sent)
                continue
        if idx % 1000 == 0: print('%d th line processed' %idx)

    label2idx = {l: i for i, l in enumerate(label_set)}
    json.dump(label2idx, open('./data/label2idx.json', 'w'))

    for k, v in total.items():
        pickle.dump(v, open(data_dir+'%s.pkl' %k, 'wb'))
        print('%s pickle saved' %k )
'''
def split_data_pre(vocab_size):
    data_dir = './data/'
    total_idx = {'train': pickle.load(open(data_dir+'train_idx.pkl','rb')),
                 'valid': pickle.load(open(data_dir+'valid_idx.pkl','rb')),
                 'test': pickle.load(open(data_dir+'test_idx.pkl','rb'))}
    total = {'train': {'id':[], 'sent':[], 'label':[]},
            'valid': {'id':[], 'sent':[], 'label':[]},
            'test': {'id':[], 'sent':[], 'label':[]}}
    stop_words = set(stopwords.words('english'))
    corpus, label_set = [], set()
    df = pd.read_csv(data_dir+'fake_or_real_news.csv', header=0, names=["id", "title", "text", "label"])
    for idx, row in df.iterrows():
        id = row[0]
        label = row[-1]
        label_set.add(label)
        sent = [w.lower() for w in word_tokenize(row[1]) if (w not in stop_words) and (re.search('[a-zA-Z0-9]', w))] + \
               [w.lower() for w in word_tokenize(row[2]) if (w not in stop_words) and (re.search('[a-zA-Z0-9]', w))]
        corpus.extend(sent)
        for k, v in total_idx.items():
            if idx in v:
                total[k]['id'].append(id)
                total[k]['label'].append(label)
                total[k]['sent'].append(sent)
                continue
        if idx % 1000 == 0: print('%d th line processed' %idx)

    label2idx = {l: i for i, l in enumerate(label_set)}
    json.dump(label2idx, open('./data/label2idx.json', 'w'))

    make_vocab(corpus, vocab_size)
    for k, v in total.items():
        pickle.dump(v, open(data_dir+'%s.pkl' %k, 'wb'))
        print('%s pickle saved' %k )
'''
def data_iter(seq_len, batch_size, vocab_size, shuffle=True, typ='train'):
    if not os.path.exists('./data/%s.pkl' %typ):
        split_data()
    data = pickle.load(open('./data/%s.pkl' %typ, 'rb'))
    label2idx = json.load(open('./data/label2idx.json', 'r'))
    sents = []
    for sent in data['sent']:
        # if sentence length < seq_len, do zero-padding
        sents.append([w for i, w in enumerate(sent) if i < seq_len] + [0]*(seq_len-len(sent)))
    data_label = [label2idx[lab] for lab in data['label']]

    dt = TensorDataset(*[torch.tensor(data['id']), torch.tensor(sents), torch.tensor(data_label)])
    return DataLoader(dt, batch_size, shuffle)

if __name__ == '__main__':
    embedding=load_glove()