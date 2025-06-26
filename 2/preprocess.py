import csv
import numpy as np
from tensorflow.keras.layers import Embedding
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from utils import logging, saveTokenizer, loadTokenizer, preprocess


def readData(config):
    """读取原始数据（TSV格式）"""
    with open(config['raw_data'], 'r', encoding=config['raw_encoding']) as f:
        raw_data = list(csv.reader(f, delimiter='\t'))  # 按制表符分割
    logging('原始数据示例', raw_data[:3])
    return raw_data


def cleanData(raw_data):
    """清洗数据（文本→预处理后的单词列表，标签→数值）"""
    sms_text = []
    sms_label = []
    label_map = {'spam': 1, 'ham': 0}  # spam=1（垃圾），ham=0（正常）
    for label, text in raw_data:
        sms_text.append(" ".join(preprocess(text)))  # 预处理后拼接为字符串
        sms_label.append(label_map[label])
    logging('预处理后文本示例', sms_text[:3])
    logging('预处理后标签示例', sms_label[:3])
    return sms_text, sms_label


def tokenize(sms_text, config):
    """生成分词器并保存"""
    tokenizer = Tokenizer(num_words=config['MAX_NUM_WORDS'])
    tokenizer.fit_on_texts(sms_text)
    saveTokenizer(tokenizer, config)
    return tokenizer


def categorical(sms_text, sms_label, config):
    """文本转序列+划分训练/验证集"""
    # 文本转数值序列
    tokenizer = tokenize(sms_text, config)
    sequences = tokenizer.texts_to_sequences(sms_text)
    X = pad_sequences(sequences, maxlen=config['MAX_SEQUENCE_LENGTH'])

    # 标签转one-hot
    y = to_categorical(np.array(sms_label))

    # 打乱数据
    indices = np.arange(X.shape[0])
    np.random.shuffle(indices)
    X, y = X[indices], y[indices]

    # 划分训练/验证集
    split_idx = int(len(X) * config['div_ratio'])
    x_train, y_train = X[:split_idx], y[:split_idx]
    x_val, y_val = X[split_idx:], y[split_idx:]
    return x_train, y_train, x_val, y_val


def train_dic(sms_text, config):
    """加载预训练词向量并生成嵌入层"""
    # 加载GloVe词向量（需提前下载glove.6B.100d.txt到../glove/）
    embedding_dic = {}
    with open('../glove/glove.6B.100d.txt', encoding='utf-8') as f:
        for line in f:
            word, *vector = line.split()
            embedding_dic[word] = np.array(vector, dtype='float32')

    # 生成词嵌入矩阵
    tokenizer = tokenize(sms_text, config)
    word_index = tokenizer.word_index
    num_words = min(config['MAX_NUM_WORDS'], len(word_index) + 1)
    embedding_matrix = np.zeros((num_words, config['EMBEDDING_DIM']))

    for word, i in word_index.items():
        if i >= config['MAX_NUM_WORDS']:
            continue
        embedding_vector = embedding_dic.get(word)
        if embedding_vector is not None:
            embedding_matrix[i] = embedding_vector

    # 定义嵌入层（固定预训练权重）
    embedding_layer = Embedding(
        input_dim=num_words,
        output_dim=config['EMBEDDING_DIM'],
        weights=[embedding_matrix],
        input_length=config['MAX_SEQUENCE_LENGTH'],
        trainable=False  # 不更新词向量
    )
    return embedding_layer


def getInputvec(raw_sentence, config):
    """将输入文本转为模型可接受的向量"""
    # 预处理输入文本
    processed_text = " ".join(preprocess(raw_sentence))

    # 文本转序列
    tokenizer = loadTokenizer(config)
    sequence = tokenizer.texts_to_sequences([processed_text])  # 输入为列表

    # 补全到固定长度
    input_vec = pad_sequences(
        sequence,
        maxlen=config['MAX_SEQUENCE_LENGTH'],
        padding='post',  # 尾部补0
        truncating='post'  # 尾部截断
    )
    return input_vec