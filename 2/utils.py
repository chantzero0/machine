import re
import numpy as np
import pickle
import os
from nltk import word_tokenize, pos_tag
from nltk.corpus import stopwords, wordnet
from nltk.stem import WordNetLemmatizer
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc
import matplotlib.pyplot as plt


# ------------------------ 文本预处理 ------------------------
def regUse(text):
    """正则清洗文本（去标点、缩写、HTML符号）"""
    text = re.sub(r"[,.?!\":]", '', text)  # 去标点
    text = re.sub(r"'\w*\s", ' ', text)  # 去缩写（如it's -> it）
    text = re.sub(r"#?&.{1,3};", '', text)  # 去HTML符号（如&nbsp;）
    return text.lower()  # 转小写


def sampleSeg(text):
    """简单分词（去停用词、短单词）"""
    tokens = [word for word in word_tokenize(text)
              if word not in stopwords.words('english') and len(word) >= 3]
    return tokens


def get_wordnet_pos(treebank_tag):
    """辅助函数：词性标签转wordnet格式"""
    if treebank_tag.startswith('J'):
        return wordnet.ADJ
    elif treebank_tag.startswith('V'):
        return wordnet.VERB
    elif treebank_tag.startswith('N'):
        return wordnet.NOUN
    elif treebank_tag.startswith('R'):
        return wordnet.ADV
    else:
        return wordnet.NOUN  # 默认名词


def lemSeg(tokens):
    """词形还原（基于词性）"""
    lemmatizer = WordNetLemmatizer()
    return [lemmatizer.lemmatize(word, pos=get_wordnet_pos(pos))
            for word, pos in pos_tag(tokens)]


def preprocess(text):
    """完整文本预处理流程：清洗→分词→词形还原"""
    text = regUse(text)
    tokens = sampleSeg(text)
    tokens = lemSeg(tokens)
    return tokens


# ------------------------ 模型工具 ------------------------
# utils.py中的saveModel函数（修改后）
def saveModel(model, name, config):
    base_path = config['base_path']
    os.makedirs(base_path, exist_ok=True)
    file_path = os.path.join(base_path, f'{name}.h5')  # 保存为HDF5格式（.h5）
    model.save(file_path)  # 使用Keras原生保存方法
    print(f'{name}模型保存成功（路径：{file_path}）')

# utils.py中的getModel函数（修改后）
import tensorflow as tf

def getModel(name, config):
    file_path = os.path.join(config['base_path'], f'{name}.h5')  # 匹配保存的.h5文件
    try:
        return tf.keras.models.load_model(file_path)  # 使用Keras原生加载方法
    except Exception as e:
        print(f'加载模型失败：{e}')
        return None

# ------------------------ 评估工具 ------------------------
def TOy_val_label(y_val):
    """将one-hot标签转为文本标签（ham/spam）"""
    return np.where(y_val[:, 1] == 1, 'spam', 'ham')


def TOy_pred_label(y_pred):
    """将模型预测概率转为文本标签"""
    pred_indices = np.argmax(y_pred, axis=1)
    return np.where(pred_indices == 1, 'spam', 'ham')


def showModel(y_true, y_pred, model_name):
    """打印混淆矩阵和分类报告"""
    print(f'\n===== {model_name} 评估结果 =====')
    print('混淆矩阵：')
    print(confusion_matrix(y_true, y_pred))
    print('\n分类报告：')
    print(classification_report(y_true, y_pred))


def print_AUC(y_true, y_pred_proba):
    """绘制AUC-ROC曲线"""
    y_scores = y_pred_proba[:, 1]  # spam类的概率
    fpr, tpr, _ = roc_curve(y_true, y_scores, pos_label='spam')
    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=(6, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2,
             label=f'ROC曲线 (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('假正例率 (FPR)')
    plt.ylabel('真正例率 (TPR)')
    plt.title('垃圾邮件分类器ROC曲线')
    plt.legend(loc="lower right")
    plt.show()


# ------------------------ 其他工具 ------------------------
def logging(title, content):
    """打印带格式的日志"""
    print(f'\n-------- {title} --------')
    print(content)


def saveTokenizer(tokenizer, config):
    """保存分词器"""
    with open(config['tokenizer_path'], 'wb') as f:
        pickle.dump(tokenizer, f)


def loadTokenizer(config):
    """加载分词器"""
    with open(config['tokenizer_path'], 'rb') as f:
        return pickle.load(f)