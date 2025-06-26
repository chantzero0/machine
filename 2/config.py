config = {
    'raw_data': '../SMSSpamCollection',      # 原始数据路径（TSV格式）
    'raw_encoding': 'utf-8',                # 数据编码
    'div_ratio': 0.7,                       # 训练集比例
    'base_path': '../models/',              # 模型保存路径
    'MAX_NUM_WORDS': 2000,                  # 词表最大单词数
    'MAX_SEQUENCE_LENGTH': 50,              # 文本最大长度（截断/补全）
    'tokenizer_path': './CNNtokenizer.pkl', # 分词器保存路径
    'EMBEDDING_DIM': 100,                   # 词向量维度（需与glove一致）
    'glove_path': '../glove/glove.6B.100d.txt'  # 添加GloVe路径配置
}