from config import config
import preprocess
from modelCNN import trainCNN
from utils import saveModel

if __name__ == '__main__':
    # 1. 读取并清洗数据
    raw_data = preprocess.readData(config)
    sms_text, sms_label = preprocess.cleanData(raw_data)

    # 2. 划分训练/验证集并生成序列
    x_train, y_train, x_val, y_val = preprocess.categorical(sms_text, sms_label, config)

    # 3. 生成预训练词嵌入层
    embedding_layer = preprocess.train_dic(sms_text, config)

    # 4. 训练CNN模型
    model = trainCNN(x_train, y_train, x_val, y_val, embedding_layer, config)

    # 5. 保存模型
    saveModel(model, "CNN", config)