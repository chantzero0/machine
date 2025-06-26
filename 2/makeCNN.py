from tensorflow.keras.layers import Input, Embedding, Conv1D, MaxPooling1D, GlobalMaxPooling1D, Dense
from tensorflow.keras.models import Model


def CNN(embedding_layer, config):
    """定义卷积神经网络模型"""
    # 输入层（接收文本序列）
    sequence_input = Input(shape=(config['MAX_SEQUENCE_LENGTH'],), dtype='int32')

    # 嵌入层（使用预训练词向量）
    embedded_sequences = embedding_layer(sequence_input)

    # 卷积+池化层（提取特征）
    x = Conv1D(128, 5, activation='relu')(embedded_sequences)  # 128个5长度的卷积核
    x = MaxPooling1D(2)(x)  # 池化窗口大小2
    x = Conv1D(128, 5, activation='relu')(x)
    x = MaxPooling1D(2)(x)
    x = Conv1D(128, 5, activation='relu')(x)
    x = GlobalMaxPooling1D()(x)  # 全局最大池化（降维）

    # 全连接层（分类）
    x = Dense(128, activation='relu')(x)  # 隐藏层
    preds = Dense(2, activation='softmax')(x)  # 输出层（2类：ham/spam）

    # 编译模型
    model = Model(sequence_input, preds)
    model.compile(
        loss='categorical_crossentropy',  # 多分类交叉熵
        optimizer='rmsprop',
        metrics=['acc']
    )
    model.summary()
    return model