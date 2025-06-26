from utils import showModel, print_AUC, TOy_val_label, TOy_pred_label
import matplotlib.pyplot as plt

def trainCNN(x_train, y_train, x_val, y_val, embedding_layer, config):
    """训练CNN模型"""
    # 初始化模型
    from makeCNN import CNN
    model = CNN(embedding_layer, config)

    # 训练模型并记录训练历史
    history = model.fit(
        x_train, y_train,
        batch_size=16,
        epochs=5,
        validation_data=(x_val, y_val)
    )

    # 验证集预测
    y_pred_proba = model.predict(x_val, batch_size=16)
    y_val_label = TOy_val_label(y_val)
    y_pred_label = TOy_pred_label(y_pred_proba)

    # 评估并可视化
    showModel(y_val_label, y_pred_label, "卷积神经网络（CNN）")
    print_AUC(y_val_label, y_pred_proba)

    # 绘制损失函数图像
    plt.figure(figsize=(10, 6))
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.show()

    return model