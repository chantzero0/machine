import numpy as np
import pandas as pd
import os
import tensorflow as tf
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer
from sklearn.model_selection import train_test_split
import pickle

# 解决GPU警告并强制使用CPU（可注释掉以下代码以尝试GPU）
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'  # 禁用GPU
physical_devices = tf.config.list_physical_devices('GPU')
if physical_devices:
    try:
        # 设置GPU内存按需增长
        tf.config.experimental.set_memory_growth(physical_devices[0], True)
    except:
        # 处理内存增长设置失败的情况
        pass

# 设置中文显示
plt.rcParams["font.family"] = ["SimHei", "WenQuanYi Micro Hei", "Heiti TC"]
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题


class Config:
    def __init__(self):
        # 关键参数：必须与训练时完全一致
        self.vocab_size = 2000  # 与训练代码中的vocab_size保持一致
        self.max_len = 50
        self.embedding_dim = 100
        self.test_size = 0.2
        self.random_state = 42

        # 路径配置
        self.data_dir = os.path.dirname(os.path.abspath(__file__))  # 当前脚本所在目录
        self.data_path = os.path.join(self.data_dir, 'SMSSpamCollection')
        self.tokenizer_path = os.path.join(self.data_dir, 'tokenizer.pkl')
        self.test_data_path = os.path.join(self.data_dir, 'test_data.csv')
        self.model_path = os.path.join(self.data_dir, 'CNN.h5')


def load_and_preprocess_data(config):
    """加载原始数据并生成测试集和分词器"""
    if not os.path.exists(config.data_path):
        raise FileNotFoundError(f"未找到原始数据集：{config.data_path}")

    # 加载原始数据集
    df = pd.read_csv(config.data_path, sep='\t', names=['label', 'text'])
    df['label'] = df['label'].map({'ham': 0, 'spam': 1})

    # 划分训练集和测试集
    train_df, test_df = train_test_split(
        df,
        test_size=config.test_size,
        random_state=config.random_state,
        stratify=df['label']
    )

    # 保存测试集
    test_df.to_csv(config.test_data_path, index=False)
    print(f"已生成测试集，保存至：{config.test_data_path}")

    # 训练tokenizer（添加OOV处理）
    tokenizer = Tokenizer(
        num_words=config.vocab_size,
        oov_token="<OOV>"  # 关键：统一OOV处理
    )
    tokenizer.fit_on_texts(train_df['text'])

    # 保存分词器
    with open(config.tokenizer_path, 'wb') as f:
        pickle.dump(tokenizer, f)
    print(f"已生成分词器，保存至：{config.tokenizer_path}")

    return tokenizer, test_df


def debug_sequence_issues(sequences, config):
    """调试序列中是否存在超出词表范围的索引"""
    max_index = 0
    out_of_range_count = 0
    out_of_range_indices = []

    for i, seq in enumerate(sequences):
        for j, idx in enumerate(seq):
            if idx >= config.vocab_size:
                out_of_range_count += 1
                out_of_range_indices.append((i, j, idx))
                if idx > max_index:
                    max_index = idx

    print(f"\n===== 序列调试信息 =====")
    print(f"总样本数: {len(sequences)}")
    print(f"发现 {out_of_range_count} 个超出词表范围的索引")
    print(f"最大索引值: {max_index}")

    if out_of_range_count > 0:
        print("\n前10个越界索引示例:")
        for i, j, idx in out_of_range_indices[:10]:
            print(f"样本 {i}, 位置 {j}: 索引 {idx}")

    return out_of_range_count > 0


def main():
    config = Config()

    # 打印配置信息用于调试
    print(f"\n===== 当前配置 =====")
    print(f"词表大小: {config.vocab_size}")
    print(f"最大序列长度: {config.max_len}")
    print(f"数据目录: {config.data_dir}")

    # 加载分词器和测试数据
    try:
        if os.path.exists(config.tokenizer_path) and os.path.exists(config.test_data_path):
            with open(config.tokenizer_path, 'rb') as f:
                tokenizer = pickle.load(f)
            test_data = pd.read_csv(config.test_data_path)
            print("已加载现有测试数据和分词器")

            # 验证加载的分词器配置
            print(f"\n===== 分词器配置 =====")
            print(f"分词器词表大小: {len(tokenizer.word_index) + 1}")  # +1 是因为0是保留索引
            print(f"分词器num_words参数: {tokenizer.num_words}")
            print(f"分词器OOV处理: {tokenizer.oov_token}")

            if tokenizer.num_words != config.vocab_size:
                print(
                    f"警告: 分词器的num_words参数({tokenizer.num_words})与配置中的vocab_size({config.vocab_size})不一致！")
                print("这可能是索引越界的原因，请确保训练和测试使用相同的配置。")
        else:
            raise FileNotFoundError("缺少测试数据或分词器文件，将重新生成")
    except Exception as e:
        print(f"加载文件失败，原因：{e}\n开始生成新的测试数据和分词器...")
        tokenizer, test_data = load_and_preprocess_data(config)

    # 加载模型
    if not os.path.exists(config.model_path):
        raise FileNotFoundError(f"未找到模型文件：{config.model_path}")
    model = load_model(config.model_path)
    print(f"已加载模型：{config.model_path}")

    # 预处理测试数据
    print("\n===== 数据预处理 =====")
    sequences = tokenizer.texts_to_sequences(test_data['text'])
    print(f"已将{len(test_data)}条文本转换为序列")

    # 调试序列问题
    has_issues = debug_sequence_issues(sequences, config)

    if has_issues:
        print("\n警告: 发现序列中存在超出词表范围的索引！")
        print("这通常是由于训练和测试使用了不同的分词器或词表配置导致的。")
        print("继续执行可能会导致错误，建议检查并确保分词器一致性。")

        # 提供修复选项
        response = input("\n是否强制截断超出范围的索引？(y/n): ")
        if response.lower() == 'y':
            print("正在截断超出范围的索引...")
            for i in range(len(sequences)):
                sequences[i] = [idx if idx < config.vocab_size else 0 for idx in sequences[i]]
            print("索引截断完成。")
        else:
            print("程序已停止。请解决词表不一致问题后重试。")
            return

    # 序列填充
    padded_sequences = pad_sequences(
        sequences,
        maxlen=config.max_len,
        padding='post',
        truncating='post'
    )
    print(f"序列填充完成，形状: {padded_sequences.shape}")

    # 模型预测
    print("\n===== 模型预测 =====")
    try:
        predictions = model.predict(padded_sequences)
        predicted_classes = np.argmax(predictions, axis=1)
        true_classes = test_data['label'].values
        print(f"预测完成，共{len(predictions)}个样本")
    except tf.errors.InvalidArgumentError as e:
        print(f"模型预测时出错：{e}")
        print("这通常是由于词表大小不匹配或分词器不一致导致的。")
        print("请确保训练和测试使用相同的vocab_size和分词器。")
        return

    # 计算混淆矩阵
    cm = confusion_matrix(true_classes, predicted_classes)

    # 提取错分样本
    false_positives = test_data[(true_classes == 0) & (predicted_classes == 1)]
    false_negatives = test_data[(true_classes == 1) & (predicted_classes == 0)]

    # 分析高置信度错分案例
    fp_mask = (true_classes == 0) & (predictions[:, 1] > 0.9)
    fn_mask = (true_classes == 1) & (predictions[:, 0] > 0.9)
    fp_high_confidence = false_positives[predictions[fp_mask, 1] > 0.9]
    fn_high_confidence = false_negatives[predictions[fn_mask, 0] > 0.9]

    # 定义错分样本分析函数
    def analyze_misclassified_samples(samples, prediction_probs, is_fp=True):
        results = []
        for i, (_, row) in enumerate(samples.iterrows()):
            text = row['text']
            true_label = row['label']
            pred_label = 1 if is_fp else 0
            confidence = prediction_probs[i][1 if is_fp else 0]

            # 提取文本特征
            word_count = len(text.split())
            exclamation_count = text.count('!')
            url_count = text.lower().count('http') + text.lower().count('www')
            dollar_count = text.count('$')
            free_count = text.lower().count('free')

            results.append({
                'text': text,
                'true_label': true_label,
                'predicted_label': pred_label,
                'confidence': confidence,
                'word_count': word_count,
                'exclamation_count': exclamation_count,
                'url_count': url_count,
                'dollar_count': dollar_count,
                'free_count': free_count
            })
        return pd.DataFrame(results)

    # 分析高置信度错分样本
    fp_analysis = analyze_misclassified_samples(
        fp_high_confidence,
        predictions[fp_mask]
    )
    fn_analysis = analyze_misclassified_samples(
        fn_high_confidence,
        predictions[fn_mask]
    )

    # 可视化错分模式
    plt.figure(figsize=(15, 10))

    # 绘制混淆矩阵
    plt.subplot(2, 2, 1)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['正常邮件', '垃圾邮件'], yticklabels=['正常邮件', '垃圾邮件'])
    plt.xlabel('预测类别')
    plt.ylabel('实际类别')
    plt.title('混淆矩阵')

    # 绘制错分样本特征分布
    plt.subplot(2, 2, 2)
    plt.hist([fp_analysis['word_count'], fn_analysis['word_count']],
             bins=20, label=['假阳性', '假阴性'], alpha=0.7)
    plt.xlabel('单词数量')
    plt.ylabel('样本数')
    plt.title('错分样本的单词数量分布')
    plt.legend()

    # 绘制特殊符号分布
    plt.subplot(2, 2, 3)
    plt.hist([fp_analysis['exclamation_count'], fn_analysis['exclamation_count']],
             bins=10, label=['假阳性', '假阴性'], alpha=0.7)
    plt.xlabel('感叹号数量')
    plt.ylabel('样本数')
    plt.title('错分样本的感叹号分布')
    plt.legend()

    # 绘制高置信度错分样本的置信度分布
    plt.subplot(2, 2, 4)
    plt.hist([fp_analysis['confidence'], fn_analysis['confidence']],
             bins=10, label=['假阳性', '假阴性'], alpha=0.7)
    plt.xlabel('预测置信度')
    plt.ylabel('样本数')
    plt.title('高置信度错分样本的置信度分布')
    plt.legend()

    plt.tight_layout()
    plt.savefig(os.path.join(config.data_dir, 'misclassification_analysis.png'), dpi=300)
    plt.show()

    # 输出分类报告
    print("\n===== 分类报告 =====")
    print(classification_report(true_classes, predicted_classes,
                                target_names=['正常邮件', '垃圾邮件']))

    # 保存错分样本分析结果
    fp_analysis.to_csv(os.path.join(config.data_dir, 'false_positives_analysis.csv'), index=False)
    fn_analysis.to_csv(os.path.join(config.data_dir, 'false_negatives_analysis.csv'), index=False)
    print(f"\n错分样本分析结果已保存至：\n"
          f"{config.data_dir}/false_positives_analysis.csv\n"
          f"{config.data_dir}/false_negatives_analysis.csv")


if __name__ == "__main__":
    main()