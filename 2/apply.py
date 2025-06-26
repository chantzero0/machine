import argparse
from utils import getModel, TOy_pred_label
from config import config
from preprocess import getInputvec


def judge(text):
    """判断输入文本是否为垃圾邮件"""
    # 文本向量化
    input_vec = getInputvec(text, config)

    # 加载模型并预测
    model = getModel('CNN', config)
    if not model:
        return "模型加载失败"

    pred_proba = model.predict(input_vec)
    pred_label = TOy_pred_label(pred_proba)[0]  # 取第一个预测结果

    return "垃圾邮件" if pred_label == 'spam' else "正常邮件"


if __name__ == '__main__':
    # 命令行参数解析
    parser = argparse.ArgumentParser(description='垃圾邮件分类器')
    parser.add_argument('-t', '--text', required=True, help='待分类的邮件文本')
    args = parser.parse_args()

    # 执行预测并输出结果
    result = judge(args.text)
    print(f'预测结果：{result}')