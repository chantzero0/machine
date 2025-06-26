import csv
from apply import judge


def test_model(test_data_path):
    correct = 0
    total = 0
    with open(test_data_path, 'r', encoding='utf-8') as f:
        reader = csv.reader(f, delimiter='\t')
        for label, text in reader:
            total += 1
            pred = judge(text)
            true_label = "垃圾邮件" if label == 'spam' else "正常邮件"
            if pred == true_label:
                correct += 1
            print(f"文本：{text}\n真实标签：{true_label}，预测结果：{pred}\n---")

    accuracy = correct / total
    print(f"\n测试完成！总样本数：{total}，准确率：{accuracy:.2%}")


if __name__ == '__main__':
    test_model('../SMSSpamCollection')  # 替换为你的测试数据路径