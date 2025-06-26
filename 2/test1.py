import unittest
import numpy as np
from unittest.mock import patch
from utils import preprocess, TOy_pred_label, loadTokenizer, getModel
from preprocess import getInputvec
from apply import judge
from config import config
import os
import shutil


class TestSpamClassifier(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        """全局测试准备（仅执行一次）"""
        # 保存原始配置（用于测试后恢复）
        cls.original_config = config.copy()
        # 创建临时目录用于测试模型/分词器存储
        cls.temp_dir = "./test_temp"
        os.makedirs(cls.temp_dir, exist_ok=True)
        config['base_path'] = cls.temp_dir
        config['tokenizer_path'] = os.path.join(cls.temp_dir, "test_tokenizer.pkl")

    @classmethod
    def tearDownClass(cls):
        """全局测试清理（仅执行一次）"""
        # 恢复原始配置
        config.update(cls.original_config)
        # 删除临时目录
        shutil.rmtree(cls.temp_dir, ignore_errors=True)

    def setUp(self):
        """每个测试用例前的准备"""
        # 示例垃圾/正常文本（与项目强相关）
        self.spam_text = "Win free cash now! Limited offer, click link."
        self.ham_text = "Reminder: Team meeting at 2 PM today in conference room A."
        self.empty_text = ""

    # ------------------------ 预处理模块测试 ------------------------
    def test_preprocess_pipeline(self):
        """测试完整预处理流程（清洗→分词→词形还原）"""
        raw_text = "Hello! It's a beautiful day. Running quickly!"
        expected_tokens = ['hello', 'beautiful', 'day', 'run', 'quickly']

        processed_tokens = preprocess(raw_text)
        self.assertEqual(processed_tokens, expected_tokens)

    def test_preprocess_stopwords_shortwords(self):
        """测试停用词和短单词过滤"""
        raw_text = "the and of for in to it go an"  # 停用词+短单词
        processed_tokens = preprocess(raw_text)
        self.assertEqual(processed_tokens, [])  # 应全部被过滤

    def test_preprocess_special_chars(self):
        """测试特殊符号和缩写处理"""
        raw_text = "!!! Earn $$$ 1000/day by working from home !!!"
        processed_tokens = preprocess(raw_text)
        self.assertIn("earn", processed_tokens)  # 清洗后保留核心词
        self.assertIn("working", processed_tokens)  # 词形还原为原形

    # ------------------------ 输入向量生成测试 ------------------------
    def test_getInputvec_shape(self):
        """测试输入向量形状（应符合模型输入要求）"""
        input_vec = getInputvec(self.ham_text, config)
        self.assertEqual(input_vec.shape, (1, config['MAX_SEQUENCE_LENGTH']))  # (1,50)

    def test_getInputvec_truncation(self):
        """测试超长文本截断逻辑（尾部截断）"""
        long_text = "important " * 60  # 生成60个"important"（长度9≥3）
        input_vec = getInputvec(long_text, config)
        # 预处理后序列长度应等于MAX_SEQUENCE_LENGTH（50）
        self.assertEqual(input_vec.shape[1], config['MAX_SEQUENCE_LENGTH'])

    # ------------------------ 标签转换函数测试 ------------------------
    def test_TOy_pred_label_spam(self):
        """测试垃圾邮件概率高时的标签转换"""
        y_pred_proba = np.array([[0.2, 0.8]])  # spam概率0.8
        pred_label = TOy_pred_label(y_pred_proba)[0]
        self.assertEqual(pred_label, "spam")

    def test_TOy_pred_label_ham(self):
        """测试正常邮件概率高时的标签转换"""
        y_pred_proba = np.array([[0.7, 0.3]])  # ham概率0.7
        pred_label = TOy_pred_label(y_pred_proba)[0]
        self.assertEqual(pred_label, "ham")

    # ------------------------ 模型推理集成测试 ------------------------
    @patch('utils.getModel')
    def test_judge_model_failure(self, mock_getModel):
        """测试模型加载失败时的错误处理"""
        mock_getModel.return_value = None
        result = judge(self.spam_text)
        self.assertEqual(result, "模型加载失败")

    def test_judge_known_spam(self):
        """测试已知垃圾邮件的正确分类（需模型已训练）"""
        # 注意：此测试需要依赖已训练的模型文件（CNN.pkl）在测试目录中
        # 实际执行前需先运行train.py生成模型
        result = judge(self.spam_text)
        self.assertEqual(result, "垃圾邮件")

    def test_judge_known_ham(self):
        """测试已知正常邮件的正确分类"""
        result = judge(self.ham_text)
        self.assertEqual(result, "正常邮件")

    def test_judge_empty_text(self):
        """测试空文本的边界情况"""
        result = judge(self.empty_text)
        self.assertEqual(result, "正常邮件")  # 预处理后无有效词，判为正常

    # ------------------------ 分词器测试 ------------------------
    def test_tokenizer_persistence(self):
        """测试分词器保存与加载的一致性"""
        # 模拟训练时的分词器生成
        from preprocess import tokenize
        test_texts = ["test sentence one", "test sentence two"]
        tokenizer = tokenize(test_texts, config)
        # 加载分词器并验证
        loaded_tokenizer = loadTokenizer(config)
        self.assertEqual(loaded_tokenizer.word_index, tokenizer.word_index)


if __name__ == '__main__':
    unittest.main()