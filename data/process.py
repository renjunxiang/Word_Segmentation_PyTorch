import os
import jieba
from keras.preprocessing.text import Tokenizer

jieba.setLogLevel('WARN')

DIR = os.path.dirname(os.path.abspath(__file__))

START_TAG = "<START>"
STOP_TAG = "<STOP>"
PAD_TAG = "<PAD>"
tag_to_ix = {
    START_TAG: 0,
    STOP_TAG: 1,
    PAD_TAG: 2,
    'B': 3, 'M': 4, 'E': 5,
    'S': 6
}


# tag_to_ix = {
#     START_TAG: 0,
#     STOP_TAG: 1,
#     'B': 2, 'M': 3, 'E': 4,
#     'S': 5
# }


class Process(object):
    def __init__(self, num_words=3000):
        self.num_words = num_words
        self.texts_seq = None
        self.texts_tags = None

    def word2tag(self, word):
        """
        词语转为序列标注
        例如:"爱"->"S","北京"->"BE","天安门"->"BME“
        :param word:
        :return:
        """
        if len(word) == 1:
            r = [tag_to_ix['S']]
        else:
            r = [tag_to_ix['B']] + (len(word) - 2) * [tag_to_ix['M']] + [tag_to_ix['E']]
        return r

    def make_tags(self, texts, sentence_len=40):
        num_words = self.num_words
        tokenizer = Tokenizer(num_words=num_words, char_level=True)
        tokenizer.fit_on_texts(texts)

        total_num = len(tokenizer.word_index)
        print('字典大小:', total_num)

        # 计算token编码字典
        if total_num > num_words:
            word_index = {}
            for word in tokenizer.word_index:
                word_id = tokenizer.word_index[word]
                if word_id <= num_words:
                    word_index.update({word: word_id})
        else:
            word_index = tokenizer.word_index
            num_words = total_num
        word_index.update({PAD_TAG: 0})

        # 文本转token编码序列,分词结果转为序列标注
        texts_seq = []
        texts_tags = []
        texts_pad = []
        for text in texts:
            # 生成序列的标注
            text = text[:sentence_len]
            text_cut = jieba.lcut(text)
            text_tags = []
            for word in text_cut:
                text_tags += self.word2tag(word)
            text_tags += [tag_to_ix[PAD_TAG]] * max(0, sentence_len - len(text))
            texts_tags.append(text_tags)

            # 生成序列的编码
            text_pad = list(text) + [PAD_TAG] * max(0, sentence_len - len(text))
            texts_pad.append(text_pad)
            text_seq = [word_index.get(char, num_words + 1) for char in text_pad]
            texts_seq.append(text_seq)

        self.num_words = num_words
        self.word_index = word_index
        self.texts_pad = texts_pad
        self.texts_seq = texts_seq
        self.texts_tags = texts_tags
