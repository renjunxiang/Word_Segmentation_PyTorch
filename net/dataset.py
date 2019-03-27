import torch
from torch.utils.data import Dataset
from flair.data import Sentence

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


# 定义数据读取方式
class DatasetRNN(Dataset):
    def __init__(self, x_seq, y_seq):
        self.x_seq = x_seq
        self.y_seq = y_seq

    def __getitem__(self, index):
        return self.x_seq[index], self.y_seq[index]

    def __len__(self):
        return len(self.x_seq)


class DatasetBERT(Dataset):
    def __init__(self, texts_pad, y_seq, embedding):
        self.embedding = embedding
        self.texts_pad = texts_pad
        self.y_seq = y_seq

    def __getitem__(self, index):
        # 数据中有一些表情乱码,bert出现oov
        try:
            text_pad = ' '.join(self.texts_pad[index])
            sentence = Sentence(text_pad)
            self.embedding.embed(sentence)
            x = torch.cat([token.embedding.unsqueeze(0) for token in sentence], dim=0)
            return x, self.y_seq[index]
        except:
            text_pad = ' '.join(['|'] * len(self.texts_pad[index]))
            sentence = Sentence(text_pad)
            self.embedding.embed(sentence)
            x = torch.cat([token.embedding.unsqueeze(0) for token in sentence], dim=0)
            return x, torch.LongTensor([tag_to_ix['S']] * len(self.texts_pad[index]))

    def __len__(self):
        return len(self.y_seq)
