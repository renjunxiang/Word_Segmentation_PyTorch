import torch
from torch.utils.data import Dataset
from flair.data import Sentence


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
    def __init__(self, texts, y_seq, embedding):
        self.embedding = embedding
        self.texts = texts
        self.y_seq = y_seq

    def __getitem__(self, index):
        text = ' '.join(self.texts[index])
        sentence = Sentence(text)

        # 数据中有一些表情乱码,bert出现oov
        try:
            self.embedding.embed(sentence)
            x = torch.Tensor([token.embedding.numpy() for token in sentence])
        except:
            x = torch.Tensor([0])
        return x, self.y_seq[index]

    def __len__(self):
        return len(self.y_seq)
