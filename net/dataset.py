import torch
from torch.utils.data import Dataset
from flair.data import Sentence
from data import tag_to_ix

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# 定义数据读取方式
class DatasetRNN(Dataset):
    def __init__(self, x_seq, y_seq):
        self.x_seq = x_seq
        self.y_seq = y_seq

    def __getitem__(self, index):
        return self.x_seq[index], self.y_seq[index]

    def __len__(self):
        return len(self.x_seq)


def collate_fn_RNN(batch):
    batch_x, batch_y, batch_len = [], [], []

    # 列表必须做拷贝,否则原始数据也会发生改变
    for i in batch:
        batch_x.append(i[0][:])
        batch_y.append(i[1][:])
        batch_len.append(len(i[0]))
    len_max = max(batch_len)

    # 按文本长度从大到小排序
    idx = sorted(range(len(batch)), key=lambda x: batch_len[x], reverse=True)
    batch_x = [batch_x[i] for i in idx]
    batch_y = [batch_y[i] for i in idx]
    batch_len = [batch_len[i] for i in idx]

    # 填充<PAD>的编码和标注0
    for i in range(len(batch)):
        batch_x[i] += [0] * (len_max - batch_len[i])
        batch_y[i] += [0] * (len_max - batch_len[i])
    batch_x = torch.LongTensor(batch_x)
    batch_y = torch.LongTensor(batch_y)

    return batch_x, batch_y


class DatasetBERT(Dataset):
    def __init__(self, texts, y_seq, embedding):
        self.embedding = embedding
        self.texts = texts
        self.y_seq = y_seq

    def __getitem__(self, index):
        # 数据中有一些表情乱码,bert出现oov
        try:
            text = ' '.join(self.texts[index])
            sentence = Sentence(text)
            self.embedding.embed(sentence)
            x = torch.cat([token.embedding.unsqueeze(0) for token in sentence], dim=0)
            return x, self.y_seq[index]
        except:
            text = ' '.join(['|'] * len(self.texts[index]))
            sentence = Sentence(text)
            self.embedding.embed(sentence)
            x = torch.cat([token.embedding.unsqueeze(0) for token in sentence], dim=0)
            return x, [tag_to_ix['S']] * len(self.texts[index])

    def __len__(self):
        return len(self.y_seq)


def collate_fn_BERT(batch):
    batch_len = [len(i[0]) for i in batch]
    len_max = max(batch_len)
    batch_x, batch_y = [], []

    # x填充torch.zeros(768),y填充0
    for i in batch:
        batch_x.append(torch.cat([i[0], torch.zeros(len_max - len(i[0]), 768)], dim=0).unsqueeze(0))
        batch_y.append(i[1] + [0] * (len_max - len(i[0])))

    batch_x = torch.cat(batch_x, dim=0)
    batch_y = torch.LongTensor(batch_y)

    return batch_x, batch_y
