import torch
import torch.nn as nn
from torchcrf import CRF

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class BiLSTM_CRF(nn.Module):
    """
    官方模板<https://pytorch.org/tutorials/beginner/nlp/advanced_tutorial.html>
    官方没有batch,速度比较慢。crf采用pytorch-crf<https://pytorch-crf.readthedocs.io>
    """

    def __init__(self, vocab_size, tag_to_ix,
                 embedding_dim=256,
                 hidden_dim=256,
                 mask=False):
        super(BiLSTM_CRF, self).__init__()
        self.vocab_size = vocab_size
        self.mask = mask
        self.hidden_dim = hidden_dim
        self.tagset_size = len(tag_to_ix)

        self.word_embeds = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim // 2,
                            num_layers=1, bidirectional=True,
                            batch_first=True)
        self.hidden2tag = nn.Linear(hidden_dim, self.tagset_size)
        self.hidden = self.init_hidden()
        self.crf = CRF(self.tagset_size, batch_first=True)

    def init_hidden(self):
        return (torch.randn(2, 1, self.hidden_dim // 2),
                torch.randn(2, 1, self.hidden_dim // 2))

    def _get_sentence_features(self, sentences):
        """
        [batch_size,time_step,char_dim],发射概率由全句语义决定
        个人觉得这种会比较好,可以和关系抽取做联合任务
        也可以直接替换成ELMo或者BERT
        :param sentences:
        :return:
        """
        embeds = self.word_embeds(sentences)
        if self.mask:
            mask_idx = 1 - torch.eq(sentences, 0)
            self.mask_idx = mask_idx
            lengths = mask_idx.sum(dim=1)
            embeds = nn.utils.rnn.pack_padded_sequence(input=embeds,
                                                       lengths=lengths,
                                                       batch_first=True)
            features, self.hidden = self.lstm(embeds)
            features = nn.utils.rnn.pad_packed_sequence(sequence=features,
                                                        batch_first=True,
                                                        padding_value=0)[0]
        else:
            features, self.hidden = self.lstm(embeds)

        return features

    def _get_sentence_feats(self, features):
        feats = self.hidden2tag(features)

        return feats

    def neg_log_likelihood(self, sentences, tags):
        """
        损失函数=所有序列得分-正确序列得分
        :param sentence:
        :param tags:
        :return:
        """
        features = self._get_sentence_features(sentences)
        feats = self._get_sentence_feats(features)
        if self.mask:
            loss = -self.crf(feats, tags, self.mask_idx, reduction='mean')
        else:
            loss = -self.crf(feats, tags, reduction='mean')

        return loss

    def _viterbi_decode(self, batch_feats):
        """
        维特比算法寻找最大得分序列，用于推断
        :param batch_feats:
        :return:
        """
        best_path = self.crf.decode(batch_feats)
        return best_path

    def forward(self, sentences):
        """
        前向传播过程
        :param sentences:
        :return:
        """
        features = self._get_sentence_features(sentences)
        feats = self._get_sentence_feats(features)
        tags = self._viterbi_decode(feats)
        return tags
