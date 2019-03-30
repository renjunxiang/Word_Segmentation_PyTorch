import torch
import torch.nn as nn
from torchcrf import CRF

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class BERT_CRF(nn.Module):
    """
    官方模板<https://pytorch.org/tutorials/beginner/nlp/advanced_tutorial.html>
    官方为cpu,要在gpu中运行,所有单独生成的tensor需要.to(device)导入gpu
    """

    def __init__(self, tag_to_ix, mask=False):
        super(BERT_CRF, self).__init__()
        self.hidden_dim = 768  # BERT最后一层维度=768
        self.mask = mask
        self.tag_to_ix = tag_to_ix
        self.tagset_size = len(tag_to_ix)

        self.hidden2tag = nn.Linear(self.hidden_dim, self.tagset_size)
        self.crf = CRF(self.tagset_size, batch_first=True)

    def _get_sentence_features(self, sentences):
        """
        用BERT抽取特征,保持结构统一直接输出,[time_step,768]
        :param sentences:
        :return:
        """
        if self.mask:
            mask_idx = 1 - torch.eq(sentences, 0)
            mask_idx = (mask_idx.sum(dim=2) > 0)
            self.mask_idx = mask_idx

        return sentences

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
        :param sentence:
        :return:
        """
        features = self._get_sentence_features(sentences)
        feats = self._get_sentence_feats(features)
        tags = self._viterbi_decode(feats)
        return tags
