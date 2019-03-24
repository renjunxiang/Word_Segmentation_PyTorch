import torch
import torch.nn as nn

# torch.manual_seed(1)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def argmax(vec):
    # return the argmax as a python int
    _, idx = torch.max(vec, 1)
    return idx.item()


# Compute log sum exp in a numerically stable way for the forward algorithm
def log_sum_exp(vec):
    max_score = vec[0, argmax(vec)]
    max_score_broadcast = max_score.view(1, -1).expand(1, vec.size()[1])
    return max_score + torch.log(torch.sum(torch.exp(vec - max_score_broadcast)))


START_TAG = "<START>"
STOP_TAG = "<STOP>"


class BiLSTM_CRF(nn.Module):
    """
    官方模板<https://pytorch.org/tutorials/beginner/nlp/advanced_tutorial.html>
    官方为cpu,要在gpu中运行,所有单独生成的tensor需要.to(device)导入gpu
    """

    def __init__(self, vocab_size, tag_to_ix,
                 embedding_dim=256,
                 hidden_dim=256):
        super(BiLSTM_CRF, self).__init__()
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.vocab_size = vocab_size
        self.tag_to_ix = tag_to_ix
        self.tagset_size = len(tag_to_ix)

        self.word_embeds = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim // 2,
                            num_layers=1, bidirectional=True)

        self.hidden2tag = nn.Linear(hidden_dim, self.tagset_size)

        self.transitions = nn.Parameter(
            torch.randn(self.tagset_size, self.tagset_size))

        self.transitions.data[tag_to_ix[START_TAG], :] = -10000
        self.transitions.data[:, tag_to_ix[STOP_TAG]] = -10000

        self.hidden = self.init_hidden()

    def init_hidden(self):
        return (torch.randn(2, 1, self.hidden_dim // 2),
                torch.randn(2, 1, self.hidden_dim // 2))

    '''
    def _get_lstm_features(self, sentence):
        """
        官方的模板是将一个文本[time_step,char_dim]转为[time_step,1,char_dim]的形式,固定每个词的发射概率
        :param sentence:
        :return:
        """
        self.hidden = self.init_hidden()
        embeds = self.word_embeds(sentence).view(len(sentence), 1, -1)
        lstm_out, self.hidden = self.lstm(embeds, self.hidden)
        lstm_out = lstm_out.view(len(sentence), self.hidden_dim)
        lstm_feats = self.hidden2tag(lstm_out)
        return lstm_feats
    '''

    def _get_sentence_features(self, sentence):
        """
        tensorflow是[batch_size,time_step,char_dim],发射概率由全句语义决定
        个人觉得这种会比较好,可以和关系抽取做联合任务
        也可以直接替换成ELMo或者BERT
        :param sentence:
        :return:
        """
        embeds = self.word_embeds(sentence).view(1, len(sentence), -1)  # tf写法
        lstm_out, self.hidden = self.lstm(embeds)
        features = lstm_out.view(len(sentence), self.hidden_dim)
        return features

    def _get_sentence_feats(self, features):
        feats = self.hidden2tag(features)
        return feats

    def _forward_alg(self, feats):
        """
        计算所有可能的隐藏状态序列得分之和
        :param feats:
        :return:
        """
        # 初始化每个状态得分
        init_alphas = torch.full((1, self.tagset_size), -10000.).to(device)
        init_alphas[0][self.tag_to_ix[START_TAG]] = 0.

        forward_var = init_alphas

        for feat in feats:
            alphas_t = []
            for next_tag in range(self.tagset_size):
                # time_setp到tag的发射概率
                emit_score = feat[next_tag].view(
                    1, -1).expand(1, self.tagset_size)
                # tags到tag的转移概率,动态规划思想
                trans_score = self.transitions[next_tag].view(1, -1)
                # score_next=score_now+transition+emit
                next_tag_var = forward_var + trans_score + emit_score
                # 计算log_sum_exp
                alphas_t.append(log_sum_exp(next_tag_var).view(1))
            # 更新这个time_step结束后的forward_var
            forward_var = torch.cat(alphas_t).view(1, -1)
        terminal_var = forward_var + self.transitions[self.tag_to_ix[STOP_TAG]]
        alpha = log_sum_exp(terminal_var)
        return alpha

    def _score_sentence(self, feats, tags):
        """
        计算给定隐藏状态的序列得分
        :param feats:
        :param tags:
        :return:
        """
        score = torch.zeros(1).to(device)
        tags = torch.cat([torch.LongTensor([self.tag_to_ix[START_TAG]]).to(device), tags])
        for i, feat in enumerate(feats):
            score = score + self.transitions[tags[i + 1], tags[i]] + feat[tags[i + 1]]
        score = score + self.transitions[self.tag_to_ix[STOP_TAG], tags[-1]]
        return score

    def neg_log_likelihood(self, sentence, tags):
        """
        损失函数=所有序列得分-正确序列得分
        :param sentence:
        :param tags:
        :return:
        """
        features = self._get_sentence_features(sentence)
        feats = self._get_sentence_feats(features)
        forward_score = self._forward_alg(feats)
        gold_score = self._score_sentence(feats, tags)
        return forward_score - gold_score

    def _viterbi_decode(self, feats):
        """
        维特比算法寻找最大得分序列，用于推断
        :param feats:
        :return:
        """
        backpointers = []

        init_vvars = torch.full((1, self.tagset_size), -10000.).to(device)
        init_vvars[0][self.tag_to_ix[START_TAG]] = 0

        forward_var = init_vvars
        for feat in feats:
            bptrs_t = []
            viterbivars_t = []

            for next_tag in range(self.tagset_size):
                next_tag_var = forward_var + self.transitions[next_tag]
                best_tag_id = argmax(next_tag_var)
                bptrs_t.append(best_tag_id)
                viterbivars_t.append(next_tag_var[0][best_tag_id].view(1))
            forward_var = (torch.cat(viterbivars_t) + feat).view(1, -1)
            backpointers.append(bptrs_t)

        terminal_var = forward_var + self.transitions[self.tag_to_ix[STOP_TAG]]
        best_tag_id = argmax(terminal_var)
        path_score = terminal_var[0][best_tag_id]

        best_path = [best_tag_id]
        for bptrs_t in reversed(backpointers):
            best_tag_id = bptrs_t[best_tag_id]
            best_path.append(best_tag_id)

        start = best_path.pop()
        assert start == self.tag_to_ix[START_TAG]
        best_path.reverse()
        return path_score, best_path

    def forward(self, sentence):
        """
        前向传播过程
        :param sentence:
        :return:
        """
        sentence_features = self._get_sentence_features(sentence)
        sentence_feats = self._get_sentence_feats(sentence_features)
        score, tag_seq = self._viterbi_decode(sentence_feats)
        return score, tag_seq
