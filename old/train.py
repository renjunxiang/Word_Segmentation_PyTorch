import torch
import torch.optim as optim
from flair.embeddings import BertEmbeddings
from net import BiLSTM_CRF, BERT_CRF, DatasetRNN, DatasetBERT
import pickle
import os

DIR = os.path.dirname(os.path.abspath(__file__))
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.manual_seed(2)

EMBEDDING_DIM = 256
HIDDEN_DIM = 256

START_TAG = "<START>"
STOP_TAG = "<STOP>"
tag_to_ix = {
    START_TAG: 0,
    STOP_TAG: 1,
    'B': 2, 'M': 3, 'E': 4,
    'S': 5
}


def train(epochs=5, feature='LSTM'):
    # 导入预处理标签
    with open(DIR + '/data/texts_tags.pkl', 'rb') as f:
        texts_tags = pickle.load(f)

    # 模型网络
    if feature == 'BERT':
        # 导入文本
        with open(DIR + '/data/texts.pkl', 'rb') as f:
            texts = pickle.load(f)

        # 导入BERT预训练模型
        embedding = BertEmbeddings('bert-base-chinese', '-1', 'mean')

        trainloader = torch.utils.data.DataLoader(
            dataset=DatasetBERT(texts[:-100], texts_tags[:-100], embedding),
            batch_size=1, shuffle=False)

        testloader = torch.utils.data.DataLoader(
            dataset=DatasetBERT(texts[-100:], texts_tags[-100:], embedding),
            batch_size=1, shuffle=False)

        model = BERT_CRF(tag_to_ix=tag_to_ix).to(device)

        # 优化器
        optimizer = optim.Adam(model.parameters(), lr=0.01, weight_decay=1e-4)

        for epoch in range(epochs):
            print('start Epoch: %d\n' % (epoch + 1))
            sum_loss = 0.0
            for i, data in enumerate(trainloader):
                model.zero_grad()
                x_seq_batch, y_seq_batch = data

                # 数据中有一些表情乱码
                if x_seq_batch.size() == torch.Size([1, 1]):
                    continue

                x_seq_batch = x_seq_batch.view([-1, 768]).to(device)
                y_seq_batch = torch.LongTensor(y_seq_batch).view([len(y_seq_batch)]).to(device)

                # 损失函数
                loss = model.neg_log_likelihood(x_seq_batch, y_seq_batch)
                loss.backward()
                optimizer.step()
                # 记录总损失
                sum_loss += loss.item()

                if (i + 1) % 100 == 0:
                    print('Epoch: %d ,batch: %d, loss = %f' % (epoch, i + 1, sum_loss / 100))
                    sum_loss = 0.0

            torch.save(model.state_dict(),
                       '%s/%s_%03d.pth' % ('./model', feature, epoch + 1))

            # 每跑完一次epoch测试一下准确率
            with torch.no_grad():
                sum_loss = 0.0
                for data in testloader:
                    x_seq_batch, y_seq_batch = data
                    x_seq_batch = x_seq_batch.view([-1, 768]).to(device)
                    y_seq_batch = torch.LongTensor(y_seq_batch).view([len(y_seq_batch)]).to(device)
                    loss = model.neg_log_likelihood(x_seq_batch, y_seq_batch)
                    sum_loss += loss.item()
                print('test loss = %f' % (sum_loss / 100))
    else:
        # 导入文本编码、词典
        with open(DIR + '/data/word_index.pkl', 'rb') as f:
            word_index = pickle.load(f)
        with open(DIR + '/data/texts_seq.pkl', 'rb') as f:
            texts_seq = pickle.load(f)

        trainloader = torch.utils.data.DataLoader(
            dataset=DatasetRNN(texts_seq[:-100], texts_tags[:-100]),
            batch_size=1, shuffle=False)

        testloader = torch.utils.data.DataLoader(
            dataset=DatasetRNN(texts_seq[-100:], texts_tags[-100:]),
            batch_size=1, shuffle=False)

        model = BiLSTM_CRF(vocab_size=len(word_index) + 2,
                           tag_to_ix=tag_to_ix,
                           embedding_dim=EMBEDDING_DIM,
                           hidden_dim=HIDDEN_DIM).to(device)

        # 优化器
        optimizer = optim.Adam(model.parameters(), lr=0.01, weight_decay=1e-4)

        for epoch in range(epochs):
            print('start Epoch: %d\n' % (epoch + 1))
            sum_loss = 0.0
            for i, data in enumerate(trainloader):
                model.zero_grad()
                x_seq_batch, y_seq_batch = data
                x_seq_batch = torch.LongTensor(x_seq_batch).view([len(x_seq_batch)]).to(device)
                y_seq_batch = torch.LongTensor(y_seq_batch).view([len(y_seq_batch)]).to(device)

                # 损失函数
                loss = model.neg_log_likelihood(x_seq_batch, y_seq_batch)
                loss.backward()
                optimizer.step()
                # 记录总损失
                sum_loss += loss.item()

                if (i + 1) % 1 == 0:
                    print('Epoch: %d ,batch: %d, loss = %f' % (epoch, i + 1, sum_loss / 1))
                    sum_loss = 0.0

            torch.save(model.state_dict(),
                       '%s/%s_%03d.pth' % ('./model', feature, epoch + 1))

            # 每跑完一次epoch测试一下准确率
            with torch.no_grad():
                sum_loss = 0.0
                for data in testloader:
                    x_seq_batch, y_seq_batch = data
                    x_seq_batch = torch.LongTensor(x_seq_batch)
                    x_seq_batch = x_seq_batch.view([len(x_seq_batch)]).to(device)
                    y_seq_batch = torch.LongTensor(y_seq_batch)
                    y_seq_batch = y_seq_batch.view([len(y_seq_batch)]).to(device)
                    loss = model.neg_log_likelihood(x_seq_batch, y_seq_batch)
                    sum_loss += loss.item()
                print('test loss = %f' % (sum_loss / 100))


if __name__ == '__main__':
    train(epochs=3, feature='BERT')
