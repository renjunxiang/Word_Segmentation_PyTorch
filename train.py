import torch
import torch.optim as optim
from flair.embeddings import BertEmbeddings
from net import BiLSTM_CRF, BERT_CRF, DatasetRNN, collate_fn_RNN, DatasetBERT, collate_fn_BERT
from data import tag_to_ix
import pickle
import os

DIR = os.path.dirname(os.path.abspath(__file__))
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.manual_seed(2)

EMBEDDING_DIM = 256
HIDDEN_DIM = 256
BATCH_SIZE = 64


def train(epochs=5, feature='LSTM', mask=False):
    # 导入预处理标签
    with open(DIR + '/data/texts_tags.pkl', 'rb') as f:
        texts_tags = pickle.load(f)

    # 模型网络
    if feature == 'BERT':
        # 导入文本
        with open(DIR + '/data/texts.pkl', 'rb') as f:
            texts_pad = pickle.load(f)

        # 导入BERT预训练模型
        embedding = BertEmbeddings('bert-base-chinese', '-1', 'mean')

        trainloader = torch.utils.data.DataLoader(
            dataset=DatasetBERT(texts_pad[:-100], texts_tags[:-100], embedding),
            batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn_BERT)

        testloader = torch.utils.data.DataLoader(
            dataset=DatasetBERT(texts_pad[-100:], texts_tags[-100:], embedding),
            batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn_BERT)

        model = BERT_CRF(tag_to_ix=tag_to_ix, mask=mask).to(device)

        # 优化器
        optimizer = optim.Adam(model.parameters(), lr=0.01, weight_decay=1e-4)

        for epoch in range(epochs):
            print('start Epoch: %d\n' % (epoch + 1))
            sum_loss = 0.0
            for i, data in enumerate(trainloader):
                model.zero_grad()
                x_seq_batch, y_seq_batch = data
                x_seq_batch = x_seq_batch.to(device)
                y_seq_batch = y_seq_batch.to(device)

                # 损失函数
                loss = model.neg_log_likelihood(x_seq_batch, y_seq_batch)
                loss.backward()
                optimizer.step()
                # 记录总损失
                sum_loss += loss.item()

                if (i + 1) % 10 == 0:
                    print('Epoch: %d ,batch: %d, loss = %f' % (epoch + 1, i + 1, sum_loss / 10))
                    sum_loss = 0.0

            torch.save(model.state_dict(),
                       './model/%s_%03d.pth' % (feature, epoch + 1))

            # 每跑完一次epoch测试一下准确率
            with torch.no_grad():
                sum_loss = 0.0
                n = 0
                for data in testloader:
                    x_seq_batch, y_seq_batch = data
                    x_seq_batch = x_seq_batch.to(device)
                    y_seq_batch = y_seq_batch.to(device)
                    loss = model.neg_log_likelihood(x_seq_batch, y_seq_batch)
                    sum_loss += loss.item()
                    n += 1
                print('test loss = %f' % (sum_loss / n))
    else:
        # 导入文本编码、词典
        with open(DIR + '/data/word_index.pkl', 'rb') as f:
            word_index = pickle.load(f)
        with open(DIR + '/data/texts_seq.pkl', 'rb') as f:
            texts_seq = pickle.load(f)

        trainloader = torch.utils.data.DataLoader(
            dataset=DatasetRNN(texts_seq[:-100], texts_tags[:-100]),
            batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn_RNN)

        testloader = torch.utils.data.DataLoader(
            dataset=DatasetRNN(texts_seq[-100:], texts_tags[-100:]),
            batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn_RNN)

        # vocab_size还有pad和unknow
        model = BiLSTM_CRF(vocab_size=len(word_index) + 2,
                           tag_to_ix=tag_to_ix,
                           embedding_dim=EMBEDDING_DIM,
                           hidden_dim=HIDDEN_DIM,
                           mask=mask).to(device)

        # 优化器
        optimizer = optim.Adam(model.parameters(), lr=0.01, weight_decay=1e-4)

        for epoch in range(epochs):
            print('start Epoch: %d\n' % (epoch + 1))
            sum_loss = 0.0
            for i, data in enumerate(trainloader):
                model.zero_grad()
                x_seq_batch, y_seq_batch = data
                x_seq_batch = x_seq_batch.to(device)
                y_seq_batch = y_seq_batch.to(device)

                # 损失函数
                loss = model.neg_log_likelihood(x_seq_batch, y_seq_batch)
                loss.backward()
                optimizer.step()
                # 记录总损失
                sum_loss += loss.item()

                if (i + 1) % 10 == 0:
                    print('Epoch: %d ,batch: %d, loss = %f' % (epoch + 1, i + 1, sum_loss / 10))
                    sum_loss = 0.0

            torch.save(model.state_dict(),
                       './model/%s_%03d.pth' % (feature, epoch + 1))

            # 每跑完一次epoch测试一下准确率
            with torch.no_grad():
                sum_loss = 0.0
                n = 0
                for data in testloader:
                    x_seq_batch, y_seq_batch = data
                    x_seq_batch = x_seq_batch.to(device)
                    y_seq_batch = y_seq_batch.to(device)
                    loss = model.neg_log_likelihood(x_seq_batch, y_seq_batch)
                    sum_loss += loss.item()
                    n += 1
                print('test loss = %f' % (sum_loss / n))


if __name__ == '__main__':
    train(epochs=5, feature='BERT', mask=True)
