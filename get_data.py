from data import load_data, Process
import os
import pickle

DIR = os.path.dirname(os.path.abspath(__file__))

texts = load_data(minlen=30)
print('样本数:%d' % len(texts))
process = Process(num_words=3000)
process.make_tags(texts)
with open(DIR + '/data/texts_tags.pkl', 'wb') as f:
    pickle.dump(process.texts_tags, f)
with open(DIR + '/data/texts.pkl', 'wb') as f:
    pickle.dump(process.texts, f)
with open(DIR + '/data/word_index.pkl', 'wb') as f:
    pickle.dump(process.word_index, f)

print(len(texts),len(process.texts_tags))
print(process.num_words)
