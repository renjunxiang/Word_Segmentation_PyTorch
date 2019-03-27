import re
import os

DIR = os.path.dirname(os.path.abspath(__file__))


def load_data(minlen=30):
    f = open(DIR + '/xiaohuangji50w_nofenci.conv',
             encoding='utf-8')
    texts = []
    line = True
    while line:
        line = f.readline()
        text = re.sub('M |\n', '', line)
        text = re.sub('\s', '|', text)
        if len(text) >= minlen:
            texts.append(text)
    f.close()
    return texts
