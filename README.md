# Word_Segmentation_PyTorch
A Simple Chinese Word Segmentation Tool

[![](https://img.shields.io/badge/Python-3.6-blue.svg)](https://www.python.org/)
[![](https://img.shields.io/badge/torch-1.0.0-brightgreen.svg)](https://pypi.org/project/torch/1.0.0)
[![](https://img.shields.io/badge/pytorch--crf-0.7.2-brightgreen.svg)](https://pypi.org/project/pytorch-crf/0.7.2)
[![](https://img.shields.io/badge/keras-2.2.0-brightgreen.svg)](https://pypi.org/project/keras/2.2.0)
[![](https://img.shields.io/badge/flair-0.4.1-brightgreen.svg)](https://pypi.org/project/flair/0.4.1/)
[![](https://img.shields.io/badge/jieba-0.39-brightgreen.svg)](https://pypi.org/project/jieba/0.39/)

## **项目简介**
最近在研究PyTorch和信息抽取，就拿分词练习下序列标注。<br>
PyTorch的crf学习的差不多了，后续研究下seq2seq生成的方式一步到位做关系抽取~<br>

## **项目更新**
### **2019.3.30**
定长预处理对于分词没问题，但考虑到后续要做实体识别，所以不能暴力的做裁剪：<br>
* 对每个batch按照最长样本做填充，默认的DataLoader没法合并不定长数据，新增了```collate_fn```函数；<br>
* LSTM文本序列和标注都填充0，网络层采用```pack_padded_sequence```和```pad_packed_sequence```；<br>
* 参考LSTM对mask的处理，BERT填充```torch.zeros([768])```似乎更合理，虽然训练的loss比"PAD"填充大而且耗时多一半，但是分词效果要好一些；<br>
* 对于loss的计算，有人认为"PAD"也是标注的一部分，好的模型应该能预测出"PAD"，个人感觉mask好像没啥影响；<br>


### **2019.3.27**
* 采用pytorch-crf模块作为crf层的batch实现，速度比官网demo要快得多，官网demo修改的老版本在old文件夹中；<br>
* 预处理使用了PAD填充，用于batch的对齐，BERT的OOV文本在Dataset中用"|"替换；<br>

### **2019.3.24**
原始版本，参考PyTorch官方的Bilstm+crf范例，如下说明：<br>
* flair的embedding用空格分割，为了避免错误预处理将空格用"|"替代，跳过BERT的OOV文本；<br>
* 对初始化的tensor增加了``` .to(device) ```以便在GPU运行；<br>
* 范例是将文本转为[time_step,1,char_dim]固定每个词的发射概率；考虑句子整体语义我改为[1,time_step,char_dim]，发射概率由全句语义决定：<br>
* 独立出```_get_lstm_features```函数，以便调用预训练模型如BERT：<br>
* 范例没有用batch，训练时逐条反向更新，效率比较低。目前找到的资料也只是在crf里用for循环累加梯度，并没有真正的利用全部显存，后续研究下pytorch-crf或者直接用seq2seq的方式<br>

## **模块结构**
结构比较简单，包括数据、预处理方法、网络模型、训练代码和测试代码：<br>
* **数据**：在data文件夹中，小黄鸡对话数据集，来源<https://github.com/fateleak/dgk_lost_conv>；<br>
* **预处理**：运行get_data.py，会在data文件夹中生成字典、文本编码和序列标注；<br>
* **网络模型**：在net文件夹中，dataset.py是DataLoader读取方式，Bilstm_crf.py和Bert_crf.py是网络结构；<br>
* **模型训练**：train.py，"BERT"和"LSTM"两种方式训练网络；<br>
* **模型推断**：test.py，"BERT"和"LSTM"两种方式进行分词；<br>
* **模型存储**：model文件夹，保存训练的模型；<br>

## **效果展示**
*通过jieba作为分词标签，我采用样本长度大于30的，训练样本只有一万六，对预测结果采用抽取实体的方式。语料涵盖的词汇范围非常小，都是日常用语，BERT的效果要好略于Bilstm，能够很好的拆分科技文本。<br>
**训练过程**<br>
![](https://github.com/renjunxiang/Word_Segmentation_PyTorch/blob/master/picture/train.png)<br>
**Bilstm**<br>
![](https://github.com/renjunxiang/Word_Segmentation_PyTorch/blob/master/picture/Bilstm.png)<br>
**BERT**<br>
![](https://github.com/renjunxiang/Word_Segmentation_PyTorch/blob/master/picture/BERT.png)<br>