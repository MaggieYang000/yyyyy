# cail2019_track2
中国法研杯CAIL2019要素抽取任务第三名


### **主要参数**

| 参数名 | 参数值 |
| :------: | :------: |
| 预训练模型 | [BERT_Base_Chinese](https://storage.googleapis.com/bert_models/2018_11_03/chinese_L-12_H-768_A-12.zip) |
| max_length(divorce) | 128 |
| max_length(labor) | 150 |
| max_length(loan) | 200 |
| batch_size | 32 |
| learning_rate | 2e-5 |
| num_train_epochs | 30 |
| alpha(focal loss) | 0.25 |
| gamma(focal loss) | 2 |
| hidden_dim(lstm) | 200 |

**方案介绍**
------
### **任务简介**
根据给定司法文书中的相关段落，识别相应的关键案情要素，其中每个句子对应的类别标签个数不定，属于多标签问题。任务共涉及三个领域，包括婚姻家庭、劳动争议、借款合同。
例如：

| 例句 | 标签 |
| :------: | :------: |
| 高雷红提出诉讼请求：1、判令被告支付原告的工资1630×0.8×4＝5216元； | ["LB2"] |
| 原告范合诉称：原告系被告处职工。 | [] |
| 5、判令被告某甲公司支付2011年9月8日至2012年8月7日未签订劳动合同的二倍工资差额16，298.37元； | ["LB9", "LB6"] |

根据数据集，选用传统的多标签分类方法。bert预训练模型使用的是google开源的bert_base_chinese.

### **任务难点**

* **正负例样本不均衡**
* **有的要素标签正例仅有几条，模型无法学习**

### **解决方案**

#### **focal loss**
减少易分类样本的权重，增加难分类样本的损失贡献值，参数见上表的alpha，gamma

#### **阈值移动**
将比赛的数据集切分为训练集和测试集。先用训练集去训练模型，
然后使用测试集去测试模型，筛选阈值；最后把所有数据拿去训练最后的提交模型，
预测阈值就采用之前筛选出来的阈值。

#### **模型优化**
最后使用的模型是BERT + RCNN，并且RCNN部分的最大池化修改为Attention。
主要方法就是将BERT的输出向量X输入BiLstm，得到一个特征向量H，最后将X和H
拼接送入Attention。

#### **规则**
规则主要是为了修正模型无法学习的要素标签，使用的方式：首先通过
标签的解释说明和包含标签的样本确定规则，规则在python中使用的是正则
表达式；然后针对需要预测的文本，我们先使用正则表达式去匹配，若是
匹配成功，则说明文本包含该规则对应的标签；最后把规则匹配出来的标签与
模型预测的标签取并集，得到最终预测要素集。

规则举例：
> ['.(保证合同|抵押合同|借款合同).(无效|不发生效力).*']
   ，对应的要素是LN12。
 
**否定词规则**

否定词规则的意思是：在采用规则修正的时候，若是句子以一些否定词结尾，规则将不生效。

举例：

> 被告五金公司辩称本案借款合同和保证合同均无效，缺乏法律依据，本院**不予采纳**。

> 实际标签: LN13 LN10

这个句子可以匹配到我们写的LN12的规则：‘.*(保证合同|抵押合同|借款合同).*(无效|不发生效力).*‘

但是因为末尾出现了不予采纳，所以该标签规则不生效，没有LN12。

#### **领域预训练**

bert模型采用的是bert_base_chinese
## 目前情况

因为改代码都是基于TensorFlow 1.x的，并且配置上好像有些问题，还没有完全跑通。

**Reference**
-----
1. [TensorFlow code and pre-trained models for BERT](https://github.com/google-research/bert)
2. [The implementation of focal loss proposed on "Focal Loss for Dense Object Detection" by KM He and support for multi-label dataset.](https://github.com/ailias/Focal-Loss-implement-on-Tensorflow)

