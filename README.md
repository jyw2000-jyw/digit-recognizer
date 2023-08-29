# Westonline3
西二在线3轮考核

# 简介

kaggle上发起的一个“Digit Recognizer”手写数字识别竞赛。链接：[kaggle手写数字识别竞赛](https://www.kaggle.com/c/digit-recognizer)

## 1.数据介绍

1. 数据文件train.csv和test.csv包含手绘数字的灰度图像，从0到9。
2. 每张图像高28像素，宽28像素，总共784像素。每个像素都有一个与之相关联的像素值，表示该像素的明度或暗度，数字越大表示越暗。这个像素值是0到255之间的整数(包括255)。
3. 训练数据集(train.csv)有785列。第一列称为“label”，是用户绘制的数字。其余的列包含关联图像的像素值。训练集中的每个像素列都有一个类似pixel x的名称，其中x是0到783之间的整数(包括0和783)。为了在图像上定位这个像素，假设我们将x分解为x = i * 28 + j，其中i和j都是0到27之间的整数，包括0和27。然后，pixel x位于一个28 x 28矩阵的第i行和第j列，(索引为0)。
构成的图片大概这种格式：

![在这里插入图片描述](https://img-blog.csdnimg.cn/20210217174612147.png)

4. 测试数据集(test.csv)除了不包含label列外，与训练集相同。
5. 对于测试集中的28000张图像，输出包含ImageId和预测数字的单行。最后作为最后提交的submission。这个方法最终获得了0.97003的评分。![在这里插入图片描述](https://img-blog.csdnimg.cn/20210217174932800.png)


## 2.代码
### 库的引入

```python
import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import precision_score
from sklearn.model_selection import train_test_split
```
### 导入数据

```python
train = pd.read_csv("train1.csv")
test = pd.read_csv("test1.csv")
```
数据集可自行下载：[数据集下载](https://www.kaggle.com/c/digit-recognizer/data)

### 数据初探

```python
train.shape
train.head()

test.shape
test.head()

numbers = train['label']
numbers.head(10)

train=train.drop('label', axis=1)
train.head()
```

```python
fre = numbers.value_counts()
fre.sort_index(inplace=True)

for x, y in enumerate(fre.values):
    plt.text(x-0.4, y, "%s" %y)
    
plt.title("Frequency Histogram of Numbers in Training Data")
plt.xlabel("Number")
plt.ylabel("Frequency")
fre.plot.bar()
plt.show()
```
![在这里插入图片描述](https://img-blog.csdnimg.cn/20210217175753567.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80NzI4MjQwNA==,size_16,color_FFFFFF,t_70)

### 绘制部分训练集图片

```python
def draw():
    for i in range(36):
        plt.subplot(6,6,i+1)
        plt.imshow(train.ix[i].values.reshape(28, 28))
    plt.show()
draw()
```
![在这里插入图片描述](https://img-blog.csdnimg.cn/20210217180007852.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80NzI4MjQwNA==,size_16,color_FFFFFF,t_70)
### 训练

```python
train = train/255
test = test/255

X_train, X_num, y_train, y_num = train_test_split(train, numbers, test_size = 0.2)

mlp = MLPClassifier()

mlp.fit(X_train,y_train)

y_test_predict=mlp.predict(X_num)
print(y_num)

y_test=y_num.values
print(y_test)

print(y_test_predict)
```
### 准确率

```python
print(mlp, metrics.classification_report(y_num, y_test_predict))

p = precision_score(y_test,y_test_predict,average=None) 
print("accuracy：",p)

print('accurcy :',metrics.accuracy_score(y_num, y_test_predict))
```
### 项目提交

```python
test_prediction=mlp.predict(test)
results = pd.Series(test_prediction,name="Label")
image_ids=pd.Series(range(1,28001),name = "ImageId")
My_submission = pd.concat([image_ids,results],axis = 1)
My_submission.to_csv("submission.csv",index=False)
My_submission.head()
```
![在这里插入图片描述](https://img-blog.csdnimg.cn/20210217180634879.png)
