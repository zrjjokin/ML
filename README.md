
#毕设汇报


----------
[TOC]

##实际问题的理解

这次的问题是关于如何预警学习成绩偏靠后的在校大学生。简单的来说，我把在校大学生分为落后生和普通学生，从机器学习角度来说，这是个明显的分类问题。所以在这个实际问题中，我需要做的就是通过机器学习算法预测在校学生是否有学业警示或留级的风险，然后再根据测试数据集来判断预测结果的正确度。
大致思路：
numpy+pandas进行数据分析和处理
sklearn用来建模

##获取数据
表名 | 描述
:-------- | ---:
基本信息表（info.csv）|基本信息表包含了入学年份、入学方式、专业类别、生源地、考试类别等信息。 
成绩表（grades.csv）|成绩表包含学年学期、年级、专业、课程名称、学分、成绩等信息。学年学期指的是学生在哪个学期选择了这门课；课程通过后，学生可得到相应的学分，学分可以衡量课程的重要程度；成绩字段指的是学生在课程中取得的最终成绩，包含平时课堂表现以及期末考试成绩，平时课堂表现占最终成绩的30%，期末考试占70%，成绩大多为百分制。
入馆记录 (lib.csv)|记录的是学生进入图书馆的时间，包括学号和时间两个字段，图书馆是本校学生学习的主要场所，进出图书馆的频率很大程度上表示学生在学业上的投入程度。
智能卡打卡记录 (card.csv)|智能卡打卡记录包含学生智能卡的使用记录，比如在餐厅、超市的消费等记录，主要字段有时间、终端编号，终端编号代表其消费地点。
借书表(books.csv)|记录学生借还书籍的时间
##处理数据
####分析数据
提供了一共五张表，信息量有点大，所以有必要分析一下数据结构，看看哪些数据有用，哪些没用。
利用pandas库导入数据加以观察
以学生成绩表为例

```
# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import re
#读入原始数据
f=open('/home/jokin/数据/毕业设计（机器学习）/毕业设计/XueYe/grades.csv','rb')
df=pd.read_csv(f,low_memory=False,usecols=[0,1,3,4,5,7,8,15,16])

```
![学生成绩表](https://raw.githubusercontent.com/zrjjokin/HTML/master/%E9%80%89%E5%8C%BA_001.png)
####数据清洗

- ***删除***
由于我最后只需要计算某学生每学期所修的总学分和学年内的加权平均分，所以好多数据没有用处，例如选课号，课程名，平时成绩，初修取得等等，我会把这些数据列信息给删除，同时我们还注意到此表中不仅存在13级学生的信息，还有14，15级的，这些多余的信息行也是不需要的，继续删除
- ***分组***
经过数据删除，我们筛选出我所需要的数据，在这些数据的基础上，继续进行类似SQL语句的查询操作，我给学生进行分组，分组的依据是学号和学期
```
#根据学号，姓名，学期进行分组
group=df.groupby(['xh','xnxq'],sort=False)
#对各列进行聚合操作
df=group.agg({'xf':'sum','kscj':'mean','mul':'sum'})
#增加‘加权平均分’列
df['weight_avg']=df['mul']/df['xf']
```
得到了新的数据（样例）
|xh(学号)|xf(获得学分)|avg(平均分)|lable(标签)|
| :-------- | --------:| -------:|:--: |
|13110031901|49.0|76.85714285714286|0|
依次对各个表进行数据清洗的操作得到各个表的样例：
入馆表：
|xh（学号）|times（一年内入馆次数）|
|:----|----:|
|13110031901|126|
早餐表：
|xh（学号）|date（一年内吃早饭次数）|
|---|---|
|13110031901|56|
借书表：
|xh|books|
|----|----|
|13110031901|47|
学生信息表：
|xh（学号）|zy（专业）|mz（民族）|sydq（生源地）|kslb（农村还是城市）|
|---|----|||
|13110031901|海洋科学|汉族|山东省|城市应届


- ***合并***
```
import numpy as np
import pandas as pd
from sklearn.utils import shuffle
#读入成绩表
grade=pd.read_csv('/home/jokin/数据/毕业设计（机器学习）/毕业设计/code/grades/grade03.csv',low_memory=False,usecols=[0,1,3,4])
#读入入馆次数表
lib=pd.read_csv('/home/jokin/数据/毕业设计（机器学习）/毕业设计/code/lib/lib01.csv',low_memory=False)
#读入早餐详情表
bf=pd.read_csv('/home/jokin/数据/毕业设计（机器学习）/毕业设计/code/card/card01.csv',low_memory=False)
#读入借书表
book=pd.read_csv('/home/jokin/数据/毕业设计（机器学习）/毕业设计/code/books/book01.csv',low_memory=False)
#读入学生信息表
info=pd.read_csv('/home/jokin/数据/毕业设计（机器学习）/毕业设计/code/info/info01.csv',low_memory=False,usecols=[0,1,3])
grade_info=pd.merge(grade,info,on='xh')
pil=pd.merge(grade_info,lib,on='xh')
pilb=pd.merge(pil,bf,on='xh')
pilbb=pd.merge(pilb,book,on='xh')
#print(pilbb.describe())
#pilbb.to_csv('data.csv',index=None)
#print(pilbb.plot())
#打乱数据
pilbb=shuffle(pilbb)
#print(pilbb)
pilbb.to_csv('data.csv',index=None)
```
 对上面所有特征进行合并操作，汇总成一张数据表
 ![训练集](https://raw.githubusercontent.com/zrjjokin/HTML/master/%E9%80%89%E5%8C%BA_003.png)
 
####数据预处理
- 编码
 
        注意到我的数据中存在非数值型数据（机器学习只能学习数值），所以必须对那些数据进行预处理。以专业名称为例，大学里面有好多专业，这些专业数据都是字符串类型变量，需要我们通过编码的形式来改变其类型。于是我采用one-hot编码方式，将每个专业编码成类似二进制的形式，例如工学的编码就是00100000
- 数据标准化

        我们都知道大多数的梯度方法（几乎所有的机器学习算法都基于此）对于数据的缩放很敏感。因此，在运行算法之前，我们应该进行标准化、规格化（归一化）。标准化是将数据按比例缩放，使之落入一个小的特定区间。 归一化是一种简化计算的方式，即将有量纲的表达式，经过变换，化为无量纲的表达式，成为纯量，把数据映射到0～1范围之内处理。Scikit-Learn库已经为其提供了相应的函数。
```
from sklearn import preprocessing
# standardize the data attributes
standardized_X = preprocessing.scale(X)
# normalize the data attributes
normalized_X = preprocessing.normalize(X)
```
##特征选取   
上面交代了好多可能对预警学业产生影响的特征，但是我们并不知道这些特征是否与预测结果相关，在机器学习上通过计算各个特征的信息增益来判断某个特征对结果的影响程度。虽然特征选择是一个相当有创造性的过程，有时候更多的是靠直觉和专业的知识，但对于特征的选取，已经有很多的算法可供直接使用。我采用了树算法计算特征的信息量。
```
from sklearn import metrics
from sklearn.ensemble import ExtraTreesClassifier
model = ExtraTreesClassifier()
model.fit(X, y)
# 打印信息增益
print(model.feature_importances_)
```
##模型建立
将训练集划分为两部分，一部分是用来做学习的，一部分是用来做预测的。由于分类算法有好多，我选取了几个有名的算法分别进行机器学习，最后通过交叉验证算法比对他们预测结果的正确率进行选择最佳模型。Scikit-Learn库已经实现了所有基本机器学习的算法。
####逻辑回归
```
from sklearn import metrics
from sklearn.linear_model import LogisticRegression
model = LogisticRegression()
model.fit(X, y)
print(model)
# 作出预测
expected = y
predicted = model.predict(X)
# 总结该模型
print(metrics.classification_report(expected, predicted))
print(metrics.confusion_matrix(expected, predicted))
```
####决策树
```
from sklearn import metrics
from sklearn.tree import DecisionTreeClassifier
# fit a CART model to the data
model = DecisionTreeClassifier()
model.fit(X, y)
print(model)
expected = y
predicted = model.predict(X)
print(metrics.classification_report(expected, predicted))
print(metrics.confusion_matrix(expected, predicted))
```
####K邻近算法
```
from sklearn import metrics
from sklearn.neighbors import KNeighborsClassifier
# fit a k-nearest neighbor model to the data
model = KNeighborsClassifier()
model.fit(X, y)
print(model)
# make predictions
expected = y
predicted = model.predict(X)
# summarize the fit of the model
print(metrics.classification_report(expected, predicted))
print(metrics.confusion_matrix(expected, predicted))
```
####支持向量机
```
from sklearn import metrics
from sklearn.svm import SVC
# fit a SVM model to the data
model = SVC()
model.fit(X, y)
print(model)
# make predictions
expected = y
predicted = model.predict(X)
# summarize the fit of the model
print(metrics.classification_report(expected, predicted))
print(metrics.confusion_matrix(expected, predicted))
```
####朴素贝叶斯算法
```
from sklearn import metrics
from sklearn.naive_bayes import GaussianNB
model = GaussianNB()
model.fit(X, y)
print(model)
# 作出预测
expected = y
predicted = model.predict(X)
# 总结该模型
print(metrics.classification_report(expected, predicted))
print(metrics.confusion_matrix(expected, predicted))
```
初步模型对比
|分类模型|正确率|
|---|---|
|Logistic Regression|90.12%|
|Decision Tree|79.67%|
|KNN|85.25%|
|SVM|82.67%|
|Naive Bayes|78.53%|
 






