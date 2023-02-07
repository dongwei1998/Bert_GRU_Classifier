## Bert+gru实现的文本分类
+ 自行修改文件配置路径 /utils/parameter.py  文件
### 介绍
+ 本项目通过 BERT模型拼接 GRU模型实现
+ 可以完后文本分类、情感分析、等任务

### 数据集
+ 数据集采用的是情感分析数据
```
文本分类
├── test 文件夹
│   ├── 类别1
│   │   ├── 1.txt
│   │   └── 2.txt
│   ├── 类别2
│   │   ├── 3.txt
│   │   └── 4.txt
├── train 文件夹
│   ├── 类别1
│   │   ├── 5.txt
│   │   └── 6.txt
│   ├── 类别2
│   │   ├── 7.txt
│   │   └── 8.txt
└── val 文件夹
    ├── 类别1
    │   ├── 9.txt
    │   └── 10.txt
    └── 类别2
        ├── 11.txt
        └── 12.txt

```
  

### 环境
+ data==0.4
+ easydict==1.9
+ Flask==2.1.0
+ matplotlib==3.5.1
+ numpy==1.19.5
+ python-dotenv==0.20.0
+ requests==2.25.1
+ six==1.15.0
+ tensorflow_gpu==2.5.0
+ tqdm==4.62.3


### 训练
    python train.py
### 服务启动
    python server.py
### 服务测试
    python flasktest.py
