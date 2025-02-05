# 图片描述生成

## 简介
本项目基于预训练的Swin-Transformer V2与BART，用于自动为图像生成中文描述。

## 使用
### 训练
请将训练集和验证集分别放置在根目录下文件夹中(默认为Train与Val文件夹)，并确保各文件夹都内包含一个名为image_descriptions.json的文件用于保存训练用描述。
调用方式如下
python imageCaption.py --epoch 你要训练的轮数 --train_set_path 训练集文件夹路径(如果要更改默认路径的话) --val_set_path 验证集文件夹路径(同前)

### 生成
请将测试集放置在根目录下文件夹中(即在根目录下某文件夹内放置图片即可)。
调用方式如下
python imageCaption.py --test_set_path 测试集文件夹路径 --epoch 0

## 依赖
详见requirements.txt