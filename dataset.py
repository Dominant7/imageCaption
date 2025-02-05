import json
import os
from torch.utils.data import Dataset
from PIL import Image

class CustomCOCODataset(Dataset):
    '''
    将自定义数据集转化为类COCO结构的Dataset
    '''
    def __init__(self, folder, image_processor, tokenizer, max_length=256):
        self.folder = folder
        self.image_processor = image_processor
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.data = []

        with open(os.path.join(folder, 'image_descriptions.json'), 'r', encoding='utf-8') as f:
            raw_data = json.load(f)
            for item in raw_data:
                image_path = os.path.join(os.getcwd(), self.folder, item['image_path'].split('/')[-1])
                description = item['description']['output']['choices'][0]['message']['content'][0]['text']
                self.data.append({
                    'image_path': image_path,
                    'caption': description
                })
    # # 定义预处理函数，用于处理图像和标注文本
    # def preprocess(items):
    #     pixel_values = image_processor(items["image"], return_tensors="pt").pixel_values.to(device)
    #     targets = self.tokenizer(
    #         [sentence["raw"] for sentence in items["sentences"]],
    #         max_length=max_length, padding="max_length", truncation=True, return_tensors="pt"
    #     ).to(device)
    #     return {'pixel_values': pixel_values, 'labels': targets["input_ids"]}


    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]

        image = Image.open(item['image_path']).convert("RGB")
        pixel_values = self.image_processor(images=image, return_tensors="pt").pixel_values

        inputs = self.tokenizer(
            item['caption'], 
            max_length=self.max_length, 
            padding="max_length", 
            truncation=True, 
            return_tensors="pt"
        )

        return {
            'pixel_values': pixel_values.squeeze(), 
            'labels': inputs['input_ids'].squeeze()
        }

class TestCOCODataset(Dataset):
    '''
    用于测试集的dataset(即生成任务), labels被设置为图像名称, 不要将这个类的对象用于训练
    '''
    def __init__(self, folder, image_processor, tokenizer, max_length=256):
        self.folder = folder
        self.image_processor = image_processor
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.data = []
        image_extensions = ('.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff')
        for root, dirs, files in os.walk(folder):
            for file in files:
                if file.lower().endswith(image_extensions):
                    self.data.append({"image_path" : os.path.join(root, file)})

                
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]

        image = Image.open(item['image_path']).convert("RGB")
        pixel_values = self.image_processor(images=image, return_tensors="pt").pixel_values

        return {
            'pixel_values': pixel_values.squeeze(), 
            'labels': item['image_path']
        }