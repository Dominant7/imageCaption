import requests
import torch
from PIL import Image
from transformers import *
from tqdm import tqdm
import os
from dataset import CustomCOCODataset, TestCOCODataset
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="本脚本用于为图片生成描述")
    parser.add_argument('--train_set_path', type=str, default='./Train', help='训练集路径')
    parser.add_argument('--val_set_path', type=str, default='./Val', help='验证集路径')    
    parser.add_argument('--test_set_path', type=str, default='./Val', help='测试集路径') # 没有测试集，默认使用验证集路径，如增加测试集可以传入类似于./Test的参数
    parser.add_argument('--epoch', type=int, default=100, help='训练轮数')
    args = parser.parse_args()

device = "cuda" if torch.cuda.is_available() else "cpu"

encoder_model = "microsoft/swinv2-base-patch4-window12-192-22k"
decoder_model = "fnlp/bart-base-chinese"

model = VisionEncoderDecoderModel.from_encoder_decoder_pretrained(
    encoder_model, decoder_model, device_map="balanced"
).to(device)
print(f"initial model eos, bos, start is {model.config.eos_token_id} | {model.config.bos_token_id} | {model.config.decoder_start_token_id}")
tokenizer = BertTokenizerFast.from_pretrained(decoder_model)
print(f"initial tokenizer eos, bos is {tokenizer.eos_token_id} | {tokenizer.bos_token_id}") 
image_processor = ViTImageProcessor.from_pretrained(encoder_model)

model.config.decoder_start_token_id = tokenizer.cls_token_id # cls未使用
model.config.pad_token_id = tokenizer.pad_token_id
model.config.eos_token_id = tokenizer.eos_token_id
model.config.bos_token_id = tokenizer.bos_token_id

# 查看eos、bos (调试用，已注释)
# print(f"model eos, bos, start is {model.config.eos_token_id} | {model.config.bos_token_id} | {model.config.decoder_start_token_id}")
# print(f"tokenizer eos, bos is {tokenizer.eos_token_id} | {tokenizer.bos_token_id}")

train_dataset = CustomCOCODataset(args.train_set_path, image_processor, tokenizer)
val_dataset = CustomCOCODataset(args.val_set_path, image_processor, tokenizer)
test_dataset = TestCOCODataset(args.test_set_path, image_processor, tokenizer)

# 测试一下数据
# for data in train_dataset:
#     print("".join(tokenizer.batch_decode(data["labels"], skip_special_tokens=True)))
#     print(data["labels"].shape)
#     break

def collate_fn(batch):
    return {
        'pixel_values': torch.stack([x['pixel_values'] for x in batch]),
        'labels': torch.stack([x['labels'] for x in batch])
    }

import evaluate
import numpy as np

rouge = evaluate.load("rouge")
bleu = evaluate.load("bleu")

def compute_metrics(eval_pred):
    preds = eval_pred.predictions
    preds = np.where(preds != -100, preds, tokenizer.pad_token_id)
    labels = eval_pred.label_ids
    pred_str = tokenizer.batch_decode(preds, skip_special_tokens=True)
    labels_str = tokenizer.batch_decode(labels, skip_special_tokens=True)
    rouge_result = rouge.compute(predictions=pred_str, references=labels_str)
    rouge_result = {k: round(v * 100, 4) for k, v in rouge_result.items()}
    bleu_result = bleu.compute(predictions=pred_str, references=labels_str)
    return {
        **rouge_result,
        "bleu": round(bleu_result["bleu"] * 100, 4),
        "gen_len": bleu_result["translation_length"] / len(preds)
    }

# 训练参数
num_epochs = args.epoch
batch_size = 10
gradient_accumulation_steps = 4

training_args = Seq2SeqTrainingArguments(
    gradient_accumulation_steps=gradient_accumulation_steps,
    fp16=True,
    predict_with_generate=True,
    num_train_epochs=num_epochs,
    evaluation_strategy="steps",
    eval_steps=5000,
    logging_steps=5000,
    save_strategy="steps",
    save_steps=5000,
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=batch_size,
    output_dir="./output",
    load_best_model_at_end=True,  
    save_total_limit=10,  
    generation_max_length = 256,
)

trainer = Seq2SeqTrainer(
    model=model,
    tokenizer=image_processor,
    args=training_args,
    compute_metrics=compute_metrics,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    data_collator=collate_fn,
)

def get_last_checkpoint(output_dir):
    # 获取输出目录中的所有检查点
    checkpoint_dirs = [d for d in os.listdir(output_dir) if d.startswith('checkpoint-')]
    if not checkpoint_dirs:
        return None

    # 根据检查点目录名排序选择最新检查点
    checkpoint_dirs.sort(key=lambda x: int(x.split('-')[-1]), reverse=True)
    return os.path.join(output_dir, checkpoint_dirs[0])

# 如果有检查点可用，则从最新的检查点恢复训练
last_checkpoint = None
if os.path.isdir(training_args.output_dir):
    last_checkpoint = get_last_checkpoint(training_args.output_dir)
    if last_checkpoint:
        print(f"Resuming training from checkpoint {last_checkpoint}")
        trainer.train(resume_from_checkpoint=last_checkpoint)
    else:
        trainer.train()
else:
    trainer.train()

output_dir = "./output"

model.save_pretrained(output_dir)
tokenizer.save_pretrained(output_dir)

# print(f"Before evaluation, decoder_start_token_id: {model.get_decoder().config.decoder_start_token_id}")

# result = trainer.evaluate(val_dataset)

# print(result)

# 生成描述函数，对于CustomCOCODataset的对象会生成原描述与生成描述的对照文件，对于TestCOCODataset生成描述与图像名称的对照文件
def generate_and_save_descriptions(model, dataset, tokenizer, output_file):
    model.eval()
    descriptions = []
    original_labels = []
    
    if isinstance(dataset, CustomCOCODataset): # 此时生成描述对照
        with torch.no_grad():
            for batch in tqdm(dataset):
                pixel_values = batch['pixel_values'].unsqueeze(0).to(device)
                labels = batch['labels'].to(device)  # 原始标签
                
                # 生成描述
                outputs = model.generate(pixel_values=pixel_values, max_length=256)
                preds = tokenizer.batch_decode(outputs, skip_special_tokens=True)
                preds = "".join(preds).replace(" ", "")
                original_texts = tokenizer.batch_decode(labels, skip_special_tokens=True)
                original_texts = "".join(original_texts)

                descriptions.append(preds)
                original_labels.append(original_texts)
            
        with open(output_file, 'w', encoding='utf-8') as f:
            for i in range(len(descriptions)):
                f.write(f"Generated: {descriptions[i]}\n")
                f.write(f"Original: {original_labels[i]}\n")
                f.write("\n")  # 空行作为分隔符

    elif isinstance(dataset, TestCOCODataset): # 此时生成名称描述对照
        with torch.no_grad():
            for batch in tqdm(dataset):
                pixel_values = batch['pixel_values'].unsqueeze(0).to(device)
                labels = batch['labels']  # 图像名称
                
                # 生成描述
                outputs = model.generate(pixel_values=pixel_values, max_length=256)
                preds = tokenizer.batch_decode(outputs, skip_special_tokens=True)
                preds = "".join(preds).replace(" ", "")

                descriptions.append(preds)
                original_labels.append(labels)
            
        with open(output_file, 'w', encoding='utf-8') as f:
            for i in range(len(descriptions)):
                f.write(f"Image Name: {original_labels[i]}\n")
                f.write(f"Generated: {descriptions[i]}\n")
                f.write("\n")  # 空行作为分隔符

    else:
        raise RuntimeError(f"期待输入dataset为CustomCOCODataset或TestCOCODataset, 然而输入dataset类型为{type(dataset)}")

# 生成并保存描述
generate_and_save_descriptions(model, test_dataset, tokenizer, 'generated_descriptions.txt')

