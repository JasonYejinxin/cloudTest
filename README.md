import os
import torch
from transformers import Blip2Processor, Blip2ForConditionalGeneration
from PIL import Image
import json
from torch.optim import AdamW
import torch.nn as nn
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

print(f"okay i start")
# 初始化 BLIP 处理器和模型
model = Blip2ForConditionalGeneration.from_pretrained("Salesforce/blip2-flan-t5-xl")
print(f"okay the 2nd sentence")
processor = Blip2Processor.from_pretrained("Salesforce/blip2-flan-t5-xl")
print(f"tekenizer is {processor.tokenizer}")
model = model.to(device)
print("okay, successfully load model")
# 设置优化器
optimizer = AdamW(filter(lambda p:p.requires_grad, model.parameters()), lr=2e-5)

# 加载 qa.json 数据
qa_json_path = "/home/airlab/Desktop/Jingwen/MAPLMTest/baseline/evaluation/qaTrain_3Q_balanced.json"  # 替换为你的qa.json路径
with open(qa_json_path, 'r') as f:
    qa_data = json.load(f)

# 加载图片 (frame 文件夹下所有图片)
def load_images_from_frame(frame_id):
    images = []
    frame_path = os.path.join("/home/airlab/Desktop/Jingwen/MAPLMTest/baseline/evaluation/data/maplm_v0.1/train", frame_id)

    if not os.path.exists(frame_path):
        print(f"Warning: Frame folder {frame_id} not found. Skipping this entry.")
        return None

    img_names = sorted(os.listdir(frame_path))  # 确保文件按顺序读取
    for img_name in img_names[:4]:  # 加载最多 4 张图片
        img_path = os.path.join(frame_path, img_name)
        if img_path.endswith('.jpg') or img_path.endswith('.png'):
            img = Image.open(img_path).convert("RGB")
            images.append(img)
    return images if images else None

# 特征聚合 (平均池化)
def aggregate_image_features(images):
    pixel_values = []
    for img in images:
        inputs = processor(images=img, return_tensors="pt").pixel_values.to(device)
        pixel_values.append(inputs.to(device))
    pixel_values = torch.cat(pixel_values, dim=0)  # [4, 3, H, W]
    return pixel_values.mean(dim=0, keepdim=True)  # [1, 3, H, W]
channel_reduction = nn.Conv2d(in_channels = 12, out_channels = 3, kernel_size = 1).to(device)
# 特征拼接
def concatenate_image_features(images):
    pixel_values = []
    for img in images:
        inputs = processor(images=img, return_tensors="pt").pixel_values.to(device)
        pixel_values.append(inputs.to(device))
    pixel_values = torch.cat(pixel_values,dim=1)
    reduced = channel_reduction(pixel_values)
    return reduced # [1, 3, H, W]

# 处理多模态数据
def process_multimodal_data(qa_data, feature_method="aggregate"):
    inputs = []
    targets = []

    for data in qa_data:
        frame = data["frame"]
        qa_list = data["QA"]

        # 加载 frame 文件夹中的图片
        images = load_images_from_frame(frame)
        if images is None:
            continue

        # 根据指定方法处理图片特征
        if feature_method == "aggregate":
            pixel_values = aggregate_image_features(images)  # 平均池化
        elif feature_method == "concatenate":
            pixel_values = concatenate_image_features(images)  # 拼接特征
        else:
            raise ValueError("Invalid feature method. Use 'aggregate' or 'concatenate'.")

        for qa in qa_list:
            question = qa["question"]
            answer = qa["answer"]

            # 处理问题和答案
            text_inputs = processor(text=question, return_tensors="pt", padding=True, truncation=True).input_ids.to(device)
            target_ids = processor(text=answer, return_tensors="pt", padding=True, truncation=True).input_ids.to(device)

            # 对齐问题和答案的长度
            target_length = text_inputs.size(1)
            if target_ids.size(1) < target_length:
                pad_length = target_length - target_ids.size(1)
                padding = torch.full((target_ids.size(0), pad_length), processor.tokenizer.pad_token_id).to(device)
                target_ids = torch.cat((target_ids, padding), dim=1)

            inputs.append((text_inputs, pixel_values))
            targets.append(target_ids)

    return inputs, targets

# 训练模型
def train_model(qa_data, epochs=20, feature_method="aggregate"):
    inputs, targets = process_multimodal_data(qa_data, feature_method)

    for epoch in range(epochs):
        print(f"Starting epoch {epoch + 1}/{epochs}")
        model.train()

    
        for i in range(len(inputs)):
            input_ids, pixel_values = inputs[i]
            target_ids = targets[i]

            try:
                outputs = model(input_ids=input_ids, labels=target_ids, pixel_values=pixel_values)
                loss = outputs.loss
           
                if loss is not None:
                  loss.backward()
                  optimizer.step()
                  optimizer.zero_grad()
            except Exception as e:
                print(f"Error occurred: {e}")
                print(f"input_ids shape: {input_ids.shape}")
                print(f"pixel_values shape: {pixel_values.shape}")
                print(f"labels shape: {target_ids.shape}")
                continue
        if loss is not None:
            print(f"Epoch {epoch + 1} completed. Loss:{loss.item()}")
        else:
            print(f"Epoch{epoch + 1} completed, but loss was not coumputed")

        # 保存模型和处理器
        if(epoch % 5 == 0)&(epoch != 0):
            model_save_path = f"./blip2_flan_t5_epoch_{epoch + 5}_{feature_method}"
            processor_save_path = f"./blip2_flan_t5_epoch_{epoch + 5}_{feature_method}"
            model.save_pretrained(model_save_path)
            processor.save_pretrained(processor_save_path)
            print(f"Model and processor saved at {model_save_path}")

# 启动训练 (指定特征处理方法：'aggregate' 或 'concatenate')
train_model(qa_data, feature_method="aggregate")  # 或者改为 "concatenate"
