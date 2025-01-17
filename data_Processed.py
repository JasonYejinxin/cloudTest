
import os
import torch
from transformers import BlipProcessor, BlipForConditionalGeneration
from PIL import Image
import json
from torch.optim import AdamW
from torch.utils.data import Dataset, DataLoader
# os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:512"

device = torch.device('cuda' if torch.cuda.is_available() else' cpu')
# 初始化 BLIP 处理器和模型
processor = BlipProcessor.from_pretrained("Salesforce/blip2-flan-t5-xl")
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip2-flan-t5-xl", ignore_mismatched_sizes=True)

model = model.to(device)

# 设置优化器
optimizer = AdamW(model.parameters(), lr=5e-5)

# 假设qa.json文件是存储在当前目录下
qa_json_path = "/home/airlab/Desktop/Jingwen/MAPLMTest/baseline/evaluation/qaTrain_balanced.json"  # 替换为你的qa.json文件路径
with open(qa_json_path, 'r') as f:
    qa_data = json.load(f)

# 载入图像（每个frame文件夹只加载第一张图片）
def load_images_from_frame(frame_id):
    images = []
    frame_path = os.path.join("/home/airlab/Desktop/Jingwen/MAPLMTest/baseline/evaluation/data/maplm_v0.1/train", frame_id)  # 指定图片路径

    # 检查文件夹是否存在
    if not os.path.exists(frame_path):
        print(f"Warning: Frame folder {frame_id} not found. Skipping this entry.")
        return None  # 返回None，表示该frame没有对应的图片

    img_names = sorted(os.listdir(frame_path))  # 确保文件按顺序读取
    for img_name in img_names[:1]:  # 只加载第一张图片
        img_path = os.path.join(frame_path, img_name)
        print(img_path)
        if img_path.endswith('.jpg') or img_path.endswith('.png'):
            img = Image.open(img_path).convert("RGB")
            images.append(img)
    return images if images else None  # 如果没有图片，返回None

# 处理图像和文本数据
def process_multimodal_data(qa_data):
    inputs = []
    targets = []

    for data in qa_data:
        question = data["question"]
        frame = data["frame"]
        answer = data["answer"]

        # 加载该 frame 下的第一张图片
        images = load_images_from_frame(frame)
        
        # 如果没有加载到图像，跳过该条数据
        if images is None:
            continue
        
        # 提取图像特征
        pixel_values = processor(images=images[0], return_tensors="pt").pixel_values.to("cuda")
        
        # 将问题转化为输入 ID
        text_inputs = processor(text=question, return_tensors="pt", padding=True, truncation=True).input_ids.to("cuda")

        # 将答案转化为目标 ID (作为监督学习的标签)
        target_ids = processor(text=answer, return_tensors="pt", padding=True, truncation=True).input_ids.to("cuda")

        # 获取问题的长度
        target_length = text_inputs.size(1)  # 获取问题的长度
        print(f"target length is{target_length}")
        # 如果目标长度小于问题长度，进行填充
        if target_ids.size(1) < target_length:
            pad_length = target_length - target_ids.size(1)
            # 创建一个pad序列（填充到目标的长度）
            padding = torch.full((target_ids.size(0), pad_length), processor.tokenizer.pad_token_id).to(device)
            target_ids = torch.cat((target_ids, padding), dim=1)
        # 如果目标序列的长度大于问题长度，截断目标序列
        # elif target_ids.size(1) > target_length:
        #     target_ids = target_ids[:, :target_length]
        print(f"length of target now is{target_ids.size(1)}")

        # 保存输入和目标
        inputs.append((text_inputs, pixel_values))
        targets.append(target_ids)

    return inputs, targets

# 训练模型
def train_model(qa_data, epochs=5, save_interval=1):
    inputs, targets = process_multimodal_data(qa_data)

    # 训练循环
    for epoch in range(epochs):  # 假设训练 5 个 epoch
        print(f"round of traning epochs is number{epoch}")
        model.train()
        for i in range(len(inputs)):
            input_ids, pixel_values = inputs[i]
            target_ids = targets[i]
            
            # print(f"pixel shape:{pixel_values},input_ids shape:{input_ids},labels shape:{target_ids}")
            # # 向模型传递输入和目标
            # outputs = model(input_ids=input_ids, pixel_values=pixel_values, labels=target_ids)
            try:
                outputs = model(input_ids=input_ids, labels=target_ids, pixel_values=pixel_values)
            except Exception as e:
                 print(f"Error occurred: {e}")
                 print(f"input_ids shape: {input_ids.shape}")
                 print(f"pixel_values shape: {pixel_values.shape}")
                 print(f"labels shape: {target_ids.shape}")
            print(f"output is correct+{target_ids}")

            loss = outputs.loss
            loss.backward()

            # 更新权重
            optimizer.step()
            optimizer.zero_grad()
            # 每个epoch结束后保存模型和处理器
            model_save_path = f"./blip2_flan_t5_epoch{epoch+1}"
            processor_save_path = f"./blip2_flan_t5_epoch{epoch+1}"

            model.save_pretrained(model_save_path)
            processor.save_pretrained(processor_save_path)

        print(f"Model and processor saved for epoch {epoch+1} at {model_save_path}.")

        # if epoch%2 == 0:
        #    print(f"Epoch {epoch+2} completed. Loss: {loss.item()}")

        #    # 每个epoch结束后保存模型和处理器
        #    model_save_path = f"./blip2_flan_t5_epoch_{epoch+2}"
        #    processor_save_path = f"./blip2_flan_t5_epoch_{epoch+2}"

        #    model.save_pretrained(model_save_path)
        #    processor.save_pretrained(processor_save_path)

        #    print(f"Model and processor saved for epoch {epoch+2} at {model_save_path}.")

# 启动训练
train_model(qa_data)
