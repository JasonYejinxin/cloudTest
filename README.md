import os
import torch
from transformers import BlipProcessor, BlipForConditionalGeneration
from PIL import Image
import torch.nn as nn

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 加载训练好的模型和处理器
model_path = "./blip2_flan_t5_epoch_5_concatenate"  # 替换为保存的模型路径
processor = BlipProcessor.from_pretrained(model_path)
model = BlipForConditionalGeneration.from_pretrained(model_path)
model = model.to(device)
model.eval()

# 加载图片（推理时加载多帧图像）
def load_images_for_inference(test_dir):
    images = []
    if not os.path.exists(test_dir):
        raise FileNotFoundError(f"Test directory {test_dir} not found.")
    
    # 遍历所有子文件夹（fr1, fr2 等）
    subfolders = sorted(os.listdir(test_dir))
    frame_names = []
    for folder in subfolders:
        folder_path = os.path.join(test_dir, folder)
        if os.path.isdir(folder_path):  # 确保是子文件夹
            img_names = sorted(os.listdir(folder_path))  # 按文件名排序
            images_in_frame = []
            for img_name in img_names[:4]:  # 加载最多4张图片
                img_path = os.path.join(folder_path, img_name)
                if img_path.endswith(".jpg") or img_path.endswith(".png"):  # 只加载jpg/png图片
                    try:
                        img = Image.open(img_path).convert("RGB")
                        images_in_frame.append(img)
                    except Exception as e:
                        print(f"Failed to open image {img_path}: {e}")
            if images_in_frame:
                images.append(images_in_frame)
                frame_names.append(folder)  # 保存帧名称
    if not images:
        raise ValueError(f"No valid images found in {test_dir}.")
    return images, frame_names

# 特征拼接（与训练保持一致）
def concatenate_image_features(images):
    pixel_values = []
    for img in images:
        inputs = processor(images=img, return_tensors="pt").pixel_values.to(device)
        pixel_values.append(inputs)
    pixel_values = torch.cat(pixel_values, dim=1)  # [1, 12, H, W]
    # 使用卷积层将通道数减少到 3，确保与模型输入一致
    channel_reduction = nn.Conv2d(in_channels=12, out_channels=3, kernel_size=1).to(device)
    reduced_pixel_values = channel_reduction(pixel_values)
    return reduced_pixel_values

# 推理函数
def generate_answer(test_dir, question, feature_method="concatenate"):
    # 加载图片
    images, frame_names = load_images_for_inference(test_dir)

    answers = {}
    # 对每一组图片进行推理
    for i, frame_images in enumerate(images):
        # 特征处理
        if feature_method == "concatenate":
            pixel_values = concatenate_image_features(frame_images)
        else:
            raise ValueError("Only 'concatenate' feature method is supported in this inference code.")

        # 处理问题
        text_inputs = processor(text=question, return_tensors="pt", padding=True, truncation=True).input_ids.to(device)

        # 推理：先做前向推理获取 logits
        with torch.no_grad():
            try:
                # 获取模型的输出 logits
                outputs = model.generate(input_ids=text_inputs, pixel_values=pixel_values, max_length=50)
                
                if outputs is None or outputs.size(0) == 0:
                    print(f"No output generated for frame {frame_names[i]}")
                    answers[frame_names[i]] = "No answer generated"
                    continue

                # 解码生成的答案
                answer = processor.tokenizer.decode(outputs[0], skip_special_tokens=True)
                answers[frame_names[i]] = answer
                print(f"Generated answer for frame {frame_names[i]}: {answer}")
            except Exception as e:
                print(f"Error occurred during inference for frame {frame_names[i]}: {e}")
                answers[frame_names[i]] = "Error during inference"
                continue

    return answers

# 示例推理
if __name__ == "__main__":
    test_dir = "/home/airlab/Desktop/Jingwen/MAPLMTest/baseline/evaluation/data/maplm_v0.1/test"  # 替换为测试数据目录
    question = "How many lanes in current road?"  # 替换为你想问的问题

    try:
        answers = generate_answer(test_dir, question)
        for frame_name, answer in answers.items():
            print(f"Frame: {frame_name}")
            print(f"Answer: {answer}")
    except Exception as e:
        print(f"Error occurred during inference: {e}")
