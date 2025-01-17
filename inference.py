import os
import torch
from transformers import BlipProcessor, BlipForConditionalGeneration
from PIL import Image
import json

# 设置设备为cuda或者cpu
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 加载保存的模型和处理器
model_path = "/home/airlab/Desktop/Jingwen/MAPLMTest/baseline/evaluation/blip2_flan_t5_epoch1"  # 保存模型的路径
processor = BlipProcessor.from_pretrained(model_path)
model = BlipForConditionalGeneration.from_pretrained(model_path)

# 将模型移至指定设备（GPU或CPU）
model = model.to(device)

# 假设qa.json文件路径
qa_json_path = "/home/airlab/Desktop/Jingwen/MAPLMTest/baseline/evaluation/qaTest.json"  # 替换为你的qa.json文件路径

# 读取qa.json文件
with open(qa_json_path, 'r') as f:
    qa_data = json.load(f)


# 载入图像（每个frame文件夹只加载第一张图片）
def load_image_from_frame(frame_id):
    images = []
    frame_path = os.path.join("/home/airlab/Desktop/Jingwen/MAPLMTest/baseline/evaluation/data/maplm_v0.1/test", frame_id)  # 替换为你的图片数据集路径

    # 检查文件夹是否存在
    if not os.path.exists(frame_path):
        print(f"Warning: Frame folder {frame_id} not found. Skipping this entry.")
        return None  # 返回None，表示该frame没有对应的图片

    img_names = sorted(os.listdir(frame_path))  # 确保文件按顺序读取
    for img_name in img_names[:1]:  # 只加载第一张图片
        img_path = os.path.join(frame_path, img_name)
        print(f"Loading image: {img_path}")
        if img_path.endswith('.jpg') or img_path.endswith('.png'):
            img = Image.open(img_path).convert("RGB")
            images.append(img)
    return images if images else None  # 如果没有图片，返回None

# 处理图像和文本数据
def test_image_qa(model, processor, frame_id, question):
    # 载入图像
    images = load_image_from_frame(frame_id)
    if images is None:
        print(f"No image found for frame {frame_id}. Skipping.")
        return None

    # 提取图像特征
    pixel_values = processor(images=images[0], return_tensors="pt").pixel_values.to(device)

    # 将问题转化为输入 ID
    text_inputs = processor(text=question, return_tensors="pt", padding=True, truncation=True).input_ids.to(device)

    # 使用模型进行推理
    with torch.no_grad():
        outputs = model(input_ids=text_inputs, pixel_values=pixel_values)

    # 打印 logits 的形状和内容
    print(f"Logits shape: {outputs.logits.shape}")
    print(f"Logits: {outputs.logits}")

    # 从输出中获取生成的答案
    predicted_ids = outputs.logits.argmax(dim=-1)  # 获取最大值的索引

    # 打印预测的 ID
    print(f"Predicted token IDs: {predicted_ids}")

    # 解码
    generated_answer = processor.decode(predicted_ids[0], skip_special_tokens=True)  # 取第一个序列的 ID 进行解码

    return generated_answer

# 遍历qa.json中的每个条目进行测试
def test_model(qa_data, model, processor):
    for data in qa_data:
        question = data["question"]
        frame = data["frame"]
        answer = data["answer"]

        print(f"Testing question: {question} for frame: {frame}")
        
        # 测试模型并获得生成的答案
        generated_answer = test_image_qa(model, processor, frame, question)
        
        if generated_answer:
            print(f"Expected Answer: {answer}")
            print(f"Generated Answer: {generated_answer}")
            print("-" * 50)

# 启动测试
test_model(qa_data, model, processor)
