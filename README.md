# 处理图像和文本数据
def test_image_qa(model, processor, frame_id, question):
    # 载入图像
    images = load_image_from_frame(frame_id)
    if not images:
        print(f"No image found for frame {frame_id}. Skipping.")
        return None

    # 将图像处理为模型输入 (如果是多张图片可以扩展为拼接或聚合)
    pixel_values = processor(images=images, return_tensors="pt", padding=True).pixel_values.to(device)

    # 将问题转化为输入 ID
    text_inputs = processor(text=question, return_tensors="pt", padding=True, truncation=True).input_ids.to(device)

    # 打印输入形状用于调试
    print(f"Text input shape: {text_inputs.shape}, Pixel values shape: {pixel_values.shape}")

    # 使用模型进行推理
    with torch.no_grad():
        generated_ids = model.generate(input_ids=text_inputs, pixel_values=pixel_values, max_length=50)

    # 打印生成的 Token ID
    print(f"Generated token IDs: {generated_ids}")

    # 解码生成的答案
    generated_answer = processor.decode(generated_ids[0], skip_special_tokens=True)  # 解码第一句
    return generated_answer
