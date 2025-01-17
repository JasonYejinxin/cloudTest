# 推理函数
def generate_answer(test_dir, question, feature_method="concatenate"):
    # 加载图片
    images, frame_names = load_images_for_inference(test_dir)

    answers = {}
    # 对每一组图片进行推理
    for i, frame_images in enumerate(images):
        print(f"Processing frame: {frame_names[i]}")
        
        # 特征处理
        try:
            if feature_method == "concatenate":
                pixel_values = concatenate_image_features(frame_images)
            else:
                raise ValueError("Only 'concatenate' feature method is supported in this inference code.")
        except Exception as e:
            print(f"Error during feature processing for frame {frame_names[i]}: {e}")
            answers[frame_names[i]] = "Error in feature processing"
            continue

        # 打印像素值形状用于调试
        print(f"Pixel values shape for frame {frame_names[i]}: {pixel_values.shape}")

        # 处理问题
        text_inputs = processor(text=question, return_tensors="pt", padding=True, truncation=True).input_ids.to(device)

        # 推理：使用 forward 方法手动调用
        with torch.no_grad():
            try:
                # 获取 logits（模型前向计算）
                outputs = model(input_ids=text_inputs, pixel_values=pixel_values)
                
                # 确保 logits 存在
                if not hasattr(outputs, "logits"):
                    print(f"Model outputs do not contain logits for frame {frame_names[i]}")
                    answers[frame_names[i]] = "No logits generated"
                    continue

                # 获取生成的 token ID（选择最大概率的 token）
                predicted_ids = outputs.logits.argmax(dim=-1)

                # 解码答案
                answer = processor.tokenizer.decode(predicted_ids[0], skip_special_tokens=True)
                answers[frame_names[i]] = answer
                print(f"Generated answer for frame {frame_names[i]}: {answer}")
            except Exception as e:
                print(f"Error occurred during inference for frame {frame_names[i]}: {e}")
                answers[frame_names[i]] = "Error during inference"
                continue

    return answers
