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



Traceback (most recent call last):
  File "/home/airlab/Desktop/Jingwen/MAPLMTest/baseline/evaluation/data_processed_concat.py", line 12, in <module>
    model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip2-flan-t5-xl")
            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/airlab/anaconda3/lib/python3.12/site-packages/transformers/modeling_utils.py", line 4264, in from_pretrained
    ) = cls._load_pretrained_model(
        ^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/airlab/anaconda3/lib/python3.12/site-packages/transformers/modeling_utils.py", line 4834, in _load_pretrained_model
    raise RuntimeError(f"Error(s) in loading state_dict for {model.__class__.__name__}:\n\t{error_msg}")
RuntimeError: Error(s) in loading state_dict for BlipForConditionalGeneration:
        size mismatch for vision_model.embeddings.class_embedding: copying a param with shape torch.Size([1, 1, 1408]) from checkpoint, the shape in current model is torch.Size([1, 1, 768]).
        size mismatch for vision_model.embeddings.position_embedding: copying a param with shape torch.Size([1, 257, 1408]) from checkpoint, the shape in current model is torch.Size([1, 577, 768]).
        size mismatch for vision_model.embeddings.patch_embedding.weight: copying a param with shape torch.Size([1408, 3, 14, 14]) from checkpoint, the shape in current model is torch.Size([768, 3, 16, 16]).
        size mismatch for vision_model.embeddings.patch_embedding.bias: copying a param with shape torch.Size([1408]) from checkpoint, the shape in current model is torch.Size([768]).
        size mismatch for vision_model.encoder.layers.0.self_attn.qkv.weight: copying a param with shape torch.Size([4224, 1408]) from checkpoint, the shape in current model is torch.Size([2304, 768]).
        size mismatch for vision_model.encoder.layers.0.self_attn.qkv.bias: copying a param with shape torch.Size([4224]) from checkpoint, the shape in current model is torch.Size([2304]).
        size mismatch for vision_model.encoder.layers.0.self_attn.projection.weight: copying a param with shape torch.Size([1408, 1408]) from checkpoint, the shape in current model is torch.Size([768, 768]).
        size mismatch for vision_model.encoder.layers.0.self_attn.projection.bias: copying a param with shape torch.Size([1408]) from checkpoint, the shape in current model is torch.Size([768]).
        size mismatch for vision_model.encoder.layers.0.layer_norm1.weight: copying a param with shape torch.Size([1408]) from checkpoint, the shape in current model is torch.Size([768]).
        size mismatch for vision_model.encoder.layers.0.layer_norm1.bias: copying a param with shape torch.Size([1408]) from checkpoint, the shape in current model is torch.Size([768]).
        size mismatch for vision_model.encoder.layers.0.mlp.fc1.weight: copying a param with shape torch.Size([6144, 1408]) from checkpoint, the shape in current model is torch.Size([3072, 768]).
        size mismatch for vision_model.encoder.layers.0.mlp.fc1.bias: copying a param with shape torch.Size([6144]) from checkpoint, the shape in current model is torch.Size([3072]).
        size mismatch for vision_model.encoder.layers.0.mlp.fc2.weight: copying a param with shape torch.Size([1408, 6144]) from checkpoint, the shape in current model is torch.Size([768, 3072]).
        size mismatch for vision_model.encoder.layers.0.mlp.fc2.bias: copying a param with shape torch.Size([1408]) from checkpoint, the shape in current model is torch.Size([768]).
        size mismatch for vision_model.encoder.layers.0.layer_norm2.weight: copying a param with shape torch.Size([1408]) from checkpoint, the shape in current model is torch.Size([768]).
        size mismatch for vision_model.encoder.layers.0.layer_norm2.bias: copying a param with shape torch.Size([1408]) from checkpoint, the shape in current model is torch.Size([768]).
        size mismatch for vision_model.encoder.layers.1.self_attn.qkv.weight: copying a param with shape torch.Size([4224, 1408]) from checkpoint, the shape in current model is torch.Size([2304, 768]).
        size mismatch for vision_model.encoder.layers.1.self_attn.qkv.bias: copying a param with shape torch.Size([4224]) from checkpoint, the shape in current model is torch.Size([2304]).
        size mismatch for vision_model.encoder.layers.1.self_attn.projection.weight: copying a param with shape torch.Size([1408, 1408]) from checkpoint, the shape in current model is torch.Size([768, 768]).
        size mismatch for vision_model.encoder.layers.1.self_attn.projection.bias: copying a param with shape torch.Size([1408]) from checkpoint, the shape in current model is torch.Size([768]).
        size mismatch for vision_model.encoder.layers.1.layer_norm1.weight: copying a param with shape torch.Size([1408]) from checkpoint, the shape in current model is torch.Size([768]).
        size mismatch for vision_model.encoder.layers.1.layer_norm1.bias: copying a param with shape torch.Size([1408]) from checkpoint, the shape in current model is torch.Size([768]).
        size mismatch for vision_model.encoder.layers.1.mlp.fc1.weight: copying a param with shape torch.Size([6144, 1408]) from checkpoint, the shape in current model is torch.Size([3072, 768]).
        size mismatch for vision_model.encoder.layers.1.mlp.fc1.bias: copying a param with shape torch.Size([6144]) from checkpoint, the shape in current model is torch.Size([3072]).
        size mismatch for vision_model.encoder.layers.1.mlp.fc2.weight: copying a param with shape torch.Size([1408, 6144]) from checkpoint, the shape in current model is torch.Size([768, 3072]).
        size mismatch for vision_model.encoder.layers.1.mlp.fc2.bias: copying a param with shape torch.Size([1408]) from checkpoint, the shape in current model is torch.Size([768]).
        size mismatch for vision_model.encoder.layers.1.layer_norm2.weight: copying a param with shape torch.Size([1408]) from checkpoint, the shape in current model is torch.Size([768]).
        size mismatch for vision_model.encoder.layers.1.layer_norm2.bias: copying a param with shape torch.Size([1408]) from checkpoint, the shape in current model is torch.Size([768]).
        size mismatch for vision_model.encoder.layers.2.self_attn.qkv.weight: copying a param with shape torch.Size([4224, 1408]) from checkpoint, the shape in current model is torch.Size([2304, 768]).
        size mismatch for vision_model.encoder.layers.2.self_attn.qkv.bias: copying a param with shape torch.Size([4224]) from checkpoint, the shape in current model is torch.Size([2304]).
        size mismatch for vision_model.encoder.layers.2.self_attn.projection.weight: copying a param with shape torch.Size([1408, 1408]) from checkpoint, the shape in current model is torch.Size([768, 768]).
        size mismatch for vision_model.encoder.layers.2.self_attn.projection.bias: copying a param with shape torch.Size([1408]) from checkpoint, the shape in current model is torch.Size([768]).
        size mismatch for vision_model.encoder.layers.2.layer_norm1.weight: copying a param with shape torch.Size([1408]) from checkpoint, the shape in current model is torch.Size([768]).
        size mismatch for vision_model.encoder.layers.2.layer_norm1.bias: copying a param with shape torch.Size([1408]) from checkpoint, the shape in current model is torch.Size([768]).
        size mismatch for vision_model.encoder.layers.2.mlp.fc1.weight: copying a param with shape torch.Size([6144, 1408]) from checkpoint, the shape in current model is torch.Size([3072, 768]).
        size mismatch for vision_model.encoder.layers.2.mlp.fc1.bias: copying a param with shape torch.Size([6144]) from checkpoint, the shape in current model is torch.Size([3072]).
        size mismatch for vision_model.encoder.layers.2.mlp.fc2.weight: copying a param with shape torch.Size([1408, 6144]) from checkpoint, the shape in current model is torch.Size([768, 3072]).
        size mismatch for vision_model.encoder.layers.2.mlp.fc2.bias: copying a param with shape torch.Size([1408]) from checkpoint, the shape in current model is torch.Size([768]).
        size mismatch for vision_model.encoder.layers.2.layer_norm2.weight: copying a param with shape torch.Size([1408]) from checkpoint, the shape in current model is torch.Size([768]).
        size mismatch for vision_model.encoder.layers.2.layer_norm2.bias: copying a param with shape torch.Size([1408]) from checkpoint, the shape in current model is torch.Size([768]).
        size mismatch for vision_model.encoder.layers.3.self_attn.qkv.weight: copying a param with shape torch.Size([4224, 1408]) from checkpoint, the shape in current model is torch.Size([2304, 768]).
        size mismatch for vision_model.encoder.layers.3.self_attn.qkv.bias: copying a param with shape torch.Size([4224]) from checkpoint, the shape in current model is torch.Size([2304]).
        size mismatch for vision_model.encoder.layers.3.self_attn.projection.weight: copying a param with shape torch.Size([1408, 1408]) from checkpoint, the shape in current model is torch.Size([768, 768]).
        size mismatch for vision_model.encoder.layers.3.self_attn.projection.bias: copying a param with shape torch.Size([1408]) from checkpoint, the shape in current model is torch.Size([768]).
        size mismatch for vision_model.encoder.layers.3.layer_norm1.weight: copying a param with shape torch.Size([1408]) from checkpoint, the shape in current model is torch.Size([768]).
        size mismatch for vision_model.encoder.layers.3.layer_norm1.bias: copying a param with shape torch.Size([1408]) from checkpoint, the shape in current model is torch.Size([768]).
        size mismatch for vision_model.encoder.layers.3.mlp.fc1.weight: copying a param with shape torch.Size([6144, 1408]) from checkpoint, the shape in current model is torch.Size([3072, 768]).
        size mismatch for vision_model.encoder.layers.3.mlp.fc1.bias: copying a param with shape torch.Size([6144]) from checkpoint, the shape in current model is torch.Size([3072]).
        size mismatch for vision_model.encoder.layers.3.mlp.fc2.weight: copying a param with shape torch.Size([1408, 6144]) from checkpoint, the shape in current model is torch.Size([768, 3072]).
        size mismatch for vision_model.encoder.layers.3.mlp.fc2.bias: copying a param with shape torch.Size([1408]) from checkpoint, the shape in current model is torch.Size([768]).
        size mismatch for vision_model.encoder.layers.3.layer_norm2.weight: copying a param with shape torch.Size([1408]) from checkpoint, the shape in current model is torch.Size([768]).
        size mismatch for vision_model.encoder.layers.3.layer_norm2.bias: copying a param with shape torch.Size([1408]) from checkpoint, the shape in current model is torch.Size([768]).
        size mismatch for vision_model.encoder.layers.4.self_attn.qkv.weight: copying a param with shape torch.Size([4224, 1408]) from checkpoint, the shape in current model is torch.Size([2304, 768]).
        size mismatch for vision_model.encoder.layers.4.self_attn.qkv.bias: copying a param with shape torch.Size([4224]) from checkpoint, the shape in current model is torch.Size([2304]).
        size mismatch for vision_model.encoder.layers.4.self_attn.projection.weight: copying a param with shape torch.Size([1408, 1408]) from checkpoint, the shape in current model is torch.Size([768, 768]).
        size mismatch for vision_model.encoder.layers.4.self_attn.projection.bias: copying a param with shape torch.Size([1408]) from checkpoint, the shape in current model is torch.Size([768]).
        size mismatch for vision_model.encoder.layers.4.layer_norm1.weight: copying a param with shape torch.Size([1408]) from checkpoint, the shape in current model is torch.Size([768]).
        size mismatch for vision_model.encoder.layers.4.layer_norm1.bias: copying a param with shape torch.Size([1408]) from checkpoint, the shape in current model is torch.Size([768]).
        size mismatch for vision_model.encoder.layers.4.mlp.fc1.weight: copying a param with shape torch.Size([6144, 1408]) from checkpoint, the shape in current model is torch.Size([3072, 768]).
        size mismatch for vision_model.encoder.layers.4.mlp.fc1.bias: copying a param with shape torch.Size([6144]) from checkpoint, the shape in current model is torch.Size([3072]).
        size mismatch for vision_model.encoder.layers.4.mlp.fc2.weight: copying a param with shape torch.Size([1408, 6144]) from checkpoint, the shape in current model is torch.Size([768, 3072]).
        size mismatch for vision_model.encoder.layers.4.mlp.fc2.bias: copying a param with shape torch.Size([1408]) from checkpoint, the shape in current model is torch.Size([768]).
        size mismatch for vision_model.encoder.layers.4.layer_norm2.weight: copying a param with shape torch.Size([1408]) from checkpoint, the shape in current model is torch.Size([768]).
        size mismatch for vision_model.encoder.layers.4.layer_norm2.bias: copying a param with shape torch.Size([1408]) from checkpoint, the shape in current model is torch.Size([768]).
        size mismatch for vision_model.encoder.layers.5.self_attn.qkv.weight: copying a param with shape torch.Size([4224, 1408]) from checkpoint, the shape in current model is torch.Size([2304, 768]).
        size mismatch for vision_model.encoder.layers.5.self_attn.qkv.bias: copying a param with shape torch.Size([4224]) from checkpoint, the shape in current model is torch.Size([2304]).
        size mismatch for vision_model.encoder.layers.5.self_attn.projection.weight: copying a param with shape torch.Size([1408, 1408]) from checkpoint, the shape in current model is torch.Size([768, 768]).
        size mismatch for vision_model.encoder.layers.5.self_attn.projection.bias: copying a param with shape torch.Size([1408]) from checkpoint, the shape in current model is torch.Size([768]).
        size mismatch for vision_model.encoder.layers.5.layer_norm1.weight: copying a param with shape torch.Size([1408]) from checkpoint, the shape in current model is torch.Size([768]).
        size mismatch for vision_model.encoder.layers.5.layer_norm1.bias: copying a param with shape torch.Size([1408]) from checkpoint, the shape in current model is torch.Size([768]).
        size mismatch for vision_model.encoder.layers.5.mlp.fc1.weight: copying a param with shape torch.Size([6144, 1408]) from checkpoint, the shape in current model is torch.Size([3072, 768]).
        size mismatch for vision_model.encoder.layers.5.mlp.fc1.bias: copying a param with shape torch.Size([6144]) from checkpoint, the shape in current model is torch.Size([3072]).
        size mismatch for vision_model.encoder.layers.5.mlp.fc2.weight: copying a param with shape torch.Size([1408, 6144]) from checkpoint, the shape in current model is torch.Size([768, 3072]).
        size mismatch for vision_model.encoder.layers.5.mlp.fc2.bias: copying a param with shape torch.Size([1408]) from checkpoint, the shape in current model is torch.Size([768]).
        size mismatch for vision_model.encoder.layers.5.layer_norm2.weight: copying a param with shape torch.Size([1408]) from checkpoint, the shape in current model is torch.Size([768]).
        size mismatch for vision_model.encoder.layers.5.layer_norm2.bias: copying a param with shape torch.Size([1408]) from checkpoint, the shape in current model is torch.Size([768]).
        size mismatch for vision_model.encoder.layers.6.self_attn.qkv.weight: copying a param with shape torch.Size([4224, 1408]) from checkpoint, the shape in current model is torch.Size([2304, 768]).
        size mismatch for vision_model.encoder.layers.6.self_attn.qkv.bias: copying a param with shape torch.Size([4224]) from checkpoint, the shape in current model is torch.Size([2304]).
        size mismatch for vision_model.encoder.layers.6.self_attn.projection.weight: copying a param with shape torch.Size([1408, 1408]) from checkpoint, the shape in current model is torch.Size([768, 768]).
        size mismatch for vision_model.encoder.layers.6.self_attn.projection.bias: copying a param with shape torch.Size([1408]) from checkpoint, the shape in current model is torch.Size([768]).
        size mismatch for vision_model.encoder.layers.6.layer_norm1.weight: copying a param with shape torch.Size([1408]) from checkpoint, the shape in current model is torch.Size([768]).
        size mismatch for vision_model.encoder.layers.6.layer_norm1.bias: copying a param with shape torch.Size([1408]) from checkpoint, the shape in current model is torch.Size([768]).
        size mismatch for vision_model.encoder.layers.6.mlp.fc1.weight: copying a param with shape torch.Size([6144, 1408]) from checkpoint, the shape in current model is torch.Size([3072, 768]).
        size mismatch for vision_model.encoder.layers.6.mlp.fc1.bias: copying a param with shape torch.Size([6144]) from checkpoint, the shape in current model is torch.Size([3072]).
        size mismatch for vision_model.encoder.layers.6.mlp.fc2.weight: copying a param with shape torch.Size([1408, 6144]) from checkpoint, the shape in current model is torch.Size([768, 3072]).
        size mismatch for vision_model.encoder.layers.6.mlp.fc2.bias: copying a param with shape torch.Size([1408]) from checkpoint, the shape in current model is torch.Size([768]).
        size mismatch for vision_model.encoder.layers.6.layer_norm2.weight: copying a param with shape torch.Size([1408]) from checkpoint, the shape in current model is torch.Size([768]).
        size mismatch for vision_model.encoder.layers.6.layer_norm2.bias: copying a param with shape torch.Size([1408]) from checkpoint, the shape in current model is torch.Size([768]).
        size mismatch for vision_model.encoder.layers.7.self_attn.qkv.weight: copying a param with shape torch.Size([4224, 1408]) from checkpoint, the shape in current model is torch.Size([2304, 768]).
        size mismatch for vision_model.encoder.layers.7.self_attn.qkv.bias: copying a param with shape torch.Size([4224]) from checkpoint, the shape in current model is torch.Size([2304]).
        size mismatch for vision_model.encoder.layers.7.self_attn.projection.weight: copying a param with shape torch.Size([1408, 1408]) from checkpoint, the shape in current model is torch.Size([768, 768]).
        size mismatch for vision_model.encoder.layers.7.self_attn.projection.bias: copying a param with shape torch.Size([1408]) from checkpoint, the shape in current model is torch.Size([768]).
        size mismatch for vision_model.encoder.layers.7.layer_norm1.weight: copying a param with shape torch.Size([1408]) from checkpoint, the shape in current model is torch.Size([768]).
        size mismatch for vision_model.encoder.layers.7.layer_norm1.bias: copying a param with shape torch.Size([1408]) from checkpoint, the shape in current model is torch.Size([768]).
        size mismatch for vision_model.encoder.layers.7.mlp.fc1.weight: copying a param with shape torch.Size([6144, 1408]) from checkpoint, the shape in current model is torch.Size([3072, 768]).
        size mismatch for vision_model.encoder.layers.7.mlp.fc1.bias: copying a param with shape torch.Size([6144]) from checkpoint, the shape in current model is torch.Size([3072]).
        size mismatch for vision_model.encoder.layers.7.mlp.fc2.weight: copying a param with shape torch.Size([1408, 6144]) from checkpoint, the shape in current model is torch.Size([768, 3072]).
        size mismatch for vision_model.encoder.layers.7.mlp.fc2.bias: copying a param with shape torch.Size([1408]) from checkpoint, the shape in current model is torch.Size([768]).
        size mismatch for vision_model.encoder.layers.7.layer_norm2.weight: copying a param with shape torch.Size([1408]) from checkpoint, the shape in current model is torch.Size([768]).
        size mismatch for vision_model.encoder.layers.7.layer_norm2.bias: copying a param with shape torch.Size([1408]) from checkpoint, the shape in current model is torch.Size([768]).
        size mismatch for vision_model.encoder.layers.8.self_attn.qkv.weight: copying a param with shape torch.Size([4224, 1408]) from checkpoint, the shape in current model is torch.Size([2304, 768]).
        size mismatch for vision_model.encoder.layers.8.self_attn.qkv.bias: copying a param with shape torch.Size([4224]) from checkpoint, the shape in current model is torch.Size([2304]).
        size mismatch for vision_model.encoder.layers.8.self_attn.projection.weight: copying a param with shape torch.Size([1408, 1408]) from checkpoint, the shape in current model is torch.Size([768, 768]).
        size mismatch for vision_model.encoder.layers.8.self_attn.projection.bias: copying a param with shape torch.Size([1408]) from checkpoint, the shape in current model is torch.Size([768]).
        size mismatch for vision_model.encoder.layers.8.layer_norm1.weight: copying a param with shape torch.Size([1408]) from checkpoint, the shape in current model is torch.Size([768]).
        size mismatch for vision_model.encoder.layers.8.layer_norm1.bias: copying a param with shape torch.Size([1408]) from checkpoint, the shape in current model is torch.Size([768]).
        size mismatch for vision_model.encoder.layers.8.mlp.fc1.weight: copying a param with shape torch.Size([6144, 1408]) from checkpoint, the shape in current model is torch.Size([3072, 768]).
        size mismatch for vision_model.encoder.layers.8.mlp.fc1.bias: copying a param with shape torch.Size([6144]) from checkpoint, the shape in current model is torch.Size([3072]).
        size mismatch for vision_model.encoder.layers.8.mlp.fc2.weight: copying a param with shape torch.Size([1408, 6144]) from checkpoint, the shape in current model is torch.Size([768, 3072]).
        size mismatch for vision_model.encoder.layers.8.mlp.fc2.bias: copying a param with shape torch.Size([1408]) from checkpoint, the shape in current model is torch.Size([768]).
        size mismatch for vision_model.encoder.layers.8.layer_norm2.weight: copying a param with shape torch.Size([1408]) from checkpoint, the shape in current model is torch.Size([768]).
        size mismatch for vision_model.encoder.layers.8.layer_norm2.bias: copying a param with shape torch.Size([1408]) from checkpoint, the shape in current model is torch.Size([768]).
        size mismatch for vision_model.encoder.layers.9.self_attn.qkv.weight: copying a param with shape torch.Size([4224, 1408]) from checkpoint, the shape in current model is torch.Size([2304, 768]).
        size mismatch for vision_model.encoder.layers.9.self_attn.qkv.bias: copying a param with shape torch.Size([4224]) from checkpoint, the shape in current model is torch.Size([2304]).
        size mismatch for vision_model.encoder.layers.9.self_attn.projection.weight: copying a param with shape torch.Size([1408, 1408]) from checkpoint, the shape in current model is torch.Size([768, 768]).
        size mismatch for vision_model.encoder.layers.9.self_attn.projection.bias: copying a param with shape torch.Size([1408]) from checkpoint, the shape in current model is torch.Size([768]).
        size mismatch for vision_model.encoder.layers.9.layer_norm1.weight: copying a param with shape torch.Size([1408]) from checkpoint, the shape in current model is torch.Size([768]).
        size mismatch for vision_model.encoder.layers.9.layer_norm1.bias: copying a param with shape torch.Size([1408]) from checkpoint, the shape in current model is torch.Size([768]).
        size mismatch for vision_model.encoder.layers.9.mlp.fc1.weight: copying a param with shape torch.Size([6144, 1408]) from checkpoint, the shape in current model is torch.Size([3072, 768]).
        size mismatch for vision_model.encoder.layers.9.mlp.fc1.bias: copying a param with shape torch.Size([6144]) from checkpoint, the shape in current model is torch.Size([3072]).
        size mismatch for vision_model.encoder.layers.9.mlp.fc2.weight: copying a param with shape torch.Size([1408, 6144]) from checkpoint, the shape in current model is torch.Size([768, 3072]).
        size mismatch for vision_model.encoder.layers.9.mlp.fc2.bias: copying a param with shape torch.Size([1408]) from checkpoint, the shape in current model is torch.Size([768]).
        size mismatch for vision_model.encoder.layers.9.layer_norm2.weight: copying a param with shape torch.Size([1408]) from checkpoint, the shape in current model is torch.Size([768]).
        size mismatch for vision_model.encoder.layers.9.layer_norm2.bias: copying a param with shape torch.Size([1408]) from checkpoint, the shape in current model is torch.Size([768]).
        size mismatch for vision_model.encoder.layers.10.self_attn.qkv.weight: copying a param with shape torch.Size([4224, 1408]) from checkpoint, the shape in current model is torch.Size([2304, 768]).
        size mismatch for vision_model.encoder.layers.10.self_attn.qkv.bias: copying a param with shape torch.Size([4224]) from checkpoint, the shape in current model is torch.Size([2304]).
        size mismatch for vision_model.encoder.layers.10.self_attn.projection.weight: copying a param with shape torch.Size([1408, 1408]) from checkpoint, the shape in current model is torch.Size([768, 768]).
        size mismatch for vision_model.encoder.layers.10.self_attn.projection.bias: copying a param with shape torch.Size([1408]) from checkpoint, the shape in current model is torch.Size([768]).
        size mismatch for vision_model.encoder.layers.10.layer_norm1.weight: copying a param with shape torch.Size([1408]) from checkpoint, the shape in current model is torch.Size([768]).
        size mismatch for vision_model.encoder.layers.10.layer_norm1.bias: copying a param with shape torch.Size([1408]) from checkpoint, the shape in current model is torch.Size([768]).
        size mismatch for vision_model.encoder.layers.10.mlp.fc1.weight: copying a param with shape torch.Size([6144, 1408]) from checkpoint, the shape in current model is torch.Size([3072, 768]).
        size mismatch for vision_model.encoder.layers.10.mlp.fc1.bias: copying a param with shape torch.Size([6144]) from checkpoint, the shape in current model is torch.Size([3072]).
        size mismatch for vision_model.encoder.layers.10.mlp.fc2.weight: copying a param with shape torch.Size([1408, 6144]) from checkpoint, the shape in current model is torch.Size([768, 3072]).
        size mismatch for vision_model.encoder.layers.10.mlp.fc2.bias: copying a param with shape torch.Size([1408]) from checkpoint, the shape in current model is torch.Size([768]).
        size mismatch for vision_model.encoder.layers.10.layer_norm2.weight: copying a param with shape torch.Size([1408]) from checkpoint, the shape in current model is torch.Size([768]).
        size mismatch for vision_model.encoder.layers.10.layer_norm2.bias: copying a param with shape torch.Size([1408]) from checkpoint, the shape in current model is torch.Size([768]).
        size mismatch for vision_model.encoder.layers.11.self_attn.qkv.weight: copying a param with shape torch.Size([4224, 1408]) from checkpoint, the shape in current model is torch.Size([2304, 768]).
        size mismatch for vision_model.encoder.layers.11.self_attn.qkv.bias: copying a param with shape torch.Size([4224]) from checkpoint, the shape in current model is torch.Size([2304]).
        size mismatch for vision_model.encoder.layers.11.self_attn.projection.weight: copying a param with shape torch.Size([1408, 1408]) from checkpoint, the shape in current model is torch.Size([768, 768]).
        size mismatch for vision_model.encoder.layers.11.self_attn.projection.bias: copying a param with shape torch.Size([1408]) from checkpoint, the shape in current model is torch.Size([768]).
        size mismatch for vision_model.encoder.layers.11.layer_norm1.weight: copying a param with shape torch.Size([1408]) from checkpoint, the shape in current model is torch.Size([768]).
        size mismatch for vision_model.encoder.layers.11.layer_norm1.bias: copying a param with shape torch.Size([1408]) from checkpoint, the shape in current model is torch.Size([768]).
        size mismatch for vision_model.encoder.layers.11.mlp.fc1.weight: copying a param with shape torch.Size([6144, 1408]) from checkpoint, the shape in current model is torch.Size([3072, 768]).
        size mismatch for vision_model.encoder.layers.11.mlp.fc1.bias: copying a param with shape torch.Size([6144]) from checkpoint, the shape in current model is torch.Size([3072]).
        size mismatch for vision_model.encoder.layers.11.mlp.fc2.weight: copying a param with shape torch.Size([1408, 6144]) from checkpoint, the shape in current model is torch.Size([768, 3072]).
        size mismatch for vision_model.encoder.layers.11.mlp.fc2.bias: copying a param with shape torch.Size([1408]) from checkpoint, the shape in current model is torch.Size([768]).
        size mismatch for vision_model.encoder.layers.11.layer_norm2.weight: copying a param with shape torch.Size([1408]) from checkpoint, the shape in current model is torch.Size([768]).
        size mismatch for vision_model.encoder.layers.11.layer_norm2.bias: copying a param with shape torch.Size([1408]) from checkpoint, the shape in current model is torch.Size([768]).
        size mismatch for vision_model.post_layernorm.weight: copying a param with shape torch.Size([1408]) from checkpoint, the shape in current model is torch.Size([768]).
        size mismatch for vision_model.post_layernorm.bias: copying a param with shape torch.Size([1408]) from checkpoint, the shape in current model is torch.Size([768]).
        You may consider adding `ignore_mismatched_sizes=True` in the model `from_pretrained` method.






    
