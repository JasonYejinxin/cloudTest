
from safetensors import safe_open

model_path = "./blip2_flan_t5_epoch_5_concatenate/model.safetensors"

# 读取 safetensors 文件
with safe_open(model_path, framework="pt", device="cpu") as f:
    for key in f.keys():
        print(f"Key: {key}, Shape: {f.get_tensor(key).shape}")






    