Traceback (most recent call last):
  File "/home/airlab/Desktop/Jingwen/MAPLMTest/baseline/evaluation/data_processed_concat.py", line 145, in <module>
    train_model(qa_data, feature_method="concatenate")  # 或者改为 "concatenate"
    ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/airlab/Desktop/Jingwen/MAPLMTest/baseline/evaluation/data_processed_concat.py", line 106, in train_model
    inputs, targets = process_multimodal_data(qa_data, feature_method)
                      ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/airlab/Desktop/Jingwen/MAPLMTest/baseline/evaluation/data_processed_concat.py", line 80, in process_multimodal_data
    pixel_values = concatenate_image_features(images)  # 拼接特征
                   ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/airlab/Desktop/Jingwen/MAPLMTest/baseline/evaluation/data_processed_concat.py", line 58, in concatenate_image_features
    pixel_values = torch.cat(pixel_values,dim=1)
                   ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
RuntimeError: CUDA error: out of memory
CUDA kernel errors might be asynchronously reported at some other API call, so the stacktrace below might be incorrect.
For debugging consider passing CUDA_LAUNCH_BLOCKING=1
Compile with `TORCH_USE_CUDA_DSA` to enable device-side assertions
