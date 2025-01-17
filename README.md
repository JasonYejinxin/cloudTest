The tokenizer class you load from this checkpoint is not the same type as the class this function is called from. It may result in unexpected tokenization. 
The tokenizer class you load from this checkpoint is 'T5Tokenizer'. 
The class this function is called from is 'BertTokenizerFast'.
Some kwargs in processor config are unused and will not have any effect: num_query_tokens. 
You are using a model of type blip-2 to instantiate a model of type blip. This is not supported for all configurations of models and can yield errors.

Traceback (most recent call last):
  File "/home/airlab/Desktop/Jingwen/MAPLMTest/baseline/evaluation/data_processed_concat.py", line 13, in <module>
    model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip2-flan-t5-xl")
            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/airlab/anaconda3/lib/python3.12/site-packages/transformers/modeling_utils.py", line 4264, in from_pretrained
    ) = cls._load_pretrained_model(
        ^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/airlab/anaconda3/lib/python3.12/site-packages/transformers/modeling_utils.py", line 4834, in _load_pretrained_model
    raise RuntimeError(f"Error(s) in loading state_dict for {model.__class__.__name__}:\n\t{error_msg}")
RuntimeError: Error(s) in loading state_dict for BlipForConditionalGeneration:

You are using a model of type blip-2 to instantiate a model of type blip. This is not supported for all configurations of models and can yield errors.

{
  "architectures": [
    "Blip2ForConditionalGeneration"
  ],
  "image_text_hidden_size": 256,
  "image_token_index": 32100,
  "initializer_factor": 1.0,
  "initializer_range": 0.02,
  "model_type": "blip-2",
  "num_query_tokens": 32,
  "qformer_config": {
    "classifier_dropout": null,
    "model_type": "blip_2_qformer"
  },
  "text_config": {
    "architectures": [
      "T5ForConditionalGeneration"
    ],
    "bos_token_id": 1,
    "classifier_dropout": 0.0,
    "d_ff": 5120,
    "d_kv": 64,
    "d_model": 2048,
    "decoder_start_token_id": 0,
    "dense_act_fn": "gelu",
    "dropout_rate": 0.1,
    "eos_token_id": 1,
    "feed_forward_proj": "gated-gelu",
    "initializer_factor": 1.0,
    "is_encoder_decoder": true,
    "is_gated_act": true,
    "layer_norm_epsilon": 1e-06,
    "model_type": "t5",
    "n_positions": 512,
    "num_decoder_layers": 24,
    "num_heads": 32,
    "num_layers": 24,
    "output_past": true,
    "pad_token_id": 0,
    "relative_attention_max_distance": 128,
    "relative_attention_num_buckets": 32,
    "task_specific_params": {
      "summarization": {
        "early_stopping": true,
        "length_penalty": 2.0,
        "max_length": 200,
        "min_length": 30,
        "no_repeat_ngram_size": 3,
        "num_beams": 4,
        "prefix": "summarize: "
      },
      "translation_en_to_de": {
        "early_stopping": true,
        "max_length": 300,
        "num_beams": 4,
        "prefix": "translate English to German: "
      },
      "translation_en_to_fr": {
        "early_stopping": true,
        "max_length": 300,
        "num_beams": 4,
        "prefix": "translate English to French: "
      },
      "translation_en_to_ro": {
        "early_stopping": true,
        "max_length": 300,
        "num_beams": 4,
        "prefix": "translate English to Romanian: "
      }
    },
    "tie_word_embeddings": false,
    "torch_dtype": "float32",
    "vocab_size": 32128
  },
  "torch_dtype": "float32",
  "transformers_version": "4.47.0.dev0",
  "use_decoder_only_language_model": false,
  "vision_config": {
    "dropout": 0.0,
    "initializer_factor": 1.0,
    "model_type": "blip_2_vision_model",
    "num_channels": 3,
    "projection_dim": 512
  }
}
