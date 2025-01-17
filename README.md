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
