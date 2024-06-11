# gguf-mindspore

This project helps users to quickly convert the ckpt files of the big models trained by ***MindSpore to GGUF*** files, which can be imported into ***Ollama*** and quickly deployed to use the corresponding big models.



## Requirements

```python
gguf==0.6.0
mindspore==2.2.14
numpy==1.26.4
```



## Quick start

Execute ***models/llama2/example.py*** to get example.gguf.



## How to convert your own big Model

To convert your own big model of MindSpore to GGUF, you need to do as follows:

1. build a new module
2. build ***metadata key-value pairs*** json file.
3. build ***layer name mapper*** json file.
4. write your ***example.py*** and ***generate your example.gguf***.
5. write your ***ModelFile*** file, and create your model on Ollama
6. run your model



## LLama2 as example

#### Step 1: Build a new module

We need to build a new module like ***llama2*** in ***./models***.



#### Step 2: Build metadata key-value pairs json file

First we need to download a llama2-7b model from hugging face, like https://huggingface.co/TheBloke/Llama-2-7B-GGUF/tree/main.

and put it in ***./models/llama2***, then we use ***models/llama2/make_gguf_meta_data_json.py*** to generate ***metadata key-value pairs json file***



#### Step 3: Build layer name mapper json file

We need to compare ***tensor infos*** between your ***ckpt file and gguf file***, and write a json file like below.

```json
{
  "model.tok_embeddings.embedding_weight": "token_embd.weight",
  "model.layers": "blk",
  "attention.wq": "attn_q",
  "attention.wk": "attn_k",
  "attention.wv": "attn_v",
  "attention.wo": "attn_output",
  "attention_norm": "attn_norm",
  "feed_forward.w1": "ffn_gate",
  "feed_forward.w3": "ffn_up",
  "feed_forward.w2": "ffn_down",
  "ffn_norm": "ffn_norm",
  "model.norm_out.weight": "output_norm.weight",
  "lm_head": "output"
}
```



#### Step 4: Write your example.py and generate your example.gguf

We just need to change the params of the Class **Llama2Writer**, and excute this file to get example.gguf

```python
if __name__ == '__main__':
    writer = Llama2Writer(metadata_json_path="llama2-7b-gguf-metadata.json",
                          layer_name_map_json_path="llama2_layer_name_map.json",
                          ckpt_file_path="llama2_7b.ckpt")
    writer.write()
```

Note: we fixed all the formats of tensors ***GGMLQuantizationType.F32*** , you can quantize using ***llama.cpp*** as you wish.



#### Step 5: Write your ModelFile file, and create your model on Ollama

You need to write a ModelFile, you can find tutorial on Google.



#### Step 6: Run your model

You can run your model on Ollama.



