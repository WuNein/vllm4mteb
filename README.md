# vllm4mteb
## Notice
### Problem 1
https://github.com/vllm-project/vllm/blob/main/vllm/model_executor/models/llama_embedding.py is not working with raw LLM!

这个脚本在没有用Sentence bert训练过的模型上是用不了的，建议参考我这个项目的MTEB

https://github.com/vllm-project/vllm/blob/4ef41b84766670c1bd8079f58d35bf32b5bcb3ab/vllm/model_executor/models/llama_embedding.py#L77 Problem is here

问题出在加载权重上。

### Problem 2 **Don't miss architecture change**
```
{
  "_name_or_path": "princeton-nlp/Sheared-LLaMA-1.3B",
  "architectures": [
    "LlamaForCausalLM"
  ],
  "bos_token_id": 1,
  "eos_token_id": 2,
  "hidden_act": "silu",
  "hidden_size": 2048,
  "initializer_range": 0.02,
  "intermediate_size": 5504,
  "max_position_embeddings": 4096,
  "model_type": "llama",
  "num_attention_heads": 16,
  "num_hidden_layers": 24,
  "num_key_value_heads": 16,
  "pad_token_id": 0,
  "pretraining_tp": 1,
  "rms_norm_eps": 1e-05,
  "rope_scaling": null,
  "tie_word_embeddings": false,
  "torch_dtype": "float32",
  "transformers_version": "4.28.1",
  "use_cache": true,
  "vocab_size": 32000
}
```
Change / 修改
```
{
  ...
  "architectures": [
    "MyLlamaEmbeddingModel"
  ],
  ...
}
```
修改成和你自定义的Embedding Class一样的architectures，目的是让vLLM认为是你新实现的模型！

```
[rank0]: File "python3.11/site-packages/vllm/worker/embedding_model_runner.py", line 122, in execute_model
[rank0]: self.model.pooler(hidden_states=hidden_states,
[rank0]: ^^^^^^^^^^^^^^^^^
[rank0]: File "python3.11/site-packages/torch/nn/modules/module.py", line 1729, in getattr
[rank0]: raise AttributeError(f"'{type(self).name}' object has no attribute '{name}'")
[rank0]: AttributeError: 'LlamaForCausalLM' object has no attribute 'pooler'
```
In case, this problem occur. 不然的话就出现这个问题！LlamaForCausalLM这个模型架构明显不对！

## New Instructions

> The project is nearing its conclusion! It now supports the latest vLLM!
![alt text](assets/image-2.png)

vLLM has basic support for Embedding, though it's not very convenient. Pooling only supports one method, which we suggest solving with a MonkeyPatch.

For the final solution, refer to `vllm4emb.ipynb`, which provides a way to solve this without modifying the codebase. The principle is to register our Embedding model. Essentially, it's just importing a few files and copying and pasting some code.

```python
from vllm import ModelRegistry
ModelRegistry.register_model("MyLlamaEmbeddingModel", MyLlamaEmbeddingModel)
```

See the detailed comments in the notebook. You need to patch the `ModelRegistry.is_embedding_model` method, as the author did not anticipate the need for registering out-of-the-box Embedding models.

> https://github.com/vllm-project/vllm/blob/388596c91437a51d428a447594e9faec340c29b2/vllm/model_executor/layers/pooler.py#L44
This implementation should support Qwen's tiktoken tokenizer, so compatibility issues are minimal.

## 新的说明

> 本项目应该迎来大结局了！支持最新的vLLM！
![alt text](assets/image-2.png)

vLLM算是正常支持Embedding了，只是支持的不太舒服。Pooling也只支持一种，这个建议MonkeyPatch来解决。

最后的解决办法请见`vllm4emb.ipynb`，提供了一种不修改codebase的方法来解决，原理就是把我们的Embedding模型注册上去，其实就是导入几个文件，复制黏贴一下就行了。

```
from vllm import ModelRegistry
ModelRegistry.register_model("MyLlamaEmbeddingModel", MyLlamaEmbeddingModel)
```

详细注释看ipynb，需要Patch `ModelRegistry.is_embedding_model` 这个方法，这个作者并没有考虑额外注册OOT的Embedding模型的需求。

> https://github.com/vllm-project/vllm/blob/388596c91437a51d428a447594e9faec340c29b2/vllm/model_executor/layers/pooler.py#L44
他这个写法应该是支持Qwen的tiktoken分词的，所以基本上兼容问题不大了。



## 旧的说明
vllm for embedding tasks

https://github.com/kongds/scaling_sentemb

This is the code to accelate the inference of scaling sentemb project.

You need to change the model info in python code.

Then

python run_array_decoder_vllm.py

I make two examples of not using vllm, and one with.

![Alt text](assets/image.png)

The result is same.

## Example

1. git clone lora from https://huggingface.co/royokong/prompteol-opt-2.7b
2. git clone opt2.7b
3. python run_array_decoder_vllm.py --lora_weight prompteol-opt-2.7b

## Dependence

vllm <= 0.22

After 0.22, they change the api. 新版本代码已测试，等待修改

I was using vllm 0.21, when initially develop this project.

## 针对最新版 Latest vllm

![Alt text](assets/image-1.png)
感谢@guankaisi的提醒，vllm的函数改了。
目前做了一个demo (vllm-new 文件)，新版的函数也可以用。晚些再改。


## 其他

我颇为确定 https://arxiv.org/pdf/2401.00368.pdf intfloat/e5-mistral-7b-instruct 没有充分利用模型，我手头就有一个STS比他高的。

用Qwen embedding的代码我改了一下，Qwen Batch的话，他是tiktoken不能取-1，要根据attention mask来算。然后Qwen这种模型，使用中文Prompt还是英文Prompt好，请自己试一下，结果跟我说说？

训练这个东西大Batch size是必须的。
