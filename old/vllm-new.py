from vllm import EngineArgs, LLMEngine, SamplingParams, RequestOutput
from vllm.sequence import SamplerOutput, SequenceData, SequenceGroupMetadata
#vllm part

engine_args = EngineArgs(model="facebook/opt-125m")
engine = LLMEngine.from_engine_args(engine_args)

sentences_batch = ['test is not a test']
batch = engine.tokenizer.tokenizer.batch_encode_plus(
                sentences_batch,
                padding=True
            )['input_ids']
seqs = []
for group_id in range(len(batch)):
    seq_data = SequenceData(list(batch[group_id]))
    seq = SequenceGroupMetadata(
        request_id=str(group_id),
        is_prompt=True,
        seq_data={group_id: seq_data},
        sampling_params=SamplingParams(top_p=0.99),
        block_tables=None,
    )
    seqs.append(seq)

input_tokens, input_positions, input_metadata, return_prompt_lens, _, _, _, _ = (
    engine.driver_worker.model_runner._prepare_prompt(seqs))

num_layers = engine.driver_worker.model_config.get_num_layers(engine.driver_worker.parallel_config)
outputs = engine.driver_worker.model_runner.model(input_ids=input_tokens,
        positions=input_positions,
        kv_caches=[(None, None)] * num_layers,
        input_metadata=input_metadata)

outputs = outputs[:, -1, :]
outputs.size()

# transformers part

from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
model = AutoModelForCausalLM.from_pretrained("facebook/opt-125m")
tokenizer = AutoTokenizer.from_pretrained("facebook/opt-125m")

sentences_batch = ['test is not a test']
batch = tokenizer.batch_encode_plus(
                sentences_batch,
                padding=True,
                return_tensors = 'pt'
        )

with torch.no_grad():
    hidden = model(output_hidden_states= True, **batch).hidden_states

outputs1= hidden[-1][:,-1,:]

# should be 1 , the same value

print(torch.nn.functional.cosine_similarity(outputs.float().cpu(),outputs1.float().cpu()))


