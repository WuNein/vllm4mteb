import argparse
import logging
import os

logging.basicConfig(level=logging.INFO)


import numpy as np
from mteb import MTEB
from transformers import AutoModel, AutoTokenizer, AutoModelForCausalLM
import torch
from copy import deepcopy


TASK_LIST_CLASSIFICATION = [
    "AmazonCounterfactualClassification",
    "AmazonPolarityClassification",
    "AmazonReviewsClassification",
    "Banking77Classification",
    "EmotionClassification",
    "ImdbClassification",
    "MassiveIntentClassification",
    "MassiveScenarioClassification",
    "MTOPDomainClassification",
    "MTOPIntentClassification",
    "ToxicConversationsClassification",
    "TweetSentimentExtractionClassification",
]

TASK_LIST_CLUSTERING = [
    "ArxivClusteringP2P",
    "ArxivClusteringS2S",
    "BiorxivClusteringP2P",
    "BiorxivClusteringS2S",
    "MedrxivClusteringP2P",
    "MedrxivClusteringS2S",
    "RedditClustering",
    "RedditClusteringP2P",
    "StackExchangeClustering",
    "StackExchangeClusteringP2P",
    "TwentyNewsgroupsClustering",
]

TASK_LIST_PAIR_CLASSIFICATION = [
    "SprintDuplicateQuestions", 
    "TwitterSemEval2015",
    "TwitterURLCorpus",
]

TASK_LIST_RERANKING = [
    "AskUbuntuDupQuestions",
    "MindSmallReranking",
    "SciDocsRR",
    "StackOverflowDupQuestions",
]

TASK_LIST_RETRIEVAL = [
    "ArguAna",
    "ClimateFEVER",
    "CQADupstackAndroidRetrieval",
    "CQADupstackEnglishRetrieval",
    "CQADupstackGamingRetrieval",
    "CQADupstackGisRetrieval",
    "CQADupstackMathematicaRetrieval",
    "CQADupstackPhysicsRetrieval",
    "CQADupstackProgrammersRetrieval",
    "CQADupstackStatsRetrieval",
    "CQADupstackTexRetrieval",
    "CQADupstackUnixRetrieval",
    "CQADupstackWebmastersRetrieval",
    "CQADupstackWordpressRetrieval",
    "DBPedia",
    "FEVER",
    "FiQA2018",
    "HotpotQA",
    "MSMARCO",
    "NFCorpus",
    "NQ",
    "QuoraRetrieval",
    "SCIDOCS",
    "SciFact",
    "Touche2020",
    "TRECCOVID",
]

TASK_LIST_STS = [
    "BIOSSES", #s2s 但是很长
    "SICK-R",
    "STS12",
    "STS13",
    "STS14",
    "STS15",
    "STS16",
    "STS17",
    "STS22", #p2p
    "STSBenchmark",
    "SummEval",
]

TASK_LIST = (
    TASK_LIST_CLASSIFICATION
    + TASK_LIST_CLUSTERING
    + TASK_LIST_PAIR_CLASSIFICATION
    + TASK_LIST_RERANKING
    + TASK_LIST_RETRIEVAL
    + TASK_LIST_STS
)

from peft import PeftModel
from vllm import LLM, SamplingParams
from vllm.sequence import SamplerOutput, SequenceData, SequenceGroupMetadata


class DecoderWrapper:
    def __init__(
        self,
        modelpath="princeton-nlp/sup-simcse-bert-base-uncased",
        lora_weight="",
        mask_embedding_sentence_template='This_passage_:_"*sent_0*"_means_in_one_word:"',
        avg=False,
        bf16 = False,
    ):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # self.tokenizer.padding_side = "left"  # Allow batched inference
        self.model = AutoModelForCausalLM.from_pretrained(
            modelpath,
            output_hidden_states=True,
            trust_remote_code=True,
            torch_dtype=torch.float16 if bf16 == False else torch.bfloat16,
            device_map="auto",
        )
        self.model = PeftModel.from_pretrained(
            self.model,
            lora_weight,
            torch_dtype=torch.float16 if bf16 == False else torch.bfloat16,
            device_map='auto',
        )
        self.model = self.model.merge_and_unload()
        self.model.save_pretrained("./temp")
        del self.model
        #vllm
        self.llm = LLM(model="./temp",tokenizer=modelpath,dtype='float16' if bf16 == False else "bfloat16")
        self.model = self.llm.llm_engine.workers[0].model #opt
        self.tokenizer = self.llm.llm_engine.tokenizer
        # self.tokenizer = AutoTokenizer.from_pretrained(
        #     modelpath, trust_remote_code=True
        # )
        self.tokenizer.pad_token_id = (
            0  # unk. we want this to be different from the eos token
        )
        self.tokenizer.padding_side = "left"
        # self.model.eval()
        self.mask_embedding_sentence_template = mask_embedding_sentence_template
        self.avg = avg
        print(self.mask_embedding_sentence_template)

    def encode(self, raw_sentences, batch_size=32, **kwargs):
        """Returns a list of embeddings for the given sentences.
        Args:
            sentences (`List[str]`): List of sentences to encode
            batch_size (`int`): Batch size for the encoding

        Returns:
            `List[np.ndarray]` or `List[tensor]`: List of embeddings for the given sentences
        """
        # if len(batch) >= 1 and len(batch[0]) >= 1 and isinstance(batch[0][0], bytes):
        #     batch = [[word.decode("utf-8") for word in s] for s in batch]

        # sentences = [" ".join(s) for s in batch]
        # input_sentences = [" ".join(s) for s in batch]
        # if max_length == 500:
        #     sentences = [
        #         self.tokenizer.decode(
        #             self.tokenizer.encode(s, add_special_tokens=False)[:max_length]
        #         )
        #         for s in sentences
        #     ]
        #     max_length = 512
        max_length = 512
        sentences = deepcopy(raw_sentences)

        if (
            # args.mask_embedding_sentence
            # and
            self.mask_embedding_sentence_template
            is not None
        ):
            # *cls*_This_sentence_of_"*sent_0*"_means*mask*.*sep+*
            template = self.mask_embedding_sentence_template
            template = (
                template.replace("_", " ").replace("*sep+*", "").replace("*cls*", "")
            )

            for i, s in enumerate(sentences):
                if len(s) > 0 and s[-1] not in ".?\"'":
                    s += "."
                s = s.replace('"', "'")
                if len(s) > 0 and "?" == s[-1]:
                    s = s[:-1] + "."
                sentences[i] = template.replace("*sent 0*", s).strip()

        all_embeddings = []
        length_sorted_idx = np.argsort([len(sen) for sen in sentences])
        sentences_sorted = [sentences[idx] for idx in length_sorted_idx]

        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        vocab_size = self.llm.llm_engine.workers[0].model.config.vocab_size
        sampling_params = SamplingParams(top_p=0.99, top_k=vocab_size - 1)
        max_num_batched_tokens = self.llm.llm_engine.workers[0].scheduler_config.max_num_batched_tokens
        max_num_seqs = self.llm.llm_engine.workers[0].scheduler_config.max_num_seqs

        for start_index in range(0, len(sentences), batch_size):
            sentences_batch = sentences_sorted[start_index:start_index+batch_size]
            batch = self.tokenizer.batch_encode_plus(
                sentences_batch,
                # return_tensors="pt",
                padding=True,
                max_length=max_length,
                truncation=max_length is not None,
            )['input_ids']
            # Move to the correct device
            # for k in batch:
            #     batch[k] = batch[k].to(self.device) if batch[k] is not None else None
            #    # Get raw embeddings
            # batch = {k: v.to(self.device) for k,v in batch.items()}
            seqs = []
            for group_id in range(len(batch)):
                seq_data = SequenceData(list(batch[group_id]))
                seq = SequenceGroupMetadata(
                    request_id=str(group_id),
                    is_prompt=True,
                    seq_data={group_id: seq_data},
                    sampling_params=sampling_params,
                    block_tables=None,
                )
                seqs.append(seq)

            input_tokens, input_positions, input_metadata = self.llm.llm_engine.workers[0]._prepare_inputs(seqs)
            num_layers = self.llm.llm_engine.workers[0].model_config.get_num_layers(self.llm.llm_engine.workers[0].parallel_config)
            with torch.no_grad():
                outputs = self.llm.llm_engine.workers[0].model.model( #opt
                # outputs = self.llm.llm_engine.workers[0].model.transformer( #falcon
                    input_ids=input_tokens,
                    positions=input_positions,
                    kv_caches=[(None, None)] * num_layers,
                    input_metadata=input_metadata,
                    cache_events=None,
                )
                outputs = outputs[:, -1, :]
            all_embeddings.extend(outputs.float().cpu().numpy())
        all_embeddings = [all_embeddings[idx] for idx in np.argsort(length_sorted_idx)]
        return all_embeddings


def parse_args():
    # Parse command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--startid", type=int)
    parser.add_argument("--endid", type=int)
    parser.add_argument("--lang", type=str, default="en")
    parser.add_argument("--taskname", type=str, default=None)
    parser.add_argument("--lora", type=str, default=None)
    parser.add_argument("--batchsize", type=int, default=32)
    args = parser.parse_args()
    return args


def main(args):
    model = DecoderWrapper(modelpath='/root/hdd/llm/opt-2.7b', lora_weight=args.lora,bf16=False)

    for task in TASK_LIST_STS:
        print("Running task: ", task)
        eval_splits = ["validation"] if task == "MSMARCO" else ["test"]
        model_name = 'opt_27_'+args.lora.split('-')[-1]
        evaluation = MTEB(tasks=[task], task_langs=[args.lang])
        evaluation.run(
            model,
            output_folder=f"results/{model_name}",
            batch_size=args.batchsize,
            eval_splits=eval_splits,
        )


if __name__ == "__main__":
    args = parse_args()
    main(args)
