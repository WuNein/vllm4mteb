import argparse
import logging
import os

logging.basicConfig(level=logging.INFO)

import numpy as np
from mteb import MTEB
from transformers import AutoModel, AutoTokenizer, AutoModelForCausalLM, GenerationConfig
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
    # "BIOSSES",
    "SICK-R",
    "STS12",
    "STS13",
    "STS14",
    "STS15",
    "STS16",
    # "STS17",
    # "STS22", #p2p
    "STSBenchmark",
    # "SummEval",
]

TASK_LIST = (
    TASK_LIST_CLASSIFICATION
    + TASK_LIST_CLUSTERING
    + TASK_LIST_PAIR_CLASSIFICATION
    + TASK_LIST_RERANKING
    + TASK_LIST_RETRIEVAL
    + TASK_LIST_STS
)

# from peft import PeftModel
from qwen_generation_utils import (
    HistoryType,
    make_context,
    decode_tokens,
    get_stop_words_ids,
    StopWordsLogitsProcessor,
)

class DecoderWrapper:
    def __init__(
        self,
        modelpath="",
        lora_weight="",
        mask_embedding_sentence_template='This_sentence_:_"of_or_relating_to_tutors_or_tutoring."_means_in_one_word:"Tutorial".This_passage_:_"*sent_0*"_means_in_one_word:"',
        avg=False,
        bf16 = False,
    ):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.max_length = 512
        self.tokenizer = AutoTokenizer.from_pretrained(
            modelpath,
            model_max_length=self.max_length,
            padding_side="right", #right in ft code
            use_fast=False,
            trust_remote_code=True,
        )
        self.tokenizer.pad_token_id = self.tokenizer.eod_id
        
        # self.tokenizer.pad_token_id = (
        #     0  # unk. we want this to be different from the eos token
        # )
        # self.tokenizer.padding_side = "left"  # Allow batched inference
        self.model = AutoModelForCausalLM.from_pretrained(
            modelpath,
            device_map="auto", trust_remote_code=True, fp16=True
        )
        self.model.generation_config = GenerationConfig.from_pretrained(modelpath, trust_remote_code=True)
        # self.model = PeftModel.from_pretrained(
        #     self.model,
        #     lora_weight,
        #     torch_dtype=torch.float16 if bf16 == False else torch.bfloat16,
        #     device_map={"": 0},
        # )
        self.model.eval()
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
        max_length = self.max_length
        sentences = deepcopy(raw_sentences)

        all_embeddings = []
        # length_sorted_idx = np.argsort([len(sen) for sen in sentences])
        # sentences_sorted = [sentences[idx] for idx in length_sorted_idx]

        for start_index in range(0, len(sentences), batch_size):
            sentences_batch = sentences[start_index:start_index+batch_size]

            max_length = 0
            input_ids = []
            self.tokenizer.pad_token_id = self.tokenizer.eod_id
            for j, sentence in enumerate(sentences_batch):
                # query = self.tokenizer.from_list_format([
                #     {'text': f"""高等学校或研究机关中指导他人学习、进修、或撰写学术论文的教师或科研人员。\n上面这句话用一个中文词来表达是：导师。\n"""},
                #     {'text': f"""{sentence}\n上面这句话用一个中文词来表达是："""},
                # ])
                query = self.tokenizer.from_list_format([
                    {'text': f"""This sentence : "of or relating to tutors or tutoring." means in one English word:Tutorial.\nThis sentence : "{sentence}" means in one English word:"""},
                ])
                raw_text, context_tokens = make_context(
                    self.tokenizer,
                    query,
                    history=None,
                    system="You are a helpful assistant.",
                    max_window_size=self.model.generation_config.max_window_size,
                    chat_format=self.model.generation_config.chat_format,
                )
                # print(raw_text)
                # input_id = self.tokenizer(sentence).input_ids
                input_ids.append(context_tokens)
                max_length = max(max_length, len(context_tokens))
            # max_length += 10
            padding_lengths = []
            for j in range(len(input_ids)):
                padding_lengths.append(max_length - len(input_ids[j]))
                input_ids[j] += [self.tokenizer.eod_id] * (max_length - len(input_ids[j]))
            
            input_ids = torch.tensor(input_ids, dtype=torch.int).to('cuda')
            outputs = []
            with torch.no_grad():
                attention_mask = input_ids.ne(self.tokenizer.eod_id)
                hidden_states = self.model(
                    input_ids= input_ids,attention_mask = attention_mask,output_hidden_states=True, return_dict=True
                ).hidden_states
                last_true_indices = torch.max(attention_mask.int().cumsum(dim=1), dim=1).indices
                outputs = [hidden_states[-1][i, index, :].float().cpu().numpy() for i, index in enumerate(last_true_indices)]

            all_embeddings.extend(outputs)
        return all_embeddings


def parse_args():
    # Parse command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--startid", type=int)
    parser.add_argument("--endid", type=int)
    parser.add_argument("--lang", type=str, default="en")
    parser.add_argument("--taskname", type=str, default=None)
    parser.add_argument("--lora", type=str, default=None)
    parser.add_argument("--batchsize", type=int, default=4)
    args = parser.parse_args()
    return args


def main(args):
    # model = DecoderWrapper(modelpath='/root/hdd/llm/Mistral-7B-v0.1', lora_weight='/root/hdd/scaling_sentemb/mis-lora/checkpoint-600',bf16=True)
    model = DecoderWrapper(modelpath='./Qwen', lora_weight=None,bf16=True)
    # model = DecoderWrapper(modelpath='/root/hdd/llm/falcon-rw-1b', lora_weight=args.lora,bf16=True)

    for task in TASK_LIST_STS:
        print("Running task: ", task)
        eval_splits = ["validation"] if task == "MSMARCO" else ["test"]
        # model_name = 'falcon_'+args.lora.split('-')[-1]
        model_name = 'Qwen'
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
