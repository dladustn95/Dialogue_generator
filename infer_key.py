import os
import logging
import random
from argparse import ArgumentParser
from itertools import chain
from pprint import pformat
import warnings
import datetime

import torch
import torch.nn.functional as F
from tqdm import tqdm

from transformers import OpenAIGPTLMHeadModel, OpenAIGPTTokenizer, GPT2LMHeadModel, GPT2Tokenizer
from train_key import build_input_from_segments
from utils import get_test_dataset_key, download_pretrained_model

from kogpt2.pytorch_kogpt2 import get_pytorch_conkogpt2_model2
from gluonnlp.data import SentencepieceTokenizer
from kogpt2.utils import get_tokenizer

from pytorch_pretrained_bert.modeling import BertModel
from pytorch_pretrained_bert.tokenization2 import BertTokenizer

from kogpt2.model.Seq2Seq import Seq2Seq

def top_filtering(logits, top_k=0., top_p=0.9, threshold=-float('Inf'), filter_value=-float('Inf')):
    """ Filter a distribution of logits using top-k, top-p (nucleus) and/or threshold filtering
        Args:
            logits: logits distribution shape (vocabulary size)
            top_k: <=0: no filtering, >0: keep only top k tokens with highest probability.
            top_p: <=0.0: no filtering, >0.0: keep only a subset S of candidates, where S is the smallest subset
                whose total probability mass is greater than or equal to the threshold top_p.
                In practice, we select the highest probability tokens whose cumulative probability mass exceeds
                the threshold top_p.
            threshold: a minimal threshold to keep logits
    """
    assert logits.dim() == 1  # Only work for batch size 1 for now - could update but it would obfuscate a bit the code
    top_k = min(top_k, logits.size(-1))
    if top_k > 0:
        # Remove all tokens with a probability less than the last token in the top-k tokens
        indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
        logits[indices_to_remove] = filter_value

    if top_p > 0.0:
        # Compute cumulative probabilities of sorted tokens
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probabilities = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

        # Remove tokens with cumulative probability above the threshold
        sorted_indices_to_remove = cumulative_probabilities > top_p
        # Shift the indices to the right to keep also the first token above the threshold
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0

        # Back to unsorted indices and set them to -infinity
        indices_to_remove = sorted_indices[sorted_indices_to_remove]
        logits[indices_to_remove] = filter_value

    indices_to_remove = logits < threshold
    logits[indices_to_remove] = filter_value

    return logits


def sample_sequence(source, attention, bert_tokenizer, model, gpt_vocab, args, current_output=None):
    bos, eos = gpt_vocab[gpt_vocab.bos_token], gpt_vocab[gpt_vocab.eos_token]
    if current_output is None:
        current_output = []

    for i in range(args.max_length):
        instance = build_input_from_segments(source, current_output, attention, bert_tokenizer, gpt_vocab, with_eos=False)

        source_ids = torch.tensor([instance["source_ids"]], device=args.device)
        target_ids = torch.tensor(instance["target_ids"], device=args.device)
        keyword_scores = torch.tensor([instance["key_scores"]], device=args.device)

        #logits = model(input_ids, token_type_ids=token_type_ids)
        logits = model(source_ids, target_ids, key_score=keyword_scores)
        if isinstance(logits, tuple):  # for gpt2 and maybe others
            logits = logits[0]
        logits = logits[-1, :] / args.temperature

        """
        if i < args.min_length:
            logits[1] = -float("inf")
            logits[316] = -float("inf")
            logits[2530] = -float("inf")
            logits[6026] = -float("inf")
            logits[47812] = -float("inf")
            logits[47440] = -float("inf")
            logits[47774] = -float("inf")
            logits[47453] = -float("inf")
        """

        logits = top_filtering(logits, top_k=args.top_k, top_p=args.top_p)
        probs = F.softmax(logits, dim=-1)

        prev = torch.topk(probs, 1)[1] if args.no_sample else torch.multinomial(probs, 1)
        if i < args.min_length and prev.item() in [bos, eos]:
            while prev.item() in [bos, eos]:
                if probs.max().item() == 1:
                    warnings.warn("Warning: model generating special token with probability 1.")
                    break  # avoid infinitely looping over special token
                elif torch.multinomial(probs, 1).item() in [bos, eos]:
                    break
                prev = torch.multinomial(probs, num_samples=1)
        elif prev[0].item() in [bos, eos]:
            break
        current_output.append(prev[0].item())
    return current_output


def run():
    parser = ArgumentParser()
    parser.add_argument("--dataset_path", type=str, default="", help="Path or url of the dataset. If empty download from S3.")
    parser.add_argument("--use_adapter", type=bool, default=True, help="Use adapter or not")
    parser.add_argument("--keyword_Module", type=str, default="", help="add, attention, ")
    parser.add_argument("--model", type=str, default="openai-gpt", help="Model type (openai-gpt or gpt2)", choices=['openai-gpt', 'gpt2'])  # anything besides gpt2 will load openai-gpt
    parser.add_argument("--model_checkpoint", type=str, default="", help="Path, url or short name of the model")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Device (cuda or cpu)")
    parser.add_argument("--bert_model_path", default="./", type=str, help="Bert pre-trained model path")
    parser.add_argument("--vocab_file", default="./vocab.korean.rawtext.list", type=str, help="The vocabulary file that the BERT model was trained on.")
    parser.add_argument("--no_sample", action='store_true', help="Set to use greedy decoding instead of sampling")
    parser.add_argument("--max_length", type=int, default=50, help="Maximum length of the output utterances")
    parser.add_argument("--min_length", type=int, default=1, help="Minimum length of the output utterances")
    parser.add_argument("--seed", type=int, default=0, help="Seed")
    parser.add_argument("--temperature", type=int, default=0.8, help="Sampling softmax temperature")
    parser.add_argument("--top_k", type=int, default=30, help="Filter top-k tokens before sampling (<=0: no filtering)")
    parser.add_argument("--top_p", type=float, default=0.9, help="Nucleus filtering (top-p) before sampling (<=0.0: no filtering)")
    parser.add_argument("--do_lower_case", action='store_true', help="Set this flag if you are using an uncased model.")

    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__file__)
    logger.info(pformat(args))

    if args.model_checkpoint == "":
        if args.model == 'gpt2':
            raise ValueError("Interacting with GPT2 requires passing a finetuned model_checkpoint")
        else:
            args.model_checkpoint = download_pretrained_model()

    if args.seed != 0:
        random.seed(args.seed)
        torch.random.manual_seed(args.seed)
        torch.cuda.manual_seed(args.seed)

    logger.info("Get pretrained model and tokenizer")

    # Load KoBERT model and tokenizer
    bert_tokenizer = BertTokenizer.from_pretrained(args.vocab_file, do_lower_case=args.do_lower_case)
    bert_model = BertModel.from_pretrained(args.bert_model_path)
    bert_model.to(args.device)

    # Load KoGPT2 model and tokenizer
    tok_path = get_tokenizer()
    gpt_model, gpt_vocab = get_pytorch_conkogpt2_model2(keyword_Module=args.keyword_Module, use_adapter=args.use_adapter)
    gpt_tokenizer = SentencepieceTokenizer(tok_path)
    gpt_model.to(args.device)

    model = Seq2Seq(bert_model, gpt_model, gpt_vocab, args)
    model.load_state_dict(torch.load(args.model_checkpoint), strict=False)
    model.eval()

    logger.info("Load test data")
    sourceList, targetList, attentionList = get_test_dataset_key(bert_tokenizer, gpt_tokenizer, gpt_vocab, args.dataset_path)

    f1 = open((args.model_checkpoint + "_output.txt"), 'w')
    for line in tqdm(zip(sourceList, targetList, attentionList), total=len(sourceList)):
        out_ids = sample_sequence(line[0], line[2], bert_tokenizer, model, gpt_vocab, args)
        out_texts = gpt_vocab.to_tokens(out_ids)
        for text in out_texts:
            f1.write(text.replace('▁', ' ').replace('</s>',' '))
        """
        for id in out_ids:
            f1.write(str(id))
            f1.write(' ')
        """
        f1.write("\n")
    f1.close()



if __name__ == "__main__":
    run()