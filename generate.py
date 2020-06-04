import os
import logging
import random
from argparse import ArgumentParser
from itertools import chain
from pprint import pformat
import warnings

import torch
import torch.nn.functional as F

from transformers import OpenAIGPTLMHeadModel, OpenAIGPTTokenizer, GPT2LMHeadModel, GPT2Tokenizer
from train import build_input_from_segments
from utils import get_dataset, download_pretrained_model

from kogpt2.pytorch_kogpt2 import get_pytorch_conkogpt2_model
from gluonnlp.data import SentencepieceTokenizer
from kogpt2.utils import get_tokenizer

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="0"
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

def sample_sequence(topic, source, vocab, tokenizer, model, args, current_output=None):
    bos, eos, speaker1, speaker2 = vocab[vocab.bos_token], vocab[vocab.eos_token], vocab[vocab.cls_token], vocab[vocab.sep_token]

    if current_output is None:
        current_output = []
    instance = build_input_from_segments(topic, source, current_output, vocab, tokenizer, with_eos=False)

    input_ids = torch.tensor(instance["input_ids"], device=args.device)
    current_output = model.generate(
        input_ids.unsqueeze(0),
        max_length=50,
        num_beams=8,
        pad_token_id=3,
        early_stopping=True,
    )

    return current_output


def run():
    parser = ArgumentParser()
    parser.add_argument("--dataset_path", type=str, default="",
                        help="Path or url of the dataset. If empty download from S3.")
    parser.add_argument("--model", type=str, default="openai-gpt", help="Model type (openai-gpt or gpt2)",
                        choices=['openai-gpt', 'gpt2'])  # anything besides gpt2 will load openai-gpt
    parser.add_argument("--model_checkpoint", type=str, default="", help="Path, url or short name of the model")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu",
                        help="Device (cuda or cpu)")

    parser.add_argument("--no_sample", action='store_true', help="Set to use greedy decoding instead of sampling")
    parser.add_argument("--max_length", type=int, default=50, help="Maximum length of the output utterances")
    parser.add_argument("--min_length", type=int, default=1, help="Minimum length of the output utterances")
    parser.add_argument("--seed", type=int, default=0, help="Seed")
    parser.add_argument("--temperature", type=int, default=0.6, help="Sampling softmax temperature")
    parser.add_argument("--top_k", type=int, default=30, help="Filter top-k tokens before sampling (<=0: no filtering)")
    parser.add_argument("--top_p", type=float, default=0.9,
                        help="Nucleus filtering (top-p) before sampling (<=0.0: no filtering)")
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

    # Load KoGPT2 model and tokenizer
    tok_path = get_tokenizer()
    model, vocab = get_pytorch_conkogpt2_model(args.model_checkpoint)
    tokenizer = SentencepieceTokenizer(tok_path)
    model.to(args.device)

    logger.info("Sample a personality")
    topicList, sourceList, targetList = get_dataset(tokenizer, vocab, args.dataset_path)

    f1 = open((args.model_checkpoint + "_output.txt"), 'w')
    for line in zip(topicList, sourceList, targetList):
        out_ids = sample_sequence(line[0], line[1], vocab, tokenizer, model, args)
        out_texts = vocab.to_tokens(out_ids.squeeze(0).tolist())
        for text in out_texts:
            f1.write(text.replace('▁', ' ').replace('</s>',' '))
        f1.write("\n")
    f1.close()



if __name__ == "__main__":
    run()