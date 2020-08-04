from datetime import datetime
import json
import logging
import os
import tarfile
import tempfile
import socket
import sys

import torch

from transformers import cached_path

PERSONACHAT_URL = "https://s3.amazonaws.com/datasets.huggingface.co/personachat/personachat_self_original.json"
HF_FINETUNED_MODEL = "https://s3.amazonaws.com/models.huggingface.co/transfer-learning-chatbot/gpt_personachat_cache.tar.gz"

logger = logging.getLogger(__file__)

def download_pretrained_model():
    """ Download and extract finetuned model from S3 """
    resolved_archive_file = cached_path(HF_FINETUNED_MODEL)
    tempdir = tempfile.mkdtemp()
    logger.info("extracting archive file {} to temp dir {}".format(resolved_archive_file, tempdir))
    with tarfile.open(resolved_archive_file, 'r:gz') as archive:
        archive.extractall(tempdir)
    return tempdir

def get_dataset_key(bert_tokenizer, gpt_tokenizer, gpt_vocab, dataset_path):
    def read(fn):
        f = open(fn, 'r', encoding="UTF-8-SIG")
        lines = []
        for line in f:
            lines.append(line.strip())

        f.close()

        return lines

    sourceList_train = []
    targetList_train = []
    attentionList_train = []
    sourceList_valid = []
    targetList_valid = []
    attentionList_valid = []

    lines = read(dataset_path + "_train.txt")
    for line in lines:
        tmp = line.split("|")
        source = bert_tokenizer.tokenize(tmp[0])
        sourceList_train.append(source)
        target = gpt_vocab[gpt_tokenizer(tmp[1])]
        targetList_train.append(target)

    lines = read(dataset_path + "_train_keyword.txt")
    for line in lines:
        tmp = []
        for i in line.split(" "):
            tmp.append(float(i))
        attentionList_train.append(tmp)

    validlines = read(dataset_path + "_valid.txt")
    for line in validlines:
        tmp = line.split("|")
        source = bert_tokenizer.tokenize(tmp[0])
        sourceList_valid.append(source)
        target = gpt_vocab[gpt_tokenizer(tmp[1])]
        targetList_valid.append(target)

    lines = read(dataset_path + "_valid_keyword.txt")
    for line in lines:
        tmp = []
        for i in line.split(" "):
            tmp.append(float(i))
        attentionList_valid.append(tmp)

    return sourceList_train, targetList_train, attentionList_train, sourceList_valid, targetList_valid, attentionList_valid

def get_test_dataset_key(bert_tokenizer, gpt_tokenizer, gpt_vocab, dataset_path):
    def read(fn):
        f = open(fn, 'r', encoding="UTF-8-SIG")
        lines = []
        for line in f:
            lines.append(line.strip())

        f.close()

        return lines

    sourceList = []
    targetList = []
    attentionList = []

    lines = read(dataset_path + "_test.txt")
    for line in lines:
        tmp = line.split("|")
        source = bert_tokenizer.tokenize(tmp[0])
        sourceList.append(source)
        target = gpt_vocab[gpt_tokenizer(tmp[1])]
        targetList.append(target)

    lines = read(dataset_path + "_test_keyword.txt")
    for line in lines:
        tmp = []
        for i in line.split(" "):
            tmp.append(float(i))
        attentionList.append(tmp)

    return sourceList, targetList, attentionList

def get_dataset(bert_tokenizer, gpt_tokenizer, gpt_vocab, dataset_path):
    def read(fn):
        f = open(fn, 'r', encoding="UTF-8-SIG")
        lines = []
        for line in f:
            lines.append(line.strip())

        f.close()

        return lines

    sourceList_train = []
    targetList_train = []
    sourceList_valid = []
    targetList_valid = []

    lines = read(dataset_path + "_train.txt")
    for line in lines:
        tmp = line.split("|")
        source = bert_tokenizer.tokenize(tmp[0])
        sourceList_train.append(source)
        target = gpt_vocab[gpt_tokenizer(tmp[1])]
        targetList_train.append(target)

    validlines = read(dataset_path + "_valid.txt")
    for line in validlines:
        tmp = line.split("|")
        source = bert_tokenizer.tokenize(tmp[0])
        sourceList_valid.append(source)
        target = gpt_vocab[gpt_tokenizer(tmp[1])]
        targetList_valid.append(target)

    return sourceList_train, targetList_train, sourceList_valid, targetList_valid

def get_test_dataset(bert_tokenizer, gpt_tokenizer, gpt_vocab, dataset_path):
    def read(fn):
        f = open(fn, 'r', encoding="UTF-8-SIG")
        lines = []
        for line in f:
            lines.append(line.strip())

        f.close()

        return lines

    sourceList = []
    targetList = []

    lines = read(dataset_path + "_test.txt")
    for line in lines:
        tmp = line.split("|")
        source = bert_tokenizer.tokenize(tmp[0])
        sourceList.append(source)
        target = gpt_vocab[gpt_tokenizer(tmp[1])]
        targetList.append(target)

    return sourceList, targetList

class AttrDict(dict):
    def __init__(self, *args, **kwargs):
        super(AttrDict, self).__init__(*args, **kwargs)
        self.__dict__ = self


def make_logdir(model_name: str, dataset_path: str, keyword_Module: str):
    """Create unique path to save results and checkpoints, e.g. runs/Sep22_19-45-59_gpu-7_gpt2"""
    # Code copied from ignite repo
    current_time = datetime.now().strftime('%b%d_%H-%M')
    data = dataset_path.split("/")[-1]
    logdir = os.path.join(
        'runs', data + '_' + current_time + '_' + model_name + '_' + keyword_Module)
    return logdir