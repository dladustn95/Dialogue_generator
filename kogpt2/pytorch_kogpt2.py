# coding=utf-8
# Copyright 2020 SKT AIX Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import hashlib
import os
import sys

import gluonnlp as nlp
import requests
import torch

from .model.torch_gpt2 import GPT2Config, GPT2LMHeadModel, GPT2LMHeadModelEncoder
from transformers import OpenAIGPTLMHeadModel, OpenAIGPTConfig
from .utils import download as _download
from .utils import tokenizer

pytorch_kogpt2 = {
    'url':
    'https://kobert.blob.core.windows.net/models/kogpt2/pytorch/pytorch_kogpt2_676e9bcfa7.params',
    'fname': 'pytorch_kogpt2_676e9bcfa7.params',
    'chksum': '676e9bcfa7'
}

kogpt2_config = {
    "initializer_range": 0.02,
    "layer_norm_epsilon": 1e-05,
    "n_ctx": 1024,
    "n_embd": 768,
    "n_head": 12,
    "n_layer": 12,
    "n_positions": 1024,
    "vocab_size": 50000,
    "output_past": True
}

kogpt2_encoder_config = {
    "initializer_range": 0.02,
    "layer_norm_epsilon": 1e-05,
    "output_hidden_states": True,
    "n_ctx": 1024,
    "n_embd": 768,
    "n_head": 12,
    "n_layer": 12,
    "n_positions": 1024,
    "vocab_size": 50000,
    "use_adapter": False,
    "output_past": True
}

engpt2_config = {
  "initializer_range": 0.02,
  "layer_norm_epsilon": 1e-05,
  "n_ctx": 1024,
  "n_embd": 768,
  "n_head": 12,
  "n_layer": 12,
  "n_positions": 1024,
  "vocab_size": 50257,
  "output_past": True
}

engpt_config = {
  "initializer_range": 0.02,
  "layer_norm_epsilon": 1e-05,
  "n_ctx": 512,
  "n_embd": 768,
  "n_head": 12,
  "n_layer": 12,
  "n_positions": 512,
  "vocab_size": 40478,
  "output_past": True
}

def get_pytorch_conkogpt2_model2(keyword_Module="", use_adapter=True, cachedir='~/kogpt2/'):
    # download model
    # download vocab
    vocab_info = tokenizer
    vocab_path = _download(vocab_info['url'],
                           vocab_info['fname'],
                           vocab_info['chksum'],
                           cachedir=cachedir)

    config = kogpt2_config
    config["use_adapter"] = use_adapter
    config["keyword_Module"] = keyword_Module
    kogpt2model = GPT2LMHeadModel(config=GPT2Config.from_dict(config))
    vocab_b_obj = nlp.vocab.BERTVocab.from_sentencepiece(vocab_path,
                                                         mask_token=None,
                                                         sep_token='<sep>',
                                                         cls_token='<cls>',
                                                         unknown_token='<unk>',
                                                         padding_token='<pad>',
                                                         bos_token='<s>',
                                                         eos_token='</s>')
    return kogpt2model, vocab_b_obj

def get_pytorch_conengpt2_model(keyword_Module="", use_adapter=True, model_class="gpt"):
    if model_class == "gpt2":
        config = engpt2_config
    else:
        config = engpt_config
    config["use_adapter"] = use_adapter
    config["keyword_Module"] = keyword_Module

    if model_class == "gpt2":
        gpt2model = GPT2LMHeadModel(config=GPT2Config.from_dict(config))
    else:
        gpt2model = OpenAIGPTLMHeadModel(config=OpenAIGPTConfig.from_dict(config))
    return gpt2model

def get_pytorch_kogpt2_encoder(cachedir='~/kogpt2/'):
    # download model
    # download vocab
    model_info = pytorch_kogpt2
    model_path = _download(model_info['url'],
                           model_info['fname'],
                           model_info['chksum'],
                           cachedir=cachedir)
    vocab_info = tokenizer
    vocab_path = _download(vocab_info['url'],
                           vocab_info['fname'],
                           vocab_info['chksum'],
                           cachedir=cachedir)

    kogpt2model = GPT2LMHeadModelEncoder(config=GPT2Config.from_dict(kogpt2_encoder_config))
    kogpt2model.load_state_dict(torch.load(model_path), strict=False)
    vocab_b_obj = nlp.vocab.BERTVocab.from_sentencepiece(vocab_path,
                                                         mask_token=None,
                                                         sep_token='<sep>',
                                                         cls_token='<cls>',
                                                         unknown_token='<unk>',
                                                         padding_token='<pad>',
                                                         bos_token='<s>',
                                                         eos_token='</s>')
    return kogpt2model, vocab_b_obj

def get_pytorch_kogpt2_model(keyword_Module="", use_adapter=True, cachedir='~/kogpt2/'):
    # download model
    model_info = pytorch_kogpt2
    model_path = _download(model_info['url'],
                           model_info['fname'],
                           model_info['chksum'],
                           cachedir=cachedir)
    # download vocab
    vocab_info = tokenizer
    vocab_path = _download(vocab_info['url'],
                           vocab_info['fname'],
                           vocab_info['chksum'],
                           cachedir=cachedir)
    return get_kogpt2_model(model_path, vocab_path, keyword_Module, use_adapter)

def get_pytorch_engpt2_model(keyword_Module="", use_adapter=True, model_class="gpt2", cachedir='~/kogpt2/'):
    if model_class == "gpt2":
        config = engpt2_config
    else:
        config = engpt_config
    config["use_adapter"] = use_adapter
    config["keyword_Module"] = keyword_Module

    if model_class == "gpt2":
        gpt2model = GPT2LMHeadModel(config=GPT2Config.from_dict(config))
    else:
        gpt2model = OpenAIGPTLMHeadModel(config=OpenAIGPTConfig.from_dict(config))
    gpt2model.load_state_dict(torch.load("./transformer_model/" + model_class + ".pth"), strict=False)
    return gpt2model


def get_kogpt2_model(model_file, vocab_file, keyword_Module="", use_adapter=True):

    config = kogpt2_config
    config["use_adapter"] = use_adapter
    config["keyword_Module"] = keyword_Module

    kogpt2model = GPT2LMHeadModel(config=GPT2Config.from_dict(config))
    kogpt2model.load_state_dict(torch.load(model_file), strict=False)
    vocab_b_obj = nlp.vocab.BERTVocab.from_sentencepiece(vocab_file,
                                                         mask_token=None,
                                                         sep_token=None,
                                                         cls_token=None,
                                                         unknown_token='<unk>',
                                                         padding_token='<pad>',
                                                         bos_token='<s>',
                                                         eos_token='</s>')
    return kogpt2model, vocab_b_obj
