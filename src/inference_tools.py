#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
inference_tools:
    Gather the HuggingFace Transformers inference tools, e.g. util functions etc.

Created on Sat Mar 11 15:04:22 2023

@author: jeremylhour
"""
import torch
from nltk.tokenize import sent_tokenize

def translate(batch, tokenizer, model, text='text'):
    """
    translate
    
    Args:
        batch ():
        tokenizer (Transformers.tokenizer):
        model (Transformers.model):
        
    Returns:
        The new dataset with a 'translation' column that have the translated text.
    """
    tokenized_batch = tokenizer(
        batch[text],
        padding="longest",
        truncation=True,
        max_length=512,
        return_tensors="pt"
    )
    with torch.no_grad():
        translation = model.generate(**tokenized_batch)
    return {
        'translation': tokenizer.batch_decode(translation, skip_special_tokens=True)
    }

def break_sentence(batch, text='text'):
    """
    break_sentences:
        Breaks a paragraph into sentences, while keeping the original paragraph index.
        
    Args:
        batch (): Must contain keys 'id' and whatever text is.
        text (str): Name of the textual column to break
        
    Returns:
        The new dataset with the broken down sentences and the appropriate paragraph id and position.
        
    Note:
        If 'elements' is empty, nothing is returned for this observation.
    """
    sentences, text_id, pos_id = [], [], []
    for i, item in zip(batch['id'], batch[text]):
        elements = sent_tokenize(item)
        sentences += elements
        text_id += [i]*len(elements)
        pos_id += range(len(elements))
        
    return {
        'sentence': sentences,
        'id': text_id,
        'pos_id': pos_id
        }

def compute_length(batch, text='text'):
    """
    compute_length:
        Computes the length of a text (nb. of characters).
        
    Args:
        batch ():
        text (str): Which key gives the text.
    
    Returns:
        The new dataset with the length of the string of text.
    """
    return {
        'length': [len(item) for item in batch[text]]
    }