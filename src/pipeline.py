#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
pipeline

Created on Thu Nov 11 09:09:08 2021

@author: jeremylhour
"""
import os
import shutil
import pandas as pd
from datetime import datetime, timedelta
import time
import yaml
import re
import uuid
import logging

from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from datasets import Dataset

from inference_tools import break_sentence, compute_length, translate

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler()
    ]
)

# ---------------------------------------------------------------------
# CONSTANTS AND SMALLER FUNCTIONS
# ---------------------------------------------------------------------     
def removeConsecutiveDuplicates(s):
    if len(s)<2:
        return s
    if s[0]!=s[1]:
        return s[0]+removeConsecutiveDuplicates(s[1:])
    return removeConsecutiveDuplicates(s[1:])

HASH_EXPR = [
    re.compile(r'\\newpage', re.DOTALL),
    re.compile(r'.*?\\begin{document}', re.DOTALL),
    re.compile(r'\\begin{figure}.*?\\end{figure}', re.DOTALL),
    re.compile(r'\\begin{equation}.*?\\end{equation}', re.DOTALL),
    re.compile(r'\\begin{equation\*}.*?\\end{equation\*}', re.DOTALL),
    re.compile(r'\\begin{align}.*?\\end{align}', re.DOTALL),
    re.compile(r'\\begin{align\*}.*?\\end{align\*}', re.DOTALL),
    re.compile(r'\\begin{table}.*?\\end{table}', re.DOTALL),
    re.compile(r'\\begin{tabular}.*?\\end{tabular}', re.DOTALL),
    re.compile(r'\\begin{float}.*?\\end{float}', re.DOTALL),
    re.compile(r'\\begin{tikz}.*?\\end{tikz}', re.DOTALL),
    re.compile(r'\\\$', re.DOTALL),
    re.compile(r'\$+.*?\$+', re.DOTALL),
    re.compile(r'\\begin{.*?}', re.DOTALL),
    re.compile(r'\\end{.*?}', re.DOTALL),
    re.compile(r'\\texttt{.*?}', re.DOTALL),
    re.compile(r'\\textbf{.*?}', re.DOTALL),
    re.compile(r'\\paragraph{.*?}', re.DOTALL),
    re.compile(r'\\q{.*?}', re.DOTALL),
    re.compile(r'\\emph{.*?}', re.DOTALL),
    re.compile(r'\\textit{.*?}', re.DOTALL),
    re.compile(r'\\label{.*?}', re.DOTALL),
    re.compile(r'\\url{.*?}', re.DOTALL),
    re.compile(r'\\cite\[*?.*?\]*?{.*?}', re.DOTALL),
    re.compile(r'\\citeauthor\[*?.*?\]*?{.*?}', re.DOTALL),
    re.compile(r'\\citeyear\[*?.*?\]*?{.*?}', re.DOTALL),
    re.compile(r'\\ref{.*?}', re.DOTALL),
    re.compile(r'\\eqref{.*?}', re.DOTALL),
    re.compile(r'\\\%', re.DOTALL),
    re.compile(r'\\item', re.DOTALL)
    ]

CACHE_DEFAULT = "~/.cache/huggingface/hub"

# ---------------------------------------------------------------------
# MAIN CLASS
# ---------------------------------------------------------------------
class Translator:
    def __init__(self, src, tgt, cache_dir: str = CACHE_DEFAULT, batch_size: int = 10):
        """
        init the translator for French to English translation
            for more language see : https://huggingface.co/Helsinki-NLP

        Args:
            src (str): source language.
            tgt (str): target language.
            cache_dir= (str): The folder where models are cached.
            batch_size (int): batch_size for Transformers inference.
        """
        self.batch_size = batch_size
        self.src, self.tgt = src, tgt
        model = "Helsinki-NLP/opus-mt-{src}-{tgt}".format(src=self.src, tgt=self.tgt)

        # Initialize the tokenizer and model
        self.tokenizer = AutoTokenizer.from_pretrained(model, use_fast=True)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model, cache_dir=cache_dir)

        # Dictionnary for substitutions
        self.subs = {
            "\oe{}": "oe"
        }

        # List for hashing
        self.hash_table = {}
        self.hash_expr = HASH_EXPR

    def replace_char(self, string):
        """
        replace_char:
            replace the specified char in the given string
        
        Args:
            string (str):
        """
        for item in self.subs:
            string = string.replace(item, self.subs[item])
        return string
        
    def hash_replace(self, match):
        """
        hash_replace:
            replace the given regex by a hash,
            enforces that the hash is translation invariant and unique
        
        Args:
            match: regular expression
        """
        if match.group() in list(self.hash_table.values()):
            hash = list(self.hash_table.keys())[list(self.hash_table.values()).index(match.group())]
        else:
            while True: 
                hash = uuid.uuid1().hex.upper()[:10]
                if (self.dummy_translate(hash) == hash) and (hash not in self.hash_table.keys()):
                    break
            self.hash_table[hash] = match.group()
        return hash
    
    def encode(self, text):
        """
        encode:
            replace the expression by the hash
            
        Args:
            text (str):
        """
        for expression in self.hash_expr:
            text = expression.sub(self.hash_replace, text)
        return text
    
    def decode(self, text):
        """
        decode:
            decode the string from the specified hash_dict
        
        Args:
            text (str): text to decode
        """
        for key, value in self.hash_table.items():
            text = text.replace(key, value)
        return text

    def dummy_translate(self, text):
        """
        dummy_translate :
            translate the chunk of given text. Only used to make sure hashed lines will be preserved.
        
        Args:
            text (str): str to translate
        """
        tokenized_text = self.tokenizer([text], return_tensors="pt")
        translation = self.model.generate(**tokenized_text)
        return self.tokenizer.batch_decode(translation, skip_special_tokens=True)[0]

    def translate_document(self, lines):
        """
        translate_document:
            Translates the document using the HuggingFace technology (datasets and transformers).
            Allows for 2x speed increase

        Args:
            lines (list of str): The lines to translate.

        Returns:
            The translated lines.
        """
        # Create a HuggingFace Dataset object
        ds = Dataset.from_dict({
            'id': range(len(lines)),
            'text': lines
        })

        # Break into sentences, sort by length (for speed-up), run inference
        ds = ds.map(break_sentence, batched=True, batch_size=self.batch_size, remove_columns=ds.column_names)
        ds = ds.map(lambda x: compute_length(x, text='sentence'), batched=True).sort('length', reverse=True)
        ds = ds.map(lambda x: translate(x, self.tokenizer, self.model, text='sentence'), batched=True, batch_size=self.batch_size).sort('id')

        # Add back the empty lines
        ds.set_format("pandas")
        df = ds[:].sort_values(['id', 'pos_id']).copy()

        missing_ids = [i for i in range(df.id.max()) if i not in df.id.unique()]
        missing_df = pd.DataFrame.from_dict({
            'id': missing_ids,
            'sentence': [' ']*len(missing_ids),
            'pos_id': [0]*len(missing_ids),
            'length': [0]*len(missing_ids),
            'translation': [' ']*len(missing_ids)
        })

        # Finally reform the original lines, including the empty ones, for 1-to-1 mapping with original document
        translated_lines = (
            pd.concat([df, missing_df], axis=0)
            .sort_values(['id', 'pos_id'])
            .groupby('id')['translation']
            .apply(lambda x: ' '.join(x))
            .to_list()
        )
        return translated_lines

    def process(self, text):
        """
        process:
            main method for processing the text
        
        Args:
            text (str): the whole text to encore and translate.
            Lines are separated by line breaks.
        """
        # 1. Encode the whole text
        logging.info("Encoding text.")
        start_time = time.time()
        encoded_text = self.replace_char(text) # replace chars
        encoded_text = self.encode(encoded_text) # encode specified regex
        logging.info(f"Encoding time: {timedelta(seconds=int(time.time() - start_time))}.")
        
        # 2. Break into lines
        lines = [item for item in encoded_text.split('\n') if item]
        
        # 3. Process the whole document at once
        logging.info("Translating lines.")
        translated_lines = self.translate_document(lines)

        # 4. Decode the lines
        logging.info("Decoding lines.")
        translated_lines = [self.decode(item) for item in translated_lines]
        return translated_lines

if __name__=='__main__':
    now = datetime.now()
    logging.info(f"\nThis script translates latex files while preserving equations and special expressions.\nLaunched on {now.strftime('%d, %b %Y, %H:%M:%S')} \n")

    logging.info("Loading config.")
    CONFIG_FILE = "config/configuration.yaml"
    with open(CONFIG_FILE, 'r') as stream:
        config = yaml.safe_load(stream)

    IN_DIR = config.get('DIR').get('IN')
    OUT_DIR = config.get('DIR').get('OUT')
    SRC = config.get('LANGUAGES').get('SRC')
    TGT = config.get('LANGUAGES').get('TGT')
    CACHE = config.get('MODEL').get('CACHE')
    
    traducteur = Translator(
        src=SRC,
        tgt=TGT,
        cache_dir=CACHE
        )

    logging.info("Loading files.")
    
    tex_files = sorted([
        file for file in os.listdir(IN_DIR) if file.endswith(".tex")
        ])
    logging.info(f"{len(tex_files)} files to be translated: {', '.join(tex_files)}")
    
    for file in tex_files:
        if os.path.exists(OUT_DIR+file):
            os.remove(OUT_DIR+file)
        
        logging.info("Processing : {}".format(file))
        start_time = time.time()
        
        # 1. Read the whole text
        with open(IN_DIR+file, 'r') as in_file:
            whole_text = ' '.join([line for line in in_file.readlines()])
        
        # 2. Process it
        translated_text = traducteur.process(whole_text)
        
        # 3. Write to file
        with open(OUT_DIR+file, 'a') as out_file:
            for line in translated_text:
                out_file.write(line+"\n")
        
        logging.info(f"Time elapsed : {timedelta(seconds=int(time.time() - start_time))}")

    logging.info("Zipping files.")
    shutil.make_archive("translated", 'zip', OUT_DIR)