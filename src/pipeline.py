#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 11 09:09:08 2021

@author: jeremylhour
"""
import os
import shutil
from datetime import datetime
import time
import yaml
from tqdm import tqdm
import re
import uuid

from transformers import AutoTokenizer, AutoModelForSeq2SeqLM


# ---------------------------------------------------------------------
# MAIN CLASS
# ---------------------------------------------------------------------
class Translator:
    def __init__(self, src, tgt, chunk_size_limit = 100):
        """
        init the translator for French to English translation
            for more language see : https://huggingface.co/Helsinki-NLP

        @param src (str): source language
        @param tgt (str): target language
        @param chunk_size_limit : if line is larger than this limit, breaks it down by sentances.
        """
        self.chunk_size_limit = chunk_size_limit
        self.src, self.tgt = src, tgt
        model = "Helsinki-NLP/opus-mt-{src}-{tgt}".format(src=self.src, tgt=self.tgt)

        # Initialize the tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model)
        # Initialize the model
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model)
        
        # out of text mode, ignore translation when in out of text mode
        self.oot_begin = [
            "\begin{equation}",
            "\begin{equation*}",
            "\begin{align}",
            "\begin{align*}",
            "\begin{figure}",
            "\begin{figure}[ht!]"
            "\begin{table}"
        ]
        self.oot_end = [
            "\end{equation}",
            "\end{equation*}",
            "\end{align}",
            "\end{align*}",
            "\end{figure}",
            "\end{table}"
        ]
        self.out_of_text_mode = False

        # Dictionnary for substitutions
        self.subs = {
            "\oe{}": "oe"
        }
        
        # List for hashing
        self.hash_table = {}
        self.hash_expr = [
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
            re.compile(r'\$.*?\$', re.DOTALL),
            re.compile(r'\\begin{.*?}', re.DOTALL),
            re.compile(r'\\end{.*?}', re.DOTALL),
            re.compile(r'\\textit{i\.e\.}', re.DOTALL),
            re.compile(r'\\textit{e\.g\.}', re.DOTALL),
            re.compile(r'\\label{.*?}', re.DOTALL),
            re.compile(r'\\url{.*?}', re.DOTALL),
            re.compile(r'\\cite{.*?}', re.DOTALL),
            re.compile(r'\\citeauthor{.*?}', re.DOTALL),
            re.compile(r'\\citeyear{.*?}', re.DOTALL),
            re.compile(r'\\ref{.*?}', re.DOTALL),
            re.compile(r'\\eqref{.*?}', re.DOTALL)
        ]
        
    def oot_switch(self, line):
        """
        oot_switch :
            turns out_of_text_mode on or off

        @param line (str):
        """
        if any([item in line for item in self.oot_begin]):
            self.out_of_text_mode = True
        if any([item in line for item in self.oot_end]):
            self.out_of_text_mode = False
        return None

    def replace_char(self, line):
        for item in self.subs:
            line = line.replace(item, self.subs[item])
        return line

    def break_line(self, line):
        """
        break_line :
            break the line if it is too long

        @param line (str): string to break
        """
        line = line.strip()
        if len(line) > self.chunk_size_limit:
            return [item for item in line.split('. ') if item]
        else:
            return [line]
        
    def hash_replace(self, match):
        """
        hash_replace :
            replace the given regex by a hash,
            enforces that the hash is translation invariant
            
        @param match : regular expression
        """
        if match.group() in list(self.hash_table.values()):
            hash = list(self.hash_table.keys())[list(self.hash_table.values()).index(match.group())]
        else:
            while True: 
                hash = uuid.uuid1().hex.upper()[:10]
                hash = removeConsecutiveDuplicates(hash)
                if self.translate(hash) == hash:
                    break
            self.hash_table[hash] = match.group()
        return hash
    
    def encode(self, text):
        """
        encode :
            replace the expression by the hash
            
        @param text (str):
        """
        for expression in self.hash_expr:
            text = expression.sub(self.hash_replace, text)
        return text
    
    def decode(self, text):
        """
        decode :
            decode the string after translation
            
        @param text (str):
        """
        for key, value in self.hash_table.items():
            text = text.replace(key, value)
        return text
        
    def translate(self, text):
        """
        translate :
            translate the chunk of given text
            
        @param text (str): str to translate
        """
        tokenized_text = self.tokenizer([text], return_tensors="pt")
        translation = self.model.generate(**tokenized_text)
        return self.tokenizer.batch_decode(translation, skip_special_tokens=True)[0]

    def process(self, text):
        """
        process :
            main method for processing the text
        """
        # 1. Encode the whole text
        print("    Encoding text...")
        start_time = time.time()
        encoded_text = self.replace_char(text) # replace chars
        encoded_text = self.encode(encoded_text) # encode regex
        print(f"    Encoding time : {(time.time() - start_time):.2f} seconds ---")
        
        # 2. Break into lines
        lines = [item for item in encoded_text.split('\n') if item]
        n = len(lines)
        
        # 3. Process each line
        empty_line = re.compile(r'^\s*$', re.DOTALL)
        translation = []
        
        with tqdm(total=n) as prog:
            for line in lines:
                if empty_line.search(line):
                    translation.append("")
                else:
                    broken_line = self.break_line(line) # break if too long
                    translated_line = [self.translate(item) for item in broken_line] # translate
                    translated_line = [self.decode(item) for item in translated_line] # decode regex
                    translation.append(' '.join(translated_line))
                prog.update(1)
        return translation
            
def removeConsecutiveDuplicates(s):
    if len(s)<2:
        return s
    if s[0]!=s[1]:
        return s[0]+removeConsecutiveDuplicates(s[1:])
    return removeConsecutiveDuplicates(s[1:])


if __name__=='__main__':
    print("\nThis script translates latex files from French to English.\n")
    now = datetime.now()
    print(f"Launched on {now.strftime('%d, %b %Y, %H:%M:%S')} \n")

    
    print("="*80)
    print("LOADING THE CONFIG, INIT TRANSLATOR")
    print("="*80)

    CONFIG_FILE = "config/configuration.yaml"
    with open(CONFIG_FILE, 'r') as stream:
        config = yaml.safe_load(stream)

    IN_DIR = config.get('DIR').get('IN')
    OUT_DIR = config.get('DIR').get('OUT')
    SRC = config.get('LANGUAGES').get('SRC')
    TGT = config.get('LANGUAGES').get('TGT')
    
    traducteur = Translator(src=SRC, tgt=TGT)

    
    print("="*80)
    print("TRANSLATE FILES")
    print("="*80)
    
    tex_files = [file for file in os.listdir(IN_DIR) if file.endswith(".tex")]
    print(str(len(tex_files))+" files to be translated : "+', '.join(tex_files))
    
    for file in tex_files:
        if os.path.exists(OUT_DIR+file):
            os.remove(OUT_DIR+file)
        
        print("\nCURRENTLY PROCESSING : {}".format(file))
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
        
        print(f"Time elapsed : {(time.time() - start_time):.2f} seconds ---")
        
        
    print("="*80)
    print("ZIPPING FILES")
    print("="*80)
    
    shutil.make_archive("translated", 'zip', OUT_DIR)