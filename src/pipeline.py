#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 11 09:09:08 2021
@author: jeremylhour
"""
import os
from datetime import datetime
import time
import yaml
from tqdm import tqdm

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
            "\begin{align*}"
        ]
        self.oot_end = [
            "\end{equation}",
            "\end{equation*}",
            "\end{align}",
            "\end{align*}"
        ]
        self.out_of_text_mode = False

        # Dictionnary for substitutions
        self.subs = {
            "\oe{}": "oe"
        }
    
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
            return [item for item in line.split('.') if item]
        else:
            return [line]

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
        if text == '\n':
            return ""
        else:
            self.oot_switch(text)
            if self.out_of_text_mode:
                return text.strip()
            else:
                text = self.replace_char(text)
                text = self.break_line(text)
                translation = [self.translate(item) for item in text]
                return ' '.join(translation)



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
        try:
            os.remove(OUT_DIR+file)
        except:
            pass

        with open(IN_DIR+file, 'r') as in_file:
            n = len(in_file.readlines())
        
        print("\nCurrently processing : {}".format(file))
        start_time = time.time()
        with open(IN_DIR+file, 'r') as in_file, open(OUT_DIR+file, 'a') as out_file:
            with tqdm(total=n) as prog:
                for line in in_file.readlines():
                    translated_text = traducteur.process(line)
                    out_file.write(translated_text+"\n")
                    prog.update(1)
        print(f"Time elapsed : {(time.time() - start_time):.2f} seconds ---")