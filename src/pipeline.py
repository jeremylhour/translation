#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 11 09:09:08 2021
@author: jeremylhour
"""
import os
from datetime import datetime
import time

from transformers import AutoTokenizer, AutoModelForSeq2SeqLM


class Translator:
    def __init__(self, chunk_size_limit = 100):
        """
        init the translator for French to English translation

        @param chunk_size_limit : if line is larger than this limit, breaks it down by sentances.
        """
        self.chunk_size_limit = chunk_size_limit

        # Initialize the tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained("Helsinki-NLP/opus-mt-fr-en")
        # Initialize the model
        self.model = AutoModelForSeq2SeqLM.from_pretrained("Helsinki-NLP/opus-mt-fr-en")
        
        # math mode, ignore translation when in math mode
        self.math_begin = [
            "\begin{equation}",
            "\begin{equation*}",
            "\begin{align}",
            "\begin{align*}"
        ]
        self.math_end = [
            "\end{equation}",
            "\end{equation*}",
            "\end{align}",
            "\end{align*}"
        ]
        self.math_mode = False

        # Dictionnary for substitutions
        self.subs = {
            "\oe{}": "oe"
        }
    
    def math_switch(self, line):
        """
        match_switch :
            turns math_mode on or off

        @param line (str):
        """
        if any([item in line for item in self.math_begin]):
            self.math_mode = True
        if any([item in line for item in self.math_end]):
            self.math_mode = False
        return None

    def replace_char(self, line):
        for item in self.subs:
            line = line.replace(item, self.subs[item])
        return line

    def break_line(self, line):
        """
        break_line :
            break the line if it's too long

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
            translate the chunk of text given
            
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
            self.math_switch(text)
            if self.math_mode:
                return text.strip()
            else:
                text = self.replace_char(text)
                text = self.break_line(text)
                translation = []
                for item in text:
                    translation.append(self.translate(item))
                return ' '.join(translation)



if __name__=='__main__':
    print("\nThis script translates latex files from French to English.")
    now = datetime.now()
    print(f"Launched on {now.strftime('%d, %b %Y, %H:%M:%S')} \n")

    print("="*80)
    print("LOADING THE CONFIG, INIT TRANSLATOR")
    print("="*80)

    IN_DIR = 'data/raw/'
    OUT_DIR = 'data/translated/'
    FILE = '01_Rappels_ML.tex'

    os.remove(OUT_DIR+FILE)

    traducteur = Translator()

    print("="*80)
    print("PROCESS DATA")
    print("="*80)

    start_time = time.time()
    with open(IN_DIR+FILE, 'r') as in_file, open(OUT_DIR+FILE, 'a') as out_file:
        for line in in_file.readlines():
            translated_text = traducteur.process(line)
            out_file.write(translated_text+"\n")
    print(f"Time elapsed : {(time.time() - start_time):.2f} seconds ---")