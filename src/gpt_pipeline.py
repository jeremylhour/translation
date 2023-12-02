"""
gpt_pipeline:
    Runs a pipeline to call Open AI API on a bunch of texts.
    Requires to have an Open AI API key as environment variable under OPENAI_API_KEY,
    and of course, credits on the Open AI website.

    See https://platform.openai.com/docs/quickstart?context=python.

Created on Sat Nov 25 2023

@author: jeremylhour
"""
import os
import shutil
from datetime import datetime, timedelta
import time
import yaml
import logging
from tqdm import tqdm

from openai import OpenAI

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

# Max number of token from Open AI is 4097 between input and output,
# let's take 1200 for the input, to be sure.
# This number is used
NUM_TOKENS_LIMIT = 1_200

# Set a limit to send a request to OpenAI.
# There's no point in sending small sequences, it's probably just section titles etc.
MAX_INPUT_TOKENS = 30

def split_with_delimiter(x: str, delimiter: str = "\section", start: bool = True):
    """
    split_with_delimiter:
        Split on a delimiter, but keep it.
        The deliminter is assumed to be at the start of the chunk.

        Notice that this is loss-less, in the sence that x == ''.join(split_with_delimiter(x)).
    
    Args:
        x (str): The string to split.
        s (str): The delimiter.
        start (bool): Whether the delimiter should be at the start or at the end of each chunks.
    """
    chunks = x.split(delimiter)
    sections = []
    for i, s_ in enumerate(chunks):
        if start:
            if i > 0:
                s_ = delimiter + s_
        else:
            if i < len(chunks):
                s_ += delimiter
        sections.append(s_)
    return sections

def split_chapter(x: str):
    """
    split_chapter:
        split chapter into sections and subsections.
        
    Args:
        x (str): The string to split.
    """
    chunks = []
    sections = split_with_delimiter(x, delimiter="\section")
    for s_ in sections:
        subsections = split_with_delimiter(s_, delimiter="\subsection")
        chunks += subsections
    return chunks

def approx_num_tokens(x: str):
    """
    approx_num_tokens:
        Approximate the number of tokens in a string,
        using the rule nb_tokens = len(x) / 4.
        Likely valid for English, but less so for other languages.

    Args:
        x (str): The string.
    """
    return len(x) // 4

def split_long_str(x: str, limit: str = NUM_TOKENS_LIMIT):
    """
    split_long_str:
        Split long strings into chunks of max limit tokens (approximation).
        This function is loss-less.

    Args:
        x (str): The string.
        limit (int): Limit (in number of tokens).
    """
    chunks = split_with_delimiter(x, delimiter="\n", start=True)
    output = []
    current_line = chunks[0]
    for c in chunks[1:]:
        if approx_num_tokens(current_line) + approx_num_tokens(c) <= limit:
            # If adding the next line does not exceed the max length, append it
            current_line += c
        else:
            # If adding the next line exceeds the max length, start a new element
            output.append(current_line)
            current_line = c

    # Append the last element
    output.append(current_line)
    return output

# ---------------------------------------------------------------------
# MAIN CLASS
# ---------------------------------------------------------------------
class OpenAICaller:
    def __init__(self, base_prompt: str, system_prompt: str, model: str = "gpt-3.5-turbo"):
        self.model = model
        self.system_prompt = system_prompt
        self.base_prompt = base_prompt
        self.client = OpenAI()

    def __call__(self, prompt: str):
        combined_input = f"{self.base_prompt} '{prompt}'"
        
        # Make a request to the OpenAI API
        completion = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": combined_input}
            ])
        return completion.choices[0].message.content

if __name__ == '__main__':
    now = datetime.now()
    logging.info(f"This script translates latex files by calling the Open AI API with relevant prompts.\
                 \nLaunched on {now.strftime('%d, %b %Y, %H:%M:%S')} \n")

    logging.info("Loading config.")
    CONFIG_FILE = "config/configuration_gpt.yaml"
    with open(CONFIG_FILE, 'r') as stream:
        config = yaml.safe_load(stream)

    IN_DIR = config.get('DIR').get('IN')
    OUT_DIR = config.get('DIR').get('OUT')
    MODEL = config.get('MODEL')
    PROMPTS = config.get('PROMPTS')

    logging.info("Loading the Open AI translator.")
    translator = OpenAICaller(
        base_prompt=PROMPTS["USER"],
        system_prompt=PROMPTS["SYSTEM"],
        model=MODEL
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

        # 2. Split it into sections and subsections
        sections = split_chapter(whole_text)
        chunks = []
        for s in sections:
            chunks += split_long_str(s, limit=NUM_TOKENS_LIMIT)

        # 3. Call Open AI model
        translated = []
        for c in tqdm(chunks):
            if approx_num_tokens(c) < MAX_INPUT_TOKENS:
                translated.append(c)
            else:
                output = translator(prompt=c)
                translated.append(output)

        translated_text = ''.join(translated)

        # 4. Write to file
        with open(OUT_DIR+file, 'a') as out_file:
            out_file.write(translated_text) 

        logging.info(f"Time elapsed : {timedelta(seconds=int(time.time() - start_time))}")

    logging.info("Zipping files.")
    shutil.make_archive("translated", 'zip', OUT_DIR)