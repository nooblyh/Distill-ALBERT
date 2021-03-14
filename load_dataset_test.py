from logging import Logger
import random
from datasets import load_dataset
import unicodedata


datasets = load_dataset('./openwebtext.py', cache_dir='cache')
train_dataset = datasets["train"]

# Log a few random samples from the training set:
for index in random.sample(range(len(train_dataset)), 50):
    Logger.info(f"Sample {index} of the training set: {train_dataset[index]}.")

def clean_text(text):
    output = []
    for char in text:
        cp = ord(char)
        if cp == 0 or cp == 0xfffd or is_control(char):
            continue
        elif ((cp >= 33 and cp <= 47) or (cp >= 58 and cp <= 64) or (cp >= 91 and cp <= 96) or (cp >= 123 and cp <= 126)):
            output.append(" ")
        elif is_whitespace(char):
            output.append(" ")
        else:
            output.append(char)
    return "".join(output)

def is_control(char):
    # These are technically control characters but we count them as whitespace
    # characters.
    if char == "\t" or char == "\n" or char == "\r":
        return False
    cat = unicodedata.category(char)
    if cat in ("Cc", "Cf"):
        return True
    return False

def is_whitespace(char):
    # \t, \n, and \r are technically contorl characters but we treat them
    # as whitespace since they are generally considered as such.
    if char == " " or char == "\t" or char == "\n" or char == "\r":
        return True
    cat = unicodedata.category(char)
    if cat == "Zs":
        return True
    return False
