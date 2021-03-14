import re
import unicodedata

filepath = "0450004-eef988a8300b327b4d4b8b8f3d327f03.txt"
with open(filepath, encoding="utf-8") as f:
    x = re.sub("\n+", "\n", f.read()).strip()
    y = re.sub("(https?|http|ftp|file):\/\/[-A-Za-z0-9+&@#/%?=~_|!:,.;]+[-A-Za-z0-9+&@#/%=~_|]", "", x)
    x = y.split('\n')

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

for i in x:
    # TODO: delete the blank line
    result = clean_text(i)
    result = re.sub(" +", " ", result).strip()
    print(result)

