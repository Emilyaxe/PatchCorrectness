import re
from tqdm import tqdm
def tokenize_for_bleu_eval(code):
    code = re.sub(r'([^A-Za-z0-9])', r' \1 ', code)
        #code = re.sub(r'([a-z])([A-Z])', r'\1 \2', code)
    code = re.sub(r'\s+', ' ', code)
    code = code.replace('"', '`')
    code = code.replace('\'', '`')
    tokens = [t for t in code.split(' ') if t]
    return tokens
f = open("dev.txt", "r")
wf = open("ndev.txt", "w")
lines = f.readlines()
Nls = []
Codes = []
isNl = True
for i in tqdm(range(len(lines))):
    if isNl:
        wf.write(lines[i].strip() + "\n")
        isNl = False
        continue
    if lines[i].strip() == "<linemask>":
        wf.write("\t".join(Codes) + "\n")
        Codes = []
        isNl = True
        continue
    code = lines[i].strip().lower()
    code_tokens = tokenize_for_bleu_eval(code)#word_tokenize(code)
    Codes.extend(code_tokens)