import os
import jsonlines
#process code
import ast
import sys
from nltk import word_tokenize
import pickle
from tqdm import tqdm
import re
import parso


def file_analysis(old_file_lines, six_quotes, hashtap):
    """标记需要删除的注释的行号，并存入列表"""
    i = 0
    for line in old_file_lines:
        # 符号 # 独占一行
        ret_1 = re.match(r"^[^\w]*#+", line)
        if ret_1:
            hashtap.append(i)
        # 符号 """ 独占一行
        ret_2 = "\"\"\"" in line
        if ret_2:
            # 如果存在类型，函数说明的 """xxxxx""" 之类的，不予删除
            #ret_2_1 = re.match(r".*\"\"\".*\"\"\".*", line)
            instr = re.match(r"\'.*\"\"\".*\'.*", line)
            
            if not instr:
                six_quotes.append(i)
            #six_quotes.append(i)
        i += 1
    # 将两个"""行号之间所有的行添加到 # 号列表中
    #print(old_file_lines)
    #print(six_quotes)
    while six_quotes != []:
        # 从列表中移出最后两个元素
        a = six_quotes.pop()
        b = six_quotes.pop()
        temp = b
        while temp <= a:
            hashtap.append(temp)
            temp += 1
    # 返回 # 号列表， 记返回需要删除的所有注释的 行号 集合
    return hashtap

ff = open("error.txt", "w")
def main(old_file):
    """ 主函数"""
    # 1，获取路径，并读取此文件
    # 1.1 获取文件名及其路径
    #old_file = old_file.decode("utf-8")
    old_file_lines = old_file.splitlines()
        # 2.2 处理文件并得到需要删除的注释的行号集合
    six_quotes, hashtap = list(), list()
    try:
        hashtap = file_analysis(old_file_lines, six_quotes, hashtap)
    except:
        ff.write("\n".join(old_file_lines))
        ff.flush()
        return None
        # 此时返回值 hashtap列表中，不仅仅包含#，还有"""的行号
    try:
        # 3，获取 注释和无注释 内容到列表中
        # 3.2 去重并排序，得到所有注释行号的列表
        comment_list = sorted(set(hashtap))
        # 3.3 创建存储(备份)注释文件内容的列表
        comment_file = list()
        for i in comment_list:
            comment = old_file_lines[i]
            comment_file.append(comment)
            # 创建与源文件总行号相同的列表 0,1,2,3...
        new_file_list = list(i for i in range(len(old_file_lines)))
            # 删除注释的行号，留下无注释的行号 的列表集合
        for i in comment_list:
            new_file_list.remove(i)
            # 3.4 创建存储（无注释）新文件内容的列表
        new_file_lines = list()
        for i in new_file_list:
            temp = old_file_lines[i]
            if temp == "\n" or temp == "":
                continue
            new_file_lines.append(temp)

    except:
        print("待处理代码中没有注释")
    else:
        return new_file_lines
wf = open("test.txt", "w")
maxlen = 0
maxlen1 = 0
x1 = 0
x2 = 0
sum1 = 0
suml = 0
invalidnum = 0
def parseF(s):
    try:
        try:
            parsed_module = parso.parse(s, error_recovery=False, version="2.7")
        except parso.parser.ParserSyntaxError:
            parsed_module = parso.parse(s, error_recovery=False, version="3.7")
        function_defs = list(parsed_module.iter_funcdefs())
    except parso.parser.ParserSyntaxError:
        for function_def in function_defs:
            function_code = function_def.get_code()
            print(function_code)
        return function_code
        print(s)
        exit(1)
            
def turns(s):
    ans = ""
    isCode = -1
    for i, x in enumerate(s):
        if x != " ":
            isCode = i
            break
    num = 0
    tmps = s[:isCode]
    if "\t" in tmps:
        num = len(tmps)
    else:
        num = int(len(tmps) / 4)
    ans = "<tab> " * num
    for i in range(isCode, len(s)):
        ans += s[i]

    return ans
for i in range(0, 1):
    f = "python_dedupe_definitions_v2.pkl"#"python_valid_" + str(i) + ".jsonl"
    
    fs = open(f, "rb")
    dic = pickle.load(fs)
    #x = 0
    #sum1 = 0
    #suml = 0
    for item in tqdm(dic):
        #print(item)
        '''wf.write("\t".join(item["code_tokens"]).replace("\n", "").replace("\r", ""))
        maxlen = max(maxlen, len(item["code_tokens"]))
        suml += len(item["code_tokens"])
        wf.write("\n")'''
        #print(item["function"])
        newfilelines = parseF(item["function"])#main(item["function"])
        if newfilelines:
            #wf.write("\t".join(item["docstring_tokens"]).replace("\n", "").replace("\r", ""))
            #wf.write("\n")
            #maxlen1 = max(maxlen1, len(item["docstring_tokens"]))
            #sum1 += len(item["docstring_tokens"])
            for xs in newfilelines:
                xs = turns(xs)
                #maxlen = max(maxlen, len(word_tokenize(xs)))
                #suml += len(word_tokenize(xs))
                x2 += 1
                wf.write(xs)
                wf.write("\n")
            wf.write("<linemask>\n")
            wf.flush()
            x1 += 1
        else:
            print("gg")
print(maxlen1, maxlen, sum1 / x1, suml / x2)
wf.close()
