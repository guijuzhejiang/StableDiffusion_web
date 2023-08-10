# coding=utf-8
# @CREATE_TIME: 2021/8/31 下午1:57
# @LAST_MODIFIED: 2021/8/31 下午1:57
# @FILE: get_text_renderer_text.py
# @AUTHOR: Ray
import pandas as pd
from collections import defaultdict

input = '/home/ray/Workspace/project/ocr/src/gso_server/supply.txt'
output = '/home/ray/Workspace/project/ocr/src/gso_server/supply_text.txt'
output_content = []

with open(input, 'r') as f:
    src_list = [i.replace('\n', '') for i in f.readlines()]
    src_len = len(src_list)

df = pd.DataFrame(src_list)
# for char_num in range(3, 10+3):
char_num = 10
loop_range = 500
count_dict = defaultdict(int)

break_flag = False
while True:
    for index in range(len(src_list)):
        first_char = src_list[index % src_len]
        chars = [first_char]
        chars.extend(df.sample(n=char_num - 1, replace=False, random_state=None)[0].values.tolist())
        chars = list(set(chars))
        for char in chars:
            count_dict[char] += 1
            if count_dict[char] == 500:
                del src_list[src_list.index(char)]
                src_len = len(src_list)
                df = pd.DataFrame(src_list)
                break_flag = True
                break

        output_content.append(''.join(chars) + '\n')

        if break_flag:
            break_flag = False
            break

    if len(df) == 10:
        break


with open(output, 'w') as f:
    f.writelines(output_content)
