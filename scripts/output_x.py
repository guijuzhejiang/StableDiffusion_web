import glob

search_dir = '/home/ray/Workspace/project/ocr/rec'
output_fp = 'generate_qinghua_labels_x.txt'
output_c = []
for src in glob.glob(f'{search_dir}/*.txt'):
    with open(src, 'r') as f1:
        c1 = f1.readlines()
        for l in c1:
            if '×' in l:
                output_c.append(l)

with open(output_fp, 'w') as f2:
    f2.writelines(output_c)