import glob
import sys


import aiosupabase
from PIL import Image
import os

import pandas as pd
from PIL import Image
import os


text2image_style_prompts = {
    0: {'label': 'none',
        'prompt': '',
        'disallow': []},
    1: {'label': '胶片',
        'prompt': 'analog film photo {prompt} . faded film,desaturated,35mm photo,grainy,vignette,vintage,Kodachrome,Lomography,stained,found footage,<lora:JuggerCineXL2:1>,Movie Still,Film Still',
        'disallow': []},
    2: {'label': '动画',
        'prompt': 'anime artwork {prompt} . anime style,key visual,vibrant,studio anime',
        'disallow': []},
    3: {'label': '电影',
        'prompt': 'cinematic film still {prompt} . shallow depth of field,vignette,high budget,bokeh,cinemascope,moody,epic,gorgeous,film grain,grainy,<lora:JuggerCineXL2:1>,Cinematic,Cinematic Shot,Cinematic Lighting',
        'disallow': []},
    4: {'label': '漫画',
        'prompt': 'comic {prompt} . graphic illustration,comic art,graphic novel art,vibrant',
        'disallow': []},
    5: {'label': '橡皮泥',
        'prompt': 'play-doh style {prompt} . sculpture,clay art,centered composition,Claymation',
        'disallow': []},
    6: {'label': '梦幻',
        'prompt': 'ethereal fantasy concept art of  {prompt} . magnificent,celestial,ethereal,painterly,epic,majestic,magical,fantasy art,cover art,dreamy',
        'disallow': []},
    7: {'label': '等距',
        'prompt': 'isometric style {prompt} . vibrant,beautiful,crisp,intricate',
        'disallow': []},
    8: {'label': '线条',
        'prompt': 'line art drawing {prompt} . professional,sleek,modern,minimalist,graphic,line art,vector graphics',
        'disallow': ['<lora:xl_more_art-full_v1:0.8>','<lora:WowifierXL-V2:0.6>','extremely detailed']},
    9: {'label': '多边形',
        'prompt': 'low-poly style {prompt} . low-poly game art,polygon mesh,jagged,blocky,wireframe edges,centered composition',
        'disallow': []},
    10: {'label': '霓虹朋克',
         'prompt': 'neonpunk style {prompt} . cyberpunk,vaporwave,neon,vibes,vibrant,stunningly beautiful,crisp,sleek,ultramodern,magenta highlights,purple shadows,high contrast,cinematic,intricate,professional,glowneon,<lora:glowneon_xl_v1:1>',
         'disallow': []},
    11: {'label': '折纸',
         'prompt': 'origami style {prompt} . paper art,pleated paper,folded,origami art,pleats,cut and fold,centered composition',
         'disallow': []},
    12: {'label': '摄影',
         'prompt': 'cinematic photo {prompt} . 35mm photograph,film,bokeh,professional,4k',
         'disallow': []},
    13: {'label': '像素风',
         'prompt': 'pixel-art {prompt} . low-res,blocky,pixel art style,8-bit graphics',
         'disallow': ['<lora:xl_more_art-full_v1:0.8>','<lora:WowifierXL-V2:0.6>','extremely detailed']},
    14: {'label': '汽车',
         'prompt': 'Automotive advertisement style {prompt} . Sleek,dynamic,professional,commercial,vehicle-focused,high-resolution,Car Photography',
         'disallow': []},
    15: {'label': '时尚杂志',
         'prompt': 'Fashion editorial style {prompt} . High fashion,trendy,stylish,editorial,magazine style,professional',
         'disallow': []},
    16: {'label': '食品摄影',
         'prompt': 'Food photography style {prompt} . Appetizing,professional,culinary,high-resolution,commercial,Food Photography',
         'disallow': []},
    17: {'label': '奢侈品',
         'prompt': 'Luxury product style {prompt} . Elegant,sophisticated,high-end,luxurious,professional',
         'disallow': []},
    18: {'label': '房屋',
         'prompt': 'Real estate photography style {prompt} . Professional,inviting,well-lit,high-resolution,property-focused,commercial,Architecture Photography',
         'disallow': []},
    19: {'label': '小商品包装',
         'prompt': 'Retail packaging style {prompt} . Vibrant,enticing,commercial,product-focused,eye-catching,professional',
         'disallow': []},
    20: {'label': '抽象派',
         'prompt': 'abstract expressionist painting {prompt} . energetic brushwork,bold colors,abstract forms,expressive,emotional',
         'disallow': ['<lora:xl_more_art-full_v1:0.8>','<lora:WowifierXL-V2:0.6>','extremely detailed']},
    21: {'label': '装饰',
         'prompt': 'Art Deco style {prompt} . geometric shapes,bold colors,luxurious,elegant,decorative,symmetrical,ornate,Interior Photography',
         'disallow': []},
    22: {'label': '色块拼图',
         'prompt': 'cubist artwork {prompt} . geometric shapes,abstract,innovative,revolutionary',
         'disallow': []},
    23: {'label': '涂鸦',
         'prompt': 'graffiti style {prompt} . street art,vibrant,urban,tag,mural',
         'disallow': []},
    24: {'label': '印象派',
         'prompt': 'impressionist painting {prompt} . loose brushwork,vibrant color,light and shadow play,captures feeling over form',
         'disallow': ['<lora:xl_more_art-full_v1:0.8>','<lora:WowifierXL-V2:0.6>','extremely detailed']},
    25: {'label': '点彩画',
         'prompt': 'pointillism style {prompt} . composed entirely of small,distinct dots of color,vibrant',
         'disallow': []},
    26: {'label': '流行艺术',
         'prompt': 'Pop Art style {prompt} . bright colors,bold outlines,popular culture themes,ironic or kitsch',
         'disallow': ['<lora:xl_more_art-full_v1:0.8>','<lora:WowifierXL-V2:0.6>','extremely detailed']},
    27: {'label': '迷幻',
         'prompt': 'psychedelic style {prompt} . vibrant colors,swirling patterns,abstract forms,surreal,trippy',
         'disallow': ['<lora:xl_more_art-full_v1:0.8>','<lora:WowifierXL-V2:0.6>','extremely detailed']},
    28: {'label': '蒸汽朋克',
         'prompt': 'steampunk style {prompt} . mechanical,intricate',
         'disallow': []},
    29: {'label': '水彩画',
         'prompt': 'watercolor painting {prompt} . vibrant,beautiful,painterly,textural,artistic',
         'disallow': []},
    30: {'label': '生物赛博',
         'prompt': 'biomechanical cyberpunk {prompt} . cybernetics,human-machine fusion,dystopian,organic meets artificial,intricate',
         'disallow': []},
    31: {'label': '赛博机器人',
         'prompt': 'cybernetic robot {prompt} . android,AI,machine,metal,wires,tech,futuristic',
         'disallow': []},
    32: {'label': '赛博城市',
         'prompt': 'cyberpunk cityscape {prompt} . neon lights,alleys,skyscrapers,futuristic,vibrant colors,high contrast',
         'disallow': []},
    33: {'label': '未来主义',
         'prompt': 'futuristic style {prompt} . sleek,modern',
         'disallow': []},
    34: {'label': '复古赛博',
         'prompt': "retro cyberpunk {prompt} . 80's inspired,synthwave,neon,vibrant,retro futurism",
         'disallow': []},
    35: {'label': '复古科幻',
         'prompt': 'retro-futuristic {prompt} . vintage sci-fi,50s and 60s style,atomic age,vibrant',
         'disallow': []},
    36: {'label': '科幻',
         'prompt': 'sci-fi style {prompt} . futuristic,technological,alien worlds,space themes,advanced civilizations',
         'disallow': []},
    37: {'label': '泡泡龙',
         'prompt': 'Bubble Bobble style {prompt} . 8-bit,cute,pixelated,fantasy,vibrant,reminiscent of Bubble Bobble game',
         'disallow': []},
    38: {'label': '赛博游戏',
         'prompt': 'cyberpunk game style {prompt} . neon,dystopian,futuristic,digital,vibrant,high contrast,reminiscent of cyberpunk genre video games',
         'disallow': []},
    39: {'label': '格斗游戏',
         'prompt': 'fighting game style {prompt} . dynamic,vibrant,action-packed,detailed character design,reminiscent of fighting video games',
         'disallow': []},
    40: {'label': '侠盗飞车',
         'prompt': 'GTA-style artwork {prompt} . satirical,exaggerated,pop art style,vibrant colors,iconic characters,action-packed',
         'disallow': []},
    41: {'label': '马里奥',
         'prompt': 'Super Mario style {prompt} . vibrant,cute,cartoony,fantasy,playful,reminiscent of Super Mario series',
         'disallow': []},
    42: {'label': '体素风',
         'prompt': 'Minecraft style {prompt} . blocky,pixelated,vibrant colors,recognizable characters and objects,game assets',
         'disallow': []},
    43: {'label': '宝可梦',
         'prompt': 'Pokémon style {prompt} . vibrant,cute,anime,fantasy,reminiscent of Pokémon series',
         'disallow': []},
    44: {'label': '街头霸王',
         'prompt': 'Street Fighter style {prompt} . vibrant,dynamic,arcade,2D fighting game,reminiscent of Street Fighter series',
         'disallow': []},
    45: {'label': '迪斯科',
         'prompt': 'disco-themed {prompt} . vibrant,groovy,retro 70s style,shiny disco balls,neon lights,dance floor',
         'disallow': []},
    46: {'label': '世界末日',
         'prompt': 'dystopian style {prompt} . bleak,post-apocalyptic,somber,dramatic',
         'disallow': []},
    47: {'label': '童话',
         'prompt': 'fairy tale {prompt} . magical,fantastical,enchanting,storybook style',
         'disallow': []},
    48: {'label': '哥特式',
         'prompt': 'gothic style {prompt},mysterious,haunting,dramatic,ornate',
         'disallow': []},
    49: {'label': '摇滚',
         'prompt': 'grunge style {prompt} . textured,distressed,vintage,edgy,punk rock vibe,dirty,noisy',
         'disallow': []},
    50: {'label': '恐怖',
         'prompt': 'horror-themed {prompt} . eerie,unsettling,spooky,suspenseful,grim',
         'disallow': []},
    51: {'label': '可爱',
         'prompt': 'kawaii style {prompt} . cute,adorable,brightly colored,cheerful,anime influence',
         'disallow': []},
    52: {'label': '魔幻',
         'prompt': 'lovecraftian horror {prompt} . eldritch,cosmic horror,unknown,mysterious,surreal',
         'disallow': []},
    53: {'label': '阴森','prompt': 'macabre style {prompt} . dark,gothic,grim,haunting',
         'disallow': []},
    54: {'label': '摩天高楼',
         'prompt': 'metropolis-themed {prompt} . urban,cityscape,skyscrapers,modern,futuristic',
         'disallow': []},
    55: {'label': '单色',
         'prompt': 'monochrome {prompt} . black and white,contrast,tone,texture',
         'disallow': []},
    56: {'label': '航海',
         'prompt': 'nautical-themed {prompt} . sea,ocean,ships,maritime,marine life',
         'disallow': []},
    57: {'label': '宇宙太空',
         'prompt': 'space-themed {prompt} . cosmic,celestial,stars,galaxies,nebulas,planets,science fiction',
         'disallow': []},
    58: {'label': '染色玻璃',
         'prompt': 'stained glass style {prompt} . vibrant,beautiful,translucent,intricate',
         'disallow': []},
    59: {'label': '时尚赛博',
         'prompt': 'techwear fashion {prompt} . futuristic,cyberpunk,sleek',
         'disallow': []},
    60: {'label': '原始部落',
         'prompt': 'tribal style {prompt} . indigenous,ethnic,traditional patterns,bold,natural colors',
         'disallow': []},
    61: {'label': '复杂单色',
         'prompt': 'zentangle {prompt} . intricate,abstract,monochrome,patterns,meditative',
         'disallow': []},
    62: {'label': '平面剪纸',
         'prompt': 'flat papercut style {prompt} . silhouette,clean cuts,paper,sharp edges,minimalist,color block',
         'disallow': ['<lora:xl_more_art-full_v1:0.8>','<lora:WowifierXL-V2:0.6>','extremely detailed']},
    63: {'label': '立体剪纸',
         'prompt': 'kirigami representation of {prompt} . 3D,paper folding,paper cutting,Japanese,intricate,symmetrical,precision,clean lines',
         'disallow': ['<lora:xl_more_art-full_v1:0.8>','<lora:WowifierXL-V2:0.6>','extremely detailed']},
    64: {'label': '纸浆',
         'prompt': 'paper mache representation of {prompt} . 3D,sculptural,textured,handmade,vibrant,fun',
         'disallow': ['<lora:xl_more_art-full_v1:0.8>','<lora:WowifierXL-V2:0.6>','extremely detailed']},
    65: {'label': '剪纸拼图',
         'prompt': 'papercut collage of {prompt} . mixed media,textured paper,overlapping,asymmetrical,abstract,vibrant',
         'disallow': ['<lora:xl_more_art-full_v1:0.8>','<lora:WowifierXL-V2:0.6>','extremely detailed']},
    66: {'label': '地外文明',
         'prompt': 'alien-themed {prompt} . extraterrestrial,cosmic,otherworldly,mysterious,sci-fi',
         'disallow': []},
    67: {'label': '黑白电影',
         'prompt': 'film noir style {prompt} . monochrome,high contrast,dramatic shadows,1940s style,mysterious,cinematic',
         'disallow': []},
    68: {'label': '高清',
         'prompt': 'HDR photo of {prompt} . High dynamic range,vivid,rich details,clear shadows and highlights,realistic,intense,enhanced contrast,Hyperdetailed Photography',
         'disallow': []}}
#
# 指定 Excel 文件路径
excel_file_path = '/home/zzg/项目文档/AIGC.xlsx'

# 创建一个逆映射字典，将label映射到key
label_to_key = {v['label']: k for k, v in text2image_style_prompts.items()}

# 定义一个函数来匹配style描述与对应的key
def match_style_to_key(style):
    if style is not None:
        if style not in label_to_key:
            print(f"Style not found: {style}")
            sys.exit(1)  # 非零退出代码通常表示程序由于某些错误而退出
        return label_to_key[style]

#在字符串的前面添加前导零
# string: 需要添加前导零的原始字符串。
# desired_length: 期望的最终字符串长度。
# fill_char (可选): 用于填充的字符,默认为'0'。
def add_leading_zeros(string, desired_length, fill_char='0'):
    return (fill_char * (desired_length - len(string))) + string

# 使用 pandas 读取 Excel 文件的特定行和列
# header=None 表示原始数据没有列名，sheet_name 参数根据你的实际情况调整
# usecols='A:E' 表示只读取 A 到 E 列，skiprows 跳过前 5 行（因为行索引从 0 开始，且不包括结束行），nrows 读取 102 行（因为不包括起始行）
df = pd.read_excel(excel_file_path, usecols='B:D', skiprows=range(1), header=None, sheet_name='文生图sample')
df = df.dropna(subset=[2])
# 使用apply函数应用自定义函数
try:
    df[2] = df[2].apply(match_style_to_key)
except SystemExit as e:
    print(f"Program exited with error: {e}")
    raise  # 这将重新抛出异常，如果需要可以进行额外的处理

user_id = 'd8f5d02e-5e36-4040-be84-15a0a2cf90e8'
url = "https://www.guijutech.com:8888/"
key = "xxx"
# supabase_client = create_client(CONFIG['supabase']['url'], CONFIG['supabase']['key'])
supabase_client = aiosupabase.Supabase
supabase_client.configure(
    url=url,
    key=key,
    debug_enabled=True,
)
# 遍历 DataFrame，并输出 A 列和 E 列的数据
# 在这里，列的索引是从 0 开始的，所以 A 列是 0，E 列是 4
for index, row in df.iterrows():
    instance_id = os.path.basename(row[1].replace('png', 'webp'))
    instance_id = add_leading_zeros(instance_id, 11, '0')
    # 首先检查相同的instance_id是否已存在
    existing = supabase_client.table("gallery").select("instance_id").eq('instance_id', instance_id).execute()
    # 如果existing为空，说明没有相同的instance_id存在，可以插入数据
    if len(existing.data)==0:
        img = Image.open(row[1])
        w, h = img.size
        config = {'width': w, 'height': h, 'style': int(row[2]), 'translate': False}
        supabase_client.table("gallery").insert({"config": config, 'prompt':row[3], 'instance_id': instance_id, 'public': True, 'category': 'text2image'}).execute()
        print(f'The instance_id:{instance_id} is inserted.')
    else:
        # instance_id已存在，不插入数据
        print(f'The instance_id:{instance_id} already exists. No data was inserted.')
# print(dict(ddd))
# image = Image.open('/home/ray/Workspace/project/demo_web_sys/guiju_dashboard/public/assets/magic_text2image/Pokémon.png')
# image.save(os.path.join('/home/ray/Workspace/project/demo_web_sys/guiju_dashboard/public/assets/magic_text2image/Pokémon.webp'))
