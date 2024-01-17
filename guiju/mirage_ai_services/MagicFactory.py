# coding=utf-8
# @Time : 2023/12/15 下午3:14
# @File : magic_wallpapaper.py
import numpy as np

prompt_style = [
    {'label': '无', 'prompt': None},
    {'label': '电影感', 'prompt': 'cinematic photo,cinematic light,film,depth of field,blurry background,bokeh,gloom'},
    {'label': '唯美写真', 'prompt': 'soft light,warm color,light leak,filmg'},
]

prompt_distance = [
    {'label': '面部特写', 'prompt': '(close-up:1.3),portrait', 'width': 512,
     'height': 512},
    {'label': '上半身照', 'prompt': '(upper body:1.3)', 'width': 512, 'height': 768},
    {'label': '全身照', 'prompt': '(full body:1.3)', 'width': 512, 'height': 1024},
]

prompt_gender = [
    {
        'label': '女',
        'prompt': '1female'
    },
    {
        'label': '男',
        'prompt': '1male'
    },
]

# prompt_age = [
#     {'label': '儿童', 'prompt': 'child'},
#     {'label': '青年', 'prompt': 'youth'},
#     {'label': '中年', 'prompt': 'middle-aged'},
#     {'label': '老年', 'prompt': 'elder'},
# ]

prompt_male_character = {
    0: {'label': '天使', 'prompt': '(huge angel wings:1.5),(angel:1.5)'},
    1: {'label': '美人鱼', 'prompt': '(mermaid:1.5)'},
    2: {'label': 'Dracula', 'prompt': '(Dracula:1.5)'},
    3: {'label': 'Spiderman', 'prompt': '(Spiderman:1.5)'},
    4: {'label': 'Superman', 'prompt': '(Superman:1.5)'},
    5: {'label': 'Iron man', 'prompt': '(Iron man:1.5)'},
    6: {'label': '滅霸', 'prompt': 'jim lee,Thanos', 'lora': [{'name': 'jim_lee_offset_right_filesize', 'scale': 0.8, 'trigger': 'jim lee,Thanos'}]},
    7: {'label': 'Deadpool', 'prompt': '(Deadpool:1.5)'},
    8: {'label': 'Thor', 'prompt': '(Thor:1.5)'},
    9: {'label': 'Black Panther', 'prompt': '(Black Panther:1.5)'},
    10: {'label': 'Captain America', 'prompt': '(Captain America:1.5)'},
    11: {'label': 'Green Arrow', 'prompt': '(Green Arrow:1.5)'},
    12: {'label': 'Batman', 'prompt': '(Batman:1.5)',
         'female': {
             'label': '女蝙蝠侠Batgirl', 'prompt': 'cassandra,cape,bodysuit,black bodysuit,belt,belt pouch,black gloves,black pants',
              'lora': [{'name': 'cassandracain-nvwls-v1', 'scale': 0.8}]
            }
         },
    13: {'label': 'Harley Quinn', 'prompt': 'pikkyharleyquinn,midriff', 'lora': [{'name': 'HarleyQuinnV2', 'scale': 0.9}]},
    14: {'label': 'Black Widow', 'prompt': 'blkwidow,black bodysuit,cleavage,black belt,superhero suit',
         'lora': [{'name': 'blackwidow-nvwls-v1', 'scale': 0.8}]},
    15: {'label': 'War Machine', 'prompt': '(War Machine:1.5)'},
    16: {'label': 'Scarlet Witch', 'prompt': '(Scarlet Witch:1.5)'},
    17: {'label': 'Moon Knight', 'prompt': '(Moon Knight:1.5)'},
    18: {'label': 'Star-Lord ', 'prompt': '(Star-Lord :1.5)'},
    19: {'label': 'Harry Potter', 'prompt': '(Harry Potter:1.5)'},
    20: {'label': 'Seiya', 'prompt': '(Seiya:1.5)'},
    21: {'label': 'Kamen Rider', 'prompt': '(Kamen Rider:1.5)'},
    22: {'label': '亚马逊女战士', 'prompt': 'wonder woman,superhero suit', 'lora': [{'name': 'ww_v1', 'scale': 0.4}]},
    23: {'label': '凤凰女(琴·格蕾)', 'prompt': 'Jean-grey', 'lora': [{'name': 'jean_grey-10', 'scale': 1}]},
    24: {'label': '猫女', 'prompt': 'DC_catwoman_comic_bodysuit,DC_catwoman_comic_cleavage',
         'lora': [{'name': 'CARTOON_DC_catwoman_comic_ownwaifu-15', 'scale': 0.7}]},
    25: {'label': '妙语Punchline', 'prompt': 'CARTOON_DC_punchline_ownwaifu', 'lora': [{'name': 'CARTOON_DC_punchline_ownwaifu-15', 'scale': 0.7}]},
    26: {'label': '女巫', 'prompt': 'enchantress,superhero suit', 'lora': [{'name': 'enchantress-10IJ2v8', 'scale': 0.7}]},
    # 27: {'label': '女蝙蝠侠Batgirl', 'prompt': 'cassandra,cape,bodysuit,black bodysuit,belt,belt pouch,black gloves,black pants',
    #      'lora': [{'name': 'cassandracain-nvwls-v1', 'scale': 0.8}]},
    27: {'label': '红灯侠（Red Lantern）', 'prompt': 'red lantern costume,superhero suit',
         'lora': [{'name': 'Red Lantern Costume_v1', 'scale': 1}]},
    28: {'label': '蓝灯侠（Blue Lantern）', 'prompt': 'blue lantern costume,superhero suit',
         'lora': [{'name': 'Blue Lantern Costume_v1', 'scale': 1}]},
    29: {'label': '绿灯侠（Green Lantern）', 'prompt': 'green lantern costume,superhero suit',
         'lora': [{'name': 'Blue Lantern Costume_v1', 'scale': 0.9}]},
    30: {'label': '大芭达(Big Barda)', 'prompt': 'bigbarda,blue and gold bodysuit,helmet,red cape',
         'lora': [{'name': 'bigbarda-11DCG', 'scale': 1}]},
    31: {'label': '蓝甲虫（Blue Beetle）', 'prompt': 'Blue_Beetle_DC,superhero suit',
         'lora': [{'name': 'Blue_Beetle_DC_v1', 'scale': 0.8}]},
    32: {'label': '假面騎士BLACK', 'prompt': 'Kamen_Rider_Black_RX,superhero suit',
         'lora': [{'name': 'Kamen_Rider_Black_RX', 'scale': 0.8}]},
    33: {'label': '白戰士(White Power Ranger)', 'prompt': 'White_Ranger,solo,black breastplate,detailed armor,superhero suit',
         'lora': [{'name': 'White_Ranger', 'scale':1}]},
    34: {'label': '沙赞（Shazam）', 'prompt': 'shazamsuit,superhero suit',
         'lora': [{'name': 'Shazamsuit_Lora_v1', 'scale': 0.6}]},
    35: {'label': '北极星(Polaris)', 'prompt': 'Polaris,bodysuit,cape,superhero suit',
         'lora': [{'name': 'polaris-10', 'scale': 0.6}]},
    36: {'label': '奥特曼(ultraman)', 'prompt': 'ultraman,[red|white]',
         'lora': [{'name': 'ultraman2', 'scale': 0.7}]},
    37: {'label': '惊奇队长(Captain Marvel)', 'prompt': 'cptMarvel,bodysuit,superhero suit',
         'lora': [{'name': 'cptmarvel-nvwls-v1', 'scale': 1}]},
    38: {'label': '女毒液(Venom)', 'prompt': 'CARTOON_MARVEL_she_venom_ownwaifu,bodysuit',
         'lora': [{'name': 'CARTOON_MARVEL_she_venom_ownwaifu-15', 'scale': 0.8}]},
    39: {'label': '三国武将', 'prompt': 'jim lee,sanguo male warrior,chinese chainmail long armour,cape',
         'lora': [{'name': 'jim_lee_offset_right_filesize', 'scale': 0.8}]},
}

prompt_female_character = {
    0: {'label': '天使', 'prompt': '(huge angel wings:1.5),(angel:1.5)'},
    1: {'label': '美人鱼', 'prompt': '(mermaid:1.5)'},
    2: {'label': 'Dracula', 'prompt': '(Dracula:1.5)'},
    3: {'label': 'Spiderman', 'prompt': '(Spiderman:1.5)'},
    4: {'label': 'Superman', 'prompt': '(Superman:1.5)'},
    5: {'label': 'Iron man', 'prompt': '(Iron man:1.5)'},
    6: {'label': 'Thanos', 'prompt': '(Thanos:1.5)'},
    7: {'label': 'Deadpool', 'prompt': '(Deadpool:1.5)'},
    8: {'label': 'Thor', 'prompt': '(Thor:1.5)'},
    9: {'label': 'Black Panther', 'prompt': '(Black Panther:1.5)'},
    10: {'label': 'Captain America', 'prompt': '(Captain America:1.5)'},
    11: {'label': 'Green Arrow', 'prompt': '(Green Arrow:1.5)'},
    12: {'label': 'Batman', 'prompt': '(Batman:1.5)'},
    13: {'label': 'Harley Quinn', 'prompt': '(Harley Quinn:1.5)'},
    14: {'label': 'Black Widow', 'prompt': '(Black Widow:1.5)'},
    15: {'label': 'War Machine', 'prompt': '(War Machine:1.5)'},
    16: {'label': 'Scarlet Witch', 'prompt': '(Scarlet Witch:1.5)'},
    17: {'label': 'Moon Knight', 'prompt': '(Moon Knight:1.5)'},
    18: {'label': 'Star-Lord ', 'prompt': '(Star-Lord :1.5)'},
    19: {'label': 'Harry Potter', 'prompt': '(Harry Potter:1.5)'},
    20: {'label': 'Seiya', 'prompt': '(Seiya:1.5)'},
    21: {'label': 'Kamen Rider', 'prompt': '(Kamen Rider:1.5)'},
    22: {'label': '亚马逊女战士', 'prompt': '(Kamen Rider:1.5)'},
}

prompt_costume = {
    0: {'label': '校服', 'prompt': '(school uniform:1.3)'},
    1: {'label': '天使', 'prompt': '(huge angel wings:1.3),(angel:1.3)'},
    2: {'label': '美人鱼', 'prompt': '(mermaid:1.3)'},
    3: {'label': '水手服套装', 'prompt': '(sailor collar:1.3)'},
    4: {'label': '水手连衣裙', 'prompt': '(sailor dress:1.3)'},
    5: {'label': '职业装', 'prompt': '(business_suit:1.3)'},
    6: {'label': '军服', 'prompt': '(garreg mach monastery uniform:1.3)'},
    7: {'label': '晚礼服', 'prompt': '(evening gown:1.3)'},
    8: {'label': '婚纱', 'prompt': '(wedding_dress:1.3)'},
    9: {'label': '毛衣', 'prompt': '(sweater:1.3)'},
    10: {'label': '长毛衣连衣裙', 'prompt': '(sweater dress:1.3)'},
    11: {'label': '短毛衣夹克', 'prompt': '(sweater jacket:1.3)'},
    12: {'label': '工装服', 'prompt': '(dungarees:1.3)'},
    13: {'label': '卫衣', 'prompt': '(hoodie:1.3)'},
    14: {'label': '披风', 'prompt': '(cloak:1.3)'},
    15: {'label': '斗篷', 'prompt': '(cape:1.3)'},
    16: {'label': '围裙', 'prompt': '(apron:1.3)'},
    17: {'label': '哥特风', 'prompt': '(gothic:1.3)'},
    18: {'label': '公主装', 'prompt': '(lolita_fashion:1.3)'},
    19: {'label': '洛丽塔', 'prompt': '(gothic_lolita:1.3)'},
    20: {'label': '长运动服', 'prompt': '(tracksuit:1.3)'},
    21: {'label': '短运动服', 'prompt': '(exercise clothing:1.3)'},
    22: {'label': '剪裁牛仔', 'prompt': '(cropped jacket :1.3)'},
    23: {'label': '睡衣', 'prompt': '(pajamas:1.3)'},
    24: {'label': '和服', 'prompt': '(japanese yukata:1.3)'},
    25: {'label': '迷你裙', 'prompt': '(miniskirt:1.3)'},
    26: {'label': '比基尼', 'prompt': '(bikini:1.3)'},
    27: {'label': '连体泳衣', 'prompt': '(swimsuit:1.3)'},
    28: {'label': '情趣内衣', 'prompt': '(sexy lingerie:1.3)'},
    29: {'label': '透明内衣', 'prompt': '(transparent underwear:1.3)'},
    30: {'label': 'T恤', 'prompt': '(t-shirt:1.3)'},
    31: {'label': '吊带', 'prompt': '(camisole:1.3)'},
    32: {'label': '圣诞装', 'prompt': '(santa dress:1.3)'},
    33: {'label': '棒球服', 'prompt': '(letterman jacket:1.3)'},
    34: {'label': '排球服', 'prompt': '(volleyball uniform:1.3)'},
    35: {'label': '足球队衣', 'prompt': '(soccer team jersey:1.3)'},
    36: {'label': '篮球服', 'prompt': '(basketball uniform:1.3)'},
    37: {'label': '体操服', 'prompt': '(athletic leotard:1.3)'},
    38: {'label': '兔女郎', 'prompt': '(playboy bunny leotard:1.3)'},
    39: {'label': '高领毛衣', 'prompt': '(turtleneck sweater:1.3)'},
    40: {'label': '旗袍', 'prompt': '(cheongsam:1.3)'},
    41: {'label': '宇航服', 'prompt': '(space suit:1.3)'},
    42: {'label': '紧身乳胶衣', 'prompt': '(latex_bodysuit:1.3)'},
    43: {'label': '赛车服', 'prompt': '(racing suit:1.3)'},
    44: {'label': '医生白大挂', 'prompt': '(lab_coat:1.3)'},
    45: {'label': '兜帽斗篷', 'prompt': '(cape hood:1.3)'},
    46: {'label': '风衣', 'prompt': '(overcoat:1.3)'},
    47: {'label': '汉服', 'prompt': '(hanfu:1.3)'},
    48: {'label': '战斗服', 'prompt': '(combat suit:1.3)'},
    49: {'label': '皮夹克', 'prompt': '(leather jacket:1.3)'},
    50: {'label': '浴袍', 'prompt': '(bathrobe:1.3)'},
    51: {'label': '盔甲', 'prompt': '(armor:1.3)'},
    52: {'label': '动力甲', 'prompt': '(power armor:1.3)'},
    53: {'label': '外骨骼', 'prompt': '(exoskeleton:1.3)'},
    54: {'label': '外骨骼机甲', 'prompt': '(Exoskeleton Mecha:1.3)'},
    55: {'label': '道袍', 'prompt': '(Taoist robe:1.3)'},
    56: {'label': '军官大衣', 'prompt': '(Army overcoat:1.3)'},
    57: {'label': '盔甲裙', 'prompt': '(armored dress:1.3)'},
    58: {'label': '空手道服', 'prompt': '(karate uniform:1.3)'},
    59: {'label': '古希腊服装', 'prompt': '(Greek clothes:1.3)'},
    60: {'label': '印第安服饰', 'prompt': '(pocahontas outfit:1.3)'},
}

prompt_scene = {
    0: {'label': '赛博朋克',
        'prompt': '(cyberpunk:1.3),(cyber sci-fi:1.3),(surreal:1.3)'},
    1: {'label': '太空',
        'prompt': 'earth,cosmic,celestial,space suit,astronaut,universe,space,science fiction,galaxy,floating,stars,nebula'},
    2: {'label': '蒸汽朋克',
        'prompt': '(steampunk:1.3),mechanical,surreal'},
    3: {'label': '机器改造人',
        'prompt': '(robot:1.3),surreal'},
    4: {'label': '赛博机器人',
        'prompt': '(cyberpunk:1.3),mechanical,surreal'},
    5: {'label': '竹林',
        'prompt': '(bamboo forest:1.3)'},
    6: {'label': '健身房',
        'prompt': '(gym background:1.3)'},
    7: {'label': '满月城堡',
        'prompt': 'a ruined medieval era fortress,surrounded by an ancient forest,nighttime,(full moon:1.3),stars'},
    8: {'label': '峡谷风光',
        'prompt': 'canyon,desert,mountains,sunset'},
    9: {'label': '运河桥上',
        'prompt': 'bridge,futuristic city in the background,ocean,mountains,sunrise'},
    10: {'label': '图书馆',
         'prompt': 'indoors,medieval themed library,candles,fireplace,windows,mountains in the background,sunset'},
    11: {'label': '山下村庄',
         'prompt': 'medieval era,village,mountains,sunrise'},
    12: {'label': '河边码头',
         'prompt': 'medieval era port city,harbor,fantasy,ocean,storm,rain,mountains'},
    13: {'label': '街拍',
         'prompt': 'public,(photographers:1.0),street,foreground,close up,outdoors,city'},
    14: {'label': '酒店',
         'prompt': 'grainy,lovehotel,LHbedpanel,scenery,couch,lamp,door,indoors,bed,table,chair'},
    15: {'label': '古罗马城',
         'prompt': 'Roman anphitheatre,Rome,Old times,beautiful,near the mountains,rocks,Caesar style,trees,perfect proportions,real colors'},
    16: {'label': '雪地森林',
         'prompt': 'enchanted forest,snow,night,stars,subsurface scattering,walking in the forest,path'},
    17: {'label': '林中小河',
         'prompt': 'path,river,waterfall,forest'},
    18: {'label': '天台',
         'prompt': 'school rooftop,building,chain-link fence,wind lift'},
    19: {'label': '星空',
         'prompt': 'sky,star,scenery,starry sky,night sky,outdoors,building,cloud,milky way,tree,city,silhouette,cityscape'},
    20: {'label': '七彩花海',
         'prompt': 'cinematic shot of alpine meadow,wildflowers,god rays'},
    21: {'label': '樱花盛开',
         'prompt': 'ray tracing,colorful,glowing light,(detailed background,complex background:1.2),cherry blossoms,park'},
    22: {'label': '巨大月亮',
         'prompt': '(full red moon:1.3),starry sky'},
    23: {'label': '废弃游乐场',
         'prompt': 'post-apocalypse,decayed amusement park,broken rides,faded colors,absurdress'},
    24: {'label': '夏威夷',
         'prompt': '(hawaii:1.3)'},
    25: {'label': '科幻世界',
         'prompt': 'cinematic SCI-FI environment,towering factories ,dystopian world,metallic architecture,cyber sci-fi'},
    26: {'label': '赛博酒吧',
         'prompt': 'cinematic SCI-FI environment,Robotic bartenders,bar,cyber sci-fi'},
    27: {'label': '飞船驾驶仓',
         'prompt': 'cinematic SCI-FI environment,swirling nebulae,starship interior,cyber sci-fi'},
    28: {'label': '沙漠遗迹',
         'prompt': 'cinematic SCI-FI environment,Ruins of an ancient alien civilization,desert,carved relics,cyber sci-fi'},
    29: {'label': '未来工厂',
         'prompt': 'cinematic SCI-FI environment,AI-operated factory,robotic ,artificial constellations,floating energy orbs,cyber sci-fi'},
    30: {'label': '篮球馆',
         'prompt': 'stage,scenery,indoors,aesthetic,stadium,basketball court'},
    31: {'label': '排球馆',
         'prompt': 'stage,scenery,indoors,aesthetic,stadium,volleyball court'},
    32: {'label': '国风山中古城',
         'prompt': 'ancient chinese style landscape painting,mountains in front,beautiful winding green river behind'},
    33: {'label': '多人办公室',
         'prompt': 'business office,window,desk,chair,computer,ceiling,ceiling light'},
    34: {'label': '林中瀑布',
         'prompt': 'diffused natural sunlight,park,woods,flowers,birds,waterfall'},
    35: {'label': '富士山下',
         'prompt': 'very beautiful landscape,Mt Fuji,(cherry blossoms:0.4)'}
}


class MagicFactory(object):
    operator = None

    # sd_model_name = 'dreamshaper_8'

    # denoising_strength_min = 0.5
    # denoising_strength_max = 1

    def __init__(self, operator):
        self.operator = operator

    def __call__(self, *args, **kwargs):
        # read params
        # params, user_id, input_image, pic_name
        params = kwargs['params']
        user_id = kwargs['user_id']
        _input_image = kwargs['input_image']
        pic_name = kwargs['pic_name']

        _style = int(params['style'])
        _distance = int(params['distance'])
        _gender = 0 if params['gender'] == 'female' else 1
        # _age = int(params['age'])
        _character = int(params['character'])
        _scene = int(params['scene'])
        _costume = int(params['costume'])
        _batch_size = int(params['batch_size'])

        _character_enable = bool(params['character_enable'])
        _costume_enable = bool(params['costume_enable'])
        _scene_enable = bool(params['scene_enable'])

        # save cache face img
        # _input_image.save(f'tmp/{self.__class__.__name__}_origin_{pic_name}_save.png')
        # _input_image = _input_image.convert('RGBA')

        _input_image_width, _input_image_height = _input_image.size

        if self.operator.update_progress(10):
            return {'success': True}

        # parse face
        face_boxes = self.operator.facer.detect_face(_input_image)
        if len(face_boxes) == 0:
            # return {'success': False, 'result': '未检测到人脸'}
            return {'success': False, 'result': 'backend.magic-factory.error.no-face'}

        elif len(face_boxes) > 1:
            # return {'success': False, 'result': '检测到多个人脸，请上传一张单人照'}
            return {'success': False, 'result': 'backend.magic-factory.error.multi-face'}

        else:
            if self.operator.update_progress(30):
                return {'succ'
                        'ess': True}

            # prompt
            positive_prompt = f'{prompt_gender[_gender]["prompt"]},{prompt_distance[_distance]["prompt"]}'
            if _character_enable:
                character_dict = prompt_male_character[_character]['female'] if 'female' in prompt_male_character[_character].keys() and _gender == 0 else prompt_male_character[_character]
                lora_enable = 'lora' in character_dict.keys()
                positive_prompt = positive_prompt + f',{character_dict["prompt"]}'
            if _costume_enable:
                positive_prompt = positive_prompt + f',{prompt_costume[_costume]["prompt"]}'
            if _scene_enable:
                positive_prompt = positive_prompt + f',{prompt_scene[_scene]["prompt"]}'
            if _style != 0:
                positive_prompt = positive_prompt + f',{prompt_style[_style]["prompt"]}'
            positive_prompt = positive_prompt + f',(best quality),(high quality),high details,masterpiece,extremely detailed,(sharp focus),(cinematic lighting),high saturation,ultra detailed,detailed background,wide view,sharp and crisp background,epic composition,intricate,solo,professional,4k'
            negative_prompt = f'(NSFW:1.3),cartoon,painting,illustration,(worst quality:2),(low quality:2),(normal quality:2),(Multiple people:1.3),bad anatomy,DeepNegative,text,error,cropped,mutation,jpeg artifacts,polar lowres,bad proportions,gross proportions,deformed body,cross-eyed,sketches,bad hands,blurry,bad feet,poorly drawn hands,extra fingers,fewer digits,extra limbs,extra arms,extra legs,malformed limbs,(fused fingers:1.3),(too many fingers:1.3),long neck,mutated hands,polar lowres,bad body,(missing fingers:1.3),missing arms,missing legs,extra digit,extra foot'

            print("-------------------factory logger-----------------")
            print(f"sd_positive_prompt: {positive_prompt}")
            print(f"sd_negative_prompt: {negative_prompt}")
            # self.operator.logging(
            #     f"[_reference_image_path][{_reference_image_path}]:\n",
            #     f"logs/sd_webui.log")
            res = self.operator.faceid_predictor(np.array(_input_image), positive_prompt, negative_prompt, _batch_size,
                                                 prompt_distance[_distance]['width'],
                                                 prompt_distance[_distance]['height'],
                                                 lora=character_dict['lora'] if lora_enable else None)

            return res
