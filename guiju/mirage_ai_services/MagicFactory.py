# coding=utf-8
# @Time : 2023/12/15 下午3:14
# @File : magic_wallpapaper.py
import numpy as np

prompt_style = [
    {'label': '无', 'prompt': None},
    {'label': '电影感',
     'prompt': 'cinematic photo,cinematic light,film,depth of field,blurry background,bokeh,gloom,(photorealistic:1.2),professional,4k'},
    {'label': '唯美写真', 'prompt': 'soft light,warm color,light leak,filmg,(photorealistic:1.2),professional,4k'},
]

prompt_distance = [
    {'label': '面部特写', 'prompt': 'close-up,detailed face,delicate face,true skin texture,portrait', 'width': 512,
     'height': 512},
    {'label': '上半身照', 'prompt': 'upper body,true skin texture,portrait', 'width': 512, 'height': 768},
    {'label': '全身照', 'prompt': 'full body', 'width': 512, 'height': 1024},
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

prompt_character = {
    0: {'label': '天使', 'prompt': '(huge angel wings:1.3),(angel:1.3)'},
    1: {'label': '美人鱼', 'prompt': '(mermaid:1.3)'},
    2: {'label': '护士', 'prompt': '(nurse:1.3)'},
    3: {'label': '空姐', 'prompt': '(stewardess:1.3)'},
    4: {'label': '学生', 'prompt': '(school uniform:1.3)'},
    5: {'label': '牧师', 'prompt': '(police:1.3)'},
}

prompt_costume = {
    0: {'label': '水手服套装', 'prompt': '(sailor collar:1.3),(sailor suite:1.3),(sailor shirt:1.3)'},
    1: {'label': '水手连衣裙', 'prompt': '(sailor dress:1.3)'},
    2: {'label': '校服', 'prompt': '(school uniform:1.3)'},
    3: {'label': '职业装', 'prompt': '(business_suit:1.3),office lady suit,'},
    4: {'label': '军服', 'prompt': '(garreg mach monastery uniform:1.3),military uniform,'},
    5: {'label': '晚礼服', 'prompt': '(evening gown:1.3),(evening dress:1.3)'},
    6: {'label': '婚纱', 'prompt': '(wedding_dress:1.3)'},
    7: {'label': '毛衣', 'prompt': '(sweater:1.3)'},
    8: {'label': '长毛衣连衣裙', 'prompt': '(sweater dress:1.3)'},
    9: {'label': '短毛衣夹克', 'prompt': '(sweater jacket:1.3)'},
    10: {'label': '工装服', 'prompt': '(dungarees:1.3)'},
    11: {'label': '卫衣', 'prompt': '(hoodie:1.3)'},
    12: {'label': '披风', 'prompt': 'cloak,'},
    13: {'label': '斗篷', 'prompt': '(cape:1.3)'},
    14: {'label': '围裙', 'prompt': '(apron:1.3)'},
    15: {'label': '哥特风', 'prompt': '(gothic:1.3)'},
    16: {'label': '公主装', 'prompt': '(lolita_fashion:1.3)'},
    17: {'label': '洛丽塔', 'prompt': '(gothic_lolita:1.3)'},
    18: {'label': '长运动服', 'prompt': '(tracksuit:1.3)'},
    19: {'label': '短运动服', 'prompt': '(exercise clothing:1.3),(gym_uniform:1.3),(sports bra:1.3)'},
    20: {'label': '剪裁牛仔', 'prompt': '(cropped jacket :1.3)'},
    21: {'label': '睡衣', 'prompt': '(pajamas:1.3)'},
    22: {'label': '和服', 'prompt': '(japanese_clothes:1.3),(kimono:1.3),japanese yukata,'},
    23: {'label': '迷你裙', 'prompt': '(miniskirt:1.3)'},
    24: {'label': '比基尼', 'prompt': '(bikini:1.3)'},
    25: {'label': '连体泳衣', 'prompt': '(swimsuit:1.3)'},
    26: {'label': '情趣内衣', 'prompt': '(sexy lingerie:1.3)'},
    27: {'label': '透明内衣', 'prompt': '(transparent underwear:1.3)'},
    28: {'label': 'T恤', 'prompt': '(t-shirt:1.3)'},
    29: {'label': '吊带', 'prompt': 'camisole'},
    30: {'label': '圣诞装', 'prompt': '(santa dress:1.3)'},
    31: {'label': '棒球服', 'prompt': '(letterman jacket:1.3)'},
    32: {'label': '排球服', 'prompt': 'volleyball uniform'},
    33: {'label': '足球队衣', 'prompt': 'Soccer team jersey,'},
    34: {'label': '汉服', 'prompt': '(hanfu:1.3)'},
    35: {'label': '体操服', 'prompt': '(athletic leotard:1.3),student training wear,'},
    36: {'label': '兔女郎', 'prompt': '(playboy bunny leotard:1.3)'},
    37: {'label': '高领毛衣', 'prompt': 'turtleneck sweater,'},
    38: {'label': '旗袍', 'prompt': 'cheongsam'},
    39: {'label': '宇航服', 'prompt': '(space suit:1.2)'},
    40: {'label': '紧身乳胶衣', 'prompt': 'latex_bodysuit,'},
    41: {'label': '赛车服', 'prompt': 'racing suit,'},
    42: {'label': '医生白大挂', 'prompt': 'lab_coat'},
    43: {'label': '兜帽斗篷', 'prompt': 'Cape hood,'},
    44: {'label': '风衣', 'prompt': 'overcoat,'},
    45: {'label': '大衣', 'prompt': 'wind coat,'},
    46: {'label': '战斗服', 'prompt': 'combat suit,'},
    47: {'label': '皮夹克', 'prompt': 'leather jacket,'},
    48: {'label': '浴袍', 'prompt': 'bathrobe,'},
    49: {'label': '盔甲', 'prompt': 'armor,'},
    50: {'label': '动力甲', 'prompt': 'power armor,'},
    51: {'label': '外骨骼', 'prompt': 'exoskeleton,'},
    52: {'label': '外骨骼机甲', 'prompt': 'Exoskeleton Mecha,'},
    53: {'label': '道袍', 'prompt': 'Taoist robe,'},
    54: {'label': '军官大衣', 'prompt': 'Army overcoat,'},
    55: {'label': '盔甲裙', 'prompt': 'armored dress,'},
    56: {'label': '空手道服', 'prompt': 'karate uniform'},
    57: {'label': '古希腊服装', 'prompt': 'Greek clothes,'},
    58: {'label': '印第安服饰', 'prompt': 'pocahontas outfit,'}
}

prompt_scene = {
    0: {'label': '太空',
        'prompt': 'earth,cosmic,celestial,space suit,astronaut,night_sky,starry_sky,universe,space,science fiction,galaxy, floating,stars,nebula'},
    1: {'label': '赛博朋克',
        'prompt': 'cyberpunk,surreal'},
    2: {'label': '蒸汽朋克',
        'prompt': 'steampunk,mechanical,surreal'},
    3: {'label': '机器改造人',
        'prompt': 'robot,surreal'},
    4: {'label': '赛博机器人',
        'prompt': 'cyberpunk,mechanical,surreal'},
    5: {'label': '竹林',
        'prompt': 'bamboo forest'},
    6: {'label': '健身房',
        'prompt': '(gym background)'},
    7: {'label': '满月城堡',
        'prompt': 'a ruined medieval era fortress, surrounded by an ancient forest, nighttime, moon, stars,'},
    8: {'label': '峡谷风光',
        'prompt': 'happy, canyon, desert, mountains, sunset'},
    9: {'label': '运河桥上',
        'prompt': 'happy, bridge, futuristic city in the background, ocean, mountains, sunrise'},
    10: {'label': '图书馆',
         'prompt': 'indoors, medieval themed library, candles, fireplace, windows, mountains in the background, sunset,'},
    11: {'label': '山下村庄',
         'prompt': 'medieval era, village, mountains, sunrise'},
    12: {'label': '河边码头',
         'prompt': 'medieval era port city, harbor, fantasy, ocean, storm, rain, mountains'},
    13: {'label': '街拍',
         'prompt': 'public, (photographers:1.0), street,foreground,close up,outdoors, city'},
    14: {'label': '酒店',
         'prompt': 'grainy, lovehotel, LHbedpanel, scenery, couch, lamp, door, indoors, bed, table,chair'},
    15: {'label': '古罗马城',
         'prompt': 'Roman anphitheatre,Rome,Old times,beautiful,near the mountains,rocks,Caesar style,trees,perfect proportions,real colors'},
    16: {'label': '雪地森林',
         'prompt': 'enchanted forest, snow,night,stars,subsurface scattering,walking in the forest,path'},
    17: {'label': '林中小河',
         'prompt': 'grass, path, river, waterfall, forest'},
    18: {'label': '天台',
         'prompt': 'school rooftop,building, chain-link fence, wind lift, skirt tug,'},
    19: {'label': '星空',
         'prompt': 'sky,star,scenery,starry sky,night sky,outdoors,building,cloud,milky way,tree,city,silhouette,cityscape'},
    20: {'label': '七彩花海', 'prompt': 'cinematic shot of alpine meadow, wildflowers, god rays'},
    21: {'label': '樱花盛开',
         'prompt': 'ray tracing,colorful,glowing light, (detailed background, complex background:1.2),cherry blossoms,park'},
    22: {'label': '巨大月亮',
         'prompt': 'full moon, sky, planet, space, starry sky'},
    23: {'label': '废弃游乐场',
         'prompt': 'post-apocalypse, decayed amusement park, broken rides, faded colors,absurdress,'},
    24: {'label': '夏威夷灯塔',
         'prompt': 'hawaii,lighthouse, see,beach,beacon'},
    25: {'label': '科幻世界',
         'prompt': 'cinematic SCI-FI environment,towering factories ,dystopian world,metallic architecture,cyber sci-fi'},
    26: {'label': '赛博酒吧',
         'prompt': 'cinematic SCI-FI environment,Robotic bartenders serve patrons in a bar with levitating seats,cyber sci-fi'},
    27: {'label': '飞船驾驶仓',
         'prompt': 'cinematic SCI-FI environment,swirling nebulae,starship interior,cyber sci-fi'},
    28: {'label': '沙漠遗迹',
         'prompt': 'cinematic SCI-FI environment, Ruins of an ancient alien civilization,desert,carved relics, cyber sci-fi'},
    29: {'label': '未来工厂',
         'prompt': 'cinematic SCI-FI environment, AI-operated factory,robotic ,artificial constellations,floating energy orbs.cyber sci-fi'},
    30: {'label': '篮球馆',
         'prompt': 'stage, scenery, indoors, wooden floor, basketball uniform,aesthetic,stadium'},
    31: {'label': '排球馆',
         'prompt': 'stage, scenery, indoors, wooden floor, volleyball uniform,aesthetic,stadium'},
    32: {'label': '国风山中古城',
         'prompt': 'ancient chinese style landscape painting, mountains in front, beautiful winding green river behind'},
    33: {'label': '多人办公室',
         'prompt': 'depth_of_field,wide_shot,id_card,strap,business office,standing,window, desk,chair,computer, ceiling, ceiling light,'},
    34: {'label': '林中瀑布',
         'prompt': 'diffused natural sunlight, park, woods, flowers, birds, waterfall'},
    35: {'label': '富士山下',
         'prompt': 'very beautiful landscape, Mt Fuji, (cherry blossoms:0.4)'}
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
            positive_prompt = f'{prompt_gender[_gender]},{prompt_distance[_distance]["prompt"]}{("," + prompt_style[_style]["prompt"]) if _style != 0 else ""},{prompt_character[_character]["prompt"]},{prompt_costume[_costume]["prompt"]},{prompt_scene[_scene]["prompt"]},(Realism),(photorealistic),realistic,(best quality),(high quality),high details,masterpiece,extremely detailed,(sharp focus),(cinematic lighting),high saturation,ultra detailed,detailed background,wide view,sharp and crisp background,epic composition,intricate,solo'
            negative_prompt = f'(NSFW:1.3),(worst quality:2), (low quality:2), (normal quality:2),bad anatomy, DeepNegative,text, error,cropped,mutation,jpeg artifacts,polar lowres, bad proportions, gross proportions,deformed body,cross-eyed,sketches,bad hands,blurry,bad feet,poorly drawn hands,extra fingers, fewer digits, extra limbs, extra arms,extra legs, malformed limbs,(fused fingers:1.5),(too many fingers:1.5), long neck,mutated hands, polar lowres, bad body,(missing fingers:1.5), missing arms, missing legs, extra digit,extra foot,'

            print("-------------------factory logger-----------------")
            print(f"sd_positive_prompt: {positive_prompt}")
            print(f"sd_negative_prompt: {negative_prompt}")
            # self.operator.logging(
            #     f"[_reference_image_path][{_reference_image_path}]:\n",
            #     f"logs/sd_webui.log")
            res = self.operator.faceid_predictor(np.array(_input_image), positive_prompt, negative_prompt, _batch_size,
                                                 prompt_distance[_distance]['width'],
                                                 prompt_distance[_distance]['height'])

            return res
