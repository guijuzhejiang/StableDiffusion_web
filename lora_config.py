import shutil

lora_model_common_dict = [
    {'lora_name': 'polyhedron_new_skin_v1.1', 'weight': 0.1, 'label': '赋予真实皮肤，带褶皱'},
    {'lora_name': 'add_detail', 'weight': 1, 'label': '增加细节'},
    {'lora_name': 'more_details', 'weight': 1.5, 'label': '增加细节'},
    # {'lora_name': 'Xian-T手部修复lora（不用controlnet也不坏手了）_v3.0', 'weight': 0.6, 'label': '手部修复'},
]

lora_gender_dict = [
    # 女美化眼睛
    '<lora:detailed_eye-10:0.1>,1girl',
    # 男美化眼睛
    '<lora:polyhedron_men_eyes:0.1>,1man,1boy'
]

# gender 男1女0
lora_model_dict = {
    0: {'lora_name': 'shojovibe_v11',
        'weight': 0.4,
        'prompt': '',
        'gender': 0,
        'label': '亚洲少女'
        },

    1: {'lora_name': 'abd',
        'weight': 1,
        'prompt': 'beautiful abd_woman,abd_body,perfect abd_face',
        'gender': 0,
        'label': '非洲'
        },

    2: {'lora_name': 'dollface',
        'weight': 1,
        'prompt': 'irish',
        'gender': 0,
        'label': '爱尔兰人',
        },

    3: {'lora_name': 'dollface',
        'weight': 1,
        'prompt': 'spanish',
        'gender': 0,
        'label': '西班牙人'
        },

    4: {'lora_name': 'edgAustralianDoll',
        'weight': 0.7,
        'prompt': 'beautiful blonde edgAus_woman',
        'gender': 0,
        'label': '澳大利亚',
        },

    5: {'lora_name': 'edgBulgarian_Doll_Likeness',
        'weight': 1,
        'prompt': 'edgBulgr_woman,edgBulgr_face,edgBulgr_body',
        'gender': 0,
        'label': '保加利亚女性',
        },

    6: {'lora_name': 'edgEgyptian_Doll',
        'weight': 0.7,
        'prompt': 'beautiful dark haired edgEgyptian_woman,perfect edgEgyptian_face,perfect edgEgyptian_body',
        'gender': 0,
        'label': '埃及',
        },

    7: {'lora_name': 'edgIndonesianDollLikeness',
        'weight': 1,
        'prompt': 'edgIndo_woman,edgIndo_face,edgIndo_body',
        'gender': 0,
        'label': '印尼',
        },

    8: {'lora_name': 'edg_LatinaDollLikeness',
        'weight': 0.8,
        'prompt': 'beautiful Lnd_woman,perfect Lnd_face,perfect Lnd_body',
        'gender': 0,
        'label': '拉丁美洲',
        },

    9: {'lora_name': 'edgPersian',
        'weight': 1,
        'prompt': 'beautiful edgPersian_woman,perfect edgPersian_face,perfect edgPersian_body',
        'gender': 0,
        'label': '波斯',
        },

    10: {'lora_name': 'edgSwedishDoll',
         'weight': 0.7,
         'prompt': 'beautiful edgSwedish_woman,perfect edgSwedish_face,perfect edgSwedish_body',
         'gender': 0,
         'label': '瑞典'
         },

    11: {'lora_name': 'EnglishDollLikeness_v10',
         'weight': 1,
         'prompt': '',
         'gender': 0,
         'label': '英国'
         },

    12: {'lora_name': 'esd',
         'weight': 1,
         'prompt': 'beautiful esd_woman,perfect esd_face,perfect esd_body',
         'gender': 0,
         'label': '西班牙'
         },

    13: {'lora_name': 'frd',
         'weight': 0.8,
         'prompt': 'beautiful frd_woman,perfect frd_face,perfect frd_body',
         'gender': 0,
         'label': '法国'
         },

    14: {'lora_name': 'BLONDBOY',
         'weight': 1,
         'prompt': 'blondboy',
         'gender': 1,
         'label': '金发男'
         },

    15: {'lora_name': 'grd',
         'weight': 1,
         'prompt': 'beautiful grd_woman,perfect grd_face,perfect grd_body',
         'gender': 0,
         'label': '德国'
         },

    16: {'lora_name': 'hld',
         'weight': 1,
         'prompt': 'beautiful hld_woman,perfect hld_face,perfect hld_body',
         'gender': 0,
         'label': '苏格兰高地'
         },

    17: {'lora_name': 'ind',
         'weight': 1,
         'prompt': 'beautiful ind_woman,perfect ind_face,perfect ind_body',
         'gender': 0,
         'label': '印度'
         },

    18: {'lora_name': 'IndonesianDollLikeness_V1',
         'weight': 0.7,
         'prompt': '',
         'gender': 0,
         'label': '印尼'
         },

    19: {'lora_name': 'ird',
         'weight': 0.8,
         'prompt': 'beautiful ird_woman,perfect ird_face,perfect ird_body',
         'gender': 0,
         'label': '爱尔兰'
         },

    20: {'lora_name': 'itd1',
         'weight': 1,
         'prompt': 'beautiful itd_woman,perfect itd_face,perfect itd_body',
         'gender': 0,
         'label': '意大利'
         },

    21: {'lora_name': 'koreanDollLikeness',
         'weight': 0.7,
         'prompt': '',
         'gender': 0,
         'label': '韩国'
         },

    22: {'lora_name': 'Korean Men Dolllikeness 1.0',
         'weight': 1,
         'prompt': '',
         'gender': 1,
         'label': '韩国'
         },

    23: {'lora_name': 'Lora-Custom-ModelLiXian',
         'weight': 0.5,
         'prompt': '',
         'gender': 1,
         'label': '亚洲'
         },

    24: {'lora_name': 'm3d',
         'weight': 1,
         'prompt': 'beautiful m3d_woman,perfect m3d_body,perfact m3d_face',
         'gender': 0,
         'label': '中东'
         },

    25: {'lora_name': 'nod',
         'weight': 0.7,
         'prompt': 'beautiful nod_woman,perfect nod_body,perfect nod_face',
         'gender': 0,
         'label': '挪威'
         },

    26: {'lora_name': 'PrettyBoy',
         'weight': 1,
         'prompt': 'pretty boy,caucasian,black,asian,indian',
         'gender': 1,
         'label': '白种人、黑人、亚洲人、印度人'
         },

    27: {'lora_name': 'rud',
         'weight': 1,
         'prompt': 'beautiful rud_woman,perfect rud_face,perfect rud_body',
         'gender': 0,
         'label': '俄罗斯'
         },

    28: {'lora_name': 'RussianDollV3',
         'weight': 0.8,
         'prompt': 'russian',
         'gender': 0,
         'label': '俄罗斯'
         },

    29: {'lora_name': 'syahasianV3',
         'weight': 1,
         'prompt': 'syahmi',
         'gender': 1,
         'label': '亚洲，东南亚'
         },

    30: {'lora_name': 'tkd',
         'weight': 0.7,
         'prompt': 'beautiful tkd_woman,perfect tkd_face,perfect tkd_body',
         'gender': 0,
         'label': '土耳其'
         },

    31: {'lora_name': 'VietnameseDollLikeness-v1.0',
         'weight': 0.7,
         'prompt': '',
         'gender': 0,
         'label': '越南'
         },
}

lora_place_dict = {
    0: {'label': '单色',
        'prompt': '(simple background:1.3),(white background:1.3)'
        },

    1: {'label': '公路风光',
        'prompt': '(scenicroad:1.3),<lora:scenicroad:1.0>,landscape,a road with many trees on both sides,Utah,Florida,California,New England,Colorado,Arizona,Texas,Oregon,Pennsylvania,Washington,outdoor,nice bokeh professional nature photography,calm atmosphere'
        },

    2: {'label': '花团锦簇',
        'prompt': '<lora:乐章五部曲-林V1:1>,blue sky,outdoor,tree,nice bokeh professional nature photography,Cute landscape,calm atmosphere,peaceful theme,sen,nature,flowers',
        },

    3: {'label': '樱花绽放',
        'prompt': 'CherryBlossom_background,<lora:CherryBlossom_v1:0.6>,cherry blossoms in bloom,outdoor,nice bokeh professional nature photography,calm atmosphere,landscape,peaceful theme',
        },

    4: {'label': '光晕',
        'prompt': 'glowingdust,bokeh,<lora:glowingdust:0.9>,outdoor,nice bokeh professional nature photography,calm atmosphere,landscape,peaceful theme',
        },

    5: {'label': '街景',
        'prompt': 'haruhizaka,kitakoumae,scenery,<lora:kitakoukou:1>,outdoor,nice bokeh professional nature photography,calm atmosphere,street,landscape,road,power lines,city,tree,building,sign,cityscape',
        },

    6: {'label': '公园',
        'prompt': 'Park_Bench_background,<lora:ParkBench_v1:0.6>,park,nice bokeh professional nature photography,calm atmosphere,landscape,peaceful theme',
        },

    7: {'label': '天台',
        'prompt': '<lora:school_rooftop_v0.1:1> school rooftop,(rooftop:1.3),nice bokeh professional nature photography,calm atmosphere,chain-link fence,building',
        },

    8: {'label': '草坪',
        'prompt': '<lora:slg_v30:1>,slg,grass,green lawn',
        },

    9: {'label': '林间小路',
        'prompt': 'slg,(forest),path,<lora:slg_v30:1>,(path in woods:1.3),outdoor,nice bokeh professional nature photography,calm atmosphere',
        },

    10: {'label': '林间溪流',
         'prompt': 'slg,forest,(river:1.3),(stream),<lora:slg_v30:1>,outdoor,nice bokeh professional nature photography,calm atmosphere',
         },

    11: {'label': '林间瀑布',
         'prompt': 'slg,(waterfall:1.3),river,<lora:slg_v30:1>,huge waterfall,outdoor,nice bokeh professional nature photography,calm atmosphere',
         },

    12: {'label': '向日葵海',
         'prompt': 'sunflower_background,<lora:Sunflower_v1:0.7>,(sunflowers),outdoor,nice bokeh professional nature photography,calm atmosphere,landscape,peaceful theme',
         },

    13: {'label': '黄昏',
         'prompt': 'sunset_scenery_background,<lora:SunsetScenery_v1:0.7>,sunset,outdoor,nice bokeh professional nature photography,calm atmosphere,landscape,peaceful theme',
         },

    14: {'label': '沙滩',
         'prompt': 'beach,<lora:Taketomijima:1>,outdoor,nice bokeh professional nature photography,calm atmosphere,landscape,peaceful theme',
         },
    15: {'label': '夏威夷热',
         'prompt': 'tropical_tiki_retreat,<lora:tropical_tiki_retreat-10:1>,outdoor,nice bokeh professional nature photography,calm atmosphere,landscape,peaceful theme,Exotic,Hawaiian,aloha',
         },
    16: {'label': '秋天的童话',
         'prompt': 'rogowoarboretum,beautiful tree with red and yellow leaves,intricate detail,sunny weather,natural lighting,very sharp,<lora:hjrogowoarboretum_v10:0.8>',
         },
    17: {'label': '图书馆',
         'prompt': 'library,scenery,book,shelf,bookshelf,box,table,office,desk,NCT2,<lora:NishinomiyaChuouTosyokan:1>',
         },
    18: {'label': '小酒馆',
         'prompt': 'murayakuba,izakaya,scenery,indoors,lamp,table,<lora:Murayakuba:1>',
         },
    19: {'label': '泳池',
         'prompt': 'shs reinopool,reinopool,pool,indoors,tile floor,tile wall,<lora:reinopool_test2:1>',
         },
    20: {'label': '日式房间',
         'prompt': 'ryokan,scenery,table,indoors,television,window,chair,cup,ceiling light,lamp,flower pot,sunlight,<lora:ryokan:1>',
         },
}


if __name__ =='__main__':
    for k, v in lora_model_dict.items():
        shutil.copy(f'/media/zzg/GJ_disk01/pretrained_model/stable-diffusion-webui/models/Lora/{v["lora_name"]}.safetensors', f'/home/zzg/workspace/pycharm/StableDiffusion_web/models/Lora/{v["lora_name"]}.safetensors')

