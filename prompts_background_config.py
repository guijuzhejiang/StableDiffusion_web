import shutil


lora_common_dict = [
    # {'lora_name': 'add_detail','weight': 1,'label': '增加细节'},
    {'lora_name': 'more_details','weight': 1.5,'label': '增加细节'},
]
promts_common_pos = 'science_fiction,fantasy,incredible,(best quality:1.2),(high quality:1.2),high details,(Realism:1.4),masterpiece,extremely detailed,extremely delicate,ultra detailed,Amazing,8k wallpaper,8k uhd,(dramatic scene),(Epic composition:1.2),strong contrast,raw photo,no humans,huge_filesize,incredibly_absurdres,absurdres,highres,motion lines,magazine cover,video game cover,intense angle,dynamic angle,high saturation,poster,'
promts_common_neg = '(NSFW:1.8),(hands),(feet),(shoes),(mask),(glove),(fingers:1.3),(arms),(legs),(toes:1.3),(digits:1.3),(hair:1.3),(humans:1.3),(face:1.3),bad_picturesm,EasyNegative,easynegative,ng_deepnegative_v1_75t,verybadimagenegative_v1.3,(worst quality:2),(low quality:2),(normal quality:2),((monochrome)),((grayscale)),sketches,bad anatomy,DeepNegative,facing away,{Multiple people},text,error,cropped,blurry,mutation,deformed,jpeg artifacts,polar lowres,bad proportions,gross proportions'
lora_place_dict = {
    #巨大战舰+星系+detail
    0: {'label': '巨大战舰',
        'prompt': "<lora:neowrsk_v2:0.7>,<lora:[LoHa] Octans八分儀 Stylev2:1>,Megalophobia,giant phobia,cloud,low angle,chaosmix,chaos,horror,neowrsk,octans,flying spacecraft,floating in the sky,spaceship,cyberpunk aesthetics,electrical storm,plasma turret fire,interstellar warfare,tension,decaying space station backdrop,ominous,devoid of illumination,nebula-filled cosmos,(huge:1.5),from below,red theme,mysterious,ethereal,sharp focus,hot pinks,and glowing purples,(giant clothes),Dramatic Lighting,Bold Coloration,Vibrant Warmth,deep shadow,astonishing level of detail,Horizon composition,universe,Hal's Mobile Castle,Huge Sky Castle,explosion,fireworks"
        },
    #赛博朋克
    1: {'label': '赛博朋克',
        'prompt': "<lora:Cyberpunk sceneV1:0.7>,Megalophobia,giant phobia,cloud,low angle,chaosmix,chaos,horror,tooimage cyberpunk futuristic city,flying spacecraft,floating in the sky,spaceship,cyberpunk aesthetics,electrical storm,plasma turret fire,interstellar warfare,tension,decaying space station backdrop,ominous,devoid of illumination,nebula-filled cosmos,(huge:1.5),from below,red theme,mysterious,ethereal,sharp focus,hot pinks,and glowing purples,(giant clothes),Dramatic Lighting,Bold Coloration,Vibrant Warmth,deep shadow,astonishing level of detail,Horizon composition,universe,Hal's Mobile Castle,Huge Sky Castle",
        },
    #未来城市
    2: {'label': '未来城市',
        'prompt': "<lora:XSArchi_127:1>,<lora:Concept_scenery_background:0.3>,solo,(zenithal angle),sunset,((by Iwan Baan)),skyscraper,japan style,arasaka tower,neon lights,cyberpunk,cyberpunk \(series\),Steam power: Steampunk works usually depict a world based on steam power,people use steam engines,steam trains,steam airships,etc.,these machines are usually powered by steam engines.coastal city,blue sky and white clouds,the sun is shining brightly,ultra-wide angle,",
        },
    #红叶古楼，效果不好
    3: {'label': '红叶古楼',
        'prompt': "<lora:绪儿-和风赛博场景 indoor:1>,cherry blossoms,east asian architecture,petals,potted plant,reflection,red flower,water,chinese clothes,wide shot,autumn leaves,",
        },
    #月下古城
    4: {'label': '月下古城',
        'prompt': "<lora:Ancient_city:1>,BJ_Ancient_city,outdoors,sky,cloud,water,tree,moon,fire,building,scenery,full_moon,stairs,mountain,architecture,east_asian_architecture,cinematic lighting,morning red,abundant,wallpaper,huge bridges",
        },
    #霸王龙，效果不好
    5: {'label': '霸王龙',
        'prompt': "<lora:侏罗纪花园_v1.0:0.7>,horror,deep shadow,large Tyrannosaurus Rex,forest,trees,huge waterfall,river",
        },
    #星际大战
    6: {'label': '星际大战',
        'prompt': "<lora:末日-宇宙（场景）_v1.0:0.6>,horror,(A huge spaceship:1.5),(solo:1.5),(A rectangular spacecraft resembling the shape of an aircraft carrier:1.5),Full of art,Cosmic galaxy background,Doomsday scenario,Crumbling earth,a volcano erupts,energy blast,The fleeing spaceship,(Epic composition:1.2)",
        },
    #未来世界
    7: {'label': '未来世界',
        'prompt': "<lora:新科幻Neo Sci-Fi_v1.0:1>,sci-fi city,modern architecture style,river,a floating city in the sky,super high-rise building,high resolution,outdoor,(day:1.2),(blue sky:1.3),water,soft lighting,(dramatic scene),(Epic composition:1.2)",
        },
    #机甲怪兽，效果不好
    8: {'label': '机甲怪兽',
         'prompt': "<lora:机甲怪兽风格lora_v1.0:0.5>,monster,Alien monsters invade Earth,Dragon-shaped monster,huge,thriller,robot,Mecha,chilling,horrifying,terrifying",
         },
    #扎古机甲，效果不好
    9: {'label': '扎古机甲',
         'prompt': "<lora:Like a Zagu mech 类似扎古的机甲_v1.0:1>,BJ_Zagu_Mecha,solo,holding,weapon,holding_weapon,gun,robot,holding_gun,mecha,one-eyed,mobile_suit,zeon,BJ_Zagu_Mecha,cinematic lighting",
         },
    #巨大机甲，效果不好
    10: {'label': '巨大机甲',
         'prompt': "<lora:拾图-机甲shitu-mecha_shitu-mechaV1:0.6>,looking at viewer,(official art),illustration,beautiful abstract background,Futurism,cyberpunk,shitu-mecha,((Picture of the battle,Sparks,Smoke,Explosion,)),full body,mysterious,photon mapping,radiosity,physically-based rendering,delicate realism,large mechanical robot construction,cityscape,citylights,besttexture,neonlight,lighteffect,alienship,battleposture,handheld,weapons,lasergun",
         },
    #红色机甲，效果不好
    11: {'label': '红色机甲',
         'prompt': "<lora:拾图-机甲shitu-mecha_shitu-mechaV1:0.6>,sharp focus,professional lighting,colorful details,iridescent colors BREAK extreme long shot of a factory,((Redism,red style):1.5),large china mechanical robot construction,microchip,computer,glowing,(shitou-mecha luminescence:1.1),wide shot,panorama,sideways,negative space,mysterious,photon mapping,radiosity,physically-based rendering,night,(Ruins:1.1)",
         },
    #古风浮岛
    12: {'label': '古风浮岛',
         'prompt': "<lora:(LIb首发)CG古风大场景类_v2.0:1>,HD,cg,Chinese CG scene,unreal 5 engine,floating island,abg,large waterfall left in the middle,rain,huge peaks",
         },
    #古风满月
    13: {'label': '古风满月',
         'prompt': "<lora:(LIb首发)CG古风大场景类_v2.0:1>,HD,cg,Chinese CG scene,unreal 5 engine,Mid-autumn,full moon,night view,plants,ancient buildings,bridge,Backlight,Creek,Clouds,Chinese architecture,brightly lit",
         },
    #古风宝塔
    14: {'label': '古风宝塔',
         'prompt': "<lora:(LIb首发)CG古风大场景类_v2.0:1>,Unreal Engine 5,CG,abg,Chinese CG scene,top down,scenery,waterfall,cloud,tree,architecture,sky,outdoors,floating island,day,east asian architecture,mountain,water,bridge,pagoda,castle,building,blue sky,tower,fog,",
         },
    #古风松
    15: {'label': '古风松',
         'prompt': "<lora:绪儿-【古风松】主题风景_v1.0:0.9>,an painting in a style of oriental painting,in the style of matte painting,layered and atmospheric landscapes,rich and immersive,quiet contemplation,dark white and green,history painting,zen-inspired,grandeur of scale,cinematic,stunning,realistic lighting and shading,vivid,vibrant,octane render,unreal engine,concept art,Cry engine,wide shot,east asian architecture,building,pagoda,scenery,pine,tree,outdoors,large bridge,water,reflection,fog,boat,pond,moss",
         },
    #满月与海，容易inpaint人物成女性
    16: {'label': '满月与海',
         'prompt': "<lora:满月与大海_v0.1:1>,ambient lighting,professional artwork,Ambient Occlusion,surrealism,illusion,only sky,unreal,depth of field,focus to the sky,silver theme,lunarYW,sea",
         },
    #死神，效果不好
    17: {'label': '死神',
         'prompt': "<lora:Ghost Concept_v1.0:0.7>,visually stunning,elegant,g0s1,faceless,cloak,robe,torn clothes,torn fabric,floating,grim reaper,black reaper,<lora:f1nt - Fantasy:0.5>,f1nt,fantasy theme,horror \(theme\),scythe,holding scythe,death,ghost,hood,",
         },
}


if __name__ =='__main__':
    for k,v in lora_place_dict.items():
        shutil.copy(f'/media/zzg/GJ_disk01/pretrained_model/stable-diffusion-webui/models/Lora/{v["lora_name"]}.safetensors',f'/home/zzg/workspace/pycharm/StableDiffusion_web/models/Lora/{v["lora_name"]}.safetensors')