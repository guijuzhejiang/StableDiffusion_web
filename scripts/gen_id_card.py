# -*- encoding: utf-8 -*-
'''
@File    :   gen_id_card.py    
@Modify Time      @Author    @Version    @Desciption
------------      -------    --------    -----------
2022/4/13 上午10:55   ray      1.0         None
'''

# import lib
import glob
import json
import os
import random
import re
import sys

import PIL.Image as PImage
from PIL import ImageFont, ImageDraw
import cv2
import numpy as np
from tqdm import tqdm

try:
    from Tkinter import *
    from ttk import *
    from tkFileDialog import *
    from tkMessageBox import *
except ImportError:
    from tkinter import *
    from tkinter.ttk import *
    from tkinter.filedialog import *
    from tkinter.messagebox import *

if getattr(sys, 'frozen', None):
    base_dir = os.path.join(sys._MEIPASS, 'usedres')
else:
    base_dir = os.path.join(os.path.dirname(__file__), 'usedres')

char_dict = []
with open('/home/ray/Workspace/project/ocr/src/ch_gso_server/lib/ocr/ppocr/utils/dict/ppocr_keys_20220401.txt',
          'r') as f:
    c = f.readlines()
    for i in c:
        char_dict.append(i.replace('\n', ''))


def changeBackground(img, img_back, zoom_size, center):
    # 缩放
    img = cv2.resize(img, zoom_size)
    rows, cols, channels = img.shape

    # 转换hsv
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    # 获取mask
    # lower_blue = np.array([78, 43, 46])
    # upper_blue = np.array([110, 255, 255])
    diff = [5, 30, 30]
    gb = hsv[0, 0]
    lower_blue = np.array(gb - diff)
    upper_blue = np.array(gb + diff)
    mask = cv2.inRange(hsv, lower_blue, upper_blue)
    # cv2.imshow('Mask', mask)

    # 腐蚀膨胀
    erode = cv2.erode(mask, None, iterations=1)
    dilate = cv2.dilate(erode, None, iterations=1)

    # 粘贴
    for i in range(rows):
        for j in range(cols):
            if dilate[i, j] == 0:  # 0代表黑色的点
                img_back[center[0] + i, center[1] + j] = img[i, j]  # 此处替换颜色，为BGR通道

    return img_back


# 删减部分小众姓氏
# firstName = "赵钱孙李周吴郑王冯陈褚卫蒋沈韩杨朱秦尤许何吕施张孔曹严华金魏陶姜戚谢邹喻水云苏潘葛奚范彭郎鲁韦昌马苗凤花方俞任袁柳鲍史唐费岑薛雷贺倪汤滕殷罗毕郝邬安常乐于时傅卞齐康伍余元卜顾孟平" \
#             "黄和穆萧尹姚邵湛汪祁毛禹狄米贝明臧计成戴宋茅庞熊纪舒屈项祝董粱杜阮席季麻强贾路娄危江童颜郭梅盛林刁钟徐邱骆高夏蔡田胡凌霍万柯卢莫房缪干解应宗丁宣邓郁单杭洪包诸左石崔吉" \
#             "龚程邢滑裴陆荣翁荀羊甄家封芮储靳邴松井富乌焦巴弓牧隗山谷车侯伊宁仇祖武符刘景詹束龙叶幸司韶黎乔苍双闻莘劳逄姬冉宰桂牛寿通边燕冀尚农温庄晏瞿茹习鱼容向古戈终居衡步都耿满弘国文东殴沃曾关红游盖益桓公晋楚闫"
# 百家姓姓氏
# 
firstName = "赵钱孙李周吴郑王冯陈褚卫蒋沈韩杨朱秦尤许何吕施张孔曹严华金魏陶姜戚谢邹喻柏水窦章云苏潘葛奚范彭郎鲁韦昌马苗凤花方俞任袁柳酆鲍史唐费廉岑薛雷贺倪汤滕殷罗毕郝邬安常乐于时傅皮卞齐康伍余元卜顾孟平" \
            "黄和穆萧尹姚邵湛汪祁毛禹狄米贝明臧计伏成戴谈宋茅庞熊纪舒屈项祝董粱杜阮蓝闵席季麻强贾路娄危江童颜郭梅盛林刁钟徐邱骆高夏蔡田樊胡凌霍虞万支柯昝管卢莫经房裘缪干解应宗丁宣贲邓郁单杭洪包诸左石崔吉钮" \
            "龚程嵇邢滑裴陆荣翁荀羊於惠甄麴家封芮羿储靳汲邴糜松井段富巫乌焦巴弓牧隗山谷车侯宓蓬全郗班仰秋仲伊宫宁仇栾暴甘钭厉戎祖武符刘景詹束龙叶幸司韶郜黎蓟薄印宿白怀蒲邰从鄂索咸籍赖卓蔺屠蒙池乔阴欎胥能苍" \
            "双闻莘党翟谭贡劳逄姬申扶堵冉宰郦雍舄璩桑桂濮牛寿通边扈燕冀郏浦尚农温别庄晏柴瞿阎充慕连茹习宦艾鱼容向古易慎戈廖庾终暨居衡步都耿满弘匡国文寇广禄阙东殴殳沃利蔚越夔隆师巩厍聂晁勾敖融冷訾辛阚那简饶空" \
            "曾毋沙乜养鞠须丰巢关蒯相查後荆红游竺权逯盖益桓公晋楚闫法汝鄢涂钦归海帅缑亢况后有琴梁丘左丘商牟佘佴伯赏南宫墨哈谯笪年爱阳佟言福百家姓终"
# 百家姓中双姓氏
firstName2 = "万俟司马上官欧阳夏侯诸葛闻人东方赫连皇甫尉迟公羊澹台公冶宗政濮阳淳于单于太叔申屠公孙仲孙轩辕令狐钟离宇文长孙慕容鲜于闾丘司徒司空亓官司寇仉督子颛孙端木巫马公西漆雕乐正壤驷公良拓跋夹谷宰父谷梁段干百里东郭南门呼延羊舌微生梁丘左丘东门西门南宫南宫"
# 女孩名字
girl = '秀娟英华慧巧美娜静淑惠珠翠雅芝玉萍红娥玲芬芳燕彩春菊兰凤洁梅琳素云莲真环雪荣爱妹霞香月莺媛艳瑞凡佳嘉琼勤珍贞莉桂娣叶璧璐娅琦晶妍茜秋珊莎锦黛青倩婷姣婉娴瑾颖露瑶怡婵雁蓓纨仪荷丹蓉眉君琴蕊薇菁梦岚苑婕馨瑗琰韵融园艺咏卿聪澜纯毓悦昭冰爽琬茗羽希宁欣飘育滢馥筠柔竹霭凝晓欢霄枫芸菲寒伊亚宜可姬舒影荔枝思丽'
# 男孩名字
boy = '伟刚勇毅俊峰强军平保东文辉力明永健世广志义兴良海山仁波宁贵福生龙元全国胜学祥才发武新利清飞彬富顺信子杰涛昌成康星光天达安岩中茂进林有坚和彪博诚先敬震振壮会思群豪心邦承乐绍功松善厚庆磊民友裕河哲江超浩亮政谦亨奇固之轮翰朗伯宏言若鸣朋斌梁栋维启克伦翔旭鹏泽晨辰士以建家致树炎德行时泰盛雄琛钧冠策腾楠榕风航弘'
# 名
name = '中笑贝凯歌易仁器义礼智信友上都卡被好无九加电金马钰玉忠孝'

firstName = [x for x in firstName if x in char_dict]
firstName2 = [x for x in firstName2 if x in char_dict]
girl = [x for x in girl if x in char_dict]
boy = [x for x in boy if x in char_dict]
name = [x for x in name if x in char_dict]


def random_name():
    # 10%的机遇生成双数姓氏
    if random.choice(range(100)) > 10:
        firstName_name = random.choice(firstName)
    else:
        i = random.choice(range(len(firstName2)))
        firstName_name = ''.join(firstName2[i:i + 2])

    sex = random.choice(range(2))
    name_1 = ""
    # 生成并返回一个名字
    if sex > 0:
        girl_name = girl[random.choice(range(len(girl)))]
        if random.choice(range(2)) > 0:
            name_1 = name[random.choice(range(len(name)))]
        return firstName_name + name_1 + girl_name
    else:
        boy_name = boy[random.choice(range(len(boy)))]
        if random.choice(range(2)) > 0:
            name_1 = name[random.choice(range(len(name)))]
        return firstName_name + name_1 + boy_name


def random_nation():
    # \t仡佬
    nation_str = '蒙古\t回\t苗\t傣\t傈僳\t藏\t壮\t朝鲜\t高山\t纳西\t布朗\t阿昌\t怒\t鄂温克\t鄂伦春\t赫哲\t门巴\t白\t保安\t布依\t达斡尔\t德昂\t东乡\t侗\t独龙\t俄罗斯族 、哈尼\t哈萨克\t基诺\t京\t景颇\t柯尔克孜\t拉祜\t黎\t珞巴\t满\t毛南\t仫佬\t普米\t羌\t撒拉\t畲\t水\t塔吉克\t塔塔尔\t土家\t土\t佤\t维吾尔\t乌孜别克\t锡伯\t瑶\t裕固\t彝\t汉\t入籍'
    nation_list = nation_str.split('\t')
    return random.choice(nation_list)


def random_num(length):
    return ''.join([str(random.randint(0, 9)) for i in range(length)])


province = []
city = []
area = []
for i in glob.glob('/home/ray/Workspace/project/ocr/src/ch_gso_server/scripts/area_id/*.txt'):
    province_str = re.sub(re.compile(r'[^\u4e00-\u9fa5]'), "", os.path.basename(i))
    if province_str != '北京' or province_str != '上海' or province_str != '港澳' or province_str != '海外':
        province_str += '省'
    province.append(province_str)
    area_dict = json.loads(open(i, 'r').read().replace("'", '"'))
    for p, v1 in area_dict.items():
        for c, v2 in v1.items():
            if isinstance(v2, str):
                continue
            else:
                city.append(re.sub(re.compile(r'[^\u4e00-\u9fa5]'), "", c))
                for a, v3 in v2.items():
                    area.append(re.sub(re.compile(r'[^\u4e00-\u9fa5]'), "", a))


def random_addr():
    # province = ["河北省", "山西省", "辽宁省", "吉林省", "黑龙江省", "江苏省", "浙江省", "安徽省", "福建省", "江西省", "山东省", "河南省", "湖北省", "湖南省",
    #             "广东省", "海南省", "四川省", "贵州省", "云南省", "陕西省", "甘肃省", "青海省", "台湾省"]
    # city = ["安康市", "安庆市", "安顺市", "安阳市", "鞍山市", "巴彦淖尔市", "巴中市", "白城市", "白山市", "白银市", "百色市", "蚌埠市", "包头市", "宝鸡市", "保定市",
    #         "保山市", "北海市", "本溪市", "滨州市", "沧州市", "昌都地区", "长春市", "长沙市", "长治市", "常德市", "常州市", "巢湖市", "朝阳市", "潮州市", "郴州市",
    #         "成都市", "承德市", "池州市", "赤峰市", "崇左市", "滁州市", "达州市", "大连市", "大庆市", "大同市", "丹东市", "德阳市", "德州市", "定西市", "东莞市",
    #         "东营市", "鄂尔多斯市", "鄂州市", "防城港市", "佛山市", "福州市", "抚顺市", "抚州市", "阜新市", "阜阳市", "甘南州", "赣州市", "固原市", "广安市", "广元市",
    #         "广州市", "贵港市", "贵阳市", "桂林市", "哈尔滨市", "哈密地区", "海北藏族自治州", "海东地区", "海口市", "邯郸市", "汉中市", "杭州市", "毫州市", "合肥市",
    #         "河池市", "河源市", "菏泽市", "贺州市", "鹤壁市", "鹤岗市", "黑河市", "衡水市", "衡阳市", "呼和浩特市", "呼伦贝尔市", "湖州市", "葫芦岛市", "怀化市",
    #         "淮安市", "淮北市", "淮南市", "黄冈市", "黄山市", "黄石市", "惠州市", "鸡西市", "吉安市", "吉林市", "济南市", "济宁市", "佳木斯市", "嘉兴市", "嘉峪关市",
    #         "江门市", "焦作市", "揭阳市", "金昌市", "金华市", "锦州市", "晋城市", "晋中市", "荆门市", "荆州市", "景德镇市", "九江市", "酒泉市", "开封市", "克拉玛依市",
    #         "昆明市", "拉萨市", "来宾市", "莱芜市", "兰州市", "廊坊市", "乐山市", "丽江市", "丽水市", "连云港市", "辽阳市", "辽源市", "聊城市", "临沧市", "临汾市",
    #         "临沂市", "柳州市", "六安市", "六盘水市", "龙岩市", "陇南市", "娄底市", "泸州市", "吕梁市", "洛阳市", "漯河市", "马鞍山市", "茂名市", "眉山市", "梅州市",
    #         "绵阳市", "牡丹江市", "内江市", "南昌市", "南充市", "南京市", "南宁市", "南平市", "南通市", "南阳市", "宁波市", "宁德市", "攀枝花市", "盘锦市", "平顶山市",
    #         "平凉市", "萍乡市", "莆田市", "濮阳市", "普洱市", "七台河市", "齐齐哈尔市", "钦州市", "秦皇岛市", "青岛市", "清远市", "庆阳市", "曲靖市", "衢州市", "泉州市",
    #         "日照市", "三门峡市", "三明市", "三亚市", "汕头市", "汕尾市", "商洛市", "商丘市", "上饶市", "韶关市", "邵阳市", "绍兴市", "深圳市", "沈阳市", "十堰市",
    #         "石家庄市", "石嘴山市", "双鸭山市", "朔州市", "四平市", "松原市", "苏州市", "宿迁市", "宿州市", "绥化市", "随州市", "遂宁市", "台州市", "太原市", "泰安市",
    #         "泰州市", "唐山市", "天水市", "铁岭市", "通化市", "通辽市", "铜川市", "铜陵市", "铜仁市", "吐鲁番地区", "威海市", "潍坊市", "渭南市", "温州市", "乌海市",
    #         "乌兰察布市", "乌鲁木齐市", "无锡市", "吴忠市", "芜湖市", "梧州市", "武汉市", "武威市", "西安市", "西宁市", "锡林郭勒盟", "厦门市", "咸宁市", "咸阳市",
    #         "湘潭市", "襄樊市", "孝感市", "忻州市", "新乡市", "新余市", "信阳市", "兴安盟", "邢台市", "徐州市", "许昌市", "宣城市", "雅安市", "烟台市", "延安市",
    #         "盐城市", "扬州市", "阳江市", "阳泉市", "伊春市", "伊犁哈萨克自治州", "宜宾市", "宜昌市", "宜春市", "益阳市", "银川市", "鹰潭市", "营口市", "永州市",
    #         "榆林市", "玉林市", "玉溪市", "岳阳市", "云浮市", "运城市", "枣庄市", "湛江市", "张家界市", "张家口市", "张掖市", "漳州市", "昭通市", "肇庆市", "镇江市",
    #         "郑州市", "中山市", "中卫市", "舟山市", "周口市", "株洲市", "珠海市", "驻马店市", "资阳市", "淄博市", "自贡市", "遵义市"]
    # area = ["伊春区", "带岭区", "南岔区", "金山屯区", "西林区", "美溪区", "乌马河区", "翠峦区", "友好区", "新青区", "上甘岭区", "五营区", "红星区", "汤旺河区",
    #         "乌伊岭区", "榆次区"]
    road = ["爱国路", "安边路", "安波路", "安德路", "安汾路", "安福路", "安国路", "安化路", "安澜路", "安龙路", "安仁路", "安顺路", "安亭路", "安图路", "安业路",
            "民主路", "民强路",
            "安义路", "安远路", "鞍山路", "鞍山支路", "澳门路", "八一路", "巴林路", "白城路", "白城南路", "白渡路", "白渡桥", "白兰路", "白水路", "白玉路", '博爱路',
            "中山一路", "中山二路",
            "百安路（方泰镇）", "百官街", "百花街", "百色路", "板泉路", "半淞园路", "包头路", "包头南路", "宝安公路", "宝安路", "宝昌路", "宝联路", "宝林路", "宝祁路",
            "宏基路",
            "宝山路", "宝通路", "宝杨路", "宝源路", "保德路", "保定路", "保屯路", "保屯路", "北艾路", "解放西路", "人民路"]
    home = ["金色家园", "耀江花园", "阳光翠竹苑", "东新大厦", "溢盈河畔别墅", "真新六街坊", "和亭佳苑", "协通公寓", "博泰新苑", "菊园五街坊", "住友嘉馨名园", "复华城市花园",
            "爱里舍花园", "岐乐花园", "金碧雅苑", "大信海岸家园", "江南名居"]
    return random.choice(province) + random.choice(city) + random.choice(area) + random.choice(road) + random.choice(
        home) + str(random.randint(1, 999)) + '号'


def random_org():
    return random.choice(area) + '公安局'


def random_period():
    return f'{str(random.randint(1900, 2030))}.%02d.%02d-{str(random.randint(1900, 2030))}.%02d.%02d' % (
        random.randint(1, 12), random.randint(1, 31), random.randint(1, 12), random.randint(1, 31))


def generator(trt_fn):
    name = random_name()
    sex = '男' if random.randint(0, 1) == 1 else '女'
    nation = random_nation()
    year = str(random.randint(1900, 2030))
    mon = str(random.randint(1, 12))
    day = str(random.randint(1, 31))
    # 签发机关
    org = random_org()
    # 有效期限
    life = random_period()
    addr = random_addr()
    idn = random_num(17) + random.choice([str(i) for i in range(10)] + ['X'])

    file_list = glob.glob('/home/ray/Documents/faces/*')
    fname = random.choice(file_list)
    # print fname
    im = PImage.open(os.path.join(base_dir, 'empty.png'))
    avatar = PImage.open(fname)  # 500x670

    name_font = ImageFont.truetype(os.path.join(base_dir, 'hei.ttf'), 72)
    other_font = ImageFont.truetype(os.path.join(base_dir, 'hei.ttf'), 60)
    bdate_font = ImageFont.truetype(os.path.join(base_dir, 'fzhei.ttf'), 60)
    id_font = ImageFont.truetype(os.path.join(base_dir, 'ocrb10bt.ttf'), 72)

    draw = ImageDraw.Draw(im)
    draw.text((630, 690), name, fill=(0, 0, 0), font=name_font)
    draw.text((630, 840), sex, fill=(0, 0, 0), font=other_font)
    draw.text((1030, 840), nation, fill=(0, 0, 0), font=other_font)
    draw.text((630, 980), year, fill=(0, 0, 0), font=bdate_font)
    draw.text((950, 980), mon, fill=(0, 0, 0), font=bdate_font)
    draw.text((1150, 980), day, fill=(0, 0, 0), font=bdate_font)
    start = 0
    loc = 1120
    while start + 11 < len(addr):
        draw.text((630, loc), addr[start:start + 11], fill=(0, 0, 0), font=other_font)
        start += 11
        loc += 100
    draw.text((630, loc), addr[start:], fill=(0, 0, 0), font=other_font)
    draw.text((950, 1475), idn, fill=(0, 0, 0), font=id_font)
    draw.text((1050, 2750), org, fill=(0, 0, 0), font=other_font)
    draw.text((1050, 2895), life, fill=(0, 0, 0), font=other_font)

    if True:
        avatar = cv2.cvtColor(np.asarray(avatar), cv2.COLOR_RGBA2BGRA)
        im = cv2.cvtColor(np.asarray(im), cv2.COLOR_RGBA2BGRA)
        im = changeBackground(avatar, im, (500, 670), (690, 1500))
        # im = PImage.fromarray(cv2.cvtColor(im, cv2.COLOR_BGRA2RGBA))
        # cv2.imwrite(trt_fn + '_front.jpg', targetImg1, [cv2.IMWRITE_JPEG_QUALITY, 80])
    # else:
    #     avatar = avatar.resize((500, 670))
    #     avatar = avatar.convert('RGBA')
    #     im.paste(avatar, (1500, 690), mask=avatar)
    #     # im = paste(avatar, im, (500, 670), (690, 1500))

    # h、w为想要截取的图片大小
    h, w = 1190, 1890
    # 坐标
    x, y = 287, 490
    x2, y2 = 280, 1902

    # im.save('color.png')
    # im.convert('L').save('bw.png')

    # img = cv2.imread('color.png')
    targetImg1 = im[(y):(y + h), (x):(x + w)]
    targetImg2 = im[(y2):(y2 + h), (x2):(x2 + w)]

    resize_tuple = (int(targetImg1.shape[1] * 0.65), int(targetImg1.shape[0] * 0.65))
    targetImg1 = cv2.resize(targetImg1, resize_tuple)
    targetImg2 = cv2.resize(targetImg2, resize_tuple)

    cv2.imwrite(trt_fn + '_front.jpg', targetImg1, [cv2.IMWRITE_JPEG_QUALITY, 80])
    cv2.imwrite(trt_fn + '_back.jpg', targetImg2, [cv2.IMWRITE_JPEG_QUALITY, 80])

    # im.save('color.png')
    # im.convert('L').save('bw.png')
    trt_dn = os.path.basename(os.path.dirname(trt_fn))

    # draw.text((630, 690), name, fill=(0, 0, 0), font=name_font)
    # draw.text((630, 840), sex, fill=(0, 0, 0), font=other_font)
    # draw.text((1030, 840), nation, fill=(0, 0, 0), font=other_font)
    # draw.text((630, 980), year, fill=(0, 0, 0), font=bdate_font)
    # draw.text((950, 980), mon, fill=(0, 0, 0), font=bdate_font)
    # draw.text((1150, 980), day, fill=(0, 0, 0), font=bdate_font)
    # start = 0
    # loc = 1120
    # while start + 11 < len(addr):
    #     draw.text((630, loc), addr[start:start + 11], fill=(0, 0, 0), font=other_font)
    #     start += 11
    #     loc += 100
    # draw.text((630, loc), addr[start:], fill=(0, 0, 0), font=other_font)
    # draw.text((950, 1475), idn, fill=(0, 0, 0), font=id_font)
    # draw.text((1050, 2750), org, fill=(0, 0, 0), font=other_font)
    # draw.text((1050, 2895), life, fill=(0, 0, 0), font=other_font)
    front_buf = []
    back_buf = []
    front_buf.append({"transcription": "姓名" + name,
                      "points": [[int((425 - x) * 0.65), int((690 - y) * 0.65)],
                                 [int(((630 - x) + name_font.size * len(name)) * 0.65), int((690 - y) * 0.65)],
                                 [int(((630 - x) + name_font.size * len(name)) * 0.65),
                                  int(((690 - y) + name_font.size) * 0.65)],
                                 [int((425 - x) * 0.65), int(((690 - y) + name_font.size) * 0.65)]],
                      "difficult": False})
    front_buf.append({"transcription": "性别" + sex,
                      "points": [[int((425 - x) * 0.65), int((840 - y) * 0.65)],
                                 [int(((630 - x) + other_font.size) * 0.65), int((840 - y) * 0.65)],
                                 [int(((630 - x) + other_font.size) * 0.65),
                                  int(((840 - y) + other_font.size) * 0.65)],
                                 [int((425 - x) * 0.65), int(((840 - y) + other_font.size) * 0.65)]],
                      "difficult": False})
    front_buf.append({"transcription": "民族" + nation,
                      "points": [[int((850 - x) * 0.65), int((840 - y) * 0.65)],
                                 [int(((1030 - x) + other_font.size * len(nation)) * 0.65), int((840 - y) * 0.65)],
                                 [int(((1030 - x) + other_font.size * len(nation)) * 0.65),
                                  int(((840 - y) + other_font.size) * 0.65)],
                                 [int((850 - x) * 0.65), int(((840 - y) + other_font.size) * 0.65)]],
                      "difficult": False})
    front_buf.append({"transcription": "出生" + year + "年" + mon + "月" + day + "日",
                      "points": [[int((425 - x) * 0.65), int((980 - y) * 0.65)],
                                 [int((1290 - x) * 0.65), int((980 - y) * 0.65)],
                                 [int((1290 - x) * 0.65), int(((980 - y) + bdate_font.size) * 0.65)],
                                 [int((425 - x) * 0.65), int(((980 - y) + bdate_font.size) * 0.65)]],
                      "difficult": False})

    if len(addr) <= 11:
        front_buf.append({"transcription": "住址" + addr,
                          "points": [[int((425 - x) * 0.65), int((1120 - y) * 0.65)],
                                     [int(((630 - x) + other_font.size * len(addr)) * 0.65), int((840 - y) * 0.65)],
                                     [int(((630 - x) + other_font.size * len(addr)) * 0.65),
                                      int(((840 - y) + other_font.size) * 0.65)],
                                     [int((425 - x) * 0.65), int(((1120 - y) + other_font.size) * 0.65)]],
                          "difficult": False})
    else:
        start = 0
        loc = 1120
        while start + 11 < len(addr):
            # draw.text((630, loc), addr[start:start + 11], fill=(0, 0, 0), font=other_font)
            if start == 0:
                front_buf.append({"transcription": "住址" + addr[start:start + 11],
                                  "points": [[int((425 - x) * 0.65), int((loc - y) * 0.65)],
                                             [int(((630 - x) + other_font.size * len(addr[start:start + 11])) * 0.65),
                                              int((loc - y) * 0.65)],
                                             [int(((630 - x) + other_font.size * len(addr[start:start + 11])) * 0.65),
                                              int(((loc - y) + other_font.size) * 0.65)],
                                             [int((425 - x) * 0.65), int(((loc - y) + other_font.size) * 0.65)]],
                                  "difficult": False})
            else:
                front_buf.append({"transcription": addr[start:start + 11],
                                  "points": [[int((630 - x) * 0.65), int((loc - y) * 0.65)],
                                             [int(((630 - x) + other_font.size * len(addr[start:start + 11]) - 30 * len(
                                                 re.sub(re.compile(r'[^\d]'), "", addr[start:start + 11]))) * 0.65),
                                              int((loc - y) * 0.65)],
                                             [int(((630 - x) + other_font.size * len(addr[start:start + 11]) - 30 * len(
                                                 re.sub(re.compile(r'[^\d]'), "", addr[start:start + 11]))) * 0.65),
                                              int(((loc - y + 10) + other_font.size) * 0.65)],
                                             [int((630 - x) * 0.65), int(((loc - y + 10) + other_font.size) * 0.65)]],
                                  "difficult": False})
            start += 11
            loc += 100

        front_buf.append({"transcription": addr[start:],
                          "points": [[int((630 - x) * 0.65), int((loc - y) * 0.65)],
                                     [int(((630 - x) + other_font.size * len(addr[start:]) - 30 * len(
                                         re.sub(re.compile(r'[^\d]'), "", addr[start:]))) * 0.65),
                                      int((loc - y) * 0.65)],
                                     [int(((630 - x) + other_font.size * len(addr[start:]) - 30 * len(
                                         re.sub(re.compile(r'[^\d]'), "", addr[start:]))) * 0.65),
                                      int(((loc - y + 10) + other_font.size) * 0.65)],
                                     [int((630 - x) * 0.65), int(((loc - y + 10) + other_font.size) * 0.65)]],
                          "difficult": False})

        front_buf.append({"transcription": "公民身份证号码",
                          "points": [[int((425 - x) * 0.65), int((1475 - y) * 0.65)],
                                     [int((795 - x) * 0.65), int((1475 - y) * 0.65)],
                                     [int((795 - x) * 0.65), int(((1475 - y) + id_font.size) * 0.65)],
                                     [int((425 - x) * 0.65), int(((1475 - y) + id_font.size) * 0.65)]],
                          "difficult": False})
        # 950, 1475
        front_buf.append({"transcription": idn,
                          "points": [[int((950 - x) * 0.65), int((1475 - y) * 0.65)],
                                     [int(((950 - x) + 44 * len(idn)) * 0.65), int((1475 - y) * 0.65)],
                                     [int(((950 - x) + 44 * len(idn)) * 0.65),
                                      int(((1475 - y) + id_font.size) * 0.65)],
                                     [int((950 - x) * 0.65), int(((1475 - y) + id_font.size) * 0.65)]],
                          "difficult": False})

        back_buf.append({"transcription": "中华人民共和国",
                         "points": [[int((930 - x2) * 0.65), int((2040 - y2) * 0.65)],
                                    [int((1945 - x2) * 0.65), int((2040 - y2) * 0.65)],
                                    [int((1945 - x2) * 0.65), int((2175 - y2) * 0.65)],
                                    [int((930 - x2) * 0.65), int((2175 - y2) * 0.65)]],
                         "difficult": False})
        back_buf.append({"transcription": "居民身份证",
                         "points": [[int((845 - x2) * 0.65), int((2225 - y2) * 0.65)],
                                    [int((2040 - x2) * 0.65), int((2225 - y2) * 0.65)],
                                    [int((2040 - x2) * 0.65), int((2405 - y2) * 0.65)],
                                    [int((845 - x2) * 0.65), int((2405 - y2) * 0.65)]],
                         "difficult": False})
        # draw.text((1050, 2750), org, fill=(0, 0, 0), font=other_font)
        # draw.text((1050, 2895), life, fill=(0, 0, 0), font=other_font)
        back_buf.append({"transcription": "签发机关" + org,
                         "points": [[int((720 - x2) * 0.65), int((2750 - y2) * 0.65)],
                                    [int(((1050 - x2) + other_font.size * len(org)) * 0.65), int((2750 - y2) * 0.65)],
                                    [int(((1050 - x2) + other_font.size * len(org)) * 0.65),
                                     int(((2750 - y2) + other_font.size) * 0.65)],
                                    [int((720 - x2) * 0.65), int(((2750 - y2) + other_font.size) * 0.65)]],
                         "difficult": False})
        back_buf.append({"transcription": "有效期限" + life,
                         "points": [[int((720 - x2) * 0.65), int((2895 - y2) * 0.65)],
                                    [int(((1050 - x2) + 31 * len(life)) * 0.65), int((2895 - y2) * 0.65)],
                                    [int(((1050 - x2) + 31 * len(life)) * 0.65),
                                     int(((2895 - y2) + other_font.size) * 0.65)],
                                    [int((720 - x2) * 0.65), int(((2895 - y2) + other_font.size) * 0.65)]],
                         "difficult": False})
    with open(os.path.join(os.path.dirname(trt_fn), 'Label_idcard_gen_20220418.txt'), 'a') as f:
        # for data in front_buf:
        f.write(
            os.path.join(trt_dn, os.path.basename(trt_fn)) + '_front.jpg\t' + str(front_buf).replace("'", '"').replace("False", 'false') + '\n')
        # for data in back_buf:
        f.write(
            os.path.join(trt_dn, os.path.basename(trt_fn)) + '_back.jpg\t' + str(back_buf).replace("'", '"').replace("False", 'false') + '\n')


if __name__ == '__main__':
    for i in tqdm(range(500)):
        generator(f'/home/ray/DataSet/ocr/ID_card/idcard_gen_20220418/%05d_idcard' % i)

# ch4_test_images/img_304.jpg	[{"transcription": "ALLWATCHES", "points": [[378, 89], [616, 84], [620, 115], [382, 120]], "difficult": false}, {"transcription": "SINC", "points": [[496, 117], [545, 114], [546, 124], [497, 127]], "difficult": false}, {"transcription": "###", "points": [[554, 113], [595, 114], [596, 125], [555, 124]], "difficult": false}, {"transcription": "###", "points": [[1004, 144], [1035, 141], [1035, 153], [1004, 156]], "difficult": false}, {"transcription": "###", "points": [[558, 206], [597, 207], [597, 218], [560, 217]], "difficult": false}, {"transcription": "###", "points": [[890, 172], [919, 168], [921, 183], [891, 185]], "difficult": false}, {"transcription": "###", "points": [[835, 180], [862, 178], [865, 190], [837, 192]], "difficult": false}, {"transcription": "###", "points": [[898, 160], [910, 160], [910, 170], [900, 170]], "difficult": false}]
