# MBartTranslator :
# Author : ParisNeo
# Description : This script translates Stable diffusion prompt from one of the 50 languages supported by MBART
#    It uses MBartTranslator class that provides a simple interface for translating text using the MBart language model.

from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch
from torch.cuda.amp import autocast
from datetime import datetime


class Translator:
    """MBartTranslator class provides a simple interface for translating text using the MBart language model.

    The class can translate between 50 languages and is based on the "facebook/mbart-large-50-many-to-many-mmt"
    pre-trained MBart model. However, it is possible to use a different MBart model by specifying its name.

    Attributes:
        model (MBartForConditionalGeneration): The MBart language model.
        tokenizer (MBart50TokenizerFast): The MBart tokenizer.
    """

    def __init__(self):
        cn_model_name = "/media/zzg/GJ_disk01/pretrained_model/Helsinki-NLP/opus-mt-zh-en"
        jp_model_name = "/media/zzg/GJ_disk01/pretrained_model/Helsinki-NLP/opus-mt-ja-en"

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print("Loading cn model")
        self.cn_model = AutoModelForSeq2SeqLM.from_pretrained(cn_model_name).to(self.device)
        print("Loading cn tokenizer")
        self.cn_tokenizer = AutoTokenizer.from_pretrained(cn_model_name)
        print("cn Translator is ready")

        print("Loading jp model")
        self.jp_model = AutoModelForSeq2SeqLM.from_pretrained(jp_model_name).to(self.device)
        print("Loading jp tokenizer")
        self.jp_tokenizer = AutoTokenizer.from_pretrained(jp_model_name)
        print("jp Translator is ready")

    def translate(self, chinese_sentence: str, lang='cn') -> str:
        """Translate the given text from the input language to the output language.

        Args:
            text (str): The text to translate.
        Returns:
            str: The translated text.
        """
        print(f'{lang} sentence: {chinese_sentence}')
        if lang == 'cn':
            # 将中文句子编码成token ids
            input_ids = self.cn_tokenizer.encode(chinese_sentence, return_tensors="pt").to(self.device)
            # 使用autocast上下文管理器来执行前向传播，在这期间自动使用float16
            with torch.no_grad():
                with autocast():
                    outputs = self.cn_model.generate(input_ids)
            # 将预测的token ids解码成英文句子
            translated_sentence = self.cn_tokenizer.decode(outputs[0], skip_special_tokens=True)

        else:
            # 将中文句子编码成token ids
            input_ids = self.jp_tokenizer.encode(chinese_sentence, return_tensors="pt").to(self.device)
            # 使用autocast上下文管理器来执行前向传播，在这期间自动使用float16
            with torch.no_grad():
                with autocast():
                    outputs = self.jp_model.generate(input_ids)
            # 将预测的token ids解码成英文句子
            translated_sentence = self.jp_tokenizer.decode(outputs[0], skip_special_tokens=True)

        print(f'translated sentence: {translated_sentence}')
        return translated_sentence


if __name__ == "__main__":
    init_start = datetime.now()
    # translator_cn = Translator("/media/zzg/GJ_disk01/pretrained_model/Helsinki-NLP/opus-mt-zh-en")
    translator_jp = Translator('/media/zzg/GJ_disk01/pretrained_model/Helsinki-NLP/opus-mt-ja-en')
    init_end = datetime.now()
    print(f'init model time is : {init_end-init_start}')
    # sentence_cn = '一只猫和一只狗在战斗，旁边的女孩在拍手叫好'
    sentence_jp = '猫と犬がケンカをしていて、隣の女の子が手を叩いている。'
    sentence_en = translator_jp.translate(sentence_jp)
    translate_end = datetime.now()
    print(f'sentence_cn:{sentence_jp}')
    print(f'sentence_en:{sentence_en}')
    print(f'translate time is : {translate_end-init_end}')
