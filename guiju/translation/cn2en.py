# MBartTranslator :
# Author : ParisNeo
# Description : This script translates Stable diffusion prompt from one of the 50 languages supported by MBART
#    It uses MBartTranslator class that provides a simple interface for translating text using the MBart language model.

from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch

class Translator:
    """MBartTranslator class provides a simple interface for translating text using the MBart language model.

    The class can translate between 50 languages and is based on the "facebook/mbart-large-50-many-to-many-mmt"
    pre-trained MBart model. However, it is possible to use a different MBart model by specifying its name.

    Attributes:
        model (MBartForConditionalGeneration): The MBart language model.
        tokenizer (MBart50TokenizerFast): The MBart tokenizer.
    """

    def __init__(self, model_name="/media/zzg/GJ_disk01/pretrained_model/Helsinki-NLP/opus-mt-zh-en"):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print("Loading model")
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name).to(self.device)
        print("Loading tokenizer")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        print("Translator is ready")

    def translate(self, chinese_sentence: str) -> str:
        """Translate the given text from the input language to the output language.

        Args:
            text (str): The text to translate.
        Returns:
            str: The translated text.
        """
        # 将中文句子编码成token ids
        input_ids = self.tokenizer.encode(chinese_sentence, return_tensors="pt").to(self.device)
        # 使用模型进行预测
        outputs = self.model.generate(input_ids)
        # 将预测的token ids解码成英文句子
        translated_sentence = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return translated_sentence

if __name__ == "__main__":
    translator = Translator()
    sentence_cn = '一只猫和一只狗在战斗，旁边的女孩在拍手叫好'
    sentence_en = translator.translate(sentence_cn)
    print(f'sentence_cn:{sentence_cn}')
    print(f'sentence_en:{sentence_en}')

