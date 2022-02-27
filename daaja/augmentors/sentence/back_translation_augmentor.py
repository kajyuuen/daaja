import torch
from daaja.augmentors.sentence.sentence_augmentor import SentenceAugmentor
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, pipeline


class BackTranslationAugmentor(SentenceAugmentor):
    def __init__(self,
                 from_model_name: str = "Helsinki-NLP/opus-mt-ja-en",
                 to_model_name: str = "Helsinki-NLP/opus-mt-en-jap",
                 device: int = -1) -> None:
        # ja -> en
        self.from_pipeline = pipeline("translation",
                                      model=AutoModelForSeq2SeqLM.from_pretrained(from_model_name),
                                      tokenizer=AutoTokenizer.from_pretrained(from_model_name),
                                      device=device)
        # en -> ja
        self.to_pipeline = pipeline("translation",
                                    model=AutoModelForSeq2SeqLM.from_pretrained(to_model_name),
                                    tokenizer=AutoTokenizer.from_pretrained(to_model_name),
                                    device=device)

    @torch.no_grad()
    def augment(self, sentence: str) -> str:
        """Back translation Augmentor

        Args:
            sentence (str): sentence

        Returns:
            str: Back translation sentence
        """
        to_text = self.from_pipeline(sentence)[0]["translation_text"]
        from_text = self.to_pipeline(to_text)[0]["translation_text"]
        return from_text
