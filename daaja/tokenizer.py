from typing import List, Optional, Tuple

from sudachipy import dictionary as sudachi_dictionary
from sudachipy import tokenizer as sudachi_tokenizer


class Tokenizer:
    def __init__(self, sudachi_mode: str = "C") -> None:
        self.tokenizer_obj = sudachi_dictionary.Dictionary().create()
        if sudachi_mode == "A":
            self.mode = sudachi_tokenizer.Tokenizer.SplitMode.A
        elif sudachi_mode == "B":
            self.mode = sudachi_tokenizer.Tokenizer.SplitMode.B
        elif sudachi_mode == "C":
            self.mode = sudachi_tokenizer.Tokenizer.SplitMode.C
        else:
            raise ValueError("Select sudachi_mode from A, B, and C.")

    def tokenize(self,
                 text: str,
                 stopwords: List[str] = [],
                 synonym_target_pos: List[str] = ["名詞", "動詞"]) -> Tuple[List[str], List[Optional[str]]]:
        m = self.tokenizer_obj.tokenize(text, self.mode)

        tokens: List[str] = [mi.surface() for mi in m]
        cleaned_tokens: List[Optional[str]] = [mi.dictionary_form() if mi.part_of_speech()[0] in synonym_target_pos else None for mi in m]
        # Tokens in which the part of speech and stopwords not in synonym_target_pos are set to None.
        selected_tokens: List[Optional[str]] = [token if (token is not None) and (token not in stopwords) else None for token in cleaned_tokens]

        return tokens, selected_tokens


if __name__ == "__main__":
    toeknizer = Tokenizer()
    toeknizer.tokenize("類似するデータを生成する記事を書いてます。", ["類似"])
