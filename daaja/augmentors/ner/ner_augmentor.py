from typing import List


class NerAugmentor:
    def __init__(self) -> None:
        pass

    def augment(self,
                tokens: List[str],
                labels: List[str]) -> str:
        raise NotImplementedError
