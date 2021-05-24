from ..proofread import predict as proofread_predict
from ..repr_checker.checker.check import ReprChecker


class Reward:
    def __init__(self):
        self.checker = ReprChecker()

    def repr_check(self, document: str, summary: str) -> float:
        # output : 0~1
        return self.checker.check(document, summary, caching=True)

    @staticmethod
    def proofread(summary: str) -> float:
        # output : 0~1
        return proofread_predict(summary)

    @staticmethod
    def length(document: str, summary: str) -> float:
        # output : -1~0
        min_len = 30  # no more reward below this length

        doc_len: int = len(document)
        sum_len: int = len(summary)
        assert doc_len > 0

        ratio: float = (sum_len / doc_len)  # 0~1
        min_ratio = (min_len / doc_len)
        if ratio < min_ratio:
            ratio = 0.5 * (min_ratio - ratio) + min_ratio
        return -ratio  # -1~0

    def get_reward(self, document: str, summary: str, log: bool = False) -> float:
        # return reward : -1 ~ 2
        r = (self.repr_check(document, summary), self.proofread(summary), self.length(document, summary))
        if log:
            print(r)
        return r[0] + r[1] + r[2]
