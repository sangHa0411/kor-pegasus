from konlpy.tag import Mecab
from six.moves import map
from six.moves import range
from rouge_score.rouge import scoring

def _lcs_table(ref, can):
    rows = len(ref)
    cols = len(can)
    lcs_table = [[0] * (cols + 1) for _ in range(rows + 1)]

    for i in range(1, rows + 1):
        for j in range(1, cols + 1):
            if ref[i - 1] == can[j - 1]:
                lcs_table[i][j] = lcs_table[i - 1][j - 1] + 1
            else:
                lcs_table[i][j] = max(lcs_table[i - 1][j], lcs_table[i][j - 1])
                
    return lcs_table

def _score_lcs(target_tokens, prediction_tokens):
    if not target_tokens or not prediction_tokens:
        return scoring.Score(precision=0, recall=0, fmeasure=0)

    lcs_table = _lcs_table(target_tokens, prediction_tokens)
    lcs_length = lcs_table[-1][-1]

    precision = lcs_length / len(prediction_tokens)
    recall = lcs_length / len(target_tokens)
    fmeasure = scoring.fmeasure(precision, recall)

    return scoring.Score(precision=precision, recall=recall, fmeasure=fmeasure)

class RougeScorer(scoring.BaseScorer):

    def __init__(self, ):
        self.tokenizer = Mecab()

    def score(self, target, prediction):
        target_tokens = self.tokenizer.morphs(target)
        prediction_tokens = self.tokenizer.morphs(prediction)
    
        result = {}       
        scores = _score_lcs(target_tokens, prediction_tokens)
        result["rougeL"] = scores
        return result
