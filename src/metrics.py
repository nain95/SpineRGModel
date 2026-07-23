"""
NLG metrics (BLEU-1..4, ROUGE-L, CIDEr) via the standard MS-COCO captioning
scorers (pycocoevalcap), fed with PTB-style tokenized text. Avoids the Java
PTBTokenizer dependency by approximating it with a regex-based tokenizer.
"""
import re

from pycocoevalcap.bleu.bleu import Bleu
from pycocoevalcap.rouge.rouge import Rouge
from pycocoevalcap.cider.cider import Cider


PUNCTUATIONS = ["''", "'", "``", "`", "-LRB-", "-RRB-", "-LCB-", "-RCB-",
                ".", "?", "!", ",", ":", "-", "--", "...", ";"]


def ptb_normalize_text(text):
    text = text.lower()

    text = text.replace('(', ' -lrb- ')
    text = text.replace(')', ' -rrb- ')
    text = text.replace('[', ' -lsb- ')
    text = text.replace(']', ' -rsb- ')
    text = text.replace('{', ' -lcb- ')
    text = text.replace('}', ' -rcb- ')

    text = re.sub(r'``', ' `` ', text)
    text = re.sub(r"''", " '' ", text)
    text = re.sub(r'"', ' " ', text)
    text = re.sub(r'([.!?,:;])', r' \1 ', text)
    text = re.sub(r'--', ' -- ', text)
    text = re.sub(r'\.\.\.', ' ... ', text)

    text = re.sub(r"'s\b", " 's", text)
    text = re.sub(r"'re\b", " 're", text)
    text = re.sub(r"'ve\b", " 've", text)
    text = re.sub(r"'ll\b", " 'll", text)
    text = re.sub(r"'d\b", " 'd", text)
    text = re.sub(r"'m\b", " 'm", text)
    text = re.sub(r"n't\b", " n't", text)

    text = re.sub(r'\s+', ' ', text)
    return text.strip()


def ptb_tokenize(text):
    normalized = ptb_normalize_text(text)
    tokens = normalized.split()
    filtered_tokens = [token for token in tokens if token not in PUNCTUATIONS]
    return filtered_tokens


def calculate_metrics(references, hypotheses):
    if not references or not hypotheses or len(references) != len(hypotheses):
        return {}

    gts = {}
    res = {}
    for idx, (ref, hyp) in enumerate(zip(references, hypotheses)):
        if ref.strip() and hyp.strip():
            gts[idx] = [' '.join(ptb_tokenize(ref))]
            res[idx] = [' '.join(ptb_tokenize(hyp))]

    if not gts:
        return {}

    try:
        bleu_scores, _ = Bleu(4).compute_score(gts, res, verbose=0)
        rougeL_score, _ = Rouge().compute_score(gts, res)
        cider_score, _ = Cider().compute_score(gts, res)

        return {
            'bleu_1': bleu_scores[0],
            'bleu_2': bleu_scores[1],
            'bleu_3': bleu_scores[2],
            'bleu_4': bleu_scores[3],
            'rougeL': rougeL_score,
            'cider': cider_score,
        }

    except Exception as e:
        print(f"Metrics calculation failed: {e}")
        return {}
