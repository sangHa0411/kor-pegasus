import os 
import json
import random
from functools import partial
from kor_rouge import RougeScorer
from tqdm import tqdm

def get_rouge(sens, scorer) :
    info = {}
    for i, sen  in enumerate(sens) :
        remain = sens[:i] + sens[i:]
        remain = " ".join(remain)

        score = round(scorer.score(remain, sen)["rougeL"].fmeasure, 4)
        info[i] = score
    return info

def split(sens, info) :
    gap_size = int(len(sens) * 0.3)
    if gap_size == 0 :
        gap_size = 1

    rank_items = sorted(info.items(), key=lambda x : x[1], reverse=True)
    rank_ids = [item[0] for item in rank_items][:gap_size]

    source = []
    target = []

    for i in range(len(sens)) :

        if i in rank_ids :
            target.append(sens[i])
            source.append("<mask_2>")
        else :
            source.append(sens[i])
    return {"document" : " ".join(source), "summary" : " ".join(target)}

def main() :

    BATCH_SIZE = 10000
    DIR_PATH = "/home/wkrtkd911/project/kor-pegasus/raw_dataset"
    rouge_scorer = RougeScorer()

    def preprocess(document, scorer) :
        info = get_rouge(document, scorer)
        data = split(document, info)
        return data

    preprocess_fn = partial(preprocess, scorer=rouge_scorer)

    file_list = os.listdir(DIR_PATH)
    file_list = [f for f in file_list if f.endswith(".json")]

    num = 0
    for file in file_list :
        path = os.path.join(DIR_PATH, file)

        print("\n" + path)
        with open(path, "r") as f :
            docs = json.load(f)

        print("Filtering Datasets")
        docs = [d for d in docs if len(d) >= 4 and len(d) <= 50]
        print("Preprocessing Datasets")
        results = [preprocess_fn(d) for d in tqdm(docs)]

        for i in range(0, len(results), BATCH_SIZE) :
            sub_results = results[i:i+BATCH_SIZE]
            save_path = os.path.join("./datasets", f"document{num}.json")
            with open(save_path, "w") as f :
                json.dump(sub_results, f, ensure_ascii=False, indent=4)
            num += 1

if __name__ == "__main__" :
    main()