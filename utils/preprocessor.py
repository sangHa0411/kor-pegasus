
import re

class Preprocessor :
    def __init__(self, ) :
        pass

    def __call__(self, dataset) :
        documents = []
        summaries = []

        size = len(dataset["document"])
        for i in range(size) :
            doc = dataset["document"][i]
            doc = re.sub("\s+", " ", doc).strip()
            documents.append(doc)

            summary = dataset["summary"][i]
            summary = re.sub("\s+", " ", summary).strip()
            summaries.append(summary)
            
        dataset["document"] = documents
        dataset["summary"] = summaries
        return dataset