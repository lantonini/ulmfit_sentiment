import numpy as np
from pathlib import Path


def imdb(path=Path("data/aclImdb/")):
    import pickle

    try:
        return pickle.load((path / "train-test.p").open("rb"))
    except FileNotFoundError:
        pass

    CLASSES = ["neg", "pos", "unsup"]

    def get_texts(path):
        texts, labels = [], []
        for idx, label in tqdm(enumerate(CLASSES)):
            for fname in tqdm((path / label).glob("*.txt"), leave=False):
                texts.append(fname.read_text())
                labels.append(idx)
        return texts, np.asarray(labels)

    trXY = get_texts(path / "train")
    teXY = get_texts(path / "test")
    data = (trXY, teXY)
    pickle.dump(data, (path / "train-test.p").open("wb"))
    return data
