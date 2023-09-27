import torch
import spacy


def bag_of_words(names):
    nlp = spacy.load("en_core_web_sm")
    unique_words = set()

    for name in names:
        doc = nlp(name)
        word_frequency = {}
        for token in doc:
            if token.text in word_frequency:
                word_frequency[token.text] += 1
            else:
                word_frequency[token.text] = 1

        bag = []
        for w in sorged(unique_words):
            if w in word_frequency:
                bag.append(word_frequency[w])
            else:
                bag.append(0)

    t = torch.tensor(bag)
    t2 = torch.tensor([1, 2, 6, 3])
    t3 = torch.cat([t, t2])


print(bag_of_words(["test", "other", "wii"]))
