import math
from nltk.corpus import wordnet as wn
import numpy as np
from roman.scene_graph.word_net_wrapper import *
from nltk.corpus.reader.wordnet import Synset, Lemma
import unittest

class TestLemmaWrapper(unittest.TestCase):

    def test__init__(self):
        """ Test initialization of LemmaWrapper. """

        dog_synset = wn.synsets("dogs")[0]
        lemmas = dog_synset.lemmas()

        words_from_lemma_wrapper = []
        for lemma in lemmas:
            words_from_lemma_wrapper.append(LemmaWrapper(lemma).get_word())

        np.testing.assert_array_equal(["dog", "domestic dog", "Canis familiaris"], words_from_lemma_wrapper)

class TestSynsetWrapper(unittest.TestCase):

    def test_get_word(self):
        """ Test get_word method of SynsetWrapper. """

        house_synset = wn.synsets("house")[0]
        print(house_synset.name())
        synset_wrapper = SynsetWrapper(house_synset)
        print(synset_wrapper)

        self.assertEqual("house", synset_wrapper.get_word())

if __name__ == "__main__":
    unittest.main()