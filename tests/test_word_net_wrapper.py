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
        synset_wrapper = SynsetWrapper(house_synset)

        self.assertEqual("house", synset_wrapper.get_word())

    def test_get_all_meronyms(self):
        """ Test get_all_meronyms method of SynsetWrapper. """

        # Test that no meronyms are found when meronym_levels is 0
        synset_wrapper: SynsetWrapper = SynsetWrapper(wn.synsets("coffee")[0])

        meronyms: list[Synset] = synset_wrapper.get_all_meronyms(False, 0)
        meronym_words: list[str] = [SynsetWrapper(meronym).get_word() for meronym in meronyms]
        np.testing.assert_array_equal([], meronym_words)

        meronyms: list[Synset] = synset_wrapper.get_all_meronyms(True, 0)
        meronym_words: list[str] = [SynsetWrapper(meronym).get_word() for meronym in meronyms]
        np.testing.assert_array_equal([], meronym_words)

        # Test that direct meronyms are found when meronym_levels is 1
        meronyms = synset_wrapper.get_all_meronyms(False, 1)
        meronym_words: list[str] = [SynsetWrapper(meronym).get_word() for meronym in meronyms]
        np.testing.assert_array_equal(sorted(['coffee bean', 'caffeine']), sorted(meronym_words))

        meronyms: list[Synset] = synset_wrapper.get_all_meronyms(True, 1)
        meronym_words: list[str] = [SynsetWrapper(meronym).get_word() for meronym in meronyms]
        np.testing.assert_array_equal(sorted(['coffee bean', 'caffeine']), sorted(meronym_words))

        # Test we get even more meronoyms when meronym_levels is 2
        meronyms = synset_wrapper.get_all_meronyms(False, 2)
        meronym_words: list[str] = [SynsetWrapper(meronym).get_word() for meronym in meronyms]
        np.testing.assert_array_equal(sorted(['coffee bean', 'caffeine']), sorted(meronym_words))
        meronyms = synset_wrapper.get_all_meronyms(True, 2)
        meronym_words: list[str] = [SynsetWrapper(meronym).get_word() for meronym in meronyms]
        np.testing.assert_array_equal(sorted(['coffee bean', 'caffeine', 'kernel']), sorted(meronym_words))


if __name__ == "__main__":
    unittest.main()