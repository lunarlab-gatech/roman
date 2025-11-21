from __future__ import annotations

import clip
import cupy as cp
from ..logger import logger
import nltk
nltk.download('wordnet')
from nltk.corpus import wordnet as wn
from nltk.corpus.reader.wordnet import Synset, Lemma
import numpy as np
import os
from pathlib import Path
from PIL import Image  # Required by clip.load, even if not used
import torch
from typeguard import typechecked

@typechecked
class LemmaWrapper():
    """Wrapper for a WordNet Lemma"""
    
    def __init__(self, lemma: Lemma):
        self.lemma = lemma

    def __str__(self) -> str:
        str_rep = f"Name: {self.get_word()}\n"
        str_rep += f"Synset: {self.lemma.synset()}\n"
        antonyms = self.lemma.antonyms()
        if len(antonyms) > 0:str_rep += f"Antonyms: {self.lemma.antonyms()}\n"
        return str_rep
    
    def get_word(self) -> str:
        """ Assuming the lemma name (ex. angle-closure_glaucoma), return the corresponding word (ex. angle-closure glaucoma)"""
        lemma_str: str = self.lemma.name()
        return lemma_str.replace("_", " ")

@typechecked
class SynsetWrapper():
    """Wrapper for a WordNet Synset"""

    def __init__(self, synset: Synset):
        self.synset: Synset = synset

    def get_word(self) -> str:
        """ Assuming the synset name (ex. montezuma.n.01), return the corresponding word (ex. montezuma)"""
        synset_str: str = self.synset.name()
        return synset_str.split('.')[0].replace("_", " ")
    
    def get_all_meronyms(self, include_meronyms: bool, meronym_levels: int = 1) -> list[Synset]:
        meronyms = []

        if meronym_levels > 0:
            meronyms = self.synset.part_meronyms()
            meronyms += self.synset.substance_meronyms()
            meronyms += self.synset.member_meronyms()

        lower_level_meronyms = []
        for meronym in meronyms:
            lower_level_meronyms += SynsetWrapper(meronym).get_all_meronyms(include_meronyms, meronym_levels-1)
        meronyms += lower_level_meronyms

        if include_meronyms:
            for hypernym in self.synset.hypernyms():
                meronyms += SynsetWrapper(hypernym).get_all_meronyms(False, meronym_levels)
        return meronyms
    
    def get_all_holonyms(self, include_hypernyms: bool, holonym_levels: int = 1) -> list[Synset]:
        holonyms = []

        if holonym_levels > 0:
            holonyms = self.synset.part_holonyms()
            holonyms += self.synset.substance_holonyms()
            holonyms += self.synset.member_holonyms()

        higher_level_holonyms = []
        for holonym in holonyms:
            higher_level_holonyms += SynsetWrapper(holonym).get_all_holonyms(include_hypernyms, holonym_levels-1)
        holonyms += higher_level_holonyms

        if include_hypernyms:
            for hypernym in self.synset.hypernyms():
                holonyms += SynsetWrapper(hypernym).get_all_holonyms(False, holonym_levels)
        return holonyms


    @staticmethod
    def synsets_as_strings(synsets: list[Synset], max_len: int = 7) -> list[str]:
        wrapped: list[SynsetWrapper] = SynsetWrapper.synset_to_wrapper(synsets)
        strings = [h.get_word() for h in wrapped]
        if len(strings) > max_len: 
            return strings[0:max_len - 1] + ["..."] + [strings[-1]]
        else: 
            return strings
        
    @staticmethod
    def all_words_in_synsets(synsets: list[Synset] | list[SynsetWrapper]) -> set[str]:
        if len(synsets) == 0:
            return set()
        
        if isinstance(synsets[0], SynsetWrapper):
            temp = []
            for synset in synsets:
                temp.append(synset.synset)
            synsets = temp

        all_words: set[str] = set()
        for synset in synsets:
            lemmas: list[LemmaWrapper] = [LemmaWrapper(x) for x in synset.lemmas()]
            for l in lemmas:
                all_words.add(l.get_word())
        return all_words

    @staticmethod
    def synset_to_wrapper(x: Synset | list[Synset]) -> SynsetWrapper | list[SynsetWrapper]:
        if isinstance(x, Synset):
            return SynsetWrapper(x)
        else:
            return [SynsetWrapper(y) for y in x]
        
    def __str__(self) -> str:
        methods_str_to_call = ["Root Hypernyms", "Hypernyms", "Instance Hypernyms",
                            "Hyponyms", "Instance Hyponyms",
                            "Part Holonyms", "Substance Holonyms", "Member Holonyms",
                            "Part Meronyms", "Substance Meronyms", "Member Meronyms",
                            "In Region Domains", "In Topic Domains", "In Usage Domains"]
        str_rep = f"Name: {self.get_word()}\n"

        # Extract lemmas
        lemma_names = [n.replace("_", " ") for n in self.synset.lemma_names()]
        str_rep += f"Lemma Names: {lemma_names}\n"

        # Print Def/Examples
        str_rep += f"Def: {self.synset.definition()}\n"
        examples = self.synset.examples()
        if len(examples) > 0:
            str_rep += f"Examples: {examples}\n"

        # Print other related words
        for method_str in methods_str_to_call:
            method_exact = method_str.lower().replace(" ", "_")
            synsets = getattr(self.synset, method_exact)()
            if len(synsets) >= 1:
                str_rep += f"{method_str}: {SynsetWrapper.synsets_as_strings(synsets)}\n"

        # Print Upwards Meronyms & Holonyms
        meronyms_upwards = self.get_all_meronyms(True)
        holonyms_upwards = self.get_all_holonyms(True)
        if len(meronyms_upwards) > 0:
            str_rep += f"Upwards Meronyms: {SynsetWrapper.synsets_as_strings(meronyms_upwards, 20)}\n"
        if len(holonyms_upwards) > 0:
            str_rep += f"Upwards Holonyms: {SynsetWrapper.synsets_as_strings(holonyms_upwards, 20)}\n"

        # Print path to root hypernym with depths
        min, max = self.synset.min_depth(), self.synset.max_depth()
        if min == max: str_rep += f"Depth: {min}\n"
        else: str_rep += f"Min Depth: {min}; Max Depth: {max}\n"
        paths = [SynsetWrapper.synsets_as_strings(x, 20) for x in self.synset.hypernym_paths()]
        str_rep += f"Hypernym Paths: {paths}\n"
        str_rep += "\n"

        return str_rep

@typechecked
class WordWrapper():
    """Wrapper for a specific word"""

    def __init__(self, word: str, synsets: list[SynsetWrapper]):
        self.word: str = word

        # Synset: a set of synonyms that share a common meaning.
        self.synsets: list[SynsetWrapper] = synsets

        # Calculate shared lemmas (words which could mean the exact same thing)
        self.shared_lemmas: set[str] = SynsetWrapper.all_words_in_synsets(self.synsets)

    def __eq__(self, other: WordWrapper):
        if self.word == other.word:
            return True
        
        if other.word in self.shared_lemmas:
            return True
        if self.word in other.shared_lemmas:
            return True
        
        return False

    @classmethod
    def from_word(cls, word: str) -> WordWrapper:
        # Get Synsets from WordNet
        word = word.replace(" ", "_")
        synsets: list[SynsetWrapper] = SynsetWrapper.synset_to_wrapper(wn.synsets(word))

        # Exclude synsets where word isn't part of the lemma list 
        # NOTE: Without this, "AS" returns synsets for "a".
        synsets_pure = []
        for syn in synsets:
            lemmas = [w.lower() for w in syn.synset.lemma_names()]
            if word.lower() in lemmas:
                synsets_pure.append(syn)
        return cls(word, synsets_pure)
    
    def __str__(self) -> str:
        str_rep = f"Printing all Synsets for {self.word}..." + '\n'
        for i, syn in enumerate(self.synsets):
            str_rep += f"===== Synset #{i} =====\n"
            str_rep += f"{syn}"
        return str_rep
    
    def get_all_meronyms(self, include_hypernyms: bool, meronym_levels: int = 1) -> set[str]:
        meronyms: list[Synset] = []
        for synset in self.synsets:
            meronyms += synset.get_all_meronyms(include_hypernyms, meronym_levels)
        return SynsetWrapper.all_words_in_synsets(meronyms)
    
    def get_all_holonyms(self, include_hypernyms: bool, holonym_levels: int = 1) -> set[str]:
        holonyms: list[Synset] = []
        for synset in self.synsets:
            holonyms += synset.get_all_holonyms(include_hypernyms, holonym_levels)
        return SynsetWrapper.all_words_in_synsets(holonyms)

    
class WordListWrapper():
    """ 
    Wrapper around a list of words; considered equal if any of
    its words are the same as any of another word list's words.
    """

    def __init__(self, words: list[WordWrapper]):
        self.words: list[WordWrapper] = words

    def __eq__(self, other: WordListWrapper):
        for word_self in self.words:
            for word_other in other.words:
                if word_self == word_other:
                    return True
        return False
    
    def __str__(self) -> str:
        str_rep = "{"
        for i, word in enumerate(self.words):
            str_rep += f"{word.word}"
            if i + 1 < len(self.words):
                str_rep += ", "
        return str_rep + "}"

    @classmethod
    def from_words(cls, words: list[str]) -> WordListWrapper:
        words_wrapped: list[WordWrapper] = []
        for word in words:
            words_wrapped.append(WordWrapper.from_word(word))
        return cls(words_wrapped)
    
    def to_list(self) -> list[str]:
        word_list: list[str] = []
        for word in self.words:
            word_list.append(word.word)
        return word_list
    
    def get_all_meronyms(self, include_hypernyms: bool, meronym_levels: int = 1) -> set[str]:
        meronyms: list[Synset] = []
        for word in self.words:
            meronyms += word.get_all_meronyms(include_hypernyms, meronym_levels)
        return SynsetWrapper.all_words_in_synsets(meronyms)
    
    def get_all_holonyms(self, include_hypernyms: bool, holonym_levels: int = 1) -> set[str]:
        holonyms: list[Synset] = []
        for word in self.words:
            holonyms += word.get_all_holonyms(include_hypernyms, holonym_levels)
        return SynsetWrapper.all_words_in_synsets(holonyms)
    
class WordNetWrapper():

    def __init__(self, word_list: list[str] | None):

        self.model = None
        self.wordnet_emb_path = Path(__file__).resolve().parent / "files" / "word_features.npy"
        logger.info(f"WordNet Embeddings Path: {self.wordnet_emb_path}")

        if word_list is None:
            # Get all synsets in wordnet, and then extract all words as the lemmas of the synsets
            all_synsets: list[Synset] = list(wn.all_synsets(pos=wn.NOUN))
            self.word_list: list[str] = list(SynsetWrapper.all_words_in_synsets(all_synsets))
        else:
            # Get meronyms and holonyms for each word and append to list
            word_list_initial: set[str] = set(word_list)
            word_list_final: set[str] = set()

            for word in word_list_initial:
                wrapped = WordWrapper.from_word(word)
                meronyms = wrapped.get_all_meronyms(True, 2)
                holonyms = wrapped.get_all_holonyms(True, 2)
                word_list_final.update(meronyms)
                word_list_final.update(holonyms)

            word_list_final.update(word_list_initial)
            self.word_list = list(word_list_final)

        self.word_list.sort()
        self.num_of_words: int = len(self.word_list)
        logger.debug(f"Final Word List: {self.word_list}")
        logger.info(f"Number of Words in dictionary: {len(self.word_list)}")

        # Check if we have access to cupy
        try:
            _ = cp.zeros(1)
            self.use_cupy = True
        except Exception:
            self.use_cupy = False

        # Calculate embeddings if we haven't already
        if self.wordnet_emb_path.exists():
            self.word_features = np.load(str(self.wordnet_emb_path))
            logger.info(f"CLIP Features loaded successfully for dictionary")
        else:
            self.word_features = self._calculate_word_embeddings(self.word_list)
            os.makedirs(os.path.dirname(self.wordnet_emb_path), exist_ok=True)
            np.save(str(self.wordnet_emb_path), self.word_features)

        # Save word features into CuPy array
        if self.use_cupy:
            self.word_features_cupy = cp.asarray(self.word_features)
            
    def _calculate_word_embeddings(self, word_list: list[str]) -> np.ndarray:
        """ Convert WordNet words into CLIP embeddings """

        # Load the CLIP model
        if self.model is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
            self.model, preprocess = clip.load("ViT-L/14", device=device)
            self.model.eval()
            batch_size = 1000

        # Get CLIP features
        word_features = np.zeros((len(word_list), 768), dtype=np.float16)
        for i in range(int(np.ceil(len(word_list) / batch_size))):
            # Tokenize the words
            word_batch = word_list[i*batch_size:(i+1)*batch_size]
            tokens = clip.tokenize(word_batch).to(device)

            # Get text features and save in mapping
            with torch.no_grad():
                text_features = self.model.encode_text(tokens).cpu().numpy()
                text_features /= np.linalg.norm(text_features, axis=1, keepdims=True)
            word_features[i*batch_size:(i+1)*batch_size] = text_features
        
        return word_features

    def map_embedding_to_words(self, emb: np.ndarray, k: int = 5) -> list[str]:
        """ Returns the top-k words that match, with first word being most likely. """

        assert k <= self.num_of_words

        if k == 1: top_k = k + 1
        else: top_k = k

        if self.use_cupy:
            similarities = self.word_features_cupy @ cp.asarray(emb)
            best_idxs_gpu = cp.argsort(similarities)[-top_k:][::-1]
            best_idxs = best_idxs_gpu.get()
        else:
            similarities = self.word_features @ emb
            best_idxs = np.argsort(similarities)[-top_k:][::-1]
        
        best_words = [self.word_list[i] for i in best_idxs]
        if k == 1:
            return [best_words[0]]
        else:
            return best_words
        
    def get_embedding_for_word(self, word: str) -> np.ndarray:
        if not word in self.word_list:
            word_feature = self._calculate_word_embeddings([word])

            self.word_list.append(word)
            self.word_features = np.vstack((self.word_features, word_feature))

        return self.word_features[self.word_list.index(word)]
    
def main():

    words = ["automotive vehicle"]
    for word in words:
        print(WordWrapper.from_word(word))

    wordnetWrapper = WordNetWrapper(["curb", "tree", "garbage can", "door", "window", "pole", "street lamp", "trunk", "wall", "sign", "crosswalk", "sidewalk", "mulch", "leaves", "grass", "retaining wall", "railing", "curbstone", "bush", "hedge" "floor marking", "stairs", "column", "car", "wheel", "bike", "street", "manhole", "parking meter", "tree pit", "fire hydrant", "road marking", "zebra strips"])
    
    wrapped = WordWrapper.from_word('car')
    print(wrapped.get_all_holonyms(True, 2))

if __name__ == "__main__":
    main()