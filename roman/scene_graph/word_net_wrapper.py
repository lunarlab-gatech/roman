from __future__ import annotations

import clip
import cupy as cp
from .logger import logger
import nltk
nltk.download('wordnet')
from nltk.corpus import wordnet as wn
from nltk.corpus.reader.wordnet import Synset, Lemma
import numpy as np
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
    
    def get_all_meronyms(self, upwards: bool) -> list[Synset]:
        meronyms = self.synset.part_meronyms()
        meronyms += self.synset.substance_meronyms()
        meronyms += self.synset.member_meronyms()
        if upwards:
            for hypernym in self.synset.hypernyms():
                meronyms += SynsetWrapper(hypernym).get_all_meronyms(upwards)
        return meronyms
    
    def get_all_holonyms(self, upwards: bool) -> list[Synset]:
        holonyms = self.synset.part_holonyms()
        holonyms += self.synset.substance_holonyms()
        holonyms += self.synset.member_holonyms()
        if upwards:
            for hypernym in self.synset.hypernyms():
                holonyms += SynsetWrapper(hypernym).get_all_holonyms(upwards)
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
    def all_words_in_synsets(synsets: list[Synset]) -> set[str]:
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
        meronyms_upwards = self.get_all_meronyms(upwards=True)
        holonyms_upwards = self.get_all_holonyms(upwards=True)
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
        self.word = word

        # Synset: a set of synonyms that share a common meaning.
        self.synsets: list[SynsetWrapper] = synsets

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
    
    def get_all_meronyms(self, upwards: bool) -> set[str]:
        meronyms: list[Synset] = []
        for synset in self.synsets:
            meronyms += synset.get_all_meronyms(upwards)
        return SynsetWrapper.all_words_in_synsets(meronyms)
    
    def get_all_holonyms(self, upwards: bool) -> set[str]:
        holonyms: list[Synset] = []
        for synset in self.synsets:
            holonyms += synset.get_all_holonyms(upwards)
        return SynsetWrapper.all_words_in_synsets(holonyms)
    
class WordNetWrapper():

    def __init__(self):
        self.wordnet_emb_path = Path('/home/dbutterfield3/roman/weights/word_features.npy')
        self._calculate_wordnet_embeddings()
        
    def _calculate_wordnet_embeddings(self):
        """ Convert WordNet words into CLIP embeddings """

        # Get all synsets in wordnet, and then extract all words as the lemmas of the synsets
        all_synsets: list[Synset] = list(wn.all_synsets(pos=wn.NOUN))
        self.word_list: list[str] = list(SynsetWrapper.all_words_in_synsets(all_synsets))
        self.word_list.sort()
        logger.debug(f"{self.word_list}")
        num_of_words: int = len(self.word_list)
        logger.info(f"Number of Words in dictionary: {len(self.word_list)}")

        # Check if we've already calculated these embeddings
        if self.wordnet_emb_path.exists():
            self.word_features = np.load(str(self.wordnet_emb_path))
            logger.info(f"CLIP Features loaded successfully for dictionary")
        else:

            # Load the CLIP model
            device = "cuda" if torch.cuda.is_available() else "cpu"
            self.model, preprocess = clip.load("ViT-L/14", device=device)
            self.model.eval()
            batch_size = 1000

            # Get CLIP features
            self.word_features = np.zeros((num_of_words, 768), dtype=np.float16)
            for i in range(int(np.ceil(num_of_words / batch_size))):
                # Tokenize the words
                word_batch = self.word_list[i*batch_size:(i+1)*batch_size]
                tokens = clip.tokenize(word_batch).to(device)

                # Get text features and save in mapping
                with torch.no_grad():
                    text_features = self.model.encode_text(tokens).cpu().numpy()
                    text_features /= np.linalg.norm(text_features, axis=1, keepdims=True)
                self.word_features[i*batch_size:(i+1)*batch_size] = text_features
            logger.info(f"CLIP Features encoded successfully for dictionary")

            # Finally, save this on the comptuer
            np.save(str(self.wordnet_emb_path), self.word_features)

        # Save word features into CuPy array
        self.word_features_cupy = cp.asarray(self.word_features)
    
    def map_embedding_to_word(self, emb: np.ndarray):
        similarities = self.word_features_cupy @ cp.asarray(emb)
        best_idxs_gpu = cp.argsort(similarities)[:-5][::-1]

        best_idxs = best_idxs_gpu.get()
        best_words = [self.word_list[i] for i in best_idxs]
        return best_words[0]
    
def main():
    words = ["shoebird"]
    for word in words:
        print(WordWrapper.from_word(word))

if __name__ == "__main__":
    main()