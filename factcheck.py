# factcheck.py

import torch
from typing import List
import numpy as np
import spacy
import re


class FactExample(object):
    """
    :param fact: A string representing the fact to make a prediction on
    :param passages: List[dict], where each dict has keys "title" and "text". "title" denotes the title of the
    Wikipedia page it was taken from; you generally don't need to use this. "text" is a chunk of text, which may or
    may not align with sensible paragraph or sentence boundaries
    :param label: S, NS, or IR for Supported, Not Supported, or Irrelevant. Note that we will ignore the Irrelevant
    label for prediction, so your model should just predict S or NS, but we leave it here so you can look at the
    raw data.
    """
    def __init__(self, fact: str, passages: List[dict], label: str):
        self.fact = fact
        self.passages = passages
        self.label = label

    def __repr__(self):
        return repr("fact=" + repr(self.fact) + "; label=" + repr(self.label) + "; passages=" + repr(self.passages))



class EntailmentModel(object):
    def __init__(self, model, tokenizer, cuda=False):
        self.model = model
        self.tokenizer = tokenizer
        self.cuda = cuda

        self.model.eval()
        self.id2label = self.model.config.id2label

        # Build the label mapping ONCE here, not on every call
        self.label_mapping = {}
        for index, label in self.id2label.items():
            label = label.lower()

            if "entailment" in label:
                self.label_mapping["entailment"] = int(index)
            elif "neutral" in label:
                self.label_mapping["neutral"] = int(index)
            elif "contradiction" in label:
                self.label_mapping["contradiction"] = int(index)

        if len(self.label_mapping) != 3:
            raise ValueError(
                f"Unexpected model labels: {self.id2label}"
            )

    def check_entailment(self, premise: str, hypothesis: str):
        with torch.inference_mode():

            # Tokenize the evidence and the fact
            inputs = self.tokenizer(
                premise,
                hypothesis,
                return_tensors='pt',
                truncation=True,
                max_length=512
            )

            # Move inputs to GPU if CUDA is enabled
            if self.cuda:
                inputs = {
                    key: value.to('cuda')
                    for key, value in inputs.items()
                }

            # Run the NLI model
            outputs = self.model(**inputs)

            # Convert logits into probabilities
            probabilities = torch.softmax(
                outputs.logits, dim=-1
            )[0].cpu().tolist()

        # Return probabilities using meaningful class names
        return {
            label: float(probabilities[index])
            for label, index in self.label_mapping.items()
        }
class FactChecker(object):
    """
    Fact checker base type
    """

    def predict(self, fact: str, passages: List[dict]) -> str:
        """
        Makes a prediction on the given sentence
        :param fact: same as FactExample
        :param passages: same as FactExample
        :return: "S" (supported) or "NS" (not supported)
        """
        raise Exception("Don't call me, call my subclasses")


class RandomGuessFactChecker(FactChecker):
    def predict(self, fact: str, passages: List[dict]) -> str:
        prediction = np.random.choice(["S", "NS"])
        return prediction


class AlwaysEntailedFactChecker(FactChecker):
    def predict(self, fact: str, passages: List[dict]) -> str:
        return "S"


class WordRecallThresholdFactChecker(FactChecker):
    def predict(self, fact: str, passages: List[dict]) -> str:
        raise Exception("Implement me")



class EntailmentFactChecker(FactChecker):

    def __init__(self, ent_model):
        self.ent_model = ent_model

        # Lightweight sentence splitter (no model download needed)
        self.nlp = spacy.blank("en")
        self.nlp.add_pipe("sentencizer")

        self.entailment_threshold = 0.36

        # Sentences sharing less than this fraction of the fact's words are skipped
        self.overlap_cutoff = 0.1

    def _words(self, text: str):
        # Lowercase and strip punctuation so "Paris," matches "paris"
        return set(re.findall(r"[a-z0-9]+", text.lower()))

    def _overlap(self, fact: str, sentence: str) -> float:
        fact_words = self._words(fact)
        sent_words = self._words(sentence)
        return len(fact_words & sent_words) / max(len(fact_words), 1)

    def max_entailment_score(self, fact: str, passages: List[dict],
                             return_sentence: bool = False):
        best_score = 0.0
        best_sentence = ""

        for passage in passages:
            passage_text = passage.get("text", "")
            if not passage_text or not passage_text.strip():
                continue

            for sent in self.nlp(passage_text).sents:
                sentence = sent.text.strip()
                if not sentence:
                    continue

                # Pruning: skip sentences with almost no word overlap
                if self._overlap(fact, sentence) < self.overlap_cutoff:
                    continue

                score = self.ent_model.check_entailment(
                    premise=sentence,
                    hypothesis=fact
                )["entailment"]

                if score > best_score:
                    best_score = score
                    best_sentence = sentence

        if return_sentence:
            return best_score, best_sentence
        return best_score

    def predict(self, fact: str, passages: List[dict]) -> str:
        score = self.max_entailment_score(fact, passages)
        return "S" if score >= self.entailment_threshold else "NS"
# OPTIONAL
class DependencyRecallThresholdFactChecker(FactChecker):
    def __init__(self):
        self.nlp = spacy.load('en_core_web_sm')

    def predict(self, fact: str, passages: List[dict]) -> str:
        raise Exception("Implement me")

    def get_dependencies(self, sent: str):
        """
        Returns a set of relevant dependencies from sent
        :param sent: The sentence to extract dependencies from
        :param nlp: The spaCy model to run
        :return: A set of dependency relations as tuples (head, label, child) where the head and child are lemmatized
        if they are verbs. This is filtered from the entire set of dependencies to reflect ones that are most
        semantically meaningful for this kind of fact-checking
        """
        # Runs the spaCy tagger
        processed_sent = self.nlp(sent)
        relations = set()
        for token in processed_sent:
            ignore_dep = ['punct', 'ROOT', 'root', 'det', 'case', 'aux', 'auxpass', 'dep', 'cop', 'mark']
            if token.is_punct or token.dep_ in ignore_dep:
                continue
            # Simplify the relation to its basic form (root verb form for verbs)
            head = token.head.lemma_ if token.head.pos_ == 'VERB' else token.head.text
            dependent = token.lemma_ if token.pos_ == 'VERB' else token.text
            relation = (head, token.dep_, dependent)
            relations.add(relation)
        return relations