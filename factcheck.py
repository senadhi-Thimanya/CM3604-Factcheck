# factcheck.py

import torch
from typing import List
import numpy as np
import spacy
import gc
import re
from nltk.stem import PorterStemmer


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

        # Map model output indices to their labels
        label_mapping = {}

        for index, label in self.id2label.items():
            label = label.lower()

            if "entailment" in label:
                label_mapping["entailment"] = int(index)
            elif "neutral" in label:
                label_mapping["neutral"] = int(index)
            elif "contradiction" in label:
                label_mapping["contradiction"] = int(index)

        if len(label_mapping) != 3:
            raise ValueError(
                f"Unexpected model labels: {self.id2label}"
            )

        # Return probabilities using meaningful class names
        result = {
            label: float(probabilities[index])
            for label, index in label_mapping.items()
        }

        del inputs, outputs
        gc.collect()

        return result

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


# ---------------------------------------------------------------------------
# Part 1: word overlap (preprocessing + scoring)
# ---------------------------------------------------------------------------

# Hard-coded so we do not depend on nltk.download() having been run on the marker's machine.
STOPWORDS = frozenset("""
a about above after again against all am an and any are aren as at be because been before being below between both
but by can cannot could couldn d did didn do does doesn doing don down during each few for from further had hadn has
hasn have haven having he her here hers herself him himself his how i if in into is isn it its itself just ll m me
mightn more most mustn my myself needn no nor not now o of off on once only or other our ours ourselves out over own
re s same shan she should shouldn so some such t than that the their theirs them themselves then there these they
this those through to too under until up ve very was wasn we were weren what when where which while who whom why will
with won would wouldn y you your yours yourself yourselves also
""".split())

_TAG_RE = re.compile(r"</?s>")           # sentence tags present in the retrieved passages
_TOKEN_RE = re.compile(r"[^\W_]+")        # runs of letters/digits (keeps years such as 1950)
_stemmer = PorterStemmer()
_stem_cache = {}


def _stem(word: str) -> str:
    stemmed = _stem_cache.get(word)
    if stemmed is None:
        stemmed = _stemmer.stem(word)
        _stem_cache[word] = stemmed
    return stemmed


def preprocess(text: str, lowercase: bool = True, remove_stopwords: bool = True, stem: bool = True) -> List[str]:
    """
    Turns raw text into a list of tokens. Each step can be switched off so it can be ablated.
    """
    text = _TAG_RE.sub(" ", text)
    if lowercase:
        text = text.lower()
    tokens = _TOKEN_RE.findall(text)
    if remove_stopwords:
        tokens = [t for t in tokens if t.lower() not in STOPWORDS]
    if stem:
        tokens = [_stem(t) for t in tokens]
    return tokens


def overlap_from_tokens(fact_tokens, passage_tokens, metric: str = "precision") -> float:
    """
    Bag-of-words (set) overlap between a fact and a passage, as a score in [0, 1].
    precision: |F & P| / |F|   (fraction of the fact's words found in the passage)
    jaccard:   |F & P| / |F | P|
    dice:      2|F & P| / (|F| + |P|)
    """
    fact_set, passage_set = set(fact_tokens), set(passage_tokens)
    if not fact_set or not passage_set:
        return 0.0
    inter = len(fact_set & passage_set)
    if metric == "precision":
        return inter / len(fact_set)
    if metric == "jaccard":
        return inter / len(fact_set | passage_set)
    if metric == "dice":
        return 2 * inter / (len(fact_set) + len(passage_set))
    raise ValueError("Unknown metric: " + metric)


def overlap_score(fact: str, text: str, metric: str = "precision", **prep_kwargs) -> float:
    """
    Agreed interface: overlap_score(fact, text) -> float in [0, 1]. Used by Part 1 and for pruning in Part 2.
    """
    return overlap_from_tokens(preprocess(fact, **prep_kwargs), preprocess(text, **prep_kwargs), metric)


class WordRecallThresholdFactChecker(FactChecker):
    """
    Predicts S if the best word-overlap score between the fact and any retrieved passage reaches a threshold.
    pool_passages=True instead compares the fact against the union of the words of all passages.
    """
    def __init__(self, threshold: float = 0.64, metric: str = "precision", lowercase: bool = True,
                 remove_stopwords: bool = True, stem: bool = True, pool_passages: bool = False):
        self.threshold = threshold
        self.metric = metric
        self.prep = dict(lowercase=lowercase, remove_stopwords=remove_stopwords, stem=stem)
        self.pool_passages = pool_passages
        self._passage_cache = {}

    def _passage_tokens(self, text: str):
        tokens = self._passage_cache.get(text)
        if tokens is None:
            tokens = set(preprocess(text, **self.prep))
            self._passage_cache[text] = tokens
        return tokens

    def score(self, fact: str, passages: List[dict]) -> float:
        fact_tokens = preprocess(fact, **self.prep)
        if not passages:
            return 0.0
        passage_sets = [self._passage_tokens(p["text"]) for p in passages]
        if self.pool_passages:
            return overlap_from_tokens(fact_tokens, set().union(*passage_sets), self.metric)
        return max(overlap_from_tokens(fact_tokens, ps, self.metric) for ps in passage_sets)

    def predict(self, fact: str, passages: List[dict]) -> str:
        return "S" if self.score(fact, passages) >= self.threshold else "NS"



class EntailmentFactChecker(FactChecker):

    def __init__(self, ent_model):
        self.ent_model = ent_model

        # Lightweight sentence splitter
        self.nlp = spacy.blank("en")
        self.nlp.add_pipe("sentencizer")

        self.entailment_threshold = 0.34

        # Skip sentences whose word overlap with the fact is below this
        # (0.0 = pruning off)
        self.pruning_cutoff = 0.2

        # Counts how many sentences went to the model vs. were pruned
        self.checked_count = 0
        self.skipped_count = 0

    def predict(self, fact: str, passages: List[dict]) -> str:

        max_entailment_score = 0.0

        # Examine each retrieved passage
        for passage in passages:

            passage_text = passage.get("text", "")

            if not passage_text or not passage_text.strip():
                continue

            # Split passage into individual sentences
            doc = self.nlp(passage_text)

            sentences = [
                sentence.text.strip()
                for sentence in doc.sents
                if sentence.text.strip()
            ]

            # Check whether any sentence supports the fact
            for sentence in sentences:

                # Prune: very low word overlap means entailment is unlikely
                if overlap_score(fact, sentence) < self.pruning_cutoff:
                    self.skipped_count += 1
                    continue

                self.checked_count += 1

                result = self.ent_model.check_entailment(
                    premise=sentence,
                    hypothesis=fact
                )

                entailment_score = result["entailment"]

                max_entailment_score = max(
                    max_entailment_score,
                    entailment_score
                )

        # Make the final binary prediction
        if max_entailment_score >= self.entailment_threshold:
            return "S"

        return "NS"



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