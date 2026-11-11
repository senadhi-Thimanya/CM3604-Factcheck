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

            # Tokenize the evidence (premise) and the fact (hypothesis)
            inputs = self.tokenizer(
                premise,
                hypothesis,
                return_tensors='pt',
                truncation=True,
                max_length=512
            )

            # Move inputs to the same device as the model (GPU or CPU)
            device = self.model.device
            inputs = {key: value.to(device) for key, value in inputs.items()}

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


# ---------------------------------------------------------------------------
# Part 2: textual entailment
# ---------------------------------------------------------------------------

_SPACE_RE = re.compile(r"\s+")
_SENTENCE_TAG_RE = re.compile(r"</?s>")     # kept separate so Part 2 does not depend on Part 1 code


def clean_passage(text: str) -> str:
    """
    Noise removal for retrieved passages: drops the <s> / </s> sentence tags and collapses
    newlines and repeated spaces, so the sentence splitter and the NLI model see plain text.
    """
    text = _SENTENCE_TAG_RE.sub(" ", text)
    return _SPACE_RE.sub(" ", text).strip()


class EntailmentFactChecker(FactChecker):
    """
    Predicts S when the best entailment probability over all sentences of all retrieved passages
    reaches a threshold. Sentences with very low word overlap with the fact are pruned (skipped).
    """

    def __init__(self, ent_model, entailment_threshold: float = 0.36, overlap_cutoff: float = 0.1,
                 overlap_method: str = "simple", clean_text: bool = False):
        self.ent_model = ent_model

        # Lightweight rule-based sentence splitter (no model download needed)
        self.nlp = spacy.blank("en")
        self.nlp.add_pipe("sentencizer")

        # Predict S when the best entailment probability is at least this value
        self.entailment_threshold = entailment_threshold

        # Skip sentences whose word overlap with the fact is below this (0.0 = pruning off)
        self.overlap_cutoff = overlap_cutoff

        # "simple":  share of the fact's lower-cased words that appear in the sentence
        # "content": the Part 1 overlap (stop words removed, stemmed)
        self.overlap_method = overlap_method

        # Remove <s> tags and extra whitespace before splitting into sentences.
        # Off by default: with the 0.36 threshold it scored 183/221 (82.8%) against 189/221 (85.5%) on raw text.
        self.clean_text = clean_text

        # How many sentences went to the model and how many were pruned
        self.checked_count = 0
        self.skipped_count = 0

    def _overlap(self, fact: str, sentence: str) -> float:
        if self.overlap_method == "content":
            fact_tokens = preprocess(fact)
            if not fact_tokens:          # fact made only of stop words: do not prune
                return 1.0
            return overlap_from_tokens(fact_tokens, preprocess(sentence), "precision")

        fact_words = set(re.findall(r"[a-z0-9]+", fact.lower()))
        sent_words = set(re.findall(r"[a-z0-9]+", sentence.lower()))
        return len(fact_words & sent_words) / max(len(fact_words), 1)

    def max_entailment_score(self, fact: str, passages: List[dict], return_sentence: bool = False):
        best_score = 0.0
        best_sentence = ""

        for passage in passages:
            passage_text = passage.get("text", "")
            if not passage_text or not passage_text.strip():
                continue

            if self.clean_text:
                passage_text = clean_passage(passage_text)

            for sent in self.nlp(passage_text).sents:
                sentence = sent.text.strip()
                if not sentence:
                    continue

                # Pruning: very low word overlap means entailment is very unlikely
                if self._overlap(fact, sentence) < self.overlap_cutoff:
                    self.skipped_count += 1
                    continue

                self.checked_count += 1

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