### 1. Tokenization and minimum token length (stopwords + Porter, precision, max passage)

| Setting | Best acc | Threshold | Plateau |
|---|---|---|---|
| regex (letters+digits) | 0.8190 | 0.64 | 0.61-0.66 |
| letters only (drop digits) | 0.7919 | 0.64 | 0.61-0.66 |
| regex + keep apostrophe words | 0.8235 | 0.64 | 0.61-0.66 |
| whitespace + strip punctuation | 0.8100 | 0.62 | 0.61-0.62 |
| regex, min token length 2 | 0.8190 | 0.64 | 0.61-0.66 |
| regex, min token length 3 | 0.8190 | 0.64 | 0.61-0.66 |

### 2. Stopword list (Porter stemming, precision)

| Setting | Best acc | Threshold | Plateau |
|---|---|---|---|
| stopwords: none | 0.7919 | 0.75 | 0.72-0.77 |
| stopwords: small | 0.8235 | 0.69 | 0.67-0.71 |
| stopwords: full | 0.8190 | 0.64 | 0.61-0.66 |

### 3. Stemming vs lemmatization, and lowercasing with the stemmer switched off

| Setting | Best acc | Threshold | Plateau |
|---|---|---|---|
| normalizer: none | 0.7873 | 0.64 | 0.61-0.66 |
| normalizer: porter | 0.8190 | 0.64 | 0.61-0.66 |
| normalizer: snowball | 0.8190 | 0.64 | 0.61-0.66 |
| normalizer: lemma | 0.8009 | 0.64 | 0.61-0.66 |
| no normalizer, lowercase=True | 0.7873 | 0.64 | 0.61-0.66 |
| no normalizer, lowercase=False | 0.7783 | 0.64 | 0.61-0.66 |

### 4. Scoring alternatives (precision-style variants)

| Setting | Best acc | Threshold | Plateau |
|---|---|---|---|
| precision, best passage (current) | 0.8190 | 0.64 | 0.61-0.66 |
| IDF-weighted precision | 0.8281 | 0.72 | 0.72-0.72 |
| precision, best single sentence | 0.7738 | 0.64 | 0.61-0.66 |
| precision, best 2-sentence window | 0.8100 | 0.64 | 0.61-0.66 |
| bigram precision | 0.7149 | 0.30 | 0.26-0.33 |
| average of unigram and bigram precision | 0.7738 | 0.43 | 0.42-0.43 |

### 5. Threshold honesty: in-sample vs grouped cross-validation (5 folds by person, 30 repeats)

| Scoring | In-sample best | Grouped CV mean +- sd | 95% bootstrap CI at chosen threshold |
|---|---|---|---|
| precision, best passage (current) | 0.8190 | 0.8078 +- 0.0148 | 0.769-0.869 |
| IDF-weighted precision | 0.8281 | 0.8106 +- 0.0140 | 0.778-0.878 |
| precision, best single sentence | 0.7738 | 0.7321 +- 0.0128 | 0.719-0.828 |
| precision, best 2-sentence window | 0.8100 | 0.8051 +- 0.0088 | 0.756-0.860 |
| bigram precision | 0.7149 | 0.7134 +- 0.0056 | 0.656-0.774 |
| average of unigram and bigram precision | 0.7738 | 0.7513 +- 0.0220 | 0.719-0.828 |
