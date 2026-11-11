| Change from best config | Metric | Lowercase | Stopwords | Stem | Passages | Best acc | Threshold | Plateau |
|---|---|---|---|---|---|---|---|---|
| (best) | precision | True | True | True | max | 0.8190 | 0.64 | 0.61-0.66 |
| lowercase -> False | precision | False | True | True | max | 0.8190 | 0.64 | 0.61-0.66 |
| stopwords -> False | precision | True | False | True | max | 0.7919 | 0.75 | 0.72-0.77 |
| stem -> False | precision | True | True | False | max | 0.7873 | 0.64 | 0.61-0.66 |
| metric -> jaccard | jaccard | True | True | True | max | 0.6968 | 0.03 | 0.03-0.03 |
| metric -> dice | dice | True | True | True | max | 0.6923 | 0.06 | 0.06-0.06 |
| passages -> pooled | precision | True | True | True | pooled | 0.7964 | 0.71 | 0.71-0.71 |
