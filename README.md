# Similarity

Match a phrase against candidates. Returns them ranked 0 to 1 and labelled MATCH, PARTIAL, or NO_MATCH.

Classical NLP only. Six signals (TF-IDF cosine, Jaccard, WordNet alignment, WordNet soft-overlap, char n-grams, word order) combined max-of-two-groups (surface vs semantic), with negation, antonym, and order-mismatch gates. No neural nets, no API calls.

## Install

    pip install -r requirements.txt

NLTK data downloads on first run.

## Use

    from matcher import Matcher
    m = Matcher(["the cat sat on the mat", "birds fly south"])
    for r in m.match("a cat on a mat", k=3):
        print(r.score, r.label, r.candidate)

## Evaluate, tune, test

    python evaluate.py   # report on data/gold_pairs.json
    python tune.py       # writes config.json only if cv beats default
    pytest
