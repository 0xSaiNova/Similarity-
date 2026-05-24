# Similarity

Match a phrase against a list of candidates. Returns them ranked 0-1
and labelled MATCH, PARTIAL, or NO_MATCH.

Classical NLP only: TF-IDF, Jaccard, WordNet, char n-grams, word order,
plus a negation gate. No neural nets, no API calls.

## Install

    pip install -r requirements.txt

NLTK data downloads on first run.

## Use

    from matcher import Matcher
    m = Matcher(["the cat sat on the mat", "birds fly south"])
    for r in m.match("a cat on a mat", k=3):
        print(r.score, r.label, r.candidate)

## Tests

    pytest
