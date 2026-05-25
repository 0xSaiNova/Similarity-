# Similarity

Match a phrase against candidates. Returns them ranked 0 to 1 and labelled MATCH, PARTIAL, or NO_MATCH.

Pluggable backends. classical (WordNet plus 6 signals, default). use (Universal Sentence Encoder v4 via TF Hub). gpt (OpenAI text embedding 3).

## Install

    pip install -r requirements.txt        # core
    pip install -r requirements-use.txt    # add USE backend (TensorFlow + TF Hub)
    pip install -r requirements-gpt.txt    # add GPT backend (openai client)

NLTK data downloads on first run. USE model loads on first use.
USE extras need Python 3.10 to 3.13 (no TensorFlow wheel for 3.14+ yet).
GPT backend reads the key from the OPENAI_API_KEY env var and caches every embedding under .cache/ so repeat runs cost nothing.

## Use

    from matcher import Matcher
    m = Matcher(["the cat sat on the mat", "birds fly south"])
    for r in m.match("a cat on a mat", k=3):
        print(r.score, r.label, r.candidate)

    python cli.py "a cat on a mat" cands.txt --backend use

## Evaluate, tune, test

    python evaluate.py     # report on data/gold_pairs.json
    python tune.py         # writes config.json only if cv beats default
    pytest                 # fast suite
    pytest --runslow       # includes USE model integration tests
