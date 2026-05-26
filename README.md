# Similarity

Match a phrase against candidates. Returns them ranked 0 to 1 and labelled MATCH, PARTIAL, or NO_MATCH.

Pluggable backends: classical (WordNet plus 6 signals, default), use (Universal Sentence Encoder v4 via TF Hub), gpt (OpenAI compatible text embeddings).

## Install

    pip install -r requirements.txt
    pip install -r requirements-use.txt
    pip install -r requirements-gpt.txt

USE needs Python 3.10 to 3.13. GPT reads OPENAI_API_KEY; set OPENAI_BASE_URL plus GPT_MODEL to route through gateways such as OpenRouter. Embeddings cache to .cache/ so repeat runs cost nothing.

## Use

    python cli.py "a cat on a mat" cands.txt --backend use
    python evaluate.py --backend classical
    python evaluate.py --compare          # classical vs use vs gpt side by side
    python tune.py --backend classical    # also use, gpt
    pytest                                # fast suite
    pytest --runslow                      # USE model integration tests

Tuned thresholds land under the backend name in config.json only when 5 fold CV beats the in code default.
