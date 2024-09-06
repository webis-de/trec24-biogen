# TREC description

Run trials with Optuna until we have 100 trials with GPT4-o mini (OpenAI) and 100 trials with Mistral-Instruct-v0.3 (Blablador). Evaluate Recall@1000, nDCG, ROUGE-1 F1 and ROUGE-L F1.

Mistral-Instruct-v0.3:

```shell
trec-biogen optimize --study-storage-path data/optuna.log --answers-path data/training12b_new.json --retrieval-measures Recall@1000 nDCG --generation-measures rouge1-f1 rougeL-f1 --progress --sample 10 --trials 100 --study-name webis
```

GPT4-o mini:

```shell
trec-biogen optimize --study-storage-path data/optuna.log --answers-path data/training12b_new.json --retrieval-measures Recall@1000 nDCG --generation-measures rouge1-f1 rougeL-f1 --progress --sample 10 --trials 100 --study-name webis-gpt
```

For GPT4-o mini / Mistral-Instruct-v0.3, respectively, take top-10 by mean score, then re-evaluate with same measures plus faithfullness, TODO, and TODO (from RAGAS).
Compute submissions for top-5 by mean score.

Mistral-Instruct-v0.3:

```shell
trec-biogen prepare-trec-submissions --study-storage-path data/optuna.log --answers-path data/training12b_new.json --questions-path data/BioGen2024topics-json.txt --submissions-path data/submissions/ --team-id webis --contact-email heinrich.merker@uni-jena.de --retrieval-measures Recall@1000 nDCG --generation-measures rouge1-f1 rougeL-f1 faithfulness answer-relevance --sample 10 --top-k 10 --study-name webis
```

GPT4-o mini:

```shell
trec-biogen prepare-trec-submissions --study-storage-path data/optuna.log --answers-path data/training12b_new.json --questions-path data/BioGen2024topics-json.txt --submissions-path data/submissions/ --team-id webis --contact-email heinrich.merker@uni-jena.de --retrieval-measures Recall@1000 nDCG --generation-measures rouge1-f1 rougeL-f1 faithfulness answer-relevance --sample 10 --top-k 10 --study-name webis-gpt
```

Afterwards, semi-automatically fix minor format issues in the generated answers:

- dangling references (regex: `\[.*?\d+.*?\]`)
- sentence splitting errors (e.g., `implantation and pregnancy. [19464684].`)
- repeated dots/spaces (regexes: `(\S)  +` and `(\S)\.\.+`)

Submit 5 runs with Mistral-Instruct-v0.3 and 5 runs with GPT4-o mini.


## Runs

### webis-1

Tag: webis-1
Generation-only? No
Document retrieval model description:
    Retrieve up to 10 PubMed articles from Elasticsearch using BM25.
    As the query, use the concatenated question, query, narrative, and simple yes-no, factual, or list answer (if question type is known and if retrieval is run after generation).
    Stopwords are not removed from the query.
    Match the query on the article's abstract text (Elasticsearch BM25).
    Exclude PubMed articles of non-peer-reviewed publication types.
    After retrieval, split passages from the retrieved article's abstract text by splitting it into sentences and returning all sentence n-grams up to 3 sentences.
    Re-rank up to 50 passages pointwise with a TCT-ColBERT model (castorini/tct_colbert-v2-hnp-msmarco).
Answer generation model description:
    Generate a summary answer for each question with DSPy using a Mistral model (Mistral-7B-Instruct-v0.3, via Blablador API).
    Give the question and the top-10 passages to the model as context (numbered by rank), and prompt the model to return a summary answer with references (by rank) given in the text.
    Using DSPy, optimize the prompt by labeled few-shot prompting with 3 examples from the BioASQ 12b train set.
    After generation, convert internal reference numbering back to PubMed IDs.
Short description:
    Use retrieval and generation as above, while augmenting both retrieval and generation independently.
Details/comments:
    Using the retrieval and generation modules as described above, independently augment the generation module with retrieval and the retrieval module with generation.
    For generation-augmented retrieval, augment 3 times while not feeding back retrieval results to the generation module.
    For retrieval-augmented generation, augment 3 times while feeding back generation results to the retrieval module.
    Retrieval is implemented using PyTerrier.
    Generation is implemented using DSPy.
Judging precedence: 1
Submitted!

### webis-2

Tag: webis-2
Generation-only? No
Document retrieval model description:
    Retrieve up to 10 PubMed articles from Elasticsearch using BM25.
    As the query, use the concatenated question, narrative, and simple yes-no, factual, or list answer (if question type is known and if retrieval is run after generation).
    Stopwords are not removed from the query.
    Match the query on the article's abstract text (Elasticsearch BM25).
    Exclude PubMed articles of non-peer-reviewed publication types.
    After retrieval, split passages from the retrieved article's abstract text by splitting it into sentences and returning all sentence n-grams up to 3 sentences.
    Re-rank up to 10 passages pointwise with a TCT-ColBERT model (castorini/tct_colbert-v2-hnp-msmarco).
    Re-rank up to 3 passages pairwise with a duoT5 model (castorini/duot5-base-msmarco).
Answer generation model description:
    Generate a summary answer for each question with DSPy using a Mistral model (Mistral-7B-Instruct-v0.3, via Blablador API).
    Give the question and the top-3 passages to the model as context (numbered by rank), and prompt the model to return a summary answer with references (by rank) given in the text.
    Use chain-of-thought prompting.
    Using DSPy, optimize the prompt by labeled few-shot prompting with 3 examples from the BioASQ 12b train set.
    After generation, convert internal reference numbering back to PubMed IDs.
Short description:
    Use retrieval and generation as above, while augmenting both retrieval and generation independently.
Details/comments:
    Using the retrieval and generation modules as described above, independently augment the generation module with retrieval and the retrieval module with generation.
    For generation-augmented retrieval, augment 3 times while not feeding back retrieval results to the generation module.
    For retrieval-augmented generation, augment 3 times while feeding back generation results to the retrieval module.
    Retrieval is implemented using PyTerrier.
    Generation is implemented using DSPy.
Judging precedence: 2
Submitted!

### webis-3

Tag: webis-3
Generation-only? No
Document retrieval model description:
    Retrieve up to 10 PubMed articles from Elasticsearch using BM25.
    As the query, use the concatenated question, narrative, and simple yes-no, factual, or list answer (if question type is known and if retrieval is run after generation).
    Stopwords are not removed from the query.
    Match the query on the article's title, abstract text, and MeSH terms (Elasticsearch BM25, both title and abstract must match, MeSH terms should match, MeSH terms matched to medical entities extracted from the query using sciSpaCy).
    Exclude PubMed articles of non-peer-reviewed publication types.
    After retrieval, split passages from the retrieved article's abstract text by splitting it into sentences and returning all sentence n-grams up to 3 sentences. The full title is also used as a separate passage.
    Re-rank up to 10 passages pointwise with a monoT5 model (castorini/monot5-base-msmarco).
Answer generation model description:
    Generate a summary answer for each question with DSPy using a Mistral model (Mistral-7B-Instruct-v0.3, via Blablador API).
    Give the question and the top-10 passages to the model as context (numbered by rank), and prompt the model to return a summary answer with references (by rank) given in the text.
    Using DSPy, optimize the prompt by labeled few-shot prompting with 3 examples from the BioASQ 12b train set.
    After generation, convert internal reference numbering back to PubMed IDs.
Short description:
    Use retrieval and generation as above, while augmenting both retrieval and generation independently.
Details/comments:
    Using the retrieval and generation modules as described above, independently augment the generation module with retrieval and the retrieval module with generation.
    For generation-augmented retrieval, augment 3 times while not feeding back retrieval results to the generation module.
    For retrieval-augmented generation, augment 3 times while feeding back generation results to the retrieval module.
    Retrieval is implemented using PyTerrier.
    Generation is implemented using DSPy.
Judging precedence: 3
Submitted!

### webis-4

Tag: webis-4
Generation-only? No
Document retrieval model description:
    Retrieve up to 10 PubMed articles from Elasticsearch using BM25.
    As the query, use the concatenated question, narrative, and simple yes-no, factual, or list answer (if question type is known and if retrieval is run after generation).
    Stopwords are not removed from the query.
    Match the query on the article's title, abstract text, and MeSH terms (Elasticsearch BM25, both title and abstract must match, MeSH terms should match, MeSH terms matched to medical entities extracted from the query using sciSpaCy).
    Exclude PubMed articles of non-peer-reviewed publication types.
    After retrieval, split passages from the retrieved article's abstract text by splitting it into sentences and returning all sentence n-grams up to 3 sentences. The full title is also used as a separate passage.
    Re-rank up to 10 passages pointwise with a monoT5 model (castorini/monot5-base-msmarco).
Answer generation model description:
    Generate a summary answer for each question with DSPy using a Mistral model (Mistral-7B-Instruct-v0.3, via Blablador API).
    Give the question and the top-10 passages to the model as context (numbered by rank), and prompt the model to return a summary answer with references (by rank) given in the text.
    Using DSPy, optimize the prompt by labeled few-shot prompting with 3 examples from the BioASQ 12b train set.
    After generation, convert internal reference numbering back to PubMed IDs.
Short description:
    Use retrieval and generation as above, while augmenting both retrieval and generation independently.
Details/comments:
    Using the retrieval and generation modules as described above, independently augment the generation module with retrieval and the retrieval module with generation.
    For generation-augmented retrieval, augment 3 times while not feeding back retrieval results to the generation module.
    For retrieval-augmented generation, augment 3 times while feeding back generation results to the retrieval module.
    Retrieval is implemented using PyTerrier.
    Generation is implemented using DSPy.
Judging precedence: 0 (run effectively uses the same config as webis-3)

### webis-5

Tag: webis-5
Generation-only? No
Document retrieval model description:
    Retrieve up to 10 PubMed articles from Elasticsearch using BM25.
    As the query, use the concatenated question, narrative, and simple yes-no, factual, or list answer (if question type is known and if retrieval is run after generation).
    Stopwords are not removed from the query.
    Match the query on the article's title and abstract text (Elasticsearch BM25, title should match and abstract must match).
    Exclude PubMed articles of non-peer-reviewed publication types.
    Exclude PubMed articles with empty title.
    After retrieval, split passages from the retrieved article's abstract text by splitting it into sentences and returning all sentence n-grams up to 3 sentences. The full title is also used as a separate passage.
    Re-rank up to 10 passages pointwise with a monoT5 model (castorini/monot5-base-msmarco).
Answer generation model description:
    Generate a summary answer for each question with DSPy using a Mistral model (Mistral-7B-Instruct-v0.3, via Blablador API).
    Give the question and the top-10 passages to the model as context (numbered by rank), and prompt the model to return a summary answer with references (by rank) given in the text.
    Using DSPy, optimize the prompt by labeled few-shot prompting with 3 examples from the BioASQ 12b train set.
    After generation, convert internal reference numbering back to PubMed IDs.
Short description:
    Use retrieval and generation as above, while augmenting both retrieval and generation independently.
Details/comments:
    Using the retrieval and generation modules as described above, independently augment the generation module with retrieval and the retrieval module with generation.
    For generation-augmented retrieval, augment 3 times while not feeding back retrieval results to the generation module.
    For retrieval-augmented generation, augment 3 times while feeding back generation results to the retrieval module.
    Retrieval is implemented using PyTerrier.
    Generation is implemented using DSPy.
Judging precedence: 4
Submitted!

### webis-6

Tag: webis-6
Generation-only?
Document retrieval model description:
Answer generation model description:
Short description:
Details/comments:
Judging precedence:

### webis-7

Tag: webis-7
Generation-only?
Document retrieval model description:
Answer generation model description:
Short description:
Details/comments:
Judging precedence:

### webis-8

Tag: webis-8
Generation-only?
Document retrieval model description:
Answer generation model description:
Short description:
Details/comments:
Judging precedence:

### webis-9

Tag: webis-9
Generation-only?
Document retrieval model description:
Answer generation model description:
Short description:
Details/comments:
Judging precedence:

### webis-10

Tag: webis-10
Generation-only?
Document retrieval model description:
Answer generation model description:
Short description:
Details/comments:
Judging precedence:

### webis-gpt-1

Tag: webis-gpt-1
Generation-only? Yes
Document retrieval model description:
    Not used.
Answer generation model description:
    Generate a summary answer for each question with DSPy using a GPT-4o mini model (gpt-4o-mini-2024-07-18).
    Give the question to the model and prompt the model to return a summary answer.
    Using DSPy, optimize the prompt by labeled few-shot prompting with 3 examples from the BioASQ 12b train set.
Short description:
    Use generation as above, while not augmenting the generation.
Details/comments:
    Generation is implemented using DSPy.
Judging precedence: 4
Submitted!

### webis-gpt-2

Tag: webis-gpt-2
Generation-only? Yes
Document retrieval model description:
    Not used.
Answer generation model description:
    Generate a summary answer for each question with DSPy using a GPT-4o mini model (gpt-4o-mini-2024-07-18).
    Give the question to the model and prompt the model to return a summary answer.
    Using DSPy, optimize the prompt by labeled few-shot prompting with 3 examples from the BioASQ 12b train set.
Short description:
    Use generation as above, while not augmenting the generation.
Details/comments:
    Generation is implemented using DSPy.
Judging precedence: 0 (run effectively uses the same config as webis-gpt-1)

### webis-gpt-3

Tag: webis-gpt-3
Generation-only? Yes
Document retrieval model description:
    Not used.
Answer generation model description:
    Generate a summary answer for each question with DSPy using a GPT-4o mini model (gpt-4o-mini-2024-07-18).
    Give the question to the model and prompt the model to return a summary answer.
    Using DSPy, optimize the prompt by labeled few-shot prompting with 3 examples from the BioASQ 12b train set.
Short description:
    Use generation as above, while not augmenting the generation.
Details/comments:
    Generation is implemented using DSPy.
Judging precedence: 0 (run effectively uses the same config as webis-gpt-1)

### webis-gpt-4

Tag: webis-gpt-4
Generation-only? No
Document retrieval model description:
    Retrieve up to 10 PubMed articles from Elasticsearch using BM25.
    As the query, use the concatenated question, narrative, and simple yes-no, factual, or list answer (if question type is known and if retrieval is run after generation).
    Stopwords are removed from the query.
    Match the query on the article's abstract text (Elasticsearch BM25).
    Exclude PubMed articles with empty abstract text.
    After retrieval, split passages from the retrieved article's abstract text by splitting it into sentences and returning all sentence n-grams up to 3 sentences.
    Re-rank up to 50 passages pointwise with a monoT5 model (castorini/monot5-base-msmarco).
Answer generation model description:
    Generate a summary answer for each question with DSPy using a GPT-4o mini model (gpt-4o-mini-2024-07-18).
    Give the question and the top-3 passages to the model as context (numbered by rank), and prompt the model to return a summary answer with references (by rank) given in the text.
    Use the unoptimized prompt from DSPy.
    After generation, convert internal reference numbering back to PubMed IDs.
Short description:
    Use retrieval and generation as above, while cross-augmenting both retrieval and generation.
Details/comments:
    Using the retrieval and generation modules as described above, cross-augment the generation module with retrieval and the retrieval module with generation (i.e., in each step, the previous retrieval result is used to augment the next generation step and vice-versa).
    Do 2 cross-augmentation steps.
    Retrieval is implemented using PyTerrier.
    Generation is implemented using DSPy.
Judging precedence: 3
Submitted!

### webis-gpt-5

Tag: webis-gpt-5
Generation-only? No
Document retrieval model description:
    Retrieve up to 10 PubMed articles from Elasticsearch using BM25.
    As the query, use the concatenated question, narrative, and simple yes-no, factual, or list answer (if question type is known and if retrieval is run after generation).
    Stopwords are removed from the query.
    Match the query on the article's abstract text (Elasticsearch BM25).
    Exclude PubMed articles with empty abstract text.
    After retrieval, split passages from the retrieved article's abstract text by splitting it into sentences and returning all sentence n-grams up to 3 sentences.
    Re-rank up to 50 passages pointwise with a monoT5 model (castorini/monot5-base-msmarco).
Answer generation model description:
    Generate a summary answer for each question with DSPy using a GPT-4o mini model (gpt-4o-mini-2024-07-18).
    Give the question and the top-3 passages to the model as context (numbered by rank), and prompt the model to return a summary answer with references (by rank) given in the text.
    Use the unoptimized prompt from DSPy.
    After generation, convert internal reference numbering back to PubMed IDs.
Short description:
    Use retrieval and generation as above, while cross-augmenting both retrieval and generation.
Details/comments:
    Using the retrieval and generation modules as described above, cross-augment the generation module with retrieval and the retrieval module with generation (i.e., in each step, the previous retrieval result is used to augment the next generation step and vice-versa).
    Do 2 cross-augmentation steps.
    Retrieval is implemented using PyTerrier.
    Generation is implemented using DSPy.
Judging precedence: 0 (run effectively uses the same config as webis-gpt-1)

### webis-gpt-6

Tag: webis-gpt-6
Generation-only? No
Document retrieval model description:
    Retrieve up to 10 PubMed articles from Elasticsearch using BM25.
    As the query, use the concatenated question, and simple yes-no, factual, or list answer (if question type is known and if retrieval is run after generation).
    Stopwords are not removed from the query.
    Match the query on the article's abstract text (Elasticsearch BM25).
    Exclude PubMed articles with empty title.
    Exclude PubMed articles of non-peer-reviewed publication types.
    After retrieval, use only the title as a passage for the article.
    Re-rank up to 10 passages pointwise with a monoT5 model (castorini/monot5-base-msmarco).
    Re-rank up to 3 passages pairwise with a duoT5 model (castorini/duot5-base-msmarco).
Answer generation model description:
    Generate a summary answer for each question with DSPy using a GPT-4o mini model (gpt-4o-mini-2024-07-18).
    Give the question and the top-3 passages to the model as context (numbered by rank), and prompt the model to return a summary answer with references (by rank) given in the text.
    Using DSPy, optimize the prompt by labeled few-shot prompting with 1 example from the BioASQ 12b train set.
    After generation, convert internal reference numbering back to PubMed IDs.
Short description:
    Use retrieval and generation as above, while augmenting both retrieval and generation independently.
Details/comments:
    Using the retrieval and generation modules as described above, independently augment the generation module with retrieval and the retrieval module with generation.
    For generation-augmented retrieval, augment 2 times while feeding back retrieval results to the generation module.
    For retrieval-augmented generation, augment 1 time while not feeding back generation results to the retrieval module.
    Retrieval is implemented using PyTerrier.
    Generation is implemented using DSPy.
Judging precedence: 3
Submitted!

### webis-gpt-7

Tag: webis-gpt-7
Generation-only? Yes
Document retrieval model description:
    Not used.
Answer generation model description:
    Generate a summary answer for each question with DSPy using a GPT-4o mini model (gpt-4o-mini-2024-07-18).
    Give the question to the model and prompt the model to return a summary answer.
    Using DSPy, optimize the prompt by labeled few-shot prompting with 3 examples from the BioASQ 12b train set.
Short description:
    Use generation as above, while not augmenting the generation.
Details/comments:
    Generation is implemented using DSPy.
Judging precedence: 0 (run effectively uses the same config as webis-gpt-1)

### webis-gpt-8

Tag: webis-gpt-8
Generation-only? Yes
Document retrieval model description:
    Not used.
Answer generation model description:
    Generate a summary answer for each question with DSPy using a GPT-4o mini model (gpt-4o-mini-2024-07-18).
    Give the question to the model and prompt the model to return a summary answer.
    Using DSPy, optimize the prompt by labeled few-shot prompting with 3 examples from the BioASQ 12b train set.
Short description:
    Use generation as above, while not augmenting the generation.
Details/comments:
    Generation is implemented using DSPy.
Judging precedence: 0 (run effectively uses the same config as webis-gpt-1)

### webis-gpt-9

Tag: webis-gpt-9
Generation-only? Yes
Document retrieval model description:
    Not used.
Answer generation model description:
    Generate a summary answer for each question with DSPy using a GPT-4o mini model (gpt-4o-mini-2024-07-18).
    Give the question to the model and prompt the model to return a summary answer.
    Using DSPy, optimize the prompt by labeled few-shot prompting with 3 examples from the BioASQ 12b train set.
Short description:
    Use generation as above, while not augmenting the generation.
Details/comments:
    Generation is implemented using DSPy.
Judging precedence: 0 (run effectively uses the same config as webis-gpt-1)

### webis-gpt-10

Tag: webis-gpt-10
Generation-only? Yes
Document retrieval model description:
    Not used.
Answer generation model description:
    Generate a summary answer for each question with DSPy using a GPT-4o mini model (gpt-4o-mini-2024-07-18).
    Give the question to the model and prompt the model to return a summary answer.
    Using DSPy, optimize the prompt by labeled few-shot prompting with 3 examples from the BioASQ 12b train set.
Short description:
    Use generation as above, while not augmenting the generation.
Details/comments:
    Generation is implemented using DSPy.
Judging precedence: 0 (run effectively uses the same config as webis-gpt-1)
