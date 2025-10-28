# Company-Classifier
Automated company labeling pipeline using keyword matching, sentence embeddings, and zero-shot classification with transformers.

How to Run
1. Run a specific method

Open the main Python file (e.g., labeling_pipeline.py) and call the main function with your preferred method(s):

```python
if __name__ == "__main__":
    main(methods=('embeddings',))
    combine_results()
    count_unlabeled_companies()
```


Available methods:

"keyword" -> rule-based keyword matching

"embeddings" -> semantic similarity with SentenceTransformer

"model" -> zero-shot classification using BART (slowest, most powerful)

To run all three methods:

main(methods=('keyword', 'embeddings', 'model'))


Each method will generate a separate CSV file in the outputs/ folder. Then a csv file combining them will be generated.
