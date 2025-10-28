# Company-Classifier
Automated company labeling pipeline using keyword matching, sentence embeddings, and zero-shot classification with transformers.

The results are found in outputs/companies_combined.csv

## How to Run

Open the main Python file and call the main function with your preferred method(s):

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
```python
main(methods=('keyword', 'embeddings', 'model'))
```

Each method will generate a separate CSV file in the outputs/ folder. Then a csv file combining them will be generated.

## Changing File Names or Paths
If your input CSV files have different names or locations, modify these parameters in the main() function call:

```python
main(
    methods=('embeddings',), 
    taxonomy_path='your_taxonomy_file.csv', 
    companies_path='your_companies_file.csv', 
    output_dir='your_output_folder'
)
```
### Performance Tips
"keyword" method takes around 2 minutes to label my dataset

"embeddings" method takes around 18 seconds to label my dataset

"model" method takes around 2 days to label my dataset. Best for smaller datasets.
