import pandas as pd
import re
import time
import os
from sentence_transformers import SentenceTransformer
from transformers import pipeline
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import torch
import glob

# --- GLOBAL SETTINGS ---

#See the full data
pd.set_option('display.max_columns', None)
pd.set_option('display.max_colwidth', None)
DEVICE='cuda' if torch.cuda.is_available() else 'cpu'

# --- UTILITIES ---

#Clearn nan values from the dataframe
def clean_nan_values(df,df_name='dataframe'):
    total_nans=df.isna().sum().sum()

    #print(f'Total NaNs in {df_name}: {total_nans} before cleaning')

    df = df.fillna('').astype(str)

    remaining_nans=df.isna().sum().sum()
    #print(f'Total NaNs in {df_name}: {remaining_nans} after cleaning')

    return df

def combine_results(companies_path='ml_insurance_challenge.csv',output_dir='outputs',output_name='companies_combined.csv'):
    companies=pd.read_csv(companies_path)
    companies['insurance_label']=[[] for _ in range(len(companies))]

    csv_files=[f for f in glob.glob(os.path.join(output_dir,'*.csv')) if '_partial' not in f]
    if not csv_files:
        print(f'[Combine] No CSV files found in {output_dir}')
        return companies

    print(f'[Combine] Found {len(csv_files)} CSV files in {output_dir}')

    for file_path in csv_files:
        print(f'[Combine] Processing {file_path}')
        df=pd.read_csv(file_path)
        label_cols=[c for c in df.columns if 'labels' in c]
        if not label_cols:
            continue

        for col in label_cols:
            for i in range(len(companies)):
                labels=df.at[i, col]
                if pd.isna(labels) or labels=='':
                    continue
                if isinstance(labels,list):
                    companies.at[i,'insurance_label'].extend(labels)
                else:
                    cleaned=str(labels).replace('[', '').replace(']', '').replace("'", '').split(',')
                    cleaned=[x.strip() for x in cleaned if x.strip()]
                    companies.at[i,'insurance_label'].extend(cleaned)

    companies['insurance_label'] = companies['insurance_label'].apply(lambda x: sorted(set(x)))

    output_path = os.path.join(output_dir, output_name)
    companies.to_csv(output_path, index=False)
    print(f'[Combine] Combined dataset saved to {output_path}')

def count_unlabeled_companies(combined_path='outputs/companies_combined.csv'):
    if not os.path.exists(combined_path):
        print(f"[Count] File not found: {combined_path}")
        return None

    df=pd.read_csv(combined_path)

    if 'insurance_label' not in df.columns:
        print(f"[Count] 'insurance_label' column not found in {combined_path}")
        return

    def parse_labels(val):
        if pd.isna(val) or val=='':
            return []
        if isinstance(val,list):
            return val
        val=str(val).strip()
        val=val.replace('[', '').replace(']', '').replace("'", '')
        parts=[x.strip() for x in val.split(',') if x.strip()]
        return parts

    df['insurance_label'] = df['insurance_label'].apply(parse_labels)

    total_companies=len(df)
    unlabeled_count=sum(len(labels)==0 for labels in df['insurance_label'])
    labeled_count=total_companies-unlabeled_count

    print(f"[Count] Total companies: {total_companies}")
    print(f"[Count] Labeled companies: {labeled_count}")
    print(f"[Count] Unlabeled companies: {unlabeled_count}")
    print(f"[Count] {unlabeled_count/total_companies:.2%} of companies have no labels.")


# --- DATA PREPROCESSING ---

GENERIC_TERMS=['services','installation','construction','manufacturing','and','processing']

#We remove repetitive words from the labels that might affect the algorithm
def clean_labels(df, column='label'):
    pattern=r"\b("+"|".join(GENERIC_TERMS)+r")\b"
    df=df.copy()
    df['clean_label']=df[column].astype(str).apply(
        lambda x: re.sub(pattern, "", x, flags=re.IGNORECASE).strip()
    )
    df['clean_label'] = df['clean_label'].str.replace(r"\s{2,}", " ", regex=True)
    return df

def row_to_text(row):
    parts=[]
    for col,val in row.items():
        if pd.isna(val):
            continue
        text=str(val).strip()

        text=text.replace('[', '').replace(']', '')
        text=text.replace("'","").replace("'","")
        text=' '.join(text.split())

        if text:
            parts.append(text)
    return '|'.join(parts).lower()

def build_company_text(company):
    parts=[]
    if company['description']:
        parts.append(company['description'])
    if company['business_tags']:
        parts.append(f"The company has the following business tags: {company['business_tags']}.")
    if company['sector']:
        parts.append(f'The company has the following sector: {company["sector"]}.')
    if company['niche']:
        parts.append(f'The company has the following niche: {company["niche"]}.')
    if company['category']:
        parts.append(f'The company has the following category: {company["category"]}.')
    return ' '.join(parts)

# --- METHODS ---

#keyword labeling
def run_keyword_search(companies, taxonomy, output_path):

    taxonomy_clean=clean_labels(taxonomy)
    labels_col=[]

    for i, row in companies.iterrows():

        company_text=str(row['full_text']).replace('|', ' ').replace(',', ' ')
        company_tokens=company_text.split()

        #Precompute company unigrams and bigrams for fast lookup
        def ngrams_local(tokens,n):
            return [' '.join(tokens[i:i+n]) for i in range(len(tokens)-n+1)]

        comp_uni=set(company_tokens)
        comp_bi=set(ngrams_local(company_tokens,2))

        rows=[]
        for _,trow in taxonomy_clean.iterrows():
            label_raw=str(trow['clean_label']).strip().lower()
            if not label_raw:
                continue

            ltoks=label_raw.split()
            label_len=len(ltoks)
            comp_ngrams=set(ngrams_local(company_tokens, label_len))

            # whole label
            whole_hit=1 if ' '.join(ltoks) in comp_ngrams else 0

            # label bigrams
            lbigrams=set(ngrams_local(ltoks, 2)) if len(ltoks)>=2 else set()
            bigram_hits=sum(1 for b in lbigrams if b in comp_bi)

            # label unigrams
            lunigrams=[u for u in ltoks if u not in GENERIC_TERMS]
            unigram_hits=sum(1 for u in set(lunigrams) if u in comp_uni)

            score=3*whole_hit+2*bigram_hits+min(unigram_hits,3)

            rows.append({
                "label": trow.get("label", label_raw),
                "score": score
            })

        df_scores=pd.DataFrame(rows).sort_values('score',ascending=False)
        labels=df_scores.loc[df_scores['score']>=2, 'label'].tolist()
        labels_col.append(labels)

        if i%1000==0:
            print(f'[Keyword] Processed{i}/{len(companies)} companies')

    companies['labels_keyword']=labels_col
    companies.to_csv(output_path, index=False)
    print(f'[Keywords] Saved results to {output_path}')

#embeddings similarity method
def run_embeddings_method(companies,taxonomy,output_path):
    model_name='sentence-transformers/all-MiniLM-L6-v2'
    model=SentenceTransformer(model_name,device=DEVICE)

    taxonomy_embeddings=model.encode(taxonomy['label'].tolist(),convert_to_numpy=True)
    company_embeddings=model.encode(companies['full_text'].tolist(),convert_to_numpy=True)

    sim_matrix=cosine_similarity(company_embeddings,taxonomy_embeddings)
    top_n=5
    threshold=0.45
    labels_col=[]
    for i,sims in enumerate(sim_matrix):
        top_idx=np.argsort(sims)[::-1][:top_n]
        labels=[taxonomy.iloc[j]['label'] for j in top_idx if sims[j]>=threshold]
        labels_col.append(labels)

    companies['labels_embedding']=labels_col
    companies.to_csv(output_path, index=False)
    print(f'[Embedding] Saved results to {output_path}')

#zero shot transformer-based method
def run_model_method(companies,taxonomy,output_path):
    model_name='facebook/bart-large-mnli'
    classifier=pipeline('zero-shot-classification',model=model_name,device=DEVICE)

    label_list=taxonomy['label'].tolist()
    batch_size=32
    threshold=0.1
    results=[]
    total_batches=(len(companies)+batch_size-1)//batch_size

    start=time.time()

    for batch_num,start_idx in enumerate(range(0,len(companies),batch_size),start=1):
        end_idx=min(start_idx+batch_size, len(companies))
        batch=companies.iloc[start_idx:end_idx]
        company_texts=[build_company_text(row) for _, row in batch.iterrows()]
        batch_results=classifier(company_texts,candidate_labels=label_list,hypothesis_template='This company provides {}.')

        if isinstance(batch_results, dict):
            batch_results=[batch_results]

        for result in batch_results:
            labels=[l for l,score in zip(result['labels'],result['scores']) if score>=threshold]
            results.append(labels)

        elapsed=time.time()-start
        remaining_batches=total_batches-batch_num
        eta_minutes=(elapsed/batch_num)*remaining_batches/60
        print(f'[Model] Batch {batch_num}/{total_batches} | Elapsed time: {elapsed/60:.1f} minutes | Remaining: ~{eta_minutes:.1f} minutes')

        #Save partial progress
        companies_partial=companies.copy()
        companies_partial['labels_model']=results+[[]]*(len(companies)-len(results))
        companies_partial.to_csv(output_path.replace('.csv','_partial.csv'), index=False)
        print(f'[Model] Saved partial results to {output_path} after batch {end_idx}')

    companies['labels_model']=results
    companies.to_csv(output_path, index=False)
    print(f'[Model] Saved results to {output_path}')

def main(methods=('keyword','embeddings','model'),
         taxonomy_path='insurance_taxonomy - insurance_taxonomy.csv',
         companies_path='ml_insurance_challenge.csv',
         output_dir='outputs'):

    print('=== Insurance Labeling Interface ===')

    taxonomy=clean_nan_values(pd.read_csv(taxonomy_path),'taxonomy')
    companies=clean_nan_values(pd.read_csv(companies_path),'companies')

    if 'full_text' not in companies.columns:
        companies['full_text']=companies.apply(row_to_text, axis=1)

    os.makedirs(output_dir, exist_ok=True)

    start=time.time()

    for method in methods:
        print(f'Running method: {method}')
        output_path=os.path.join(output_dir,f'companies_labels_{method}.csv')

        if method=='keyword':
            run_keyword_search(companies.copy(),taxonomy,output_path)
        elif method=='embeddings':
            run_embeddings_method(companies.copy(),taxonomy,output_path)
        elif method=='model':
            run_model_method(companies.copy(),taxonomy,output_path)
        else:
            print(f'Unknown method: {method}')

    total_time=time.time()-start
    print(f'\nFinished all selected methods in {total_time/60:.1f} minutes')

if __name__=='__main__':
    main(methods=('embeddings',))
    combine_results()
    #uncomment if you want to see how man companies were labeled
    #count_unlabeled_companies()