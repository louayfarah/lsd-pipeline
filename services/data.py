# Load data
# Preprocess data
# Clean data

import pandas as pd
import numpy as np

# 1) Load English and French data
df_en = pd.read_csv('data/openbook_qa_en.csv', index_col=0)
df_fr = pd.read_csv('data/openbook_qa_fr.csv', index_col=0)

# 2) Preprocess English: drop missing, strip, dedupe
df_en = (
    df_en
    .dropna(subset=['question_stem', 'answerKey'])
    .assign(question_stem=lambda d: d['question_stem'].str.strip())
)

# 3) Compute stem_length and sort
df_en['stem_length'] = df_en['question_stem'].str.len()
df_en = df_en.sort_values('stem_length').reset_index(drop=True)

# 4) Select 1000 evenly‐spaced rows over sorted index
N = len(df_en)
indices = np.linspace(0, N - 1, 1000).astype(int)
sample_en = df_en.iloc[indices].reset_index(drop=True)

# 5) Save the English sample
sample_en.to_csv('data/openbook_qa_en_sample1000_by_length.csv', index=False)

# 6) Filter French by the same IDs, preserving order
#    (assumes 'id' is a column in both frames)
ids = sample_en['id'].tolist()
df_fr = (
    df_fr
    .dropna(subset=['question_stem', 'answerKey'])
    .assign(question_stem=lambda d: d['question_stem'].str.strip())
)
# select and reindex
sample_fr = (
    df_fr.set_index('id')
         .loc[ids]             # pick rows in same order
         .reset_index()
)

# 7) Compute stem_length and sort
sample_fr['stem_length'] = sample_fr['question_stem'].str.len()

# 7) Save the French sample
sample_fr.to_csv('data/openbook_qa_fr_sample1000_by_length.csv', index=False)

print("English sample:", sample_en.shape, "Question length: ", sample_en['stem_length'].describe())
print("French sample:", sample_fr.shape, "Question length: ", sample_fr['stem_length'].describe())
