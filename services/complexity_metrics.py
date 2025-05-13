# Calculate complexity metrics
import numpy as np
import pandas as pd
import nltk
from nltk.corpus import wordnet as wn
import re
import spacy
from transformers import AutoModel, AutoTokenizer
import torch
import textstat

from util.query_templates import make_mcq_prompt

# load English model once
nlp_en = spacy.load("en_core_web_sm")

# load French model once
nlp_fr = spacy.load("fr_core_news_sm")

# load the model and tokenizer once
tokenizer = AutoTokenizer.from_pretrained("xlm-roberta-base")
model = AutoModel.from_pretrained("xlm-roberta-base")


# Semantic complexity metrics

def polysemy_score(text, lang="en"):
    tokens = [w.lower() for w in re.findall(r"\w+", text) if w.isalpha()]

    synset_counts = []
    for t in tokens:
        if lang == "en":
            syns = wn.synsets(t)
        elif lang == "fr":
            syns = wn.synsets(t, lang="fra")
        else:
            syns = []

        if syns:
            synset_counts.append(len(syns))

    # Return total polysemy (or 0 if none)
    return float(sum(synset_counts)) if synset_counts else 0.0


def named_entity_score(text, lang="en"):

    nlp = nlp_en if lang == "en" else nlp_fr
    doc = nlp(text)
    ne_count = len(doc.ents)
    # normalize by token count to get density
    token_count = len([tok for tok in doc if not tok.is_punct])
    return ne_count if token_count else 0.0


def embedding_dispersion(text, lang="en", truncation=True, max_length=512):
    # Tokenize (truncated to model’s limit)
    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=truncation,
        max_length=max_length,
    )
    with torch.no_grad():
        out = model(**inputs, output_hidden_states=True)

    # Extract embeddings: (seq_len, dim)
    embeddings = out.last_hidden_state[0]

    # Normalize each embedding to unit length
    norms    = embeddings.norm(dim=1, keepdim=True)
    emb_norm = embeddings / norms

    # Cosine similarity matrix: (seq_len, seq_len)
    sim_matrix = emb_norm @ emb_norm.T

    # Convert to distances (1 - cosine similarity)
    dist_matrix = 1.0 - sim_matrix

    # Mask out the diagonal
    n = dist_matrix.size(0)
    mask = ~torch.eye(n, dtype=torch.bool)

    # Return all off-diagonal distances as a flat tensor
    return dist_matrix[mask]



# Grammatical complexity metrics

def readability_scores(text, lang="en"):
    return {
        "flesch_kincaid_grade": textstat.flesch_kincaid_grade(text),
        "gunning_fog": textstat.gunning_fog(text),
        # add others as you like
    }


def clause_count(text, lang="en"):
    nlp = nlp_en if lang == "en" else nlp_fr
    doc = nlp(text)
    # count tokens that head a SBAR subordinate clause
    subords = sum(1 for tok in doc if tok.dep_ == "mark")
    # total clauses ≈ number of finite verbs
    finite_verbs = sum(1 for tok in doc if tok.pos_ == "VERB" and tok.morph.get("VerbForm") == ["Fin"])
    return {"subordinate_clauses": subords, "finite_verbs": finite_verbs}



# Syntactic complexity metrics

def max_dep_depth(text, lang="en"):
    nlp = nlp_en if lang == "en" else nlp_fr
    doc = nlp(text)
    def depth(tok):
        return 1 + (depth(tok.head) if tok.head != tok else 0)
    return max(depth(tok) for tok in doc)


def avg_dep_length(text, lang="en"):
    nlp = nlp_en if lang == "en" else nlp_fr
    doc = nlp(text)
    dists = [abs(tok.i - tok.head.i) for tok in doc if tok.dep_ != "ROOT"]
    return sum(dists) / len(dists) if dists else 0.0



# Main function to compute complexity

def compute_complexity(
        text, 
        idx,
        lang="en"
):
    # Semantic
    poly_score = polysemy_score(text, lang)
    ne_score   = named_entity_score(text, lang)
    dists = embedding_dispersion(text, lang)
    max_dist = dists.max().item()
    mean_dist = dists.mean().item()
    sem_score = np.mean([poly_score, ne_score, max_dist, mean_dist])

    # Grammatical
    rd = readability_scores(text, lang)
    rd_flesch = rd["flesch_kincaid_grade"]
    rd_gunning = rd["gunning_fog"]
    clauses = clause_count(text, lang)
    clauses_subord = clauses["subordinate_clauses"]
    clauses_finite = clauses["finite_verbs"]
    gram_score = np.mean([rd_flesch, 
                          rd_gunning, 
                          clauses_subord, 
                          clauses_finite])

    # Syntactic
    max_depth = max_dep_depth(text, lang)
    avg_length = avg_dep_length(text, lang)
    syn_score = np.mean([max_depth, avg_length])

    print("Finished computing complexity for index:", idx)
    return {
        "semantic": sem_score,
        "grammatical": gram_score,
        "syntactic": syn_score,
    }


def minmax_normalize(series):
    return (series - series.min()) / (series.max() - series.min())


def augment_complexity(input_csv, output_csv, lang="en"):
    df = pd.read_csv(input_csv)
    
    # 1) compute raw scores
    # TODO: compute complexity of all prompt not just question_stem, using make_mcq_prompt(stem, choices)
    for idx, row in df.iterrows():
        # compute complexity for question_stem
        stem = row["question_stem"]
        obj = eval(row["choices"], {"array": np.array})
        texts = obj["text"].tolist()
        labels = obj["label"].tolist()
        choices = dict(zip(labels, texts))
        prompt = make_mcq_prompt(stem, choices)
        complexity = compute_complexity(prompt, idx, lang)

        df.at[idx, "semantic"] = complexity["semantic"]
        df.at[idx, "grammatical"] = complexity["grammatical"]
        df.at[idx, "syntactic"] = complexity["syntactic"]
        # compute humanScore and clarity
        df.at[idx, "humanScore"] = row["humanScore"]
        df.at[idx, "clarity"] = row["clarity"]
        df.at[idx, "mcq_prompt_number_tokens"] = len(prompt.split())
    
    # 2) normalize each complexity column
    for col in ("semantic","grammatical","syntactic"):
        df[f"{col}_norm"] = minmax_normalize(df[col])
    
    # 3) normalize humanScore & clarity
    df["humanScore_norm"] = minmax_normalize(df["humanScore"])
    df["clarity_norm"]    = minmax_normalize(df["clarity"])
    
    # 4) combined complexity
    # weights: semantic .3, grammatical .3, syntactic .3, humanScore .05, clarity .05
    df["combined_complexity"] = (
        0.3*df["semantic_norm"]
      + 0.3*df["grammatical_norm"]
      + 0.3*df["syntactic_norm"]
      + 0.05*df["humanScore_norm"]
      + 0.05*df["clarity_norm"]
    )
    
    # Sort by combined complexity
    df = df.sort_values("combined_complexity").reset_index(drop=True)

    # 5) Select 100 rows evenly spaced over sorted index
    N = len(df)
    indices = np.linspace(0, N - 1, 100).astype(int)
    df = df.iloc[indices].reset_index(drop=True)
    
    # 6) Save the augmented data
    df.to_csv(output_csv, index=False)
    print(f"Written augmented data to {output_csv}")


augment_complexity(
    "data/openbook_qa_en_sample1000_by_length.csv",
    "data/openbook_qa_en_sample100_by_complexity.csv",
    lang="en"
)
augment_complexity(
    "data/openbook_qa_fr_sample1000_by_length.csv",
    "data/openbook_qa_fr_sample100_by_complexity.csv",
    lang="fr"
)