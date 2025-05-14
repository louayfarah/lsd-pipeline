# Query Complexity & Latency Analysis in English–French QA

This repository contains code to investigate how query linguistic complexity affects response time in a bilingual (English/French) open‐book QA setting. We build on the OpenBookQA dataset, computing multifactor complexity metrics, measuring prompt‐evaluation times with two open‐source LLMs, and analyzing correlations and trade‐offs.

---

## 🚀 Getting Started

### 1. Clone & Install


git clone https://github.com/louayfarah/lsd-pipeline.git
cd lsd-pipeline
pip install -r requirements.txt


Make sure you have Docker installed for LibreTranslate (if translating) and Ollama running locally (or point to your Ollama server).

### 2. Download & Prepare NLP Models


python -m spacy download en_core_web_sm
python -m spacy download fr_core_news_sm
python -u src/scripts/nltk_setup.py


This will fetch SpaCy and NLTK WordNet resources.

---

## 🎯 Pipeline Overview

All steps are orchestrated in run_pipeline.sh.

---

## 🔧 Configuration

* LibreTranslate: If you wish to re‐translate English → French, start a Docker container:

  
  docker run -d -p 5000:5000 libretranslate/libretranslate
  
* Ollama client: Ensure Ollama is installed and your models (`llama3.2:3b`, `deepseek-r1:7b`) are available locally.

---

## 📈 Analysis & Results

After running the pipeline, open the Jupyter notebooks:

* analysis_en.ipynb
* analysis_fr.ipynb

They cover:

1. MCQ prompt‐evaluation time vs. complexity
2. Fact prompt‐evaluation time vs. complexity
3. Model‐to‐model latency comparisons
4. Accuracy vs. complexity
5. Token‐count effects
6. Complexity decile trends


---

## 📜 Citation

If you use this work, please cite:

> Farah, L., & Daassi, M. A. (2025). *Investigating Query Complexity and Prompt Evaluation Latency in English–French QA with OpenBookQA.*

---

## ❓ Contact

* Louay Farah <[l.farah@innopolis.university](mailto:l.farah@innopolis.university)>
* Mohamed Aymen Daassi <[m.daassi@innopolis.university](mailto:m.daassi@innopolis.university)>

Happy exploring! 🚀