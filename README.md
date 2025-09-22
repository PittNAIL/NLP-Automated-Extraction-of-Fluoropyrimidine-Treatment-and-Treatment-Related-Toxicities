# Automated Extraction of Fluoropyrimidine Treatment & Toxicities from Clinical Notes

> Rule-based • Classical ML • Deep Learning (BERT/ClinicalBERT) • LLM (zero-shot & error-analysis prompting)

This repository hosts code accompanying the manuscript:

**Automated Extraction of Fluoropyrimidine Treatment and Treatment-Related Toxicities from Clinical Notes Using Natural Language Processing**  
Xizhi Wu, Madeline S. Kreider, Philip E. Empey, Chenyu Li, Yanshan Wang

---

## Table of Contents
- [Overview](#overview)
- [Tasks & Labels](#tasks--labels)
- [Pipelines](#pipelines)
  - [Rule-based (MedTagger)](#rule-based-medtagger)
  - [Machine Learning (LR / SVM / RF)](#machine-learning-lr--svm--rf)
  - [Deep Learning (BERT / ClinicalBERT)](#deep-learning-bert--clinicalbert)
  - [LLM (Zero-shot / Error-analysis prompting)](#llm-zero-shot--error-analysis-prompting)
- [License](#license)
- [Acknowledgements](#acknowledgements)

---

## Overview

We compare four families of NLP methods for extracting **fluoropyrimidine (FP) treatment** and **treatment-related toxicities** from oncology clinical notes:

1. **Rule-based** with MedTagger (regex + custom context/negation rules).  
2. **Classical ML**: Logistic Regression (LR), Linear SVM, Random Forest (RF) using bag-of-words / TF-IDF.  
3. **Deep Learning**: BERT and ClinicalBERT sentence classifiers.  
4. **LLM prompting**: LLaMA-3.1-8B (local) with **zero-shot** and **error-analysis prompting**.

All methods use the same 80:20 train–test split and are evaluated with **precision, recall, and (weighted) F1**.

---

## Tasks & Labels

Binary sentence-level classification across **five categories**:

- **Drug of interest** (capecitabine, 5-FU; brand names; combination regimens & abbreviations)
- **Arrhythmia**
- **Heart Failure (HF)**
- **Valvular Complications**
- **HFS treatment/prevention therapies** (topicals & uridine triacetate)

See `resources/keywords/` for curated terminology lists and `resources/prompts/` for LLM prompts.

---

## Pipelines

1. **Rule-based** with MedTagger

A deterministic NLP pipeline built with regular expressions and custom context rules (negation, uncertainty, experiencer). Clinical experts curated terminology lists, and rules were tuned on 80% of the dataset. MedTagger executed these regex rules to extract mentions of fluoropyrimidine treatment and toxicities.

2. **Classical ML (LR / SVM / RF)** 

A supervised classification approach with one binary model per toxicity category. Text is preprocessed (tokenization, normalization) and transformed into features:

TF-IDF with sublinear scaling for linear models (Logistic Regression, SVM).

Count-based n-grams for Random Forest.
Class weighting addressed imbalance, and models were evaluated using precision, recall, and F1.

3. **Deep Learning (BERT / ClinicalBERT)**

Neural classifiers using transformer-based embeddings. Input sentences are tokenized and passed through pretrained BERT or ClinicalBERT. Each category is trained independently, with ClinicalBERT providing domain-specific adaptation. Predictions are converted into binary outputs and evaluated against the gold standard.

4. **LLM prompting (LLMs: Zero-shot & Error-analysis prompting)**

Inference-only classification with LLaMA 3.1 8B using prompt engineering:

Zero-shot prompting: binary classification prompts with terminology lists and explicit yes/no outputs.

Error-analysis prompting: improved prompts with chain-of-thought reasoning examples, created by analyzing systematic misclassifications. These enhanced prompts helped LLMs better capture indirect or nuanced clinical evidence.