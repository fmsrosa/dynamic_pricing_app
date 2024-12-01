# Gender Classification App from Names Using BiLSTM and Attention Mechanism (Pytorch | Streamlit)

## Project Overview

This project is a two-part system for gender classification from names. It leverages a **Bidirectional LSTM** (BiLSTM) combined with a simple **Attention Mechanism**, using **PyTorch** for training and prediction.

### First Project: Gender Classification with BiLSTM and Attention Mechanism (PyTorch)

Classified names by gender using a Bidirectional LSTM combined with a simple attention mechanism, using PyTorch.

The dataset consists of first names, which are encoded into integer sequences. The model processes these sequences to predict gender, with the output being a probability of male or female.

Performance is evaluated using metrics like accuracy, F1 score, and ROC AUC.

Training is available at notebooks/01.0-fmsrosa-gender-classification-bilstm-attention.ipynb or at Kaggle.

---

### Second Project: Streamlit App for Name Gender Prediction

The second part of the project provides a **Streamlit** web application, which allows users to input a name and predict its gender using the previously trained model.

#### Features:
1. **Tokenizer**: 
   - Encodes names from strings to tensor representations (using integer sequences).
   - Decodes tensors back to strings.
2. **Model Loading**: 
   - Loads the trained BiLSTM model from the first project.
3. **Streamlit Web App**: 
   - A user-friendly interface where users can input a name, get the gender prediction, and view the results.

---

## How to Use

### Prerequisites

Make sure you have **Poetry** and **pyenv** installed. If you don't have **Poetry**, you can install it using the following command:

```bash
curl -sSL https://install.python-poetry.org | python3 -
```

For `pyenv` follow the instructions at [https://github.com/pyenv/pyenv?tab=readme-ov-file#installation](https://github.com/pyenv/pyenv?tab=readme-ov-file#installation).

### Installation

Set local Python environement and install Poetry environment
```bash
pyenv local 3.12
poetry install
```

### Execution
Run Streamlit app with
```bash
poetry run streamlit run app/app.py 
```
---

## Repository structure

```
├── .gitignore                                    <- Directories and files to be ignored by git
├── .poetry.lock                                  <- Current poetry settings.
├── .pre-commit-config.yaml                       <- Settings for pre-commit
├── .python-version                               <- Current Python version
│
│
├── app
│   └── app.py                                    <- Streamlit App
│
├── model
│   ├── bidirectional_attention_lstm_model.pth    <- Model state dict
│   ├── bidirectional_attention_lstm_model.py     <- Script to load model
│   ├── char_to_int.pkl                           <- Pickle dump encoding characters to integers.
│   ├── tokenizer.pkl                             <- Pickle dump of tokenizer.
│   └── tokenizer_creation.py                     <- Script to create tokenizer
│
├── name_to_gender_app                            <- Package folder
│   └── __init__.py                               <- Init file representing package.
│    
│    
├── notebooks                                     <- Jupyter notebooks. Naming convention is a zero followed by a number,
│                                                    if the number has single digit, or just the number if it has more
│                                                    than one digit, followed by the creator's initials, and a short
│                                                    `-` delimited description, e.g.`1.0-jqp-initial-data-exploration`.
│
├── pyproject.toml                                <- Project configuration file with package metadata.
│
│
└── README.md                                     <- The top-level README for developers using this project.
```
