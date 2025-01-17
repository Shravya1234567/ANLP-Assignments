# Advanced NLP | Assignment 1  

This implements three different types of language models using PyTorch: a Neural Network-based Language Model, an RNN-based Language Model using LSTM, and a Transformer Decoder-based Language Model.

Each model is trained and evaluated on the **Auguste_Maquet** corpus, using **GloVe 100d** pre-trained embeddings. The performance of the models is measured using perplexity scores on both training and test data.

### pre-processing

**[preprocess_utils.py](./preprocess_utils.py)**: Helper functions for tokenization, splitting data, handling unknown words, and loading GloVe embeddings.

### Neural Network-based Language Model (5-gram Context)
- **[NNLM.py](./NNLM.py)**: Main script that trains and tests the Neural Network language model.
- **[NNLM_utils.py](./NNLM_utils.py)**: Utility functions for data preprocessing, model training, and evaluation.

#### Training the Model

To train the NNLM on the preprocessed data, run:

```bash
python NNLM.py
```

#### Model Evaluation

Train, test and validation perplexity scores are computed and saved in separate text files:

- `2021101051-LM1-train-perplexity.txt`
- `2021101051-LM1-val-perplexity.txt`
- `2021101051-LM1-test-perplexity.txt`
---

### RNN-based Language Model using LSTM
- **[LSTM.py](./LSTM.py)**: Main script that trains and tests the LSTM language model.
- **[LSTM_utils.py](./LSTM_utils.py)**: Utility functions for data preprocessing, model training, and evaluation.

#### Training the Model

To train the LSTM model run:

```bash
python LSTM.py
```

#### Model Evaluation

Train, test and validation perplexity scores are computed and saved in separate text files:

- `2021101051-LM2-train-perplexity.txt`
- `2021101051-LM2-val-perplexity.txt`
- `2021101051-LM2-test-perplexity.txt`

---

### Transformer Decoder-based Language Model
- **[transformer.py](./transformer.py)**: Main script that trains and tests the Transformer Decoder language model.
- **[transformers_utils.py](./transformer_utils.py)**: Utility functions for data preprocessing, model training, and evaluation.

#### Training the Model

To train the Transformer Decoder model, run:

```bash
python transformer.py
```

#### Model Evaluation

Perplexity scores are computed for the training, validation, and test datasets and saved in separate text files:

- `2021101051-LM3-train-perplexity.txt`
- `2021101051-LM3-val-perplexity.txt`
- `2021101051-LM3-test-perplexity.txt`

---

### Trained Models
Link to the trained models: [link](https://iiitaphyd-my.sharepoint.com/:f:/g/personal/kukkapalli_shravya_students_iiit_ac_in/Eqaa20F7Vr5Nhylg71-Pf_UBWUbsemcLyPgtxVM_n3-vWQ?e=qn3bqw)