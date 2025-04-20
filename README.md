# NLP-Deep-Learning-Neural-Machine-Translations-RNN

## Overview
This repository builds a sequence‑to‑sequence (Seq2Seq) neural machine translation (NMT) system with attention. Using Python and PyTorch, I construct a bidirectional LSTM encoder, unidirectional LSTM decoder, and multiplicative global attention (Luong et al., 2015) to translate Cherokee to English.

## Key Features & Highlights
- **Custom Embedding Layer** (`model_embeddings.py`): Initialized source and target `nn.Embedding` modules with proper handling of padding tokens, demonstrating mastery of PyTorch’s embedding API.
- **Seq2Seq Architecture with Attention** (`nmt_model.py`): Implemented a bidirectional LSTM encoder, LSTMCell decoder, and global attention mechanism (projection, score computation, context vector, and combined output), showcasing skills in deep RNN modeling.
- **Data Utilities** (`utils.py`): Developed functions to read SentencePiece‑tokenized corpora, pad batches of variable‑length sentences, and iterate batches in length‑sorted order for efficient training.
- **Training Workflow** (`run.sh`): Orchestrated GPU‑accelerated training with early stopping (patience = 1, max trials = 5), reporting cumulative loss, perplexity, and validation evaluation at each epoch.
- **Evaluation Metrics**: Measured held‑out performance via development perplexity and final BLEU score on the test set, demonstrating quantitative analysis of translation quality.

## Results
| Metric                  | Value             |
|-------------------------|------------------:|
| Development Perplexity  |            39.39  |
| Test BLEU Score         |            12.07  |

*Trained for 7 epochs with early stopping after 5 trials.*

## Project Structure
- **src/submission/model_embeddings.py**: Defines `ModelEmbeddings` for source and target tokens.  
- **src/submission/nmt_model.py**: Contains `NMT` class with `encode`, `decode`, and `step` methods implementing encoder‑decoder and attention. Includes `beam_search` for inference.  
- **src/submission/utils.py**: Utility functions: `read_corpus` (SentencePiece loading), `pad_sents`, `batch_iter`, and BLEU computation.  
- **src/submission/__init__.py**: Exposes model classes and helper functions.  
- **run.sh**: Shell script to generate vocab (`vocab`), train on CPU/GPU, validate, and test, producing `test_outputs.txt` for submission.  

## Insights
- **Attention Effectiveness**: The global attention mechanism substantially improves translation by focusing decoder predictions on relevant source words.  
- **Early Stopping for Efficiency**: With patience = 1 and max trials = 5, training halted at epoch 7, preventing overfitting and saving GPU time.  
- **BLEU and Perplexity Trade-offs**: While a low dev perplexity (39.4) shows the model predicts individual tokens reliably, the modest BLEU score (12.1) underscores that token-level accuracy doesn’t always translate to fluent, coherent sentences. Closing this gap often requires not just further training (e.g. deeper networks, larger embeddings) but also enhancements at the sequence level—such as beam‑search tuning, coverage mechanisms, or length normalization—to better align token‑level confidence with end‑to‑end translation quality.

## Assignment Requirements Met
- **Development Perplexity**: Required ≤45; achieved 39.39 (lower is better).
- **Test BLEU Score**: Required ≥10; achieved 12.07 (higher is better).
- Completed all required coding tasks and generated `test_outputs.txt` via `run.sh test_gpu`, fulfilling submission deliverables.

