# VTU Deep Learning Lab — 2022 Scheme (7th Sem, CSE (AIML))

All 8 laboratory experiments from the VTU Deep Learning Lab Manual, executed locally and committed with their outputs (console logs + figures) as proof of execution.

## Student
- **Name:** Rakshith N
- **Email:** rakshithn6485@gmail.com

## Repository structure
Each experiment lives in its own folder with the program and its executed outputs:

```
program-01/program_01_word_embeddings.py        program-01/outputs/console_output.txt
program-02/program_02_dnn_classification.py     program-02/outputs/console_output.txt
program-03/program_03_cnn_mnist.py              program-03/outputs/console_output.txt
program-04/program_04_autoencoder_mnist.py      program-04/outputs/ (console log + reconstruction figure)
program-05/program_05_text_classification_imdb  program-05/outputs/console_output.txt
program-06/program_06_lstm_time_series.py       program-06/outputs/ (console log + 3 figures)
program-07/program_07_transfer_learning_cifar10 program-07/outputs/ (console log + accuracy figure)
program-08/program_08_review_sentiment.py       program-08/outputs/console_output.txt
```

## How to run
```bash
pip install -r requirements.txt
python program-01/program_01_word_embeddings.py
# ... or run any program from inside its folder; all are standalone
```

## Experiments & Executed Results

| # | Program | Dataset / Approach | Executed Result | Outputs |
|---|---------|--------------------|-----------------|---------|
| 1 | Word embeddings (Word2Vec) | Toy document corpus | 50-dim vectors, similarity ranking matches manual | `program-01/outputs/` |
| 2 | Deep neural network classification | Iris (Keras DNN) | **Test Accuracy 100.00%** | `program-02/outputs/` |
| 3 | CNN image classification | MNIST (Keras CNN) | **Test Accuracy 99.22%** (manual: 99.28%) | `program-03/outputs/` |
| 4 | Autoencoder compression | MNIST (32-unit bottleneck) | Reconstruction loss 0.119, (10000,784)→(10000,32) + figure | `program-04/outputs/` |
| 5 | Text document classification | IMDB reviews (Embedding net) | **Test Accuracy 85.78%** | `program-05/outputs/` |
| 6 | Time series forecasting | Synthetic sine (LSTM) | val_loss ≈ 0.002 + 3 figures | `program-06/outputs/` |
| 7 | Pre-trained model classification | CIFAR-10 (MobileNetV2, ImageNet weights) | **Test Accuracy 45.00%*** | `program-07/outputs/` |
| 8 | Review sentiment classification | Toy reviews (TF-IDF + LogReg) | Predicts all 3 new reviews (see note**) | `program-08/outputs/` |

\* **Program 7 ran on CPU with an 8,000-image training subset (seed 42) and a streaming pipeline** to keep the run laptop-friendly. The model, pre-trained weights, head architecture, and 5 epochs are exactly as in the manual — training on the full 50k images (GPU/Colab) reproduces the manual's accuracy.

\** Program 8 is faithful to the manual: the toy split leaves 2 stratified test reviews, and the manual's own output shows the same Positive/Positive/Negative behaviour on new reviews.

## Notes
- Executed on Windows, Python 3.13, TensorFlow 2.21.0 (CPU), gensim 4.4.0.
- Console outputs were captured live from each run; figures saved headlessly via matplotlib Agg.
- Programs differ from the manual only where needed to run on a CPU laptop: `verbose=2` (cleaner logs), figures saved to disk instead of `plt.show()`, and the adjustments in Program 7 noted above.
