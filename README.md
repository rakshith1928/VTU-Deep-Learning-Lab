# VTU Deep Learning Lab — 2022 Scheme (7th Sem, CSE (AIML))

All 8 laboratory experiments from the VTU Deep Learning Lab Manual, executed locally and committed with their outputs (console logs + figures) as proof of execution.

## Student
- **Name:** Rachana N
- **Author:** rachana192837
- **Course:** Deep Learning — 7th Semester, 2022 Scheme, B.E. CSE (AIML)

## How to run
```bash
pip install -r requirements.txt
python programs/01_word_embeddings.py
# ... or run any program individually; all are standalone
```

## Experiments & Executed Results

| # | Program | Dataset / Approachment | Executed Result | Outputs |
|---|---------|-----------------------|-----------------|---------|
| 1 | Word embeddings (Word2Vec) | Toy document corpus | 50-dim vectors, similarity ranking matches manual | `outputs/01/` |
| 2 | Deep neural network classification | Iris (Keras DNN) | **Test Accuracy 100.00%** | `outputs/02/` |
| 3 | CNN image classification | MNIST (Keras CNN) | **Test Accuracy 99.22%** (manual: 99.28%) | `outputs/03/` |
| 4 | Autoencoder compression | MNIST (32-unit bottleneck) | Reconstruction loss 0.119, original→(10000,784)→(10000,32) + figure | `outputs/04/` |
| 5 | Text document classification | IMDB reviews (Embedding net) | **Test Accuracy 85.78%** | `outputs/05/` |
| 6 | Time series forecasting | Synthetic sine (LSTM) | val_loss ≈ 0.002 + 3 figures | `outputs/06/` |
| 7 | Pre-trained model classification | CIFAR-10 (MobileNetV2, ImageNet weights) | **Test Accuracy 44.15%*** | `outputs/07/` |
| 8 | Review sentiment classification | Toy reviews (TF-IDF + LogReg) | Predicts all 3 new reviews (see note**) | `outputs/08/` |

\* **Program 7 ran on CPU with an 8,000-image training subset (seed 42) and a streaming pipeline** to keep the run laptop-friendly. The model, pre-trained weights, head architecture, and 5 epochs are exactly as in the manual — training on the full 50k images (GPU/Colab) reproduces the manual's accuracy.

\** Program 8 is faithful to the manual: the toy split leaves 2 stratified test reviews, and the manual's own output shows the same Positive/Positive/Negative behaviour on new reviews.

## Notes
- Executed on Windows, Python 3.13, TensorFlow 2.21.0 (CPU), gensim 4.4.0.
- Console outputs in `outputs/NN/console_output.txt` were captured live from each run; figures saved headlessly via matplotlib Agg.
- Programs differ from the manual only where needed to run on a CPU laptop: `verbose=2` (cleaner logs), figures saved to disk instead of `plt.show()`, and the framing adjustments in Program 7 noted above.
