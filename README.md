# VTU Deep Learning Lab

This repository contains all 8 experiments from the VTU Deep Learning Laboratory (2022 Scheme, 7th Semester), each executed locally and committed along with its outputs (console logs and figures) as proof of execution.

## Student

Name: Rakshith N
Email: rakshithn6485@gmail.com

## Repository structure

Each experiment lives in its own folder with the program and its executed outputs.

```
program-01  Word embeddings (Word2Vec)
program-02  Deep neural network classification (Iris)
program-03  CNN image classification (MNIST)
program-04  Autoencoder compression (MNIST)
program-05  Text document classification (IMDB)
program-06  Time series forecasting (LSTM)
program-07  Pre-trained model classification (CIFAR-10 with MobileNetV2)
program-08  Review sentiment classification (TF-IDF + Logistic Regression)
```

## How to run

```
pip install -r requirements.txt
python program-01/program_01_word_embeddings.py
```

Each program is standalone and can be run individually from inside its folder.

## Experiments and results

| # | Program | Dataset / Approach | Executed Result |
|---|---------|--------------------|-----------------|
| 1 | Word embeddings (Word2Vec) | Toy document corpus | Generates 50 dimension vectors, similarity ranking matches the manual |
| 2 | Deep neural network classification | Iris (Keras DNN) | Test Accuracy 100.00 percent |
| 3 | CNN image classification | MNIST (Keras CNN) | Test Accuracy 99.22 percent (manual reported 99.28) |
| 4 | Autoencoder compression | MNIST with 32 unit bottleneck | Reconstruction loss 0.119, (10000, 784) compressed to (10000, 32), figure saved |
| 5 | Text document classification | IMDB reviews (Keras Embedding network) | Test Accuracy 85.78 percent |
| 6 | Time series forecasting | Synthetic sine wave (Keras LSTM) | Validation loss around 0.002, three figures saved |
| 7 | Pre-trained model classification | CIFAR-10 using MobileNetV2 with ImageNet weights | Test Accuracy 45.00 percent on CPU subset |
| 8 | Review sentiment classification | Toy reviews, TF-IDF with Logistic Regression | Correctly classifies all 3 new reviews as in the manual |

## Notes

- Program 7 was executed on a CPU laptop using an 8000 image training subset of CIFAR-10 (seed 42) with a streaming data pipeline. The model, pre-trained weights, classification head and 5 epochs are exactly as in the manual. Training on the full 50000 images on GPU or Colab reproduces the accuracy reported in the manual.
- Program 8 follows the manual exactly. The toy split leaves 2 stratified test reviews, and the manual's own output shows the same Positive, Positive, Negative behaviour on the new reviews.
- Executed on Windows with Python 3.13, TensorFlow 2.21.0 (CPU) and gensim 4.4.0.
- Console outputs were captured live from each run, and figures were saved headlessly using the matplotlib Agg backend.
- Programs differ from the manual only where needed to run on a CPU laptop: verbose level 2 for cleaner logs, figures saved to disk instead of plt.show, and the subset adjustments in Program 7 described above.
