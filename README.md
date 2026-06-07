# Spotify Lyric Sentiment Engine

A supervised Machine Learning pipeline utilizing Random Forests and NLP vectorization to classify the emotional valence (positiveness) of music lyrics using a dataset of 900,000+ Spotify tracks.

## Technical Architecture Overview
* **Data Scale:** Balanced subset of 450,000 tracks optimized to eliminate classification bias.
* **Feature Engineering:** Customized tokenization pipeline paired with a 1,500-feature TF-IDF Vectorizer.
* **Classification Engine:** Parallelized Random Forest Classifier optimizing macro-F1 performance.

## Getting Started
1. Clone the repository.

```bash
git clone [https://github.com/yourusername/spotify-sentiment-engine.git](https://github.com/yourusername/spotify-sentiment-engine.git)
cd spotify-sentiment-engine
pip install -r requirements.txt
```

2. Place `spotify_dataset.csv` in a `data/` directory.
3. Run `python main.py`.


