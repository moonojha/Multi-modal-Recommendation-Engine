# Multimodal E-commerce Recommendation Engine

A content-based product recommendation system that combines image and text embeddings to find visually and semantically similar fashion products.


## Overview

This project builds a recommendation engine over a subset of the H&M Personalized Fashion Recommendations dataset, combining computer vision (product images) and NLP (product descriptions) into a shared embedding space to power similarity-based recommendations.


## Tech Stack

- **Image embeddings:** PyTorch, pretrained ResNet50 (ImageNet weights, final classification layer removed)

- **Text embeddings:** `sentence-transformers` (`all-MiniLM-L6-v2`)

- **Projection / shared space:** TensorFlow/Keras, trained via contrastive loss

- **Retrieval:** scikit-learn `BallTree`

- **Data:** H&M Personalized Fashion Recommendations (Kaggle)




- **Dataset subset:** the full dataset includes 105K+ products and a 28GB image folder. Rather than downloading everything, this project uses `articles.csv` plus a stratified sample of ~965 products (8 per category across 131 product categories), with images selectively downloaded via the Kaggle API for just those sampled products. This keeps the project reproducible without requiring tens of GB of storage.

- **No collaborative filtering / purchase history:** `customers.csv` and `transactions_train.csv` were deliberately excluded. This is a **content-based** recommender (image + text similarity), not a personalized/behavioral one.

- **Contrastive learning for shared embedding space:** since image embeddings (2048-dim) and text embeddings (384-dim) live in incompatible spaces, two small projection networks were trained with a contrastive loss (similar in spirit to CLIP) so that each product's image and text projections land close together in a shared 256-dim space, while different products' embeddings are pushed apart.

- **Fusion by averaging:** each product's final embedding is the average of its trained image and text projections which is a simple fusion strategy.

- **No trained ranking model:** the original plan included a deep ranking model on top of BallTree retrieval. This was deliberately scoped out because a proper learned ranker needs real interaction/click/purchase signals to train on meaningfully, which this project intentionally excludes. Retrieval via BallTree on the shared embedding space is the final recommendation step.


## Running Locally

**1. Install dependencies**

pip install -r requirements.txt

**2. Set up Kaggle credentials**

Get an API token at [kaggle.com/settings](https://www.kaggle.com/settings), place it at `~/.kaggle/kaggle.json`.

**3. Download data**

kaggle competitions download -c h-and-m-personalized-fashion-recommendations -f articles.csv
unzip articles.csv.zip -d data/

Run the notebook to sample products and selectively download corresponding images (avoids downloading the full 28GB image set).

**4. Run the notebook**

jupyter notebook notebooks/your_notebook.ipynb

This runs the full pipeline: image embedding, text embedding, projection training, and retrieval.


## Evaluation

Recommendations were spot-checked qualitatively across 7 query products spanning different categories, checking whether the top-5 retrieved products were visually and semantically similar to the query.

**Results were generally strong and thematically coherent:**

| Query | Category | Top Recommendations |
|---|---|---|
| Bag Simon Rolltop | Bag | Backpack, Cross-body bag, Backpack, Weekend/Gym bag ×2 — all bags |
| Knot Bitter Top | Bikini top | Bikini top ×2, Swimwear bottom, Swimwear set ×2 — all swimwear |
| Small basic scrunchie | Hair ties | Hair clip, Hair string ×2, Hair ties ×2 — all hair accessories |
| PE TYRA BOTTOM | Underwear bottom | Swimwear bottom ×3, unlabeled ×2 — all "bottoms" |
| Flirty Cora Pouch | Other accessories | Wallet ×3, Other accessories, Cross-body bag — all small accessories |
| Beau shorts | Shorts | Shorts ×2, Skirt, Underwear bottom, Swimwear bottom — all lower-body garments |

Across these 6 queries, recommendations consistently stayed within the correct **broad product theme** (bags, swimwear, hair accessories, bottoms, small accessories), even when the exact subcategory label differed (e.g. "Bag" → "Backpack"/"Cross-body bag" rather than literally "Bag"). This suggests the trained embedding space captures meaningful higher-level product semantics rather than memorizing narrow labels.

**One notable outlier:** a query for "RUBBER ANIMALS" (Accessories) returned a mix of soft toys and cosmetics sharing an "animal" theme in their names, rather than staying within accessories. This suggests the model can sometimes latch onto a strong thematic signal (e.g. shared wording or visual motif) that overrides category boundaries which is an interesting edge case rather than a failure, but worth noting as a case where thematic similarity and category similarity diverge.


## Limitations & Next Steps

- **No formal quantitative evaluation** — current evaluation is a manual inspection of retrieved recommendations. A category-match-rate metric (similar to Precision@k in my other [RAG project](https://github.com/moonojha/RAGAIAssistant/blob/main/notebooks/AI_Research_Assistant_RAG.ipynb)) or a proper offline evaluation using held-out interaction data would give a more rigorous benchmark.
- **No trained ranking model** — retrieval currently relies purely on nearest-neighbor search in the shared embedding space. A learned re-ranker (using real click/purchase signals, if available) would likely improve result ordering.
- **Small subset (~965 products)** — the full dataset has 105K+ products, and scaling up the sample size would improve recommendation diversity and better test the system's robustness across categories.
- **Simple average fusion** — combining image and text embeddings via a straight average is a simple baseline. however, a learned fusion layer could better weight each modality's contribution.
- **No train/validation split during contrastive training.** With ~965 samples, training loss dropped to 0.076 over 20 epochs which is a strong fit, but without a held-out validation set, it's not possible to fully rule out overfitting to this specific sample rather than learning generalizable cross-modal similarity. Qualitative recommendation results were reassuring on this front, but a proper validation split would be needed to confirm generalization.


## Dataset

[H&M Personalized Fashion Recommendations](https://www.kaggle.com/competitions/h-and-m-personalized-fashion-recommendations) — Kaggle competition dataset.
