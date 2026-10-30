# App-review-sentiment-analysis-group-project

**Tools**: Python, Keras, GloVe, LSTM, Conv1D

We (a group of 4) built this project to predict 1-5 star ratings of Google Play reviews. Each member scraped one app and built their own models. This repo contains **my** work (TikTok reviews).

## My contribution
- Scraped 200,000+ TikTok reviews with `google_play_scraper`
- Built 2 models: a Conv1D model and a two-branch Keras Functional API model merging a GloVe-LSTM branch with a Conv1D branch.
- Individual portions in the report and the model analysis on its limitations and proposed improvements.

## Shared pipeline (team)
Text cleaning, a 10,000-word vocabulary with 100-token sequences, class capping to reduce the skew in ratings, and a stratified 80/20 split. 

## Results (5-class, chance = 20%)
| Model | Validation | Test |
|---|---|---|
| Conv1D | 58.25% | 57.86% |
| GloVe-LSTM + Conv1D (frozen GloVe) | 59.45% | 59.18% |

**Cross-app testing:** my models were also tested on teammates' Netflix,
Telegram and Spotify sets. Scores there looked higher, but those sets were
dominated by 1 and 5 star reviews, so accuracy alone overstates performance.

## Limitations and next steps
- Weak on 2 and 3 star reviews and on sarcasm.
- Next: better class balancing and context-aware models.
