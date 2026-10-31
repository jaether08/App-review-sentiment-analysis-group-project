**Table of Contents**
1. Overview
2. Data Collection and Processing (Group)
3. Develop Sentiment Analysis Models
4. Model Evaluation Using Testing Data
5. Make Prediction with Best Model
6. Summary + Improvements
---
# 1. Overview

(Jaydon)

The objective is to develop and compare deep learning models that can
predict a user's Google Play Store star rating, from 1 to 5 stars, based
only on the text input in a review. This makes the task more challenging
because the model has to distinguish between five different levels of
sentiment rather than only three broad categories. For example, a review
that deserves a 4-star rating may contain mostly positive comments but
also mention a few minor issues, while a 5-star review is usually much
more enthusiastic. Predicting the exact rating allows the model to
capture these subtle differences in sentiment.

The assignment was completed as a group, with each team member
collecting reviews from a different mobile application on the Google
Play Store. The four selected applications were Netflix, Spotify,
Telegram and TikTok. Although all four datasets contain app reviews,
each application has different features, user expectations and common
complaints. For example, Netflix reviews often discuss video streaming
quality and subscription pricing, while Spotify reviews are more likely
to focus on music recommendations and advertisements. Telegram reviews
may mention messaging features, privacy or updates, whereas TikTok
reviews frequently discuss content recommendations, videos and user
experience. By using different applications, our group was able to build
separate models and later compare how well they performed on reviews
from other apps instead of only the app they were originally trained on.

The assignment followed the ML universal workflow to ensure consistency
throughout the development process. The first stage involved collecting
real review data directly from the Google Play Store using the
google_play_scraper Python package. This approach allowed the group to
work with recent and realistic user reviews instead of relying on an
existing cleaned dataset.

To keep evaluation fair between team members, the group agreed on a
shared cleaning baseline, though each member applied it a little
differently depending on what their dataset needed. Most datasets had
URLs, HTML entities and repeated characters cleaned out before
vectorisation, and reviews with no useful text left after cleaning were
removed. For the Netflix dataset, cleaning was handled by
TextVectorization itself, which lowercases text and strips punctuation
as part of building its vocabulary, so a separate cleaning function
wasn't needed on top of that.

After cleaning, each team member converted the text into numerical form,
mostly using Keras' TextVectorization layer. Deep learning models cannot
process raw text directly, so each review had to be transformed into
sequences of numbers representing individual words. These sequences were
then padded to a fixed length so that every input had the same
dimensions before being used for training. Each dataset was also divided
into training and testing sets, allowing model performance to be
evaluated on unseen data.

The next stage focused on building and improving several deep learning
models independently. Each of us followed the same development strategy
by first creating a simple baseline model before gradually increasing
the model complexity. Once signs of overfitting appeared, different
regularisation techniques such as dropout, L2 regularisation and Early
Stopping were introduced to improve generalisation. Hyperparameters
including learning rate, batch size, sequence length and network
architecture were also tuned to achieve better performance. Different
architectures were explored, including convolutional neural networks,
recurrent neural networks and models using either pretrained GloVe word
embeddings or embeddings learned during training. This systematic
process allowed each model to be fairly compared while understanding how
different design choices affected performance.

The final stage of the project involved evaluating every model on both
its own test dataset and the test datasets collected by the other team
members. Before making predictions on another dataset, the reviews were
cleaned using the same preprocessing steps and encoded using the
vectoriser or tokenizer belonging to the model being evaluated. This
cross-app evaluation provided a more realistic assessment of how well
each model could generalise beyond the application it was trained on. A
model that performs well only on its own dataset may simply have learned
patterns specific to that application, while a model that performs
consistently across different apps demonstrates a stronger understanding
of the relationship between review text and user ratings. The
best-performing model was selected using the macro F1 score because it
evaluates all five rating classes equally, even when the dataset
contains class imbalance.

# 2. Data Collection and Processing (Group)

## Why this set of apps was selected

Our team selected four different apps — Telegram, TikTok, Netflix, and
Spotify — one per team member (Lakshya, Jaether, Jaydon, and Alex
respectively). These apps were chosen because they span four distinct
app categories: messaging, short-form social video, video streaming, and
music streaming. This was intentional. A sentiment analysis model
trained only on reviews from a single app risks learning patterns
specific to that app's user base and complaint types (e.g. "buffering"
for a streaming app, or "chat bugs" for a messaging app) rather than
generalizable sentiment language. By each working on a different
category, the group could compare how well similar modeling approaches
transferred across genuinely different review domains, and
cross-evaluate each other's trained models in Step 3 against data they
hadn't been trained on.

Picking four different types of apps also gave the team a fair way to
test if a model built for one kind of app could still work on a totally
different kind of app. If everyone had picked apps from the same
category, like four different games, the reviews might all use similar
words and complaints. This would make it harder to know if a model was
actually learning about sentiment in general, or just learning the
specific words used in that one category. By spreading out across four
very different categories, the team made sure that any model that did
well across all four apps was probably picking up on real sentiment
signals, not just memorising one type of complaint.

Within that shared rationale, each member also had an app-specific
reason for their pick. Alex chose Spotify because the music-streaming
community spans a huge range of genres and use cases, which was expected
to produce a correspondingly diverse range of sentiment and complaint
types to train on. The other three apps were chosen primarily to fill
out the four categories and were pulled using the google-play-scraper
package, giving each member raw review text paired with its
corresponding 1–5 star rating.

Once each member had picked their app, the next step was actually
pulling the review data down from the Google Play Store. This was done
using a Python package called google-play-scraper, which lets you
request reviews for any app just by knowing its app ID. Each review that
came back included the text the user wrote, along with the star rating
they gave, from 1 to 5. This gave every team member the same basic shape
of data to work with, just for a different app, which made it much
easier to later combine everyone's cleaning and preprocessing code into
one shared pipeline.

| **Score** | **Lakshya (Telegram)** | **Jaydon (Netflix, before cleaning)** | **Jaether (TikTok)** | **Alex (Spotify)** |
|----|----|----|----|----|
| 1 | 79,087 | 41,700 | 24,115 | 22,907 |
| 2 | 8,427 | 5,963 | 5,349 | 5,055 |
| 3 | 8,141 | 5,711 | 6,304 | 5,674 |
| 4 | 10,604 | 6,550 | 6,694 | 9,625 |
| 5 | 93,941 | 40,267 | 57,738 | 56,937 |

Each collected app's raw reviews contained the free-text review body
(content) and the accompanying star rating (score, 1 to 5), pulled using
the google-play-scraper package. Each review's text was cleaned using a
shared function: the text was lowercased, then every character that was
not a lowercase letter or whitespace was replaced with a space using a
regular expression, followed by trimming leading and trailing
whitespace. This removed punctuation, digits, emojis, and non-Latin
characters, while deliberately leaving reviews containing non-English
text in the dataset rather than removing them, since real app reviews
are naturally multilingual and removing them would have made the dataset
less representative of the real-world data the model would eventually
need to handle.

This cleaning step matters a lot more than it might seem at first. Raw
review text from the Play Store is messy: people use capital letters
randomly, add emojis, mix in numbers, and sometimes barely use
punctuation at all. If this text were fed straight into a model without
any cleaning, the model would have to deal with a huge number of tiny,
meaningless differences, like the same word appearing as both "Good" and
"good", which a computer would otherwise treat as two completely
different words. Lowercasing and stripping out anything that isn't a
plain letter makes sure the model only has to deal with the actual words
themselves, not all the extra noise around them.

Once cleaned, the text was converted into numeric tensors using Keras's
TextVectorization layer. This layer was adapted directly on the cleaned
review text, building a vocabulary capped at the 10,000 most frequent
tokens (max_words), and used to convert every review into a fixed-length
sequence of 100 integer token IDs (maxlen), padding shorter reviews and
truncating longer ones. The corresponding star ratings, originally on a
1-5 scale, were shifted to a zero-indexed range and converted into
one-hot vectors using to_categorical, to match the five-unit softmax
output layer used by every model.

This step is needed because neural networks cannot read plain text
directly. A model only understands numbers, so every word first has to
be turned into a number before anything else can happen. The
TextVectorization layer handles this by looking at all the cleaned
reviews and building a list of the most common words, then giving each
one its own number. Any review can then be turned into a list of these
numbers, one number per word, in the same order the words appeared.
Since not every review is the same length, shorter reviews are padded
with extra zeros at the end, and longer reviews are cut off, so that
every review ends up being exactly the same length before it goes into a
model.

Finally, the data needed to be sampled before training. Raw review
counts pulled directly from the Play Store are naturally imbalanced,
with certain star ratings, typically 2, 3, and 4 stars, appearing far
less often than 1 and 5-star reviews. Rather than forcing every class to
an identical count, the balancing step first determined the smallest
class's review count, then capped every other class at three times that
number using a cap_multiplier of 3, or the maximum available if fewer
existed. This retains most of the dataset's natural shape while still
preventing the largest classes from completely dominating training,
striking a middle ground between a fully balanced dataset and the raw,
heavily skewed one. The combined, capped dataset was then shuffled
before the train/test split. This split reserved 20% of the data for
testing, using stratified sampling so that the same proportion of each
star rating present in the balanced dataset was preserved in both the
training and test sets, ensuring evaluation was not distorted by a
disproportionate class in either split.

Sampling the data in this way is important because most people who leave
app reviews are either very happy or very unhappy, so 1-star and 5-star
reviews are usually much more common than the middle ratings. If a model
were trained on this as-is, it could simply learn to guess the most
common rating most of the time and still look like it was doing well,
without ever really learning what makes a review positive, negative, or
somewhere in between. By capping the biggest classes and keeping a
stratified split, the team made sure every model saw a more even mix of
all five star ratings, and that the final testing step was a fair and
honest measure of how well each model could actually tell the different
ratings apart.

# 3. Develop Sentiment Analysis Models

Jaydon

All four models were trained on the same train/validation split, with
validation_split=0.2 held out from the training set, allowing training
and validation accuracy/loss curves to be tracked and compared across
every epoch.

**Model 1 — Bidirectional LSTM with pretrained GloVe embeddings**

**Baseline model**: The baseline used a single small Bidirectional
(LSTM(16)) layer on top of the GloVe embedding, with no dropout or L2
regularisation. It was trained for 10 epochs and reached a training
accuracy of 77.3%, but validation accuracy plateaued and then declined,
falling from a peak of 60.2% (epoch 3) down to 57.0% by epoch 10.
Validation loss bottomed out at 1.059 (epoch 3) then climbed steadily to
1.277, while training loss kept falling to 0.664, a clear, fast-widening
overfitting gap once the model had enough capacity to memorise the
training reviews.

**Regularisation attempts**:\
**Attempt 1**: Fixed the GloVe file path check so the embedding matrix
could never silently load as all-zeros, then scaled the architecture up
to two stacked Bidirectional LSTM layers (64 then 32 units) with L2
(0.001) and Dropout (0.5) between them.\
**Attempt 2**: Raised the Adam learning rate from 5e-4 to 1e-3 to speed
up convergence within the fixed epoch budget, and increased
EarlyStopping patience from 3 to 5 so the slower-converging recurrent
layers had more room before stopping.

This produced a training accuracy of 67.3% against a validation accuracy
of 58.7%, with training loss of 0.903 and validation loss of 1.199 by
the final epoch (training stopped early at epoch 7).

**Accuracy gap**: 8.6%\
**Loss gap**: -0.296

**Model 2 — 1D Convolutional network with embeddings trained from
scratch**

**Baseline model**: The baseline used a single small Conv1D (16,5) layer
with GlobalMaxPooling1D, no regularisation, and a from-scratch
embedding. It overfit even faster than Model 1's baseline: training
accuracy reached 87.2% by epoch 10 while validation accuracy peaked at
60.0% (epoch 2) and then dropped to 55.6%. Validation loss bottomed at
1.076 before rising sharply to 1.542, while training loss fell all the
way to 0.429 — the largest raw overfitting gap of any baseline, since a
from-scratch embedding with no regularisation had free rein to memorise
the training vocabulary.

**Regularisation attempts**:\
**Attempt 1**: Scaled up from a single small Conv1D(16) to two stacked
Conv1D(64) blocks with L2 (0.001) and Dropout (0.5) after each, plus an
intermediate Dense(64, L2=0.001) layer before the softmax output, giving
the model more capacity but with regularisation to keep it in check.\
**Attempt 2**: Added ReduceLROnPlateau (factor 0.5, patience 2)
alongside the same EarlyStopping patience-5 change as Model 1, so the
learning rate would automatically drop once validation loss plateaued
instead of overshooting.

This produced a training accuracy of 66.6% against a validation accuracy
of 57.8%, with training loss of 0.938 and validation loss of 1.217 by
the final epoch (training stopped early at epoch 7).

**Accuracy gap**: 8.8%\
**Loss gap**: -0.280

**Model 3 — Bidirectional GRU with pretrained GloVe embeddings**

**Baseline model**: The baseline used a single small Bidirectional
(GRU(16)) layer on the same GloVe embeddings as Model 1, with no
regularisation. Training accuracy reached 77.3% by epoch 10, while
validation accuracy peaked at 59.8% (epoch 2) then fell to 56.5%.
Validation loss bottomed at 1.066 before climbing to 1.333, a slightly
smaller gap than Model 1's LSTM baseline, hinting even at this small
scale that GRU cells might generalise a touch better on this dataset.

**Regularisation attempts**:\
**Attempt 1**: Same GloVe path fix as Model 1, then scaled up to two
stacked Bidirectional GRU layers (64 then 32 units) with L2 (0.001) and
Dropout (0.5), keeping the embedding choice constant to isolate the
effect of cell type (GRU vs. LSTM).\
**Attempt 2**: Raised the learning rate to 1e-3 and increased
EarlyStopping patience to 5, matching Model 1.

This produced a training accuracy of 68.7% against a validation accuracy
of 56.9%, with training loss of 0.877 and validation loss of 1.227 by
the final epoch (training stopped early at epoch 8).

**Accuracy gap**: 11.8%\
**Loss gap**: -0.350

**Model 4 — Hybrid Conv1D + Bidirectional LSTM with from-scratch
embeddings**

**Baseline model**: The baseline used a small Conv1D(16,5) +
MaxPooling1D(4) block feeding into a single Bidirectional (LSTM(16))
layer, no regularisation, from-scratch embedding. Training accuracy
reached 86.3% by epoch 10 while validation accuracy peaked at 59.7%
(epoch 1) then fell to 55.4%. Validation loss bottomed at 1.071 before
rising sharply to 1.809, the steepest loss divergence of any baseline,
since the hybrid architecture combines the memorisation capacity of both
a conv block and a recurrent layer with nothing to hold it back.

**Regularisation attempts**:\
**Attempt 1**: Scaled the Conv1D block from 16 to 64 filters with L2
(0.001), added Dropout (0.4) after pooling, and widened the LSTM from 16
to 32 units with its own L2 (0.001) regularisation and Dropout (0.5)
afterward.\
**Attempt 2**: Added a Dense(32, L2=0.001) + Dropout(0.5) block before
the output layer, and applied ReduceLROnPlateau plus the same
EarlyStopping patience-5 change used in Model 2.

This produced a training accuracy of 68.8% against a validation accuracy
of 58.4%, with training loss of 0.888 and validation loss of 1.234 by
the final epoch (training stopped early at epoch 7).

**Accuracy gap**: 10.4%\
**Loss gap**: -0.346

Across all four models, the baselines showed the expected fast, severe
overfitting (train accuracy 77–87% vs. a validation peak of only ~60%
within 1-3 epochs, then declining val accuracy from there), confirming
that unregularised models overfit quickly on this dataset regardless of
architecture. Regularisation (L2, Dropout, ReduceLROnPlateau,
EarlyStopping) closed a meaningful share of that gap in every case,
though a residual 8.6-11.8 percentage-point acc gap remained, consistent
with the finding that **Model 1** (smallest acc gap, 8.6%) and **Model
2** (smallest loss gap, -0.280) were the closest to generalising on
their own validation split, though as Section 4 shows, this didn't fully
predict which model generalised best across all four apps, motivating
macro F1 as the fairer selection metric rather than raw validation
accuracy alone.

Lakshya

**Model 1 – Embedding + Conv1D**

Model 1 was my simple baseline model that consists of an Embedding
layer, one Conv1D layer, GlobalMaxPooling1D, and a Dense(5, softmax)
output with no dropout or regularisation. I trained the model with
RMSprop for 10 epochs. This model was left untuned as it was meant to
show how badly an unregularised model overfits on this dataset. Training
accuracy climbed steadily throughout, reaching 76.15% by the final
epoch, while validation accuracy peaked around epoch 5 (60.53%) before
decreasing back down to 56.88% by epoch 10. Validation loss kept
climbing from around 1.06 to 1.1874 while training loss kept falling.

Acc gap: 19.27% \| Loss gap: -0.5036

Test accuracy came out to 55.42%, with a test loss of 1.1988. Based on
the per-class breakdown, 5-star reviews were the easiest to get right
with an F1 of 0.74, while 2-star and 3-star were the weakest (F1 of 0.14
and 0.23), showing the model struggles most with reviews that don't
distinctly show whether the person liked it or not.

**Model 2 - Embedding + Conv1D (Regularised)**

I attempted to address the overfitting issue in Model 1. Since Model 1
had no regularisation at all, three mechanisms were added together:
SpatialDropout1D(0.3) directly after the embedding layer (chosen over
normal Dropout since it zeroes out entire embedding dimensions rather
than scattered individual values, which works much better for text), L2
regularisation of 0.001 on the Conv1D layer, and Dropout(0.5).
EarlyStopping (patience=3, monitoring val_loss) was also added so
training would stop automatically once overfitting started happening. I
set epochs to 30, but EarlyStopping stopped it at epoch 11. At that
point, training accuracy was 59.60% and validation accuracy was 59.99%,
with a training loss of 1.0811 against a validation loss of 1.0749.

Acc gap: -0.39% \| Loss gap: 0.0062

This combination worked well on the first attempt, essentially closing
the gap completely, with validation performance matching or slightly
exceeding training. Test accuracy reached 59.19%, with a test loss of
1.0775. However, the per-class breakdown shows a real cost to this
regularisation: 2-star F1 dropped to just 0.01, meaning the model almost
never correctly identifies a 2-star review anymore.

**Model 3 - GloVe Embedding + LSTM**

This model's tuning focus was different from the others. Instead of
solving the overfitting issue, I wanted to find out if pre-trained GloVe
embeddings would give the model better accuracy than training embeddings
from scratch. In the architecture, I used a frozen 100-dimensional GloVe
embedding into an LSTM(64) with L2 regularizer (1e-4), Dropout(0.5), a
Dense(32) with L2 (1e-4), Dropout(0.3), and the softmax output, trained
for 25 epochs with a batch size of 128 and the same EarlyStopping
settings as Model 2.

I initially built a separate preprocessing pipeline for this model to
support the GloVe vocabulary, which caused a mismatch: the embedding
matrix was mapped to a different vocabulary than the one my actual
training data was tokenized with, meaning the pre-trained vectors didn't
correspond to the correct words at all. Fixing this — reusing the same
word_index and vocabulary as Models 1 and 2 instead of a separate
Tokenizer — made a real difference. EarlyStopping now stopped training
at epoch 11, with training accuracy at 57.61% and validation accuracy at
56.89%, training loss of 1.1062 against a validation loss of 1.1144.

Acc gap: 0.72% \| Loss gap: -0.0082

Test accuracy improved substantially after the fix, from 40.92% up to
56.74% (test loss 1.1136). That said, 2-star and 3-star F1 are still
both 0.00 even after the fix — this remaining collapse is likely no
longer about the vocabulary bug, but instead the combination of a frozen
embedding (which can't adapt to this specific dataset) with the
smallest, most imbalanced classes in the dataset.

**Model 4 - Bidirectional GRU**

Initially, there was a version with much more regularisation — a smaller
embedding, a smaller LSTM(32), L2 as high as 0.01, recurrent_dropout of
0.3, Dropout of 0.6, and a low learning rate of 1e-4, on the assumption
that a recurrent layer would need heavy regularisation to avoid
overfitting the way Model 1 had. This did not work — training and
validation accuracy both flattened around 19-20%, meaning the model was
just randomly guessing across five classes, with validation loss flat at
around 1.61. This showed the model had been regularised so much it was
unable to learn anything at all.

To fix this, every regularisation value was changed at once: the layer
became a Bidirectional GRU(64) (up from LSTM(32)), L2 was reduced to
0.0001, recurrent_dropout lowered to 0.2, Dropout lowered to 0.4, and
the learning rate raised to 1e-3 using Adam. This also reuses the same
corrected GloVe embedding matrix from Model 3, but with trainable=True,
letting the embedding fine-tune itself to this dataset's vocabulary
rather than staying frozen. EarlyStopping stopped training at epoch 9,
with training accuracy at 63.90% and validation accuracy at 61.44%,
training loss of 0.9824 against a validation loss of 1.0636.

Acc gap: 2.46% \| Loss gap: -0.0812

This confirmed the model could now actually learn, with both curves
climbing together. This model produced the best result of all four, with
test accuracy improving from 45.44% before the vocabulary fix to
**60.94%** after it, with a test loss of 1.0616, and was saved as the
final best model (text_model_best.h5).

**Group Evaluation Across Apps**

As part of the group's cross-evaluation, the final best model (Model 4)
was tested against each teammate's own held-out test set, since we each
collected reviews for a different app. The results were:

| **App**  | **Test Accuracy** | **Test Loss** |
|----------|-------------------|---------------|
| Netflix  | 61.87%            | 1.0660        |
| Telegram | 57.70%            | 1.1463        |
| TikTok   | 59.77%            | 1.1545        |
| Spotify  | 60.94%            | 1.0616        |

The model performs fairly consistently across all four apps, staying
within a roughly 4-point accuracy band (57.70%–61.87%), with Netflix
performing best and Telegram slightly weaker. This consistency is a good
sign — it suggests the model has learned genuinely transferable
sentiment patterns rather than overfitting to vocabulary specific to any
single app, though the gap toward Telegram may reflect that
messaging-app reviews use different language patterns than
entertainment/streaming apps.

Across all four models and all four apps, one pattern held throughout:
5-star reviews were consistently the easiest to classify, while 2-star
and 3-star reviews were consistently the hardest, regardless of
architecture or which app's data was being evaluated.

Jaether

Both models followed the universal workflow: develop a baseline model,
scale until it overfits, then regularise and tune hyperparameters to
close the train-validation gap. Both models were trained on the same
train/validation split. 30% of the training set was held out with
validation_split=0.3, allowing training and validation accuracy and loss
curves to be tracked and compared for every iteration.  

**Model 1** - Embedding + Conv1D (Sequential API)

**<u>Baseline model</u>:** The baseline model used a from-scratch
embedding layer (Embedding(32)) that was fed into one Conv1D(32, 7)
layer, GlobalAveragePooling1D, and an output Dense(5, softmax) layer. It
was trained for 10 epochs and reached a training accuracy of 65.89%, but
validation accuracy plateaued around 58.6%. Validation loss began
climbing from 1.1486 to 1.2390 after epoch 4, while training loss kept
falling. This is a clear sign of overfitting.

To address this, regularisation attempts were made. 

**<u>Attempt 1</u>**: Added Dropout layers of 0.3 and 0.5 and L2
regularisation of 0.001 around the smaller Conv1D(16) layer. This
considerably reduced the overfitting gap, but the final validation
accuracy came out to **58.25%** with a validation loss of **1.199**,
which is much closer to the training curve.

**<u>Attempt 2</u>**: I reduced the embedding dimension to 16 (from 32),
dropped the filters from 16 to 8, and increased dropout to 0.5 and 0.6
from 0.3 and 0.5, respectively. I also strengthened L2 regularisation to
0.005 and added EarlyStopping monitoring val_loss with a patience of 3.
This produced a **54.36%** training accuracy, **57.48%** validation
accuracy, training loss of **1.270**, and validation loss of **1.228**.
This stronger regularisation, however, pushed the model too far in the
other direction, as it resulted in underfitting. This told me that I had
implemented excessive regularisation. The next attempt will focus on
scaling up. 

**<u>Attempt 3</u>**: I partially restored the model’s capacity by
increasing the embedding dimension back to 32 and had 2 Conv1D layers
with 32, then 16 filters. L2 regularisation is back to 0.001 for both
Conv1D layers, and dropout is now set to 0.3 and 0.4. The model is
trained for up to 25 epochs with EarlyStopping of the same parameters as
the previous attempt. Results were: val_acc: **57.30%**, val_loss:
**1.250**. This is a mild but more acceptable overfitting. 

**Acc gap: ~0.85%** (58.15% - 57.30%)

**Loss gap: ~0.081** (1.1695 – 1.2503)

**<u>Attempt 4</u>**: Following attempt 3, the focus will now be to
close the train/validation gap. To achieve that, I reduced the filter in
the first Conv1D layer from 32 to 24. I used a strong L2 of 0.003 but a
higher dropout rate of 0.4 and 0.5. This gave a validation accuracy of
**56.05%**, a loss of **1.275**, and a smaller overfitting gap. 

**Acc gap: ~ -0.86%** (55.19% - 56.05%)

**Loss gap: ~ -0.031**

Across these attempts, the final epoch values of validation accuracy and
loss were compared directly. The best-performing regularised variant was
still Attempt 1 with a validation accuracy of 58.25% and validation loss
of 1.199, and it was saved as text_model_1.h5.

**<u>Model 2 - Functional API (GloVe + LSTM merged with Conv1D)</u>**

**<u>Baseline model</u>**: The second model utilises the Keras’
Functional API to build a two-branch architecture. Branch A made use of
GloVe’s 200-dimensional embeddings. It was fed into an LSTM (32,
recurrent_dropout=0.3). Branch B used two Conv1D layers with max
pooling. The two branches were concatenated, then passed through
Dropout, and an output Dense (5, softmax) layer with L2 regularisation
at 0.001. The model immediately overfits, with its training accuracy
exceeding validation accuracy. 

Regularisation Attempts:

**<u>Attempt 1</u>**: I shrank Branch B’s embedding dimension from 100
to 64 and added a dropout layer (0.35) before the first Conv1D layer.
This reduced the branch’s network capacity and led to a slight narrowing
of the train-validation gap. This attempt’s validation accuracy peaked
at **59.1%** (loss **1.146**). 

**<u>Attempt 2</u>:** I changed EarlyStopping to monitor val_acc instead
of val_loss, increased the patience to 3, and slightly increased the
last dropout to 0.4, aiming to stop the training at the point where the
model’s generalisation abilities peak instead of the lowest loss.
Validation accuracy peaked at epoch 7 at **59.1%** with a loss of
**1.142**.

**<u>Attempt 3</u>:** I froze the GloVe embedding. This left only the
LSTM and Branch B weights trainable. This was done to check if the GloVe
embedding made a difference to the model’s accuracy. This produced a
higher validation accuracy of **59.45%**, which outperformed Attempt 2.
This suggests that keeping the pre-trained vector fixed, rather than
letting it update freely, helped the model to generalise better.

**<u>Attempt 4</u>:** To further fine-tune the model, I chose to keep
the first keep the GloVe embedding frozen for 10 epochs, then unfreeze
it for another 10 epochs at a lower learning rate of 1e-5. This was done
to let the LSTM branch learn from the fixed pre-trained vector first
before letting it adapt to the review-specific vocabulary without
destroying its pre-trained structure. This generated a **59.18%**
validation accuracy in phase 1 and **59.09%** in phase 2.

Ultimately, after comparing all the attempts’ validation accuracy and
loss, attempt 3 was chosen and saved as the final Model 2
(\`text_model_2\`). This model achieved a <u>test</u> accuracy of
**59.45%** and a <u>test</u> loss of 1.1457.

Alex

<u>Overview</u>

Wanting to have some diversity within the models I was training, I
decided on comparing the performance of Recurrent Neural Networks (RNN)
against the performance of a 1 dimensional convolutional neural network
(CNN).

<u>Starting (baseline) models – discovering class imbalance</u>

I would begin by building and training a baseline RNN and Conv1D model,
making use of LSTM layers and their ability to retain memory of word
info, avoiding the vanishing gradient problem. At first the model showed
an extremely rigid learning process, with its validation accuracy
jumping from extreme low and high values (epoch 8 val_acc=0.45, epoch 9
val_acc=0.02, epoch 10 val_acc=0.6). I later found that this was due to
my highly imbalanced dataset, the model was fitting to the most frequent
class almost never predicting the less frequent classes. This would
result in huge jumps between predicting some of the 1 & 5 score
validation reviews on point but then getting the 2-4 score reviews
completely wrong. In the end the model showed a validation accuracy of
0.71, or so it seemed. Using more accurate metrics (e.g. precision,
recall, f1-score etc.) I found that both my models had a 0.00 recall and
0.00 precision on both scores 2 and 3.

<u>Fixing class imbalance with “compute_class_weight” addon</u>

Looking for ways I could distribute my dataset more evenly, I found
scikit learns “compute_class_weight” addon, a function that essentially
punishes a model for ignoring less frequency classes during training. At
the time my second model (1d CNN model) was the only model with dropout
and regularization, this was due to it largely overfitting during
baseline training (val_acc and train_acc split was 0.25). Once the
addition of the compute_class_weight function was introduced my CNN
model adjusted well (no extreme values, validation accuracy averaging
around 0.4), while my RNN model, began rigorously shooting between
values (epoch 9 val_acc = 0.55 epoch 10 val_acc=0.67). One thing I
failed to account for was the compute_class_weight function to introduce
large gradient distortions to any models without regularization. Though
this was an easy fix, just by adding regularization to my RNN model. I
didn’t trust the addon accurately guide my model, which is what lead me
to my second and final decision.

<u>Fixing class imbalance with selective class sampling</u>

Instead of trying to change the distribution of the dataset after
sampling it, I could instead just resample the original dataset in
whatever distribution I wanted. As said before, I would set the count of
the least frequent score as a threshold and cap the number of more
frequent scores samples using that threshold multiplied by a cap
multiplier (min_class_count\*3). This gave me a clean way to experiment
with a more skewed or less skewed distribution and it’s affect on my
models performance.

<u>Improving accuracy with more complex word vectors</u>

After that issue was resolved, my performance accuracies became more
realistic. However, they were considerably low, averaging around 0.4-0.5
per training. I would increase the maximum length of sentences to be
used (maxlen) from the original 20 to 100 to make my dataset and
training data more specific, giving way for my model to better fit to
data and spot its underlying patterns. Pairing this with increasing the
dimensionality of the vectors I was feeding my model, by converting from
Glove50d to Glove200d.

<u>Saving time with callbacks</u>

Despite all of the techniques and changes I made bringing the average
validation accuracy of my models to 0.5 and above, it was costing me a
heavy amount of computational power and time. That is when I made use of
callbacks to monitor my models performance during training, spotting
when the model wasn’t making any progress, and halting training, saving
the best model at the same time. This had saved me an enormous amount of
time and unnecessary computing power lost for no reason from both of my
models.

<u>Adding and comparing Gated Recurrent Units (GRU) with LSTM</u>

When thinking of ways to experiment further, I thought back to RNNs and
how GRUs were also an alternative to fixing the vanishing gradient
problem, different from LSTMs as they did not have a cell state making
them slightly faster when it came to extracting useful patterns and word
info. Creating a mirror copy of the LSTM model except with GRU layers
instead of LSTM. It’s training performance showed almost identical to
the LSTM with one major difference, it had reached the performance level
significantly faster (within 14 epochs compared to LSTM which took 22
epochs) proving what I had learnt.

# 4. Model Evaluation Using Testing Data

Jaydon

When tested on unseen Netflix review data, the four models produced
similar overall accuracy but differed noticeably on macro F1. Model 1
(BiLSTM + GloVe) achieved 60.32% accuracy and the highest macro F1 of
0.3776, while Model 2 (Conv1D + scratch embeddings) performed the worst
on macro F1 at 0.3003 despite a similar 59.54% accuracy. Model 3
(BiGRU + GloVe) reached 58.93% accuracy with a macro F1 of 0.3609, and
Model 4 (Conv1D + BiLSTM + scratch embeddings) reached 59.78% accuracy
with a macro F1 of 0.3294. Model 1 also achieved the highest mean macro
F1 across its own and all teammates' test sets (0.3792), ahead of Model
3 (0.3576), Model 4 (0.3353), and Model 2 (0.3087), showing the most
consistent generalisation across apps.

The classification reports showed a similar pattern for all models. They
classified 1-star and 5-star reviews well but struggled badly with the
2, 3, and 4-star classes, with 2-star F1 collapsing to 0.00 for every
model. These middle ratings contain more mixed sentiment and were less
represented in the dataset, making it much harder to classify
accurately.

Model 1 was chosen for further fine-tuning as Model 5, using a lower
learning rate (1e-4) so the pretrained weights were adjusted gently
rather than overwritten. This improved test accuracy to 61.74% and
raised macro F1 slightly to 0.3832, without the bias toward the majority
classes seen in earlier fine-tuning attempts. Since fine-tuning improved
macro F1 over the best-of-4 baseline, Model 5 was kept as the final
recommended model.

Lakshya

Across all four models, test performance closely tracked validation
performance, which helps to confirm that non of the results are
overfitting on the validation set. Model 4 (Bidirectional GRU) was the
clear best performer at 60.94% test accuracy and 1.0616 test loss, ahead
of Model 2 (59.19%), Model 3 (56.74%), and Model 1 (55.42%). Model 1’s
test accuracy was much lower Models 2 ,3 and 4 improved from the
regularisation and, for Models 3 and 4, correctly aligned GloVe
embeddings. A consistent weakness across every model was poor
performance on 2-star and 3-star reviews, metrics like ‘F1’ and ‘Recall’
helping to know whether the model accuracy is trustworthy or not, while
1-star and 5-star reviews were classified reliably throughout. In the
group cross-evaluation, Model 4 was tested against each teammate's own
app data, scoring 61.87% (Netflix), 60.94% (Spotify), 59.77% (TikTok),
and 57.70% (Telegram) which is a consistent 4-point range suggesting the
model generalises reasonably well across the different app reviews
rather than overfitting to any single one. Model 4 is recommended as the
best model overall, given its highest accuracy, lowest loss, and
demonstrated cross-app consistency.

Jaether

Both saved models were evaluated on the test set using model.evaluate().
Model 1 achieved a test accuracy of **57.86%** and a test loss of
**1.2107.** While Model 2 achieved a test accuracy of **59.18%** and a
test loss of **1.1457**.

Both models landed close to their respective validation accuracies.
Model 1’s validation was 58.25% against its test at 57.86%, and Model
2’s validation was 59.45% against its test at 59.18%. This indicates
that neither overfit significantly, with validation performance closely
matching test results. 

Comparing the two, Model 2 outperformed Model 1 on both metrics, roughly
1.2 percentage points higher in accuracy with a lower test loss. This
was expected, since Model 2’s structure is more capable. It made use of
pre-trained GloVe embeddings, which already carry general knowledge from
a much larger dataset, combining with an LSTM that can capture word
sequence and context. Model 1, by contrast, used only from-scratch
embeddings trained on the much smaller TikTok reviews dataset, with no
component to track how words relate to one another in sequence. 

Recommendation: **Model 2**. It achieved higher test accuracy and lower
test loss, and its architecture is better suited for sentiment analysis,
where sequential context matters. While the improvement is moderate, it
is consistent across both accuracy and loss.

Alex

Running all my models through my test set, my GRU was the best
performing as expected. With the highest overall accuracy, and decent
recall among all classes

Uploading the different CSV files containing reviews from my teammate’s
respective apps (Netflix, Telegram, and TikTok), I intended to use all
the reviews for test data. I created a function to process all of the
raw review data individually to fit my best model (GRU), before creating
my predictions and classification report. Through the classification
report, I’d be able to identify both my model’s overall accuracy and its
performance in predicting each individual score with metrics like
“recall” and “f1-score” helping me avoid any misleading accuracy values.

Looking at my model’s performance over each app, In terms of surface
level accuracy it performed the best on the Telegram app with a (test
acc=0.71, test loss=1.07) followed by Netflix (test acc=0.70, test
loss=1.00) and TikTok (test acc=0.68, test loss=1.07). I suspect that
this is because Spotify is a music app, thus reviews are likely to be
music related. Netflix being a streaming app and TikTok being a content
app will likely have conflicting reviews while Telegram is a simple
social media app where user reviews won’t have any film or content
specific language.

Team model performance

<table style="width:94%;">
<colgroup>
<col style="width: 12%" />
<col style="width: 20%" />
<col style="width: 20%" />
<col style="width: 20%" />
<col style="width: 20%" />
</colgroup>
<thead>
<tr>
<th></th>
<th style="text-align: left;">Netflix</th>
<th>Telegram</th>
<th>TikTok</th>
<th>Spotify</th>
</tr>
</thead>
<tbody>
<tr>
<td>Jaydon</td>
<td style="text-align: left;">Acc: 61.64%<br />
Loss: 1.0748</td>
<td><p>Acc: 72.18%</p>
<p>Loss: 0.9868</p></td>
<td><p>Acc: 65.80%</p>
<p>Loss: 1.0737</p></td>
<td><p>Acc: 68.84%</p>
<p>Loss: 0.9569</p></td>
</tr>
<tr>
<td>Lakshya</td>
<td style="text-align: left;"><p>Acc: 61.87%</p>
<p>Loss: 1.0660</p></td>
<td><p>Acc: 57.70%</p>
<p>Loss: 1.1463</p></td>
<td><p>Acc: 59.77%</p>
<p>Loss: 1.1545</p></td>
<td><p>Acc: 60.94%</p>
<p>Loss: 1.0616</p></td>
</tr>
<tr>
<td>Jaether</td>
<td style="text-align: left;"><p>Acc: 71.71%</p>
<p>Loss: 0.9475</p></td>
<td><p>Acc: 74.32%</p>
<p>Loss: 0.9251</p></td>
<td><p>Acc: 59.18%</p>
<p>Loss: 1.1457</p></td>
<td><p>Acc: 69.70%</p>
<p>Loss: 0.9796</p></td>
</tr>
<tr>
<td>Alex</td>
<td style="text-align: left;"><p>Acc: 69.98%</p>
<p>Loss: 1.0016</p></td>
<td><p>Acc: 71.02%</p>
<p>Loss: 1.0716</p></td>
<td><p>Acc: 67.97%</p>
<p>Loss: 1.0691</p></td>
<td><p>Acc: 73.61%</p>
<p>Loss: 0.8603</p></td>
</tr>
</tbody>
</table>

Reviewing the performances of all our models on each of our test sets,
we found Alex’s model had the highest accuracy across all the apps.
However, concluding that his model was objectively the best would be a
premature decision, because of the largely imbalanced classes inside of
the datasets we had chosen to use. A high accuracy did not necessarily
mean a better model, as if 1-2 classes took up 80% of the data labels, a
model that only learnt to predict those 2 classes could achieve an 80%
accuracy without guessing the other classes at a single time. We
investigated using different metrics like recall, and f1-score to get
detailed evaluations of the model when predicting each possible score.
We found that despite having a high accuracy across all the apps, the
individual recall and f1-score were only high for extreme scores (1 and
5) while the rest remained extremely low. Despite this we still chose to
go with Alex’s model as our best model.

# 5. Make Prediction with Best Model

(Alex)

Our chosen best model is a recurrent neural network that makes use Gated
Recurrent Unit (GRU) layers, trained on review data put through a
vectorizer of parameters (max_token=10000, output_mode=’int’,
output_sequence=100). To apply this model to a real life input we would
need to:

1.  Receive the input from the user

2.  Convert said input into a string array

3.  Vectorize said array using the same vectorizer used to adapt the
    chosen models training data (converting the input into a numeric
    tensor)

4.  Run the vectorized input through our best model and collect the list
    of confidence percentages for each possible class

5.  Create a list of all possible scores to help identify which
    confidence percentage belongs to which class

We chose to run our model through 5 different prompts (each representing
1 class) set to test the model’s ability to identify every class based
on a given input, and 1 vague prompt using positive language despite
being a negative review.

Prompt 1: (Score 1 – clearly negative) "This app is a complete mess. It
crashes every single time I try to open it, and when it does work, half
the features don't even load properly. I've wasted so much time trying
to get this to function. Absolutely not worth the storage space."

Prompt 2: (Score 2 – mostly negative, some redeeming point) "Not a great
experience overall. It has potential but the bugs make it frustrating to
use daily. Customer support never got back to me when I reported the
issue. Might give it one more chance before giving up completely."

Prompt 3: (Score 3 - mixed) "It's okay, does what it says most of the
time. Some features work well, others feel half-finished. Nothing
special but nothing terrible either. Wouldn't recommend against it,
wouldn't rave about it either."

Prompt 4: (Score 4 – mostly positive, minor complaints) "Really solid
app, I use it almost every day now. The interface is clean and it's fast
most of the time. Only issue is occasional lag during peak hours, but
it's a minor inconvenience compared to how useful this has become for
me."

Prompt 5: (Score 5 – clearly positive) "Absolutely love this app! It's
intuitive, reliable, and has genuinely made my daily routine so much
easier. The developers clearly care about user experience - updates keep
making it better. Couldn't ask for more."

Vague prompt: (Score 1 – positive on the surface, negative intent) "Oh
wonderful, another update that broke everything that used to work
perfectly. Really impressive how they managed to make a simple app feel
like rocket science. Truly a masterclass in how to ruin something that
didn't need fixing."

We recorded the following:

Prompt 1 (Prediction: 1, Confidence: 0.63)

The model easily identified the correct score of the first prompt with a
much higher confidence than the other classes, with the second highest
being 2 with a confidence of 0.26

Prompt 2 (Prediction: 2, Confidence: 0.38)

The model was able to predict the correct score. However, confidence was
noticeably lower than with the prediction made with prompt 1, with the
second highest being 3 with a confidence of 0.31

Prompt 3 (Prediction: 3, Confidence: 0.35)

Despite correctly identifying the correct score, it was with a very low
confidence with the second highest being 4 with a confidence of 0.30.
This was expected as we acknowledged that our chosen best model is
weaker when it comes to analysing more mixed reviews.

Prompt 4 (Prediction: 4, Confidence: 0.71)

Correct score predicted, and with the highest confidence observed thus
far, very notable considering the correct score was not a fully extreme
end with the second highest score being 5 with a confidence of 0.20

Prompt 5 (Prediction: 5, Confidence: 0.71)

Since most users leave positive reviews, our chosen model was easily
able to identify an extremely positive review with a high confidence.
Second highest score was 4 with a confidence of 0.27

Vague Prompt (Prediction: 4, Confidence: 0.41)

Despite it being made with middle-to-low confidence, the completely
wrong score was predicted. This is a clear sign that our model was
incapable of seeing through sarcasm and mostly relies on “on the
surface” sentiment. Perhaps this could be fixed though training with
explicitly labelled sarcastic reviews? Through this we discovered that
sarcasm detection is a well-known problem faced by sentiment analysis
model trained on straightforward positive and negative word
associations.

After evaluating the model’s performance when exposed to each prompt. It
is clear that the model was much more capable of predicting extreme
scores than middle scores making them inherently harder to separate
cleanly. This is expected as this model consistently showed a weaker
recall for the 2 and 3 scores. It is also worth noting that the models
prediction to the sarcastic review was not only wrong but wrong in the
positive direction (predicted score: 4) this is evidence of the training
data being heavily skewed towards 5-star reviews and a clear bias in the
models predictions.

Across all six prompts, the model reliably identified sentiment at the
extremes, showed reduced confidence on middle-ground reviews (consistent
with earlier findings), and failed entirely on sarcasm, revealing a real
limitation of surface-level sentiment modelling.

# 6. Summary + Improvements

(Jaether)

This assignment sets out to build a sentiment classification model
capable of predicting star ratings (1-5) from Netflix, Spotify,
Telegram, and TikTok app review texts, comparing Recurrent Neural
Network (RNN) and 1D Convolutional Neural Network (CNN).

Through iterative experimentations which include addressing class
imbalance via compute_class_weight and selective sampling, extending
sequence length and word vector dimensionality, and comparing LSTM to
GRU architectures. After all that, our best-performing model emerged as
a GRU-based RNN, trained on data vectorized with max_tokens=10000 and an
output_sequence of 100. GRU was selected over LSTM because it reached
comparable accuracy in significantly fewer epochs (14 vs 22), making it
the more efficient choice for this task without any measurable loss in
its performance. This efficiency gain is particularly valuable given the
earlier finding that model development was heavily constrained by
computational cost, meaning GRU’s faster convergence offers practical
benefits beyond raw accuracy alone.

When applied to real-word review text, this model proved reliable at
identifying reviews with clearly extremely sentiment, correctly
classifying both strongly negative and strongly positive reviews with
high confidence (0.63 and 0.71 respectively), and showing unexpectedly
strong confidence (0.71) on a moderately positive review as well. This
suggests that the model has learnt robust decision boundaries at the
outer edges of the ratings scale, which is likely reinforced by these
classes being well-represented and distinct in the sense that it’s
clearly positive or negative language, within the training data. In
practical terms, this means the model can be trusted to flag reviews
that are unambiguously positive or negative with a reasonably low risk
of misclassification.

However, the model’s performance notably dropped off for moderate
reviews, correctly predicting scores 2 and 3 but with far lower
confidence (0.38 and 0.35), reflecting a narrower margin between
adjacent classes that are harder to separate on word choice alone. This
is a direct consequence of the class imbalance challenges identified
during model development, even after mitigation through class weighting
and resampling. Minority classes remain inherently harder to classify
with certainty, since there is simply less linguistic variety in the
training data to draw firm boundaries from. More importantly, the model
failed horribly on a sarcastic review, misclassifying a clearly negative
review as a relatively high score, 4 (positive) with moderate confidence
(0.41). This is strong evidence that the model, despite its
architectural improvements, still relies primarily on surface-level word
associations rather than contextual or tonal understandings, and likely
retains some residual bias toward positive sentiment inherited from an
originally imbalanced dataset.

Considering both together, these results show that our chosen GRU model
performs well as a practical tool for flagging reviews at the extremes
of sentiment, where confidence is high and predictions are consistent,
but is less dependable for nuanced or ambiguous reviews. Its 2 main
limitations are (1) mid-range sentiment classification; and (2) sarcasm
detection. Both point towards the same underlying cause of, that the
model learns direct word-to-sentiment associations instead of deeper
contextual meanings, and this weakness becomes most visible when
surface-level language and true intent diverge, like in the sarcastic
review.

**Improvement**:

There are several different directions in which the model can be further
improved.

First, sarcasm detection could be addressed by deliberately including or
increasing the number of sarcastic or ironic reviews in the dataset, and
ensure that they are explicitly labelled as such, so that the model can
start to learn linguistic cues. This would be things like exaggerated
praised paired with contextually negative outcomes, instead of relying
purely on word polarity.

Second, more work on class balancing could help close the confidence gap
on moderate reviews. While compute_class_weight and selective sampling
both improved performance over the baseline variant, a more
sophisticated approach such as synthetic oversampling (e.g. SMOTE-style
techniques adapted for text) or targeted data collection for
underrepresented classed could help yield more decisive predictions.

Thirdly, exploring more context-aware architectures would likely improve
the model’s ability to capture context and tone across a sentence rather
than treating words largely in isolation, which is an already known
limitation of GRU and LSTM-based approaches when handling nuanced or
sarcastic language.

Finally, an ensemble approach could be implemented using Kera’s
functional API to combine our GRU model’s strength at extreme
classification with a secondary, sarcasm-aware or context-sensitive
branch. For example. 2 parallel branches processing the same vectorized
input, with their outputs merged (e.g. with concatenation) before a
final classification layer. This would offer a practical middle ground,
allowing confident predictions to be handled quickly and efficiently,
while harder or lower-confidence cases benefit from the added
context-sensitivity of the new second branch.

Overall, this project showcases that careful handling of class imbalance
and thoughtful architecture selection can produce a genuinely useful
sentiment classification model that is capable of reliably identifying
clear-cut positive and negative reviews. At the same time, its struggle
with mid-range sentiments and sarcastic reviews highlight a larger
challenge in sentiment analysis. And that is, distinguishing what is
explicitly said and what it means. Addressing this gap through more
diverse and thoughtfully labelled training data, alongside more
context-aware architectures would be the clearing path moving forward
for improving the model’s real-world reliability.
