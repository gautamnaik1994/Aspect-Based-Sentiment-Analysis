---
title: Aspect Based Sentiment Analysis
date: 2025-05-20
slug: aspect-based-sentiment-analysis
updatedDate: 2025-05-20
description: A case study project highlighting the techniques and methods used to implement Aspect Based Sentiment Analysis using Deep Learning.
publish: true
featuredPost: true
tags:
  - python
categories:
  - Data Science
  - Deep Learning
keywords:
    - aspect based sentiment analysis
    - sentiment analysis
    - deep learning
    - bert
    - pytorch
    - nlp
    - transformers
    - data science
    - machine learning
bannerImage: aspect-based-sentiment-analysis.png
---


## Introduction

Ever found yourself in a product manager's shoes at a tech company, grappling with mountains of unstructured customer reviews after a new launch? You're not just looking for a 'happy or sad' tally; you want to pinpoint exactly *what* customers love or loathe. So, what's your next move? Standard sentiment analysis often falls short, giving you a high-level 'happy or sad' score but missing the granular insights you truly need.

This is precisely where **Aspect-Based Sentiment Analysis (ABSA)** shines.

Think of Aspect-Based Sentiment Analysis as a precision tool: it lets you drill down into customer reviews to extract specific product or service aspects and then determine the sentiment tied to each one. For example, if you have a review that says "The battery life is great, but the camera quality is terrible," Aspect-Based Sentiment Analysis would allow you to identify "battery life" as an aspect with positive sentiment and "camera quality" as an aspect with negative sentiment.

### Why is Aspect-Based Sentiment Analysis Important?

ABSA moves you beyond a generic 'happy or unhappy' score, empowering you to pinpoint *exactly* which features customers adore or despise. This granular insight is invaluable for making data-driven decisions, whether you're prioritizing product improvements, crafting marketing campaigns, or refining customer support strategies. By analyzing customer reviews over time, you can also track whether sentiment towards specific aspects is improving or declining, which in turn helps you make informed decisions about product development and marketing strategies.

## Project Overview and Challenges

The core challenge I faced in this project stemmed from its inherently multi-stage nature. Typically, to achieve Aspect-Based Sentiment Analysis, you'd perform two distinct tasks: first, extracting the aspects from the reviews, and then determining the sentiment associated with each extracted aspect. This proved particularly challenging because, in real-world scenarios, explicit aspect labels are rarely available within raw review texts.

### Why Use BERT?

Traditional models look at words in isolation, but BERT understands the context of each word in a sentence. This helps it identify aspects and their sentiments more accurately, especially when the same word can mean different things in different contexts. Its pre-training on vast amounts of text data makes it incredibly powerful for a wide range of NLP tasks, including this complex multi-task ABSA problem.

### Initial Approach: Multi-Step Analysis

My initial approach involved a classic two-stage pipeline: first, I extracted aspects from reviews using one pre-trained BERT model, and *then* I employed a separate pre-trained BERT model to classify the sentiment of each identified aspect.

While this approach was effective, it came with notable drawbacks: significant processing time, a need for substantial manual optimization, and an increased project complexity owing to the deployment of two distinct models.

### Moving Towards a Single-Step Approach

To overcome these limitations, I then explored a **single-step approach**. The goal was to use a single pre-trained BERT model to simultaneously extract aspects and their associated sentiment, offering a simpler and faster alternative to the multi-step methodology. This streamlined process, while requiring a more complex architecture and potentially more training data, promised significant efficiency gains.

## Neural Network Pipeline

```
Input Text
   |
[Tokenizer]
   |
[Input IDs, Attention Mask]
   |
[BERT Model]
   |
[Aspect Extraction Head] ---> Aspect Tags (BIO)
   |
[Aspect Extraction]
   |
[Sentiment Classification Head]
   |
[Sentiment for Each Aspect]
```

## How Does the Model Work?

1. The review text is split into tokens (words or subwords).
2. Each token is converted to an ID and passed to BERT, a language model that understands context.
3. BERT outputs a representation for each token.
4. The first head (aspect extraction) predicts which tokens are part of an aspect (using BIO tags).
5. For each detected aspect, the second head (sentiment classifier) predicts if the sentiment is positive, negative, or neutral.

## Dataset Used

The dataset used in this project is the SemEval 2014 Task 4 dataset, which contains customer reviews of laptops. The dataset is divided into three parts: the training set, the test set, and the validation set.
The dataset is available on the [SemEval website](https://alt.qcri.org/semeval2014/task4/). The dataset is in the form of a JSONL file, with each row representing a review and its associated aspect and sentiment labels. The aspect categories include "battery", "camera", "design", "performance", and "price". The sentiment labels are "positive", "negative", and "neutral".

Following is the format of the training data.

```json
{
    "text": "now i ' m really bummed that i have a very nice looking chromebook with a beautiful screen that is totally unusable .",
    "labels": [
        {"aspect": "chromebook", "opinion": "nice", "polarity": "positive", "category": "LAPTOP#DESIGN_FEATURES"},
        {"aspect": "chromebook", "opinion": "bummed", "polarity": "negative", "category": "LAPTOP#OPERATION_PERFORMANCE"},
        {"aspect": "chromebook", "opinion": "unusable", "polarity": "negative", "category": "LAPTOP#OPERATION_PERFORMANCE"},
        {"aspect": "screen", "opinion": "beautiful", "polarity": "positive", "category": "DISPLAY#OPERATION_PERFORMANCE"}
    ]
}
```

## Dataset Preprocessing

Before feeding the data into our deep learning models, thorough preprocessing is crucial. This involved several key steps:

1. **Tokenization:** Converting text into numerical tokens that the BERT model can understand. I used the `BertTokenizer` from Hugging Face for this.
2. **Aspect Span Identification:** Identifying the start and end positions of each aspect within the tokenized sentence.
3. **Sentiment Mapping:** Converting sentiment labels (e.g., "positive", "negative") into numerical representations.
4. **Creating Masks:** Generating aspect masks to highlight the relevant tokens for each aspect during training.
5. **Padding and Truncation:** Ensuring all input sequences have a uniform length by padding shorter sequences and truncating longer ones.

Here's an example of the final preprocessed data structure for a single review:

```yaml
Tokens: ['[CLS]', 'now', 'i', "'", 'm', 'really', 'bum', '##med', 'that', 'i', 'have', 'a', 'very', 'nice', 'looking', 'chrome', '##book', 'with', 'a', 'beautiful', 'screen', 'that', 'is', 'totally', 'un', '##usa', '##ble', '.', '[SEP]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]']
Input IDs: [101, 2085, 1045, 1005, 1049, 2428, 26352, 7583, 2008, 1045, 2031, 1037, 2200, 3835, 2559, 18546, 8654, 2007, 1037, 3376, 3898, 2008, 2003, 6135, 4895, 10383, 3468, 1012, 102, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
Attention Mask: [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
Aspect Spans: [[15, 16], [20, 20]]
Polarities: [1, 1]
Token Labels: ['O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'B-ASP', 'I-ASP', 'O', 'O', 'O', 'B-ASP', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O']
```

### Explanation of the output

- **Tokens**: The input text is tokenized into individual words and sub-words. The special tokens `[CLS]` and `[SEP]` are added to the beginning and end of the sequence, respectively.
- **Input IDs**: Each token is mapped to its corresponding ID in the BERT vocabulary. Padding tokens are added to ensure that all sequences have the same length.
- **Attention Mask**: A binary mask indicating which tokens are actual words (1) and which are padding (0). This is used to ignore padding tokens during model training and evaluation.
- **Aspect Spans**: The start and end indices of the aspect tokens in the tokenized input.
- **Polarities**: The sentiment polarity associated with each aspect. In this case, both aspects have a positive polarity.
- **Token Labels**: The labels for each token in the input sequence. The aspect tokens are labeled as `B-ASP` (beginning of aspect) and `I-ASP` (inside aspect), while other tokens are labeled as `O` (outside aspect).

## Multi Step Approach

The implemented aspect-based sentiment analysis involved a two-stage process. Initially, aspects were extracted from reviews using a pre-trained BERT model. Subsequently, a separate pre-trained BERT model was employed to classify the sentiment of each identified aspect. While effective, this methodology presented drawbacks including significant processing time, substantial manual optimization, and increased project complexity due to the use of two distinct models.

## Single Step Approach

A single pre-trained BERT model was used to simultaneously extract aspects and their associated sentiment, offering a simpler and faster alternative to multi-step approaches. This single-step method required a more complex architecture and more training data but streamlined the process. The following methods were employed to implement this approach.

### Model Architecture

**Aspect Extraction Head**: Because I will be extracting the aspects and determining the sentiment associated with each aspect, I will be using multi-head neural network. The first head will be used to extract the aspects. This will be powered by Bert model.  
**Sentiment Classification Head**: The second head will be used to determine the sentiment associated with each aspect. The output of the first head will be used as the input to the second head. The final output will be a list of aspects and their associated sentiment polarities. I tried 3 different methods to implement the sentiment classification head. The methods are as follows:

- **Mean Pooling**
- **Multi-Head Attention**
- **Simple Attention and CLS Embedding**

I will discuss each of these methods in detail in the following sections.

## Using Mean Pooling

<Aside>

### What is Mean Pooling?

Mean pooling is a technique used to aggregate the output embeddings of a sequence of tokens into a single vector representation. In the context of aspect-based sentiment analysis, mean pooling can be used to obtain a fixed-size representation for an aspect term by averaging the embeddings of the tokens that make up the aspect.

Let’s say you have the sentence:

```
The battery life of this laptop is amazing.
```

Suppose the aspect term is **"battery life"**, and its tokenized positions in the sequence are **5 and 6**.

Assume the output of your BERT model for this sentence (i.e., sequence_output) gives a hidden size of 4 for simplicity (normally it's 768). Here's what the embeddings for each token might look like:

```python
import torch

# Simulated BERT output: shape [1, seq_len=8, hidden_size=4]
sequence_output = torch.tensor([[
    [0.1, 0.2, 0.3, 0.4],  # token 0
    [0.2, 0.1, 0.0, 0.3],  # token 1
    [0.5, 0.4, 0.1, 0.2],  # token 2
    [0.3, 0.6, 0.7, 0.1],  # token 3
    [0.0, 0.0, 0.0, 0.0],  # token 4
    [0.9, 0.8, 0.7, 0.6],  # token 5 → 'battery'
    [0.7, 0.6, 0.5, 0.4],  # token 6 → 'life'
    [0.1, 0.3, 0.2, 0.2],  # token 7
]])
```

To get the mean pooled vector for the aspect **"battery life"**, we take the average of the embeddings at positions **[5, 6]**:

```python
# Extract and mean pool over the aspect span
aspect_span = sequence_output[0, 5:7]  # shape [2, 4]
mean_pooled = aspect_span.mean(dim=0)
print(mean_pooled)
# Output: tensor([0.8000, 0.7000, 0.6000, 0.5000])
```

This vector represents the average embedding of the aspect term **"battery life"** and is passed into the sentiment classifier to predict whether the sentiment is positive, neutral, or negative.

</Aside>

### Model Architecture

Follow is the model architecture for Multi-Head Neural Network using Mean Pooling. The `self.classifier` is the Aspect Extraction Head and `self.sentiment_classifier` is the Sentiment Classification Head. The `forward` method takes the input IDs and attention mask as input and returns the logits and sequence output.  
Sequence output is the contextualized embeddings for each token in the input sequence. The logits are the output of the Aspect Extraction Head.

```python
class AspectDetectionModel(nn.Module):
    def __init__(self):
        super(AspectDetectionModel, self).__init__()
        self.bert = AutoModel.from_pretrained("bert-base-uncased")
        self.dropout = nn.Dropout(0.3)
        self.classifier = nn.Linear(self.bert.config.hidden_size, len(label2id)) # Aspect Extraction Head
        self.sentiment_classifier = nn.Linear(self.bert.config.hidden_size, 3)  # Sentiment Classification Head

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        sequence_output = self.dropout(outputs.last_hidden_state)  # [B, L, H]
        logits = self.classifier(sequence_output)  # [B, L, num_labels]
        return logits, sequence_output
```

Following is the training loop for the model. It takes the input IDs, attention mask, and labels as input and returns the loss. If you notice, I am using sentiment loss as well. The sentiment loss is calculated using the mean pooling of the sequence output.

```python
# Training Loop Snippet
num_epochs = 20

for epoch in range(num_epochs):
    total_aspect_train_loss = 0
    total_sentiment_train_loss = 0

    model.train()
    total_train_loss = 0

    for batch in train_dataloader:
        input_ids = batch["input_ids"].to("mps")
        attention_mask = batch["attention_mask"].to("mps")
        labels = batch["labels"].to("mps")
        
        # logits contains the aspect extraction head output and sequence_output contains the contextualized embeddings
        logits, sequence_output = model(input_ids, attention_mask)
        loss = criterion(logits.view(-1, len(label2id)), labels.view(-1))
        total_aspect_train_loss += loss.item()

        sentiment_losses = []
        for i in range(len(input_ids)):
            for aspect_index, sentiment in zip(batch["aspects_index"][i], batch["aspects_sentiment"][i]):
                if aspect_index[1] >= sequence_output.size(1):
                    continue
                # Assume the aspect span is a list of words ['chrome', '##book'], BIO tags are ['B-ASP', 'I-ASP'], and indices are [15, 16]
                # Each word in the aspect span have its own embedding in the sequence output
                # We take the mean of the embeddings for the aspect span
                # aspect_index = [15, 16] means we take the mean of sequence_output[i, 15:17]
                pooled = sequence_output[i, aspect_index[0]:aspect_index[1]+1].mean(dim=0)
                sentiment_logits = model.sentiment_classifier(pooled.unsqueeze(0))
                sentiment_target = torch.tensor([sentiment], dtype=torch.long).to("mps")
                sentiment_loss = criterion(sentiment_logits.view(-1, 3), sentiment_target)
                sentiment_losses.append(sentiment_loss)


        if sentiment_losses:
            sentiment_loss = torch.stack(sentiment_losses).mean()
        else:
            sentiment_loss = torch.tensor(0.0).to("mps")

        total_sentiment_train_loss += sentiment_loss.item()
        total_loss = aspect_loss + sentiment_loss

        total_loss.backward()
        optimizer.step()
        optimizer.zero_grad()
```

Following is the evaluation loop for the model

```python

def extract_aspect_spans(pred_labels):
    """
    Extracts aspect spans from the predicted labels.
    Args:
        pred_labels (list): List of predicted labels (BIO tags).
    Returns:
        spans (list): List of aspect spans, where each span is a list of [start, end] indices.
    """
    spans = []
    i = 0
    while i < len(pred_labels):
        if pred_labels[i] == 1:  # B-ASP
            start = i
            i += 1
            while i < len(pred_labels) and pred_labels[i] == 2:  # I-ASP
                i += 1
            end = i - 1
            spans.append([start, end])
        else:
            i += 1
    return spans

# Validation Loop Snippet

total_aspect_val_loss = 0
total_sentiment_val_loss = 0

with torch.no_grad():
    model.eval()

    for batch in val_dataloader:
        input_ids = batch["input_ids"].to("mps")
        attention_mask = batch["attention_mask"].to("mps")
        labels = batch["labels"].to("mps")

        logits, sequence_output = model(input_ids, attention_mask)
        aspect_loss = criterion(logits.view(-1, len(label2id)), labels.view(-1))
        total_aspect_val_loss += aspect_loss.item()

        sentiment_losses = []
        preds = torch.argmax(logits, dim=2)
        for i in range(len(input_ids)):
            # During training, I used the existing aspect spans from the training data. But during evaluation, 
            # I will extract the aspect spans from the predicted labels
            # This is because the aspect spans are not available in the validation data
            # I will use the predicted labels to extract the aspect spans
            # The predicted labels are in the form of BIO tags 
            aspects = extract_aspect_spans(preds[i].cpu().tolist())
            for aspect_index, sentiment in zip(batch["aspects_index"][i], batch["aspects_sentiment"][i]):
                if aspect_index in aspects and aspect_index[1] < sequence_output.size(1):
                    pooled = sequence_output[i, aspect_index[0]:aspect_index[1]+1].mean(dim=0)
                    sentiment_logits = model.sentiment_classifier(pooled.unsqueeze(0))
                    sentiment_target = torch.tensor([sentiment], dtype=torch.long).to("mps")
                    sentiment_loss = criterion(sentiment_logits.view(-1, 3), sentiment_target)
                    sentiment_losses.append(sentiment_loss)

        if sentiment_losses:
            sentiment_loss = torch.stack(sentiment_losses).mean()
        else:
            sentiment_loss = torch.tensor(0.0).to("mps")

        total_sentiment_val_loss += sentiment_loss.item()
```

In above code you can see that I am using the `extract_aspect_spans` function to extract the aspect spans from the predicted labels. The predicted labels are in the form of BIO tags. The `extract_aspect_spans` function takes the predicted labels as input and returns the aspect spans.

### Model Performance

This model performed poorly even during the training phase for the sentiment detection section. This might be because I was trying to directly use the mean of the aspect embeddings to get the sentiment logits. The mean pooling method does not take into account the context of the aspect term in the sentence.

For example, if the aspect term is "battery life", the mean pooling method will take the mean of the embeddings for "battery" and "life" without considering the context of the sentence. This might lead to poor performance in sentiment detection.

## Using MultiHead Attention

Since the previous method was not able to learn sentiment associated with each aspect, I decided to try a different approach. I used multi-head attention to focus on different parts of the aspect term.
This allows the model to capture more complex relationships between the tokens in the input sequence.  I used the pooled aspect embedding as the query in the attention mechanism, while the original token embeddings serve as both keys and values. The attention layer produces an attended output, emphasizing information most relevant to the aspect.

### Why MultiHead Attention?

Multi-head attention is a mechanism that allows the model to focus on different parts of the input sequence when making predictions. It does this by using multiple attention heads, each of which learns to focus on different aspects of the input. This allows the model to capture more complex relationships between the tokens in the input sequence.

In the context of aspect-based sentiment analysis, multi-head attention can be used to focus on different parts of the aspect term when making predictions. For example, if the aspect term is "battery life", the model can learn to focus on different parts of the aspect term (e.g., "battery" and "life") when making predictions. This allows the model to capture more complex relationships between the tokens in the input sequence.

**Following is the model architecture code**

```python
class SentimentClassifier(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.attention = nn.MultiheadAttention(embed_dim=hidden_size, num_heads=1, batch_first=True)
        self.dropout = nn.Dropout(0.3)
        self.norm = nn.LayerNorm(hidden_size)
        self.classifier = nn.Linear(hidden_size, 3)  # Sentiment classes: pos, neg, neutral

    def forward(self, token_embeddings, aspect_mask):
        # token_embeddings: [B, L, H], aspect_mask: [B, L]
        aspect_mask = aspect_mask.unsqueeze(-1).expand_as(token_embeddings)  # [B, L, H]
         # mask out non-aspect tokens, average aspect token embeddings
        aspect_embeddings = token_embeddings * aspect_mask  # Zero out non-aspect tokens
        # take mean of aspect embeddings
        aspect_pooled = aspect_embeddings.sum(dim=1) / (aspect_mask.sum(dim=1) + 1e-8)  # [B, H]
        query = aspect_pooled.unsqueeze(1)  # [B, 1, H]
        key = value = token_embeddings  # [B, L, H]

        attended_output, attn_weights = self.attention(query, key, value)  # [B, 1, H]
        attended_output = self.dropout(attended_output)
        attended_output = self.norm(attended_output)

        logits = self.classifier(attended_output.squeeze(1))  # [B, 3]
        return logits, attn_weights 


class AspectDetectionModel(nn.Module):
    def __init__(self):
        super(AspectDetectionModel, self).__init__()
        self.bert = AutoModel.from_pretrained("bert-base-uncased")
        self.dropout = nn.Dropout(0.3)
        self.token_classifier = nn.Linear(self.bert.config.hidden_size, len(label2id))
        self.sentiment_classifier = SentimentClassifier(hidden_size=self.bert.config.hidden_size)

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        sequence_output = self.dropout(outputs.last_hidden_state)  # [B, L, H]
        token_logits = self.token_classifier(sequence_output)  # For aspect term tagging (BIO)
        return token_logits, sequence_output
```

The `forward` method for `SentimentClassifier` takes the token embeddings and aspect mask as input and returns the sentiment logits and attention weights. The sentiment logits is consumed inside the training loop to calculate the sentiment loss.  

I am generating the aspect mask for each aspect span. The main job of aspect mask is to be zero out the non-aspect tokens. After that, the aspect embeddings are added to get the aspect pooled embeddings, and then divided by the number of aspect tokens.
This is to build the query matrix for the multi-head attention. The key and value matrices are the same as the token embeddings. The output of the multi-head attention is then passed to the Linear layer to get the sentiment logits. The sentiment loss is then calculated using the CrossEntropyLoss function. The sentiment loss is added to the aspect loss to get the total loss. The total loss is then backpropagated to update the model weights.

**Following is the training loop for the model.**

```python
num_epochs = 10

for epoch in range(num_epochs):
    total_aspect_train_loss = 0
    total_sentiment_train_loss = 0

    model.train()
    total_train_loss = 0

    for batch in train_dataloader:
        input_ids = batch["input_ids"].to("mps")
        attention_mask = batch["attention_mask"].to("mps")
        labels = batch["labels"].to("mps")

        token_logits, sequence_output = model(input_ids, attention_mask)
        aspect_loss = criterion(token_logits.view(-1, len(label2id)), labels.view(-1))
        total_aspect_train_loss += aspect_loss.item()

        sentiment_losses = []
        for i in range(len(input_ids)):
            for aspect_index, sentiment in zip(batch["aspects_index"][i], batch["aspects_sentiment"][i]):
                if aspect_index[1] >= sequence_output.size(1):
                    continue
                # build a zero tensor of the same size as input_ids
                aspect_mask = torch.zeros_like(input_ids, dtype=torch.float).to("mps")
                # set the aspect span to 1
                aspect_mask[i, aspect_index[0]:aspect_index[1]+1] = 1
                sentiment_logits, _ = model.sentiment_classifier(
                    sequence_output[i].unsqueeze(0), aspect_mask[i].unsqueeze(0)
                )
                sentiment_target = torch.tensor([sentiment], dtype=torch.long).to("mps")
                sentiment_loss = criterion(sentiment_logits, sentiment_target)
                sentiment_losses.append(sentiment_loss)

        if sentiment_losses:
            sentiment_loss = torch.stack(sentiment_losses).mean()
        else:
            sentiment_loss = torch.tensor(0.0).to("mps")

        total_sentiment_train_loss += sentiment_loss.item()
        total_loss = aspect_loss + sentiment_loss

        total_loss.backward()
        optimizer.step()
        optimizer.zero_grad()
```

The validation loop is similar to the validation loop in the previous method where `extract_aspect_spans` function is used to extract the aspect spans from the predicted labels

## Using Simple Attention and CLS Embedding

The Multi-head attention module is a bit complex and requires a lot of parameters to be tuned. So I decided to try a simpler approach. I used the CLS token embedding and simple attention mechanism to get the sentiment logits.

<Aside>

### What is CLS Token?

The CLS token is a special token used in BERT and other transformer-based models to represent the entire input sequence. It is added at the beginning of the input sequence. The output embedding of the CLS token is expected to capture the overall meaning of the input sequence. In other words, the CLS token captures the context of the entire input sequence as a single vector.
Following is the code to extract the CLS token embedding from the BERT model output:

```python
cls_embedding = bert.last_hidden_state[:, 0, :]
```

The `bert.last_hidden_state` is the output of the BERT model. The `[:, 0, :]` indexing extracts the CLS token embedding from the output. The shape of the CLS token embedding is `[batch_size, hidden_size]`. The hidden size is usually 768 for BERT-base model.

</Aside>

**Following is the model architecture code**

```python
class SentimentClassifier(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.dropout = nn.Dropout(0.3)
        self.norm = nn.LayerNorm(hidden_size*2)
        self.classifier = nn.Linear(hidden_size*2, 3)  # Sentiment classes: pos, neg, neutral
        self.attention_param = nn.Linear(hidden_size,1)

    def forward(self, token_embeddings, aspect_mask):

        cls_embedding = token_embeddings[:, 0, :]

        expaned_aspect_mask = aspect_mask.unsqueeze(-1).expand_as(token_embeddings)  # [B, L, H]
        aspect_embeddings = token_embeddings * expaned_aspect_mask  # Zero out non-aspect tokens

        attention_score = self.attention_param(aspect_embeddings).squeeze(-1) # [B, L]
        attention_score = attention_score.masked_fill(aspect_mask == 0, -1e9) # Mask non-aspect tokens
        attention_score = torch.softmax(attention_score, dim=1) # [B, L]

        aspect_pooled = torch.bmm(attention_score.unsqueeze(1), token_embeddings).squeeze(1)  # [B, H]
        combined = torch.concat([aspect_pooled, cls_embedding ], dim=1)

        combined =  self.dropout(combined)
        combined = self.norm(combined)  # [B, H]

        logits = self.classifier(combined)  # [B, 3]
        return logits , attention_score


class AspectDetectionModel(nn.Module):
    def __init__(self):
        super(AspectDetectionModel, self).__init__()
        self.bert = AutoModel.from_pretrained("bert-base-uncased")
        self.dropout = nn.Dropout(0.3)
        self.token_classifier = nn.Linear(self.bert.config.hidden_size, len(label2id))
        self.sentiment_classifier = SentimentClassifier(hidden_size=self.bert.config.hidden_size)

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        sequence_output = self.dropout(outputs.last_hidden_state)  # [B, L, H]

        token_logits = self.token_classifier(sequence_output)  # For aspect term tagging (BIO)
        return token_logits, sequence_output

```

In the above code, I am extracting the CLS token embedding from the bert model output.  Along with that I am extracting the aspect pooled embedding using the attention score.
The forward method processes the input as follows: it first extracts the [CLS] token embedding, which often represents the overall sentence meaning. The aspect mask is expanded to match the shape of the token embeddings, and non-aspect tokens are zeroed out. The model then computes attention scores for each token using the attention_param layer, masking out non-aspect tokens by assigning them a very large negative value (so their softmax probability becomes near zero). These attention scores are normalized with softmax, and used to compute a weighted sum (pooling) of the token embeddings, focusing on the aspect tokens.

The pooled aspect embedding is concatenated with the [CLS] embedding, passed through dropout and layer normalization, and finally through the classifier to produce sentiment logits. The method returns both the logits and the attention scores, allowing for interpretability of which tokens contributed most to the sentiment prediction. This design enables the model to focus on specific aspects within a sentence when determining sentiment, rather than treating the sentence as a whole.

**Following is the training loop for the model.**

```python
num_epochs = 10

for epoch in range(num_epochs):
    total_aspect_train_loss = 0
    total_sentiment_train_loss = 0
    total_aspect_val_loss = 0
    total_sentiment_val_loss = 0

    model.train()
    total_train_loss = 0

    for batch in train_dataloader:
        input_ids = batch["input_ids"].to("mps")
        attention_mask = batch["attention_mask"].to("mps")
        labels = batch["labels"].to("mps")

        token_logits, sequence_output = model(input_ids, attention_mask)
        aspect_loss = criterion(token_logits.view(-1, len(label2id)), labels.view(-1))
        total_aspect_train_loss += aspect_loss.item()

        sentiment_losses = []
        for i in range(len(input_ids)):
            for aspect_index, sentiment in zip(batch["aspects_index"][i], batch["aspects_sentiment"][i]):
                if aspect_index[1] >= sequence_output.size(1):
                    continue
                aspect_mask = torch.zeros_like(input_ids, dtype=torch.float).to("mps")
                aspect_mask[i, aspect_index[0]:aspect_index[1]+1] = 1
                sentiment_logits, _ = model.sentiment_classifier(
                    sequence_output[i].unsqueeze(0), aspect_mask[i].unsqueeze(0)
                )
                sentiment_target = torch.tensor([sentiment], dtype=torch.long).to("mps")
                sentiment_loss = criterion(sentiment_logits, sentiment_target)
                sentiment_losses.append(sentiment_loss)

        if sentiment_losses:
            sentiment_loss = torch.stack(sentiment_losses).mean()
        else:
            sentiment_loss = torch.tensor(0.0).to("mps")

        total_sentiment_train_loss += sentiment_loss.item()
        total_loss = aspect_loss + sentiment_loss

        total_loss.backward()
        optimizer.step()
        optimizer.zero_grad()

```

The validation loop is similar to the validation loop in the previous method where `extract_aspect_spans` function is used to extract the aspect spans from the predicted labels.

## Performance

The above model that used Simple Attentions and CLS Embedding gave the best output. Following is the output of the model on the some of sample reviews from Amazon website.

**Example Review**

```
This remote control car is fun, fast, and easy to handle—perfect for kids! The build quality is sturdy and it runs smoothly on different surfaces. Battery life is decent and the controls are very responsive. A great gift for kids!"
```

**Output**

<table>
    <thead>
        <tr>
            <th>BIO Tag</th>
            <th>Word</th>
            <th>Sentiment</th>
        </tr>
    </thead>
    <tbody>
        <tr>
            <td>O</td>
            <td>this</td>
            <td></td>
        </tr>
        <tr>
            <td>B-ASP</td>
            <td>remote</td>
            <td>positive</td>
        </tr>
        <tr>
            <td>I-ASP</td>
            <td>control</td>
            <td>positive</td>
        </tr>
        <tr>
            <td>I-ASP</td>
            <td>car</td>
            <td>positive</td>
        </tr>
        <tr>
            <td>O</td>
            <td>is</td>
            <td></td>
        </tr>
        <tr>
            <td>O</td>
            <td>fun</td>
            <td></td>
        </tr>
        <tr>
            <td>O</td>
            <td>,</td>
            <td></td>
        </tr>
        <tr>
            <td>O</td>
            <td>fast</td>
            <td></td>
        </tr>
        <tr>
            <td>O</td>
            <td>,</td>
            <td></td>
        </tr>
        <tr>
            <td>O</td>
            <td>and</td>
            <td></td>
        </tr>
        <tr>
            <td>O</td>
            <td>easy</td>
            <td></td>
        </tr>
        <tr>
            <td>O</td>
            <td>to</td>
            <td></td>
        </tr>
        <tr>
            <td>O</td>
            <td>handle</td>
            <td></td>
        </tr>
        <tr>
            <td>O</td>
            <td>—</td>
            <td></td>
        </tr>
        <tr>
            <td>O</td>
            <td>perfect</td>
            <td></td>
        </tr>
        <tr>
            <td>O</td>
            <td>for</td>
            <td></td>
        </tr>
        <tr>
            <td>O</td>
            <td>kids</td>
            <td></td>
        </tr>
        <tr>
            <td>O</td>
            <td>!</td>
            <td></td>
        </tr>
        <tr>
            <td>O</td>
            <td>the</td>
            <td></td>
        </tr>
        <tr>
            <td>B-ASP</td>
            <td>build</td>
            <td>positive</td>
        </tr>
        <tr>
            <td>I-ASP</td>
            <td>quality</td>
            <td>positive</td>
        </tr>
        <tr>
            <td>O</td>
            <td>is</td>
            <td></td>
        </tr>
        <tr>
            <td>O</td>
            <td>sturdy</td>
            <td></td>
        </tr>
        <tr>
            <td>O</td>
            <td>and</td>
            <td></td>
        </tr>
        <tr>
            <td>O</td>
            <td>it</td>
            <td></td>
        </tr>
        <tr>
            <td>O</td>
            <td>runs</td>
            <td></td>
        </tr>
        <tr>
            <td>O</td>
            <td>smoothly</td>
            <td></td>
        </tr>
        <tr>
            <td>O</td>
            <td>on</td>
            <td></td>
        </tr>
        <tr>
            <td>O</td>
            <td>different</td>
            <td></td>
        </tr>
        <tr>
            <td>O</td>
            <td>surfaces</td>
            <td></td>
        </tr>
        <tr>
            <td>O</td>
            <td>.</td>
            <td></td>
        </tr>
        <tr>
            <td>B-ASP</td>
            <td>battery</td>
            <td>positive</td>
        </tr>
        <tr>
            <td>I-ASP</td>
            <td>life</td>
            <td>positive</td>
        </tr>
        <tr>
            <td>O</td>
            <td>is</td>
            <td></td>
        </tr>
        <tr>
            <td>O</td>
            <td>decent</td>
            <td></td>
        </tr>
        <tr>
            <td>O</td>
            <td>and</td>
            <td></td>
        </tr>
        <tr>
            <td>O</td>
            <td>the</td>
            <td></td>
        </tr>
        <tr>
            <td>B-ASP</td>
            <td>controls</td>
            <td>positive</td>
        </tr>
        <tr>
            <td>O</td>
            <td>are</td>
            <td></td>
        </tr>
        <tr>
            <td>O</td>
            <td>very</td>
            <td></td>
        </tr>
        <tr>
            <td>O</td>
            <td>responsive</td>
            <td></td>
        </tr>
        <tr>
            <td>O</td>
            <td>.</td>
            <td></td>
        </tr>
        <tr>
            <td>O</td>
            <td>a</td>
            <td></td>
        </tr>
        <tr>
            <td>O</td>
            <td>great</td>
            <td></td>
        </tr>
        <tr>
            <td>O</td>
            <td>gift</td>
            <td></td>
        </tr>
        <tr>
            <td>O</td>
            <td>for</td>
            <td></td>
        </tr>
        <tr>
            <td>O</td>
            <td>kids</td>
            <td></td>
        </tr>
        <tr>
            <td>O</td>
            <td>!</td>
            <td></td>
        </tr>
    </tbody>
</table>

**Example Review**

```
Car quality is very nice but the controller sucks . The controller of this car do not works properly and the final in the controller do not rotate fully it only rotate like button
```

**Output**

<table>
    <thead>
        <tr>
            <th>BIO Tag</th>
            <th>Word</th>
            <th>Sentiment</th>
        </tr>
    </thead>
    <tbody>
        <tr>
            <td>B-ASP</td>
            <td>car</td>
            <td>positive</td>
        </tr>
        <tr>
            <td>I-ASP</td>
            <td>quality</td>
            <td>positive</td>
        </tr>
        <tr>
            <td>O</td>
            <td>is</td>
            <td></td>
        </tr>
        <tr>
            <td>O</td>
            <td>very</td>
            <td></td>
        </tr>
        <tr>
            <td>O</td>
            <td>nice</td>
            <td></td>
        </tr>
        <tr>
            <td>O</td>
            <td>but</td>
            <td></td>
        </tr>
        <tr>
            <td>O</td>
            <td>the</td>
            <td></td>
        </tr>
        <tr>
            <td>B-ASP</td>
            <td>controller</td>
            <td>negative</td>
        </tr>
        <tr>
            <td>O</td>
            <td>sucks</td>
            <td></td>
        </tr>
        <tr>
            <td>O</td>
            <td>.</td>
            <td></td>
        </tr>
        <tr>
            <td>O</td>
            <td>the</td>
            <td></td>
        </tr>
        <tr>
            <td>B-ASP</td>
            <td>controller</td>
            <td>negative</td>
        </tr>
        <tr>
            <td>I-ASP</td>
            <td>of</td>
            <td>negative</td>
        </tr>
        <tr>
            <td>O</td>
            <td>this</td>
            <td></td>
        </tr>
        <tr>
            <td>O</td>
            <td>car</td>
            <td></td>
        </tr>
        <tr>
            <td>O</td>
            <td>do</td>
            <td></td>
        </tr>
        <tr>
            <td>O</td>
            <td>not</td>
            <td></td>
        </tr>
        <tr>
            <td>O</td>
            <td>works</td>
            <td></td>
        </tr>
        <tr>
            <td>O</td>
            <td>properly</td>
            <td></td>
        </tr>
        <tr>
            <td>O</td>
            <td>and</td>
            <td></td>
        </tr>
        <tr>
            <td>O</td>
            <td>the</td>
            <td></td>
        </tr>
        <tr>
            <td>O</td>
            <td>final</td>
            <td></td>
        </tr>
        <tr>
            <td>O</td>
            <td>in</td>
            <td></td>
        </tr>
        <tr>
            <td>O</td>
            <td>the</td>
            <td></td>
        </tr>
        <tr>
            <td>B-ASP</td>
            <td>controller</td>
            <td>negative</td>
        </tr>
        <tr>
            <td>O</td>
            <td>do</td>
            <td></td>
        </tr>
        <tr>
            <td>O</td>
            <td>not</td>
            <td></td>
        </tr>
        <tr>
            <td>O</td>
            <td>rotate</td>
            <td></td>
        </tr>
        <tr>
            <td>O</td>
            <td>fully</td>
            <td></td>
        </tr>
        <tr>
            <td>O</td>
            <td>it</td>
            <td></td>
        </tr>
        <tr>
            <td>O</td>
            <td>only</td>
            <td></td>
        </tr>
        <tr>
            <td>O</td>
            <td>rotate</td>
            <td></td>
        </tr>
        <tr>
            <td>O</td>
            <td>like</td>
            <td></td>
        </tr>
        <tr>
            <td>O</td>
            <td>button</td>
            <td></td>
        </tr>
    </tbody>
</table>

## Challenges and Limitations

Even with the advancements of deep learning and sophisticated models like BERT, Aspect-Based Sentiment Analysis presents its own set of challenges:

- **Overlapping Aspects:** Sometimes, aspects overlap or are not clearly separated in text, making their individual extraction and sentiment assignment tricky.
- **Domain-Specific Language:** Slang, jargon, or product-specific terms can be hard for a general-purpose model to understand without extensive fine-tuning on domain-specific data.
- **Data Requirements:** While BERT helps, deep learning models still need a significant amount of labeled data to perform well, especially for multi-task learning, and collecting such data can be time-consuming and expensive.
- **Computation:** Training and fine-tuning these large transformer-based models requires substantial computing power, often necessitating GPUs.

## Further Reading

- [BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding](https://arxiv.org/abs/1810.04805)
- [Hugging Face Transformers Documentation](https://huggingface.co/docs/transformers/index)
- [Aspect-Based Sentiment Analysis Survey](https://arxiv.org/abs/2107.04625)
