# Aspect-Based Sentiment Analysis using BERT and Attention

An in-depth exploration of multi-step and single-step approaches for aspect term extraction and sentiment classification. This project leverages transformer-based architectures (BERT) with custom pooling and attention mechanisms to achieve state-of-the-art results on SemEval 2014 Task 4.

## Overview

This project implements and compares three methods for ABSA:

- Mean Pooling over aspect token embeddings

- Multi-Head Attention on token embeddings with learned aspect heads

- Simple Attention using a BERT+CLS-based classifier
