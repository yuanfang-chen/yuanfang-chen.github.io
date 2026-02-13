---
layout: post
title:  "Flash Attention"
# date:   2025-12-08 11:18:26 -0800
categories: CUDA
typora-root-url: ..
---

## Self-attention

Yes, self-attention can be restricted to only check previous tokens, but this is specifically known as 

**Masked Self-Attention** or **Causal Self-Attention**, rather than the default "vanilla" self-attention. 

- **Vanilla Self-Attention (Encoder):** Allows a token to look at all other tokens in a sequence, both past and future (bidirectional).
- **Masked/Causal Self-Attention (Decoder):** Restricts attention to only the current and previous tokens. 

Here is a breakdown of why and how this is done: 

Why Restrict Future Tokens? 

Masked self-attention is crucial for **autoregressive models** (like GPT) to maintain causality during training. 

- **Preventing "Cheating":** If a model is training to predict the next word in a sentence, allowing it to see that word in the input would mean it's not actually learning to predict.
- **Autoregressive Property:** It ensures that when generating text, the model only relies on the context it has already generated. 

How Masked Self-Attention Works 

1. **Attention Matrix:** During the calculation of attention scores (![img](data:image/gif;base64,R0lGODlhAQABAIAAAP///wAAACH5BAEAAAAALAAAAAABAAEAAAICRAEAOw==)QKTcap Q cap K to the cap T-th power𝑄𝐾𝑇), a token at position ![img](data:image/gif;base64,R0lGODlhAQABAIAAAP///wAAACH5BAEAAAAALAAAAAABAAEAAAICRAEAOw==)ii𝑖 calculates scores for all tokens in the sequence.
2. **Masking:** A mask is applied to the attention matrix, replacing the scores of future tokens with a very large negative number (e.g., ![img](data:image/gif;base64,R0lGODlhAQABAIAAAP///wAAACH5BAEAAAAALAAAAAABAAEAAAICRAEAOw==)−∞negative infinity−∞).
3. **Softmax:** When the softmax function is applied, these large negative numbers become 0, meaning the model assigns zero weight to future tokens. 

Summary Table 

| Attention Type | Look at Future?     | Primary Use Case                            |
| -------------- | ------------------- | ------------------------------------------- |
| **Vanilla**    | Yes (Bidirectional) | Encoder-only (e.g., BERT) for understanding |
| **Masked**     | No (Unidirectional) | Decoder-only (e.g., GPT) for generation     |

While the transformer architecture processes sequences in parallel, masking ensures that the information flow during training mimics the step-by-step, left-to-right generation of inference. 



## Local/sparse attention

GPT-3

## Multi-query attention (MQA)

![image-20260204113303374](./assets/images/image-20260204113303374.png)

## Grouped-query attention (GQA)

Llama 2 and 3

## MLA

Deepseek



## DeepSeek Sparse Attention (DSA)

## Linear Attention





## Flash Attention

Flash Attention is a popular method and implementation that provides significant speedups for both training and inference of Transformer LLMs on GPUs. It speeds up the attention calculation by optimizing what values are loaded and moved between a GPU’s shared memory (SRAM) and high bandwidth memory (HBM). It is described in detail in the papers “FlashAttention: Fast and memory-efficient exact attention with IO-awareness” and the subsequent “FlashAttention-2: Faster attention with better parallelism and work partitioning”.



[A Case Study in CUDA Kernel Fusion: Implementing FlashAttention-2 on NVIDIA Hopper Architecture using the CUTLASS Library](https://arxiv.org/abs/2312.11918)



FlashAttention v3 manually optimizes data movement
using byte permute and warp shuffle instructions to bypass
shared memory in layout conversions—an approach that has
not yet been implemented in DL compilers.



## Native Sparse Attention(NSA)



## Flex Attention

https://pytorch.org/blog/flexattention/



