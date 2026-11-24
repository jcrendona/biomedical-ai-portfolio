# Local LLM inference for domain Q&A (GPT-2 vs Llama-2-13B-chat)

An exploratory notebook that compares a small pre-trained language model with a quantised chat model, run entirely on a single Colab T4 GPU. The questions are domain-specific, about electrical muscle stimulation (EMS) training, in Spanish.

## What the notebook does

1. **GPT-2 baseline** (Hugging Face `transformers`, TensorFlow): beam-search generation for a factual prompt. The output degenerates into repetition ("It's the capital of Colombia. It's the capital of Colombia…"). This is a known failure mode of small models that are not instruction-tuned.
2. **Llama-2-13B-chat, GGML q5_1** (`llama-cpp-python` with cuBLAS): downloads the 9.8 GB quantised model, offloads layers to the GPU and answers an EMS-physiology question with a chat prompt template. The answer is structured and in Spanish, covering low-, medium- and high-frequency effects. The notebook also contains utilities to split the dialogue and log answers.

## Why it is here

It shows practical familiarity with model quantisation, GPU offloading and prompt templates for running open models under tight hardware limits. That knowledge is relevant to on-premise or privacy-preserving clinical NLP.

## Honest notes

- No fine-tuning or training takes place. This is inference and prompt engineering only.
- Generated answers were not fact-checked against the literature and should not be read as medical guidance.

## Stack
Python · Hugging Face Transformers · llama-cpp-python (GGML, cuBLAS) · Google Colab T4
