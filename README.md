# Low-Resource NMT for African Languages: Fine-Tuning NLLB-200 with LoRA, Augmentation and Quality Control

**2nd Place – DSN Bootcamp In-House Hackathon**

## Overview

This project addresses neural machine translation (NMT) for low-resource African languages (Yoruba, Igbo, Hausa) into English.
The solution combines:

* Fine-tuning the multilingual model NLLB-200 Distilled 600M using parameter-efficient training (LoRA).
* Rigorous data quality control: normalization, deduplication, and noise filtering.
* Data augmentation via external corpora (JW300, FLORES-101) and back-translation.
* Curriculum learning via length-ratio filtering.
* Robust evaluation using BLEU and WER (word error rate).
* Optimized training for GPU (NVIDIA T4) with fp16 precision.

## Achievement

This approach placed **2nd** in the DSN Bootcamp In-House Hackathon.
It demonstrates strong performance on a real-world low-resource translation challenge and offers a reusable pipeline for similar tasks.

## Motivation

African languages are under-represented in modern translation systems. Bridging this gap enables inclusive digital tools, supports linguistic diversity, and promotes equitable access to AI technologies.
By combining distilled multilingual models, LoRA, and robust data augmentation, this work provides a scalable solution for low-resource NMT.

## Methodology

### 1. Data Preparation and Quality Control

* Load training and testing data; remove missing values and duplicates.
* Normalize text using Unicode NFKC.
* Filter out noise: very short/long sentences and repeated characters.
* Compute **length ratio = target_length / max(source_length, 1)** and filter to 0.5–3.0.
* Split by language (Yoruba, Igbo, Hausa) with an 80/20 train/validation ratio.

### 2. Data Augmentation

* Load JW300 (sample ~5%) and FLORES-101 corpora.
* Normalize and clean augmented data.
* Merge augmented data with the training set.
* Apply limited back-translation from English to African languages to generate pseudo-parallel data.

### 3. Tokenization and Dataset Setup

* Model: `facebook/nllb-200-distilled-600M`.
* Set tokenizer target language (`tgt_lang = eng_Latn`) and dynamic source language mapping (`lang_to_code`).
* Convert pandas DataFrames into `datasets.Dataset` and `DatasetDict`.
* Map a preprocessing function to tokenize input-output pairs.

### 4. Model Adaptation via LoRA

* Configure LoRA with `r=8`, `lora_alpha=32`, and `lora_dropout=0.1`, targeting the `q_proj` and `v_proj` modules.
* Use `get_peft_model()` so only adapter parameters are trained, improving efficiency.

### 5. Training

* Use `Seq2SeqTrainer` and `Seq2SeqTrainingArguments` with the following key parameters:

  * Learning rate: 3e-4
  * Batch size: 8 (train), 16 (eval)
  * Gradient accumulation: 2
  * Epochs: 2
  * Weight decay: 0.01
  * Warm-up steps: 200
  * FP16 precision enabled
  * Evaluation and checkpointing every 1000 steps
  * Early stopping patience: 3
* Evaluation metric: WER (via `evaluate.load('wer')`).
* Save best model to `./final_model`.

### 6. Inference and Submission

* Load adapter-tuned model via `PeftModel.from_pretrained()` if available.
* For each test example, set source language dynamically.
* Generate translations with beam search (`num_beams=4`, `temperature=0.5`, `length_penalty=1.0`, `repetition_penalty=1.1`).
* Post-process output text to clean punctuation and spacing.
* Save submission file with columns (`ID`, `Output text`) to `submission.csv`.

## Why This Works

* A strong multilingual backbone (NLLB-200 Distilled) captures low-resource linguistic features effectively.
* LoRA enables efficient fine-tuning with low computational cost.
* Data augmentation increases diversity and robustness while maintaining data quality.
* Combining BLEU and WER captures both semantic and structural translation quality.

## Results Summary

* Ranked **2nd place** in the DSN Bootcamp In-House Hackathon.
* Consistent performance across Yoruba, Igbo, and Hausa.
* Demonstrated a replicable, efficient approach suitable for research or deployment.

## Repository Structure

```
├── README.md                # This file  
├── standard-approach.ipynb  # Cleaned notebook  
├── model/                   # Trained model and adapters  
└── results/                 # Outputs and submission file  
```

## Getting Started

1. Clone the repository.
2. Install dependencies:

   ```bash
   pip install transformers datasets evaluate jiwer peft torch opustools-pkg
   ```
3. Update `data_dir` in the notebook to your dataset path.
4. Run notebook cells in sequence: cleaning → augmentation → tokenization → training → inference → submission.

## Notes and Tips

* Adjust batch size and accumulation steps based on GPU memory.
* The `length_ratio` filter helps prevent extreme mismatches that harm model stability.
* Limit pseudo back-translation to small subsets to prevent noise.
* Track BLEU and WER for a balanced view of translation quality.

## References

* NLLB Paper: [https://arxiv.org/abs/2207.04672](https://arxiv.org/abs/2207.04672)
* LoRA: Hu et al., “Low-Rank Adaptation of Large Language Models” (2021)
* FLORES-101: [https://github.com/facebookresearch/flores](https://github.com/facebookresearch/flores)
* JW300: [https://opus.nlpl.eu/JW300.php](https://opus.nlpl.eu/JW300.php)

## Author

**Oluwaferanmi Oladepo [github.com/Eniiyanu](github.com/Eniiyanu)**
