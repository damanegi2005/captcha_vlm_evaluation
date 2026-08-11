# CAPTCHA VLM Evaluation

This repository contains the code developed during a semester-long research project at INRIA / INSA Lyon. The project evaluates the performance of Vision-Language Models (VLMs) on different CAPTCHA tasks using automated prompting and quantitative evaluation metrics.

## Project Overview

The project aims to evaluate how well VLMs solve different CAPTCHA types. The evaluation pipeline automatically:

- Loads CAPTCHA images and instructions
- Sends requests to the OpenAI API
- Compares model predictions with ground truth
- Computes task-specific evaluation metrics
- Saves the results for further analysis

## CAPTCHA Categories

- OCR
- Object Selection
- Coordinate Localization
- Click Order
- Arithmetic
- Common Sense Reasoning
- Rule-based Filtering

## Evaluation Metrics

- Exact Match Accuracy
- Character Accuracy (OCR)
- F1 Score (Object Selection)
- IoU (Bounding Box Tasks)
- Latency

## Technologies

- Python
- OpenAI GPT-4.1-mini API
- Pandas

## Repository Structure

```
code/
dataset/
results/
README.md
```

## Notes

- The original CAPTCHA datasets are not included in this repository due to research and licensing considerations.
- An OpenAI API key is required to run the evaluation scripts.
