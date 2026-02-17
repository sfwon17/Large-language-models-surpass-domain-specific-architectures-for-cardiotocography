# Large-language-models-surpass-domain-specific-architectures-for-cardiotocography
In this study, we present the first comprehensive benchmark of state-of-the-art architectures for automated antepartum CTG classification. Over 2,500 20-minutes recordings were used to evaluate over 15 models spanning domain-specific, time-series, foundation, and language-model categories under a unified framework. 

## Overview 
This repository provides example code for fine-tuning Llama models that achieved the best performance. It also includes additional examples, such as the complete instruction set used with GPT-5 mini. The code can be used in conjunction with your antepartum dataset for evaluation purposes. Other publicly available models may be evaluated using their respective codebases, which are accessible online. Please refer to the associated publications for further details.

## Result
| Model                                  | Trainable Parameters | Model Category             | Training Method                           | Input Format           | Sampling Rate |
| -------------------------------------- | -------------------- | -------------------------- | ----------------------------------------- | ---------------------- | ------------- |
| **Domain-Specific Models**             |                      |                            |                                           |                        |               |
| PatchCTG                               | 7.3M                 | CTG-specific               | Trained from scratch                      | Structured time-series | 1 Hz          |
| Conv-PatchCTG                          | 7.3M                 | CTG-specific               | Trained from scratch                      | Structured time-series | 1 Hz          |
| 1D SE-ResNet                           | 45M                  | CTG-specific               | Trained from scratch                      | Structured time-series | 1 Hz          |
| NeuroFetalNet                          | 2.4M                 | CTG-specific               | Trained from scratch                      | Structured time-series | 1 Hz          |
| **General-Purpose Time-Series Models** |                      |                            |                                           |                        |               |
| Informer                               | 1.1M                 | Time-series transformer    | Trained from scratch                      | Structured time-series | 1 Hz          |
| Non-stationary Transformer             | 0.4M                 | Time-series transformer    | Trained from scratch                      | Structured time-series | 1 Hz          |
| TimesNet                               | 8.7M                 | Time-series CNN            | Trained from scratch                      | Structured time-series | 1 Hz          |
| **Time-Series Foundation Models**      |                      |                            |                                           |                        |               |
| Moment                                 | 0.3B                 | Pre-trained time-series FM | Full fine-tuning                          | Structured time-series | 1 Hz          |
| Mantis                                 | 8M                   | Pre-trained time-series FM | Full fine-tuning                          | Structured time-series | 1 Hz          |
| **Large Language Models**              |                      |                            |                                           |                        |               |
| Llama 3 (1B)                           | 9M                   | LLM                        | QLoRA fine-tuning                         | Text (full-length CTG) | 0.5 Hz        |
| Llama 3 (3B)                           | 11M                  | LLM                        | QLoRA fine-tuning                         | Text (full-length CTG) | 0.5 Hz        |
| Llama 3 (8B)                           | 13M                  | LLM                        | QLoRA fine-tuning                         | Text (full-length CTG) | 0.5 Hz        |
| GPT-5-mini (Simple)                    | Unknown              | Proprietary LLM            | One-shot prompting                        | Text (full-length CTG) | 1 Hz          |
| GPT-5-mini (Detailed)                  | Unknown              | Proprietary LLM            | One-shot prompting                        | Text (full-length CTG) | 1 Hz          |
| **Time-LLM Hybrid**                    |                      |                            |                                           |                        |               |
| Time-LLM (Llama 1B)                    | 0.1B                 | LLM hybrid                 | Frozen LLM + alignment training           | Structured + text      | 0.5 Hz        |
| Time-LLM (Llama 3B)                    | 0.1B                 | LLM hybrid                 | Frozen LLM + alignment training           | Structured + text      | 0.5 Hz        |
| Time-LLM (Llama 8B)                    | 0.1B                 | LLM hybrid                 | Frozen LLM + alignment training           | Structured + text      | 0.5 Hz        |
| **Contrastive Learning (CL) Models**   |                      |                            |                                           |                        |               |
| CNN Encoder (CL)                       | 1.5M                 | ResNet-style CNN           | Contrastive pre-training + classification | Structured time-series | 0.5 Hz        |
| Llama 1B (CL-embedding)                | 22M                  | LLM hybrid                 | CL pre-training + QLoRA fine-tuning       | Structured embeddings  | 0.5 Hz        |
| Llama 3B (CL-embedding)                | 48M                  | LLM hybrid                 | CL pre-training + QLoRA fine-tuning       | Structured embeddings  | 0.5 Hz        |
| Llama 8B (CL-embedding)                | 58M                  | LLM hybrid                 | CL pre-training + QLoRA fine-tuning       | Structured embeddings  | 0.5 Hz        |

## Notes
1. The study and dataset are intended for antepartum evaluation, specifically for CTG recordings obtained prior to the onset of labor. Model performance may differ in the intrapartum setting.

2. The study was conducted using CTG data split at the patient level to prevent data leakage. We strongly encourage users to include as many unique patients as possible. Our training set comprises more than 3,000 distinct patients.

3. Due to the input representation requirements of Llama models, the time-series data are formatted as textual sequences rather than processed in a conventional numerical time-series format.
