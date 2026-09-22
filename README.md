<p align="center">
  <img src="cerist_logo.webp" width="45" align="left">
  <img src="kalima_logo.png" width="65" align="right" style="vertical-align: -5px;">

  <h1 align="center">AraManuscript-VL: Arabic Manuscript Recognition</h1>
</p>

<p align="center">
  Fine-tuning <strong>Qwen2.5-VL-3B-Instruct</strong> for recognizing text from historical Arabic manuscript lines.
</p>


<p align="center">
  <a href="https://huggingface.co/Qwen/Qwen2.5-VL-3B-Instruct">
    <img src="https://img.shields.io/badge/Base%20Model-Qwen2.5--VL--3B--Instruct-blue" alt="Base Model">
  </a>
  <a href="https://huggingface.co/dsaru-cerist/AraManuscript-VL">
    <img src="https://img.shields.io/badge/Fine--Tuned%20Model-Hugging%20Face-yellow" alt="Fine-Tuned Model">
  </a>
  <a href="https://huggingface.co/spaces/dsaru-cerist/AraManuscript-VL-demo">
    <img src="https://img.shields.io/badge/Live%20Demo-Hugging%20Face%20Space-orange" alt="Live Demo">
  </a>
</p>

<p align="center">
  <img src="manuscript_line.png" alt="Arabic historical manuscript" width="350">
</p>

---

## Overview

Historical Arabic manuscripts are challenging for automatic handwritten text recognition due to variations in handwriting, historical writing conventions, degraded documents, faded ink, image noise, and differences in document quality.

This project adapts Qwen2.5-VL-3B-Instruct for line-level Arabic manuscript recognition. Given an image of a manuscript line, the model generates its corresponding Arabic transcription.

The adaptation is performed using **LoRA** through the Hugging Face **PEFT** framework.

---

## Results

| Metric | Score |
|:---|---:|
| **Character Error Rate (CER)** | **6.58%** |
| **Word Error Rate (WER)** | **22.34%** |

Lower values indicate better recognition performance.

## Fine-Tuning

Training was performed on paired manuscript images and their corresponding Arabic transcriptions.

### Configuration

| Parameter | Value |
|---|---:|
| Rank | 128 |
| Alpha | 256 |
| Dropout | 0.05 |
| Learning rate | 1e-5 |
| Batch size | 1 |
| Gradient accumulation | 16 |
| Epochs | 2 |
| Precision | BF16 |
| LR scheduler | Cosine |

---

## Datasets

The model was trained and evaluated using the following publicly available datasets:

- **Kalima Dataset** (Bouchal et al., 2025 [[**1**]](#ref1))
- **KHATT Dataset** (Mahmoud et al., 2014 [[**2**]](#ref2))
- **Hcima Dataset** (official website) https://hicma.net/goal.html
- **Muharaf Dataset** ([Hugging Face]([https://huggingface.co/datasets/aamijar/muharaf-public]))
- **Nakba Dataset** ([Hugging Face]([https://huggingface.co/datasets/U4RASD/omar-al-saleh-manuscripts-full]))
- **Baybars Dataset** ([Hugging Face]([https://huggingface.co/datasets/calfa-ai/baybars]))
- **Iskandar Dataset** ([Hugging Face](https://huggingface.co/datasets/calfa-ai/iskandar))
- **RASM Dataset** (Keinan-Schoonbaert and British Library, 2019 [[**3**]](#ref3)))
- **RASAM-1 Dataset** ([Hugging Face](https://huggingface.co/datasets/calfa-ai/RASAM-1))
- **RASAM-2 Dataset** ([Hugging Face](https://huggingface.co/datasets/calfa-ai/RASAM-2))
- **Egypt word Dataset** ([Hugging Face](https://huggingface.co/datasets/OmarMDiab/Egyptian-Handwriting-Dataset))

## Model Integration

The fine-tuned model is integrated into **Kalima OCR**, a comprehensive AI platform for processing and digitizing historical Arabic manuscripts and heritage documents.

Kalima OCR provides a wide range of AI-powered capabilities, including automatic handwriting recognition, manuscript analysis and comparison, intelligent manuscript interaction, heritage content summarization, Quranic text verification and diacritization, Hadith referencing, poetry verification, biographical text processing, and audio transcription.

<p>
  <a href="https://www.kalima-ocr.com/#demo" style="color: black; text-decoration: none;">
    <strong>Explore Kalima OCR and its functionalities </strong>
  </a>
</p>

## Training

Install the required dependencies:

```bash
pip install -r requirements.txt
```

## Citation

If you use this model, fine-tuning code, or results in your research, please cite this work:

```bibtex
@software{dsaru-cerist2026qwen_arabic_manuscript,
  author  = {Khawla Belgacem and Ahror Belaid},
  title   = {Qwen2.5-VL Arabic Manuscript Recognition},
  year    = {2026},
  url     = {https://huggingface.co/dsaru-cerist/AraManuscript-VL}
}
```
## References

<a id="ref1"></a>

[1] H. Bouchal, A. Belaid, and F. Meziane, "Towards accurate recognition of historical Arabic manuscripts: A novel dataset and a generalizable pipeline," *ACM Transactions on Asian and Low-Resource Language Information Processing*, vol. 24, no. 10, pp. 1–30, 2025.

<a id="ref2"></a>

[2] S. A. Mahmoud, I. Ahmad, W. G. Al-Khatib, M. Alshayeb, M. T. Parvez, V. Märgner, and G. A. Fink, "KHATT: An open Arabic offline handwritten text database," *Pattern Recognition*, vol. 47, no. 3, pp. 1096–1112, 2014. doi:10.1016/j.patcog.2013.08.009.

<a id="ref3"></a>

[3] A. Keinan-Schoonbaert and British Library, Ground Truth Transcriptions for Training OCR of Historical Arabic Handwritten Texts, 2019. doi:10.23636/1135.
