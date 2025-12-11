# Agentic Receipts: Ensembles, OCR, and Anomaly Decisions

_By Emily, John, Luke, Michael, and Raghu — for Dr. Ghosh’s class_

_By Emily, John, Luke, Michael, and Raghu_

> “AI agents have moved from experimental to an essential part of an organization’s tech stack. But how are enterprises actually using them?”

![Header image](https://raw.githubusercontent.com/RogueTex/StreamingDataforModelTraining/main/assets/images/pipeline_summary.png)

## TL;DR
- Problem: Automate receipt understanding—classify, extract vendor/date/total, flag suspicious cases, and route to APPROVE / REVIEW / REJECT.
- Approach: Four ensembles (classification, OCR, field extraction, anomaly detection) orchestrated by a LangGraph agentic pipeline with retries and human-in-the-loop.
- Results (100-test set): Classification 98% accuracy; Field Extraction 99.08% accuracy; Anomaly Ensemble 98% accuracy (F1 0.98, AUC 0.99); OCR ~75% avg confidence.
- Demo: Hugging Face Spaces: `https://huggingface.co/spaces/Rogue2003/Receipt_Agent`.
- Code: https://github.com/RogueTex/StreamingDataforModelTraining

## Why We Built This
Receipts show up as scans, photos, PDFs, and screenshots. Fixed pipelines break on edge cases—blurry photos, missing fields, odd layouts. We needed something that adapts, explains its choices, and improves with feedback without retraining end-to-end. The cost of a bad approval is high; slow reviews stall reimbursements.

### Stakes & edge cases
Bad approvals cost money; slow reviews stall reimbursements. Edge cases include multi-currency, taxes included/excluded, logo-only vendors, handwritten totals, partial crops, and low-light captures.

## Data & Features (Receipts)
Every receipt is distilled into signals that feed the pipeline:
- **Sources:** Synthetic receipts (varied vendors, currencies, lighting, skew) plus a 100-receipt held-out test set that mimics internal-like scenarios for quick iteration.
- **Eight anomaly features:** amount, log_amount, vendor length, date validity, number of items, hour, amount per item, weekend flag.
- **OCR tokens with bounding boxes** feed LayoutLMv3 to preserve text + layout + visual cues.
- **Raw images** power the ViT/ResNet classifiers and the multi-OCR ensemble.
- **Human feedback loop:** corrections in the Gradio app tune vendor/date patterns and anomaly labels over time.

### Pre-Processing & Exploration
- Image cleanup: resize, denoise, deskew on low-confidence cases; normalize to RGB.
- OCR normalization: merge overlapping boxes (IoU>0.5), lowercase/strip tokens, normalize dates/amounts.
- Synthetic validation: spot-check skew/lighting/handwritten variants before training to ensure coverage.

## The Four Ensembles (In Plain English)

### 1) Document Classification (ViT + ResNet + Stacking)
- Base models: ViT-Tiny, fine-tuned ViT-10k, ResNet18.
- Meta-learner: XGBoost (LogReg/RandomForest backups).
- Why: ViT sees global layout; ResNet catches textures; a learned meta-learner balances them.
- Outcome: 98% accuracy on receipts vs. non-receipts.

**Tiny snippet (intuition):**
```python
# features = [P_vit, P_resnet, P_vit10k, conf_vit, conf_resnet, ...]
final_prob = xgboost_meta.predict_proba(scaler.transform(features))
```

### 2) OCR Ensemble (EasyOCR + TrOCR + PaddleOCR + Tesseract)
- Combine by region (IoU>0.5), then vote by confidence.
- Why: Different OCR engines fail on different fonts/angles; fusion reduces single-engine bias.
- Outcome: ~75% average confidence on challenging receipts.
![OCR evaluation](https://raw.githubusercontent.com/RogueTex/StreamingDataforModelTraining/main/assets/images/ocr_evaluation.png)

### 3) Field Extraction Ensemble (LayoutLMv3 + Regex + Position + NER)
- Weights: LayoutLM 35%, Regex 25%, Position 20%, NER 20% with a 1.2× agreement bonus.
- Why: LayoutLM understands spatial context; regex nails dates/amounts; position handles standard layouts; NER helps with logos/names.
- Outcome: 99.08% accuracy on vendor/date/total.
![Field extraction](https://raw.githubusercontent.com/RogueTex/StreamingDataforModelTraining/main/assets/images/layoutlm_field_extraction.png)

### 4) Anomaly Detection Ensemble (Isolation Forest + XGBoost + HistGradientBoosting + One-Class SVM)
- Weights: 35%, 30%, 20%, 15% respectively.
- Decision: Weighted average score plus majority vote (≥2 of 4 must flag) to avoid single-model veto.
- Why: Different models catch different “weirdness”: outliers, learned fraud patterns, NaN-robust boundaries.
- Outcome: 98.0% accuracy, F1 0.98, AUC 0.99 (conservative).
![Anomaly evaluation](https://raw.githubusercontent.com/RogueTex/StreamingDataforModelTraining/main/assets/images/anomaly_detection_evaluation.png)
![Anomaly comparison](https://raw.githubusercontent.com/RogueTex/StreamingDataforModelTraining/main/assets/images/anomaly_model_comparison.png)
![Anomaly confusion matrix](https://raw.githubusercontent.com/RogueTex/StreamingDataforModelTraining/main/assets/images/anomaly_confusion_matrix.png)

## The Agentic Pipeline (LangGraph)
Instead of a brittle linear flow, we run a stateful graph with retries and conditional routing:
```
[INGEST] → [CLASSIFY] → [OCR] → [EXTRACT] → [ANOMALY] → [ROUTE]
```
- Shared state: carries image, OCR results, extracted fields, anomaly result, decision, and a processing log for explainability.
- Conditional routing: if not a receipt, skip to ROUTE and REJECT.
- Retries: OCR retries with image enhancement if confidence < 0.7.
- Human-in-the-loop: REVIEW queue; every 5 corrections trigger model updates (LayoutLM tuning, anomaly retrain, weight adjustments).
![Pipeline summary](https://raw.githubusercontent.com/RogueTex/StreamingDataforModelTraining/main/assets/images/pipeline_summary.png)

**Decision logic (simplified):**
- Not a receipt → REJECT
- Anomaly detected → REVIEW
- Confidence > 90% and no anomalies → APPROVE
- Confidence > 70% → APPROVE
- Else → REVIEW

## Results (100-Sample Test)
| Component | Result |
|-----------|--------|
| Document Classification | 98% accuracy |
| LayoutLM Field Extraction | 99.08% accuracy |
| OCR | ~75% avg confidence |
| Anomaly Detection | 98.0% accuracy, F1 0.98, AUC 0.99 |
| Ensemble Benefit | ~+9% vs. best single model |

### Anomaly Model Breakdown
| Model | Accuracy | F1 | AUC |
|-------|---------|----|-----|
| Isolation Forest | 78.2% | 0.79 | 0.84 |
| XGBoost | 89.5% | 0.87 | 0.93 |
| HistGradientBoosting | 87.3% | 0.85 | 0.90 |
| One-Class SVM | 76.4% | 0.78 | 0.82 |
| **Ensemble** | **98.0%** | **0.98** | **0.99** |

### Failure cases we observed
- Handwritten totals and logo-only vendors without text.
- Extremely low-light or skewed captures where OCR confidence stays low even after enhancement.
- Rare date formats not in regex list (often caught by LayoutLM, but not always).

## Key Design Choices
- LoRA for ViT fine-tuning: ~0.1% parameters trained → faster adaptation, less overfit.
- Optuna + LR Finder: Bayesian search for LR/weight decay/warmup plus quick LR range test for stable starts.
- Confidence-weighted fusion everywhere: OCR, field extraction, anomalies all use weighted voting to avoid single points of failure.
- Explainability built-in: Each stage logs reasons (e.g., “High amount $50,000” or “Invalid vendor”) to support review.

## What Makes It “Agentic”
- Adaptive: Retries and conditional skips prevent brittle failures.
- Stateful: Decisions consider classification, OCR confidence, and anomalies together.
- Feedback-aware: Human corrections update vendors, date formats, anomaly labels, and ensemble weights.
- Composable: Nodes (ingest/classify/ocr/extract/anomaly/route) can be swapped or extended.

## How to Run It
- Colab (full pipeline): Open `NewVerPynbAgent.ipynb` from the repo (GPU recommended).
- Demo (Spaces): Hugging Face Spaces Gradio app: `https://huggingface.co/spaces/Rogue2003/Receipt_Agent`.
- Local: `pip install -r requirements.txt` (or use `huggingface_spaces/requirements.txt`) and run `huggingface_spaces/app.py`.

## Visuals You Can Reuse (paths relative to repo)
- `assets/images/pipeline_summary.png` (end-to-end view)
- `assets/images/anomaly_detection_evaluation.png` (performance)
- `assets/images/anomaly_model_comparison.png` (model comparison)
- `assets/images/anomaly_confusion_matrix.png` (confusion matrix)
- `assets/images/ocr_evaluation.png` (OCR comparisons)
- `assets/images/layoutlm_field_extraction.png` (field extraction example)

## Lessons Learned
- Ensembles beat single models across classification, extraction, and anomaly detection.
- Confidence and agreement matter: weighting + majority votes reduce false positives without missing true anomalies.
- Agentic orchestration is worth it: retries + conditional routing prevent brittle failures.
- Human feedback closes the loop: every correction improves vendors, date formats, and anomaly labels.

## Conclusion
The agentic, feedback-aware ensemble let us keep accuracy high on messy receipt inputs while explaining why we approved, reviewed, or rejected. Strong baselines (ViT/ResNet/LayoutLM), lightweight preprocessing, and a tight feedback loop mean fewer brittle failures without constant full retrains. Next steps focus on multilingual OCR, richer anomaly signals, and automatic threshold tuning per customer.

## Future Work
- Swap EasyOCR with newer OCR (e.g., TrOCR full-time) and measure impact.
- Add receipt-language detection for multilingual pipelines.
- Expand anomaly features (merchant frequency drift, geo/time coherence).
- Auto-calibrate thresholds per customer with small labeled batches.

## References
- LangGraph documentation: https://python.langchain.com/docs/langgraph
- LayoutLMv3 paper: https://arxiv.org/abs/2204.08376
- EasyOCR: https://github.com/JaidedAI/EasyOCR
- XGBoost: https://arxiv.org/abs/1603.02754

## Links
- Demo (Spaces): `https://huggingface.co/spaces/Rogue2003/Receipt_Agent`
- Repo: https://github.com/RogueTex/StreamingDataforModelTraining
- Notebook (Colab badge in README): `NewVerPynbAgent.ipynb`

---
Professional, narrative-style, with images and final results summarized for a broad Medium audience.
