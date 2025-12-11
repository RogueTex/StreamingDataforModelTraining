# Receipts, Ensembles, and an Agentic Brain: How We Built a Trustworthy Pipeline

_By Emily, John, Luke, Michael, and Raghu — for Dr. Ghosh’s class_

> We set out to keep expense approvals fast and honest: separate receipts from noise, pull out the fields that matter, and flag the weird ones without drowning reviewers.

![Pipeline summary](https://raw.githubusercontent.com/RogueTex/StreamingDataforModelTraining/main/assets/images/pipeline_summary.png)

## TL;DR
- **Goal:** Approve/review/reject receipts with speed and confidence.
- **Approach:** Four ensembles (classification, OCR, field extraction, anomaly detection) orchestrated by a **LangGraph agent** that retries, routes, and listens to human feedback.
- **Data:** Synthetic receipts plus a 100-receipt held-out set; feedback data captured in the app.
- **Results:** 98% doc classification, 99.08% field extraction, ~75% OCR confidence, 98% anomaly F1/AUC.
- **Try it:** Hugging Face Spaces demo and full repo links below.

## Why It Matters
Bad approvals waste money; slow reviews frustrate teams. Receipts arrive as scans, phone photos, PDFs, and screenshots—often skewed, dim, or missing fields. A brittle pipeline breaks; an agentic one adapts, explains, and improves.

## Data and Pre-Processing
- **Sources:** Synthetic receipts (varied vendors, currencies, lighting, skew, handwriting) plus a 100-receipt held-out test set to approximate real cases.
- **Cleanup:** Resize, denoise, deskew when OCR confidence drops; normalize to RGB.
- **OCR normalization:** Merge overlapping boxes (IoU>0.5), lowercase/strip tokens, normalize dates/amounts.
- **Signals:** Eight anomaly features (amount, log_amount, vendor length, date validity, item count, hour, amount per item, weekend), OCR tokens + boxes for LayoutLMv3, and raw images for ViT/ResNet and multi-OCR.
- **Feedback loop:** Gradio app captures reviewer corrections; those update vendor/date patterns and anomaly labels.

## The Four Ensembles (Plain English)
1) **Document Classification (ViT + ResNet + stacking)**  
   - Global layout + texture cues; meta-learner balances them.  
   - Outcome: **98%** accuracy.

2) **OCR Ensemble (EasyOCR + TrOCR + PaddleOCR + Tesseract)**  
   - Group by overlapping boxes, vote by confidence.  
   - Outcome: ~**75%** average confidence on tough receipts.  
   ![OCR evaluation — fusion lifts confidence on skewed/low-light receipts](https://raw.githubusercontent.com/RogueTex/StreamingDataforModelTraining/main/assets/images/ocr_evaluation.png)

3) **Field Extraction (LayoutLMv3 + Regex + Position + NER)**  
   - Weights 35/25/20/20 with a 1.2× agreement bonus.  
   - Outcome: **99.08%** accuracy on vendor/date/total.  
   ![Field extraction — ensemble agreement stabilizes vendor/date/total](https://raw.githubusercontent.com/RogueTex/StreamingDataforModelTraining/main/assets/images/layoutlm_field_extraction.png)

4) **Anomaly Detection (Isolation Forest + XGBoost + HistGB + One-Class SVM)**  
   - Weighted average + majority vote (≥2 of 4 must flag).  
   - Outcome: **98.0%** accuracy, F1 **0.98**, AUC **0.99**.  
   ![Anomaly evaluation — ensemble bumps AUC vs individual models](https://raw.githubusercontent.com/RogueTex/StreamingDataforModelTraining/main/assets/images/anomaly_detection_evaluation.png)  
   ![Anomaly comparison — weighted vote outperforms single models](https://raw.githubusercontent.com/RogueTex/StreamingDataforModelTraining/main/assets/images/anomaly_model_comparison.png)  
   ![Anomaly confusion matrix — balanced approvals vs reviews/rejects](https://raw.githubusercontent.com/RogueTex/StreamingDataforModelTraining/main/assets/images/anomaly_confusion_matrix.png)

## Agentic Orchestration (LangGraph)
- **Graph, not a line:** `[INGEST] → [CLASSIFY] → [OCR] → [EXTRACT] → [ANOMALY] → [ROUTE]`
- **Conditional routing:** If not a receipt, jump to ROUTE and reject; otherwise continue.
- **Retries:** OCR re-runs with image enhancement if confidence < 0.7.
- **Shared state:** Carries image, OCR results, extracted fields, anomaly score, decision, and a processing log for explainability.
- **Feedback-aware:** Every few corrections triggers updates to patterns, thresholds, and ensemble weights.

## Results on 100-Receipt Test Set
| Component | Result |
|-----------|--------|
| Document Classification | 98% accuracy |
| LayoutLM Field Extraction | 99.08% accuracy |
| OCR | ~75% avg confidence |
| Anomaly Detection | 98.0% accuracy, F1 0.98, AUC 0.99 |
| Ensemble Benefit | ~+9% vs. best single model |

### Failure Cases We Saw
- Handwritten totals and logo-only vendors with little text.
- Very low-light or highly skewed images where OCR remains weak even after enhancement.
- Rare date formats outside the regex list (LayoutLM helps but can still miss).

### How Feedback Improves It
- Reviewers correct vendor/date/total; patterns update and boost regex/NER matches. Example: when “Starbuks” is corrected to “Starbucks,” vendor fuzzy-matching is updated so the next misspelling auto-resolves.
- Anomaly labels from reviewers recalibrate the weighted vote and thresholds.
- Next receipts with similar patterns get routed more accurately (fewer false reviews/rejects).

## How to Run It
- **Colab pipeline:** Open `NewVerPynbAgent.ipynb` (GPU recommended).
- **Deployed Demo :** https://huggingface.co/spaces/Rogue2003/Receipt_Agent
- **Local:** `pip install -r requirements.txt` (or `huggingface_spaces/requirements.txt`) and run `huggingface_spaces/app.py`.

## Outcomes
- Ensembles beat single models across classification, extraction, and anomaly detection.
- Confidence-weighted votes and agreement checks reduce brittle failures.
- Agentic orchestration (routing + retries) matters as much as model choice.
- Human feedback closes the loop without constant full retrains.

## Challenges
- OCR brittleness on low-light, skewed, or handwritten receipts.
- Currency and language coverage is narrow; multilingual OCR and locale-aware parsing are needed.
- Business rules differ by organization (thresholds, vendors, categories), so policies must be configurable.

## Conclusion and Next Steps
This agentic, feedback-aware stack keeps receipt decisions fast, explainable, and resilient to messy inputs—exactly what we set out to deliver for Dr. Ghosh’s Advanced Machine Learning course (Fall ’25). Thank you, Dr. Ghosh, for your guidance throughout the semester. Next up: stronger multilingual OCR, richer anomaly features (frequency drift, geo/time coherence), and per-customer threshold auto-tuning.

## References
- LangGraph docs: https://python.langchain.com/docs/langgraph
- LayoutLMv3: https://arxiv.org/abs/2204.08376
- EasyOCR: https://github.com/JaidedAI/EasyOCR
- XGBoost: https://arxiv.org/abs/1603.02754

## Links
- **Demo:** https://huggingface.co/spaces/Rogue2003/Receipt_Agent
- **Repo:** https://github.com/RogueTex/StreamingDataforModelTraining
- **Notebook:** `NewVerPynbAgent.ipynb`
