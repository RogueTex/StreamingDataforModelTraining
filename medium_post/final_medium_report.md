# Smart Doc Approver: Building an Agentic Receipt Pipeline That Actually Works!

*By Emily Caraher, Luke Hartfield, John MacDonald, Michael Ovassapian, Raghu Subramanian*  
*MIS 382N — Advanced Machine Learning | Dr. Joydeep Ghosh*

> We set out to keep expense approvals fast and honest: separate receipts from noise, pull out the fields that matter, and flag the weird ones without drowning reviewers.

![Pipeline summary](https://raw.githubusercontent.com/RogueTex/StreamingDataforModelTraining/main/assets/images/header_image.png)

## TL;DR
We built an end‑to‑end, agentic receipt‑processing system that:
Classifies images as receipts vs. non‑receipts
Runs OCR with automatic retries on low‑quality images
Extracts vendor, date, total and amount using a 4‑way ensemble
Detects suspicious receipts with an anomaly ensemble
Routes documents to approve / review / reject using a LangGraph workflow
Learns from human corrections via a Gradio UI
- **Try it:** [Hugging Face Spaces demo](https://huggingface.co/spaces/Rogue2003/Receipt_Agent) and full repo links below.

---

## Problem Statement
Businesses process thousands of receipts and invoices every day. Someone has to review these documents, extract key fields, and make approval decisions. The status quo is:
- Time-consuming: reviewers spend hours on repetitive checks.
- Error-prone: manual entry and brittle rules miss edge cases.
- Non-adaptive: traditional pipelines cannot learn from corrections.

The stakes are real: bad approvals waste money; slow reviews frustrate teams. Receipts arrive as scans, photos, PDFs, and screenshots—often skewed, dim, or missing fields. A brittle pipeline breaks; an agentic one adapts, explains, and improves.

---

## Data Collection & Description
- **Sources:** Real document datasets (RVL-CDIP for classification pretrain/fine-tune; SROIE and CORD for receipts and field annotations) plus synthetic receipts (varied vendors, currencies, lighting, skew, handwriting) and a 100-receipt held-out set that mimics real scenarios for quick iteration.
- **Signals:** Eight anomaly features (amount, log_amount, vendor length, date validity, item count, hour, amount per item, weekend); OCR tokens + boxes for LayoutLMv3; raw images for ViT/ResNet and multi-OCR.
- **Feedback loop:** The Gradio app captures reviewer corrections; those updates feed regex/NER/vendor patterns and anomaly labels.

### Pre-Processing & Exploration
- Image cleanup: resize, denoise, deskew when OCR confidence drops; normalize to RGB.
- OCR normalization: merge overlapping boxes (IoU>0.5), lowercase/strip tokens, normalize dates/amounts.
- Synthetic validation: spot-check skew/lighting/handwritten variants before training to ensure coverage.

---

## System Overview: An Agentic Multi-Model Pipeline
Instead of a brittle linear flow, we run a **stateful LangGraph** with shared state, retries, and conditional routing:
```
[INGEST] → [CLASSIFY] → [OCR] → [EXTRACT] → [ANOMALY] → [ROUTE]
```
- **Conditional routing:** If not a receipt, skip to ROUTE and reject; otherwise continue.
- **Retries:** OCR re-runs with image enhancement if confidence < 0.7.
- **Shared state:** Carries image, OCR results, extracted fields, anomaly score, decision, and a processing log for explainability.
- **Feedback-aware:** Every few corrections triggers updates to patterns, thresholds, and ensemble weights.



## Ensemble Outlook
We layer four decisions to stay robust: classification filters out non-receipts; OCR turns pixels into text; field extraction pulls vendor/date/total with layout-aware cues; anomaly detection flags suspicious or incomplete receipts before approval. Each layer votes within itself, then hands richer signals to the next, so no single weak step breaks the pipeline. 

The graphic below shows the ensemble approach we use to solve the problem end to end.

![Ensemble outlook](https://raw.githubusercontent.com/RogueTex/StreamingDataforModelTraining/main/assets/images/Ensemble_Outlook.png)

### The Four Ensembles!
1) **Document Classification (ViT + ResNet + stacking)**  
   - Global layout + texture cues; meta-learner balances them.  
   - Outcome: **98%** accuracy on single models; the ensemble gates to 100% on our validation/test set, ensuring only receipts flow downstream.
   ![Classifier evaluation](https://raw.githubusercontent.com/RogueTex/StreamingDataforModelTraining/main/assets/images/classifier_evaluation.png)

2) **OCR Ensemble (EasyOCR + TrOCR + PaddleOCR + Tesseract)**  
   - Group by overlapping boxes, vote by confidence.  
   - Outcome: ~**75%** average confidence on tough receipts.  
   ![OCR evaluation — fusion lifts confidence on skewed/low-light receipts](https://raw.githubusercontent.com/RogueTex/StreamingDataforModelTraining/main/assets/images/ocr_evaluation.png)

   Shown below is OCR extraction on a sample receipt; fields are overlaid with color-coded boxes by confidence to highlight what the ensemble trusts most.
   ![OCR + LayoutLM demonstration](https://raw.githubusercontent.com/RogueTex/StreamingDataforModelTraining/main/assets/images/OCR_plus_LayoutLMdemonstrationSumamry.png)

3) **Field Extraction (LayoutLMv3 + Regex + Position + NER)**  
   - Weights 35/25/20/20 with a 1.2× agreement bonus.  
   - Outcome: **99.08%** accuracy on vendor/date/total.  
   ![Field extraction — ensemble agreement stabilizes vendor/date/total](https://raw.githubusercontent.com/RogueTex/StreamingDataforModelTraining/main/assets/images/layoutlm_field_extraction.png)

4) **Anomaly Detection (Isolation Forest + XGBoost + HistGB + One-Class SVM)**  
   - Weighted average + majority vote (≥2 of 4 must flag).  
   - Outcome: **98.0%** accuracy, F1 **0.98**, AUC **0.99**; the final ensemble flagged essentially all anomalies in our test set.
   ![Anomaly evaluation — ensemble bumps AUC vs individual models](https://raw.githubusercontent.com/RogueTex/StreamingDataforModelTraining/main/assets/images/anomaly_detection_evaluation.png)  

We run anomaly detection to catch outliers, missing fields, and suspicious patterns before anything is auto-approved. Examples we flag:

| Anomaly Type            | Example                     |
|-------------------------|-----------------------------|
| Unusually high amounts  | $50,000 for coffee          |
| Missing critical fields | No vendor name              |
| Invalid dates           | February 30th, 2025         |
| Suspicious timing       | 3 AM transactions           |
| Data integrity issues   | Negative totals             |
| Statistical outliers    | 500 items on one receipt    |


### Inside Each Ensemble
For classification, we blend ViT-Tiny (LoRA-finetuned for global layout), a fine-tuned ViT-10k, and ResNet18 for texture, then stack them with XGBoost so a meta-learner can trust the right signals. OCR combines EasyOCR, TrOCR (fine-tuned on receipts), PaddleOCR, and Tesseract, leaning on weighted voting because each engine fails on different fonts and angles. Field extraction mixes LayoutLMv3 (fine-tuned), regex for dates/amounts, positional heuristics for common layouts, and NER for vendors, weighted 35/25/20/20 with a 1.2× agreement bonus to reward consensus. Anomalies use Isolation Forest (outliers), XGBoost (supervised patterns), HistGradientBoosting (robust to missingness), and One-Class SVM (boundary), with a weighted vote plus a majority gate to avoid lone-model vetoes.

### Fine-Tuning & Search for better results!
- **LoRA on vision models:** Used for ViT-Tiny (classification) and LayoutLMv3 (extraction) to keep tuning lightweight—~0.1% of parameters trained, faster convergence, less overfit to receipts.
- **TrOCR fine-tune:** We fine-tuned TrOCR on receipt-like text to handle tough fonts/angles; combined via weighted voting so it boosts, not breaks, OCR.
- **Optuna + LR Finder:** Searched LR/weight decay/warmup with Optuna, then confirmed stable starts with a quick LR range test before long runs.
- **Stacking/weights learned on val:** XGBoost meta-learner and ensemble weights were calibrated on validation splits to trust the right model per case.
- **Confidence gates and retries:** OCR retries with enhancement if confidence < 0.7; extraction uses agreement bonuses to reward consensus and avoid lone-model drift.

### Why We Ensemble
Receipts are messy and diverse—fonts, crops, lighting, vendors, and handwritten quirks all vary. No single model wins everywhere, so we use voting and confidence-weighted fusion to raise the floor and avoid brittle failures. Agreement bonuses and majority gates prevent a lone bad call, while stacking learns which signals to trust for each case.

---

## Results!
| Component | Result |
|-----------|--------|
| Document Classification | 98% accuracy |
| LayoutLM Field Extraction | 99.08% accuracy |
| OCR | ~75% avg confidence |
| Anomaly Detection | 98.0% accuracy, F1 0.98, AUC 0.99 |
| Ensemble Benefit | ~+9% vs. best single model |

Below is a walkthrough of the earlier sample receipt: the key fields were detected, the anomalies were cleared, and the agent approved it end to end.

![End-to-end results summary](https://raw.githubusercontent.com/RogueTex/StreamingDataforModelTraining/main/assets/images/AGENTIC_WORFLOW_FINAL_RESULT_SUMMARY.png)

### Failure Cases We Saw
- Handwritten totals and logo-only vendors with little text.
- Very low-light or highly skewed images where OCR remains weak even after enhancement.
- Rare date formats outside the regex list (LayoutLM helps but can still miss).

### How Feedback Improves It
- Reviewers correct vendor/date/total; patterns update and boost regex/NER matches. Example: when “Starbuks” is corrected to “Starbucks,” vendor fuzzy-matching is updated so the next misspelling auto-resolves.
- Anomaly labels from reviewers recalibrate the weighted vote and thresholds.
- Next receipts with similar patterns get routed more accurately (fewer false reviews/rejects).

---

## Key Design Choices
- **LoRA for ViT fine-tuning:** Train ~0.1% of parameters; faster adaptation, less overfit.
- **Optuna + LR Finder:** Bayesian search for LR/weight decay/warmup + quick LR range test for stable starts.
- **Confidence-weighted fusion everywhere:** OCR, field extraction, anomalies all use weighted voting to avoid single points of failure.
- **Explainability built-in:** Each stage logs reasons (e.g., “High amount $50,000” or “Invalid vendor”) to support review.

---

## Now, What Makes It “Agentic”?
We orchestrate the whole pipeline as an agentic workflow: the graph routes receipts through classification → OCR → extraction → anomaly, retries when confidence is low, and stops early for non-receipts. 

The graphic below show how the agent stitches these pieces together end to end.

![Agentic workflow](https://raw.githubusercontent.com/RogueTex/StreamingDataforModelTraining/main/assets/images/agentic_Workflow.png)

The image below shows each agent’s results on the same receipt, with feedback entry points for the relevant sections.
![Agentic workflow (full view)](https://raw.githubusercontent.com/RogueTex/StreamingDataforModelTraining/main/assets/images/Agentic_full_worfklow_result_2.png)
- **Adaptive:** Retries with enhanced images; conditional skips for non-receipts.
- **Stateful:** Decisions consider classification, OCR confidence, and anomalies together.
- **Feedback-aware:** Human corrections update vendors, date formats, anomaly labels, and ensemble weights.
- **Composable:** Nodes (ingest/classify/ocr/extract/anomaly/route) can be swapped or extended.

### Agentic Implementation (LangGraph)
We wired the pipeline as a LangGraph state machine so every node reads/writes a shared state and the graph decides what to do next:
```python
from langgraph.graph import StateGraph, END
from typing import TypedDict, List, Dict

class AgentState(TypedDict):
    image: bytes
    ocr_results: List[Dict]
    extracted_fields: Dict
    anomaly: Dict
    decision: str
    log: List[str]

def classify(state): ...
def run_ocr(state): ...
def extract_fields(state): ...
def detect_anomaly(state): ...
def route(state):
    if not state["decision"]:  # set in anomaly step
        return "route"
    return END

graph = StateGraph(AgentState)
graph.add_node("classify", classify)
graph.add_node("ocr", run_ocr)
graph.add_node("extract", extract_fields)
graph.add_node("anomaly", detect_anomaly)
graph.add_node("route", route)

graph.add_edge("classify", "ocr")
graph.add_edge("ocr", "extract")
graph.add_edge("extract", "anomaly")
graph.add_conditional_edges(
    "anomaly",
    lambda s: s["decision"],  # APPROVE / REVIEW / REJECT
    {"APPROVE": "route", "REVIEW": "route", "REJECT": "route"},
)

workflow = graph.compile()
result = workflow.invoke({"image": uploaded_bytes})
```
- **Shared state:** `image`, `ocr_results`, `extracted_fields`, `anomaly`, `decision`, plus a processing log for explainability.
- **Conditional edges:** After anomaly, the decision drives routing (approve/review/reject) without re-running upstream steps.
- **Retrials:** OCR node retries with enhanced images if confidence < 0.7 before handing off to extraction.
- **Feedback hook:** User corrections (vendor/date/total/anomaly) feed back into patterns and weights, so the next invocation starts smarter.

---

## How to Run It
- **Colab (full pipeline):** Open `NewVerPynbAgent.ipynb` (GPU recommended).
- **Demo (Spaces):** https://huggingface.co/spaces/Rogue2003/Receipt_Agent
- **Local:** `pip install -r requirements.txt` (or `huggingface_spaces/requirements.txt`) and run `huggingface_spaces/app.py`.

---

## Key Outcomes
- **Ensembles beat single models** across classification, extraction, and anomaly detection.
- **Confidence and agreement matter:** weighting + majority votes reduce false positives without missing true anomalies.
- **Agentic orchestration** (routing + retries) matters as much as model choice.
- **Human feedback closes the loop** without constant full retrains.

## Conclusion and Next Steps
This agentic, feedback-aware stack keeps receipt decisions fast, explainable, and resilient to messy inputs—exactly what we set out to deliver for Dr. Ghosh’s Advanced Machine Learning course (Fall ’25). 

Thank you, Dr. Ghosh, for your guidance throughout the semester. 
Next up: stronger multilingual OCR, richer anomaly features (frequency drift, geo/time coherence), and per-customer threshold auto-tuning.

---

## Links
- **Demo (Spaces):** https://huggingface.co/spaces/Rogue2003/Receipt_Agent
- **Repo:** https://github.com/RogueTex/StreamingDataforModelTraining
- **Notebook:** `NewVerPynbAgent.ipynb`

## References
- LangGraph docs: https://python.langchain.com/docs/langgraph
- LayoutLMv3: https://arxiv.org/abs/2204.08376
- EasyOCR: https://github.com/JaidedAI/EasyOCR
- XGBoost: https://arxiv.org/abs/1603.02754
