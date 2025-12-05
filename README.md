# Receipt Processing Pipeline with AI Agents

An intelligent document processing pipeline using Vision Transformers (ViT), LayoutLMv3, and LangGraph for automated receipt classification, field extraction, and anomaly detection.

## 🚀 Quick Start

### Open in Google Colab
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/RogueTex/StreamingDataforModelTraining/blob/main/NewVerPynbAgent.ipynb)

### Features
- **Document Classification**: ViT-based classifier to identify receipts vs other documents
- **Field Extraction**: LayoutLMv3 for extracting vendor, date, total, and line items
- **Anomaly Detection**: Isolation Forest to detect suspicious receipts
- **AI Agent Workflow**: LangGraph-powered intelligent processing pipeline
- **Gradio Demo**: Interactive web interface for testing

## 📁 Project Structure

```
StreamingDataforModelTraining/
├── NewVerPynbAgent.ipynb    # Main notebook with full pipeline
├── models/                   # Trained model files (.pt)
│   ├── rvl_classifier.pt    # ViT Document Classifier (~21 MB)
│   ├── layoutlm_extractor.pt # LayoutLM Field Extractor (~478 MB)
│   └── anomaly_detector.pt  # Anomaly Detection Model (~1.5 MB)
├── data/                     # Dataset cache and synthetic data
└── README.md
```

## 🔧 Models

After running the notebook, models will be saved to the `models/` directory:

| Model | Size | Description |
|-------|------|-------------|
| `rvl_classifier.pt` | ~21 MB | ViT-based document classifier |
| `layoutlm_extractor.pt` | ~478 MB | LayoutLMv3 field extraction |
| `anomaly_detector.pt` | ~1.5 MB | Isolation Forest anomaly detector |

## 📊 Datasets Used

- **RVL-CDIP**: Document classification (optional, uses synthetic if unavailable)
- **CORD**: Receipt understanding dataset
- **FUNSD**: Form understanding dataset
- **SROIE**: Receipt OCR dataset (optional)

## 🛠️ Requirements

- Python 3.8+
- PyTorch
- Transformers (HuggingFace)
- EasyOCR
- LangGraph
- Gradio

## 📝 Usage

1. Open the notebook in Google Colab (click badge above)
2. Run all cells to train models
3. Use the Gradio interface to test with your own receipts
4. Download trained models to `models/` folder

## 🔗 Links

- [LangGraph Documentation](https://langchain-ai.github.io/langgraph/)
- [LayoutLMv3 Paper](https://arxiv.org/abs/2204.08387)
- [Vision Transformer](https://arxiv.org/abs/2010.11929)

## 📄 License

MIT License
