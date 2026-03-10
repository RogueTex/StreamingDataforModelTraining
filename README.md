# Streaming Data for Model Training

A data streaming pipeline for efficient model training on large datasets.

## Overview

This project implements streaming data ingestion for ML model training, allowing models to train on data larger than available RAM by streaming batches directly from storage.

## Architecture

```
Data Source → Stream Processor → Batch Generator → Model Trainer
```

## Getting Started

### Prerequisites
- Python 3.10+
- - Jupyter Notebook
  - - Required packages (see notebooks for imports)
   
    - ### Running
   
    - 1. Open the notebook in Jupyter or Google Colab
      2. 2. Configure your data source path
         3. 3. Run cells sequentially
           
            4. ## Features
           
            5. - Memory-efficient data streaming
               - - Configurable batch sizes
                 - - Support for multiple data formats
                  
                   - ## License
                  
                   - MIT
