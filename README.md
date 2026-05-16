# 🫁 Multi-Modal Chest X-Ray Intelligence System

Developed as part of the **DSAI 413 Course (Assignment 2)**. This project implements an independent dual-mode Vision-Language medical pipeline featuring automated **Radiology Report Generation** and a **Retrieval-Augmented Generation (RAG) Clinical QA System**.

---

## 🛠️ System Architecture

The pipeline decouples image understanding and linguistic reasoning workflows into two isolated execution modes:

1. **Report Generation Mode (VLM Direct Strategy):** Processes raw diagnostic images directly through a 4-bit NF4-quantized `MedGemma-4b-it` backbone using an structured clinical prompt matrix to produce clean preliminary descriptions.
2. **Clinical QA Mode (Multi-Modal RAG):** Builds a cross-modal search index via an in-memory `Qdrant` engine. Compares late-interaction patch document embeddings (`ColPali`) against a global textual alignment baseline (`CLIP`) before contextual generation.

---

## 📂 Repository Structure

```text
multi-modal-cxr-system/
│
├── data/                      # Local storage partition for JSON records
│   └── medical_qa_dataset.json
│
├── src/                       # Pipeline core packages
│   ├── __init__.py            # Package tree allocation
│   ├── data_processor.py      # MIMIC-CXR loaders and synthetic generation
│   ├── retrieval_engine.py    # Quantized vector matrix indices and engines
│   ├── generation_engine.py   # MedGemma inference execution layers
│   └── evaluation.py          # Empirical accuracy validation module
│
├── app.py                     # Primary Web Demo app wrapper (Gradio framework)
├── requirements.txt           # Unified package constraints lock
└── README.md                  # Project execution documentation

