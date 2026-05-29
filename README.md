# NLP-to-API Command Translator

A high-fidelity system that translates natural language commands (e.g., "Email Priya about the report") into structured JSON API payloads. This project features a fine-tuned transformer model and a custom multi-turn slot-filling interface.

##  Features
- **Transformer Core:** Fine-tuned `FLAN-T5-base` model trained on a custom synthetic dataset of 5,000+ API command pairs.
- **Robust Parsing:** Uses a linearized intermediate format to ensure the model adheres to strict API schemas.
- **Entity Alignment:** Custom fuzzy-matching logic to ensure names and locations are correctly grounded from the original input.
- **Slot Filling:** Interactive Streamlit UI that identifies missing parameters and asks the user to fill them before execution.
- **Multi-Intent Support:** Can handle compound commands like "Set an alarm for 7 AM and email the professor."

## Tech Stack
- **Language:** Python 3.11
- **ML Framework:** PyTorch, HuggingFace Transformers
- **Frontend:** Streamlit
- **Model:** Google FLAN-T5-base

##  Project Structure
- `app.py`: The Streamlit chat interface and slot-filling logic.
- `train.py`: Training script optimized for local GPUs (RTX 4060) with Gradient Checkpointing.
- `inference.py`: Local inference engine with multiple decoding strategies (Beam Search, Greedy).
- `generate_dataset.py`: Synthetic data engine using regex and template-based augmentation.

##  Getting Started
1. **Install dependencies:** `pip install -r requirements.txt`
2. **Train the model:** Run `python train.py` (requires CUDA).
3. **Launch the App:** `streamlit run app.py`
