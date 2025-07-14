# 🎯 Customer Support Ticket Classification System

This project classifies customer support chats based on **Intent**, **Tone**, and **Resolution** using a **Random Forest classifier** and **TF-IDF vectorization**.

---

## 📁 Code Directory Structure

```
CA_Code/
│
├── main.py                     # Entry point of the application
├── Config.py                   # Configuration file
├── preprocess.py               # Data preprocessing and cleaning
├── embeddings.py               # TF-IDF embedding logic
├── data_loader.py              # Data loading and handling
│
├── modelling/
│   ├── modelling.py            # Model training and evaluation
│   └── data_model.py           # Data schema and utilities
│
├── model/
│   ├── base.py                 # Base model class
│   └── randomforest.py         # Random Forest implementation
│
└── data/
    ├── AppGallery.csv          # Sample dataset 1
    └── Purchasing.csv          # Sample dataset 2
```

---

## ▶️ How to Run

1. **Install dependencies:**

   ```bash
   pip install -r requirements.txt
   ```

2. **Run the main script:**

   ```bash
   python3 main.py
   ```

---

## ✨ Features

* ✅ Data cleaning and deduplication
* ✅ TF-IDF vectorization
* ✅ Chained multi-label classification:
  → **Intent** → **Tone** → **Resolution**
* ✅ Output CSV files grouped by mailbox

---

## ⚙️ Configuration

Modify `Config.py` to set your custom column names:

```python
INTERACTION_CONTENT = "Interaction content"
TICKET_SUMMARY      = "Ticket Summary"
CLASS_COL           = "Intent"
GROUPED             = "Mailbox"
```

---

## 🌐 Optional: Translation

Supports multilingual data using Facebook’s **M2M100** model.
To translate non-English text to English, use:

```python
translate_to_en()
```

Available in `preprocess.py`.
