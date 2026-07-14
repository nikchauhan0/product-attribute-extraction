# 🛍️ Product Attribute Extraction using NLP

An end-to-end **Natural Language Processing (NLP)** project that automatically extracts structured product attributes from unstructured fashion product descriptions using **TF-IDF**, **Logistic Regression**, and **Flask REST API**.

The system converts plain text product descriptions into structured JSON attributes such as category, color, fabric, neckline, sleeve, length, silhouette, and embellishment.

---

## 📌 Features

- Extracts 8 structured product attributes
- Multi-model machine learning pipeline
- TF-IDF based text vectorization
- Logistic Regression classifiers
- Flask REST API
- JSON response
- Saved ML models using Joblib
- Easy to extend with additional attributes

---

## 📊 Example

### Input

```text
Black silk maxi dress with round neck and sleeveless design
```

### Output

```json
{
    "category": "Dress",
    "color": "Black",
    "fabric": "Silk",
    "neckline": "Round Neck",
    "sleeve": "Sleeveless",
    "length": "Maxi",
    "silhouette": "Unknown",
    "embellishment": "Solid"
}
```

---

# 🏗️ Project Architecture

```
                   Product Description
                            │
                            ▼
                    Data Preprocessing
                            │
                            ▼
                    Label Normalization
                            │
                            ▼
                     TF-IDF Vectorizer
                            │
                            ▼
      ┌─────────────────────────────────────────┐
      │      Independent ML Models              │
      │                                         │
      │  Category Model                         │
      │  Color Model                            │
      │  Fabric Model                           │
      │  Neckline Model                         │
      │  Sleeve Model                           │
      │  Length Model                           │
      │  Silhouette Model                       │
      │  Embellishment Model                    │
      └─────────────────────────────────────────┘
                            │
                            ▼
                    Structured JSON Output
```

---

# 📂 Project Structure

```
product_attribute_extractor/

├── app.py
├── train.py
├── requirements.txt
├── README.md

├── data/
│   └── products.csv

└── models/
    ├── vectorizer.pkl
    ├── category_model.pkl
    ├── color_model.pkl
    ├── fabric_model.pkl
    ├── neckline_model.pkl
    ├── sleeve_model.pkl
    ├── length_model.pkl
    ├── silhouette_model.pkl
    └── embellishment_model.pkl
```

---

# 🧠 Machine Learning Pipeline

### 1. Dataset Preparation

- Manually labeled dataset
- 50+ fashion product descriptions
- 8 target attributes

### 2. Data Preprocessing

- Missing value handling
- Label normalization
- Text cleaning
- Stop-word removal
- Lowercase conversion

### 3. Feature Engineering

TF-IDF Vectorizer

- Unigrams & Bigrams
- English stop-word removal
- Sparse feature representation

### 4. Model Training

A separate **Logistic Regression** classifier is trained for each attribute.

| Model | Target |
|--------|---------|
| Category Model | Category |
| Color Model | Color |
| Fabric Model | Fabric |
| Neckline Model | Neckline |
| Sleeve Model | Sleeve |
| Length Model | Length |
| Silhouette Model | Silhouette |
| Embellishment Model | Embellishment |

### 5. Model Serialization

All trained models are saved using **Joblib**.

```
models/
├── vectorizer.pkl
├── category_model.pkl
├── color_model.pkl
├── ...
```

---

# 🚀 API

## Endpoint

```
POST /extract
```

### Request

```json
{
    "description": "Black silk maxi dress with round neck and sleeveless design"
}
```

### Response

```json
{
    "category": "Dress",
    "color": "Black",
    "fabric": "Silk",
    "neckline": "Round Neck",
    "sleeve": "Sleeveless",
    "length": "Maxi",
    "silhouette": "Unknown",
    "embellishment": "Solid"
}
```

---

# ⚙️ Installation

Clone the repository

```bash
git clone https://github.com/yourusername/product-attribute-extraction.git
```

Move into the project

```bash
cd product-attribute-extraction
```

Install dependencies

```bash
pip install -r requirements.txt
```

---

# ▶️ Train the Models

```bash
python train.py
```

This will:

- Load the dataset
- Normalize labels
- Train all classifiers
- Evaluate the models
- Save `.pkl` files

---

# ▶️ Run the Flask API

```bash
python app.py
```

Server:

```
http://127.0.0.1:5000
```

---

# 📈 Evaluation

The project evaluates every classifier using:

- Attribute-level Accuracy
- Weighted F1 Score

this is the table that describe the precision, recall, acuracy and f1 score of this model:

Training model for: category, color, fabric, neckline, sleeve, length, silhouette, embellishment

========================================

Training model for: category
category Accuracy: 0.55
category F1 Score: 0.50  

Training model for: color
color Accuracy: 0.64
color F1 Score: 0.69      

Training model for: fabric
fabric Accuracy: 0.45
fabric F1 Score: 0.40       

Training model for: neckline
neckline Accuracy: 0.45
neckline F1 Score: 0.48

Training model for: sleeve
sleeve F1 Score: 0.75

Training model for: length
length Accuracy: 0.91
length F1 Score: 0.87

Training model for: silhouette
silhouette Accuracy: 1.00
silhouette F1 Score: 1.00

Training model for: embellishment
embellishment Accuracy: 0.91
embellishment F1 Score: 0.87

---

# 🛠️ Common failure cases

- The model struggles when product descriptions use terms that were not present in the training dataset.
- Rare categories with only a few training examples are more likely to be misclassified.
- If a description omits an attribute (for example, fabric or sleeve), the model may predict the most common value       instead of Unknown.
- Similar terms such as "Kurti" and "Kurta" or "Navy Blue" and "Blue" can cause confusion without label normalization.
- The small dataset size (50 products) limits the model's ability to generalize to unseen descriptions.

---
---

# 🛠️ Tech Stack

- Python
- Pandas
- NumPy
- Scikit-learn
- Flask
- Joblib

---

# 📌 Challenges

- Small dataset size
- Class imbalance
- Similar attribute names
- Missing product attributes
- Limited vocabulary coverage

---

# 🚀 Future Improvements

- Expand dataset size
- Fine-tune BERT/DistilBERT
- Multi-language support
- Confidence scores
- Docker deployment
- Batch prediction API

---

# 👨‍💻 Author

**Nikhil Chauhan**

Python Developer | Machine Learning Enthusiast
