# Pneumonia Detection on Chest X-Ray Images Using Deep Learning

**Project Reference:** Ref X-Ray Exp  
**Authors:** Eda AYDIN, Nesibe Betül DÖNER, Okan TOPAL, Berkay DİRİL  
**Collaborating Partners:** İstanbul Büyükşehir Belediyesi, Zemin İstanbul, Kodluyoruz, United States Mission Turkey, University of Virginia Biocomplexity Institute.

---

## 📌 6 Project Phases (PDF Table of Contents)
1. **01 Business Understanding:** Pneumonia etiology, COVID-19 pulmonary impact, and radiograph pattern distinction (clear vs lobar consolidation vs diffuse interstitial).
2. **02 Data Understanding:** Kaggle pediatric dataset analysis (5,863 scans, Guangzhou Women and Children's Medical Center), class balance ($74.3\%$ Pneumonic vs $25.7\%$ Normal), and $150 \times 150$ grayscale normalization.
3. **03 Data Augmentation:** Real-time transformation lab testing rotations ($30^\circ$), zoom ($20\%$), shifts ($10\%$), and horizontal flips to prevent overfitting.
4. **04 CNN Modelling & Inference:** Custom 5-stage Keras CNN with BatchNormalization and `ReduceLROnPlateau` scheduling; live interactive radiograph classifier.
5. **05 Model Evaluation:** Interactive accuracy/loss curves over 15 epochs, confusion matrix ($4047 / 226 / 52 / 1531$), and precision/recall/F1 diagnostic reports.
6. **06 Proof & Error Analysis:** Analysis of correctly classified vs misclassified cases, clinical false-negative inspection, and references.

---

## 🚀 Quickstart & Execution

### 1. Environment Setup
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install --upgrade pip
pip install -r requirements.txt