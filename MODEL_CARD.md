# Model Card — DL-005 Pneumonia X-Ray Classification Demo

## Model identity
- **Name:** Pneumonia X-Ray Classification Demo
- **Project ID:** DL-005
- **Model family:** MobileNetV2 transfer learning with a sigmoid binary-classification head
- **Input:** 224 × 224 RGB image derived from a chest X-ray
- **Output classes:** NORMAL-like (0), PNEUMONIA-like (1)
- **Status:** Educational prototype; not clinically validated

## Intended use
Teaching image preprocessing, transfer learning, class-wise evaluation, confusion matrices, and error inspection in a controlled classroom setting.

## Out-of-scope use
Clinical diagnosis, treatment, triage, screening, patient management, or ruling out pneumonia. Do not use predictions to make healthcare decisions.

## Data
Document the exact dataset version, download URL, license/terms, citation, class counts, split method, and any patient-ID checks in the project report. The commonly used public dataset contains pediatric X-rays; it does not establish performance for adults, other hospitals, scanners, or populations.

## Evaluation
Fill in after training. Do not invent or pre-fill results.
- Test sample count:
- NORMAL count / PNEUMONIA count:
- Sensitivity:
- Specificity:
- Precision / recall / F1:
- ROC-AUC:
- Confusion matrix (TN, FP, FN, TP):
- Number of false negatives inspected:
- Notes on false-negative patterns:

## Known limitations and risks
- Dataset may be imbalanced.
- Models may learn shortcuts or source artifacts rather than robust disease features.
- Differences in age, equipment, image quality, clinical setting, and patient population may change performance.
- Labels may contain errors.
- A model score is not a calibrated probability of disease.
- Test metrics from one dataset do not prove clinical safety or generalization.

## Human oversight
A qualified healthcare professional must evaluate medical concerns. This software must not be used as a substitute for professional judgment.

## Reporting checklist
- [ ] Dataset license and citation reviewed
- [ ] Patient-level split verified where IDs are available
- [ ] Class balance reported for each split
- [ ] Sensitivity and specificity reported
- [ ] False negatives inspected
- [ ] Test set kept separate from model tuning
- [ ] Non-clinical warning shown in the interface and report
