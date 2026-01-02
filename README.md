<div align="center">

# The Importance of Color in Facial Recognition Models

</div>


<div align="center">
    <img src="https://img.shields.io/badge/Made%20with-Python-306998?style=for-the-badge&logo=python&logoColor=white">
    <img src="https://img.shields.io/badge/Made%20with-Jupyter-F37726?style=for-the-badge&logo=Jupyter&logoColor=white">
</div>

<br/>

<div align="center">
    <img src="https://img.shields.io/badge/License-Academic%20Use-blue?style=flat">
    <img src="https://img.shields.io/badge/Focus-Fairness%20%26%20Bias-critical?style=flat">
</div>

---

## Project Background

Face recognition systems are increasingly deployed in socially sensitive contexts, yet their robustness to variations in appearance, particularly skin color, remains uneven. This project investigates how **synthetic skin color manipulation**, applied independently of facial geometry, affects the performance and fairness of face recognition models. By isolating color from structure, the study aims to identify whether models encode identity through invariant facial features or rely on chromatic and texture-based cues that may introduce bias.

---

## Models Evaluated

The following face recognition / detection models are analyzed:

- **MediaPipe Face Detection**  
  Landmark-driven, trained on the FairFace dataset.
- **MTCNN**  
  CNN-based detector with strong reliance on color-channel information.
- **dlib HOG**  
  Gradient-based detector sensitive to luminance and edge contrast.

---

## Dataset

Experiments are conducted on the **FairFace dataset**, which provides balanced annotations across:

- Age groups  
- Sex  
- Race  

Skin recoloring is applied while preserving the **facial silhouette**, without modifying internal landmarks (eyes, nose, mouth), ensuring that geometry remains constant.

---

## Methodology Overview

1. Evaluate original images (baseline performance).
2. Generate multiple skin color and shade variants per image.
3. Measure detection success per model and variant.
4. Aggregate results by age, sex, and race.
5. Analyze **failure recovery** cases where recolored images succeed after original failures (Section 5.5).

All experiments and visualizations are implemented in a single Jupyter Notebook.

---

## Key Findings

- **MediaPipe** demonstrates near color-invariance, with performance remaining close to baseline across all colors, shades, and demographic groups.  
- **MTCNN** exhibits strong dependence on red and blue channels, failing disproportionately for green-dominant colors.  
- **dlib HOG** is highly sensitive to luminance and contrast, benefiting most from lighter color variants.

Failure recovery analysis shows that **color perturbation can rescue a significant number of original failures**, particularly for dlib HOG and MediaPipe, highlighting how non-geometric cues influence detection.

From a socio-technical perspective, these results emphasize that **robustness to color is essential for fairness**, as reliance on chromatic features can amplify disparities under real-world variations in skin tone, lighting, or appearance.

---

## Dependencies & Execution

This project was developed using **Python** and **Jupyter Notebook**.

### Setup Instructions

1. Create a virtual environment (recommended).
2. Install dependencies:
   ```bash
   pip install -r requirements.txt

Exact package versions may vary depending on platform-specific installation constraints (notably for **dlib** and **tensorflow**). The versions listed in **requirements.txt** correspond to tested, compatible configurations.

---

### Ethical Note

This work is intended for academic research and analysis.
The findings highlight how design and training choices in face recognition systems can unintentionally encode bias, reinforcing the importance of transparency, dataset balance, and fairness-aware evaluation.

---

## Authorship

- **Author** &#8594; [Daniela Osório](https://github.com/BOB-o-rabanete)
- **Course** &#8594; Artificial Intelligence and Society
  [[M.IA001](https://sigarra.up.pt/fcup/pt/ucurr_geral.ficha_uc_view?pv_ocorrencia_id=559505)]
- **Universitys** &#8594; Faculty of Sciences and Faculty of Engineering, University of Porto 

