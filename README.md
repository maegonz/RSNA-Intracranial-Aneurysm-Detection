# Intracranial Aneurysm Detection

Intracranial aneurysms affect nearly 3% of the global population, yet up to 50% are only diagnosed after a rupture—often leading to severe outcomes or death. Each year, they contribute to approximately 500,000 deaths worldwide, with half of the victims under the age of 50.

This project aims to develop **machine learning and deep learning models** capable of detecting and localizing aneurysms across **multi-modal brain imaging**, including:

* **CT Angiography**
* **MR Angiography**
* **T1-weighted post-contrast MRI**
* **T2-weighted MRI**

Below are example visualizations from different modalities and model outputs (replace these placeholders with your real images):

![T2 Example](examples/output.png) 

## Overview

The objective of this project is to build robust and generalizable models based on UNet architecture and DICE evaluation metric that can:

1. **Identify** the presence of intracranial aneurysms
2. **Localize** their precise regions within the brain
3. **Generalize** across multiple imaging modalities, scanners, and acquisition protocols

Ultimately, these models could serve as **decision-support tools** to aid radiologists in **early aneurysm detection**, enabling **timely intervention** and **saving lives**.

This project is part of the Kaggle **RSNA Intracranial Aneurysm Detection** challenge, hosted by:

* **Radiological Society of North America (RSNA)**
* **American Society of Neuroradiology (ASNR)**
* **Society of Neurointerventional Surgery (SNIS)**
* **European Society of Neuroradiology (ESNR)**

The dataset includes **real-world clinical variations** from multiple institutions, scanner vendors, and imaging protocols — making it a benchmark for developing models that perform well in diverse clinical scenarios.

## Structure
```bash
├── data/              # Data importation and processing files
├── model/             # Custom UNet and architecture modules
├── main.ipynb         # Jupyter notebook used for exploration
└── README.md          # This document
```

## 📊 Evaluation Metrics

* **Dice Coefficient (DICE)**


---

## 🤝 Acknowledgements

Special thanks to:

* **RSNA**, **ASNR**, **SNIS**, **ESNR**
* All participating clinicians and institutions
* Open-source contributors advancing medical imaging AI


## License

This project is released under the [MIT License](LICENSE).
