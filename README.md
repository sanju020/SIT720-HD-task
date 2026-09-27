\# SIT720 – DASMcC Cardiovascular Disease Classification Reproduction



\## Author



Sanjay Sathyakumar



\## Project Overview



This repository contains my SIT720 11.1 HD Machine Learning Research project.



The project reproduces and critically evaluates the following published study:



N. Sinha et al., "DASMcC: Data Augmented SMOTE Multi-Class Classifier for Prediction of Cardiovascular Diseases Using Time Series Features," IEEE Access, 2023.



The study uses the PTB-XL ECG dataset to classify five diagnostic groups:



\- NORM – Normal ECG

\- MI – Myocardial Infarction

\- STTC – ST/T Change

\- CD – Conduction Disturbance

\- HYP – Hypertrophy



The reproduction includes ECG preprocessing, feature extraction, class balancing using SMOTE, machine-learning classification and comparison with the published results.



A second experimental solution was also developed using robust statistical ECG features to reduce feature dimensionality and computational cost.



\---



\## Dataset



The project uses PTB-XL version 1.0.3.



Included data files:



\- `ptbxl\_database.csv`

\- `scp\_statements.csv`

\- `ptbxl\_single\_label.csv`



The complete ECG waveform dataset is not stored in this GitHub repository because of its size.



The raw PTB-XL waveform files can be downloaded from PhysioNet:



https://physionet.org/content/ptb-xl/1.0.3/



The project uses the 100 Hz ECG records contained in the `records100` directory.



After downloading, the expected directory structure is:



```text

ptb-xl/

│

├── records100/

├── ptbxl\_database.csv

├── scp\_statements.csv

└── Python scripts

