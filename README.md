Adobe India Hackathon Challenge
# PDF Feature Extraction and XGBoost Model Pipeline

## Overview
This project provides a pipeline for extracting text features from PDF documents, training an XGBoost machine learning model on the extracted features, and predicting labels for new PDF data. The pipeline includes scripts for processing PDFs, feature extraction, model training, and label prediction.

## Features
- Extract detailed text features from PDFs using pdfminer.six
- Process multiple PDFs in a folder and combine extracted features into a CSV file
- Train an XGBoost model on labeled data
- Predict labels for new PDF data using the trained model
- Save model artifacts and label encoders for reuse

## Installation
1. Clone the repository or download the project files.
2. Ensure you have Python 3.7+ installed.
3. Install the required Python packages using pip:

bash
pip install -r requirements.txt


## Usage

### 1. Extract Features from PDFs
Place your PDF files in a folder (default is InputPDF folder).  
Run the feature extraction script:

bash
python app/extract_features.py


This will process all PDFs in the folder and output a CSV file (combined_unlabeled.csv) with extracted features.

### 2. Train the XGBoost Model
Use the training script to train the model on labeled data:

bash
python app/train_xgboost_model.py


This script will save the trained model and label encoder in the model/ directory.

### 3. Predict Labels for New Data
Use the prediction script to predict labels on new data:

bash
python app/predict_labels.py


The predicted labels will be saved in the app/OUTPUT/ directory.

## Project Structure


.
├── app/
│   ├── extract_features.py       # Extract features from PDFs
│   ├── train_xgboost_model.py    # Train the XGBoost model
│   ├── predict_labels.py         # Predict labels using the trained model
│   ├── oversampling.PY           # (Optional) Oversampling script for data balancing
│   ├── structure_jsonoutput.py   # (Optional) Script for structuring JSON output
│   ├── INPUT/                    # Folder for input CSV and PDF files
│   ├── OUTPUT/                   # Folder for output JSON and CSV files
├── model/
│   ├── label_encoder.pkl         # Saved label encoder
│   ├── xgb_model.pkl             # Saved XGBoost model
├── process_pdfs.py               # (Optional) Additional PDF processing script
├── requirements.txt              # Python dependencies


## Notes
- Modify the PDF_FOLDER and OUTPUT_CSV variables in app/extract_features.py as needed to point to your PDF input folder and desired output CSV file.
- Ensure the model/ directory exists before training or predicting.
- The scripts assume a consistent data format for training and prediction.

## License
This project is provided as-is without any warranty.
