import os
import pandas as pd
import json
import kagglehub

def setup_data():
    print("Downloading dataset...")
    path = kagglehub.dataset_download("simhadrisadaram/mimic-cxr-dataset")
    
    root_files = os.listdir(path)
    csv_files = [f for f in root_files if f.endswith('.csv')]

    if not csv_files:
        raise FileNotFoundError(f"No CSV files were found in the root of {path}")

    target_csv = os.path.join(path, csv_files[0])
    df = pd.read_csv(target_csv, nrows=250)
    
    return df, path

def create_synthetic_qa(df, output_path='data/medical_qa_dataset.json'):
    qa_data = []
    for _, row in df.iterrows():
        report = str(row['text'])
        
        if "pleural effusion" in report.lower():
            q = "Is there any sign of fluid in the lungs (pleural effusion)?"
            a = "Yes, the report indicates pleural effusion."
        elif "normal" in report.lower():
            q = "Are there any significant findings in this chest X-ray?"
            a = "No, the findings are within normal limits."
        else:
            q = "What is the primary clinical finding in this image?"
            a = f"The findings suggest: {report[:100]}..."

        qa_data.append({
            "image_path": row['image'],
            "question": q,
            "answer": a,
            "report": report
        })
        
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(qa_data, f)
        
    print(f"Dataset generated with {len(qa_data)} items!")
    return qa_data