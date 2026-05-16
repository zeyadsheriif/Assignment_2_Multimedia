import ast
import os
import torch
from tqdm import tqdm

def run_evaluation(qa_dataset, retrieval_engine):
    device = retrieval_engine.device
    client = retrieval_engine.client
    COLLECTION_NAME = "chest_xray_rag"
    
    print("\n--- Evaluating Retrieval Architectures Clinical Accuracy ---")

    filename_to_report = {}
    for item in qa_dataset:
        raw = str(item['image_path'])
        clean = ast.literal_eval(raw)[0] if raw.startswith('[') else raw
        fname = os.path.basename(clean)
        filename_to_report[fname] = item['report'].lower()

    clip_filenames = [os.path.basename(p) for p in retrieval_engine.clip_paths]
    valid_samples = []
    for item in qa_dataset:
        raw = str(item['image_path'])
        clean = ast.literal_eval(raw)[0] if raw.startswith('[') else raw
        if os.path.basename(clean) in clip_filenames:
            q = item['question'].lower()
            if "fluid" in q or "normal" in q:
                valid_samples.append(item)

    test_samples = valid_samples[:50]
    print(f"Testing validation metrics on {len(test_samples)} clinical queries...")

    colpali_correct = 0
    for item in tqdm(test_samples, desc="ColPali Metric Pass"):
        query = item['question']
        target_concept = "pleural effusion" if "fluid" in query.lower() else "normal"
        colpali_top_filename = None
        
        try:
            q_batch = retrieval_engine.retrieval_processor.process_queries([query]).to(device)
            with torch.no_grad():
                q_emb = retrieval_engine.retrieval_model(**q_batch)[0].cpu().float().numpy().tolist()
            
            try:
                search_result = client.query_points(collection_name=COLLECTION_NAME, query=q_emb, limit=1).points
            except AttributeError:
                search_result = client.search(collection_name=COLLECTION_NAME, query_vector=q_emb, limit=1)
            
            if search_result:
                colpali_top_filename = os.path.basename(search_result[0].payload['image_path'])
        except Exception as e:
            print(f"\n[ERROR] ColPali failed on verification step: {str(e)}")
            
        colpali_report = filename_to_report.get(colpali_top_filename, "")
        colpali_correct += int(target_concept in colpali_report)

    clip_correct = 0
    for item in tqdm(test_samples, desc="CLIP Metric Pass"):
        query = item['question']
        target_concept = "pleural effusion" if "fluid" in query.lower() else "normal"
        clip_top_filename = None
        
        try:
            clip_inputs = retrieval_engine.clip_processor(text=[query], return_tensors="pt", padding=True).to(device)
            dummy_img = torch.zeros((1, 3, 224, 224)).to(device)
            
            with torch.no_grad():
                outputs = retrieval_engine.clip_model(
                    input_ids=clip_inputs['input_ids'], 
                    attention_mask=clip_inputs['attention_mask'], 
                    pixel_values=dummy_img
                )
                clip_q_emb = outputs.text_embeds
                clip_q_emb = clip_q_emb / clip_q_emb.norm(p=2, dim=-1, keepdim=True)
                
            similarities = (clip_q_emb @ retrieval_engine.clip_image_embs_tensor.T).squeeze(0)
            best_idx = similarities.argmax().item()
            clip_top_filename = os.path.basename(retrieval_engine.clip_paths[best_idx])
        except Exception as e:
            print(f"\n[ERROR] CLIP failed on verification step: {str(e)}")
        
        clip_report = filename_to_report.get(clip_top_filename, "")
        clip_correct += int(target_concept in clip_report)

    print("\n" + "="*45)
    print(f" COLPALI CLINICAL ACCURACY: {(colpali_correct / len(test_samples))*100:.2f}%")
    print(f" CLIP CLINICAL ACCURACY:    {(clip_correct / len(test_samples))*100:.2f}%")
    print("="*45)