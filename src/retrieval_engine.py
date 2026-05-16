import ast
import os
import torch
from PIL import Image
from tqdm import tqdm
from colpali_engine.models import ColPali, ColPaliProcessor
from qdrant_client import QdrantClient, models
from transformers import CLIPModel, CLIPProcessor, BitsAndBytesConfig

COLLECTION_NAME = "chest_xray_rag"

class DualRetrievalEngine:
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.client = QdrantClient(":memory:")
        
        self.clip_image_embs_tensor = None
        self.clip_image_embs_list = []
        self.clip_paths = []
        
        self.bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16
        )
        
        self.load_models()

    def load_models(self):
        print("Loading ColPali in 4-bit to save VRAM...")
        colpali_name = "vidore/colpali-v1.3"
        self.retrieval_model = ColPali.from_pretrained(
            colpali_name, 
            quantization_config=self.bnb_config, 
            device_map=self.device
        ).eval()
        self.retrieval_processor = ColPaliProcessor.from_pretrained(colpali_name)

        print("Loading CLIP...")
        clip_name = "openai/clip-vit-base-patch32"
        self.clip_model = CLIPModel.from_pretrained(clip_name).to(self.device).eval()
        self.clip_processor = CLIPProcessor.from_pretrained(clip_name)

    def index_images_dual(self, qa_data, dataset_path):
        self.clip_image_embs_list = []
        self.clip_paths = []
        
        print("Scanning directory for your specific images...")
        target_filenames = set()
        for item in qa_data:
            raw = str(item['image_path'])
            clean_path = ast.literal_eval(raw)[0] if raw.startswith('[') else raw
            target_filenames.add(os.path.basename(clean_path))

        image_lookup = {}
        for root, dirs, files in os.walk(dataset_path):
            found_in_folder = set(files).intersection(target_filenames)
            for f in found_in_folder:
                image_lookup[f] = os.path.join(root, f)

        if not image_lookup:
            print("CRITICAL ERROR: No images found.")
            return

        try:
            self.client.delete_collection(COLLECTION_NAME)
        except Exception:
            pass

        self.client.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=models.VectorParams(
                size=128, 
                distance=models.Distance.COSINE,
                multivector_config=models.MultiVectorConfig(comparator=models.MultiVectorComparator.MAX_SIM)
            ),
        )

        successful_colpali = 0
        successful_clip = 0
        
        for idx, item in enumerate(tqdm(qa_data, desc="Dual Indexing (ColPali + CLIP)")):
            try:
                raw_path_str = str(item['image_path'])
                clean_path = ast.literal_eval(raw_path_str)[0] if raw_path_str.startswith('[') else raw_path_str
                filename = os.path.basename(clean_path)

                if filename not in image_lookup:
                    continue
                    
                full_image_path = image_lookup[filename]
                img = Image.open(full_image_path).convert("RGB")
                item['image_path'] = full_image_path 

                try:
                    inputs = self.retrieval_processor.process_images([img]).to(self.device)
                    with torch.no_grad():
                        embeddings = self.retrieval_model(**inputs)
                    
                    vec = embeddings[0] if not isinstance(embeddings, tuple) else embeddings[0][0]
                    
                    self.client.upsert(
                        collection_name=COLLECTION_NAME,
                        points=[models.PointStruct(id=idx, vector=vec.cpu().float().numpy().tolist(), payload=item)]
                    )
                    successful_colpali += 1
                except Exception:
                    pass

                try:
                    clip_inputs = self.clip_processor(images=img, return_tensors="pt").to(self.device)
                    dummy_text = torch.tensor([[49406, 49407]]).to(self.device)
                    
                    with torch.no_grad():
                        outputs = self.clip_model(pixel_values=clip_inputs['pixel_values'], input_ids=dummy_text)
                        c_emb_raw = outputs.image_embeds
                        c_emb = c_emb_raw / c_emb_raw.norm(p=2, dim=-1, keepdim=True) 
                    
                    self.clip_image_embs_list.append(c_emb)
                    self.clip_paths.append(full_image_path)
                    successful_clip += 1
                except Exception as e:
                    print(f"\nCLIP Error on {filename}: {str(e)}")
                    pass
                    
            except Exception:
                continue

        if self.clip_image_embs_list:
            self.clip_image_embs_tensor = torch.cat(self.clip_image_embs_list)

        print(f"\nIndexing Complete! ColPali: {successful_colpali} | CLIP: {successful_clip}")