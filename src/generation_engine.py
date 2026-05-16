import torch
from transformers import AutoProcessor, AutoModelForImageTextToText, BitsAndBytesConfig
from huggingface_hub import login

def initialize_generator(hf_token):
    login(hf_token)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    gen_model_id = "google/medgemma-4b-it"

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16
    )

    processor = AutoProcessor.from_pretrained(gen_model_id)
    model = AutoModelForImageTextToText.from_pretrained(
        gen_model_id,
        quantization_config=bnb_config, 
        device_map=device
    ).eval()
    
    return model, processor, device

def generate_answer(image, prompt_text, model, processor, device):
    clean_prompt = prompt_text.replace("answer en: ", "").strip()
    
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": clean_prompt}
            ]
        }
    ]
    
    text = processor.apply_chat_template(messages, add_generation_prompt=True)
    inputs = processor(text=text, images=image, return_tensors="pt").to(device)
    
    with torch.no_grad():
        output = model.generate(**inputs, max_new_tokens=250)

    input_len = inputs["input_ids"].shape[-1]
    result = processor.decode(output[0][input_len:], skip_special_tokens=True).strip()
    
    return result