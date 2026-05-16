import gradio as gr
from PIL import Image
import torch
import json
from src.generation_engine import initialize_generator, generate_answer


class ChestXrayAI:
    def report_mode(self, image):
        if image is None:
            raise gr.Error("Please upload an image first.")
        try:
            torch.cuda.empty_cache()
            prompt = "answer en: Write a detailed medical description of the findings in this chest x-ray."
            generated_text = generate_answer(image, prompt, gen_model, gen_processor, device)  # type: ignore
            return f"PRELIMINARY RADIOLOGY REPORT\n{'-'*30}\n{generated_text.capitalize()}"
        except Exception as e:
            raise gr.Error(f"Report Error: {str(e)}")

    def qa_mode(self, query, retriever_type):
        pass

engine = ChestXrayAI()

with gr.Blocks(title="Dual-Mode Chest X-Ray AI System") as demo:
    gr.Markdown("# 🫁 Multi-Modal Chest X-Ray Intelligence System")
    gr.Markdown("### Developed for DSAI 413 - Dual-Engine Clinical Assistant Pipeline")

    with gr.Tab("Report Generation Mode"):
        img_input = gr.Image(type="pil", label="Upload X-ray")
        report_output = gr.Textbox(label="Generated Medical Report")
        btn_report = gr.Button("Generate Report")
        btn_report.click(engine.report_mode, inputs=img_input, outputs=report_output)

    with gr.Tab("Clinical QA Mode (RAG Comparison)"):
        query_input = gr.Textbox(label="Ask a clinical question (e.g., 'Is there fluid?')")
        retriever_toggle = gr.Radio(
            choices=["ColPali (Mandatory Late-Interaction)", "CLIP (Baseline Compression Mode)"],
            value="ColPali (Mandatory Late-Interaction)",
            label="Select Active Vision Retrieval Engine Engine Model"
        )
        
        with gr.Row():
            answer_output = gr.Textbox(label="AI Answer Output")
            source_img = gr.Image(label="Retrieved Context Evidence X-ray")
            
        btn_qa = gr.Button("Execute RAG Pipeline Query")
        btn_qa.click(
            engine.qa_mode, 
            inputs=[query_input, retriever_toggle], 
            outputs=[answer_output, source_img]
        )

if __name__ == "__main__":
    demo.launch(share=True)