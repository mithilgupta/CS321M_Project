import os
from dotenv import load_dotenv
load_dotenv()

from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import torch

def build_prompt(question, retrieved_chunks):
    context_parts = []
    for i, chunk in enumerate(retrieved_chunks, start=1):
        context_parts.append(
            f"[Context {i}]\n"
            f"Document Title: {chunk['title']}\n"
            f"Document ID: {chunk['doc_id']}\n"
            f"Section ID: {chunk['section_id']}\n"
            f"Text: {chunk['text']}\n"
        )
    context_text = "\n\n".join(context_parts)
    prompt = f"""You are a question-answering assistant.
Use ONLY the retrieved context below to answer the question.
If the answer is not supported by the retrieved context, say:
"I do not have enough information from the retrieved context."

Retrieved context:
{context_text}

Question:
{question}

Answer:
"""
    return prompt


class AnthropicGenerator:
    def __init__(self, model_name):
        import anthropic
        self.model_name = model_name
        self.client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])

    def generate(self, prompt):
        response = self.client.messages.create(
            model=self.model_name,
            max_tokens=256,
            messages=[{"role": "user", "content": prompt}]
        )
        return response.content[0].text.strip()


class HFLocalGenerator:
    def __init__(self, model_name):
        self.model_name = model_name
        self.pipe = pipeline(
            "text-generation",
            model=model_name,
            tokenizer=model_name,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            device_map="auto"
        )

    def generate(self, prompt):
        out = self.pipe(
            prompt,
            max_new_tokens=256,
            do_sample=False,
            temperature=0.0,
            return_full_text=False
        )
        return out[0]["generated_text"].strip()
