# app.py
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import uvicorn
import os

app = FastAPI(title="Production Chatbot API")

class ChatInput(BaseModel):
    message: str

class ChatResponse(BaseModel):
    response: str

# Global variables for model and tokenizer
model = None
tokenizer = None

def load_model():
    global model, tokenizer
    
    # Using Mistral 7B - good balance of performance and resource usage
    model_name = "mistralai/Mistral-7B-Instruct-v0.1"
    
    # Load in 4-bit quantization to reduce memory usage while maintaining quality
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        load_in_4bit=True
    )

@app.on_event("startup")
async def startup_event():
    load_model()

@app.post("/chat", response_model=ChatResponse)
async def chat(chat_input: ChatInput):
    try:
        # Prepare the prompt
        prompt = f"<s>[INST] {chat_input.message} [/INST]"
        
        # Generate response
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        outputs = model.generate(
            inputs["input_ids"],
            max_length=512,
            temperature=0.7,
            do_sample=True,
            top_p=0.9,
            num_return_sequences=1,
            pad_token_id=tokenizer.eos_token_id,
        )
        
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        # Clean up the response
        response = response.split("[/INST]")[-1].strip()
        
        return ChatResponse(response=response)
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=8080)
