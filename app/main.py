from fastapi import FastAPI
from transformers import pipeline
from pydantic import BaseModel

# 1. Tell FastAPI what the incoming JSON looks like
class GenerateRequest(BaseModel):
    prompt: str

app = FastAPI()

# 2. Load the AI model when the app starts 
# (Renamed to 'generator' to avoid overwriting the import)
generator = pipeline(task="text-generation", model="facebook/opt-125m")

# 3. Define the endpoint
@app.post("/generate")
def generate_text(request: GenerateRequest):
    # Pass the incoming text to the AI model
    ai_output = generator(request.prompt)
    
    # ai_output looks like: [{'generated_text': 'Once upon a time...'}]
    # We need to extract just the string from that list/dictionary combo
    final_text = ai_output[0]['generated_text']
    
    # Return it in the exact JSON format requested
    return {"response": final_text}

@app.get("/health")
def health():
    return {"status": "ok"}