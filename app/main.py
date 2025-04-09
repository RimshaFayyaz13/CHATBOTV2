from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
from RagBot import chatbot_function
from fastapi.middleware.cors import CORSMiddleware
import os

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://verified-arriving-dingo.ngrok-free.app"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/")
async def serve_ui():
    return FileResponse(os.path.join("static", "index.html"))

class UserInput(BaseModel):
    message: str

# @app.post("/chat")
# async def chat(input_data: UserInput):
#     response = chatbot_function(input_data.message)
#     print("DEBUG chatbot_function response:", response)
    
#     # response is just a string, return it directly
#     return {"reply": response}


@app.post("/chat")

async def chat(input_data: UserInput):
    response = chatbot_function(input_data.message)
    
    result_text = response["response"]["result"]
    # response is just a string, return it directly
    return {"reply": result_text}

