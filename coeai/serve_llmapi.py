


# from fastapi import FastAPI, Header, HTTPException, Request
# from fastapi.responses import StreamingResponse
# from pydantic import BaseModel
# from typing import Optional, Dict, Any, List, Union
# import requests
# import logging
# import json

# app = FastAPI(title="Ollama Proxy API", version="1.1")

# # ===================== 🔐 CONFIG =====================
# logging.basicConfig(
#     filename="userapi.log",
#     level=logging.INFO,
#     format="%(asctime)s | %(levelname)s | %(message)s",
# )

# API_KEYS = {
#     "coeai-Ldiz6OdBwMJiSAnTwuX7dUKrV0aM9cEa": "user1",
#     "coeai-7OgsyrKRr7C-Zx-sNMlL_vNWLZ6bIjy5": "user2",
#     "mysecretkeycoeaiqwerty": "admin"
# }

# AVAILABLE_MODELS = [
#     "gpt-oss:latest",
#     "gpt-oss:120b",
#     "llama4:16x17b",
#     "llama4:latest",
#     "tinyllama:latest",
#     "deepseek-r1:70b"
# ]

# OLLAMA_HOST = "http://127.0.0.1:11434"

# # ===================== 📦 DATA MODELS =====================
# class Message(BaseModel):
#     role: str
#     content: Union[str, List[Dict[str, Any]]]

# class GenerateRequest(BaseModel):
#     model: str
#     prompt: Optional[str] = None
#     messages: Optional[List[Message]] = None
#     max_tokens: Optional[int] = 512
#     temperature: Optional[float] = 0.7
#     top_p: Optional[float] = 1.0
#     stream: bool = False
#     extra: Optional[Dict[str, Any]] = None

# # ===================== 🔧 HELPERS =====================
# def authenticate(x_api_key: str, client_ip: str) -> str:
#     user = API_KEYS.get(x_api_key)
#     if not user:
#         logging.warning(f"❌ Unauthorized attempt with API key '{x_api_key}' from {client_ip}")
#         raise HTTPException(status_code=401, detail="Unauthorized")
#     return user

# def log_request(user: str, api_key: str, client_ip: str, endpoint: str, payload: dict, backend_url: str):
#     log_msg = (
#         f"\n--- Incoming Request ---\n"
#         f"User: {user}\n"
#         f"API Key: {api_key}\n"
#         f"IP Address: {client_ip}\n"
#         f"Endpoint: {endpoint}\n"
#         f"Backend URL: {backend_url}\n"
#         f"Payload Sent:\n{json.dumps(payload, indent=2)}\n"
#         f"------------------------"
#     )
#     logging.info(log_msg)

# # ===================== 🚀 ENDPOINTS =====================
# @app.get("/models")
# async def list_models(request: Request, x_api_key: str = Header(..., alias="X-API-Key")):
#     client_ip = request.client.host
#     user = authenticate(x_api_key, client_ip)
#     return {"user": user, "models": AVAILABLE_MODELS}

# @app.post("/generate")
# async def generate_text(
#     request_data: GenerateRequest,
#     request: Request,
#     x_api_key: str = Header(..., alias="X-API-Key")
# ):
#     client_ip = request.client.host
#     user = authenticate(x_api_key, client_ip)

#     if request_data.model not in AVAILABLE_MODELS:
#         raise HTTPException(status_code=400, detail=f"Model '{request_data.model}' not supported")

#     backend_url = f"{OLLAMA_HOST}/api/chat"

#     # Build Ollama payload
#     if request_data.messages:
#         ollama_payload = {
#             "model": request_data.model,
#             "messages": [m.dict() for m in request_data.messages],
#             "stream": request_data.stream,
#             "options": {
#                 "temperature": request_data.temperature,
#                 "top_p": request_data.top_p,
#                 "num_predict": request_data.max_tokens
#             }
#         }
#     else:
#         ollama_payload = {
#             "model": request_data.model,
#             "messages": [{"role": "user", "content": request_data.prompt}],
#             "stream": request_data.stream,
#             "options": {
#                 "temperature": request_data.temperature,
#                 "top_p": request_data.top_p,
#                 "num_predict": request_data.max_tokens
#             }
#         }

#     log_request(user, x_api_key, client_ip, "/generate", ollama_payload, backend_url)

#     try:
#         with requests.post(backend_url, json=ollama_payload, stream=True, timeout=600) as r:
#             r.raise_for_status()

#             if request_data.stream:
#                 # Stream Ollama chunks directly
#                 def event_stream():
#                     for line in r.iter_lines():
#                         if line:
#                             yield line + b"\n"
#                 return StreamingResponse(event_stream(), media_type="application/json")

#             else:
#                 # Non-stream mode → single JSON response
#                 resp_json = r.json()
#                 logging.info(f"✅ Response for {user} @ {client_ip}: {str(resp_json)[:300]}...")
#                 return {
#                     "user": user,
#                     "model": request_data.model,
#                     "response": resp_json.get("message", {}).get("content") or resp_json
#                 }

#     except requests.exceptions.RequestException as e:
#         logging.error(f"🔥 Network/Ollama error for {user} @ {client_ip}: {str(e)}")
#         raise HTTPException(status_code=502, detail=f"Ollama backend unavailable: {str(e)}")
#     except Exception as e:
#         logging.error(f"🔥 Unexpected error for {user} @ {client_ip}: {str(e)}")
#         raise HTTPException(status_code=500, detail=f"Internal error: {str(e)}")



# from fastapi import FastAPI, Header, HTTPException, Request, UploadFile, File, Form
# from pydantic import BaseModel
# from typing import Optional, List, Union, Dict, Any
# import requests
# import logging
# import json
# import base64
# import datetime

# app = FastAPI(title="Ollama Proxy API", version="1.3")

# # ----------------- CONFIG -----------------
# logging.basicConfig(
#     filename="userapi.log",
#     level=logging.INFO,
#     format="%(asctime)s | %(levelname)s | %(message)s",
# )

# API_KEYS = {
#     "coeai-Ldiz6OdBwMJiSAnTwuX7dUKrV0aM9cEa": "user1",
#     "mysecretkeycoeaiqwerty": "admin"
# }

# AVAILABLE_MODELS = [
#     "gpt-oss:latest",
#     "gpt-oss:120b",
#     "llama4:16x17b",
#     "llama4:latest",
#     "tinyllama:latest",
#     "deepseek-r1:70b"  # hypothetical multimodal
# ]

# OLLAMA_CHAT_URL = "http://127.0.0.1:11434/v1/chat/completions"

# # ----------------- DATA MODELS -----------------
# class Message(BaseModel):
#     role: str
#     content: Union[str, List[Dict[str, Any]]]

# # ----------------- ROUTES -----------------
# @app.get("/models")
# async def list_models(request: Request, x_api_key: str = Header(..., alias="X-API-Key")):
#     client_ip = request.client.host
#     user = API_KEYS.get(x_api_key)
#     if not user:
#         raise HTTPException(status_code=401, detail="Unauthorized")
#     return {"user": user, "models": AVAILABLE_MODELS}


# @app.post("/generate")
# async def generate(
#     request: Request,
#     x_api_key: str = Header(..., alias="X-API-Key"),
#     model: str = Form(...),
#     prompt: Optional[str] = Form(None),
#     max_tokens: Optional[int] = Form(512),
#     temperature: Optional[float] = Form(0.7),
#     top_p: Optional[float] = Form(1.0),
#     stream: Optional[bool] = Form(False),
#     messages: Optional[str] = Form(None),  # JSON string
#     files: Optional[List[UploadFile]] = File(None)
# ):
#     client_ip = request.client.host
#     user = API_KEYS.get(x_api_key)
#     if not user:
#         raise HTTPException(status_code=401, detail="Unauthorized")

#     if model not in AVAILABLE_MODELS:
#         raise HTTPException(status_code=400, detail=f"Model '{model}' not supported")

#     # Parse messages JSON
#     parsed_messages = None
#     if messages:
#         try:
#             parsed_messages = json.loads(messages)
#         except Exception:
#             raise HTTPException(status_code=400, detail="Invalid messages JSON format")

#     # If no messages provided, fall back to simple prompt
#     if not parsed_messages:
#         parsed_messages = [{"role": "user", "content": [{"type": "text", "text": prompt}]}]

#     # Inject uploaded files as image content blocks
#     if files:
#         for file in files:
#             content = await file.read()
#             b64 = base64.b64encode(content).decode("utf-8")
#             # Append an image block to the last user message
#             if parsed_messages:
#                 if isinstance(parsed_messages[-1]["content"], list):
#                     parsed_messages[-1]["content"].append({
#                         "type": "image",
#                         "image_url": {"url": f"data:image/png;base64,{b64}"}
#                     })
#                 else:
#                     parsed_messages[-1]["content"] = [
#                         {"type": "text", "text": parsed_messages[-1]["content"]},
#                         {"type": "image", "image_url": {"url": f"data:image/png;base64,{b64}"}}
#                     ]

#     # Build payload
#     payload: Dict[str, Any] = {
#         "model": model,
#         "messages": parsed_messages,
#         "stream": stream,
#         "options": {
#             "temperature": temperature,
#             "top_p": top_p,
#             "num_predict": max_tokens
#         }
#     }

#     # Log request
#     logging.info(
#         f"\n--- Incoming Request ---\n"
#         f"User: {user}\n"
#         f"IP: {client_ip}\n"
#         f"Model: {model}\n"
#         f"Payload: {json.dumps(payload, indent=2)[:1000]}...\n"
#         f"------------------------"
#     )

#     try:
#         res = requests.post(OLLAMA_CHAT_URL, json=payload, timeout=600)
#         res.raise_for_status()
#         return res.json()
#     except requests.exceptions.RequestException as e:
#         logging.error(f"🔥 Ollama network error for {user} @ {client_ip}: {str(e)}")
#         raise HTTPException(status_code=502, detail=f"Ollama backend unavailable: {str(e)}")
#     except Exception as e:
#         logging.error(f"🔥 Unexpected error for {user} @ {client_ip}: {str(e)}")
#         raise HTTPException(status_code=500, detail=f"Internal error: {str(e)}")






# from fastapi import FastAPI, Header, HTTPException, Request, UploadFile, File, Form
# from fastapi.responses import StreamingResponse
# from pydantic import BaseModel
# from typing import Optional, List, Union, Dict, Any
# import requests
# import logging
# import json
# import base64
# import datetime

# app = FastAPI(title="Ollama Proxy API", version="1.4")

# # ----------------- CONFIG -----------------
# logging.basicConfig(
#     filename="userapi.log",
#     level=logging.INFO,
#     format="%(asctime)s | %(levelname)s | %(message)s",
# )

# API_KEYS = {
#     "coeai-Ldiz6OdBwMJiSAnTwuX7dUKrV0aM9cEa": "user1",
#     "mysecretkeycoeaiqwerty": "admin"
# }

# OLLAMA_BASE_URL = "http://127.0.0.1:11434"
# OLLAMA_CHAT_URL = f"{OLLAMA_BASE_URL}/v1/chat/completions"
# OLLAMA_MODELS_URL = f"{OLLAMA_BASE_URL}/api/tags"

# # ----------------- LOAD MODELS DYNAMICALLY -----------------
# def load_models() -> List[str]:
#     try:
#         res = requests.get(OLLAMA_MODELS_URL, timeout=10)
#         res.raise_for_status()
#         models = [m["name"] for m in res.json().get("models", [])]
#         logging.info(f"✅ Loaded models dynamically: {models}")
#         return models
#     except Exception as e:
#         logging.error(f"❌ Could not fetch models from Ollama: {str(e)}")
#         return []

# AVAILABLE_MODELS = load_models()

# # ----------------- DATA MODELS -----------------
# class Message(BaseModel):
#     role: str
#     content: Union[str, List[Dict[str, Any]]]

# # ----------------- ROUTES -----------------
# @app.get("/models")
# async def list_models(request: Request, x_api_key: str = Header(..., alias="X-API-Key")):
#     client_ip = request.client.host
#     user = API_KEYS.get(x_api_key)
#     if not user:
#         raise HTTPException(status_code=401, detail="Unauthorized")
#     return {"user": user, "models": AVAILABLE_MODELS}


# @app.post("/generate")
# async def generate(
#     request: Request,
#     x_api_key: str = Header(..., alias="X-API-Key"),
#     model: str = Form(...),
#     prompt: Optional[str] = Form(None),
#     max_tokens: Optional[int] = Form(512),
#     temperature: Optional[float] = Form(0.7),
#     top_p: Optional[float] = Form(1.0),
#     stream: Optional[bool] = Form(False),
#     messages: Optional[str] = Form(None),  # JSON string
#     files: Optional[List[UploadFile]] = File(None)
# ):
#     client_ip = request.client.host
#     user = API_KEYS.get(x_api_key)
#     if not user:
#         raise HTTPException(status_code=401, detail="Unauthorized")

#     if model not in AVAILABLE_MODELS:
#         raise HTTPException(status_code=400, detail=f"Model '{model}' not supported")

#     # Parse messages JSON
#     parsed_messages = None
#     if messages:
#         try:
#             parsed_messages = json.loads(messages)
#         except Exception:
#             raise HTTPException(status_code=400, detail="Invalid messages JSON format")

#     # If no messages provided, fall back to simple prompt
#     if not parsed_messages:
#         parsed_messages = [{"role": "user", "content": [{"type": "text", "text": prompt}]}]

#     # Inject uploaded files as image content blocks
#     if files:
#         for file in files:
#             content = await file.read()
#             b64 = base64.b64encode(content).decode("utf-8")
#             if parsed_messages:
#                 if isinstance(parsed_messages[-1]["content"], list):
#                     parsed_messages[-1]["content"].append({
#                         "type": "image",
#                         "image_url": {"url": f"data:image/png;base64,{b64}"}
#                     })
#                 else:
#                     parsed_messages[-1]["content"] = [
#                         {"type": "text", "text": parsed_messages[-1]["content"]},
#                         {"type": "image", "image_url": {"url": f"data:image/png;base64,{b64}"}}
#                     ]

#     # Build payload
#     payload: Dict[str, Any] = {
#         "model": model,
#         "messages": parsed_messages,
#         "stream": stream,
#         "options": {
#             "temperature": temperature,
#             "top_p": top_p,
#             "num_predict": max_tokens
#         }
#     }

#     # Log request (truncate messages for safety)
#     safe_payload = {
#         "model": model,
#         "messages": str(parsed_messages)[:200] + "...",
#         "options": payload["options"]
#     }
#     logging.info(f"📥 Request from {user}@{client_ip}: {json.dumps(safe_payload, indent=2)}")

#     try:
#         if stream:
#             def event_stream():
#                 with requests.post(OLLAMA_CHAT_URL, json=payload, stream=True, timeout=(10, 600)) as r:
#                     r.raise_for_status()
#                     for line in r.iter_lines():
#                         if line:
#                             yield line.decode("utf-8") + "\n"
#             return StreamingResponse(event_stream(), media_type="text/event-stream")

#         else:
#             res = requests.post(OLLAMA_CHAT_URL, json=payload, timeout=(10, 600))
#             res.raise_for_status()
#             return res.json()

#     except requests.exceptions.RequestException as e:
#         logging.error(f"🔥 Ollama network error for {user}@{client_ip}: {str(e)}")
#         raise HTTPException(status_code=502, detail=f"Ollama backend unavailable: {str(e)}")
#     except Exception as e:
#         logging.error(f"🔥 Unexpected error for {user}@{client_ip}: {str(e)}")
#         raise HTTPException(status_code=500, detail=f"Internal error: {str(e)}")











from fastapi import FastAPI, Header, HTTPException, Request, UploadFile, File, Form
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from typing import Optional, List, Union, Dict, Any
import requests
import logging
import json
import base64
import os
from dotenv import load_dotenv

# ----------------- LOAD .env -----------------
load_dotenv()  # automatically loads variables from .env

API_KEYS = {
    os.getenv("API_KEY_USER1"): "user1",
    os.getenv("API_KEY_ADMIN"): "admin"
}

OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://127.0.0.1:11434")
OLLAMA_CHAT_URL = f"{OLLAMA_BASE_URL}/v1/chat/completions"
OLLAMA_MODELS_URL = f"{OLLAMA_BASE_URL}/api/tags"

# ----------------- CONFIG -----------------
app = FastAPI(title="Ollama Proxy API", version="1.7")

logging.basicConfig(
    filename="userapi.log",
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)

# ----------------- LOAD MODELS DYNAMICALLY -----------------
def load_models() -> List[str]:
    try:
        res = requests.get(OLLAMA_MODELS_URL, timeout=10)
        res.raise_for_status()
        models = [m["name"] for m in res.json().get("models", [])]
        logging.info(f"✅ Loaded models dynamically: {models}")
        return models
    except Exception as e:
        logging.error(f"❌ Could not fetch models from Ollama: {str(e)}")
        return []

AVAILABLE_MODELS = load_models()

# ----------------- DATA MODELS -----------------
class ImageURL(BaseModel):
    url: str

class ContentBlock(BaseModel):
    type: str
    image_url: Optional[ImageURL] = None
    text: Optional[str] = None

class Message(BaseModel):
    role: str
    content: Union[str, List[ContentBlock]]

# ----------------- ROUTES -----------------
@app.get("/models")
async def list_models(request: Request, x_api_key: str = Header(..., alias="X-API-Key")):
    client_ip = request.client.host
    user = API_KEYS.get(x_api_key)
    if not user:
        raise HTTPException(status_code=401, detail="Unauthorized")
    return {"user": user, "models": AVAILABLE_MODELS}


@app.post("/generate")
async def generate(
    request: Request,
    x_api_key: str = Header(..., alias="X-API-Key"),
    model: str = Form(...),
    inference_type: str = Form("text-to-text"),  # text-to-text or image-to-text
    prompt: Optional[str] = Form(None),
    max_tokens: Optional[int] = Form(512),
    temperature: Optional[float] = Form(0.7),
    top_p: Optional[float] = Form(1.0),
    stream: Optional[bool] = Form(False),
    messages: Optional[str] = Form(None),  # JSON string
    files: Optional[List[UploadFile]] = File(None)
):
    client_ip = request.client.host
    user = API_KEYS.get(x_api_key)
    if not user:
        raise HTTPException(status_code=401, detail="Unauthorized")

    if model not in AVAILABLE_MODELS:
        raise HTTPException(status_code=400, detail=f"Model '{model}' not supported")

    # ---------------- Validate inference_type ----------------
    if inference_type not in ["text-to-text", "image-to-text"]:
        raise HTTPException(status_code=400, detail="Invalid inference_type, must be 'text-to-text' or 'image-to-text'")

    # Restrict image-to-text to llama4:16x17b
    if inference_type == "image-to-text":
        if model != "llama4:16x17b":
            raise HTTPException(
                status_code=400,
                detail="Image-to-text inference is only supported on 'llama4:16x17b'"
            )
        if not files or len(files) == 0:
            raise HTTPException(status_code=400, detail="No image uploaded for image-to-text inference")

    # ---------------- Parse messages ----------------
    parsed_messages: List[Dict[str, Any]] = []
    if messages:
        try:
            parsed_messages = json.loads(messages)
        except Exception:
            raise HTTPException(status_code=400, detail="Invalid messages JSON format")

    if not parsed_messages:
        parsed_messages = [{
            "role": "user",
            "content": [{"type": "text", "text": prompt}]
        }]

    # ---------------- Inject image content blocks ----------------
    if inference_type == "image-to-text":
        for file in files:
            content = await file.read()
            b64 = base64.b64encode(content).decode("utf-8")
            if isinstance(parsed_messages[-1]["content"], list):
                parsed_messages[-1]["content"].append({
                    "type": "image_url",
                    "image_url": {"url": f"data:image/png;base64,{b64}"}
                })
            else:
                parsed_messages[-1]["content"] = [
                    {"type": "text", "text": parsed_messages[-1]["content"]},
                    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}}
                ]

    # ---------------- Build payload ----------------
    payload: Dict[str, Any] = {
        "model": model,
        "messages": parsed_messages,
        "stream": stream,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "top_p": top_p
    }

    # Log request (truncate messages for safety)
    safe_payload = {
        "model": model,
        "messages": str(parsed_messages)[:200] + "...",
        "temperature": temperature,
        "top_p": top_p,
        "max_tokens": max_tokens,
        "inference_type": inference_type
    }
    logging.info(f"📥 Request from {user}@{client_ip}: {json.dumps(safe_payload, indent=2)}")

    # ---------------- Call Ollama ----------------
    try:
        if stream:
            def event_stream():
                with requests.post(OLLAMA_CHAT_URL, json=payload, stream=True, timeout=(10, 600)) as r:
                    r.raise_for_status()
                    for line in r.iter_lines():
                        if line:
                            yield line.decode("utf-8") + "\n"
            return StreamingResponse(event_stream(), media_type="text/event-stream")
        else:
            res = requests.post(OLLAMA_CHAT_URL, json=payload, timeout=(10, 600))
            res.raise_for_status()
            return res.json()

    except requests.exceptions.RequestException as e:
        logging.error(f"🔥 Ollama network error for {user}@{client_ip}: {str(e)}")
        raise HTTPException(status_code=502, detail=f"Ollama backend unavailable: {str(e)}")
    except Exception as e:
        logging.error(f"🔥 Unexpected error for {user}@{client_ip}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Internal error: {str(e)}")
