# main.py
"""
FastAPI app with two endpoints:
 - POST /generate  -> generate email JSON (one or more variants) from lead data
 - POST /send      -> send email (expects recipient + subject + body) using SMTP helper

This implementation:
 - The /generate endpoint runs the model `variants` times (separate calls).
 - Each call uses a temperature tweaked by +/-10% per step around the base temperature.
 - The prompt DOES NOT embed 'variants' or 'creativity' — repetition and temperature spread produce variation.

Environment variables required:
  OPENROUTER_API_KEY
  SMTP_HOST, SMTP_PORT, SMTP_USER, SMTP_PASS, FROM_EMAIL (for sending)

Run:
  pip install fastapi uvicorn requests python-dotenv pydantic
  uvicorn main:app --reload
"""

from typing import Optional, List, Any
import os
import json
import re
import textwrap
import requests
import uuid

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from dotenv import load_dotenv

# load .env if present
load_dotenv()

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
if not OPENROUTER_API_KEY:
    raise RuntimeError("Set OPENROUTER_API_KEY env var before running (get it from openrouter.ai).")

OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
MODEL = os.getenv("MODEL", "mistralai/mistral-7b-instruct:free")

# Demo recipient email (for demonstration purposes - all emails go here)
DEMO_RECIPIENT_EMAIL = os.getenv("DEMO_RECIPIENT_EMAIL", "goledc123@gmail.com")

# import SMTP helper
from smtp_send import send_email_resend

app = FastAPI(title="AI Email Outreach Backend (generate/send)")

# Configure CORS (Cross-Origin Resource Sharing)
# Allow requests from frontend
ALLOWED_ORIGINS_ENV = os.getenv("ALLOWED_ORIGINS", "*")
if ALLOWED_ORIGINS_ENV == "*":
    ALLOWED_ORIGINS = ["*"]
else:
    # Split by comma and strip whitespace for multiple origins
    ALLOWED_ORIGINS = [origin.strip() for origin in ALLOWED_ORIGINS_ENV.split(",")]

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,  # List of allowed origins, or ["*"] for all
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods (GET, POST, PUT, DELETE, etc.)
    allow_headers=["*"],  # Allows all headers
)

# ----------------- Models -----------------
class Lead(BaseModel):
    first_name: str
    last_name: Optional[str] = None
    company: Optional[str] = None
    title: Optional[str] = None
    zip: Optional[str] = None
    insight: Optional[str] = None
    business_specialization: Optional[str] = "Life insurance, Tax Preparation, Credit Repair, and Business Startup services for Women"

class GenerateRequest(BaseModel):
    lead: Lead
    temperature: Optional[float] = Field(0.7, ge=0.0, le=1.0)
    max_tokens: Optional[int] = 500
    tone: Optional[str] = "Friendly"
    variants: Optional[int] = Field(1, ge=1, le=10)  # 1..10 variants

class VariantModel(BaseModel):
    id: str
    subject: str
    body: str
    cta_text: Optional[str] = None
    confidence: Optional[float] = None
    used_tokens: Optional[List[str]] = None
    raw_text: Optional[str] = None

class GenerateResponse(BaseModel):
    variants: List[VariantModel]
    raw_text: Optional[str] = None  # raw was kept in each variant.raw_text

class SendRequest(BaseModel):
    recipient_email: str
    subject: str
    body: str
    plain_text: Optional[str] = None
    lead: Optional[Lead] = None
    consent_snapshot: Optional[dict] = None

# ----------------- Prompt builder -----------------
def build_prompt(lead: dict, tone: str) -> list:
    """
    Build the chat messages for a single-email generation.
    This prompt intentionally does NOT include variants or temperature.
    """
    system = {
        "role": "system",
        "content": (
            "You are a world-class B2C copywriter trained in Apollo.io's high-conversion email framework. "
            "Write short, crystal-clear, and highly converting outreach emails targeted at busy customers. "
            "You are writing on behalf of The Policy Boss, a company specializing in Life insurance, Tax Preparation, Credit Repair, and Business Startup services for Women. "
            "\n\nConversion rules (must follow):\n"
            "1. Subject: 4–6 words, mobile-friendly, curiosity-driven, and directly relevant to the prospect's pain or goal.\n"
            "2. Clarity: Target a 5th-grade reading level. Use simple language, minimal adverbs, and sentences < 15 words.\n"
            "3. Structure: Hook (personalization) → Specific pain → Clear value proposition → Single low-friction CTA.\n"
            "4. Engagement: Personalize to the lead's name, title, and company. Mention 'The Policy Boss' naturally in the email body where appropriate. Add short social proof relevant to the business specialization if possible.\n"
            "5. Length: 70–100 words total in the body. Keep 3–4 short paragraphs (1–2 lines each).\n"
            "6. Output: Strict JSON with keys: subject, body, cta_text. No extra keys or commentary.\n"
            "\nImportant: Use the business specialization (provided by the user) to tailor the message, and naturally incorporate The Policy Boss as the company name."
        )
    }

    user_instructions = (f"""
Lead details:
{json.dumps(lead, indent=2)}

Company: The Policy Boss
Tone: {tone}

Task:
Write **one** personalized outreach email tailored to the lead and the business specialization above.
- Write the email on behalf of The Policy Boss (include the company name naturally in the email body).
- Make the subject 4–6 words and directly relevant to the lead's role or business specialization.
- Body must be 70–100 words, 3–4 short paragraphs, sentences under 15 words, and at a 5th-grade reading level.
- Include a one-line social proof if available or a short, believable benchmark (e.g., "Clients cut churn 20-30%").
- Make the text visually good looking using html tags add bold italic, new lines etc.
- End with a single clear, low-friction CTA that matches the CTA field.
"""
    "IMPORTANT INSTRUCTIONS FOR THE RESPONSE FORMAT:\n"
        "1) RETURN ONLY a single JSON object that is directly parsable by standard JSON parsers (e.g., python's json.loads).\n"
        "2) The JSON MUST start with '{' and end with '}' and contain the exact keys described below.\n"
        "3) Do NOT include any surrounding explanatory text, headings, or bullet points.\n"
        "4) Do NOT wrap the JSON in markdown/code fences (```) or add language tags like ```json. Return the raw JSON only. that is very important if you do you lose\n"
        "5) If a value contains newlines, keep them inside the JSON string values only.\n\n"
        "Provide your response in the following json format (example keys and types):\n"
        "    {\n"
        "        \"subject\": the subject line of the email,\n"
        "        \"body\": the HTML body of the email"
        "        \"cta_text\": a clear cta"
        "    }\n\n"
    )

    return [system, {"role": "user", "content": textwrap.dedent(user_instructions)}]
# ----------------- Helpers -----------------
def extract_first_json(text: str):
    # remove spaces in the start and the end
    email = text.strip()
    # remove the  ```json and  ``` in the email with 
    email = email.replace("```json", "")
    cleaned = email.replace("```", "")
    try:
        return json.loads(cleaned)
    except Exception:
        raise ValueError("Could not parse JSON from model response.")

def call_openrouter(messages: list, model: str, temperature: float, max_tokens: int) -> str:
    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    resp = requests.post(OPENROUTER_URL, headers=headers, json=payload, timeout=30)
    try:
        resp.raise_for_status()
    except Exception as e:
        raise RuntimeError(f"OpenRouter API error: {e} - {resp.text[:400]}")
    data = resp.json()
    content = data["choices"][0]["message"]["content"]
    return content

def compute_spread_temperatures(base_temp: float, count: int) -> list:
    """
    Compute a list of `count` temperatures centered on base_temp, each step +/-10% of base.
    Example: base=0.6, count=3 -> [0.54, 0.6, 0.66] (approx)
    Clamp to [0.0, 1.0].
    """
    base = float(base_temp or 0.7)
    if count <= 1:
        return [max(0.0, min(1.0, base))]

    center = (count - 1) / 2.0
    temps = []
    for i in range(count):
        offset_steps = i - center
        temp = base * (1.0 + 0.10 * offset_steps)
        temp = max(0.0, min(1.0, temp))
        temps.append(round(temp, 3))
    return temps

# ----------------- Routes -----------------
@app.post("/generate", response_model=GenerateResponse)
def generate_endpoint(req: GenerateRequest):
    """
    Generate one or more personalized email variants (subject, body, cta_text) for a lead.
    Calls the model `variants` times; each call uses a tweaked temperature.
    """
    print("working")
    lead_dict = req.lead.model_dump() if hasattr(req.lead, "model_dump") else req.lead.dict()
    tone = req.tone or "Friendly"
    variants_count = int(req.variants or 1)
    base_temp = float(req.temperature or 0.7)
    max_tokens = int(req.max_tokens or 500)

    messages = build_prompt(lead_dict, tone)
    temps = compute_spread_temperatures(base_temp, variants_count)

    collected = []
    for idx, temp in enumerate(temps):
        try:
            print(f"--- Calling OpenRouter for variant {idx+1} with temp {temp} ---")
            raw = call_openrouter(messages, MODEL, temp, max_tokens)
            print(f"--- Raw response for variant {idx+1} ---")
            print(raw)
            print("------------------------------------------")
        except Exception as e:
            raise HTTPException(status_code=503, detail=f"OpenRouter error on variant {idx+1}: {e}")

        # parse JSON
        try:
            print(f"--- Parsing JSON for variant {idx+1} ---")
            parsed = extract_first_json(raw)
            print(f"--- Parsed JSON for variant {idx+1} ---")
            print(parsed)
            print("----------------------------------------")
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Failed to parse model JSON for variant {idx+1}: {e}. Raw: {raw[:1000]}")

        # normalize to dict (we expect single object)
        variant_obj = None
        if isinstance(parsed, dict):
            variant_obj = parsed
        elif isinstance(parsed, list) and len(parsed) > 0 and isinstance(parsed[0], dict):
            variant_obj = parsed[0]
        else:
            raise HTTPException(status_code=500, detail=f"Unexpected model response type for variant {idx+1}. Raw: {raw[:1000]}")

        subject = variant_obj.get("subject")
        body = variant_obj.get("body")
        cta = variant_obj.get("cta_text") or variant_obj.get("cta") or None

        if not subject or not body:
            raise HTTPException(status_code=500, detail=f"Model returned invalid variant {idx+1}. Missing subject/body. Raw: {raw[:1000]}")

        collected.append(
            VariantModel(
                id=variant_obj.get("id") or str(uuid.uuid4()),
                subject=subject,
                body=body,
                cta_text=cta,
                confidence=variant_obj.get("confidence"),
                used_tokens=variant_obj.get("used_tokens") or variant_obj.get("usedTokens"),
                raw_text=json.dumps(variant_obj)
            )
        )

    if not collected:
        raise HTTPException(status_code=500, detail="No valid variants produced.")

    return GenerateResponse(variants=collected, raw_text=None)

@app.post("/send")
def send_endpoint(req: SendRequest):
    """
    Send an email. Expects recipient_email, subject and body (HTML). Returns status.
    """
    if not req.recipient_email or not req.subject or not req.body:
        raise HTTPException(status_code=400, detail="recipient_email, subject and body are required.")

    try:
        # For demo purposes, always send to DEMO_RECIPIENT_EMAIL regardless of req.recipient_email
        ok = send_email_resend(to=DEMO_RECIPIENT_EMAIL, subject=req.subject, html_body=req.body)
        if not ok:
            raise HTTPException(status_code=500, detail="SMTP send failed.")
        return {"status": "success", "message": f"Email sent to {DEMO_RECIPIENT_EMAIL} (demo mode)"}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
