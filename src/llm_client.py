# llm_client.py
import requests
import json
from loguru import logger
from typing import Optional
from src.schemas import DetectedTask
from pydantic import ValidationError


class OllamaClient:
    def __init__(self, base_url="http://localhost:11434", model="mixtral-7b", timeout=30):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.timeout = timeout

    def build_prompt(self, raw_subject: str, raw_body: str, headers: dict) -> str:
        # Carefully instruct model to reply with strict JSON only matching the schema.
        prompt = f"""
You are an assistant that extracts meeting/task scheduling details from an email. 
Return ONLY a single JSON object with keys exactly: task_name, due_date, time, priority, description.
- due_date: ISO 8601 date (YYYY-MM-DD) if you can determine a date, otherwise null.
- time: HH:MM (24h) and timezone if possible, or null.
- priority: one of "low","medium","high" — default to "medium" if no clue.
- description: short (max 200 chars) summary.
Do not output any prose, explanation, or additional keys. If information is ambiguous, infer reasonably (e.g. "tomorrow at 6pm" => actual date) and prefer future date relative to the email Date header.

EMAIL SUBJECT:
{raw_subject}

EMAIL HEADERS:
{json.dumps(headers, ensure_ascii=False, indent=2)}

EMAIL BODY:
{raw_body}

Now output the JSON object as described.
"""
        return prompt

    def generate_json(self, subject: str, body: str, headers: dict, max_retries=2) -> Optional[DetectedTask]:
        prompt = self.build_prompt(subject, body, headers)
        url = f"{self.base_url}/api/generate"
        payload = {"model": self.model, "prompt": prompt, "max_tokens": 300}
        for attempt in range(max_retries + 1):
            try:
                r = requests.post(url, json=payload, timeout=self.timeout)
                r.raise_for_status()
                resp = r.json()
                # Ollama might return a structure; many Ollama servers return {"id":..., "result": {"content": "..."}}
                # Try to find text output:
                text = None
                # Inspect common shapes:
                if isinstance(resp, dict):
                    # try ['choices'][0]['message']['content']
                    if "choices" in resp:
                        ch = resp["choices"]
                        if ch and isinstance(ch, list) and "message" in ch[0]:
                            text = ch[0]["message"].get("content")
                    if not text and "result" in resp and isinstance(resp["result"], dict):
                        # try result.output or result.content
                        text = resp["result"].get("content") or resp["result"].get("output")
                # fallback: if response is text/plain
                if text is None:
                    # Try as string (some Ollama servers return plain text)
                    text = r.text
                # isolate JSON substring (try to parse)
                import re
                m = re.search(r'(\{[\s\S]*\})', text)
                json_text = m.group(1) if m else text.strip()
                data = json.loads(json_text)
                # Validate & coerce via Pydantic
                # Convert due_date/time formats if needed in utils, but try direct parse
                detected = DetectedTask(**data)
                return detected
            except (requests.RequestException, json.JSONDecodeError, ValidationError) as e:
                logger.warning(f"Ollama parse/req error attempt {attempt}: {e}")
                if attempt == max_retries:
                    logger.error("Ollama failed to produce valid JSON after retries.")
                    return None
        return None
