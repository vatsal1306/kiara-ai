#!/usr/bin/env python3
"""
gmail_meetings_extractor.py

Updated behavior (per your request):
- On first run (no last-seen saved), fetch ONLY the latest email, save its message id + timestamp, and do NOT process anything else.
- On subsequent runs (every POLL_INTERVAL seconds), fetch all emails received AFTER the saved timestamp (using Gmail 'after:' search),
  fetch their full messages, sort them chronologically (oldest->newest), process each, and finally update the saved last-seen id/timestamp
  to the newest message processed in that run.

Other behavior is unchanged:
- Emails are not modified.
- Uses local Ollama.
- Uses IST and ISO 8601 normalization.
"""

import os
import time
import json
import logging
import base64
import re
from typing import Optional, Dict, Any, Set, Tuple, List
from datetime import datetime
from datetime import timedelta

import requests
from dateutil import parser as dt_parser
import pytz
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
from dotenv import load_dotenv

# Google API imports
from google.oauth2.credentials import Credentials
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError

# --- Config & logging -------------------------------------------------------
load_dotenv()

CLIENT_ID = os.getenv("CLIENT_ID")
CLIENT_SECRET = os.getenv("CLIENT_SECRET")
REFRESH_TOKEN = os.getenv("REFRESH_TOKEN")
ACCESS_TOKEN = os.getenv("ACCESS_TOKEN")  # optional, not required if refresh token is present

OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3")
OLLAMA_API_URL = os.getenv("OLLAMA_API_URL", "http://localhost:11434/api/generate")

POLL_INTERVAL = int(os.getenv("POLL_INTERVAL", "300"))  # seconds
LAST_SEEN_STORE = os.getenv("LAST_SEEN_STORE", "last_seen.json")
PROCESSED_STORE = os.getenv("PROCESSED_STORE", "processed_ids.json")  # kept for backward compatibility (unused)
OUTPUT_DIR = os.getenv("OUTPUT_DIR", "outputs")

# kiara
KIA_API_URL = os.getenv("KIA_API_URL", "https://api.kiarai.co/api/tasks/create")
KIA_API_USER_ID = os.getenv("KIA_API_USER_ID", "b5402f3c-c867-4565-9cce-0b7c99efddeb")
KIA_API_KEY = os.getenv("KIA_API_KEY", None)  # optional Bearer token if required by API
DEFAULT_REMINDER_MINUTES = int(os.getenv("DEFAULT_REMINDER_MINUTES", "10"))  # 5/10/30 allowed by BE
DEFAULT_DURATION_MINUTES = int(os.getenv("DEFAULT_DURATION_MINUTES", "60"))  # how long to assume if end_time not provided

PRIORITY_MAP = {"low": 0, "medium": 1, "high": 2}


# Gmail token endpoint
TOKEN_URI = "https://oauth2.googleapis.com/token"

# Target timezone
TARGET_TZ = pytz.timezone("Asia/Kolkata")

# Logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger("gmail_meetings_extractor")


# --- Last-seen store -------------------------------------------------------
def load_last_seen(path: str) -> Optional[dict]:
    """
    Returns dict: {"last_seen_id": str, "last_seen_ts": int_seconds}
    or None if not present.
    """
    if not os.path.exists(path):
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
            if not isinstance(data, dict):
                return None
            return data
    except Exception:
        logger.exception("Failed to load last-seen file; ignoring and treating as first run.")
        return None


def save_last_seen(path: str, message_id: str, ts_seconds: int) -> None:
    tmp = f"{path}.tmp"
    payload = {"last_seen_id": message_id, "last_seen_ts": int(ts_seconds)}
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    os.replace(tmp, path)
    logger.info("Persisted last-seen id=%s ts=%d", message_id, ts_seconds)


# --- Gmail helpers ---------------------------------------------------------
def build_gmail_service() -> Any:
    if not (CLIENT_ID and CLIENT_SECRET and REFRESH_TOKEN):
        raise RuntimeError("Missing CLIENT_ID / CLIENT_SECRET / REFRESH_TOKEN in environment.")
    creds = Credentials(
        token=None,
        refresh_token=REFRESH_TOKEN,
        token_uri=TOKEN_URI,
        client_id=CLIENT_ID,
        client_secret=CLIENT_SECRET,
        scopes=["https://www.googleapis.com/auth/gmail.readonly"]
    )
    service = build("gmail", "v1", credentials=creds, cache_discovery=False)
    return service


@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=1, max=10),
       retry=retry_if_exception_type(HttpError))
def list_message_ids(service, q: Optional[str] = None, max_results: Optional[int] = None) -> list:
    """
    Return list of message ids (from messages.list) matching optional query q.
    """
    ids = []
    request = service.users().messages().list(userId="me", q=q, labelIds=["INBOX"], maxResults=max_results)
    while request is not None:
        resp = request.execute()
        for m in resp.get("messages", []):
            ids.append(m["id"])
        request = service.users().messages().list_next(request, resp)
    return ids


@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=1, max=10),
       retry=retry_if_exception_type(Exception))
def get_message_full_by_id(service, message_id: str) -> dict:
    """
    Fetch full message resource (includes 'internalDate' in milliseconds).
    """
    return service.users().messages().get(userId="me", id=message_id, format="full").execute()


@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=1, max=10),
       retry=retry_if_exception_type(Exception))
def get_latest_message_id(service) -> Optional[str]:
    """
    Get the latest message id in INBOX by requesting maxResults=1.
    Note: Gmail's messages.list tends to return newest-first; this is the pragmatic approach.
    """
    resp = service.users().messages().list(userId="me", labelIds=["INBOX"], maxResults=1).execute()
    msgs = resp.get("messages", [])
    if not msgs:
        return None
    return msgs[0]["id"]


# --- Email parsing (unchanged) --------------------------------------------
def _fix_b64_padding(s: str) -> str:
    s = s.replace("-", "+").replace("_", "/")
    padding = len(s) % 4
    if padding:
        s += "=" * (4 - padding)
    return s


def extract_plain_text_from_payload(payload: dict) -> str:
    text_parts = []

    def walk(part):
        mime = part.get("mimeType", "")
        body = part.get("body", {})
        if mime == "text/plain" and body.get("data"):
            raw = body["data"]
            try:
                decoded = base64.urlsafe_b64decode(_fix_b64_padding(raw)).decode("utf-8", errors="replace")
                text_parts.append(decoded)
            except Exception:
                logger.exception("Failed to decode text/plain part.")
        elif mime == "text/html" and body.get("data"):
            raw = body["data"]
            try:
                decoded = base64.urlsafe_b64decode(_fix_b64_padding(raw)).decode("utf-8", errors="replace")
                stripped = re.sub(r"<[^>]+>", " ", decoded)
                text_parts.append(stripped)
            except Exception:
                logger.exception("Failed to decode text/html part.")
        for p in part.get("parts", []) or []:
            walk(p)

    walk(payload)
    return "\n".join(text_parts).strip()


def parse_headers(headers: list) -> dict:
    d = {}
    for h in headers:
        name = h.get("name", "").lower()
        value = h.get("value", "")
        d[name] = value
    return {
        "from": d.get("from", ""),
        "to": d.get("to", ""),
        "cc": d.get("cc", ""),
        "subject": d.get("subject", ""),
        "date": d.get("date", "")
    }


def preprocess_email_body(body: str) -> str:
    if not body:
        return ""
    lines = body.splitlines()
    cleaned = []
    for ln in lines:
        if ln.strip().startswith(">"):
            continue
        if re.match(r"On .* wrote:", ln):
            break
        if ln.strip() == "--":
            break
        cleaned.append(ln)
    text = "\n".join(cleaned)
    text = re.sub(r"Sent from my .*", "", text, flags=re.IGNORECASE)
    return text.strip()


# --- Ollama / LLM interaction (unchanged) ---------------------------------
def build_llm_prompt(email_meta: dict, body_text: str, reference_dt_iso: str) -> str:
    instruction = f"""
You are a strict JSON extractor. Given the email metadata and body below,
extract meeting information if the email proposes a meeting or requests scheduling.

Important:
- ALWAYS return exactly one JSON object and nothing else.
- Use ISO 8601 for dates and times.
- Date field SHOULD be "due_date": "YYYY-MM-DD" in IST.
- Time field SHOULD be "time": "HH:MM:SS+05:30" (24-hour, include +05:30).
- If no date or no time can be determined, use null for that field.
- Priority must be one of "low", "medium", "high". If not specified, default to "medium".
- If you detect multiple candidate times, choose the earliest upcoming option relative to the reference datetime below.
- Extract meeting links (Zoom/Google Meet/etc.) and include them in the "description" if found.
- Provide a concise "task_name" and a short "description".
- Do not include additional fields.
- Do not hallucinate.
- Do not return anything other than the JSON object.

Reference (for resolving relative dates/times):
- Use this reference datetime (IST): {reference_dt_iso}

Email metadata (JSON):
{json.dumps(email_meta, ensure_ascii=False, indent=2)}

Email body (preprocessed plain text):
\"\"\"{body_text}\"\"\"

Return JSON with these fields:
{{"message_id": "<gmail-message-id>",
 "task_name": <string or null>,
 "due_date": <"YYYY-MM-DD" or null>,
 "time": <"HH:MM:SS+05:30" or null>,
 "priority": <"low"|"medium"|"high">,
 "description": <string or null>
}}
"""
    return instruction.strip()


@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=1, max=10),
       retry=retry_if_exception_type(Exception))
def call_ollama_generate(prompt: str, model: str = OLLAMA_MODEL, timeout: int = 120) -> str:
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "temperature": 0.0,
        "max_tokens": 512
    }
    breakpoint()
    resp = requests.post(OLLAMA_API_URL, json=payload, timeout=timeout)
    logger.info(f"LLM response status: {resp.status_code}")
    resp.raise_for_status()
    return resp.text


def extract_json_from_text(text: str) -> Optional[dict]:
    text = text.strip()
    text = json.loads(text)
    try:
        return json.loads(text['response'])
    except Exception:
        text = text['response']
        start = text.find('{')
        if start == -1:
            return None
        depth = 0
        end = None
        for i in range(start, len(text)):
            if text[i] == '{':
                depth += 1
            elif text[i] == '}':
                depth -= 1
                if depth == 0:
                    end = i + 1
                    break
        if end:
            json_str = text[start:end]
            try:
                return json.loads(json_str)
            except Exception:
                m = re.search(r'(\{.*\})', text, flags=re.DOTALL)
                if m:
                    try:
                        return json.loads(m.group(1))
                    except Exception:
                        return None
        return None


# --- Normalization & validation (unchanged) --------------------------------
def normalize_date_time_fields(parsed: dict, reference_dt: datetime) -> dict:
    out = {}
    out["message_id"] = parsed.get("message_id")
    tn = parsed.get("task_name")
    out["task_name"] = tn if isinstance(tn, str) and tn.strip() else None
    pr = (parsed.get("priority") or "").lower()
    if pr not in ("low", "medium", "high"):
        pr = "medium"
    out["priority"] = pr
    desc = parsed.get("description")
    out["description"] = desc if isinstance(desc, str) and desc.strip() else None

    dd = parsed.get("due_date")
    if dd and isinstance(dd, str):
        try:
            parsed_date = dt_parser.isoparse(dd)
            out["due_date"] = parsed_date.date().isoformat()
        except Exception:
            try:
                parsed_date = dt_parser.parse(dd, default=reference_dt)
                out["due_date"] = parsed_date.date().isoformat()
            except Exception:
                out["due_date"] = None
    else:
        out["due_date"] = None

    t_field = parsed.get("time")
    if t_field and isinstance(t_field, str):
        try:
            dt = dt_parser.isoparse(t_field)
            dt_ist = dt.astimezone(TARGET_TZ)
            out["time"] = dt_ist.strftime("%H:%M:%S%z")
            out["time"] = out["time"][:-2] + ":" + out["time"][-2:]
        except Exception:
            try:
                parsed_time = dt_parser.parse(t_field, default=reference_dt)
                parsed_time = TARGET_TZ.localize(parsed_time.replace(tzinfo=None)) if parsed_time.tzinfo is None else parsed_time.astimezone(TARGET_TZ)
                out["time"] = parsed_time.strftime("%H:%M:%S%z")
                out["time"] = out["time"][:-2] + ":" + out["time"][-2:]
            except Exception:
                out["time"] = None
    else:
        out["time"] = None

    return out

def _parse_time_to_hhmm(time_str: Optional[str], due_date: Optional[str]) -> Optional[str]:
    """
    time_str expected like "HH:MM:SS+05:30" or ISO time. If present, return "HH:MM".
    If time_str is None but due_date present, return default "09:00".
    If both missing return None.
    """
    if time_str:
        try:
            dt = dt_parser.isoparse(time_str)
            dt_ist = dt.astimezone(TARGET_TZ)
            return dt_ist.strftime("%H:%M")
        except Exception:
            try:
                # Try parsing just as time
                parsed_time = dt_parser.parse(time_str)
                parsed_time = TARGET_TZ.localize(parsed_time.replace(tzinfo=None)) if parsed_time.tzinfo is None else parsed_time.astimezone(TARGET_TZ)
                return parsed_time.strftime("%H:%M")
            except Exception:
                return None
    # fallback default time if date exists
    if due_date:
        return "09:00"
    return None

def _compute_end_time(start_hhmm: Optional[str], duration_minutes: int = DEFAULT_DURATION_MINUTES) -> Optional[str]:
    if not start_hhmm:
        return None
    try:
        hh, mm = map(int, start_hhmm.split(":"))
        start_dt = datetime.now(TARGET_TZ).replace(hour=hh, minute=mm, second=0, microsecond=0)
        end_dt = start_dt + timedelta(minutes=duration_minutes)
        return end_dt.strftime("%H:%M")
    except Exception:
        return None

@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=1, max=8),
       retry=retry_if_exception_type(Exception))
def call_create_task_api(payload: dict, timeout: int = 15) -> dict:
    headers = {"Content-Type": "application/json"}
    if KIA_API_KEY:
        headers["Authorization"] = f"Bearer {KIA_API_KEY}"
    resp = requests.post(KIA_API_URL, json=payload, headers=headers, timeout=timeout)
    resp.raise_for_status()
    return resp.json()

def on_meeting_detected(task: dict, email_meta: dict, full_msg: dict) -> None:
    """
    Called when a meeting is detected. This implementation will call the Kiarai task-create API.
    Customize as required.
    """
    msg_id = task.get("message_id", "unknown")
    logger.info("Meeting detected for message %s: %s", msg_id, task)

    # Prepare fields for API payload
    name = task.get("task_name") or email_meta.get("subject") or f"Gmail task {msg_id}"
    due_date = task.get("due_date")  # expected "YYYY-MM-DD" or None
    # normalized time in task['time'] is expected like "HH:MM:SS+05:30"
    start_time_hhmm = _parse_time_to_hhmm(task.get("time"), due_date)
    end_time_hhmm = None
    if start_time_hhmm:
        end_time_hhmm = _compute_end_time(start_time_hhmm, DEFAULT_DURATION_MINUTES)

    # map priority string to integer expected by API
    priority_str = (task.get("priority") or "medium").lower()
    priority_int = PRIORITY_MAP.get(priority_str, 1)

    # reminder time: ensure allowed values (5,10,30). fallback to default.
    reminder = DEFAULT_REMINDER_MINUTES
    if reminder not in (5, 10, 30):
        reminder = 10

    payload = {
        "user_id": KIA_API_USER_ID,
        "task_type": "gmail",
        "name": name,
        "due_date": due_date or datetime.now(TARGET_TZ).date().isoformat(),
        "start_time": start_time_hhmm or "09:00",
        "end_time": end_time_hhmm,  # can be None
        "priority": priority_int,
        "reminder_time": reminder,
        "description": task.get("description") or email_meta.get("subject") or None
    }

    # Remove keys with None if API prefers absence over null (safe to keep null too; adapt as needed)
    cleaned_payload = {k: v for k, v in payload.items() if v is not None}

    try:
        logger.info("Creating task in remote API for message %s: %s", msg_id, cleaned_payload)
        api_resp = call_create_task_api(cleaned_payload)
        # persist response for traceability
        meetings_dir = os.path.join(OUTPUT_DIR, "meetings")
        os.makedirs(meetings_dir, exist_ok=True)
        api_out_path = os.path.join(meetings_dir, f"{msg_id}_api.json")
        with open(api_out_path, "w", encoding="utf-8") as fh:
            json.dump({"request": cleaned_payload, "response": api_resp, "created_at": datetime.now(TARGET_TZ).isoformat()}, fh, indent=2, ensure_ascii=False)
        logger.info("Task created for message %s, API response saved to %s", msg_id, api_out_path)
    except Exception:
        logger.exception("Failed to create task for message %s via API; saving request for inspection.", msg_id)
        # save the failed request for debugging
        debug_path = os.path.join(OUTPUT_DIR, "meetings", f"{msg_id}_api_failed.json")
        os.makedirs(os.path.dirname(debug_path), exist_ok=True)
        with open(debug_path, "w", encoding="utf-8") as fh:
            json.dump({"request": cleaned_payload, "error_at": datetime.now(TARGET_TZ).isoformat()}, fh, indent=2, ensure_ascii=False)

# --- Processing a full message (new helper to avoid double-fetch) ----------
def process_full_message(full_msg: dict) -> Optional[dict]:
    """
    Accepts a full message resource (from messages.get with format=full),
    extracts content and calls the LLM, then normalizes and saves output.
    Returns normalized task dict or None on failure.
    """

    # ------------------ user callback: handle detected meetings ------------------
    # def on_meeting_detected(task: dict, email_meta: dict, full_msg: dict) -> None:
    #     """
    #     User-implementable callback invoked when a meeting is detected.
    #
    #     Parameters:
    #     - task: normalized task dict (message_id, task_name, due_date, time, priority, description)
    #     - email_meta: {subject, from, to, cc, received_at}
    #     - full_msg: original Gmail full message resource (useful for headers, threadId, raw payload, etc.)
    #
    #     Default behavior: log and persist a short artifact to outputs/meetings/<message_id>.json.
    #     Customize this function to call calendars, webhooks, databases, etc.
    #     """
    #     msg_id = task.get("message_id", "unknown")
    #     logger.info("Meeting detected for message %s: %s", msg_id, task)
    #
    #     # default persistence (non-blocking-ish): write a copy under outputs/meetings/
    #     try:
    #         meetings_dir = os.path.join(OUTPUT_DIR, "meetings")
    #         os.makedirs(meetings_dir, exist_ok=True)
    #         out_path = os.path.join(meetings_dir, f"{msg_id}.json")
    #         payload = {
    #             "detected_at": datetime.now(TARGET_TZ).isoformat(),
    #             "task": task,
    #             "email_meta": email_meta,
    #             # optionally include minimal reference to full message (not entire payload to avoid huge files)
    #             "message_ref": {"id": full_msg.get("id"), "threadId": full_msg.get("threadId")}
    #         }
    #         with open(out_path, "w", encoding="utf-8") as fh:
    #             json.dump(payload, fh, indent=2, ensure_ascii=False)
    #         logger.info("Saved meeting artifact to %s", out_path)
    #     except Exception:
    #         logger.exception("Failed to persist meeting artifact for %s", msg_id)

    message_id = full_msg.get("id")
    payload = full_msg.get("payload", {})
    headers = parse_headers(payload.get("headers", []) + full_msg.get("payload", {}).get("headers", []))
    raw_body = extract_plain_text_from_payload(payload)
    pre_body = preprocess_email_body(raw_body)

    raw_date_header = headers.get("date", "")
    try:
        parsed_received = dt_parser.parse(raw_date_header)
        if parsed_received.tzinfo is None:
            parsed_received = TARGET_TZ.localize(parsed_received)
        parsed_received_ist = parsed_received.astimezone(TARGET_TZ)
    except Exception:
        parsed_received_ist = datetime.now(TARGET_TZ)

    reference_iso = parsed_received_ist.isoformat()

    email_meta = {
        "subject": headers.get("subject"),
        "from": headers.get("from"),
        "to": headers.get("to"),
        "cc": headers.get("cc"),
        "received_at": reference_iso
    }

    prompt = build_llm_prompt({**email_meta, "message_id": message_id}, pre_body, reference_iso)
    try:
        logger.info("Calling LLM")
        llm_out_text = call_ollama_generate(prompt, model=OLLAMA_MODEL)
        logger.info("LLM call successful")
    except Exception:
        logger.exception("LLM call failed for message %s", message_id)
        return None

    parsed = extract_json_from_text(llm_out_text)
    if not parsed:
        logger.warning("Could not parse JSON from LLM for message %s; saving raw output.", message_id)
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        fallback_path = os.path.join(OUTPUT_DIR, f"{message_id}_raw.txt")
        with open(fallback_path, "w", encoding="utf-8") as f:
            f.write(llm_out_text)
        return None

    # Parse internalDate for reference (if present)
    internal_ms = None
    try:
        internal_ms = int(full_msg.get("internalDate"))  # milliseconds
    except Exception:
        internal_ms = None

    parsed_received_dt = parsed_received_ist if parsed_received_ist is not None else datetime.now(TARGET_TZ)
    normalized = normalize_date_time_fields(parsed, parsed_received_dt)
    if not normalized.get("message_id"):
        normalized["message_id"] = message_id

    # --- invoke user callback if a meeting was detected ---
    meeting_found = any([
        normalized.get("task_name"),
        normalized.get("due_date"),
        normalized.get("time"),
        normalized.get("description")
    ])
    if meeting_found:
        try:
            logger.info("Meeting found for message %s", message_id)
            # email_meta was built earlier in this function
            on_meeting_detected(normalized, email_meta, full_msg)
        except Exception:
            logger.exception("on_meeting_detected callback failed for message %s", message_id)

    # Save output artifact
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    out_path = os.path.join(OUTPUT_DIR, f"{message_id}.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({
            "extracted_at": datetime.now(TARGET_TZ).isoformat(),
            "email_meta": email_meta,
            "task": normalized,
            "llm_raw_output": llm_out_text[:4000],
            "internalDate_ms": internal_ms
        }, f, indent=2, ensure_ascii=False)
    logger.info("Saved extracted info for %s -> %s", message_id, out_path)
    return normalized

def filter_messages_after_ms(full_msgs: List[dict], last_seen_ts_sec: int) -> List[dict]:
    """
    Return only messages whose internalDate (ms) is strictly greater than last_seen_ts_sec*1000.
    Keeps original chronological order (assuming full_msgs already sorted ascending).
    """
    cutoff_ms = int(last_seen_ts_sec) * 1000
    filtered = []
    for m in full_msgs:
        try:
            ts = int(m.get("internalDate", 0))
        except Exception:
            ts = 0
        if ts > cutoff_ms:
            filtered.append(m)
    return filtered


# --- Helper: fetch and sort full messages by internalDate -------------------
def fetch_full_messages_sorted(service, message_ids: List[str]) -> List[dict]:
    """
    Given a list of message IDs, fetch full resources and return a list sorted by internalDate asc.
    If internalDate is missing, treat it as 0 and place them first.
    """
    full_msgs = []
    for mid in message_ids:
        try:
            full = get_message_full_by_id(service, mid)
            full_msgs.append(full)
        except Exception:
            logger.exception("Failed to fetch full message %s; skipping.", mid)
    # sort by internalDate (ms) ascending
    def get_ts(m):
        try:
            return int(m.get("internalDate", 0))
        except Exception:
            return 0
    full_msgs.sort(key=get_ts)
    return full_msgs


# --- Main loop: initial + subsequent runs ----------------------------------
def main_loop():
    service = build_gmail_service()
    last_seen = load_last_seen(LAST_SEEN_STORE)
    if not last_seen:
        # First run: fetch only latest message id, save its id + timestamp, and exit loop (wait for next cycle).
        logger.info("No last-seen found: performing initial bootstrap (fetch latest message only).")
        try:
            latest_id = get_latest_message_id(service)
            if not latest_id:
                logger.info("No messages found in INBOX on initial bootstrap.")
                # nothing to save; just sleep and continue to next iteration
                time.sleep(POLL_INTERVAL)
            else:
                latest_full = get_message_full_by_id(service, latest_id)
                try:
                    internal_ms = int(latest_full.get("internalDate"))
                    ts_sec = internal_ms // 1000
                except Exception:
                    # fallback: parse Date header
                    headers = parse_headers(latest_full.get("payload", {}).get("headers", []))
                    date_header = headers.get("date", "")
                    try:
                        parsed_date = dt_parser.parse(date_header)
                        if parsed_date.tzinfo is None:
                            parsed_date = TARGET_TZ.localize(parsed_date)
                        ts_sec = int(parsed_date.timestamp())
                    except Exception:
                        ts_sec = int(datetime.now(TARGET_TZ).timestamp())
                save_last_seen(LAST_SEEN_STORE, latest_id, ts_sec)
                logger.info("Bootstrap complete. Saved last-seen id=%s ts=%d", latest_id, ts_sec)
        except Exception:
            logger.exception("Initial bootstrap failed; will retry on next loop.")
        # enter the polling loop from here on
    else:
        logger.info("Loaded last-seen: %s", last_seen)

    # Enter continuous polling loop
    try:
        while True:
            try:
                last_seen = load_last_seen(LAST_SEEN_STORE)
                if last_seen:
                    after_ts = int(last_seen.get("last_seen_ts", 0))
                    q = f"in:inbox after:{after_ts}"
                    logger.info("Listing messages with query: %s", q)
                    ids = list_message_ids(service, q=q)
                else:
                    # If last_seen disappeared for some reason, fall back to getting latest id only (no processing)
                    logger.warning("No last-seen on periodic run; performing bootstrap again and skipping processing.")
                    ids = []
                    latest_id = get_latest_message_id(service)
                    if latest_id:
                        latest_full = get_message_full_by_id(service, latest_id)
                        try:
                            internal_ms = int(latest_full.get("internalDate"))
                            ts_sec = internal_ms // 1000
                        except Exception:
                            headers = parse_headers(latest_full.get("payload", {}).get("headers", []))
                            date_header = headers.get("date", "")
                            try:
                                parsed_date = dt_parser.parse(date_header)
                                if parsed_date.tzinfo is None:
                                    parsed_date = TARGET_TZ.localize(parsed_date)
                                ts_sec = int(parsed_date.timestamp())
                            except Exception:
                                ts_sec = int(datetime.now(TARGET_TZ).timestamp())
                        save_last_seen(LAST_SEEN_STORE, latest_id, ts_sec)

                if not ids:
                    logger.info("No new messages found after last-seen.")
                else:
                    # Fetch full messages and sort chronologically (oldest -> newest)
                    full_msgs = fetch_full_messages_sorted(service, ids)
                    logger.info("Fetched %d messages since last seen", len(full_msgs))
                    last_seen = load_last_seen(LAST_SEEN_STORE) or {}
                    last_seen_ts_sec = int(last_seen.get("last_seen_ts", 0))
                    full_msgs = filter_messages_after_ms(full_msgs, last_seen_ts_sec)
                    logger.info("Filtered %d messages since last seen", len(full_msgs))
                    processed_count = 0
                    newest_ts_ms = None
                    newest_msg_id = None

                    for full in full_msgs:
                        mid = full.get("id")
                        try:
                            internal_ms = int(full.get("internalDate", 0))
                        except Exception:
                            internal_ms = 0
                        # Process the full message
                        res = process_full_message(full)
                        if res is not None:
                            processed_count += 1
                        # track newest
                        if newest_ts_ms is None or internal_ms > newest_ts_ms:
                            newest_ts_ms = internal_ms
                            newest_msg_id = mid

                    logger.info("Processed %d new messages this cycle.", processed_count)

                    # update last-seen to the newest message we fetched this run (even if processing failed)
                    if newest_msg_id and newest_ts_ms:
                        save_last_seen(LAST_SEEN_STORE, newest_msg_id, newest_ts_ms // 1000)

            except Exception:
                logger.exception("Error during polling cycle; will continue on next cycle.")

            time.sleep(POLL_INTERVAL)

    except KeyboardInterrupt:
        logger.info("Interrupted by user; exiting.")


if __name__ == "__main__":
    main_loop()
