"""
smtp_send.py

Simple, robust SMTP sender (Gmail-compatible).
Requirements: built-in smtplib, email
Environment variables expected:
    SENDER_EMAIL        - the sending Gmail address (you@example.com)
    GMAIL_APP_PASSWORD  - Gmail App Password (NOT your normal password)
Optional:
    SMTP_HOST (default "smtp.gmail.com")
    SMTP_PORT (default 587)
"""

import json
import os
import time
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from typing import Optional
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

SENDER_EMAIL = os.getenv("SENDER_EMAIL")
GMAIL_APP_PASSWORD = os.getenv("GMAIL_APP_PASSWORD")
SMTP_HOST = os.getenv("SMTP_HOST", "smtp.gmail.com")
SMTP_PORT = int(os.getenv("SMTP_PORT", "587"))

if not SENDER_EMAIL or not GMAIL_APP_PASSWORD:
    raise RuntimeError("Set SENDER_EMAIL and GMAIL_APP_PASSWORD environment variables in .env file.")


def _connect_smtp(host: str = SMTP_HOST, port: int = SMTP_PORT, timeout: int = 30):
    """
    Create and return a connected SMTP client with STARTTLS enabled.
    Caller should quit() the returned server when done.
    """
    server = smtplib.SMTP(host, port, timeout=timeout)
    server.ehlo()
    server.starttls()
    server.ehlo()
    server.login(SENDER_EMAIL, GMAIL_APP_PASSWORD)
    return server


def send_email_smtp(
    recipient_email: str,
    subject: str,
    plain_body: str,
    html_body: Optional[str] = None,
    from_name: Optional[str] = None,
    reply_to: Optional[str] = None,
    max_retries: int = 3,
    retry_delay_sec: float = 2.0,
) -> bool:
    """
    Send an email via SMTP.
    - recipient_email: final recipient address
    - subject: subject line (string)
    - plain_body: plain-text version (required)
    - html_body: optional HTML version (recommended if you want formatting)
    - from_name: optional display name for sender (e.g., "Haitham <you@example.com>")
    - reply_to: optional reply-to address
    - Returns True on success, False otherwise.
    """
    if from_name:
        from_header = f"{from_name} <{SENDER_EMAIL}>"
    else:
        from_header = SENDER_EMAIL

    # Build MIME message
    if html_body:
        msg = MIMEMultipart("alternative")
        msg.attach(MIMEText(plain_body, "plain"))
        msg.attach(MIMEText(html_body, "html"))
    else:
        msg = MIMEMultipart()
        msg.attach(MIMEText(plain_body, "plain"))
    msg["From"] = from_header
    msg["To"] = recipient_email
    msg["Subject"] = subject
    if reply_to:
        msg["Reply-To"] = reply_to

    attempt = 0
    last_exc = None
    while attempt < max_retries:
        try:
            server = _connect_smtp()
            server.sendmail(SENDER_EMAIL, recipient_email, msg.as_string())
            server.quit()
            print(f"✅ Email sent to {recipient_email} | Subject: {subject}")
            return True
        except Exception as e:
            last_exc = e
            attempt += 1
            wait = retry_delay_sec * attempt
            print(f"⚠️ Send attempt {attempt} failed for {recipient_email}: {e}. Retrying in {wait}s...")
            time.sleep(wait)

    print(f"❌ All {max_retries} send attempts failed for {recipient_email}. Last error: {last_exc}")
    return False

def load_json_file(file_path: str):
    """Loads data from a JSON file."""
    with open(file_path, "r") as f:
        return json.load(f)

