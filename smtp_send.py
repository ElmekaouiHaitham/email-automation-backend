import resend
import os
from dotenv import load_dotenv

def send_email_resend(to, subject, html_body, from_email=None):
    
    load_dotenv()
    resend.api_key = "re_FaWeHWQ8_3FLvmVpRX2DFasHmhrnHXpd5"
    from_email = from_email or "Haitham <onboarding@resend.dev>"
    params = {
        "from": from_email,
        "to": [to],
        "subject": subject,
        "html": html_body
    }
    email = resend.Emails.send(params)
    print("✅ Email sent:", email)
    return email.status == "sent"
