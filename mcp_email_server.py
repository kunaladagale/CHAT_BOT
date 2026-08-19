"""
mcp_email_server.py
===================
A tiny MCP (Model Context Protocol) server that exposes ONE tool: send_email.

This runs as a SEPARATE program. Your LangGraph app connects to it over MCP
(stdio) and calls its `send_email` tool remotely — that's the difference from a
local @tool: the email code lives here, in its own process, not inside the app.

Run it directly to sanity-check it:   python mcp_email_server.py
(it will just sit waiting for an MCP client — that's correct.)

Requires:  pip install mcp
"""

import os
import smtplib
from email.message import EmailMessage

from dotenv import load_dotenv
load_dotenv()   # so SMTP_* credentials are read from your .env

from mcp.server.fastmcp import FastMCP

# The server's name. The tools it exposes are discovered by the client.
mcp = FastMCP("email")


def _smtp_send(to: str, subject: str, body: str) -> str:
    """Deliver the email via SMTP. If SMTP_* creds are missing, SIMULATE the
    send so the whole flow can be tested without real credentials."""
    host = os.getenv("SMTP_HOST")
    user = os.getenv("SMTP_USER")
    pwd = os.getenv("SMTP_PASSWORD")
    port = int(os.getenv("SMTP_PORT", "587"))
    sender = os.getenv("SMTP_FROM", user or "")

    if not (host and user and pwd):
        return (f"[SIMULATED SEND] No SMTP_* credentials in .env, so nothing was "
                f"actually emailed. Would have sent to {to} | subject: {subject!r}.")
    try:
        msg = EmailMessage()
        msg["From"] = sender
        msg["To"] = to
        msg["Subject"] = subject
        msg.set_content(body)
        with smtplib.SMTP(host, port, timeout=20) as smtp:
            smtp.starttls()
            smtp.login(user, pwd)
            smtp.send_message(msg)
        return f"Email successfully sent to {to}."
    except Exception as exc:  # never crash the MCP server
        return f"Error: could not send the email -- {exc}"


@mcp.tool()
def send_email(to: str, subject: str, body: str) -> str:
    """Send an email to a recipient.

    Args:
        to: Recipient email address, e.g. "someone@example.com".
        subject: Short subject line.
        body: The full email body in plain text.

    Returns:
        Confirmation that the email was sent, or an error message.
    """
    return _smtp_send(to, subject, body)


if __name__ == "__main__":
    # stdio transport: the LangGraph app launches this program and talks to it
    # over standard input/output. This is the simplest MCP transport.
    mcp.run(transport="stdio")