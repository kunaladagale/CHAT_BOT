"""
tools.py
========
All external capabilities the agent can call live here, in one place.

Rules:
  1. The DOCSTRING is the routing logic — write it for the model.
  2. NEVER raise. Catch everything and return a plain-English error string.

Each tool is a thin @tool wrapper around a @traceable `_..._impl` function, so
LangSmith shows a nested tree. impl and wrapper are kept separate because
@traceable injects a `config=None` kwarg that would otherwise pollute the tool
schema the LLM sees.
"""

from __future__ import annotations

# --- standard library ---
import ast
import math
import operator
import os
import re
import smtplib
import time
import urllib.parse
from email.message import EmailMessage

# --- third party ---
import requests
import feedparser
from gtts import gTTS
from langchain_core.tools import tool
from langgraph.types import interrupt          # human-in-the-loop primitive

# --- local ---
import media                                    # audio hand-off to the UI

# --- LangSmith @traceable, with a no-op fallback (MUST be defined before use) ---
try:
    from langsmith import traceable
except ImportError:  # pragma: no cover
    def traceable(*d_args, **d_kwargs):          # type: ignore[misc]
        """No-op stand-in so the app runs without langsmith installed."""
        if len(d_args) == 1 and callable(d_args[0]) and not d_kwargs:
            return d_args[0]                      # bare @traceable
        return lambda fn: fn                      # @traceable(...)

from rag import RAG                              # RAG store for document/YouTube tools

HTTP_TIMEOUT = 15       # seconds -- never let a dead API hang the whole chat
AUDIO_DIR = "audio_out"


# ===========================================================================
# 1. CALCULATOR
# ===========================================================================
_BIN_OPS = {
    ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul,
    ast.Div: operator.truediv, ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod, ast.Pow: operator.pow,
}
_UNARY_OPS = {ast.UAdd: operator.pos, ast.USub: operator.neg}
_ALLOWED_NAMES = {
    "pi": math.pi, "e": math.e, "tau": math.tau, "inf": math.inf,
    "abs": abs, "round": round, "min": min, "max": max, "sum": sum, "pow": pow,
    "sqrt": math.sqrt,
    "cbrt": lambda x: math.copysign(abs(x) ** (1 / 3), x),
    "exp": math.exp, "log": math.log, "log2": math.log2, "log10": math.log10,
    "factorial": math.factorial, "gcd": math.gcd, "lcm": math.lcm,
    "floor": math.floor, "ceil": math.ceil, "trunc": math.trunc,
    "degrees": math.degrees, "radians": math.radians, "hypot": math.hypot,
    "sin": math.sin, "cos": math.cos, "tan": math.tan,
    "asin": math.asin, "acos": math.acos, "atan": math.atan, "atan2": math.atan2,
    "sinh": math.sinh, "cosh": math.cosh, "tanh": math.tanh,
}


def _safe_eval(node: ast.AST):
    if isinstance(node, ast.Expression):
        return _safe_eval(node.body)
    if isinstance(node, ast.Constant):
        if isinstance(node.value, (int, float)) and not isinstance(node.value, bool):
            return node.value
        raise ValueError(f"only numbers are allowed, got {node.value!r}")
    if isinstance(node, ast.BinOp):
        op = _BIN_OPS.get(type(node.op))
        if op is None:
            raise ValueError(f"operator {type(node.op).__name__} is not allowed")
        left, right = _safe_eval(node.left), _safe_eval(node.right)
        if isinstance(node.op, ast.Pow) and abs(right) > 1000:
            raise ValueError("exponent too large (limit is 1000)")
        return op(left, right)
    if isinstance(node, ast.UnaryOp):
        op = _UNARY_OPS.get(type(node.op))
        if op is None:
            raise ValueError(f"unary {type(node.op).__name__} is not allowed")
        return op(_safe_eval(node.operand))
    if isinstance(node, ast.Name):
        if node.id in _ALLOWED_NAMES:
            return _ALLOWED_NAMES[node.id]
        raise ValueError(f"unknown name '{node.id}'")
    if isinstance(node, ast.Call):
        if not isinstance(node.func, ast.Name):
            raise ValueError("only direct function calls are allowed")
        fn = _ALLOWED_NAMES.get(node.func.id)
        if fn is None or not callable(fn):
            raise ValueError(f"unknown function '{node.func.id}'")
        if node.keywords:
            raise ValueError("keyword arguments are not supported")
        return fn(*[_safe_eval(a) for a in node.args])
    if isinstance(node, (ast.List, ast.Tuple)):
        return [_safe_eval(el) for el in node.elts]
    if isinstance(node, ast.Compare):
        left = _safe_eval(node.left)
        for op_node, comparator in zip(node.ops, node.comparators):
            right = _safe_eval(comparator)
            cmp = {
                ast.Lt: operator.lt, ast.LtE: operator.le,
                ast.Gt: operator.gt, ast.GtE: operator.ge,
                ast.Eq: operator.eq, ast.NotEq: operator.ne,
            }.get(type(op_node))
            if cmp is None or not cmp(left, right):
                return False
            left = right
        return True
    raise ValueError(f"expression element {type(node).__name__} is not allowed")


@traceable(run_type="parser", name="safe_math_eval")
def _evaluate_expression(expr: str):
    return _safe_eval(ast.parse(expr, mode="eval"))


@traceable(run_type="tool", name="calculator")
def _calculator_impl(expression: str) -> str:
    expr = (expression or "").strip().rstrip("=").strip()
    if not expr:
        return "Error: empty expression."
    expr = expr.replace("^", "**").replace("×", "*").replace("÷", "/")
    expr = re.sub(r"(?<=\d),(?=\d{3}\b)", "", expr)
    try:
        result = _evaluate_expression(expr)
    except ZeroDivisionError:
        return "Error: division by zero."
    except (ValueError, SyntaxError, TypeError, OverflowError) as exc:
        return f"Error: could not evaluate '{expression}' -- {exc}"
    except Exception as exc:  # noqa: BLE001
        return f"Error: unexpected failure evaluating '{expression}' -- {exc}"
    if isinstance(result, float):
        if result.is_integer() and abs(result) < 1e15:
            return f"{expr} = {int(result)}"
        return f"{expr} = {result:.12g}"
    return f"{expr} = {result}"


@tool
def calculator(expression: str) -> str:
    """Evaluate a mathematical expression and return the exact numeric result.

    ALWAYS use this for arithmetic instead of computing in your head.

    Supported: + - * / // % ** parentheses, and sqrt cbrt exp log log2 log10
    factorial gcd lcm abs round floor ceil min max sum pow hypot degrees radians
    and trig (sin cos tan asin acos atan atan2 sinh cosh tanh). Constants: pi, e, tau.

    Args:
        expression: A pure math expression, e.g. "2 + 3 * 4", "sqrt(144)/3",
            "log(1000, 10)". No equals sign, units, or thousands commas.

    Returns:
        The result as a string, or a clear error message.
    """
    return _calculator_impl(expression)


# ===========================================================================
# 2. WEATHER  (Open-Meteo -- no API key)
# ===========================================================================
GEOCODE_URL = "https://geocoding-api.open-meteo.com/v1/search"
FORECAST_URL = "https://api.open-meteo.com/v1/forecast"
_WMO = {
    0: "Clear sky", 1: "Mainly clear", 2: "Partly cloudy", 3: "Overcast",
    45: "Fog", 48: "Depositing rime fog",
    51: "Light drizzle", 53: "Moderate drizzle", 55: "Dense drizzle",
    56: "Light freezing drizzle", 57: "Dense freezing drizzle",
    61: "Slight rain", 63: "Moderate rain", 65: "Heavy rain",
    66: "Light freezing rain", 67: "Heavy freezing rain",
    71: "Slight snowfall", 73: "Moderate snowfall", 75: "Heavy snowfall",
    77: "Snow grains",
    80: "Slight rain showers", 81: "Moderate rain showers", 82: "Violent rain showers",
    85: "Slight snow showers", 86: "Heavy snow showers",
    95: "Thunderstorm", 96: "Thunderstorm with slight hail",
    99: "Thunderstorm with heavy hail",
}


def _describe(code) -> str:
    try:
        return _WMO.get(int(code), f"Weather code {code}")
    except (TypeError, ValueError):
        return "Unknown conditions"


@traceable(run_type="retriever", name="geocode_location")
def _geocode(place: str) -> dict | None:
    candidates = [place]
    if "," in place:
        candidates.append(place.split(",")[0].strip())
    for candidate in candidates:
        resp = requests.get(
            GEOCODE_URL,
            params={"name": candidate, "count": 5, "language": "en", "format": "json"},
            timeout=HTTP_TIMEOUT,
        )
        resp.raise_for_status()
        results = resp.json().get("results") or []
        if not results:
            continue
        if "," in place:
            wanted = place.split(",", 1)[1].strip().lower()
            for r in results:
                if wanted in (r.get("country", "") or "").lower() or \
                   wanted == (r.get("country_code", "") or "").lower():
                    return r
        return results[0]
    return None


@traceable(run_type="retriever", name="open_meteo_forecast")
def _fetch_forecast(lat: float, lon: float, days: int) -> dict:
    params = {
        "latitude": lat, "longitude": lon,
        "current": ("temperature_2m,apparent_temperature,relative_humidity_2m,"
                    "precipitation,weather_code,wind_speed_10m"),
        "timezone": "auto",
    }
    if days > 0:
        params["daily"] = ("weather_code,temperature_2m_max,temperature_2m_min,"
                           "precipitation_sum,precipitation_probability_max")
        params["forecast_days"] = days
    resp = requests.get(FORECAST_URL, params=params, timeout=HTTP_TIMEOUT)
    resp.raise_for_status()
    return resp.json()


@traceable(run_type="tool", name="get_weather")
def _get_weather_impl(location: str, forecast_days: int = 0) -> str:
    place = (location or "").strip()
    if not place:
        return "Error: no location given. Ask the user which city they mean."
    try:
        days = max(0, min(int(forecast_days), 7))
    except (TypeError, ValueError):
        days = 0
    try:
        match = _geocode(place)
    except requests.RequestException as exc:
        return f"Error: could not reach the geocoding service -- {exc}"
    except ValueError:
        return "Error: the geocoding service returned an unreadable response."
    if match is None:
        return (f"Error: could not find a place called '{place}'. "
                f"Ask the user to check the spelling or add a country.")
    lat, lon = match["latitude"], match["longitude"]
    pretty = ", ".join(p for p in (match.get("name"), match.get("admin1"),
                                   match.get("country")) if p)
    try:
        data = _fetch_forecast(lat, lon, days)
    except requests.RequestException as exc:
        return f"Error: could not reach the weather service -- {exc}"
    except ValueError:
        return "Error: the weather service returned an unreadable response."
    cur = data.get("current", {}) or {}
    lines = [
        f"Weather for {pretty}  (local time {cur.get('time', 'n/a')}, "
        f"timezone {data.get('timezone', 'n/a')})",
        f"  Conditions : {_describe(cur.get('weather_code'))}",
        f"  Temperature: {cur.get('temperature_2m')} °C "
        f"(feels like {cur.get('apparent_temperature')} °C)",
        f"  Humidity   : {cur.get('relative_humidity_2m')} %",
        f"  Precip now : {cur.get('precipitation')} mm",
        f"  Wind       : {cur.get('wind_speed_10m')} km/h",
    ]
    daily = data.get("daily")
    if daily and daily.get("time"):
        lines.append("")
        lines.append("Daily forecast:")
        for i, date in enumerate(daily["time"]):
            lines.append(
                f"  {date}: {_describe(daily['weather_code'][i])}, "
                f"{daily['temperature_2m_min'][i]}–{daily['temperature_2m_max'][i]} °C, "
                f"precip {daily['precipitation_sum'][i]} mm "
                f"({daily['precipitation_probability_max'][i]}% chance)"
            )
    return "\n".join(lines)


@tool
def get_weather(location: str, forecast_days: int = 0) -> str:
    """Get the CURRENT weather, and optionally a daily forecast, for any place.

    Use for any weather/temperature/rain/humidity/wind question. Data is live.

    Args:
        location: Plain city/district name works best ("Pune", "London").
            Add a country to disambiguate ("Hyderabad, Pakistan"). No coordinates.
        forecast_days: 0 (default) for now, up to 7 for a week ahead.

    Returns:
        A readable weather report, or an error message.
    """
    return _get_weather_impl(location, forecast_days)


# ===========================================================================
# 3. STOCK PRICE  (yfinance -- no API key)
# ===========================================================================
_TICKER_ALIASES = {
    "apple": "AAPL", "microsoft": "MSFT", "google": "GOOGL", "alphabet": "GOOGL",
    "amazon": "AMZN", "meta": "META", "facebook": "META", "tesla": "TSLA",
    "nvidia": "NVDA", "netflix": "NFLX", "intel": "INTC", "amd": "AMD",
    "reliance": "RELIANCE.NS", "tcs": "TCS.NS", "infosys": "INFY.NS",
    "infy": "INFY.NS", "hdfc bank": "HDFCBANK.NS", "hdfcbank": "HDFCBANK.NS",
    "icici bank": "ICICIBANK.NS", "icicibank": "ICICIBANK.NS",
    "sbi": "SBIN.NS", "state bank of india": "SBIN.NS",
    "wipro": "WIPRO.NS", "itc": "ITC.NS", "tata motors": "TATAMOTORS.NS",
    "bitcoin": "BTC-USD", "ethereum": "ETH-USD",
    "nifty": "^NSEI", "nifty 50": "^NSEI", "sensex": "^BSESN",
    "s&p 500": "^GSPC", "sp500": "^GSPC", "nasdaq": "^IXIC", "dow jones": "^DJI",
}


@traceable(run_type="retriever", name="yfinance_history")
def _fetch_history(ticker_symbol: str):
    import yfinance as yf
    ticker = yf.Ticker(ticker_symbol)
    hist = ticker.history(period="2d", interval="1d")
    if hist is None or hist.empty:
        hist = ticker.history(period="5d", interval="1d")
    return ticker, hist


@traceable(run_type="tool", name="get_stock_price")
def _get_stock_price_impl(symbol: str) -> str:
    raw = (symbol or "").strip()
    if not raw:
        return "Error: no ticker symbol given."
    ticker_symbol = _TICKER_ALIASES.get(raw.lower(), raw.upper())
    try:
        import yfinance  # noqa: F401
    except ImportError:
        return "Error: the yfinance package is not installed. Run: pip install yfinance"
    try:
        ticker, hist = _fetch_history(ticker_symbol)
        if hist is None or hist.empty:
            return (f"Error: no market data found for '{ticker_symbol}'. "
                    f"Indian stocks need a .NS or .BO suffix (e.g. RELIANCE.NS).")
        last = hist.iloc[-1]
        price = float(last["Close"])
        day_high = float(last["High"])
        day_low = float(last["Low"])
        volume = int(last["Volume"]) if last["Volume"] == last["Volume"] else 0
        as_of = hist.index[-1].strftime("%Y-%m-%d")
        prev_close = float(hist.iloc[-2]["Close"]) if len(hist) > 1 else None
        currency = ""
        try:
            info = getattr(ticker, "fast_info", None)
            if info is not None:
                currency = (info.get("currency") if hasattr(info, "get")
                            else getattr(info, "currency", "")) or ""
        except Exception:  # noqa: BLE001
            pass
        cur = f" {currency.upper()}" if currency else ""
        lines = [f"{ticker_symbol} — as of {as_of}",
                 f"  Price        : {price:,.2f}{cur}"]
        if prev_close:
            change = price - prev_close
            pct = (change / prev_close) * 100
            arrow = "▲" if change >= 0 else "▼"
            lines.append(f"  Day change   : {arrow} {change:+,.2f} ({pct:+.2f}%)")
            lines.append(f"  Prev close   : {prev_close:,.2f}{cur}")
        lines.append(f"  Day range    : {day_low:,.2f} – {day_high:,.2f}{cur}")
        lines.append(f"  Volume       : {volume:,}")
        return "\n".join(lines)
    except Exception as exc:  # noqa: BLE001
        return (f"Error: could not fetch the price for '{ticker_symbol}' -- {exc}. "
                f"The symbol may be wrong, or Yahoo Finance may be unavailable.")


@tool
def get_stock_price(symbol: str) -> str:
    """Get the latest market price for a stock, index, ETF, or cryptocurrency.

    Data is live -- never answer a price question from memory.

    Args:
        symbol: Ticker like "AAPL". Indian stocks need ".NS"/".BO"
            (e.g. "RELIANCE.NS"). Crypto: "BTC-USD". Indices: "^NSEI", "^GSPC".
            A plain company name like "Apple" is also accepted.

    Returns:
        Price, day change, range and volume as text, or an error message.
    """
    return _get_stock_price_impl(symbol)


# ===========================================================================
# 4. SEND EMAIL  (human-in-the-loop: pauses for approval before sending)
# ===========================================================================
@traceable(run_type="tool", name="smtp_send")
def _smtp_send(to: str, subject: str, body: str) -> str:
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
    except Exception as exc:  # noqa: BLE001
        return f"Error: could not send the email -- {exc}"


@tool
def send_email(to: str, subject: str, body: str) -> str:
    """Send an email to a recipient.

    IMPORTANT: this does NOT send immediately — it pauses for human approval.
    Use whenever the user asks to send, write, or email someone.

    Args:
        to: Recipient email address.
        subject: Short subject line.
        body: The full email body in plain text.

    Returns:
        Confirmation the email was sent, or a note that it was rejected/failed.
    """
    decision = interrupt({
        "type": "email_approval",
        "to": to, "subject": subject, "body": body,
    })
    if isinstance(decision, dict):
        action = decision.get("action", "reject")
        to = decision.get("to", to)
        subject = decision.get("subject", subject)
        body = decision.get("body", body)
        feedback = decision.get("feedback", "")
    else:
        action = str(decision)
        feedback = ""
    if action == "approve":
        return _smtp_send(to, subject, body)
    if action == "revise":
        return (
            "The user did NOT approve the email. They requested these changes:\n"
            f"{feedback}\n\n"
            "Rewrite the email applying this feedback, then call send_email again "
            "with the improved subject and body. Write the body as clean PLAIN "
            "TEXT (short paragraphs, simple '-' bullets). No HTML unless asked."
        )
    return "The email was NOT sent — the user rejected the draft."


# ===========================================================================
# 5. DOCUMENT SEARCH  (Agentic RAG)
# ===========================================================================
@tool
def search_documents(query: str, source: str = "") -> str:
    """Search the user's UPLOADED documents / loaded YouTube transcripts and
    return relevant passages with citations.

    Use ONLY when the question is about content the user uploaded or a YouTube
    video they loaded. For general knowledge, answer normally.

    Args:
        query: What to look up.
        source: Optional exact source name to restrict the search.

    Returns:
        Relevant passages tagged with [S#] citations, or a note if nothing is
        found or nothing has been loaded.
    """
    return RAG.search(query, source or None)


# ===========================================================================
# 6. YOUTUBE TRANSCRIPT
# ===========================================================================
@tool
def add_youtube_video(url: str) -> str:
    """Load a YouTube video's transcript so it can be searched and questioned.
    Call whenever the user provides a YouTube link/URL.

    Args:
        url: A YouTube URL (youtube.com/watch?v=..., youtu.be/..., shorts, etc.).

    Returns:
        Confirmation the transcript was loaded, or an error message.
    """
    return RAG.ingest_youtube(url)


# ===========================================================================
# 7. TEXT TO SPEECH
# ===========================================================================
@traceable(run_type="tool", name="text_to_speech")
def _tts_impl(text: str) -> str:
    text = (text or "").strip()
    if not text:
        return "Error: there is no text to convert to audio."
    os.makedirs(AUDIO_DIR, exist_ok=True)
    path = os.path.join(AUDIO_DIR, f"speech_{int(time.time())}.mp3")
    try:
        gTTS(text=text, lang="en").save(path)      # needs internet, no API key
    except Exception as exc:  # noqa: BLE001
        return f"Error: text-to-speech failed -- {exc}"
    media.set_last_audio(path)
    return "Audio generated — shown to the user with a player and a download button."


@tool
def text_to_speech(text: str) -> str:
    """Convert text into spoken audio (MP3). Use whenever the user asks for the
    answer as audio/voice, or to listen to / hear the response.

    Args:
        text: The exact text to speak aloud — normally your full answer.

    Returns:
        Confirmation the audio was generated (a player + download appear in-app).
    """
    return _tts_impl(text)


# ===========================================================================
# 8. LATEST NEWS  (Google News RSS -- no API key)
# ===========================================================================
@traceable(run_type="tool", name="get_news")
def _news_impl(topic: str = "", limit: int = 8) -> str:
    if topic:
        url = ("https://news.google.com/rss/search?q="
               + urllib.parse.quote(topic) + "&hl=en-US&gl=US&ceid=US:en")
    else:
        url = "https://news.google.com/rss?hl=en-US&gl=US&ceid=US:en"
    try:
        feed = feedparser.parse(url)
    except Exception as exc:  # noqa: BLE001
        return f"Error: could not fetch news -- {exc}"
    entries = (feed.entries or [])[:limit]
    if not entries:
        return "No news headlines were found."
    lines = []
    for i, e in enumerate(entries, 1):
        title = e.get("title", "").strip()
        published = e.get("published", "")
        link = e.get("link", "")
        line = f"{i}. {title}"
        if published:
            line += f"  ({published})"
        if link:
            line += f"\n   {link}"
        lines.append(line)
    return "Latest headlines:\n" + "\n".join(lines)


@tool
def get_news(topic: str = "") -> str:
    """Fetch the latest top news headlines. Use for current news / latest events
    / 'what's happening', optionally on a topic.

    Args:
        topic: Optional keyword ("AI", "India economy"). Empty = top headlines.

    Returns:
        A numbered list of recent headlines with links, to summarize/answer from.
    """
    return _news_impl(topic)


# ===========================================================================
# Registry the graph imports.
# NOTE: send_email is intentionally NOT here — email is served by the MCP
# server (mcp_email_server.py) and loaded in langgraph_backend.py. The local
# send_email above is kept only as a reference/fallback and is not registered.
# ===========================================================================
TOOLS = [calculator, get_weather, get_stock_price,
         search_documents, add_youtube_video, text_to_speech, get_news]

TOOL_LABELS = {
    "calculator": "🧮 Calculator",
    "get_weather": "🌤️ Weather",
    "get_stock_price": "📈 Stock price",
    "send_email": "✉️ Send email",
    "search_documents": "📄 Document search",
    "add_youtube_video": "▶️ YouTube transcript",
    "text_to_speech": "🔊 Text to speech",
    "get_news": "📰 Latest news",
}


if __name__ == "__main__":
    print(calculator.invoke({"expression": "sqrt(144) * 3 + 2**10"}))