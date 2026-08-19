"""
tools.py
========
All external capabilities the agent can call live here, in one place.

Every function decorated with @tool becomes something the LLM can choose to
invoke. Two rules matter more than the code itself:

  1. The DOCSTRING is the routing logic. The model reads it to decide whether
     to call the tool and what to put in each argument. Write it for the model,
     not for a human reviewer.
  2. NEVER raise. A tool that raises kills the graph run. Always catch and
     return a plain-English error string -- the LLM can read that, apologise,
     or retry with different arguments.

LangSmith tracing
-----------------
Each tool is a thin @tool wrapper around a @traceable `_..._impl` function,
plus @traceable child spans on the expensive internal steps (geocoding, the
HTTP fetch, the math evaluation). So in LangSmith you get a nested tree and can
see exactly which step was slow or wrong instead of staring at one opaque box.

IMPORTANT — why impl and wrapper are separate functions:
@traceable injects a `config=None` keyword into the signature it exposes. If
you stack it directly under @tool, LangChain reads that signature and hands the
LLM a bogus `config` parameter to fill in. Keeping them apart gives you full
tracing AND a clean tool schema.

No API keys are needed for any of these:
  - calculator      -> pure local Python
  - get_weather     -> Open-Meteo (free, unlimited, worldwide)
  - get_stock_price -> yfinance / Yahoo Finance (free)
"""

from __future__ import annotations

import ast
import math
import operator
import re

import requests
from langchain_core.tools import tool
import os
import smtplib
from email.message import EmailMessage
from langgraph.types import interrupt   # the human-in-the-loop primitive


# Youtube
@tool
def add_youtube_video(url: str) -> str:
    """Load a YouTube video's transcript so its content can be searched and
    questioned. Call this whenever the user provides a YouTube link/URL.

    Args:
        url: A YouTube URL (youtube.com/watch?v=..., youtu.be/..., shorts, etc.).

    Returns:
        A confirmation that the transcript was loaded, or an error message.
    """
    return RAG.ingest_youtube(url)






# --- LangSmith @traceable, with a no-op fallback ---------------------------
# If langsmith isn't installed (or tracing is off) the app must still run.
try:
    from langsmith import traceable
except ImportError:  # pragma: no cover
    def traceable(*d_args, **d_kwargs):          # type: ignore[misc]
        """No-op stand-in so the app runs without langsmith installed."""
        if len(d_args) == 1 and callable(d_args[0]) and not d_kwargs:
            return d_args[0]                      # used as bare @traceable
        return lambda fn: fn                      # used as @traceable(...)


HTTP_TIMEOUT = 15  # seconds -- never let a dead API hang the whole chat

#RAG:-
from rag import RAG   # add with the other imports at the top
@tool
def search_documents(query: str, source: str = "") -> str:
    """Search the user's UPLOADED documents and return relevant passages with citations.

    Use this ONLY when the user's question is about the content of documents they
    have uploaded (their file, PDF, report, contract, notes, etc.). For general
    knowledge questions, do NOT use this tool — answer normally.

    Args:
        query: What to look up in the uploaded documents.
        source: Optional exact filename to restrict the search to one document.

    Returns:
        Relevant passages tagged with [S#] citations, or a note if nothing
        relevant is found or no documents have been uploaded.
    """
    return RAG.search(query, source or None)






# ===========================================================================
# 1. CALCULATOR
# ===========================================================================
# We do NOT use eval(). eval("__import__('os').system('rm -rf /')") is a real
# thing an LLM could be tricked into emitting. Instead we parse the expression
# into an AST and walk it, allowing only arithmetic nodes and a whitelist of
# math functions.

_BIN_OPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
}

_UNARY_OPS = {
    ast.UAdd: operator.pos,
    ast.USub: operator.neg,
}

# Whitelisted names. Anything not in here is rejected.
_ALLOWED_NAMES = {
    # constants
    "pi": math.pi,
    "e": math.e,
    "tau": math.tau,
    "inf": math.inf,
    # general
    "abs": abs,
    "round": round,
    "min": min,
    "max": max,
    "sum": sum,
    "pow": pow,
    # math module
    "sqrt": math.sqrt,
    "cbrt": lambda x: math.copysign(abs(x) ** (1 / 3), x),
    "exp": math.exp,
    "log": math.log,          # log(x) natural, log(x, base) too
    "log2": math.log2,
    "log10": math.log10,
    "factorial": math.factorial,
    "gcd": math.gcd,
    "lcm": math.lcm,
    "floor": math.floor,
    "ceil": math.ceil,
    "trunc": math.trunc,
    "degrees": math.degrees,
    "radians": math.radians,
    "hypot": math.hypot,
    # trig
    "sin": math.sin, "cos": math.cos, "tan": math.tan,
    "asin": math.asin, "acos": math.acos, "atan": math.atan, "atan2": math.atan2,
    "sinh": math.sinh, "cosh": math.cosh, "tanh": math.tanh,
}


def _safe_eval(node: ast.AST):
    """Recursively evaluate a whitelisted AST node."""
    if isinstance(node, ast.Expression):
        return _safe_eval(node.body)

    # numbers: 3, 3.5
    if isinstance(node, ast.Constant):
        if isinstance(node.value, (int, float)) and not isinstance(node.value, bool):
            return node.value
        raise ValueError(f"only numbers are allowed, got {node.value!r}")

    # 2 + 3, 2 ** 10, ...
    if isinstance(node, ast.BinOp):
        op = _BIN_OPS.get(type(node.op))
        if op is None:
            raise ValueError(f"operator {type(node.op).__name__} is not allowed")
        left, right = _safe_eval(node.left), _safe_eval(node.right)
        # guard against 2**99999999 freezing the process
        if isinstance(node.op, ast.Pow) and abs(right) > 1000:
            raise ValueError("exponent too large (limit is 1000)")
        return op(left, right)

    # -5, +5
    if isinstance(node, ast.UnaryOp):
        op = _UNARY_OPS.get(type(node.op))
        if op is None:
            raise ValueError(f"unary {type(node.op).__name__} is not allowed")
        return op(_safe_eval(node.operand))

    # pi, e, sqrt, ...
    if isinstance(node, ast.Name):
        if node.id in _ALLOWED_NAMES:
            return _ALLOWED_NAMES[node.id]
        raise ValueError(f"unknown name '{node.id}'")

    # sqrt(16), log(100, 10)
    if isinstance(node, ast.Call):
        if not isinstance(node.func, ast.Name):
            raise ValueError("only direct function calls are allowed")
        fn = _ALLOWED_NAMES.get(node.func.id)
        if fn is None or not callable(fn):
            raise ValueError(f"unknown function '{node.func.id}'")
        if node.keywords:
            raise ValueError("keyword arguments are not supported")
        return fn(*[_safe_eval(a) for a in node.args])

    # [1, 2, 3] -- so sum([1,2,3]) and max([4,9]) work
    if isinstance(node, (ast.List, ast.Tuple)):
        return [_safe_eval(el) for el in node.elts]

    # 1 < 2, 3 == 3
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
    """Parse + evaluate, as its own LangSmith span so you can see what the
    normalised expression actually was when a result looks wrong."""
    tree = ast.parse(expr, mode="eval")
    return _safe_eval(tree)


@traceable(run_type="tool", name="calculator")
def _calculator_impl(expression: str) -> str:
    """Real body of the calculator tool (traced as its own LangSmith span)."""
    expr = (expression or "").strip().rstrip("=").strip()
    if not expr:
        return "Error: empty expression."

    # tolerate a few things the model commonly emits
    expr = expr.replace("^", "**").replace("×", "*").replace("÷", "/")
    # strip THOUSANDS separators only (1,234,567 -> 1234567) so we don't
    # destroy real argument commas in calls like log(1000, 10)
    expr = re.sub(r"(?<=\d),(?=\d{3}\b)", "", expr)

    try:
        result = _evaluate_expression(expr)
    except ZeroDivisionError:
        return "Error: division by zero."
    except (ValueError, SyntaxError, TypeError, OverflowError) as exc:
        return f"Error: could not evaluate '{expression}' -- {exc}"
    except Exception as exc:  # noqa: BLE001 -- tools must never raise
        return f"Error: unexpected failure evaluating '{expression}' -- {exc}"

    # tidy float noise: 0.30000000000000004 -> 0.3
    # %.12g is used rather than round(x, 10) because rounding to 10 decimal
    # places destroys small magnitudes (1e-18 would print as 0). %g rounds to
    # 12 SIGNIFICANT digits instead, so tiny and huge values both survive.
    if isinstance(result, float):
        if result.is_integer() and abs(result) < 1e15:
            return f"{expr} = {int(result)}"
        return f"{expr} = {result:.12g}"
    return f"{expr} = {result}"


@tool
def calculator(expression: str) -> str:
    """Evaluate a mathematical expression and return the exact numeric result.

    ALWAYS use this for arithmetic instead of computing in your head -- language
    models make silent arithmetic mistakes, this tool does not.

    Supported: + - * / // % ** parentheses, and the functions
    sqrt cbrt exp log log2 log10 factorial gcd lcm abs round floor ceil
    min max sum pow hypot degrees radians and all trig functions
    (sin cos tan asin acos atan atan2 sinh cosh tanh).
    Constants: pi, e, tau.

    Args:
        expression: A pure math expression, e.g. "2 + 3 * 4",
            "sqrt(144) / 3", "(1500 * 1.18) - 200", "factorial(10)",
            "log(1000, 10)", "sin(radians(30))".
            Do NOT include an equals sign, variable assignments, units,
            currency symbols, or comma separators in numbers
            (write 1234567, not 1,234,567).

    Returns:
        The result as a string, or a clear error message if the expression
        is invalid.
    """
    return _calculator_impl(expression)


# ===========================================================================
# 2. WEATHER  (Open-Meteo -- no API key required)
# ===========================================================================

GEOCODE_URL = "https://geocoding-api.open-meteo.com/v1/search"
FORECAST_URL = "https://api.open-meteo.com/v1/forecast"

# Open-Meteo returns a WMO weather code; translate it to something readable.
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
    """Resolve a place name to coordinates. Own LangSmith span -- this is where
    'wrong city' bugs come from, so you want to see its input and output."""
    # Open-Meteo's geocoder chokes on "City, Country", so try the full string
    # first and fall back to just the part before the comma.
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

        # if the user wrote "X, Country", prefer the result in that country
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
    """Hit the Open-Meteo forecast endpoint. Own span so you can see latency."""
    params = {
        "latitude": lat,
        "longitude": lon,
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
    """Real body of the weather tool (traced as its own LangSmith span)."""
    place = (location or "").strip()
    if not place:
        return "Error: no location given. Ask the user which city or district they mean."

    try:
        days = max(0, min(int(forecast_days), 7))
    except (TypeError, ValueError):
        days = 0

    # --- step 1: turn the place name into coordinates -----------------------
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
    pretty = ", ".join(
        p for p in (match.get("name"), match.get("admin1"), match.get("country")) if p
    )

    # --- step 2: fetch the weather -----------------------------------------
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
    """Get the CURRENT weather, and optionally a daily forecast, for any place
    on Earth -- city, town, district, or country.

    Use this whenever the user asks about weather, temperature, rain, humidity,
    or wind for a named place. The data is live, so use it even for places you
    think you know.

    Args:
        location: Place name. A plain city or district name works best, e.g.
            "Pune", "Nashik", "London", "Tokyo". Add a country to disambiguate
            when a name is common, e.g. "Springfield, United States" or
            "Hyderabad, Pakistan". Do NOT pass coordinates.
        forecast_days: How many days of daily forecast to include, 0 to 7.
            Use 0 (default) for "what's the weather right now".
            Use 1 for today's high/low, 2 for today and tomorrow,
            7 for a week ahead.

    Returns:
        A readable weather report, or an error message if the place could not
        be found or the service is unreachable.
    """
    return _get_weather_impl(location, forecast_days)


# ===========================================================================
# 3. STOCK PRICE  (yfinance -- no API key required)
# ===========================================================================

# Friendly names the model is likely to receive -> real ticker symbols.
# Indian stocks need the .NS (NSE) or .BO (BSE) suffix on Yahoo Finance.
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
    """Pull recent daily bars from Yahoo Finance as its own LangSmith span."""
    import yfinance as yf

    ticker = yf.Ticker(ticker_symbol)
    # 2 days of history gives us today's price AND the previous close,
    # which is what we need to compute the day change.
    hist = ticker.history(period="2d", interval="1d")
    if hist is None or hist.empty:
        hist = ticker.history(period="5d", interval="1d")
    return ticker, hist


@traceable(run_type="tool", name="get_stock_price")
def _get_stock_price_impl(symbol: str) -> str:
    """Real body of the stock tool (traced as its own LangSmith span)."""
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
                    f"Check the symbol -- Indian stocks need a .NS or .BO suffix "
                    f"(e.g. RELIANCE.NS).")

        last = hist.iloc[-1]
        price = float(last["Close"])
        day_high = float(last["High"])
        day_low = float(last["Low"])
        volume = int(last["Volume"]) if last["Volume"] == last["Volume"] else 0
        as_of = hist.index[-1].strftime("%Y-%m-%d")

        prev_close = float(hist.iloc[-2]["Close"]) if len(hist) > 1 else None

        # currency is nice-to-have; never let it break the call
        currency = ""
        try:
            info = getattr(ticker, "fast_info", None)
            if info is not None:
                currency = (info.get("currency") if hasattr(info, "get")
                            else getattr(info, "currency", "")) or ""
        except Exception:  # noqa: BLE001
            pass

        cur = f" {currency.upper()}" if currency else ""
        lines = [
            f"{ticker_symbol} — as of {as_of}",
            f"  Price        : {price:,.2f}{cur}",
        ]
        if prev_close:
            change = price - prev_close
            pct = (change / prev_close) * 100
            arrow = "▲" if change >= 0 else "▼"
            lines.append(f"  Day change   : {arrow} {change:+,.2f} ({pct:+.2f}%)")
            lines.append(f"  Prev close   : {prev_close:,.2f}{cur}")
        lines.append(f"  Day range    : {day_low:,.2f} – {day_high:,.2f}{cur}")
        lines.append(f"  Volume       : {volume:,}")
        return "\n".join(lines)

    except Exception as exc:  # noqa: BLE001 -- tools must never raise
        return (f"Error: could not fetch the price for '{ticker_symbol}' -- {exc}. "
                f"The symbol may be wrong, or Yahoo Finance may be temporarily "
                f"unavailable.")


@tool
def get_stock_price(symbol: str) -> str:
    """Get the latest market price for a stock, index, ETF, or cryptocurrency.

    Use this for any question about a share price, how a stock is doing today,
    or the level of a market index. The data is live -- never answer a price
    question from memory.

    Args:
        symbol: The ticker symbol, e.g. "AAPL", "MSFT", "TSLA".
            For Indian stocks add the exchange suffix: ".NS" for NSE
            (e.g. "RELIANCE.NS", "TCS.NS", "INFY.NS") or ".BO" for BSE.
            For crypto use the pair form, e.g. "BTC-USD".
            For indices: "^NSEI" (Nifty 50), "^BSESN" (Sensex),
            "^GSPC" (S&P 500), "^IXIC" (Nasdaq).
            A plain company name like "Apple" or "Reliance" is also accepted
            and will be resolved automatically.

    Returns:
        The current price, day change, day range and volume as text, or an
        error message if the symbol is unknown or the market data is
        unavailable.
    """
    return _get_stock_price_impl(symbol)


# ===========================================================================
# 4. SEND EMAIL  (human-in-the-loop: pauses for approval before sending)
# ===========================================================================

@traceable(run_type="tool", name="smtp_send")
def _smtp_send(to: str, subject: str, body: str) -> str:
    """Actually deliver the email via SMTP (own LangSmith span).
    Reads credentials from .env. If they're missing it SIMULATES the send so
    you can test the whole approval flow without configuring SMTP."""
    host = os.getenv("SMTP_HOST")
    user = os.getenv("SMTP_USER")
    pwd  = os.getenv("SMTP_PASSWORD")
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
    except Exception as exc:  # noqa: BLE001 -- tools must never raise
        return f"Error: could not send the email -- {exc}"


@tool
def send_email(to: str, subject: str, body: str) -> str:
    """Send an email to a recipient.

    IMPORTANT: this does NOT send immediately. It PAUSES and shows the drafted
    email to the human, who must approve it before it is delivered. Use this
    whenever the user asks to send, write, or email someone.

    Args:
        to: Recipient email address, e.g. "someone@example.com".
        subject: Short subject line.
        body: The full email body in plain text.

    Returns:
        Confirmation the email was sent, or a note that it was rejected/failed.
    """
    # --- HUMAN IN THE LOOP -------------------------------------------------
    # interrupt() pauses the ENTIRE graph here and hands this draft to the UI.
    # Execution resumes only when the app calls Command(resume=<decision>).
    decision = interrupt({
        "type": "email_approval",
        "to": to,
        "subject": subject,
        "body": body,
    })

    # `decision` is whatever the app passed to Command(resume=...). We accept an
    # edited draft too, so the user can tweak the email before approving.
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
            "with the improved subject and body. Write the body as clean, "
            "well-formatted PLAIN TEXT (short paragraphs, simple '-' bullets). "
            "Do NOT use HTML tags unless the user explicitly asked for HTML."
        )

    return "The email was NOT sent — the user rejected the draft."

# ===========================================================================
# The registry the graph imports. Add a new @tool above, list it here, done.
# ===========================================================================
TOOLS = [calculator, get_weather, get_stock_price, search_documents, add_youtube_video]

TOOL_LABELS = {
    "calculator": "🧮 Calculator",
    "get_weather": "🌤️ Weather",
    "get_stock_price": "📈 Stock price",
    "send_email": "✉️ Send email",
    "search_documents": "📄 Document search",
    "add_youtube_video": "▶️ YouTube transcript",
}

if __name__ == "__main__":
    # quick smoke test:  python tools.py
    print(calculator.invoke({"expression": "sqrt(144) * 3 + 2**10"}))
    print()
    print(get_weather.invoke({"location": "Pune", "forecast_days": 2}))
    print()
    print(get_stock_price.invoke({"symbol": "RELIANCE.NS"}))