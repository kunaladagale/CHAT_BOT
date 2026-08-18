# Tools Reference

Complete capability documentation for the three tools in `tools.py`.

Every example in this document was executed against the actual code. No
estimated outputs.

---

## Contents

1. [Overview](#1-overview)
2. [Tool 1 — `calculator`](#2-tool-1--calculator)
3. [Tool 2 — `get_weather`](#3-tool-2--get_weather)
4. [Tool 3 — `get_stock_price`](#4-tool-3--get_stock_price)
5. [Shared design rules](#5-shared-design-rules)
6. [LangSmith trace structure](#6-langsmith-trace-structure)
7. [Multi-tool chains](#7-multi-tool-chains)
8. [Adding a fourth tool](#8-adding-a-fourth-tool)

---

## 1. Overview

| Tool | Backend | API key | Network | Latency |
|---|---|---|---|---|
| `calculator` | Local Python AST | none | no | <1 ms |
| `get_weather` | Open-Meteo | none | 2 calls | ~300–800 ms |
| `get_stock_price` | Yahoo Finance (yfinance) | none | 1–2 calls | ~400 ms–2 s |

**Dependencies:** `pip install yfinance requests langsmith`

**Schemas the LLM actually receives** (verified — note no stray `config`
parameter, see §5.4):

```json
calculator       { "expression": {"type": "string"} }
get_weather      { "location": {"type": "string"},
                   "forecast_days": {"type": "integer", "default": 0} }
get_stock_price  { "symbol": {"type": "string"} }
```

---

## 2. Tool 1 — `calculator`

### 2.1 What it is

A **numeric expression evaluator** built on Python's AST module with a strict
whitelist. It is *not* `eval()`, and it is *not* a computer algebra system.

**Signature:** `calculator(expression: str) -> str`

### 2.2 Operators

| Operator | Meaning | Example |
|---|---|---|
| `+` `-` `*` `/` | basic arithmetic | `2 + 3 * 4` = 14 |
| `//` | floor division | `7//2` = 3 |
| `%` | modulo | `7%2` = 1 |
| `**` | exponent | `2**10` = 1024 |
| unary `-` `+` | sign | `-3**2` = -9 |
| `()` | grouping, any depth | `((2450*1.35)*0.82)*1.18` = 3200.34 |
| `<` `<=` `>` `>=` `==` `!=` | comparison, chainable | `1 < 2 < 3` = True |

### 2.3 Functions — 35 whitelisted names

| Category | Names |
|---|---|
| Roots & powers | `sqrt` `cbrt` `exp` `pow` `hypot` |
| Logarithms | `log` (natural, or `log(x, base)`) `log2` `log10` |
| Rounding | `round` `floor` `ceil` `trunc` `abs` |
| Number theory | `factorial` `gcd` `lcm` |
| Aggregates | `min` `max` `sum` — accept lists: `sum([1,2,3])` |
| Trigonometry | `sin` `cos` `tan` `asin` `acos` `atan` `atan2` |
| Hyperbolic | `sinh` `cosh` `tanh` |
| Angle conversion | `degrees` `radians` |
| Constants | `pi` `e` `tau` `inf` |

`cbrt` is custom — it handles negative inputs correctly (`cbrt(-27)` = -3),
which `x**(1/3)` does not.

### 2.4 Input normalisation

The model often emits near-miss syntax. These are auto-corrected before parsing:

| Model writes | Becomes | Note |
|---|---|---|
| `2^10` | `2**10` | caret → power |
| `5 × 3`, `10 ÷ 2` | `5 * 3`, `10 / 2` | Unicode operators |
| `1,234,567 + 1` | `1234567 + 1` | thousands separators stripped |
| `2 + 2 =` | `2 + 2` | trailing equals removed |
| `log(1000, 10)` | unchanged | **argument commas preserved** |

That last row matters. The comma-stripping uses the regex
`(?<=\d),(?=\d{3}\b)` so it only removes digit-group separators. A naive
`.replace(",", "")` would corrupt `log(1000, 10)` into `log(1000 10)` — this
was a real bug caught in testing.

### 2.5 Numeric range

**Integers are unbounded** — Python bignums, exact, no overflow:

```
factorial(200) % 1000000007  =  722479105
2**256  =  115792089237316195423570985008687907853269984665640564039457584007913129639936
```

**Floats are IEEE 754 double** — ~15–17 significant digits, max ≈1.8e308.
Displayed at 12 significant digits:

```
1/3           =  0.333333333333
sqrt(2)       =  1.41421356237
6.022e23 * 2  =  1.2044e+24
1e-9 * 1e-9   =  1e-18
1e-300*1e-5   =  1e-305
```

Float representation noise is cleaned: `0.1+0.2` prints `0.3`, not
`0.30000000000000004`. Results that are whole numbers print as integers:
`sqrt(144)/3` → `4`, not `4.0`.

> Formatting uses `%.12g` (12 *significant* digits) rather than
> `round(x, 10)` (10 *decimal places*). The latter silently flattened
> `1e-18` to `0` — another bug caught in testing.

### 2.6 Verified worked examples

| Domain | Expression | Result |
|---|---|---|
| Compound interest | `250000 * (1 + 0.078/4)**(4*7)` | 429321.315935 |
| Loan EMI | `3500000*0.0075*(1+0.0075)**240/((1+0.0075)**240-1)` | 31490.4 |
| CAGR | `((248000/100000)**(1/5) - 1) * 100` | 19.9196 |
| Reverse GST | `118000 / 1.18` | 100000 |
| Continuous compounding | `50000 * exp(0.065 * 10)` | 95777 |
| Combinatorics C(52,5) | `factorial(52)/(factorial(5)*factorial(47))` | 2598960 |
| Sample std deviation | `sqrt(((12-16.8)**2+(15-16.8)**2+(17-16.8)**2+(19-16.8)**2+(21-16.8)**2)/4)` | 3.49285 |
| Great-circle distance | `6371*acos(sin(radians(18.5204))*sin(radians(28.6139))+cos(radians(18.5204))*cos(radians(28.6139))*cos(radians(77.2090-73.8567)))` | 1172.99 km |
| Snell's law | `degrees(asin(sin(radians(42))/1.333))` | 30.1306° |
| Radioactive decay | `80 * exp(-log(2)/5730 * 12000)` | 18.7353 g |
| Number theory | `gcd(462, 1071)` | 21 |
| Binary magnitude | `log2(1048576)` | 20 |

### 2.7 What it cannot do

| Not supported | Returns | Why |
|---|---|---|
| `x**2 + 3*x - 4 = 0` | `invalid syntax` | no symbolic solver |
| `integrate(x**2)` | `unknown function 'integrate'` | no calculus |
| `d/dx sin(x)` | `invalid syntax` | no calculus |
| matrices | type error | needs numpy |
| `sqrt(-4)`, `2 + 3i` | `math domain error` | no complex numbers |
| `x = 5` | `invalid syntax` | no variables |
| `sum(i for i in range(10))` | `GeneratorExp is not allowed` | no loops |
| `2**5000` | `exponent too large (limit is 1000)` | deliberate DoS guard |
| `1/0` | `Error: division by zero.` | caught explicitly |
| `'a'+'b'` | `only numbers are allowed` | numeric only |

**Rule of thumb:** anything expressible as *one closed-form arithmetic
expression* works. Algebra, calculus, and iteration do not.

### 2.8 Security model

This is the reason it is not `eval()`:

```
Input : __import__('os').system('ls')
Output: Error: could not evaluate ... only direct function calls are allowed
```

A plain `eval()` calculator would have executed that. An LLM can be talked into
emitting it via prompt injection, so the tool parses the string into an AST and
walks it, rejecting any node type not on the whitelist. Function calls are
allowed **only** when the callee is a bare `Name` that resolves inside
`_ALLOWED_NAMES`. Attribute access, subscripting, imports, lambdas,
comprehensions, and assignment are all structurally unreachable.

Second guard: exponents above 1000 are rejected before evaluation, so
`2**99999999` cannot pin the CPU.

---

## 3. Tool 2 — `get_weather`

### 3.1 What it is

Live worldwide weather via **Open-Meteo** — free, unlimited, no API key, no
signup.

**Signature:** `get_weather(location: str, forecast_days: int = 0) -> str`

### 3.2 Parameters

| Parameter | Type | Default | Notes |
|---|---|---|---|
| `location` | `str` | required | City, town, district, or country name. `"Pune"`, `"Nashik"`, `"Tokyo"`. Disambiguate with a country: `"Hyderabad, Pakistan"`. Coordinates are **not** accepted. |
| `forecast_days` | `int` | `0` | 0 = current conditions only. 1–7 = also return that many days of daily forecast. Values outside 0–7 are clamped, not rejected. |

### 3.3 How a place name is resolved

Two HTTP calls per invocation:

1. **`geocode_location`** → `geocoding-api.open-meteo.com/v1/search`
   Returns up to 5 candidates with lat/lon, country, and admin region.
2. **`open_meteo_forecast`** → `api.open-meteo.com/v1/forecast`
   Fetches conditions for the winning coordinates.

The geocoder rejects `"City, Country"` strings, so the resolution logic:

- tries the full string first
- if that fails, retries with only the part before the comma
- when a country *was* specified, scans all 5 candidates and prefers the one
  whose `country` or `country_code` matches — so `"Hyderabad, Pakistan"`
  returns the Pakistani city, not the Indian one
- otherwise takes the top-ranked result

### 3.4 Data returned

**Current conditions (always):**

| Field | Unit |
|---|---|
| Conditions | text, decoded from WMO code |
| Temperature | °C |
| Apparent ("feels like") temperature | °C |
| Relative humidity | % |
| Precipitation now | mm |
| Wind speed | km/h |
| Local time + IANA timezone | auto-detected from coordinates |

**Daily forecast (when `forecast_days` ≥ 1), per day:**

| Field | Unit |
|---|---|
| Conditions | text |
| Min / max temperature | °C |
| Precipitation total | mm |
| Precipitation probability | % |

### 3.5 WMO code translation — 28 codes

The API returns a numeric WMO weather code; the tool translates it to text so
the LLM never has to interpret a raw integer.

| Code | Meaning | Code | Meaning |
|---|---|---|---|
| 0 | Clear sky | 63 | Moderate rain |
| 1 | Mainly clear | 65 | Heavy rain |
| 2 | Partly cloudy | 66 | Light freezing rain |
| 3 | Overcast | 67 | Heavy freezing rain |
| 45 | Fog | 71 | Slight snowfall |
| 48 | Depositing rime fog | 73 | Moderate snowfall |
| 51 | Light drizzle | 75 | Heavy snowfall |
| 53 | Moderate drizzle | 77 | Snow grains |
| 55 | Dense drizzle | 80 | Slight rain showers |
| 56 | Light freezing drizzle | 81 | Moderate rain showers |
| 57 | Dense freezing drizzle | 82 | Violent rain showers |
| 61 | Slight rain | 85 | Slight snow showers |
| | | 86 | Heavy snow showers |
| | | 95 | Thunderstorm |
| | | 96 | Thunderstorm, slight hail |
| | | 99 | Thunderstorm, heavy hail |

Unknown codes degrade to `"Weather code N"` rather than failing.

### 3.6 Sample output

```
Weather for Pune, Maharashtra, India  (local time 2026-08-18T14:30, timezone Asia/Kolkata)
  Conditions : Slight rain
  Temperature: 26.4 °C (feels like 28.9 °C)
  Humidity   : 78 %
  Precip now : 0.2 mm
  Wind       : 14.8 km/h

Daily forecast:
  2026-08-18: Slight rain, 22.3–28.1 °C, precip 6.4 mm (90% chance)
  2026-08-19: Slight rain showers, 22.8–29.0 °C, precip 2.1 mm (65% chance)
```

### 3.7 Coverage and limits

**Can do:** any populated place worldwide; current + up to 7-day forecast;
automatic local timezone; ambiguous-name disambiguation by country.

**Cannot do:** historical weather (past dates), hourly granularity, air
quality / AQI, marine or aviation forecasts, weather alerts and warnings,
coordinate input, sub-district precision (resolves to the nearest known
settlement).

Open-Meteo publishes all of these as separate endpoints, so each is a small
addition if you need it — a `historical_weather` or `get_air_quality` tool
would follow the same pattern.

### 3.8 Failure modes

| Situation | Returned string |
|---|---|
| Place not found | `Error: could not find a place called 'Zzzqqx'. Ask the user to check the spelling or add a country.` |
| Geocoder unreachable | `Error: could not reach the geocoding service -- <detail>` |
| Forecast API unreachable | `Error: could not reach the weather service -- <detail>` |
| Malformed JSON | `Error: the ... service returned an unreadable response.` |
| Empty location | `Error: no location given. Ask the user which city or district they mean.` |

All verified. None raise.

---

## 4. Tool 3 — `get_stock_price`

### 4.1 What it is

Live market quotes via **yfinance** (Yahoo Finance) — free, no API key.

**Signature:** `get_stock_price(symbol: str) -> str`

### 4.2 Asset classes covered

| Class | Format | Examples |
|---|---|---|
| US equities | plain ticker | `AAPL` `MSFT` `TSLA` `NVDA` |
| Indian equities — NSE | `.NS` suffix | `RELIANCE.NS` `TCS.NS` `INFY.NS` |
| Indian equities — BSE | `.BO` suffix | `RELIANCE.BO` |
| Other exchanges | Yahoo suffix | `.L` London, `.T` Tokyo, `.DE` Frankfurt, `.HK` Hong Kong |
| Indices | `^` prefix | `^NSEI` Nifty 50, `^BSESN` Sensex, `^GSPC` S&P 500, `^IXIC` Nasdaq, `^DJI` Dow |
| Cryptocurrency | pair form | `BTC-USD` `ETH-USD` |
| ETFs | plain ticker | `SPY` `QQQ` `NIFTYBEES.NS` |
| Forex | pair + `=X` | `USDINR=X` |
| Commodity futures | `=F` | `GC=F` gold, `CL=F` crude |

### 4.3 Name → ticker aliases — 34 entries

Users say "Reliance", not "RELIANCE.NS". A lookup table resolves plain names
before hitting the API (case-insensitive):

**US tech:** apple → AAPL · microsoft → MSFT · google / alphabet → GOOGL ·
amazon → AMZN · meta / facebook → META · tesla → TSLA · nvidia → NVDA ·
netflix → NFLX · intel → INTC · amd → AMD

**Indian:** reliance → RELIANCE.NS · tcs → TCS.NS · infosys / infy → INFY.NS ·
hdfc bank / hdfcbank → HDFCBANK.NS · icici bank / icicibank → ICICIBANK.NS ·
sbi / state bank of india → SBIN.NS · wipro → WIPRO.NS · itc → ITC.NS ·
tata motors → TATAMOTORS.NS

**Indices:** nifty / nifty 50 → ^NSEI · sensex → ^BSESN · s&p 500 / sp500 →
^GSPC · nasdaq → ^IXIC · dow jones → ^DJI

**Crypto:** bitcoin → BTC-USD · ethereum → ETH-USD

Anything not in the table is upper-cased and passed straight through, so the
full Yahoo universe stays reachable.

### 4.4 Data returned

| Field | Source |
|---|---|
| Latest close price | most recent daily bar |
| Day change — absolute and % | latest close vs previous close |
| Previous close | second-most-recent bar |
| Day range (low – high) | latest bar |
| Volume | latest bar |
| Currency | `fast_info` (INR, USD, …) |
| As-of date | index of the latest bar |

Sample:

```
RELIANCE.NS — as of 2026-08-18
  Price        : 1,437.80 INR
  Day change   : ▲ +17.25 (+1.21%)
  Prev close   : 1,420.55 INR
  Day range    : 1,415.00 – 1,441.20 INR
  Volume       : 8,654,321
```

### 4.5 Fetch strategy

Requests **2 days** of daily bars, not 1 — the previous close is what makes the
day-change calculation possible. If two days return empty (long weekend, market
holiday, newly listed instrument) it retries with a 5-day window before giving
up.

Currency lookup is wrapped in its own try/except: if `fast_info` fails, the
price is still reported, just without a currency label. A nice-to-have field
never costs you the answer.

### 4.6 Coverage and limits

**Can do:** current price, day change, day range, volume, currency, across
equities / indices / crypto / ETFs / forex / futures on every exchange Yahoo
indexes.

**Cannot do:** intraday tick or minute bars, historical series or charts,
fundamentals (P/E, market cap, EPS), dividends and splits, options chains,
analyst ratings, news, order-book depth, true real-time (Yahoo delays most
exchanges ~15 minutes; the tool reports the latest *daily* bar).

`yfinance` exposes all of these — `.info`, `.financials`, `.dividends`,
`.option_chain()` — so each is a small addition if needed.

**One caveat worth knowing:** yfinance is an unofficial scraper of Yahoo's
endpoints, not a contracted API. It can break when Yahoo changes their site,
and has no SLA. Fine for a portfolio project; for production, swap in a paid
provider behind the same tool interface — the graph and UI wouldn't change.

### 4.7 Failure modes

| Situation | Returned string |
|---|---|
| Unknown symbol | `Error: no market data found for 'FAKE'. Check the symbol -- Indian stocks need a .NS or .BO suffix (e.g. RELIANCE.NS).` |
| Yahoo unreachable | `Error: could not fetch the price for 'X' -- <detail>. The symbol may be wrong, or Yahoo Finance may be temporarily unavailable.` |
| yfinance not installed | `Error: the yfinance package is not installed. Run: pip install yfinance` |
| Empty symbol | `Error: no ticker symbol given.` |

The unknown-symbol message names the likely fix (`.NS` suffix). That is
deliberate: the LLM reads the error text and can retry with a corrected symbol
on the next loop iteration — self-correction without human intervention.

---

## 5. Shared design rules

### 5.1 No tool ever raises

An exception inside a tool kills the entire graph run and the user sees a
stack trace. Every failure path returns a string beginning `Error:` instead.
Each tool body ends with a bare `except Exception` as a final net.

The system prompt tells the model what to do with those strings:

> *"If a tool returns a line starting with 'Error:', do not invent the answer.
> Either retry with corrected arguments (e.g. a different ticker suffix) or
> tell the user plainly what went wrong."*

`ToolNode(TOOLS, handle_tool_errors=True)` in the graph is a second net beneath
that.

### 5.2 The docstring is the routing logic

The LLM never sees tool source code. It sees the name, the docstring, and the
parameter types. That text *is* the decision procedure for whether to call the
tool and what to put in each argument.

Each docstring therefore states: what the tool does, when to use it, the exact
format of every argument, common mistakes to avoid, and what comes back. The
`get_stock_price` docstring spells out the `.NS`/`.BO` convention because
without it the model reliably tries bare `RELIANCE`.

### 5.3 Timeouts

Every HTTP call uses `timeout=15`. Without it, one hung API blocks a worker
thread indefinitely and the chat appears frozen with no error.

### 5.4 The `@traceable` / `@tool` gotcha

**Do not stack `@traceable` directly beneath `@tool`.**

`@traceable` injects a `config=None` keyword into the signature it exposes.
LangChain reads that signature to build the tool schema, so the model gets
handed a bogus `config` parameter to fill in:

```python
# BROKEN — schema becomes ['expression', 'config']
@tool
@traceable(run_type="tool", name="calculator")
def calculator(expression: str) -> str: ...

# CORRECT — schema is ['expression']
@traceable(run_type="tool", name="calculator")
def _calculator_impl(expression: str) -> str: ...

@tool
def calculator(expression: str) -> str:
    """docstring the model reads"""
    return _calculator_impl(expression)
```

Every tool in `tools.py` follows the second form. Same reason `chat_node` in
`langgraph_backend.py` delegates to a traceable `_decide()` rather than being
decorated itself — LangGraph inspects node signatures to decide what to pass in.

If `langsmith` is not installed, a no-op fallback decorator keeps everything
running.

---

## 6. LangSmith trace structure

```
chat_turn                          ← run_name set in the frontend config
├── chat_node_reasoning            ← LLM decides which tools to call
├── calculator                     ← @traceable tool span
│   └── safe_math_eval             ← shows the NORMALISED expression
├── get_weather
│   ├── geocode_location           ← where "wrong city" bugs live
│   └── open_meteo_forecast        ← where latency lives
├── get_stock_price
│   └── yfinance_history
└── chat_node_reasoning            ← LLM writes the final answer
```

Runs are grouped into a conversation thread via `metadata: {thread_id}`.

**Debugging rule:** when a number comes out wrong, read the tool span's
**input**, not its output. Nine times out of ten the tool computed correctly
and the LLM built the wrong expression — usually dropping parentheses in
EMI-style formulas. That is a system-prompt problem, not a tool problem.

---

## 7. Multi-tool chains

The graph loops `chat_node → tools → chat_node`, so tools can be chained where
a later call's arguments depend on an earlier call's output. Prompts that
exercise this:

| Prompt | Chain |
|---|---|
| "Price of RELIANCE.NS now, and what would 350 shares be worth after an 18% gain?" | `get_stock_price` → `calculator` |
| "Compare TCS.NS and INFY.NS — percentage difference?" | `get_stock_price` ×2 → `calculator` |
| "Check Pune's actual temperature and convert it to Fahrenheit." | `get_weather` → `calculator` |
| "If Nifty is at X and I need 12% annual returns, what level in 3 years?" | `get_stock_price` → `calculator` |

Verified behaviour: a single turn requesting `calculator` + `get_weather`
produced the message sequence

```
HumanMessage → AIMessage(tool_calls=[calculator, get_weather])
             → ToolMessage → ToolMessage
             → AIMessage(final answer)
```

That sequence *is* the loop having executed. Parallel calls in one round run
together; dependent calls take separate rounds.

`recursion_limit: 12` in the frontend caps this at roughly 5 tool rounds per
turn, so a confused model cannot loop forever.

---

## 8. Adding a fourth tool

1. Write the `_impl` + `@tool` pair in `tools.py` (§5.4 pattern).
2. Append it to the `TOOLS` list at the bottom of the file.
3. Add an emoji entry to `TOOL_LABELS`.

The graph, the Streamlit tool boxes, and the tracing all pick it up
automatically — no changes to the other two files.

**Spend your effort on the docstring.** It is the routing logic. A vague
docstring produces random tool calls no amount of graph tuning will fix.

Natural next additions, in rough order of value:

| Tool | Why | Cost |
|---|---|---|
| `get_current_datetime` | models genuinely don't know today's date | trivial |
| `web_search` | everything past the training cutoff | needs Tavily/Brave key |
| `run_python` | subsumes the calculator; unlocks stats, matrices, iteration | needs a real sandbox — Docker or E2B, not a whitelist |
| `sympy_solve` | equations, calculus, symbolic algebra | `pip install sympy` |
| `historical_weather` | past dates | same Open-Meteo pattern |
| `get_stock_history` | charts and trends | already in yfinance |

Note the ordering principle: **few general tools beat many narrow ones.**
`run_python` would replace `calculator` entirely. Tool-selection accuracy
degrades past roughly 20 tools, so resist adding a narrow tool when an existing
general one already covers the case.