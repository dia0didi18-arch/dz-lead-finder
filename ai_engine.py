"""Lead scoring + outreach generation with three interchangeable providers.

- Google Gemini     -> official `google-genai` SDK
- Groq (Free)       -> OpenAI-compatible REST call through httpx
- OpenRouter (Free) -> OpenAI-compatible REST call through httpx

Whatever the provider, the model receives the SAME system prompt (SYSTEM_PROMPT),
the SAME per-lead user prompt, and must answer with the SAME JSON object
(LeadAnalysis). The response goes through one shared parser/cleaner.

Model names change often, so instead of trusting one hardcoded name,
validate_api_key() asks each provider which models the key can really use
and builds a fallback chain from the *_PREFERRED lists below.
"""
import json
import re
import time
from typing import Dict, List, Optional, Set, Tuple

import httpx
from google import genai
from google.genai import errors as genai_errors
from google.genai import types
from pydantic import BaseModel, Field

GEMINI = "Google Gemini"
GROQ = "Groq (Free)"
OPENROUTER = "OpenRouter (Free)"
PROVIDERS = [GEMINI, GROQ, OPENROUTER]

# Preference order (best first). Edit freely; unavailable names are skipped automatically.
# Gemini: the free tier gives many more requests/day on Flash-Lite than on Flash.
GEMINI_PREFERRED = [
    "gemini-3.1-flash-lite", "gemini-3.5-flash-lite", "gemini-flash-lite-latest",
    "gemini-3.6-flash", "gemini-3.8-flash", "gemini-flash-latest",
    "gemini-2.5-flash-lite", "gemini-2.5-flash",
]
# Groq free plan: gpt-oss models + qwen. (8K tokens/min, 1,000 requests/day.)
GROQ_PREFERRED = ["openai/gpt-oss-120b", "qwen/qwen3.8-27b", "openai/gpt-oss-20b"]
GROQ_BASE = "https://api.groq.com/openai/v1"
# OpenRouter: the exact list of free models changes weekly; discovered live from /models.
OPENROUTER_PREFERRED = [
    "meta-llama/llama-3.3-70b-instruct:free", "openai/gpt-oss-120b:free",
    "qwen/qwen3-next-80b-a3b-instruct:free", "google/gemma-3-27b-it:free",
    "nvidia/nemotron-3-super-120b-a12b:free",
]
OPENROUTER_ROUTER = "openrouter/free"  # picks any currently-available free model: last resort
OPENROUTER_BASE = "https://openrouter.ai/api/v1"

_STATIC_CHAIN = {
    GEMINI: GEMINI_PREFERRED,
    GROQ: GROQ_PREFERRED,
    OPENROUTER: OPENROUTER_PREFERRED + [OPENROUTER_ROUTER],
}
_DAILY_RE = re.compile(r"per[- ]?day|daily", re.I)
_DEAD_MODEL_RE = re.compile(
    r"decommission|no longer (?:available|supported)|not found|does not exist|model_not_found|"
    r"unknown model|invalid model|deprecated|No endpoints found for", re.I)
PACE_MIN_TOKENS = 3500  # Groq: below this many tokens left in the minute, wait for the reset


class AIError(Exception):
    pass


class AIAuthError(AIError):
    """Invalid / unauthorized API key."""


class AIConfigError(AIError):
    """Account setting blocks the request (e.g. OpenRouter privacy toggles). Retrying won't help."""


class AIQuotaError(AIError):
    """Rate limit hit. fatal=True means a DAILY cap: stop the batch, retrying won't help today."""

    def __init__(self, message: str, fatal: bool = False):
        super().__init__(message)
        self.fatal = fatal


class _ModelUnavailable(Exception):
    """This model name is gone / not allowed for this key: use the next one."""


class _ModelBusy(Exception):
    """Temporarily unusable: rate-limited (rate=True) or failing upstream (5xx / empty answer)."""

    def __init__(self, message: str, daily: bool = False, rate: bool = False):
        super().__init__(message)
        self.daily = daily
        self.rate = rate


class LeadAnalysis(BaseModel):
    lead_score: int = Field(description="Potential value from 1 to 100")
    scoring_reason: str = Field(description="1-2 short sentences in Arabic naming the single main revenue gap")
    custom_message_darija: str = Field(description="Primary WhatsApp pitch in Algerian Darija (Arabic script)")
    custom_message_franco_arabe: str = Field(description="Same pitch written in Franco-Arabe (Latin script, Algerian mix)")
    custom_message_french: str = Field(description="Same pitch in professional Business French")
    custom_followup_darija: str = Field(description="Soft value-first follow-up in Algerian Darija, to send 3 days later")


class AIClient:
    """One handle for any provider. Gemini uses the SDK client; the others use httpx."""

    def __init__(self, provider: str, api_key: str, http: Optional[httpx.Client] = None):
        if provider not in PROVIDERS:
            raise AIError(f"Unknown AI provider: {provider}")
        # tolerate copy/paste accidents: spaces, quotes, "Bearer " prefix
        key = re.sub(r"^(?i:bearer)\s+", "", (api_key or "").strip().strip("\"'")).strip()
        if not key:
            raise AIAuthError(f"{provider} API key is empty.")
        self.provider = provider
        self.api_key = key
        self.gemini = genai.Client(api_key=key) if provider == GEMINI else None
        self.http = None if provider == GEMINI else (
            http or httpx.Client(timeout=httpx.Timeout(60.0, connect=15.0))
        )
        self.chain: List[str] = []   # filled by validate_api_key()
        self.dead: Set[str] = set()  # names that proved unavailable this session
        self.active: Optional[str] = None

    def candidates(self) -> List[str]:
        base = self.chain or _STATIC_CHAIN[self.provider]
        names = [m for m in base if m not in self.dead]
        if self.active in names:
            names.remove(self.active)
            names.insert(0, self.active)
        return names

    @property
    def short(self) -> str:
        return self.provider.split(" (")[0]


def make_client(provider: str, api_key: str) -> AIClient:
    return AIClient(provider, api_key)


# --------------------------------------------------------------------------
# Gemini transport (SDK)
# --------------------------------------------------------------------------
def _gemini_retry_delay(text: str) -> Optional[float]:
    m = re.search(r"retry in (\d+(?:\.\d+)?)s|retryDelay\W+(\d+(?:\.\d+)?)s", text)
    return float(m.group(1) or m.group(2)) if m else None


def _is_bad_gemini_key(code, text: str) -> bool:
    return code == 401 or (code == 400 and re.search(r"API key|API_KEY|credential", text, re.I) is not None)


def _generate(client: AIClient, contents, config=None, retries: int = 2) -> Tuple[object, str]:
    """Call Gemini through the fallback chain. Returns (response, model_used)."""
    last_error = "no model available"
    quota_hit = daily_seen = False
    forbidden = zero_quota = tried = 0

    for model in client.candidates():
        tried += 1
        for attempt in range(retries + 1):
            try:
                resp = client.gemini.models.generate_content(model=model, contents=contents, config=config)
                client.active = model
                return resp, model
            except genai_errors.APIError as e:
                code = getattr(e, "code", None)
                text = str(e)
                last_error = f"{model} -> {code}: {text[:160]}"
                if _is_bad_gemini_key(code, text):
                    raise AIAuthError("Gemini rejected the API key. Create a new one at aistudio.google.com/apikey "
                                      "(new keys start with 'AQ.', old ones with 'AIza'), and run: "
                                      "pip install -U google-genai")
                if code == 403:
                    forbidden += 1  # key/model not allowed: try the next model
                    break
                if code == 404:
                    client.dead.add(model)  # retired or unavailable for this key
                    break
                if code == 429:
                    if re.search(r"limit:\s*0\b", text):
                        client.dead.add(model)  # free tier has zero quota for this model
                        zero_quota += 1
                        break
                    quota_hit = True
                    if _DAILY_RE.search(text):
                        daily_seen = True
                        break  # daily cap on this model: next one
                    if attempt < retries:
                        time.sleep(min(_gemini_retry_delay(text) or 3 * (attempt + 1), 45))
                        continue
                    break
                if code in (500, 503) and attempt < retries:
                    time.sleep(3 * (attempt + 1))
                    continue
                break  # other 4xx (e.g. model doesn't support JSON mode): next model
            except Exception as e:  # network failure, DNS, etc.
                raise AIError(f"Could not reach the Gemini API: {e}")

    if forbidden and forbidden == tried:
        raise AIAuthError("The key was rejected (403). Check the key and that the Gemini API is enabled for it.")
    if quota_hit:
        raise AIQuotaError(
            "Gemini rate limit reached on all models. Wait a minute or raise the AI pause.",
            fatal=daily_seen,
        )
    if zero_quota and zero_quota == tried:
        raise AIError("The free tier has no quota for these Gemini models right now (limit: 0). "
                      "Try again after the daily reset or enable billing.")
    raise AIError(f"No Gemini model available. Last error: {last_error}")


def _discover_gemini(client: AIClient) -> List[str]:
    """List the models this key can call (models.list costs no quota) and order them."""
    try:
        names = []
        for m in client.gemini.models.list():
            actions = getattr(m, "supported_actions", None) or getattr(m, "supported_generation_methods", None) or []
            name = (getattr(m, "name", "") or "").replace("models/", "", 1)
            if name.startswith("gemini") and (not actions or "generateContent" in actions):
                names.append(name)
    except genai_errors.APIError as e:
        code, text = getattr(e, "code", None), str(e)
        if _is_bad_gemini_key(code, text) or code == 403:
            raise AIAuthError("Gemini rejected the API key. Create a new one at aistudio.google.com/apikey "
                              "(new keys start with 'AQ.'), and run: pip install -U google-genai")
        if code == 429:
            return []
        raise AIError(f"Gemini model list failed ({code}): {text[:160]}")
    except Exception as e:
        raise AIError(f"Could not reach the Gemini API: {e}")

    chain = [p for p in GEMINI_PREFERRED if p in names]
    # Newer GA names we don't know yet: lite first, newest first.
    def version(n: str):
        m = re.match(r"gemini-(\d+(?:\.\d+)?)-", n)
        return float(m.group(1)) if m else 0.0
    extras = [n for n in names if n not in chain and re.fullmatch(r"gemini-\d+(?:\.\d+)?-flash(?:-lite)?", n)]
    extras.sort(key=lambda n: (0 if n.endswith("-lite") else 1, -version(n)))
    return (chain + extras)[:6]


# --------------------------------------------------------------------------
# Groq / OpenRouter transport (OpenAI-compatible REST through httpx)
# --------------------------------------------------------------------------
def _headers(client: AIClient) -> Dict[str, str]:
    h = {"Authorization": f"Bearer {client.api_key}", "Content-Type": "application/json"}
    if client.provider == OPENROUTER:  # optional attribution headers recommended by OpenRouter
        h["HTTP-Referer"] = "http://localhost:8501"
        h["X-Title"] = "DZ Lead Finder"
    return h


def _parse_duration(text: str) -> float:
    """'7.66s' / '2m59.56s' / '120ms' -> seconds."""
    total = 0.0
    for num, unit in re.findall(r"(\d+(?:\.\d+)?)(ms|h|m|s)", text or ""):
        total += float(num) * {"ms": 0.001, "s": 1, "m": 60, "h": 3600}[unit]
    return total


def _retry_after(resp: httpx.Response) -> Optional[float]:
    try:
        return float(resp.headers.get("retry-after", ""))
    except ValueError:
        return None


def _pace_groq(resp: httpx.Response) -> None:
    """Groq's free plan allows only 8K tokens/minute: wait for the window to reset."""
    try:
        remaining = int(resp.headers.get("x-ratelimit-remaining-tokens", ""))
    except ValueError:
        return
    if remaining < PACE_MIN_TOKENS:
        time.sleep(min(_parse_duration(resp.headers.get("x-ratelimit-reset-tokens", "")) + 0.5, 60))


def _payload(client: AIClient, model: str, system: str, user: str) -> Dict:
    p = {
        "model": model,
        "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
        "temperature": 0.7,
        "response_format": {"type": "json_object"},
    }
    if client.provider == GROQ:
        p["max_completion_tokens"] = 1500
        if model.startswith("openai/gpt-oss"):
            p["reasoning_effort"] = "low"  # reasoning tokens count against the 8K TPM budget
    else:
        p["max_tokens"] = 1500
        if "gpt-oss" in model:
            p["reasoning"] = {"effort": "low"}
    return p


def _chat_once(client: AIClient, model: str, system: str, user: str, retries: int = 2) -> str:
    base = GROQ_BASE if client.provider == GROQ else OPENROUTER_BASE
    name = client.short
    payload = _payload(client, model, system, user)
    attempt = 0
    while attempt <= retries:
        try:
            r = client.http.post(f"{base}/chat/completions", headers=_headers(client), json=payload)
        except httpx.TimeoutException:
            if attempt < retries:
                time.sleep(2 * (attempt + 1))
                attempt += 1
                continue
            raise AIError(f"{name} timed out. Try again in a moment.")
        except httpx.HTTPError as e:
            raise AIError(f"Could not reach {name}: {e}")

        code, body = r.status_code, r.text[:500]
        if code == 200:
            try:
                data = r.json()
            except ValueError:
                raise AIError(f"{name} returned a non-JSON response.")
            err = data.get("error") if isinstance(data, dict) else None
            if err and not data.get("choices"):  # OpenRouter can wrap upstream errors in a 200
                code = int(err["code"]) if isinstance(err, dict) and str(err.get("code", "")).isdigit() else 502
                body = str(err)[:500]
            else:
                try:
                    content = data["choices"][0]["message"]["content"]
                except (KeyError, IndexError, TypeError):
                    raise AIError(f"{name} returned an unexpected response shape.")
                if content and content.strip():
                    if client.provider == GROQ:
                        _pace_groq(r)
                    return content
                code, body = 502, "empty completion (the model ran out of tokens while thinking)"

        low = body.lower()
        if code in (400, 404) and "response_format" in payload and re.search(
                r"response_format|json|structured|support.*parameter|parameter.*support", body, re.I) \
                and not _DEAD_MODEL_RE.search(body):
            payload.pop("response_format")  # no JSON mode on this route: rely on the prompt
            continue
        if code == 404 and "data policy" in low:
            raise AIConfigError(
                "OpenRouter blocks free models by your privacy settings. Open "
                "https://openrouter.ai/settings/privacy and enable the free-model options "
                "('providers that may train on inputs' and free model publication), then retry.")
        if code in (401, 402) or (code == 403 and not re.search(r"model|permission", low)):
            if code == 402:
                raise AIError(f"{name}: insufficient credits for this model.")
            raise AIAuthError(f"{name} rejected the API key (HTTP {code}). Check it and try again.")
        if code in (400, 403, 404) and (_DEAD_MODEL_RE.search(body) or code in (403, 404)):
            raise _ModelUnavailable(f"{model}: HTTP {code} {body[:120]}")
        if code == 413:
            raise AIError(f"{name}: request too large for this plan's per-minute token limit.")
        if code in (429, 500, 502, 503, 504):
            daily = code == 429 and bool(_DAILY_RE.search(body))
            if daily:
                if client.provider == OPENROUTER:  # the free-model daily cap is account-wide
                    raise AIQuotaError("OpenRouter daily free-model cap reached (50/day without credits).", fatal=True)
                raise _ModelBusy(f"{model}: daily cap", daily=True, rate=True)
            if attempt < retries:
                time.sleep(min(_retry_after(r) or 3 * (attempt + 1), 30))
                attempt += 1
                continue
            raise _ModelBusy(f"{model}: HTTP {code} {body[:100]}", rate=(code == 429))
        raise AIError(f"{name} HTTP {code}: {body[:150]}")
    raise AIError(f"{name}: no valid response after {retries + 1} attempts.")


def _chat_completion(client: AIClient, system: str, user: str) -> str:
    """Try each candidate model until one answers; remember the one that works."""
    last = "no model available"
    busy: List[_ModelBusy] = []
    tried = 0
    for model in client.candidates():
        tried += 1
        try:
            text = _chat_once(client, model, system, user)
            client.active = model
            return text
        except _ModelUnavailable as e:
            client.dead.add(model)
            last = str(e)
        except _ModelBusy as e:
            busy.append(e)
            last = str(e)
    if any(b.rate for b in busy):
        raise AIQuotaError(
            f"{client.short} rate limit reached on every model. Raise the pause between AI calls.",
            fatal=len(busy) == tried and all(b.daily for b in busy),
        )
    if busy:  # 5xx / empty answers, not a quota problem: say what really happened
        raise AIError(f"{client.short} gave no usable answer on any model. Last error: {last}")
    raise AIError(f"No {client.short} model is available for this key. Last error: {last}")


def _discover_groq(client: AIClient) -> Tuple[List[str], bool]:
    r = client.http.get(f"{GROQ_BASE}/models", headers=_headers(client))
    if r.status_code in (401, 403):
        raise AIAuthError("Invalid Groq API key.")
    if r.status_code != 200:
        return [], False
    ids = [m.get("id", "") for m in r.json().get("data", [])]
    chain = [m for m in GROQ_PREFERRED if m in ids]
    if not chain:  # unknown future names: any plausible chat model
        skip = re.compile(r"guard|whisper|orpheus|compound|safeguard|tts|embed", re.I)
        chain = [i for i in ids if re.search(r"gpt-oss|qwen|llama", i, re.I) and not skip.search(i)][:3]
    return chain, True


def _discover_openrouter(client: AIClient) -> List[str]:
    r = client.http.get(f"{OPENROUTER_BASE}/models")  # public list
    if r.status_code != 200:
        return []
    skip = re.compile(r"vl\b|vision|coder|guard|safeguard|embed|image|audio|tts|ocr|music", re.I)
    big = re.compile(r"120b|235b|80b|70b|72b|405b|397b|35b|32b|30b|27b|super|ultra", re.I)
    free = []
    for m in r.json().get("data", []):
        mid = m.get("id", "")
        pricing = m.get("pricing") or {}
        is_free = mid.endswith(":free") or (str(pricing.get("prompt")) in ("0", "0.0")
                                            and str(pricing.get("completion")) in ("0", "0.0"))
        if not is_free or mid == OPENROUTER_ROUTER or skip.search(mid):
            continue
        if (m.get("context_length") or 0) < 16000:
            continue
        arch = m.get("architecture") or {}
        if "text" not in (arch.get("output_modalities") or ["text"]):
            continue
        params = m.get("supported_parameters") or []
        score = OPENROUTER_PREFERRED.index(mid) if mid in OPENROUTER_PREFERRED else 50
        if mid not in OPENROUTER_PREFERRED:
            score -= 10 if big.search(mid) else 0
            score -= 5 if ("response_format" in params or "structured_outputs" in params) else 0
        free.append((score, mid))
    free.sort()
    return [mid for _, mid in free[:5]] + [OPENROUTER_ROUTER]


# --------------------------------------------------------------------------
# Key validation + model discovery (no completions, so no free-tier requests are spent)
# --------------------------------------------------------------------------
def validate_api_key(client: AIClient) -> str:
    """Raise on a bad key; build the model chain; return a short label of what will be used."""
    if client.provider == GEMINI:
        chain = _discover_gemini(client)
        client.chain = chain or list(GEMINI_PREFERRED)
        return f"Gemini · {client.chain[0]}"

    name = client.short
    try:
        if client.provider == GROQ:
            chain, listed = _discover_groq(client)
            if listed and not chain:
                raise AIError("This Groq key can't use any supported chat model (free plan: "
                              "openai/gpt-oss-120b, openai/gpt-oss-20b, qwen/qwen3.8-27b).")
            client.chain = chain or list(GROQ_PREFERRED)
            return f"Groq · {client.chain[0]}"

        r = client.http.get(f"{OPENROUTER_BASE}/auth/key", headers=_headers(client))
        if r.status_code in (401, 403):
            raise AIAuthError("Invalid OpenRouter API key.")
        client.chain = _discover_openrouter(client) or list(_STATIC_CHAIN[OPENROUTER])
        return f"OpenRouter · {client.chain[0]}"
    except httpx.HTTPError as e:
        raise AIError(f"Could not reach {name}: {e}")
    except (ValueError, AttributeError):
        client.chain = client.chain or list(_STATIC_CHAIN[client.provider])
        return f"{name} · {client.chain[0]}"


# --------------------------------------------------------------------------
# Prompts (identical for every provider)
# --------------------------------------------------------------------------
SYSTEM_PROMPT = """You are a senior B2B copywriter and consultative sales strategist with 10+ years of experience
selling digital services to owners of small local businesses in Algeria. Your WhatsApp messages get replies
because they sound like a thoughtful local professional, not like a mass blast.

STEP 1 - SCORE (1-100)
Higher when the business is clearly active (many reviews), reachable on WhatsApp, and the digital health audit
shows fixable weak spots (missing website, dead website, low reviews, no WhatsApp) that the user's offer can close.
Lower when it looks inactive, has almost no data, or is already well served online.

STEP 2 - TARGET THE WEAK SPOT
The user message contains a DIGITAL HEALTH AUDIT with an explicit list of WEAK SPOTS, ordered by severity.
Pick the FIRST weak spot that the offer can realistically fix and make it the heart of the pitch:
- "Aucun site web" / "Pas de vrai site web" / "Site web hors service" -> pitch the website (discovery, trust, booking).
- "Note Google faible" / "Note Google très faible" / "Quasi aucun avis Google" / "Peu d'avis Google" -> pitch
  review collection and reputation management.
- "Pas de ligne WhatsApp" -> pitch a WhatsApp Business line / click-to-chat setup.
- "Fiche Google probablement non réclamée" -> pitch claiming and optimizing the Google profile.
If several apply, choose the one that matches THE OFFER best. Never invent facts: no traffic numbers, no
competitor names, no promised results. Phrase the weak spot as a short observation ("quand j'ai cherché votre
business sur Google, j'ai vu que..."), never as an accusation. Put this single focus, in 1-2 short Arabic
sentences, in scoring_reason.

STEP 3 - WRITE THE PITCH in THREE versions that all say the same thing:
- custom_message_darija: Algerian Darija in Arabic script (natural, not Moroccan/Egyptian/formal; French
  loanwords like site, réservation, WhatsApp are fine).
- custom_message_franco_arabe: the SAME message in Franco-Arabe, i.e. Algerian Darija written in Latin
  letters with the usual Algerian spelling (ki, ch, ntuma, yezi, c bon ...). No formal Arabic script.
- custom_message_french: the SAME message in professional Business French, warm tone, vouvoiement.
Structure for each version:
  1. Greet the owner/business by name (Darija/Franco-Arabe: "السلام عليكم" / "Salam"; French: "Bonjour")
     and anchor locally using the city/wilaya from the address.
  2. One genuine, specific compliment tied to the real data (rating or review count) ONLY if it is real.
  3. The weak spot from Step 2, as a short observation.
  4. One concrete outcome in plain words (more calls, bookings or orders). No jargon, no price, no list of services.
  5. A single low-friction question as the call to action (yes/no, or an offer of a free 2-minute idea or mock-up).
  6. Signature as described in the SENDER section of the user message.

STEP 4 - WRITE THE 3-DAY FOLLOW-UP (custom_followup_darija)
Sent three days later to someone who has not replied. Soft and value-first: share ONE small, genuinely useful
tip related to the weak spot that they can apply for free. Do not mention that they did not reply, do not
guilt-trip, give a polite easy way out, end with a light question, and sign as in SENDER.

BANNED: generic openers and spam phrases such as "نتمنى أن تكونوا بخير", "نقدم لكم خدماتنا", "عرض استثنائي",
"Dear Sir/Madam", "j'espère que vous allez bien", "agence leader", "offre exceptionnelle", "profitez".
No bullet points, no hashtags, no links, no ALL CAPS, at most one emoji per message.
NEVER output placeholders such as [Votre Nom], [Nom], [Your Name] or {name}.

LENGTH: each pitch version max 55 words, follow-up max 40 words.

OUTPUT FORMAT
Return ONLY one JSON object, with no markdown fences and no commentary, containing exactly these keys:
{"lead_score": <integer 1-100>, "scoring_reason": "<Arabic text>", "custom_message_darija": "<text>",
 "custom_message_franco_arabe": "<text>", "custom_message_french": "<text>",
 "custom_followup_darija": "<text>"}"""


def _build_user_prompt(lead: Dict, offer: str, sender_name: str) -> str:
    website = lead.get("website") or "NONE"
    site_active = lead.get("site_active") or "Unknown"
    site_type = lead.get("site_type") or "Unknown"
    site_line = "No (the business lists no website)" if site_active == "No website" else site_active

    weak = lead.get("health_weak_spots") or []
    if isinstance(weak, str):  # stored as a joined string in some exports
        weak = [w.strip() for w in weak.split(";") if w.strip()]
    weak_block = "\n".join(f"- {w}" for w in weak) if weak else "- (no major weak spot found)"

    if sender_name:
        sender_block = (
            f'The messages are sent by: "{sender_name}".\n'
            "Introduce yourself naturally ONCE in each message using this exact spelling "
            '(Latin letters are fine inside Darija; if it contains "|", read it as "name from agency"). '
            "End each message with this identity as the signature."
        )
    else:
        sender_block = "No sender name was provided. Do NOT invent a name and do NOT write a signature."

    return f"""THE OFFER: {offer}

SENDER
{sender_block}

PROSPECT (scraped from Google Maps)
- Name: {lead.get('name')}
- Search category: {lead.get('keyword')} in {lead.get('city')}
- Address: {lead.get('address') or 'unknown'}
- Google rating: {lead.get('rating') if lead.get('rating') is not None else 'unknown'}
- Number of Google reviews: {lead.get('reviews') or 0}
- Maps URL: {lead.get('maps_url')}

DIGITAL HEALTH AUDIT (automated check)
- Website listed: {website}
- Website active: {site_line}
- Link type: {site_type}
- Own real website (not a social page): {bool(lead.get('has_website'))}
- Reachable on WhatsApp: {bool(lead.get('has_whatsapp'))}
- Digital health score: {lead.get('health_score', 'n/a')}/100
- WEAK SPOTS (in order of severity):
{weak_block}

Analyse this prospect and return the JSON object. The pitch MUST address the FIRST weak spot
that the offer can fix."""


# --------------------------------------------------------------------------
# Shared completion / parsing
# --------------------------------------------------------------------------
def _complete(client: AIClient, system: str, user: str) -> str:
    """Send the prompts to the selected provider and return the raw JSON text."""
    if client.provider == GEMINI:
        config = types.GenerateContentConfig(
            system_instruction=system,
            response_mime_type="application/json",
            response_schema=LeadAnalysis,
            temperature=0.7,
        )
        resp, _model = _generate(client, user, config)
        text = getattr(resp, "text", None)
        if text:
            return text
        parsed = getattr(resp, "parsed", None)
        if parsed is not None:
            return parsed.model_dump_json()
        raise AIError("Gemini returned an empty response (it may have been blocked).")
    return _chat_completion(client, system, user)


def _extract_json(text: str) -> Dict:
    t = re.sub(r"<think>.*?</think>", "", text or "", flags=re.S | re.I).strip()  # reasoning models
    t = re.sub(r"^```(?:json)?\s*|\s*```$", "", t, flags=re.I)
    try:
        data = json.loads(t)
    except json.JSONDecodeError:
        m = re.search(r"\{.*\}", t, re.S)  # models sometimes wrap the object in prose
        if not m:
            raise
        data = json.loads(m.group(0))
    if not isinstance(data, dict):
        raise ValueError("model did not return a JSON object")
    return data


_PLACEHOLDER_RE = re.compile(r"\[[^\[\]]{1,60}\]|\{[^{}]{1,60}\}")
_NAME_HINTS = ("nom", "name", "اسم", "sender", "agence", "agency", "signature", "expéditeur")


def _clean_message(text: str, sender_name: str) -> str:
    """Safety net: swap leftover name placeholders for the real sender, drop other ones."""
    def repl(m: re.Match) -> str:
        inner = m.group(0).lower()
        return sender_name if sender_name and any(h in inner for h in _NAME_HINTS) else ""

    text = _PLACEHOLDER_RE.sub(repl, str(text or ""))
    text = re.sub(r"[ \t]{2,}", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _parse_analysis(text: str, sender_name: str) -> Dict:
    data = _extract_json(text)
    missing = [k for k in LeadAnalysis.model_fields if k not in data]
    if missing:
        raise ValueError(f"missing keys: {', '.join(missing)}")
    out = {
        "lead_score": max(1, min(100, int(float(data["lead_score"])))),
        "scoring_reason": str(data["scoring_reason"]).strip(),
    }
    for k in ("custom_message_darija", "custom_message_franco_arabe", "custom_message_french",
              "custom_followup_darija"):
        out[k] = _clean_message(data[k], sender_name)
    if not out["custom_message_darija"]:
        raise ValueError("empty Darija pitch")
    return out


def fallback_analysis(reason: str) -> Dict:
    return {
        "lead_score": 0,
        "scoring_reason": f"AI error: {reason}",
        "custom_message_darija": "",
        "custom_message_franco_arabe": "",
        "custom_message_french": "",
        "custom_followup_darija": "",
    }


def score_lead(client: AIClient, lead: Dict, offer: str, sender_name: str = "") -> Dict:
    """Score one lead with the selected provider.

    Raises AIAuthError (bad key), AIConfigError (account setting) and AIQuotaError(fatal=True)
    (daily cap) so the caller can stop the batch; any other failure returns a fallback dict
    whose scoring_reason explains what went wrong.
    """
    sender_name = (sender_name or "").strip()
    user = _build_user_prompt(lead, offer, sender_name)
    last_err = "unknown error"
    for _ in range(2):  # second pass only for malformed output
        try:
            return _parse_analysis(_complete(client, SYSTEM_PROMPT, user), sender_name)
        except (AIAuthError, AIConfigError):
            raise
        except AIQuotaError as e:
            if e.fatal:
                raise
            return fallback_analysis(str(e))
        except AIError as e:
            return fallback_analysis(str(e))
        except (json.JSONDecodeError, ValueError, TypeError, KeyError, AttributeError) as e:
            last_err = f"bad answer from the model ({e})"
            time.sleep(1)
    return fallback_analysis(last_err)
