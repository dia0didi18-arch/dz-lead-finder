"""Google Maps scraper (async Playwright) + Digital Health Audit engine.

Google Maps CSS selectors drift over time. The `data-item-id` based selectors
(phone, address, website) are the most stable; the class-based ones
(h1.DUwDvf, div.F7nice, a.hfpxzc) are the first thing to check if the scraper
suddenly returns empty fields.

Every returned lead already passed four gates:
  1. not in leads.db (when a KnownLeads index is given), checked from the result
     link BEFORE the place page is opened, and again after reading name/phone;
  2. strict Algerian mobile regex (+213 5/6/7 + 8 digits);
  3. wilaya boundary check: the address must mention the requested city or
     wilaya, so results pulled in from a neighbouring region are dropped;
  4. not a duplicate of another lead found in the same run.

The Digital Health Audit enriches each lead with has_website / is_unclaimed /
has_whatsapp plus a 0-100 health score and a list of weak spots that the AI
pitch then targets explicitly.
"""
import asyncio
import random
import re
import subprocess
import sys
from datetime import datetime
from typing import Callable, Dict, List, Optional, Tuple
from urllib.parse import quote_plus, urlparse

import httpx
from playwright.async_api import Error as PlaywrightError
from playwright.async_api import Page
from playwright.async_api import TimeoutError as PWTimeout
from playwright.async_api import async_playwright

from utils import MOBILE_E164_RE, KnownLeads, _norm, normalize_algerian_phone, place_key, whatsapp_link

ProgressCB = Callable[[float, str], None]


class ScraperError(Exception):
    """Raised for user-facing scraping failures."""


def _ensure_chromium_installed() -> None:
    """Ensure Playwright Chromium browser binaries are installed on cloud hosts."""
    try:
        subprocess.run([sys.executable, "-m", "playwright", "install", "chromium"], check=True)
    except Exception as e:
        print(f"[Playwright Auto-Install Warning]: {e}")


# --------------------------------------------------------------------------
# Wilaya boundary enforcement
# --------------------------------------------------------------------------
WILAYAS = [
    "Adrar", "Chlef", "Laghouat", "Oum El Bouaghi", "Batna", "Béjaïa", "Biskra",
    "Béchar", "Blida", "Bouira", "Tamanrasset", "Tébessa", "Tlemcen", "Tiaret",
    "Tizi Ouzou", "Alger", "Djelfa", "Jijel", "Sétif", "Saïda", "Skikda",
    "Sidi Bel Abbès", "Annaba", "Guelma", "Constantine", "Médéa", "Mostaganem",
    "M'Sila", "Mascara", "Ouargla", "Oran", "El Bayadh", "Illizi", "Bordj Bou Arréridj",
    "Boumerdès", "El Tarf", "Tindouf", "Tissemsilt", "El Oued", "Khenchela",
    "Souk Ahras", "Tipaza", "Mila", "Aïn Defla", "Naâma", "Aïn Témouchent",
    "Ghardaïa", "Relizane", "Timimoun", "Bordj Badji Mokhtar", "Ouled Djellal",
    "Béni Abbès", "In Salah", "In Guezzam", "Touggourt", "Djanet", "El M'Ghair", "El Meniaa",
]

WILAYA_ALIASES = {
    "algiers": "Alger", "alger": "Alger", "algiers centre": "Alger",
    "oran": "Oran", "wahran": "Oran", "constantine": "Constantine",
    "blida": "Blida", "sétif": "Sétif", "setif": "Sétif",
    "annaba": "Annaba", "béjaia": "Béjaïa", "bejaia": "Béjaïa",
    "tlemcen": "Tlemcen", "tizi ouzou": "Tizi Ouzou",
    "sidi bel abbes": "Sidi Bel Abbès", "skikda": "Skikda",
    "jijel": "Jijel", "batna": "Batna", "biskra": "Biskra",
    "mostaganem": "Mostaganem", "boumerdes": "Boumerdès", "boumerdès": "Boumerdès",
    "tipaza": "Tipaza", "chlef": "Chlef", "laghouat": "Laghouat",
    "ouargla": "Ouargla", "ghardaia": "Ghardaïa", "ghardaïa": "Ghardaïa",
    "medea": "Médéa", "médéa": "Médéa", "relizane": "Relizane",
}


def resolve_wilaya(city: str) -> Optional[str]:
    norm = _norm(city)
    for key, wilaya in WILAYA_ALIASES.items():
        if _norm(key) == norm:
            return wilaya
    for wilaya in WILAYAS:
        w = _norm(wilaya)
        if w in norm or norm in w:
            return wilaya
    return None


def build_maps_query(keyword: str, city: str, wilaya: Optional[str] = None) -> str:
    location = f"{city}, {wilaya}" if wilaya and _norm(wilaya) not in _norm(city) else city
    return f"{keyword} {location}, Algérie"


def _inside_wilaya(address: str, city: str, wilaya: Optional[str]) -> bool:
    norm_addr = _norm(address)
    for token in (wilaya, city):
        t = _norm(token)
        if t and t in norm_addr:
            return True
    return not city and not wilaya


async def _pause(delay_range: Tuple[float, float]) -> None:
    await asyncio.sleep(random.uniform(*delay_range))


async def _text(page: Page, selector: str) -> str:
    loc = page.locator(selector).first
    try:
        if await loc.count() == 0:
            return ""
        return (await loc.inner_text(timeout=2500)).strip()
    except Exception:
        return ""


async def _attr(page: Page, selector: str, attr: str) -> str:
    loc = page.locator(selector).first
    try:
        if await loc.count() == 0:
            return ""
        return (await loc.get_attribute(attr, timeout=2500)) or ""
    except Exception:
        return ""


async def _accept_consent(page: Page) -> None:
    labels = ("Accept all", "Tout accepter", "I agree", "J'accepte", "Reject all", "Tout refuser")
    for label in labels:
        try:
            btn = page.get_by_role("button", name=label)
            if await btn.count():
                await btn.first.click(timeout=3000)
                await page.wait_for_load_state("domcontentloaded")
                return
        except Exception:
            continue


async def _collect_place_urls(
    page: Page,
    target: int,
    city: str,
    known: KnownLeads,
    stats: Dict,
    delay_range: Tuple[float, float],
    progress: Optional[ProgressCB],
) -> List[str]:
    feed = page.locator('div[role="feed"]')
    try:
        await feed.wait_for(timeout=20000)
    except PWTimeout:
        if "/maps/place/" in page.url:
            if known.seen_candidate(page.url, city):
                stats["skipped"] += 1
                return []
            return [page.url]
        raise ScraperError(
            "No results list found. Check the keyword/city, your connection, "
            "or whether Google is showing a CAPTCHA (try again with headless OFF)."
        )

    candidates: Dict[str, None] = {}
    seen_all = set()
    stagnant = 0
    while len(candidates) < target and stagnant < 4:
        hrefs = await page.eval_on_selector_all("a.hfpxzc", "els => els.map(e => e.href)")
        before = len(seen_all)
        for h in hrefs:
            if "/maps/place/" not in h:
                continue
            key = place_key(h)
            if key in seen_all:
                continue
            seen_all.add(key)
            if known.seen_candidate(h, city):
                stats["skipped"] += 1
            else:
                candidates.setdefault(h, None)
        stagnant = stagnant + 1 if len(seen_all) == before else 0
        if progress:
            progress(0.1, f"Searching... {len(candidates)} new places, {stats['skipped']} skipped (already saved)")
        if len(candidates) >= target:
            break
        await feed.evaluate("el => el.scrollBy(0, el.scrollHeight)")
        await _pause(delay_range)
    return list(candidates)[:target]


def _to_float(s: str) -> Optional[float]:
    m = re.search(r"\d+(?:[.,]\d+)?", s or "")
    return float(m.group(0).replace(",", ".")) if m else None


async def _extract_place(page: Page, url: str) -> Dict:
    await page.goto(url, wait_until="domcontentloaded", timeout=45000)
    await page.locator("h1").first.wait_for(timeout=15000)

    name = await _text(page, "h1.DUwDvf") or await _text(page, "h1")

    phone_id = await _attr(page, 'button[data-item-id^="phone:tel:"]', "data-item-id")
    phone_raw = phone_id.split("phone:tel:", 1)[1] if "phone:tel:" in phone_id else ""

    addr = await _attr(page, 'button[data-item-id="address"]', "aria-label")
    address = re.sub(r"^[^:]+:\s*", "", addr).strip() if addr else ""

    website = await _attr(page, 'a[data-item-id="authority"]', "href")

    rating, reviews = None, 0
    block = await _text(page, "div.F7nice")
    if block:
        rating = _to_float(block.split("\n")[0])
        m = re.search(r"\(([\d\s,.\u202f\u00a0]+)\)", block)
        if m:
            reviews = int(re.sub(r"\D", "", m.group(1)) or 0)

    return {
        "name": name,
        "phone_raw": phone_raw,
        "address": address,
        "rating": rating,
        "reviews": reviews,
        "website": website,
        "maps_url": url,
    }


async def scrape_google_maps(
    keyword: str,
    city: str,
    max_leads: int,
    progress: Optional[ProgressCB] = None,
    headless: bool = True,
    delay_range: Tuple[float, float] = (3.0, 7.0),
    known: Optional[KnownLeads] = None,
    stats: Optional[Dict] = None,
    wilaya: Optional[str] = None,
) -> List[Dict]:
    # ضمان وجود Chromium مسبقاً قبل فتح المتصفح
    _ensure_chromium_installed()

    known = known or KnownLeads(enabled=False)
    stats = stats if stats is not None else {}
    stats.update(visited=0, skipped=0, non_mobile=0, failed=0, out_of_area=0)

    resolved = resolve_wilaya(wilaya or city)
    query = build_maps_query(keyword, city, resolved)
    search_url = f"https://www.google.com/maps/search/{quote_plus(query)}?hl=fr"
    results: List[Dict] = []
    target = min(max(max_leads * 3, max_leads), 300)

    try:
        async with async_playwright() as pw:
            browser = await pw.chromium.launch(
                headless=headless,
                args=["--disable-blink-features=AutomationControlled", "--lang=fr-FR"],
            )
            context = await browser.new_context(
                locale="fr-FR", viewport={"width": 1366, "height": 900}
            )
            page = await context.new_page()
            page.set_default_timeout(20000)

            try:
                if progress:
                    progress(0.02, f"Opening Google Maps ({query})...")
                await page.goto(search_url, wait_until="domcontentloaded", timeout=45000)
                await _accept_consent(page)
                await _pause(delay_range)

                urls = await _collect_place_urls(page, target, city, known, stats, delay_range, progress)

                for url in urls:
                    stats["visited"] += 1
                    try:
                        data = await _extract_place(page, url)
                    except (PWTimeout, PlaywrightError):
                        stats["failed"] += 1
                    else:
                        e164, _ = normalize_algerian_phone(data.pop("phone_raw", ""))
                        if not _inside_wilaya(data.get("address", ""), city, resolved):
                            stats["out_of_area"] += 1
                        elif not (e164 and MOBILE_E164_RE.fullmatch(e164)):
                            stats["non_mobile"] += 1
                        elif known.seen_lead(data["name"], city, e164):
                            stats["skipped"] += 1
                        else:
                            data.update(
                                phone=e164,
                                phone_type="mobile",
                                whatsapp_link=whatsapp_link(e164, "mobile"),
                                scraped_at=datetime.now().isoformat(timespec="seconds"),
                            )
                            known.remember(data["name"], city, e164, url)
                            results.append(data)

                    if progress:
                        progress(
                            0.15 + 0.85 * min(1.0, max(len(results) / max_leads, stats["visited"] / len(urls))),
                            f"Extracting: {len(results)}/{max_leads} valid mobile leads "
                            f"({stats['visited']}/{len(urls)} visited, {stats['non_mobile']} non-mobile, "
                            f"{stats['out_of_area']} outside area)",
                        )
                    if len(results) >= max_leads:
                        break
                    await _pause(delay_range)
            finally:
                await browser.close()
    except ScraperError:
        raise
    except PWTimeout as e:
        raise ScraperError(f"Network timeout while loading Google Maps: {e}")
    except PlaywrightError as e:
        msg = str(e)
        if "Executable doesn't exist" in msg or "playwright install" in msg:
            raise ScraperError(
                "Playwright browser binaries are missing. Run: playwright install chromium"
            )
        raise ScraperError(f"Browser error: {msg.splitlines()[0]}")

    if not results:
        raise ScraperError(
            f"No new mobile leads found around {city}. Visited {stats['visited']} listings, "
            f"skipped {stats['skipped']} already saved, dropped {stats['non_mobile']} without a mobile number, "
            f"{stats['out_of_area']} outside the {resolved or city} area, {stats['failed']} failed to load. "
            "Try another keyword/city or turn the duplicate skip off."
        )
    return results


def _new_loop() -> asyncio.AbstractEventLoop:
    return asyncio.ProactorEventLoop() if sys.platform == "win32" else asyncio.new_event_loop()


def run_scrape(*args, **kwargs) -> List[Dict]:
    loop = _new_loop()
    try:
        return loop.run_until_complete(scrape_google_maps(*args, **kwargs))
    finally:
        loop.close()


# --------------------------------------------------------------------------
# Digital Health Audit: website reachability + a 0-100 health score
# --------------------------------------------------------------------------
SOCIAL_DOMAINS = (
    "facebook.com", "fb.com", "instagram.com", "tiktok.com", "linkedin.com",
    "youtube.com", "x.com", "twitter.com", "linktr.ee", "wa.me",
)
_UA = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/124.0 Safari/537.36"
)

W_WEBSITE_OWN_ACTIVE = 30
W_WEBSITE_OWN_DEAD = 10
W_RATING_GOOD = 20
W_RATING_MID = 10
W_REVIEWS_MANY = 20
W_REVIEWS_SOME = 10
W_WHATSAPP = 15
W_CLAIMED = 15


def compute_digital_health(
    website: Optional[str],
    site_active: str,
    rating: Optional[float],
    reviews_count: int,
    has_whatsapp: bool,
) -> Dict:
    website = (website or "").strip()
    site_active = site_active or "Unknown"
    rating = rating or 0.0
    try:
        reviews_count = int(reviews_count or 0)
    except (TypeError, ValueError):
        reviews_count = 0

    has_website = bool(website) and site_active != "No website"
    host = urlparse(website if website.startswith(("http://", "https://")) else "https://" + (website or "x")).netloc
    host = host[4:] if host.startswith("www.") else host
    is_social_only = bool(website) and any(host == d or host.endswith("." + d) for d in SOCIAL_DOMAINS)

    score, weak = 0, []
    if has_website and site_active == "Yes" and not is_social_only:
        score += W_WEBSITE_OWN_ACTIVE
    elif has_website and is_social_only:
        score += 5
        weak.append("Pas de vrai site web (page réseau social seulement)")
    elif has_website and site_active == "No":
        score += W_WEBSITE_OWN_DEAD
        weak.append("Site web hors service")
    else:
        weak.append("Aucun site web")

    if rating >= 4.2:
        score += W_RATING_GOOD
    elif rating >= 3.5:
        score += W_RATING_MID
        weak.append("Note Google faible")
    elif rating > 0:
        score += 3
        weak.append("Note Google très faible")
    else:
        weak.append("Aucune note Google")

    if reviews_count >= 25:
        score += W_REVIEWS_MANY
    elif reviews_count >= 5:
        score += W_REVIEWS_SOME
        weak.append("Peu d'avis Google")
    else:
        weak.append("Quasi aucun avis Google")

    if has_whatsapp:
        score += W_WHATSAPP
    else:
        weak.append("Pas de ligne WhatsApp")

    is_unclaimed = not website and not has_whatsapp
    if is_unclaimed:
        weak.append("Fiche Google probablement non réclamée")
    else:
        score += W_CLAIMED

    return {
        "has_website": has_website and not is_social_only,
        "is_unclaimed": is_unclaimed,
        "has_whatsapp": bool(has_whatsapp),
        "health_score": max(0, min(100, score)),
        "health_weak_spots": weak,
    }


async def _check_website(client: httpx.AsyncClient, sem: asyncio.Semaphore, url: str) -> Dict:
    if not url or not str(url).strip():
        return {"site_active": "No website", "site_type": "None"}
    url = str(url).strip()
    candidates = [url] if url.lower().startswith(("http://", "https://")) else ["https://" + url, "http://" + url]

    host = urlparse(candidates[0]).netloc.lower()
    host = host[4:] if host.startswith("www.") else host
    is_social = any(host == d or host.endswith("." + d) for d in SOCIAL_DOMAINS)
    site_type = "Social page only" if is_social else "Own website"

    active = False
    async with sem:
        for cand in candidates:
            try:
                async with client.stream("GET", cand) as r:
                    if r.status_code < 400 or r.status_code in (401, 403, 429):
                        active = True
                        break
            except Exception:
                continue
    return {"site_active": "Yes" if active else "No", "site_type": site_type}


async def audit_leads(leads: List[Dict], concurrency: int = 8, timeout: float = 8.0) -> List[Dict]:
    sem = asyncio.Semaphore(concurrency)
    async with httpx.AsyncClient(
        follow_redirects=True, timeout=timeout, verify=False, headers={"User-Agent": _UA}
    ) as client:
        site_results = list(await asyncio.gather(*(_check_website(client, sem, l.get("website", "")) for l in leads)))

    audits = []
    for lead, site in zip(leads, site_results):
        has_whatsapp = bool(lead.get("phone")) and lead.get("phone_type") == "mobile"
        health = compute_digital_health(
            website=lead.get("website", ""),
            site_active=site["site_active"],
            rating=lead.get("rating"),
            reviews_count=lead.get("reviews", 0),
            has_whatsapp=has_whatsapp,
        )
        audits.append({**site, **health})
    return audits


def run_audit(leads: List[Dict]) -> List[Dict]:
    loop = _new_loop()
    try:
        return loop.run_until_complete(audit_leads(leads))
    finally:
        loop.close()
