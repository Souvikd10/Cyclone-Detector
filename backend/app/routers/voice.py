"""
Voice & Conversational AI Agent Router:
  POST /voice/token — Mints short-lived ephemeral token for Gemini Live WebSocket API.
  POST /voice/chat  — Direct conversational cyclone intelligence powered by Google Gemini API.
                      Supports English, Hindi, and Bengali with live storm telemetry, exact user location proximity, and dashboard action execution.
"""

import json
import logging
import math
import re
import uuid
from datetime import datetime, timedelta, timezone
from typing import Any, Optional

import httpx
from fastapi import APIRouter, HTTPException, status
from pydantic import BaseModel, Field

from app.config import get_settings
from app.core.mock_data import get_mock_storms

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/voice", tags=["Voice Agent"])


# ── Schemas for Token Minting ──

class VoiceTokenRequest(BaseModel):
    client_session_id: Optional[str] = Field(None, description="Optional unique client session ID")
    model: Optional[str] = Field(None, description="Target Gemini Live model")


class VoiceTokenResponse(BaseModel):
    token: str = Field(..., description="Ephemeral token name or simulated token")
    mode: str = Field(..., description="'live' if real Gemini Live token, 'simulated' if local fallback")
    model: str = Field(..., description="Active model identifier")
    expires_in: int = Field(1800, description="Token validity in seconds")
    message: str = Field(..., description="Status explanation")


# ── Schemas for Conversational Chat ──

class UserLocation(BaseModel):
    lat: float = Field(..., description="Latitude")
    lon: float = Field(..., description="Longitude")
    city: Optional[str] = Field(None, description="City or coastal locality")
    state: Optional[str] = Field(None, description="State / Province")
    country: Optional[str] = Field("India", description="Country")


class ChatMessage(BaseModel):
    role: str = Field(..., description="'user' or 'assistant' or 'model'")
    content: str


class VoiceChatRequest(BaseModel):
    message: str = Field(..., description="User message or voice transcript")
    history: Optional[list[ChatMessage]] = Field(default_factory=list, description="Recent conversation history")
    current_storm_id: Optional[str] = Field(None, description="Active storm ID on the dashboard")
    user_location: Optional[UserLocation] = Field(None, description="Current geographic location of the user")
    language: Optional[str] = Field(None, description="Preferred language: 'en', 'hi', 'bn', or None for auto-detect")
    model: Optional[str] = Field(None, description="Optional override for Gemini model")


class ToolExecutionItem(BaseModel):
    id: str
    name: str
    args: dict[str, Any] = {}


class VoiceChatResponse(BaseModel):
    text: str = Field(..., description="AI response text")
    language: str = Field("en", description="Detected language: en, hi, bn")
    tool_executions: list[ToolExecutionItem] = Field(default_factory=list, description="Dashboard actions to execute")
    model: str = Field(..., description="Model used")
    source: str = Field("gemini", description="'gemini' or 'fallback'")


# ── Language Detection Helper ──

def detect_language(text: str) -> str:
    # Bengali Unicode range: \u0980-\u09FF
    if re.search(r"[\u0980-\u09FF]", text):
        return "bn"
    # Devanagari (Hindi) Unicode range: \u0900-\u097F
    if re.search(r"[\u0900-\u097F]", text):
        return "hi"

    lower = text.lower()
    # Bengali phonetic / Romanized words
    if any(w in lower for w in [
        "kemon", "ache", "achhe", "khobor", "ki", "naam", "ghurnijhor", "tofan", "dekhao",
        "dekhaw", "bondho", "kothay", "kobe", "hobe", "asbe", "bengali", "bangla", "amader", "ekhane", "bipod"
    ]):
        return "bn"

    # Hindi phonetic / Romanized words
    if any(w in lower for w in [
        "kya", "kaise", "kaisa", "kahan", "hai", "dikhao", "toofan", "chhatisgarh", "odisha",
        "batao", "chal", "raha", "hawa", "alert", "baarish", "samay", "hindi", "mere yahan", "khatra", "paas"
    ]):
        return "hi"

    return "en"


def calculate_distance_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Haversine formula to compute great-circle distance between two points in km."""
    R = 6371.0
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = (
        math.sin(dlat / 2) ** 2
        + math.cos(math.radians(lat1))
        * math.cos(math.radians(lat2))
        * math.sin(dlon / 2) ** 2
    )
    return round(R * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a)), 1)


def calculate_bearing(lat1: float, lon1: float, lat2: float, lon2: float) -> str:
    """Calculates 16-point compass direction from point 1 to point 2."""
    dlon = math.radians(lon2 - lon1)
    y = math.sin(dlon) * math.cos(math.radians(lat2))
    x = math.cos(math.radians(lat1)) * math.sin(math.radians(lat2)) - (
        math.sin(math.radians(lat1)) * math.cos(math.radians(lat2)) * math.cos(dlon)
    )
    compass_bearing = (math.degrees(math.atan2(y, x)) + 360) % 360
    directions = [
        "North", "North-Northeast", "Northeast", "East-Northeast",
        "East", "East-Southeast", "Southeast", "South-Southeast",
        "South", "South-Southwest", "Southwest", "West-Southwest",
        "West", "West-Northwest", "Northwest", "North-Northwest"
    ]
    return directions[round(compass_bearing / 22.5) % 16]


def _build_storm_context(user_loc: Optional[UserLocation] = None) -> str:
    """Build a formatted summary of tracked storms with live distances to the user."""
    try:
        storms = get_mock_storms()
    except Exception:
        storms = []

    if not storms:
        return "No active tropical cyclones currently tracked in the North Indian Ocean basin. Monitoring feeds are awaiting model data ingestion."

    storm_items = []
    for s in storms:
        s_id = str(s.get("id", ""))
        name = s.get("name", "Unknown Storm")
        basin = s.get("basin", "NIO")
        cat = s.get("category", "CS")
        lat = float(s.get("current_lat", 0.0))
        lon = float(s.get("current_lon", 0.0))
        wind = float(s.get("max_wind_speed", 0.0))
        pres = float(s.get("min_pressure", 1000.0))
        status_val = s.get("status", "active")

        dist_km = None
        bearing = None
        hazard_note = ""
        if user_loc is not None:
            dist_km = calculate_distance_km(user_loc.lat, user_loc.lon, lat, lon)
            bearing = calculate_bearing(user_loc.lat, user_loc.lon, lat, lon)
            if dist_km < 150:
                hazard_note = f"IMMEDIATE DANGER ZONE (Direct core threat, ~{dist_km} km away)"
            elif dist_km < 350:
                hazard_note = f"HIGH WARNING ZONE (Outer storm gale winds & heavy rainbands, ~{dist_km} km away)"
            elif dist_km < 700:
                hazard_note = f"REGIONAL WATCH ZONE (~{dist_km} km away)"
            else:
                hazard_note = f"DISTANT (~{dist_km} km away, low direct impact)"

        storm_items.append({
            "name": name,
            "id": s_id,
            "basin": basin,
            "category": cat,
            "lat": lat,
            "lon": lon,
            "wind": wind,
            "pressure": pres,
            "status": status_val,
            "dist_km": dist_km if dist_km is not None else 99999,
            "bearing": bearing,
            "hazard_note": hazard_note,
        })

    if user_loc is not None:
        storm_items.sort(key=lambda x: x["dist_km"])

    lines = []
    for item in storm_items:
        dist_str = f" | DISTANCE FROM USER: {item['dist_km']} km to the {item['bearing']} | Hazard: {item['hazard_note']}" if user_loc else ""
        lines.append(
            f"- {item['name']} (ID: {item['id']}): Category={item['category']}, Status={item['status']}, "
            f"Wind={item['wind']} kts (~{int(item['wind'] * 1.852)} km/h), Central Pressure={item['pressure']} hPa, "
            f"Coordinates=({item['lat']}°N, {item['lon']}°E){dist_str}"
        )
    return "\n".join(lines)


# ── POST /voice/token ──

@router.post(
    "/token",
    response_model=VoiceTokenResponse,
    summary="Mint an ephemeral token for Gemini Live API",
)
async def mint_voice_token(body: Optional[VoiceTokenRequest] = None):
    settings = get_settings()
    api_key = (settings.gemini_api_key or "").strip()
    target_model = (body.model if body and body.model else None) or settings.gemini_live_model

    if not api_key:
        logger.info("[Voice] GEMINI_API_KEY is not set. Issuing simulated fallback token.")
        return VoiceTokenResponse(
            token="simulated-local-session-token",
            mode="simulated",
            model=target_model,
            expires_in=3600,
            message="GEMINI_API_KEY is not configured yet. Running in simulated assistant mode.",
        )

    try:
        now_utc = datetime.now(timezone.utc)
        expire_time = (now_utc + timedelta(minutes=30)).strftime("%Y-%m-%dT%H:%M:%SZ")
        new_session_expire_time = (now_utc + timedelta(minutes=5)).strftime("%Y-%m-%dT%H:%M:%SZ")

        url = "https://generativelanguage.googleapis.com/v1alpha/auth_tokens"
        headers = {
            "Content-Type": "application/json",
            "x-goog-api-key": api_key,
        }
        token_request_body = {
            "uses": 1,
            "expireTime": expire_time,
            "newSessionExpireTime": new_session_expire_time,
        }

        async with httpx.AsyncClient(timeout=10.0) as client:
            resp = await client.post(url, headers=headers, json=token_request_body)

        if resp.status_code == 200:
            data = resp.json()
            token_name = data.get("name") or data.get("token") or "token-minted"
            logger.info(f"[Voice] Ephemeral token minted successfully: {token_name[:20]}...")
            return VoiceTokenResponse(
                token=token_name,
                mode="live",
                model=target_model,
                expires_in=1800,
                message="Live ephemeral token minted successfully.",
            )
        else:
            logger.warning(
                f"[Voice] Google auth_tokens returned HTTP {resp.status_code}. Fallback mode active."
            )
            return VoiceTokenResponse(
                token="simulated-fallback-token",
                mode="simulated",
                model=target_model,
                expires_in=1800,
                message=f"Gemini API returned status {resp.status_code}. Fallback mode active.",
            )

    except Exception as exc:
        logger.error(f"[Voice] Failed to contact Gemini token endpoint: {exc}. Using simulated mode.")
        return VoiceTokenResponse(
            token="simulated-fallback-token",
            mode="simulated",
            model=target_model,
            expires_in=1800,
            message=f"Token error: {str(exc)}. Fallback mode active.",
        )


# ── POST /voice/chat ──

@router.post(
    "/chat",
    response_model=VoiceChatResponse,
    summary="Real-time Multilingual Gemini Cyclone Intelligence Conversation",
    description=(
        "Uses Google Gemini with active cyclone intelligence telemetry to answer user queries, "
        "provide weather/storm updates, explain tracks & intensity, and execute dashboard actions."
    ),
)
async def voice_chat(body: VoiceChatRequest):
    settings = get_settings()
    api_key = (settings.gemini_api_key or "").strip()
    detected_lang = body.language or detect_language(body.message)

    # Pick model
    model_name = body.model or settings.gemini_chat_model or "models/gemini-2.5-flash"
    if not model_name.startswith("models/"):
        clean_model = model_name.split("/")[-1]
    else:
        clean_model = model_name

    storm_context = _build_storm_context(body.user_location)

    user_loc_block = ""
    if body.user_location:
        u = body.user_location
        loc_name = u.city or "User's Area"
        if u.state:
            loc_name += f", {u.state}"
        user_loc_block = f"""
USER'S EXACT GEOGRAPHIC LOCATION & PROXIMITY TELEMETRY:
- User is currently situated at: {loc_name} ({u.lat:.2f}°N, {u.lon:.2f}°E).
- LOCALIZATION GUIDELINES:
  Whenever the user asks about "my area", "here", "amader ekhane", "mere yahan", "apne shehar", "kahan hai", "nearest storm", "safe", "danger", "weather", or localized concerns:
  1. Base your answer directly on their distance to the storms in the telemetry above.
  2. If storms are active: clearly state the distance in kilometers from {loc_name} to the nearest cyclone and provide threat status (Immediate Danger Zone [<150 km], High Warning Zone [150-350 km], Regional Watch [350-700 km], or Safe Distance [>700 km]).
  3. If NO storms are currently active in the North Indian Ocean basin: reassure them that {loc_name} is currently clear of cyclonic threats and satellite monitoring systems are active.
"""

    system_instruction = f"""You are the official Cyclone Intelligence AI Assistant for the SIH-2K26 Cyclone Prediction & Decision-Support System, powered by Google Gemini.
You specialize in tropical meteorology, cyclone detection, tracking, intensity forecasting, rapid intensification (RI), landfall prediction, early warning alerts (IMD 4-stage alert protocol: Pre-Cyclone Watch, Cyclone Alert, Cyclone Warning, Post-Landfall Outlook), and coastal safety advisories for the North Indian Ocean (Bay of Bengal and Arabian Sea).

Language & Communication:
- You natively understand and communicate in English, Hindi (हिन्दी), and Bengali (বাংলা).
- You also understand Romanized transliterations (e.g. "naam ki", "kaisa hai", "toofan kothay ache", "wind speed").
- Always respond in the exact language/script the user used (Bengali for Bengali, Hindi for Hindi, English for English).
- Be authoritative, scientifically accurate, and concise (2-4 sentences suitable for a dashboard assistant and speech-to-text readouts).
{user_loc_block}
Current Active Storms in the System:
{storm_context}

Dashboard Actions:
If the user asks to view/show/navigate to a specific storm, toggle map layers, open alerts, or view replay, append an action command tag at the end of your response:
- To navigate to a storm: [ACTION:navigate_to_storm:{{"storm_id":"<storm_id_or_name>"}}]
- To toggle map layers: [ACTION:toggle_map_layer:{{"layer":"forecast_cone|cyclone_mask|impact_zone|track","visible":true}}]
- To start historical replay: [ACTION:start_replay:{{"storm_id":"<storm_id_or_name>"}}]
- To open alert panel: [ACTION:open_alert_panel:{{}}]
"""

    if not api_key:
        logger.warning("[Voice] GEMINI_API_KEY is not configured; using offline cyclone knowledge.")
        return _offline_cyclone_fallback(body.message, detected_lang, body.user_location)

    # Format conversation contents for Gemini REST API
    contents = []
    for turn in (body.history or [])[-6:]:
        role = "user" if turn.role == "user" else "model"
        contents.append({"role": role, "parts": [{"text": turn.content}]})

    contents.append({"role": "user", "parts": [{"text": body.message}]})

    payload = {
        "systemInstruction": {"parts": [{"text": system_instruction}]},
        "contents": contents,
        "generationConfig": {
            "temperature": 0.3,
            "maxOutputTokens": 800,
        },
    }

    target_models = [clean_model]
    if clean_model != "gemini-2.5-flash":
        target_models.append("gemini-2.5-flash")

    last_error = None
    for candidate_model in target_models:
        gemini_url = f"https://generativelanguage.googleapis.com/v1beta/models/{candidate_model}:generateContent?key={api_key}"
        try:
            async with httpx.AsyncClient(timeout=12.0) as client:
                resp = await client.post(gemini_url, json=payload, headers={"Content-Type": "application/json"})

            if resp.status_code == 200:
                res_data = resp.json()
                candidates = res_data.get("candidates", [])
                if candidates:
                    parts = candidates[0].get("content", {}).get("parts", [])
                    raw_text = "".join(p.get("text", "") for p in parts)

                    # Extract any [ACTION:name:{args}]
                    tool_executions: list[ToolExecutionItem] = []
                    action_matches = re.findall(r"\[ACTION:([a-zA-Z_]+):(\{.*?\})\]", raw_text)
                    for action_name, args_str in action_matches:
                        try:
                            parsed_args = json.loads(args_str)
                        except Exception:
                            parsed_args = {}
                        tool_executions.append(
                            ToolExecutionItem(
                                id="call_" + uuid.uuid4().hex[:6],
                                name=action_name,
                                args=parsed_args,
                            )
                        )

                    # Clean action tags from user-facing text
                    clean_text = re.sub(r"\[ACTION:[a-zA-Z_]+:\{.*?\}\]", "", raw_text).strip()

                    return VoiceChatResponse(
                        text=clean_text,
                        language=detected_lang,
                        tool_executions=tool_executions,
                        model=candidate_model,
                        source="gemini",
                    )
            else:
                last_error = f"Gemini API {candidate_model} returned {resp.status_code}: {resp.text}"
                logger.warning(last_error)

        except Exception as exc:
            last_error = str(exc)
            logger.error(f"[Voice] Error calling Gemini API {candidate_model}: {exc}")

    # Fallback if Gemini calls failed
    logger.warning(f"[Voice] Gemini call failed ({last_error}), using fallback.")
    return _offline_cyclone_fallback(body.message, detected_lang, body.user_location)


def _offline_cyclone_fallback(
    message: str,
    lang: str,
    user_loc: Optional[UserLocation] = None
) -> VoiceChatResponse:
    """Local intelligent fallback if Gemini API is unreachable or on standby."""
    q = message.lower()
    tool_executions = []

    try:
        storms = get_mock_storms()
    except Exception:
        storms = []

    # 1. If NO storms are currently tracked in the system
    if not storms:
        if user_loc:
            city_label = user_loc.city or "your area"
            if lang == "bn":
                text = f"{city_label} ({user_loc.lat:.2f}°N, {user_loc.lon:.2f}°E) অঞ্চলে বর্তমানে কোনো সক্রিয় ঘূর্ণিঝড়ের প্রভাব বা সতর্কতা নেই। উত্তর ভারত মহাসাগর অববাহিকা বর্তমানে শান্ত রয়েছে। আপনার এআই মডেল থেকে নতুন পূর্বাভাস আসার সাথে সাথে তা প্রদর্শিত হবে।"
            elif lang == "hi":
                text = f"{city_label} ({user_loc.lat:.2f}°N, {user_loc.lon:.2f}°E) क्षेत्र में वर्तमान में किसी भी सक्रिय चक्रवात का कोई खतरा नहीं है। उत्तर हिंद महासागर बेसिन शांत है। जैसे ही आपके नए एআই मॉडल से डेटा प्राप्त होगा, सिस्टम तुरंत सतर्क करेगा।"
            else:
                text = f"From {city_label} ({user_loc.lat:.2f}°N, {user_loc.lon:.2f}°E), there are currently no active tropical cyclones tracked in the North Indian Ocean basin. Your area is clear of cyclonic threats. The dashboard and AI intelligence modules are on standby ready for your model data feeds."
        else:
            if lang == "bn":
                text = "বর্তমানে উত্তর ভারত মহাসাগরে কোনো সক্রিয় ঘূর্ণিঝড় চিহ্নিত নেই। ড্যাশবোর্ড এবং এআই সহকারী প্রস্তুত রয়েছে — আপনার এআই মডেলের ডেটাসেট সংযুক্ত হলেই তাৎক্ষণিক পূর্বাভাসের তথ্য এখানে প্রদর্শিত হবে।"
            elif lang == "hi":
                text = "वर्तमान में उत्तर हिंद महासागर में कोई सक्रिय चक्रवात नहीं है। सिस्टम पूरी तरह तैयार है और आपके एআই मॉडल पाइपलाइन से डेटा आते ही लाइव विश्लेषण शुरू हो जाएगा।"
            else:
                text = "Currently, no active tropical cyclones are tracked in the North Indian Ocean basin. The dashboard is on standby and ready to ingest live cyclone predictions from your AI model."

        return VoiceChatResponse(
            text=text,
            language=lang,
            tool_executions=[],
            model="offline-fallback",
            source="fallback",
        )

    # 2. If storms are present, evaluate dynamically
    is_area_query = any(w in q for w in [
        "near", "nearest", "closest", "area", "here", "kahan", "mere yahan", "amader ekhane",
        "bipod", "khatra", "danger", "safe", "distance", "city", "forecast"
    ])

    if is_area_query and user_loc is not None:
        ranked = []
        for s in storms:
            s_lat = float(s.get("current_lat", 0.0))
            s_lon = float(s.get("current_lon", 0.0))
            dist = calculate_distance_km(user_loc.lat, user_loc.lon, s_lat, s_lon)
            bearing = calculate_bearing(user_loc.lat, user_loc.lon, s_lat, s_lon)
            ranked.append((dist, bearing, s))

        ranked.sort(key=lambda x: x[0])
        closest_dist, closest_bearing, closest_storm = ranked[0]
        s_name = closest_storm.get("name", "Cyclone")
        s_cat = closest_storm.get("category", "CS")
        s_wind = closest_storm.get("max_wind_speed", 75)
        wind_kmh = int(s_wind * 1.852)
        city_label = user_loc.city or "your area"

        if closest_dist < 150:
            threat_en = "Immediate Danger Zone"
            threat_hi = "गंभीर खतरा क्षेत्र"
            threat_bn = "চরম বিপদ অঞ্চল"
            advisory_en = "IMD Emergency Warning: Severe gale winds and storm surges expected. Move to certified cyclone shelters."
            advisory_hi = "IMD आपातकालीन चेतावनी: अत्यधिक तेज़ हवाएं और भारी बारिश। तुरंत सुरक्षित पक्के आश्रयों में जाएं।"
            advisory_bn = "IMD জরুরি সতর্কতা: প্রবল ঝড়ো হাওয়া ও জলোচ্ছ্বাসের আশঙ্কা। নিচু এলাকা ছেড়ে দ্রুত আশ্রয়কেন্দ্রে চলে যান।"
        elif closest_dist < 350:
            threat_en = "High Warning Zone"
            threat_hi = "उच्च चेतावनी क्षेत्र"
            threat_bn = "উচ্চ সতর্কতা অঞ্চল"
            advisory_en = "IMD Warning: Outer rainbands and squally winds likely. Fishermen are strictly advised not to venture into sea."
            advisory_hi = "IMD चेतावनी: तूफानी बारिश और तेज़ हवाएं चलेंगी। मछुआरों को समुद्र में न जाने की सख्त सलाह है।"
            advisory_bn = "IMD সতর্কতা: ভারী বৃষ্টি ও দমকা হাওয়া বয়ে যাবে। মৎস্যজীবীদের সমুদ্রে যাওয়ার ওপর নিষেধাজ্ঞা জারি করা হয়েছে।"
        elif closest_dist < 700:
            threat_en = "Regional Watch Zone"
            threat_hi = "क्षेत्रीय निगरानी क्षेत्र"
            threat_bn = "আঞ্চলিক নজরদারি অঞ্চল"
            advisory_en = "IMD Watch: Expect overcast skies and moderate coastal showers."
            advisory_hi = "IMD निगरानी: बादल छाए रहेंगे और तटीय क्षेत्रों में बारिश हो सकती है।"
            advisory_bn = "IMD নজরদারি: মেঘলা আকাশ ও উপকূলীয় অঞ্চলে বৃষ্টির সম্ভাবনা।"
        else:
            threat_en = "Safe Distance"
            threat_hi = "सुरक्षित दूरी"
            threat_bn = "নিরাপদ দূরত্ব"
            advisory_en = "No direct cyclonic hazard expected at your coordinates."
            advisory_hi = "वर्तमान में आपके स्थान पर सीधे चक्रवात का कोई बड़ा खतरा नहीं है।"
            advisory_bn = "আপনার স্থানে সরাসরি কোনো ঘূর্ণিঝড়ের প্রভাবের আশঙ্কা নেই।"

        tool_executions.append(
            ToolExecutionItem(id="call_nav", name="navigate_to_storm", args={"storm_id": s_name})
        )

        if lang == "bn":
            text = f"{city_label} ({user_loc.lat:.2f}°N, {user_loc.lon:.2f}°E) থেকে নিকটতম ঘূর্ণিঝড় হল {s_name} ({s_cat}), যা প্রায় {closest_dist} কিমি {closest_bearing} দিকে অবস্থান করছে। বিপদের মাত্রা: {threat_bn}। বাতাসের গতিবেগ {s_wind} নট (~{wind_kmh} কিমি/ঘণ্টা)। {advisory_bn}"
        elif lang == "hi":
            text = f"{city_label} ({user_loc.lat:.2f}°N, {user_loc.lon:.2f}°E) से आपके सबसे नजदीकी चक्रवात {s_name} ({s_cat}) है, जो लगभग {closest_dist} किमी {closest_bearing} दिशा में है। खतरा स्तर: {threat_hi}। हवा की रफ्तार {s_wind} नॉट्स (~{wind_kmh} किमी/घंटा) है। {advisory_hi}"
        else:
            text = f"From {city_label} ({user_loc.lat:.2f}°N, {user_loc.lon:.2f}°E), the closest cyclone is {s_name} ({s_cat}), approximately {closest_dist} km to the {closest_bearing}. Status: {threat_en}. Sustained winds: {s_wind} kts (~{wind_kmh} km/h). {advisory_en}"

        return VoiceChatResponse(
            text=text,
            language=lang,
            tool_executions=tool_executions,
            model="offline-fallback",
            source="fallback",
        )

    # 3. Dynamic match against any ingested storm name
    for s in storms:
        s_name = s.get("name", "")
        if s_name.lower() in q:
            tool_executions.append(
                ToolExecutionItem(id="call_nav", name="navigate_to_storm", args={"storm_id": s_name})
            )
            s_cat = s.get("category", "CS")
            s_wind = s.get("max_wind_speed", 0)
            s_pres = s.get("min_pressure", 1000)
            text = f"{s_name} is currently a {s_cat} with sustained winds of {s_wind} knots and central pressure of {s_pres} hPa."
            return VoiceChatResponse(
                text=text,
                language=lang,
                tool_executions=tool_executions,
                model="offline-fallback",
                source="fallback",
            )

    # Default fallback
    first_storm_name = storms[0].get("name", "Active Cyclone")
    text = f"I am your Cyclone Intelligence Assistant. Currently tracking {len(storms)} system(s) including {first_storm_name}. Ask for proximity, danger level, or track forecasts."
    return VoiceChatResponse(
        text=text,
        language=lang,
        tool_executions=tool_executions,
        model="offline-fallback",
        source="fallback",
    )
