from __future__ import annotations

import base64
import datetime as dt
import json
import os
import re
import urllib.parse
import urllib.request
from dataclasses import dataclass, field
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any

from talent_scouting_intel.utils.io import ensure_parent, read_csv, resolve_path, write_csv
from talent_scouting_intel.utils.math_utils import clamp01

NAME_RE = re.compile(r"[^a-z0-9]+")
TOKEN_RE = re.compile(r"[a-z0-9]+")
SPOTIFY_ARTIST_RE = re.compile(r"spotify\.com/artist/([A-Za-z0-9]+)")
SPOTIFY_TRACK_RE = re.compile(r"spotify\.com/track/([A-Za-z0-9]+)")
MUSICBRAINZ_ID_RE = re.compile(
    r"\b([0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12})\b",
    flags=re.IGNORECASE,
)

STOPWORDS = {
    "and",
    "the",
    "with",
    "from",
    "your",
    "that",
    "this",
    "into",
    "over",
    "around",
    "about",
    "artist",
    "track",
    "song",
    "music",
    "moment",
    "breakout",
    "early",
    "very",
    "more",
    "less",
    "when",
    "where",
    "what",
    "then",
    "than",
    "their",
    "they",
    "them",
    "has",
    "have",
    "had",
    "was",
    "were",
    "for",
    "not",
    "you",
}

VECTOR_KEYS = ("shortform", "streaming", "video", "network", "depth", "geo")
KNOWN_GENRES = {
    "pop",
    "indie pop",
    "singer-songwriter",
    "country-pop",
    "indie folk",
    "alt rock",
    "other",
}


@dataclass
class IdentityInfo:
    spotify_artist_id: str = ""
    spotify_track_id: str = ""
    musicbrainz_artist_id: str = ""
    source: str = ""
    confidence: float = 0.0


@dataclass
class AffinityRecord:
    artist_name: str
    norm_artist_name: str
    primary_genre: str
    priority: int
    affinity_anchor: str
    reason: str
    data_confidence: float
    source_url: str
    identity: IdentityInfo = field(default_factory=IdentityInfo)


@dataclass
class BreakoutTemplate:
    artist_name: str
    norm_artist_name: str
    primary_genre: str
    breakout_window_estimate: str
    breakout_track_or_moment: str
    platform_path: str
    regions_momentum: str
    notes: str
    evidence_url_primary: str
    evidence_url_secondary: str
    vector: dict[str, float]
    identity: IdentityInfo = field(default_factory=IdentityInfo)


@dataclass
class PriorContext:
    enabled: bool
    affinity_records: list[AffinityRecord]
    breakout_templates: list[BreakoutTemplate]
    affinity_by_norm: dict[str, AffinityRecord]
    genre_profiles: dict[str, dict[str, float]]
    genre_profile_mass: dict[str, float]
    genre_baseline: dict[str, float]
    global_profile: dict[str, float]
    global_profile_mass: float
    identity_map: dict[str, IdentityInfo]


def _empty_context() -> PriorContext:
    return PriorContext(
        enabled=False,
        affinity_records=[],
        breakout_templates=[],
        affinity_by_norm={},
        genre_profiles={},
        genre_profile_mass={},
        genre_baseline={},
        global_profile={},
        global_profile_mass=1.0,
        identity_map={},
    )


def _normalize(text: str) -> str:
    return NAME_RE.sub("", str(text).strip().lower())


def _coalesce(row: dict[str, Any], *keys: str, default: str = "") -> str:
    for key in keys:
        value = str(row.get(key, "")).strip()
        if value:
            return value
    return default


def _normalize_genre(raw: str) -> str:
    text = str(raw or "").strip().lower()
    if not text:
        return "other"
    text = text.replace("_", " ").replace("/", " ").replace("+", " ").replace("|", " ")
    tokens = [token for token in TOKEN_RE.findall(text) if token]
    joined = " ".join(tokens)
    checks = [
        ("country pop", "country-pop"),
        ("singer songwriter", "singer-songwriter"),
        ("indie folk", "indie folk"),
        ("indie pop", "indie pop"),
        ("alt rock", "alt rock"),
        ("alternative rock", "alt rock"),
        ("pop", "pop"),
        ("folk", "indie folk"),
        ("rock", "alt rock"),
    ]
    for pattern, label in checks:
        if pattern in joined:
            return label
    return "other"


def _tokenize(text: str) -> list[str]:
    out: list[str] = []
    for token in TOKEN_RE.findall(str(text).lower()):
        if len(token) < 3:
            continue
        if token in STOPWORDS:
            continue
        if token.isdigit():
            continue
        out.append(token)
    return out


def _float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return default


def _int(value: Any, default: int = 0) -> int:
    try:
        return int(float(value))
    except Exception:
        return default


def _profile_mass(profile: dict[str, float], top_n: int = 40) -> float:
    if not profile:
        return 1.0
    return max(1e-6, sum(sorted(profile.values(), reverse=True)[:top_n]))


def _profile_overlap(tokens: list[str], profile: dict[str, float], mass: float) -> float:
    if not tokens or not profile:
        return 0.0
    score = sum(profile.get(token, 0.0) for token in set(tokens))
    return clamp01(score / max(1e-6, mass))


def _name_similarity(a: str, b: str) -> float:
    if not a or not b:
        return 0.0
    if a == b:
        return 1.0
    return SequenceMatcher(a=a, b=b).ratio()


def _extract_spotify_artist_id(source_url: str) -> str:
    match = SPOTIFY_ARTIST_RE.search(source_url or "")
    if not match:
        return ""
    return str(match.group(1))


def _extract_spotify_track_id(source_url: str) -> str:
    match = SPOTIFY_TRACK_RE.search(source_url or "")
    if not match:
        return ""
    return str(match.group(1))


def _extract_musicbrainz_id(text: str) -> str:
    match = MUSICBRAINZ_ID_RE.search(text or "")
    if not match:
        return ""
    return str(match.group(1)).lower()


def _split_prefixed_id(raw_id: str, prefix: str) -> str:
    value = str(raw_id or "").strip()
    if value.startswith(prefix):
        return value[len(prefix) :]
    return ""


def _safe_request_json(
    url: str,
    *,
    headers: dict[str, str] | None = None,
    data: bytes | None = None,
    timeout: int = 4,
) -> dict[str, Any]:
    req = urllib.request.Request(url, data=data, headers=headers or {})
    with urllib.request.urlopen(req, timeout=timeout) as response:
        payload = response.read().decode("utf-8", errors="ignore")
    obj = json.loads(payload)
    return obj if isinstance(obj, dict) else {}


def _spotify_token(client_id: str, client_secret: str) -> str | None:
    token_url = "https://accounts.spotify.com/api/token"
    auth = base64.b64encode(f"{client_id}:{client_secret}".encode("utf-8")).decode("ascii")
    headers = {
        "Authorization": f"Basic {auth}",
        "Content-Type": "application/x-www-form-urlencoded",
    }
    data = urllib.parse.urlencode({"grant_type": "client_credentials"}).encode("utf-8")
    try:
        payload = _safe_request_json(token_url, headers=headers, data=data)
    except Exception:
        return None
    token = payload.get("access_token")
    return str(token) if token else None


def _spotify_search_artist(name: str, token: str) -> tuple[str, float]:
    params = urllib.parse.urlencode({"q": name, "type": "artist", "limit": 1})
    url = f"https://api.spotify.com/v1/search?{params}"
    headers = {
        "Authorization": f"Bearer {token}",
        "User-Agent": "tsi-priors/0.1",
    }
    try:
        payload = _safe_request_json(url, headers=headers)
    except Exception:
        return "", 0.0

    items = ((payload.get("artists") or {}).get("items")) or []
    if not isinstance(items, list) or not items:
        return "", 0.0

    top = items[0] if isinstance(items[0], dict) else {}
    artist_id = str(top.get("id", "")).strip()
    popularity = _float(top.get("popularity", 0.0))
    if not artist_id:
        return "", 0.0
    return artist_id, clamp01(popularity / 100.0)


def _musicbrainz_search_artist(name: str) -> tuple[str, float]:
    params = urllib.parse.urlencode({"query": f"artist:{name}", "fmt": "json", "limit": 1})
    url = f"https://musicbrainz.org/ws/2/artist/?{params}"
    headers = {
        "User-Agent": "tsi-priors/0.1 (research@localhost)",
        "Accept": "application/json",
    }
    try:
        payload = _safe_request_json(url, headers=headers, timeout=5)
    except Exception:
        return "", 0.0

    artists = payload.get("artists", [])
    if not isinstance(artists, list) or not artists:
        return "", 0.0
    top = artists[0] if isinstance(artists[0], dict) else {}
    mbid = str(top.get("id", "")).strip()
    score = _float(top.get("score", 0.0)) / 100.0
    if not mbid:
        return "", 0.0
    return mbid, clamp01(score)


def _load_identity_cache(path: Path) -> dict[str, dict[str, Any]]:
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    if not isinstance(payload, dict):
        return {}
    artists = payload.get("artists", {})
    return artists if isinstance(artists, dict) else {}


def _write_identity_cache(path: Path, artists: dict[str, dict[str, Any]]) -> None:
    ensure_parent(path)
    payload = {"version": 1, "updated_at": dt.datetime.now(dt.UTC).isoformat(), "artists": artists}
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=True), encoding="utf-8")


def _best_known_identity(
    artist_name: str,
    known: dict[str, IdentityInfo],
    min_similarity: float,
) -> IdentityInfo | None:
    norm = _normalize(artist_name)
    exact = known.get(norm)
    if exact and (exact.spotify_artist_id or exact.musicbrainz_artist_id):
        return exact

    best_key = ""
    best_score = 0.0
    for key, value in known.items():
        if not (value.spotify_artist_id or value.musicbrainz_artist_id):
            continue
        score = _name_similarity(norm, key)
        if score > best_score:
            best_score = score
            best_key = key

    if best_key and best_score >= min_similarity:
        value = known[best_key]
        return IdentityInfo(
            spotify_artist_id=value.spotify_artist_id,
            spotify_track_id=value.spotify_track_id,
            musicbrainz_artist_id=value.musicbrainz_artist_id,
            source=f"{value.source}:fuzzy",
            confidence=best_score,
        )
    return None


def _resolve_identity_map(
    artist_names: set[str],
    known: dict[str, IdentityInfo],
    config: dict[str, Any],
    root: Path,
) -> dict[str, IdentityInfo]:
    priors_cfg = config.get("priors", {})
    id_cfg = priors_cfg.get("identity_resolution", {})
    if not bool(priors_cfg.get("enabled", True)) or not bool(id_cfg.get("enabled", True)):
        return {}

    cache_path_value = config.get("paths", {}).get("priors_identity_cache", "outputs/state/user_priors_identity_cache.json")
    cache_path = resolve_path(str(cache_path_value), root)
    cache_rows = _load_identity_cache(cache_path)
    min_similarity = _float(id_cfg.get("min_name_similarity", 0.92))

    resolved: dict[str, IdentityInfo] = {}
    unresolved: list[tuple[str, str]] = []

    for artist_name in sorted(artist_names):
        norm = _normalize(artist_name)
        if not norm:
            continue

        known_info = _best_known_identity(artist_name, known, min_similarity)
        if known_info and (known_info.spotify_artist_id or known_info.musicbrainz_artist_id):
            resolved[norm] = known_info
            continue

        cached = cache_rows.get(norm, {})
        cached_info = IdentityInfo(
            spotify_artist_id=str(cached.get("spotify_artist_id", "")).strip(),
            spotify_track_id=str(cached.get("spotify_track_id", "")).strip(),
            musicbrainz_artist_id=str(cached.get("musicbrainz_artist_id", "")).strip(),
            source=str(cached.get("source", "cache")).strip(),
            confidence=_float(cached.get("confidence", 0.0)),
        )
        if cached_info.spotify_artist_id or cached_info.musicbrainz_artist_id:
            resolved[norm] = cached_info
            continue

        unresolved.append((artist_name, norm))

    if not bool(id_cfg.get("network_lookups", False)):
        for artist_name, norm in unresolved:
            resolved[norm] = IdentityInfo(source="unresolved", confidence=0.0)
            cache_rows[norm] = {
                "artist_name": artist_name,
                "spotify_artist_id": "",
                "spotify_track_id": "",
                "musicbrainz_artist_id": "",
                "source": "unresolved",
                "confidence": 0.0,
                "last_attempted": dt.datetime.now(dt.UTC).isoformat(),
            }
        _write_identity_cache(cache_path, cache_rows)
        return resolved

    max_unresolved = max(0, _int(id_cfg.get("max_unresolved_per_run", 12), 12))
    unresolved = unresolved[:max_unresolved]

    spotify_token: str | None = None
    use_spotify = bool(id_cfg.get("use_spotify_api", False))
    if use_spotify:
        cid = os.getenv(str(id_cfg.get("spotify_client_id_env", "SPOTIFY_CLIENT_ID")), "")
        csecret = os.getenv(str(id_cfg.get("spotify_client_secret_env", "SPOTIFY_CLIENT_SECRET")), "")
        if cid and csecret:
            spotify_token = _spotify_token(cid, csecret)

    use_musicbrainz = bool(id_cfg.get("use_musicbrainz_api", False))

    for artist_name, norm in unresolved:
        info = IdentityInfo(source="unresolved", confidence=0.0)

        if spotify_token and use_spotify:
            sp_id, sp_conf = _spotify_search_artist(artist_name, spotify_token)
            if sp_id:
                info.spotify_artist_id = sp_id
                info.source = "spotify_api"
                info.confidence = sp_conf

        if use_musicbrainz and not info.musicbrainz_artist_id:
            mbid, mb_conf = _musicbrainz_search_artist(artist_name)
            if mbid:
                info.musicbrainz_artist_id = mbid
                if not info.source or info.source == "unresolved":
                    info.source = "musicbrainz_api"
                    info.confidence = mb_conf
                else:
                    info.source = f"{info.source}+musicbrainz_api"
                    info.confidence = max(info.confidence, mb_conf)

        resolved[norm] = info
        cache_rows[norm] = {
            "artist_name": artist_name,
            "spotify_artist_id": info.spotify_artist_id,
            "spotify_track_id": info.spotify_track_id,
            "musicbrainz_artist_id": info.musicbrainz_artist_id,
            "source": info.source,
            "confidence": round(info.confidence, 6),
            "last_attempted": dt.datetime.now(dt.UTC).isoformat(),
        }

    _write_identity_cache(cache_path, cache_rows)
    return resolved


def _known_identities_from_features(feature_rows: list[dict[str, Any]]) -> tuple[dict[str, IdentityInfo], dict[str, list[tuple[str, str]]]]:
    artists: dict[str, IdentityInfo] = {}
    tracks_by_artist: dict[str, list[tuple[str, str]]] = {}

    for row in feature_rows:
        artist_name = str(row.get("artist_name", "")).strip()
        track_name = str(row.get("track_name", "")).strip()
        artist_norm = _normalize(artist_name)
        track_norm = _normalize(track_name)
        if not artist_norm:
            continue

        artist_id = str(row.get("artist_id", "")).strip()
        track_id = str(row.get("track_id", "")).strip()
        metadata_text = str(row.get("metadata_text", ""))

        spotify_artist_id = _split_prefixed_id(artist_id, "sp_")
        spotify_track_id = _split_prefixed_id(track_id, "sp_")
        mbid_artist = _split_prefixed_id(artist_id, "mb_")
        if not mbid_artist:
            mbid_artist = _extract_musicbrainz_id(metadata_text)

        existing = artists.get(artist_norm, IdentityInfo())
        if spotify_artist_id and not existing.spotify_artist_id:
            existing.spotify_artist_id = spotify_artist_id
            existing.source = "pipeline"
            existing.confidence = 1.0
        if mbid_artist and not existing.musicbrainz_artist_id:
            existing.musicbrainz_artist_id = mbid_artist
            existing.source = "pipeline"
            existing.confidence = 1.0
        artists[artist_norm] = existing

        if artist_norm and track_norm and spotify_track_id:
            tracks_by_artist.setdefault(artist_norm, []).append((track_norm, spotify_track_id))

    return artists, tracks_by_artist


def _resolve_track_hint(
    artist_name: str,
    text_hint: str,
    known_tracks: dict[str, list[tuple[str, str]]],
) -> str:
    artist_norm = _normalize(artist_name)
    hint_norm = _normalize(text_hint)
    if not artist_norm or not hint_norm:
        return ""
    candidates = known_tracks.get(artist_norm, [])
    if not candidates:
        return ""

    # Fast path for direct containment.
    for track_norm, track_id in candidates:
        if track_norm and (track_norm in hint_norm or hint_norm in track_norm):
            return track_id

    # Fallback to fuzzy.
    best_id = ""
    best_sim = 0.0
    for track_norm, track_id in candidates:
        sim = _name_similarity(track_norm, hint_norm)
        if sim > best_sim:
            best_sim = sim
            best_id = track_id
    if best_sim >= 0.9:
        return best_id
    return ""


def _template_vector(
    primary_genre: str,
    platform_path: str,
    breakout_moment: str,
    regions_momentum: str,
    notes: str,
) -> dict[str, float]:
    text = " ".join([platform_path, breakout_moment, notes]).lower()
    regions = str(regions_momentum or "").lower()

    vec = {
        "shortform": 0.1,
        "streaming": 0.22,
        "video": 0.1,
        "network": 0.15,
        "depth": 0.1,
        "geo": 0.1,
    }

    shortform_terms = ["tiktok", "short-form", "short form", "viral", "sound trend", "reel", "shorts", "fyp"]
    streaming_terms = ["spotify", "streaming", "playlist", "listener", "save", "algorithmic"]
    video_terms = ["youtube", "video", "channel", "live session", "visualizer"]
    network_terms = ["press", "radio", "credits", "feature", "collab", "tour", "label", "newsletter", "blog"]
    depth_terms = ["on repeat", "lyrics", "emotional", "fan", "repeat listening", "chorus", "story"]

    if any(term in text for term in shortform_terms):
        vec["shortform"] += 0.45
        vec["streaming"] += 0.12
    if any(term in text for term in streaming_terms):
        vec["streaming"] += 0.34
    if any(term in text for term in video_terms):
        vec["video"] += 0.35
        vec["streaming"] += 0.05
    if any(term in text for term in network_terms):
        vec["network"] += 0.35
    if any(term in text for term in depth_terms):
        vec["depth"] += 0.3

    geo_markets = sum(1 for market in ["us", "uk", "au", "se"] if market in regions)
    if geo_markets == 1:
        vec["geo"] += 0.18
    elif geo_markets >= 2:
        vec["geo"] += 0.33

    genre = str(primary_genre or "").strip().lower()
    if genre in {"singer-songwriter", "indie folk"}:
        vec["depth"] += 0.08
    if genre in {"pop", "indie pop"}:
        vec["shortform"] += 0.08
        vec["streaming"] += 0.06
    if genre == "country-pop":
        vec["geo"] += 0.08
        vec["network"] += 0.04
    if genre == "alt rock":
        vec["network"] += 0.08
        vec["video"] += 0.05

    return {key: clamp01(vec[key]) for key in VECTOR_KEYS}


def _cosine_similarity(a: dict[str, float], b: dict[str, float]) -> float:
    dot = sum(a[key] * b[key] for key in VECTOR_KEYS)
    mag_a = sum(a[key] * a[key] for key in VECTOR_KEYS) ** 0.5
    mag_b = sum(b[key] * b[key] for key in VECTOR_KEYS) ** 0.5
    if mag_a <= 0 or mag_b <= 0:
        return 0.0
    return clamp01(dot / (mag_a * mag_b))


def _template_recency_weight(window_estimate: str) -> float:
    text = str(window_estimate or "").strip()
    if not text:
        return 1.0
    year = 0
    try:
        year = int(text[:4])
    except Exception:
        year = 0
    if year <= 0:
        return 1.0
    current_year = dt.date.today().year
    delta = current_year - year
    if delta <= 2:
        return 1.03
    if delta <= 6:
        return 1.0
    return 0.96


def _build_resolution_rows(
    affinity_records: list[AffinityRecord],
    breakout_templates: list[BreakoutTemplate],
) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for row in affinity_records:
        out.append(
            {
                "dataset": "affinity_artists",
                "artist_name": row.artist_name,
                "primary_genre": row.primary_genre,
                "spotify_artist_id": row.identity.spotify_artist_id,
                "spotify_track_id": row.identity.spotify_track_id,
                "musicbrainz_artist_id": row.identity.musicbrainz_artist_id,
                "resolution_source": row.identity.source,
                "resolution_confidence": round(row.identity.confidence, 6),
            }
        )
    for row in breakout_templates:
        out.append(
            {
                "dataset": "historical_breakouts",
                "artist_name": row.artist_name,
                "primary_genre": row.primary_genre,
                "spotify_artist_id": row.identity.spotify_artist_id,
                "spotify_track_id": row.identity.spotify_track_id,
                "musicbrainz_artist_id": row.identity.musicbrainz_artist_id,
                "resolution_source": row.identity.source,
                "resolution_confidence": round(row.identity.confidence, 6),
            }
        )
    out.sort(key=lambda item: (str(item["dataset"]), str(item["artist_name"]).lower()))
    return out


def _declared_columns(dictionary_rows: list[dict[str, Any]], dataset_file_name: str) -> set[str]:
    expected: set[str] = set()
    for row in dictionary_rows:
        if str(row.get("file", "")).strip() != dataset_file_name:
            continue
        column = str(row.get("column", "")).strip()
        if column:
            expected.add(column)
    return expected


def build_prior_context(
    config: dict[str, Any],
    project_root: Path,
    feature_rows: list[dict[str, Any]],
) -> PriorContext:
    priors_cfg = config.get("priors", {})
    if not bool(priors_cfg.get("enabled", True)):
        return _empty_context()

    paths_cfg = config.get("paths", {})
    affinity_path = resolve_path(str(paths_cfg.get("affinity_artists", "data/user/affinity_artists_updated.csv")), project_root)
    breakout_path = resolve_path(str(paths_cfg.get("breakout_templates", "data/user/historical_breakouts_updated.csv")), project_root)
    dictionary_path = resolve_path(str(paths_cfg.get("user_data_dictionary", "data/user/datasets_data_dictionary.csv")), project_root)

    affinity_rows = read_csv(affinity_path)
    breakout_rows = read_csv(breakout_path)
    dictionary_rows = read_csv(dictionary_path)

    if dictionary_rows:
        affinity_expected = _declared_columns(dictionary_rows, affinity_path.name)
        breakout_expected = _declared_columns(dictionary_rows, breakout_path.name)
        if affinity_expected:
            for row in affinity_rows:
                for column in affinity_expected:
                    row.setdefault(column, "")
        if breakout_expected:
            for row in breakout_rows:
                for column in breakout_expected:
                    row.setdefault(column, "")

    if not affinity_rows and not breakout_rows:
        return _empty_context()

    known_identities, known_tracks = _known_identities_from_features(feature_rows)
    artist_names = {
        str(row.get("artist_name", "")).strip()
        for row in affinity_rows + breakout_rows
        if str(row.get("artist_name", "")).strip()
    }
    identity_map = _resolve_identity_map(artist_names, known_identities, config, project_root)

    affinity_records: list[AffinityRecord] = []
    for row in affinity_rows:
        artist_name = str(row.get("artist_name", "")).strip()
        if not artist_name:
            continue
        norm = _normalize(artist_name)
        identity = identity_map.get(norm, IdentityInfo())

        source_url = _coalesce(row, "source_url", default="")
        source_artist_id = _extract_spotify_artist_id(source_url)
        source_track_id = _extract_spotify_track_id(source_url)

        if source_artist_id and not identity.spotify_artist_id:
            identity.spotify_artist_id = source_artist_id
            identity.source = identity.source or "source_url"
            identity.confidence = max(identity.confidence, 0.9)
        if source_track_id and not identity.spotify_track_id:
            identity.spotify_track_id = source_track_id

        track_hint = _coalesce(row, "affinity_anchor", "defining_song", "anchor_track", default="")
        if track_hint and not identity.spotify_track_id:
            identity.spotify_track_id = _resolve_track_hint(artist_name, track_hint, known_tracks)

        raw_genre = _coalesce(row, "primary_genre", "genre", "lane", default="other")
        normalized_genre = _normalize_genre(raw_genre)
        data_confidence = _float(row.get("data_confidence", 0.0), 0.0)
        if data_confidence <= 0:
            # Prioritize curated rows if explicit confidence is unavailable.
            data_confidence = clamp01(0.55 + 0.08 * max(1, min(5, _int(row.get("priority", 3), 3))))

        affinity_records.append(
            AffinityRecord(
                artist_name=artist_name,
                norm_artist_name=norm,
                primary_genre=normalized_genre if normalized_genre in KNOWN_GENRES else "other",
                priority=max(1, min(5, _int(row.get("priority", 3), 3))),
                affinity_anchor=track_hint,
                reason=_coalesce(row, "reason", "notes", default=""),
                data_confidence=clamp01(data_confidence),
                source_url=source_url,
                identity=identity,
            )
        )

    breakout_templates: list[BreakoutTemplate] = []
    for row in breakout_rows:
        artist_name = str(row.get("artist_name", "")).strip()
        if not artist_name:
            continue
        norm = _normalize(artist_name)
        identity = identity_map.get(norm, IdentityInfo())

        evidence_primary = _coalesce(row, "evidence_url_primary", default="")
        evidence_secondary = _coalesce(row, "evidence_url_secondary", default="")

        source_artist_id = _extract_spotify_artist_id(evidence_primary) or _extract_spotify_artist_id(evidence_secondary)
        source_track_id = _extract_spotify_track_id(evidence_primary) or _extract_spotify_track_id(evidence_secondary)
        if source_artist_id and not identity.spotify_artist_id:
            identity.spotify_artist_id = source_artist_id
            identity.source = identity.source or "evidence_url"
            identity.confidence = max(identity.confidence, 0.9)
        if source_track_id and not identity.spotify_track_id:
            identity.spotify_track_id = source_track_id

        breakout_moment = _coalesce(row, "breakout_track_or_moment", "track_name", "notes", default="")
        if breakout_moment and not identity.spotify_track_id:
            identity.spotify_track_id = _resolve_track_hint(artist_name, breakout_moment, known_tracks)

        primary_genre = _normalize_genre(_coalesce(row, "primary_genre", "lane", default="other"))
        window_estimate = _coalesce(row, "breakout_window_estimate", "approx_window", default="")
        platform_path = _coalesce(row, "platform_path", "primary_platform", default="")
        regions = _coalesce(row, "regions_momentum", "region", default="")

        template = BreakoutTemplate(
            artist_name=artist_name,
            norm_artist_name=norm,
            primary_genre=primary_genre if primary_genre in KNOWN_GENRES else "other",
            breakout_window_estimate=window_estimate,
            breakout_track_or_moment=breakout_moment,
            platform_path=platform_path,
            regions_momentum=regions,
            notes=str(row.get("notes", "")).strip(),
            evidence_url_primary=evidence_primary,
            evidence_url_secondary=evidence_secondary,
            vector=_template_vector(
                primary_genre,
                platform_path,
                breakout_moment,
                regions,
                str(row.get("notes", "")),
            ),
            identity=identity,
        )
        breakout_templates.append(template)

    affinity_by_norm = {record.norm_artist_name: record for record in affinity_records}
    genre_profiles: dict[str, dict[str, float]] = {}
    genre_profile_mass: dict[str, float] = {}
    genre_baseline: dict[str, float] = {}
    global_profile: dict[str, float] = {}

    by_genre_records: dict[str, list[AffinityRecord]] = {}
    for record in affinity_records:
        by_genre_records.setdefault(record.primary_genre, []).append(record)
        tokens = _tokenize(" ".join([record.artist_name, record.affinity_anchor, record.reason]))
        weight = ((record.priority - 1) / 4.0) * (0.5 + 0.5 * record.data_confidence)
        for token in tokens:
            global_profile[token] = global_profile.get(token, 0.0) + weight
            genre_profiles.setdefault(record.primary_genre, {})
            genre_profiles[record.primary_genre][token] = genre_profiles[record.primary_genre].get(token, 0.0) + weight

    for genre, rows in by_genre_records.items():
        avg_priority = sum((row.priority - 1) / 4.0 for row in rows) / max(1, len(rows))
        avg_conf = sum(row.data_confidence for row in rows) / max(1, len(rows))
        genre_baseline[genre] = clamp01(0.65 * avg_priority + 0.35 * avg_conf)
        genre_profile_mass[genre] = _profile_mass(genre_profiles.get(genre, {}))

    if affinity_records:
        avg_priority = sum((row.priority - 1) / 4.0 for row in affinity_records) / len(affinity_records)
        avg_conf = sum(row.data_confidence for row in affinity_records) / len(affinity_records)
        genre_baseline["*"] = clamp01(0.65 * avg_priority + 0.35 * avg_conf)
    else:
        genre_baseline["*"] = 0.0
    global_mass = _profile_mass(global_profile)

    resolution_path_value = paths_cfg.get("priors_resolution_csv", "outputs/user_priors_resolution.csv")
    resolution_path = resolve_path(str(resolution_path_value), project_root)
    write_csv(resolution_path, _build_resolution_rows(affinity_records, breakout_templates))

    return PriorContext(
        enabled=True,
        affinity_records=affinity_records,
        breakout_templates=breakout_templates,
        affinity_by_norm=affinity_by_norm,
        genre_profiles=genre_profiles,
        genre_profile_mass=genre_profile_mass,
        genre_baseline=genre_baseline,
        global_profile=global_profile,
        global_profile_mass=global_mass,
        identity_map=identity_map,
    )


def _candidate_path_vector(row: dict[str, Any]) -> dict[str, float]:
    shortform_proxy = clamp01(_float(row.get("shortform_proxy_score", 0.0)))
    echo_score = clamp01(_float(row.get("echo_score", row.get("cross_platform_score", 0.0))))
    momentum = clamp01(_float(row.get("momentum_score", 0.0)))
    acceleration = clamp01(_float(row.get("acceleration_score", 0.0)))
    cross_platform = clamp01(_float(row.get("cross_platform_score", 0.0)))
    network = clamp01(_float(row.get("network_score", 0.0)))
    depth = clamp01(_float(row.get("depth_score", 0.0)))
    geo = clamp01(_float(row.get("geo_score", 0.0)))
    knowledge = clamp01(_float(row.get("knowledge_graph_score", 0.0)))
    specificity = clamp01(_float(row.get("comment_specificity", 0.0)))

    return {
        "shortform": clamp01(0.72 * shortform_proxy + 0.28 * echo_score),
        "streaming": clamp01(0.44 * momentum + 0.56 * acceleration),
        "video": clamp01(0.68 * cross_platform + 0.32 * echo_score),
        "network": clamp01(0.76 * network + 0.24 * knowledge),
        "depth": clamp01(0.82 * depth + 0.18 * specificity),
        "geo": geo,
    }


def _best_affinity_match(
    artist_name: str,
    context: PriorContext,
    min_similarity: float,
) -> tuple[AffinityRecord | None, float]:
    norm = _normalize(artist_name)
    exact = context.affinity_by_norm.get(norm)
    if exact:
        return exact, 1.0

    best: AffinityRecord | None = None
    best_sim = 0.0
    for record in context.affinity_records:
        sim = _name_similarity(norm, record.norm_artist_name)
        if sim > best_sim:
            best_sim = sim
            best = record

    if best and best_sim >= min_similarity:
        return best, best_sim
    return None, 0.0


def _score_affinity(
    row: dict[str, Any],
    candidate_genre: str,
    context: PriorContext,
    config: dict[str, Any],
) -> dict[str, Any]:
    affinity_cfg = config.get("priors", {}).get("affinity", {})
    direct_w = _float(affinity_cfg.get("direct_match_weight", 0.6), 0.6)
    genre_w = _float(affinity_cfg.get("genre_baseline_weight", 0.25), 0.25)
    text_w = _float(affinity_cfg.get("text_profile_weight", 0.15), 0.15)

    min_similarity = _float(config.get("priors", {}).get("identity_resolution", {}).get("min_name_similarity", 0.92), 0.92)
    match, similarity = _best_affinity_match(str(row.get("artist_name", "")), context, min_similarity)

    direct_score = 0.0
    match_artist = ""
    match_genre = ""
    match_reason = ""
    match_priority = 0
    match_spotify_artist_id = ""
    match_spotify_track_id = ""
    match_mbid = ""

    if match:
        priority_norm = (match.priority - 1) / 4.0
        if not candidate_genre:
            genre_align = 0.6
        elif match.primary_genre == candidate_genre:
            genre_align = 1.0
        elif match.primary_genre == "other":
            genre_align = 0.68
        else:
            genre_align = 0.42
        direct_score = clamp01(
            (0.52 * priority_norm + 0.28 * match.data_confidence + 0.2 * genre_align) * max(0.75, similarity)
        )
        match_artist = match.artist_name
        match_genre = match.primary_genre
        match_reason = match.reason
        match_priority = match.priority
        match_spotify_artist_id = match.identity.spotify_artist_id
        match_spotify_track_id = match.identity.spotify_track_id
        match_mbid = match.identity.musicbrainz_artist_id

    genre_key = candidate_genre if candidate_genre in context.genre_baseline else "*"
    genre_baseline = context.genre_baseline.get(genre_key, context.genre_baseline.get("*", 0.0))

    profile = context.genre_profiles.get(candidate_genre, context.global_profile)
    mass = context.genre_profile_mass.get(candidate_genre, context.global_profile_mass)
    text_tokens = _tokenize(
        " ".join(
            [
                str(row.get("track_name", "")),
                str(row.get("artist_name", "")),
                str(row.get("metadata_text", "")),
            ]
        )
    )
    text_profile_score = _profile_overlap(text_tokens, profile, mass)

    affinity_score = clamp01(direct_w * direct_score + genre_w * genre_baseline + text_w * text_profile_score)
    return {
        "affinity_score": round(affinity_score, 6),
        "affinity_direct_score": round(direct_score, 6),
        "affinity_genre_baseline": round(genre_baseline, 6),
        "affinity_text_score": round(text_profile_score, 6),
        "affinity_match_artist": match_artist,
        "affinity_match_genre": match_genre,
        "affinity_match_priority": int(match_priority),
        "affinity_match_similarity": round(similarity, 6),
        "affinity_match_reason": match_reason,
        "affinity_spotify_artist_id": match_spotify_artist_id,
        "affinity_spotify_track_id": match_spotify_track_id,
        "affinity_musicbrainz_artist_id": match_mbid,
    }


def _score_path_similarity(
    row: dict[str, Any],
    candidate_genre: str,
    context: PriorContext,
    config: dict[str, Any],
) -> dict[str, Any]:
    templates = context.breakout_templates
    if not templates:
        return {
            "path_similarity_score": 0.0,
            "path_template_artist": "",
            "path_template_genre": "",
            "path_template_window": "",
            "path_template_path": "",
            "path_template_notes": "",
            "path_template_similarity": 0.0,
            "path_template_spotify_artist_id": "",
            "path_template_spotify_track_id": "",
            "path_template_musicbrainz_artist_id": "",
        }

    path_cfg = config.get("priors", {}).get("path_similarity", {})
    genre_bonus = _float(path_cfg.get("genre_match_bonus", 0.12), 0.12)
    top_k = max(1, _int(path_cfg.get("top_k_templates", 3), 3))

    candidate_vec = _candidate_path_vector(row)
    scored: list[tuple[float, BreakoutTemplate]] = []
    for template in templates:
        sim = _cosine_similarity(candidate_vec, template.vector)
        if candidate_genre and template.primary_genre:
            if candidate_genre == template.primary_genre:
                sim *= 1.0 + genre_bonus
            elif template.primary_genre != "other":
                sim *= max(0.8, 1.0 - genre_bonus * 0.5)
        sim *= _template_recency_weight(template.breakout_window_estimate)
        scored.append((clamp01(sim), template))

    scored.sort(key=lambda item: item[0], reverse=True)
    top = scored[:top_k]
    weights = [1.0 / (idx + 1) for idx in range(len(top))]
    weight_total = sum(weights) or 1.0
    path_similarity_score = sum(score * weight for (score, _template), weight in zip(top, weights, strict=True)) / weight_total

    top_score, top_template = top[0]
    return {
        "path_similarity_score": round(clamp01(path_similarity_score), 6),
        "path_template_artist": top_template.artist_name,
        "path_template_genre": top_template.primary_genre,
        "path_template_window": top_template.breakout_window_estimate,
        "path_template_path": top_template.platform_path,
        "path_template_notes": top_template.notes,
        "path_template_similarity": round(clamp01(top_score), 6),
        "path_template_spotify_artist_id": top_template.identity.spotify_artist_id,
        "path_template_spotify_track_id": top_template.identity.spotify_track_id,
        "path_template_musicbrainz_artist_id": top_template.identity.musicbrainz_artist_id,
    }


def score_priors(
    row: dict[str, Any],
    candidate_genre: str,
    context: PriorContext,
    config: dict[str, Any],
) -> dict[str, Any]:
    if not context.enabled:
        return {
            "affinity_score": 0.0,
            "affinity_direct_score": 0.0,
            "affinity_genre_baseline": 0.0,
            "affinity_text_score": 0.0,
            "affinity_match_artist": "",
            "affinity_match_genre": "",
            "affinity_match_priority": 0,
            "affinity_match_similarity": 0.0,
            "affinity_match_reason": "",
            "affinity_spotify_artist_id": "",
            "affinity_spotify_track_id": "",
            "affinity_musicbrainz_artist_id": "",
            "path_similarity_score": 0.0,
            "path_template_artist": "",
            "path_template_genre": "",
            "path_template_window": "",
            "path_template_path": "",
            "path_template_notes": "",
            "path_template_similarity": 0.0,
            "path_template_spotify_artist_id": "",
            "path_template_spotify_track_id": "",
            "path_template_musicbrainz_artist_id": "",
        }

    affinity = _score_affinity(row, candidate_genre, context, config)
    path = _score_path_similarity(row, candidate_genre, context, config)
    merged = dict(affinity)
    merged.update(path)
    return merged
