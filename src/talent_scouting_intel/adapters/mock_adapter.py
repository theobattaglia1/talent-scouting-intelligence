from __future__ import annotations

import datetime as dt
import math
import random
from typing import Any

from talent_scouting_intel.models.types import Snapshot


RNG = random.Random(42)
REGIONS = ["Los Angeles", "Nashville", "Austin", "New York", "London", "Toronto", "Berlin", "Sydney"]

SPECIFIC_COMMENTS = [
    "that second verse lyric hit hard",
    "chorus melody is stuck in my head",
    "on repeat all week",
    "the bridge payoff is insane",
    "this storytelling feels cinematic",
    "the vocal crack on that line is perfect",
    "cant stop listening to this hook",
]

GENERIC_COMMENTS = ["fire", "nice", "wow", "cool", "love this", "great song"]

TASTEMAKERS = [
    {"id": "tmk_001", "name": "Bedroom Pop Radar", "platform": "spotify", "quality": 0.76},
    {"id": "tmk_002", "name": "Indie Folk Window", "platform": "spotify", "quality": 0.72},
    {"id": "tmk_003", "name": "Alt Rock Basement", "platform": "youtube", "quality": 0.69},
    {"id": "tmk_004", "name": "Songwriter Signal", "platform": "youtube", "quality": 0.78},
    {"id": "tmk_005", "name": "Nashville New Heat", "platform": "spotify", "quality": 0.8},
    {"id": "tmk_006", "name": "Late Night Finders", "platform": "youtube", "quality": 0.58},
]

TRACKS = [
    {
        "track_id": "trk_001",
        "track_name": "Neon Afternoon",
        "artist_id": "art_001",
        "artist_name": "Lena Vale",
        "artist_followers": 18500,
        "genre_hint": "indie pop dreamy synth",
        "pattern": "compound_breaker",
        "release_offset": 10,
        "collaborators": ["prod_mika"],
        "metadata_text": "indie pop bedroom synth hook chorus",
    },
    {
        "track_id": "trk_002",
        "track_name": "Backroad Static",
        "artist_id": "art_002",
        "artist_name": "Clay Mercer",
        "artist_followers": 39000,
        "genre_hint": "country-pop crossover",
        "pattern": "geo_breaker",
        "release_offset": 16,
        "collaborators": ["writer_rin"],
        "metadata_text": "country-pop nashville crossover chorus",
    },
    {
        "track_id": "trk_003",
        "track_name": "Paper Lantern",
        "artist_id": "art_003",
        "artist_name": "Mira Rowe",
        "artist_followers": 14000,
        "genre_hint": "singer-songwriter acoustic",
        "pattern": "steady_emerging",
        "release_offset": 8,
        "collaborators": ["prod_mika"],
        "metadata_text": "singer-songwriter lyric acoustic intimate",
    },
    {
        "track_id": "trk_004",
        "track_name": "Glass Motel",
        "artist_id": "art_004",
        "artist_name": "Hollow June",
        "artist_followers": 27000,
        "genre_hint": "alt rock guitar",
        "pattern": "compound_breaker",
        "release_offset": 19,
        "collaborators": ["gtr_sam"],
        "metadata_text": "alt rock riff guitar chorus",
    },
    {
        "track_id": "trk_005",
        "track_name": "Blue Thread",
        "artist_id": "art_005",
        "artist_name": "Northline Choir",
        "artist_followers": 9000,
        "genre_hint": "indie folk americana",
        "pattern": "steady_emerging",
        "release_offset": 14,
        "collaborators": ["writer_rin"],
        "metadata_text": "indie folk americana harmonies",
    },
    {
        "track_id": "trk_006",
        "track_name": "One Day Fire",
        "artist_id": "art_006",
        "artist_name": "VYRA",
        "artist_followers": 51000,
        "genre_hint": "pop",
        "pattern": "spike_only",
        "release_offset": 6,
        "collaborators": ["prod_flash"],
        "metadata_text": "pop anthem hook",
    },
    {
        "track_id": "trk_007",
        "track_name": "Push Campaign",
        "artist_id": "art_007",
        "artist_name": "Aster Dawn",
        "artist_followers": 120000,
        "genre_hint": "pop",
        "pattern": "paid_push",
        "release_offset": 12,
        "collaborators": ["prod_flash"],
        "metadata_text": "pop radio campaign",
    },
    {
        "track_id": "trk_008",
        "track_name": "Playlist Lift",
        "artist_id": "art_008",
        "artist_name": "The Sun Hours",
        "artist_followers": 60000,
        "genre_hint": "indie pop",
        "pattern": "playlist_dependent",
        "release_offset": 20,
        "collaborators": ["prod_mika"],
        "metadata_text": "indie pop synth",
    },
    {
        "track_id": "trk_009",
        "track_name": "Quiet Harvest",
        "artist_id": "art_009",
        "artist_name": "Rowan Pike",
        "artist_followers": 7200,
        "genre_hint": "indie folk singer-songwriter",
        "pattern": "fading",
        "release_offset": 5,
        "collaborators": [],
        "metadata_text": "indie folk singer-songwriter acoustic",
    },
    {
        "track_id": "trk_010",
        "track_name": "Cityline Bloom",
        "artist_id": "art_010",
        "artist_name": "June Arcade",
        "artist_followers": 22000,
        "genre_hint": "indie pop alt rock",
        "pattern": "compound_breaker",
        "release_offset": 18,
        "collaborators": ["gtr_sam", "prod_mika"],
        "metadata_text": "indie pop alt rock guitar synth",
    },
]

TRACK_EVENTS = {
    "trk_001": [(2, "tmk_001", "playlist_add"), (5, "tmk_004", "feature")],
    "trk_002": [(3, "tmk_005", "playlist_add")],
    "trk_003": [(2, "tmk_004", "feature")],
    "trk_004": [(3, "tmk_003", "feature"), (7, "tmk_001", "playlist_add")],
    "trk_005": [(2, "tmk_002", "playlist_add")],
    "trk_006": [(12, "tmk_006", "feature")],
    "trk_007": [(4, "tmk_006", "feature")],
    "trk_008": [(10, "tmk_001", "playlist_add")],
    "trk_010": [(1, "tmk_001", "playlist_add"), (5, "tmk_003", "feature")],
}


def _is_seeded(track: dict[str, Any], seed_urls: list[str]) -> bool:
    if not seed_urls:
        return False
    lowered = [url.lower() for url in seed_urls]
    return any(
        token in " ".join(lowered)
        for token in [track["track_id"].lower(), track["artist_name"].lower(), track["track_name"].lower()]
    )


def _base_metric(pattern: str, platform: str, day: int) -> float:
    if day < 0:
        return 0.0

    lag = {
        "spotify": 0,
        "youtube": 1,
        "tiktok": -2,
        "instagram": -1,
    }[platform]
    d = day + lag
    if d < 0:
        return 0.0

    def curve(base: float, rate: float, accel_start: int | None = None, accel: float = 0.0) -> float:
        extra = 0.0
        if accel_start is not None and d > accel_start:
            extra = accel * (d - accel_start) ** 2
        return base * math.exp(rate * d + extra)

    if pattern == "compound_breaker":
        if platform == "tiktok":
            return curve(180.0, 0.055, 16, 0.00045)
        if platform == "youtube":
            return curve(150.0, 0.048, 20, 0.0003)
        if platform == "instagram":
            return curve(130.0, 0.046, 18, 0.00025)
        return curve(120.0, 0.042, 22, 0.0004)

    if pattern == "geo_breaker":
        if platform == "spotify":
            return curve(150.0, 0.039, 15, 0.00042)
        if platform == "tiktok":
            return curve(120.0, 0.032, 22, 0.0001)
        return curve(110.0, 0.034, 22, 0.00012)

    if pattern == "steady_emerging":
        return curve(120.0, 0.031, 28, 0.00012)

    if pattern == "spike_only":
        baseline = curve(130.0, 0.024, None, 0.0)
        if platform == "spotify":
            spike = 1.0 + 4.5 * math.exp(-((d - 20.0) ** 2) / (2.0 * 1.4**2))
            return baseline * spike
        return baseline * (1.0 + 0.15 * math.exp(-((d - 20.0) ** 2) / (2.0 * 2.0**2)))

    if pattern == "paid_push":
        baseline = curve(170.0, 0.022, None, 0.0)
        if platform == "spotify":
            push = 1.0 + 3.2 * math.exp(-((d - 16.0) ** 2) / (2.0 * 1.5**2))
            return baseline * push
        return baseline * (1.0 + 0.07 * math.exp(-((d - 16.0) ** 2) / (2.0 * 2.4**2)))

    if pattern == "playlist_dependent":
        if platform == "spotify":
            return curve(160.0, 0.036, 12, 0.0002)
        return curve(110.0, 0.018, None, 0.0)

    if pattern == "fading":
        rise = 110.0 * math.exp(0.045 * min(d, 16))
        decay = math.exp(-0.035 * max(0, d - 16))
        return rise * decay

    return curve(120.0, 0.024, None, 0.0)


def _engagement_ratios(pattern: str, platform: str) -> tuple[float, float, float]:
    if pattern in {"compound_breaker", "geo_breaker", "steady_emerging"}:
        return (0.08, 0.0065, 0.012)
    if pattern == "fading":
        return (0.06, 0.004, 0.007)
    if pattern == "playlist_dependent":
        return (0.05, 0.0014, 0.003)
    if pattern == "paid_push":
        return (0.04, 0.0009, 0.002)
    if pattern == "spike_only":
        return (0.05, 0.0011, 0.0025)
    return (0.06, 0.003, 0.006)


def _region_distribution(pattern: str, day: int, metric: int) -> dict[str, int]:
    if metric <= 0:
        return {}
    if pattern == "geo_breaker":
        if day < 20:
            weights = [0.05, 0.58, 0.14, 0.07, 0.06, 0.04, 0.03, 0.03]
        elif day < 35:
            weights = [0.12, 0.33, 0.14, 0.12, 0.1, 0.08, 0.06, 0.05]
        else:
            weights = [0.16, 0.22, 0.14, 0.14, 0.12, 0.1, 0.07, 0.05]
    else:
        weights = [0.16, 0.11, 0.12, 0.16, 0.14, 0.11, 0.1, 0.1]

    values = {region: int(metric * weight) for region, weight in zip(REGIONS, weights, strict=True)}
    diff = metric - sum(values.values())
    if diff != 0:
        values[REGIONS[0]] += diff
    return values


def _pick_comments(pattern: str, count: int) -> list[str]:
    if count <= 0:
        return []
    comments: list[str] = []
    specific_share = 0.55 if pattern in {"compound_breaker", "geo_breaker", "steady_emerging"} else 0.15
    for _ in range(min(15, count)):
        if RNG.random() < specific_share:
            comments.append(RNG.choice(SPECIFIC_COMMENTS))
        else:
            comments.append(RNG.choice(GENERIC_COMMENTS))
    return comments


def _platform_value(platform: str, raw: float) -> int:
    scale = {
        "spotify": 1.0,
        "youtube": 1.6,
        "tiktok": 2.9,
        "instagram": 1.3,
    }[platform]
    return max(0, int(raw * scale * (0.97 + RNG.random() * 0.06)))


def _event_lookup(track_id: str, day_since_release: int) -> tuple[str | None, str | None, str]:
    for event_day, tastemaker_id, event_type in TRACK_EVENTS.get(track_id, []):
        if day_since_release == event_day:
            tm = next((item for item in TASTEMAKERS if item["id"] == tastemaker_id), None)
            if tm:
                return tm["id"], tm["name"], event_type
    return None, None, "none"


def _playlist_adds(pattern: str, event_type: str, metric: int, day: int) -> int:
    base = int(metric * 0.002)
    if event_type == "playlist_add":
        base += 50 + int(metric * 0.01)
    if pattern == "playlist_dependent" and day in {14, 15, 16}:
        base += 220
    return max(0, base)


def _creator_reuse(pattern: str, platform: str, day: int, metric: int) -> int:
    if platform != "tiktok":
        return int(metric * 0.001)
    factor = 0.003
    if pattern in {"compound_breaker", "geo_breaker"}:
        factor = 0.007 + 0.0001 * max(0, day - 10)
    elif pattern in {"paid_push", "spike_only"}:
        factor = 0.0013
    return int(metric * factor)


def generate_mock_snapshots(
    config: dict[str, Any],
    *,
    days: int = 70,
    end_date: dt.date | None = None,
    seed_urls: list[str] | None = None,
) -> list[dict[str, Any]]:
    _ = config
    seed_urls = seed_urls or []
    final_day = end_date or dt.date.today()
    start_day = final_day - dt.timedelta(days=days - 1)
    records: list[dict[str, Any]] = []

    for track in TRACKS:
        release_date = start_day + dt.timedelta(days=track["release_offset"])
        seeded = _is_seeded(track, seed_urls)

        for d in range(days):
            current_day = start_day + dt.timedelta(days=d)
            day_since_release = (current_day - release_date).days
            for platform in ["spotify", "youtube", "tiktok", "instagram"]:
                raw_metric = _base_metric(track["pattern"], platform, day_since_release)
                metric = _platform_value(platform, raw_metric)

                like_ratio, comment_ratio, share_ratio = _engagement_ratios(track["pattern"], platform)
                likes = int(metric * like_ratio)
                comments = int(metric * comment_ratio)
                shares = int(metric * share_ratio)
                views = metric if platform != "spotify" else int(metric * 1.05)
                streams = metric if platform == "spotify" else int(metric * 0.35)
                listeners = int(streams * 0.62)

                tastemaker_id, tastemaker_name, event_type = _event_lookup(track["track_id"], day_since_release)
                if tastemaker_id:
                    tm_platform = next(
                        item["platform"] for item in TASTEMAKERS if item["id"] == tastemaker_id
                    )
                    if tm_platform != platform:
                        tastemaker_id, tastemaker_name, event_type = None, None, "none"

                playlist_adds = _playlist_adds(track["pattern"], event_type, metric, day_since_release)
                creator_reuse = _creator_reuse(track["pattern"], platform, day_since_release, metric)
                region_metrics = _region_distribution(track["pattern"], day_since_release, streams)
                comments_text = _pick_comments(track["pattern"], comments)

                source = "organic"
                if track["pattern"] == "paid_push" and platform == "spotify" and day_since_release in {16, 17, 18}:
                    source = "paid_like"

                snapshot = Snapshot(
                    date=current_day.isoformat(),
                    platform=platform,
                    track_id=track["track_id"],
                    track_name=track["track_name"],
                    artist_id=track["artist_id"],
                    artist_name=track["artist_name"],
                    artist_followers=track["artist_followers"],
                    release_date=release_date.isoformat(),
                    genre_hint=track["genre_hint"],
                    views=views,
                    likes=likes,
                    comments=comments,
                    shares=shares,
                    streams=streams,
                    listeners=listeners,
                    playlist_adds=playlist_adds,
                    creator_reuse=creator_reuse,
                    region_metrics=region_metrics,
                    tastemaker_id=tastemaker_id,
                    tastemaker_name=tastemaker_name,
                    event_type=event_type,
                    source=source,
                    comments_text=comments_text,
                    collaborators=track["collaborators"],
                    manual_seeded=seeded,
                    metadata_text=track["metadata_text"],
                )
                records.append(snapshot.to_dict())
    return records
