from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(slots=True)
class Snapshot:
    date: str
    platform: str
    track_id: str
    track_name: str
    artist_id: str
    artist_name: str
    artist_followers: int
    release_date: str
    genre_hint: str
    views: int
    likes: int
    comments: int
    shares: int
    streams: int
    listeners: int
    playlist_adds: int
    creator_reuse: int
    region_metrics: dict[str, int]
    tastemaker_id: str | None = None
    tastemaker_name: str | None = None
    event_type: str = "none"
    source: str = "organic"
    comments_text: list[str] = field(default_factory=list)
    collaborators: list[str] = field(default_factory=list)
    manual_seeded: bool = False
    metadata_text: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "date": self.date,
            "platform": self.platform,
            "track_id": self.track_id,
            "track_name": self.track_name,
            "artist_id": self.artist_id,
            "artist_name": self.artist_name,
            "artist_followers": self.artist_followers,
            "release_date": self.release_date,
            "genre_hint": self.genre_hint,
            "views": self.views,
            "likes": self.likes,
            "comments": self.comments,
            "shares": self.shares,
            "streams": self.streams,
            "listeners": self.listeners,
            "playlist_adds": self.playlist_adds,
            "creator_reuse": self.creator_reuse,
            "region_metrics": self.region_metrics,
            "tastemaker_id": self.tastemaker_id,
            "tastemaker_name": self.tastemaker_name,
            "event_type": self.event_type,
            "source": self.source,
            "comments_text": self.comments_text,
            "collaborators": self.collaborators,
            "manual_seeded": self.manual_seeded,
            "metadata_text": self.metadata_text,
        }
