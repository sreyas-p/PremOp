"""YouTube tools, built on the Data API v3.

What is *not* here, and why: watch history. YouTube removed the `HL` history
playlist from the Data API years ago, and there is no replacement endpoint — so
"take notes on what I watched" is not implementable as stated. The workable
substitutes are the ones below: liked videos, subscriptions, and any playlist
the user maintains. Point the agent at one of those instead.

Captions are also not fetchable for videos the user doesn't own (the
captions.download endpoint requires ownership), so notes are built from title,
description, and metadata rather than transcript text.
"""

from __future__ import annotations

from typing import Any

from anthropic import beta_tool

from ..integrations.google_auth import youtube
from ..semantic import remember_text

_MAX_DESCRIPTION_CHARS = 2000
_LIKED_VIDEOS_PLAYLIST = "LL"


def _format_items(items: list[dict[str, Any]]) -> str:
    lines = []
    for item in items:
        snippet = item.get("snippet", {})
        video_id = (
            snippet.get("resourceId", {}).get("videoId")
            or item.get("id", {}).get("videoId")
            or item.get("id")
        )
        lines.append(
            f"- {snippet.get('title', '[untitled]')}\n"
            f"  id={video_id} | channel: {snippet.get('videoOwnerChannelTitle') or snippet.get('channelTitle', '?')}"
            f" | published: {snippet.get('publishedAt', '?')}"
        )
    return "\n".join(lines)


@beta_tool
def youtube_search(query: str, max_results: int = 10) -> str:
    """Search YouTube for videos matching a query.

    Args:
        query: What to search for.
        max_results: How many videos to return, 1-50. Defaults to 10.
    """
    max_results = max(1, min(int(max_results), 50))
    response = (
        youtube()
        .search()
        .list(part="snippet", q=query, type="video", maxResults=max_results)
        .execute()
    )
    items = response.get("items", [])
    if not items:
        return f"No videos matched {query!r}."
    return f"{len(items)} result(s) for {query!r}:\n" + _format_items(items)


@beta_tool
def youtube_video_details(video_id: str) -> str:
    """Get a video's title, channel, description, and statistics.

    This is the primary source for note-taking: transcripts are not available
    through the API for videos the user does not own, so the description and
    metadata are what there is to work from.

    Args:
        video_id: A YouTube video ID, e.g. "dQw4w9WgXcQ".
    """
    response = (
        youtube()
        .videos()
        .list(part="snippet,statistics,contentDetails", id=video_id)
        .execute()
    )
    items = response.get("items", [])
    if not items:
        return f"No video found with id {video_id!r}."

    video = items[0]
    snippet = video["snippet"]
    stats = video.get("statistics", {})
    description = snippet.get("description", "")
    if len(description) > _MAX_DESCRIPTION_CHARS:
        description = description[:_MAX_DESCRIPTION_CHARS] + "\n[description truncated]"

    remember_text(
        "youtube", video_id, snippet.get("title", ""),
        f"{snippet.get('title')} by {snippet.get('channelTitle')}. {description}",
    )

    remember_text(
        "youtube", video_id, snippet.get("title", ""),
        f"{snippet.get('title')} by {snippet.get('channelTitle')}. {description}",
    )

    return (
        f"Title: {snippet.get('title')}\n"
        f"Channel: {snippet.get('channelTitle')}\n"
        f"Published: {snippet.get('publishedAt')}\n"
        f"Duration: {video.get('contentDetails', {}).get('duration')}\n"
        f"Views: {stats.get('viewCount', '?')} | Likes: {stats.get('likeCount', '?')}\n"
        f"Tags: {', '.join(snippet.get('tags', [])) or '[none]'}\n"
        f"URL: https://www.youtube.com/watch?v={video_id}\n\n"
        f"Description:\n{description or '[empty]'}"
    )


@beta_tool
def youtube_liked_videos(max_results: int = 20) -> str:
    """List the videos the signed-in user has liked, newest first.

    Watch history is not retrievable through the API. This is the closest
    available signal for "what the user has been watching".

    Args:
        max_results: How many videos to return, 1-50. Defaults to 20.
    """
    return youtube_playlist_items(_LIKED_VIDEOS_PLAYLIST, max_results)


@beta_tool
def youtube_playlist_items(playlist_id: str, max_results: int = 20) -> str:
    """List the videos in a playlist.

    Args:
        playlist_id: A playlist ID. "LL" is the signed-in user's liked videos.
            Note that Watch Later ("WL") is not accessible through the API.
        max_results: How many videos to return, 1-50. Defaults to 20.
    """
    max_results = max(1, min(int(max_results), 50))
    response = (
        youtube()
        .playlistItems()
        .list(part="snippet", playlistId=playlist_id, maxResults=max_results)
        .execute()
    )
    items = response.get("items", [])
    if not items:
        return f"Playlist {playlist_id!r} is empty or not accessible."
    return f"{len(items)} video(s) in playlist {playlist_id}:\n" + _format_items(items)


@beta_tool
def youtube_subscriptions(max_results: int = 25) -> str:
    """List the channels the signed-in user subscribes to.

    Args:
        max_results: How many channels to return, 1-50. Defaults to 25.
    """
    max_results = max(1, min(int(max_results), 50))
    response = (
        youtube()
        .subscriptions()
        .list(part="snippet", mine=True, maxResults=max_results)
        .execute()
    )
    items = response.get("items", [])
    if not items:
        return "No subscriptions found."
    lines = [
        f"- {item['snippet']['title']} "
        f"(channel id={item['snippet']['resourceId']['channelId']})"
        for item in items
    ]
    return f"{len(items)} subscription(s):\n" + "\n".join(lines)


TOOLS = {
    "youtube_search": youtube_search,
    "youtube_video_details": youtube_video_details,
    "youtube_liked_videos": youtube_liked_videos,
    "youtube_playlist_items": youtube_playlist_items,
    "youtube_subscriptions": youtube_subscriptions,
}
