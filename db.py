"""
Database helper functions for Elirum.

This module contains simple wrapper functions around the Supabase Python
client.  They demonstrate how to read and write application data while
leveraging row‑level security (RLS) to ensure each user only accesses
their own records.  Functions return the Supabase response object so
callers can inspect data and error fields.
"""
from __future__ import annotations

from typing import Any, Dict, Optional

from supabase_client import supabase


def get_user_by_id(user_id: str) -> Optional[Dict[str, Any]]:
    """Retrieve a user record by its unique ID.

    The users table should enforce RLS so that only the authenticated
    user can read their own record.
    """
    response = supabase.from_("users").select("*").eq("id", user_id).single().execute()
    if response.error:
        return None
    return response.data  # type: ignore[no-any-return]


def create_video_record(user_id: str, file_name: str, url: str) -> Any:
    """Insert a new video record.

    Args:
        user_id: The authenticated user's ID.
        file_name: The original name of the uploaded file.
        url: A public URL or reference to the file in storage.

    Returns:
        The result of the insert operation.
    """
    payload = {
        "user_id": user_id,
        "file_name": file_name,
        "url": url,
    }
    return supabase.from_("videos").insert(payload).execute()
