"""
src/auth.py — Authentication logic
Handles registration, login, logout, and session management.
Passwords are hashed with bcrypt. Accounts are locked after 5 failed attempts.
"""

import re
import bcrypt
import streamlit as st
from datetime import datetime, timezone
from sqlalchemy import text

from src.database import get_engine

# ── Constants ───────────────────────────────────────────────────────────────
MAX_FAILED_ATTEMPTS = 5


# ── Validation helpers ───────────────────────────────────────────────────────

def _is_valid_email(email: str) -> bool:
    pattern = r"^[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+$"
    return bool(re.match(pattern, email.strip()))


def _is_strong_password(password: str) -> tuple[bool, str]:
    if len(password) < 8:
        return False, "Password must be at least 8 characters."
    if not re.search(r"[A-Za-z]", password):
        return False, "Password must contain at least one letter."
    if not re.search(r"\d", password):
        return False, "Password must contain at least one digit."
    return True, ""


# ── Core auth functions ──────────────────────────────────────────────────────

def register_user(username: str, email: str, password: str, confirm: str) -> tuple[bool, str]:
    """
    Validate and register a new user.
    Returns (True, "ok") on success or (False, error_message) on failure.
    """
    username = username.strip()
    email    = email.strip().lower()

    # — Input validation —
    if not username or len(username) < 3:
        return False, "Username must be at least 3 characters."
    if len(username) > 80:
        return False, "Username must be 80 characters or fewer."
    if not _is_valid_email(email):
        return False, "Please enter a valid email address."
    ok, msg = _is_strong_password(password)
    if not ok:
        return False, msg
    if password != confirm:
        return False, "Passwords do not match."

    # — Hash password —
    hashed = bcrypt.hashpw(password.encode("utf-8"), bcrypt.gensalt()).decode("utf-8")

    # — Insert into DB —
    engine = get_engine()
    try:
        with engine.begin() as conn:
            conn.execute(
                text("""
                    INSERT INTO users (username, email, hashed_password)
                    VALUES (:username, :email, :hashed_password)
                """),
                {"username": username, "email": email, "hashed_password": hashed},
            )
        return True, "ok"
    except Exception as exc:
        err = str(exc).lower()
        if "unique" in err or "duplicate" in err:
            if "email" in err:
                return False, "An account with this email already exists."
            if "username" in err:
                return False, "This username is already taken."
            return False, "An account with those details already exists."
        return False, f"Registration failed: {exc}"


def login_user(email: str, password: str) -> tuple[bool, dict | str]:
    """
    Authenticate a user by email + password.
    Returns (True, user_dict) on success or (False, error_message) on failure.
    """
    email = email.strip().lower()

    if not email or not password:
        return False, "Please enter your email and password."

    engine = get_engine()
    try:
        with engine.begin() as conn:
            row = conn.execute(
                text("""
                    SELECT id, username, email, hashed_password,
                           created_at, last_login, failed_attempts, is_locked
                    FROM   users
                    WHERE  email = :email
                """),
                {"email": email},
            ).fetchone()
    except Exception as exc:
        return False, f"Database error: {exc}"

    if row is None:
        return False, "No account found with that email address."

    user = dict(row._mapping)

    if user["is_locked"]:
        return False, (
            "⚠️ Your account has been locked after too many failed attempts. "
            "Please contact support to unlock it."
        )

    # — Verify password —
    if not bcrypt.checkpw(password.encode("utf-8"), user["hashed_password"].encode("utf-8")):
        # Increment failed attempts
        new_attempts = user["failed_attempts"] + 1
        lock         = new_attempts >= MAX_FAILED_ATTEMPTS
        try:
            with engine.begin() as conn:
                conn.execute(
                    text("""
                        UPDATE users
                        SET    failed_attempts = :attempts,
                               is_locked       = :locked
                        WHERE  id = :id
                    """),
                    {"attempts": new_attempts, "locked": lock, "id": user["id"]},
                )
        except Exception:
            pass  # Don't surface DB errors on password failure

        remaining = MAX_FAILED_ATTEMPTS - new_attempts
        if lock:
            return False, (
                "⚠️ Account locked after 5 failed attempts. "
                "Please contact support."
            )
        return False, (
            f"Incorrect password. "
            f"{remaining} attempt{'s' if remaining != 1 else ''} remaining before lockout."
        )

    # — Success: reset failed attempts, update last_login —
    now = datetime.now(timezone.utc)
    try:
        with engine.begin() as conn:
            conn.execute(
                text("""
                    UPDATE users
                    SET    failed_attempts = 0,
                           is_locked       = FALSE,
                           last_login      = :now
                    WHERE  id = :id
                """),
                {"now": now, "id": user["id"]},
            )
    except Exception:
        pass  # Don't block login if this fails

    user["last_login"] = now
    return True, user


def logout_user():
    """Clear all auth-related session state keys."""
    for key in ("authenticated", "user_id", "username", "email", "last_login"):
        st.session_state.pop(key, None)


def is_authenticated() -> bool:
    """Return True if the current session has an authenticated user."""
    return bool(st.session_state.get("authenticated", False))


def get_current_user() -> dict:
    """Return the current user's session info (or empty dict)."""
    if not is_authenticated():
        return {}
    return {
        "user_id":   st.session_state.get("user_id"),
        "username":  st.session_state.get("username"),
        "email":     st.session_state.get("email"),
        "last_login": st.session_state.get("last_login"),
    }


def _set_session(user: dict):
    """Persist auth data into st.session_state."""
    st.session_state["authenticated"] = True
    st.session_state["user_id"]       = user["id"]
    st.session_state["username"]      = user["username"]
    st.session_state["email"]         = user["email"]
    st.session_state["last_login"]    = user.get("last_login")
