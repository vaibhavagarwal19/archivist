from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.core.security import decode_access_token
from app.db.session import get_db
from app.models.user import User


bearer_scheme = HTTPBearer(auto_error=False)


def _get_or_create_guest_user(db: Session) -> User:
    guest_email = "guest@example.com"
    guest_user = db.scalar(select(User).where(User.email == guest_email))
    if guest_user:
        return guest_user

    legacy_guest = db.scalar(select(User).where(User.full_name == "Guest Workspace"))
    if legacy_guest:
        legacy_guest.email = guest_email
        legacy_guest.hashed_password = "guest-access-disabled"
        legacy_guest.is_active = True
        db.add(legacy_guest)
        db.commit()
        db.refresh(legacy_guest)
        return legacy_guest

    guest_user = User(
        email=guest_email,
        full_name="Guest Workspace",
        hashed_password="guest-access-disabled",
        is_active=True,
    )
    db.add(guest_user)
    db.commit()
    db.refresh(guest_user)
    return guest_user


def get_current_user(
    db: Session = Depends(get_db),
    credentials: HTTPAuthorizationCredentials | None = Depends(bearer_scheme),
) -> User:
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )

    if credentials is None:
        return _get_or_create_guest_user(db)
    if credentials.scheme.lower() != "bearer":
        raise credentials_exception

    try:
        payload = decode_access_token(credentials.credentials)
        user_id = int(payload.get("sub"))
    except (ValueError, TypeError):
        raise credentials_exception

    user = db.get(User, user_id)
    if not user or not user.is_active:
        raise credentials_exception
    return user
