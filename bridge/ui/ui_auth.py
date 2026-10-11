"""Username and password for the bridge web UI (standard library only).

* The first time the UI is opened there is no password yet: the visitor (from the local network only) chooses one.
  The default user name is `admin`.
* The password is stored as a salted scrypt hash in `<config>/auth.json` together with a random signing key; the file is
  chmod 600. Sessions are signed cookies (HMAC-SHA256) that stop working when the key changes, which happens whenever the
  password is changed.
* Failed logins are slowed down per client address.
"""
from __future__ import annotations

import base64
import hashlib
import hmac
import ipaddress
import json
import os
import secrets
import tempfile
import threading
import time

SCRYPT = (2 ** 14, 8, 1)                      # n, r, p: about 16 MB and 50 ms
MIN_LENGTH = 8
SESSION_DAYS = 14
COMMON = {"password", "passw0rd", "12345678", "123456789", "1234567890", "qwertyui", "qwerty123", "admin123", "administrator",
          "letmein1", "welcome1", "iloveyou", "11111111", "00000000", "password1", "password123", "changeme"}


def validate_password(password: str, username: str = "") -> list[str]:
    errs = []
    if not isinstance(password, str) or len(password) < MIN_LENGTH:
        errs.append(f"The password must be at least {MIN_LENGTH} characters.")
    elif len(password) > 200:
        errs.append("The password is too long (200 characters at most).")
    elif password.lower() in COMMON or password.lower() == (username or "").lower() or len(set(password)) < 4:
        errs.append("That password is too easy to guess: choose something else.")
    return errs


def hash_password(password: str) -> str:
    salt = os.urandom(16)
    n, r, p = SCRYPT
    digest = hashlib.scrypt(password.encode(), salt=salt, n=n, r=r, p=p, dklen=32)
    return f"scrypt${n}${r}${p}${salt.hex()}${digest.hex()}"


def check_hash(password: str, stored: str) -> bool:
    try:
        _, n, r, p, salt, digest = stored.split("$")
        got = hashlib.scrypt(password.encode(), salt=bytes.fromhex(salt), n=int(n), r=int(r), p=int(p), dklen=32)
        return hmac.compare_digest(got, bytes.fromhex(digest))
    except Exception:                          # noqa: BLE001
        return False


def is_local_address(host: str | None, trusted: list | tuple = ()) -> bool:
    """Not a public internet address (private, loopback, link-local, carrier-grade NAT such as Tailscale), or inside one
    of the networks the owner listed in `ui.trusted_networks` (for example a LAN that uses a non-standard range).
    The first-time setup page is only offered to such addresses."""
    try:
        ip = ipaddress.ip_address((host or "").split("%")[0])
    except ValueError:
        return False
    if ip.version == 6 and ip.ipv4_mapped:
        ip = ip.ipv4_mapped
    for net in trusted or ():
        try:
            if ip in ipaddress.ip_network(str(net).strip(), strict=False):
                return True
        except ValueError:
            continue
    return not ip.is_global


class Throttle:
    """Per client: after 5 failures a lock-out that doubles from 30 s up to 15 minutes."""

    def __init__(self, free: int = 5):
        self.free, self.lock = free, threading.Lock()
        self.fails: dict[str, tuple[int, float]] = {}

    def wait_s(self, key: str) -> int:
        with self.lock:
            n, until = self.fails.get(key, (0, 0.0))
            return max(0, int(until - time.time() + 0.999)) if n >= self.free else 0

    def failed(self, key: str) -> None:
        with self.lock:
            n, _ = self.fails.get(key, (0, 0.0))
            n += 1
            lock = min(900, 30 * 2 ** (n - self.free)) if n >= self.free else 0
            self.fails[key] = (n, time.time() + lock)

    def ok(self, key: str) -> None:
        with self.lock:
            self.fails.pop(key, None)


class AuthStore:
    def __init__(self, directory: str):
        self.dir = directory
        self.path = os.path.join(directory, "auth.json")
        self._lock = threading.Lock()
        self._dummy = hash_password(secrets.token_hex(8))      # so an unknown user costs as much time as a wrong password

    def _read(self) -> dict:
        try:
            with open(self.path, encoding="utf-8") as f:
                data = json.load(f)
            return data if isinstance(data, dict) else {}
        except (OSError, ValueError):
            return {}

    def _write(self, data: dict) -> None:
        os.makedirs(self.dir, exist_ok=True)
        fd, tmp = tempfile.mkstemp(dir=self.dir, prefix=".auth.", suffix=".tmp")
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as f:
            json.dump(data, f, indent=2)
            f.write("\n")
        try:
            os.chmod(tmp, 0o600)
        except OSError:
            pass
        os.replace(tmp, self.path)

    def writable(self) -> bool:
        return os.access(self.dir, os.W_OK) and (not os.path.exists(self.path) or os.access(self.path, os.W_OK))

    @property
    def configured(self) -> bool:
        d = self._read()
        return bool(d.get("password") and d.get("secret") and d.get("username"))

    @property
    def username(self) -> str:
        return self._read().get("username", "")

    def setup(self, username: str, password: str) -> list[str]:
        username = (username or "").strip()
        with self._lock:
            if self.configured:
                return ["A password is already set."]
            errs = []
            if not 1 <= len(username) <= 40 or not all(c.isalnum() or c in "._-" for c in username):
                errs.append("The user name may contain letters, digits, dot, dash and underscore (1 to 40 characters).")
            errs += validate_password(password, username)
            if errs:
                return errs
            self._write({"username": username, "password": hash_password(password), "secret": secrets.token_hex(32),
                         "created": int(time.time())})
        return []

    def verify(self, username: str, password: str) -> bool:
        d = self._read()
        ok_user = hmac.compare_digest((username or "").encode(), d.get("username", "").encode())
        ok_pass = check_hash(password or "", d.get("password") or self._dummy)
        return bool(d.get("password")) and ok_user and ok_pass

    def change_password(self, current: str, new: str) -> list[str]:
        with self._lock:
            d = self._read()
            if not d.get("password") or not check_hash(current or "", d["password"]):
                return ["The current password is not correct."]
            errs = validate_password(new, d.get("username", ""))
            if errs:
                return errs
            d.update(password=hash_password(new), secret=secrets.token_hex(32), password_changed=int(time.time()))
            self._write(d)             # a new signing key: every other session is signed out
        return []

    def reset(self) -> bool:
        """Forget the password (the next visitor of the setup page chooses a new one). Used by the command line."""
        try:
            os.remove(self.path)
            return True
        except OSError:
            return False

    # ------------------------------------------------------------------ sessions
    def _sign(self, payload: str, secret: str) -> str:
        return hmac.new(secret.encode(), payload.encode(), hashlib.sha256).hexdigest()

    def new_session(self) -> str:
        d = self._read()
        payload = base64.urlsafe_b64encode(json.dumps(
            {"u": d.get("username", ""), "exp": int(time.time()) + SESSION_DAYS * 86400, "n": secrets.token_hex(6)}).encode()).decode()
        return payload + "." + self._sign(payload, d.get("secret", ""))

    def session_user(self, cookie: str | None) -> str | None:
        if not cookie or "." not in cookie:
            return None
        d = self._read()
        if not d.get("secret") or not d.get("username"):
            return None
        payload, _, sig = cookie.rpartition(".")
        if not hmac.compare_digest(sig, self._sign(payload, d["secret"])):
            return None
        try:
            data = json.loads(base64.urlsafe_b64decode(payload.encode()))
        except Exception:                          # noqa: BLE001
            return None
        return data.get("u") if data.get("exp", 0) > time.time() and data.get("u") == d["username"] else None
