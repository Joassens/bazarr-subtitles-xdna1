# Security notes

These services are meant for a trusted home or lab network. Please do not expose them to the internet.

* **npusubs has no authentication.** Compose binds it to `127.0.0.1` by default; set `NPUSUBS_BIND` to a LAN address only
  if you need to.
* **The bridge web UI** needs a user name and password. The first visitor from the **local network** chooses the password
  (default user `admin`; at least 8 characters, easy ones are refused); until then nothing else in the UI is reachable
  and the setup page is not offered to public addresses. The password is stored as a salted scrypt hash in
  `config/auth.json` (chmod 600); sessions are signed `HttpOnly`, `SameSite=Strict` cookies that stop working when the
  password changes; failed logins are slowed down per client address (5 free tries, then 30 s doubling to 15 min). Every
  write also needs an `X-Requested-With: ui` header (CSRF guard) and a writable config folder. There is **no TLS**: put a
  reverse proxy in front if you need HTTPS or remote access. Forgot the password: `python subcfg/subcfg.py reset-password`.
* **The optional API token** (`ui.token`, header `X-Token`) is for scripts only: it opens the JSON API, never the web page,
  and cannot change the password. It is compared in constant time. Without `ui.token` there is no API token.
* **Secrets** (API keys, the Hugging Face token, the UI token) belong in `secrets.env` or the environment, never in the
  JSON files and never in git. The UI never returns secret values and refuses to edit them. Other settings (URLs, for
  example) are shown and stored as typed, so do not put credentials into them.
* The **idle gate** (`tools/idle_gate.py`) forwards requests to your LLM server without authentication: bind it to a
  LAN address. The bridge's `/busy` endpoint is open and returns job counts only.
* The **host agent** (`tools/hostmem_agent.py`) reads `/proc/meminfo` and the GPU's busy percentage only, takes no input and has no
  authentication: bind it to a LAN address.
* Do not mount the config folder read-write into anything you do not trust: whoever can write `bridge.json` controls
  which URLs the bridge calls.

To report a vulnerability, open a private security advisory on the repository, or contact the maintainer directly.
