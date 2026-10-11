# Security notes

These services are meant for a trusted home or lab network. Please do not expose them to the internet.

* **npusubs has no authentication.** Compose binds it to `127.0.0.1` by default; set `NPUSUBS_BIND` to a LAN address only
  if you need to.
* **The bridge web UI** is open to anyone who can reach it **unless `ui.token` is set**. With a token (create one with
  `python subcfg/subcfg.py init-token`, see docs/web-ui-getting-started.md), the page and the JSON API need it
  (cookie, `X-Token` header or `?token=`), and every write also needs an `X-Requested-With: ui` header
  (CSRF guard) and a writable config folder. The token is compared in constant time. There is no TLS and no login page:
  put a reverse proxy in front if you need that. Without a token the UI is read-only.
* **Secrets** (API keys, the Hugging Face token, the UI token) belong in `secrets.env` or the environment, never in the
  JSON files and never in git. The UI never returns secret values and refuses to edit them. Other settings (URLs, for
  example) are shown and stored as typed, so do not put credentials into them.
* The **resource guard agent** (`tools/hostmem_agent.py`) reads `/proc/meminfo` only, takes no input and has no
  authentication: bind it to a LAN address.
* Do not mount the config folder read-write into anything you do not trust: whoever can write `bridge.json` controls
  which URLs the bridge calls.

To report a vulnerability, open a private security advisory on the repository, or contact the maintainer directly.
