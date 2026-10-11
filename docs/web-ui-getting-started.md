# Web UI: getting started (and how to get the access token)

The bridge serves a web UI at `http://<bridge-host>:9500/ui`. You can **look** at it without doing anything. To
**change settings** (edit, profiles, playground, restore) you need an **access token**. This page shows how to get one.

## What you need first

* The bridge running with the `ui/` folder and `subcfg.py` mounted, and your **config folder** mounted **read-write**
  (see [bridge/docker-compose.example.yml](../bridge/docker-compose.example.yml): `../config:/config`, without `:ro`).
* A config folder. If you do not have one yet: copy [config.example](../config.example/README.md) to `config`.

## Get a token (one command)

Run this from the project folder on the machine that has the config folder (needs Python 3, nothing to install):

```bash
python subcfg/subcfg.py init-token --dir ./config --url http://<bridge-host>:9500
```

No Python on the host? Run it through the bridge container instead (same folder as your compose file):

```bash
docker compose run --rm --no-deps --entrypoint python whisperbridge /app/subcfg.py init-token --dir /config --url http://<bridge-host>:9500
```

It does three things, and you can run it again safely:

1. Creates a random token and stores it in `config/secrets.env` as `WEBUI_TOKEN=...` (readable by you only).
2. Points the bridge at it: `"ui": { "token": "${WEBUI_TOKEN}" }` in `config/bridge.json`.
3. Prints the **login address**, for example `http://<bridge-host>:9500/ui?token=...`.

Then **restart the bridge once** (`docker compose restart whisperbridge`) and open the login address in your browser.
The browser remembers you for 30 days (a cookie), so you only need that address once per browser. After that you can
just open `http://<bridge-host>:9500/ui`.

### The same by hand

```bash
openssl rand -hex 24                      # prints a random token: copy it
echo 'WEBUI_TOKEN=<paste the token>' >> config/secrets.env
chmod 600 config/secrets.env
```

and in `config/bridge.json` add (keep the other settings):

```json
{ "schema_version": 1, "ui": { "token": "${WEBUI_TOKEN}" } }
```

Restart the bridge and open `http://<bridge-host>:9500/ui?token=<the token>`.

## Lost the token, or want another

The token is just the `WEBUI_TOKEN=` line in `config/secrets.env`. To see it, open that file. To replace it (for
example after you shared a screenshot with it in it):

```bash
python subcfg/subcfg.py init-token --dir ./config --rotate --url http://<bridge-host>:9500
```

and restart the bridge. Every browser has to log in again with the new address.

## If it says "Read-only"

The UI tells you why at the top of the Configuration, Profiles and Playground pages:

| Message | What to do |
|---|---|
| No access token is set | Do the steps above (token + restart). |
| The config folder is mounted read-only | In the compose file mount it as `../config:/config` (no `:ro`) and restart. |
| The configuration layer is off | The bridge found no config folder or no schemas: check the `/config` mount and that `config/schemas/` exists. |

Other messages you may see: **401 token required**: open the login address again (the cookie expired or the token
changed). **409 file changed**: someone (or you in another tab) saved in the meantime: reload the page. **422**: the value
is not allowed; the message under the field says why.

## Good to know

* There is no TLS and no user accounts: the token is a shared password. Keep the bridge on your home network and do not
  forward its port to the internet. Put a reverse proxy with HTTPS in front if you need remote access.
* Secrets (API keys, the Hugging Face token, this token) are never shown or edited in the UI: they live in
  `secrets.env`.
* Every save keeps the previous version (History tab), so a wrong change is one click to undo.
* Without a token the UI stays read-only for everybody who can reach it.
