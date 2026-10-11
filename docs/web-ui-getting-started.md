# Web UI: getting started

The bridge serves a web UI at `http://<bridge-host>:9500/ui`. It asks for a **user name and password**. You do not have to
configure anything: the first time you open it, it asks you to choose the password.

## What you need

* The bridge running with the `ui/` folder and `subcfg.py` mounted, and your **config folder** mounted **read-write**
  (see [bridge/docker-compose.example.yml](../bridge/docker-compose.example.yml): `../config:/config`, without `:ro`).
  The password is stored in that folder, so the bridge must be able to write there.
* A config folder. If you do not have one yet: copy [config.example](../config.example/README.md) to `config`.

## First time: choose a password

1. Open `http://<bridge-host>:9500/ui` in your browser.
2. You land on **"First time here"**. The user name is `admin` (you can change it); choose a password of at least 8
   characters, type it twice and press *Save password and continue*.
3. You are signed in. The browser remembers you for 14 days.

Things to know:

* The first-time page is **only offered to computers on your local network**, and only until a password is set. Do it
  right after you start the bridge, so nobody else on your network gets there first.
* Easy passwords (`password`, `12345678`, the user name itself, a single repeated character) are refused.
* The password is never stored in plain text: only a salted scrypt hash in `config/auth.json` (readable by you only).

## Every day

Open `http://<bridge-host>:9500/ui`: if you are not signed in you get the sign-in page. After too many wrong attempts
(5) the page makes you wait (30 seconds, doubling up to 15 minutes).

* **Change the password:** *Change password* at the top right. Everybody else is signed out when you do.
* **Sign out:** *Sign out* at the top right.

## Forgot the password

On the machine that holds the config folder:

```bash
python subcfg/subcfg.py reset-password --dir ./config
```

(or through the bridge container: `docker compose run --rm --no-deps --entrypoint python whisperbridge /app/subcfg.py reset-password --dir /config`).
Then open the web UI again: you get the "First time here" page and choose a new password. Your settings are untouched.
Because this needs access to the machine, only someone who already controls it can do it.

## Scripts: an API token (optional)

A script that calls the JSON API (for example `curl`) does not sign in with a password. Give it a token instead:

```bash
python subcfg/subcfg.py init-token --dir ./config
```

This stores a random `WEBUI_TOKEN` in `config/secrets.env` and points `ui.token` in `bridge.json` at it. Restart the bridge
once. Use it as a header:

```bash
curl -H "X-Token: <token>" http://<bridge-host>:9500/ui/api/status
curl -H "X-Token: <token>" -H "X-Requested-With: ui" -H "Content-Type: application/json" \
     -d '{"changes": {"translation.review": false}}' http://<bridge-host>:9500/ui/api/config/bridge
```

The token works for the API only (never for the web page), cannot change the password, and you can replace it with
`init-token --rotate`. Without `ui.token` there is no API token and scripts are not let in.

## If it says "Read-only" or "needs a config folder"

| Message | What to do |
|---|---|
| The config folder is mounted read-only | In the compose file mount it as `../config:/config` (no `:ro`) and restart. |
| The web UI needs a config folder | The bridge found no config folder: check the `/config` mount and that `config/schemas/` exists. |
| 409 file changed | Someone (or you in another tab) saved in the meantime: reload the page. |
| 422 | The value is not allowed; the message under the field says why. |

## Good to know

* There is no TLS: the password travels unencrypted on your network. Keep the bridge on your home network and do not
  forward its port to the internet. Put a reverse proxy with HTTPS in front if you need remote access (the session cookie
  becomes `Secure` automatically behind HTTPS).
* Secrets (API keys, the Hugging Face token) are never shown or edited in the UI: they live in `secrets.env`.
* Every save keeps the previous version (History tab), so a wrong change is one click to undo.
