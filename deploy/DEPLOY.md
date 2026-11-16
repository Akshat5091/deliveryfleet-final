# Deploying DeliveryFleet MAS to a Hugging Face Space

You do not need Docker on your computer. Hugging Face builds the Docker image
in the cloud from the `Dockerfile` in this repo. The first build takes several
minutes (it installs the Python packages, trains the delay model and downloads
a small embedding model).

You need: Git (already installed if `git --version` works) and `uv`.

## 1. Make a Hugging Face account and a Space

1. Sign up at https://huggingface.co/join and confirm your email.
2. Open https://huggingface.co/new-space.
3. Space name: `deliveryfleet-mas` (any name is fine). SDK: **Docker**
   (choose the "Blank" template). Hardware: **CPU basic** (free). Visibility:
   your choice. Click **Create Space**.
4. Note the Space address, `https://huggingface.co/spaces/<your-username>/deliveryfleet-mas`.

## 2. Add the Gemini key as a secret

1. In the Space, open **Settings**, then **Variables and secrets**.
2. Click **New secret**. Name: `GEMINI_API_KEY`. Value: your key. Save.

The key is never stored in the image or in git. Do not put it in any file.

## 3. Install the Hugging Face command-line tool

In PowerShell:

```powershell
uv tool install huggingface_hub
hf --version
```

(`huggingface_hub` already includes the `hf` command. Older guides write
`huggingface_hub[cli]`; the current release has no `cli` extra, so the plain
name is correct. If `hf` is not found, run `uv tool update-shell` and open a
new PowerShell window.)

## 4. Log in

Create a token at https://huggingface.co/settings/tokens with **Write**
permission, then run:

```powershell
hf auth login
```

Paste the token when asked.

## 5. Upload the project to the Space

Hugging Face rejects PNG images sent with a plain `git push`, and the
evaluation charts are PNGs, so the project is uploaded with the `hf` tool.
First make a clean copy that contains only committed files: `.env` and your
keys are never committed, so they can never be uploaded. The Space also needs
the `README.md` from `deploy/huggingface/` at its root (it holds the Space
settings), so that file is copied over the project README in the copy only.

Run these from the project folder (the one containing `Dockerfile`).
Replace `<your-username>`.

```powershell
$dir = "..\space-upload-$(Get-Date -Format yyyyMMdd-HHmm)"
git archive --format=zip -o "$dir.zip" HEAD
Expand-Archive "$dir.zip" $dir
Copy-Item deploy\huggingface\README.md "$dir\README.md"
hf upload <your-username>/deliveryfleet-mas $dir . --repo-type space --commit-message "Deploy"
```

Notes:

- Commit your changes first: the copy is made from the last commit.
- Check the copy before uploading: `Get-ChildItem $dir -Force` must not list a
  `.env` file.

## 6. Watch it build

Open the Space page. The **Building** badge turns to **Running** when the image
is ready (the **Logs** tab shows progress). The app then waits until its MCP
and route services answer, so allow a minute after "Running" on a cold start.

## 7. Update later

Commit the change, then run the same five commands from step 5. Each upload
creates a new commit on the Space and Hugging Face rebuilds the image.

Free Spaces go to sleep after inactivity and wake on the next visit. Data in
`runtime/` (the SQLite log, for example) is lost on restart; the model and
policy index are rebuilt from the image.
