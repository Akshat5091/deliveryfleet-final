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
hf auth login --add-to-git-credential
```

Paste the token when asked. `--add-to-git-credential` lets `git push` reuse it.

## 5. Push the project to the Space

A Space needs a `README.md` with a small YAML header at the **root of the
Space repo**. This project keeps that file in `deploy/huggingface/README.md`
so your normal repo stays clean. The simplest way: make a throw-away branch
that adds this one file at the root, and push only that branch to the Space.
Your `main` branch is never changed.

Run these from the project folder (the one containing `Dockerfile`).
Replace `<your-username>`.

```powershell
git remote add space https://huggingface.co/spaces/<your-username>/deliveryfleet-mas
git checkout -b space-deploy
Copy-Item deploy/huggingface/README.md README.md
git add -A
git commit -m "Deploy to Hugging Face Space"
git push --force space space-deploy:main
git checkout main
```

Notes:

- `--force` is needed only the first time, because the new Space already has
  its own starter commit that your history does not contain.
- `git add -A` respects `.gitignore`, so `.env` and `runtime/` are not pushed.
  Check with `git status` before committing that no `.env` file is listed.
- Any file you changed but have not committed on `main` is carried onto
  `space-deploy`. Commit or stash first if you do not want that.

## 6. Watch it build

Open the Space page. The **Building** badge turns to **Running** when the image
is ready (the **Logs** tab shows progress). The app then waits until its MCP
and route services answer, so allow a minute after "Running" on a cold start.

## 7. Update later

```powershell
git checkout space-deploy
git merge main
git push space space-deploy:main
git checkout main
```

Free Spaces go to sleep after inactivity and wake on the next visit. Data in
`runtime/` (the SQLite log, for example) is lost on restart; the model and
policy index are rebuilt from the image.
