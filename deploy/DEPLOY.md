# Deploying DeliveryFleet MAS to Azure Container Apps

One Docker container runs everything: the website and API (port 7860, public), the MCP tool
server (8001) and the ADK Route Agent service (8002), both internal. Azure builds the image in
the cloud from the `Dockerfile`, so Docker is not needed on your computer.

Why Azure: since 2026 Hugging Face needs a paid PRO plan to host Docker Spaces, and free hosts
with 512 MB of memory are too small (the app uses about 1 GB). Azure for Students needs no credit
card: it gives $100 of credit for 12 months, and Container Apps also has a monthly free allowance
(180,000 vCPU-seconds, 360,000 GiB-seconds, 2 million requests). When the credit runs out the
services stop; nothing is ever charged to you.

## 1. Get Azure for Students (once)

1. Open https://azure.microsoft.com/free/students and choose **Start free**.
2. Sign in with a Microsoft account and verify with your university email address.
3. When the Azure portal opens, you have a subscription called "Azure for Students".

## 2. Install the Azure CLI (once)

In your own PowerShell window (it asks for administrator permission):

```powershell
winget install --exact --id Microsoft.AzureCLI
```

Close and reopen PowerShell, then check `az version`.

(No install? Use Azure Cloud Shell in the portal instead: upload the ZIP from step 4, unzip it and run
the same commands from step 5 there.)

## 3. Sign in and prepare (once)

```powershell
az login
az extension add --name containerapp --upgrade --allow-preview true
az provider register --namespace Microsoft.App
az provider register --namespace Microsoft.OperationalInsights
```

## 4. Make a clean copy of the project

Only committed files are copied, so `.env` and your keys can never be uploaded. Run from the
project folder (the one containing `Dockerfile`):

```powershell
$dir = "..\deploy-copy"
Remove-Item $dir, "$dir.zip" -Recurse -Force -ErrorAction SilentlyContinue
git archive --format=zip -o "$dir.zip" HEAD
Expand-Archive "$dir.zip" $dir
Get-ChildItem $dir -Force -Filter ".env*"   # must print nothing
```

## 5. Build and deploy

```powershell
cd ..\deploy-copy
az containerapp up --name deliveryfleet --resource-group deliveryfleet-rg --location centralindia --source . --ingress external --target-port 7860
```

This creates the resource group, a container registry and a Container Apps environment, builds the
image from the `Dockerfile` in the cloud (about 10 to 15 minutes the first time) and prints the
public URL (`https://deliveryfleet.<random>.centralindia.azurecontainerapps.io`). If `centralindia`
is not allowed for your subscription, use `eastus`.

## 6. Add the Gemini key as a secret, give the app enough memory

Replace `PASTE-YOUR-KEY` with your own Gemini key. The key is stored as an encrypted Container Apps
secret, never in the code or the image.

```powershell
az containerapp secret set --name deliveryfleet --resource-group deliveryfleet-rg --secrets gemini-key=PASTE-YOUR-KEY
az containerapp update --name deliveryfleet --resource-group deliveryfleet-rg --set-env-vars GEMINI_API_KEY=secretref:gemini-key --cpu 1.0 --memory 2.0Gi --min-replicas 0 --max-replicas 1
```

`--min-replicas 0` lets the app sleep when nobody uses it, so it costs almost nothing. The first visit
after a sleep takes one to two minutes while the container starts.

## 7. Check it

Open `<URL>/api/health`. All of `web`, `mcp_server`, `a2a_route_service` and `llm_key` must be `true`.
Then open the URL itself: the Runs tab already lists the 72 benchmark runs and the demo cases (they are
copied in from `deploy/seed_runs.sqlite3` on the first start).

## On demo day

Keep one copy running so the first click is instant, then let it sleep again afterwards:

```powershell
az containerapp update --name deliveryfleet --resource-group deliveryfleet-rg --min-replicas 1
az containerapp update --name deliveryfleet --resource-group deliveryfleet-rg --min-replicas 0
```

## Update later

Commit the change, then repeat step 4 and run the `az containerapp up` line from step 5 again. The
secret, memory and scaling settings are kept.

Notes:

- New bookings made on the live site are lost when the container restarts; the saved runs come back.
- A live event stream is cut after about four minutes by Azure's front door; the page then switches
  to polling and still shows the finished run.
