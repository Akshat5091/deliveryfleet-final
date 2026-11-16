# Deploying DeliveryFleet MAS to Azure Container Apps

Live site: https://deliveryfleet-mas.gentledune-e77ea6e0.malaysiawest.azurecontainerapps.io

One Docker container runs everything: the website and API (port 7860, public), the MCP tool
server (8001) and the ADK Route Agent service (8002), both internal.

| Piece | Where | Name |
|---|---|---|
| Image build | GitHub Actions, private repo | `Akshat5091/deliveryfleet-final`, `.github/workflows/build-image.yml` |
| Image store | Azure Container Registry, East Asia | `deliveryfleetmasacr.azurecr.io/deliveryfleet` |
| Running app | Azure Container Apps, Malaysia West | app `deliveryfleet-mas`, environment `deliveryfleet-mas-env` |
| Resource group | Azure | `deliveryfleet-rg` |

Why this shape:

- Hugging Face now needs a paid PRO plan to host Docker Spaces; free hosts with 512 MB of memory
  are too small (the app uses about 1 GB).
- Azure for Students needs no card: $100 of credit for 12 months, plus the Container Apps monthly
  free allowance (180,000 vCPU-seconds, 360,000 GiB-seconds, 2 million requests). When the credit
  runs out, services stop; nothing is charged.
- The student subscription only allows five regions (East Asia, India South Central, Malaysia
  West, Poland Central, Indonesia Central). Container Apps is not offered in India South Central,
  and the Gemini API does not serve Hong Kong (East Asia), so the app runs in Malaysia West.
- Azure blocks cloud image builds (ACR Tasks) on student subscriptions, so GitHub Actions builds
  the image and pushes it to the registry.

## One-time setup (already done)

1. Azure for Students activated with the university email; Azure CLI installed (`winget install
   --exact --id Microsoft.AzureCLI`) and `az login`.
2. `az extension add --name containerapp --upgrade --allow-preview true`, then register
   `Microsoft.App`, `Microsoft.OperationalInsights` and `Microsoft.ContainerRegistry` with
   `az provider register --namespace <name>`.
3. Registry: `az acr create -n deliveryfleetmasacr -g deliveryfleet-rg -l eastasia --sku Basic --admin-enabled true`.
4. GitHub repository secrets `ACR_USERNAME` and `ACR_PASSWORD` set from `az acr credential show`
   (the GitHub CLI needs the `workflow` scope: `gh auth refresh -h github.com -s workflow`).
5. Container app created from the image with the Gemini key as a secret:

```powershell
az containerapp create --name deliveryfleet-mas --resource-group deliveryfleet-rg `
  --environment deliveryfleet-mas-env --image deliveryfleetmasacr.azurecr.io/deliveryfleet:latest `
  --registry-server deliveryfleetmasacr.azurecr.io --registry-username <acr user> --registry-password <acr password> `
  --ingress external --target-port 7860 --cpu 1.0 --memory 2.0Gi --min-replicas 0 --max-replicas 1 `
  --secrets gemini-key=<your Gemini key> --env-vars GEMINI_API_KEY=secretref:gemini-key
```

## Update the live site

1. Commit and push to `master`. GitHub builds the image in about five minutes
   (`gh run watch` or the repository's Actions tab).
2. Point the app at the new image (the tag is the commit id):

```powershell
az containerapp update --name deliveryfleet-mas --resource-group deliveryfleet-rg --image deliveryfleetmasacr.azurecr.io/deliveryfleet:<commit id>
```

The secret, memory and scaling settings are kept.

## Check it

Open `<URL>/api/health`: `web`, `mcp_server`, `a2a_route_service` and `llm_key` must all be `true`.
The Runs tab lists the 72 benchmark runs and the demo cases (copied in from
`deploy/seed_runs.sqlite3` on the first start of each container).

## On demo day

The app sleeps when unused (`--min-replicas 0`), and the first visit after a sleep takes one to
two minutes. Keep one copy awake before the demo, and let it sleep again afterwards:

```powershell
az containerapp update --name deliveryfleet-mas --resource-group deliveryfleet-rg --min-replicas 1
az containerapp update --name deliveryfleet-mas --resource-group deliveryfleet-rg --min-replicas 0
```

## Change the Gemini key

```powershell
az containerapp secret set --name deliveryfleet-mas --resource-group deliveryfleet-rg --secrets gemini-key=<new key>
az containerapp revision restart --name deliveryfleet-mas --resource-group deliveryfleet-rg --revision <name from: az containerapp revision list -n deliveryfleet-mas -g deliveryfleet-rg -o table>
```

Notes:

- Runs made on the live site are lost when the container restarts; the saved runs come back.
- A live event stream can be cut by Azure's front door on long runs; the page then switches to
  polling and still shows the finished run.
