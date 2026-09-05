# MLflow as a Dockerized Service — Development & Google Colab Integration

> **Purpose:** A practical reference documenting how to run MLflow as a Docker service, preserve existing experiments/artifacts, connect local training to the MLflow server over HTTP, and connect Google Colab training to the same local MLflow instance through a temporary public tunnel.

---

## 1. The Goal

The objective was to build an MLOps development environment where:

- MLflow runs as a **Docker service**.
- MLflow's database remains persistent on the local machine.
- MLflow artifacts remain persistent on the local machine.
- Local training code communicates with MLflow through the **MLflow Tracking Server HTTP API**.
- Google Colab can also communicate with the same MLflow server.
- Existing MLflow experiments and runs are preserved.
- The setup can later be extended toward VPS-based training/serving.

The final development architecture is:

```text
                         INTERNET
                            │
                            │ HTTPS
                            ▼
                    ┌─────────────────┐
                    │ Cloudflare      │
                    │ Quick Tunnel    │
                    └────────┬────────┘
                             │
                             │ HTTP
                             ▼
┌───────────────────────────────────────────────────────┐
│                    LOCAL MACHINE                      │
│                                                       │
│   ┌───────────────────────────────────────────────┐   │
│   │              Docker MLflow                    │   │
│   │              localhost:5000                   │   │
│   │                                               │   │
│   │  MLflow Tracking Server                       │   │
│   └───────────────────┬───────────────────────────┘   │
│                       │                               │
│              ┌────────┴────────┐                      │
│              ▼                 ▼                       │
│       mlflow/mlflow.db    mlflow/mlruns/              │
│          Metadata            Artifacts                 │
│                                                       │
└───────────────────────────────────────────────────────┘
                             ▲
                             │
                             │ HTTPS
                             │
                    ┌────────┴────────┐
                    │   Google Colab  │
                    │                 │
                    │ Training code   │
                    └─────────────────┘
```

---

# 2. Important Mental Model

The most important concept learned during this process is:

> **Training code should communicate with MLflow through the MLflow Tracking Server. It should not directly access the MLflow database or artifact directory.**

The architecture is:

```text
Training Code
     │
     │ MLflow HTTP API
     ▼
MLflow Tracking Server
     │
     ├── Backend Store
     │       │
     │       └── SQLite database
     │
     └── Artifact Store
             │
             └── mlruns/
```

Therefore:

### Training code does NOT do this:

```text
Colab ───────► mlflow.db
```

### Instead:

```text
Colab ───────► MLflow Server ───────► mlflow.db
                         │
                         └──────────► artifacts
```

This separation becomes especially important when training, tracking, serving, and storage eventually run on different machines.

---

# 3. Initial Project Structure

The project is:

```text
open_food_mlops/
```

The relevant Docker/MLflow structure was organized as:

```text
open_food_mlops/
├── docker/
│   ├── Dockerfile.api
│   ├── Dockerfile.mlflow
│   ├── Dockerfile.pipeline
│   └── nginx.conf
├── docker-compose.yml
├── mlflow/
│   ├── mlflow.db
│   └── mlruns/
└── ...
```

The important decision was:

- Keep `Dockerfile.mlflow` inside `docker/`.
- Keep `docker-compose.yml` at the project root.
- Create a dedicated `mlflow/` directory for persistent MLflow state.

---

# 4. Why the `mlflow/` Directory Matters

Originally, the database and artifacts were located elsewhere.

They were moved into:

```text
mlflow/
├── mlflow.db
└── mlruns/
```

This gives us a clean separation between:

```text
Application source code
```

and:

```text
MLflow persistent state
```

More importantly, these files are mounted into the Docker container.

Therefore, destroying/recreating the MLflow container does **not** destroy the database or artifacts.

---

# 5. Final Dockerfile

The final `docker/Dockerfile.mlflow` is:

```dockerfile
FROM python:3.12-slim

RUN pip install --no-cache-dir \
    mlflow==3.15.2 \
    "anyio<4"

WORKDIR /mlflow

EXPOSE 5000

CMD ["mlflow", "server", \
     "--host", "0.0.0.0", \
     "--port", "5000", \
     "--backend-store-uri", "sqlite:////mlflow/mlflow.db", \
     "--artifacts-destination", "/mlflow/mlruns", \
     "--serve-artifacts", \
     "--allowed-hosts", "*"]
```

## Explanation

### Base image

```dockerfile
FROM python:3.12-slim
```

A lightweight Python image is sufficient for the MLflow server.

---

### Install MLflow

```dockerfile
RUN pip install --no-cache-dir \
    mlflow==3.15.2 \
    "anyio<4"
```

MLflow was explicitly pinned:

```text
mlflow==3.15.2
```

Pinning versions helps prevent unexpected changes when rebuilding the image.

`anyio<4` was included because of compatibility issues encountered with the environment.

---

### Working directory

```dockerfile
WORKDIR /mlflow
```

This makes `/mlflow` the working directory inside the container.

---

### Port

```dockerfile
EXPOSE 5000
```

MLflow runs on port `5000`.

---

### Backend store

```text
--backend-store-uri sqlite:////mlflow/mlflow.db
```

This tells MLflow to use:

```text
/mlflow/mlflow.db
```

inside the container.

That path is connected to the host's:

```text
./mlflow/mlflow.db
```

through Docker Compose.

The SQLite database stores MLflow metadata such as:

- experiments
- runs
- parameters
- metrics
- tags
- run metadata

---

### Artifact destination

```text
--artifacts-destination /mlflow/mlruns
```

This tells MLflow where artifacts should be stored.

The directory is connected to:

```text
./mlflow/mlruns/
```

on the host.

---

### Server-side artifact serving

```text
--serve-artifacts
```

This allows the MLflow Tracking Server to handle artifact access instead of requiring the training machine to directly access the artifact filesystem.

This is especially important once training happens remotely.

---

### Allowed hosts

```text
--allowed-hosts "*"
```

This was added because MLflow rejected requests arriving through the public tunnel with:

```text
Invalid Host header - possible DNS rebinding attack detected
```

For this temporary development/learning environment, allowing all hosts solved the issue.

**Do not blindly use this configuration for a production deployment.**

For production, the hostname should be restricted to the actual expected domain/host.

---

# 6. Final Docker Compose Configuration

The final relevant section of `docker-compose.yml` is:

```yaml
services:

  mlflow:
    build:
      context: .
      dockerfile: docker/Dockerfile.mlflow

    ports:
      - "5000:5000"

    volumes:
      - ./mlflow/mlflow.db:/mlflow/mlflow.db
      - ./mlflow/mlruns:/mlflow/mlruns
```

---

# 7. Understanding the Volume Mapping

The first mapping:

```yaml
- ./mlflow/mlflow.db:/mlflow/mlflow.db
```

means:

```text
Host:
open_food_mlops/mlflow/mlflow.db

        │
        │ Docker volume mount
        ▼

Container:
/mlflow/mlflow.db
```

The second:

```yaml
- ./mlflow/mlruns:/mlflow/mlruns
```

means:

```text
Host:
open_food_mlops/mlflow/mlruns/

        │
        │ Docker volume mount
        ▼

Container:
/mlflow/mlruns/
```

Therefore:

```text
Docker container
     │
     ├── writes metadata
     │       ↓
     │   host mlflow.db
     │
     └── writes artifacts
             ↓
         host mlruns/
```

---

# 8. Starting MLflow

Start the service with:

```bash
docker compose up -d mlflow
```

Check its status:

```bash
docker compose ps
```

The MLflow service should expose:

```text
0.0.0.0:5000 -> 5000/tcp
```

The MLflow UI is then available locally at:

```text
http://localhost:5000
```

---

# 9. Verifying Existing Experiments

One of the most important tests was opening the MLflow UI after containerization.

The existing experiments were still visible.

This proved that:

```text
Docker MLflow
      │
      ▼
existing mlflow.db
```

was working correctly.

The existing artifacts were also available.

This confirmed that the Docker volume configuration preserved the previous MLflow state.

---

# 10. Local Training → Docker MLflow

Before connecting Colab, local training was tested.

The project's configuration originally contained a SQLite tracking URI similar to:

```yaml
tracking:
  backend: mlflow
  tracking_uri: sqlite:///mlflow.db
```

The important change was to stop having the training process access SQLite directly.

Instead, the training code uses the MLflow HTTP server:

```yaml
tracking:
  backend: mlflow
  tracking_uri: http://localhost:5000
```

The same principle applies to the other configuration file.

---

# 11. Why HTTP Is Better Here

With:

```text
sqlite:///mlflow.db
```

the training process is talking directly to the database.

With:

```text
http://localhost:5000
```

the training process talks to:

```text
MLflow Tracking Server
```

and MLflow handles the backend database and artifacts.

This gives us:

```text
Python training
      │
      │ HTTP
      ▼
MLflow Server
      │
      ├── SQLite
      └── artifacts
```

rather than:

```text
Python training
      │
      └── SQLite
```

The first architecture is what allows remote training later.

---

# 12. Testing Local Training

The training pipeline was run using:

```bash
python pipelines/training_flow.py \
    --config config/experiment.yaml
```

The run succeeded.

Afterward:

- The MLflow UI showed the run.
- The database was updated.
- The artifact was stored in the correct location.

This established that the local development workflow was correct.

---

# 13. The Next Challenge: Google Colab

The next goal was:

```text
Google Colab
      │
      │ MLflow
      ▼
Local MLflow server
```

However, this does **not** work:

```python
mlflow.set_tracking_uri("http://localhost:5000")
```

inside Colab.

Why?

Because `localhost` inside Colab refers to the **Colab machine**, not the developer's laptop.

So:

```text
Colab localhost:5000
```

and:

```text
Laptop localhost:5000
```

are two completely different machines.

---

# 14. Correct Colab Architecture

The solution was to expose the local MLflow server through a temporary tunnel:

```text
Google Colab
      │
      │ HTTPS
      ▼
Public tunnel
      │
      │
      ▼
Laptop localhost:5000
      │
      ▼
Docker MLflow
```

This means Colab only needs to know the public HTTPS URL.

---

# 15. First Attempt: Cloudflare Quick Tunnel

Cloudflare Quick Tunnel was initially started with:

```bash
cloudflared tunnel --url http://localhost:5000
```

It successfully generated a public URL similar to:

```text
https://something.trycloudflare.com
```

However, the connectivity diagnostics showed:

```text
UDP Connectivity
region1 ... PASS

UDP Connectivity
region2 ... FAIL

TCP Connectivity
region1 ... PASS

TCP Connectivity
region2 ... FAIL
```

Cloudflare reported:

```text
Allow outbound QUIC traffic on port 7844 or use HTTP2.
```

Therefore the problem was related to network connectivity, not MLflow itself.

---

# 16. Cloudflare Solution: Force HTTP/2

Instead of using the default QUIC protocol, the tunnel was started using:

```bash
cloudflared tunnel \
    --protocol http2 \
    --url http://localhost:5000
```

The important option is:

```text
--protocol http2
```

This forces the tunnel to use HTTP/2 rather than QUIC.

The tunnel then worked correctly.

---

# 17. The MLflow Host Header Problem

The first Colab request produced:

```text
MlflowException:
API request to endpoint
/api/2.0/mlflow/experiments/search
failed with error code 403
```

The important part of the response was:

```text
Invalid Host header - possible DNS rebinding attack detected
```

This was extremely useful diagnostically.

It meant:

```text
Colab
   │
   ▼
Tunnel
   │
   ▼
MLflow
```

was actually working.

The request successfully reached MLflow.

MLflow itself rejected the hostname.

---

# 18. Fixing the Host Header Problem

The MLflow server was configured with:

```text
--allowed-hosts "*"
```

For this temporary development environment, this allows requests coming through the dynamically generated tunnel hostname.

After rebuilding the image and restarting MLflow, the public tunnel could successfully communicate with MLflow.

For production, the allowed host should be restricted instead of using:

```text
*
```

---

# 19. ngrok Attempt

ngrok was also considered as an alternative tunnel.

The command used was:

```bash
ngrok http 5000
```

However, ngrok rejected the connection:

```text
ERROR: authentication failed:
We do not allow agents to connect to ngrok from your IP address
```

with:

```text
ERR_NGROK_9040
```

This was an ngrok-side IP/network restriction.

It was not an MLflow problem.

Therefore, Cloudflare Tunnel was used instead.

---

# 20. Final Cloudflare Command

The working tunnel command was:

```bash
cloudflared tunnel \
    --protocol http2 \
    --url http://localhost:5000
```

It produced a temporary public URL similar to:

```text
https://something.trycloudflare.com
```

That URL was then used from Colab.

---

# 21. Testing MLflow From Colab

MLflow was installed in Colab using the same version:

```python
!pip install mlflow==3.15.2
```

Then:

```python
import mlflow

mlflow.set_tracking_uri(
    "https://YOUR-TUNNEL-URL.trycloudflare.com"
)

print(mlflow.get_tracking_uri())
```

The connection was tested using:

```python
print(mlflow.search_experiments())
```

The following experiments were successfully returned:

```text
testing-mlflow-as-service
open-food-mlops-v4
open-food-mlops-v2
open-food-mlops
pipeline_correctness_checking
Default
```

This was a major confirmation.

It proved that Colab was communicating with the MLflow server running on the local machine.

---

# 22. Final Remote Experiment Test

A small experiment was created from Colab before connecting the full training pipeline.

```python
import mlflow

MLFLOW_URI = "https://YOUR-TUNNEL-URL.trycloudflare.com"

mlflow.set_tracking_uri(MLFLOW_URI)

mlflow.set_experiment("colab-test")

with mlflow.start_run():
    mlflow.log_param("environment", "google-colab")
    mlflow.log_param("developer", "ibrahim")

    mlflow.log_metric("accuracy", 0.95)
    mlflow.log_metric("loss", 0.05)

    with open("hello_from_colab.txt", "w") as f:
        f.write("Hello from Google Colab!")

    mlflow.log_artifact("hello_from_colab.txt")

    print("Run ID:", mlflow.active_run().info.run_id)
```

This successfully created the run.

---

# 23. What This Experiment Proved

The artifact:

```text
hello_from_colab.txt
```

was created inside Google Colab.

But the artifact was uploaded through MLflow and stored by the local MLflow server.

The complete flow was:

```text
Google Colab
     │
     │ mlflow.log_artifact()
     ▼
Cloudflare Tunnel
     │
     ▼
Docker MLflow
     │
     ▼
mlflow/mlruns/
```

Likewise, parameters and metrics were sent to the MLflow Tracking Server and persisted in the backend database.

Therefore:

```text
Colab
  │
  ├── training
  ├── parameters
  ├── metrics
  └── artifacts
          │
          ▼
       MLflow
          │
          ├── SQLite metadata
          └── local artifacts
```

---

# 24. Final Working Architecture

At this point, the complete development environment is:

```text
                         ┌──────────────────────┐
                         │     Google Colab      │
                         │                      │
                         │  Training / Notebook  │
                         └──────────┬───────────┘
                                    │
                                    │ HTTPS
                                    ▼
                         ┌──────────────────────┐
                         │ Cloudflare Quick     │
                         │ Tunnel               │
                         │ HTTP/2               │
                         └──────────┬───────────┘
                                    │
                                    │
                                    ▼
┌────────────────────────────────────────────────────────────┐
│                       LOCAL MACHINE                         │
│                                                            │
│  ┌──────────────────────────────────────────────────────┐  │
│  │                 Docker MLflow                        │  │
│  │                                                      │  │
│  │              localhost:5000                          │  │
│  │                                                      │  │
│  │             MLflow Tracking Server                   │  │
│  └─────────────────────────┬────────────────────────────┘  │
│                            │                               │
│                     ┌──────┴───────┐                       │
│                     │              │                       │
│                     ▼              ▼                       │
│              ┌────────────┐  ┌─────────────┐              │
│              │ mlflow.db  │  │   mlruns/   │              │
│              │            │  │             │              │
│              │ Metadata   │  │  Artifacts  │              │
│              └────────────┘  └─────────────┘              │
│                                                            │
└────────────────────────────────────────────────────────────┘
```

---

# 25. Local Training and Colab Training Can Coexist

The architecture now supports both:

### Local training

```text
Local Python
     │
     ▼
localhost:5000
     │
     ▼
Docker MLflow
```

### Colab training

```text
Colab
     │
     ▼
Cloudflare Tunnel
     │
     ▼
localhost:5000
     │
     ▼
Docker MLflow
```

Both ultimately use the same MLflow server.

Therefore both can see the same experiments.

---

# 26. Why This Architecture Is Valuable

The major lesson is that MLflow is acting as a **service**, not merely as a Python library or local SQLite database.

The training environment can change:

```text
Laptop
Colab
GPU server
VPS
Cloud VM
```

while the MLflow Tracking Server can remain somewhere else.

The communication boundary is:

```text
MLflow Tracking URI
```

For example:

```text
http://localhost:5000
```

locally, or:

```text
https://some-public-host
```

when accessed remotely.

The training code does not need to know where the SQLite database physically exists.

---

# 27. Troubleshooting Reference

## Problem: Existing experiments disappeared

Check the Docker volume mappings.

Correct:

```yaml
volumes:
  - ./mlflow/mlflow.db:/mlflow/mlflow.db
  - ./mlflow/mlruns:/mlflow/mlruns
```

Also verify the MLflow server's backend path:

```text
sqlite:////mlflow/mlflow.db
```

Do not accidentally create a new database inside the container.

---

## Problem: `localhost:5000` doesn't work from Colab

This is expected.

Colab's:

```text
localhost
```

is not your laptop.

Use a publicly reachable tunnel URL.

---

## Problem: Cloudflare Quick Tunnel reports QUIC failures

Try forcing HTTP/2:

```bash
cloudflared tunnel \
    --protocol http2 \
    --url http://localhost:5000
```

---

## Problem: MLflow returns:

```text
403
Invalid Host header - possible DNS rebinding attack detected
```

The request reached MLflow, but MLflow rejected the hostname.

For this temporary development environment, use:

```text
--allowed-hosts "*"
```

For production, configure a specific trusted hostname instead.

---

## Problem: ngrok returns:

```text
ERR_NGROK_9040
```

This indicates that ngrok is refusing the agent connection from the current IP/network.

It is not an MLflow configuration error.

Try another tunneling solution rather than changing the MLflow database configuration.

---

# 28. Debugging Strategy Learned

When troubleshooting distributed services, don't change everything at once.

Test one layer at a time.

### Layer 1 — MLflow itself

```text
http://localhost:5000
```

If the UI works, MLflow is alive.

### Layer 2 — Database

Check that existing experiments are visible.

### Layer 3 — Artifacts

Create/log an artifact and verify it appears under:

```text
mlflow/mlruns/
```

### Layer 4 — Local training

Run the pipeline against:

```text
http://localhost:5000
```

### Layer 5 — Tunnel

Test:

```text
public tunnel → localhost:5000
```

### Layer 6 — Colab

Test:

```python
mlflow.search_experiments()
```

### Layer 7 — Remote artifact logging

Run:

```python
mlflow.log_artifact(...)
```

### Layer 8 — Full training pipeline

Only after all previous layers work should the full training pipeline be moved to Colab.

This approach makes errors much easier to isolate.

---

# 29. What We Have NOT Done Yet

This setup is intentionally a **development/learning architecture**.

We have not yet introduced:

- PostgreSQL
- MinIO
- S3/object storage
- HTTPS certificates managed by ourselves
- authentication
- production Cloudflare tunnels
- VPS deployment
- Kubernetes
- distributed workers
- model registry deployment architecture
- production monitoring

These are not currently necessary.

The purpose of the current setup is to understand the fundamental architecture first.

---

# 30. Future Production Architecture

Later, when the project moves toward production, the architecture can evolve into something like:

```text
                    Training
                       │
                       ▼
              MLflow Tracking Server
                       │
                ┌──────┴──────┐
                ▼             ▼
           PostgreSQL     Object Storage
           Metadata        Artifacts
                │             │
                └──────┬──────┘
                       │
                       ▼
                 Model Registry
                       │
                       ▼
                  VPS / Server
                       │
                       ▼
                 Model Serving
```

The important thing is that this is an **evolution** of the architecture already learned, not a completely different concept.

---

# 31. Key Concepts to Remember

### MLflow Tracking Server

The HTTP service that training code communicates with.

```text
Training → MLflow Server
```

### Backend Store

Stores MLflow metadata.

Current development choice:

```text
SQLite
```

### Artifact Store

Stores files generated by runs.

Current development choice:

```text
mlflow/mlruns/
```

### Tracking URI

Tells the MLflow client where the Tracking Server is.

Local:

```text
http://localhost:5000
```

Remote:

```text
https://<public-host>
```

### Docker Volume

Keeps MLflow state outside the container.

```text
Host mlflow/
       ↕
Container /mlflow/
```

### Tunnel

Provides temporary network access from the Internet to the local MLflow server.

```text
Internet → Tunnel → localhost:5000
```

---

# 32. Final Checklist

When recreating this setup in the future:

- [ ] Create `mlflow/`.
- [ ] Put `mlflow.db` inside `mlflow/`.
- [ ] Put `mlruns/` inside `mlflow/`.
- [ ] Keep `Dockerfile.mlflow` inside `docker/`.
- [ ] Keep `docker-compose.yml` at project root.
- [ ] Mount `mlflow.db` into the container.
- [ ] Mount `mlruns/` into the container.
- [ ] Run MLflow on port `5000`.
- [ ] Configure training to use `http://localhost:5000`.
- [ ] Verify existing experiments.
- [ ] Verify local training.
- [ ] Verify artifacts.
- [ ] Start a tunnel for remote training.
- [ ] If Cloudflare QUIC fails, try `--protocol http2`.
- [ ] If MLflow reports an invalid Host header, configure `--allowed-hosts`.
- [ ] Test `mlflow.search_experiments()` from the remote machine.
- [ ] Test a tiny remote run.
- [ ] Test remote artifact logging.
- [ ] Only then connect the full training pipeline.

---

# 33. The One-Sentence Mental Model

> **Training can happen anywhere, but it should communicate with MLflow through the Tracking Server; the Tracking Server is responsible for writing metadata to the backend store and handling artifacts.**

Current setup:

```text
                 TRAINING
              ┌─────────────┐
              │ Local/Colab │
              └──────┬──────┘
                     │
                     │ HTTP
                     ▼
              ┌─────────────┐
              │   MLflow    │
              │   Server    │
              └──────┬──────┘
                     │
             ┌───────┴────────┐
             ▼                ▼
        ┌─────────┐      ┌──────────┐
        │ SQLite  │      │  mlruns  │
        │  DB     │      │ Artifacts│
        └─────────┘      └──────────┘
```

This is the foundation for the next stage of the `open_food_mlops` project.