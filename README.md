## Running This Project with Docker

This project provides Dockerfiles and a `docker-compose.yml` for running its main components in isolated containers. Below are the project-specific instructions and requirements for using Docker with this repository.

### Project-Specific Docker Requirements

- **Python Versions:**
  - Root and config services use `python:3.11-slim`.
  - The `een/mcp` service uses `python:3.13-slim`.
- **Dependencies:**
  - Root and config images install scientific Python libraries (e.g., `numpy`, `pandas`, `matplotlib`, `seaborn`).
  - The `een/mcp` service uses Poetry (`1.8.5`) for dependency management and expects `pyproject.toml` and `poetry.lock` in the project root.
- **User Setup:**
  - All containers run as non-root users for security.

### Environment Variables

- No required environment variables are specified in the Dockerfiles or compose file by default.
- If you have environment-specific settings, you can uncomment the `env_file` lines in the `docker-compose.yml` and provide the appropriate `.env` files in the respective directories.

### Build and Run Instructions

1. **Clone the repository and ensure Docker and Docker Compose are installed.**
2. **Build and start all services:**
   ```sh
   docker compose up --build
   ```
   This will build and run the following services:
   - `python-root`: Runs the demo script (`run_demo.py`) in the root context.
   - `python-config`: Runs `mcp_consciousness_server.py` from the `config` directory.
   - `python-een-mcp`: Runs the `een.mcp.unity_server` module by default.

3. **Customizing Service Behavior:**
   - To run a different server in the `python-een-mcp` service, override the `entrypoint` or `CMD` in the compose file or at runtime.

### Special Configuration

- **Poetry:** The `een/mcp` service requires `pyproject.toml` and `poetry.lock` in the project root for dependency installation.
- **No external services** (databases, caches, etc.) are required or configured by default.
- **No persistent volumes** are defined; all data is ephemeral unless you add volumes.

### Ports

- **No ports are exposed by default** in any service. If you need to expose ports (e.g., for server access), add the `ports` section to the relevant service in `docker-compose.yml`.

---

_This section was updated to reflect the current Docker-based setup and usage for this project. For more details on individual services or advanced configuration, refer to the documentation in the `docs/` directory or the respective service directories._
