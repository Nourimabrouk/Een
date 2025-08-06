#!/usr/bin/env python3
"""
Een Cloud Deployment Script
===========================

Deploy the Een framework to various cloud platforms for global access.
"""

import os
import sys
import subprocess
import json
import yaml
from pathlib import Path
from typing import Dict, Any, List
import argparse


class EenCloudDeploy:
    def __init__(self):
        self.project_root = Path(__file__).parent
        self.deployment_configs = {
            "aws": {
                "service": "AWS Lambda + API Gateway",
                "region": "us-east-1",
                "runtime": "python3.11",
                "memory": "512MB",
                "timeout": 30,
            },
            "gcp": {
                "service": "Google Cloud Functions",
                "region": "us-central1",
                "runtime": "python311",
                "memory": "512MB",
                "timeout": 540,
            },
            "azure": {
                "service": "Azure Functions",
                "region": "East US",
                "runtime": "python3.11",
                "memory": "512MB",
                "timeout": 300,
            },
            "heroku": {
                "service": "Heroku",
                "buildpack": "heroku/python",
                "dyno": "basic",
            },
            "railway": {"service": "Railway", "region": "us-east-1"},
            "render": {"service": "Render", "region": "us-east-1", "plan": "free"},
        }

    def deploy(self, platform: str, environment: str = "production"):
        """Deploy to specified platform"""
        print(f"🚀 Deploying Een to {platform.upper()}")
        print("=" * 50)

        if platform not in self.deployment_configs:
            print(f"❌ Unsupported platform: {platform}")
            print(f"Supported platforms: {', '.join(self.deployment_configs.keys())}")
            return False

        try:
            if platform == "aws":
                return self.deploy_aws(environment)
            elif platform == "gcp":
                return self.deploy_gcp(environment)
            elif platform == "azure":
                return self.deploy_azure(environment)
            elif platform == "heroku":
                return self.deploy_heroku(environment)
            elif platform == "railway":
                return self.deploy_railway(environment)
            elif platform == "render":
                return self.deploy_render(environment)
        except Exception as e:
            print(f"❌ Deployment failed: {e}")
            return False

    def deploy_aws(self, environment: str):
        """Deploy to AWS"""
        print("☁️  Deploying to AWS...")

        # Create AWS Lambda function
        lambda_function = {
            "FunctionName": f"een-framework-{environment}",
            "Runtime": "python3.11",
            "Handler": "lambda_handler.handler",
            "Code": {"ZipFile": "lambda_function.zip"},
            "Role": "arn:aws:iam::ACCOUNT:role/lambda-role",
            "Timeout": 30,
            "MemorySize": 512,
            "Environment": {
                "Variables": {"ENVIRONMENT": environment, "PYTHONPATH": "/var/task"}
            },
        }

        # Create Lambda handler
        lambda_handler_content = '''import json
import sys
from pathlib import Path

# Add project to path
sys.path.insert(0, '/var/task')

from een_server import app
from fastapi import Request
from mangum import Mangum

handler = Mangum(app)

def lambda_handler(event, context):
    """AWS Lambda handler for Een Framework"""
    return handler(event, context)
'''

        lambda_path = self.project_root / "lambda_handler.py"
        with open(lambda_path, "w") as f:
            f.write(lambda_handler_content)

        # Create deployment package
        self.create_deployment_package()

        print("✅ AWS deployment configuration created")
        print("📝 Next steps:")
        print("1. Install AWS CLI: pip install awscli")
        print("2. Configure AWS credentials: aws configure")
        print(
            "3. Deploy: aws lambda create-function --cli-input-json lambda_function.json"
        )

        return True

    def deploy_gcp(self, environment: str):
        """Deploy to Google Cloud Platform"""
        print("☁️  Deploying to GCP...")

        # Create Cloud Function configuration
        function_config = {
            "name": f"een-framework-{environment}",
            "runtime": "python311",
            "entry_point": "main",
            "memory": "512MB",
            "timeout": "540s",
            "region": "us-central1",
            "trigger_http": True,
            "allow_unauthenticated": True,
        }

        # Create main function
        main_function_content = '''import functions_framework
from een_server import app

@functions_framework.http
def main(request):
    """Google Cloud Function entry point"""
    return app(request)
'''

        main_path = self.project_root / "main.py"
        with open(main_path, "w") as f:
            f.write(main_function_content)

        # Create requirements for Cloud Functions
        cloud_requirements = [
            "functions-framework==3.*",
            "fastapi==0.100.0",
            "uvicorn==0.23.0",
            "mangum==0.17.0",
        ]

        cloud_req_path = self.project_root / "requirements-cloud.txt"
        with open(cloud_req_path, "w") as f:
            f.write("\n".join(cloud_requirements))

        print("✅ GCP deployment configuration created")
        print("📝 Next steps:")
        print("1. Install Google Cloud SDK")
        print("2. Authenticate: gcloud auth login")
        print(
            "3. Deploy: gcloud functions deploy een-framework --runtime python311 --trigger-http"
        )

        return True

    def deploy_azure(self, environment: str):
        """Deploy to Azure"""
        print("☁️  Deploying to Azure...")

        # Create Azure Function configuration
        host_json = {
            "version": "2.0",
            "logging": {
                "applicationInsights": {
                    "samplingSettings": {"isEnabled": True, "excludedTypes": "Request"}
                }
            },
            "extensionBundle": {
                "id": "Microsoft.Azure.Functions.ExtensionBundle",
                "version": "[3.*, 4.0.0)",
            },
        }

        host_path = self.project_root / "host.json"
        with open(host_path, "w") as f:
            json.dump(host_json, f, indent=2)

        # Create function app
        function_app_content = '''import azure.functions as func
import logging
from een_server import app

app = func.FunctionApp()

@app.function_name(name="EenFramework")
@app.route(route="een")
def een_framework(req: func.HttpRequest) -> func.HttpResponse:
    """Azure Function for Een Framework"""
    return app(req)
'''

        function_path = self.project_root / "function_app.py"
        with open(function_path, "w") as f:
            f.write(function_app_content)

        print("✅ Azure deployment configuration created")
        print("📝 Next steps:")
        print("1. Install Azure CLI")
        print("2. Login: az login")
        print(
            "3. Deploy: az functionapp create --name een-framework --consumption-plan-location eastus"
        )

        return True

    def deploy_heroku(self, environment: str):
        """Deploy to Heroku"""
        print("☁️  Deploying to Heroku...")

        # Create Procfile
        procfile_content = """web: uvicorn een_server:app --host 0.0.0.0 --port $PORT
"""

        procfile_path = self.project_root / "Procfile"
        with open(procfile_path, "w") as f:
            f.write(procfile_content)

        # Create runtime.txt
        runtime_content = """python-3.11.0
"""

        runtime_path = self.project_root / "runtime.txt"
        with open(runtime_path, "w") as f:
            f.write(runtime_content)

        # Create app.json
        app_json = {
            "name": "Een Framework",
            "description": "Unity Mathematics and Consciousness Computing",
            "repository": "https://github.com/Nourimabrouk/Een",
            "keywords": ["python", "mathematics", "consciousness", "unity"],
            "env": {"PYTHONPATH": {"description": "Python path", "value": "/app"}},
            "buildpacks": [{"url": "heroku/python"}],
        }

        app_json_path = self.project_root / "app.json"
        with open(app_json_path, "w") as f:
            json.dump(app_json, f, indent=2)

        print("✅ Heroku deployment configuration created")
        print("📝 Next steps:")
        print("1. Install Heroku CLI")
        print("2. Login: heroku login")
        print("3. Create app: heroku create een-framework")
        print("4. Deploy: git push heroku main")

        return True

    def deploy_railway(self, environment: str):
        """Deploy to Railway"""
        print("☁️  Deploying to Railway...")

        # Create railway.json
        railway_config = {
            "$schema": "https://railway.app/railway.schema.json",
            "build": {"builder": "NIXPACKS"},
            "deploy": {
                "startCommand": "uvicorn een_server:app --host 0.0.0.0 --port $PORT",
                "healthcheckPath": "/health",
                "healthcheckTimeout": 100,
                "restartPolicyType": "ON_FAILURE",
                "restartPolicyMaxRetries": 10,
            },
        }

        railway_path = self.project_root / "railway.json"
        with open(railway_path, "w") as f:
            json.dump(railway_config, f, indent=2)

        print("✅ Railway deployment configuration created")
        print("📝 Next steps:")
        print("1. Install Railway CLI: npm install -g @railway/cli")
        print("2. Login: railway login")
        print("3. Deploy: railway up")

        return True

    def deploy_render(self, environment: str):
        """Deploy to Render"""
        print("☁️  Deploying to Render...")

        # Create render.yaml
        render_config = {
            "services": [
                {
                    "type": "web",
                    "name": "een-framework",
                    "env": "python",
                    "plan": "free",
                    "buildCommand": "pip install -r requirements.txt",
                    "startCommand": "uvicorn een_server:app --host 0.0.0.0 --port $PORT",
                    "healthCheckPath": "/health",
                    "envVars": [
                        {"key": "PYTHONPATH", "value": "/opt/render/project/src"}
                    ],
                }
            ]
        }

        render_path = self.project_root / "render.yaml"
        with open(render_path, "w") as f:
            yaml.dump(render_config, f, default_flow_style=False)

        print("✅ Render deployment configuration created")
        print("📝 Next steps:")
        print("1. Connect your GitHub repository to Render")
        print("2. Create a new Web Service")
        print("3. Deploy automatically on push")

        return True

    def create_deployment_package(self):
        """Create deployment package for cloud platforms"""
        print("📦 Creating deployment package...")

        # Create .dockerignore
        dockerignore_content = """__pycache__
*.pyc
*.pyo
*.pyd
.Python
env
pip-log.txt
pip-delete-this-directory.txt
.tox
.coverage
.coverage.*
.cache
nosetests.xml
coverage.xml
*.cover
*.log
.git
.mypy_cache
.pytest_cache
.hypothesis
.DS_Store
"""

        dockerignore_path = self.project_root / ".dockerignore"
        with open(dockerignore_path, "w") as f:
            f.write(dockerignore_content)

        # Create .gitignore additions
        gitignore_additions = """
# Cloud deployment
*.zip
lambda_function.json
.env.local
.railway
"""

        gitignore_path = self.project_root / ".gitignore"
        if gitignore_path.exists():
            with open(gitignore_path, "a") as f:
                f.write(gitignore_additions)

        print("✅ Deployment package created")

    def create_global_deployment_script(self):
        """Create a script to deploy to all platforms"""
        print("🌍 Creating global deployment script...")

        global_deploy_content = '''#!/usr/bin/env python3
"""
Een Global Deployment Script
============================

Deploy to all supported cloud platforms.
"""

import subprocess
import sys
from pathlib import Path

def deploy_all():
    """Deploy to all platforms"""
    platforms = ["aws", "gcp", "azure", "heroku", "railway", "render"]
    
    print("🚀 Een Global Deployment")
    print("=" * 40)
    
    for platform in platforms:
        print(f"\\n📦 Deploying to {platform.upper()}...")
        try:
            result = subprocess.run([
                sys.executable, "cloud_deploy.py", 
                "--platform", platform, 
                "--environment", "production"
            ], check=True)
            print(f"✅ {platform.upper()} deployment successful")
        except subprocess.CalledProcessError:
            print(f"❌ {platform.upper()} deployment failed")
    
    print("\\n🎉 Global deployment completed!")

if __name__ == "__main__":
    deploy_all()
'''

        global_deploy_path = self.project_root / "deploy_all.py"
        with open(global_deploy_path, "w") as f:
            f.write(global_deploy_content)

        if os.name != "nt":  # Not Windows
            os.chmod(global_deploy_path, 0o755)

        print("✅ Global deployment script created")


def main():
    parser = argparse.ArgumentParser(
        description="Deploy Een Framework to cloud platforms"
    )
    parser.add_argument(
        "--platform",
        choices=["aws", "gcp", "azure", "heroku", "railway", "render", "all"],
        required=True,
        help="Cloud platform to deploy to",
    )
    parser.add_argument(
        "--environment",
        default="production",
        choices=["development", "staging", "production"],
        help="Deployment environment",
    )

    args = parser.parse_args()

    deployer = EenCloudDeploy()

    if args.platform == "all":
        deployer.create_global_deployment_script()
        print("🌍 Global deployment script created!")
        print("Run: python deploy_all.py")
    else:
        success = deployer.deploy(args.platform, args.environment)
        if success:
            print(f"✅ {args.platform.upper()} deployment configuration created!")
        else:
            print(f"❌ {args.platform.upper()} deployment failed!")


if __name__ == "__main__":
    main()
