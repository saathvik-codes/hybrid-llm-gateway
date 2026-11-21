"""
AWS Lambda entry point: the same FastAPI app behind API Gateway (HTTP API), through the Mangum adapter.
Deployed by template.yaml (AWS SAM). gRPC is off on Lambda (GRPC_PORT=0): API Gateway speaks HTTP only.
"""
from mangum import Mangum

from app.main import app

handler = Mangum(app, lifespan="auto")
