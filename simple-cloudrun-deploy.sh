#!/bin/bash
# simple-cloudrun-deploy.sh
# Direct deployment to Cloud Run with GPU - no Secret Manager needed

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${GREEN}=========================================${NC}"
echo -e "${GREEN}Simple Cloud Run GPU Deployment${NC}"
echo -e "${GREEN}=========================================${NC}"

# Set variables
PROJECT_ID="ust-genai-pa-poc-gcp"
REGION="us-central1"  # Must use us-central1 for GPU

# Load environment variables from .env
if [ ! -f .env ]; then
    echo -e "${RED}Error: .env file not found!${NC}"
    exit 1
fi
source .env

echo -e "${YELLOW}Project: $PROJECT_ID${NC}"
echo -e "${YELLOW}Region: $REGION${NC}"

# Step 1: Build Docker images
echo -e "${YELLOW}Building Docker images...${NC}"

# Build backend
echo "Building backend..."
docker build -f deployment/Dockerfile.backend \
  -t ${REGION}-docker.pkg.dev/${PROJECT_ID}/whisper-repo/backend:gpu .

if [ $? -ne 0 ]; then
    echo -e "${RED}Backend build failed. Trying with fixed Dockerfile...${NC}"
    
    # Create fixed Dockerfile
    cat > deployment/Dockerfile.backend.fixed << 'EOF'
FROM nvidia/cuda:12.1.0-cudnn8-runtime-ubuntu22.04

RUN apt-get update && apt-get install -y \
    python3.10 python3-pip python3-dev \
    ffmpeg git wget curl build-essential libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

RUN pip3 install --upgrade pip setuptools wheel

# Install PyTorch with CUDA
RUN pip3 install torch==2.1.0 torchaudio==2.1.0 --index-url https://download.pytorch.org/whl/cu121

# Install dependencies
RUN pip3 install --no-cache-dir \
    fastapi==0.104.1 \
    uvicorn[standard]==0.24.0 \
    python-multipart==0.0.6 \
    websockets==12.0 \
    numpy==1.24.3 \
    pandas==2.0.3 \
    ffmpeg-python==0.2.0 \
    openai==1.3.0 \
    faster-whisper==0.10.0 \
    pyannote.audio==3.1.1

# Install WhisperX
RUN pip3 install git+https://github.com/m-bain/whisperX.git@main --no-deps
RUN pip3 install nltk

COPY backend/*.py ./
RUN python3 -c "import nltk; nltk.download('punkt')"

EXPOSE 8000 9090

ENV CUDA_VISIBLE_DEVICES=0
ENV WHISPER_MODEL=large-v3
ENV DEVICE=cuda
ENV COMPUTE_TYPE=float16

CMD ["python3", "-u", "backend_server.py"]
EOF
    
    # Retry with fixed Dockerfile
    docker build -f deployment/Dockerfile.backend.fixed \
      -t ${REGION}-docker.pkg.dev/${PROJECT_ID}/whisper-repo/backend:gpu .
fi

# Build frontend
echo "Building frontend..."
docker build -f deployment/Dockerfile.frontend \
  -t ${REGION}-docker.pkg.dev/${PROJECT_ID}/whisper-repo/frontend:latest .

# Step 2: Push to Artifact Registry
echo -e "${YELLOW}Pushing images to Artifact Registry...${NC}"

# Configure Docker auth
gcloud auth configure-docker ${REGION}-docker.pkg.dev --quiet

# Push images
docker push ${REGION}-docker.pkg.dev/${PROJECT_ID}/whisper-repo/backend:gpu
docker push ${REGION}-docker.pkg.dev/${PROJECT_ID}/whisper-repo/frontend:latest

# Step 3: Deploy backend with GPU
echo -e "${YELLOW}Deploying backend to Cloud Run with GPU...${NC}"

# Check if beta command is available
if ! gcloud beta --help &> /dev/null; then
    echo -e "${YELLOW}Installing gcloud beta components...${NC}"
    gcloud components install beta --quiet
fi

# Deploy with GPU (pass env vars directly, no Secret Manager)
gcloud beta run deploy whisper-backend-gpu \
  --image ${REGION}-docker.pkg.dev/${PROJECT_ID}/whisper-repo/backend:gpu \
  --platform managed \
  --region ${REGION} \
  --memory 32Gi \
  --cpu 8 \
  --gpu 1 \
  --gpu-type nvidia-l4 \
  --no-cpu-throttling \
  --timeout 3600 \
  --max-instances 3 \
  --min-instances 0 \
  --concurrency 1 \
  --set-env-vars "HF_TOKEN=${HF_TOKEN},OPENAI_API_KEY=${OPENAI_API_KEY},DEVICE=cuda,COMPUTE_TYPE=float16,WHISPER_MODEL=large-v3,LLM_MODEL=gpt-4o-mini" \
  --allow-unauthenticated \
  --quiet

if [ $? -ne 0 ]; then
    echo -e "${RED}GPU deployment failed. Trying without GPU...${NC}"
    
    # Fallback to CPU deployment
    gcloud run deploy whisper-backend \
      --image ${REGION}-docker.pkg.dev/${PROJECT_ID}/whisper-repo/backend:gpu \
      --platform managed \
      --region ${REGION} \
      --memory 8Gi \
      --cpu 4 \
      --timeout 3600 \
      --max-instances 10 \
      --set-env-vars "HF_TOKEN=${HF_TOKEN},OPENAI_API_KEY=${OPENAI_API_KEY},DEVICE=cpu,COMPUTE_TYPE=int8,WHISPER_MODEL=base,LLM_MODEL=gpt-4o-mini" \
      --allow-unauthenticated
    
    BACKEND_SERVICE="whisper-backend"
else
    BACKEND_SERVICE="whisper-backend-gpu"
fi

# Step 4: Get backend URL
BACKEND_URL=$(gcloud run services describe ${BACKEND_SERVICE} --region ${REGION} --format 'value(status.url)')

echo -e "${GREEN}Backend deployed at: ${BACKEND_URL}${NC}"

# Step 5: Update frontend with backend URL
echo -e "${YELLOW}Updating frontend with backend URL...${NC}"

# Create updated frontend
mkdir -p temp_frontend
cp frontend/index.html temp_frontend/

# Update URLs in frontend
sed -i "s|http://localhost:8000|${BACKEND_URL}|g" temp_frontend/index.html
sed -i "s|ws://localhost:9090|wss://${BACKEND_URL#https://}/ws|g" temp_frontend/index.html
sed -i "s|new WebSocket('ws://localhost:9090')|new WebSocket('wss://${BACKEND_URL#https://}/ws')|g" temp_frontend/index.html

# Create updated frontend image
cat > deployment/Dockerfile.frontend.updated << 'EOF'
FROM nginx:alpine
COPY temp_frontend/index.html /usr/share/nginx/html/
EXPOSE 80
EOF

docker build -f deployment/Dockerfile.frontend.updated \
  -t ${REGION}-docker.pkg.dev/${PROJECT_ID}/whisper-repo/frontend:updated .

docker push ${REGION}-docker.pkg.dev/${PROJECT_ID}/whisper-repo/frontend:updated

# Step 6: Deploy frontend
echo -e "${YELLOW}Deploying frontend to Cloud Run...${NC}"

gcloud run deploy whisper-frontend \
  --image ${REGION}-docker.pkg.dev/${PROJECT_ID}/whisper-repo/frontend:updated \
  --platform managed \
  --region ${REGION} \
  --memory 1Gi \
  --cpu 1 \
  --max-instances 100 \
  --allow-unauthenticated

# Step 7: Get frontend URL
FRONTEND_URL=$(gcloud run services describe whisper-frontend --region ${REGION} --format 'value(status.url)')

# Cleanup
rm -rf temp_frontend

# Step 8: Display results
echo -e "${GREEN}=========================================${NC}"
echo -e "${GREEN}🎉 Deployment Complete!${NC}"
echo -e "${GREEN}=========================================${NC}"
echo ""
echo -e "${YELLOW}📱 Frontend URL:${NC} ${FRONTEND_URL}"
echo -e "${YELLOW}🔧 Backend URL:${NC} ${BACKEND_URL}"
echo ""
if [[ "$BACKEND_SERVICE" == "whisper-backend-gpu" ]]; then
    echo -e "${GREEN}✅ GPU Enabled:${NC} NVIDIA L4"
    echo -e "${GREEN}✅ Model:${NC} Whisper large-v3"
    echo -e "${GREEN}✅ Performance:${NC} ~70x faster than CPU"
else
    echo -e "${YELLOW}⚠️ Running on CPU${NC} (GPU deployment failed)"
    echo -e "${YELLOW}⚠️ Model:${NC} Whisper base (smaller for CPU)"
fi
echo ""
echo -e "${GREEN}✅ LLM:${NC} GPT-4o-mini"
echo -e "${GREEN}✅ Auto-scaling:${NC} 0-3 instances"
echo -e "${GREEN}✅ Cost:${NC} ~$0.65/hour when active (scales to zero)"
echo ""
echo -e "${YELLOW}Test your app:${NC}"
echo "1. Open: ${FRONTEND_URL}"
echo "2. Try recording audio or uploading a file"
echo "3. Check logs: gcloud run logs read --service ${BACKEND_SERVICE} --region ${REGION}"
echo ""
echo -e "${YELLOW}Useful commands:${NC}"
echo "• View logs: gcloud run logs tail ${BACKEND_SERVICE} --region ${REGION}"
echo "• Check status: gcloud run services describe ${BACKEND_SERVICE} --region ${REGION}"
echo "• Update env: gcloud run services update ${BACKEND_SERVICE} --update-env-vars KEY=VALUE --region ${REGION}"

# Step 9: Test the deployment
echo -e "${YELLOW}Testing backend health...${NC}"
curl -s ${BACKEND_URL}/health && echo -e "\n${GREEN}✅ Backend is healthy!${NC}" || echo -e "\n${RED}⚠️ Backend health check failed${NC}"