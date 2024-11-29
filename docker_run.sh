# 1. 检查 Docker 环境
docker info


# 2. 测试容器

# 2.1 单次测试
docker run -it --gpus all --rm -p 8000:8000 -p 8001:8001 -p 8002:8002 -v ${pwd}/model_repository:/models -v ${pwd}/workspace:/workspace -v ${pwd}/data:/workspace/data nvcr.io/nvidia/tritonserver:24.10-py3 /bin/bash
tritonserver --model-repository=/models --model-control-mode=poll --repository-poll-secs=30

# 2.2 常驻后台
docker run -d --gpus all -p 8000:8000 -p 8001:8001 -p 8002:8002 -v ${pwd}/model_repository:/models -v ${pwd}/workspace:/workspace -v ${pwd}/data:/workspace/data nvcr.io/nvidia/tritonserver:24.10-py3 tail -f /dev/null
docker stop [CONTAINER_ID]
docker start [CONTAINER_ID]
docker exec -it [CONTAINER_ID] /bin/bash
tritonserver --model-repository=/models --model-control-mode=poll --repository-poll-secs=30


# 3. 运行容器
docker run -d --shm-size=1g --gpus all -p 8000:8000 -p 8001:8001 -p 8002:8002 -v ${pwd}/model_repository:/models -v ${pwd}/workspace:/workspace nvcr.io/nvidia/tritonserver:24.10-py3 -v ${pwd}/data:/workspace/data tritonserver --model-repository=/models
