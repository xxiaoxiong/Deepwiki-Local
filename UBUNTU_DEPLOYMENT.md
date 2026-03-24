# Ubuntu 环境部署指南

## 镜像文件信息

- **文件名**: `deepwiki-latest.tar`
- **大小**: ~976 MB (1,022,929,408 字节)
- **创建时间**: 2026-03-24

## 部署步骤

### 1. 传输镜像文件到 Ubuntu 服务器

使用 SCP 或其他文件传输工具将镜像文件传输到 Ubuntu 服务器：

```bash
# 从 Windows 传输到 Ubuntu（在 PowerShell 中执行）
scp deepwiki-latest.tar user@ubuntu-server:/path/to/destination/

# 或使用 WinSCP、FileZilla 等图形化工具
```

### 2. 在 Ubuntu 服务器上导入镜像

```bash
# SSH 登录到 Ubuntu 服务器
ssh user@ubuntu-server

# 导入 Docker 镜像
docker load -i /path/to/deepwiki-latest.tar

# 验证镜像已导入
docker images | grep deepwiki
```

### 3. 运行容器

#### 方式一：使用 docker run 命令

```bash
docker run -d \
  --name deepwiki \
  -p 3001:3001 \
  -p 8001:8001 \
  -e PORT=8001 \
  -e WEB_PORT=3001 \
  -e NODE_ENV=production \
  -e SERVER_BASE_URL=http://localhost:8001 \
  -e OPENAI_API_KEY=not-needed \
  -e OPENAI_BASE_URL=http://localhost:8000/v1 \
  -e VLLM_API_KEY=not-needed \
  -e VLLM_BASE_URL=http://localhost:8000/v1 \
  -e TIKTOKEN_CACHE_DIR=/app/tiktoken_cache \
  -e NEXT_TELEMETRY_DISABLED=1 \
  -e TRANSFORMERS_OFFLINE=1 \
  -e HF_HUB_OFFLINE=1 \
  -v ~/.adalflow:/root/.adalflow \
  --memory=6g \
  --memory-reservation=2g \
  deepwiki:latest
```

#### 方式二：使用 docker-compose（推荐）

1. 将 `docker-compose.yml` 文件也传输到 Ubuntu 服务器

2. 在包含 `docker-compose.yml` 的目录中执行：

```bash
# 启动服务
docker-compose up -d

# 查看日志
docker-compose logs -f

# 停止服务
docker-compose down
```

### 4. 验证部署

```bash
# 检查容器状态
docker ps | grep deepwiki

# 查看容器日志
docker logs deepwiki

# 测试服务
curl http://localhost:3001
curl http://localhost:8001/health
```

### 5. 访问应用

- **前端界面**: http://服务器IP:3001
- **后端 API**: http://服务器IP:8001

## 环境变量说明

| 变量名 | 默认值 | 说明 |
|--------|--------|------|
| `PORT` | 8001 | 后端 API 端口 |
| `WEB_PORT` | 3001 | 前端服务端口 |
| `OPENAI_BASE_URL` | http://localhost:8000/v1 | OpenAI 兼容 API 地址 |
| `VLLM_BASE_URL` | http://localhost:8000/v1 | vLLM 服务地址 |
| `LOG_LEVEL` | INFO | 日志级别 |

## 持久化数据

容器会将数据持久化到以下位置：
- `~/.adalflow` - 存储仓库和嵌入向量数据

## 资源限制

- **内存上限**: 6GB
- **内存预留**: 2GB

根据实际情况调整 `--memory` 和 `--memory-reservation` 参数。

## 故障排查

### 容器无法启动

```bash
# 查看详细日志
docker logs deepwiki --tail 100

# 检查端口占用
netstat -tlnp | grep -E '3001|8001'
```

### 端口访问问题

```bash
# 检查防火墙规则
sudo ufw status
sudo ufw allow 3001
sudo ufw allow 8001

# 或使用 iptables
sudo iptables -I INPUT -p tcp --dport 3001 -j ACCEPT
sudo iptables -I INPUT -p tcp --dport 8001 -j ACCEPT
```

### 清理和重启

```bash
# 停止并删除容器
docker stop deepwiki
docker rm deepwiki

# 重新运行容器（使用上面的 docker run 命令）
```

## 更新镜像

当有新版本时：

1. 传输新的 tar 文件到服务器
2. 停止并删除旧容器
3. 删除旧镜像：`docker rmi deepwiki:latest`
4. 导入新镜像：`docker load -i deepwiki-latest.tar`
5. 重新启动容器

## 备份数据

```bash
# 备份持久化数据
tar -czf adalflow-backup-$(date +%Y%m%d).tar.gz ~/.adalflow
```
