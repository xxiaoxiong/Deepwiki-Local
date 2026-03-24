# Deepwiki Ubuntu 部署指南

本指南将帮助你在 Ubuntu 服务器上部署 Deepwiki Docker 镜像，并允许其他电脑通过网络访问。

## 前置要求

- Ubuntu 服务器（推荐 20.04 或更高版本）
- Docker 和 Docker Compose 已安装
- 至少 6GB 可用内存
- 网络访问权限

## 步骤 1: 传输镜像文件

将导出的镜像文件 `deepwiki-latest.tar` (约 976 MB) 传输到 Ubuntu 服务器。

### 方法 A: 使用 SCP
```bash
# 在 Windows PowerShell 中执行
scp deepwiki-latest.tar username@ubuntu-server-ip:/home/username/
```

### 方法 B: 使用 U盘或其他存储设备
直接复制文件到 Ubuntu 服务器

## 步骤 2: 在 Ubuntu 服务器上加载镜像

```bash
# SSH 登录到 Ubuntu 服务器
ssh username@ubuntu-server-ip

# 加载 Docker 镜像
docker load -i deepwiki-latest.tar

# 验证镜像已加载
docker images | grep deepwiki
```

你应该看到类似输出：
```
deepwiki    latest    5650d7a1b341    xxx    1GB
```

## 步骤 3: 创建部署目录和配置文件

```bash
# 创建项目目录
mkdir -p ~/deepwiki
cd ~/deepwiki

# 创建 docker-compose.yml 文件
cat > docker-compose.yml << 'EOF'
name: deepwiki
services:
  deepwiki:
    image: deepwiki:latest
    ports:
      - "3001:3001"
      - "8001:8001"
    environment:
      - PORT=8001
      - WEB_PORT=3001
      - NODE_ENV=production
      - SERVER_BASE_URL=http://localhost:8001
      - LOG_LEVEL=INFO
      - LOG_FILE_PATH=api/logs/application.log
      # OpenAI/DeepSeek API 配置
      - OPENAI_API_KEY=${OPENAI_API_KEY:-your-api-key-here}
      - OPENAI_BASE_URL=${OPENAI_BASE_URL:-https://api.deepseek.com/v1}
      # vLLM 配置（如果使用本地 vLLM）
      - VLLM_API_KEY=${VLLM_API_KEY:-not-needed}
      - VLLM_BASE_URL=${VLLM_BASE_URL:-http://localhost:8000/v1}
      # 离线/内网环境设置
      - TIKTOKEN_CACHE_DIR=/app/tiktoken_cache
      - NEXT_TELEMETRY_DISABLED=1
      - TRANSFORMERS_OFFLINE=1
      - HF_HUB_OFFLINE=1
    volumes:
      - ~/.adalflow:/root/.adalflow      # 持久化数据
    restart: unless-stopped
    # 资源限制
    mem_limit: 6g
    mem_reservation: 2g
    # 健康检查
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8001/health"]
      interval: 60s
      timeout: 10s
      retries: 3
      start_period: 30s
EOF
```

## 步骤 4: 配置环境变量（可选）

如果你需要自定义配置，创建 `.env` 文件：

```bash
cat > .env << 'EOF'
# API 配置
OPENAI_API_KEY=your-actual-api-key
OPENAI_BASE_URL=https://api.deepseek.com/v1

# 端口配置（如果需要修改）
PORT=8001
WEB_PORT=3001

# 日志级别
LOG_LEVEL=INFO
EOF
```

## 步骤 5: 启动服务

```bash
# 启动容器
docker-compose up -d

# 查看容器状态
docker-compose ps

# 查看日志
docker-compose logs -f
```

## 步骤 6: 配置防火墙（允许外部访问）

### 使用 UFW（Ubuntu 默认防火墙）

```bash
# 允许 3001 端口（前端）
sudo ufw allow 3001/tcp

# 允许 8001 端口（后端 API）
sudo ufw allow 8001/tcp

# 重新加载防火墙
sudo ufw reload

# 查看防火墙状态
sudo ufw status
```

### 使用 iptables

```bash
# 允许 3001 和 8001 端口
sudo iptables -A INPUT -p tcp --dport 3001 -j ACCEPT
sudo iptables -A INPUT -p tcp --dport 8001 -j ACCEPT

# 保存规则
sudo netfilter-persistent save
```

## 步骤 7: 从其他电脑访问

在其他电脑的浏览器中访问：

```
http://ubuntu-server-ip:3001
```

将 `ubuntu-server-ip` 替换为你的 Ubuntu 服务器的实际 IP 地址。

### 查找服务器 IP 地址

```bash
# 在 Ubuntu 服务器上执行
ip addr show | grep inet
# 或
hostname -I
```

## 常用管理命令

```bash
# 查看容器状态
docker-compose ps

# 查看实时日志
docker-compose logs -f

# 停止服务
docker-compose down

# 重启服务
docker-compose restart

# 更新配置后重新启动
docker-compose down
docker-compose up -d

# 查看容器资源使用情况
docker stats deepwiki-deepwiki-1
```

## 故障排查

### 1. 容器无法启动

```bash
# 查看详细日志
docker-compose logs

# 检查端口是否被占用
sudo netstat -tulpn | grep -E '3001|8001'
```

### 2. 无法从外部访问

```bash
# 检查容器是否正在运行
docker ps

# 检查端口映射
docker port deepwiki-deepwiki-1

# 检查防火墙规则
sudo ufw status
sudo iptables -L -n

# 测试本地访问
curl http://localhost:3001
curl http://localhost:8001/health
```

### 3. 内存不足

如果服务器内存不足，可以调整 `docker-compose.yml` 中的内存限制：

```yaml
mem_limit: 4g        # 降低到 4GB
mem_reservation: 1g  # 降低到 1GB
```

## 性能优化建议

1. **使用反向代理（Nginx）**
   - 提供 HTTPS 支持
   - 负载均衡
   - 缓存静态资源

2. **配置域名**
   - 使用域名代替 IP 地址访问
   - 配置 DNS 解析

3. **启用日志轮转**
   ```bash
   # 限制 Docker 日志大小
   # 在 docker-compose.yml 中添加：
   logging:
     driver: "json-file"
     options:
       max-size: "10m"
       max-file: "3"
   ```

## 数据备份

重要数据存储在 `~/.adalflow` 目录中，定期备份：

```bash
# 备份数据
tar -czf deepwiki-backup-$(date +%Y%m%d).tar.gz ~/.adalflow

# 恢复数据
tar -xzf deepwiki-backup-YYYYMMDD.tar.gz -C ~/
```

## 安全建议

1. 修改默认端口（如果需要）
2. 配置 HTTPS（使用 Nginx + Let's Encrypt）
3. 限制访问 IP（使用防火墙规则）
4. 定期更新镜像和系统
5. 使用强密码保护 API 密钥

## 技术支持

- 镜像大小: ~976 MB
- 前端端口: 3001
- 后端 API 端口: 8001
- 数据持久化: ~/.adalflow

如有问题，请检查日志文件或联系技术支持。
