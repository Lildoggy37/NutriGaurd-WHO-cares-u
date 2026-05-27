NutriGuard-Copilot/
├── app/
│   ├── gateway/          # Redis 限流、语义缓存网关
│   ├── graph/            # LangGraph Supervisor 状态机
│   ├── mcp/              # MCP 统一工具服务器（Redis锁、外部接口）
│   └── rag/              # 语义切块与 Qdrant 检索核心
├── data/                 # 膳食指南、食物热量等原始语料
├── tests/                # Ragas 自动化评测试卷
└── README.md             # 顶级的开源说明文档（附带 Ragas 优化前后对比图）