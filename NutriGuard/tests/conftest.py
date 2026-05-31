import os
import sys
import pytest

# 确保能导入项目根目录的模块
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 国内 HuggingFace 镜像，加速模型下载
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

# 避免 PyTorch + sentence_transformers 多线程在 Windows 上 segfault
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

# 注册自定义标记
pytest_plugins = []


def pytest_configure(config):
    config.addinivalue_line("markers", "integration: 需要真实 LLM API 调用的集成测试（默认跳过）")


@pytest.fixture(scope="session")
def corpus_path():
    return os.path.join(os.path.dirname(__file__), "..", "data", "mock_corpus.md")


@pytest.fixture(scope="session")
def model_dir():
    return os.path.join(os.path.dirname(__file__), "..", "models")


@pytest.fixture(scope="session")
def bge_model_path(model_dir):
    p = os.path.join(model_dir, "bge-large-zh-v1.5")
    if os.path.isdir(p):
        return p
    return "BAAI/bge-large-zh-v1.5"


@pytest.fixture(scope="session")
def reranker_model_path(model_dir):
    p = os.path.join(model_dir, "bge-reranker-v2-m3")
    if os.path.isdir(p):
        return p
    return "BAAI/bge-reranker-v2-m3"


@pytest.fixture(scope="session")
def eval_dataset():
    """手写 Ground Truth 测试集，用于 RAG 离线评测"""
    return [
        ("糙米饭的升糖指数是多少？", "糙米饭"),
        ("孕妇查出糖尿病，该怎么控制？", "妊娠期糖尿病"),
        ("得了痛风，饮食上要注意啥？", "痛风"),
        ("我今天吃了燕麦片，纤维素高吗？", "燕麦片"),
        ("糖尿病初期的保守治疗手段是什么", "生活方式干预"),
    ]
