# Vercel Serverless 入口
# Vercel 只在 api/、app/、src/ 等固定位置探测入口文件，
# 本项目是 src 布局（src/sichu/app/main.py），因此在这里转发复用现有应用。
import sys
from pathlib import Path

# 把 src 目录加入模块搜索路径
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from sichu.app.main import app  # noqa: E402,F401
