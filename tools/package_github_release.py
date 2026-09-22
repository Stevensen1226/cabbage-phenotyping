#!/usr/bin/env python3
"""Build GitHub-release archives for the cabbage paper project."""
from __future__ import annotations
import hashlib, json
from datetime import date
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile

ROOT = Path(r"E:\Cabbage")
STAMP = date.today().strftime("%Y%m%d")
DIST = ROOT / "dist" / f"cabbage_github_release_{STAMP}"
DIST.mkdir(parents=True, exist_ok=True)
SOURCE_DIRS = ["cabbage_pheno", "configs", "docs", "tools", "PointNet2_Ours", "RandLANet_Ours", "PointGroup_Ours", "SoftGroup-main", "SoftGroup_Ours"]
SOURCE_FILES = [".editorconfig", ".gitignore", "README.md", "requirements.txt", "setup.py", "train.py", "main.py", "evaluate.py", "commands.txt"]
MODEL_CODE_EXT = {".py", ".yaml", ".yml", ".json", ".md", ".txt", ".sh", ".bat", ".cpp", ".cu", ".c", ".h", ".hpp", ".toml", ".cfg", ".ini"}
EXCLUDE_PARTS = {"__pycache__", ".git", ".idea", ".vscode", "build", "dist", "exp", "checkpoints", "logs", "log", "output", "visualizations"}
EXCLUDE_EXT = {".pyc", ".pyo", ".so", ".pyd", ".dll", ".lib", ".exp", ".obj", ".o", ".pth", ".pt", ".ckpt", ".onnx", ".npy", ".npz", ".log", ".tfevents", ".png", ".jpg", ".jpeg", ".gif", ".pdf", ".docx", ".zip", ".7z", ".rar"}

def is_excluded(path: Path, model_tree: bool = False) -> bool:
    rel = path.relative_to(ROOT)
    if any(part in EXCLUDE_PARTS for part in rel.parts):
        return True
    if path.suffix.lower() in EXCLUDE_EXT or path.name.startswith(".tmp_"):
        return True
    if model_tree and path.suffix and path.suffix.lower() not in MODEL_CODE_EXT:
        return True
    return False

def iter_source_files():
    for name in SOURCE_FILES:
        p = ROOT / name
        if p.is_file():
            yield p, p.relative_to(ROOT).as_posix()
    models = {"PointNet2_Ours", "RandLANet_Ours", "PointGroup_Ours", "SoftGroup-main", "SoftGroup_Ours"}
    for dname in SOURCE_DIRS:
        base = ROOT / dname
        if not base.exists():
            continue
        for p in base.rglob("*"):
            if p.is_file() and not is_excluded(p, model_tree=dname in models):
                yield p, p.relative_to(ROOT).as_posix()

def iter_raw_dataset_files():
    base = ROOT / "evalaute_test"
    if base.exists():
        for p in sorted(base.iterdir()):
            if p.is_file() and ("_gt.txt" in p.name or (p.suffix.lower() == ".ply" and "step1_sor" not in p.name)):
                yield p, f"evalaute_test/{p.name}"
    base = ROOT / "e_data" / "train"
    if base.exists():
        for p in sorted(base.iterdir()):
            if p.is_file() and (p.suffix.lower() == ".ply" or p.name.endswith("_gt.txt")):
                yield p, f"e_data/train/{p.name}"
    for rel in ("data/raw", "data/unlabel"):
        base = ROOT / rel
        if base.exists():
            for p in sorted(base.rglob("*")):
                if p.is_file():
                    yield p, f"{rel}/{p.relative_to(base).as_posix()}"

def iter_prepared_dataset_files():
    for rel, pattern in [("e_data/pointgroup_format/train", "*.pth"), ("data/processed_blocks/train", "*.npy")]:
        base = ROOT / rel
        if base.exists():
            for p in sorted(base.glob(pattern)):
                if p.is_file():
                    yield p, f"{rel}/{p.name}"

def split_texts():
    split_path = ROOT / "evalaute_test" / "split.json"
    split = json.loads(split_path.read_text(encoding="utf-8")) if split_path.exists() else {}
    eval_dir = ROOT / "evalaute_test"
    eval_names = sorted(p.name for p in eval_dir.glob("cloudR*.ply") if "step1_sor" not in p.name) if eval_dir.exists() else []
    def lines(key):
        names = split.get(key, [])
        return "\n".join(names) + ("\n" if names else "")
    return {
        "splits/train.txt": lines("train"),
        "splits/val.txt": lines("val"),
        "splits/test.txt": lines("test"),
        "splits/eval.txt": "\n".join(eval_names) + ("\n" if eval_names else ""),
        "splits/protocol.json": json.dumps(split, indent=2),
    }

def write_zip(path: Path, entries, extra_texts=None):
    extra_texts = extra_texts or {}
    seen, count, total = set(), 0, 0
    with ZipFile(path, "w", compression=ZIP_DEFLATED, compresslevel=6, allowZip64=True) as zf:
        for src, arcname in entries:
            if arcname in seen:
                continue
            seen.add(arcname)
            zf.write(src, arcname)
            count += 1
            total += src.stat().st_size
        for arcname, text in extra_texts.items():
            if arcname not in seen:
                zf.writestr(arcname, text)
                seen.add(arcname)
    print(f"built {path.name}: {count} files, {total/(1024**3):.3f} GB input, {path.stat().st_size/(1024**3):.3f} GB zip")

def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


UPLOAD_GUIDE = """# GitHub 上传说明

原始点云和训练格式文件不应直接进入普通 Git 历史。推荐将下面三个 ZIP 上传到 GitHub Releases，或使用 Git LFS。

## 推荐流程
1. 先将源码目录提交到仓库。
2. 在 GitHub Releases 新建版本。
3. 上传 `cabbage_source_*.zip`、`cabbage_dataset_raw_*.zip`、`cabbage_dataset_prepared_*.zip`。
4. 同时上传 `SHA256SUMS.txt` 用于校验。
5. 发布前补充软件许可证，并确认点云数据允许公开分发。
"""

SOURCE_README = """# 源码包说明

包含训练、评估、点云预处理、实例聚类、后处理与论文绘图脚本。数据和模型检查点未包含在内。
"""

RAW_README = """# 原始数据集

- `evalaute_test/`：评估点云与逐点实例标签
- `e_data/train/`：训练点云与标签
- `data/raw/`：原始采集及标注数据
- `data/unlabel/`：未标注点云
- `splits/`：训练、评估及测试文件列表
"""

PREPARED_README = """# 预处理训练数据

- `e_data/pointgroup_format/train/*.pth`：PointGroup/SoftGroup 输入
- `data/processed_blocks/train/*.npy`：分块训练数据
"""

source_zip = DIST / f"cabbage_source_{STAMP}.zip"
raw_zip = DIST / f"cabbage_dataset_raw_{STAMP}.zip"
prepared_zip = DIST / f"cabbage_dataset_prepared_{STAMP}.zip"
write_zip(source_zip, iter_source_files(), {"GITHUB_UPLOAD_GUIDE.md": UPLOAD_GUIDE, "README_RELEASE.md": SOURCE_README})
write_zip(raw_zip, iter_raw_dataset_files(), {"README_DATASET.md": RAW_README, **split_texts()})
write_zip(prepared_zip, iter_prepared_dataset_files(), {"README_PREPARED_DATASET.md": PREPARED_README})

archives = [source_zip, raw_zip, prepared_zip]
lines, manifest = [], {"date": STAMP, "archives": []}
for p in archives:
    digest = sha256(p)
    lines.append(f"{digest}  {p.name}")
    manifest["archives"].append({"name": p.name, "size_bytes": p.stat().st_size, "sha256": digest})
(DIST / "SHA256SUMS.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
(DIST / "GITHUB_UPLOAD_GUIDE.md").write_text(UPLOAD_GUIDE, encoding="utf-8")
(DIST / "release_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
print(f"release directory: {DIST}")
print((DIST / "SHA256SUMS.txt").read_text(encoding="utf-8"))
