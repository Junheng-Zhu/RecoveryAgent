import hashlib
import json
from pathlib import Path

VERSION = "0.2.0"

# 用稳定键顺序和紧凑格式序列化 JSON，拒绝 NaN/Infinity，作为指纹与落盘统一表示。
def canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)

# 对规范化 JSON 计算 SHA-256；比较的是内容，不依赖对象地址或输出目录。
def fingerprint(value):
    return hashlib.sha256(canonical(value).encode("utf-8")).hexdigest()

# 写入带缩进的 UTF-8 JSON 报告或配置，先确保父目录存在。
def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2,
                               allow_nan=False) + "\n", encoding="utf-8")

# 逐条写入规范化 JSON，固定换行符以支持两次生成逐字节比较。
def write_jsonl(path, values):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as stream:
        for value in values:
            stream.write(canonical(value) + "\n")

# 严格逐行读取：拒绝空行、非法 JSON、非对象行和非有限数值，并标注出错行号。
def read_jsonl(path):
    values = []
    with Path(path).open(encoding="utf-8") as stream:
        for number, line in enumerate(stream, 1):
            if not line.strip():
                raise ValueError(f"{path}:{number}: blank JSONL line")
            try:
                value = json.loads(line, parse_constant=lambda x: (_ for _ in ()).throw(ValueError(x)))
            except (ValueError, TypeError) as error:
                raise ValueError(f"{path}:{number}: invalid JSON") from error
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{number}: expected object")
            values.append(value)
    return values
