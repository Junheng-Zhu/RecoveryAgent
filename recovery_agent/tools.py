import ast
import operator
import math
from pathlib import PurePosixPath
from .models import ToolError

SCHEMAS = {
    "list_files": (set(), {"prefix"}),
    "read_file": ({"path"}, set()),
    "write_file": ({"path", "content"}, set()),
    "query_table": ({"table"}, {"record_id"}),
    "update_record": ({"table", "record_id", "values"}, {"create"}),
    "calculate": ({"expression"}, set()),
}

# 校验工具名、必填/可选字段及参数类型；数据库更新值必须能表示为有限 JSON。
# 这里只做结构校验，资源是否存在以及能否修改由 Environment 在执行时检查。
def validate(call):
    if call.name not in SCHEMAS:
        raise ToolError("UNKNOWN_TOOL", "Unknown tool")
    required, optional = SCHEMAS[call.name]
    if not isinstance(call.arguments, dict):
        raise ToolError("INVALID_ARGUMENT", "Arguments must be an object")
    missing = required - call.arguments.keys()
    extra = call.arguments.keys() - required - optional
    if missing or extra:
        raise ToolError("INVALID_ARGUMENT", "Invalid fields",
                        missing=sorted(missing), extra=sorted(extra))
    for key, value in call.arguments.items():
        kind = {"content": str, "values": dict, "create": bool}.get(key, str)
        if not isinstance(value, kind):
            raise ToolError("INVALID_ARGUMENT", "Wrong argument type", field=key)
    if "values" in call.arguments:
        import json
        try:
            json.dumps(call.arguments["values"], allow_nan=False)
        except (TypeError, ValueError):
            raise ToolError("INVALID_ARGUMENT", "Values must be finite JSON")

# 限定相对 POSIX 路径，拒绝盘符、反斜杠、绝对路径和 .. 路径穿越。
def safe_path(path):
    p = PurePosixPath(path)
    if not path or p.is_absolute() or ".." in p.parts or "\\" in path or ":" in path:
        raise ToolError("INVALID_ARGUMENT", "Unsafe relative path", field="path")
    return p.as_posix()

# 通过 AST 解释数值表达式，仅支持数值常量及 +、-、*、/，不执行 Python eval。
# 表达式过长、非法语法、除零或非有限结果统一转为参数错误。
def calculate(expression):
    if len(expression) > 200:
        raise ToolError("INVALID_ARGUMENT", "Expression too long")
    ops = {ast.Add: operator.add, ast.Sub: operator.sub,
           ast.Mult: operator.mul, ast.Div: operator.truediv}
    # 递归解释允许的 AST 节点；bool、函数调用、属性访问和未授权运算符都会被拒绝。
    def visit(node):
        if isinstance(node, ast.Constant) and type(node.value) in (int, float):
            return node.value
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub)):
            return visit(node.operand) * (-1 if isinstance(node.op, ast.USub) else 1)
        if isinstance(node, ast.BinOp) and type(node.op) in ops:
            return ops[type(node.op)](visit(node.left), visit(node.right))
        raise ValueError("Only arithmetic literals and + - * / are supported")
    try:
        value = visit(ast.parse(expression, mode="eval").body)
        if not math.isfinite(value):
            raise ValueError("Non-finite result")
        return value
    except (ValueError, SyntaxError, ZeroDivisionError, OverflowError, RecursionError) as e:
        raise ToolError("INVALID_ARGUMENT", str(e), field="expression")
