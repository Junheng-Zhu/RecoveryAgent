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

def safe_path(path):
    p = PurePosixPath(path)
    if not path or p.is_absolute() or ".." in p.parts or "\\" in path or ":" in path:
        raise ToolError("INVALID_ARGUMENT", "Unsafe relative path", field="path")
    return p.as_posix()

def calculate(expression):
    if len(expression) > 200:
        raise ToolError("INVALID_ARGUMENT", "Expression too long")
    ops = {ast.Add: operator.add, ast.Sub: operator.sub,
           ast.Mult: operator.mul, ast.Div: operator.truediv}
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
