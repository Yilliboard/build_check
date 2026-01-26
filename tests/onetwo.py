# ──────────────────────────────────────────────────────────────
# Most common compromise — good enough for most teams
# ──────────────────────────────────────────────────────────────

from pathlib import Path
import yaml
import pprint

def yaml_to_python_literal(
    yaml_file: str | Path,
    py_file: str | Path = None,
    var_name: str = "CONFIG",
    indent: int = 4,           # ← 2 or 4 are the most popular choices
    width: int = 88,           # try 80–100
    compact: bool = False
) -> None:
    yaml_path = Path(yaml_file)
    data = yaml.safe_load(yaml_path.read_text(encoding="utf-8"))

    if py_file is None:
        py_file = yaml_path.with_suffix(".py").with_stem(yaml_path.stem + "_config")

    py_path = Path(py_file)

    pretty = pprint.pformat(
        data,
        indent=indent,
        width=width,
        compact=compact
    )

    header = f"""# Generated from {yaml_path.name}
# Conversion date: auto-generated
# Original YAML → Python dict literal
# (indentation & wrapping will not be 100% identical)

"""

    py_path.write_text(header + f"{var_name} = {pretty}\n", encoding="utf-8")
    print(f"Wrote → {py_path}")


# Usage
yaml_to_python_literal("your-config.yaml", indent=4, width=100)
