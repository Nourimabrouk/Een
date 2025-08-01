import importlib
import argparse
import os
import sys
from types import ModuleType
from typing import Any


def dynamic_import(path: str) -> Any:
    """Dynamically import ``path`` which should be in dotted notation ``pkg.module:callable``.

    Returns the resolved attribute (callable/class/variable).
    """
    if ":" in path:
        module_path, attr_name = path.split(":", 1)
    else:
        module_path, attr_name = path, None

    try:
        module: ModuleType = importlib.import_module(module_path)
    except ImportError as exc:
        print(f"[agent_runner] Could not import module '{module_path}': {exc}", file=sys.stderr)
        sys.exit(1)

    if attr_name is None:
        return module

    try:
        return getattr(module, attr_name)
    except AttributeError:
        print(f"[agent_runner] Module '{module_path}' has no attribute '{attr_name}'.", file=sys.stderr)
        sys.exit(1)


def main() -> None:
    parser = argparse.ArgumentParser(description="Spawn any agent within the codebase via dotted path.")
    parser.add_argument("path", help="Dotted path to agent callable (e.g. src.agents.omega.orchestrator:OmegaOrchestrator)")
    parser.add_argument("--json-config", dest="config", help="Path to JSON config passed to agent __init__() or run()", default=None)
    parser.add_argument("--kwargs", nargs="*", help="Extra keyword args in key=value form to pass to the agent.")
    args = parser.parse_args()

    # Load kwargs
    kwargs = {}
    for kv in args.kwargs or []:
        if "=" not in kv:
            parser.error(f"Malformed --kwargs entry '{kv}'. Expected key=value.")
        k, v = kv.split("=", 1)
        kwargs[k] = v

    # If a JSON config is provided, load and merge into kwargs
    if args.config:
        import json
        with open(args.config, "r", encoding="utf-8") as f:
            cfg = json.load(f)
        kwargs.update(cfg)

    obj = dynamic_import(args.path)

    # Decide how to invoke the object
    if callable(obj):
        # Case 1: Object itself is callable (function or class)
        if isinstance(obj, type):
            # Class → instantiate then call .run() or __call__ if available
            instance = obj(**kwargs)
            if hasattr(instance, "run"):
                result = instance.run()
            elif callable(instance):
                result = instance()
            else:
                print("[agent_runner] Instance has no 'run' method and is not callable; nothing to execute.", file=sys.stderr)
                sys.exit(1)
        else:
            # Plain function
            result = obj(**kwargs)
    else:
        print("[agent_runner] Target object is not callable.", file=sys.stderr)
        sys.exit(1)

    if result is not None:
        # Print JSON serialisable result if possible
        try:
            import json
            print(json.dumps(result, indent=2, default=str))
        except (TypeError, ValueError):
            print(result)


if __name__ == "__main__":
    main()