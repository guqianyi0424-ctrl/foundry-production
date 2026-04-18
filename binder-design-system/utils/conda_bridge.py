"""
跨 conda 环境调用工具
binder(Python3.10) -> foundry(Python3.12)

调用优先级:
1. conda run -n foundry (标准方式)
2. 直接使用 foundry 环境的 bin/ 目录下的 console script
3. 降级为 mock 模式
"""
import os
import subprocess
import shutil
from pathlib import Path
from typing import List, Optional, Tuple


FOUNDRY_ENV = "foundry"

_foundry_prefix: Optional[str] = None
_foundry_checked: bool = False
_foundry_available: bool = False
_use_conda_run: bool = False


def _find_conda_env_prefix(env_name: str) -> Optional[str]:
    try:
        result = subprocess.run(
            ["conda", "env", "list", "--json"],
            capture_output=True, text=True, timeout=15
        )
        if result.returncode != 0:
            return None
        import json
        data = json.loads(result.stdout)
        for env_path in data.get("envs", []):
            if env_path.endswith(env_name) or Path(env_path).name == env_name:
                return env_path
    except Exception:
        pass

    common_prefixes = [
        os.path.expanduser(f"~/miniconda3/envs/{env_name}"),
        os.path.expanduser(f"~/anaconda3/envs/{env_name}"),
        f"/opt/conda/envs/{env_name}",
        f"/root/miniconda3/envs/{env_name}",
        f"/root/anaconda3/envs/{env_name}",
    ]
    for p in common_prefixes:
        if os.path.isdir(p):
            return p

    return None


def _find_python_in_prefix(prefix: str) -> Optional[str]:
    python_path = os.path.join(prefix, "bin", "python")
    if os.path.isfile(python_path):
        return python_path
    python_path = os.path.join(prefix, "python.exe")
    if os.path.isfile(python_path):
        return python_path
    return None


def _find_cli_script(prefix: str, cli_name: str) -> Optional[str]:
    script_path = os.path.join(prefix, "bin", cli_name)
    if os.path.isfile(script_path):
        return script_path
    script_path = os.path.join(prefix, "Scripts", cli_name + ".exe")
    if os.path.isfile(script_path):
        return script_path
    return None


def check_foundry() -> Tuple[bool, Optional[str]]:
    global _foundry_prefix, _foundry_checked, _foundry_available, _use_conda_run

    if _foundry_checked:
        return _foundry_available, _foundry_prefix

    _foundry_checked = True

    try:
        result = subprocess.run(
            ["conda", "run", "-n", FOUNDRY_ENV, "--no-banner", "python", "-c",
             "import rfd3; import mpnn; import rf3; print('OK')"],
            capture_output=True, text=True, timeout=30
        )
        if result.returncode == 0 and "OK" in result.stdout:
            _foundry_available = True
            _use_conda_run = True
            _foundry_prefix = None
            return True, None
    except Exception:
        pass

    env_prefix = _find_conda_env_prefix(FOUNDRY_ENV)
    if env_prefix:
        python_path = _find_python_in_prefix(env_prefix)
        if python_path:
            try:
                result = subprocess.run(
                    [python_path, "-c", "import rfd3; import mpnn; import rf3; print('OK')"],
                    capture_output=True, text=True, timeout=30
                )
                if result.returncode == 0 and "OK" in result.stdout:
                    _foundry_available = True
                    _use_conda_run = False
                    _foundry_prefix = env_prefix
                    return True, env_prefix
            except Exception:
                pass

    _foundry_available = False
    return False, None


def run_foundry_cli(cli_args: List[str], timeout: int = 1800) -> subprocess.CompletedProcess:
    available, env_prefix = check_foundry()

    if not available:
        raise RuntimeError("foundry 环境不可用")

    if _use_conda_run:
        cmd = ["conda", "run", "-n", FOUNDRY_ENV, "--no-banner"] + cli_args
    elif env_prefix:
        cli_name = cli_args[0]
        remaining = cli_args[1:]
        cli_script = _find_cli_script(env_prefix, cli_name)
        if cli_script:
            cmd = [cli_script] + remaining
        else:
            python_path = _find_python_in_prefix(env_prefix)
            if not python_path:
                raise RuntimeError(f"foundry 环境 Python 未找到: {env_prefix}")
            cmd = [python_path] + cli_args
    else:
        raise RuntimeError("foundry 环境不可用")

    return subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        timeout=timeout,
    )


def is_foundry_available() -> bool:
    available, _ = check_foundry()
    return available
