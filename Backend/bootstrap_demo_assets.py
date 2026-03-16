import os
import shutil
from pathlib import Path


BACKEND_ROOT = Path(__file__).resolve().parent
DEFAULT_MODEL_NAME = "llama-3-sqlcoder-8b-Q4_K_M.gguf"
DEFAULT_DB_NAME = "dhis2.sqlite"


def env_path(name: str, default: str = "") -> Path:
    raw = os.environ.get(name, default).strip()
    if not raw:
        return Path()
    path = Path(raw)
    if path.is_absolute():
        return path
    return (BACKEND_ROOT / path).resolve()


def copy_if_missing(src: Path, dst: Path) -> None:
    if not src.exists():
        raise FileNotFoundError(f"Missing source asset: {src}")
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        print(f"Exists: {dst}")
        return
    print(f"Copying {src.name} -> {dst}")
    shutil.copy2(src, dst)


def main() -> None:
    bundled_model = env_path("BUNDLED_MODEL_PATH", f"models/{DEFAULT_MODEL_NAME}")
    bundled_db = env_path("BUNDLED_DB_PATH", f"Database/{DEFAULT_DB_NAME}")

    runtime_root = env_path("ANDROID_RUNTIME_ROOT", "android_runtime")
    runtime_model = env_path("ANDROID_MODEL_PATH", str(runtime_root / "models" / DEFAULT_MODEL_NAME))
    runtime_db = env_path("ANDROID_SQLITE_DB_PATH", str(runtime_root / "Database" / DEFAULT_DB_NAME))

    copy_if_missing(bundled_model, runtime_model)
    copy_if_missing(bundled_db, runtime_db)

    print("Set these runtime variables for the app:")
    print(f"ANDROID_MODEL_PATH={runtime_model}")
    print(f"ANDROID_SQLITE_DB_PATH={runtime_db}")


if __name__ == "__main__":
    main()
