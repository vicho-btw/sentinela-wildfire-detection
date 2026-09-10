from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys
import zipfile


CONFIG_PATH = Path(__file__).resolve().parent / "run_config.json"
CONFIG: dict[str, object] = {}
if CONFIG_PATH.exists():
    with CONFIG_PATH.open(encoding="utf-8") as handle:
        CONFIG = json.load(handle)


def config_value(name: str, default: object) -> object:
    return os.getenv(name, CONFIG.get(name, default))


def env_bool(name: str, default: bool) -> bool:
    value = config_value(name, default)
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "y", "on"}


def env_int(name: str, default: int) -> int:
    value = config_value(name, default)
    return default if value is None or str(value).strip() == "" else int(value)


def run(cmd: list[str], cwd: Path) -> None:
    print("\n$ " + " ".join(cmd), flush=True)
    subprocess.run(cmd, cwd=cwd, check=True)


def find_project_root() -> Path:
    zip_candidates = [
        Path("/kaggle/input/sentinela-models/Sentinela-ModelS.zip"),
        Path("/kaggle/input/sentinela-models/sentinela-models.zip"),
    ]
    for archive in zip_candidates:
        if archive.exists():
            extract_root = Path("/kaggle/working/sentinela_repo")
            extract_root.mkdir(parents=True, exist_ok=True)
            marker = extract_root / ".extracted"
            if not marker.exists():
                print(f"Extracting {archive} to {extract_root}", flush=True)
                with zipfile.ZipFile(archive) as zf:
                    zf.extractall(extract_root)
                marker.write_text("ok\n", encoding="utf-8")
            for candidate in (extract_root / "Sentinela-ModelS", extract_root):
                if all((candidate / name).exists() for name in ("scripts", "src", "configs", "requirements.txt")):
                    return candidate

    candidates = [
        Path(str(config_value("SENTINELA_PROJECT_ROOT", ""))),
        Path("/kaggle/input/sentinela-models/Sentinela-ModelS"),
        Path("/kaggle/input/sentinela-models"),
        Path("/kaggle/working/Sentinela-ModelS"),
        Path.cwd(),
    ]
    for candidate in candidates:
        if not str(candidate):
            continue
        if all((candidate / name).exists() for name in ("scripts", "src", "configs", "requirements.txt")):
            return candidate
    raise FileNotFoundError(
        "Could not find Sentinela-ModelS. Upload the repo as a Kaggle Dataset "
        "or set SENTINELA_PROJECT_ROOT."
    )


def configure_firms_key() -> None:
    if os.getenv("FIRMS_MAP_KEY"):
        return
    try:
        from kaggle_secrets import UserSecretsClient  # type: ignore

        secret = UserSecretsClient().get_secret("FIRMS_MAP_KEY")
        if secret:
            os.environ["FIRMS_MAP_KEY"] = secret
    except Exception as exc:  # noqa: BLE001
        print(f"FIRMS_MAP_KEY secret lookup skipped: {exc}", flush=True)
    if not os.getenv("FIRMS_MAP_KEY"):
        raise RuntimeError("Set Kaggle Secret FIRMS_MAP_KEY before running label download.")


def main() -> None:
    project_root = find_project_root()
    working_root = Path("/kaggle/working")
    data_root = Path(str(config_value("SENTINELA_DATA_ROOT", str(working_root / "goes_fire"))))
    region = str(config_value("REGION", "south_america"))
    temporal_offsets = str(config_value("TEMPORAL_OFFSETS", "-30,-20,-10,0"))
    start_date = str(config_value("START_DATE", "2024-01-01"))
    end_date = str(config_value("END_DATE", "2026-05-11"))
    min_confidence = str(config_value("MIN_CONFIDENCE", "nominal"))
    max_firms_rows = env_int("MAX_FIRMS_ROWS", 2000)
    samples_per_class = env_int("SAMPLES_PER_CLASS", 1000)
    max_goes_files = env_int("MAX_GOES_FILES", 8)
    max_targets_per_file = env_int("MAX_TARGETS_PER_FILE", 0)
    hard_negative_ratio = str(config_value("HARD_NEGATIVE_RATIO", "0.5"))
    variant = str(config_value("VARIANT", "n"))
    epochs = env_int("EPOCHS", 1)
    batch_size = env_int("BATCH_SIZE", 16)
    num_workers = env_int("NUM_WORKERS", 2)
    drop_uncertain = env_bool("DROP_UNCERTAIN", False)

    run_label_download = env_bool("RUN_LABEL_DOWNLOAD", True)
    run_sample_build = env_bool("RUN_SAMPLE_BUILD", True)
    run_train = env_bool("RUN_TRAIN", True)
    run_evaluate = env_bool("RUN_EVALUATE", True)
    install_requirements = env_bool("INSTALL_REQUIREMENTS", True)

    data_root.mkdir(parents=True, exist_ok=True)
    out_dir = working_root / "models" / "regional" / region
    eval_json = data_root / region / "evaluation" / "best_test.json"

    print(
        json.dumps(
            {
                "project_root": str(project_root),
                "data_root": str(data_root),
                "region": region,
                "temporal_offsets": temporal_offsets,
                "samples_per_class": samples_per_class,
                "max_goes_files": max_goes_files,
                "variant": variant,
                "epochs": epochs,
            },
            indent=2,
        ),
        flush=True,
    )

    if install_requirements:
        run([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"], cwd=project_root)

    if run_label_download:
        configure_firms_key()
        cmd = [
            sys.executable,
            "scripts/download_firms_bootstrap_labels.py",
            "--region",
            region,
            "--data-root",
            str(data_root),
            "--min-confidence",
            min_confidence,
            "--max-rows",
            str(max_firms_rows),
            "--start-date",
            start_date,
            "--end-date",
            end_date,
        ]
        run(cmd, cwd=project_root)

        labels_path = data_root / region / "raw" / "labels" / "firms_bootstrap_events.csv"
        run(
            [
                sys.executable,
                "scripts/ingest_regional_labels.py",
                "--region",
                region,
                "--data-root",
                str(data_root),
                "--labels",
                str(labels_path),
            ],
            cwd=project_root,
        )

    if run_sample_build:
        cmd = [
            sys.executable,
            "scripts/build_regional_goes_samples_streaming.py",
            "--region",
            region,
            "--data-root",
            str(data_root),
            "--samples-per-class",
            str(samples_per_class),
            "--hard-negative-ratio",
            str(hard_negative_ratio),
            f"--temporal-offsets={temporal_offsets}",
            "--start-date",
            start_date,
            "--end-date",
            end_date,
        ]
        if max_goes_files > 0:
            cmd += ["--max-goes-files", str(max_goes_files)]
        if max_targets_per_file > 0:
            cmd += ["--max-targets-per-file", str(max_targets_per_file)]
        run(cmd, cwd=project_root)

    if run_train:
        cmd = [
            sys.executable,
            "scripts/train_regional_goes.py",
            "--region",
            region,
            "--data-root",
            str(data_root),
            "--out-dir",
            str(out_dir),
            "--epochs",
            str(epochs),
            "--batch-size",
            str(batch_size),
            "--num-workers",
            str(num_workers),
            "--variant",
            variant,
            f"--temporal-offsets={temporal_offsets}",
        ]
        if drop_uncertain:
            cmd.append("--drop-uncertain")
        run(cmd, cwd=project_root)

    if run_evaluate:
        checkpoint = out_dir / "best.pt"
        run(
            [
                sys.executable,
                "scripts/evaluate_regional_goes.py",
                "--region",
                region,
                "--data-root",
                str(data_root),
                "--checkpoint",
                str(checkpoint),
                "--split",
                "test",
                "--batch-size",
                str(batch_size),
                "--num-workers",
                str(num_workers),
                "--out-json",
                str(eval_json),
            ],
            cwd=project_root,
        )

        run(
            [
                sys.executable,
                "scripts/write_regional_benchmark_report.py",
                "--region",
                region,
                "--data-root",
                str(data_root),
                "--eval-json",
                str(eval_json),
                "--history-json",
                str(out_dir / "history.json"),
            ],
            cwd=project_root,
        )

    artifacts = {
        "best_checkpoint": str(out_dir / "best.pt"),
        "latest_checkpoint": str(out_dir / "latest.pt"),
        "history": str(out_dir / "history.json"),
        "evaluation": str(eval_json),
        "report": str(data_root / region / "reports" / "regional_benchmark.md"),
        "manifest": str(data_root / region / "manifest.csv"),
    }
    print(json.dumps({"artifacts": artifacts}, indent=2), flush=True)


if __name__ == "__main__":
    main()
