"""
Grinsztajn 벤치마크 데이터셋 다운로드 스크립트.
OpenML API를 통해 데이터를 받아 로컬 캐시에 저장한다.
"""

import os
import time
import openml
import pandas as pd
from pathlib import Path
from dataset_registry import CLASSIFICATION_DATASETS, REGRESSION_DATASETS

CACHE_DIR = Path.home() / ".cache" / "ml_agent" / "datasets"
CACHE_DIR.mkdir(parents=True, exist_ok=True)

# OpenML 캐시 경로 통일
openml.config.set_root_cache_directory(str(CACHE_DIR / "_openml_cache"))


def download_dataset(name: str, openml_id: int, task: str) -> bool:
    save_path = CACHE_DIR / task / f"{name}.parquet"
    save_path.parent.mkdir(parents=True, exist_ok=True)

    if save_path.exists():
        print(f"  [SKIP] {name} (already cached)")
        return True

    try:
        dataset = openml.datasets.get_dataset(
            openml_id,
            download_data=True,
            download_qualities=False,
            download_features_meta_data=False,
        )
        target = dataset.default_target_attribute
        if not target:
            print(f"  [WARN] {name}: default_target_attribute not set, skipping")
            return False

        X, y, _, _ = dataset.get_data(
            target=target,
            dataset_format="dataframe",
        )

        if y is None:
            print(f"  [WARN] {name}: target column not found, skipping")
            return False

        df = X.copy()
        df["__target__"] = y
        df.to_parquet(save_path, index=False)
        print(f"  [OK]   {name} — {df.shape[0]:,} rows x {df.shape[1]-1} features → {save_path}")
        return True

    except Exception as e:
        print(f"  [FAIL] {name} (id={openml_id}): {e}")
        return False


def main():
    results = {"ok": [], "skip": [], "fail": []}

    print(f"\n{'='*50}")
    print(f"Classification datasets ({len(CLASSIFICATION_DATASETS)})")
    print(f"{'='*50}")
    for name, oid in CLASSIFICATION_DATASETS.items():
        ok = download_dataset(name, oid, "classification")
        key = "ok" if ok else "fail"
        results[key].append(name)
        time.sleep(0.5)  # OpenML API rate limit 방지

    print(f"\n{'='*50}")
    print(f"Regression datasets ({len(REGRESSION_DATASETS)})")
    print(f"{'='*50}")
    for name, oid in REGRESSION_DATASETS.items():
        ok = download_dataset(name, oid, "regression")
        key = "ok" if ok else "fail"
        results[key].append(name)
        time.sleep(0.5)

    print(f"\n{'='*50}")
    print(f"완료: {len(results['ok'])}개 성공, {len(results['fail'])}개 실패")
    if results["fail"]:
        print(f"실패 목록: {results['fail']}")
    print(f"저장 위치: {CACHE_DIR}")


if __name__ == "__main__":
    main()
