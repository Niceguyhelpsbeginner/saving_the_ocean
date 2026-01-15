"""
변환된 이미지 파일 제거 및 원본 파일 재생성
"""
import os
from pathlib import Path

print("=" * 80)
print("Removing transformed files and regenerating originals")
print("=" * 80)

# 변환된 파일 찾기 및 삭제
output_dir = "analysis_output"
transformed_files = []

for root, dirs, files in os.walk(output_dir):
    for file in files:
        if '_transformed.png' in file:
            transformed_files.append(os.path.join(root, file))

print(f"\n1. Removing {len(transformed_files)} transformed files...")
print("-" * 80)

for file_path in transformed_files:
    try:
        os.remove(file_path)
        print(f"  ✓ Removed: {file_path}")
    except Exception as e:
        print(f"  ✗ Error removing {file_path}: {e}")

print(f"\n✓ Removed {len(transformed_files)} transformed files")

# 재생성할 스크립트 목록
scripts_to_run = [
    "visualize_east_sea_current.py",
    "detailed_analysis.py",
    "visualize_eddies.py",
    "visualize_eddies_north_pacific.py",
    "analyze_current_trash_correlation.py",
    "optimize_trash_collection_route.py"
]

print(f"\n2. Regenerating original visualizations...")
print("-" * 80)
print(f"Scripts to run: {len(scripts_to_run)}")

for idx, script in enumerate(scripts_to_run, 1):
    if os.path.exists(script):
        print(f"\n[{idx}/{len(scripts_to_run)}] Running {script}...")
        print("-" * 80)
        os.system(f"python {script}")
    else:
        print(f"[{idx}/{len(scripts_to_run)}] ✗ Script not found: {script}")

print("\n" + "=" * 80)
print("Regeneration completed!")
print("=" * 80)


