"""
모든 시각화 파일을 135도 시계방향 회전 + 좌우반전
"""
import os
from PIL import Image
import numpy as np
from pathlib import Path

print("=" * 80)
print("Transforming All Visualizations")
print("135° clockwise rotation + horizontal flip")
print("=" * 80)

# analysis_output 폴더에서 모든 PNG 파일 찾기
output_dir = "analysis_output"
png_files = []

for root, dirs, files in os.walk(output_dir):
    for file in files:
        if file.endswith('.png'):
            png_files.append(os.path.join(root, file))

print(f"\nFound {len(png_files)} PNG files")
print("-" * 80)

# 각 파일 처리
for idx, file_path in enumerate(png_files, 1):
    try:
        print(f"\n[{idx}/{len(png_files)}] Processing: {file_path}")
        
        # 이미지 로드
        img = Image.open(file_path)
        img_array = np.array(img)
        
        # 135도 시계방향 회전
        # PIL의 rotate는 반시계방향이므로 -135도 사용
        img_rotated = img.rotate(-135, expand=True, fillcolor='white')
        
        # 좌우반전
        img_flipped = img_rotated.transpose(Image.FLIP_LEFT_RIGHT)
        
        # 원본 파일명에 _transformed 추가
        path_obj = Path(file_path)
        new_filename = path_obj.stem + '_transformed' + path_obj.suffix
        new_path = path_obj.parent / new_filename
        
        # 저장
        img_flipped.save(new_path, 'PNG', dpi=(300, 300))
        print(f"  ✓ Saved: {new_path}")
        
    except Exception as e:
        print(f"  ✗ Error processing {file_path}: {e}")

print("\n" + "=" * 80)
print("Transformation completed!")
print(f"Transformed files saved with '_transformed' suffix.")
print("=" * 80)


