"""
KHOA NetCDF 파일 분석 스크립트
"""
from netCDF4 import Dataset
import numpy as np
from pathlib import Path

# 파일 경로
file_path = "dataset/KHOA_SCU_L4_Z004_D01_U20251118_EastSea.nc"

print("=" * 80)
print("KHOA NetCDF 파일 분석")
print("=" * 80)
print(f"\n파일: {file_path}\n")

try:
    # NetCDF 파일 열기
    nc = Dataset(file_path, 'r')
    
    print("1. 전역 속성(Global Attributes)")
    print("-" * 80)
    for attr in nc.ncattrs():
        print(f"{attr}: {nc.getncattr(attr)}")
    
    print("\n\n2. 차원(Dimensions) 정보")
    print("-" * 80)
    for dim_name, dim in nc.dimensions.items():
        print(f"{dim_name}: {dim.size} (무제한: {dim.isunlimited()})")
    
    print("\n\n3. 변수(Variables) 정보")
    print("-" * 80)
    for var_name, var in nc.variables.items():
        print(f"\n{var_name}:")
        print(f"  차원: {var.dimensions}")
        print(f"  형태: {var.shape}")
        print(f"  데이터 타입: {var.dtype}")
        print(f"  속성:")
        for attr in var.ncattrs():
            attr_value = var.getncattr(attr)
            if isinstance(attr_value, (str, int, float)):
                print(f"    {attr}: {attr_value}")
            elif isinstance(attr_value, np.ndarray) and attr_value.size <= 10:
                print(f"    {attr}: {attr_value}")
            else:
                print(f"    {attr}: {type(attr_value).__name__} (크기: {len(attr_value) if hasattr(attr_value, '__len__') else 'N/A'})")
        
        # 데이터 통계 (작은 배열인 경우에만)
        if var.size > 0 and var.size < 1000000:  # 100만개 이하만
            try:
                data = var[:]
                if isinstance(data, np.ndarray):
                    valid_data = data[~np.isnan(data)] if data.dtype.kind == 'f' else data
                    if len(valid_data) > 0:
                        print(f"  데이터 통계:")
                        print(f"    유효 데이터 포인트: {len(valid_data)}/{data.size}")
                        if data.dtype.kind == 'f':  # 부동소수점
                            print(f"    최소값: {np.nanmin(data):.4f}")
                            print(f"    최대값: {np.nanmax(data):.4f}")
                            print(f"    평균값: {np.nanmean(data):.4f}")
                            if len(valid_data) > 1:
                                print(f"    표준편차: {np.nanstd(data):.4f}")
                        else:  # 정수형
                            print(f"    최소값: {np.min(data)}")
                            print(f"    최대값: {np.max(data)}")
                            print(f"    평균값: {np.mean(data):.4f}")
                        
                        # 작은 배열의 경우 샘플 데이터 표시
                        if data.size <= 20:
                            print(f"    데이터 값: {data}")
                        elif len(data.shape) == 1 and len(data) <= 50:
                            print(f"    첫 10개 값: {data[:10]}")
            except Exception as e:
                print(f"  데이터 읽기 오류: {e}")
    
    print("\n\n4. 파일 구조 요약")
    print("-" * 80)
    print(f"차원 개수: {len(nc.dimensions)}")
    print(f"변수 개수: {len(nc.variables)}")
    print(f"전역 속성 개수: {len(nc.ncattrs())}")
    
    # 좌표 변수 식별
    coord_vars = []
    data_vars = []
    for var_name, var in nc.variables.items():
        if var_name in nc.dimensions or len(var.dimensions) == 1:
            coord_vars.append(var_name)
        else:
            data_vars.append(var_name)
    
    print(f"\n좌표 변수: {coord_vars}")
    print(f"데이터 변수: {data_vars}")
    
    nc.close()
    
    print("\n" + "=" * 80)
    print("분석 완료!")
    print("=" * 80)
    
except Exception as e:
    print(f"\n오류 발생: {e}")
    import traceback
    traceback.print_exc()
