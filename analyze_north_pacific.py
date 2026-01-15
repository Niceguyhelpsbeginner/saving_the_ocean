"""
North Pacific NetCDF 파일 구조 분석
"""
from netCDF4 import Dataset
import numpy as np

file_path = "dataset/north_pacific.nc"

print("=" * 80)
print("North Pacific NetCDF 파일 구조 분석")
print("=" * 80)

try:
    nc = Dataset(file_path, 'r')
    
    print("\n1. 전역 속성(Global Attributes)")
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
    
    nc.close()
    
except Exception as e:
    print(f"\n오류 발생: {e}")
    import traceback
    traceback.print_exc()


