"""
데이터 범위 확인 스크립트
"""
from netCDF4 import Dataset
import numpy as np

file_path = "dataset/north_pacific.nc"

print("=" * 80)
print("데이터 범위 상세 확인")
print("=" * 80)

try:
    nc = Dataset(file_path, 'r')
    
    lat = nc.variables['lat'][:]
    lon = nc.variables['lon'][:]
    
    print(f"\n위도 (lat):")
    print(f"  개수: {len(lat)}")
    print(f"  최소값: {lat.min():.4f}°N")
    print(f"  최대값: {lat.max():.4f}°N")
    print(f"  첫 5개: {lat[:5]}")
    print(f"  마지막 5개: {lat[-5:]}")
    
    print(f"\n경도 (lon):")
    print(f"  개수: {len(lon)}")
    print(f"  최소값: {lon.min():.4f}°E")
    print(f"  최대값: {lon.max():.4f}°E")
    print(f"  첫 5개: {lon[:5]}")
    print(f"  마지막 5개: {lon[-5:]}")
    
    print(f"\n전역 속성에서 확인된 범위:")
    print(f"  image_upperleft_latitude: {nc.getncattr('image_upperleft_latitude')}")
    print(f"  image_lowerright_latitude: {nc.getncattr('image_lowerright_latitude')}")
    print(f"  image_upperleft_longitude: {nc.getncattr('image_upperleft_longitude')}")
    print(f"  image_lowerright_longitude: {nc.getncattr('image_lowerright_longitude')}")
    
    print(f"\n격자 크기:")
    print(f"  Number_of_Pixel: {nc.getncattr('Number_of_Pixel')}")
    print(f"  실제 데이터 크기: {len(lat)} × {len(lon)}")
    
    # 데이터 커버리지 확인
    print(f"\n데이터 커버리지:")
    print(f"  위도 범위: {lat.max() - lat.min():.2f}도")
    print(f"  경도 범위: {lon.max() - lon.min():.2f}도")
    print(f"  총 면적: 약 {(lat.max() - lat.min()) * (lon.max() - lon.min()):.0f} 평방도")
    
    # 북태평양 전체 범위와 비교
    print(f"\n북태평양 일반적인 범위:")
    print(f"  위도: 약 0°N ~ 60°N")
    print(f"  경도: 약 100°E ~ 260°E (또는 100°E ~ 100°W)")
    print(f"  현재 데이터는 서태평양-중태평양 지역만 포함")
    
    nc.close()
    
except Exception as e:
    print(f"\n오류 발생: {e}")
    import traceback
    traceback.print_exc()


