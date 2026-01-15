"""
지도 시각화 테스트 - 실제 육지가 보이는지 확인
"""
from netCDF4 import Dataset
import numpy as np
import matplotlib.pyplot as plt

# 데이터 로드
nc = Dataset("dataset/KHOA_SCU_L4_Z004_D01_U20251118_EastSea.nc", 'r')
lat = nc.variables['lat'][:]
lon = nc.variables['lon'][:]
mask = nc.variables['mask'][:]
lon_grid, lat_grid = np.meshgrid(lon, lat)

print(f"Latitude range: {lat.min():.2f}°N to {lat.max():.2f}°N")
print(f"Longitude range: {lon.min():.2f}°E to {lon.max():.2f}°E")
print(f"Mask: land={np.sum(mask==0)}, water={np.sum(mask==1)}")

# 지도 그리기
fig, ax = plt.subplots(figsize=(12, 10))

# 육지 영역 (mask=0이 육지)
land = np.where(mask == 0, 1, np.nan)
water = np.where(mask == 1, 1, np.nan)

# 바다 영역 (파란색)
ax.contourf(lon_grid, lat_grid, water, levels=[0.5, 1.5], colors='lightblue', alpha=0.5)

# 육지 영역 (회색)
ax.contourf(lon_grid, lat_grid, land, levels=[0.5, 1.5], colors='lightgray', alpha=0.8)

# 육지 경계선
ax.contour(lon_grid, lat_grid, mask, levels=[0.5], colors='black', linewidths=2)

# 주요 도시 위치 표시 (참고용)
cities = {
    'Busan': (35.1796, 129.0756),
    'Ulsan': (35.5384, 129.3114),
    'Pohang': (36.0322, 129.3650),
    'Gangneung': (37.7519, 128.8761),
    'Sokcho': (38.2070, 128.5918),
    'Niigata (Japan)': (37.9161, 139.0364),
    'Akita (Japan)': (39.7186, 140.1024),
}

for city_name, (city_lat, city_lon) in cities.items():
    if lat.min() <= city_lat <= lat.max() and lon.min() <= city_lon <= lon.max():
        ax.plot(city_lon, city_lat, 'ro', markersize=8)
        ax.annotate(city_name, (city_lon, city_lat), xytext=(5, 5), 
                   textcoords='offset points', fontsize=9)

ax.set_xlim(lon.min(), lon.max())
ax.set_ylim(lat.min(), lat.max())
ax.set_xlabel('Longitude (°E)', fontsize=12)
ax.set_ylabel('Latitude (°N)', fontsize=12)
ax.set_title('East Sea Region Map\n(Korean East Coast & Japanese West Coast)', 
            fontsize=14, fontweight='bold')
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("test_map.png", dpi=300, bbox_inches='tight')
print("Test map saved to test_map.png")
plt.close()

nc.close()


