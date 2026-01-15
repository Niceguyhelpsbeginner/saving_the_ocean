"""
동해 해류 데이터 시각화
"""
from netCDF4 import Dataset
import numpy as np
import matplotlib.pyplot as plt
import os
from setup_korean_font import setup_korean_font

# 한글 폰트 설정
setup_korean_font()

print("=" * 80)
print("East Sea Current Data Visualization")
print("=" * 80)

# 데이터 로드
current_file = "dataset/KHOA_SCU_L4_Z004_D01_U20251118_EastSea.nc"

print("\n1. Loading data...")
print("-" * 80)

nc = Dataset(current_file, 'r')
lat = nc.variables['lat'][:]
lon = nc.variables['lon'][:]
u = nc.variables['u'][:]  # 동서 방향 해류 속도
v = nc.variables['v'][:]  # 남북 방향 해류 속도
ssh = nc.variables['ssh'][:]  # 해수면 높이
mask = nc.variables['mask'][:]  # 육지/바다 마스크 (0=육지, 1=바다)

lon_grid, lat_grid = np.meshgrid(lon, lat)

print(f"Latitude range: {lat.min():.2f}°N to {lat.max():.2f}°N")
print(f"Longitude range: {lon.min():.2f}°E to {lon.max():.2f}°E")
print(f"Grid size: {len(lat)} × {len(lon)}")
print(f"Land pixels: {np.sum(mask == 0)}")
print(f"Water pixels: {np.sum(mask == 1)}")

# 해류 속도 계산
speed = np.sqrt(u**2 + v**2)

# 출력 디렉토리
output_dir = "analysis_output/east_sea_current"
os.makedirs(output_dir, exist_ok=True)

print("\n2. Creating visualizations...")
print("-" * 80)

# 1. 기본 지도 - 육지와 바다 구분
fig, ax = plt.subplots(figsize=(14, 10))

# 육지 영역 (mask=0이 육지)
land = np.where(mask == 0, 1, np.nan)
water = np.where(mask == 1, 1, np.nan)

# 바다 영역 (연한 파란색)
ax.contourf(lon_grid, lat_grid, water, levels=[0.5, 1.5], 
           colors='lightblue', alpha=0.6, zorder=1)

# 육지 영역 (회색)
ax.contourf(lon_grid, lat_grid, land, levels=[0.5, 1.5], 
           colors='lightgray', alpha=0.9, zorder=2)

# 육지 경계선 (검은색 두꺼운 선)
ax.contour(lon_grid, lat_grid, mask, levels=[0.5], 
          colors='black', linewidths=2.5, zorder=3)

ax.set_xlim(lon.min(), lon.max())
ax.set_ylim(lat.min(), lat.max())
ax.set_xlabel('Longitude (°E)', fontsize=12)
ax.set_ylabel('Latitude (°N)', fontsize=12)
ax.set_title('East Sea Region Map\n(Korean East Coast & Japanese West Coast)', 
            fontsize=14, fontweight='bold')
ax.grid(True, alpha=0.3, linestyle='--')
plt.tight_layout()
plt.savefig(f"{output_dir}/01_base_map.png", dpi=300, bbox_inches='tight')
print(f"✓ Base map saved: {output_dir}/01_base_map.png")
plt.close()

# 2. 해류 속도 지도
fig, ax = plt.subplots(figsize=(14, 10))

# 육지 먼저
ax.contourf(lon_grid, lat_grid, land, levels=[0.5, 1.5], 
           colors='lightgray', alpha=0.9, zorder=1)
ax.contour(lon_grid, lat_grid, mask, levels=[0.5], 
          colors='black', linewidths=2.5, zorder=2)

# 해류 속도 (바다 영역만)
speed_masked = np.ma.masked_where(np.isnan(speed) | (mask == 0), speed)
im = ax.contourf(lon_grid, lat_grid, speed_masked, levels=20, 
                cmap='jet', alpha=0.7, extend='both', zorder=3)

ax.set_xlim(lon.min(), lon.max())
ax.set_ylim(lat.min(), lat.max())
ax.set_xlabel('Longitude (°E)', fontsize=12)
ax.set_ylabel('Latitude (°N)', fontsize=12)
ax.set_title('Ocean Current Speed - East Sea', fontsize=14, fontweight='bold')
plt.colorbar(im, ax=ax, label='Current Speed (m/s)')
plt.tight_layout()
plt.savefig(f"{output_dir}/02_current_speed.png", dpi=300, bbox_inches='tight')
print(f"✓ Current speed map saved: {output_dir}/02_current_speed.png")
plt.close()

# 3. 해류 벡터 필드
fig, ax = plt.subplots(figsize=(14, 10))

# 육지 먼저
ax.contourf(lon_grid, lat_grid, land, levels=[0.5, 1.5], 
           colors='lightgray', alpha=0.9, zorder=1)
ax.contour(lon_grid, lat_grid, mask, levels=[0.5], 
          colors='black', linewidths=2.5, zorder=2)

# 해류 속도 배경 (옅게)
speed_masked = np.ma.masked_where(np.isnan(speed) | (mask == 0), speed)
im = ax.contourf(lon_grid, lat_grid, speed_masked, levels=20, 
                cmap='Blues', alpha=0.4, extend='both', zorder=3)

# 해류 벡터 (서브샘플링)
skip = 4
u_sub = u[::skip, ::skip]
v_sub = v[::skip, ::skip]
lon_sub = lon_grid[::skip, ::skip]
lat_sub = lat_grid[::skip, ::skip]
valid = ~np.isnan(u_sub) & ~np.isnan(v_sub) & (mask[::skip, ::skip] == 1)

# 벡터 색상을 속도에 따라
speed_sub = np.sqrt(u_sub**2 + v_sub**2)
ax.quiver(lon_sub[valid], lat_sub[valid], u_sub[valid], v_sub[valid],
          speed_sub[valid], cmap='jet', scale=0.3, width=0.003, 
          alpha=0.8, zorder=4)

ax.set_xlim(lon.min(), lon.max())
ax.set_ylim(lat.min(), lat.max())
ax.set_xlabel('Longitude (°E)', fontsize=12)
ax.set_ylabel('Latitude (°N)', fontsize=12)
ax.set_title('Ocean Current Vector Field - East Sea', fontsize=14, fontweight='bold')
plt.colorbar(im, ax=ax, label='Current Speed (m/s)')
plt.tight_layout()
plt.savefig(f"{output_dir}/03_current_vectors.png", dpi=300, bbox_inches='tight')
print(f"✓ Current vectors map saved: {output_dir}/03_current_vectors.png")
plt.close()

# 4. 해수면 높이 (SSH)
fig, ax = plt.subplots(figsize=(14, 10))

# 육지 먼저
ax.contourf(lon_grid, lat_grid, land, levels=[0.5, 1.5], 
           colors='lightgray', alpha=0.9, zorder=1)
ax.contour(lon_grid, lat_grid, mask, levels=[0.5], 
          colors='black', linewidths=2.5, zorder=2)

# SSH (바다 영역만)
ssh_masked = np.ma.masked_where(np.isnan(ssh) | (mask == 0), ssh)
im = ax.contourf(lon_grid, lat_grid, ssh_masked, levels=20, 
                cmap='viridis', alpha=0.7, extend='both', zorder=3)
ax.contour(lon_grid, lat_grid, ssh_masked, levels=20, 
          colors='gray', linewidths=0.5, alpha=0.5, zorder=4)

ax.set_xlim(lon.min(), lon.max())
ax.set_ylim(lat.min(), lat.max())
ax.set_xlabel('Longitude (°E)', fontsize=12)
ax.set_ylabel('Latitude (°N)', fontsize=12)
ax.set_title('Sea Surface Height (SSH) - East Sea', fontsize=14, fontweight='bold')
plt.colorbar(im, ax=ax, label='SSH (m)')
plt.tight_layout()
plt.savefig(f"{output_dir}/04_ssh.png", dpi=300, bbox_inches='tight')
print(f"✓ SSH map saved: {output_dir}/04_ssh.png")
plt.close()

# 5. 통합 패널
fig, axes = plt.subplots(2, 2, figsize=(16, 14))

# (0,0) 기본 지도
axes[0, 0].contourf(lon_grid, lat_grid, water, levels=[0.5, 1.5], 
                   colors='lightblue', alpha=0.6, zorder=1)
axes[0, 0].contourf(lon_grid, lat_grid, land, levels=[0.5, 1.5], 
                   colors='lightgray', alpha=0.9, zorder=2)
axes[0, 0].contour(lon_grid, lat_grid, mask, levels=[0.5], 
                  colors='black', linewidths=2, zorder=3)
axes[0, 0].set_xlim(lon.min(), lon.max())
axes[0, 0].set_ylim(lat.min(), lat.max())
axes[0, 0].set_title('Base Map', fontsize=12, fontweight='bold')
axes[0, 0].set_xlabel('Longitude (°E)')
axes[0, 0].set_ylabel('Latitude (°N)')
axes[0, 0].grid(True, alpha=0.3)

# (0,1) 해류 속도
axes[0, 1].contourf(lon_grid, lat_grid, land, levels=[0.5, 1.5], 
                   colors='lightgray', alpha=0.9, zorder=1)
axes[0, 1].contour(lon_grid, lat_grid, mask, levels=[0.5], 
                  colors='black', linewidths=1.5, zorder=2)
im1 = axes[0, 1].contourf(lon_grid, lat_grid, speed_masked, levels=20, 
                         cmap='jet', alpha=0.7, extend='both', zorder=3)
axes[0, 1].set_xlim(lon.min(), lon.max())
axes[0, 1].set_ylim(lat.min(), lat.max())
axes[0, 1].set_title('Current Speed', fontsize=12, fontweight='bold')
axes[0, 1].set_xlabel('Longitude (°E)')
axes[0, 1].set_ylabel('Latitude (°N)')
plt.colorbar(im1, ax=axes[0, 1], label='m/s')

# (1,0) 해류 벡터
axes[1, 0].contourf(lon_grid, lat_grid, land, levels=[0.5, 1.5], 
                   colors='lightgray', alpha=0.9, zorder=1)
axes[1, 0].contour(lon_grid, lat_grid, mask, levels=[0.5], 
                  colors='black', linewidths=1.5, zorder=2)
im2 = axes[1, 0].contourf(lon_grid, lat_grid, speed_masked, levels=20, 
                         cmap='Blues', alpha=0.4, extend='both', zorder=3)
axes[1, 0].quiver(lon_sub[valid], lat_sub[valid], u_sub[valid], v_sub[valid],
                  speed_sub[valid], cmap='jet', scale=0.3, width=0.003, 
                  alpha=0.8, zorder=4)
axes[1, 0].set_xlim(lon.min(), lon.max())
axes[1, 0].set_ylim(lat.min(), lat.max())
axes[1, 0].set_title('Current Vectors', fontsize=12, fontweight='bold')
axes[1, 0].set_xlabel('Longitude (°E)')
axes[1, 0].set_ylabel('Latitude (°N)')
plt.colorbar(im2, ax=axes[1, 0], label='m/s')

# (1,1) SSH
axes[1, 1].contourf(lon_grid, lat_grid, land, levels=[0.5, 1.5], 
                   colors='lightgray', alpha=0.9, zorder=1)
axes[1, 1].contour(lon_grid, lat_grid, mask, levels=[0.5], 
                  colors='black', linewidths=1.5, zorder=2)
im3 = axes[1, 1].contourf(lon_grid, lat_grid, ssh_masked, levels=20, 
                         cmap='viridis', alpha=0.7, extend='both', zorder=3)
axes[1, 1].set_xlim(lon.min(), lon.max())
axes[1, 1].set_ylim(lat.min(), lat.max())
axes[1, 1].set_title('Sea Surface Height', fontsize=12, fontweight='bold')
axes[1, 1].set_xlabel('Longitude (°E)')
axes[1, 1].set_ylabel('Latitude (°N)')
plt.colorbar(im3, ax=axes[1, 1], label='m')

plt.suptitle('East Sea Ocean Current Data Visualization', 
            fontsize=16, fontweight='bold', y=0.995)
plt.tight_layout()
plt.savefig(f"{output_dir}/05_comprehensive.png", dpi=300, bbox_inches='tight')
print(f"✓ Comprehensive panel saved: {output_dir}/05_comprehensive.png")
plt.close()

# 데이터 요약
print("\n3. Data Summary")
print("-" * 80)
print(f"Current Speed:")
print(f"  Min: {np.nanmin(speed):.4f} m/s")
print(f"  Max: {np.nanmax(speed):.4f} m/s")
print(f"  Mean: {np.nanmean(speed):.4f} m/s")
print(f"\nSea Surface Height:")
print(f"  Min: {np.nanmin(ssh):.4f} m")
print(f"  Max: {np.nanmax(ssh):.4f} m")
print(f"  Mean: {np.nanmean(ssh):.4f} m")

nc.close()

print("\n" + "=" * 80)
print("Visualization completed!")
print(f"All maps saved in '{output_dir}' folder.")
print("=" * 80)

