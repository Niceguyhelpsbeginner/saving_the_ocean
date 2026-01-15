"""
추가 유의미한 분석들
"""
import pandas as pd
import numpy as np
from netCDF4 import Dataset
import matplotlib.pyplot as plt
from scipy.interpolate import griddata
from scipy.ndimage import gaussian_filter
import os
from setup_korean_font import setup_korean_font

setup_korean_font()

print("=" * 80)
print("Additional Meaningful Analysis")
print("=" * 80)

# 데이터 로드
trash_file = "dataset/trash/eastsea_vessel_threat_coastallitter-2022.csv"
current_file = "dataset/KHOA_SCU_L4_Z004_D01_U20251118_EastSea.nc"

trash_df = pd.read_csv(trash_file, encoding='utf-8')
trash_df['LAT'] = (trash_df['STR_LA'] + trash_df['END_LA']) / 2
trash_df['LON'] = (trash_df['STR_LO'] + trash_df['END_LO']) / 2

area_trash = trash_df.groupby('INVS_AREA_NM').agg({
    'IEM_CNT': 'sum',
    'LAT': 'mean',
    'LON': 'mean'
}).reset_index()
area_trash.columns = ['AREA', 'TOTAL_TRASH', 'LAT', 'LON']

nc = Dataset(current_file, 'r')
lat = nc.variables['lat'][:]
lon = nc.variables['lon'][:]
u = nc.variables['u'][:]
v = nc.variables['v'][:]
ssh = nc.variables['ssh'][:]
mask = nc.variables['mask'][:]
lon_grid, lat_grid = np.meshgrid(lon, lat)

def get_current_at_location(lat_val, lon_val, lat_grid, lon_grid, data):
    if lat_val < lat.min() or lat_val > lat.max() or lon_val < lon.min() or lon_val > lon.max():
        return np.nan
    valid_mask = ~np.isnan(data)
    if np.sum(valid_mask) == 0:
        return np.nan
    valid_lat = lat_grid[valid_mask]
    valid_lon = lon_grid[valid_mask]
    valid_data = data[valid_mask]
    try:
        interpolated = griddata((valid_lon.flatten(), valid_lat.flatten()), 
                               valid_data.flatten(), (lon_val, lat_val), 
                               method='nearest')
        return interpolated if not np.isnan(interpolated) else np.nan
    except:
        lat_idx = np.argmin(np.abs(lat - lat_val))
        lon_idx = np.argmin(np.abs(lon - lon_val))
        if 0 <= lat_idx < len(lat) and 0 <= lon_idx < len(lon):
            return data[lat_idx, lon_idx]
        return np.nan

output_dir = "analysis_output/additional_analysis"
os.makedirs(output_dir, exist_ok=True)

# 분석 1: 해류 발산/수렴 분석 (Divergence Analysis)
print("\n1. Current Divergence/Convergence Analysis")
print("-" * 80)

lat_rad = np.deg2rad(lat_grid)
dlat_m = 111000.0
dlon_m = 111000.0 * np.cos(lat_rad)
dlat_deg = lat[1] - lat[0] if len(lat) > 1 else 0.25
dlon_deg = lon[1] - lon[0] if len(lon) > 1 else 0.25
dx = dlon_deg * dlon_m
dy = dlat_deg * dlat_m

u_masked = np.where((mask == 1) & ~np.isnan(u), u, np.nan)
v_masked = np.where((mask == 1) & ~np.isnan(v), v, np.nan)

# 발산도 계산: ∇·v = ∂u/∂x + ∂v/∂y
du_dx = np.gradient(u_masked, axis=1) / dx
dv_dy = np.gradient(v_masked, axis=0) / dy
divergence = du_dx + dv_dy

# 수렴 지역 (음수 발산도) = 쓰레기가 모이는 지역
convergence_mask = (divergence < 0) & (mask == 1)
convergence_strength = np.where(convergence_mask, -divergence, 0)
convergence_strength = np.ma.masked_where(mask == 0, convergence_strength)

print(f"Convergence areas (trash accumulation zones): {np.sum(convergence_mask)} pixels")
print(f"Average convergence strength: {np.nanmean(convergence_strength):.2e} s^-1")

# 분석 2: 쓰레기 집적 위험도 지수 개발
print("\n2. Trash Accumulation Risk Index")
print("-" * 80)

speed = np.sqrt(u**2 + v**2)
speed_masked = np.ma.masked_where(np.isnan(speed) | (mask == 0), speed)

# Vorticity 계산
dv_dx = np.gradient(v_masked, axis=1) / dx
du_dy = np.gradient(u_masked, axis=0) / dy
vorticity = dv_dx - du_dy
vorticity_masked = np.ma.masked_where(np.isnan(vorticity) | (mask == 0), vorticity)

# 위험도 지수 구성 요소 정규화
speed_norm = (speed_masked - np.nanmin(speed_masked)) / (np.nanmax(speed_masked) - np.nanmin(speed_masked))
vorticity_norm = (np.abs(vorticity_masked) - np.nanmin(np.abs(vorticity_masked))) / \
                 (np.nanmax(np.abs(vorticity_masked)) - np.nanmin(np.abs(vorticity_masked)))
convergence_norm = (convergence_strength - np.nanmin(convergence_strength)) / \
                  (np.nanmax(convergence_strength) - np.nanmin(convergence_strength))

# 위험도 지수 = 느린 해류(0.4) + 높은 vorticity(0.3) + 수렴(0.3)
risk_index = (1 - speed_norm) * 0.4 + vorticity_norm * 0.3 + convergence_norm * 0.3
risk_index = np.ma.masked_where(mask == 0, risk_index)

print(f"Risk index range: {np.nanmin(risk_index):.3f} to {np.nanmax(risk_index):.3f}")
print(f"High risk areas (>= 0.7): {np.sum(risk_index >= 0.7)} pixels")

# 분석 3: 쓰레기 이동 역추적 (Backward Trajectory)
print("\n3. Backward Trajectory Analysis")
print("-" * 80)

def backward_trajectory(start_lat, start_lon, days=7, dt_hours=6):
    """쓰레기 역추적 (해류 방향 반대로)"""
    trajectory = [(start_lat, start_lon)]
    current_lat, current_lon = start_lat, start_lon
    
    for day in range(days):
        for hour in range(0, 24, dt_hours):
            u_curr = get_current_at_location(current_lat, current_lon, lat_grid, lon_grid, u)
            v_curr = get_current_at_location(current_lat, current_lon, lat_grid, lon_grid, v)
            
            if np.isnan(u_curr) or np.isnan(v_curr):
                break
            
            # 해류 방향 반대로 이동
            dt_seconds = dt_hours * 3600
            dlat = -v_curr * dt_seconds / 111000.0
            dlon = -u_curr * dt_seconds / (111000.0 * np.cos(np.radians(current_lat)))
            
            current_lat += dlat
            current_lon += dlon
            
            if current_lat < lat.min() or current_lat > lat.max() or \
               current_lon < lon.min() or current_lon > lon.max():
                break
            
            trajectory.append((current_lat, current_lon))
    
    return trajectory

# 주요 쓰레기 집적 지역에서 역추적
top_areas = area_trash.nlargest(3, 'TOTAL_TRASH')
backward_trajectories = {}

for idx, row in top_areas.iterrows():
    traj = backward_trajectory(row['LAT'], row['LON'], days=7)
    backward_trajectories[row['AREA']] = traj
    print(f"\n{row['AREA']} backward trajectory:")
    print(f"  Current location: ({row['LAT']:.4f}°N, {row['LON']:.4f}°E)")
    if len(traj) > 1:
        print(f"  Possible origin (7 days ago): ({traj[-1][0]:.4f}°N, {traj[-1][1]:.4f}°E)")

# 분석 4: 쓰레기 밀도 예측 지도
print("\n4. Trash Density Prediction Map")
print("-" * 80)

# 실제 쓰레기 위치를 기반으로 밀도 추정
from scipy.spatial.distance import cdist

# 격자점에서 실제 쓰레기 지점까지의 거리 계산
grid_points = np.column_stack([lat_grid.flatten(), lon_grid.flatten()])
trash_points = area_trash[['LAT', 'LON']].values
trash_counts = area_trash['TOTAL_TRASH'].values

# 거리 기반 가중치로 밀도 추정
predicted_density = np.zeros(len(grid_points))
for i, grid_point in enumerate(grid_points):
    if mask.flatten()[i] == 1:  # 바다 영역만
        distances = cdist([grid_point], trash_points)[0]
        # 가까운 쓰레기 지점일수록 높은 가중치
        weights = np.exp(-distances * 10)  # 거리 스케일 조정
        predicted_density[i] = np.sum(weights * trash_counts)

predicted_density = predicted_density.reshape(lat_grid.shape)
predicted_density = np.ma.masked_where(mask == 0, predicted_density)

print(f"Predicted density range: {np.nanmin(predicted_density):.2f} to {np.nanmax(predicted_density):.2f}")

# 시각화
print("\n5. Creating visualizations...")
print("-" * 80)

# 1. 발산/수렴 지도
fig, ax = plt.subplots(figsize=(14, 10))

land = np.where(mask == 0, 1, np.nan)
ax.contourf(lon_grid, lat_grid, land, levels=[0.5, 1.5], 
           colors='lightgray', alpha=0.9, zorder=1)
ax.contour(lon_grid, lat_grid, mask, levels=[0.5], 
          colors='black', linewidths=2.5, zorder=2)

divergence_masked = np.ma.masked_where(np.isnan(divergence) | (mask == 0), divergence)
im = ax.contourf(lon_grid, lat_grid, divergence_masked, levels=30, 
                cmap='RdBu_r', alpha=0.7, extend='both', zorder=3)

# 수렴 지역 강조
ax.contour(lon_grid, lat_grid, convergence_strength, levels=5, 
          colors='red', linewidths=1.5, alpha=0.6, zorder=4)

ax.scatter(area_trash['LON'], area_trash['LAT'], 
          s=area_trash['TOTAL_TRASH']*2, c='yellow', 
          edgecolors='black', linewidths=2, zorder=5, label='Trash Sites')

ax.set_xlim(lon.min(), lon.max())
ax.set_ylim(lat.min(), lat.max())
ax.set_xlabel('Longitude (°E)', fontsize=12)
ax.set_ylabel('Latitude (°N)', fontsize=12)
ax.set_title('Current Divergence/Convergence\n(Red: Convergence zones = Trash accumulation)', 
            fontsize=14, fontweight='bold')
ax.legend(loc='upper right', fontsize=10)
plt.colorbar(im, ax=ax, label='Divergence (s⁻¹)')
plt.tight_layout()
plt.savefig(f"{output_dir}/divergence_convergence.png", dpi=300, bbox_inches='tight')
print(f"✓ Divergence/convergence map saved: {output_dir}/divergence_convergence.png")
plt.close()

# 2. 위험도 지수 지도
fig, ax = plt.subplots(figsize=(14, 10))

ax.contourf(lon_grid, lat_grid, land, levels=[0.5, 1.5], 
           colors='lightgray', alpha=0.9, zorder=1)
ax.contour(lon_grid, lat_grid, mask, levels=[0.5], 
          colors='black', linewidths=2.5, zorder=2)

im = ax.contourf(lon_grid, lat_grid, risk_index, levels=20, 
                cmap='YlOrRd', alpha=0.8, extend='max', zorder=3)

ax.scatter(area_trash['LON'], area_trash['LAT'], 
          s=area_trash['TOTAL_TRASH']*3, c=area_trash['TOTAL_TRASH'],
          cmap='Reds', alpha=0.8, edgecolors='darkred', linewidths=2,
          zorder=5, label='Actual Trash Sites')

ax.set_xlim(lon.min(), lon.max())
ax.set_ylim(lat.min(), lat.max())
ax.set_xlabel('Longitude (°E)', fontsize=12)
ax.set_ylabel('Latitude (°N)', fontsize=12)
ax.set_title('Trash Accumulation Risk Index\n(Slow Current + High Vorticity + Convergence)', 
            fontsize=14, fontweight='bold')
ax.legend(loc='upper right', fontsize=10)
plt.colorbar(im, ax=ax, label='Risk Index')
plt.tight_layout()
plt.savefig(f"{output_dir}/risk_index.png", dpi=300, bbox_inches='tight')
print(f"✓ Risk index map saved: {output_dir}/risk_index.png")
plt.close()

# 3. 역추적 경로
fig, ax = plt.subplots(figsize=(14, 10))

ax.contourf(lon_grid, lat_grid, land, levels=[0.5, 1.5], 
           colors='lightgray', alpha=0.9, zorder=1)
ax.contour(lon_grid, lat_grid, mask, levels=[0.5], 
          colors='black', linewidths=2.5, zorder=2)

speed_masked = np.ma.masked_where(np.isnan(speed) | (mask == 0), speed)
im = ax.contourf(lon_grid, lat_grid, speed_masked, levels=20, 
                cmap='Blues', alpha=0.4, extend='both', zorder=3)

colors = plt.cm.tab10(np.linspace(0, 1, len(backward_trajectories)))
for (area_name, traj), color in zip(backward_trajectories.items(), colors):
    if len(traj) > 1:
        traj_lons = [t[1] for t in traj]
        traj_lats = [t[0] for t in traj]
        ax.plot(traj_lons, traj_lats, '--', linewidth=2.5, alpha=0.8, 
               color=color, label=f'{area_name} (backward)', zorder=4)
        ax.scatter(traj_lons[0], traj_lats[0], s=150, c=color, 
                  marker='o', edgecolors='black', linewidths=2, zorder=5, label='Current')
        ax.scatter(traj_lons[-1], traj_lats[-1], s=150, c=color, 
                  marker='*', edgecolors='black', linewidths=2, zorder=5, label='Possible Origin')

ax.set_xlim(lon.min(), lon.max())
ax.set_ylim(lat.min(), lat.max())
ax.set_xlabel('Longitude (°E)', fontsize=12)
ax.set_ylabel('Latitude (°N)', fontsize=12)
ax.set_title('Backward Trajectory Analysis\n(Where did the trash come from?)', 
            fontsize=14, fontweight='bold')
ax.legend(loc='upper right', fontsize=9, ncol=2)
plt.colorbar(im, ax=ax, label='Current Speed (m/s)')
plt.tight_layout()
plt.savefig(f"{output_dir}/backward_trajectory.png", dpi=300, bbox_inches='tight')
print(f"✓ Backward trajectory saved: {output_dir}/backward_trajectory.png")
plt.close()

# 4. 쓰레기 밀도 예측 지도
fig, ax = plt.subplots(figsize=(14, 10))

ax.contourf(lon_grid, lat_grid, land, levels=[0.5, 1.5], 
           colors='lightgray', alpha=0.9, zorder=1)
ax.contour(lon_grid, lat_grid, mask, levels=[0.5], 
          colors='black', linewidths=2.5, zorder=2)

# 예측 밀도 스무딩
predicted_smooth = gaussian_filter(predicted_density, sigma=2.0)
predicted_smooth = np.ma.masked_where(mask == 0, predicted_smooth)

im = ax.contourf(lon_grid, lat_grid, predicted_smooth, levels=20, 
                cmap='Reds', alpha=0.7, extend='max', zorder=3)

# 실제 쓰레기 위치
ax.scatter(area_trash['LON'], area_trash['LAT'], 
          s=area_trash['TOTAL_TRASH']*3, c='yellow', 
          edgecolors='black', linewidths=2, zorder=5, label='Actual Sites')

ax.set_xlim(lon.min(), lon.max())
ax.set_ylim(lat.min(), lat.max())
ax.set_xlabel('Longitude (°E)', fontsize=12)
ax.set_ylabel('Latitude (°N)', fontsize=12)
ax.set_title('Predicted Trash Density Map\n(Based on Distance from Known Sites)', 
            fontsize=14, fontweight='bold')
ax.legend(loc='upper right', fontsize=10)
plt.colorbar(im, ax=ax, label='Predicted Density')
plt.tight_layout()
plt.savefig(f"{output_dir}/predicted_density.png", dpi=300, bbox_inches='tight')
print(f"✓ Predicted density map saved: {output_dir}/predicted_density.png")
plt.close()

# 5. 통합 분석 패널
fig, axes = plt.subplots(2, 2, figsize=(16, 14))

# 발산/수렴
axes[0, 0].contourf(lon_grid, lat_grid, land, levels=[0.5, 1.5], 
                   colors='lightgray', alpha=0.9, zorder=1)
axes[0, 0].contour(lon_grid, lat_grid, mask, levels=[0.5], 
                  colors='black', linewidths=1.5, zorder=2)
im1 = axes[0, 0].contourf(lon_grid, lat_grid, divergence_masked, levels=20, 
                         cmap='RdBu_r', alpha=0.7, extend='both', zorder=3)
axes[0, 0].set_xlim(lon.min(), lon.max())
axes[0, 0].set_ylim(lat.min(), lat.max())
axes[0, 0].set_title('Divergence/Convergence', fontsize=11, fontweight='bold')
axes[0, 0].set_xlabel('Longitude (°E)')
axes[0, 0].set_ylabel('Latitude (°N)')
plt.colorbar(im1, ax=axes[0, 0], label='s⁻¹')

# 위험도 지수
axes[0, 1].contourf(lon_grid, lat_grid, land, levels=[0.5, 1.5], 
                   colors='lightgray', alpha=0.9, zorder=1)
axes[0, 1].contour(lon_grid, lat_grid, mask, levels=[0.5], 
                  colors='black', linewidths=1.5, zorder=2)
im2 = axes[0, 1].contourf(lon_grid, lat_grid, risk_index, levels=20, 
                         cmap='YlOrRd', alpha=0.8, extend='max', zorder=3)
axes[0, 1].scatter(area_trash['LON'], area_trash['LAT'], 
                 s=area_trash['TOTAL_TRASH']*2, c='red', 
                 alpha=0.7, edgecolors='black', linewidths=1, zorder=4)
axes[0, 1].set_xlim(lon.min(), lon.max())
axes[0, 1].set_ylim(lat.min(), lat.max())
axes[0, 1].set_title('Risk Index', fontsize=11, fontweight='bold')
axes[0, 1].set_xlabel('Longitude (°E)')
axes[0, 1].set_ylabel('Latitude (°N)')
plt.colorbar(im2, ax=axes[0, 1], label='Risk')

# 역추적
axes[1, 0].contourf(lon_grid, lat_grid, land, levels=[0.5, 1.5], 
                   colors='lightgray', alpha=0.9, zorder=1)
axes[1, 0].contour(lon_grid, lat_grid, mask, levels=[0.5], 
                  colors='black', linewidths=1.5, zorder=2)
for (area_name, traj), color in zip(list(backward_trajectories.items())[:3], 
                                    plt.cm.tab10(np.linspace(0, 1, 3))):
    if len(traj) > 1:
        traj_lons = [t[1] for t in traj]
        traj_lats = [t[0] for t in traj]
        axes[1, 0].plot(traj_lons, traj_lats, '--', linewidth=2, 
                       alpha=0.7, color=color, label=area_name[:8])
axes[1, 0].set_xlim(lon.min(), lon.max())
axes[1, 0].set_ylim(lat.min(), lat.max())
axes[1, 0].set_title('Backward Trajectory', fontsize=11, fontweight='bold')
axes[1, 0].set_xlabel('Longitude (°E)')
axes[1, 0].set_ylabel('Latitude (°N)')
axes[1, 0].legend(loc='upper right', fontsize=8)

# 예측 밀도
axes[1, 1].contourf(lon_grid, lat_grid, land, levels=[0.5, 1.5], 
                   colors='lightgray', alpha=0.9, zorder=1)
axes[1, 1].contour(lon_grid, lat_grid, mask, levels=[0.5], 
                  colors='black', linewidths=1.5, zorder=2)
im4 = axes[1, 1].contourf(lon_grid, lat_grid, predicted_smooth, levels=20, 
                         cmap='Reds', alpha=0.7, extend='max', zorder=3)
axes[1, 1].scatter(area_trash['LON'], area_trash['LAT'], 
                 s=area_trash['TOTAL_TRASH']*2, c='yellow', 
                 edgecolors='black', linewidths=1, zorder=4)
axes[1, 1].set_xlim(lon.min(), lon.max())
axes[1, 1].set_ylim(lat.min(), lat.max())
axes[1, 1].set_title('Predicted Density', fontsize=11, fontweight='bold')
axes[1, 1].set_xlabel('Longitude (°E)')
axes[1, 1].set_ylabel('Latitude (°N)')
plt.colorbar(im4, ax=axes[1, 1], label='Density')

plt.suptitle('Additional Meaningful Analysis', fontsize=14, fontweight='bold', y=0.995)
plt.tight_layout()
plt.savefig(f"{output_dir}/comprehensive_additional.png", dpi=300, bbox_inches='tight')
print(f"✓ Comprehensive additional analysis saved: {output_dir}/comprehensive_additional.png")
plt.close()

# 결과 요약
with open(f"{output_dir}/additional_analysis_summary.txt", "w", encoding="utf-8") as f:
    f.write("=" * 80 + "\n")
    f.write("Additional Meaningful Analysis Summary\n")
    f.write("=" * 80 + "\n\n")
    
    f.write("1. Divergence/Convergence Analysis\n")
    f.write("-" * 80 + "\n")
    f.write(f"Convergence areas (trash accumulation): {np.sum(convergence_mask)} pixels\n")
    f.write(f"Average convergence strength: {np.nanmean(convergence_strength):.2e} s^-1\n")
    f.write("Interpretation: Negative divergence (convergence) indicates areas where\n")
    f.write("currents converge, leading to trash accumulation.\n\n")
    
    f.write("2. Risk Index\n")
    f.write("-" * 80 + "\n")
    f.write("Components:\n")
    f.write("  - Slow current (40%%): Areas with slower currents accumulate more trash\n")
    f.write("  - High vorticity (30%%): Eddy regions trap trash\n")
    f.write("  - Convergence (30%%): Converging currents collect trash\n")
    f.write(f"High risk areas (>= 0.7): {np.sum(risk_index >= 0.7)} pixels\n\n")
    
    f.write("3. Backward Trajectory\n")
    f.write("-" * 80 + "\n")
    f.write("Tracing back where trash might have come from:\n")
    for area_name, traj in backward_trajectories.items():
        if len(traj) > 1:
            f.write(f"  {area_name}: Possible origin at ({traj[-1][0]:.4f}°N, {traj[-1][1]:.4f}°E)\n")
    
    f.write("\n4. Key Insights\n")
    f.write("-" * 80 + "\n")
    f.write("- Convergence zones are key indicators of trash accumulation\n")
    f.write("- Risk index combines multiple factors for better prediction\n")
    f.write("- Backward trajectory helps identify trash sources\n")
    f.write("- Predicted density map can guide monitoring efforts\n")

print(f"\n✓ Additional analysis summary saved: {output_dir}/additional_analysis_summary.txt")

nc.close()

print("\n" + "=" * 80)
print("Additional analysis completed!")
print(f"All results saved in '{output_dir}' folder.")
print("=" * 80)


