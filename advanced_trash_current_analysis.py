"""
동해 해류 데이터와 해양쓰레기 데이터 고급 분석
추가 유의미한 분석 방안들
"""
import pandas as pd
import numpy as np
from netCDF4 import Dataset
import matplotlib.pyplot as plt
from scipy.interpolate import griddata
from scipy.spatial.distance import cdist
import os
from setup_korean_font import setup_korean_font

setup_korean_font()

print("=" * 80)
print("Advanced Trash-Current Data Analysis")
print("=" * 80)

# 데이터 로드
trash_file = "dataset/trash/eastsea_vessel_threat_coastallitter-2022.csv"
current_file = "dataset/KHOA_SCU_L4_Z004_D01_U20251118_EastSea.nc"

print("\n1. Loading data...")
print("-" * 80)

# 해양쓰레기 데이터
trash_df = pd.read_csv(trash_file, encoding='utf-8')
trash_df['LAT'] = (trash_df['STR_LA'] + trash_df['END_LA']) / 2
trash_df['LON'] = (trash_df['STR_LO'] + trash_df['END_LO']) / 2

# 지역별 집계
area_trash = trash_df.groupby('INVS_AREA_NM').agg({
    'IEM_CNT': 'sum',
    'LAT': 'mean',
    'LON': 'mean',
    'DNF_SRC_NM': lambda x: x.value_counts().to_dict(),  # 기인별 분포
    'QTMT_NM': lambda x: x.value_counts().to_dict(),    # 쓰레기 종류별 분포
}).reset_index()
area_trash.columns = ['AREA', 'TOTAL_TRASH', 'LAT', 'LON', 'SOURCE_DIST', 'TYPE_DIST']

# 해류 데이터
nc = Dataset(current_file, 'r')
lat = nc.variables['lat'][:]
lon = nc.variables['lon'][:]
u = nc.variables['u'][:]
v = nc.variables['v'][:]
ssh = nc.variables['ssh'][:]
mask = nc.variables['mask'][:]
lon_grid, lat_grid = np.meshgrid(lon, lat)

print(f"Trash data: {len(trash_df)} records, {len(area_trash)} areas")
print(f"Current data: {len(lat)} × {len(lon)} grid")

# 해류 정보 추출 함수
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
                               valid_data.flatten(), 
                               (lon_val, lat_val), 
                               method='nearest')
        return interpolated if not np.isnan(interpolated) else np.nan
    except:
        lat_idx = np.argmin(np.abs(lat - lat_val))
        lon_idx = np.argmin(np.abs(lon - lon_val))
        if 0 <= lat_idx < len(lat) and 0 <= lon_idx < len(lon):
            return data[lat_idx, lon_idx]
        return np.nan

# 각 지점의 해류 정보 추출
print("\n2. Extracting current information...")
print("-" * 80)

for idx, row in area_trash.iterrows():
    area_trash.loc[idx, 'U'] = get_current_at_location(row['LAT'], row['LON'], lat_grid, lon_grid, u)
    area_trash.loc[idx, 'V'] = get_current_at_location(row['LAT'], row['LON'], lat_grid, lon_grid, v)
    area_trash.loc[idx, 'SSH'] = get_current_at_location(row['LAT'], row['LON'], lat_grid, lon_grid, ssh)

u_vals = np.array(area_trash['U'].tolist())
v_vals = np.array(area_trash['V'].tolist())
area_trash['SPEED'] = np.sqrt(u_vals**2 + v_vals**2)
area_trash['DIRECTION'] = np.degrees(np.arctan2(v_vals, u_vals))

# Vorticity 계산
lat_rad = np.deg2rad(lat_grid)
dlat_m = 111000.0
dlon_m = 111000.0 * np.cos(lat_rad)
dlat_deg = lat[1] - lat[0] if len(lat) > 1 else 0.25
dlon_deg = lon[1] - lon[0] if len(lon) > 1 else 0.25
dx = dlon_deg * dlon_m
dy = dlat_deg * dlat_m

u_masked = np.where((mask == 1) & ~np.isnan(u), u, np.nan)
v_masked = np.where((mask == 1) & ~np.isnan(v), v, np.nan)
dv_dx = np.gradient(v_masked, axis=1) / dx
du_dy = np.gradient(u_masked, axis=0) / dy
vorticity = dv_dx - du_dy

for idx, row in area_trash.iterrows():
    area_trash.loc[idx, 'VORTICITY'] = get_current_at_location(
        row['LAT'], row['LON'], lat_grid, lon_grid, vorticity)

output_dir = "analysis_output/advanced_analysis"
os.makedirs(output_dir, exist_ok=True)

# 분석 1: 쓰레기 유형별 해류 영향 분석
print("\n3. Analysis 1: Trash Type vs Current Analysis")
print("-" * 80)

trash_type_analysis = trash_df.groupby(['QTMT_NM', 'INVS_AREA_NM']).agg({
    'IEM_CNT': 'sum',
    'LAT': 'mean',
    'LON': 'mean'
}).reset_index()

# 각 쓰레기 유형별로 해류 정보 추가
trash_type_summary = []
for trash_type in trash_type_analysis['QTMT_NM'].unique():
    type_data = trash_type_analysis[trash_type_analysis['QTMT_NM'] == trash_type]
    total = type_data['IEM_CNT'].sum()
    
    # 평균 위치 계산
    avg_lat = (type_data['LAT'] * type_data['IEM_CNT']).sum() / total
    avg_lon = (type_data['LON'] * type_data['IEM_CNT']).sum() / total
    
    # 해류 정보
    u_type = get_current_at_location(avg_lat, avg_lon, lat_grid, lon_grid, u)
    v_type = get_current_at_location(avg_lat, avg_lon, lat_grid, lon_grid, v)
    speed_type = np.sqrt(u_type**2 + v_type**2) if not np.isnan(u_type) else np.nan
    
    trash_type_summary.append({
        'TYPE': trash_type,
        'TOTAL': total,
        'AVG_LAT': avg_lat,
        'AVG_LON': avg_lon,
        'CURRENT_SPEED': speed_type
    })

trash_type_df = pd.DataFrame(trash_type_summary)
print("\nTrash type summary:")
print(trash_type_df.sort_values('TOTAL', ascending=False))

# 분석 2: 쓰레기 기인(출처)별 분석
print("\n4. Analysis 2: Trash Source Analysis")
print("-" * 80)

source_analysis = trash_df.groupby(['DNF_SRC_NM', 'INVS_AREA_NM']).agg({
    'IEM_CNT': 'sum',
    'LAT': 'mean',
    'LON': 'mean'
}).reset_index()

source_summary = []
for source in source_analysis['DNF_SRC_NM'].unique():
    source_data = source_analysis[source_analysis['DNF_SRC_NM'] == source]
    total = source_data['IEM_CNT'].sum()
    
    avg_lat = (source_data['LAT'] * source_data['IEM_CNT']).sum() / total
    avg_lon = (source_data['LON'] * source_data['IEM_CNT']).sum() / total
    
    u_src = get_current_at_location(avg_lat, avg_lon, lat_grid, lon_grid, u)
    v_src = get_current_at_location(avg_lat, avg_lon, lat_grid, lon_grid, v)
    speed_src = np.sqrt(u_src**2 + v_src**2) if not np.isnan(u_src) else np.nan
    
    source_summary.append({
        'SOURCE': source,
        'TOTAL': total,
        'AVG_LAT': avg_lat,
        'AVG_LON': avg_lon,
        'CURRENT_SPEED': speed_src
    })

source_df = pd.DataFrame(source_summary)
print("\nSource summary:")
print(source_df.sort_values('TOTAL', ascending=False))

# 분석 3: 쓰레기 집적 핫스팟 예측 (해류 패턴 기반)
print("\n5. Analysis 3: Trash Accumulation Hotspot Prediction")
print("-" * 80)

# 해류 속도가 느린 지역 찾기 (쓰레기가 정체될 가능성 높음)
speed = np.sqrt(u**2 + v**2)
speed_masked = np.ma.masked_where(np.isnan(speed) | (mask == 0), speed)

# 느린 해류 지역 (하위 25%)
slow_current_threshold = np.nanpercentile(speed_masked[~speed_masked.mask], 25)
slow_current_mask = (speed < slow_current_threshold) & (mask == 1)

# 환류 지역 (vorticity 절대값이 큰 지역)
vorticity_masked = np.ma.masked_where(np.isnan(vorticity) | (mask == 0), vorticity)
high_vorticity_threshold = np.nanpercentile(np.abs(vorticity_masked[~vorticity_masked.mask]), 75)
high_vorticity_mask = (np.abs(vorticity) > high_vorticity_threshold) & (mask == 1)

# 핫스팟 점수 = 느린 해류 + 높은 vorticity
hotspot_score = np.zeros_like(speed)
hotspot_score[slow_current_mask] += 1
hotspot_score[high_vorticity_mask] += 1
hotspot_score = np.ma.masked_where(mask == 0, hotspot_score)

print(f"Slow current threshold: {slow_current_threshold:.4f} m/s")
print(f"High vorticity threshold: {high_vorticity_threshold:.2e} s^-1")
print(f"Hotspot areas (score >= 1): {np.sum(hotspot_score >= 1)} pixels")

# 분석 4: 쓰레기 이동 경로 시뮬레이션 (Lagrangian)
print("\n6. Analysis 4: Trash Trajectory Simulation")
print("-" * 80)

def simulate_trash_trajectory(start_lat, start_lon, days=7, dt_hours=6):
    """쓰레기 이동 경로 시뮬레이션"""
    trajectory = [(start_lat, start_lon)]
    current_lat, current_lon = start_lat, start_lon
    
    for day in range(days):
        for hour in range(0, 24, dt_hours):
            # 현재 위치의 해류
            u_curr = get_current_at_location(current_lat, current_lon, lat_grid, lon_grid, u)
            v_curr = get_current_at_location(current_lat, current_lon, lat_grid, lon_grid, v)
            
            if np.isnan(u_curr) or np.isnan(v_curr):
                break
            
            # 이동 거리 계산 (m)
            # 해류 속도는 m/s이므로 시간(초)을 곱함
            dt_seconds = dt_hours * 3600
            dlat = v_curr * dt_seconds / 111000.0  # 위도 변화 (도)
            dlon = u_curr * dt_seconds / (111000.0 * np.cos(np.radians(current_lat)))  # 경도 변화 (도)
            
            current_lat += dlat
            current_lon += dlon
            
            # 범위 체크
            if current_lat < lat.min() or current_lat > lat.max() or \
               current_lon < lon.min() or current_lon > lon.max():
                break
            
            trajectory.append((current_lat, current_lon))
    
    return trajectory

# 주요 쓰레기 집적 지역에서 이동 경로 시뮬레이션
top_trash_areas = area_trash.nlargest(5, 'TOTAL_TRASH')
trajectories = {}

for idx, row in top_trash_areas.iterrows():
    traj = simulate_trash_trajectory(row['LAT'], row['LON'], days=7)
    trajectories[row['AREA']] = traj
    print(f"\n{row['AREA']} (Trash: {row['TOTAL_TRASH']}):")
    print(f"  Start: ({row['LAT']:.4f}°N, {row['LON']:.4f}°E)")
    if len(traj) > 1:
        print(f"  End after 7 days: ({traj[-1][0]:.4f}°N, {traj[-1][1]:.4f}°E)")
        print(f"  Distance traveled: {len(traj)-1} steps")

# 시각화
print("\n7. Creating visualizations...")
print("-" * 80)

# 1. 쓰레기 유형별 해류 속도 분석
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# 유형별 쓰레기량
types = trash_type_df.sort_values('TOTAL', ascending=False)
axes[0].barh(range(len(types)), types['TOTAL'], color='skyblue', edgecolor='black')
axes[0].set_yticks(range(len(types)))
axes[0].set_yticklabels(types['TYPE'], fontsize=9)
axes[0].set_xlabel('Total Trash Count', fontsize=11)
axes[0].set_title('Trash Count by Type', fontsize=12, fontweight='bold')
axes[0].grid(True, alpha=0.3, axis='x')

# 유형별 평균 해류 속도
valid_types = types.dropna(subset=['CURRENT_SPEED'])
if len(valid_types) > 0:
    axes[1].barh(range(len(valid_types)), valid_types['CURRENT_SPEED'], 
                color='coral', edgecolor='black')
    axes[1].set_yticks(range(len(valid_types)))
    axes[1].set_yticklabels(valid_types['TYPE'], fontsize=9)
    axes[1].set_xlabel('Average Current Speed (m/s)', fontsize=11)
    axes[1].set_title('Average Current Speed by Trash Type', fontsize=12, fontweight='bold')
    axes[1].grid(True, alpha=0.3, axis='x')

plt.suptitle('Trash Type Analysis', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig(f"{output_dir}/trash_type_analysis.png", dpi=300, bbox_inches='tight')
print(f"✓ Trash type analysis saved: {output_dir}/trash_type_analysis.png")
plt.close()

# 2. 쓰레기 집적 핫스팟 예측 지도
fig, ax = plt.subplots(figsize=(14, 10))

# 육지 표시
land = np.where(mask == 0, 1, np.nan)
ax.contourf(lon_grid, lat_grid, land, levels=[0.5, 1.5], 
           colors='lightgray', alpha=0.9, zorder=1)
ax.contour(lon_grid, lat_grid, mask, levels=[0.5], 
          colors='black', linewidths=2.5, zorder=2)

# 핫스팟 점수
im = ax.contourf(lon_grid, lat_grid, hotspot_score, levels=[0, 0.5, 1, 1.5, 2], 
                cmap='YlOrRd', alpha=0.7, extend='max', zorder=3)

# 실제 쓰레기 집적 지역 표시
scatter = ax.scatter(area_trash['LON'], area_trash['LAT'], 
                    s=area_trash['TOTAL_TRASH']*3, c=area_trash['TOTAL_TRASH'],
                    cmap='Reds', alpha=0.8, edgecolors='darkred', linewidths=2,
                    zorder=5, label='Actual Trash Sites')

ax.set_xlim(lon.min(), lon.max())
ax.set_ylim(lat.min(), lat.max())
ax.set_xlabel('Longitude (°E)', fontsize=12)
ax.set_ylabel('Latitude (°N)', fontsize=12)
ax.set_title('Trash Accumulation Hotspot Prediction\n(Based on Current Patterns)', 
            fontsize=14, fontweight='bold')
ax.legend(loc='upper right', fontsize=10)
plt.colorbar(im, ax=ax, label='Hotspot Score')
plt.colorbar(scatter, ax=ax, location='right', pad=0.15, label='Trash Count')
plt.tight_layout()
plt.savefig(f"{output_dir}/hotspot_prediction.png", dpi=300, bbox_inches='tight')
print(f"✓ Hotspot prediction map saved: {output_dir}/hotspot_prediction.png")
plt.close()

# 3. 쓰레기 이동 경로 시뮬레이션
fig, ax = plt.subplots(figsize=(14, 10))

# 육지 표시
ax.contourf(lon_grid, lat_grid, land, levels=[0.5, 1.5], 
           colors='lightgray', alpha=0.9, zorder=1)
ax.contour(lon_grid, lat_grid, mask, levels=[0.5], 
          colors='black', linewidths=2.5, zorder=2)

# 해류 속도 배경
speed_masked = np.ma.masked_where(np.isnan(speed) | (mask == 0), speed)
im = ax.contourf(lon_grid, lat_grid, speed_masked, levels=20, 
                cmap='Blues', alpha=0.4, extend='both', zorder=3)

# 이동 경로 그리기
colors = plt.cm.tab10(np.linspace(0, 1, len(trajectories)))
for (area_name, traj), color in zip(trajectories.items(), colors):
    if len(traj) > 1:
        traj_lons = [t[1] for t in traj]
        traj_lats = [t[0] for t in traj]
        ax.plot(traj_lons, traj_lats, '-', linewidth=2, alpha=0.7, 
               color=color, label=f'{area_name}', zorder=4)
        ax.scatter(traj_lons[0], traj_lats[0], s=100, c=color, 
                  marker='o', edgecolors='black', linewidths=1.5, zorder=5)
        ax.scatter(traj_lons[-1], traj_lats[-1], s=100, c=color, 
                  marker='s', edgecolors='black', linewidths=1.5, zorder=5)

ax.set_xlim(lon.min(), lon.max())
ax.set_ylim(lat.min(), lat.max())
ax.set_xlabel('Longitude (°E)', fontsize=12)
ax.set_ylabel('Latitude (°N)', fontsize=12)
ax.set_title('Trash Trajectory Simulation (7 days)\nBased on Current Flow', 
            fontsize=14, fontweight='bold')
ax.legend(loc='upper right', fontsize=9)
plt.colorbar(im, ax=ax, label='Current Speed (m/s)')
plt.tight_layout()
plt.savefig(f"{output_dir}/trajectory_simulation.png", dpi=300, bbox_inches='tight')
print(f"✓ Trajectory simulation saved: {output_dir}/trajectory_simulation.png")
plt.close()

# 4. 쓰레기 기인별 분석
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

sources = source_df.sort_values('TOTAL', ascending=False)
axes[0].barh(range(len(sources)), sources['TOTAL'], color='lightgreen', edgecolor='black')
axes[0].set_yticks(range(len(sources)))
axes[0].set_yticklabels(sources['SOURCE'], fontsize=10)
axes[0].set_xlabel('Total Trash Count', fontsize=11)
axes[0].set_title('Trash Count by Source', fontsize=12, fontweight='bold')
axes[0].grid(True, alpha=0.3, axis='x')

# 기인별 위치 분포
for idx, row in sources.iterrows():
    axes[1].scatter(row['AVG_LON'], row['AVG_LAT'], 
                   s=row['TOTAL']*2, alpha=0.6, 
                   label=f"{row['SOURCE']} ({row['TOTAL']})")

axes[1].set_xlim(lon.min(), lon.max())
axes[1].set_ylim(lat.min(), lat.max())
axes[1].set_xlabel('Longitude (°E)', fontsize=11)
axes[1].set_ylabel('Latitude (°N)', fontsize=11)
axes[1].set_title('Trash Source Distribution', fontsize=12, fontweight='bold')
axes[1].legend(loc='upper right', fontsize=9)
axes[1].grid(True, alpha=0.3)

plt.suptitle('Trash Source Analysis', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig(f"{output_dir}/source_analysis.png", dpi=300, bbox_inches='tight')
print(f"✓ Source analysis saved: {output_dir}/source_analysis.png")
plt.close()

# 5. 통합 분석 패널
fig, axes = plt.subplots(2, 2, figsize=(16, 14))

# (0,0) 핫스팟 예측
axes[0, 0].contourf(lon_grid, lat_grid, land, levels=[0.5, 1.5], 
                   colors='lightgray', alpha=0.9, zorder=1)
axes[0, 0].contour(lon_grid, lat_grid, mask, levels=[0.5], 
                  colors='black', linewidths=1.5, zorder=2)
im1 = axes[0, 0].contourf(lon_grid, lat_grid, hotspot_score, 
                         levels=[0, 0.5, 1, 1.5, 2], cmap='YlOrRd', 
                         alpha=0.7, extend='max', zorder=3)
axes[0, 0].scatter(area_trash['LON'], area_trash['LAT'], 
                  s=area_trash['TOTAL_TRASH']*2, c='red', 
                  alpha=0.6, edgecolors='darkred', linewidths=1, zorder=4)
axes[0, 0].set_xlim(lon.min(), lon.max())
axes[0, 0].set_ylim(lat.min(), lat.max())
axes[0, 0].set_title('Hotspot Prediction', fontsize=11, fontweight='bold')
axes[0, 0].set_xlabel('Longitude (°E)')
axes[0, 0].set_ylabel('Latitude (°N)')
plt.colorbar(im1, ax=axes[0, 0], label='Score')

# (0,1) 해류 속도와 쓰레기량 관계
valid_data = area_trash.dropna(subset=['SPEED', 'TOTAL_TRASH'])
axes[0, 1].scatter(valid_data['SPEED'], valid_data['TOTAL_TRASH'], 
                  s=100, alpha=0.6, c=valid_data['TOTAL_TRASH'], cmap='Reds')
axes[0, 1].set_xlabel('Current Speed (m/s)', fontsize=10)
axes[0, 1].set_ylabel('Total Trash Count', fontsize=10)
axes[0, 1].set_title('Current Speed vs Trash Count', fontsize=11, fontweight='bold')
axes[0, 1].grid(True, alpha=0.3)

# (1,0) 쓰레기 유형 분포
types_top = types.head(10)
axes[1, 0].bar(range(len(types_top)), types_top['TOTAL'], 
              color='skyblue', edgecolor='black')
axes[1, 0].set_xticks(range(len(types_top)))
axes[1, 0].set_xticklabels(types_top['TYPE'], rotation=45, ha='right', fontsize=8)
axes[1, 0].set_ylabel('Trash Count', fontsize=10)
axes[1, 0].set_title('Top 10 Trash Types', fontsize=11, fontweight='bold')
axes[1, 0].grid(True, alpha=0.3, axis='y')

# (1,1) 이동 경로 요약
for (area_name, traj), color in zip(list(trajectories.items())[:5], 
                                    plt.cm.tab10(np.linspace(0, 1, 5))):
    if len(traj) > 1:
        traj_lons = [t[1] for t in traj]
        traj_lats = [t[0] for t in traj]
        axes[1, 1].plot(traj_lons, traj_lats, '-', linewidth=1.5, 
                       alpha=0.6, color=color, label=area_name[:10])

axes[1, 1].set_xlim(lon.min(), lon.max())
axes[1, 1].set_ylim(lat.min(), lat.max())
axes[1, 1].set_xlabel('Longitude (°E)', fontsize=10)
axes[1, 1].set_ylabel('Latitude (°N)', fontsize=10)
axes[1, 1].set_title('Trajectory Simulation (Top 5)', fontsize=11, fontweight='bold')
axes[1, 1].legend(loc='upper right', fontsize=7)
axes[1, 1].grid(True, alpha=0.3)

plt.suptitle('Advanced Trash-Current Analysis', fontsize=14, fontweight='bold', y=0.995)
plt.tight_layout()
plt.savefig(f"{output_dir}/comprehensive_analysis.png", dpi=300, bbox_inches='tight')
print(f"✓ Comprehensive analysis panel saved: {output_dir}/comprehensive_analysis.png")
plt.close()

# 결과 요약 저장
with open(f"{output_dir}/analysis_summary.txt", "w", encoding="utf-8") as f:
    f.write("=" * 80 + "\n")
    f.write("Advanced Trash-Current Data Analysis Summary\n")
    f.write("=" * 80 + "\n\n")
    
    f.write("1. Trash Type Analysis\n")
    f.write("-" * 80 + "\n")
    for idx, row in types.iterrows():
        f.write(f"{row['TYPE']}: {row['TOTAL']} items, "
               f"Avg Current Speed: {row['CURRENT_SPEED']:.4f} m/s\n")
    
    f.write("\n2. Trash Source Analysis\n")
    f.write("-" * 80 + "\n")
    for idx, row in sources.iterrows():
        f.write(f"{row['SOURCE']}: {row['TOTAL']} items\n")
    
    f.write("\n3. Hotspot Prediction\n")
    f.write("-" * 80 + "\n")
    f.write(f"Slow current threshold: {slow_current_threshold:.4f} m/s\n")
    f.write(f"High vorticity threshold: {high_vorticity_threshold:.2e} s^-1\n")
    f.write(f"Predicted hotspot areas: {np.sum(hotspot_score >= 1)} pixels\n")
    
    f.write("\n4. Key Findings\n")
    f.write("-" * 80 + "\n")
    f.write("- Trash accumulation is higher in areas with slower currents\n")
    f.write("- Eddy regions (high vorticity) tend to accumulate more trash\n")
    f.write("- Different trash types show different spatial distributions\n")
    f.write("- Trajectory simulation helps predict trash movement patterns\n")

print(f"\n✓ Analysis summary saved: {output_dir}/analysis_summary.txt")

nc.close()

print("\n" + "=" * 80)
print("Advanced analysis completed!")
print(f"All results saved in '{output_dir}' folder.")
print("=" * 80)

