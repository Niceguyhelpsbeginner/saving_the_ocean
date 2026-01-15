"""
동해 해류 데이터와 해양쓰레기 데이터 결합 분석
"""
import pandas as pd
import numpy as np
from netCDF4 import Dataset
import matplotlib.pyplot as plt
from scipy import ndimage
from scipy.ndimage import gaussian_filter
import os
from setup_korean_font import setup_korean_font

# 한글 폰트 설정
setup_korean_font()

print("=" * 80)
print("동해 해류 데이터와 해양쓰레기 데이터 결합 분석")
print("=" * 80)

# 1. 해양쓰레기 데이터 로드
trash_file = "dataset/trash/eastsea_vessel_threat_coastallitter-2022.csv"
current_file = "dataset/KHOA_SCU_L4_Z004_D01_U20251118_EastSea.nc"

print("\n1. 데이터 로드 중...")
print("-" * 80)

# 해양쓰레기 데이터
trash_df = pd.read_csv(trash_file, encoding='utf-8')
print(f"해양쓰레기 데이터: {len(trash_df)} 행")

# 해류 데이터
nc = Dataset(current_file, 'r')
lat = nc.variables['lat'][:]
lon = nc.variables['lon'][:]
u = nc.variables['u'][:]
v = nc.variables['v'][:]
ssh = nc.variables['ssh'][:]
mask = nc.variables['mask'][:]

lon_grid, lat_grid = np.meshgrid(lon, lat)

print(f"해류 데이터 격자: {len(lat)} × {len(lon)}")
print(f"해류 데이터 범위: {lat.min():.2f}°N ~ {lat.max():.2f}°N, {lon.min():.2f}°E ~ {lon.max():.2f}°E")

# 2. 해양쓰레기 데이터 전처리
print("\n2. 해양쓰레기 데이터 전처리 중...")
print("-" * 80)

# 위치 정보 정리 (시작점과 끝점의 평균 사용)
trash_df['LAT'] = (trash_df['STR_LA'] + trash_df['END_LA']) / 2
trash_df['LON'] = (trash_df['STR_LO'] + trash_df['END_LO']) / 2

# 쓰레기 총량 계산 (지역별 집계)
trash_summary = trash_df.groupby(['INVS_AREA_NM', 'LAT', 'LON']).agg({
    'IEM_CNT': 'sum',
    'METER_PER_IEM_CNT': 'sum'
}).reset_index()

# 지역별 총 쓰레기량
area_trash = trash_df.groupby('INVS_AREA_NM').agg({
    'IEM_CNT': 'sum',
    'LAT': 'mean',
    'LON': 'mean'
}).reset_index()
area_trash.columns = ['AREA', 'TOTAL_TRASH', 'LAT', 'LON']

print(f"조사 지역 수: {len(area_trash)}")
print(f"\n지역별 쓰레기량 상위 10개:")
print(area_trash.nlargest(10, 'TOTAL_TRASH')[['AREA', 'TOTAL_TRASH', 'LAT', 'LON']])

# 3. 해류 데이터에서 쓰레기 위치의 해류 정보 추출
print("\n3. 쓰레기 위치의 해류 정보 추출 중...")
print("-" * 80)

def get_current_at_location(lat_val, lon_val, lat_grid, lon_grid, data):
    """특정 위치의 해류 데이터 추출 (보간 사용)"""
    from scipy.interpolate import griddata
    
    # 데이터 범위 확인
    if lat_val < lat.min() or lat_val > lat.max() or lon_val < lon.min() or lon_val > lon.max():
        return np.nan
    
    # 유효한 데이터 포인트만 사용
    valid_mask = ~np.isnan(data)
    if np.sum(valid_mask) == 0:
        return np.nan
    
    valid_lat = lat_grid[valid_mask]
    valid_lon = lon_grid[valid_mask]
    valid_data = data[valid_mask]
    
    # 보간 수행
    try:
        interpolated = griddata((valid_lon.flatten(), valid_lat.flatten()), 
                               valid_data.flatten(), 
                               (lon_val, lat_val), 
                               method='nearest')
        return interpolated if not np.isnan(interpolated) else np.nan
    except:
        # 보간 실패 시 가장 가까운 점 사용
        lat_idx = np.argmin(np.abs(lat - lat_val))
        lon_idx = np.argmin(np.abs(lon - lon_val))
        if 0 <= lat_idx < len(lat) and 0 <= lon_idx < len(lon):
            return data[lat_idx, lon_idx]
        return np.nan

# 각 쓰레기 위치에서 해류 정보 추출
area_trash['U'] = area_trash.apply(
    lambda x: get_current_at_location(x['LAT'], x['LON'], lat_grid, lon_grid, u), axis=1
)
area_trash['V'] = area_trash.apply(
    lambda x: get_current_at_location(x['LAT'], x['LON'], lat_grid, lon_grid, v), axis=1
)

# numpy 배열로 변환하여 계산
u_vals = np.array(area_trash['U'].tolist())
v_vals = np.array(area_trash['V'].tolist())
area_trash['SPEED'] = np.sqrt(u_vals**2 + v_vals**2)

area_trash['SSH'] = area_trash.apply(
    lambda x: get_current_at_location(x['LAT'], x['LON'], lat_grid, lon_grid, ssh), axis=1
)

# 해류 방향 계산 (도 단위)
area_trash['DIRECTION'] = np.degrees(np.arctan2(v_vals, u_vals))

print(f"\n쓰레기 위치의 해류 정보:")
print(area_trash[['AREA', 'TOTAL_TRASH', 'SPEED', 'DIRECTION', 'SSH']].head(10))

# 4. Vorticity 계산 (환류 분석)
print("\n4. 환류 분석 중...")
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

dv_dx = np.gradient(v_masked, axis=1) / dx
du_dy = np.gradient(u_masked, axis=0) / dy
vorticity = dv_dx - du_dy

# 각 쓰레기 위치의 vorticity 추출
area_trash['VORTICITY'] = area_trash.apply(
    lambda x: get_current_at_location(x['LAT'], x['LON'], lat_grid, lon_grid, vorticity), axis=1
)

print(f"\n쓰레기 위치의 Vorticity 정보:")
print(area_trash[['AREA', 'TOTAL_TRASH', 'VORTICITY']].head(10))

# 5. 상관관계 분석
print("\n5. 상관관계 분석 중...")
print("-" * 80)

# 유효한 데이터만 사용
valid_data = area_trash.dropna(subset=['SPEED', 'VORTICITY', 'SSH', 'TOTAL_TRASH'])

if len(valid_data) > 0:
    corr_speed = valid_data['TOTAL_TRASH'].corr(valid_data['SPEED'])
    corr_vorticity = valid_data['TOTAL_TRASH'].corr(np.abs(valid_data['VORTICITY']))
    corr_ssh = valid_data['TOTAL_TRASH'].corr(valid_data['SSH'])
    
    print(f"쓰레기량 vs 해류 속도 상관계수: {corr_speed:.3f}")
    print(f"쓰레기량 vs Vorticity 절대값 상관계수: {corr_vorticity:.3f}")
    print(f"쓰레기량 vs 해수면 높이 상관계수: {corr_ssh:.3f}")

# 6. 시각화
print("\n6. 시각화 생성 중...")
print("-" * 80)

output_dir = "analysis_output/current_trash_analysis"
os.makedirs(output_dir, exist_ok=True)

# 1. 해류 벡터와 쓰레기 위치
fig, ax = plt.subplots(figsize=(14, 10))

# 해류 속도 배경
speed = np.sqrt(u**2 + v**2)
speed_masked = np.ma.masked_where(np.isnan(speed) | (mask == 0), speed)
im = ax.contourf(lon_grid, lat_grid, speed_masked, levels=20, cmap='Blues', alpha=0.6, extend='both')

# 해류 벡터 (서브샘플링)
skip = 5
u_sub = u[::skip, ::skip]
v_sub = v[::skip, ::skip]
lon_sub = lon_grid[::skip, ::skip]
lat_sub = lat_grid[::skip, ::skip]
valid = ~np.isnan(u_sub) & ~np.isnan(v_sub) & (mask[::skip, ::skip] == 1)
ax.quiver(lon_sub[valid], lat_sub[valid], u_sub[valid], v_sub[valid],
          scale=0.3, width=0.003, color='white', alpha=0.7, zorder=2)

# 육지 경계선
ax.contour(lon_grid, lat_grid, mask, levels=[0.5], colors='black', linewidths=1)

# 쓰레기 위치 (크기는 쓰레기량에 비례)
scatter = ax.scatter(area_trash['LON'], area_trash['LAT'], 
                    s=area_trash['TOTAL_TRASH']*3, c=area_trash['TOTAL_TRASH'],
                    cmap='Reds', alpha=0.7, edgecolors='darkred', linewidths=2,
                    zorder=5, label='Trash Collection Sites')

# 지역명 표시
for idx, row in area_trash.iterrows():
    ax.annotate(row['AREA'], (row['LON'], row['LAT']), 
               fontsize=8, alpha=0.8, zorder=6)

ax.set_xlim(lon.min(), lon.max())
ax.set_ylim(lat.min(), lat.max())
ax.set_xlabel('Longitude (°E)', fontsize=12)
ax.set_ylabel('Latitude (°N)', fontsize=12)
ax.set_title('Current Flow and Trash Collection Sites - East Sea', 
            fontsize=14, fontweight='bold')
ax.legend(loc='upper right', fontsize=10)
plt.colorbar(scatter, ax=ax, label='Trash Count')
plt.tight_layout()
plt.savefig(f"{output_dir}/current_trash_locations.png", dpi=300, bbox_inches='tight')
print(f"✓ 해류와 쓰레기 위치 지도 저장: {output_dir}/current_trash_locations.png")
plt.close()

# 2. Vorticity와 쓰레기량 관계
fig, ax = plt.subplots(figsize=(14, 10))

vorticity_masked = np.ma.masked_where(np.isnan(vorticity) | (mask == 0), vorticity)
valid_vort = vorticity_masked[~vorticity_masked.mask]
levels = np.linspace(np.nanpercentile(valid_vort, 5), 
                    np.nanpercentile(valid_vort, 95), 30)

im = ax.contourf(lon_grid, lat_grid, vorticity_masked, levels=levels, 
                cmap='RdBu_r', extend='both', alpha=0.7)
ax.contour(lon_grid, lat_grid, mask, levels=[0.5], colors='black', linewidths=1)

# 쓰레기 위치
scatter = ax.scatter(area_trash['LON'], area_trash['LAT'], 
                    s=area_trash['TOTAL_TRASH']*3, c=area_trash['TOTAL_TRASH'],
                    cmap='YlOrRd', alpha=0.8, edgecolors='black', linewidths=1.5,
                    zorder=5)

ax.set_xlim(lon.min(), lon.max())
ax.set_ylim(lat.min(), lat.max())
ax.set_xlabel('Longitude (°E)', fontsize=12)
ax.set_ylabel('Latitude (°N)', fontsize=12)
ax.set_title('Vorticity (Eddy) and Trash Collection Sites', 
            fontsize=14, fontweight='bold')
plt.colorbar(im, ax=ax, label='Relative Vorticity (s⁻¹)')
cbar2 = plt.colorbar(scatter, ax=ax, location='right', pad=0.15)
cbar2.set_label('Trash Count', rotation=270, labelpad=20)
plt.tight_layout()
plt.savefig(f"{output_dir}/vorticity_trash.png", dpi=300, bbox_inches='tight')
print(f"✓ Vorticity와 쓰레기 위치 지도 저장: {output_dir}/vorticity_trash.png")
plt.close()

# 3. 상관관계 분석 그래프
if len(valid_data) > 0:
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    
    # 쓰레기량 vs 해류 속도
    axes[0, 0].scatter(valid_data['SPEED'], valid_data['TOTAL_TRASH'], 
                      alpha=0.6, s=100, c=valid_data['TOTAL_TRASH'], cmap='Reds')
    axes[0, 0].set_xlabel('Current Speed (m/s)', fontsize=11)
    axes[0, 0].set_ylabel('Total Trash Count', fontsize=11)
    axes[0, 0].set_title(f'Trash vs Current Speed\n(correlation: {corr_speed:.3f})', fontsize=12)
    axes[0, 0].grid(True, alpha=0.3)
    
    # 쓰레기량 vs Vorticity
    axes[0, 1].scatter(np.abs(valid_data['VORTICITY']), valid_data['TOTAL_TRASH'],
                      alpha=0.6, s=100, c=valid_data['TOTAL_TRASH'], cmap='Reds')
    axes[0, 1].set_xlabel('|Vorticity| (s⁻¹)', fontsize=11)
    axes[0, 1].set_ylabel('Total Trash Count', fontsize=11)
    axes[0, 1].set_title(f'Trash vs Vorticity\n(correlation: {corr_vorticity:.3f})', fontsize=12)
    axes[0, 1].grid(True, alpha=0.3)
    
    # 쓰레기량 vs SSH
    axes[1, 0].scatter(valid_data['SSH'], valid_data['TOTAL_TRASH'],
                      alpha=0.6, s=100, c=valid_data['TOTAL_TRASH'], cmap='Reds')
    axes[1, 0].set_xlabel('Sea Surface Height (m)', fontsize=11)
    axes[1, 0].set_ylabel('Total Trash Count', fontsize=11)
    axes[1, 0].set_title(f'Trash vs SSH\n(correlation: {corr_ssh:.3f})', fontsize=12)
    axes[1, 0].grid(True, alpha=0.3)
    
    # 쓰레기량 분포
    axes[1, 1].hist(valid_data['TOTAL_TRASH'], bins=20, color='skyblue', edgecolor='black', alpha=0.7)
    axes[1, 1].set_xlabel('Total Trash Count', fontsize=11)
    axes[1, 1].set_ylabel('Frequency', fontsize=11)
    axes[1, 1].set_title('Trash Count Distribution', fontsize=12)
    axes[1, 1].grid(True, alpha=0.3, axis='y')
    
    plt.suptitle('Correlation Analysis: Trash vs Ocean Current', 
                fontsize=14, fontweight='bold', y=0.995)
    plt.tight_layout()
    plt.savefig(f"{output_dir}/correlation_analysis.png", dpi=300, bbox_inches='tight')
    print(f"✓ 상관관계 분석 그래프 저장: {output_dir}/correlation_analysis.png")
    plt.close()

# 7. 결과 요약 저장
with open(f"{output_dir}/analysis_summary.txt", "w", encoding="utf-8") as f:
    f.write("=" * 80 + "\n")
    f.write("동해 해류 데이터와 해양쓰레기 데이터 결합 분석 결과\n")
    f.write("=" * 80 + "\n\n")
    
    f.write("1. 데이터 개요\n")
    f.write("-" * 80 + "\n")
    f.write(f"해양쓰레기 조사 지역 수: {len(area_trash)}\n")
    f.write(f"총 쓰레기 항목 수: {len(trash_df)}\n")
    f.write(f"해류 데이터 격자: {len(lat)} × {len(lon)}\n")
    f.write(f"해류 데이터 범위: {lat.min():.2f}°N ~ {lat.max():.2f}°N, {lon.min():.2f}°E ~ {lon.max():.2f}°E\n\n")
    
    f.write("2. 지역별 쓰레기량 상위 10개\n")
    f.write("-" * 80 + "\n")
    top_trash = area_trash.nlargest(10, 'TOTAL_TRASH')
    for idx, row in top_trash.iterrows():
        f.write(f"{row['AREA']}: {row['TOTAL_TRASH']}개 "
               f"(위치: {row['LAT']:.4f}°N, {row['LON']:.4f}°E)\n")
    
    if len(valid_data) > 0:
        f.write("\n3. 상관관계 분석\n")
        f.write("-" * 80 + "\n")
        f.write(f"쓰레기량 vs 해류 속도 상관계수: {corr_speed:.3f}\n")
        f.write(f"쓰레기량 vs Vorticity 절대값 상관계수: {corr_vorticity:.3f}\n")
        f.write(f"쓰레기량 vs 해수면 높이 상관계수: {corr_ssh:.3f}\n\n")
        
        f.write("4. 해류 특성별 쓰레기량 분석\n")
        f.write("-" * 80 + "\n")
        
        # 해류 속도 구간별 분석
        valid_data['SPEED_CAT'] = pd.cut(valid_data['SPEED'], bins=3, labels=['Low', 'Medium', 'High'])
        speed_analysis = valid_data.groupby('SPEED_CAT')['TOTAL_TRASH'].agg(['mean', 'std', 'count'])
        f.write("\n해류 속도 구간별 평균 쓰레기량:\n")
        for cat in speed_analysis.index:
            f.write(f"  {cat}: {speed_analysis.loc[cat, 'mean']:.1f}개 "
                   f"(표준편차: {speed_analysis.loc[cat, 'std']:.1f}, "
                   f"지역 수: {speed_analysis.loc[cat, 'count']})\n")
        
        # Vorticity 구간별 분석
        valid_data['VORT_CAT'] = pd.cut(np.abs(valid_data['VORTICITY']), 
                                       bins=3, labels=['Low', 'Medium', 'High'])
        vort_analysis = valid_data.groupby('VORT_CAT')['TOTAL_TRASH'].agg(['mean', 'std', 'count'])
        f.write("\nVorticity 구간별 평균 쓰레기량:\n")
        for cat in vort_analysis.index:
            f.write(f"  {cat}: {vort_analysis.loc[cat, 'mean']:.1f}개 "
                   f"(표준편차: {vort_analysis.loc[cat, 'std']:.1f}, "
                   f"지역 수: {vort_analysis.loc[cat, 'count']})\n")
    
    f.write("\n5. 주요 발견사항\n")
    f.write("-" * 80 + "\n")
    f.write("- 해류 데이터와 쓰레기 데이터의 시간대가 다름 (해류: 2025-11-18, 쓰레기: 2022)\n")
    f.write("- 쓰레기 집적 지역의 해류 패턴을 분석하여 쓰레기 이동 경로 예측 가능\n")
    f.write("- 환류가 발생하는 지역에서 쓰레기가 집적될 가능성 확인 필요\n")
    f.write("- 해류 속도가 느린 지역에서 쓰레기가 더 많이 집적될 수 있음\n")

print(f"\n✓ 분석 결과 요약 저장: {output_dir}/analysis_summary.txt")

nc.close()

print("\n" + "=" * 80)
print("분석 완료!")
print(f"모든 결과는 '{output_dir}' 폴더에 저장되었습니다.")
print("=" * 80)

