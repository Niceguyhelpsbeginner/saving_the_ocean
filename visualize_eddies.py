"""
환류(Eddy) 시각화 스크립트
소용돌이도(Vorticity)를 계산하여 환류 지점을 찾고 시각화
"""
from netCDF4 import Dataset
import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage
from scipy.ndimage import gaussian_filter
import os
from setup_korean_font import setup_korean_font

# 한글 폰트 설정
setup_korean_font()

# 파일 경로
file_path = "dataset/KHOA_SCU_L4_Z004_D01_U20251118_EastSea.nc"

print("=" * 80)
print("환류(Eddy) 시각화 분석")
print("=" * 80)

try:
    # NetCDF 파일 열기
    nc = Dataset(file_path, 'r')
    
    # 데이터 읽기
    lat = nc.variables['lat'][:]
    lon = nc.variables['lon'][:]
    u = nc.variables['u'][:]  # 동서 방향 해류 속도
    v = nc.variables['v'][:]  # 남북 방향 해류 속도
    ssh = nc.variables['ssh'][:]  # 해수면 높이
    mask = nc.variables['mask'][:]  # 육지/바다 마스크
    
    # 좌표 그리드 생성
    lon_grid, lat_grid = np.meshgrid(lon, lat)
    
    # 위도, 경도를 미터 단위로 변환 (대략적)
    # 1도 위도 ≈ 111,000 m
    # 1도 경도 ≈ 111,000 * cos(위도) m
    lat_rad = np.deg2rad(lat_grid)
    dlat_m = 111000.0  # 미터
    dlon_m = 111000.0 * np.cos(lat_rad)  # 위도에 따라 변함
    
    # 해상도 (도 단위)
    dlat_deg = lat[1] - lat[0] if len(lat) > 1 else 0.25
    dlon_deg = lon[1] - lon[0] if len(lon) > 1 else 0.25
    
    # 미터 단위로 변환
    dx = dlon_deg * dlon_m  # 경도 방향 격자 간격 (미터)
    dy = dlat_deg * dlat_m  # 위도 방향 격자 간격 (미터)
    
    # 데이터 마스킹 (육지 제거)
    u_masked = np.where((mask == 1) & ~np.isnan(u), u, np.nan)
    v_masked = np.where((mask == 1) & ~np.isnan(v), v, np.nan)
    
    # Vorticity (소용돌이도) 계산: ζ = ∂v/∂x - ∂u/∂y
    # numpy.gradient를 사용하여 편미분 계산
    dv_dx = np.gradient(v_masked, axis=1) / dx  # ∂v/∂x
    du_dy = np.gradient(u_masked, axis=0) / dy  # ∂u/∂y
    
    # Vorticity 계산
    vorticity = dv_dx - du_dy
    
    # Coriolis 파라미터 (f = 2Ωsin(φ))
    omega = 7.2921e-5  # 지구 자전 각속도 (rad/s)
    f = 2 * omega * np.sin(np.deg2rad(lat_grid))  # Coriolis 파라미터
    
    # 상대적 소용돌이도 (Relative Vorticity)
    relative_vorticity = vorticity
    
    # 절대 소용돌이도 (Absolute Vorticity) = 상대 소용돌이도 + Coriolis 파라미터
    absolute_vorticity = relative_vorticity + f
    
    # 정규화된 상대 소용돌이도 (Normalized Relative Vorticity)
    # ζ/f로 정규화하여 환류 강도를 측정
    normalized_vorticity = np.where(f != 0, relative_vorticity / np.abs(f), np.nan)
    
    # Vorticity 마스킹
    vorticity_masked = np.ma.masked_where(
        np.isnan(vorticity) | (mask == 0) | np.isinf(vorticity), 
        vorticity
    )
    
    # 출력 디렉토리 생성
    output_dir = "analysis_output"
    os.makedirs(output_dir, exist_ok=True)
    
    print("\n1. Vorticity 통계")
    print("-" * 80)
    valid_vort = vorticity_masked[~vorticity_masked.mask]
    if len(valid_vort) > 0:
        print(f"상대 소용돌이도 범위: {np.nanmin(valid_vort):.2e} ~ {np.nanmax(valid_vort):.2e} s^-1")
        print(f"평균: {np.nanmean(valid_vort):.2e} s^-1")
        print(f"표준편차: {np.nanstd(valid_vort):.2e} s^-1")
        
        # 양수/음수 vorticity 개수 (시계방향/반시계방향 환류)
        positive_vort = np.sum((valid_vort > 0) & ~np.isnan(valid_vort))
        negative_vort = np.sum((valid_vort < 0) & ~np.isnan(valid_vort))
        print(f"\n시계방향 환류 (양수 vorticity): {positive_vort} 개")
        print(f"반시계방향 환류 (음수 vorticity): {negative_vort} 개")
    
    # 환류 중심 찾기 (국소적 최대/최소 vorticity)
    print("\n2. 환류 중심 탐지 중...")
    print("-" * 80)
    
    # Vorticity를 스무딩하여 노이즈 제거
    vorticity_smooth = np.where(~np.isnan(vorticity_masked), vorticity_masked, 0)
    vorticity_smooth = gaussian_filter(vorticity_smooth, sigma=1.0)
    vorticity_smooth = np.ma.masked_where(vorticity_masked.mask, vorticity_smooth)
    
    # 국소적 최대/최소 찾기
    threshold = np.nanstd(valid_vort) * 1.5  # 표준편차의 1.5배를 임계값으로
    
    # 양수 vorticity의 국소적 최대값 (시계방향 환류)
    from scipy.ndimage import maximum_filter, minimum_filter
    local_max = maximum_filter(vorticity_smooth, size=5)
    positive_eddies = (vorticity_smooth == local_max) & (vorticity_smooth > threshold)
    
    # 음수 vorticity의 국소적 최소값 (반시계방향 환류)
    local_min = minimum_filter(vorticity_smooth, size=5)
    negative_eddies = (vorticity_smooth == local_min) & (vorticity_smooth < -threshold)
    
    # 환류 중심 좌표 추출
    positive_indices = np.where(positive_eddies)
    negative_indices = np.where(negative_eddies)
    
    print(f"탐지된 시계방향 환류 중심: {len(positive_indices[0])} 개")
    print(f"탐지된 반시계방향 환류 중심: {len(negative_indices[0])} 개")
    
    # 시각화
    print("\n3. 시각화 생성 중...")
    print("-" * 80)
    
    # 1. Vorticity 지도
    fig, ax = plt.subplots(figsize=(14, 10))
    
    # Vorticity 컨투어
    levels = np.linspace(np.nanpercentile(valid_vort, 5), 
                        np.nanpercentile(valid_vort, 95), 30)
    im = ax.contourf(lon_grid, lat_grid, vorticity_masked, 
                     levels=levels, cmap='RdBu_r', extend='both')
    
    # 육지 경계선
    ax.contour(lon_grid, lat_grid, mask, levels=[0.5], colors='black', linewidths=1.5)
    
    # 환류 중심 표시
    if len(positive_indices[0]) > 0:
        ax.scatter(lon_grid[positive_indices], lat_grid[positive_indices], 
                  c='red', s=100, marker='o', edgecolors='darkred', 
                  linewidths=2, label=f'시계방향 환류 ({len(positive_indices[0])}개)', zorder=5)
    
    if len(negative_indices[0]) > 0:
        ax.scatter(lon_grid[negative_indices], lat_grid[negative_indices], 
                  c='blue', s=100, marker='s', edgecolors='darkblue', 
                  linewidths=2, label=f'반시계방향 환류 ({len(negative_indices[0])}개)', zorder=5)
    
    # 해류 벡터 (서브샘플링)
    skip = 4
    u_sub = u_masked[::skip, ::skip]
    v_sub = v_masked[::skip, ::skip]
    lon_sub = lon_grid[::skip, ::skip]
    lat_sub = lat_grid[::skip, ::skip]
    valid = ~np.isnan(u_sub) & ~np.isnan(v_sub)
    
    ax.quiver(lon_sub[valid], lat_sub[valid], u_sub[valid], v_sub[valid],
              scale=0.3, width=0.003, color='white', alpha=0.6, zorder=3)
    
    ax.set_xlabel('Longitude (°E)', fontsize=12)
    ax.set_ylabel('Latitude (°N)', fontsize=12)
    ax.set_title('Relative Vorticity and Eddy Detection - 2025-11-18', 
                fontsize=14, fontweight='bold')
    ax.legend(loc='upper right', fontsize=10)
    plt.colorbar(im, ax=ax, label='Relative Vorticity (s⁻¹)')
    plt.tight_layout()
    plt.savefig(f"{output_dir}/eddy_vorticity.png", dpi=300, bbox_inches='tight')
    print(f"✓ Vorticity 및 환류 중심 지도 저장: {output_dir}/eddy_vorticity.png")
    plt.close()
    
    # 2. SSH와 환류 중심 비교
    fig, ax = plt.subplots(figsize=(14, 10))
    
    # SSH 컨투어
    ssh_levels = np.linspace(np.nanmin(ssh), np.nanmax(ssh), 20)
    im = ax.contourf(lon_grid, lat_grid, ssh, levels=ssh_levels, 
                     cmap='viridis', extend='both', alpha=0.7)
    ax.contour(lon_grid, lat_grid, ssh, levels=ssh_levels, 
              colors='gray', linewidths=0.5, alpha=0.5)
    
    # 육지 경계선
    ax.contour(lon_grid, lat_grid, mask, levels=[0.5], colors='black', linewidths=1.5)
    
    # 환류 중심 표시
    if len(positive_indices[0]) > 0:
        ax.scatter(lon_grid[positive_indices], lat_grid[positive_indices], 
                  c='red', s=150, marker='o', edgecolors='darkred', 
                  linewidths=2, label=f'Clockwise Eddy ({len(positive_indices[0])})', zorder=5)
    
    if len(negative_indices[0]) > 0:
        ax.scatter(lon_grid[negative_indices], lat_grid[negative_indices], 
                  c='blue', s=150, marker='s', edgecolors='darkblue', 
                  linewidths=2, label=f'Counter-clockwise Eddy ({len(negative_indices[0])})', zorder=5)
    
    ax.set_xlabel('Longitude (°E)', fontsize=12)
    ax.set_ylabel('Latitude (°N)', fontsize=12)
    ax.set_title('Sea Surface Height (SSH) with Eddy Centers - 2025-11-18', 
                fontsize=14, fontweight='bold')
    ax.legend(loc='upper right', fontsize=10)
    plt.colorbar(im, ax=ax, label='SSH (m)')
    plt.tight_layout()
    plt.savefig(f"{output_dir}/eddy_ssh.png", dpi=300, bbox_inches='tight')
    print(f"✓ SSH와 환류 중심 지도 저장: {output_dir}/eddy_ssh.png")
    plt.close()
    
    # 3. 통합 패널: Vorticity, SSH, 속도 벡터
    fig, axes = plt.subplots(2, 2, figsize=(16, 14))
    
    # Vorticity
    im1 = axes[0, 0].contourf(lon_grid, lat_grid, vorticity_masked, 
                             levels=levels, cmap='RdBu_r', extend='both')
    axes[0, 0].contour(lon_grid, lat_grid, mask, levels=[0.5], colors='black', linewidths=0.5)
    if len(positive_indices[0]) > 0:
        axes[0, 0].scatter(lon_grid[positive_indices], lat_grid[positive_indices], 
                          c='red', s=50, marker='o', edgecolors='darkred', linewidths=1)
    if len(negative_indices[0]) > 0:
        axes[0, 0].scatter(lon_grid[negative_indices], lat_grid[negative_indices], 
                          c='blue', s=50, marker='s', edgecolors='darkblue', linewidths=1)
    axes[0, 0].set_title('Relative Vorticity', fontsize=12, fontweight='bold')
    axes[0, 0].set_xlabel('Longitude (°E)')
    axes[0, 0].set_ylabel('Latitude (°N)')
    plt.colorbar(im1, ax=axes[0, 0], label='s⁻¹')
    
    # SSH
    im2 = axes[0, 1].contourf(lon_grid, lat_grid, ssh, levels=ssh_levels, 
                             cmap='viridis', extend='both')
    axes[0, 1].contour(lon_grid, lat_grid, mask, levels=[0.5], colors='black', linewidths=0.5)
    if len(positive_indices[0]) > 0:
        axes[0, 1].scatter(lon_grid[positive_indices], lat_grid[positive_indices], 
                          c='red', s=50, marker='o', edgecolors='darkred', linewidths=1)
    if len(negative_indices[0]) > 0:
        axes[0, 1].scatter(lon_grid[negative_indices], lat_grid[negative_indices], 
                          c='blue', s=50, marker='s', edgecolors='darkblue', linewidths=1)
    axes[0, 1].set_title('Sea Surface Height', fontsize=12, fontweight='bold')
    axes[0, 1].set_xlabel('Longitude (°E)')
    axes[0, 1].set_ylabel('Latitude (°N)')
    plt.colorbar(im2, ax=axes[0, 1], label='m')
    
    # 속도 벡터 필드
    speed = np.sqrt(u_masked**2 + v_masked**2)
    speed_masked = np.ma.masked_where(np.isnan(speed) | (mask == 0), speed)
    im3 = axes[1, 0].contourf(lon_grid, lat_grid, speed_masked, levels=20, 
                             cmap='jet', extend='both', alpha=0.7)
    axes[1, 0].contour(lon_grid, lat_grid, mask, levels=[0.5], colors='black', linewidths=0.5)
    skip2 = 3
    u_sub2 = u_masked[::skip2, ::skip2]
    v_sub2 = v_masked[::skip2, ::skip2]
    lon_sub2 = lon_grid[::skip2, ::skip2]
    lat_sub2 = lat_grid[::skip2, ::skip2]
    valid2 = ~np.isnan(u_sub2) & ~np.isnan(v_sub2)
    axes[1, 0].quiver(lon_sub2[valid2], lat_sub2[valid2], u_sub2[valid2], v_sub2[valid2],
                     scale=0.3, width=0.003, color='white', alpha=0.8)
    if len(positive_indices[0]) > 0:
        axes[1, 0].scatter(lon_grid[positive_indices], lat_grid[positive_indices], 
                          c='red', s=50, marker='o', edgecolors='darkred', linewidths=1)
    if len(negative_indices[0]) > 0:
        axes[1, 0].scatter(lon_grid[negative_indices], lat_grid[negative_indices], 
                          c='blue', s=50, marker='s', edgecolors='darkblue', linewidths=1)
    axes[1, 0].set_title('Current Speed & Vectors', fontsize=12, fontweight='bold')
    axes[1, 0].set_xlabel('Longitude (°E)')
    axes[1, 0].set_ylabel('Latitude (°N)')
    plt.colorbar(im3, ax=axes[1, 0], label='m/s')
    
    # 정규화된 Vorticity
    norm_vort_masked = np.ma.masked_where(
        np.isnan(normalized_vorticity) | (mask == 0) | np.isinf(normalized_vorticity),
        normalized_vorticity
    )
    norm_levels = np.linspace(np.nanpercentile(norm_vort_masked[~norm_vort_masked.mask], 5),
                             np.nanpercentile(norm_vort_masked[~norm_vort_masked.mask], 95), 30)
    im4 = axes[1, 1].contourf(lon_grid, lat_grid, norm_vort_masked, 
                             levels=norm_levels, cmap='RdBu_r', extend='both')
    axes[1, 1].contour(lon_grid, lat_grid, mask, levels=[0.5], colors='black', linewidths=0.5)
    if len(positive_indices[0]) > 0:
        axes[1, 1].scatter(lon_grid[positive_indices], lat_grid[positive_indices], 
                          c='red', s=50, marker='o', edgecolors='darkred', linewidths=1)
    if len(negative_indices[0]) > 0:
        axes[1, 1].scatter(lon_grid[negative_indices], lat_grid[negative_indices], 
                          c='blue', s=50, marker='s', edgecolors='darkblue', linewidths=1)
    axes[1, 1].set_title('Normalized Vorticity (ζ/f)', fontsize=12, fontweight='bold')
    axes[1, 1].set_xlabel('Longitude (°E)')
    axes[1, 1].set_ylabel('Latitude (°N)')
    plt.colorbar(im4, ax=axes[1, 1], label='ζ/f')
    
    plt.suptitle('Eddy Detection Analysis - 2025-11-18', fontsize=16, fontweight='bold', y=0.995)
    plt.tight_layout()
    plt.savefig(f"{output_dir}/eddy_comprehensive.png", dpi=300, bbox_inches='tight')
    print(f"✓ 통합 환류 분석 패널 저장: {output_dir}/eddy_comprehensive.png")
    plt.close()
    
    # 환류 중심 정보 저장
    with open(f"{output_dir}/eddy_locations.txt", "w", encoding="utf-8") as f:
        f.write("=" * 80 + "\n")
        f.write("환류 중심 위치 정보\n")
        f.write("=" * 80 + "\n\n")
        f.write(f"분석 날짜: 2025-11-18\n")
        f.write(f"탐지 임계값: {threshold:.2e} s⁻¹\n\n")
        
        f.write("시계방향 환류 (양수 vorticity) 중심:\n")
        f.write("-" * 80 + "\n")
        if len(positive_indices[0]) > 0:
            for i in range(len(positive_indices[0])):
                idx_lat, idx_lon = positive_indices[0][i], positive_indices[1][i]
                eddy_lat = lat_grid[idx_lat, idx_lon]
                eddy_lon = lon_grid[idx_lat, idx_lon]
                eddy_vort = vorticity_smooth[idx_lat, idx_lon]
                eddy_ssh = ssh[idx_lat, idx_lon]
                f.write(f"{i+1}. 위치: ({eddy_lat:.4f}°N, {eddy_lon:.4f}°E), "
                       f"Vorticity: {eddy_vort:.2e} s⁻¹, SSH: {eddy_ssh:.4f} m\n")
        else:
            f.write("탐지된 환류 없음\n")
        
        f.write("\n반시계방향 환류 (음수 vorticity) 중심:\n")
        f.write("-" * 80 + "\n")
        if len(negative_indices[0]) > 0:
            for i in range(len(negative_indices[0])):
                idx_lat, idx_lon = negative_indices[0][i], negative_indices[1][i]
                eddy_lat = lat_grid[idx_lat, idx_lon]
                eddy_lon = lon_grid[idx_lat, idx_lon]
                eddy_vort = vorticity_smooth[idx_lat, idx_lon]
                eddy_ssh = ssh[idx_lat, idx_lon]
                f.write(f"{i+1}. 위치: ({eddy_lat:.4f}°N, {eddy_lon:.4f}°E), "
                       f"Vorticity: {eddy_vort:.2e} s⁻¹, SSH: {eddy_ssh:.4f} m\n")
        else:
            f.write("탐지된 환류 없음\n")
    
    print(f"✓ 환류 위치 정보 저장: {output_dir}/eddy_locations.txt")
    
    nc.close()
    
    print("\n" + "=" * 80)
    print("환류 시각화 완료!")
    print(f"모든 결과는 '{output_dir}' 폴더에 저장되었습니다.")
    print("=" * 80)
    
except Exception as e:
    print(f"\n오류 발생: {e}")
    import traceback
    traceback.print_exc()

