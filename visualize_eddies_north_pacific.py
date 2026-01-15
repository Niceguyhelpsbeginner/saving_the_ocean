"""
북태평양 환류(Eddy) 시각화 및 분석 스크립트
환류가 생기기 쉬운 지점을 분석
"""
from netCDF4 import Dataset
import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage
from scipy.ndimage import gaussian_filter, maximum_filter, minimum_filter
import os
from setup_korean_font import setup_korean_font

# 한글 폰트 설정
setup_korean_font()

# 파일 경로
file_path = "dataset/north_pacific.nc"

print("=" * 80)
print("북태평양 환류(Eddy) 분석 - 환류가 생기기 쉬운 지점 탐지")
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
    
    print(f"\n데이터 범위:")
    print(f"  위도: {lat.min():.2f}°N ~ {lat.max():.2f}°N")
    print(f"  경도: {lon.min():.2f}°E ~ {lon.max():.2f}°E")
    print(f"  격자 크기: {len(lat)} × {len(lon)}")
    
    # 좌표 그리드 생성
    lon_grid, lat_grid = np.meshgrid(lon, lat)
    
    # 위도, 경도를 미터 단위로 변환
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
    dv_dx = np.gradient(v_masked, axis=1) / dx  # ∂v/∂x
    du_dy = np.gradient(u_masked, axis=0) / dy  # ∂u/∂y
    
    # Vorticity 계산
    vorticity = dv_dx - du_dy
    
    # Coriolis 파라미터 (f = 2Ωsin(φ))
    omega = 7.2921e-5  # 지구 자전 각속도 (rad/s)
    f = 2 * omega * np.sin(np.deg2rad(lat_grid))  # Coriolis 파라미터
    
    # 상대적 소용돌이도
    relative_vorticity = vorticity
    
    # 정규화된 상대 소용돌이도 (ζ/f)
    normalized_vorticity = np.where(f != 0, relative_vorticity / np.abs(f), np.nan)
    
    # Vorticity 마스킹
    vorticity_masked = np.ma.masked_where(
        np.isnan(vorticity) | (mask == 0) | np.isinf(vorticity), 
        vorticity
    )
    
    # 출력 디렉토리 생성
    output_dir = "analysis_output/north_pacific"
    os.makedirs(output_dir, exist_ok=True)
    
    print("\n1. Vorticity 통계")
    print("-" * 80)
    valid_vort = vorticity_masked[~vorticity_masked.mask]
    if len(valid_vort) > 0:
        print(f"상대 소용돌이도 범위: {np.nanmin(valid_vort):.2e} ~ {np.nanmax(valid_vort):.2e} s⁻¹")
        print(f"평균: {np.nanmean(valid_vort):.2e} s⁻¹")
        print(f"표준편차: {np.nanstd(valid_vort):.2e} s⁻¹")
        
        # 양수/음수 vorticity 개수
        positive_vort = np.sum((valid_vort > 0) & ~np.isnan(valid_vort))
        negative_vort = np.sum((valid_vort < 0) & ~np.isnan(valid_vort))
        print(f"\n시계방향 환류 영역 (양수 vorticity): {positive_vort} 개 픽셀")
        print(f"반시계방향 환류 영역 (음수 vorticity): {negative_vort} 개 픽셀")
    
    # 환류 중심 찾기
    print("\n2. 환류 중심 탐지 중...")
    print("-" * 80)
    
    # Vorticity를 스무딩하여 노이즈 제거
    vorticity_smooth = np.where(~np.isnan(vorticity_masked), vorticity_masked, 0)
    vorticity_smooth = gaussian_filter(vorticity_smooth, sigma=1.5)
    vorticity_smooth = np.ma.masked_where(vorticity_masked.mask, vorticity_smooth)
    
    # 임계값 설정 (표준편차의 배수로 조정 가능)
    threshold_multiplier = 1.5
    threshold = np.nanstd(valid_vort) * threshold_multiplier
    
    # 양수 vorticity의 국소적 최대값 (시계방향 환류)
    local_max = maximum_filter(vorticity_smooth, size=7)
    positive_eddies = (vorticity_smooth == local_max) & (vorticity_smooth > threshold)
    
    # 음수 vorticity의 국소적 최소값 (반시계방향 환류)
    local_min = minimum_filter(vorticity_smooth, size=7)
    negative_eddies = (vorticity_smooth == local_min) & (vorticity_smooth < -threshold)
    
    # 환류 중심 좌표 추출
    positive_indices = np.where(positive_eddies)
    negative_indices = np.where(negative_eddies)
    
    print(f"탐지 임계값: ±{threshold:.2e} s⁻¹")
    print(f"탐지된 시계방향 환류 중심: {len(positive_indices[0])} 개")
    print(f"탐지된 반시계방향 환류 중심: {len(negative_indices[0])} 개")
    
    # 환류 강도 분석 (vorticity의 절대값)
    eddy_strength = np.abs(vorticity_smooth)
    eddy_strength_masked = np.ma.masked_where(vorticity_masked.mask, eddy_strength)
    
    # 환류 발생 가능성 지수 (Vorticity의 절대값이 클수록 환류 발생 가능성 높음)
    eddy_probability = np.clip(eddy_strength_masked / np.nanpercentile(eddy_strength_masked[~eddy_strength_masked.mask], 95), 0, 2)
    
    # 시각화
    print("\n3. 시각화 생성 중...")
    print("-" * 80)
    
    # 1. Vorticity 지도와 환류 중심
    fig, ax = plt.subplots(figsize=(16, 10))
    
    # Vorticity 컨투어
    levels = np.linspace(np.nanpercentile(valid_vort, 2), 
                        np.nanpercentile(valid_vort, 98), 40)
    im = ax.contourf(lon_grid, lat_grid, vorticity_masked, 
                     levels=levels, cmap='RdBu_r', extend='both')
    
    # 육지 경계선
    ax.contour(lon_grid, lat_grid, mask, levels=[0.5], colors='black', linewidths=1)
    
    # 환류 중심 표시
    if len(positive_indices[0]) > 0:
        ax.scatter(lon_grid[positive_indices], lat_grid[positive_indices], 
                  c='red', s=120, marker='o', edgecolors='darkred', 
                  linewidths=2, label=f'Clockwise Eddy ({len(positive_indices[0])})', zorder=5)
    
    if len(negative_indices[0]) > 0:
        ax.scatter(lon_grid[negative_indices], lat_grid[negative_indices], 
                  c='blue', s=120, marker='s', edgecolors='darkblue', 
                  linewidths=2, label=f'Counter-clockwise Eddy ({len(negative_indices[0])})', zorder=5)
    
    # 해류 벡터 (서브샘플링)
    skip = 6
    u_sub = u_masked[::skip, ::skip]
    v_sub = v_masked[::skip, ::skip]
    lon_sub = lon_grid[::skip, ::skip]
    lat_sub = lat_grid[::skip, ::skip]
    valid = ~np.isnan(u_sub) & ~np.isnan(v_sub)
    
    ax.quiver(lon_sub[valid], lat_sub[valid], u_sub[valid], v_sub[valid],
              scale=0.2, width=0.002, color='white', alpha=0.5, zorder=3)
    
    # 전체 데이터 범위 명시적으로 설정
    ax.set_xlim(lon.min(), lon.max())
    ax.set_ylim(lat.min(), lat.max())
    ax.set_xlabel('Longitude (°E)', fontsize=12)
    ax.set_ylabel('Latitude (°N)', fontsize=12)
    ax.set_title(f'Relative Vorticity and Eddy Detection - North Pacific\n({lat.min():.1f}°N - {lat.max():.1f}°N, {lon.min():.1f}°E - {lon.max():.1f}°E)', 
                fontsize=14, fontweight='bold')
    ax.legend(loc='upper right', fontsize=11)
    plt.colorbar(im, ax=ax, label='Relative Vorticity (s⁻¹)')
    plt.tight_layout()
    plt.savefig(f"{output_dir}/eddy_vorticity.png", dpi=300, bbox_inches='tight')
    print(f"✓ Vorticity 및 환류 중심 지도 저장: {output_dir}/eddy_vorticity.png")
    plt.close()
    
    # 2. 환류 발생 가능성 지도
    fig, ax = plt.subplots(figsize=(16, 10))
    
    # 환류 발생 가능성 컨투어
    prob_levels = np.linspace(0, 2, 30)
    im = ax.contourf(lon_grid, lat_grid, eddy_probability, 
                     levels=prob_levels, cmap='YlOrRd', extend='max')
    
    # 육지 경계선
    ax.contour(lon_grid, lat_grid, mask, levels=[0.5], colors='black', linewidths=1)
    
    # 환류 중심 표시
    if len(positive_indices[0]) > 0:
        ax.scatter(lon_grid[positive_indices], lat_grid[positive_indices], 
                  c='red', s=150, marker='o', edgecolors='darkred', 
                  linewidths=2, label=f'Clockwise Eddy ({len(positive_indices[0])})', zorder=5)
    
    if len(negative_indices[0]) > 0:
        ax.scatter(lon_grid[negative_indices], lat_grid[negative_indices], 
                  c='blue', s=150, marker='s', edgecolors='darkblue', 
                  linewidths=2, label=f'Counter-clockwise Eddy ({len(negative_indices[0])})', zorder=5)
    
    # 전체 데이터 범위 명시적으로 설정
    ax.set_xlim(lon.min(), lon.max())
    ax.set_ylim(lat.min(), lat.max())
    ax.set_xlabel('Longitude (°E)', fontsize=12)
    ax.set_ylabel('Latitude (°N)', fontsize=12)
    ax.set_title(f'Eddy Formation Probability - North Pacific\n({lat.min():.1f}°N - {lat.max():.1f}°N, {lon.min():.1f}°E - {lon.max():.1f}°E)', 
                fontsize=14, fontweight='bold')
    ax.legend(loc='upper right', fontsize=11)
    plt.colorbar(im, ax=ax, label='Eddy Probability Index')
    plt.tight_layout()
    plt.savefig(f"{output_dir}/eddy_probability.png", dpi=300, bbox_inches='tight')
    print(f"✓ 환류 발생 가능성 지도 저장: {output_dir}/eddy_probability.png")
    plt.close()
    
    # 3. SSH와 환류 중심 비교
    fig, ax = plt.subplots(figsize=(16, 10))
    
    # SSH 컨투어
    ssh_levels = np.linspace(np.nanmin(ssh), np.nanmax(ssh), 25)
    im = ax.contourf(lon_grid, lat_grid, ssh, levels=ssh_levels, 
                     cmap='viridis', extend='both', alpha=0.8)
    ax.contour(lon_grid, lat_grid, ssh, levels=ssh_levels, 
              colors='gray', linewidths=0.3, alpha=0.4)
    
    # 육지 경계선
    ax.contour(lon_grid, lat_grid, mask, levels=[0.5], colors='black', linewidths=1)
    
    # 환류 중심 표시
    if len(positive_indices[0]) > 0:
        ax.scatter(lon_grid[positive_indices], lat_grid[positive_indices], 
                  c='red', s=150, marker='o', edgecolors='darkred', 
                  linewidths=2, label=f'Clockwise Eddy ({len(positive_indices[0])})', zorder=5)
    
    if len(negative_indices[0]) > 0:
        ax.scatter(lon_grid[negative_indices], lat_grid[negative_indices], 
                  c='blue', s=150, marker='s', edgecolors='darkblue', 
                  linewidths=2, label=f'Counter-clockwise Eddy ({len(negative_indices[0])})', zorder=5)
    
    # 전체 데이터 범위 명시적으로 설정
    ax.set_xlim(lon.min(), lon.max())
    ax.set_ylim(lat.min(), lat.max())
    ax.set_xlabel('Longitude (°E)', fontsize=12)
    ax.set_ylabel('Latitude (°N)', fontsize=12)
    ax.set_title(f'Sea Surface Height (SSH) with Eddy Centers - North Pacific\n({lat.min():.1f}°N - {lat.max():.1f}°N, {lon.min():.1f}°E - {lon.max():.1f}°E)', 
                fontsize=14, fontweight='bold')
    ax.legend(loc='upper right', fontsize=11)
    plt.colorbar(im, ax=ax, label='SSH (m)')
    plt.tight_layout()
    plt.savefig(f"{output_dir}/eddy_ssh.png", dpi=300, bbox_inches='tight')
    print(f"✓ SSH와 환류 중심 지도 저장: {output_dir}/eddy_ssh.png")
    plt.close()
    
    # 4. 통합 패널
    fig, axes = plt.subplots(2, 2, figsize=(18, 14))
    
    # Vorticity
    im1 = axes[0, 0].contourf(lon_grid, lat_grid, vorticity_masked, 
                             levels=levels, cmap='RdBu_r', extend='both')
    axes[0, 0].contour(lon_grid, lat_grid, mask, levels=[0.5], colors='black', linewidths=0.5)
    if len(positive_indices[0]) > 0:
        axes[0, 0].scatter(lon_grid[positive_indices], lat_grid[positive_indices], 
                          c='red', s=60, marker='o', edgecolors='darkred', linewidths=1)
    if len(negative_indices[0]) > 0:
        axes[0, 0].scatter(lon_grid[negative_indices], lat_grid[negative_indices], 
                          c='blue', s=60, marker='s', edgecolors='darkblue', linewidths=1)
    axes[0, 0].set_xlim(lon.min(), lon.max())
    axes[0, 0].set_ylim(lat.min(), lat.max())
    axes[0, 0].set_title('Relative Vorticity', fontsize=12, fontweight='bold')
    axes[0, 0].set_xlabel('Longitude (°E)')
    axes[0, 0].set_ylabel('Latitude (°N)')
    plt.colorbar(im1, ax=axes[0, 0], label='s⁻¹')
    
    # 환류 발생 가능성
    im2 = axes[0, 1].contourf(lon_grid, lat_grid, eddy_probability, 
                             levels=prob_levels, cmap='YlOrRd', extend='max')
    axes[0, 1].contour(lon_grid, lat_grid, mask, levels=[0.5], colors='black', linewidths=0.5)
    if len(positive_indices[0]) > 0:
        axes[0, 1].scatter(lon_grid[positive_indices], lat_grid[positive_indices], 
                          c='red', s=60, marker='o', edgecolors='darkred', linewidths=1)
    if len(negative_indices[0]) > 0:
        axes[0, 1].scatter(lon_grid[negative_indices], lat_grid[negative_indices], 
                          c='blue', s=60, marker='s', edgecolors='darkblue', linewidths=1)
    axes[0, 1].set_xlim(lon.min(), lon.max())
    axes[0, 1].set_ylim(lat.min(), lat.max())
    axes[0, 1].set_title('Eddy Formation Probability', fontsize=12, fontweight='bold')
    axes[0, 1].set_xlabel('Longitude (°E)')
    axes[0, 1].set_ylabel('Latitude (°N)')
    plt.colorbar(im2, ax=axes[0, 1], label='Probability Index')
    
    # SSH
    im3 = axes[1, 0].contourf(lon_grid, lat_grid, ssh, levels=ssh_levels, 
                             cmap='viridis', extend='both')
    axes[1, 0].contour(lon_grid, lat_grid, mask, levels=[0.5], colors='black', linewidths=0.5)
    if len(positive_indices[0]) > 0:
        axes[1, 0].scatter(lon_grid[positive_indices], lat_grid[positive_indices], 
                          c='red', s=60, marker='o', edgecolors='darkred', linewidths=1)
    if len(negative_indices[0]) > 0:
        axes[1, 0].scatter(lon_grid[negative_indices], lat_grid[negative_indices], 
                          c='blue', s=60, marker='s', edgecolors='darkblue', linewidths=1)
    axes[1, 0].set_xlim(lon.min(), lon.max())
    axes[1, 0].set_ylim(lat.min(), lat.max())
    axes[1, 0].set_title('Sea Surface Height', fontsize=12, fontweight='bold')
    axes[1, 0].set_xlabel('Longitude (°E)')
    axes[1, 0].set_ylabel('Latitude (°N)')
    plt.colorbar(im3, ax=axes[1, 0], label='m')
    
    # 해류 속도
    speed = np.sqrt(u_masked**2 + v_masked**2)
    speed_masked = np.ma.masked_where(np.isnan(speed) | (mask == 0), speed)
    im4 = axes[1, 1].contourf(lon_grid, lat_grid, speed_masked, levels=25, 
                             cmap='jet', extend='both')
    axes[1, 1].contour(lon_grid, lat_grid, mask, levels=[0.5], colors='black', linewidths=0.5)
    skip2 = 4
    u_sub2 = u_masked[::skip2, ::skip2]
    v_sub2 = v_masked[::skip2, ::skip2]
    lon_sub2 = lon_grid[::skip2, ::skip2]
    lat_sub2 = lat_grid[::skip2, ::skip2]
    valid2 = ~np.isnan(u_sub2) & ~np.isnan(v_sub2)
    axes[1, 1].quiver(lon_sub2[valid2], lat_sub2[valid2], u_sub2[valid2], v_sub2[valid2],
                     scale=0.2, width=0.002, color='white', alpha=0.6)
    if len(positive_indices[0]) > 0:
        axes[1, 1].scatter(lon_grid[positive_indices], lat_grid[positive_indices], 
                          c='red', s=60, marker='o', edgecolors='darkred', linewidths=1)
    if len(negative_indices[0]) > 0:
        axes[1, 1].scatter(lon_grid[negative_indices], lat_grid[negative_indices], 
                          c='blue', s=60, marker='s', edgecolors='darkblue', linewidths=1)
    axes[1, 1].set_xlim(lon.min(), lon.max())
    axes[1, 1].set_ylim(lat.min(), lat.max())
    axes[1, 1].set_title('Current Speed & Vectors', fontsize=12, fontweight='bold')
    axes[1, 1].set_xlabel('Longitude (°E)')
    axes[1, 1].set_ylabel('Latitude (°N)')
    plt.colorbar(im4, ax=axes[1, 1], label='m/s')
    
    plt.suptitle(f'Eddy Detection and Formation Probability Analysis - North Pacific\nData Coverage: {lat.min():.1f}°N - {lat.max():.1f}°N, {lon.min():.1f}°E - {lon.max():.1f}°E', 
                fontsize=16, fontweight='bold', y=0.995)
    plt.tight_layout()
    plt.savefig(f"{output_dir}/eddy_comprehensive.png", dpi=300, bbox_inches='tight')
    print(f"✓ 통합 환류 분석 패널 저장: {output_dir}/eddy_comprehensive.png")
    plt.close()
    
    # 환류 중심 정보 저장
    with open(f"{output_dir}/eddy_locations.txt", "w", encoding="utf-8") as f:
        f.write("=" * 80 + "\n")
        f.write("북태평양 환류 중심 위치 정보\n")
        f.write("=" * 80 + "\n\n")
        f.write(f"분석 날짜: 2025-11-27\n")
        f.write(f"탐지 임계값: ±{threshold:.2e} s⁻¹\n")
        f.write(f"위도 범위: {lat.min():.2f}°N ~ {lat.max():.2f}°N\n")
        f.write(f"경도 범위: {lon.min():.2f}°E ~ {lon.max():.2f}°E\n\n")
        
        f.write("시계방향 환류 (양수 vorticity) 중심:\n")
        f.write("-" * 80 + "\n")
        if len(positive_indices[0]) > 0:
            # Vorticity 강도로 정렬
            eddy_data = []
            for i in range(len(positive_indices[0])):
                idx_lat, idx_lon = positive_indices[0][i], positive_indices[1][i]
                eddy_lat = lat_grid[idx_lat, idx_lon]
                eddy_lon = lon_grid[idx_lat, idx_lon]
                eddy_vort = vorticity_smooth[idx_lat, idx_lon]
                eddy_ssh = ssh[idx_lat, idx_lon]
                eddy_prob = eddy_probability[idx_lat, idx_lon]
                eddy_data.append((eddy_vort, eddy_lat, eddy_lon, eddy_ssh, eddy_prob))
            
            # Vorticity 강도 순으로 정렬
            eddy_data.sort(reverse=True)
            
            for i, (vort, eddy_lat, eddy_lon, eddy_ssh, eddy_prob) in enumerate(eddy_data):
                f.write(f"{i+1}. 위치: ({eddy_lat:.4f}°N, {eddy_lon:.4f}°E), "
                       f"Vorticity: {vort:.2e} s⁻¹, SSH: {eddy_ssh:.4f} m, "
                       f"Probability: {eddy_prob:.2f}\n")
        else:
            f.write("탐지된 환류 없음\n")
        
        f.write("\n반시계방향 환류 (음수 vorticity) 중심:\n")
        f.write("-" * 80 + "\n")
        if len(negative_indices[0]) > 0:
            eddy_data = []
            for i in range(len(negative_indices[0])):
                idx_lat, idx_lon = negative_indices[0][i], negative_indices[1][i]
                eddy_lat = lat_grid[idx_lat, idx_lon]
                eddy_lon = lon_grid[idx_lat, idx_lon]
                eddy_vort = vorticity_smooth[idx_lat, idx_lon]
                eddy_ssh = ssh[idx_lat, idx_lon]
                eddy_prob = eddy_probability[idx_lat, idx_lon]
                eddy_data.append((abs(eddy_vort), eddy_lat, eddy_lon, eddy_ssh, eddy_prob))
            
            # Vorticity 강도 순으로 정렬
            eddy_data.sort(reverse=True)
            
            for i, (vort, eddy_lat, eddy_lon, eddy_ssh, eddy_prob) in enumerate(eddy_data):
                f.write(f"{i+1}. 위치: ({eddy_lat:.4f}°N, {eddy_lon:.4f}°E), "
                       f"Vorticity: {eddy_vort:.2e} s⁻¹, SSH: {eddy_ssh:.4f} m, "
                       f"Probability: {eddy_prob:.2f}\n")
        else:
            f.write("탐지된 환류 없음\n")
        
        # 환류 발생 가능성이 높은 지역 요약
        f.write("\n\n환류 발생 가능성이 높은 지역 (상위 10개):\n")
        f.write("-" * 80 + "\n")
        high_prob_indices = np.unravel_index(
            np.argsort(eddy_probability.flatten())[::-1][:10], 
            eddy_probability.shape
        )
        for i in range(min(10, len(high_prob_indices[0]))):
            idx_lat, idx_lon = high_prob_indices[0][i], high_prob_indices[1][i]
            if not eddy_probability.mask[idx_lat, idx_lon]:
                prob_lat = lat_grid[idx_lat, idx_lon]
                prob_lon = lon_grid[idx_lat, idx_lon]
                prob_val = eddy_probability[idx_lat, idx_lon]
                prob_vort = vorticity_smooth[idx_lat, idx_lon]
                f.write(f"{i+1}. 위치: ({prob_lat:.4f}°N, {prob_lon:.4f}°E), "
                       f"Probability: {prob_val:.2f}, Vorticity: {prob_vort:.2e} s⁻¹\n")
    
    print(f"✓ 환류 위치 정보 저장: {output_dir}/eddy_locations.txt")
    
    nc.close()
    
    print("\n" + "=" * 80)
    print("북태평양 환류 분석 완료!")
    print(f"모든 결과는 '{output_dir}' 폴더에 저장되었습니다.")
    print("=" * 80)
    
except Exception as e:
    print(f"\n오류 발생: {e}")
    import traceback
    traceback.print_exc()

