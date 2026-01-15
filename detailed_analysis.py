"""
KHOA NetCDF 파일 상세 분석 및 시각화
"""
from netCDF4 import Dataset
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
import os
from setup_korean_font import setup_korean_font

# 한글 폰트 설정
setup_korean_font()

# 파일 경로
file_path = "dataset/KHOA_SCU_L4_Z004_D01_U20251118_EastSea.nc"

print("=" * 80)
print("KHOA NetCDF 파일 상세 분석 및 시각화")
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
    
    # 해류 속도 계산
    speed = np.sqrt(u**2 + v**2)
    
    # 출력 디렉토리 생성
    output_dir = "analysis_output"
    os.makedirs(output_dir, exist_ok=True)
    
    print("\n1. 데이터 요약")
    print("-" * 80)
    print(f"위도 범위: {lat.min():.4f}°N ~ {lat.max():.4f}°N")
    print(f"경도 범위: {lon.min():.4f}°E ~ {lon.max():.4f}°E")
    print(f"공간 해상도: 0.25° × 0.25°")
    print(f"격자 크기: {len(lat)} × {len(lon)}")
    print(f"\n해류 속도 (u):")
    print(f"  범위: {np.nanmin(u):.4f} ~ {np.nanmax(u):.4f} m/s")
    print(f"  평균: {np.nanmean(u):.4f} m/s")
    print(f"\n해류 속도 (v):")
    print(f"  범위: {np.nanmin(v):.4f} ~ {np.nanmax(v):.4f} m/s")
    print(f"  평균: {np.nanmean(v):.4f} m/s")
    print(f"\n해류 속도 크기:")
    print(f"  범위: {np.nanmin(speed):.4f} ~ {np.nanmax(speed):.4f} m/s")
    print(f"  평균: {np.nanmean(speed):.4f} m/s")
    print(f"\n해수면 높이 (SSH):")
    print(f"  범위: {np.nanmin(ssh):.4f} ~ {np.nanmax(ssh):.4f} m")
    print(f"  평균: {np.nanmean(ssh):.4f} m")
    
    # 육지/바다 비율
    water_pixels = np.sum(mask == 1)
    land_pixels = np.sum(mask == 0)
    total_pixels = mask.size
    print(f"\n육지/바다 분포:")
    print(f"  바다: {water_pixels} 픽셀 ({water_pixels/total_pixels*100:.1f}%)")
    print(f"  육지: {land_pixels} 픽셀 ({land_pixels/total_pixels*100:.1f}%)")
    
    # 유효 데이터 비율
    valid_u = np.sum(~np.isnan(u))
    valid_v = np.sum(~np.isnan(v))
    print(f"\n유효 데이터:")
    print(f"  u 변수: {valid_u}/{u.size} ({valid_u/u.size*100:.1f}%)")
    print(f"  v 변수: {valid_v}/{v.size} ({valid_v/v.size*100:.1f}%)")
    
    # 좌표 그리드 생성
    lon_grid, lat_grid = np.meshgrid(lon, lat)
    
    # 시각화
    print("\n\n2. 시각화 생성 중...")
    print("-" * 80)
    
    # 1. 해수면 높이 (SSH)
    fig, ax = plt.subplots(figsize=(12, 8))
    im = ax.contourf(lon_grid, lat_grid, ssh, levels=20, cmap='viridis', extend='both')
    ax.contour(lon_grid, lat_grid, mask, levels=[0.5], colors='black', linewidths=1)
    ax.set_xlabel('경도 (°E)', fontsize=12)
    ax.set_ylabel('위도 (°N)', fontsize=12)
    ax.set_title('해수면 높이 (SSH) - 2025-11-18', fontsize=14, fontweight='bold')
    plt.colorbar(im, ax=ax, label='해수면 높이 (m)')
    plt.tight_layout()
    plt.savefig(f"{output_dir}/ssh_map.png", dpi=300, bbox_inches='tight')
    print(f"✓ SSH 지도 저장: {output_dir}/ssh_map.png")
    plt.close()
    
    # 2. 해류 속도 크기
    fig, ax = plt.subplots(figsize=(12, 8))
    speed_masked = np.ma.masked_where(np.isnan(speed) | (mask == 0), speed)
    im = ax.contourf(lon_grid, lat_grid, speed_masked, levels=20, cmap='jet', extend='both')
    ax.contour(lon_grid, lat_grid, mask, levels=[0.5], colors='black', linewidths=1)
    ax.set_xlabel('경도 (°E)', fontsize=12)
    ax.set_ylabel('위도 (°N)', fontsize=12)
    ax.set_title('해류 속도 크기 - 2025-11-18', fontsize=14, fontweight='bold')
    plt.colorbar(im, ax=ax, label='속도 (m/s)')
    plt.tight_layout()
    plt.savefig(f"{output_dir}/current_speed.png", dpi=300, bbox_inches='tight')
    print(f"✓ 해류 속도 지도 저장: {output_dir}/current_speed.png")
    plt.close()
    
    # 3. 해류 벡터 필드 (서브샘플링)
    fig, ax = plt.subplots(figsize=(12, 8))
    # 벡터를 보기 쉽게 서브샘플링 (5개마다)
    skip = 5
    u_sub = u[::skip, ::skip]
    v_sub = v[::skip, ::skip]
    lon_sub = lon_grid[::skip, ::skip]
    lat_sub = lat_grid[::skip, ::skip]
    speed_sub = speed[::skip, ::skip]
    
    # 속도 크기로 색상 지정
    speed_masked_sub = np.ma.masked_where(np.isnan(speed_sub) | (mask[::skip, ::skip] == 0), speed_sub)
    im = ax.contourf(lon_grid, lat_grid, speed_masked, levels=20, cmap='jet', alpha=0.6, extend='both')
    ax.contour(lon_grid, lat_grid, mask, levels=[0.5], colors='black', linewidths=1)
    
    # 벡터 그리기
    valid = ~np.isnan(u_sub) & ~np.isnan(v_sub) & (mask[::skip, ::skip] == 1)
    ax.quiver(lon_sub[valid], lat_sub[valid], u_sub[valid], v_sub[valid], 
              speed_sub[valid], cmap='jet', scale=0.5, width=0.003, angles='xy')
    
    ax.set_xlabel('경도 (°E)', fontsize=12)
    ax.set_ylabel('위도 (°N)', fontsize=12)
    ax.set_title('해류 벡터 필드 - 2025-11-18', fontsize=14, fontweight='bold')
    plt.colorbar(im, ax=ax, label='속도 (m/s)')
    plt.tight_layout()
    plt.savefig(f"{output_dir}/current_vectors.png", dpi=300, bbox_inches='tight')
    print(f"✓ 해류 벡터 필드 저장: {output_dir}/current_vectors.png")
    plt.close()
    
    # 4. u 성분 (동서 방향)
    fig, ax = plt.subplots(figsize=(12, 8))
    u_masked = np.ma.masked_where(np.isnan(u) | (mask == 0), u)
    im = ax.contourf(lon_grid, lat_grid, u_masked, levels=20, cmap='RdBu_r', extend='both')
    ax.contour(lon_grid, lat_grid, mask, levels=[0.5], colors='black', linewidths=1)
    ax.set_xlabel('경도 (°E)', fontsize=12)
    ax.set_ylabel('위도 (°N)', fontsize=12)
    ax.set_title('동서 방향 해류 속도 (u) - 2025-11-18', fontsize=14, fontweight='bold')
    plt.colorbar(im, ax=ax, label='속도 (m/s)')
    plt.tight_layout()
    plt.savefig(f"{output_dir}/u_component.png", dpi=300, bbox_inches='tight')
    print(f"✓ u 성분 지도 저장: {output_dir}/u_component.png")
    plt.close()
    
    # 5. v 성분 (남북 방향)
    fig, ax = plt.subplots(figsize=(12, 8))
    v_masked = np.ma.masked_where(np.isnan(v) | (mask == 0), v)
    im = ax.contourf(lon_grid, lat_grid, v_masked, levels=20, cmap='RdBu_r', extend='both')
    ax.contour(lon_grid, lat_grid, mask, levels=[0.5], colors='black', linewidths=1)
    ax.set_xlabel('경도 (°E)', fontsize=12)
    ax.set_ylabel('위도 (°N)', fontsize=12)
    ax.set_title('남북 방향 해류 속도 (v) - 2025-11-18', fontsize=14, fontweight='bold')
    plt.colorbar(im, ax=ax, label='속도 (m/s)')
    plt.tight_layout()
    plt.savefig(f"{output_dir}/v_component.png", dpi=300, bbox_inches='tight')
    print(f"✓ v 성분 지도 저장: {output_dir}/v_component.png")
    plt.close()
    
    # 6. 통합 패널
    fig, axes = plt.subplots(2, 2, figsize=(16, 14))
    
    # SSH
    im1 = axes[0, 0].contourf(lon_grid, lat_grid, ssh, levels=20, cmap='viridis', extend='both')
    axes[0, 0].contour(lon_grid, lat_grid, mask, levels=[0.5], colors='black', linewidths=0.5)
    axes[0, 0].set_title('해수면 높이 (SSH)', fontsize=12, fontweight='bold')
    axes[0, 0].set_xlabel('경도 (°E)')
    axes[0, 0].set_ylabel('위도 (°N)')
    plt.colorbar(im1, ax=axes[0, 0], label='m')
    
    # 해류 속도
    im2 = axes[0, 1].contourf(lon_grid, lat_grid, speed_masked, levels=20, cmap='jet', extend='both')
    axes[0, 1].contour(lon_grid, lat_grid, mask, levels=[0.5], colors='black', linewidths=0.5)
    axes[0, 1].set_title('해류 속도 크기', fontsize=12, fontweight='bold')
    axes[0, 1].set_xlabel('경도 (°E)')
    axes[0, 1].set_ylabel('위도 (°N)')
    plt.colorbar(im2, ax=axes[0, 1], label='m/s')
    
    # u 성분
    im3 = axes[1, 0].contourf(lon_grid, lat_grid, u_masked, levels=20, cmap='RdBu_r', extend='both')
    axes[1, 0].contour(lon_grid, lat_grid, mask, levels=[0.5], colors='black', linewidths=0.5)
    axes[1, 0].set_title('동서 방향 해류 (u)', fontsize=12, fontweight='bold')
    axes[1, 0].set_xlabel('경도 (°E)')
    axes[1, 0].set_ylabel('위도 (°N)')
    plt.colorbar(im3, ax=axes[1, 0], label='m/s')
    
    # v 성분
    im4 = axes[1, 1].contourf(lon_grid, lat_grid, v_masked, levels=20, cmap='RdBu_r', extend='both')
    axes[1, 1].contour(lon_grid, lat_grid, mask, levels=[0.5], colors='black', linewidths=0.5)
    axes[1, 1].set_title('남북 방향 해류 (v)', fontsize=12, fontweight='bold')
    axes[1, 1].set_xlabel('경도 (°E)')
    axes[1, 1].set_ylabel('위도 (°N)')
    plt.colorbar(im4, ax=axes[1, 1], label='m/s')
    
    plt.suptitle('KHOA 표층 해류 데이터 분석 - 2025-11-18', fontsize=16, fontweight='bold', y=0.995)
    plt.tight_layout()
    plt.savefig(f"{output_dir}/summary_panel.png", dpi=300, bbox_inches='tight')
    print(f"✓ 통합 패널 저장: {output_dir}/summary_panel.png")
    plt.close()
    
    # 통계 정보를 텍스트 파일로 저장
    with open(f"{output_dir}/data_summary.txt", "w", encoding="utf-8") as f:
        f.write("=" * 80 + "\n")
        f.write("KHOA 표층 해류 데이터 분석 요약\n")
        f.write("=" * 80 + "\n\n")
        f.write(f"파일명: {file_path}\n")
        f.write(f"날짜: 2025-11-18\n")
        f.write(f"지역: 동해 (East Sea)\n\n")
        f.write("공간 정보:\n")
        f.write(f"  위도 범위: {lat.min():.4f}°N ~ {lat.max():.4f}°N\n")
        f.write(f"  경도 범위: {lon.min():.4f}°E ~ {lon.max():.4f}°E\n")
        f.write(f"  공간 해상도: 0.25° × 0.25°\n")
        f.write(f"  격자 크기: {len(lat)} × {len(lon)}\n\n")
        f.write("해류 속도 통계:\n")
        f.write(f"  u (동서): {np.nanmin(u):.4f} ~ {np.nanmax(u):.4f} m/s (평균: {np.nanmean(u):.4f} m/s)\n")
        f.write(f"  v (남북): {np.nanmin(v):.4f} ~ {np.nanmax(v):.4f} m/s (평균: {np.nanmean(v):.4f} m/s)\n")
        f.write(f"  속도 크기: {np.nanmin(speed):.4f} ~ {np.nanmax(speed):.4f} m/s (평균: {np.nanmean(speed):.4f} m/s)\n\n")
        f.write("해수면 높이 (SSH):\n")
        f.write(f"  범위: {np.nanmin(ssh):.4f} ~ {np.nanmax(ssh):.4f} m\n")
        f.write(f"  평균: {np.nanmean(ssh):.4f} m\n")
        f.write(f"  표준편차: {np.nanstd(ssh):.4f} m\n\n")
        f.write("데이터 품질:\n")
        f.write(f"  u 변수 유효 데이터: {valid_u}/{u.size} ({valid_u/u.size*100:.1f}%)\n")
        f.write(f"  v 변수 유효 데이터: {valid_v}/{v.size} ({valid_v/v.size*100:.1f}%)\n")
        f.write(f"  바다 픽셀: {water_pixels} ({water_pixels/total_pixels*100:.1f}%)\n")
        f.write(f"  육지 픽셀: {land_pixels} ({land_pixels/total_pixels*100:.1f}%)\n")
    
    print(f"✓ 데이터 요약 저장: {output_dir}/data_summary.txt")
    
    nc.close()
    
    print("\n" + "=" * 80)
    print("상세 분석 완료!")
    print(f"모든 결과는 '{output_dir}' 폴더에 저장되었습니다.")
    print("=" * 80)
    
except Exception as e:
    print(f"\n오류 발생: {e}")
    import traceback
    traceback.print_exc()

