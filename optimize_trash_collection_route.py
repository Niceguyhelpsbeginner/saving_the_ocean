"""
Ocean Current-Based Trash Collection Route Optimization Algorithm
"""
import pandas as pd
import numpy as np
from netCDF4 import Dataset
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist
from scipy.interpolate import griddata
import os
from itertools import permutations
import warnings
from setup_korean_font import setup_korean_font
warnings.filterwarnings('ignore')

# 한글 폰트 설정
setup_korean_font()

print("=" * 80)
print("Ocean Current-Based Trash Collection Route Optimization Algorithm")
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

# 지역별 쓰레기량 집계
area_trash = trash_df.groupby('INVS_AREA_NM').agg({
    'IEM_CNT': 'sum',
    'LAT': 'mean',
    'LON': 'mean'
}).reset_index()
area_trash.columns = ['AREA', 'TOTAL_TRASH', 'LAT', 'LON']
area_trash = area_trash.sort_values('TOTAL_TRASH', ascending=False).reset_index(drop=True)

print(f"Number of collection sites: {len(area_trash)}")

# 해류 데이터
nc = Dataset(current_file, 'r')
lat = nc.variables['lat'][:]
lon = nc.variables['lon'][:]
u = nc.variables['u'][:]
v = nc.variables['v'][:]
ssh = nc.variables['ssh'][:]
mask = nc.variables['mask'][:]
lon_grid, lat_grid = np.meshgrid(lon, lat)

print(f"Current data grid: {len(lat)} × {len(lon)}")

# 해류 정보 추출 함수
def get_current_at_location(lat_val, lon_val, lat_grid, lon_grid, data):
    """특정 위치의 해류 데이터 추출 (보간)"""
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

print("Current information extraction completed")

# 최적화 알고리즘 클래스
class TrashCollectionOptimizer:
    def __init__(self, locations_df, vessel_speed=5.0):
        """
        locations_df: 쓰레기 수거 지점 정보 (LAT, LON, TOTAL_TRASH, U, V, SPEED, VORTICITY)
        vessel_speed: 선박 기본 속도 (m/s, 기본값 5.0 m/s ≈ 9.7 knots)
        """
        self.locations = locations_df.copy()
        self.vessel_speed = vessel_speed  # m/s
        self.n = len(locations_df)
        self.distance_matrix = None
        self.cost_matrix = None
        
    def calculate_distance_matrix(self):
        """지점 간 거리 행렬 계산 (직선 거리)"""
        coords = self.locations[['LAT', 'LON']].values
        # Haversine 거리 계산
        R = 6371000  # 지구 반지름 (미터)
        lat1, lon1 = np.radians(coords[:, 0]), np.radians(coords[:, 1])
        
        distances = np.zeros((self.n, self.n))
        for i in range(self.n):+
            for j in range(self.n):
                if i != j:
                    lat2, lon2 = np.radians(coords[j, 0]), np.radians(coords[j, 1])
                    dlat = lat2 - lat1[i]
                    dlon = lon2 - lon1[i]
                    a = np.sin(dlat/2)**2 + np.cos(lat1[i]) * np.cos(lat2) * np.sin(dlon/2)**2
                    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1-a))
                    distances[i, j] = R * c
        
        self.distance_matrix = distances
        return distances
    
    def calculate_current_aided_cost(self, from_idx, to_idx):
        """해류를 고려한 이동 비용 계산"""
        if self.distance_matrix is None:
            self.calculate_distance_matrix()
        
        distance = self.distance_matrix[from_idx, to_idx]
        
        # 출발지와 목적지의 중간 지점에서 해류 추정
        lat_from = self.locations.iloc[from_idx]['LAT']
        lon_from = self.locations.iloc[from_idx]['LON']
        lat_to = self.locations.iloc[to_idx]['LAT']
        lon_to = self.locations.iloc[to_idx]['LON']
        
        # 중간 지점
        lat_mid = (lat_from + lat_to) / 2
        lon_mid = (lon_from + lon_to) / 2
        
        # 중간 지점의 해류
        u_mid = get_current_at_location(lat_mid, lon_mid, lat_grid, lon_grid, u)
        v_mid = get_current_at_location(lat_mid, lon_mid, lat_grid, lon_grid, v)
        
        if np.isnan(u_mid) or np.isnan(v_mid):
            u_mid, v_mid = 0, 0
        
        # 이동 방향 벡터
        lat_rad_from = np.radians(lat_from)
        lat_rad_to = np.radians(lat_to)
        lon_rad_from = np.radians(lon_from)
        lon_rad_to = np.radians(lon_to)
        
        dlat_rad = lat_rad_to - lat_rad_from
        dlon_rad = lon_rad_to - lon_rad_from
        
        # 방향 벡터 (정규화)
        direction_lat = dlat_rad / np.sqrt(dlat_rad**2 + dlon_rad**2) if dlat_rad != 0 or dlon_rad != 0 else 0
        direction_lon = dlon_rad / np.sqrt(dlat_rad**2 + dlon_rad**2) if dlat_rad != 0 or dlon_rad != 0 else 0
        
        # 해류 성분 (위도/경도 방향)
        current_lat = v_mid / 111000.0  # m/s를 도/초로 변환 (대략적)
        current_lon = u_mid / (111000.0 * np.cos(np.radians(lat_mid)))
        
        # 해류가 이동 방향과 같은 방향이면 도움, 반대면 방해
        current_aid = direction_lat * current_lat + direction_lon * current_lon
        
        # 유효 속도 = 선박 속도 + 해류 도움
        effective_speed = self.vessel_speed + current_aid * 111000.0  # 대략적 변환
        
        if effective_speed <= 0:
            effective_speed = self.vessel_speed * 0.5  # 최소 속도 보장
        
        # 이동 시간
        travel_time = distance / effective_speed
        
        return travel_time
    
    def calculate_cost_matrix(self):
        """해류를 고려한 비용 행렬 계산"""
        self.cost_matrix = np.zeros((self.n, self.n))
        
        for i in range(self.n):
            for j in range(self.n):
                if i != j:
                    # 기본 거리 비용
                    distance_cost = self.distance_matrix[i, j] / 1000  # km
                    
                    # 해류 도움 비용 (시간 단위)
                    current_cost = self.calculate_current_aided_cost(i, j)
                    
                    # 쓰레기량 가중치 (쓰레기가 많은 곳을 우선 방문)
                    trash_weight = 1.0 / (1.0 + self.locations.iloc[j]['TOTAL_TRASH'] / 100.0)
                    
                    # Vorticity 가중치 (환류 지역 우선)
                    vorticity_weight = 1.0 / (1.0 + abs(self.locations.iloc[j]['VORTICITY']) * 1e6)
                    
                    # 종합 비용
                    self.cost_matrix[i, j] = (distance_cost * 0.3 + 
                                            current_cost * 0.4 + 
                                            trash_weight * 0.2 + 
                                            vorticity_weight * 0.1)
                else:
                    self.cost_matrix[i, j] = np.inf
        
        return self.cost_matrix
    
    def greedy_route(self, start_idx=0):
        """Greedy 알고리즘으로 경로 생성"""
        if self.cost_matrix is None:
            self.calculate_cost_matrix()
        
        route = [start_idx]
        unvisited = set(range(self.n)) - {start_idx}
        
        current = start_idx
        total_cost = 0
        
        while unvisited:
            # 쓰레기량과 비용을 고려한 우선순위 계산
            best_next = None
            best_score = np.inf
            
            for next_idx in unvisited:
                cost = self.cost_matrix[current, next_idx]
                trash = self.locations.iloc[next_idx]['TOTAL_TRASH']
                
                # 쓰레기량이 많을수록, 비용이 적을수록 좋음
                score = cost / (1 + trash / 100.0)
                
                if score < best_score:
                    best_score = score
                    best_next = next_idx
            
            route.append(best_next)
            total_cost += self.cost_matrix[current, best_next]
            unvisited.remove(best_next)
            current = best_next
        
        return route, total_cost
    
    def nearest_neighbor_route(self, start_idx=0):
        """Nearest Neighbor 알고리즘"""
        if self.cost_matrix is None:
            self.calculate_cost_matrix()
        
        route = [start_idx]
        unvisited = set(range(self.n)) - {start_idx}
        current = start_idx
        total_cost = 0
        
        while unvisited:
            nearest = min(unvisited, key=lambda x: self.cost_matrix[current, x])
            route.append(nearest)
            total_cost += self.cost_matrix[current, nearest]
            unvisited.remove(nearest)
            current = nearest
        
        return route, total_cost
    
    def priority_based_route(self, start_idx=0):
        """쓰레기량 우선순위 기반 경로"""
        if self.cost_matrix is None:
            self.calculate_cost_matrix()
        
        # 쓰레기량 순으로 정렬
        priority_order = self.locations.sort_values('TOTAL_TRASH', ascending=False).index.tolist()
        
        # 시작점을 첫 번째로
        if start_idx in priority_order:
            priority_order.remove(start_idx)
        priority_order = [start_idx] + priority_order
        
        route = []
        total_cost = 0
        
        for i in range(len(priority_order) - 1):
            from_idx = priority_order[i]
            to_idx = priority_order[i + 1]
            route.append(from_idx)
            total_cost += self.cost_matrix[from_idx, to_idx]
        
        route.append(priority_order[-1])
        
        return route, total_cost
    
    def optimize_route(self, method='greedy', start_idx=0):
        """최적화된 경로 생성"""
        if self.distance_matrix is None:
            self.calculate_distance_matrix()
        if self.cost_matrix is None:
            self.calculate_cost_matrix()
        
        if method == 'greedy':
            return self.greedy_route(start_idx)
        elif method == 'nearest':
            return self.nearest_neighbor_route(start_idx)
        elif method == 'priority':
            return self.priority_based_route(start_idx)
        else:
            return self.greedy_route(start_idx)

# 최적화 실행
print("\n3. Running optimization algorithms...")
print("-" * 80)

optimizer = TrashCollectionOptimizer(area_trash, vessel_speed=5.0)  # 5 m/s ≈ 9.7 knots

# 여러 방법으로 경로 생성
methods = ['greedy', 'nearest', 'priority']
routes = {}
costs = {}

for method in methods:
    route, cost = optimizer.optimize_route(method=method, start_idx=0)
    routes[method] = route
    costs[method] = cost
    print(f"\n{method.upper()} method:")
    print(f"  Total cost: {cost:.2f}")
    print(f"  Route: {' -> '.join([area_trash.iloc[i]['AREA'] for i in route])}")

# 최적 경로 선택
best_method = min(costs.keys(), key=lambda x: costs[x])
best_route = routes[best_method]
best_cost = costs[best_method]

print(f"\nOptimal route ({best_method.upper()} method):")
print(f"Total cost: {best_cost:.2f}")

# 시각화
print("\n4. Generating visualizations...")
print("-" * 80)

output_dir = "analysis_output/route_optimization"
os.makedirs(output_dir, exist_ok=True)

# 1. 해류와 최적 경로
fig, ax = plt.subplots(figsize=(14, 10))

# 먼저 육지 영역 표시 (배경)
land_mask = np.where(mask == 0, 1, np.nan)
ax.contourf(lon_grid, lat_grid, land_mask, levels=[0.5, 1.5], 
           colors='lightgray', alpha=0.9, zorder=1)
ax.contour(lon_grid, lat_grid, mask, levels=[0.5], colors='black', 
          linewidths=2.5, zorder=2)

# 해류 속도 배경 (바다 영역만)
speed = np.sqrt(u**2 + v**2)
speed_masked = np.ma.masked_where(np.isnan(speed) | (mask == 0), speed)
im = ax.contourf(lon_grid, lat_grid, speed_masked, levels=20, 
                cmap='Blues', alpha=0.5, extend='both', zorder=3)

# 해류 벡터 (바다 영역만)
skip = 5
u_sub = u[::skip, ::skip]
v_sub = v[::skip, ::skip]
lon_sub = lon_grid[::skip, ::skip]
lat_sub = lat_grid[::skip, ::skip]
valid = ~np.isnan(u_sub) & ~np.isnan(v_sub) & (mask[::skip, ::skip] == 1)
ax.quiver(lon_sub[valid], lat_sub[valid], u_sub[valid], v_sub[valid],
          scale=0.3, width=0.003, color='white', alpha=0.6, zorder=4)

# 쓰레기 수거 지점 (경로 위에 표시되도록 zorder 조정)
scatter = ax.scatter(area_trash['LON'], area_trash['LAT'], 
                    s=area_trash['TOTAL_TRASH']*2, c=area_trash['TOTAL_TRASH'],
                    cmap='Reds', alpha=0.7, edgecolors='darkred', linewidths=2,
                    zorder=5, label='Collection Sites')

# 최적 경로 그리기 (화살표 포함)
route_lons = [area_trash.iloc[i]['LON'] for i in best_route]
route_lats = [area_trash.iloc[i]['LAT'] for i in best_route]
route_lons.append(route_lons[0])  # 시작점으로 복귀
route_lats.append(route_lats[0])

# 경로 선 그리기
for i in range(len(route_lons) - 1):
    ax.plot([route_lons[i], route_lons[i+1]], [route_lats[i], route_lats[i+1]], 
           'r-', linewidth=3, alpha=0.9, zorder=6)
    
    # 화살표 그리기
    ax.annotate('', xy=(route_lons[i+1], route_lats[i+1]), 
               xytext=(route_lons[i], route_lats[i]),
               arrowprops=dict(arrowstyle='->', lw=2.5, color='red', alpha=0.9),
               zorder=7)

ax.plot([], [], 'r-', linewidth=3, alpha=0.8, label='Optimal Route')

# 시작점 표시
ax.scatter(route_lons[0], route_lats[0], s=400, c='green', marker='*', 
          edgecolors='darkgreen', linewidths=3, zorder=10, label='Start/End')

# 각 지점에 순서 번호 표시
for order, idx in enumerate(best_route, 1):
    row = area_trash.iloc[idx]
    # 번호를 원 안에 표시
    ax.scatter(row['LON'], row['LAT'], s=500, c='yellow', 
              edgecolors='black', linewidths=3, zorder=8, alpha=0.95)
    ax.annotate(str(order), (row['LON'], row['LAT']), 
               fontsize=14, fontweight='bold', ha='center', va='center',
               color='black', zorder=9)

# 지점명 표시 (번호 옆에 작게)
for idx, row in area_trash.iterrows():
    route_order = best_route.index(idx) + 1 if idx in best_route else None
    if route_order:
        # 번호 위에 지점명 표시
        ax.annotate(row['AREA'], (row['LON'], row['LAT']), 
                   xytext=(0, 25), textcoords='offset points',
                   fontsize=7, alpha=0.9, zorder=10, ha='center',
                   bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8, edgecolor='black'))

ax.set_xlim(lon.min(), lon.max())
ax.set_ylim(lat.min(), lat.max())
ax.set_xlabel('Longitude (°E)', fontsize=12)
ax.set_ylabel('Latitude (°N)', fontsize=12)
ax.set_title(f'Optimal Trash Collection Route ({best_method.upper()})\nEast Sea Region (Korean & Japanese Coasts)\nTotal Cost: {best_cost:.2f}', 
            fontsize=14, fontweight='bold')
ax.legend(loc='upper right', fontsize=10)
plt.colorbar(scatter, ax=ax, label='Trash Count (items)')
plt.tight_layout()
plt.savefig(f"{output_dir}/optimal_route.png", dpi=300, bbox_inches='tight')
print(f"✓ Optimal route map saved: {output_dir}/optimal_route.png")
plt.close()

# 2. 모든 방법 비교
fig, axes = plt.subplots(1, 3, figsize=(18, 6))

for idx, method in enumerate(methods):
    ax = axes[idx]
    route = routes[method]
    
    im = ax.contourf(lon_grid, lat_grid, speed_masked, levels=20, cmap='Blues', alpha=0.5, extend='both')
    ax.contour(lon_grid, lat_grid, mask, levels=[0.5], colors='black', linewidths=1)
    # 육지 영역 채우기
    land_mask = np.where(mask == 0, 1, np.nan)
    ax.contourf(lon_grid, lat_grid, land_mask, levels=[0.5, 1.5], colors='lightgray', alpha=0.6, zorder=1)
    
    # 경로
    route_lons = [area_trash.iloc[i]['LON'] for i in route]
    route_lats = [area_trash.iloc[i]['LAT'] for i in route]
    route_lons.append(route_lons[0])
    route_lats.append(route_lats[0])
    
    # 경로 선과 화살표
    for i in range(len(route_lons) - 1):
        ax.plot([route_lons[i], route_lons[i+1]], [route_lats[i], route_lats[i+1]], 
               'r-', linewidth=2, alpha=0.8, zorder=4)
        ax.annotate('', xy=(route_lons[i+1], route_lats[i+1]), 
                   xytext=(route_lons[i], route_lats[i]),
                   arrowprops=dict(arrowstyle='->', lw=1.5, color='red', alpha=0.8),
                   zorder=5)
    
    # 지점 표시
    ax.scatter(area_trash['LON'], area_trash['LAT'], 
              s=area_trash['TOTAL_TRASH']*1.5, c=area_trash['TOTAL_TRASH'],
              cmap='Reds', alpha=0.7, edgecolors='darkred', linewidths=1, zorder=5)
    
    # 시작점 표시
    ax.scatter(route_lons[0], route_lats[0], s=200, c='green', marker='*', 
              edgecolors='darkgreen', linewidths=2, zorder=6)
    
    # 순서 번호 표시
    for order, idx in enumerate(route, 1):
        row = area_trash.iloc[idx]
        ax.scatter(row['LON'], row['LAT'], s=300, c='yellow', 
                  edgecolors='black', linewidths=1.5, zorder=7, alpha=0.9)
        ax.annotate(str(order), (row['LON'], row['LAT']), 
                   fontsize=10, fontweight='bold', ha='center', va='center',
                   color='black', zorder=8)
    
    ax.set_xlim(lon.min(), lon.max())
    ax.set_ylim(lat.min(), lat.max())
    ax.set_title(f'{method.upper()}\nCost: {costs[method]:.2f}', fontsize=11, fontweight='bold')
    ax.set_xlabel('Longitude (°E)', fontsize=9)
    ax.set_ylabel('Latitude (°N)', fontsize=9)

plt.suptitle('Route Optimization Methods Comparison', fontsize=14, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig(f"{output_dir}/route_comparison.png", dpi=300, bbox_inches='tight')
print(f"✓ Route comparison map saved: {output_dir}/route_comparison.png")
plt.close()

# 결과 저장
with open(f"{output_dir}/optimization_results.txt", "w", encoding="utf-8") as f:
    f.write("=" * 80 + "\n")
    f.write("해류 기반 쓰레기 수거 최적화 동선 결과\n")
    f.write("=" * 80 + "\n\n")
    
    f.write("1. 최적 경로 정보\n")
    f.write("-" * 80 + "\n")
    f.write(f"방법: {best_method.upper()}\n")
    f.write(f"총 비용: {best_cost:.2f}\n\n")
    
    f.write("경로 순서:\n")
    total_distance = 0
    for i in range(len(best_route)):
        current_idx = best_route[i]
        next_idx = best_route[(i + 1) % len(best_route)]
        
        area_name = area_trash.iloc[current_idx]['AREA']
        trash_count = area_trash.iloc[current_idx]['TOTAL_TRASH']
        distance = optimizer.distance_matrix[current_idx, next_idx] / 1000  # km
        
        f.write(f"{i+1}. {area_name} (Trash: {trash_count} items)")
        if i < len(best_route) - 1:
            f.write(f" -> Distance: {distance:.2f} km\n")
        else:
            f.write(f" -> Return to start: {distance:.2f} km\n")
        
        total_distance += distance
    
    f.write(f"\nTotal travel distance: {total_distance:.2f} km\n\n")
    
    f.write("2. All Methods Comparison\n")
    f.write("-" * 80 + "\n")
    for method in methods:
        f.write(f"{method.upper()}: Cost {costs[method]:.2f}\n")
    
    f.write("\n3. Algorithm Description\n")
    f.write("-" * 80 + "\n")
    f.write("- GREEDY: Greedy algorithm considering cost and trash quantity\n")
    f.write("- NEAREST: Sequentially visit the nearest points\n")
    f.write("- PRIORITY: Visit in order of trash quantity (highest first)\n")
    f.write("\nCost calculation factors:\n")
    f.write("- Distance (30%%): Straight-line distance between points\n")
    f.write("- Current assistance (40%%): Time savings when riding ocean currents\n")
    f.write("- Trash quantity weight (20%%): Priority to areas with more trash\n")
    f.write("- Vorticity weight (10%%): Priority to eddy regions\n")

print(f"\n✓ Optimization results saved: {output_dir}/optimization_results.txt")

nc.close()

print("\n" + "=" * 80)
print("Optimization completed!")
print(f"All results saved in '{output_dir}' folder.")
print("=" * 80)

