"""
한글 폰트 설정 유틸리티
"""
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import platform

def setup_korean_font():
    """한글 폰트 설정"""
    system = platform.system()
    
    if system == 'Windows':
        # Windows에서 사용 가능한 한글 폰트 찾기
        font_list = ['Malgun Gothic', 'NanumGothic', 'NanumBarunGothic', 
                    'Gulim', 'Batang', 'Gungsuh']
        
        for font_name in font_list:
            try:
                # 폰트가 설치되어 있는지 확인
                font_path = fm.findfont(fm.FontProperties(family=font_name))
                if font_path:
                    plt.rcParams['font.family'] = font_name
                    plt.rcParams['axes.unicode_minus'] = False  # 마이너스 기호 깨짐 방지
                    print(f"Korean font set to: {font_name}")
                    return True
            except:
                continue
        
        # 기본 설정
        plt.rcParams['font.family'] = 'Malgun Gothic'
        plt.rcParams['axes.unicode_minus'] = False
        print("Using default Korean font: Malgun Gothic")
        return True
    
    elif system == 'Darwin':  # macOS
        plt.rcParams['font.family'] = 'AppleGothic'
        plt.rcParams['axes.unicode_minus'] = False
        print("Korean font set to: AppleGothic")
        return True
    
    else:  # Linux
        plt.rcParams['font.family'] = 'NanumGothic'
        plt.rcParams['axes.unicode_minus'] = False
        print("Korean font set to: NanumGothic")
        return True

if __name__ == "__main__":
    setup_korean_font()
    print("\nAvailable fonts:")
    fonts = [f.name for f in fm.fontManager.ttflist]
    korean_fonts = [f for f in fonts if any(k in f for k in ['Gothic', 'Malgun', 'Nanum', 'Gulim', 'Batang'])]
    for font in sorted(set(korean_fonts))[:10]:
        print(f"  - {font}")


