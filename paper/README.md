# 해류 데이터 기반 해양쓰레기 집적 예측 및 수거 최적화 연구

## 논문 파일

- `ocean_trash_current_analysis_paper.md`: 메인 논문 파일 (마크다운 형식)

## 논문 개요

본 논문은 동해 해역의 해류 데이터와 해양쓰레기 데이터를 결합하여 다음과 같은 연구를 수행하였다:

1. **해류 패턴 분석**: 소용돌이도(vorticity), 발산/수렴(divergence/convergence) 분석
2. **상관관계 분석**: 해류 변수와 쓰레기 집적량 간의 상관관계 분석
3. **집적 예측**: 해류 패턴을 활용한 쓰레기 집적 지역 예측
4. **경로 최적화**: 해류를 고려한 효율적인 쓰레기 수거 경로 최적화

## 주요 결과

- 해류 속도와 쓰레기량: **-0.280** (음의 상관관계)
- 소용돌이도와 쓰레기량: **0.349** (양의 상관관계)
- 수렴 지역: **704개 픽셀** 식별
- 최적 수거 경로: **860.49 km**

## 논문 구조

1. 서론: 연구 배경, 목적, 필요성
2. 데이터 및 방법론: 데이터셋 설명, 알고리즘, 수식
3. 결과 및 분석: 상관관계, 경로 최적화, 예측 결과
4. 논의: 결과 해석, 한계점, 향후 연구 방향
5. 결론: 연구 요약 및 기여도

## 사용된 데이터

- 해류 데이터: `KHOA_SCU_L4_Z004_D01_U20251118_EastSea.nc`
- 쓰레기 데이터: `eastsea_vessel_threat_coastallitter-2022.csv`

## 분석 스크립트

모든 분석은 다음 Python 스크립트로 수행되었다:
- `analyze_current_trash_correlation.py`: 상관관계 분석
- `optimize_trash_collection_route.py`: 경로 최적화
- `visualize_eddies.py`: 환류 시각화
- `advanced_trash_current_analysis.py`: 고급 분석
- `additional_meaningful_analysis.py`: 추가 분석

## 논문 작성 가이드

이 논문은 학술 논문 형식으로 작성되었으며, 다음을 포함한다:
- 초록 (Abstract)
- 서론 (Introduction)
- 데이터 및 방법론 (Data and Methodology)
- 결과 및 분석 (Results and Analysis)
- 논의 (Discussion)
- 결론 (Conclusion)
- 참고문헌 (References)
- 부록 (Appendix)

논문을 PDF로 변환하려면 Pandoc 또는 다른 마크다운 변환 도구를 사용할 수 있다.

