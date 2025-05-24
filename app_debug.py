# ✅ debug_test.py (VSCode에서 실행만 하면 됨)
import asyncio
import pandas as pd
from app import predict_and_schedule_split, CombinedInput

# 테스트 입력 데이터 구성 (예시)
lstm_input = [{"year": 2025, "month": 5, "day": 18, "hour": h, 
               "solar battery_kW": 1.0, "instantaneous_generation_kW": 0.5, "region": 1} for h in range(24)]

ppo_input = [{"year": 2025, "month": 5, "day": 18, "hour": h,
              "total load forecast": 1.1, "total load actual": 1.0,
              "price day ahead": 0.2, "price actual": 0.25} for h in range(24)]

# 요청 객체 생성
test_data = CombinedInput(lstm_input=lstm_input, ppo_input=ppo_input)

# FastAPI의 POST 엔드포인트 함수 직접 호출 (비동기)
result = asyncio.run(predict_and_schedule_split(test_data))

# 결과 출력
print(result)