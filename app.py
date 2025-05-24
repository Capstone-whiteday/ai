from fastapi import FastAPI
from pydantic import BaseModel
from typing import List, Dict
import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model
import os
from stable_baselines3 import PPO
import zipfile
import gym
from gym import spaces

app = FastAPI()

# 모델 로딩
# ✅ compile=False: 추론 전용으로 로딩
lstm_model_path = "solar_predict_model.h5"
lstm_model = load_model(
    lstm_model_path,
    compile=False
) if os.path.exists(lstm_model_path) else None

ppo_model_path = "ppo_solar_model.zip"
ppo_model = PPO.load(ppo_model_path) if os.path.exists(ppo_model_path) else None

# ✅ 요청 스키마
class CombinedInput(BaseModel):
    lstm_input: List[Dict]
    ppo_input: List[Dict]

# 요청 스키마
class CombinedInput(BaseModel):
    lstm_input: List[Dict]
    ppo_input: List[Dict]

# ✅ PPO 환경 정의
class EnergyChargingEnv(gym.Env):
    def __init__(self, df):
        super().__init__()
        self.df = df.reset_index(drop=True)
        self.max_steps = len(self.df)
        self.current_step = 0
        self.battery_capacity = 10.0
        self.battery_level = np.random.uniform(2.0, 8.0)
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32)
        self.observation_space = spaces.Box(low=0, high=1.0, shape=(5,), dtype=np.float32)

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.current_step = 0
        self.battery_level = np.random.uniform(2.0, 8.0)
        obs = self._get_state()
        return obs, {}
    
    def _get_state(self):
        if self.current_step >= len(self.df):
            return np.zeros(self.observation_space.shape, dtype=np.float32)
        row = self.df.iloc[self.current_step]
        return np.array([
            row['total load forecast'],
            row['total load actual'],
            row['price day ahead'],
            row['price actual'],
            self.battery_level / self.battery_capacity,
        ], dtype=np.float32)

    def step(self, action):
        if self.current_step >= len(self.df):
            return self._get_state(), 0.0, True, False, {}

        action_val = float(np.clip(action[0], -1.0, 1.0))
        row = self.df.iloc[self.current_step]
        predict_e = max(row['predict_solar_norm'], 0.01) * 1000
        price = row['price actual'] * 100
        real_action = np.clip(action_val * 0.5, -0.5, 0.5)
        soc = self.battery_level / self.battery_capacity

        reward, energy_delta = 0.0, 0.0
        if real_action > 0.05 and soc < 1.0:
            energy_delta = min(real_action * predict_e, self.battery_capacity - self.battery_level)
            self.battery_level += energy_delta
            reward = -energy_delta * price * 0.1 + 2.0
        elif real_action < -0.05 and soc > 0:
            energy_delta = min(abs(real_action) * predict_e, self.battery_level)
            self.battery_level -= energy_delta
            reward = energy_delta * price * 0.1 + 0.1
        else:
            reward = -0.05

            self.current_step += 1
        terminated = self.current_step >= self.max_steps
        return self._get_state(), reward, terminated, False, {}

# ✅ 통합 API
@app.post("/predict_and_schedule_split")
async def predict_and_schedule_split(data: CombinedInput):
    if lstm_model is None or ppo_model is None:
        return {"error": "모델이 로딩되지 않았습니다."}

    # 1️⃣ LSTM 예측
    df_lstm = pd.DataFrame(data.lstm_input)
    try:
        X = df_lstm.drop(columns=["hour", "region"]).values.reshape(1, 24, -1)
        predict_solar = lstm_model.predict(X).flatten()
    except Exception as e:
        return {"error": f"LSTM 입력 오류: {str(e)}"}

    # 2️⃣ PPO 입력 병합
    df_ppo = pd.DataFrame(data.ppo_input)
    if len(df_ppo) != 24:
        return {"error": "PPO 입력은 24시간 데이터여야 합니다."}
    df_ppo["predict_solar"] = predict_solar

    # ✅ 정규화 (hour만 유지)
    df_ppo["hour_norm"] = df_ppo["hour"] / 23.0
    df_ppo["predict_solar_norm"] = df_ppo["predict_solar"] / 5.0

     # 3️⃣ PPO 스케줄링
    env = EnergyChargingEnv(df_ppo)
    obs, _ = env.reset()

    schedule = []
    for i in range(24):
        action, _ = ppo_model.predict(obs, deterministic=True)
        obs, reward, terminated, _, _ = env.step(action)

        real_action = float(np.clip(action[0], -1.0, 1.0) * 0.5)
        power_kw = abs(real_action) * max(df_ppo["predict_solar"][i], 0.01)
        power_payment = power_kw * df_ppo["price actual"][i]

        schedule.append({
            "hour": int(df_ppo["hour"][i]),
            "predict_solar": round(df_ppo["predict_solar"][i], 4),
            "action": round(real_action, 3),
            "powerKw": round(power_kw, 4),
            "powerpayment": round(power_payment, 2)
        })

        if terminated:
            break

    return {"result": schedule}