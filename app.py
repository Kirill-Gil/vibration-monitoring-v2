"""
Система комплексного онлайн-мониторинга и вибродиагностики
электроприводных насосных агрегатов ДНС 0.4-6 кВ.

Модуль ИИ: Интеграция с OpenRouter API (free tier) с кэшированием,
троттлингом по событиям и бесшовным fallback на локальные правила.
"""

import asyncio
from collections import deque
from datetime import datetime, timedelta
import io
import json
import math
import os
import time
from typing import Any, Dict, List, Optional
from urllib.error import HTTPError, URLError
from urllib.request import Request as URLRequest, urlopen

from fastapi import FastAPI, File, HTTPException, Request, UploadFile
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
import numpy as np
from pydantic import BaseModel
from scipy import signal
from scipy.io import loadmat

# Автоматическая загрузка .env файла без обязательной зависимости python-dotenv
def load_env_file():
    env_path = os.path.join(os.path.dirname(__file__), ".env")
    if os.path.exists(env_path):
        with open(env_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    k, v = line.split("=", 1)
                    k = k.strip()
                    v = v.strip().strip('"').strip("'")
                    if k not in os.environ:
                        os.environ[k] = v

load_env_file()

app = FastAPI(title="Industrial DNS Vibration & MCSA Diagnostic Platform")
templates = Jinja2Templates(directory="templates")

# ==========================================
# 1. ПАСПОРТ ОБОРУДОВАНИЯ
# ==========================================
SUPPORTS = ["ED_NDE", "ED_DE", "PUMP_DE", "PUMP_NDE"]
CHANNELS = ["H", "V", "A"]

class MachinePassport:
    def __init__(self):
        self.motor_name = "Асинхронный ЭД А4-400Х-4МУ3"
        self.voltage_kv = 6.0
        self.power_kw = 630.0
        self.nominal_current = 75.0
        self.pole_pairs = 2
        self.rated_rpm = 1796.0

        # SKF 6205
        self.bearing_type = "SKF 6205"
        self.nb_balls = 9
        self.ball_diameter_mm = 7.94
        self.pitch_diameter_mm = 39.04
        self.contact_angle_deg = 0.0

        # ГОСТ Р ИСО 20816-1/3
        self.iso_warn = 2.8
        self.iso_alarm = 4.5
        self.iso_catastrophic = 6.75

    def calculate_kinematic_frequencies(self, rpm: float) -> Dict[str, float]:
        fr = rpm / 60.0
        gamma = (self.ball_diameter_mm / self.pitch_diameter_mm) * math.cos(math.radians(self.contact_angle_deg))
        bpfo = (self.nb_balls / 2.0) * fr * (1.0 - gamma)
        bpfi = (self.nb_balls / 2.0) * fr * (1.0 + gamma)
        bsf = (self.pitch_diameter_mm / (2.0 * self.ball_diameter_mm)) * fr * (1.0 - gamma**2)
        ftf = (fr / 2.0) * (1.0 - gamma)
        return {
            "fr": round(fr, 2),
            "BPFO": round(bpfo, 2),
            "BPFI": round(bpfi, 2),
            "BSF": round(bsf, 2),
            "FTF": round(ftf, 2),
        }

passport = MachinePassport()

# ==========================================
# 2. БУФЕРЫ ТЕЛЕМЕТРИИ И ДЕДУПЛИКАЦИЯ
# ==========================================
class StorageManager:
    def __init__(self):
        self.daily_telemetry = deque(maxlen=17280)  # 24ч с шагом 5 сек
        self.last_history_save = 0.0
        self.incidents_30d: List[Dict[str, Any]] = []
        self.active_faults: Dict[str, Dict[str, Any]] = {}

    def add_telemetry_point(self, point: Dict[str, Any]):
        now = time.time()
        if now - self.last_history_save >= 5.0:
            self.daily_telemetry.append({
                "timestamp": datetime.fromtimestamp(now).strftime("%H:%M:%S"),
                "max_rms": point["max_rms"],
                "temp_max": point["temp_max"],
                "current_l1": point["currents"]["L1"],
                "status": point["machine_status"]
            })
            self.last_history_save = now

    def process_incident(self, fault_key: Optional[str], severity: str, description: str, unit: str, value: float):
        now_dt = datetime.now()
        if not fault_key or severity == "NORMAL":
            for active_k in list(self.active_faults.keys()):
                for inc in reversed(self.incidents_30d):
                    if inc["fault_key"] == active_k and inc["cleared_time"] is None:
                        inc["cleared_time"] = now_dt.strftime("%d.%m.%Y %H:%M:%S")
                        inc["duration_sec"] = int((now_dt - inc["dt_obj"]).total_seconds())
                        break
                del self.active_faults[active_k]
            return

        # Дедупликация: запись создается 1 раз при переходе порога
        if fault_key not in self.active_faults:
            incident_record = {
                "id": len(self.incidents_30d) + 1,
                "fault_key": fault_key,
                "timestamp": now_dt.strftime("%d.%m.%Y %H:%M:%S"),
                "dt_obj": now_dt,
                "severity": severity,
                "unit": unit,
                "description": description,
                "value": round(value, 2),
                "cleared_time": None,
                "duration_sec": None
            }
            self.incidents_30d.append(incident_record)
            self.active_faults[fault_key] = incident_record

            cutoff = now_dt - timedelta(days=30)
            self.incidents_30d = [inc for inc in self.incidents_30d if inc["dt_obj"] >= cutoff]

storage = StorageManager()

# ==========================================
# 3. ЦОС И ФИЛЬТРАЦИЯ
# ==========================================
class SignalProcessor:
    def __init__(self):
        self.cw = 0
        self.ca = 0
        self.confirmed_status = "NORMAL"
        self.temp_history = {sup: deque(maxlen=60) for sup in SUPPORTS}
        self.last_temp_calc = time.time()
        self.dt_dt_values = {sup: 0.0 for sup in SUPPORTS}

    def spatial_noise_cancellation(self, signals_dict: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        v_common = (signals_dict["ED_DE_V"] + signals_dict["PUMP_DE_V"] +
                    signals_dict["ED_NDE_V"] + signals_dict["PUMP_NDE_V"]) / 4.0
        cleaned = {}
        for ch_name, sig in signals_dict.items():
            if ch_name.endswith("_V"):
                cleaned[ch_name] = sig - 0.7 * v_common
            else:
                cleaned[ch_name] = sig
        return cleaned

    def compute_envelope(self, raw_accel: np.ndarray, fs: int = 12000, band=(2000, 5000), max_freq=500.0):
        nyq = 0.5 * fs
        low = min(max(band[0] / nyq, 0.01), 0.98)
        high = min(max(band[1] / nyq, low + 0.01), 0.99)

        b, a = signal.butter(4, [low, high], btype='band')
        filtered = signal.filtfilt(b, a, raw_accel)

        analytic = signal.hilbert(filtered)
        envelope = np.abs(analytic)
        envelope -= np.mean(envelope)

        n = len(envelope)
        fft_vals = np.fft.rfft(envelope * signal.windows.hann(n))
        freqs = np.fft.rfftfreq(n, d=1.0/fs)
        amplitudes = (2.0 / n) * np.abs(fft_vals)

        mask = freqs <= max_freq
        f_sub = freqs[mask]
        a_sub = amplitudes[mask]

        kinematics = passport.calculate_kinematic_frequencies(passport.rated_rpm)
        detected_peaks = {}
        for name in ["BPFO", "BPFI", "BSF", "FTF"]:
            target_f = kinematics[name]
            f_win = (f_sub >= target_f * 0.98) & (f_sub <= target_f * 1.02)
            detected_peaks[name] = float(np.max(a_sub[f_win])) if np.any(f_win) else 0.0

        max_a = np.max(a_sub) if len(a_sub) > 0 and np.max(a_sub) > 0 else 1.0
        return f_sub.tolist(), (a_sub / max_a).tolist(), detected_peaks

    def compute_mcsa(self, current_sig: np.ndarray, fs: int = 4000):
        n = len(current_sig)
        fft_vals = np.fft.rfft(current_sig * signal.windows.hann(n))
        freqs = np.fft.rfftfreq(n, d=1.0/fs)
        power = (2.0 / n) * np.abs(fft_vals)

        mask_50 = (freqs >= 48.0) & (freqs <= 52.0)
        idx_f0 = np.argmax(power[mask_50]) if np.any(mask_50) else 0
        f0 = freqs[mask_50][idx_f0] if np.any(mask_50) else 50.0
        p0 = power[mask_50][idx_f0] if np.any(mask_50) else 1.0

        fr = passport.rated_rpm / 60.0
        s = max(0.001, (f0 - passport.pole_pairs * fr) / f0 if f0 > 0 else 0.015)
        fb_left = f0 * (1.0 - 2 * s)
        fb_right = f0 * (1.0 + 2 * s)

        def get_db(f_target):
            w = (freqs >= f_target - 0.5) & (freqs <= f_target + 0.5)
            if np.any(w):
                return 20.0 * math.log10(max(np.max(power[w]) / (p0 + 1e-6), 1e-4))
            return -80.0

        delta_l = max(get_db(fb_left), get_db(fb_right))
        mask_plot = (freqs >= 35.0) & (freqs <= 65.0)
        p_norm = power / (p0 + 1e-6)
        db_plot = 20.0 * np.log10(np.maximum(p_norm[mask_plot], 1e-4))
        return freqs[mask_plot].tolist(), db_plot.tolist(), delta_l, round(s, 4)

    def update_temperatures(self, raw_temps: Dict[str, float]) -> Dict[str, float]:
        now = time.time()
        dt_sec = max(now - self.last_temp_calc, 1.0)
        self.last_temp_calc = now

        for sup, val in raw_temps.items():
            self.temp_history[sup].append((now, val))
            if len(self.temp_history[sup]) >= 2:
                t_old, val_old = self.temp_history[sup][0]
                delta_t = now - t_old
                if delta_t >= 2.0:
                    self.dt_dt_values[sup] = round(((val - val_old) / delta_t) * 60.0, 2)
        return self.dt_dt_values

    def apply_persistence_filter(self, raw_status: str, max_rms: float) -> str:
        if max_rms >= passport.iso_catastrophic:
            self.confirmed_status = "ALARM"
            return "ALARM"

        if raw_status == "ALARM":
            self.ca += 1
            self.cw += 1
        elif raw_status == "WARNING":
            self.cw += 1
            self.ca = 0
        else:
            if max_rms < 0.9 * passport.iso_warn:
                self.cw = 0
                self.ca = 0
                self.confirmed_status = "NORMAL"

        if self.ca >= 2:
            self.confirmed_status = "ALARM"
        elif self.cw >= 3 and self.confirmed_status != "ALARM":
            self.confirmed_status = "WARNING"

        return self.confirmed_status

processor = SignalProcessor()

# ==========================================
# 4. ТЕХНОЛОГИЧЕСКИЙ ДВИЖОК
# ==========================================
class RuntimeEngine:
    def __init__(self):
        self.sim_load_pct = 85.0
        self.active_defect = "NORMAL"

    def generate_hardware_stream(self) -> Dict[str, Any]:
        fs = 12000
        duration = 0.5
        t = np.linspace(0, duration, int(fs * duration), endpoint=False)
        fr = passport.rated_rpm / 60.0
        kinematics = passport.calculate_kinematic_frequencies(passport.rated_rpm)

        raw_channels = {}
        for sup in SUPPORTS:
            for ch in CHANNELS:
                name = f"{sup}_{ch}"
                noise = 0.08 * np.random.randn(len(t))
                harmonic_1x = 0.15 * np.sin(2 * np.pi * fr * t)
                raw_channels[name] = noise + harmonic_1x

        if self.active_defect == "BEARING_OUTER":
            period = int(fs / kinematics["BPFO"])
            pulses = np.zeros_like(t)
            pulses[::period] = 1.2
            res = np.exp(-1200 * t[:int(fs * 0.01)]) * np.sin(2 * np.pi * 3150 * t[:int(fs * 0.01)])
            raw_channels["ED_DE_V"] += 2.5 * np.convolve(pulses, res, mode='same')
        elif self.active_defect == "COUPLING_MISALIGN":
            harmonic_2x = 0.9 * np.sin(2 * np.pi * 2 * fr * t)
            raw_channels["ED_DE_H"] += harmonic_2x
            raw_channels["PUMP_DE_H"] += 0.85 * harmonic_2x
            raw_channels["ED_DE_A"] += 0.7 * harmonic_2x

        cleaned_channels = processor.spatial_noise_cancellation(raw_channels)

        rms_values = {}
        for name, sig in cleaned_channels.items():
            vel = np.cumsum(sig) * (9810.0 / fs)
            vel -= np.mean(vel)
            rms = float(np.sqrt(np.mean(vel**2))) * (self.sim_load_pct / 100.0)
            if self.active_defect == "BEARING_OUTER" and "ED_DE" in name:
                rms += 3.8
            elif self.active_defect == "COUPLING_MISALIGN" and ("ED_DE" in name or "PUMP_DE" in name):
                rms += 2.4
            rms_values[name] = round(rms, 2)

        t_curr = np.linspace(0, duration, int(4000 * duration), endpoint=False)
        base_current = passport.nominal_current * (self.sim_load_pct / 100.0)
        i_l1 = base_current * math.sqrt(2) * np.sin(2 * np.pi * 50.0 * t_curr) + 0.5 * np.random.randn(len(t_curr))

        if self.active_defect == "ROTOR_BAR":
            s = (50.0 - passport.pole_pairs * fr) / 50.0
            fb = 50.0 * (1.0 - 2 * s)
            i_l1 += 0.18 * base_current * np.sin(2 * np.pi * fb * t_curr)

        temps = {
            "ED_NDE": 52.0 + 10.0 * (self.sim_load_pct / 100.0),
            "ED_DE": 58.0 + 12.0 * (self.sim_load_pct / 100.0),
            "PUMP_DE": 61.0 + 14.0 * (self.sim_load_pct / 100.0),
            "PUMP_NDE": 54.0 + 10.0 * (self.sim_load_pct / 100.0)
        }
        if self.active_defect == "THERMAL_LOCK":
            temps["PUMP_DE"] += 28.5

        dt_dt = processor.update_temperatures(temps)
        f_env, a_env, detected_peaks = processor.compute_envelope(cleaned_channels["ED_DE_V"], fs=fs)
        f_mcsa, db_mcsa, delta_l, slip = processor.compute_mcsa(i_l1, fs=4000)

        max_rms = max(rms_values.values())
        raw_status = "ALARM" if max_rms > passport.iso_alarm else ("WARNING" if max_rms > passport.iso_warn else "NORMAL")
        machine_status = processor.apply_persistence_filter(raw_status, max_rms)

        fault_key = None
        desc = ""
        unit = "Опора 1"
        if self.active_defect == "BEARING_OUTER" or detected_peaks["BPFO"] > 0.30:
            fault_key = "FAULT_BEARING_BPFO"
            desc = f"Разрушение наружного кольца подшипника привода ЭД (BPFO {kinematics['BPFO']} Гц)"
            unit = "Опора 2 (ЭД-DE)"
        elif self.active_defect == "COUPLING_MISALIGN":
            fault_key = "FAULT_COUPLING"
            desc = "Механическая несоосность валов двигателя и насоса (перекос муфты)"
            unit = "Муфта (Опоры 2-3)"
        elif self.active_defect == "ROTOR_BAR" or delta_l > -35.0:
            fault_key = "FAULT_ROTOR"
            desc = f"Повреждение стержней беличьей клетки ротора (MCSA боковая {round(delta_l, 1)} дБ)"
            unit = "Ротор ЭД"
        elif self.active_defect == "THERMAL_LOCK" or max(dt_dt.values()) > 2.0:
            fault_key = "FAULT_THERMAL"
            desc = f"Острое смазочное голодание подшипника (dT/dt={max(dt_dt.values())} °C/мин)"
            unit = "Опора 3 (Насос-DE)"
        elif machine_status in ["WARNING", "ALARM"]:
            fault_key = "FAULT_HIGH_VIBRATION"
            desc = f"Превышение предельных уровней виброскорости ГОСТ ({max_rms} мм/с)"
            unit = max(rms_values, key=rms_values.get)

        storage.process_incident(fault_key, machine_status, desc, unit, max_rms)

        telemetry_snapshot = {
            "timestamp": datetime.now().strftime("%H:%M:%S"),
            "max_rms": max_rms,
            "temp_max": max(temps.values()),
            "raw_status": raw_status,
            "machine_status": machine_status,
            "filter_cw": processor.cw,
            "filter_ca": processor.ca,
            "rms_channels": rms_values,
            "temperatures": temps,
            "dt_dt": dt_dt,
            "currents": {
                "L1": round(float(np.mean(np.abs(i_l1)) * 1.11), 1),
                "L2": round(float(np.mean(np.abs(i_l1)) * 1.10), 1),
                "L3": round(float(np.mean(np.abs(i_l1)) * 1.12), 1)
            },
            "slip": slip,
            "delta_l_mcsa": round(delta_l, 1),
            "envelope": {
                "freqs": f_env[::4],
                "amps": a_env[::4],
                "kinematics": kinematics,
                "detected_peaks": {k: round(v, 3) for k, v in detected_peaks.items()}
            },
            "mcsa_chart": {
                "freqs": f_mcsa,
                "db": db_mcsa
            },
            "active_fault": storage.active_faults.get(fault_key),
            "fault_key": fault_key
        }

        storage.add_telemetry_point(telemetry_snapshot)
        return telemetry_snapshot

engine = RuntimeEngine()

# =========================================================================
# 5. НОВЫЙ ИИ-ВАЛИДАТОР: OPENROUTER API С КЭШЕМ И ТРОТТЛИНГОМ (50 ЗАПРОСОВ)
# =========================================================================
class AIValidator:
    def __init__(self):
        self.mode = "OPENROUTER"  # 'OPENROUTER' или 'LOCAL'
        self.model = os.getenv("OPENROUTER_MODEL", "meta-llama/llama-3.2-3b-instruct:free")
        self.endpoint = "https://openrouter.ai/api/v1/chat/completions"

        # Кэш результатов анализа
        self.cached_verdict = "Инициализация аналитического блока..."
        self.cached_source = "LOCAL_EXPERT"
        self.last_analysis_time = 0.0
        self.last_status = "NORMAL"
        self.last_defect_key = None
        self.daily_requests_count = 0
        self.last_counter_date = datetime.now().date()
        self.is_fetching = False

    def _reset_counter_if_new_day(self):
        today = datetime.now().date()
        if today != self.last_counter_date:
            self.daily_requests_count = 0
            self.last_counter_date = today

    def get_local_verdict(self, snapshot: Dict[str, Any]) -> str:
        """Детерминированная экспертная оценка по таблице 3 методички."""
        status = snapshot["machine_status"]
        max_v = snapshot["max_rms"]
        max_t = snapshot["temp_max"]
        dt_dt_max = max(snapshot["dt_dt"].values())
        active_f = snapshot.get("active_fault")

        if status == "NORMAL":
            return (
                f"✅ [Локальные правила] Агрегат {passport.motor_name} работает в пределах нормы. "
                f"Максимальная виброскорость {max_v} мм/с (Зона A/B ГОСТ Р ИСО 20816). "
                f"Тепловой баланс стабилен: {max_t:.1f} °C, dT/dt={dt_dt_max} °C/мин. "
                f"Электрические дефекты ротора отсутствуют (боковые гармоники тока < -45 дБ). "
                "Рекомендация: Продолжать непрерывную эксплуатацию по регламенту CBM."
            )
        else:
            defect_name = active_f["description"] if active_f else "Превышение вибрационных норм"
            unit_name = active_f["unit"] if active_f else "Контур насоса"
            action = "Немедленный технологический останов" if status == "ALARM" else "Внеочередная центровка/виброналадка"
            return (
                f"⚠️ [Локальные правила] Зафиксировано отклонение! Статус: {status}. "
                f"Максимальная виброскорость: {max_v} мм/с. "
                f"Дефект: {defect_name} (Узел: {unit_name}). "
                f"Предписание: {action}."
            )

    def _call_openrouter_sync(self, prompt: str, api_key: str) -> str:
        """Синхронный сетевой вызов к OpenRouter (выполняется в пуле потоков)."""
        payload = {
            "model": self.model,
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "Ты — ведущий эксперт по надежности и вибродиагностике насосных агрегатов 0.4-6 кВ на объектах нефтедобычи. "
                        "Оценивай показатели по ГОСТ Р ИСО 20816-1-2021 и методу спектра огибающей. "
                        "Отвечай строго на русском языке, технически грамотно и предельно лаконично (3-4 предложения). "
                        "Обязательно дай диагноз узла и четкое предписание дежурному инженеру."
                    )
                },
                {"role": "user", "content": prompt}
            ],
            "max_tokens": 250,
            "temperature": 0.2
        }

        req = URLRequest(
            self.endpoint,
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
                "HTTP-Referer": "http://localhost:8000",
                "X-Title": "DNS Motor Diagnostics Platform"
            }
        )

        with urlopen(req, timeout=8.0) as response:
            res_data = json.loads(response.read().decode("utf-8"))
            return res_data["choices"][0]["message"]["content"].strip()

    async def analyze_snapshot(self, snapshot: Dict[str, Any], force: bool = False) -> Dict[str, Any]:
        """
        Интеллектуальная маршрутизация с троттлингом:
        - Возвращает кэш, если прошло мало времени и статус не менялся.
        - Вызывает OpenRouter только при смене статуса или раз в 30 сек.
        - Fallback на локальные правила при отсутствии ключа или лимитах.
        """
        self._reset_counter_if_new_day()
        now = time.time()
        current_status = snapshot["machine_status"]
        current_defect = snapshot.get("fault_key")

        # Проверка необходимости обращения к нейросети
        status_changed = (current_status != self.last_status) or (current_defect != self.last_defect_key)
        time_elapsed = now - self.last_analysis_time
        min_cooldown = 15.0  # Защита от спама: не чаще раза в 15 секунд при дребезге
        periodic_trigger = (time_elapsed >= 30.0)

        should_call_llm = force or (status_changed and time_elapsed >= min_cooldown) or periodic_trigger

        # Если вызывать ИИ не требуется или фоновый запрос уже идет — отдаем свежий кэш
        if not should_call_llm or self.is_fetching:
            return {
                "source": self.cached_source,
                "verdict": self.cached_verdict,
                "is_anomaly": current_status != "NORMAL",
                "requests_today": self.daily_requests_count,
                "model": self.model,
                "quota_left": max(0, 50 - self.daily_requests_count)
            }

        # Проверяем наличие ключа OpenRouter
        api_key = os.getenv("OPENROUTER_API_KEY", "").strip()

        # Если ключ не задан или выбран локальный режим — используем правила
        if self.mode == "LOCAL" or not api_key:
            self.cached_source = "LOCAL_EXPERT (Ключ OpenRouter не задан)" if not api_key else "LOCAL_EXPERT"
            self.cached_verdict = self.get_local_verdict(snapshot)
            self.last_analysis_time = now
            self.last_status = current_status
            self.last_defect_key = current_defect
            return {
                "source": self.cached_source,
                "verdict": self.cached_verdict,
                "is_anomaly": current_status != "NORMAL",
                "requests_today": self.daily_requests_count,
                "model": self.model,
                "quota_left": 50 - self.daily_requests_count
            }

        # Если суточный лимит 50 запросов исчерпан — переходим на локальные правила
        if self.daily_requests_count >= 50:
            self.cached_source = "LOCAL_EXPERT (Суточный лимит OpenRouter 50 исчерпан)"
            self.cached_verdict = self.get_local_verdict(snapshot)
            self.last_analysis_time = now
            return {
                "source": self.cached_source,
                "verdict": self.cached_verdict,
                "is_anomaly": current_status != "NORMAL",
                "requests_today": self.daily_requests_count,
                "model": self.model,
                "quota_left": 0
            }

        # Формируем компактный промпт для LLM
        prompt = (
            f"Телеметрия насосного агрегата: {passport.motor_name} ({passport.power_kw} кВт, {passport.voltage_kv} кВ, {passport.rated_rpm} об/мин). "
            f"Макс. вибрация: {snapshot['max_rms']} мм/с (Норма <= {passport.iso_warn}, Авария > {passport.iso_alarm}). "
            f"Макс. температура: {snapshot['temp_max']} °C, скорость роста: {max(snapshot['dt_dt'].values())} °C/мин. "
            f"MCSA токов: {snapshot['delta_l_mcsa']} дБ (скольжение s={snapshot['slip']}). "
            f"Пики спектра огибающей подшипника (g): {snapshot['envelope']['detected_peaks']}. "
            f"Аппаратный статус системы: {current_status}. "
            f"Зафиксированный инцидент: {snapshot.get('active_fault', {}).get('description', 'Дефектов нет')}."
        )

        try:
            self.is_fetching = True
            # Асинхронный вызов в потоке, чтобы не тормозить 1-секундный поток телеметрии
            llm_text = await asyncio.to_thread(self._call_openrouter_sync, prompt, api_key)
            self.daily_requests_count += 1
            self.cached_source = f"OpenRouter [{self.model}]"
            self.cached_verdict = llm_text
            self.last_analysis_time = now
            self.last_status = current_status
            self.last_defect_key = current_defect
        except HTTPError as e:
            err_code = e.code
            err_msg = f"HTTP {err_code}"
            if err_code == 401:
                err_msg = "Неверный OPENROUTER_API_KEY (401)"
            elif err_code == 429:
                err_msg = "Превышен лимит запросов OpenRouter Rate Limit (429)"
            self.cached_source = f"LOCAL_EXPERT (Ошибка OpenRouter: {err_msg})"
            self.cached_verdict = self.get_local_verdict(snapshot)
            self.last_analysis_time = now
        except Exception as e:
            self.cached_source = f"LOCAL_EXPERT (Сбой сети: {type(e).__name__})"
            self.cached_verdict = self.get_local_verdict(snapshot)
            self.last_analysis_time = now
        finally:
            self.is_fetching = False

        return {
            "source": self.cached_source,
            "verdict": self.cached_verdict,
            "is_anomaly": current_status != "NORMAL",
            "requests_today": self.daily_requests_count,
            "model": self.model,
            "quota_left": max(0, 50 - self.daily_requests_count)
        }

ai_validator = AIValidator()

# ==========================================
# 6. REST API И МАРШРУТЫ
# ==========================================
class PassportUpdate(BaseModel):
    motor_name: str
    power_kw: float
    voltage_kv: float
    rated_rpm: float
    bearing_type: str
    nb_balls: int
    ball_diameter_mm: float
    pitch_diameter_mm: float
    contact_angle_deg: float
    iso_warn: float
    iso_alarm: float

class SimControl(BaseModel):
    sim_load_pct: float
    active_defect: str
    rated_rpm: float

class AIConfig(BaseModel):
    ai_mode: str
    openrouter_key: Optional[str] = None
    model_name: Optional[str] = None

@app.get("/", response_class=HTMLResponse)
async def get_dashboard(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.get("/api/telemetry")
async def get_telemetry():
    snapshot = engine.generate_hardware_stream()
    ai_analysis = await ai_validator.analyze_snapshot(snapshot, force=False)
    return {
        "telemetry": snapshot,
        "ai": ai_analysis,
        "passport": {
            "motor_name": passport.motor_name,
            "power_kw": passport.power_kw,
            "voltage_kv": passport.voltage_kv,
            "rated_rpm": passport.rated_rpm,
            "bearing_type": passport.bearing_type,
            "kinematics": passport.calculate_kinematic_frequencies(passport.rated_rpm)
        }
    }

@app.post("/api/ai/force_eval")
async def force_ai_evaluation():
    """Принудительный вызов анализа ИИ по кнопке пользователя."""
    snapshot = engine.generate_hardware_stream()
    ai_analysis = await ai_validator.analyze_snapshot(snapshot, force=True)
    return ai_analysis

@app.post("/api/ai/config")
async def configure_ai(cfg: AIConfig):
    ai_validator.mode = cfg.ai_mode
    if cfg.openrouter_key:
        os.environ["OPENROUTER_API_KEY"] = cfg.openrouter_key.strip()
    if cfg.model_name:
        ai_validator.model = cfg.model_name.strip()
    return {
        "status": "success",
        "mode": ai_validator.mode,
        "model": ai_validator.model,
        "has_key": bool(os.getenv("OPENROUTER_API_KEY"))
    }

@app.get("/api/history/24h")
async def get_24h_history():
    return list(storage.daily_telemetry)

@app.get("/api/incidents/30d")
async def get_30d_incidents():
    clean_list = []
    for inc in reversed(storage.incidents_30d):
        c = dict(inc)
        c.pop("dt_obj", None)
        clean_list.append(c)
    return clean_list

@app.post("/api/passport/update")
async def update_passport(data: PassportUpdate):
    passport.motor_name = data.motor_name
    passport.power_kw = data.power_kw
    passport.voltage_kv = data.voltage_kv
    passport.rated_rpm = data.rated_rpm
    passport.bearing_type = data.bearing_type
    passport.nb_balls = data.nb_balls
    passport.ball_diameter_mm = data.ball_diameter_mm
    passport.pitch_diameter_mm = data.pitch_diameter_mm
    passport.contact_angle_deg = data.contact_angle_deg
    passport.iso_warn = data.iso_warn
    passport.iso_alarm = data.iso_alarm
    passport.iso_catastrophic = data.iso_alarm * 1.5
    return {"status": "success"}

@app.post("/api/simulation/control")
async def control_simulation(data: SimControl):
    engine.sim_load_pct = data.sim_load_pct
    engine.active_defect = data.active_defect
    passport.rated_rpm = data.rated_rpm
    return {"status": "success"}

@app.post("/api/diagnose/file")
async def analyze_file(file: UploadFile = File(...)):
    content = await file.read()
    filename = file.filename.lower()
    try:
        if filename.endswith(".mat"):
            mat = loadmat(io.BytesIO(content))
            key = next((k for k in mat.keys() if any(x in k for x in ["DE_time", "BA_time", "FE_time"])), None)
            if not key:
                key = [k for k in mat.keys() if not k.startswith("__")][0]
            raw_data = mat[key].flatten()
        elif filename.endswith(".csv") or filename.endswith(".txt"):
            text = content.decode("utf-8")
            raw_data = np.genfromtxt(io.StringIO(text), delimiter=",")
            if raw_data.ndim > 1:
                raw_data = raw_data[:, -1]
        else:
            raise HTTPException(status_code=400, detail="Формат должен быть .mat или .csv")

        f_env, a_env, detected_peaks = processor.compute_envelope(raw_data, fs=12000)
        kinematics = passport.calculate_kinematic_frequencies(passport.rated_rpm)
        found_f = f_env[np.argmax(a_env)]
        error = abs(found_f - kinematics["BPFO"])

        return {
            "status": "success",
            "filename": file.filename,
            "points_count": len(raw_data),
            "calculated_bpfo": kinematics["BPFO"],
            "detected_frequency": round(found_f, 2),
            "deviation_hz": round(error, 2),
            "conclusion": f"Пик огибающей: {round(found_f,1)} Гц. Отклонение от BPFO: {round(error,2)} Гц."
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)