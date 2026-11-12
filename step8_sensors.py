import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import IsolationForest

rng = np.random.default_rng(7)
times = pd.date_range("2026-11-12 00:00", periods=288, freq="5min")   # 24 hours
hours = np.array(times.hour + times.minute / 60)

# Normal behaviour of three simulated sensors
pressure = 3.5 + 0.15 * np.sin(2 * np.pi * hours / 24) + rng.normal(0, 0.05, 288)   # pipe pressure, bar
power = 400 + 150 * np.exp(-((hours - 19) ** 2) / 6) + 80 * np.exp(-((hours - 9) ** 2) / 6) \
        + rng.normal(0, 8, 288)                                                      # transformer load, kW
water = 5 + rng.normal(0, 0.8, 288)                                                  # water in a drain, cm

# Faults we inject (the system does not know these windows)
FAULTS = {"Pipe pressure": (150, 162), "Power load": (210, 220), "Drain water level": (100, 130)}
pressure[150:162] -= np.linspace(0.5, 1.6, 12)    # slow pipe burst
power[210:220] *= 1.55                            # transformer overload
water[100:130] += np.linspace(0, 40, 30)          # drain gets blocked


def detect(values, contamination=0.06):
    s = pd.Series(values)
    feats = pd.DataFrame({
        "value": s,
        "delta": s.diff().fillna(0),
        "dev": s - s.rolling(24, min_periods=1).median(),
    })
    model = IsolationForest(contamination=contamination, random_state=0)
    return model.fit_predict(feats) == -1


signals = {"Pipe pressure": (pressure, "bar"), "Power load": (power, "kW"),
           "Drain water level": (water, "cm")}
fig, axes = plt.subplots(3, 1, figsize=(12, 9), sharex=True)
for ax, (name, (values, unit)) in zip(axes, signals.items()):
    flags = detect(values)
    a, b = FAULTS[name]
    caught = int(flags[a:b].sum())
    false_alarms = int(flags.sum() - caught)
    print(f"{name}: caught {caught} of {b - a} fault readings, {false_alarms} false alarms")
    ax.plot(times, values, color="#4c78a8", linewidth=1)
    ax.scatter(times[flags], np.array(values)[flags], color="red", s=18, label="anomaly flagged")
    ax.axvspan(times[a], times[b - 1], color="orange", alpha=0.2, label="real fault (injected)")
    ax.set_ylabel(f"{name} ({unit})")
    ax.legend(loc="upper left")
plt.tight_layout()
plt.savefig("sensor_anomalies.png", dpi=130)

pd.DataFrame({"time": times, "pipe_pressure_bar": pressure, "power_load_kw": power,
              "drain_water_cm": water}).to_csv("sensor_data.csv", index=False)
print("Saved sensor_anomalies.png and sensor_data.csv")