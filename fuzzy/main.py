import numpy as np
from skfuzzy import control as ctrl
from skfuzzy.membership import trimf

# AC with 3 inputs:
# - temperature (current)
# - humidity (current)
# - target_temperature
# 2 Outputs: 
# - mode (adjust_temperature, dehumidify)
# - fan_speed (0-100)

temperature = ctrl.Antecedent(np.arange(0, 35, 1), "temperature")
humidity = ctrl.Antecedent(np.arange(0, 101, 1), "humidity")
target_temperature = ctrl.Antecedent(np.arange(17, 31, 1), "target_temperature")

fan_speed = ctrl.Consequent(np.arange(0, 101, 1), "fan_speed")
mode = ctrl.Consequent(np.arange(0, 4.1, 0.1), "mode")

temperature.automf(5, names=["cold", "cool", "comfortable", "warm", "hot"])
humidity.automf(3, names=["dry", "comfortable", "sticky"])
target_temperature.automf(3, names=["low", "medium", "high"])

fan_speed["very_slow"] = trimf(fan_speed.universe, [0, 0, 30])
fan_speed["slow"]      = trimf(fan_speed.universe, [20, 35, 50])
fan_speed["standard"]  = trimf(fan_speed.universe, [40, 55, 70])
fan_speed["fast"]      = trimf(fan_speed.universe, [60, 75, 90])
fan_speed["very_fast"] = trimf(fan_speed.universe, [70, 90, 100])

mode["adjust_temperature"] = trimf(mode.universe, [0, 0, 2])
mode["dehumidify"] = trimf(mode.universe, [2, 4, 4])


def get_rules() -> list[ctrl.Rule]:
    return [
        # Rules for mode
        ctrl.Rule(humidity["dry"] | humidity["comfortable"], mode["adjust_temperature"]),
        ctrl.Rule(humidity["sticky"], mode["dehumidify"]),

        # Rules for fan_speed based on temperature and target_temperature
        ctrl.Rule(temperature["cold"] & target_temperature["low"], fan_speed["very_fast"]),
        ctrl.Rule(temperature["cool"] & target_temperature["low"], fan_speed["standard"]),
        ctrl.Rule(temperature["comfortable"] & target_temperature["low"], fan_speed["very_slow"]),
        ctrl.Rule(temperature["warm"] & target_temperature["low"], fan_speed["standard"]),
        ctrl.Rule(temperature["hot"] & target_temperature["low"], fan_speed["fast"]),

        ctrl.Rule(temperature["cold"] & target_temperature["medium"], fan_speed["very_fast"]),
        ctrl.Rule(temperature["cool"] & target_temperature["medium"], fan_speed["fast"]),
        ctrl.Rule(temperature["comfortable"] & target_temperature["medium"], fan_speed["standard"]),
        ctrl.Rule(temperature["warm"] & target_temperature["medium"], fan_speed["slow"]),
        ctrl.Rule(temperature["hot"] & target_temperature["medium"], fan_speed["very_fast"]),

        ctrl.Rule(temperature["cold"] & target_temperature["high"], fan_speed["very_fast"]),
        ctrl.Rule(temperature["cool"] & target_temperature["high"], fan_speed["very_fast"]),
        ctrl.Rule(temperature["comfortable"] & target_temperature["high"], fan_speed["fast"]),
        ctrl.Rule(temperature["warm"] & target_temperature["high"], fan_speed["standard"]),
        ctrl.Rule(temperature["hot"] & target_temperature["high"], fan_speed["slow"]),

        # Rules for mode influencing fan_speed
        ctrl.Rule(mode["dehumidify"], fan_speed["very_fast"] % 0.6),
        ctrl.Rule(mode["adjust_temperature"], fan_speed["standard"] % 0.4),
    ]


ac_ctrl = ctrl.ControlSystem(get_rules())

ac = ctrl.ControlSystemSimulation(ac_ctrl)


def compute(temperature: float, humidity: float, target_temperature: float) -> tuple[float, float]:
    ac.input["temperature"] = temperature
    ac.input["humidity"] = humidity
    ac.input["target_temperature"] = target_temperature
    ac.compute()

    fan_speed = ac.output["fan_speed"]
    mode = ac.output["mode"]

    return mode, fan_speed


# 144 results, for better visualization redirect main.py output to a txt file
# Also, mode is displayed as quantitative value instead of category label
def main() -> None:
    temperatures = [6, 12, 18, 24, 30, 34]
    humidities = [15, 30, 45, 60, 75, 90]
    targets = [18, 22, 26, 30]

    print(f"{'Temp':>6} {'Hum':>6} {'Target':>8} {'Mode':>12} {'Fan Speed':>12}")
    print("-" * 50)

    for t in temperatures:
        for h in humidities:
            for target in targets:
                mode, fan_speed = compute(t, h, target)
                print(f"{t:6.1f} {h:6.1f} {target:8.1f} {mode:12.2f} {fan_speed:12.2f}")


if __name__ == "__main__":
    main()
