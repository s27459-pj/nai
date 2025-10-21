import numpy as np
from skfuzzy import control as ctrl

# AC with 3 inputs:
# - temperature (current)
# - humidity (current)
# - target_temperature
# 2 Outputs: 
# - mode (fan_only, heating, cooling, dehumidifying)
# - fan_speed (0-100)

temperature = ctrl.Antecedent(np.arange(0, 35, 1), "temperature")
humidity = ctrl.Antecedent(np.arange(0, 101, 1), "humidity")
target_temperature = ctrl.Antecedent(np.arange(17, 31, 1), "target_temperature")
mode = ctrl.Consequent(np.arange(0, 4, 1), "mode")
fan_speed = ctrl.Consequent(np.arange(0, 101, 1), "fan_speed")

temperature.automf(5, names=["cold", "cool", "comfortable", "warm", "hot"])
humidity.automf(3, names=["dry", "comfortable", "sticky"])
target_temperature.automf(3, names=["low", "medium", "high"])

# Note: "fan_only" mode is defined in the automf but is not used in the rules. Maybe
# we could define 2 membership functions manually instead of using automf.
mode.automf(3, names=["fan_only", "adjust_temperature", "dehumidify"])
fan_speed.automf(5, names=["very slow", "slow", "standard", "fast", "very fast"])


def get_rules() -> list[ctrl.Rule]:
    return [
        # Rules for mode
        ctrl.Rule(humidity["dry"] | humidity["comfortable"], mode["adjust_temperature"]),
        ctrl.Rule(humidity["sticky"], mode["dehumidify"]),

        # Rules for fan_speed based on temperature and target_temperature
        ctrl.Rule(temperature["cold"] & target temperature["low"], fan_speed["very fast"]),
        ctrl.Rule(temperature["cool"] & target_temperature["low"], fan_speed["standard"]),
        ctrl.Rule(temperature["comfortable"] & target_temperature["low"], fan_speed["very slow"]),
        ctrl.Rule(temperature["warm"] & target_temperature["low"], fan_speed["standard"]),
        ctrl.Rule(temperature["hot"] & target_temperature["low"], fan_speed["very fast"]),

        ctrl.Rule(temperature["cold"] & target temperature["medium"], fan_speed["very fast"]),
        ctrl.Rule(temperature["cool"] & target_temperature["medium"], fan_speed["fast"]),
        ctrl.Rule(temperature["comfortable"] & target_temperature["medium"], fan_speed["standard"]),
        ctrl.Rule(temperature["warm"] & target_temperature["medium"], fan_speed["slow"]),
        ctrl.Rule(temperature["hot"] & target_temperature["medium"], fan_speed["very fast"]),

        ctrl.Rule(temperature["cold"] & target temperature["high"], fan_speed["very fast"]),
        ctrl.Rule(temperature["cool"] & target_temperature["high"], fan_speed["very fast"]),
        ctrl.Rule(temperature["comfortable"] & target_temperature["high"], fan_speed["fast"]),
        ctrl.Rule(temperature["warm"] & target_temperature["high"], fan_speed["standard"]),
        ctrl.Rule(temperature["hot"] & target_temperature["high"], fan_speed["slow"]),

        # # Target meets the actual temperature - don't run the AC
        # ctrl.Rule(temperature["low"] & target_temperature["low"], fan_speed["low"]),
        # ctrl.Rule(temperature["medium"] & target_temperature["medium"], fan_speed["low"]),
        # ctrl.Rule(temperature["high"] & target_temperature["high"], fan_speed["low"]),
        # # Run on medium fan_speed if temperature is not too far fan_only
        # ctrl.Rule(temperature["medium"] & target_temperature["low"], fan_speed["medium"]),
        # ctrl.Rule(temperature["high"] & target_temperature["medium"], fan_speed["medium"]),
        # # Run on high fan_speed if temperature is too far fan_onlygit
        # ctrl.Rule(temperature["high"] & target_temperature["low"], fan_speed["high"]),
    ]


ac_ctrl = ctrl.ControlSystem(get_rules())

ac = ctrl.ControlSystemSimulation(ac_ctrl)


def compute(temperature: float, humidity: float, target_temperature: float) -> float:
    ac.input["temperature"] = temperature
    ac.input["humidity"] = humidity
    ac.input["target_temperature"] = target_temperature
    ac.compute()
    return ac.output["fan_speed"]


def main() -> None:
    for i in range(17, 31):
        print(f"temp: 30, target: {i}, fan_speed:", compute(30, 0, i))


if __name__ == "__main__":
    main()
