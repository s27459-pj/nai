import numpy as np
from skfuzzy import control as ctrl

# AC with 3 inputs:
# - temperature (current)
# - humidity (current)
# - target_temperature
# Output: power (0-100)

temperature = ctrl.Antecedent(np.arange(17, 31, 1), "temperature")
humidity = ctrl.Antecedent(np.arange(0, 101, 1), "humidity")
target_temperature = ctrl.Antecedent(np.arange(17, 31, 1), "target_temperature")
power = ctrl.Consequent(np.arange(0, 101, 1), "power")

temperature.automf(3, names=["low", "medium", "high"])
humidity.automf(3, names=["low", "medium", "high"])
target_temperature.automf(3, names=["low", "medium", "high"])
power.automf(3, names=["low", "medium", "high"])


def get_rules() -> list[ctrl.Rule]:
    return [
        # Target meets the actual temperature - don't run the AC
        ctrl.Rule(temperature["low"] & target_temperature["low"], power["low"]),
        ctrl.Rule(temperature["medium"] & target_temperature["medium"], power["low"]),
        ctrl.Rule(temperature["high"] & target_temperature["high"], power["low"]),
        # Run on medium power if temperature is not too far off
        ctrl.Rule(temperature["medium"] & target_temperature["low"], power["medium"]),
        ctrl.Rule(temperature["high"] & target_temperature["medium"], power["medium"]),
        # Run on high power if temperature is too far off
        ctrl.Rule(temperature["high"] & target_temperature["low"], power["high"]),
    ]


ac_ctrl = ctrl.ControlSystem(get_rules())

ac = ctrl.ControlSystemSimulation(ac_ctrl)


def compute(temperature: float, humidity: float, target_temperature: float) -> float:
    ac.input["temperature"] = temperature
    # ac.input["humidity"] = humidity
    ac.input["target_temperature"] = target_temperature
    ac.compute()
    return ac.output["power"]


def main() -> None:
    for i in range(17, 31):
        print(f"temp: 30, target: {i}, power:", compute(30, 0, i))


if __name__ == "__main__":
    main()
