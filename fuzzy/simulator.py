"""
See README.md for running instructions, game rules and authors.
"""

from time import sleep
from typing import NoReturn, final

from control import compute

DEFAULT_INITIAL_TEMPERATURE = 31.0
DEFAULT_INITIAL_HUMIDITY = 80.0
DEFAULT_INITIAL_TARGET_TEMPERATURE = 21.0


def prompt_initial_value(name: str, default: float) -> float:
    while True:
        user_input = input(f"Enter initial {name} (leave empty for default {default}): ")

        if user_input.strip() == "":
            return default

        try:
            return float(user_input)
        except ValueError:
            print(f"Invalid {name} value. Please enter a valid number.")


@final
class Simulation:
    def __init__(self, temperature: float, humidity: float, target_temperature: float):
        self.temperature = temperature
        self.humidity = humidity
        self.target_temperature = target_temperature
        self.iteration = 0

    def run(self) -> NoReturn:
        while True:
            mode, fan_speed = compute(self.temperature, self.humidity, self.target_temperature)

            if mode < 1.0:
                print(
                    f"Mode: Adjusting temperature from {self.temperature:.2f} to {self.target_temperature:.2f}"
                )
                diff = self.target_temperature - self.temperature
                # Simulate adjusting the temperature by 10%
                self.temperature += diff * 0.05
            else:
                print(f"Mode: Dehumidifying from {self.humidity:.2f}")
                # Simulate dehumidifying by 1%
                self.humidity -= self.humidity * 0.01

            self.iteration += 1
            self.print_state()
            print()
            sleep(1)

    def print_state(self) -> None:
        print(f"[{self.iteration:>3}]", end=" ")
        print(f"Temperature: {self.temperature:.2f}", end="   ")
        print(f"Humidity: {self.humidity:.2f}", end="   ")
        print(f"Target Temperature: {self.target_temperature}")


def main() -> None:
    temperature = prompt_initial_value("temperature", DEFAULT_INITIAL_TEMPERATURE)
    humidity = prompt_initial_value("humidity", DEFAULT_INITIAL_HUMIDITY)
    target_temperature = prompt_initial_value(
        "target temperature", DEFAULT_INITIAL_TARGET_TEMPERATURE
    )

    simulation = Simulation(temperature, humidity, target_temperature)
    print(f"Starting simulation with {temperature=}, {humidity=}, {target_temperature=}")
    print()

    try:
        simulation.run()
    except KeyboardInterrupt:
        print(f"Stopping simulation after {simulation.iteration} iterations")
        pass


if __name__ == "__main__":
    main()
