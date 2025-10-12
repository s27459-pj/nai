from __future__ import annotations

from easyAI.AI.Negamax import Negamax
from easyAI.Player import AI_Player, Human_Player

from game import BestGameEverMade


def main() -> None:
    ai = Negamax(7)
    game = BestGameEverMade([Human_Player(), AI_Player(ai)])

    try:
        _ = game.play()
    except KeyboardInterrupt:
        print("\nBye")
        pass


if __name__ == "__main__":
    main()
