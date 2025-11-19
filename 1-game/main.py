"""
See README.md for running instructions, game rules and authors.
"""

from __future__ import annotations

from easyAI.AI.Negamax import Negamax
from easyAI.Player import AI_Player, Human_Player

from game import BestGameEverMade


def main() -> None:
    ai = Negamax(7)
    game = BestGameEverMade([Human_Player(), AI_Player(ai)])

    hist = game.play()

    last_game_state = hist[-1]
    assert isinstance(last_game_state, BestGameEverMade)

    if last_game_state.player_1_score > last_game_state.player_2_score:
        print("Player 1 wins!")
    elif last_game_state.player_1_score < last_game_state.player_2_score:
        print("Player 2 wins!")
    else:
        print("It's a tie!")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nBye")
        pass
