from __future__ import annotations

from typing import TYPE_CHECKING, Literal, final, override

from easyAI import TwoPlayerGame
from easyAI.AI.Negamax import Negamax
from easyAI.Player import AI_Player, Human_Player

if TYPE_CHECKING:
    from easyAI.Player import Player
else:
    from easyAI import TwoPlayerGame as UntypedTwoPlayerGame

    # Add a stub generic specifier to the actual TwoPlayerGame implementation
    class TwoPlayerGame[Move](UntypedTwoPlayerGame): ...


type Move = Literal[1, 2, 3]


@final
class GameOfBones(TwoPlayerGame[Move]):
    def __init__(self, players: list[Player]) -> None:
        self.players = players
        self.current_player = 1
        self.pile = 20

    @override
    @classmethod
    def possible_moves(cls) -> list[Move]:
        return [1, 2, 3]

    @override
    def make_move(self, move: Move) -> None:
        self.pile -= move

    @override
    def is_over(self) -> bool:
        return self.pile <= 0

    def scoring(self) -> int:
        return 100 if self.pile <= 0 else 0

    def show(self) -> None:
        print(f"{self.pile} bones left in the pile")


def main() -> None:
    ai = Negamax(13)
    game = GameOfBones([Human_Player(), AI_Player(ai)])
    _ = game.play()


if __name__ == "__main__":
    main()
