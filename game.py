from __future__ import annotations

from enum import Enum
from typing import TYPE_CHECKING, Literal, final, override

from easyAI import TwoPlayerGame

if TYPE_CHECKING:
    from easyAI.Player import Player
    from easyAI.TwoPlayerGame import PlayerIndex
else:
    from easyAI import TwoPlayerGame as UntypedTwoPlayerGame

    # Add a stub generic specifier to the actual TwoPlayerGame implementation
    class TwoPlayerGame[Move](UntypedTwoPlayerGame): ...

# region Constants

# Amount of columns on the board
BOARD_COLUMNS = 7
# Amount of rows on the board
BOARD_ROWS = 5
# Amount of tokens each player starts the game with
INITIAL_PLAYER_TOKENS = 20

# endregion Constants

type Move = Literal[1, 2, 3, 4, 5, 6, 7]


class BoardElement(str, Enum):
    WHITE = "w"
    BLACK = "b"


@final
class BestGameEverMade(TwoPlayerGame[Move]):
    def __init__(self, players: list[Player], initial_player: PlayerIndex = 1) -> None:
        self.players = players
        self.current_player = initial_player
        # First dimension is the column, second dimension is the row
        # Columns are ordered from bottom to top
        self.board: list[list[BoardElement]] = [[] for _ in range(BOARD_COLUMNS)]
        self.player_1_remaining_tokens = INITIAL_PLAYER_TOKENS
        self.player_2_remaining_tokens = INITIAL_PLAYER_TOKENS
        self.player_1_score = 0
        self.player_2_score = 0

    @override
    def possible_moves(self) -> list[Move]:
        moves: list[Move] = [1, 2, 3, 4, 5, 6, 7]
        for idx, column in enumerate(self.board, start=1):
            if self._column_is_full(column):
                moves.remove(idx)  # pyright: ignore[reportArgumentType]
        return moves

    @override
    def make_move(self, move: Move) -> None:
        selected_column = self.board[move - 1]
        if len(selected_column) >= BOARD_ROWS:
            raise ValueError(f"Column {move - 1} is full")

        selected_column.append(self._current_player_board_element)
        self.use_current_player_token()

        added_at_index = len(selected_column) - 1
        cells_to_capture = self._find_tokens_to_capture(selected_column, added_at_index)
        if cells_to_capture is not None:
            for index in cells_to_capture:
                selected_column[index] = self._current_player_board_element

        cells_to_score = self._find_tokens_to_score(selected_column)
        if cells_to_score is not None:
            self.add_score_to_current_player(cells_to_score)
            for _ in range(cells_to_score):
                _ = selected_column.pop()

        # FIXME: In the case: (B, W, B, B), the `W` token is captured
        #        by the double `B` token, which shouldn't happen

    def _find_tokens_to_capture(
        self,
        column: list[BoardElement],
        below: int,
    ) -> list[int] | None:
        if below <= 1:
            return None

        below_added = column[:below]
        consecutive_opponent_tokens = 0
        for cell in reversed(below_added):
            if cell == self._current_player_board_element:
                consecutive_opponent_tokens = 0
            elif cell != self._current_player_board_element:
                consecutive_opponent_tokens += 1

            if consecutive_opponent_tokens == 2:
                return [below - 2, below - 1]

        return None

    def _find_tokens_to_score(self, column: list[BoardElement]) -> int | None:
        if len(column) <= 2:
            return None

        # TODO: Allow scoring horizontally

        consecutive_tokens = 1
        player = column[0]
        for cell in column[1:]:
            if cell == player:
                consecutive_tokens += 1
            elif cell != player:
                consecutive_tokens = 1

        if consecutive_tokens >= 3:
            return consecutive_tokens

        return None

    @property
    def _current_player_board_element(self) -> BoardElement:
        return BoardElement.WHITE if self.current_player == 1 else BoardElement.BLACK

    def add_score_to_current_player(self, score: int) -> None:
        if self.current_player == 1:
            self.player_1_score += score
        elif self.current_player == 2:
            self.player_2_score += score

    def use_current_player_token(self) -> None:
        if self.current_player == 1:
            self.player_1_remaining_tokens -= 1
        elif self.current_player == 2:
            self.player_2_remaining_tokens -= 1

    @override
    def is_over(self) -> bool:
        if self.player_1_remaining_tokens == 0 and self.player_2_remaining_tokens == 0:
            return True

        return self._board_is_full

    @property
    def _board_is_full(self) -> bool:
        return all(self._column_is_full(column) for column in self.board)

    @staticmethod
    def _column_is_full(column: list[BoardElement]) -> bool:
        return len(column) == BOARD_ROWS

    def scoring(self) -> int:
        if self.current_player == 1:
            return self.player_1_score
        if self.current_player == 2:
            return self.player_2_score
        raise ValueError(f"Invalid player: {self.current_player}")

    def show(self) -> None:
        # Player scores and left moves
        status_parts = [
            f"Player 1: {self.player_1_score} points ({self.player_1_remaining_tokens}/{INITIAL_PLAYER_TOKENS})",
            f"Player 2: {self.player_2_score} points ({self.player_2_remaining_tokens}/{INITIAL_PLAYER_TOKENS})",
        ]
        print(", ".join(status_parts))

        # Board header
        print("   ", end="")
        for col in range(BOARD_COLUMNS):
            print(col + 1, end="  ")
        print()

        # Board rows and columns
        for row in range(BOARD_ROWS):
            print(row + 1, end=" ")
            for col in range(BOARD_COLUMNS):
                column = self.board[col]
                value = column[row].value if len(column) > row else " "
                print(f"[{value}]", end="")
            print()
