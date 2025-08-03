from typeguard import typechecked

class IDManager:
    """ Manages identification numbers for graph nodes"""
    def __init__(self):
        self._next_id = 0

    @typechecked
    def assign_id(self) -> int:
        """Assign a new ID to a node."""

        new_id = self._next_id
        self._next_id += 1
        return new_id

