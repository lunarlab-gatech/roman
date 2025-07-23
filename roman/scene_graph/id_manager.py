from typeguard import typechecked

class IDManager:
    """ Manages identification numbers for graph nodes"""
    def __init__(self):
        self._next_id = 0
        #self._used_ids = set()
        #self._free_ids = set()

    @typechecked
    def assign_id(self) -> int:
        """Assign a new ID to a node."""
        # if self._free_ids:
        #     new_id = self._free_ids.pop()
        #else:
        new_id = self._next_id
        self._next_id += 1
        #self._used_ids.add(new_id)
        return new_id

    # @typechecked
    # def release_id(self, id_: int) -> None:
    #     """Release an ID back to the pool."""
    #     if id_ in self._used_ids:
    #         self._used_ids.remove(id_)
    #         #self._free_ids.add(id_)
    #     else:
    #         raise ValueError(f"ID {id_} is not currently in use.")

    # @typechecked
    # def is_used(self, id_: int) -> bool:
    #     """Check if an ID is currently in use."""
    #     return id_ in self._used_ids

    # @typechecked
    # def get_used_ids(self) -> set[int]:
    #     """Return a set of currently used IDs."""
    #     return set(self._used_ids)

    # @typechecked
    # def get_free_ids(self) -> set[int]:
    #     """Return a set of currently free IDs."""
    #     return set(self._free_ids)