import heapq
from typeguard import typechecked

class IDManager:
    next_id: int
    free_ids: list[int]
    in_use: set

    def __init__(self):
        self.next_id = 0
        self.free_ids = []
        self.in_use = set()

    @typechecked
    def acquire(self) -> int:
        if len(self.free_ids) > 0:
            new_id = heapq.heappop(self.free_ids)
        else:
            new_id = self.next_id
            self.next_id += 1
        self.in_use.add(new_id)
        return new_id

    @typechecked
    def release(self, id: int):
        if id in self.in_use:
            self.in_use.remove(id)
            heapq.heappush(self.free_ids, id)
        else: 
            raise ValueError(f"Attempted release of id not in use: {id}")

    @typechecked
    def current_ids(self) -> list[int]:
        return sorted(self.in_use)