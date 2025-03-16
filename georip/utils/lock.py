from collections import deque
from time import sleep


class Lock:
    """
    A custom lock implementation that ensures exclusive access through an ID-based queueing mechanism.

    This lock allows multiple processes/threads to request access sequentially. Each acquire call
    receives a unique ID, and the lock can only be freed using the correct ID, ensuring strict order.

    Attributes:
        _locked (bool): Indicates whether the lock is currently acquired.
        _queue (Deque): A queue to maintain the order of lock acquisition requests.
        _next_id (int): The next ID to assign when `acquire` is called.
    """

    def __init__(self):
        """
        Initialize the Lock instance.

        - Sets the lock state to unlocked.
        - Initializes an empty queue to track lock requests.
        - Starts the ID counter at 0.
        """
        self._locked = False
        self._queue = deque()
        self._next_id = 0

    def acquire(self):
        """
        Request access to the lock.

        - Adds the current request ID to the queue.
        - Waits until the lock is free and this request ID is next in line.
        - Acquires the lock and returns the assigned ID.

        Returns:
            int: A unique ID representing the lock acquisition request.

        Example:
            >>> lock = Lock()
            >>> id1 = lock.acquire()  # Process acquires lock
        """
        id = self._next_id
        self._queue.append(id)
        self._next_id += 1

        while self.is_locked() or self._queue[0] != id:
            sleep(1 / 50)

        self._lock()
        return id

    def free(self, id):
        """
        Release the lock if the provided ID matches the next request in the queue.

        Parameters:
            id (int): The ID assigned during the lock acquisition.

        Raises:
            ValueError: If the provided ID does not match the next request in the queue.

        Example:
            >>> lock.free(id1)  # Free the lock using the correct ID
        """
        if self._queue[0] != id:
            raise ValueError("Attempt to free unlocked lock using invalid ID")

        self._queue.popleft()
        self._unlock()

        if self._queue:
            self._next_id = self._queue[-1] + 1
        else:
            self._next_id = 0

    def is_locked(self):
        """
        Check whether the lock is currently acquired.

        Returns:
            bool: True if the lock is acquired, False otherwise.

        Example:
            >>> lock.is_locked()
            False
        """
        return self._locked

    def _lock(self):
        """
        Internal method to acquire the lock.

        - Ensures the lock is not already locked before setting `_locked` to True.

        Raises:
            AssertionError: If the lock is already locked.
        """
        assert not self._locked, "Lock is already locked"
        self._locked = True

    def _unlock(self):
        """
        Internal method to release the lock.

        - Ensures the lock is currently locked before setting `_locked` to False.

        Raises:
            AssertionError: If the lock is already unlocked.
        """
        assert self._locked, "Lock is already unlocked"
        self._locked = False
