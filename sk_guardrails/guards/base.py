class BaseGuard:
    def validate(self, text: str) -> bool:
        """Return True if text passes the guard."""
        raise NotImplementedError
