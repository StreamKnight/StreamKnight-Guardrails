class Engine:
    def __init__(self, guards):
        self.guards = guards

    def run(self, text):
        # All guards must pass
        return all(guard.validate(text) for guard in self.guards)
