class InvalidInputException(Exception):
    """
    Custom error for handling invalid inputs when loaded in a pydantic model
    """
    def __init__(self, value: str, message: str) -> None:
        self.value = value
        self.message = message
        super().__init__(message)
