
class Document:
    """
    Represents a document with text and a name.
    """

    def __init__(self, text, name):
        """
        Initializes a new Document instance.

        Args:
            text (str): The text content of the document.
            name (str): The name of the document.
        """
        self.text = text
        self.name = name
    