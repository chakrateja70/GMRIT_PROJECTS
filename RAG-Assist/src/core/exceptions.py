class DocumentFolderNotFoundException(Exception):
    """Raised when the document folder is not found."""
    def __init__(self, folder_path: str):
        self.folder_path = folder_path
        super().__init__(f"Document folder not found: {folder_path}")

class DocumentProcessingException(Exception):
    """Raised when there is an error in document processing."""
    pass

class NoChunksToEmbedException(Exception):
    """Raised when there are no chunks to embed."""
    def __init__(self):
        super().__init__("No valid chunks found to create embeddings")


class EmbeddingModelException(Exception):
    """Raised when the embedding model encounters an error."""
    pass

class PineconeInitializationException(Exception):
    """Raised when Pinecone initialization fails."""
    pass
class PineconeQueryException(Exception):
    """Raised when a Pinecone query fails."""
    pass


class PineconeUpsertException(Exception):
    """Raised when a Pinecone upsert operation fails."""
    pass

class LLMServiceAPIException(Exception):
    """Raised when there is an API error with the LLM service."""
    pass

class LLMServiceUnexpectedException(Exception):
    """Raised when there is an unexpected error with the LLM service."""
    pass