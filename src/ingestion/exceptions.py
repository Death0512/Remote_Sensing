class IngestionError(Exception):
    pass


class InvalidMetadataError(IngestionError):
    pass


class InvalidGeoreferenceError(InvalidMetadataError):
    pass


class DataUnavailableError(IngestionError):
    pass


class DataCorruptedError(IngestionError):
    pass
