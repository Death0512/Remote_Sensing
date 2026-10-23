class ModelRegistryError(Exception):
    pass


class ModelUnavailableError(ModelRegistryError):
    pass


class ModelChecksumMismatchError(ModelRegistryError):
    pass
