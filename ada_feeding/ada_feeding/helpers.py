"""
This module defines a number of helper functions that are reused throughout the
Ada Feeding project.
"""

# Standard imports
from typing import Any


def import_from_string(import_string: str) -> Any:
    """
    Imports a module from a string.

    Parameters
    ----------
    import_string: The string to import, e.g., "ada_feeding_msgs.action.MoveTo"
        This should be in the format "package.module.class"

    Returns
    -------
    module: The imported module.

    Raises
    ------
    NameError: If the import string is invalid.
    ImportError: If the module cannot be imported.
    """
    try:
        import_package, import_module, import_class = import_string.split(".", 3)
    except Exception as exc:
        raise NameError(
            'Invalid import string %s. Except "package.module.class" e.g., "ada_feeding_msgs.action.MoveTo"'
            % import_string
        ) from exc
    try:
        return getattr(
            getattr(
                __import__("%s.%s" % (import_package, import_module)),
                import_module,
            ),
            import_class,
        )
    except Exception as exc:
        raise ImportError("Error importing %s" % (import_string)) from exc
