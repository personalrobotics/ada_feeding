"""
Helper functions for the nano_bridge package.
"""

# Standard imports
from typing import Any


def import_from_string(import_string: str) -> Any:
    """
    Imports a module from a string.

    Parameters
    ----------
    import_string: The string to import, e.g., "sensor_msgs.msg.CompressedImage"
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
            f'Invalid import string {import_string}. Except "package.module.class" '
            'e.g., "ada_feeding_msgs.action.MoveTo"'
        ) from exc
    try:
        return getattr(
            getattr(
                __import__(f"{import_package}.{import_module}"),
                import_module,
            ),
            import_class,
        )
    except Exception as exc:
        raise ImportError(f"Error importing {import_string}") from exc
