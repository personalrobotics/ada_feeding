"""
This module defines a number of helper functions that are reused throughout the
Ada Feeding project.
"""

# Standard imports
from typing import Any

# Third-party imports
import py_trees


def get_from_blackboard_with_default(
    blackboard: py_trees.blackboard.Client, key: str, default: Any
) -> Any:
    """
    Gets a value from the blackboard, returning a default value if the key is not set.

    Parameters
    ----------
    blackboard: The blackboard client.
    key: The key to get.
    default: The default value to return if the key is not set.

    Returns
    -------
    value: The value of the key if it is set, otherwise the default value.
    """
    try:
        return getattr(blackboard, key)
    except KeyError:
        return default


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
