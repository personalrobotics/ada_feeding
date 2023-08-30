"""
This module defines a number of helper functions that are reused throughout the
Ada Feeding project.
"""

# Standard imports
from typing import Any, Set

# Third-party imports
import py_trees

# These prefixes are used to separate namespaces from each other.
# For example, if we have a behavior `foo` that has a position goal constraint,
# an orientaiton path constraint, and a move to behavior, the namespaces
# of each behavior (and their corresponding blackboards) will be:
#   - foo.position_goal_constraint
#   - foo.orientation_path_constraint
#   - foo.move_to
# Note that this norm can only be used if each MoveTo behavior has maximally
# one constraint of each type. When creating a behavior with multiple constraints
# of the same time, you'll have to create custom namespaces.
POSITION_GOAL_CONSTRAINT_NAMESPACE_PREFIX = "position_goal_constraint"
ORIENTATION_GOAL_CONSTRAINT_NAMESPACE_PREFIX = "orientation_goal_constraint"
JOINT_GOAL_CONSTRAINT_NAMESPACE_PREFIX = "joint_goal_constraint"
POSITION_PATH_CONSTRAINT_NAMESPACE_PREFIX = "position_path_constraint"
ORIENTATION_PATH_CONSTRAINT_NAMESPACE_PREFIX = "orientation_path_constraint"
JOINT_PATH_CONSTRAINT_NAMESPACE_PREFIX = "joint_path_constraint"
MOVE_TO_NAMESPACE_PREFIX = "move_to"


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


# pylint: disable=dangerous-default-value
# A mutable default value is okay since we don't change it in this function.
def set_to_blackboard(
    blackboard: py_trees.blackboard.Client,
    key: str,
    value: Any,
    keys_to_not_write_to_blackboard: Set[str] = set(),
) -> None:
    """
    Sets a value to the blackboard.

    Parameters
    ----------
    blackboard: The blackboard client.
    key: The key to set.
    value: The value to set.
    keys_to_not_write_to_blackboard: A set of keys that should not be written
        to the blackboard. Note that the keys need to be exact e.g., "move_to.cartesian,"
        "position_goal_constraint.tolerance," "orientation_goal_constraint.tolerance,"
        etc.
    """
    if key not in keys_to_not_write_to_blackboard:
        blackboard.set(key, value)


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
