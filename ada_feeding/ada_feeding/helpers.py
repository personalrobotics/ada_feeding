"""
This module defines a number of helper functions that are reused throughout the
Ada Feeding project.
"""

# Standard imports
from threading import Lock
from typing import Any, Optional, Set, Tuple

# Third-party imports
import py_trees
from py_trees.common import Access
from pymoveit2 import MoveIt2
from pymoveit2.robots import kinova
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.node import Node

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
CLEAR_CONSTRAINTS_NAMESPACE_PREFIX = "clear_constraints"
POSITION_GOAL_CONSTRAINT_NAMESPACE_PREFIX = "position_goal_constraint"
ORIENTATION_GOAL_CONSTRAINT_NAMESPACE_PREFIX = "orientation_goal_constraint"
JOINT_GOAL_CONSTRAINT_NAMESPACE_PREFIX = "joint_goal_constraint"
POSITION_PATH_CONSTRAINT_NAMESPACE_PREFIX = "position_path_constraint"
ORIENTATION_PATH_CONSTRAINT_NAMESPACE_PREFIX = "orientation_path_constraint"
JOINT_PATH_CONSTRAINT_NAMESPACE_PREFIX = "joint_path_constraint"
MOVE_TO_NAMESPACE_PREFIX = "move_to"


def get_moveit2_object(
    blackboard: py_trees.blackboard.Client,
    node: Optional[Node] = None,
) -> Tuple[MoveIt2, Lock]:
    """
    Gets the MoveIt2 object and its corresponding lock from the blackboard.
    If they do not exist on the blackboard, they are created.

    Parameters
    ----------
    blackboard: The blackboard client. Any blackboard client can be used, as
        this function: (a) uses absolute paths; and (b) registers the keys if
        they are not already registered.
    node: The ROS2 node that the MoveIt2 object should be associated with, if
        we need to create it from scratch. If None, this function will not create
        the MoveIt2 object if it doesn't exist, and will instead raise a KeyError.

    Returns
    -------
    moveit2: The MoveIt2 object.
    lock: The lock for the MoveIt2 object.

    Raises
    -------
    KeyError: if the MoveIt2 object does not exist and node is None.
    """
    # These Blackboard keys are used to store the single, global MoveIt2 object
    # and its corresponding lock. Note that it is important that these keys start with
    # a "/" because to indicate it is an absolute path, so all behaviors can access
    # the same object.
    moveit2_blackboard_key = "/moveit2"
    moveit2_lock_blackboard_key = "/moveit2_lock"

    # First, register the MoveIt2 object and its corresponding lock for READ access
    if not blackboard.is_registered(moveit2_blackboard_key, Access.READ):
        blackboard.register_key(moveit2_blackboard_key, Access.READ)
    if not blackboard.is_registered(moveit2_lock_blackboard_key, Access.READ):
        blackboard.register_key(moveit2_lock_blackboard_key, Access.READ)

    # Second, check if the MoveIt2 object and its corresponding lock exist on the
    # blackboard. If they do not, register the blackboard for WRITE access to those
    # keys and create them.
    try:
        moveit2 = blackboard.get(moveit2_blackboard_key)
        lock = blackboard.get(moveit2_lock_blackboard_key)
    except KeyError as exc:
        # If no node is passed in, raise an error.
        if node is None:
            raise KeyError("MoveIt2 object does not exist on the blackboard") from exc

        # If a node is passed in, create a new MoveIt2 object and lock.
        node.get_logger().info(
            "MoveIt2 object and lock do not exist on the blackboard. Creating them now."
        )
        blackboard.register_key(moveit2_blackboard_key, Access.WRITE)
        blackboard.register_key(moveit2_lock_blackboard_key, Access.WRITE)
        # TODO: Assess whether ReentrantCallbackGroup is necessary for MoveIt2.
        callback_group = ReentrantCallbackGroup()
        moveit2 = MoveIt2(
            node=node,
            joint_names=kinova.joint_names(),
            base_link_name=kinova.base_link_name(),
            end_effector_name="forkTip",
            group_name=kinova.MOVE_GROUP_ARM,
            callback_group=callback_group,
        )
        lock = Lock()
        blackboard.set(moveit2_blackboard_key, moveit2)
        blackboard.set(moveit2_lock_blackboard_key, lock)

    return moveit2, lock


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
