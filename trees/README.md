# Behavior Tree Objects

## debug.xml

Use this file for Behavior Tree scratch that won't be picked up by Git. It has been added to the .gitignore file. Use `git add -f` to override.

## default.xml

Default tree to run when not running `feeding.launch`. Currently has a simple debug sequence.

## feeding.xml

Runs the Feeding Demo. See the [Miro board](https://miro.com/app/board/uXjVPF445sc=/) for a more thorough diagram.

## helpers.xml

Small 1-5 nodes trees for common operations, including:

* `ParamToWorld`: Given `objectName` (string), loads the following rosparam format into the Aikido World:
```
workspace:
  <objectName>:
    urdf: package://path/to/urdf.urdf
    pos: [x, y, z] # Relative to world origin
    quat: [w, x, y, z] # Relative to world origin
```
* `MoveToConfig`: Combines `PlanToConfig` and `ExecuteTraj`
* `MoveToOffset`: Combines `PlanToOffset` and `ExecuteTraj`
* `MoveToPose`: Combines `PlanToPose` and `ExecuteTraj`

## Tree TODOs

* Update `default.xml` to act as a `simple_trajectories` to be used to test system setup.
