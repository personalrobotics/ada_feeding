# Behavior Tree Objects

## debug.xml

Use this file for Behavior Tree scratch that won't be picked up by Git. It will be added to the .gitignore file. Use `git add -f` to override.

## default.xml

Default tree to run when not running `feeding.launch`. Currently has a simple debug sequence. TODO (egordon): update to act as a `simple_trajectories` to be used to test system setup.

## feeding.xml

Runs the Feeding Demo. See the [Miro board](https://miro.com/app/board/uXjVPF445sc=/) for a more thorough diagram.

It relies on sub-trees in the `feeding` folder.

## Tree TODOs

* Small trees for moving the robot (i.e. combining planning and execute operations)
* Small tree wrapping WorldAddUpdate (see `feeding.xml`)
* Update trees to explicitly use BT_v4 root header
* Separate acquisition and transfer into different sub-tree files.
