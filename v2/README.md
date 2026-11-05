# ADA Feeding 2.0

ADA Feeding 2.0 rebuilds this system on [ssrobot](https://github.com/personalrobotics/ssrobot). It is developed here, alongside the current system at the repository root, the way [Geodude 4.0](https://github.com/personalrobotics/geodude/tree/main/v4) is developed alongside Geodude 3.x. It replaces the current system once it covers the same feeding experience (Milestone 5). It has no code yet, and Milestone 1 adds the first.

## Boundaries

| Part | Where it lives | Rule |
|---|---|---|
| Feeding app, meaning the feeding service and its web UI | `v2/` (Milestone 1) | No ROS clients, topics, or message types |
| ssrobot | [personalrobotics/ssrobot](https://github.com/personalrobotics/ssrobot), pinned by commit | Used only through its public interfaces, never copied in |
| ADA integration, meaning ADA's robot package and runtime profiles | [personalrobotics/ada_assets](https://github.com/personalrobotics/ada_assets), and profiles in `v2/` (Milestone 2) | Robot-specific values live here, not in the feeding app |
| ROS adapter, which turns ssrobot commands into calls to ADA's ROS 2 system | `v2/`, as its own package (Milestone 2) | The only part of `v2/` that imports ROS |
| Current system | The repository root and `feeding_web_interface/` | Receives fixes but not redesigns until cutover, and `v2/` never imports it |
| Experiments | [`experiments/`](../experiments) | May import anything, and nothing imports it |
