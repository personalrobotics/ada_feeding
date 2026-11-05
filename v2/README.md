# ADA Feeding 2.0

ADA Feeding 2.0 rebuilds this system on [ssrobot](https://github.com/personalrobotics/ssrobot). It is developed here, alongside the current system at the repository root, the way [Geodude 4.0](https://github.com/personalrobotics/geodude/tree/main/v4) is developed alongside Geodude 3.x, and replaces it once it covers the same feeding experience (Milestone 5). It has no code yet; Milestone 1 adds the first.

## Boundaries

| Part | Where it lives | Rule |
|---|---|---|
| Feeding app: the feeding service and its web UI | `v2/` (Milestone 1) | No ROS clients, topics, or message types |
| ssrobot | [personalrobotics/ssrobot](https://github.com/personalrobotics/ssrobot), pinned by commit | Used only through its public interfaces, never copied in |
| ADA integration: ADA's robot package and runtime profiles | [personalrobotics/ada_assets](https://github.com/personalrobotics/ada_assets), and profiles in `v2/` (Milestone 2) | Robot-specific values live here, not in the feeding app |
| ROS adapter: ssrobot commands to ADA's ROS 2 system | `v2/`, as its own package (Milestone 2) | The only part of `v2/` that imports ROS |
| Current system | The repository root and `feeding_web_interface/` | Unchanged until cutover; `v2/` never imports it |
| Experiments | [`experiments/`](../experiments) | May import anything; nothing imports it |
