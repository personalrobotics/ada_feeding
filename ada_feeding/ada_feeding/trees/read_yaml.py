import yaml

with open('../../config/feeding_goal_config.yaml', 'r') as file:
    parameter_service = yaml.safe_load(file)
location_goal = parameter_service['above_plate']
print(location_goal)
