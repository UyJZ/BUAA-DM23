import os
import pandas as pd

script_dir = os.path.dirname(os.path.realpath(__file__))
script_dir = os.path.join(script_dir, '..')
graphPath = os.path.join(script_dir, 'database', 'data', 'road_graph.gpickle')
traj_path = os.path.join(script_dir, 'database','data', 'traj.csv')
fmm_path = os.path.join(script_dir, 'database','fmm', 'fmm_all_fields.csv')
grouped_traj_path = os.path.join(script_dir, 'database', 'output', 'grouped_traj.csv')
road_path = os.path.join(script_dir, 'database', 'data', 'road.csv')

output_path_holiday_0 = os.path.join(script_dir, 'database', 'output', 'speeds_holiday_0.csv')
output_path_holiday_1 = os.path.join(script_dir, 'database', 'output', 'speeds_holiday_1.csv')
result_output_path = os.path.join(script_dir, 'database', 'output', 'result_df.csv')
merged_output_path = os.path.join(script_dir, 'database', 'output', 'merged_df.csv')
train_set_path_holiday_0 = os.path.join(script_dir, 'database', 'output', 'train_set_holiday_0.csv')
train_set_path_holiday_1 = os.path.join(script_dir, 'database', 'output', 'train_set_holiday_1.csv')
model_holiday_0_path = os.path.join(script_dir, 'database', 'output', 'random_forest_model_holiday_0.pkl')
model_holiday_1_path = os.path.join(script_dir, 'database', 'output', 'random_forest_model_holiday_1.pkl')
fmm_out_path = os.path.join(script_dir, 'database', 'output', 'fmm_df.csv')
predict_path = os.path.join(script_dir, 'database', 'output', 'predict_df.csv')
grouped_traj_output_path = os.path.join(script_dir, 'database', 'output', 'grouped_traj.csv')
grid_assignment_path = os.path.join(script_dir, 'database', 'ETA', 'grid_assignment.csv')
neighboring_roads_path = os.path.join(script_dir, 'database', 'ETA', 'neighboring_roads.csv')