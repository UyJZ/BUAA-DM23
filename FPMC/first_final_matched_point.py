import pandas as pd
import numpy as np
import os

cwd = os.path.dirname(os.path.realpath(__file__))

df = pd.read_csv(cwd + '/../database/fmm/fmm_jump.csv', delimiter=';')
df = df.sort_values(by='id')

start_points = df['pgeom'].astype(str).apply(lambda x: x.split(',')[0].replace('LINESTRING(', ''))
final_points = df['pgeom'].astype(str).apply(lambda x: x.split(',')[-1].replace(')', ''))

start_points = start_points.apply(lambda x: tuple(map(float, x.split())))
final_points = final_points.apply(lambda x: tuple(map(float, x.split())))

valid_start_points = [(df.loc[i, 'id'], point) for i, point in enumerate(start_points.tolist()) if len(point) == 2]
valid_final_points = [(df.loc[i, 'id'], point) for i, point in enumerate(final_points.tolist()) if len(point) == 2]

# 合并成 first_final_matched_point
first_final_matched_point = np.column_stack((np.array([point[1][0] for point in valid_start_points]),
                                             np.array([point[1][1] for point in valid_final_points])))

# 打印结果
print("First and Final Matched Points:")
print(first_final_matched_point)


np.save(cwd+'/../database/ETA/point.npy', first_final_matched_point)