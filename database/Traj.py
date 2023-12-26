import json

class Trajectory:
    def __init__(self, traj_id, data):
        self.traj_id = traj_id
        self.id_list = [item['id'] for item in data]
        self.time_list = [item['time'] for item in data]
        self.entity_id_list = [item['entity_id'] for item in data]
        self.coordinates_list = [item['coordinates'] for item in data]
        self.current_dis_list = [item['current_dis'] for item in data]
        self.speeds_list = [item['speeds'] for item in data]
        self.holidays_list = [item['holidays'] for item in data]

    def __str__(self):
        return f"Trajectory {self.traj_id}: {len(self.id_list)} data points"

# 从文件中读取 JSON 数据
with open('/mnt/f/codes/Python/BUAA-DM23/all_traj_data.json', 'r') as json_file:
    json_data = json.load(json_file)

# 创建 Trajectory 对象列表
trajectories = [Trajectory(item['traj_id'], item['data']) for item in json_data]

# 打印每个轨迹对象的信息
for traj in trajectories:
    print(traj.traj_id)
    print(traj.id_list)
    print(traj.time_list)
    print(traj.entity_id_list)
    # 其他属性的访问方式类似
