import csv


# 将fmm结果转化成FPMC的数据集
# 同一个车有多条路径
# entity_id => road_id road_id ...
# 生成训练测试集
# 用traj.csv建立entity_id和traj_id的关系
# 用fmm_all_field建立traj_id和road_id链
# 生成预测集
# 用jump_task建立entity_id和traj_id的关系
# 用fmm_jump.csv建立traj_id和road_id链的关系

def form_data(dirname, road_num):
    fmm_all_fields_f = dirname + '/' + 'fmm_all_fields.csv'
    traj_f = dirname + '/' + 'traj.csv'

    entity_id2traj_id = {}
    with open(traj_f, 'r') as file:
        csv_reader = csv.reader(file, delimiter=',')
        i = 0
        for row in csv_reader:
            if i == 0:
                i = i + 1
                continue
            entity_id = row[2]
            traj_id = row[3]

            if entity_id not in entity_id2traj_id.keys():
                entity_id2traj_id[entity_id] = {traj_id}
            else:
                entity_id2traj_id[entity_id].add(traj_id)
        file.close()

    traj_id2road_id = {}
    with open(fmm_all_fields_f, 'r') as file:
        csv_reader = csv.reader(file, delimiter=';')
        i = 0
        for row in csv_reader:
            if i > 0 and row[6] != '':
                str1 = row[6]
                if not ',' in row[6]:
                    str1 = str1 + ' ' + str1
                traj_id2road_id[row[0]] = str1
            i = i + 1
        file.close()

    with open(dirname + '/' + 'idxseq.txt', 'w') as file:
        for entity_id in entity_id2traj_id.keys():
            for traj_id in entity_id2traj_id[entity_id]:
                if traj_id in traj_id2road_id.keys():
                    file.write(entity_id + ' ' + traj_id + ' ' + traj_id2road_id[traj_id].replace(',', ' ') + '\n')

        file.close()

    with open(dirname + '/' + 'item_idx_list.txt', 'w') as file:
        file.write("\"Item_Index\"" + '\n')
        for i in range(0, road_num):
            file.write(str(i) + '\n')
        file.close()

    with open(dirname + '/' + 'user_idx_list.txt', 'w') as file:
        file.write("\"User_Index\"" + '\n')
        for i in entity_id2traj_id.keys():
            file.write(str(i) + '\n')
        file.close()

    # ------------------------------------------------------------------------------------------------------------

    jump_task_f = dirname + '/' + 'jump_task.csv'
    entity_id2traj_id = {}
    with open(jump_task_f, 'r') as file:
        csv_reader = csv.reader(file, delimiter=',')
        i = 0
        for row in csv_reader:
            if i == 0:
                i = i + 1
                continue
            entity_id = row[2]
            traj_id = row[3]

            if entity_id not in entity_id2traj_id.keys():
                entity_id2traj_id[entity_id] = {traj_id}
            else:
                entity_id2traj_id[entity_id].add(traj_id)
        file.close()

    # jump_task:
    fmm_jump_f = dirname + '/' + 'fmm_jump.csv'
    traj_id2road_id = []
    with open(fmm_jump_f, 'r') as file:
        csv_reader = csv.reader(file, delimiter=';')
        i = 0
        for row in csv_reader:
            if i > 0 and row[6] != '':
                str1 = row[6]
                if ',' not in row[6]:
                    str1 = str1 + ' ' + str1
                traj_id2road_id.append((row[0], str1))
            i = i + 1
        file.close()

    with open(dirname + '/' + 'jump.txt', 'w') as file:
        for e in traj_id2road_id:
            traj_id = e[0]
            for entity_id in entity_id2traj_id.keys():
                if traj_id in entity_id2traj_id[entity_id]:
                    file.write(entity_id + ' ' + traj_id + ' ' + e[1].replace(',', ' ') + '\n')
                    break

        file.close()


if __name__ == '__main__':
    form_data('data/', 38027)
