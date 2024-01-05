import sys
sys.path.append("../ETA")
import SpeedPredictor as SP
speedPredictor = SP.SpeedPredictor()
mean_speed_per_road = speedPredictor.predict_speed(road_ids, start_speed, hour, holiday)

# 检测current_distance是否为None，是则开始计算
print("start calculate distance and coordinate")
with open('jump_task.csv',"w") as file :
    csv_writer = csv.writer(file)
    csv_writer.writerow(["id","time","entity_id","traj_id","coordinates","current_dis","speeds","holidays"])
    for i in range(len(jump_task_data)):
        id2,t2,entity_id,traj_id,coordinate2,current_distance2,speeds2,holidays2 = jump_task_data[i]
        if coordinate2 == None:
            id1,t1,_,_,coordinate1,current_distance1,speeds1,holidays1 = jump_task_data[i - 1]
            delta_t = time_difference(t1,t2)
            # 有可能出现fmm_jump匹配失败无法预测
            if (entity_id,traj_id) not in predict_data.keys() :
                current_distance2 = 114514
                coordinate2 = (114514,114514)
            else :
                road_id1,road_id2 = predict_data[(entity_id,traj_id)]
                if road_id1 == road_id2 : # TODO 毒瘤情况
                    print("{}youdu".format(id2))
                    current_distance2 = 114514
                    coordinate2 = (114514,114514)
                else :
                    cross = cross_data[(road_id1,road_id2)]
                    # 倒数第二条路段的速度v1
                    v1 = mean_speed_per_road[-2,traj_id2idx[traj_id]] * 60 / 1000
                    print(v1-speeds1)
                    remainDis = getRemainDis(road_id1,cross,coordinate1)
                    t4road1 = remainDis / v1
                    if t4road1 < delta_t : # 说明开到road2上了
                        print("change")
                        # 最后一条路段的速度v2
                        v2 = mean_speed_per_road[-1,traj_id2idx[traj_id]] * 60 / 1000
                        print(v2-speeds2)
                        dis4road2 = v2 * (delta_t - t4road1)
                        coordinate2 = getCoordinate(road_id2,cross,dis4road2)
                        current_distance2 = current_distance1 + remainDis + dis4road2
                    else : # 说明没时间把road1开完
                        print("remain")
                        # 计算距离cross的距离
                        d = remainDis - v1 * delta_t
                        coordinate2 = getCoordinate(road_id1,cross,d)
                        current_distance2 = current_distance1 + v1 * delta_t

                
            
        # 将结果写入jump_task.csv
        # 这里没写回原文件,写在当前目录下的jump_task.csv中
        row = [id2,t2,entity_id,traj_id,"[{},{}]".format(coordinate2[0],coordinate2[1]),current_distance2,speeds2,holidays2]
        csv_writer.writerow(row)
    file.close
            

        
            


            

            


















        
        