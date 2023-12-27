import pandas as pd
from shapely.geometry import LineString, Point

def getRes(df, p : Point):
    df['geometry'] = df['coordinates'].apply(lambda x: LineString(eval(x)))

    # 计算给定坐标到每条路径的距离
    df['nearest_point'] = df['geometry'].apply(lambda line: line.interpolate(line.project(p)))

    # 添加距离列
    df['nearest_point_distance'] = df.apply(lambda row: row['geometry'].distance(row['nearest_point']), axis=1)

    # 按照距离升序排序
    df = df.sort_values(by='nearest_point_distance')
    
    return df.iloc[0]

def Point2SrcDst(src_x : float, src_y : float, dst_x : float, dst_y : float):
    src = Point(src_x, src_y)
    dst = Point(dst_x, dst_y)
    # 读取 CSV 文件
    df = pd.read_csv('/mnt/f/codes/Python/BUAA-DM23/database/data/road.csv')
    
    res_src = getRes(df, src)
    res_dst = getRes(df, dst)
    from_x = res_src['nearest_point'].x
    from_y = res_src['nearest_point'].y
    to_x = res_dst['nearest_point'].x
    to_y = res_dst['nearest_point'].y
    return from_x, from_y, to_x, to_y
    
    
def dijkstra(from_x : float, from_y : float, to_x : float, to_y : float):
    pass

if __name__ == "__main__":
    from_x, from_y, to_x, to_y = Point2SrcDst(116.490013,39.892902, 116.511292,39.904655)
    print(from_x, from_y, to_x, to_y)
