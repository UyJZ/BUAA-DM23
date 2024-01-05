amap_key = '8122351ca4498dd9af86b6b1f21c6a47'

import requests

def calculate_driving_time(api_key, origin, destination):
    url = "https://restapi.amap.com/v5/direction/driving"
    params = {
        "key": api_key,
        "origin": origin,
        "destination": destination,
        "output": "json",
        "strategy": 0,  # 0: 速度优先
        "show_fields" : 'cost'
    }

    response = requests.get(url, params=params)
    data = response.json()
    print(data)

    if data.get("status") == "1" and int(data.get("count", 0)) > 0:
        total_duration = sum(int(step.get('cost', 0).get("duration", 0)) for path in data["route"]["paths"] for step in path["steps"])
        return (total_duration / 60)
    else:
        print("Error in API response")
        return None

if __name__ == "__main__":
    # 替换为你的高德地图 API Key
    amap_api_key = amap_key

    # 替换为起点和终点的经纬度
    origin_coords = "116.335503,39.763187"  # 举例：北京天安门
    destination_coords = "116.34182,39.771339"  # 举例：北京西站

    total_driving_time = calculate_driving_time(amap_api_key, origin_coords, destination_coords)

    if total_driving_time is not None:
        print(f"Total driving time: {total_driving_time} mins")
