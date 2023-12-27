amap_key = '8122351ca4498dd9af86b6b1f21c6a47'

api_url = 'https://restapi.amap.com/v5/direction/driving'

import requests
import json

def driving_route_planning():

    params = {
        "key": amap_key,
        "origin": "116.450546,39.943771",
        "destination": "116.440742,40.018944",
        "strategy": 0,
        'show_fields': 'polyline',
    }


    try:
        response = requests.get(api_url, params=params)
        result = response.json()
        if result.get("status") == "1":
            print("请求成功")
            print("方案总数:", result["count"])
            print(result)
            
            for route in result["route"]:
                print("方案距离:", route["distance"], "米")
                print("预计出租车费用:", route.get("taxi_cost", "未知"), "元")

                for step in route.get("paths", []):
                    print("行驶指示:", step.get("instruction", "未知"))
                    print("进入道路方向:", step.get("orientation", "未知"))
                    print("分段道路名称:", step.get("road_name", "未知"))
                    print("分段距离信息:", step.get("distance"), "米")

                print("\n" + "=" * 50 + "\n")
        else:
            print("请求失败，错误信息:", result.get("info", "未知"))

    except Exception as e:
        print("请求异常:", e)


if __name__ == "__main__":
    driving_route_planning()
