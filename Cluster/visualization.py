import csv
import argparse
import pickle
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument("--csv", type=str, default="database/data/road.csv", help="path to the csv file that contains polylines wanted")
parser.add_argument("--polyline-index", type=int, default=1, help="index of polyline column.")
parser.add_argument("--wkt", type=bool, default=False, help="whether the polyline is stored in wkt format. Or as the format in database/data/road.csv")
parser.add_argument("--output", type=str, default="Cluster/map.html", help="output path of the generated html file")
parser.add_argument("--label", type=str, default="Cluster/pkl_label_eps0.4_mst500.cluster")

offset = []   # 之后可以把这个offset试出来获得更好的可视化效果.

color_bar = [
    '#FF0000',   # -1开始，红色表示噪声点.
    '#3C7CB2',
    '#4daf48',
    '#994da5',
    '#ff7f03',
    '#fffd35',
    '#a55728',
    '#f880c1',
    '#999999',
]

before = \
'''
<!doctype html>
<html>

<head>
  <meta charset="utf-8">
  <meta http-equiv="X-UA-Compatible" content="IE=edge">
  <meta name="viewport" content="initial-scale=1.0, user-scalable=no, width=device-width">
  <title>HELLO，AMAP!</title>
  <link rel="stylesheet" href="https://a.amap.com/jsapi_demos/static/demo-center/css/demo-center.css" />
  <style>
    html,
    body,
    #container {
      height: 100%;
      width: 100%;
    }

    .amap-icon img,
    .amap-marker-content img {
      width: 25px;
      height: 34px;
    }

    .cus_info_window {
      background-color: #fff;
      padding: 10px;
    }
  </style>
</head>

<body>
  <div id="container"></div>
  <script type="text/javascript" src="https://webapi.amap.com/maps?v=2.0&key=692a38180fd6dce577d2e206e522d8e3"></script>
  <script type="text/javascript">
    // 创建地图实例
    var map = new AMap.Map("container", {
      zoom: 13,
      center: [116.39, 39.92],
      resizeEnable: true
    });

    // 创建点覆盖物
    var marker = new AMap.Marker({
      position: new AMap.LngLat(116.39, 39.92),
      icon: '//a.amap.com/jsapi_demos/static/demo-center/icons/poi-marker-default.png',
      offset: new AMap.Pixel(-13, -30)
    });
    map.add(marker);

    // 创建信息窗体
    var infoWindow = new AMap.InfoWindow({
      isCustom: true,  // 使用自定义窗体
      content: '<div class="cus_info_window">HELLO,AMAP!</div>', // 信息窗体的内容可以是任意 html 片段
      offset: new AMap.Pixel(16, -45)
    });
    var onMarkerClick = function (e) {
      infoWindow.open(map, e.target.getPosition()); // 打开信息窗体
      // e.target 就是被点击的 Marker
    }

    marker.on('click', onMarkerClick); // 绑定 click 事件
'''
after = \
'''
  </script>
</body>

</html>
'''

def main():
    args = parser.parse_args()
    infilepath = args.csv
    polyindex = args.polyline_index
    isWKT = args.wkt
    outfilepath = args.output
    label_path = args.label
    
    infile = open(infilepath, "r", encoding="utf-8-sig")
    reader = csv.reader(infile)

    with open(label_path, "rb") as f:
        label = pickle.load(f)
    n_clusters = label.max() - label.min() + 1
    lineArrs = ['' for _ in range(n_clusters)]
    idx = 0
    with open(outfilepath, "w", encoding="utf-8-sig") as outfile:
        outfile.write(before)
        first = True
        #polylines = ""
        for line in reader:
            if first:
                first = False
                continue
            else:
                polyline = line[polyindex]
                l = int(label[idx]) + 1  # 转化为从0开始.
                idx += 1
                if not isWKT:
                    lineArrs[l] += polyline + ","
                    #polylines += polyline + ","

        for i in range(n_clusters):
            middle = \
'''
    var lineArr{0} = [{1}];
    var polyline = new AMap.Polyline({{
      path: lineArr{0},          // 设置线覆盖物路径
      strokeColor: "{2}", // 线颜色
      strokeWeight: 5,        // 线宽
      strokeStyle: "solid",   // 线样式
    }});
    map.add(polyline);
'''.format(i, lineArrs[i], color_bar[i])     # 之后再修改颜色等等.
            outfile.write(middle)
        
        outfile.write(after)
    
    infile.close()
    

if __name__=='__main__':
    main()