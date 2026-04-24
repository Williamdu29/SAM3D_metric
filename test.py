import requests as req

item_set_id=2604081753388495 # 物品批次号
item_id=2604081753592761 # 物品ID
best_frame=1775642189184 # 最佳帧时间戳（毫秒）
k=[384.8974914550781, 0, 326.8973083496094, 0, 384.4713745117187, 240.4253845214844, 0, 0, 1] # 内参
resp = req.post("http://192.168.112.169:58860/api/inference",json={"item_set_id":item_set_id,"item_id":item_id,"timestamp":best_frame,"k":k})
print(resp.json())

