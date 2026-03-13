import json
import random
import sys
import time
from pathlib import Path
from typing import List

import requests
import urllib3
import yaml

urllib3.disable_warnings()

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
from config_utils import get_resolved_paths

CONF_PATH = PROJECT_ROOT / "config" / "conf.yaml"
PASSWORDS_PATH = PROJECT_ROOT / "config" / "passwords.yaml"

session = requests.session()


def load_conf():
    with open(CONF_PATH) as f:
        return yaml.safe_load(f)


header = {
    "application/json": "application/json"
}

def get_token(endpoint, username, password):
    """
    获取token
    :param endpoint: 请求地址
    :param username: 用戶名
    :param password: 密碼
    :return:
    """
    req_url = "https://{}/thirdData/login".format(endpoint)
    req_json = {
        "userName": username,
        "systemCode": password
    }
    resp = session.post(url=req_url, json=req_json, verify=False)
    if resp.status_code == 200:
        header["XSRF-TOKEN"] = resp.headers["xsrf-token"]
        print("login success")
    else:
        print(f"login failed：{resp.json()}")
        sys.exit(1)
    return resp


def get_station_list(endpoint, page_num: int = 1, get_total: bool = False):
    """
    获取电站列表
    :param endpoint:请求地址
    :return:
    """
    req_url = "https://{}/thirdData/stations".format(endpoint)
    req_json = {
        "pageNo": page_num
    }
    resp = session.post(url=req_url, json=req_json, verify=False, headers=header)
    if resp.status_code == 200 and resp.json()["success"] is True:
        print("get station list success")
        if get_total:
            return resp.json()["data"]["total"]
        else:
            return [each_station["plantCode"] for each_station in resp.json()["data"]["list"]]
    else:
        print(f"get station list failed：{resp.json()}")
        sys.exit(1)


def get_device_list(endpoint, station_codes: List[str]):
    """
    获取设备列表
    :param endpoint:请求地址
    :param station_codes: 电站dn
    :return:
    """
    req_url = "https://{}/thirdData/getDevList".format(endpoint)
    req_json = {
        "stationCodes": ",".join(station_codes),
    }
    resp = session.post(url=req_url, json=req_json, verify=False, headers=header)
    if resp.status_code == 200 and resp.json()["success"] is True:
        print("get device list success")
        return resp.json()["data"]
    else:
        print(f"get device list failed：{resp.json()}")
        sys.exit(1)


def get_real_kpi(endpoint, inv_device_dev_ids: List[str], dev_type_id: int):
    """
    获取设备实时数据
    :param endpoint: 请求地址
    :param inv_device_dev_ids: 逆变器设备id
    :param dev_type_id： 设备类型id
    """
    req_url = "https://{}/thirdData/getDevRealKpi".format(endpoint)
    req_json = {
        "devIds": ",".join(inv_device_dev_ids),
        "devTypeId": dev_type_id
    }
    resp = session.post(url=req_url, json=req_json, verify=False, headers=header)
    if resp.status_code == 200 and resp.json()["success"] is True:
        print("get real kpi success")
        return resp.json()["data"]
    else:
        print(f"get real kpi failed：{resp.json()}")
        # sys.exit(1)


def main(endpoint, username, password):
    """
    主程序
    :param endpoint:
    :param username:
    :param password:
    :return:
    """
    conf = load_conf()
    paths = get_resolved_paths(conf, PROJECT_ROOT)
    out_dir = paths["pv_download"]
    get_token(endpoint, username, password)
    station_total = get_station_list(endpoint, get_total=True)
    print("总电站数：", station_total)
    station_codes = []
    for page in range(1, station_total // 100 + 2):
        print(f"获取第{page}页的电站")
        station_codes.extend(get_station_list(endpoint, page_num=page))
        # 防止限流
        time.sleep(random.randint(3, 10))

    # 获取设备列表，单次最多传入100个电站
    house_inv_dev_ids = []  # 户用逆变器
    pv_inv_dev_ids = []  # PV逆变器
    for i in range(0, len(station_codes), 100):
        device_list = get_device_list(endpoint, station_codes[i: i + 100])
        house_inv_dev_ids.extend(
            [str(each_device["id"]) for each_device in device_list if each_device["devTypeId"] == 1])
        pv_inv_dev_ids.extend(
            [str(each_device["id"]) for each_device in device_list if each_device["devTypeId"] == 38])
        # 防止限流
        time.sleep(random.randint(3, 10))

    # 获取设备实时数据，单次最多100个设备
    # 户用逆变器实时数据
    inv_active_power = {}
    print(f"获取户用逆变器实时功率, 逆变器个数：{len(house_inv_dev_ids)}")
    for i in range(0, len(house_inv_dev_ids), 100):
        house_kpi = get_real_kpi(endpoint, house_inv_dev_ids[i: i + 100], 1)
        for each_data in house_kpi:
            inv_active_power[each_data["sn"]] = each_data["dataItemMap"]["active_power"]
        time.sleep(random.randint(3, 10))

    # PV逆变器实时数据获取
    print(f"获取PV逆变器实时功率, 逆变器个数：{len(pv_inv_dev_ids)}")
    for i in range(0, len(pv_inv_dev_ids), 100):
        pv_kpi = get_real_kpi(endpoint, pv_inv_dev_ids[i: i + 100], 38)
        for each_data in pv_kpi:
            inv_active_power[each_data["sn"]] = each_data["dataItemMap"]["active_power"]
        time.sleep(random.randint(3, 10))

    for each_inv, each_active_power in inv_active_power.items():
        print(f"{each_inv}：{each_active_power}")

    # 结果写入文件
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "device_active_power.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(inv_active_power, f, indent=4, ensure_ascii=False)
    print(f"Saved: {out_path}")



def load_passwords():
    """Read endpoint, username, password from config/passwords.yaml (pv section)."""
    if not PASSWORDS_PATH.exists():
        print(f"Missing {PASSWORDS_PATH}. Create it with pv.endpoint, pv.username, pv.password.")
        sys.exit(1)
    with open(PASSWORDS_PATH) as f:
        data = yaml.safe_load(f)
    pv = data.get("pv") or {}
    endpoint = pv.get("endpoint", "").strip()
    username = (pv.get("username") or "").strip()
    password = (pv.get("password") or "").strip()
    if not endpoint or not username or not password:
        print("Set pv.endpoint, pv.username, and pv.password in config/passwords.yaml")
        sys.exit(1)
    return endpoint, username, password


if __name__ == "__main__":
    endpoint, username, password = load_passwords()
    main(endpoint, username, password)