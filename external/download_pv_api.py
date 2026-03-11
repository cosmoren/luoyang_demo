from __future__ import annotations
import os
import json
import random
import sys
import time
from pathlib import Path
from typing import List
from datetime import datetime, timedelta, timezone

import pandas as pd
import requests
import urllib3
import yaml
from tqdm import tqdm

urllib3.disable_warnings()
from urllib.parse import quote, urlparse

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CONF_PATH = PROJECT_ROOT / "config" / "conf.yaml"
PASSWORDS_PATH = PROJECT_ROOT / "config" / "passwords.yaml"


def load_conf():
    with open(CONF_PATH) as f:
        return yaml.safe_load(f)


def load_passwords():
    """Read endpoint, username, password from config/passwords.yaml (pv section)."""
    if not PASSWORDS_PATH.exists():
        print(f"Missing {PASSWORDS_PATH}. Create it with pv.endpoint, pv.username, pv.password.")
        sys.exit(1)
    with open(PASSWORDS_PATH) as f:
        data = yaml.safe_load(f)
    pv = (data or {}).get("pv") or {}
    endpoint = (pv.get("endpoint") or "").strip()
    username = (pv.get("username") or "").strip()
    password = (pv.get("password") or "").strip()
    if not endpoint or not username or not password:
        print("Set pv.endpoint, pv.username, and pv.password in config/passwords.yaml")
        sys.exit(1)
    return endpoint, username, password


def resolve_path(p: str) -> Path:
    path = Path(p)
    return path if path.is_absolute() else (PROJECT_ROOT / p).resolve()


DEVICE_METAINFOS = [
    'stationCode',
    'plantName',
    'plantAddress',
    'devName',
    'latitude_device',
    'longitude_device',
    'capacity'
]

DEVICE_DATA_FIELDS = [
    'inverter_state',
    'efficiency',
    'temperature',
    'power_factor',
    'elec_freq',
    'active_power',
    'reactive_power',
    'day_cap',
    'mppt_power',
    'total_cap',
    'mppt_total_cap',
]





class NorthAPI:
    header ={
        "application/json": "application/json"
    }

    def __init__(
        self, 
        endpoint, 
        username, 
        password,
        auth_credential: str=None
    ):
        self.endpoint = endpoint
        self.username = username
        self.password = password
        self.proxies = self.set_proxy(auth_credential)
        print(self.proxies)
        self.session = requests.session()
        self.get_token()

    @staticmethod
    def set_proxy(credential):
        if credential is None:
            return credential
        account = json.load(open(credential))
        proxy_url = "proxy.huawei.com"
        proxy_port = 8080
        return {
            "http": f"dddd",
            "https": f"dddd",
        }

    @staticmethod
    def sleep():
        time.sleep(random.randint(3, 10))

    def get_token(self):
        """



        
        
        """
        req_url = "https://{}/thirdData/login".format(self.endpoint)
        req_json = {
            "userName": self.username,
            "systemCode": self.password,
        }
        resp = self.session.post(url=req_url, json=req_json, verify=False)
        if resp.status_code == 200:
            self.header["XSRF-TOKEN"] = resp.headers["xsrf-token"]
            print("login success")
        else:
            print(f"login failed：{resp.json()}")
            sys.exit(1)
        return resp

    def get_station_list(self, page_num: int = 1, get_total: bool = False):
        """



        """
        req_url = "https://{}/thirdData/stations".format(self.endpoint)
        req_json = {
            "pageNo": page_num
        }
        resp = self.session.post(url=req_url, json=req_json, proxies=self.proxies, verify=False, headers=self.header)
        if resp.status_code == 200 and resp.json()["success"] is True:
            print("get station list success")
            if get_total:
                return resp.json()["data"]["total"]
            else:
                return resp.json()["data"]["list"]
                #
        else:
            print(f"get station list failed：{resp.json()}")
            sys.exit(1)
        
    def get_device_list(self, station_codes: List[str]):
        
        req_url = "https://{}/thirdData/getDevList".format(self.endpoint)
        req_json = {
            "stationCodes": ",".join(station_codes),
        }
        resp = self.session.post(url=req_url, json=req_json, proxies=self.proxies, verify=False, headers=self.header)
        if resp.status_code == 200 and resp.json()["success"] is True:
            print("get device list success")
            return resp.json()["data"]
        else:
            print(f"get device list failed: {resp.json()}")
            sys.exit(1)

    def get_real_kpi(self, dev_ids: List[str], dev_type_id: int):
        """

        """
        req_url = "https://{}/thirdData/getDevRealKpi".format(self.endpoint)
        req_json = {
            "devIds": ",".join(dev_ids),
            "devTypeId": dev_type_id,
        }
        resp = self.session.post(url=req_url, json=req_json, proxies=self.proxies, verify=False, headers=self.header)
        if resp.status_code == 200 and resp.json()["success"] is True:
            print("get real kpi success")
            return resp.json()["data"]
        else:
            print(f"get real kpi failed：{resp.json()}")
            sys.exit(1)
        # return resp
    def get_history_kpi(self, dev_ids: List[str], dev_type_id: int, start_time: str, end_time: str):


        req_url = "https://{}/thirdData/getDevHistoryKpi".format(self.endpoint)
        req_json = {
            "devIds": ",".join(dev_ids),
            "devTypeId": dev_type_id,
            "startTime": start_time,
            "endTime": end_time,
        }
        resp = self.session.post(url=req_url, json=req_json, proxies=self.proxies, verify=False, headers=self.header)
        if resp.status_code == 200 and resp.json()["success"] is True:
            print("get history kpi success")
            return resp.json()["data"]
        else:
            print(f"get history kpi failed：{resp.json()}")
            sys.exit(1)
        # return resp
class TimeParser:
    @staticmethod
    def dt2ms(dt):
        return dt.timestamp() * 1000.0

    @staticmethod
    def ms2dt(ms):
        return datetime.fromtimestamp(ms / 1000.0)

class NorthFetcher(TimeParser):
    device_map = {
        1: "组串式逆变器",
        2: "数采",
        10: "环境检测仪",
        38: "户用逆变器"
    }

    def __init__(
        self, 
        endpoint, 
        username, 
        password,
        credential: str=None,
        station_path: str=None,
        device_path: str=None,
        station_uniq_id: str='plantCode',
        device_uniq_id: str='devDn',
        devtype: int=1,
        station_meta_save_dir: str=None
    ):
        self.api = NorthAPI(endpoint, username, password, credential)
        self.station_uniq_id = station_uniq_id
        self.device_uniq_id = device_uniq_id
        self.devtype = devtype

        if device_path is not None:
            self.device_df = self.load_df(device_path)
        else:
            self.station_df, self.device_df = self.fetch_devices(station_path)
            if station_meta_save_dir is not None:
                self.station_df.to_csv(station_meta_save_dir, f'{username}-stations.csv', 
                                        index=False, encoding='utf-8-sig')
                self.device_df.to_csv(os.path.join(station_meta_save_dir, f'{username}-devices.csv'), 
                                        index=False, encoding='utf-8-sig')

    @staticmethod
    def load_df(path: str):
        if isinstance(path, pd.DataFrame):
            return path
        else:
            try:
                df = pd.read_csv(path)
            except:
                df = pd.read_excel(path)
            return df

    def fetch_stations(self, path: str=None):
        """
        """
        if path is not None:
            return self.load_df(path)
        station_total = self.api.get_station_list(get_total=True)
        print ("Total stations: ", station_total)
        stations = []
        for page in range(1, station_total // 100 + 2):
            print(f"Fetching page {page} of {station_total // 100 + 1}")
            stations.extend(self.api.get_station_list(page_num=page))
            self.api.sleep()
        self.api.sleep()
        return pd.DataFrame(stations)

    def fetch_devices(self, path: str=None):
        """
        """
        station_df = self.fetch_stations(path)
        devices = []
        for i in range(0, len(station_df), 100):
            device_list = self.api.get_device_list(station_df[self.station_uniq_id].values[i: i + 100])
            devices.extend([d for d in device_list if d["devTypeId"] == self.devtype])
            self.api.sleep()
            
        device_df = pd.DataFrame(devices)
        device_df = pd.merge(station_df, device_df, 
                            left_on=self.station_uniq_id, right_on='stationCode', how='inner',
                            suffixes=('_station', '_device'))
        return station_df, device_df

    def fetch_history_by_date(self, date: datetime, metainfos, datafields):
        year, month, day = date.year, date.month, date.day
        datalist = []
        print (f"获取组串式逆变器式时功率，逆变器个数： {len(self.device_df)}")
        for ind, dev in tqdm(self.device_df.iterrows()):
            kpi_data = self.api.get_history_kpi(
                dev['devDn'], self.devtype,
                self.dt2ms(datetime(year, month, day, 0, 0, 0)),
                self.dt2ms(datetime(year, month, day, 23, 59, 59)))
            for data_by_time in kpi:
                dev_data = {k: dev[k] for k in metainfos}
                dev_data.update({
                    'collectTime': data_by_time['collectTime'],
                    'devDn': data_by_time['devDn'],
                })
                dev_data.update({f: data_by_time['dataItems'][f] for f in datafields})
                datalist.append(dev_data)
        df = pd.DataFrame(datalist)
        for col in ['collectTime']:
            df[col] = list(map(lambda x: x if pd.isna(x) else self.ms2dt(x), df[col]))
        return df

    def fetch_history(self, start, end, save_dir, metainfos, datafields=None):
        dates = pd.data_range(start=start, end=end).tolist()
        for date in dates:
            save_path = os.path.join(save_dir, '{}-{}.csv'.format(self.device_map[self.devtype], date.strftime("%Y-%m-%d")))
            if os.path.exists(save_path):
                continue
            df = self.fetch_history_by_date(date, 
                            DEVICE_METAINFOS if metainfos is None else metainfos, 
                            DEVICE_DATA_FIELDS if datafields is None else datafields)
            df.to_csv(save_path, encoding='utf-8-sig', index=False)
    
    def fetch_real(self, save_dir, collect_time):
        
        dev_ids = []
        for ind, dev in tqdm(self.device_df.iterrows()):
            dev_ids.append(dev['devDn'])

        data_list = []
        for i in range(0, len(dev_ids), 100):
            real_kpi = self.api.get_real_kpi(dev_ids[i: i + 100], 1)
            # ['devId', 'sn', 'dataItemMap']
            # [1000000333620909, ES2230149107, ]
            for j, each_real_kpi in enumerate(real_kpi):
                dev_line = {}
                dev_line['devDn'] = dev_ids[i: i + 100][j]
                for key in DEVICE_DATA_FIELDS:
                    dev_line[key] = each_real_kpi['dataItemMap'][key]
                dev_line['collectTime'] = collect_time
                data_list.append(dev_line)    
            
        df = pd.DataFrame(data_list)
        df.to_csv(save_dir, index=False, encoding='utf-8-sig')
            
                    

def next_five_minute() -> datetime:
    now = datetime.now(timezone.utc)
    minute = (now.minute // 5 + 1) * 5
    if minute >= 60:
        return (now.replace(minute=0, second=0, microsecond=0) + timedelta(hours=1))
    return now.replace(minute=minute, second=0, microsecond=0)


if __name__ == "__main__":
    endpoint, username, password = load_passwords()
    conf = load_conf()
    paths = conf.get("paths", {})
    pv_download = paths.get("pv_download", "data/pv")
    save_dir = resolve_path(pv_download)
    device_path_raw = paths.get("pv_device_path")
    station_meta_raw = paths.get("pv_station_meta")
    device_path = str(resolve_path(device_path_raw)) if device_path_raw else None
    station_meta_save_dir = str(resolve_path(station_meta_raw)) if station_meta_raw else None

    dev_type = 1
    north_fetcher = NorthFetcher(
        endpoint=endpoint,
        username=username,
        password=password,
        credential=None,
        device_path=device_path,
        station_meta_save_dir=station_meta_save_dir)
    # north_fetcher.fetch_history(start='2025-01-01', end='2025-12-31', save_dir='./data/history')


    while True:
        next_run = next_five_minute()
        now_utc = datetime.now(timezone.utc)
        wait_sec = (next_run - now_utc).total_seconds()
        print(next_run, now_utc)
        if wait_sec > 0:
            time.sleep(wait_sec)

        capture = datetime.now(timezone.utc)
        filename = capture.strftime("%Y%m%d_%H%M") + ".csv"
        save_path = save_dir / filename
        north_fetcher.fetch_real(save_dir=save_path, collect_time=capture)