"""
逐个检查 /work/datasets/luoyang_SPMF/pv_sf1 下所有 csv 文件的
collectTime 和 local_solar_time 字段是否存在不合法的时间格式。
若存在则打印文件名、行号(从 1 开始计 header)、对应原始行内容以及具体出错字段。
"""

import csv
import os
from datetime import datetime

DATA_DIR = "/work/datasets/luoyang_SPMF/pv_sf1"
TIME_COLUMNS = ("collectTime", "local_solar_time")
TIME_FORMATS = ("%Y-%m-%d %H:%M:%S", "%Y-%m-%d %H:%M:%S.%f")


def is_valid_time(value: str) -> bool:
    if value is None:
        return False
    value = value.strip()
    if not value:
        return False
    for fmt in TIME_FORMATS:
        try:
            datetime.strptime(value, fmt)
            return True
        except ValueError:
            continue
    return False


def check_file(path: str) -> int:
    bad_count = 0
    with open(path, "r", encoding="utf-8", newline="") as f:
        reader = csv.reader(f)
        try:
            header = next(reader)
        except StopIteration:
            return 0

        col_idx = {}
        for col in TIME_COLUMNS:
            if col in header:
                col_idx[col] = header.index(col)

        if not col_idx:
            print(f"[WARN] {path} 缺少时间列, 跳过")
            return 0

        for line_no, row in enumerate(reader, start=2):
            for col, idx in col_idx.items():
                if idx >= len(row):
                    print(
                        f"[BAD] {path} | line {line_no} | column={col} | "
                        f"missing field | row={row}"
                    )
                    bad_count += 1
                    continue
                value = row[idx]
                if not is_valid_time(value):
                    print(
                        f"[BAD] {path} | line {line_no} | column={col} | "
                        f"value={value!r} | row={row}"
                    )
                    bad_count += 1
    return bad_count


def main():
    if not os.path.isdir(DATA_DIR):
        print(f"目录不存在: {DATA_DIR}")
        return

    files = sorted(
        os.path.join(DATA_DIR, name)
        for name in os.listdir(DATA_DIR)
        if name.lower().endswith(".csv")
    )
    print(f"待检查 csv 文件数量: {len(files)}")

    total_bad = 0
    bad_files = 0
    for i, path in enumerate(files, start=1):
        bad = check_file(path)
        if bad > 0:
            bad_files += 1
            total_bad += bad
        if i % 1 == 0:
            print(f"  已检查 {i}/{len(files)} 个文件, 累计错误 {total_bad} 条")

    print("=" * 60)
    print(f"检查完成: 共 {len(files)} 个文件, "
          f"含错误文件 {bad_files} 个, 错误时间总数 {total_bad}")


if __name__ == "__main__":
    main()
