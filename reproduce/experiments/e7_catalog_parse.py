"""E7 探查：解析 Apollo 事件目录，提取 1976-01 事件"""
import csv, os

base = r"G:\SeisY\docs\dssrr_paper\experiments\apollo_catalog"

# 1) dm arrivals 全貌
with open(os.path.join(base, "nakamura_2005_dm_arrivals.csv"), encoding="utf-8") as f:
    lines = [l.rstrip("\n") for l in f if l.strip()]
print("dm_arrivals lines:", len(lines), "first:", lines[0][:60], "last:", lines[-1][:60])

# 2) m_arrivals 1976-01 事件
with open(os.path.join(base, "nakamura_1983_m_arrivals.csv"), encoding="utf-8") as f:
    rd = csv.reader(f)
    header = next(rd)
    rows = [r for r in rd]
print("\nm_arrivals total:", len(rows))
jan = [r for r in rows if len(r) > 2 and r[0] == "1976" and r[1].startswith("00")]
print("1976-Jan meteoroid events:", len(jan))
for r in jan:
    print("  ", r[:6])

# 3) sm_arrivals 全量
with open(os.path.join(base, "nakamura_1983_sm_arrivals.csv"), encoding="utf-8") as f:
    rd = csv.reader(f)
    header = next(rd)
    rows = [r for r in rd]
jan = [r for r in rows if len(r) > 2 and r[0] == "1976" and r[1].startswith("00")]
print("\nsm_arrivals total:", len(rows), "1976-Jan:", len(jan))
for r in jan:
    print("  ", r[:6])

# 4) ai_arrivals
with open(os.path.join(base, "nakamura_1983_ai_arrivals.csv"), encoding="utf-8") as f:
    rd = csv.reader(f)
    header = next(rd)
    rows = [r for r in rd]
print("\nai_arrivals total:", len(rows))
for r in rows[:12]:
    print("  ", r[:8])
