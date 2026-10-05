import argparse
from pathlib import Path
from .common import ROOT, CONFIG, read_json


def main():
    parser = argparse.ArgumentParser(description="独立 UAV-FD/F550 模型与证据工具")
    parser.add_argument("command", choices=["sources", "audit", "calibrate", "replay", "vibration", "demo", "validate", "report", "test"])
    parser.add_argument("--data-dir", type=Path, default=ROOT / "UAV-FD")
    parser.add_argument("--out", type=Path, default=ROOT / "output_uavfd_f550")
    parser.add_argument("--config", type=Path, default=CONFIG / "analysis.json")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    if args.command == "test":
        parser.error("最终实飞测试处于锁定状态，本轮不提供解锁开关。")
    if args.command == "sources":
        from .sources import run
        run(args.out / "source_snapshots")
    elif args.command == "audit":
        from .audit import run
        run(args.data_dir, args.out, read_json(args.out / "release.json"))
    elif args.command == "report":
        from .reports import run
        run(args.out)
    else:
        from .analysis import run
        run(args.command, args.data_dir, args.out, read_json(args.config))


if __name__ == "__main__":
    main()
