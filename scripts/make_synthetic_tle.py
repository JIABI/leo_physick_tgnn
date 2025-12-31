from __future__ import annotations
import argparse, math
from datetime import datetime, timezone

MU = 398600.4418  # km^3/s^2
R_E = 6378.137  # km


def tle_checksum(line: str) -> str:
    s = 0
    for ch in line[:68]:
        if ch.isdigit():
            s += int(ch)
        elif ch == "-":
            s += 1
    return str(s % 10)


def format_epoch(dt: datetime) -> str:
    # YYDDD.DDDDDDDD
    year = dt.year % 100
    start = datetime(dt.year, 1, 1, tzinfo=timezone.utc)
    doy = (dt - start).total_seconds() / 86400.0 + 1.0
    return f"{year:02d}{doy:012.8f}"


def mean_motion_rev_per_day(alt_km: float) -> float:
    a = R_E + alt_km
    n = math.sqrt(MU / (a ** 3))  # rad/s
    return n * 86400.0 / (2.0 * math.pi)  # rev/day


def make_tle_lines(
        satnum: int,
        epoch: datetime,
        inc_deg: float,
        raan_deg: float,
        ecc: float,
        argp_deg: float,
        mean_anom_deg: float,
        mm_rev_day: float,
        revnum: int = 1,
) -> tuple[str, str]:
    epoch_str = format_epoch(epoch)

    # NOTE: 这里把 B*、drag 等都设为 0（合成仿真足够用）
    line1 = f"1 {satnum:05d}U 25001A   {epoch_str}  .00000000  00000-0  00000-0 0  999"
    line1 = line1.ljust(68)
    line1 = line1[:68] + tle_checksum(line1)

    ecc7 = f"{ecc:.7f}".split(".")[1][:7]  # 7 digits, no dot
    line2 = (
        f"2 {satnum:05d} {inc_deg:8.4f} {raan_deg:8.4f} {ecc7:7s} "
        f"{argp_deg:8.4f} {mean_anom_deg:8.4f} {mm_rev_day:11.8f}{revnum:5d}"
    )
    line2 = line2.ljust(68)
    line2 = line2[:68] + tle_checksum(line2)
    return line1, line2


def generate(out_path: str, N: int, alt_km: float, inc_deg: float, planes: int, ecc: float, seed: int):
    # 让每个 plane 的卫星数尽量均匀
    sats_per_plane = math.ceil(N / planes)
    epoch = datetime.now(timezone.utc)
    mm = mean_motion_rev_per_day(alt_km)

    k = 0
    with open(out_path, "w", encoding="utf-8") as f:
        for p in range(planes):
            raan = (360.0 * p / planes) % 360.0
            for q in range(sats_per_plane):
                if k >= N:
                    break
                satnum = 90000 + k + 1
                M = (360.0 * q / sats_per_plane) % 360.0
                name = f"SYN-{satnum:05d}"
                l1, l2 = make_tle_lines(
                    satnum=satnum,
                    epoch=epoch,
                    inc_deg=inc_deg,
                    raan_deg=raan,
                    ecc=ecc,
                    argp_deg=0.0,
                    mean_anom_deg=M,
                    mm_rev_day=mm,
                    revnum=1,
                )
                f.write(name + "\n")
                f.write(l1 + "\n")
                f.write(l2 + "\n")
                k += 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--N", type=int, required=True)
    ap.add_argument("--out", type=str, required=True)
    ap.add_argument("--alt_km", type=float, default=550.0)
    ap.add_argument("--inc_deg", type=float, default=53.0)
    ap.add_argument("--planes", type=int, default=72)
    ap.add_argument("--ecc", type=float, default=0.0001)
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()
    generate(args.out, args.N, args.alt_km, args.inc_deg, args.planes, args.ecc, args.seed)
    print(f"[OK] wrote {args.N} sats to {args.out}")


if __name__ == "__main__":
    main()