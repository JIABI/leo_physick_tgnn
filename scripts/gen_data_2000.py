from pathlib import Path
import datetime, math


def tle_checksum(line_68: str) -> str:
    s = 0
    for ch in line_68[:68]:
        if ch.isdigit():
            s += int(ch)
        elif ch == '-':
            s += 1
    return str(s % 10)


def format_epoch(dt: datetime.datetime) -> str:
    year = dt.year % 100
    start = datetime.datetime(dt.year, 1, 1, tzinfo=dt.tzinfo)
    day_of_year = (dt - start).days + 1
    frac_day = (dt - (start + datetime.timedelta(days=day_of_year - 1))).total_seconds() / 86400.0
    return f"{year:02d}{day_of_year:03d}{frac_day:0.8f}"[:14]


def bstar_field(x: float) -> str:
    if x == 0.0:
        return " 00000+0"
    sign = '-' if x < 0 else ' '
    x = abs(x)
    exp10 = int(math.floor(math.log10(x)))
    mant = x / (10 ** exp10)
    mant_int = int(round(mant * 1e5))
    if mant_int >= 1000000:
        mant_int //= 10
        exp10 += 1
    exp = exp10 - 5
    exp = max(-9, min(9, exp))
    exp_sign = '-' if exp < 0 else '+'
    return f"{sign}{mant_int:05d}{exp_sign}{abs(exp):1d}"


def make_tle(name, satnum, epoch, inc_deg, raan_deg, ecc, argp_deg, ma_deg, mm_rev_per_day, revnum, bstar=0.0):
    intl = "25999A  "
    epoch_str = format_epoch(epoch)

    line1_wo = (
        f"1 {satnum:05d}U {intl}{epoch_str}  "
        f".00000000  00000-0 {bstar_field(bstar)} 0  999"
    ).ljust(68)[:68]
    line1 = line1_wo + tle_checksum(line1_wo)

    ecc_str = f"{ecc:.7f}".split(".")[1]
    line2_wo = (
        f"2 {satnum:05d} "
        f"{inc_deg:8.4f} {raan_deg:8.4f} {ecc_str:7s} "
        f"{argp_deg:8.4f} {ma_deg:8.4f} {mm_rev_per_day:11.8f}{revnum:5d}"
    ).ljust(68)[:68]
    line2 = line2_wo + tle_checksum(line2_wo)

    return f"{name}\n{line1}\n{line2}\n"


def generate(path, planes=40, sats_per_plane=50, inc_deg=53.0, mean_motion=15.05, ecc=0.0002, argp_deg=0.0,
             epoch=None, satnum_start=40000):
    if epoch is None:
        epoch = datetime.datetime(2025, 12, 25, 0, 0, 0, tzinfo=datetime.timezone.utc)
    out = []
    idx = 0
    for p in range(planes):
        raan = (p * 360.0 / planes) % 360.0
        for k in range(sats_per_plane):
            satnum = satnum_start + idx
            name = f"SYN-LEO-P{p + 1:03d}-S{k + 1:03d}"
            ma = (k * 360.0 / sats_per_plane) % 360.0
            revnum = 1 + (k % 9999)
            out.append(make_tle(name, satnum, epoch, inc_deg, raan, ecc, argp_deg, ma, mean_motion, revnum))
            idx += 1
    Path(path).write_text("".join(out), encoding="utf-8")
    print("Wrote", path, "sats", idx)


if __name__ == "__main__":
    generate("data/synthetic_leo_2000.tle", planes=40, sats_per_plane=50)  # 40*50=2000