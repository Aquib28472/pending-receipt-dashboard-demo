"""Generate synthetic source tables for the Pending Receipt Dashboard demo.

Everything here is invented: customers, products, owners, documents, tracking numbers.
Scenarios are built so that each status the model can produce occurs, and an
expected_status.csv is written so the model's classification can be checked afterwards.
"""
from __future__ import annotations

import argparse
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

# movement codes used in the demo model
ISS, RCP, RET, WHR = "ISS", "RCP", "RET", "WHR"   # issue to transit, receipt, return dispatch, warehouse receipt of a return

DC_PLANTS = ["DC01", "DC02", "DC03", "DC04"]      # distribution centres, receipt doc = delivery
FIELD_PLANTS = ["FLD1", "FLD2", "FLD3"]           # field plants, receipt doc = material document; FLD3 belongs to DIV-B
DIV_B_CODE = "10"                                 # product division that maps to DIV-B, everything else is DIV-A

STATUS_SORT = {"pending": 1, "possible": 2, "return_transit": 3, "verify": 4,
               "overdue": 5, "old_return": 6, "old_stock": 7}

CARRIERS = ["FastFreight", "BlueLine", "AeroExpress", "RoadRunner"]
OWNERS = ["MARLOW QUINN", "TESSA BRANDT", "RAVI OKAFOR", "NOOR CASTELLAN", "ELIAS VARGA",
          "MINA HALLORAN", "JONAS PRYCE", "INES DUARTE", "CALLUM REYES", "SANA WHITLOCK"]
REGION_NAMES = ["North", "South", "East", "West", "Central"]
CUST_A = ["Northwind", "Harbor", "Summit", "Lakeside", "Pioneer", "Cedar", "Bluebird", "Granite", "Maple",
          "Riverbend", "Oakridge", "Silverline", "Kestrel", "Foxglove", "Ironwood"]
CUST_B = ["Clinics", "Health", "Care Group", "Medical Center", "Hospital", "Surgical", "Labs", "Supply Co"]
PRODUCT_KINDS = ["Catheter", "Guidewire", "Sheath", "Balloon", "Stent graft", "Introducer", "Syringe set", "Valve kit"]


def digits(rng, n: int) -> str:
    return "".join(str(d) for d in rng.integers(0, 10, n))


def build(seed: int = 7, dc_keys: int = 230, field_keys: int = 480, today: date | None = None):
    rng = np.random.default_rng(seed)
    today = today or date.today()

    # ---- products -------------------------------------------------------------------
    products = []
    for i in range(44):
        div = DIV_B_CODE if i % 4 == 0 else ("20" if i % 2 else "30")
        sku = str(500100 + i * 37)
        products.append({"sku": sku, "div": div,
                         "desc": f"{PRODUCT_KINDS[i % len(PRODUCT_KINDS)]} {3 + (i % 9)}F {60 + (i * 7) % 140} cm"})

    # ---- customers, regions, territories -------------------------------------------
    customers = []
    for i in range(170):
        name = f"{CUST_A[i % len(CUST_A)]} {CUST_B[(i // len(CUST_A)) % len(CUST_B)]}"
        if i >= len(CUST_A) * len(CUST_B):
            name += f" {i}"
        customers.append({
            "no": "2" + digits(rng, 9), "sold_to": "1" + digits(rng, 9), "name": name,
            "ta_region": f"A-R{rng.integers(1, 5)}", "terr_a": f"A-T{rng.integers(1, 15):02d}",
            "tis_region": f"B-R{rng.integers(1, 4)}", "terr_b": f"B-T{rng.integers(1, 10):02d}",
        })
    field_accounts = []
    for t in range(1, 15):
        field_accounts.append({
            "no": "80000" + digits(rng, 5), "name": f"Field stock A-T{t:02d}", "sold_to": None,
            "ta_region": f"A-R{(t - 1) % 4 + 1}", "terr_a": f"A-T{t:02d}",
            "tis_region": f"B-R{(t - 1) % 3 + 1}", "terr_b": f"B-T{(t - 1) % 9 + 1:02d}",
        })
    for a in field_accounts:
        a["sold_to"] = a["no"]

    # ---- owner alignment ------------------------------------------------------------
    align = []
    oi = 0
    for bu, prefix, n_regions, n_terr in (("DIV-A", "A", 4, 14), ("DIV-B", "B", 3, 9)):
        for r in range(1, n_regions + 1):
            owner = OWNERS[oi % len(OWNERS)]
            consign = owner + ("_OPEN" if oi % 5 == 4 else "")
            oi += 1
            for t in range(1, n_terr + 1):
                if (t - 1) % n_regions != (r - 1) % n_regions:
                    continue
                align.append({
                    "BU": bu, "AREA": f"Area {prefix}{r}", "Region ID": f"{prefix}-R{r}",
                    "Region Name": f"{REGION_NAMES[(r - 1) % 5]} ({prefix}{r})",
                    "Region Manager": f"Manager {prefix}{r}", "Territory ID": f"{prefix}-T{t:02d}",
                    "Territory Name": f"Territory {prefix}{t:02d}", "Field Associate": f"Associate {prefix}{t:02d}",
                    "Email Address": f"associate.{prefix.lower()}{t:02d}@example.com", "Acct #": "",
                    "OWNER - REP": owner, "OWNER - CONSIGN": consign, "REP": owner, "CONSIGN": consign,
                })
    alignment = pd.DataFrame(align)

    # ---- scenario engine ------------------------------------------------------------
    matdoc, fi, ret, trk, hist, expected = [], [], [], [], [], []
    used_serial = set()

    def new_serial():
        while True:
            s = digits(rng, 9)
            if s not in used_serial:
                used_serial.add(s)
                return s

    def old_date(min_months=26, max_months=60):
        return today - timedelta(days=int(rng.integers(min_months * 30, max_months * 30)))

    def age_days():
        return int(min(700, rng.gamma(1.6, 35))) + 1  # skewed young, long tail

    def pick_product(plant):
        pool = [p for p in products if (p["div"] == DIV_B_CODE) == (plant == "FLD3")] if plant in FIELD_PLANTS else products
        return pool[int(rng.integers(0, len(pool)))]

    def add_scenario(plant, scenario):
        is_dc = plant in DC_PLANTS
        prod = pick_product(plant)
        sku_padded = prod["sku"].zfill(18) if rng.random() < 0.6 else prod["sku"]
        batch = "L" + digits(rng, 7)
        vendor = ("V" + digits(rng, 7)) if rng.random() < 0.35 else ""
        n = int(rng.integers(1, 7)) if rng.random() < 0.85 else int(rng.integers(7, 20))
        cust = customers[int(rng.integers(0, len(customers)))] if is_dc else field_accounts[int(rng.integers(0, len(field_accounts)))]
        mdoc = "49" + digits(rng, 8) if is_dc else "4" + digits(rng, 9)
        delivery = "08" + digits(rng, 8)
        ref_issue = delivery if is_dc else digits(rng, 10)
        rcp_ref = delivery if is_dc else mdoc
        base = {"plant": plant, "sku": sku_padded, "batch": batch, "div": prod["div"]}

        def issue_doc(d):
            matdoc.append({"movement_type": ISS, "posting_date": d, "reference": ref_issue, "matdoc_no": mdoc,
                           "sku": sku_padded, "batch": batch, "plant": plant, "product_division": prod["div"], "quantity": n})

        def receipt_doc(d, q):
            matdoc.append({"movement_type": RCP, "posting_date": d, "reference": rcp_ref, "matdoc_no": "",
                           "sku": sku_padded, "batch": batch, "plant": plant, "product_division": prod["div"], "quantity": q})

        def tracking(p_has=0.93):
            if is_dc and rng.random() < p_has:
                nums = ", ".join("1Z" + digits(rng, 12) for _ in range(int(rng.integers(1, 3))))
                trk.append({"delivery_no": delivery, "carrier": CARRIERS[int(rng.integers(0, 4))], "tracking": nums})
            elif is_dc:
                trk.append({"delivery_no": delivery, "carrier": CARRIERS[int(rng.integers(0, 4))], "tracking": ""})

        def history(a, c, d, b):
            hist.append({"plant": plant, "sku": sku_padded, "batch": batch, "a": a, "c": c, "d": d, "b": b})

        if scenario in ("pending", "possible"):
            d_issue = today - timedelta(days=age_days())
            issue_doc(d_issue)
            if rng.random() < 0.22 and n > 1:  # partly scanned
                receipt_doc(min(today, d_issue + timedelta(days=int(rng.integers(1, 6)))), int(rng.integers(1, n)))
            tracking()
            if scenario == "possible":
                ret.append({"plant": plant, "sku": sku_padded, "batch": batch, "quantity": 2 * int(rng.integers(1, max(2, n // 2 + 1)))})
            if not is_dc:
                history(n, 0, 0, 0)
        elif scenario == "verify":
            d_issue = today - timedelta(days=age_days() + 5)
            issue_doc(d_issue)
            receipt_doc(min(today, d_issue + timedelta(days=int(rng.integers(1, 9)))), n)
            tracking()
            if not is_dc:
                history(n, n, 0, 0)
        elif scenario == "return_transit":
            ret.append({"plant": plant, "sku": sku_padded, "batch": batch, "quantity": 2 * n})
            history(0, 0, n, 0)
        elif scenario == "overdue":
            if not is_dc:
                history(n + int(rng.integers(0, 3)), 0, 0, 0)
        elif scenario == "old_return":
            history(n, n, n, 0)
        elif scenario == "old_stock":
            if rng.random() < 0.6:
                history(n, n, int(rng.integers(0, 3)), 0)
                hist[-1]["b"] = hist[-1]["d"]
        else:
            raise ValueError(scenario)

        bu = "DIV-B" if (plant == "FLD3" or (is_dc and prod["div"] == DIV_B_CODE)) else "DIV-A"
        for _ in range(n):
            serial = new_serial()
            fi.append({"branch": plant, "customer_no": cust["no"], "sku": prod["sku"], "description": prod["desc"],
                       "batch": batch, "vendor_batch": vendor, "serial": serial,
                       "expiry": today + timedelta(days=int(rng.integers(-20, 900))),
                       "customer_name": cust["name"], "territory": cust["terr_a"], "ta_region": cust["ta_region"],
                       "tis_region": cust["tis_region"], "tis_territory": cust["terr_b"], "sold_to": cust["sold_to"],
                       "qty_in_transit": 1})
            expected.append({"serial": serial, "plant": plant, "bu": bu, "expected_sort": STATUS_SORT[scenario]})

    dc_mix = (["pending"] * 60 + ["verify"] * 8 + ["overdue"] * 32)
    fld_mix = (["pending"] * 25 + ["possible"] * 5 + ["return_transit"] * 8 + ["verify"] * 5 +
               ["overdue"] * 12 + ["old_return"] * 24 + ["old_stock"] * 21)
    for _ in range(dc_keys):
        add_scenario(DC_PLANTS[int(rng.integers(0, 4))], dc_mix[int(rng.integers(0, len(dc_mix)))])
    for _ in range(field_keys):
        add_scenario(FIELD_PLANTS[int(rng.choice(3, p=[0.38, 0.34, 0.28]))], fld_mix[int(rng.integers(0, len(fld_mix)))])

    # noise: documents and history for units that are not in transit any more
    for _ in range(300):
        plant = (DC_PLANTS + FIELD_PLANTS)[int(rng.integers(0, 7))]
        prod = pick_product(plant)
        b = "N" + digits(rng, 7)
        d = today - timedelta(days=int(rng.integers(30, 700)))
        ref = ("08" + digits(rng, 8)) if plant in DC_PLANTS else digits(rng, 10)
        md = "4" + digits(rng, 9)
        q = int(rng.integers(1, 5))
        matdoc.append({"movement_type": ISS, "posting_date": d, "reference": ref, "matdoc_no": md, "sku": prod["sku"],
                       "batch": b, "plant": plant, "product_division": prod["div"], "quantity": q})
        matdoc.append({"movement_type": RCP, "posting_date": d + timedelta(days=2), "reference": ref if plant in DC_PLANTS else md,
                       "matdoc_no": "", "sku": prod["sku"], "batch": b, "plant": plant, "product_division": prod["div"], "quantity": q})

    # a few deliveries that are not tied to any in-transit unit
    for _ in range(150):
        trk.append({"delivery_no": "08" + digits(rng, 8), "carrier": CARRIERS[int(rng.integers(0, 4))],
                    "tracking": "1Z" + digits(rng, 12)})

    return {
        "matdoc": pd.DataFrame(matdoc), "field_inventory": pd.DataFrame(fi), "returns": pd.DataFrame(ret),
        "tracking": pd.DataFrame(trk), "field_history": pd.DataFrame(hist),
        "owner_alignment": alignment, "expected_status": pd.DataFrame(expected),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--today", default="2026-10-23", help="anchor date for every generated date (YYYY-MM-DD)")
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    tables = build(args.seed, today=date.fromisoformat(args.today))
    for name, df in tables.items():
        target = out / ("expected_status.csv" if name == "expected_status" else f"{name}.csv")
        df.to_csv(target, index=False)
        print(f"{target.name}: {len(df):,} rows")
    exp = tables["expected_status"]
    print("expected status counts (lines):", exp["expected_sort"].value_counts().sort_index().to_dict())


if __name__ == "__main__":
    main()
