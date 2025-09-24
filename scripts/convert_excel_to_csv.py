import os, glob
import pandas as pd

SRC = r"data/raw/tennisdata/excel"
DST = r"data/raw/tennisdata"
os.makedirs(DST, exist_ok=True)

def convert_one(x_path: str):
    xls = pd.ExcelFile(x_path)
    for sheet in xls.sheet_names:
        df = pd.read_excel(x_path, sheet_name=sheet)
        base = os.path.splitext(os.path.basename(x_path))[0]
        out = os.path.join(DST, f"{base}_{sheet}.csv")
        df.to_csv(out, index=False, encoding="utf-8")
        print("wrote", out)

def main():
    paths = glob.glob(os.path.join(SRC, "*.xls*"))
    if not paths:
        print("no excel files in", SRC)
        return
    for p in paths:
        convert_one(p)

if __name__ == "__main__":
    main()
