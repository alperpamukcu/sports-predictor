import os
import pandas as pd
from src.data.fetch_tennis import list_venues

def main():
    data = list_venues()
    df = pd.DataFrame(data)

    out_dir = os.path.join("data", "external")
    os.makedirs(out_dir, exist_ok=True)

    out_path = os.path.join(out_dir, "venues_snapshot.csv")

    df.to_csv(out_path, index=False, encoding="utf-8")
    print(f"Wrote {len(df)} rows -> {out_path}")
    
    try:
        print(df.head(5).to_string())
    except Exception:
        pass

if __name__ == "__main__":
    main()
