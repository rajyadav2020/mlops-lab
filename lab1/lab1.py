import sys
from pathlib import Path
 
print("Python:", sys.version.split()[0])
print("Folder:", Path.cwd().name)
 
for name in ["numpy", "pandas", "sklearn"]:
    try:
        __import__(name)
        print(name, "- ok")
    except ImportError:
        print(name, "- missing")


#importing the numpy panadas and sklearn if they are imported send ok else name with missing 

#building the dataset
import csv
from pathlib import Path
import numpy as np

SEED = 42
N_ROWS = 600
DATA = Path("..") / "data" / "delivery_times.csv"


def make_delivery_csv(path=DATA):
    rng = np.random.default_rng(SEED)

    distance_km = np.round(
        rng.uniform(0.5, 12.0, N_ROWS), 2
    )

    prep_time_min = np.round(
        rng.uniform(5, 30, N_ROWS), 0
    )

    traffic_level = rng.integers(
        1, 4, N_ROWS
    )

    rain = rng.binomial(
        1, 0.25, N_ROWS
    )

    delivery_min = np.round(
        6.0
        + 3.1 * distance_km
        + 0.65 * prep_time_min
        + 4.2 * traffic_level
        + 5.5 * rain
        + rng.normal(0, 2.5, N_ROWS),
        1
    )

    path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    with path.open(
        "w",
        newline="",
        encoding="utf-8"
    ) as fh:

        w = csv.writer(fh)

        w.writerow([
            "distance_km",
            "prep_time_min",
            "traffic_level",
            "rain",
            "delivery_min"
        ])

        for i in range(N_ROWS):
            w.writerow([
                distance_km[i],
                int(prep_time_min[i]),
                int(traffic_level[i]),
                int(rain[i]),
                delivery_min[i]
            ])

    return path


if not DATA.exists():
    make_delivery_csv()

print("dataset ready:", DATA)



# running the python
import sys
from pathlib import Path

print("Python version:", sys.version.split()[0])
print("Python program:", sys.executable)
print("Inside .venv :", ".venv" in sys.executable)


# checking the installed libraries
from importlib.metadata import version

LIBRARIES = [
    "numpy",
    "pandas",
    "scikit-learn",
    "matplotlib"
]

for name in LIBRARIES:
    print(name, version(name))


# creating the requirements.txt
WORK = Path("work")

WORK.mkdir(exist_ok=True)

lines = [
    f"{name}=={version(name)}"
    for name in LIBRARIES
]

(WORK / "requirements.txt").write_text(
    "\n".join(lines) + "\n"
)

print(
    (WORK / "requirements.txt").read_text()
)


# random numbers without the seed 
careless = np.random.default_rng()

print(
    np.round(
        careless.uniform(0, 10, 3),
        2
    )
)


# random numbers with a seed 
first = np.random.default_rng(42).uniform(
    0, 10, 3
)

second = np.random.default_rng(42).uniform(
    0, 10, 3
)

print(np.round(first, 2))
print(np.round(second, 2))

print(
    "identical:",
    np.array_equal(first, second)
)


# build the dataset twice and compare 
import hashlib


def sha256_of(path):
    return hashlib.sha256(
        Path(path).read_bytes()
    ).hexdigest()


make_delivery_csv(
    WORK / "run_a.csv"
)

make_delivery_csv(
    WORK / "run_b.csv"
)

print(
    "identical files:",
    sha256_of(WORK / "run_a.csv")
    == sha256_of(WORK / "run_b.csv")
)

# inspect the dataset 
import pandas as pd

orders = pd.read_csv(DATA)

print(orders.shape)

print(orders.head())

print(
    orders.describe().round(1)
)


# record the run 
import json

run_info = {
    "python": sys.version.split()[0],
    "seed": SEED,
    "rows": len(orders),
    "data_sha256": sha256_of(DATA),
    "libraries": {
        n: version(n)
        for n in LIBRARIES
    }
}

(
    WORK / "run_info.json"
).write_text(
    json.dumps(
        run_info,
        indent=2
    )
)

print(
    json.dumps(
        run_info,
        indent=2
    )
)




# ================================================

# Task 1 - change seed to 7
import numpy as np

SEED = 7
N_ROWS = 600

rng = np.random.default_rng(SEED)

distance_km = np.round(
    rng.uniform(0.5, 12.0, N_ROWS),
    2
)

print("First three values:")
print(distance_km[:3])


# Task 2 create my requirements.txt 
from importlib.metadata import version
from pathlib import Path

LIBRARIES = [
    "numpy",
    "pandas",
    "scikit-learn"
]

WORK = Path("work")
WORK.mkdir(exist_ok=True)

lines = [
    f"{name}=={version(name)}"
    for name in LIBRARIES
]

(WORK / "my_requirements.txt").write_text(
    "\n".join(lines) + "\n"
)

print(
    (WORK / "my_requirements.txt").read_text()
)

# Task 3 - fingerprint a CSV 

import hashlib
import pandas as pd
from pathlib import Path


def fingerprint(path):
    path = Path(path)

    df = pd.read_csv(path)

    sha256 = hashlib.sha256(
        path.read_bytes()
    ).hexdigest()

    return {
        "rows": len(df),
        "sha256": sha256,
        "seed": SEED
    }


result = fingerprint(
    WORK / "run_a.csv"
)

print(result)