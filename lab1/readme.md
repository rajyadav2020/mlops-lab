MLOps Lab 1 --- Your MLOps Workbench

Aim / Objective

Find out which Python is actually running your code.

Write down the exact library versions your project needs.

Make a program that gives the same answer every single time.

Step 0 --- Check the Workbench

Run the provided code to:

Print the Python version.

Print the current working folder.

Check whether numpy, pandas, and sklearn are installed.

Expected

Python: 3.12.x
Folder: P01-workbench
numpy - ok
pandas - ok
sklearn - ok

Step 0b --- Build the Shared Dataset

Create delivery_times.csv using:

SEED = 42

N_ROWS = 600

The dataset must contain:

distance_km

prep_time_min

traffic_level

rain

delivery_min

Save it under:

../data/delivery_times.csv

The dataset should only be generated if it does not already exist.

Expected

dataset ready: ../data/delivery_times.csv

Step 1 --- Which Python Is Running This Notebook?

Display:

Python version

Exact Python executable path

Whether the executable is inside .venv

Expected

Python version: 3.12.x
Python program: ...\.venv\Scripts\python.exe
Inside .venv : True

Step 2 --- What Is Installed in This Environment?

Check the installed versions of:

numpy
pandas
scikit-learn
matplotlib

Use Python's installed package metadata to print the exact versions.

Step 3 --- Freeze Versions into requirements.txt

Create a work directory and generate:

work/requirements.txt

Each library must be pinned using:

library==version

The file should contain:

numpy==<installed version>
pandas==<installed version>
scikit-learn==<installed version>
matplotlib==<installed version>

Step 4 --- Random Numbers Change Every Time

Create a random-number generator without specifying a seed.

Generate three random numbers between 0 and 10.

Run the cell more than once and observe that the values can change.

Step 5 --- A Seed Makes Randomness Repeat

Create two random generators using:

seed = 42

Generate three numbers from each and verify that they are identical.

Expected

[7.74 4.39 8.59]
[7.74 4.39 8.59]
identical: True

Step 6 --- Build the Delivery Dataset Twice

Create:

work/run_a.csv
work/run_b.csv

Use the same deterministic dataset-generation function for both files.

Create a SHA-256 function and compare the two file hashes.

Expected

identical files: True

Step 7 --- Look at the Data

Load:

../data/delivery_times.csv

using pandas.

Display:

Dataset shape

First five rows

Descriptive statistics

Expected shape

(600, 5)

Step 8 --- Record What You Ran

Create a dictionary containing:

python
seed
rows
data_sha256
libraries

Save it as:

work/run_info.json

Print the JSON file contents.

The record should capture the Python version, seed, row count, dataset
checksum, and installed library versions.

Step 9 --- Save Your Work in Git

Initialize a Git repository inside:

work/

Configure:

user.name  = SCSE3040 Student
user.email = student@bennett.edu.in

Add:

requirements.txt
run_info.json

Create a commit with the message:

P01: pinned requirements and run record

Finally, display the Git log using the one-line format.

Your Tasks

T1 --- Change the Seed

Change the seed from:

42

to:

7

Rebuild the distance_km values and report the first three values.

Expected first three values

[7.69 10.82  9.42]

T2 --- Create Your Own Pinned Requirements File

Create:

work/my_requirements.txt

It must contain only these three libraries:

numpy
pandas
scikit-learn

Pin each library to its currently installed version.

Format:

numpy==<installed version>
pandas==<installed version>
scikit-learn==<installed version>

T3 --- Fingerprint a Run

Write a function:

fingerprint(path)

The function must return a dictionary containing:

rows
sha256
seed

Example structure:

{
    "rows": 600,
    "sha256": "...",
    "seed": 42
}
