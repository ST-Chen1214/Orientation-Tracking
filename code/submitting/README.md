```bash
How to Run
1) Training Set (Orientation Tracking + Ground Truth Comparison)
python orient_pgd.py 

Change dataset ID (1–9):

Open orient_pgd.py

Go to line 138

Replace the dataset number with the one you want (1 to 9)

Example:

dataset_id = "3"  # change to "1"..."9"


2) Test Data
python orient_pgd_testdata.py

Change dataset ID (10–11):

Open orient_pgd_testdata.py

Go to line 125

Replace the dataset number with the one you want (10 or 11)

Example:

dataset_id = "10"  # change to "10" or "11"


3) Panorama Reconstruction
python run_panorama.py 

Change dataset ID:

Open run_panorama.py

Go to line 28

Replace dataset number with the one you want

Example:

dataset_id = "2" # change to 1, 2, 8, 9, 10, 11