"""Creates the synthetic ID-registration practice dataset used in every notebook.
Nothing here is real data: every value is generated at random (but the same every run)."""

# numpy for random numbers, pandas for tables
import numpy as np, pandas as pd

# A random number generator with a fixed "seed" (42): same seed = same data every time you run it.
rng = np.random.default_rng(42)
n = 5000                                  # number of applications to create

# The regions centers can belong to
regions = ["Addis Ababa", "Oromia", "Amhara", "Tigray", "Sidama", "SNNP", "Afar", "Somali"]

# Build the centers table: 40 rows, one per enrollment center.
centers = pd.DataFrame({
    "center_id": range(1, 41),                       # IDs 1..40
    "region": rng.choice(regions, 40),               # pick a random region for each center
    "lat": rng.uniform(4, 14, 40).round(4),          # random latitude inside Ethiopia's range
    "lon": rng.uniform(34, 46, 40).round(4),         # random longitude
    "staff": rng.integers(3, 25, 40),                # 3 to 24 staff per center
})

# Build the applications table: one row per applicant.
apps = pd.DataFrame({
    "app_id": range(1, n + 1),                                          # IDs 1..5000
    "center_id": rng.integers(1, 41, n),                                # which center they applied at
    "age": rng.integers(16, 85, n).astype(float),                       # age 16..84 (float so it can hold NaN later)
    "gender": rng.choice(["F", "M"], n),
    "channel": rng.choice(["walk-in", "mobile-unit", "appointment"],    # how they applied,
                          n, p=[.5, .2, .3]),                           # with these probabilities
    "docs_complete": rng.choice([0, 1], n, p=[.25, .75]),               # 1 = documents complete (75%)
    "biometric_quality": rng.normal(0.8, 0.12, n).clip(0, 1).round(3),  # fingerprint/photo quality 0..1
    "submitted": pd.to_datetime("2026-01-01")                           # submission date: Jan 1 2026
                 + pd.to_timedelta(rng.integers(0, 300, n), "D"),       # plus 0..299 random days
})

# Bring each center's staff number onto its applications (needed for the formula below).
apps = apps.merge(centers[["center_id", "staff"]], on="center_id")

# Processing days follows a hidden rule plus noise, so later models have a real pattern to find:
# complete docs = 5 days faster, better biometrics = faster, more staff = faster.
apps["processing_days"] = (12 - 5 * apps.docs_complete - 8 * (apps.biometric_quality - .8)
                           - 0.15 * apps.staff + rng.normal(0, 2, n)).clip(1).round(1)

# Rejected (1) if biometrics are poor, or (40% of the time) if documents are incomplete.
apps["rejected"] = ((apps.biometric_quality < .65) |
                    (apps.docs_complete == 0) & (rng.random(n) < .4)).astype(int)

# Add realistic mess for the cleaning lessons:
apps.loc[rng.choice(n, 150, replace=False), "age"] = np.nan          # 150 missing ages
apps.loc[rng.choice(n, 60, replace=False), "channel"] = "Walk-In "   # 60 badly typed labels
apps = pd.concat([apps, apps.sample(40, random_state=1)])            # 40 duplicate rows

# Save both tables as CSV files (staff is dropped: it belongs to centers, not applications).
apps.drop(columns="staff").to_csv("applications.csv", index=False)
centers.to_csv("centers.csv", index=False)
print("Created applications.csv", apps.shape, "and centers.csv", centers.shape)
