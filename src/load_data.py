import pandas as pd

normal = pd.read_parquet(
    "data/tep_fault_free_training.parquet"
)

faulty = pd.read_parquet(
    "data/tep_faulty_training_runs01-20.parquet"
)

print("Normal shape:")
print(normal.shape)

print("\nFaulty shape:")
print(faulty.shape)

print("\nNormal columns:")
print(normal.columns.tolist())

print("\nFirst 5 rows of normal data:")
print(normal.head())

channels = (
    [f"xmeas_{i}" for i in range(1, 42)]
    +
    [f"xmv_{i}" for i in range(1, 12)]
)

print("\nNumber of channels:")
print(len(channels))

print("\nFirst few channel names:")
print(channels[:5])


train = normal[
    normal["simulationRun"].between(1, 300)
].copy()

val = normal[
    normal["simulationRun"].between(301, 400)
].copy()

test = normal[
    normal["simulationRun"].between(401, 500)
].copy()

print("\nTrain shape:", train.shape)
print("Validation shape:", val.shape)
print("Test shape:", test.shape)

mu = train[channels].mean()

sigma = train[channels].std(ddof=1)

print("\nFirst 5 training means:")
print(mu.head())

print("\nFirst 5 training std:")
print(sigma.head())

Z_train = (train[channels] - mu) / sigma
Z_val = (val[channels] - mu) / sigma
Z_test = (test[channels] - mu) / sigma
Z_faulty = (faulty[channels] - mu) / sigma

print("\nStandardized train shape:")
print(Z_train.shape)

print("\nFirst 5 standardized rows:")
print(Z_train.head())
