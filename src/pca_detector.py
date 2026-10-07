import pandas as pd
import numpy as np
from sklearn.decomposition import PCA


# =========================================================
# 1. Load data
# =========================================================

normal = pd.read_parquet(
    "data/tep_fault_free_training.parquet"
)

faulty = pd.read_parquet(
    "data/tep_faulty_training_runs01-20.parquet"
)


# =========================================================
# 2. Define the 52 process channels
# =========================================================

channels = (
    [f"xmeas_{i}" for i in range(1, 42)]
    +
    [f"xmv_{i}" for i in range(1, 12)]
)

print("Number of channels:", len(channels))


# =========================================================
# 3. Split normal data into train / validation / test
# =========================================================

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


# =========================================================
# 4. Standardize using TRAINING statistics only
# =========================================================

mu = train[channels].mean()
sigma = train[channels].std(ddof=1)

Z_train = (train[channels] - mu) / sigma
Z_val = (val[channels] - mu) / sigma
Z_test = (test[channels] - mu) / sigma
Z_faulty = (faulty[channels] - mu) / sigma


# =========================================================
# 5. Sanity check
# =========================================================

print("\n--- Debug Z_train ---")

print("Shape:")
print(Z_train.shape)

print("\nAny NaN?")
print(Z_train.isna().any().any())

print("\nAny infinity?")
print(np.isinf(Z_train.to_numpy()).any())

print("\nLargest absolute value:")
print(np.abs(Z_train.to_numpy()).max())

print("\nFirst 5 means:")
print(Z_train.mean().head())

print("\nFirst 5 standard deviations:")
print(Z_train.std(ddof=1).head())


# =========================================================
# 6. Find number of PCA components needed for 90% variance
# =========================================================

pca_full = PCA(
    svd_solver="full"
)

pca_full.fit(Z_train)

explained = pca_full.explained_variance_ratio_

cumulative = np.cumsum(explained)

k = np.argmax(
    cumulative >= 0.90
) + 1

print("\nNumber of principal components needed:")
print(k)

print("\nCumulative explained variance:")
print(cumulative[k - 1])


# =========================================================
# 7. Fit final PCA using k components
# =========================================================

pca = PCA(
    n_components=k,
    svd_solver="full"
)

pca.fit(Z_train)

print("\nPCA fitted.")

print("PCA components shape:")
print(pca.components_.shape)


# =========================================================
# 8. Inspect transformed training data
# =========================================================

T_train = pca.transform(
    Z_train
)

print("\nPCA transformed training data shape:")
print(T_train.shape)

print("\nFirst transformed sample:")
print(T_train[0])


# =========================================================
# 9. T² score
# =========================================================

def t2_score(Z, pca):

    # Map original 52-dimensional data
    # into k PCA coordinates
    T = pca.transform(Z)

    # Standardized squared distance
    # inside PCA space
    score = np.sum(
        (T ** 2)
        / pca.explained_variance_,
        axis=1
    )

    return score


# =========================================================
# 10. SPE score
# =========================================================

def spe_score(Z, pca):

    # Transform into PCA space
    T = pca.transform(Z)

    # Reconstruct back to 52 dimensions
    Z_hat = pca.inverse_transform(T)

    # Part not explained by PCA
    residual = (
        Z.to_numpy()
        - Z_hat
    )

    # Sum squared reconstruction errors
    score = np.sum(
        residual ** 2,
        axis=1
    )

    return score


# =========================================================
# 11. Calculate PCA anomaly scores
# =========================================================

t2_train = t2_score(
    Z_train,
    pca
)

t2_val = t2_score(
    Z_val,
    pca
)

t2_test = t2_score(
    Z_test,
    pca
)

t2_faulty = t2_score(
    Z_faulty,
    pca
)


spe_train = spe_score(
    Z_train,
    pca
)

spe_val = spe_score(
    Z_val,
    pca
)

spe_test = spe_score(
    Z_test,
    pca
)

spe_faulty = spe_score(
    Z_faulty,
    pca
)

print("\nT2 train shape:")
print(t2_train.shape)

print("\nSPE train shape:")
print(spe_train.shape)


# =========================================================
# 12. Determine thresholds using VALIDATION data
# =========================================================

t2_threshold = np.quantile(
    t2_val,
    0.99
)

spe_threshold = np.quantile(
    spe_val,
    0.99
)

print("\nT2 threshold:")
print(t2_threshold)

print("\nSPE threshold:")
print(spe_threshold)


# =========================================================
# 13. Check simple threshold exceedance rates
# =========================================================

t2_test_exceed = (
    t2_test > t2_threshold
)

spe_test_exceed = (
    spe_test > spe_threshold
)

t2_faulty_exceed = (
    t2_faulty > t2_threshold
)

spe_faulty_exceed = (
    spe_faulty > spe_threshold
)

print("\nTest T2 exceed rate:")
print(t2_test_exceed.mean())

print("\nTest SPE exceed rate:")
print(spe_test_exceed.mean())

print("\nFaulty T2 exceed rate:")
print(t2_faulty_exceed.mean())

print("\nFaulty SPE exceed rate:")
print(spe_faulty_exceed.mean())


# =========================================================
# 14. Three consecutive exceedances = alarm
# =========================================================

def three_consecutive_alarm(exceed):

    exceed = np.asarray(
        exceed,
        dtype=bool
    )

    alarm = np.zeros_like(
        exceed,
        dtype=bool
    )

    alarm[2:] = (
        exceed[:-2]
        &
        exceed[1:-1]
        &
        exceed[2:]
    )

    return alarm


# =========================================================
# 15. Apply alarms separately inside each NORMAL run
# =========================================================

def make_normal_alarm_result(
    df,
    scores,
    threshold
):

    result = df[
        [
            "simulationRun",
            "sample"
        ]
    ].copy()

    result["score"] = scores

    result["exceed"] = (
        result["score"]
        > threshold
    )

    result["alarm"] = False

    for run_id, group in result.groupby(
        "simulationRun",
        sort=False
    ):

        idx = group.index

        alarms = three_consecutive_alarm(
            group["exceed"].to_numpy()
        )

        result.loc[
            idx,
            "alarm"
        ] = alarms

    return result


# =========================================================
# 16. Apply alarms separately inside each FAULTY run
# =========================================================

def make_fault_alarm_result(
    df,
    scores,
    threshold
):

    result = df[
        [
            "faultNumber",
            "simulationRun",
            "sample"
        ]
    ].copy()

    result["score"] = scores

    result["exceed"] = (
        result["score"]
        > threshold
    )

    result["alarm"] = False

    for key, group in result.groupby(
        [
            "faultNumber",
            "simulationRun"
        ],
        sort=False
    ):

        idx = group.index

        alarms = three_consecutive_alarm(
            group["exceed"].to_numpy()
        )

        result.loc[
            idx,
            "alarm"
        ] = alarms

    return result


# =========================================================
# 17. Build alarm results
# =========================================================

t2_test_result = make_normal_alarm_result(
    test,
    t2_test,
    t2_threshold
)

spe_test_result = make_normal_alarm_result(
    test,
    spe_test,
    spe_threshold
)

t2_fault_result = make_fault_alarm_result(
    faulty,
    t2_faulty,
    t2_threshold
)

spe_fault_result = make_fault_alarm_result(
    faulty,
    spe_faulty,
    spe_threshold
)


# =========================================================
# 18. False alarm rate on normal TEST data
# =========================================================

t2_false_alarm_rate = (
    t2_test_result["alarm"].mean()
)

spe_false_alarm_rate = (
    spe_test_result["alarm"].mean()
)

print("\nT2 false alarm rate:")
print(t2_false_alarm_rate)

print("\nSPE false alarm rate:")
print(spe_false_alarm_rate)


# =========================================================
# 19. Evaluate each fault
# =========================================================

def evaluate_faults(result):

    rows = []

    for fault_number, fault_df in result.groupby(
        "faultNumber"
    ):

        detection_rates = []
        delays = []
        runs_missed = 0

        for run_id, run_df in fault_df.groupby(
            "simulationRun"
        ):

            # Fault begins after sample 20
            faulty_period = run_df[
                run_df["sample"] > 20
            ].copy()

            # Detection rate for this run
            detection_rate = (
                faulty_period["alarm"].mean()
            )

            detection_rates.append(
                detection_rate
            )

            # Find first alarm
            alarm_rows = faulty_period[
                faulty_period["alarm"]
            ]

            if len(alarm_rows) == 0:

                runs_missed += 1

            else:

                first_alarm_sample = (
                    alarm_rows[
                        "sample"
                    ].iloc[0]
                )

                delay_minutes = (
                    first_alarm_sample
                    - 20
                ) * 3

                delays.append(
                    delay_minutes
                )

        mean_detection_rate = np.mean(
            detection_rates
        )

        if len(delays) > 0:

            median_delay = np.median(
                delays
            )

        else:

            median_delay = np.nan

        rows.append({
            "fault": int(fault_number),
            "detection_rate":
                mean_detection_rate,
            "median_delay_min":
                median_delay,
            "runs_missed":
                runs_missed
        })

    return pd.DataFrame(rows)


# =========================================================
# 20. Evaluate T2 and SPE
# =========================================================

t2_metrics = evaluate_faults(
    t2_fault_result
)

t2_metrics["detector"] = "T2"


spe_metrics = evaluate_faults(
    spe_fault_result
)

spe_metrics["detector"] = "SPE"


print("\nT2 metrics:")
print(t2_metrics)

print("\nSPE metrics:")
print(spe_metrics)


# =========================================================
# 21. Save PCA scores
# =========================================================

val_scores = val[
    [
        "simulationRun",
        "sample"
    ]
].copy()

val_scores["faultNumber"] = 0
val_scores["T2"] = t2_val
val_scores["SPE"] = spe_val


test_scores = test[
    [
        "simulationRun",
        "sample"
    ]
].copy()

test_scores["faultNumber"] = 0
test_scores["T2"] = t2_test
test_scores["SPE"] = spe_test


fault_scores = faulty[
    [
        "faultNumber",
        "simulationRun",
        "sample"
    ]
].copy()

fault_scores["T2"] = t2_faulty
fault_scores["SPE"] = spe_faulty


scores_pca = pd.concat(
    [
        val_scores[
            [
                "faultNumber",
                "simulationRun",
                "sample",
                "T2",
                "SPE"
            ]
        ],
        test_scores[
            [
                "faultNumber",
                "simulationRun",
                "sample",
                "T2",
                "SPE"
            ]
        ],
        fault_scores[
            [
                "faultNumber",
                "simulationRun",
                "sample",
                "T2",
                "SPE"
            ]
        ]
    ],
    ignore_index=True
)

scores_pca.to_parquet(
    "results/scores_pca.parquet",
    index=False
)

print(
    "\nSaved results/scores_pca.parquet"
)


# =========================================================
# 22. Save PCA thresholds temporarily
# =========================================================

thresholds_pca = pd.DataFrame({
    "detector": [
        "T2",
        "SPE"
    ],
    "threshold": [
        t2_threshold,
        spe_threshold
    ]
})

thresholds_pca.to_csv(
    "results/thresholds_pca_temp.csv",
    index=False
)

print(
    "Saved results/thresholds_pca_temp.csv"
)