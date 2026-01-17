import sys
import os
import io
import pickle
import boto3
import numpy as np

from awsglue.utils import getResolvedOptions
from awsglue.context import GlueContext
from awsglue.job import Job
from pyspark.context import SparkContext

from pyspark.sql import functions as F
from pyspark.sql import types as T
from pyspark.sql.functions import udf
from pyspark.sql.types import IntegerType
from pyspark.ml.functions import vector_to_array

from pyspark.ml import Pipeline
from pyspark.ml.feature import (
    StringIndexer,
    OneHotEncoder,
    VectorAssembler,
    StandardScaler,
    PCA as SparkPCA
)
from pyspark.ml.clustering import KMeans
from pyspark.ml.evaluation import ClusteringEvaluator

# plotting (driver-only)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

# sklearn IsolationForest (driver fit + distributed scoring)
from sklearn.ensemble import IsolationForest


# -------------------------
# S3 helpers
# -------------------------
def s3_parse(s3_path: str):
    assert s3_path.startswith("s3://")
    p = s3_path.replace("s3://", "", 1).split("/", 1)
    bucket = p[0]
    key = p[1] if len(p) > 1 else ""
    return bucket, key

def s3_upload_bytes(data: bytes, s3_path: str, content_type: str = None):
    s3 = boto3.client("s3")
    bucket, key = s3_parse(s3_path)
    extra = {"ContentType": content_type} if content_type else {}
    s3.put_object(Bucket=bucket, Key=key, Body=data, **extra)

def s3_upload_file(local_path: str, s3_path: str):
    s3 = boto3.client("s3")
    bucket, key = s3_parse(s3_path)
    s3.upload_file(local_path, bucket, key)

def ensure_prefix(p: str) -> str:
    return p if p.endswith("/") else (p + "/")


# -------------------------
# Core parsing helpers
# -------------------------
def safe_bool(col):
    """Casts common string/int representations to boolean; unknown -> null."""
    return (
        F.when(F.col(col).isNull(), F.lit(None).cast("boolean"))
         .when(F.upper(F.col(col).cast("string")) == "TRUE", F.lit(True))
         .when(F.upper(F.col(col).cast("string")) == "FALSE", F.lit(False))
         .when(F.col(col).cast("string") == "1", F.lit(True))
         .when(F.col(col).cast("string") == "0", F.lit(False))
         .otherwise(F.col(col).cast("boolean"))
    )

def clean_parse_timestamp(df, colname: str):
    """
    Matches your local logic:
    - strip trailing 'Z'
    - parse ISO-like strings
    Spark equivalent: to_timestamp with patterns; invalid -> null.
    """
    s = F.col(colname).cast("string")
    s = F.when(s.isNull(), F.lit(None)) \
         .when(F.upper(s).isin("NA", "N/A", "NULL", "NONE", ""), F.lit(None)) \
         .otherwise(F.regexp_replace(s, "Z$", ""))

    # Try a few common patterns (covers most ISO8601-ish values)
    ts = F.coalesce(
        F.to_timestamp(s, "yyyy-MM-dd'T'HH:mm:ss"),
        F.to_timestamp(s, "yyyy-MM-dd HH:mm:ss"),
        F.to_timestamp(s)  # fallback parser
    )
    return df.withColumn(colname, ts)


# -------------------------
# 1) Read
# -------------------------
def read_data(glueContext: GlueContext, input_path: str):
    spark = glueContext.spark_session

    df = (
        spark.read
             .option("header", "true")
             .option("inferSchema", "true")     # critical
             .option("mode", "PERMISSIVE")
             .csv(input_path)
    )

    # normalize column names: strip spaces + standardize
    new_cols = [c.strip() for c in df.columns]
    for old, new in zip(df.columns, new_cols):
        if old != new:
            df = df.withColumnRenamed(old, new)

    return df



# -------------------------
# 2) Inspect/Clean (aligned)
# -------------------------
def inspect_clean_data(df):
    # Drop duplicates
    df = df.dropDuplicates()

    # Drop rows where transaction_country is null (aligned with local)
    if "transaction_country" in df.columns:
        df = df.filter(F.col("transaction_country").isNotNull())

    # Fill merchant_category for withdrawal with "first non-null withdrawal category"
    if all(c in df.columns for c in ["merchant_category", "transaction_type"]):
        first_row = (
            df.filter((F.col("transaction_type") == "withdrawal") & F.col("merchant_category").isNotNull())
              .select("merchant_category")
              .limit(1)
              .collect()
        )
        if first_row:
            default_cat = first_row[0]["merchant_category"]
            df = df.withColumn(
                "merchant_category",
                F.when((F.col("transaction_type") == "withdrawal") & F.col("merchant_category").isNull(), F.lit(default_cat))
                 .otherwise(F.col("merchant_category"))
            )

    # IMPORTANT: do NOT globally fill everything with "NA" in Spark (it breaks typing).
    # Instead: fill only string columns with "NA" to mimic your Pandas fillna('NA')
    str_cols = [f.name for f in df.schema.fields if isinstance(f.dataType, T.StringType)]
    if str_cols:
        df = df.na.fill("NA", subset=str_cols)

    # transaction_datetime: strip Z + parse
    if "transaction_datetime" in df.columns:
        df = clean_parse_timestamp(df, "transaction_datetime")

    # booleans
    for b in ["is_fraud", "is_online", "is_international"]:
        if b in df.columns:
            df = df.withColumn(b, safe_bool(b))

    # df = df[~(df['is_international'] == 'NA')] then astype(bool)
    # Here we already cast; also drop nulls created by invalid values.
    if "is_international" in df.columns:
        df = df.filter(F.col("is_international").isNotNull())

    # Derived flags
    if all(c in df.columns for c in ["is_online", "is_fraud"]):
        df = df.withColumn("is_online_fraud", (F.col("is_online") & F.col("is_fraud")))
    if all(c in df.columns for c in ["is_international", "is_fraud"]):
        df = df.withColumn("is_international_fraud", (F.col("is_international") & F.col("is_fraud")))

    # Reset index not needed in Spark
    return df


# -------------------------
# 3) EDA (aligned logic + plots to S3)
# -------------------------
def data_eda_and_feature_prep(df, output_prefix: str):
    """
    Aligns with your local data_eda():
    - adds month/dayofweek/hour
    - converts is_international/is_online to int
    - creates dummies for transaction_type, merchant_category, entry_mode
    - returns the Spark DF with encoded columns (vector columns, Spark-native)
    Also creates a subset of plots similar to your bar charts + scatter.
    """
    output_prefix = ensure_prefix(output_prefix)
    plots_prefix = output_prefix + "plots/"

    # time parts (only if timestamp exists)
    if "transaction_datetime" in df.columns:
        df = df.withColumn("transaction_month", F.month("transaction_datetime")) \
               .withColumn("transaction_dayofweek", F.dayofweek("transaction_datetime")) \
               .withColumn("transaction_hour", F.hour("transaction_datetime"))

    # bar plots for fraud by month/day/hour (Spark agg -> pandas small -> plot)
    def bar_plot(group_col: str, fname: str, title: str):
        if not all(c in df.columns for c in [group_col, "is_fraud"]):
            return

        agg = (df.filter(F.col(group_col).isNotNull())
                 .groupBy(group_col)
                 .agg(F.sum(F.col("is_fraud").cast("int")).alias("fraud_count"))
                 .orderBy(group_col))
        pdf = agg.toPandas()
        if pdf.shape[0] == 0:
            return

        plt.figure(figsize=(8, 4))
        sns.barplot(x=group_col, y="fraud_count", data=pdf)
        plt.title(title)
        plt.tight_layout()
        local = f"/tmp/{fname}"
        plt.savefig(local, dpi=200, bbox_inches="tight")
        plt.close()
        s3_upload_file(local, plots_prefix + fname)

    bar_plot("transaction_month", "transaction_month_bar_plot.png", "Fraud Count by Month")
    bar_plot("transaction_dayofweek", "transaction_dayofweek_bar_plot.png", "Fraud Count by Day of Week (Spark: 1=Sun..7=Sat)")
    bar_plot("transaction_hour", "transaction_hour_bar_plot.png", "Fraud Count by Hour")
    bar_plot("merchant_category", "merchant_category_bar_plot.png", "Fraud Count by merchant_category")
    bar_plot("transaction_type", "transaction_type_bar_plot.png", "Fraud Count by transaction_type")
    bar_plot("merchant_name", "merchant_name_bar_plot.png", "Fraud Count by merchant_name")

    # scatter plot cardholder_age vs transaction_amount
    if all(c in df.columns for c in ["cardholder_age", "transaction_amount"]):
        samp = (df.select("cardholder_age", "transaction_amount")
                  .dropna()
                  .sample(withReplacement=False, fraction=0.05, seed=42)
                  .limit(5000)
                  .toPandas())
        if samp.shape[0] > 0:
            plt.figure(figsize=(6, 4))
            plt.scatter(samp["cardholder_age"], samp["transaction_amount"], s=10)
            plt.title("cardholder_age vs transaction_amount (sample)")
            plt.tight_layout()
            local = "/tmp/sactter_plot_tr_crd_age.png"
            plt.savefig(local, dpi=200, bbox_inches="tight")
            plt.close()
            s3_upload_file(local, plots_prefix + "sactter_plot_tr_crd_age.png")

    # Cast is_international, is_online to int for plotting/feature
    for c in ["is_international", "is_online"]:
        if c in df.columns:
            df = df.withColumn(c, F.col(c).cast("int"))

    cat_cols = []
    for c in ["transaction_type", "merchant_category", "entry_mode"]:
        if c in df.columns:
            cat_cols.append(c)

    # Spark-native one-hot:
    # (Equivalent feature power vs pandas dummy columns; not literally separate columns)
    indexers = [StringIndexer(inputCol=c, outputCol=f"{c}_idx", handleInvalid="keep") for c in cat_cols]
    encoders = [OneHotEncoder(inputCol=f"{c}_idx", outputCol=f"{c}_ohe", dropLast=False) for c in cat_cols]

    if cat_cols:
        pipe = Pipeline(stages=indexers + encoders)
        ohe_model = pipe.fit(df)
        df = ohe_model.transform(df)

    return df


# -------------------------
# Feature selection aligned with your pandas logic
# -------------------------
def numeric_feature_columns_for_ml(df, exclude_cols=None):
    """
    Mimic pandas: df.select_dtypes(include='number')
    In Spark: numeric types include int/long/float/double/decimal.
    Exclude: is_fraud (label), and anything else passed.
    """
    if exclude_cols is None:
        exclude_cols = set()
    else:
        exclude_cols = set(exclude_cols)

    numeric_types = (T.IntegerType, T.LongType, T.ShortType, T.ByteType, T.FloatType, T.DoubleType, T.DecimalType)
    cols = []
    for f in df.schema.fields:
        if isinstance(f.dataType, numeric_types) and f.name not in exclude_cols:
            cols.append(f.name)
    return cols


# -------------------------
# 4) KMeans (aligned)
# -------------------------
def kmeans_clustering(df, n_clusters: int, pca_components: int, random_state: int, output_prefix: str):
    """
    - features: numeric columns after EDA+dummies (Spark OHE vectors + numeric cols)
    - scaling -> PCA -> KMeans -> add cluster_labels
    - silhouette score
    - plot PCA scatter to S3
    """
    output_prefix = ensure_prefix(output_prefix)
    plots_prefix = output_prefix + "plots/"

    # Build feature set:
    # 1) numeric columns (exclude label)
    numeric_cols = numeric_feature_columns_for_ml(df, exclude_cols={"is_fraud"})

    # 2) OHE vector cols for transaction_type/merchant_category/entry_mode
    ohe_cols = [c for c in df.columns if c.endswith("_ohe")]

    if len(numeric_cols) == 0 and len(ohe_cols) == 0:
        raise RuntimeError("No features available for KMeans.")

    assembler_inputs = numeric_cols + ohe_cols
    assembler = VectorAssembler(inputCols=assembler_inputs, outputCol="features_raw", handleInvalid="keep")
    scaler = StandardScaler(inputCol="features_raw", outputCol="features_scaled", withMean=True, withStd=True)
    pca = SparkPCA(k=pca_components, inputCol="features_scaled", outputCol="features_pca")
    kmeans = KMeans(k=n_clusters, seed=random_state, featuresCol="features_pca", predictionCol="cluster_labels")

    pipe = Pipeline(stages=[assembler, scaler, pca, kmeans])
    model = pipe.fit(df)
    pred = model.transform(df)

    # silhouette
    evaluator = ClusteringEvaluator(featuresCol="features_pca", predictionCol="cluster_labels", metricName="silhouette")
    silhouette = evaluator.evaluate(pred)
    print(f"[KMEANS] silhouette={silhouette:.4f} (k={n_clusters}, pca={pca_components})")

    # PCA scatter (sample)
    sample = (pred.select("features_pca", "cluster_labels")
                  .sample(False, 0.05, seed=42)
                  .limit(10000)
                  .toPandas())
    if sample.shape[0] > 0:
        # features_pca is a vector
        X = np.vstack(sample["features_pca"].apply(lambda v: np.array(v.toArray())).values)
        plt.figure(figsize=(7, 5))
        plt.scatter(X[:, 0], X[:, 1], c=sample["cluster_labels"].values, s=8)
        plt.title("KMeans clustering (PCA space) – sample")
        plt.tight_layout()
        local = "/tmp/kmeans_clustering.png"
        plt.savefig(local, dpi=200, bbox_inches="tight")
        plt.close()
        s3_upload_file(local, plots_prefix + "kmeans_clustering.png")

    return pred, model, silhouette


# -------------------------
# 5) Isolation Forest (aligned)
# -------------------------
def isolation_forest_model(df, contamination: float, random_state: int, max_samples, output_prefix: str):
    """
    Align with local:
    - features = numeric columns (drop cluster_labels)
    - StandardScaler
    - tune n_estimators list by max recall on is_fraud:
        recall = TP / total_fraud where TP = predicted anomaly (-1) AND is_fraud == true
    - then fit best model and attach predictions
    Implementation:
    - Fit scaler in Spark
    - Train iForest on SAMPLE on driver
    - Distributed scoring via pandas_udf + broadcast
    - Evaluate recall in Spark for each n_estimators
    """
    output_prefix = ensure_prefix(output_prefix)

    # Features: numeric cols excluding is_fraud and cluster_labels
    numeric_cols = numeric_feature_columns_for_ml(df, exclude_cols={"is_fraud", "cluster_labels"})
    if len(numeric_cols) == 0:
        raise RuntimeError("No numeric feature columns for IsolationForest.")
    

    assembler = VectorAssembler(inputCols=numeric_cols, outputCol="isf_features_raw", handleInvalid="keep")
    scaler = StandardScaler(inputCol="isf_features_raw", outputCol="isf_features_scaled", withMean=True, withStd=True)
    pipe = Pipeline(stages=[assembler, scaler])
    feat_model = pipe.fit(df)
    dff = feat_model.transform(df)

    # Total fraud count (denominator)
    if "is_fraud" not in dff.columns:
        raise RuntimeError("is_fraud column required for evaluation.")
    total_fraud = dff.filter(F.col("is_fraud") == F.lit(True)).count()
    if total_fraud == 0:
        print("[WARN] No fraud rows found; iForest recall metric is undefined. Proceeding without tuning.")
        total_fraud = 1

    # Sample for driver training
    # (modest as dataset is small)
    sample_pdf = (
        dff.select("isf_features_scaled")
           .dropna()
           .sample(False, 0.2, seed=random_state)
           .limit(50000)
           .toPandas()
    )
    if sample_pdf.shape[0] == 0:
        raise RuntimeError("Empty sample for iForest training.")

    X_train = np.vstack(sample_pdf["isf_features_scaled"].apply(lambda v: np.array(v.toArray())).values)


    n_est_list = [50, 100, 150, 200, 250]
    best_n = n_est_list[0]
    best_f1 = -1.0

    # UDF scorer factory
    def score_with_model_bytes(df_in, model_bytes_bc):
        # Create a UDF that takes an array<double> and returns int
        @udf(returnType=IntegerType())
        def pred_udf(arr):
            mdl = pickle.loads(model_bytes_bc.value)
            if arr is None:
                return None
            Xb = np.array(arr, dtype=float).reshape(1, -1)
            return int(mdl.predict(Xb)[0])
        # Convert VectorUDT -> array<double> first
        df2 = df_in.withColumn("isf_arr", vector_to_array(F.col("isf_features_scaled")))
        return df2.withColumn("predictions", pred_udf(F.col("isf_arr"))).drop("isf_arr")

    sc = df.sql_ctx.sparkSession.sparkContext

    for n_est in n_est_list:
        isf = IsolationForest(
            n_estimators=n_est,
            contamination=contamination,
            random_state=random_state,
            max_samples=max_samples
        )
        isf.fit(X_train)

        model_bytes = pickle.dumps(isf)
        bc = sc.broadcast(model_bytes)

        scored = score_with_model_bytes(dff, bc)

        # TP = predictions == -1 AND is_fraud == True
        tp = scored.filter((F.col("predictions") == -1) & (F.col("is_fraud") == True)).count()
        fp = scored.filter((F.col("predictions") == -1) & (F.col("is_fraud") == False)).count()
        fn = scored.filter((F.col("predictions") == 1) & (F.col("is_fraud") == True)).count()

        precision = tp / float(tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / float(tp + fn) if (tp + fn) > 0 else 0.0
        f1_score = (
            2 * precision * recall / (precision + recall)
            if (precision + recall) > 0 else 0.0
        )

        print(f"[ISF METRICS] n_estimators={n_est}, TP={tp}, FP={fp}, FN={fn}")
        print(f"Precision={precision:.4f}, Recall={recall:.4f}, F1={f1_score:.4f}")

        if f1_score > best_f1:
            best_f1 = f1_score
            best_n = n_est

        bc.unpersist()

    print(f"[ISF] Best n_estimators={best_n} with f1_score={best_f1:.4f}")

    # Fit final model and attach predictions
    final_isf = IsolationForest(
        n_estimators=best_n,
        contamination=contamination,
        random_state=random_state,
        max_samples=max_samples
    )
    final_isf.fit(X_train)
    bc_final = sc.broadcast(pickle.dumps(final_isf))
    scored_final = score_with_model_bytes(dff, bc_final)

    # Add binary 0/1 style df_pred like local (1 if anomaly else 0)
    scored_final = scored_final.withColumn("df_pred", F.when(F.col("predictions") == -1, F.lit(1)).otherwise(F.lit(0)))

    return scored_final

def drop_unsupported_for_csv(df):
    """
    CSV supports only primitive types. Drop complex columns like:
    struct/map/array and Spark ML vectors (VectorUDT shows up as struct).
    """
    keep_cols = []
    drop_cols = []

    for f in df.schema.fields:
        dt = f.dataType
        # Primitive types are OK
        if isinstance(dt, (T.StringType, T.BooleanType, T.ByteType, T.ShortType, T.IntegerType,
                           T.LongType, T.FloatType, T.DoubleType, T.DecimalType,
                           T.TimestampType, T.DateType)):
            keep_cols.append(f.name)
        else:
            drop_cols.append(f.name)

    if drop_cols:
        print("[INFO] Dropping unsupported CSV columns:", drop_cols)

    return df.select(*keep_cols)



# -------------------------
# 6) Write output
# -------------------------
def write_single_csv_exact_name(df, output_prefix: str, final_filename: str):
    """
    Writes exactly ONE CSV object to S3 with the exact filename.
    Spark writes to folders, so we:
      - coalesce(1) -> temp folder
      - locate part-*.csv
      - copy to final_filename
      - delete temp folder objects
    """
    output_prefix = ensure_prefix(output_prefix)
    assert final_filename.endswith(".csv")

    # 1) Write to temp folder (one part)
    temp_folder = output_prefix + "_tmp_single_csv_final/"
    (
        df.coalesce(1)
          .write.mode("overwrite")
          .option("header", True)
          .csv(temp_folder)
    )

    # 2) Find the single part file in temp folder
    s3 = boto3.client("s3")
    bucket, temp_key_prefix = s3_parse(temp_folder)

    # List objects under temp prefix
    resp = s3.list_objects_v2(Bucket=bucket, Prefix=temp_key_prefix)
    if "Contents" not in resp:
        raise RuntimeError(f"No objects found under temp folder: {temp_folder}")

    part_keys = [obj["Key"] for obj in resp["Contents"] if obj["Key"].endswith(".csv")]
    if len(part_keys) != 1:
        # Sometimes list_objects_v2 paginates; handle pagination robustly
        paginator = s3.get_paginator("list_objects_v2")
        part_keys = []
        for page in paginator.paginate(Bucket=bucket, Prefix=temp_key_prefix):
            for obj in page.get("Contents", []):
                if obj["Key"].endswith(".csv"):
                    part_keys.append(obj["Key"])

    if len(part_keys) != 1:
        raise RuntimeError(f"Expected 1 part csv, found {len(part_keys)} under {temp_folder}: {part_keys}")

    part_key = part_keys[0]

    # 3) Copy to final exact filename
    final_s3_path = output_prefix + final_filename
    final_bucket, final_key = s3_parse(final_s3_path)

    # (Same bucket expected; if different bucket, copy still works with correct perms)
    s3.copy_object(
        Bucket=final_bucket,
        Key=final_key,
        CopySource={"Bucket": bucket, "Key": part_key},
        ContentType="text/csv"
    )

    # 4) (Optional) delete temp folder objects
    # Delete everything under temp prefix (part file + _SUCCESS etc.)
    delete_keys = []
    paginator = s3.get_paginator("list_objects_v2")
    for page in paginator.paginate(Bucket=bucket, Prefix=temp_key_prefix):
        for obj in page.get("Contents", []):
            delete_keys.append({"Key": obj["Key"]})

    # Batch delete in chunks of 1000
    for i in range(0, len(delete_keys), 1000):
        s3.delete_objects(Bucket=bucket, Delete={"Objects": delete_keys[i:i+1000]})

    print(f"[OK] Wrote single CSV: {final_s3_path}")


# -------------------------
# Main
# -------------------------
def main():
    args = getResolvedOptions(
        sys.argv,
        [
            "JOB_NAME",
            "input_path",
            "output_path",
            "contamination",
            "random_state",
            "max_samples",
            "n_clusters",
            "pca_components"
        ]
    )

    input_path = args["input_path"]
    output_path = ensure_prefix(args["output_path"])

    contamination = float(args["contamination"])
    random_state = int(args["random_state"])
    max_samples_raw = args["max_samples"]
    max_samples = max_samples_raw if max_samples_raw == "auto" else int(max_samples_raw)

    n_clusters = int(args["n_clusters"])
    pca_components = int(args["pca_components"])

    sc = SparkContext.getOrCreate()
    glueContext = GlueContext(sc)
    spark = glueContext.spark_session

    job = Job(glueContext)
    job.init(args["JOB_NAME"], args)

    # 1) Read
    df = read_data(glueContext, input_path)
    print("[INFO] Input schema:")
    df.printSchema()

    # 2) Clean
    df = inspect_clean_data(df)

    # 3) EDA + dummy-like encoding
    df = data_eda_and_feature_prep(df, output_path)

    # 4) KMeans
    df, km_model, sil = kmeans_clustering(
        df,
        n_clusters=n_clusters,
        pca_components=pca_components,
        random_state=random_state,
        output_prefix=output_path
    )
    # 5) Isolation Forest
    df = isolation_forest_model(
        df,
        contamination=contamination,
        random_state=random_state,
        max_samples=max_samples,
        output_prefix=output_path
    )
    # 6) Write final output
    df_csv = drop_unsupported_for_csv(df)
    write_single_csv_exact_name(df_csv, output_path, "final_credit_card_data.csv")


    job.commit()
    print("[OK] Glue Spark job completed.")


if __name__ == "__main__":
    main()
