from pyspark.sql import SparkSession
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.regression import RandomForestRegressor
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.linalg import Vectors
from pyspark.sql.functions import udf
from pyspark.sql.types import StringType

# Fungsi untuk mengonversi vector menjadi string
def vector_to_string(vector):
    return str(vector)

# Mendaftarkan UDF untuk mengonversi vector ke string
vector_to_string_udf = udf(vector_to_string, StringType())

def init_spark(app_name="ML_Pertanian"):
    print("[0] Inisialisasi Spark Session dengan konfigurasi memori tambahan...")
    return SparkSession.builder \
        .appName(app_name) \
        .config("spark.driver.memory", "4g") \
        .config("spark.executor.memory", "4g") \
        .config("spark.sql.shuffle.partitions", "100") \
        .config("spark.default.parallelism", "100") \
        .getOrCreate()

def read_data_from_silver(spark, path):
    print("[1] Membaca data dari Silver Layer...")
    df = spark.read.option("header", True).option("inferSchema", True).csv(path)
    print(f">>> Total record ditemukan: {df.count()}")
    return df

def prepare_features_and_target(df, fitur, target):
    print("[2] Persiapan fitur dan target...")

    kolom_tersedia = df.columns
    fitur_valid = [f for f in fitur if f in kolom_tersedia]

    if not fitur_valid:
        raise ValueError("❌ Tidak ada kolom fitur yang valid ditemukan.")
    if target not in kolom_tersedia:
        raise ValueError(f"❌ Kolom target '{target}' tidak ditemukan.")

    df_clean = df.dropna(subset=fitur_valid + [target])
    print(f">>> Jumlah record setelah pembersihan: {df_clean.count()}")

    assembler = VectorAssembler(inputCols=fitur_valid, outputCol="features")
    df_vector = assembler.transform(df_clean).select("provinsi", "tahun", "komoditas", "produksi_ton", "features", target).cache()

    return df_vector

def train_model(df_vector, target):
    print("[3] Membagi data menjadi train dan test (80:20)...")
    train_data, test_data = df_vector.randomSplit([0.8, 0.2], seed=42)

    # Optional sampling jika data sangat besar (misal lebih dari 1 juta record)
    if train_data.count() > 1_000_000:
        print(">>> Sampling data untuk pelatihan karena data sangat besar...")
        train_data = train_data.sample(fraction=0.5, seed=42)

    print("[4] Melatih model Random Forest Regressor (numTrees=30, maxDepth=7)...")
    rf = RandomForestRegressor(labelCol=target, featuresCol="features", numTrees=30, maxDepth=7)
    model = rf.fit(train_data)

    return model, test_data

def evaluate_model(model, test_data, target):
    print("[5] Mengevaluasi performa model...")
    predictions = model.transform(test_data)

    evaluator = RegressionEvaluator(labelCol=target, predictionCol="prediction", metricName="rmse")
    rmse = evaluator.evaluate(predictions)
    print(f">>> Root Mean Squared Error (RMSE): {rmse:.4f}")

    return predictions

def save_predictions(predictions, output_path):
    print("[6] Menyimpan hasil prediksi ke Gold Layer (CSV)...")

    # Menghapus kolom 'features' dan menyiapkan output
    predictions = predictions.drop("features")

    # Menyimpan hasil prediksi ke CSV
    predictions.select("provinsi", "tahun", "komoditas", "produksi_ton", "prediction") \
        .write.mode("overwrite") \
        .option("header", "true") \
        .csv(output_path)
    
    print(">>> Hasil prediksi berhasil disimpan ke CSV.")

if __name__ == "__main__":
    print("=== [MULAI] Proses Machine Learning Pertanian ===")

    spark = init_spark()

    try:
        # Ganti dengan path HDFS
        input_path = "hdfs://localhost:9000/mnt/e/bigdata-hadoop/silver/pertanian/data_clean"
        output_path = "/mnt/e/bigdata-hadoop/gold/pertanian/predictions"

        fitur = ['luas_panen_ha', 'curah_hujan_mm', 'kelembapan_%', 'suhu_celsius']
        target = 'produksi_ton'

        df_spark = read_data_from_silver(spark, input_path)
        df_vector = prepare_features_and_target(df_spark, fitur, target)
        model, test_data = train_model(df_vector, target)
        predictions = evaluate_model(model, test_data, target)
        save_predictions(predictions, output_path)

    except Exception as e:
        print(f"[❌ ERROR] Terjadi kesalahan: {str(e)}")

    finally:
        spark.stop()
        print(">>> Spark session ditutup.")
        print("=== [SELESAI] Proses Machine Learning Pertanian ===")
