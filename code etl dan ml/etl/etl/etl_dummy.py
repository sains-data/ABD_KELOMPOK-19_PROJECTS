from pyspark.sql import SparkSession
from pyspark.sql.functions import col
import os

def main():
    spark = SparkSession.builder \
        .appName("ETL_Pertanian") \
        .config("spark.sql.adaptive.enabled", "true") \
        .getOrCreate()

    print("=== MULAI ETL PROCESS ===")

    # Definisikan path input dan output untuk HDFS
    input_path = "hdfs://namenode_host:8020/bigdata/bronze/pertanian/data_pertanian_sumatera_3juta.csv"
    output_path = "hdfs://namenode_host:8020/bigdata/silver/pertanian/data_clean/etl_data_dummy"

    try:
        # Cek apakah file input ada di HDFS
        if not spark._jvm.org.apache.hadoop.fs.FileSystem.get(spark._jsc.hadoopConfiguration()).exists(spark._jvm.org.apache.hadoop.fs.Path(input_path)):
            print(f"Error: File input tidak ditemukan di HDFS path: {input_path}")
            spark.stop()
            return

        print(f"1. Membaca data dari Bronze Layer di: {input_path}")

        # Membaca CSV dengan header dan infer schema otomatis dari HDFS
        df = spark.read.csv(input_path, header=True, inferSchema=True)

        total_records = df.count()
        print(f"Total record awal: {total_records}")

        print("Schema data:")
        df.printSchema()

        print("Contoh data (5 baris pertama):")
        df.show(5, truncate=False)

        print("2. Membersihkan data (drop baris dengan nilai NULL)...")
        df_clean = df.dropna()
        clean_count = df_clean.count()
        print(f"Record setelah cleaning: {clean_count} (hilang {total_records - clean_count} record)")

        print("3. Menambah kolom 'produktivitas' = produksi_ton / luas_panen_ha...")
        df_transformed = df_clean.withColumn("produktivitas", col("produksi_ton") / col("luas_panen_ha"))

        print("Filter data dengan produktivitas > 0...")
        df_final = df_transformed.filter(col("produktivitas") > 0)
        final_count = df_final.count()
        print(f"Record final setelah filter produktivitas > 0: {final_count} (hilang {clean_count - final_count} record)")

        print(f"4. Menyimpan hasil ke Silver Layer di: {output_path}")
        df_final.coalesce(1) \
                .write.mode("overwrite") \
                .option("header", "true") \
                .csv(output_path)

        print("=== ETL PROCESS SELESAI ===")

    except Exception as e:
        print(f"Error saat ETL process: {str(e)}")

    finally:
        spark.stop()
        print("Spark session dihentikan")

if __name__ == "__main__":
    main()
