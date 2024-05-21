# Databricks notebook source
# Se importan las bibliotecas necesarias 
import pyspark
from pyspark.context import SparkContext
from pyspark.sql import SQLContext , Row
from pyspark.sql.functions import *
from pyspark.sql.types import *
import pandas as pd
from pyspark.ml.feature import DenseVector,StandardScaler

# COMMAND ----------

# se levanta la sesion pyspark para hacer uso de los metodos y herramientas que dispone 
# Se obtiene o crea el contexto Spark
sc= SparkContext.getOrCreate()
# Se crea el contexto SQL sobre el contexto Spark
sql_sc = SQLContext(sc)
sc

# COMMAND ----------

# MAGIC %fs
# MAGIC
# MAGIC ls dbfs:/FileStore/

# COMMAND ----------

#url = "default.nypd_arrest_data__year_to_date__20240402_csv"
#arrestos_DFP= pd.read_csv(url,sep=",")
arrestos_DFS = spark.read.format("csv").option("header", "true").load("dbfs:/FileStore/shared_uploads/al_salamanca@javeriana.edu.co/NYPD_Arrest_Data__Year_to_Date__20240402.csv")
# Se muestra una vista previa de las primeras filas del DataFrame
arrestos_DFS.head()

# COMMAND ----------

#arrestos_DFS= sql_sc.createDataFrame(arrestos_DFP)
# Se muestra una vista previa de las primeras 5 filas del DataFrame Spark
arrestos_DFS.show(5)

# COMMAND ----------

arrestos_DFS.printSchema()

# COMMAND ----------

display(arrestos_DFS)

# COMMAND ----------

pobresa_DFS = spark.read.format("csv").option("header", "true").load("dbfs:/FileStore/shared_uploads/al_salamanca@javeriana.edu.co/NYCgov_Poverty_Measure_Data__2018__20240402.csv")
pobresa_DFS.show(5)

# COMMAND ----------

#pobresa_DFS= sql_sc.createDataFrame(pobreza_DFP)
# Se muestra una vista previa de las primeras 5 filas del DataFrame Spark
#pobresa_DFS.show(5)

# COMMAND ----------

display(pobresa_DFS)

# COMMAND ----------

arrestos_DFS = arrestos_DFS.replace("(No value)",None)
arrestos_DFS = arrestos_DFS.replace("UNKNOWN",None)
arrestos_DFS = arrestos_DFS.replace("(null)",None)

# COMMAND ----------

arrestos_DFS.select(
    [count(when(isnan(c) | col(c).isNull() | (col(c) == "(No value)") | (col(c) == "UNKNOWN")| (col(c) == "(null)"), c)).alias(c) for c in arrestos_DFS.columns]
).toPandas()


# COMMAND ----------

moda_law_cat_cd = arrestos_DFS.groupBy("LAW_CAT_CD").count().orderBy("count", ascending=False).first()[0]

print(moda_law_cat_cd)

# COMMAND ----------

# Reemplazar los valores nulos en "LAW_CAT_CD" con la moda
arrestos_DFS = arrestos_DFS.withColumn("LAW_CAT_CD", when(arrestos_DFS["LAW_CAT_CD"].isNull(), moda_law_cat_cd).otherwise(arrestos_DFS["LAW_CAT_CD"]))

# COMMAND ----------

# Calcular la moda de la columna "PERP_RACE"
moda_perp_race = arrestos_DFS.groupBy("PERP_RACE").count().orderBy("count", ascending=False).first()[0]

# Reemplazar los valores nulos en "PERP_RACE" con la moda
arrestos_DFS = arrestos_DFS.withColumn("PERP_RACE", when(arrestos_DFS["PERP_RACE"].isNull(), moda_perp_race).otherwise(arrestos_DFS["PERP_RACE"]))

# COMMAND ----------

arrestos_DFS.count()

# COMMAND ----------

# Eliminar filas con valores faltantes en las columnas especificadas
arrestos_DFS = arrestos_DFS.dropna(subset=["PD_DESC", "KY_CD", "OFNS_DESC", "LAW_CODE"])

# COMMAND ----------

arrestos_DFS.count()

# COMMAND ----------

arrestos_DFS.select(
    [count(when(isnan(c) | col(c).isNull() | (col(c) == "(No value)") | (col(c) == "UNKNOWN")| (col(c) == "(null)"), c)).alias(c) for c in arrestos_DFS.columns]
).toPandas()


# COMMAND ----------

pobresa_DFS = pobresa_DFS.replace("(No value)",None)
pobresa_DFS = pobresa_DFS.replace("UNKNOWN",None)
pobresa_DFS = pobresa_DFS.replace("(null)",None)

# COMMAND ----------

pobresa_DFS.select(
    [count(when(isnan(c) | col(c).isNull() | (col(c) == "(No value)") | (col(c) == "UNKNOWN")| (col(c) == "(null)"), c)).alias(c) for c in pobresa_DFS.columns]
).toPandas()

# COMMAND ----------


# Calcular el total de filas en el DataFrame
total_filas = pobresa_DFS.count()

# Calcular el número de valores nulos en cada columna
null_counts_df = pobresa_DFS.select(
    [count(when(isnan(c) | col(c).isNull() | (col(c) == "(No value)") | (col(c) == "UNKNOWN") | (col(c) == "(null)"), c)).alias(c) for c in pobresa_DFS.columns]
)

# Convertir el DataFrame de PySpark a un DataFrame de pandas
null_counts_pandas = null_counts_df.toPandas()

# Calcular el porcentaje de valores nulos en cada columna
null_percentage_df = (null_counts_pandas / total_filas) * 100
# Configurar pandas para mostrar todas las filas y columnas
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
# Mostrar el DataFrame con los porcentajes de valores nulos
print(null_percentage_df)

# COMMAND ----------

# Lista de columnas a eliminar
columnas_a_eliminar = ["JWTR", "ENG", "WKW"]

# Eliminar las columnas
pobresa_DFS = pobresa_DFS.drop(*columnas_a_eliminar)

# COMMAND ----------


# Calcular la moda de la columna "PERP_RACE"
moda_schl = pobresa_DFS.groupBy("SCHL").count().orderBy("count", ascending=False).first()[0]

# Reemplazar los valores nulos en "PERP_RACE" con la moda
pobresa_DFS = pobresa_DFS.withColumn("SCHL", when(pobresa_DFS["SCHL"].isNull(), moda_schl).otherwise(pobresa_DFS["SCHL"]))

# COMMAND ----------

pobresa_DFS.select(
    [count(when(isnan(c) | col(c).isNull() | (col(c) == "(No value)") | (col(c) == "UNKNOWN")| (col(c) == "(null)"), c)).alias(c) for c in pobresa_DFS.columns]
).toPandas()

# COMMAND ----------



# COMMAND ----------

# Calcular los ingresos promedio por nivel educativo
avg_income_by_education = pobresa_DFS.groupBy("EducAttain").agg(avg("PreTaxIncome_PU").alias("AvgIncomeByEducation"))

# Mostrar el resultado
avg_income_by_education.show()

# COMMAND ----------

from pyspark.sql.functions import col

# Calcula la moda de la columna "EducAttain"
mode_value = pobresa_DFS.select("EducAttain").groupBy("EducAttain").count().orderBy(col("count").desc()).first()[0]

# Imputa los valores faltantes en "EducAttain" con la moda
pobresa_DFS = pobresa_DFS.fillna(mode_value, subset=["EducAttain"])


# COMMAND ----------

from pyspark.sql.functions import col

# Calcula la moda de la columna "LANX" excluyendo los valores nulos
mode_value = pobresa_DFS.filter(col("LANX").isNotNull()) \
                        .groupBy("LANX").count() \
                        .orderBy(col("count").desc()) \
                        .first()[0]

# Imputa los valores nulos en "LANX" con la moda
pobresa_DFS = pobresa_DFS.fillna(mode_value, subset=["LANX"])


# COMMAND ----------

from pyspark.sql.functions import col

# Calcula la moda de la columna "MSP" excluyendo los valores nulos
mode_value_msp = pobresa_DFS.filter(col("MSP").isNotNull()) \
                             .groupBy("MSP").count() \
                             .orderBy(col("count").desc()) \
                             .first()[0]

# Imputa los valores nulos en "MSP" con la moda
pobresa_DFS = pobresa_DFS.fillna(mode_value_msp, subset=["MSP"])


# COMMAND ----------

pobresa_DFS.select(
    [count(when(isnan(c) | col(c).isNull() | (col(c) == "(No value)") | (col(c) == "UNKNOWN")| (col(c) == "(null)"), c)).alias(c) for c in pobresa_DFS.columns]
).toPandas()

# COMMAND ----------

# Lista de columnas a eliminar
columns_to_drop = [
    'EST_PovGap', 'EST_PovGapIndex', 'NYCgov_REL', 'NYCgov_Threshold', 'Off_Threshold', 
    'Povunit_ID', 'Povunit_Rel', 'AgeCateg', 'DS', 'ESR', 'HHT', 'INTP_adj', 'JWTR', 
    'MRGP_adj', 'NP', 'REL', 'RELP', 'RELSHIPP', 'RETP_adj', 'RNTP_adj', 'SERIALNO', 
    'WGTP', 'WKHP', 'WKW', 'JWTRNS', 'WKWN'
]

# Eliminar las columnas especificadas
pobresa_DFS = pobresa_DFS.drop(*columns_to_drop)

# Mostrar las primeras filas del DataFrame resultante
pobresa_DFS.show(5)

# COMMAND ----------


# Filtrar por edad mayor de 18 años
pobresa_DFS = pobresa_DFS.filter(col("AGEP") >= 18)

# Agrupar por la columna de edad (AGEP) y contar el número de personas en cada grupo
age_group_counts = pobresa_DFS.groupBy("AGEP").count().orderBy("AGEP")

# Mostrar el resultado
age_group_counts.show()


# COMMAND ----------

# Suponiendo que tienes un SparkSession llamado spark y que ya tienes un DataFrame imputed_spark_df cargado

# Filtrar por ciudadanos (CitizenStatus = 1 o 2)
pobresa_DFS = pobresa_DFS.filter(col("CitizenStatus").isin([1, 2]))

citizens_group_counts = pobresa_DFS.groupBy("CitizenStatus").count()

# Mostrar el resultado
citizens_group_counts.show()


# COMMAND ----------

from pyspark.ml.feature import StringIndexer

# Creamos el StringIndexer
indexer = StringIndexer(inputCol="MAR", outputCol="MaritalStatusIndex")

# Ajustamos y transformamos los datos
pobresa_DFS_indexed = indexer.fit(pobresa_DFS).transform(pobresa_DFS)

# Mostramos el resultado
pobresa_DFS_indexed.select("MAR", "MaritalStatusIndex").show(20)


# COMMAND ----------

pobresa_DFS_indexed.printSchema()

# COMMAND ----------

from pyspark.sql.types import DoubleType
from pyspark.ml.feature import VectorAssembler, MinMaxScaler

# Columna que deseas normalizar
column_to_normalize = "PreTaxIncome_PU"

# Convertir la columna a tipo Double
pobresa_DFS_indexed = pobresa_DFS_indexed.withColumn(column_to_normalize, pobresa_DFS_indexed[column_to_normalize].cast(DoubleType()))

# Ensamblar las características en un vector denso
assembler = VectorAssembler(inputCols=[column_to_normalize], outputCol=column_to_normalize + "_vector")
pobresa_DFS_assembled = assembler.transform(pobresa_DFS_indexed)

# Crear el MinMaxScaler
scaler = MinMaxScaler(inputCol=column_to_normalize + "_vector", outputCol=column_to_normalize + "_scaled")

# Ajustar el scaler al conjunto de datos y transformar los datos
scaler_model = scaler.fit(pobresa_DFS_assembled)
pobresa_DFS_scaled = scaler_model.transform(pobresa_DFS_assembled)

# Mostrar el DataFrame con la columna escalada
pobresa_DFS_scaled.select(column_to_normalize, column_to_normalize + "_scaled").show()


# COMMAND ----------


# Calcular los ingresos promedio por nivel educativo
avg_income_by_education = pobresa_DFS_scaled.groupBy("EducAttain").agg(avg("PreTaxIncome_PU").alias("AvgIncomeByEducation"))

# Unir los ingresos promedio con el DataFrame original
pobresa_DFS_scaled = pobresa_DFS_scaled.join(avg_income_by_education, "EducAttain", "left")

# Calcular los ingresos normalizados
pobresa_DFS = pobresa_DFS_scaled.withColumn("NormalizedIncomeByEducation", col("PreTaxIncome_PU") / col("AvgIncomeByEducation"))

# Mostrar el resultado
pobresa_DFS.select("EducAttain", "PreTaxIncome_PU", "AvgIncomeByEducation", "NormalizedIncomeByEducation").show()

# COMMAND ----------

pobresa_DFS.printSchema()

# COMMAND ----------

from pyspark.sql.types import IntegerType, DoubleType

# Cambiar el tipo de datos de las columnas
pobresa_DFS = pobresa_DFS.withColumn("PWGTP", pobresa_DFS["PWGTP"].cast(IntegerType()))
pobresa_DFS = pobresa_DFS.withColumn("AGEP", pobresa_DFS["AGEP"].cast(IntegerType()))
pobresa_DFS = pobresa_DFS.withColumn("EST_Childcare", pobresa_DFS["EST_Childcare"].cast(DoubleType()))
pobresa_DFS = pobresa_DFS.withColumn("EST_Commuting", pobresa_DFS["EST_Commuting"].cast(DoubleType()))
pobresa_DFS = pobresa_DFS.withColumn("EST_FICAtax", pobresa_DFS["EST_FICAtax"].cast(DoubleType()))
pobresa_DFS = pobresa_DFS.withColumn("EST_HEAP", pobresa_DFS["EST_HEAP"].cast(DoubleType()))
pobresa_DFS = pobresa_DFS.withColumn("EST_Housing", pobresa_DFS["EST_Housing"].cast(DoubleType()))
pobresa_DFS = pobresa_DFS.withColumn("EST_IncomeTax", pobresa_DFS["EST_IncomeTax"].cast(DoubleType()))
pobresa_DFS = pobresa_DFS.withColumn("EST_MOOP", pobresa_DFS["EST_MOOP"].cast(DoubleType()))
pobresa_DFS = pobresa_DFS.withColumn("EST_Nutrition", pobresa_DFS["EST_Nutrition"].cast(DoubleType()))
pobresa_DFS = pobresa_DFS.withColumn("NYCgov_Income", pobresa_DFS["NYCgov_Income"].cast(DoubleType()))
pobresa_DFS = pobresa_DFS.withColumn("TotalWorkHrs_PU", pobresa_DFS["TotalWorkHrs_PU"].cast(IntegerType()))
pobresa_DFS = pobresa_DFS.withColumn("SPORDER", pobresa_DFS["TotalWorkHrs_PU"].cast(IntegerType()))

# Verificar los cambios
pobresa_DFS.printSchema()


# COMMAND ----------

from pyspark.ml.stat import Correlation
from pyspark.ml.feature import VectorAssembler\

# Seleccionar las columnas numéricas para calcular la correlación
numeric_cols = [col for col, dtype in pobresa_DFS.dtypes if dtype in ("int", "double")]

# Crear un VectorAssembler para ensamblar las columnas en un solo vector
assembler = VectorAssembler(inputCols=numeric_cols, outputCol="features")
assembled_data = assembler.transform(pobresa_DFS).select("features")

# Calcular la matriz de correlación
correlation_matrix = Correlation.corr(assembled_data, "features").head()

# Extraer la matriz de correlación
corr_matrix = correlation_matrix[0].toArray()

# Convertir la matriz de correlación a un DataFrame de PySpark para su visualización
corr_matrix_df = spark.createDataFrame(corr_matrix.tolist(), numeric_cols)

# Mostrar la matriz de correlación
corr_matrix_df.show(truncate=False)


# COMMAND ----------

import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

# Convertir la matriz de correlación a un DataFrame de pandas
corr_matrix_pd = pd.DataFrame(corr_matrix, columns=numeric_cols, index=numeric_cols)

# Crear un mapa de calor de la matriz de correlación utilizando Seaborn
plt.figure(figsize=(10, 8))
sns.heatmap(corr_matrix_pd, annot=True, cmap='coolwarm', fmt=".2f", linewidths=.5)
plt.title('Matriz de Correlación')
plt.show()


# COMMAND ----------

# Columnas que deseas eliminar
columns_to_drop = ["EST_FICATEX", "EST_IncomeTax", "NYCgov_Income", "PreTaxIncome_PU","EST_FICAtax"]

# Eliminar las columnas del DataFrame
pobresa_DFS = pobresa_DFS.drop(*columns_to_drop)

# COMMAND ----------

from pyspark.ml.feature import StringIndexer
from pyspark.ml.stat import Correlation
from pyspark.ml.feature import VectorAssembler
import pandas as pd

# Seleccionar las columnas categóricas
columns = ["Ethnicity", "TEN", "SEX", "SCH", "MSP", "LANX",  "CIT", "DIS"]

# Convertir las columnas categóricas en índices numéricos utilizando StringIndexer
indexers = [StringIndexer(inputCol=column, outputCol=column+"_index").fit(pobresa_DFS) for column in columns]

# Aplicar los indexers al DataFrame
indexed_df = pobresa_DFS
for indexer in indexers:
    indexed_df = indexer.transform(indexed_df)

# Seleccionar las columnas indexadas
indexed_columns = [column + "_index" for column in columns]

# Crear un ensamblador de vectores para las columnas indexadas
assembler = VectorAssembler(inputCols=indexed_columns, outputCol="features")

# Aplicar el ensamblador al DataFrame
assembled_df = assembler.transform(indexed_df).select("features")

# Calcular la matriz de correlación utilizando el coeficiente de Cramér V
correlation_matrix = Correlation.corr(assembled_df, "features", "pearson").collect()[0][0]

# Convertir la matriz de correlación a un DataFrame de pandas para visualización
correlation_matrix_pd = pd.DataFrame(correlation_matrix.toArray(), columns=columns, index=columns)

# Mostrar la matriz de correlación
print("Matriz de correlación para variables categóricas:")
print(correlation_matrix_pd)


# COMMAND ----------

# Crear el heatmap utilizando Seaborn
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix_pd, annot=True, cmap="coolwarm", fmt=".2f", linewidths=0.5)
plt.title("Matriz de correlación para variables categóricas")
plt.xlabel("Variables")
plt.ylabel("Variables")
plt.show()

# COMMAND ----------



# COMMAND ----------

pobresa_DFS.printSchema()


# COMMAND ----------

display(pobresa_DFS)

# COMMAND ----------

from pyspark.ml.feature import MinMaxScaler
from pyspark.ml.feature import VectorAssembler

# Seleccionar las columnas a normalizar
columns_to_normalize = ["PWGTP", "AGEP", "EST_Childcare", "EST_Commuting", "EST_HEAP", 
                        "EST_Housing", "EST_MOOP", "EST_Nutrition"]

# Ensamblar las columnas seleccionadas en un vector
assembler = VectorAssembler(inputCols=columns_to_normalize, outputCol="features")
assembled_df = assembler.transform(pobresa_DFS)

# Inicializar el escalador MinMaxScaler
scaler = MinMaxScaler(inputCol="features", outputCol="scaled_features")

# Ajustar el escalador a los datos
scaler_model = scaler.fit(assembled_df)

# Aplicar la transformación de escala a los datos
pandas_udf = scaler_model.transform(assembled_df)

# Seleccionar solo las columnas originales y la columna escalada
pandas_udf.select(columns_to_normalize + ["scaled_features"]).show()


# COMMAND ----------

display(pandas_udf)

# COMMAND ----------

pandas_udf.printSchema()

# COMMAND ----------

from pyspark.sql.functions import col
from pyspark.ml.feature import StringIndexer, VectorAssembler, MinMaxScaler
from pyspark.ml import Pipeline
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
import pandas as pd
import matplotlib.pyplot as plt

# Seleccionar columnas relevantes
columns = ["Ethnicity", "AGEP", "SEX", "NYCgov_Pov_Stat", "Boro", "CitizenStatus", "EducAttain", "FamType_PU", "FTPTWork", "TotalWorkHrs_PU", "CIT", "DIS", "SCH", "TEN"]

# Filtrar DataFrame para mantener solo las columnas seleccionadas
df = pandas_udf.select(columns)

# Indexar columnas categóricas
indexers = [StringIndexer(inputCol=column, outputCol=column+"_index").fit(df) for column in ["Ethnicity", "SEX", "NYCgov_Pov_Stat", "Boro", "CitizenStatus", "EducAttain", "FamType_PU", "FTPTWork", "CIT", "DIS", "SCH", "TEN"]]

# Ensamblador de características para todas las características numéricas e indexadas
assembler = VectorAssembler(
    inputCols=["Ethnicity_index", "AGEP", "SEX_index", "Boro_index", "CitizenStatus_index", "EducAttain_index", "FamType_PU_index", "FTPTWork_index", "TotalWorkHrs_PU", "CIT_index", "DIS_index", "SCH_index", "TEN_index"],
    outputCol="features"
)

# Escalador de características
scaler = MinMaxScaler(inputCol="features", outputCol="scaled_features")

# Crear el Pipeline de preprocesamiento
pipeline = Pipeline(stages=indexers + [assembler, scaler])

# Ajustar el Pipeline al DataFrame
model = pipeline.fit(df)
indexed_df = model.transform(df)

# Seleccionar las columnas de características escaladas y la etiqueta
final_df = indexed_df.select("scaled_features", col("NYCgov_Pov_Stat_index").alias("label"))

# Dividir los datos en conjuntos de entrenamiento y prueba
train_df, test_df = final_df.randomSplit([0.8, 0.2], seed=42)

# Definir el modelo Random Forest
rf = RandomForestClassifier(featuresCol="scaled_features", labelCol="label", numTrees=100, maxDepth=10)

# Entrenar el modelo
rf_model = rf.fit(train_df)

# Realizar predicciones en el conjunto de prueba
predictions = rf_model.transform(test_df)

# Evaluar el modelo
evaluator = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction", metricName="accuracy")
accuracy = evaluator.evaluate(predictions)
print(f"Accuracy del modelo Random Forest: {accuracy:.4f}")

# Extraer las importancias de características del modelo
importances = rf_model.featureImportances.toArray()
feature_names = ["Ethnicity_index", "AGEP", "SEX_index", "Boro_index", "CitizenStatus_index", "EducAttain_index", "FamType_PU_index", "FTPTWork_index", "TotalWorkHrs_PU", "CIT_index", "DIS_index", "SCH_index", "TEN_index"]

# Crear un DataFrame para las importancias
importance_df = pd.DataFrame(list(zip(feature_names, importances)), columns=["Feature", "Importance"])

# Ordenar por importancia
importance_df = importance_df.sort_values(by="Importance", ascending=False)

# Visualizar las importancias de características
plt.figure(figsize=(10, 6))
plt.barh(importance_df["Feature"], importance_df["Importance"])
plt.xlabel("Importance")
plt.ylabel("Feature")
plt.title("Feature Importances in Random Forest Model")
plt.show()


# COMMAND ----------

arrestos_DFS.printSchema()

# COMMAND ----------

display(arrestos_DFS)

# COMMAND ----------

from pyspark.sql import functions as F

# Convertir los tipos de columnas
arrestos_DFS = arrestos_DFS \
    .withColumn("X_COORD_CD", F.col("X_COORD_CD").cast("double")) \
    .withColumn("Y_COORD_CD", F.col("Y_COORD_CD").cast("double")) \
    .withColumn("Latitude", F.col("Latitude").cast("double")) \
    .withColumn("Longitude", F.col("Longitude").cast("double")) \
    .withColumn("KY_CD", F.col("KY_CD").cast("int")) \
    .withColumn("JURISDICTION_CODE", F.col("JURISDICTION_CODE").cast("int"))

# Mostrar el esquema actualizado para verificar los cambios
arrestos_DFS.printSchema()

# Mostrar algunas filas del DataFrame para verificar los cambios
arrestos_DFS.show(5)

# COMMAND ----------

# Realizar groupBy por grupo de edad y contar el número de arrestos en cada grupo
arrestos_por_edad = arrestos_DFS.groupBy("AGE_GROUP").count()

# Ordenar los resultados por el número de arrestos en orden descendente
arrestos_por_edad = arrestos_por_edad.orderBy(F.desc("count"))

# Mostrar los resultados
arrestos_por_edad.show()

# COMMAND ----------

# Filtrar las filas donde el grupo de edad no sea "<18"
arrestos_DFS = arrestos_DFS.filter(F.col("AGE_GROUP") != "<18")

# Mostrar algunas filas del DataFrame para verificar los cambios
arrestos_DFS.show(5)

# COMMAND ----------

# Realizar groupBy por grupo de edad y contar el número de arrestos en cada grupo
arrestos_por_edad = arrestos_DFS.groupBy("AGE_GROUP").count()

# Ordenar los resultados por el número de arrestos en orden descendente
arrestos_por_edad = arrestos_por_edad.orderBy(F.desc("count"))

# Mostrar los resultados
arrestos_por_edad.show()

# COMMAND ----------

display(arrestos_DFS)

# COMMAND ----------

from pyspark.sql.functions import isnan, when, count, col

# Contar los valores nulos o valores específicos en cada columna
null_counts = arrestos_DFS.select(
    [count(when(col(c).isNull() | (col(c) == "(No value)") | (col(c) == "UNKNOWN") | (col(c) == "(null)"), c)).alias(c) for c in arrestos_DFS.columns]
).toPandas()

# Mostrar el conteo de valores nulos o específicos en cada columna
print(null_counts)


# COMMAND ----------

from pyspark.sql.functions import substring

# Extraer el mes de la cadena de fecha (formato: dd/MM/yyyy)
arrestos_DFS = arrestos_DFS.withColumn(
    "MONTH_ARREST",
    substring("ARREST_DATE", 4, 2)  # Extraer los caracteres del mes
)

# Mostrar el DataFrame con el mes extraído de la cadena de fecha
arrestos_DFS.show(5)


# COMMAND ----------

# Convertir la columna MONTH_ARREST a tipo entero
arrestos_DFS = arrestos_DFS.withColumn("MONTH_ARREST", arrestos_DFS["MONTH_ARREST"].cast("integer"))

# Mostrar el DataFrame con la columna MONTH_ARREST convertida a entero
arrestos_DFS.show(5)


# COMMAND ----------

from pyspark.sql.functions import udf
from pyspark.sql.types import StringType
from datetime import datetime

# Definir una función UDF para obtener el día de la semana
def obtener_dia_semana(fecha_str):
    # Convertir la cadena de fecha a un objeto de fecha
    fecha_obj = datetime.strptime(fecha_str, "%m/%d/%Y")
    # Obtener el día de la semana como un número (lunes=0, martes=1, ..., domingo=6)
    dia_semana_num = fecha_obj.weekday()
    # Mapear el número del día de la semana a su nombre correspondiente
    dias_semana = ["Lunes", "Martes", "Miércoles", "Jueves", "Viernes", "Sábado", "Domingo"]
    return dias_semana[dia_semana_num]

# Registrar la función UDF
obtener_dia_semana_udf = udf(obtener_dia_semana, StringType())

# Crear una columna con el día de la semana
arrestos_DFS = arrestos_DFS.withColumn(
    "DIA_SEMANA",
    obtener_dia_semana_udf("ARREST_DATE")
)

# Mostrar el DataFrame con la nueva columna de día de la semana
arrestos_DFS.show(5)


# COMMAND ----------

from pyspark.ml.feature import StringIndexer

# Indexar la columna de días de la semana
indexer = StringIndexer(inputCol="DIA_SEMANA", outputCol="DIA_SEMANA_INDEX")
arrestos_DFS = indexer.fit(arrestos_DFS).transform(arrestos_DFS)

# Mostrar el DataFrame con la columna de días de la semana indexada
arrestos_DFS.show(5)


# COMMAND ----------

# Seleccionar las columnas numéricas relevantes
numeric_cols = ["KY_CD", "X_COORD_CD", "Y_COORD_CD", "Latitude", "Longitude", "DIA_SEMANA_INDEX"]

# Agregar la columna "MONTH_ARREST"
numeric_cols.append("MONTH_ARREST")

# Assembler para convertir las columnas numéricas en un solo vector
assembler = VectorAssembler(inputCols=numeric_cols, outputCol="features")
df_assembled = assembler.transform(arrestos_DFS).select("features")

# Calcular la matriz de correlación
corr_matrix = Correlation.corr(df_assembled, "features").head()
corr_matrix = corr_matrix[0].toArray()

# Crear una matriz de correlación como un DataFrame de Pandas
import pandas as pd
correlation_df = pd.DataFrame(corr_matrix, columns=numeric_cols, index=numeric_cols)

# Graficar la matriz de correlación
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_df, annot=True, cmap='coolwarm', fmt=".2f")
plt.title("Matriz de correlación")
plt.show()

# COMMAND ----------

arrestos_DFS.printSchema()

# COMMAND ----------

from pyspark.ml.feature import StringIndexer, VectorAssembler
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml import Pipeline
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
import matplotlib.pyplot as plt

# Convertir características categóricas en índices numéricos usando StringIndexer
indexers = [StringIndexer(inputCol=col, outputCol=col+"_index") for col in ["PERP_RACE", "MONTH_ARREST", "DIA_SEMANA_INDEX", "AGE_GROUP", "ARREST_BORO", "PERP_SEX"]]

# Convertir la columna LAW_CAT_CD en índices numéricos
indexer = StringIndexer(inputCol="LAW_CAT_CD", outputCol="label")

# Seleccionar las características y la variable objetivo
features = ["PERP_RACE_index", "MONTH_ARREST_index", "DIA_SEMANA_INDEX_index", "AGE_GROUP_index", "ARREST_BORO_index", "PERP_SEX_index"]
target = "label"

# Crear un VectorAssembler para combinar las características en una columna de vectores
assembler = VectorAssembler(inputCols=features, outputCol="features")

# Inicializar el modelo de Random Forest
rf = RandomForestClassifier(featuresCol="features", labelCol=target, numTrees=10)

# Crear un pipeline para encadenar las transformaciones y el modelo
pipeline = Pipeline(stages=indexers + [indexer, assembler, rf])

# Dividir los datos en conjunto de entrenamiento y prueba
(train_data, test_data) = arrestos_DFS.randomSplit([0.8, 0.2], seed=42)

# Entrenar el modelo
model = pipeline.fit(train_data)

# Realizar predicciones en el conjunto de prueba
predictions = model.transform(test_data)

# Inicializar el evaluador
evaluator = MulticlassClassificationEvaluator(labelCol=target, predictionCol="prediction", metricName="accuracy")

# Calcular la precisión del modelo
accuracy = evaluator.evaluate(predictions)
print(f"Accuracy: {accuracy:.4f}")

# Calcular otras métricas de evaluación
evaluator.setMetricName("weightedPrecision")
weighted_precision = evaluator.evaluate(predictions)
print(f"Weighted Precision: {weighted_precision:.4f}")

evaluator.setMetricName("weightedRecall")
weighted_recall = evaluator.evaluate(predictions)
print(f"Weighted Recall: {weighted_recall:.4f}")

evaluator.setMetricName("f1")
f1_score = evaluator.evaluate(predictions)
print(f"F1 Score: {f1_score:.4f}")

# Obtener la importancia de características del modelo
importances = model.stages[-1].featureImportances

# Obtener el nombre de las características
feature_names = features

# Crear un gráfico de barras para visualizar la importancia de las características
plt.figure(figsize=(10, 6))
plt.barh(feature_names, importances, color='skyblue')
plt.xlabel('Importancia')
plt.ylabel('Características')
plt.title('Importancia de las características en la clasificación de la gravedad de los delitos')
plt.gca().invert_yaxis()  # Invertir el eje y para que las características más importantes estén arriba
plt.show()


# COMMAND ----------

from pyspark.ml.feature import StringIndexer, VectorAssembler
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml import Pipeline
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
import matplotlib.pyplot as plt

# Convertir características categóricas en índices numéricos usando StringIndexer
indexers = [StringIndexer(inputCol=col, outputCol=col+"_index") for col in ["PERP_RACE", "MONTH_ARREST", "DIA_SEMANA_INDEX", "AGE_GROUP", "ARREST_BORO", "PERP_SEX"]]

# Convertir la columna LAW_CAT_CD en índices numéricos
indexer = StringIndexer(inputCol="LAW_CAT_CD", outputCol="label")

# Seleccionar las características y la variable objetivo
features = ["PERP_RACE_index", "MONTH_ARREST_index", "DIA_SEMANA_INDEX_index", "AGE_GROUP_index", "ARREST_BORO_index", "PERP_SEX_index"]
target = "label"

# Crear un VectorAssembler para combinar las características en una columna de vectores
assembler = VectorAssembler(inputCols=features, outputCol="features")

# Inicializar el modelo de Random Forest
rf = RandomForestClassifier(featuresCol="features", labelCol=target, numTrees=50)

# Crear un pipeline para encadenar las transformaciones y el modelo
pipeline = Pipeline(stages=indexers + [indexer, assembler, rf])

# Dividir los datos en conjunto de entrenamiento y prueba
(train_data, test_data) = arrestos_DFS.randomSplit([0.8, 0.2], seed=42)

# Entrenar el modelo
model = pipeline.fit(train_data)

# Realizar predicciones en el conjunto de prueba
predictions = model.transform(test_data)

# Inicializar el evaluador
evaluator = MulticlassClassificationEvaluator(labelCol=target, predictionCol="prediction", metricName="accuracy")

# Calcular la precisión del modelo
accuracy = evaluator.evaluate(predictions)
print(f"Accuracy: {accuracy:.4f}")

# Calcular otras métricas de evaluación
evaluator.setMetricName("weightedPrecision")
weighted_precision = evaluator.evaluate(predictions)
print(f"Weighted Precision: {weighted_precision:.4f}")

evaluator.setMetricName("weightedRecall")
weighted_recall = evaluator.evaluate(predictions)
print(f"Weighted Recall: {weighted_recall:.4f}")

evaluator.setMetricName("f1")
f1_score = evaluator.evaluate(predictions)
print(f"F1 Score: {f1_score:.4f}")

# Obtener la importancia de características del modelo
importances = model.stages[-1].featureImportances

# Obtener el nombre de las características
feature_names = features

# Crear un gráfico de barras para visualizar la importancia de las características
plt.figure(figsize=(10, 6))
plt.barh(feature_names, importances, color='skyblue')
plt.xlabel('Importancia')
plt.ylabel('Características')
plt.title('Importancia de las características en la clasificación de la gravedad de los delitos')
plt.gca().invert_yaxis()  # Invertir el eje y para que las características más importantes estén arriba
plt.show()


# COMMAND ----------

from pyspark.ml.feature import StringIndexer, VectorAssembler
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml import Pipeline
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
import matplotlib.pyplot as plt

# Convertir características categóricas en índices numéricos usando StringIndexer
indexers = [StringIndexer(inputCol=col, outputCol=col+"_index") for col in ["PERP_RACE", "MONTH_ARREST", "DIA_SEMANA_INDEX", "AGE_GROUP", "ARREST_BORO", "PERP_SEX"]]

# Convertir la columna LAW_CAT_CD en índices numéricos
indexer = StringIndexer(inputCol="LAW_CAT_CD", outputCol="label")

# Seleccionar las características y la variable objetivo
features = ["PERP_RACE_index", "MONTH_ARREST_index", "DIA_SEMANA_INDEX_index", "AGE_GROUP_index", "ARREST_BORO_index", "PERP_SEX_index"]
target = "label"

# Crear un VectorAssembler para combinar las características en una columna de vectores
assembler = VectorAssembler(inputCols=features, outputCol="features")

# Inicializar el modelo de Random Forest
rf = RandomForestClassifier(featuresCol="features", labelCol=target, numTrees=20,maxDepth=20)

# Crear un pipeline para encadenar las transformaciones y el modelo
pipeline = Pipeline(stages=indexers + [indexer, assembler, rf])

# Dividir los datos en conjunto de entrenamiento y prueba
(train_data, test_data) = arrestos_DFS.randomSplit([0.8, 0.2], seed=42)

# Entrenar el modelo
model = pipeline.fit(train_data)

# Realizar predicciones en el conjunto de prueba
predictions = model.transform(test_data)

# Inicializar el evaluador
evaluator = MulticlassClassificationEvaluator(labelCol=target, predictionCol="prediction", metricName="accuracy")

# Calcular la precisión del modelo
accuracy = evaluator.evaluate(predictions)
print(f"Accuracy: {accuracy:.4f}")

# Calcular otras métricas de evaluación
evaluator.setMetricName("weightedPrecision")
weighted_precision = evaluator.evaluate(predictions)
print(f"Weighted Precision: {weighted_precision:.4f}")

evaluator.setMetricName("weightedRecall")
weighted_recall = evaluator.evaluate(predictions)
print(f"Weighted Recall: {weighted_recall:.4f}")

evaluator.setMetricName("f1")
f1_score = evaluator.evaluate(predictions)
print(f"F1 Score: {f1_score:.4f}")

# Obtener la importancia de características del modelo
importances = model.stages[-1].featureImportances

# Obtener el nombre de las características
feature_names = features

# Crear un gráfico de barras para visualizar la importancia de las características
plt.figure(figsize=(10, 6))
plt.barh(feature_names, importances, color='skyblue')
plt.xlabel('Importancia')
plt.ylabel('Características')
plt.title('Importancia de las características en la clasificación de la gravedad de los delitos')
plt.gca().invert_yaxis()  # Invertir el eje y para que las características más importantes estén arriba
plt.show()


# COMMAND ----------


