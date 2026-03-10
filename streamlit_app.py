import warnings
warnings.filterwarnings("ignore")

from datetime import timedelta
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


# ============================================================
# CONFIG
# ============================================================
st.set_page_config(page_title="Autoplastek Predictivo", layout="wide")

RANDOM_STATE = 42
N_LAGS = 4
TEST_SIZE = 0.20


# ============================================================
# HELPERS
# ============================================================
def normalize_col_name(col: str) -> str:
    col = str(col).strip().replace("\n", " ")
    while "  " in col:
        col = col.replace("  ", " ")
    return col.upper()


def clean_numeric(series: pd.Series) -> pd.Series:
    if series.dtype == object:
        s = (
            series.astype(str)
            .str.replace("%", "", regex=False)
            .str.replace(",", ".", regex=False)
            .str.strip()
        )
        s = s.replace({"": np.nan, "NAN": np.nan, "NONE": np.nan})
        return pd.to_numeric(s, errors="coerce")
    return pd.to_numeric(series, errors="coerce")


def parse_time_to_minutes(series: pd.Series) -> pd.Series:
    s = series.astype(str).str.strip()
    p1 = pd.to_datetime(s, format="%H:%M:%S", errors="coerce")
    p2 = pd.to_datetime(s, format="%H:%M", errors="coerce")
    final = p1.fillna(p2)
    return final.dt.hour * 60 + final.dt.minute


def clasificar_riesgo(valor):
    if pd.isna(valor):
        return "SIN DATO"
    if valor >= 20:
        return "CRITICO"
    if valor >= 10:
        return "ALTO"
    if valor >= 5:
        return "MEDIO"
    return "BAJO"


def riesgo_emoji(valor):
    if pd.isna(valor):
        return "⚪"
    if valor >= 20:
        return "🔴"
    if valor >= 10:
        return "🟠"
    if valor >= 5:
        return "🟡"
    return "🟢"


def recomendacion_mantenimiento(valor):
    if pd.isna(valor):
        return "Sin información suficiente"
    if valor >= 20:
        return "Mantenimiento inmediato"
    if valor >= 10:
        return "Programar mantenimiento preventivo"
    if valor >= 5:
        return "Monitoreo cercano"
    return "Operación normal con seguimiento"


# ============================================================
# CARGA DE EXCEL
# ============================================================
def load_excel_sheets(excel_source):
    all_sheets = pd.read_excel(excel_source, sheet_name=None)
    sheet_map = {str(k).strip().upper(): k for k in all_sheets.keys()}

    captura_key = None
    total_key = None

    for k_upper, original in sheet_map.items():
        if "CAPTURA" in k_upper and "TRS" in k_upper:
            captura_key = original
        if "TOTAL" == k_upper or "TOTAL" in k_upper:
            total_key = original

    if captura_key is None:
        raise ValueError("No encontré la hoja 'Captura TRS'.")
    if total_key is None:
        raise ValueError("No encontré la hoja 'TOTAL'.")

    return all_sheets[captura_key].copy(), all_sheets[total_key].copy(), captura_key, total_key


def prepare_captura_trs(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [normalize_col_name(c) for c in df.columns]

    rename_map = {
        "NO.": "NO",
        "FECHA": "FECHA",
        "SEM": "SEM",
        "MAQ": "MAQ",
        "TURNO": "TURNO",
        "GAP": "GAP",
        "HR INICIO": "HR_INICIO",
        "HR FIN": "HR_FIN",
        "PIEZAS OK": "PIEZAS_OK",
        "PIEZAS NOK": "PIEZAS_NOK",
        "TOTAL PIEZAS": "TOTAL_PIEZAS",
        "PZS OBJ": "PZS_OBJ",
        "% SCRAP": "SCRAP_PCT",
        "T.U.": "TU",
        "TRS%": "TRS_CAPTURA_PCT",
        "TRS %": "TRS_CAPTURA_PCT",
        "NO CAL %": "NO_CAL_PCT",
        "D2 MIN": "D2_MIN",
        "D2%": "D2_CAPTURA_PCT",
        "D2 %": "D2_CAPTURA_PCT",
    }

    df = df.rename(columns={c: rename_map[c] for c in df.columns if c in rename_map})

    required = ["FECHA", "SEM", "MAQ", "TURNO", "GAP"]
    for c in required:
        if c not in df.columns:
            raise ValueError(f"En 'Captura TRS' falta la columna {c}")

    df["FECHA"] = pd.to_datetime(df["FECHA"], dayfirst=True, errors="coerce")

    numeric_cols = [
        "SEM", "MAQ", "TURNO", "PIEZAS_OK", "PIEZAS_NOK", "TOTAL_PIEZAS", "PZS_OBJ",
        "SCRAP_PCT", "TU", "TRS_CAPTURA_PCT", "NO_CAL_PCT", "D2_MIN", "D2_CAPTURA_PCT"
    ]
    for c in numeric_cols:
        if c in df.columns:
            df[c] = clean_numeric(df[c])

    df["HR_INICIO_MIN"] = parse_time_to_minutes(df["HR_INICIO"]) if "HR_INICIO" in df.columns else np.nan
    df["HR_FIN_MIN"] = parse_time_to_minutes(df["HR_FIN"]) if "HR_FIN" in df.columns else np.nan

    df["DURACION_MIN"] = df["HR_FIN_MIN"] - df["HR_INICIO_MIN"]
    df.loc[df["DURACION_MIN"] < 0, "DURACION_MIN"] += 24 * 60

    if "TOTAL_PIEZAS" not in df.columns and {"PIEZAS_OK", "PIEZAS_NOK"}.issubset(df.columns):
        df["TOTAL_PIEZAS"] = df["PIEZAS_OK"].fillna(0) + df["PIEZAS_NOK"].fillna(0)

    if {"PIEZAS_NOK", "TOTAL_PIEZAS"}.issubset(df.columns):
        df["TASA_NOK_PCT"] = np.where(
            df["TOTAL_PIEZAS"] > 0,
            (df["PIEZAS_NOK"] / df["TOTAL_PIEZAS"]) * 100,
            np.nan
        )
    else:
        df["TASA_NOK_PCT"] = np.nan

    if {"PIEZAS_OK", "PZS_OBJ"}.issubset(df.columns):
        df["CUMPLIMIENTO_OBJ_PCT"] = np.where(
            df["PZS_OBJ"] > 0,
            (df["PIEZAS_OK"] / df["PZS_OBJ"]) * 100,
            np.nan
        )
    else:
        df["CUMPLIMIENTO_OBJ_PCT"] = np.nan

    if {"TOTAL_PIEZAS", "DURACION_MIN"}.issubset(df.columns):
        df["PIEZAS_X_HORA"] = np.where(
            df["DURACION_MIN"] > 0,
            df["TOTAL_PIEZAS"] / (df["DURACION_MIN"] / 60),
            np.nan
        )
    else:
        df["PIEZAS_X_HORA"] = np.nan

    excluded = {
        "NO", "FECHA", "SEM", "MAQ", "TURNO", "GAP", "HR_INICIO", "HR_FIN",
        "PIEZAS_OK", "PIEZAS_NOK", "TOTAL_PIEZAS", "PZS_OBJ", "SCRAP_PCT", "TU",
        "TRS_CAPTURA_PCT", "NO_CAL_PCT", "D2_MIN", "D2_CAPTURA_PCT",
        "HR_INICIO_MIN", "HR_FIN_MIN", "DURACION_MIN", "TASA_NOK_PCT",
        "CUMPLIMIENTO_OBJ_PCT", "PIEZAS_X_HORA"
    }
    desc_candidates = [c for c in df.columns if c not in excluded]
    if desc_candidates:
        df["DESCRIPCION_PROC"] = df[desc_candidates[0]].astype(str)

    return df


def prepare_total(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [normalize_col_name(c) for c in df.columns]

    rename_map = {
        "FECHA": "FECHA",
        "SEM": "SEM",
        "MAQ": "MAQ",
        "GAP": "GAP",
        "TURNO": "TURNO",
        "TRS %": "TRS_TOTAL_PCT",
        "PP %": "PP_PCT",
        "PO %": "PO_PCT",
        "AP %": "AP_PCT",
        "D2 %": "D2_TOTAL_PCT",
        "AMAQ %": "AMAQ_PCT",
        "AMOL %": "AMOL_PCT",
        "NOCAL %": "NOCAL_PCT",
        "CM %": "CM_PCT",
        "MP %": "MP_PCT",
        "NO TRS %": "NO_TRS_PCT",
        "SUMA": "SUMA_PCT",
        "OBJ": "OBJ_PCT"
    }

    df = df.rename(columns={c: rename_map[c] for c in df.columns if c in rename_map})

    required = ["FECHA", "SEM", "MAQ", "TURNO", "GAP"]
    for c in required:
        if c not in df.columns:
            raise ValueError(f"En 'TOTAL' falta la columna {c}")

    df["FECHA"] = pd.to_datetime(df["FECHA"], dayfirst=True, errors="coerce")

    numeric_cols = [
        "SEM", "MAQ", "TURNO", "TRS_TOTAL_PCT", "PP_PCT", "PO_PCT", "AP_PCT",
        "D2_TOTAL_PCT", "AMAQ_PCT", "AMOL_PCT", "NOCAL_PCT", "CM_PCT",
        "MP_PCT", "NO_TRS_PCT", "SUMA_PCT", "OBJ_PCT"
    ]
    for c in numeric_cols:
        if c in df.columns:
            df[c] = clean_numeric(df[c])

    return df


def merge_sheets(captura: pd.DataFrame, total: pd.DataFrame) -> pd.DataFrame:
    keys = ["FECHA", "SEM", "MAQ", "GAP", "TURNO"]

    df = pd.merge(captura, total, on=keys, how="outer", suffixes=("", "_TOT"))
    df = df.dropna(subset=["FECHA", "MAQ"]).copy()
    df = df.sort_values(["MAQ", "FECHA", "TURNO"]).reset_index(drop=True)

    df["ANIO"] = df["FECHA"].dt.year
    df["MES"] = df["FECHA"].dt.month
    df["DIA"] = df["FECHA"].dt.day
    df["DIA_SEMANA"] = df["FECHA"].dt.weekday
    df["SEMANA_ANIO"] = df["FECHA"].dt.isocalendar().week.astype(int)
    df["ES_FIN_SEMANA"] = df["DIA_SEMANA"].isin([5, 6]).astype(int)

    return df


def aggregate_daily_by_machine(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    group_cols = ["FECHA", "MAQ"]

    sum_cols = [
        "PIEZAS_OK", "PIEZAS_NOK", "TOTAL_PIEZAS", "PZS_OBJ", "D2_MIN", "DURACION_MIN"
    ]

    mean_cols = [
        "SEM", "TU",
        "TRS_CAPTURA_PCT", "NO_CAL_PCT", "D2_CAPTURA_PCT",
        "TRS_TOTAL_PCT", "PP_PCT", "PO_PCT", "AP_PCT", "D2_TOTAL_PCT",
        "AMAQ_PCT", "AMOL_PCT", "NOCAL_PCT", "CM_PCT", "MP_PCT", "NO_TRS_PCT",
        "HR_INICIO_MIN", "HR_FIN_MIN"
    ]

    first_cols = [
        "GAP", "DESCRIPCION_PROC"
    ]

    agg_dict = {}

    for c in sum_cols:
        if c in df.columns:
            agg_dict[c] = "sum"

    for c in mean_cols:
        if c in df.columns:
            agg_dict[c] = "mean"

    for c in first_cols:
        if c in df.columns:
            agg_dict[c] = "first"

    if "TURNO" in df.columns:
        agg_dict["TURNO"] = "nunique"

    daily = df.groupby(group_cols, as_index=False).agg(agg_dict)

    if "TURNO" in daily.columns:
        daily = daily.rename(columns={"TURNO": "TURNOS_TRABAJADOS"})

    if {"PIEZAS_NOK", "TOTAL_PIEZAS"}.issubset(daily.columns):
        daily["SCRAP_PCT"] = np.where(
            daily["TOTAL_PIEZAS"] > 0,
            (daily["PIEZAS_NOK"] / daily["TOTAL_PIEZAS"]) * 100,
            np.nan
        )
        daily["TASA_NOK_PCT"] = np.where(
            daily["TOTAL_PIEZAS"] > 0,
            (daily["PIEZAS_NOK"] / daily["TOTAL_PIEZAS"]) * 100,
            np.nan
        )

    if {"PIEZAS_OK", "PZS_OBJ"}.issubset(daily.columns):
        daily["CUMPLIMIENTO_OBJ_PCT"] = np.where(
            daily["PZS_OBJ"] > 0,
            (daily["PIEZAS_OK"] / daily["PZS_OBJ"]) * 100,
            np.nan
        )

    if {"TOTAL_PIEZAS", "DURACION_MIN"}.issubset(daily.columns):
        daily["PIEZAS_X_HORA"] = np.where(
            daily["DURACION_MIN"] > 0,
            daily["TOTAL_PIEZAS"] / (daily["DURACION_MIN"] / 60),
            np.nan
        )

    daily["ANIO"] = daily["FECHA"].dt.year
    daily["MES"] = daily["FECHA"].dt.month
    daily["DIA"] = daily["FECHA"].dt.day
    daily["DIA_SEMANA"] = daily["FECHA"].dt.weekday
    daily["SEMANA_ANIO"] = daily["FECHA"].dt.isocalendar().week.astype(int)
    daily["ES_FIN_SEMANA"] = daily["DIA_SEMANA"].isin([5, 6]).astype(int)

    return daily.sort_values(["MAQ", "FECHA"]).reset_index(drop=True)


# ============================================================
# TARGETS Y FEATURES
# ============================================================
def get_target_columns(df):
    ordered_targets = [
        "SCRAP_PCT",
        "TRS_CAPTURA_PCT",
        "NO_CAL_PCT",
        "D2_CAPTURA_PCT",
        "TRS_TOTAL_PCT",
        "PP_PCT",
        "PO_PCT",
        "AP_PCT",
        "D2_TOTAL_PCT",
        "AMAQ_PCT",
        "AMOL_PCT",
        "NOCAL_PCT",
        "CM_PCT",
        "MP_PCT",
        "NO_TRS_PCT",
    ]
    return [c for c in ordered_targets if c in df.columns]

def create_lag_features_multioutput(df: pd.DataFrame, target_cols: list[str], n_lags: int) -> pd.DataFrame:
    df = df.copy()

    feature_candidates = [
        "SEM", "PIEZAS_OK", "PIEZAS_NOK", "TOTAL_PIEZAS", "PZS_OBJ", "TU",
        "HR_INICIO_MIN", "HR_FIN_MIN", "DURACION_MIN",
        "TASA_NOK_PCT", "CUMPLIMIENTO_OBJ_PCT", "PIEZAS_X_HORA",
        "D2_MIN"
    ] + target_cols

    feature_cols = [c for c in feature_candidates if c in df.columns]

    grp = df.groupby("MAQ", group_keys=False)

    for col in feature_cols:
        for lag in range(1, n_lags + 1):
            df[f"{col}_LAG{lag}"] = grp[col].shift(lag)

        df[f"{col}_ROLL_MEAN_{n_lags}"] = grp[col].shift(1).rolling(n_lags).mean().reset_index(level=0, drop=True)
        df[f"{col}_ROLL_STD_{n_lags}"] = grp[col].shift(1).rolling(n_lags).std().reset_index(level=0, drop=True)

    for col in target_cols:
        df[f"TARGET_{col}"] = grp[col].shift(-1)

    return df


def build_pipeline(X: pd.DataFrame, y: pd.DataFrame):
    numeric_features = X.select_dtypes(include=[np.number]).columns.tolist()
    categorical_features = X.select_dtypes(exclude=[np.number]).columns.tolist()

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", SimpleImputer(strategy="median"), numeric_features),
            ("cat", Pipeline([
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("onehot", OneHotEncoder(handle_unknown="ignore"))
            ]), categorical_features),
        ]
    )

    model = RandomForestRegressor(
        n_estimators=500,
        max_depth=18,
        min_samples_leaf=2,
        min_samples_split=4,
        random_state=RANDOM_STATE,
        n_jobs=-1
    )

    pipeline = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("model", model)
    ])

    pipeline.fit(X, y)
    return pipeline


def evaluar_modelo_multioutput(y_true: pd.DataFrame, y_pred: np.ndarray, target_cols: list[str]):
    y_pred_df = pd.DataFrame(y_pred, columns=target_cols, index=y_true.index).clip(lower=0, upper=100)

    mae_global = mean_absolute_error(y_true, y_pred_df)
    rmse_global = np.sqrt(mean_squared_error(y_true, y_pred_df))
    r2_global = r2_score(y_true, y_pred_df, multioutput="uniform_average")

    detalle = []
    for col in target_cols:
        yt = y_true[col]
        yp = y_pred_df[col]
        detalle.append({
            "Variable": col,
            "MAE": round(mean_absolute_error(yt, yp), 3),
            "RMSE": round(np.sqrt(mean_squared_error(yt, yp)), 3),
            "R2": round(r2_score(yt, yp), 3)
        })

    detalle_df = pd.DataFrame(detalle).sort_values("R2", ascending=False)
    return {
        "MAE_GLOBAL": mae_global,
        "RMSE_GLOBAL": rmse_global,
        "R2_GLOBAL": r2_global,
        "DETALLE": detalle_df,
        "Y_PRED_DF": y_pred_df
    }


def entrenar_modelo(df):
    target_cols = get_target_columns(df)
    if not target_cols:
        raise ValueError("No encontré columnas porcentuales para predecir.")

    df_model = create_lag_features_multioutput(df, target_cols, N_LAGS)

    base_features = [
    "MAQ", "GAP", "TURNO", "ANIO", "MES", "DIA",
    "DIA_SEMANA", "SEMANA_ANIO", "ES_FIN_SEMANA"
    ]
    if "DESCRIPCION_PROC" in df_model.columns:
        base_features.append("DESCRIPCION_PROC")

    derived_features = [
        c for c in df_model.columns
        if any(tag in c for tag in ["_LAG", "_ROLL_MEAN_", "_ROLL_STD_"])
    ]
    usable_features = [c for c in base_features + derived_features if c in df_model.columns]

    target_future_cols = [f"TARGET_{c}" for c in target_cols]
    df_model = df_model.dropna(subset=target_future_cols, how="any").copy()
    df_model = df_model.sort_values(["FECHA", "MAQ"]).reset_index(drop=True)

    split_index = int(len(df_model) * (1 - TEST_SIZE))
    train_df = df_model.iloc[:split_index].copy()
    test_df = df_model.iloc[split_index:].copy()

    X_train = train_df[usable_features]
    X_test = test_df[usable_features]

    y_train = train_df[target_future_cols].copy()
    y_test = test_df[target_future_cols].copy()

    rename_back = {f"TARGET_{c}": c for c in target_cols}
    y_train = y_train.rename(columns=rename_back)
    y_test = y_test.rename(columns=rename_back)

    train_mask = ~y_train.isna().any(axis=1)
    test_mask = ~y_test.isna().any(axis=1)

    X_train = X_train.loc[train_mask].copy()
    y_train = y_train.loc[train_mask].copy()

    X_test = X_test.loc[test_mask].copy()
    y_test = y_test.loc[test_mask].copy()
    test_df = test_df.loc[test_mask].copy()

    if len(X_train) == 0 or len(y_train) == 0:
        raise ValueError("No hay suficientes datos limpios para entrenar el modelo.")

    if len(X_test) == 0 or len(y_test) == 0:
        raise ValueError("No hay suficientes datos limpios para evaluar el modelo.")

    pipeline = build_pipeline(X_train, y_train)
    y_pred = pipeline.predict(X_test)

    metricas = evaluar_modelo_multioutput(y_test, y_pred, target_cols)
    y_pred_df = metricas["Y_PRED_DF"]

    prioridad = "NO_TRS_PCT" if "NO_TRS_PCT" in target_cols else target_cols[0]
    test_eval = test_df[["MAQ", "FECHA"]].copy()
    test_eval["REAL_PRIORIDAD"] = y_test[prioridad].values
    test_eval["PRED_PRIORIDAD"] = y_pred_df[prioridad].values
    test_eval["ERROR_ABS"] = (test_eval["REAL_PRIORIDAD"] - test_eval["PRED_PRIORIDAD"]).abs()

    mae_por_maquina = (
        test_eval.groupby("MAQ")["ERROR_ABS"]
        .mean()
        .reset_index()
        .rename(columns={"ERROR_ABS": "ERROR_ESPERADO_MAE"})
        .sort_values("ERROR_ESPERADO_MAE", ascending=False)
    )

    mae_global = test_eval["ERROR_ABS"].mean()

    historico_pred = test_df[["FECHA", "MAQ", "TURNO"]].copy()
    for col in target_cols:
        historico_pred[f"REAL_{col}"] = y_test[col].values
        historico_pred[f"PRED_{col}"] = y_pred_df[col].values

    return pipeline, usable_features, target_cols, metricas, mae_por_maquina, mae_global, historico_pred


# ============================================================
# PREDICCION FUTURA
# ============================================================
def crear_fila_futura_para_maquina(df, maquina, fecha_objetivo, usable_features, target_cols):
    hist = df[df["MAQ"] == maquina].sort_values(["FECHA", "TURNO"]).copy()
    if hist.empty:
        return None

    ultima = hist.iloc[-1].copy()
    fila = {
    "MAQ": maquina,
    "GAP": ultima["GAP"] if "GAP" in hist.columns else "NA",
    "TURNO": ultima["TURNO"] if "TURNO" in hist.columns else 1,
    "ANIO": fecha_objetivo.year,
    "MES": fecha_objetivo.month,
    "DIA": fecha_objetivo.day,
    "DIA_SEMANA": fecha_objetivo.weekday(),
    "SEMANA_ANIO": int(fecha_objetivo.isocalendar().week),
    "ES_FIN_SEMANA": 1 if fecha_objetivo.weekday() in [5, 6] else 0
    }

    if "DESCRIPCION_PROC" in hist.columns:
        fila["DESCRIPCION_PROC"] = ultima["DESCRIPCION_PROC"]

    base_for_history = [
        "SEM", "PIEZAS_OK", "PIEZAS_NOK", "TOTAL_PIEZAS", "PZS_OBJ", "TU",
        "HR_INICIO_MIN", "HR_FIN_MIN", "DURACION_MIN",
        "TASA_NOK_PCT", "CUMPLIMIENTO_OBJ_PCT", "PIEZAS_X_HORA",
        "D2_MIN"
    ] + target_cols

    for col in base_for_history:
        if col in hist.columns:
            values = hist[col].dropna().tolist()
            recent = values[-N_LAGS:] if len(values) >= N_LAGS else values

            for lag in range(1, N_LAGS + 1):
                fila[f"{col}_LAG{lag}"] = recent[-lag] if len(recent) >= lag else np.nan

            if len(recent) > 0:
                fila[f"{col}_ROLL_MEAN_{N_LAGS}"] = float(np.mean(recent))
                fila[f"{col}_ROLL_STD_{N_LAGS}"] = float(np.std(recent, ddof=0))
            else:
                fila[f"{col}_ROLL_MEAN_{N_LAGS}"] = np.nan
                fila[f"{col}_ROLL_STD_{N_LAGS}"] = np.nan

    fila_df = pd.DataFrame([fila])

    for c in usable_features:
        if c not in fila_df.columns:
            fila_df[c] = np.nan

    return fila_df[usable_features]


def predecir_fecha(df, pipeline, usable_features, target_cols, mae_por_maquina, mae_global, fecha_objetivo):
    maquinas = sorted(df["MAQ"].dropna().unique().tolist())
    resultados = []

    for maquina in maquinas:
        x_fut = crear_fila_futura_para_maquina(df, maquina, fecha_objetivo, usable_features, target_cols)
        if x_fut is None:
            continue

        pred = np.clip(pipeline.predict(x_fut)[0], 0, 100)

        row = {
            "Fecha solicitada": fecha_objetivo.date(),
            "Máquina": maquina,
        }

        for col, val in zip(target_cols, pred):
            row[col] = round(float(val), 2)

        principal = row["NO_TRS_PCT"] if "NO_TRS_PCT" in row else row[target_cols[0]]

        fila_error = mae_por_maquina[mae_por_maquina["MAQ"] == maquina]
        error_esperado = float(fila_error["ERROR_ESPERADO_MAE"].iloc[0]) if not fila_error.empty else float(mae_global)

        row["Error esperado ±%"] = round(error_esperado, 2)
        row["Semáforo"] = riesgo_emoji(principal)
        row["Riesgo"] = clasificar_riesgo(principal)
        row["Recomendación"] = recomendacion_mantenimiento(principal)

        resultados.append(row)

    res = pd.DataFrame(resultados)
    orden = ["Fecha solicitada", "Máquina"] + target_cols + ["Error esperado ±%", "Semáforo", "Riesgo", "Recomendación"]
    orden = [c for c in orden if c in res.columns]
    res = res[orden]

    sort_col = "NO_TRS_PCT" if "NO_TRS_PCT" in res.columns else target_cols[0]
    return res.sort_values(sort_col, ascending=False).reset_index(drop=True)


def simular_produccion_por_piezas(
    df,
    pipeline,
    usable_features,
    target_cols,
    mae_por_maquina,
    mae_global,
    fecha_objetivo,
    piezas_planeadas,
    maquina=None
):
    if maquina is None or maquina == "Todas":
        maquinas = sorted(df["MAQ"].dropna().unique().tolist())
    else:
        maquinas = [maquina]

    resultados = []

    for maq in maquinas:
        x_fut = crear_fila_futura_para_maquina(
            df=df,
            maquina=maq,
            fecha_objetivo=fecha_objetivo,
            usable_features=usable_features,
            target_cols=target_cols
        )

        if x_fut is None:
            continue

        if "TOTAL_PIEZAS_LAG1" in x_fut.columns:
            x_fut.loc[:, "TOTAL_PIEZAS_LAG1"] = piezas_planeadas

        if "PZS_OBJ_LAG1" in x_fut.columns:
            x_fut.loc[:, "PZS_OBJ_LAG1"] = piezas_planeadas

        hist = df[df["MAQ"] == maq].sort_values("FECHA").copy()

        if not hist.empty:
            ultima = hist.iloc[-1]

            piezas_ok_est = piezas_planeadas
            piezas_nok_est = 0

            if "SCRAP_PCT" in hist.columns and pd.notna(ultima.get("SCRAP_PCT", np.nan)):
                scrap_hist = max(float(ultima["SCRAP_PCT"]), 0.0)
                piezas_nok_est = round(piezas_planeadas * scrap_hist / 100.0)
                piezas_ok_est = max(piezas_planeadas - piezas_nok_est, 0)

            if "PIEZAS_OK_LAG1" in x_fut.columns:
                x_fut.loc[:, "PIEZAS_OK_LAG1"] = piezas_ok_est

            if "PIEZAS_NOK_LAG1" in x_fut.columns:
                x_fut.loc[:, "PIEZAS_NOK_LAG1"] = piezas_nok_est

            if "TASA_NOK_PCT_LAG1" in x_fut.columns:
                x_fut.loc[:, "TASA_NOK_PCT_LAG1"] = (piezas_nok_est / piezas_planeadas * 100) if piezas_planeadas > 0 else 0

            if "CUMPLIMIENTO_OBJ_PCT_LAG1" in x_fut.columns:
                x_fut.loc[:, "CUMPLIMIENTO_OBJ_PCT_LAG1"] = (piezas_ok_est / piezas_planeadas * 100) if piezas_planeadas > 0 else 0

        pred = np.clip(pipeline.predict(x_fut)[0], 0, 100)

        row = {
            "Fecha solicitada": fecha_objetivo.date(),
            "Máquina": maq,
            "Piezas planeadas": int(piezas_planeadas),
        }

        for col, val in zip(target_cols, pred):
            row[col] = round(float(val), 2)

        error_cols_prioridad = [c for c in ["SCRAP_PCT", "NO_TRS_PCT", "NOCAL_PCT", "D2_TOTAL_PCT", "CM_PCT", "MP_PCT"] if c in target_cols]
        if error_cols_prioridad:
            dominante = max(error_cols_prioridad, key=lambda c: row.get(c, -1))
            row["Tipo de error dominante"] = dominante
            row["% error dominante"] = row[dominante]
        else:
            dominante = target_cols[0]
            row["Tipo de error dominante"] = dominante
            row["% error dominante"] = row[dominante]

        row["Piezas con error esperadas"] = round(piezas_planeadas * row["% error dominante"] / 100.0)

        fila_error = mae_por_maquina[mae_por_maquina["MAQ"] == maq]
        error_esperado = float(fila_error["ERROR_ESPERADO_MAE"].iloc[0]) if not fila_error.empty else float(mae_global)
        row["Error esperado ±%"] = round(error_esperado, 2)
        row["Semáforo"] = riesgo_emoji(row["% error dominante"])
        row["Riesgo"] = clasificar_riesgo(row["% error dominante"])
        row["Recomendación"] = recomendacion_mantenimiento(row["% error dominante"])

        resultados.append(row)

    res = pd.DataFrame(resultados)

    if res.empty:
        return res

    orden = (
        ["Fecha solicitada", "Máquina", "Piezas planeadas"]
        + target_cols
        + ["Tipo de error dominante", "% error dominante", "Piezas con error esperadas", "Error esperado ±%", "Semáforo", "Riesgo", "Recomendación"]
    )
    orden = [c for c in orden if c in res.columns]
    return res[orden].sort_values("% error dominante", ascending=False).reset_index(drop=True)


def predecir_semana(df, pipeline, usable_features, target_cols, mae_por_maquina, mae_global, anio, semana):
    primer_dia = pd.Timestamp.fromisocalendar(anio, semana, 1)
    dias = [primer_dia + timedelta(days=i) for i in range(7)]

    acumulado = []
    for fecha_objetivo in dias:
        pred_dia = predecir_fecha(
            df=df,
            pipeline=pipeline,
            usable_features=usable_features,
            target_cols=target_cols,
            mae_por_maquina=mae_por_maquina,
            mae_global=mae_global,
            fecha_objetivo=fecha_objetivo
        )
        if not pred_dia.empty:
            acumulado.append(pred_dia)

    if not acumulado:
        return pd.DataFrame(), pd.DataFrame()

    detalle = pd.concat(acumulado, ignore_index=True)

    agg = {c: "mean" for c in target_cols}
    agg["Error esperado ±%"] = "mean"

    resumen = detalle.groupby("Máquina", as_index=False).agg(agg)

    principal_col = "NO_TRS_PCT" if "NO_TRS_PCT" in resumen.columns else target_cols[0]
    resumen["Semáforo"] = resumen[principal_col].apply(riesgo_emoji)
    resumen["Riesgo"] = resumen[principal_col].apply(clasificar_riesgo)
    resumen["Recomendación"] = resumen[principal_col].apply(recomendacion_mantenimiento)

    for c in target_cols + ["Error esperado ±%"]:
        if c in resumen.columns:
            resumen[c] = resumen[c].round(2)

    return resumen.sort_values(principal_col, ascending=False), detalle


# ============================================================
# UI HELPERS
# ============================================================
def rename_targets_for_ui(df: pd.DataFrame) -> pd.DataFrame:
    rename_map = {
        "SCRAP_PCT": "% SCRAP",
        "TRS_CAPTURA_PCT": "TRS Captura %",
        "NO_CAL_PCT": "NO CAL %",
        "D2_CAPTURA_PCT": "D2 Captura %",
        "TRS_TOTAL_PCT": "TRS Total %",
        "PP_PCT": "PP %",
        "PO_PCT": "PO %",
        "AP_PCT": "AP %",
        "D2_TOTAL_PCT": "D2 %",
        "AMAQ_PCT": "AMAQ %",
        "AMOL_PCT": "AMOL %",
        "NOCAL_PCT": "NOCAL %",
        "CM_PCT": "CM %",
        "MP_PCT": "MP %",
        "NO_TRS_PCT": "NO TRS %",
        "REAL_SCRAP_PCT": "Real % SCRAP",
        "PRED_SCRAP_PCT": "Pred % SCRAP",
        "REAL_TRS_CAPTURA_PCT": "Real TRS Captura %",
        "PRED_TRS_CAPTURA_PCT": "Pred TRS Captura %",
        "REAL_NO_CAL_PCT": "Real NO CAL %",
        "PRED_NO_CAL_PCT": "Pred NO CAL %",
        "REAL_D2_CAPTURA_PCT": "Real D2 Captura %",
        "PRED_D2_CAPTURA_PCT": "Pred D2 Captura %",
        "REAL_TRS_TOTAL_PCT": "Real TRS Total %",
        "PRED_TRS_TOTAL_PCT": "Pred TRS Total %",
        "REAL_PP_PCT": "Real PP %",
        "PRED_PP_PCT": "Pred PP %",
        "REAL_PO_PCT": "Real PO %",
        "PRED_PO_PCT": "Pred PO %",
        "REAL_AP_PCT": "Real AP %",
        "PRED_AP_PCT": "Pred AP %",
        "REAL_D2_TOTAL_PCT": "Real D2 %",
        "PRED_D2_TOTAL_PCT": "Pred D2 %",
        "REAL_AMAQ_PCT": "Real AMAQ %",
        "PRED_AMAQ_PCT": "Pred AMAQ %",
        "REAL_AMOL_PCT": "Real AMOL %",
        "PRED_AMOL_PCT": "Pred AMOL %",
        "REAL_NOCAL_PCT": "Real NOCAL %",
        "PRED_NOCAL_PCT": "Pred NOCAL %",
        "REAL_CM_PCT": "Real CM %",
        "PRED_CM_PCT": "Pred CM %",
        "REAL_MP_PCT": "Real MP %",
        "PRED_MP_PCT": "Pred MP %",
        "REAL_NO_TRS_PCT": "Real NO TRS %",
        "PRED_NO_TRS_PCT": "Pred NO TRS %",
    }
    cols_to_rename = {c: rename_map[c] for c in df.columns if c in rename_map}
    return df.rename(columns=cols_to_rename)


def traducir_nombre_error(nombre):
    mapa = {
        "SCRAP_PCT": "% SCRAP",
        "TRS_CAPTURA_PCT": "TRS Captura %",
        "NO_CAL_PCT": "NO CAL %",
        "D2_CAPTURA_PCT": "D2 Captura %",
        "TRS_TOTAL_PCT": "TRS Total %",
        "PP_PCT": "PP %",
        "PO_PCT": "PO %",
        "AP_PCT": "AP %",
        "D2_TOTAL_PCT": "D2 %",
        "AMAQ_PCT": "AMAQ %",
        "AMOL_PCT": "AMOL %",
        "NOCAL_PCT": "NOCAL %",
        "CM_PCT": "CM %",
        "MP_PCT": "MP %",
        "NO_TRS_PCT": "NO TRS %",
    }
    return mapa.get(nombre, nombre)


def filtrar_por_maquina(df_resultados: pd.DataFrame, maquina_seleccionada):
    if maquina_seleccionada == "Todas":
        return df_resultados
    return df_resultados[df_resultados["Máquina"] == maquina_seleccionada].reset_index(drop=True)


# ============================================================
# UI
# ============================================================
st.title("Dashboard predictivo Autoplastek")
st.write("Predicción multivariable por máquina con consolidación diaria, ranking, semáforo y simulación por piezas.")

uploaded_file = st.file_uploader("Sube tu archivo Excel (.xlsx)", type=["xlsx"])

if uploaded_file is None:
    st.info("Sube tu archivo Excel para comenzar.")
    st.stop()

try:
    captura_raw, total_raw, captura_name, total_name = load_excel_sheets(uploaded_file)
    st.success(f"Hojas detectadas: {captura_name} y {total_name}")

    captura = prepare_captura_trs(captura_raw)
    total = prepare_total(total_raw)
    df = merge_sheets(captura, total)

except Exception as e:
    st.error(f"No se pudo cargar el Excel: {e}")
    st.stop()

with st.expander("Vista previa del dataset combinado"):
    st.dataframe(df, use_container_width=True)

with st.spinner("Entrenando modelo multisalida..."):
    pipeline, usable_features, target_cols, metricas, mae_por_maquina, mae_global, historico_pred = entrenar_modelo(df)

st.subheader("Métricas globales del modelo")
c1, c2, c3 = st.columns(3)
c1.metric("RMSE global", f"{metricas['RMSE_GLOBAL']:.3f}")
c2.metric("MAE global", f"{metricas['MAE_GLOBAL']:.3f}")
c3.metric("R² global", f"{metricas['R2_GLOBAL']:.3f}")

st.subheader("R² por columna")
detalle_metricas_ui = rename_targets_for_ui(metricas["DETALLE"].copy())
st.dataframe(detalle_metricas_ui, use_container_width=True)

st.subheader("Error esperado por máquina")
st.dataframe(mae_por_maquina, use_container_width=True)

st.subheader("Parámetros de consulta")
col1, col2 = st.columns(2)
with col1:
    modo = st.radio("Tipo de consulta", ["Por fecha", "Por semana"], horizontal=True)
with col2:
    maquinas_disponibles = ["Todas"] + sorted(df["MAQ"].dropna().unique().tolist())
    maquina_filtro = st.selectbox("Filtrar máquina", maquinas_disponibles)

if modo == "Por fecha":
    fecha_usuario = st.date_input(
        "Selecciona la fecha a consultar",
        value=df["FECHA"].max().date()
    )

    if st.button("Generar predicción por fecha"):
        resultados = predecir_fecha(
            df=df,
            pipeline=pipeline,
            usable_features=usable_features,
            target_cols=target_cols,
            mae_por_maquina=mae_por_maquina,
            mae_global=mae_global,
            fecha_objetivo=pd.Timestamp(fecha_usuario)
        )

        resultados_ui = rename_targets_for_ui(resultados.copy())
        resultados_ui = filtrar_por_maquina(resultados_ui, maquina_filtro)

        st.subheader(f"Predicción por máquina para {fecha_usuario}")
        st.dataframe(resultados_ui, use_container_width=True)

        st.subheader("Ranking de máquinas más críticas")
        ranking_col = "NO TRS %" if "NO TRS %" in resultados_ui.columns else [c for c in resultados_ui.columns if "%" in c][0]
        ranking = resultados_ui[["Máquina", ranking_col, "Semáforo", "Riesgo", "Recomendación"]].sort_values(ranking_col, ascending=False)
        st.dataframe(ranking.head(15), use_container_width=True)

        top = ranking.head(10)
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.bar(top["Máquina"].astype(str), top[ranking_col])
        ax.set_title(f"Top máquinas por {ranking_col} predicho")
        ax.set_xlabel("Máquina")
        ax.set_ylabel(ranking_col)
        plt.xticks(rotation=45)
        st.pyplot(fig)

        csv = resultados_ui.to_csv(index=False).encode("utf-8")
        st.download_button(
            "Descargar predicción por fecha",
            data=csv,
            file_name=f"prediccion_maquinas_{fecha_usuario}.csv",
            mime="text/csv"
        )

else:
    anios_disponibles = sorted(df["ANIO"].dropna().unique().tolist())
    anio_sel = st.selectbox("Año", anios_disponibles, index=len(anios_disponibles) - 1)
    semana_sel = st.selectbox("Semana ISO", list(range(1, 54)))

    if st.button("Generar predicción por semana"):
        resumen, detalle = predecir_semana(
            df=df,
            pipeline=pipeline,
            usable_features=usable_features,
            target_cols=target_cols,
            mae_por_maquina=mae_por_maquina,
            mae_global=mae_global,
            anio=int(anio_sel),
            semana=int(semana_sel)
        )

        resumen_ui = rename_targets_for_ui(resumen.copy())
        detalle_ui = rename_targets_for_ui(detalle.copy())

        resumen_ui = filtrar_por_maquina(resumen_ui, maquina_filtro)
        detalle_ui = filtrar_por_maquina(detalle_ui, maquina_filtro)

        st.subheader(f"Resumen por máquina - semana {semana_sel} del {anio_sel}")
        st.dataframe(resumen_ui, use_container_width=True)

        st.subheader("Ranking semanal de máquinas más críticas")
        ranking_col = "NO TRS %" if "NO TRS %" in resumen_ui.columns else [c for c in resumen_ui.columns if "%" in c][0]
        ranking = resumen_ui[["Máquina", ranking_col, "Semáforo", "Riesgo", "Recomendación"]].sort_values(ranking_col, ascending=False)
        st.dataframe(ranking.head(15), use_container_width=True)

        top = ranking.head(10)
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.bar(top["Máquina"].astype(str), top[ranking_col])
        ax.set_title(f"Top máquinas por {ranking_col} semanal")
        ax.set_xlabel("Máquina")
        ax.set_ylabel(ranking_col)
        plt.xticks(rotation=45)
        st.pyplot(fig)

        st.subheader("Detalle diario de la semana")
        st.dataframe(detalle_ui, use_container_width=True)

        csv = resumen_ui.to_csv(index=False).encode("utf-8")
        st.download_button(
            "Descargar resumen semanal",
            data=csv,
            file_name=f"prediccion_maquinas_semana_{anio_sel}_{semana_sel}.csv",
            mime="text/csv"
        )

st.subheader("Simulación por número de piezas planeadas")

colp1, colp2, colp3 = st.columns(3)

with colp1:
    fecha_sim = st.date_input(
        "Fecha para simulación",
        value=df["FECHA"].max().date(),
        key="fecha_sim"
    )

with colp2:
    maquina_sim = st.selectbox(
        "Máquina para simulación",
        ["Todas"] + sorted(df["MAQ"].dropna().unique().tolist()),
        key="maquina_sim"
    )

with colp3:
    piezas_planeadas = st.number_input(
        "Piezas planeadas para el día",
        min_value=1,
        value=1000,
        step=1
    )

if st.button("Simular producción diaria"):
    simulacion = simular_produccion_por_piezas(
        df=df,
        pipeline=pipeline,
        usable_features=usable_features,
        target_cols=target_cols,
        mae_por_maquina=mae_por_maquina,
        mae_global=mae_global,
        fecha_objetivo=pd.Timestamp(fecha_sim),
        piezas_planeadas=int(piezas_planeadas),
        maquina=maquina_sim
    )

    if simulacion.empty:
        st.warning("No se pudo generar la simulación.")
    else:
        simulacion_ui = rename_targets_for_ui(simulacion.copy())

        if "Tipo de error dominante" in simulacion_ui.columns:
            simulacion_ui["Tipo de error dominante"] = simulacion_ui["Tipo de error dominante"].apply(traducir_nombre_error)

        st.dataframe(simulacion_ui, use_container_width=True)

        ranking_sim = simulacion_ui[[
            "Máquina",
            "Piezas planeadas",
            "Tipo de error dominante",
            "% error dominante",
            "Piezas con error esperadas",
            "Semáforo",
            "Riesgo",
            "Recomendación"
        ]].sort_values("% error dominante", ascending=False)

        st.subheader("Resumen de simulación")
        st.dataframe(ranking_sim, use_container_width=True)

        fig, ax = plt.subplots(figsize=(10, 5))
        top_sim = ranking_sim.head(10)
        ax.bar(top_sim["Máquina"].astype(str), top_sim["Piezas con error esperadas"])
        ax.set_title("Piezas con error esperadas por máquina")
        ax.set_xlabel("Máquina")
        ax.set_ylabel("Piezas con error esperadas")
        plt.xticks(rotation=45)
        st.pyplot(fig)

        csv_sim = simulacion_ui.to_csv(index=False).encode("utf-8")
        st.download_button(
            "Descargar simulación",
            data=csv_sim,
            file_name=f"simulacion_piezas_{fecha_sim}.csv",
            mime="text/csv"
        )

st.subheader("Histórico real vs predicción")
historico_ui = rename_targets_for_ui(historico_pred.copy())

variables_hist = []
for c in historico_ui.columns:
    if c.startswith("Real ") and c.replace("Real ", "Pred ") in historico_ui.columns:
        variables_hist.append(c.replace("Real ", ""))

colh1, colh2 = st.columns(2)
with colh1:
    maquina_hist = st.selectbox("Máquina para histórico", sorted(historico_ui["MAQ"].dropna().unique().tolist()))
with colh2:
    variable_hist = st.selectbox("Variable a comparar", variables_hist)

hist_f = historico_ui[historico_ui["MAQ"] == maquina_hist].sort_values("FECHA").copy()

real_col = f"Real {variable_hist}"
pred_col = f"Pred {variable_hist}"

if not hist_f.empty and real_col in hist_f.columns and pred_col in hist_f.columns:
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(hist_f["FECHA"], hist_f[real_col], marker="o", label="Real")
    ax.plot(hist_f["FECHA"], hist_f[pred_col], marker="o", label="Predicción")
    ax.set_title(f"Máquina {maquina_hist} - {variable_hist}: real vs predicción")
    ax.set_xlabel("Fecha")
    ax.set_ylabel(variable_hist)
    ax.legend()
    plt.xticks(rotation=45)
    st.pyplot(fig)

    tabla_hist = hist_f[["FECHA", "MAQ", "TURNO", real_col, pred_col]].copy()
    tabla_hist["ERROR_ABS"] = (tabla_hist[real_col] - tabla_hist[pred_col]).abs().round(2)
    st.dataframe(tabla_hist, use_container_width=True)
