import json, joblib
import numpy as np, pandas as pd
import streamlit as st
import tensorflow as tf
import matplotlib.pyplot as plt

st.set_page_config(page_title="Predictive Maintenance", layout="wide")


@st.cache_resource
def load():
    meta = json.load(open('artifacts/meta.json'))
    lstm = tf.keras.models.load_model('artifacts/lstm.keras')
    ae = tf.keras.models.load_model('artifacts/ae.keras')
    iso = joblib.load('artifacts/iso.joblib')
    df = pd.read_csv('artifacts/test_processed.csv')
    rul = pd.read_csv('artifacts/rul_test.csv')['RUL'].values
    return meta, lstm, ae, iso, df, rul


meta, lstm, ae, iso, df, rul_test = load()
sensors, af, W = meta['sensors'], meta['af'], meta['W']
RUL_MAX, THR_IF, THR_AE = meta['RUL_MAX'], meta['thr_if'], meta['thr_ae']


@st.cache_data
def score_engine(unit):
    d = df[df.unit == unit].reset_index(drop=True)
    d['anom_if'] = -iso.score_samples(d[af])
    X = d[af].values.astype('float32')
    d['anom_ae'] = np.mean((X - ae.predict(X, verbose=0)) ** 2, axis=1)

    arr = d[sensors].values.astype('float32')
    pad = np.vstack([np.repeat(arr[:1], W - 1, axis=0), arr])
    wins = np.stack([pad[i:i + W] for i in range(len(arr))])
    d['rul_pred'] = lstm.predict(wins, verbose=0).ravel() * RUL_MAX

    last = d['cycle'].max()
    d['rul_true'] = np.minimum(rul_test[unit - 1] + (last - d['cycle']), RUL_MAX)
    return d


def flag3(series, thr, t):
    s = series.iloc[max(0, t - 2):t + 1]
    return len(s) == 3 and bool((s > thr).all())


# ---- Sidebar ----
st.sidebar.header("Parametrlər")
unit = st.sidebar.selectbox("Mühərrik (unit)", sorted(df.unit.unique()))
red_rul = st.sidebar.slider("Qırmızı: RUL <", 10, 60, 30)
yel_rul = st.sidebar.slider("Sarı: RUL <", 30, 100, 60)

d = score_engine(unit)
t = st.slider("Cari dövr (cycle)", 1, len(d), len(d)) - 1
now = d.iloc[t]

# ---- Status ----
if_flag = flag3(d['anom_if'], THR_IF, t)
ae_flag = flag3(d['anom_ae'], THR_AE, t)

if now['rul_pred'] < red_rul or (if_flag and ae_flag):
    status, color = "QIRMIZI: təcili təmir planlaşdırın", "#d62728"
elif if_flag or now['rul_pred'] < yel_rul:
    status, color = "SARI: diqqətlə izləyin", "#e6a700"
else:
    status, color = "YAŞIL: normal", "#2ca02c"

st.title("Sənaye Avadanlığının Proqnozlaşdırıcı Təmiri")
st.markdown(
    f"<div style='padding:16px;border-radius:8px;background:{color};"
    f"color:white;font-size:22px'>{status}</div>",
    unsafe_allow_html=True,
)

c1, c2, c3, c4 = st.columns(4)
c1.metric("Proqnoz RUL (dövr)", f"{now['rul_pred']:.0f}")
c2.metric("Real RUL (həddlənmiş)", f"{now['rul_true']:.0f}")
c3.metric("IF anomaliya", "Bəli" if if_flag else "Xeyr")
c4.metric("AE anomaliya", "Bəli" if ae_flag else "Xeyr")

# ---- Qrafiklər ----
cyc = d['cycle'][:t + 1]
col1, col2 = st.columns(2)

with col1:
    fig, ax = plt.subplots(figsize=(6, 3.5))
    ax.plot(cyc, d['rul_true'][:t + 1], '--', label='Real RUL')
    ax.plot(cyc, d['rul_pred'][:t + 1], label='LSTM proqnoz')
    ax.axhline(red_rul, color='r', ls=':', lw=1)
    ax.set_xlabel('Dövr')
    ax.set_ylabel('RUL')
    ax.legend()
    ax.set_title('Qalıq ömür')
    st.pyplot(fig)

with col2:
    fig, ax = plt.subplots(figsize=(6, 3.5))
    ax.plot(cyc, d['anom_if'][:t + 1], label='Isolation Forest')
    ax.axhline(THR_IF, color='r', ls='--', lw=1, label='IF həddi')
    ax2 = ax.twinx()
    ax2.plot(cyc, d['anom_ae'][:t + 1], color='g', alpha=.6, label='Autoencoder')
    ax2.axhline(THR_AE, color='g', ls='--', lw=1)
    ax.set_xlabel('Dövr')
    ax.set_ylabel('IF skor')
    ax2.set_ylabel('AE xəta')
    ax.set_title('Anomaliya skoru')
    ax.legend(loc='upper left')
    st.pyplot(fig)

st.subheader("Sensor trendləri")
pick = st.multiselect("Sensorlar", sensors, default=sensors[:3])
if pick:
    st.line_chart(d.set_index('cycle').loc[:t + 1, pick])
