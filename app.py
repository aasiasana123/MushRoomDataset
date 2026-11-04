"""
Deep Learning & NLP Exercise - Streamlit App
---------------------------------------------
Task 1: CNN vs RNN classification on the Mushroom dataset (TensorFlow/Keras)
Task 2: Tokenizing exercise_text.json with a Hugging Face Transformers tokenizer

Run locally:   streamlit run app.py
Deployment:    push app.py, requirements.txt, mushrooms.csv, exercise_text.json to GitHub,
               then deploy on Streamlit Community Cloud.
"""

import json
import os
import time
from collections import Counter
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

st.set_page_config(page_title="CNN vs RNN + HF Tokenizer", page_icon="🍄", layout="wide")

SEED = 42
CNN_CFG = {"name": "CNN", "optimizer": "Adam", "batch_size": 30, "epochs": 10}
RNN_CFG = {"name": "RNN", "optimizer": "RMSprop", "batch_size": 40, "epochs": 20}
CLASS_LABELS = ["Edible (e)", "Poisonous (p)"]


# ----------------------------------------------------------------------------
# Helpers: locating / loading files
# ----------------------------------------------------------------------------
def find_file(filename):
    """Look for a file next to app.py or in the working directory."""
    for base in (Path(__file__).parent, Path.cwd()):
        candidate = base / filename
        if candidate.exists():
            return candidate
    return None


@st.cache_data(show_spinner=False)
def read_csv_path(path):
    return pd.read_csv(path)


def get_dataframe(uploaded):
    if uploaded is not None:
        return pd.read_csv(uploaded)
    path = find_file("mushrooms.csv")
    if path is not None:
        return read_csv_path(str(path))
    return None


def get_json_text(uploaded):
    if uploaded is not None:
        return json.load(uploaded).get("text", "")
    path = find_file("exercise_text.json")
    if path is not None:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f).get("text", "")
    return None


# ----------------------------------------------------------------------------
# Task 1: preprocessing
# ----------------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def preprocess(df):
    le = LabelEncoder()
    y = le.fit_transform(df["class"])
    X = pd.get_dummies(df.drop(columns=["class"])).astype("float32")

    X_train, X_test, y_train, y_test = train_test_split(
        X.values, y, test_size=0.30, random_state=SEED, stratify=y
    )
    # (samples, timesteps, channels) - required by Conv1D and LSTM
    X_train_seq = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
    X_test_seq = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)
    return {
        "classes": list(le.classes_),
        "n_encoded_features": X.shape[1],
        "X_train": X_train_seq,
        "X_test": X_test_seq,
        "y_train": y_train,
        "y_test": y_test,
        "y_test_labels": le.inverse_transform(y_test),
        "le": le,
    }


# ----------------------------------------------------------------------------
# Task 1: models
# ----------------------------------------------------------------------------
def build_cnn(n_features):
    import tensorflow as tf
    from tensorflow.keras import layers, models

    model = models.Sequential(
        [
            layers.Input(shape=(n_features, 1)),
            layers.Conv1D(32, 3, activation="relu"),
            layers.MaxPooling1D(2),
            layers.Conv1D(64, 3, activation="relu"),
            layers.MaxPooling1D(2),
            layers.Flatten(),
            layers.Dense(64, activation="relu"),
            layers.Dropout(0.3),
            layers.Dense(1, activation="sigmoid"),
        ],
        name="CNN_Model",
    )
    model.compile(
        optimizer=tf.keras.optimizers.Adam(),
        loss="binary_crossentropy",
        metrics=["accuracy"],
    )
    return model


def build_rnn(n_features):
    import tensorflow as tf
    from tensorflow.keras import layers, models

    model = models.Sequential(
        [
            layers.Input(shape=(n_features, 1)),
            layers.LSTM(64),
            layers.Dropout(0.3),
            layers.Dense(32, activation="relu"),
            layers.Dense(1, activation="sigmoid"),
        ],
        name="RNN_Model",
    )
    model.compile(
        optimizer=tf.keras.optimizers.RMSprop(),
        loss="binary_crossentropy",
        metrics=["accuracy"],
    )
    return model


def model_summary_text(model):
    lines = []
    model.summary(print_fn=lambda line, **kwargs: lines.append(line))
    return "\n".join(lines)


def architecture_blocks(model):
    blocks = []
    for layer in model.layers:
        try:
            shape = str(tuple(layer.output.shape))
        except Exception:
            shape = ""
        blocks.append((layer.__class__.__name__, shape))
    return blocks


def make_progress_callback(bar, status, total_epochs, label):
    import tensorflow as tf

    class _Progress(tf.keras.callbacks.Callback):
        def on_epoch_end(self, epoch, logs=None):
            logs = logs or {}
            bar.progress((epoch + 1) / total_epochs)
            status.write(
                f"**{label}** - epoch {epoch + 1}/{total_epochs} | "
                f"loss: {logs.get('loss', 0):.4f} | acc: {logs.get('accuracy', 0):.4f} | "
                f"val_loss: {logs.get('val_loss', 0):.4f} | val_acc: {logs.get('val_accuracy', 0):.4f}"
            )

    return _Progress()


def train_model(builder, cfg, data, bar, status):
    import tensorflow as tf

    tf.keras.utils.set_random_seed(SEED)
    n_features = data["X_train"].shape[1]
    model = builder(n_features)
    summary = model_summary_text(model)
    blocks = architecture_blocks(model)

    start = time.time()
    history = model.fit(
        data["X_train"],
        data["y_train"],
        batch_size=cfg["batch_size"],
        epochs=cfg["epochs"],
        validation_data=(data["X_test"], data["y_test"]),
        verbose=0,
        callbacks=[make_progress_callback(bar, status, cfg["epochs"], cfg["name"])],
    )
    elapsed = time.time() - start

    y_prob = model.predict(data["X_test"], verbose=0).ravel()
    y_pred = (y_prob > 0.5).astype(int)
    test_loss, test_acc = model.evaluate(data["X_test"], data["y_test"], verbose=0)

    return {
        "cfg": cfg,
        "history": history.history,
        "y_prob": y_prob,
        "y_pred": y_pred,
        "test_loss": float(test_loss),
        "test_acc": float(accuracy_score(data["y_test"], y_pred)),
        "train_time": elapsed,
        "summary": summary,
        "blocks": blocks,
        "n_params": int(model.count_params()),
    }


# ----------------------------------------------------------------------------
# Task 1: plots
# ----------------------------------------------------------------------------
def plot_class_distribution(df):
    fig, ax = plt.subplots(figsize=(5, 3.8))
    counts = df["class"].value_counts().sort_index()
    ax.bar(counts.index, counts.values, color=["green", "red"])
    for i, v in enumerate(counts.values):
        ax.text(i, v + 20, str(v), ha="center", fontweight="bold")
    ax.set_title("Class Distribution (e = edible, p = poisonous)")
    ax.set_xlabel("Class")
    ax.set_ylabel("Count")
    fig.tight_layout()
    return fig


def plot_feature_distribution(df, column):
    fig, ax = plt.subplots(figsize=(7, 3.8))
    pd.crosstab(df[column], df["class"]).plot(kind="bar", ax=ax, color=["green", "red"])
    ax.set_title(f"'{column}' by class")
    ax.set_xlabel(column)
    ax.set_ylabel("Count")
    ax.tick_params(axis="x", rotation=0)
    fig.tight_layout()
    return fig


def plot_architecture(blocks, title, color):
    n = len(blocks)
    fig, ax = plt.subplots(figsize=(3.6, 0.85 * n + 0.6))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, n)
    ax.axis("off")
    for i, (name, shape) in enumerate(blocks):
        y = n - i - 1
        ax.add_patch(plt.Rectangle((0.08, y + 0.15), 0.84, 0.7, color=color, alpha=0.85, ec="black"))
        ax.text(0.5, y + 0.5, f"{name}\n{shape}", ha="center", va="center", fontsize=8.5, color="white", fontweight="bold")
        if i < n - 1:
            ax.annotate("", xy=(0.5, y + 0.15), xytext=(0.5, y), arrowprops=dict(arrowstyle="->"))
    ax.set_title(title, fontsize=11, fontweight="bold")
    fig.tight_layout()
    return fig


def plot_training_curves(res_cnn, res_rnn):
    fig, axes = plt.subplots(2, 2, figsize=(13, 8))
    for col, (res, name) in enumerate([(res_cnn, "CNN"), (res_rnn, "RNN")]):
        h = res["history"]
        axes[0, col].plot(h["accuracy"], marker="o", label="Train")
        axes[0, col].plot(h["val_accuracy"], marker="s", label="Test")
        axes[0, col].set_title(f"{name} Accuracy")
        axes[0, col].set_xlabel("Epoch")
        axes[0, col].set_ylabel("Accuracy")
        axes[0, col].legend()
        axes[0, col].grid(alpha=0.3)

        axes[1, col].plot(h["loss"], marker="o", label="Train")
        axes[1, col].plot(h["val_loss"], marker="s", label="Test")
        axes[1, col].set_title(f"{name} Loss")
        axes[1, col].set_xlabel("Epoch")
        axes[1, col].set_ylabel("Loss")
        axes[1, col].legend()
        axes[1, col].grid(alpha=0.3)
    fig.tight_layout()
    return fig


def plot_confusion_matrices(y_true, res_cnn, res_rnn, classes):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    for ax, res, name, cmap in zip(axes, [res_cnn, res_rnn], ["CNN (Adam)", "RNN (RMSprop)"], ["Blues", "Oranges"]):
        cm = confusion_matrix(y_true, res["y_pred"])
        sns.heatmap(cm, annot=True, fmt="d", cmap=cmap, cbar=False,
                    xticklabels=classes, yticklabels=classes, ax=ax)
        ax.set_title(f"Confusion Matrix - {name}")
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
    fig.tight_layout()
    return fig


def plot_accuracy_comparison(res_cnn, res_rnn):
    accs = [res_cnn["test_acc"], res_rnn["test_acc"]]
    fig, ax = plt.subplots(figsize=(5.5, 4))
    bars = ax.bar(["CNN", "RNN"], accs, color=["steelblue", "darkorange"])
    ax.set_ylim(max(0, min(accs) - 0.02), 1.005)
    ax.set_ylabel("Test Accuracy")
    ax.set_title("Test Accuracy Comparison")
    for b, v in zip(bars, accs):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.0005, f"{v:.4f}", ha="center", fontweight="bold")
    fig.tight_layout()
    return fig


def report_df(y_true, y_pred):
    rep = classification_report(y_true, y_pred, target_names=CLASS_LABELS, output_dict=True, digits=4)
    return pd.DataFrame(rep).T.round(4)


# ----------------------------------------------------------------------------
# Task 2: tokenizer
# ----------------------------------------------------------------------------
def get_hf_token():
    token = os.environ.get("HF_TOKEN")
    if token:
        return token
    try:
        return st.secrets.get("HF_TOKEN")
    except Exception:
        return None


@st.cache_resource(show_spinner=False)
def load_tokenizer(model_name, local_dir, token):
    from transformers import AutoTokenizer

    if local_dir:
        return AutoTokenizer.from_pretrained(local_dir, local_files_only=True)

    last_error = None
    for attempt in range(1, 4):
        try:
            return AutoTokenizer.from_pretrained(model_name, token=token or None)
        except Exception as e:  # rate limits / connection problems
            last_error = e
            time.sleep(attempt * 5)
    raise last_error


# ----------------------------------------------------------------------------
# UI
# ----------------------------------------------------------------------------
st.title("🍄 Deep Learning & NLP Exercise")
st.caption("Task 1: CNN vs RNN on the Mushroom dataset (TensorFlow/Keras)  |  Task 2: Hugging Face tokenization")

with st.sidebar:
    st.header("Data files")
    st.write("The app auto-loads `mushrooms.csv` and `exercise_text.json` if they sit next to `app.py`. Otherwise upload them here.")
    up_csv = st.file_uploader("mushrooms.csv", type=["csv"])
    up_json = st.file_uploader("exercise_text.json", type=["json"])
    st.divider()
    st.markdown(
        f"**Task 1 settings (fixed by the assignment)**\n\n"
        f"- Split: 70% train / 30% test\n"
        f"- CNN: {CNN_CFG['optimizer']}, batch {CNN_CFG['batch_size']}, {CNN_CFG['epochs']} epochs\n"
        f"- RNN: {RNN_CFG['optimizer']}, batch {RNN_CFG['batch_size']}, {RNN_CFG['epochs']} epochs"
    )

tab1, tab2 = st.tabs(["Task 1: CNN vs RNN", "Task 2: HF Tokenization"])

# =============================== TASK 1 =====================================
with tab1:
    df = get_dataframe(up_csv)
    if df is None:
        st.warning("`mushrooms.csv` not found. Upload it in the sidebar or place it next to app.py.")
    else:
        st.subheader("1. Data exploration")
        c1, c2, c3 = st.columns(3)
        c1.metric("Rows", f"{df.shape[0]:,}")
        c2.metric("Columns", df.shape[1])
        c3.metric("Missing values", int(df.isnull().sum().sum()))

        with st.expander("Preview data", expanded=True):
            st.dataframe(df.head(20))
        with st.expander("Unique values per column"):
            st.dataframe(df.nunique().rename("Unique values").to_frame())

        left, right = st.columns(2)
        with left:
            st.pyplot(plot_class_distribution(df))
        with right:
            feature = st.selectbox("Feature to visualize", [c for c in df.columns if c != "class"],
                                   index=list(df.columns).index("odor") - 1 if "odor" in df.columns else 0)
            st.pyplot(plot_feature_distribution(df, feature))

        st.subheader("2. Preprocessing")
        data = preprocess(df)
        st.write(
            f"- Target `class` encoded: **{dict(zip(data['classes'], range(len(data['classes']))))}**\n"
            f"- 22 categorical features one-hot encoded into **{data['n_encoded_features']}** binary features\n"
            f"- Reshaped to `(samples, {data['n_encoded_features']}, 1)` for Conv1D / LSTM input\n"
            f"- Train shape: `{data['X_train'].shape}` | Test shape: `{data['X_test'].shape}`"
        )

        st.subheader("3. Train and evaluate")
        if st.button("Train CNN and RNN", type="primary"):
            try:
                import tensorflow  # noqa: F401
            except ImportError:
                st.error("TensorFlow is not installed. Add `tensorflow-cpu` to requirements.txt.")
                st.stop()

            st.write("**Model I - CNN**")
            bar1, stat1 = st.progress(0.0), st.empty()
            res_cnn = train_model(build_cnn, CNN_CFG, data, bar1, stat1)

            st.write("**Model II - RNN**")
            bar2, stat2 = st.progress(0.0), st.empty()
            res_rnn = train_model(build_rnn, RNN_CFG, data, bar2, stat2)

            st.session_state["results"] = {"cnn": res_cnn, "rnn": res_rnn, "n_rows": len(df)}
            st.success("Training complete.")

        results = st.session_state.get("results")
        if results is None:
            st.info("Click **Train CNN and RNN** to run both experiments (takes a few minutes on CPU).")
        else:
            res_cnn, res_rnn = results["cnn"], results["rnn"]
            y_test = data["y_test"]

            st.subheader("4. Model architectures")
            a1, a2 = st.columns(2)
            with a1:
                st.pyplot(plot_architecture(res_cnn["blocks"], "CNN", "#1f77b4"))
                with st.expander("CNN model summary"):
                    st.code(res_cnn["summary"])
            with a2:
                st.pyplot(plot_architecture(res_rnn["blocks"], "RNN (LSTM)", "#ff7f0e"))
                with st.expander("RNN model summary"):
                    st.code(res_rnn["summary"])

            st.subheader("5. Training curves")
            st.pyplot(plot_training_curves(res_cnn, res_rnn))

            st.subheader("6. Predictions of both models")
            pred_df = pd.DataFrame(
                {
                    "Actual": data["y_test_labels"],
                    "CNN_Prediction": data["le"].inverse_transform(res_cnn["y_pred"]),
                    "RNN_Prediction": data["le"].inverse_transform(res_rnn["y_pred"]),
                    "CNN_Prob(p)": res_cnn["y_prob"].round(4),
                    "RNN_Prob(p)": res_rnn["y_prob"].round(4),
                }
            )
            st.dataframe(pred_df.head(50))
            st.download_button("Download all predictions (CSV)", pred_df.to_csv(index=False).encode(),
                               file_name="predictions.csv", mime="text/csv")

            st.subheader("7. Confusion matrices")
            st.pyplot(plot_confusion_matrices(y_test, res_cnn, res_rnn, CLASS_LABELS))

            st.subheader("8. Classification reports")
            r1, r2 = st.columns(2)
            with r1:
                st.markdown("**CNN (Adam, batch 30, 10 epochs)**")
                st.dataframe(report_df(y_test, res_cnn["y_pred"]))
            with r2:
                st.markdown("**RNN (RMSprop, batch 40, 20 epochs)**")
                st.dataframe(report_df(y_test, res_rnn["y_pred"]))

            st.subheader("9. Performance comparison")
            comp = pd.DataFrame(
                {
                    "Model": ["CNN", "RNN"],
                    "Optimizer": ["Adam", "RMSprop"],
                    "Batch Size": [CNN_CFG["batch_size"], RNN_CFG["batch_size"]],
                    "Epochs": [CNN_CFG["epochs"], RNN_CFG["epochs"]],
                    "Parameters": [res_cnn["n_params"], res_rnn["n_params"]],
                    "Train Time (s)": [round(res_cnn["train_time"], 1), round(res_rnn["train_time"], 1)],
                    "Test Loss": [round(res_cnn["test_loss"], 4), round(res_rnn["test_loss"], 4)],
                    "Test Accuracy": [round(res_cnn["test_acc"], 4), round(res_rnn["test_acc"], 4)],
                    "Misclassified": [int((y_test != res_cnn["y_pred"]).sum()), int((y_test != res_rnn["y_pred"]).sum())],
                }
            )
            cc1, cc2 = st.columns([3, 2])
            with cc1:
                st.dataframe(comp)
            with cc2:
                st.pyplot(plot_accuracy_comparison(res_cnn, res_rnn))

            acc_c, acc_r = res_cnn["test_acc"], res_rnn["test_acc"]
            if acc_c > acc_r:
                st.success(f"Higher test accuracy: **CNN** ({acc_c:.4f} vs RNN {acc_r:.4f}). Recommended model: **CNN**.")
            elif acc_r > acc_c:
                st.success(f"Higher test accuracy: **RNN** ({acc_r:.4f} vs CNN {acc_c:.4f}). Recommended model: **RNN**.")
            else:
                faster = "CNN" if res_cnn["train_time"] <= res_rnn["train_time"] else "RNN"
                st.success(
                    f"Both models tied at {acc_c:.4f} test accuracy. Recommended model: **{faster}** "
                    f"(it trained faster with equal accuracy)."
                )

# =============================== TASK 2 =====================================
with tab2:
    st.subheader("Tokenize exercise_text.json with a Hugging Face tokenizer")
    text_from_file = get_json_text(up_json)

    if text_from_file is None:
        st.warning("`exercise_text.json` not found. Upload it in the sidebar or place it next to app.py.")
    else:
        text = st.text_area("Text loaded from JSON (editable)", value=text_from_file, height=220)
        st.write(f"Characters: **{len(text)}** | Words: **{len(text.split())}**")

        col_a, col_b = st.columns(2)
        with col_a:
            model_name = st.selectbox(
                "Tokenizer",
                ["bert-base-uncased", "distilbert-base-uncased", "gpt2", "roberta-base"],
            )
        with col_b:
            local_dir = st.text_input(
                "Optional: local tokenizer folder (use if Hugging Face rate-limits you with 429)",
                value="",
            )
        max_len = st.slider("Max length (truncation)", 32, 512, 512, step=32)

        if st.button("Tokenize", type="primary"):
            try:
                with st.spinner("Loading tokenizer..."):
                    tokenizer = load_tokenizer(model_name, local_dir.strip() or None, get_hf_token())
                st.session_state["tok"] = {
                    "name": model_name if not local_dir.strip() else local_dir.strip(),
                    "cls": tokenizer.__class__.__name__,
                    "vocab": tokenizer.vocab_size,
                    "special": tokenizer.special_tokens_map,
                    "tokens": tokenizer.tokenize(text),
                    "ids": tokenizer.convert_tokens_to_ids(tokenizer.tokenize(text)),
                    "encoded": dict(tokenizer(text, truncation=True, max_length=max_len)),
                    "decoded": tokenizer.decode(
                        tokenizer(text, truncation=True, max_length=max_len)["input_ids"],
                        skip_special_tokens=True,
                    ),
                    "words": {w: tokenizer.tokenize(w) for w in text.split()[:12]},
                }
            except Exception as e:
                st.error(f"Could not load the tokenizer: {e}")
                st.info(
                    "If this is a **429 Too Many Requests** error, add a free Hugging Face token as `HF_TOKEN` "
                    "(Streamlit Cloud: App settings > Secrets), wait a few minutes and retry, "
                    "or download the tokenizer files and put the folder path in the box above."
                )

        tok = st.session_state.get("tok")
        if tok:
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Tokenizer", tok["cls"])
            m2.metric("Vocabulary size", f"{tok['vocab']:,}")
            m3.metric("Tokens (no special)", len(tok["tokens"]))
            m4.metric("input_ids length", len(tok["encoded"]["input_ids"]))
            st.caption(f"Model: {tok['name']} | Special tokens: {tok['special']}")

            st.markdown("**Tokens**")
            st.write(tok["tokens"])

            st.markdown("**Token to ID table**")
            table = pd.DataFrame({"Position": range(len(tok["tokens"])), "Token": tok["tokens"], "Token ID": tok["ids"]})
            st.dataframe(table)
            st.download_button("Download token table (CSV)", table.to_csv(index=False).encode(),
                               file_name="tokens.csv", mime="text/csv")

            st.markdown("**Full encoding (with special tokens, attention mask)**")
            st.json({k: v for k, v in tok["encoded"].items()})

            st.markdown("**Decoded back to text**")
            st.write(tok["decoded"])

            st.markdown("**How words split into sub-words**")
            st.dataframe(pd.DataFrame({"Word": list(tok["words"].keys()),
                                       "Sub-word tokens": [", ".join(v) for v in tok["words"].values()]}))

            st.markdown("**Top 15 most frequent tokens**")
            freq = Counter(tok["tokens"]).most_common(15)
            labels, counts = zip(*freq)
            fig, ax = plt.subplots(figsize=(10, 4))
            ax.bar(labels, counts, color="steelblue")
            ax.set_xlabel("Token")
            ax.set_ylabel("Frequency")
            ax.set_title("Top 15 Most Frequent Tokens")
            ax.tick_params(axis="x", rotation=45)
            fig.tight_layout()
            st.pyplot(fig)
