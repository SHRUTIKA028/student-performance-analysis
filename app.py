import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import LabelEncoder

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Student Performance Analysis",
    page_icon="🎓",
    layout="wide",
)

# ── Global colour palette ─────────────────────────────────────────────────────
COLORS = ["#6366f1", "#f59e0b", "#10b981", "#ef4444", "#3b82f6", "#ec4899", "#14b8a6", "#f97316"]
PLOTLY_TEMPLATE = "plotly_white"

# ── Custom CSS – vibrant gradient sidebar + metric cards ─────────────────────
st.markdown("""
<style>
/* Gradient sidebar */
[data-testid="stSidebar"] {
    background: linear-gradient(160deg, #1e1b4b 0%, #312e81 40%, #4f46e5 100%);
}
[data-testid="stSidebar"] * { color: #e0e7ff !important; }
[data-testid="stSidebar"] .stRadio label { font-size: 1rem; font-weight: 600; }

/* Metric cards */
[data-testid="metric-container"] {
    background: linear-gradient(135deg, #6366f1 0%, #8b5cf6 100%);
    border-radius: 12px;
    padding: 16px;
    color: white !important;
    box-shadow: 0 4px 15px rgba(99,102,241,0.4);
}
[data-testid="metric-container"] * { color: white !important; }

/* Tabs */
button[data-baseweb="tab"] {
    font-weight: 600;
    font-size: 0.95rem;
}

/* Page title gradient */
h1 { background: linear-gradient(90deg,#6366f1,#ec4899); -webkit-background-clip:text; -webkit-text-fill-color:transparent; }
</style>
""", unsafe_allow_html=True)

# ── Load & cache data ─────────────────────────────────────────────────────────
@st.cache_data
def load_data():
    df = pd.read_csv("StudentPerformanceFactors.csv")
    return df

@st.cache_data
def preprocess(df):
    df = df.copy()
    for col in df.select_dtypes(include="object").columns:
        df[col] = df[col].fillna(df[col].mode()[0])
    le = LabelEncoder()
    cat_cols = df.select_dtypes(include="object").columns.tolist()
    for col in cat_cols:
        df[col] = le.fit_transform(df[col])
    return df, cat_cols

df_raw = load_data()
df_enc, cat_cols = preprocess(df_raw)

# ── Sidebar navigation ────────────────────────────────────────────────────────
st.sidebar.image("https://img.icons8.com/fluency/96/graduation-cap.png", width=72)
st.sidebar.title("Student Performance")
st.sidebar.markdown("---")
page = st.sidebar.radio(
    "Navigate",
    ["📊 Overview", "📈 EDA", "🤖 Prediction", "🔍 Predict My Score"],
)
st.sidebar.markdown("---")
st.sidebar.caption("🎓 Built with Streamlit & scikit-learn")

# ═══════════════════════════════════════════════════════════════════════════════
# PAGE 1 – OVERVIEW
# ═══════════════════════════════════════════════════════════════════════════════
if page == "📊 Overview":
    st.title("📊 Dataset Overview")
    st.markdown("A quick snapshot of the **Student Performance Factors** dataset.")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("🧑‍🎓 Total Students", f"{df_raw.shape[0]:,}")
    c2.metric("📋 Features", df_raw.shape[1] - 1)
    c3.metric("🏆 Avg Exam Score", f"{df_raw['Exam_Score'].mean():.1f}")
    c4.metric("❓ Missing Values", int(df_raw.isnull().sum().sum()))

    st.divider()

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("📄 First 5 Rows")
        st.dataframe(df_raw.head().style.background_gradient(cmap="Purples"), use_container_width=True)

    with col2:
        st.subheader("🗂️ Column Info")
        info_df = pd.DataFrame({
            "Column": df_raw.columns,
            "Type": df_raw.dtypes.values,
            "Missing": df_raw.isnull().sum().values,
            "Unique": df_raw.nunique().values,
        })
        st.dataframe(
            info_df.style.bar(subset=["Missing"], color="#ef4444")
                         .bar(subset=["Unique"], color="#6366f1"),
            use_container_width=True,
        )

    st.divider()
    st.subheader("📐 Descriptive Statistics")
    st.dataframe(
        df_raw.describe().T.style.background_gradient(cmap="YlOrRd", axis=1),
        use_container_width=True,
    )

# ═══════════════════════════════════════════════════════════════════════════════
# PAGE 2 – EDA
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "📈 EDA":
    st.title("📈 Exploratory Data Analysis")

    tab1, tab2, tab3, tab4 = st.tabs([
        "🎯 Score Distribution", "🔥 Correlations", "📦 Category Breakdown", "🔵 Feature vs Score"
    ])

    # Tab 1 – Score distribution
    with tab1:
        st.subheader("Exam Score Distribution")
        fig = px.histogram(
            df_raw, x="Exam_Score", nbins=30,
            color_discrete_sequence=["#6366f1"],
            labels={"Exam_Score": "Exam Score"},
            title="Distribution of Exam Scores",
            template=PLOTLY_TEMPLATE,
        )
        fig.update_traces(marker_line_color="white", marker_line_width=1)
        fig.update_layout(bargap=0.05, title_font_size=18, title_font_color="#6366f1")
        st.plotly_chart(fig, use_container_width=True)

        col1, col2 = st.columns(2)
        with col1:
            fig2 = px.box(
                df_raw, y="Exam_Score", title="Exam Score Box Plot",
                color_discrete_sequence=["#ec4899"],
                template=PLOTLY_TEMPLATE,
            )
            fig2.update_layout(title_font_color="#ec4899")
            st.plotly_chart(fig2, use_container_width=True)
        with col2:
            stats = df_raw["Exam_Score"].describe()
            st.subheader("📊 Score Statistics")
            stat_labels = {"count": "🔢 Count", "mean": "📊 Mean", "std": "📉 Std Dev",
                           "min": "⬇️ Min", "25%": "25th %ile", "50%": "50th %ile (Median)",
                           "75%": "75th %ile", "max": "⬆️ Max"}
            for k, v in stats.items():
                label = stat_labels.get(k, k)
                st.markdown(f"**{label}:** `{v:.2f}`")

    # Tab 2 – Correlations
    with tab2:
        st.subheader("🔥 Correlation Heatmap")
        num_df = df_enc.select_dtypes(include="number")
        fig, ax = plt.subplots(figsize=(12, 8))
        fig.patch.set_facecolor("#0f0f1a")
        ax.set_facecolor("#0f0f1a")
        sns.heatmap(
            num_df.corr(), annot=True, fmt=".2f",
            cmap="plasma", ax=ax, annot_kws={"size": 7},
            linewidths=0.5, linecolor="#1e1b4b",
        )
        plt.title("Feature Correlation Matrix", color="white", fontsize=14, pad=12)
        plt.xticks(color="white", fontsize=7)
        plt.yticks(color="white", fontsize=7)
        st.pyplot(fig)
        plt.close()

        st.subheader("Top Correlations with Exam Score")
        corr_series = num_df.corr()["Exam_Score"].drop("Exam_Score").sort_values(ascending=False)
        fig3 = px.bar(
            x=corr_series.values, y=corr_series.index,
            orientation="h",
            labels={"x": "Correlation", "y": "Feature"},
            color=corr_series.values,
            color_continuous_scale="Turbo",
            title="Feature Correlation with Exam Score",
            template=PLOTLY_TEMPLATE,
        )
        fig3.update_layout(title_font_color="#6366f1")
        st.plotly_chart(fig3, use_container_width=True)

    # Tab 3 – Category breakdown
    with tab3:
        st.subheader("Exam Score by Category")
        cat_feature = st.selectbox(
            "Select a categorical feature",
            [c for c in df_raw.select_dtypes(include="object").columns],
        )
        fig4 = px.box(
            df_raw, x=cat_feature, y="Exam_Score",
            color=cat_feature,
            color_discrete_sequence=COLORS,
            title=f"Exam Score Distribution by {cat_feature}",
            template=PLOTLY_TEMPLATE,
        )
        fig4.update_layout(showlegend=False)
        st.plotly_chart(fig4, use_container_width=True)

        avg_score = df_raw.groupby(cat_feature)["Exam_Score"].mean().reset_index()
        avg_score.columns = [cat_feature, "Average Exam Score"]
        fig5 = px.bar(
            avg_score, x=cat_feature, y="Average Exam Score",
            color=cat_feature, color_discrete_sequence=COLORS,
            title=f"Average Exam Score by {cat_feature}",
            template=PLOTLY_TEMPLATE,
            text_auto=".1f",
        )
        fig5.update_traces(textposition="outside")
        fig5.update_layout(showlegend=False)
        st.plotly_chart(fig5, use_container_width=True)

    # Tab 4 – Feature vs Score scatter
    with tab4:
        st.subheader("Numeric Feature vs Exam Score")
        num_cols = df_raw.select_dtypes(include="number").columns.tolist()
        num_cols = [c for c in num_cols if c != "Exam_Score"]
        x_feat = st.selectbox("Select a numeric feature", num_cols)
        fig6 = px.scatter(
            df_raw, x=x_feat, y="Exam_Score",
            opacity=0.45, trendline="ols",
            color_discrete_sequence=["#10b981"],
            trendline_color_override="#ef4444",
            title=f"{x_feat} vs Exam Score",
            template=PLOTLY_TEMPLATE,
        )
        fig6.update_traces(marker=dict(size=5))
        st.plotly_chart(fig6, use_container_width=True)

# ═══════════════════════════════════════════════════════════════════════════════
# PAGE 3 – MODEL TRAINING
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "🤖 Prediction":
    st.title("🤖 Model Training & Evaluation")

    col_a, col_b = st.columns(2)
    with col_a:
        model_choice = st.selectbox(
            "Choose a model",
            ["Random Forest", "Gradient Boosting", "Linear Regression"],
        )
    with col_b:
        test_size = st.slider("Test set size (%)", 10, 40, 20, step=5)

    if st.button("🚀 Train Model", type="primary"):
        with st.spinner("Training… please wait ⏳"):
            X = df_enc.drop("Exam_Score", axis=1)
            y = df_enc["Exam_Score"]
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size / 100, random_state=42
            )

            if model_choice == "Random Forest":
                model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
            elif model_choice == "Gradient Boosting":
                model = GradientBoostingRegressor(n_estimators=100, random_state=42)
            else:
                model = LinearRegression()

            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)

            mae  = mean_absolute_error(y_test, y_pred)
            rmse = np.sqrt(mean_squared_error(y_test, y_pred))
            r2   = r2_score(y_test, y_pred)

        st.success("✅ Model trained successfully!")

        c1, c2, c3 = st.columns(3)
        c1.metric("📉 MAE",  f"{mae:.2f}")
        c2.metric("📐 RMSE", f"{rmse:.2f}")
        c3.metric("🎯 R² Score", f"{r2:.4f}")

        st.divider()
        col1, col2 = st.columns(2)

        with col1:
            st.subheader("🔵 Actual vs Predicted")
            fig, ax = plt.subplots(facecolor="#0f172a")
            ax.set_facecolor("#0f172a")
            ax.scatter(y_test, y_pred, alpha=0.45, color="#6366f1", edgecolors="none", s=18)
            ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()],
                    color="#f59e0b", linewidth=2, linestyle="--")
            ax.set_xlabel("Actual Score", color="white")
            ax.set_ylabel("Predicted Score", color="white")
            ax.set_title("Actual vs Predicted", color="white", fontsize=13)
            ax.tick_params(colors="white")
            for spine in ax.spines.values():
                spine.set_edgecolor("#334155")
            st.pyplot(fig)
            plt.close()

        with col2:
            st.subheader("🟣 Residuals Distribution")
            residuals = y_test - y_pred
            fig2, ax2 = plt.subplots(facecolor="#0f172a")
            ax2.set_facecolor("#0f172a")
            ax2.hist(residuals, bins=30,
                     color="#ec4899", edgecolor="#0f172a", alpha=0.85)
            ax2.axvline(0, color="#f59e0b", linestyle="--", linewidth=2)
            ax2.set_xlabel("Residual", color="white")
            ax2.set_ylabel("Count", color="white")
            ax2.set_title("Residuals", color="white", fontsize=13)
            ax2.tick_params(colors="white")
            for spine in ax2.spines.values():
                spine.set_edgecolor("#334155")
            st.pyplot(fig2)
            plt.close()

        if model_choice in ["Random Forest", "Gradient Boosting"]:
            st.divider()
            st.subheader("⭐ Feature Importance")
            fi = pd.Series(model.feature_importances_, index=X.columns).sort_values(ascending=False)
            fig3 = px.bar(
                fi, orientation="v", title="Feature Importance",
                labels={"value": "Importance", "index": "Feature"},
                color=fi.values,
                color_continuous_scale="Turbo",
                template=PLOTLY_TEMPLATE,
            )
            fig3.update_layout(title_font_color="#6366f1", coloraxis_showscale=False)
            st.plotly_chart(fig3, use_container_width=True)

# ═══════════════════════════════════════════════════════════════════════════════
# PAGE 4 – PREDICT MY SCORE
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "🔍 Predict My Score":
    st.title("🔍 Predict a Student's Exam Score")
    st.info("Fill in the student details below and click **Predict** to get an estimated exam score.")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("##### 📚 Academic Factors")
        hours_studied       = st.slider("Hours Studied (per week)", 1, 44, 20)
        attendance          = st.slider("Attendance (%)", 60, 100, 80)
        sleep_hours         = st.slider("Sleep Hours (per night)", 4, 10, 7)
        previous_scores     = st.slider("Previous Scores", 50, 100, 75)
        tutoring_sessions   = st.slider("Tutoring Sessions (per month)", 0, 8, 1)
        physical_activity   = st.slider("Physical Activity (hrs/week)", 0, 6, 3)

    with col2:
        st.markdown("##### 🏠 Personal & Family")
        parental_involvement    = st.selectbox("Parental Involvement", ["Low", "Medium", "High"])
        access_to_resources     = st.selectbox("Access to Resources", ["Low", "Medium", "High"])
        extracurricular         = st.selectbox("Extracurricular Activities", ["Yes", "No"])
        motivation_level        = st.selectbox("Motivation Level", ["Low", "Medium", "High"])
        internet_access         = st.selectbox("Internet Access", ["Yes", "No"])
        family_income           = st.selectbox("Family Income", ["Low", "Medium", "High"])

    with col3:
        st.markdown("##### 🏫 School & Environment")
        teacher_quality         = st.selectbox("Teacher Quality", ["Low", "Medium", "High"])
        school_type             = st.selectbox("School Type", ["Public", "Private"])
        peer_influence          = st.selectbox("Peer Influence", ["Positive", "Neutral", "Negative"])
        learning_disabilities   = st.selectbox("Learning Disabilities", ["Yes", "No"])
        parental_education      = st.selectbox("Parental Education Level", ["High School", "College", "Postgraduate"])
        distance_from_home      = st.selectbox("Distance from Home", ["Near", "Moderate", "Far"])
        gender                  = st.selectbox("Gender", ["Male", "Female"])

    st.divider()
    if st.button("🎯 Predict Score", type="primary"):
        input_dict = {
            "Hours_Studied": hours_studied,
            "Attendance": attendance,
            "Parental_Involvement": parental_involvement,
            "Access_to_Resources": access_to_resources,
            "Extracurricular_Activities": extracurricular,
            "Sleep_Hours": sleep_hours,
            "Previous_Scores": previous_scores,
            "Motivation_Level": motivation_level,
            "Internet_Access": internet_access,
            "Tutoring_Sessions": tutoring_sessions,
            "Family_Income": family_income,
            "Teacher_Quality": teacher_quality,
            "School_Type": school_type,
            "Peer_Influence": peer_influence,
            "Physical_Activity": physical_activity,
            "Learning_Disabilities": learning_disabilities,
            "Parental_Education_Level": parental_education,
            "Distance_from_Home": distance_from_home,
            "Gender": gender,
        }
        input_df = pd.DataFrame([input_dict])

        df_combined = pd.concat([df_raw.drop("Exam_Score", axis=1), input_df], ignore_index=True)
        for col in df_combined.select_dtypes(include="object").columns:
            df_combined[col] = LabelEncoder().fit_transform(df_combined[col].astype(str))

        input_encoded = df_combined.iloc[[-1]]

        X = df_enc.drop("Exam_Score", axis=1)
        y = df_enc["Exam_Score"]
        model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
        model.fit(X, y)

        predicted_score = model.predict(input_encoded)[0]

        # Grade logic
        if predicted_score >= 80:
            grade = "A — Excellent 🌟"
            badge_color = "#10b981"
            progress_color = "green"
        elif predicted_score >= 70:
            grade = "B — Good 👍"
            badge_color = "#3b82f6"
            progress_color = "blue"
        elif predicted_score >= 60:
            grade = "C — Average 📘"
            badge_color = "#f59e0b"
            progress_color = "orange"
        else:
            grade = "D — Needs Improvement 📉"
            badge_color = "#ef4444"
            progress_color = "red"

        st.subheader("🎓 Prediction Result")
        r1, r2, r3 = st.columns(3)
        r1.metric("🏆 Predicted Score", f"{predicted_score:.1f} / 100")
        r2.markdown(
            f"<div style='background:{badge_color};border-radius:10px;padding:14px 18px;"
            f"text-align:center;color:white;font-size:1.1rem;font-weight:700;margin-top:8px'>"
            f"{grade}</div>",
            unsafe_allow_html=True,
        )
        r3.markdown(
            f"<div style='background:#1e293b;border-radius:10px;padding:14px 18px;"
            f"text-align:center;color:#94a3b8;font-size:0.9rem;margin-top:8px'>"
            f"Percentile rank: <b style='color:white'>{predicted_score:.0f}th</b></div>",
            unsafe_allow_html=True,
        )

        st.progress(int(min(predicted_score, 100)))
        st.caption(f"Score of **{predicted_score:.1f}** places the student in the **{grade}** bracket.")
