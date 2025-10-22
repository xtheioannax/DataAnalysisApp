import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from services.data import load_student_data
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

st.set_page_config(page_title="Student Data Dashboard", layout="wide")
st.page_link("pages/Home_Page.py", label="↩︎ Back to Home")
st.title("Student Data Dashboard")

# Load data 
student_data = load_student_data()

# Sidebar filters
st.sidebar.header("Filters")


age_min, age_max = int(student_data["age"].min()), int(student_data["age"].max())
majors = sorted(student_data["major"].dropna().unique())
regions = sorted(student_data["region"].dropna().unique())
score_cols = ["math_score", "science_score", "english_score"]
score_filters = {}
for col in score_cols:
    min_val, max_val = float(student_data[col].min()), float(student_data[col].max())
    score_filters[col] = st.sidebar.slider(
        f"{col.replace('_', ' ').title()}",
        min_value=min_val,
        max_value=max_val,
        value=(min_val, max_val),
        step=1.0
    )


sel_age = st.sidebar.slider(
    "Age Range",
    min_value=age_min,
    max_value=age_max,
    value=(age_min, age_max)
)

sel_major  = st.sidebar.multiselect("Select one or more majors", majors, default=majors)

sel_region = st.sidebar.multiselect("Select one or more regions", regions, default=regions)



filtered = student_data.copy()

filtered = filtered[
    (filtered["age"].between(*sel_age)) &
    (filtered["major"].isin(sel_major)) &
    (filtered["region"].isin(sel_region)) &
    (filtered["math_score"].between(*score_filters["math_score"])) &
    (filtered["science_score"].between(*score_filters["science_score"])) &
    (filtered["english_score"].between(*score_filters["english_score"]))
]

st.caption(f"Filtered rows: {len(filtered)}")

if filtered.empty:
    st.warning("No data matches the selected filters. Try broadening your selection.")
    st.stop()

# Metrics
st.subheader("Metrics")

total_students = len(filtered)
avg_gpa = filtered["gpa"].mean()
avg_study_hours = filtered["study_hours"].mean()
avg_math = filtered["math_score"].mean()
avg_sci = filtered["science_score"].mean()
avg_eng = filtered["english_score"].mean()

col1, col2, col3, col4, col5, col6 = st.columns(6)
col1.metric("Total Students", f"{total_students:,}")
col2.metric("Avg GPA", f"{avg_gpa:.2f}")
col3.metric("Avg Study Hours", f"{avg_study_hours:.1f}")
col4.metric("Avg Math", f"{avg_math:.1f}")
col5.metric("Avg Science", f"{avg_sci:.1f}")
col6.metric("Avg English", f"{avg_eng:.1f}")

# Diagrams

with st.expander("Statistical Distributions"):
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**GPA Distribution**")
        fig, ax = plt.subplots()
        ax.hist(filtered["gpa"], bins=20, color="skyblue", edgecolor="black")
        ax.set_xlabel("GPA")
        ax.set_ylabel("Frequency")
        ax.grid(True, linestyle="--", alpha=0.6)
        st.pyplot(fig)

    with col2:
        st.markdown("**Study Hours Distribution**")
        fig, ax = plt.subplots()
        ax.hist(filtered["study_hours"], bins=20, color="lightgreen", edgecolor="black")
        ax.set_xlabel("Study Hours")
        ax.set_ylabel("Frequency")
        ax.grid(True, linestyle="--", alpha=0.6)
        st.pyplot(fig)

    with col1:
        # Boxplots for score distributions
        st.markdown("**Subject Score Distributions**")
        fig, ax = plt.subplots()
        sns.boxplot(data=filtered[["math_score", "science_score", "english_score"]], ax=ax, palette="Set2")
        ax.set_ylabel("Score")
        st.pyplot(fig)

with st.expander("Correlation Analysis"):
    numeric_cols = ["math_score", "science_score", "english_score", "study_hours", "gpa", "age"]
    corr = filtered[numeric_cols].corr()

    col1, col2 = st.columns(2)
    with col1:
        fig, ax = plt.subplots(figsize=(6, 4))
        sns.heatmap(corr, annot=True, cmap="coolwarm", center=0, ax=ax)
        ax.set_title("Correlation Matrix (Filtered Data)")
        st.pyplot(fig)
    with col2:
        if "gpa" in corr.columns:
            top_corr = corr["gpa"].sort_values(ascending=False).drop("gpa")
            st.write("**Features most correlated with GPA:**")
            st.bar_chart(top_corr)

with st.expander("Regression Modeling: Study Hours → GPA"):
    X = filtered[["study_hours"]]
    y = filtered["gpa"]

    # Ensure there’s enough data to fit the model
    if len(filtered) > 1 and X["study_hours"].nunique() > 1:
        model = LinearRegression()
        model.fit(X, y)
        y_pred = model.predict(X)
        r2 = r2_score(y, y_pred)

        col1, col2 = st.columns(2)
        with col1:
            fig, ax = plt.subplots()
            ax.scatter(X, y, color="blue", alpha=0.6, label="Observed")
            ax.plot(X, y_pred, color="red", label="Regression Line")
            ax.set_xlabel("Study Hours")
            ax.set_ylabel("GPA")
            ax.legend()
            ax.grid(True, linestyle="--", alpha=0.6)
            st.pyplot(fig)
        with col2:
            st.markdown(f"**R² = {r2:.3f}** — goodness of fit (how much GPA variance study hours explain).")
    else:
        st.info("Not enough variation in filtered data to fit a regression model.")

with st.expander("Interactive regression"):
    mode = st.radio("Model type", ["Single predictor → GPA", "Multiple predictors → GPA"], horizontal=True)

    # Numeric candidates for predictors
    numeric_candidates = ["age", "study_hours", "math_score", "science_score", "english_score"]

    if mode == "Single predictor → GPA":
        pred = st.selectbox("Predictor (X)", numeric_candidates, index=numeric_candidates.index("study_hours") if "study_hours" in numeric_candidates else 0)

        # Guard: need >1 unique X & Y values
        if filtered[pred].nunique() < 2 or filtered["gpa"].nunique() < 2:
            st.info("Not enough variation in the selected data to fit a regression line.")
        else:
            X = filtered[[pred]].values
            y = filtered["gpa"].values

            # Train/test split for honest R²
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            model = LinearRegression().fit(X_train, y_train)

            y_hat = model.predict(X)
            r2_tr = r2_score(y_train, model.predict(X_train))
            r2_te = r2_score(y_test, model.predict(X_test))

            slope = float(model.coef_[0])
            intercept = float(model.intercept_)

            col1, col2 = st.columns(2)
            with col1:
                # Plot scatter + fitted line
                fig, ax = plt.subplots()
                ax.scatter(X.ravel(), y, alpha=0.6, label="Observed")
                # draw line across the X range
                x_line = np.linspace(X.min(), X.max(), 50).reshape(-1, 1)
                ax.plot(x_line, model.predict(x_line), label="Fitted line", linewidth=2)
                ax.set_xlabel(pred.replace("_", " ").title())
                ax.set_ylabel("GPA")
                ax.grid(True, linestyle="--", alpha=0.6)
                ax.legend()
                st.pyplot(fig)

            with col2:
                # Residuals vs fitted
                resid = y - y_hat
                fig, ax = plt.subplots()
                ax.scatter(y_hat, resid, alpha=0.6)
                ax.axhline(0, linestyle="--")
                ax.set_xlabel("Fitted GPA")
                ax.set_ylabel("Residuals")
                ax.set_title("Residuals vs Fitted")
                ax.grid(True, linestyle="--", alpha=0.6)
                st.pyplot(fig)

            with col1:
                st.markdown(
                    f"""
        **Model:** GPA = {intercept:.3f} + {slope:.3f} × {pred}  
        **R² (train):** {r2_tr:.3f} &nbsp;&nbsp; **R² (test):** {r2_te:.3f}
        """
                )

    else:  # Multiple predictors
        preds = st.multiselect(
            "Predictors (X)",
            numeric_candidates,
            default=["study_hours", "math_score"]
        )

        if len(preds) < 1:
            st.info("Select at least one predictor to fit a multiple regression.")
        elif any(filtered[p].nunique() < 2 for p in preds) or filtered["gpa"].nunique() < 2:
            st.info("Not enough variation in one or more variables to fit the model.")
        else:
            X = filtered[preds].values
            y = filtered["gpa"].values

            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            model = LinearRegression().fit(X_train, y_train)

            y_hat = model.predict(X)
            r2_tr = r2_score(y_train, model.predict(X_train))
            r2_te = r2_score(y_test, model.predict(X_test))

            # Coefficients table
            coefs = pd.DataFrame({
                "Predictor": preds,
                "Coefficient": model.coef_
            }).sort_values("Predictor")
            col1, col2 = st.columns(2)
            with col2:
                st.markdown(f"**Intercept:** {model.intercept_:.3f}  |  **R² (train):** {r2_tr:.3f}  |  **R² (test):** {r2_te:.3f}")
                st.dataframe(coefs.sort_values("Coefficient", ascending=False), use_container_width=True)

            with col1:
                # Residuals vs fitted
                fig, ax = plt.subplots()
                ax.scatter(y_hat, y - y_hat, alpha=0.6)
                ax.axhline(0, linestyle="--")
                ax.set_xlabel("Fitted GPA")
                ax.set_ylabel("Residuals")
                ax.set_title("Residuals vs Fitted")
                ax.grid(True, linestyle="--", alpha=0.6)
                st.pyplot(fig)