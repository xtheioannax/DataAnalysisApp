import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from services.data import load_insurance_data
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

st.set_page_config(page_title="Insurance Data Dashboard", layout="wide")
st.page_link("pages/Home_Page.py", label="↩︎ Back to Home")
st.title("Insurance Data Dashboard")

insurance_data = load_insurance_data()

# ---------- Guards ----------
needed = {"age","sex","bmi","children","smoker","region","charges"}
missing = needed - set(insurance_data.columns.str.lower())
if missing:
    insurance_data.columns = [c.lower() for c in insurance_data.columns]
    missing = needed - set(insurance_data.columns)
if missing:
    st.error(f"Missing expected column(s): {', '.join(sorted(missing))}")
    st.stop()

# Sidebar Filters 
st.sidebar.header("Filters")

age_min, age_max = int(insurance_data["age"].min()), int(insurance_data["age"].max())
bmi_min, bmi_max = float(insurance_data["bmi"].min()), float(insurance_data["bmi"].max())
ch_min, ch_max   = float(insurance_data["charges"].min()), float(insurance_data["charges"].max())

sel_age = st.sidebar.slider("Age range", age_min, age_max, (age_min, age_max))
sel_bmi = st.sidebar.slider("BMI range", float(bmi_min), float(bmi_max), (float(bmi_min), float(bmi_max)))
sel_charges = st.sidebar.slider("Charges range", float(ch_min), float(ch_max), (float(ch_min), float(ch_max)))

sex_vals = sorted(insurance_data["sex"].dropna().unique().tolist())
smoker_vals = sorted(insurance_data["smoker"].dropna().unique().tolist())
region_vals = sorted(insurance_data["region"].dropna().unique().tolist())

sel_sex = st.sidebar.multiselect("Sex", sex_vals, default=sex_vals)
sel_smoker = st.sidebar.multiselect("Smoker", smoker_vals, default=smoker_vals)
sel_region = st.sidebar.multiselect("Region", region_vals, default=region_vals)

# Children: let users pick any distinct counts
children_vals = sorted(insurance_data["children"].dropna().unique().tolist())
sel_children = st.sidebar.multiselect("Children", children_vals, default=children_vals)

# Apply filters 
filtered = insurance_data[
    (insurance_data["age"].between(sel_age[0], sel_age[1])) &
    (insurance_data["bmi"].between(sel_bmi[0], sel_bmi[1])) &
    (insurance_data["charges"].between(sel_charges[0], sel_charges[1])) &
    (insurance_data["sex"].isin(sel_sex)) &
    (insurance_data["smoker"].isin(sel_smoker)) &
    (insurance_data["region"].isin(sel_region)) &
    (insurance_data["children"].isin(sel_children))
].copy()

st.caption(f"Filtered rows: {len(filtered)}")
if filtered.empty:
    st.warning("No data matches your filters. Adjust in the sidebar.")
    st.stop()

st.subheader("Metrics")
c1, c2, c3, c4 = st.columns(4)
c1.metric("Avg. Charges", f"${filtered['charges'].mean():,.0f}")
c2.metric("Median Charges", f"${filtered['charges'].median():,.0f}")
c3.metric("Avg. BMI", f"{filtered['bmi'].mean():.1f}")
smoker_pct = (filtered["smoker"].eq("yes").mean()*100) if "yes" in filtered["smoker"].unique().tolist() else (filtered["smoker"].astype(str).str.lower().eq("yes").mean()*100)
c4.metric("Smokers", f"{smoker_pct:.1f}%")


# Charts 
with st.expander("Charts"):
    col1, col2 = st.columns(2)
    with col1:
        # 1) Charges distribution
        st.subheader("Charges Distribution")
        fig, ax = plt.subplots()
        ax.hist(filtered["charges"], bins=30)
        ax.set_xlabel("Charges"); ax.set_ylabel("Count")
        st.pyplot(fig)

    with col2:
        # 2) Average charges by region
        st.subheader("Average Charges by Region")
        fig, ax = plt.subplots()
        filtered.groupby("region")["charges"].mean().sort_values().plot(kind="bar", ax=ax)
        ax.set_xlabel("Region"); ax.set_ylabel("Avg Charges")
        st.pyplot(fig)

    with col1:
        # 3) Charges vs BMI (colored by smoker where possible)
        st.subheader("Charges vs BMI")
        fig, ax = plt.subplots()
        if set(filtered["smoker"].unique()) >= {"yes", "no"}:
            df_yes = filtered[filtered["smoker"] == "yes"]
            df_no  = filtered[filtered["smoker"] == "no"]
            ax.scatter(df_no["bmi"], df_no["charges"], label="Non-smoker")
            ax.scatter(df_yes["bmi"], df_yes["charges"], label="Smoker")
            ax.legend()
        else:
            ax.scatter(filtered["bmi"], filtered["charges"])
        ax.set_xlabel("BMI"); ax.set_ylabel("Charges")
        st.pyplot(fig)
    with col2:
        # 4) Charges by children (box-ish via describe or bar)
        st.subheader("Average Charges by Number of Children")
        fig, ax = plt.subplots()
        filtered.groupby("children")["charges"].mean().plot(kind="bar", ax=ax)
        ax.set_xlabel("Children"); ax.set_ylabel("Avg Charges")
        st.pyplot(fig)

st.header("Statistical Analysis")
with st.expander("Statistical Distributions"):
    col1, col2 = st.columns(2)

    with col1:
        st.caption("Charges distribution")
        fig, ax = plt.subplots()
        sns.histplot(filtered["charges"], bins=30, kde=True, ax=ax)
        ax.set_xlabel("Charges")
        ax.set_ylabel("Count")
        st.pyplot(fig)

    with col2:
        st.caption("BMI distribution")
        fig, ax = plt.subplots()
        sns.histplot(filtered["bmi"], bins=30, kde=True, color="orange", ax=ax)
        ax.set_xlabel("BMI")
        ax.set_ylabel("Count")
        st.pyplot(fig)

with st.expander("Correlation Analysis"):
    col1, col2 = st.columns(2)
    with col1:
        numeric_cols = filtered.select_dtypes(include=[np.number])
        corr = numeric_cols.corr()

        fig, ax = plt.subplots()
        sns.heatmap(corr, annot=True, cmap="coolwarm", center=0, ax=ax)
        ax.set_title("Correlation Matrix (Numeric Features)")
        st.pyplot(fig)
    with col2:
        if "charges" in corr.columns:
            top_corr = corr["charges"].sort_values(ascending=False).drop("charges")
            st.write("**Features most correlated with charges:**")
            st.bar_chart(top_corr)

with st.expander("Regression Modeling: Predicting Charges"):
    encoded = pd.get_dummies(
        filtered[["age", "bmi", "children", "smoker", "sex", "region", "charges"]],
        drop_first=True,
    )

    X = encoded.drop(columns="charges")
    y = encoded["charges"]

    # Train a simple linear regression model
    model = LinearRegression()
    model.fit(X, y)
    y_pred = model.predict(X)

    # Evaluate fit (R²)
    r2 = model.score(X, y)
    st.metric("Model R²", f"{r2:.3f}")
    col1, col2 = st.columns(2)
    with col1:
        # Scatter: predicted vs actual
        fig, ax = plt.subplots()
        ax.scatter(y, y_pred, alpha=0.5)
        ax.plot([y.min(), y.max()], [y.min(), y.max()], "r--")
        ax.set_xlabel("Actual Charges")
        ax.set_ylabel("Predicted Charges")
        ax.set_title("Predicted vs Actual Charges")
        st.pyplot(fig)

    with col2:
        coefs = pd.Series(model.coef_, index=X.columns).sort_values(key=abs, ascending=False)
        st.write("**Factors influencing predicted charges:**")
        st.dataframe(coefs)