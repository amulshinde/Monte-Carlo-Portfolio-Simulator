import streamlit as st
import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt

st.set_page_config(page_title="Monte Carlo Portfolio Simulator", layout="wide")

st.title("Monte Carlo Portfolio Simulator")

# =========================
# Sidebar Inputs
# =========================

st.sidebar.header("Simulation Inputs")

tickers = st.sidebar.text_input(
    "Stock Tickers (comma separated)",
    "AAPL,MSFT,GOOGL"
)

weights = st.sidebar.text_input(
    "Portfolio Weights (comma separated)",
    "0.4,0.4,0.2"
)

num_simulations = st.sidebar.slider(
    "Number of Simulations",
    1000,
    20000,
    1000
)

days = st.sidebar.slider(
    "Simulation Period (days)",
    30,
    252,
    252
)

initial_investment = st.sidebar.number_input(
    "Initial Portfolio Value",
    value=10000
)

tab1, tab2, tab3, tab4 = st.tabs(
    ["Portfolio Paths", "Return Distribution", "Statistics", "About"]
)

# About tab always visible
with tab4:

    st.markdown("""
                ## About This App

                The **Monte Carlo Portfolio Simulator** helps investors explore how their portfolio might perform under many possible future market conditions. Instead of predicting a single outcome, the app generates thousands of simulated scenarios using historical market data and statistical modeling.

                By running these simulations, investors can better understand the potential range of portfolio outcomes and evaluate both **expected returns and downside risks** before making investment decisions.

                ### What This Tool Helps You Do

                - **Visualize potential future portfolio paths** across thousands of simulated market scenarios  
                - **Estimate expected portfolio returns** based on historical data patterns  
                - **Measure the probability of loss** and evaluate downside risk  
                - **Understand the likelihood of portfolio growth or survival over time**

                ### Why Monte Carlo Simulation Matters

                Financial markets are inherently uncertain. Monte Carlo simulation provides a practical way to model that uncertainty by generating a large number of possible future outcomes. Rather than relying on a single forecast, this approach helps investors see the **distribution of possible results**, including best-case, typical, and worst-case scenarios.

                ### Professional Use in Finance

                Monte Carlo simulation is widely used by **investment banks, hedge funds, and quantitative analysts** for portfolio analysis, risk management, and financial forecasting. By evaluating thousands of potential market scenarios, professionals can better understand portfolio risk and make more informed investment decisions.""")

run_button = st.sidebar.button("Run Simulation")

# =========================
# Run Simulation
# =========================

# Run simulation when button pressed
if run_button:

    with st.spinner("Running Monte Carlo simulations..."):

        tickers = [t.strip() for t in tickers.split(",")]
        weights = np.array([float(w) for w in weights.split(",")])
        weights = weights / np.sum(weights)

        data = yf.download(tickers, start="2020-01-01")["Close"]

        returns = data.pct_change().dropna()

        mean_returns = returns.mean()
        cov_matrix = returns.cov()

        results = np.zeros((days, num_simulations))

        for i in range(num_simulations):

            portfolio_value = initial_investment
            path = []

            for d in range(days):

                random_returns = np.random.multivariate_normal(
                    mean_returns,
                    cov_matrix
                )

                portfolio_return = np.dot(weights, random_returns)

                portfolio_value = portfolio_value * (1 + portfolio_return)

                path.append(portfolio_value)

            results[:, i] = path

    st.success("Simulation completed successfully!")

    final_values = results[-1]

    expected_value = np.mean(final_values)
    probability_of_loss = np.sum(final_values < initial_investment) / len(final_values)
    survival_probability = 1 - probability_of_loss

    # Populate tabs
    with tab1:

        st.subheader("Simulated Portfolio Paths")

        fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot(results[:, :100], alpha=0.3)
        ax.set_xlabel("Days")
        ax.set_ylabel("Portfolio Value")

        st.pyplot(fig)

    with tab2:

        st.subheader("Final Portfolio Value Distribution")

        fig2, ax2 = plt.subplots(figsize=(10, 4))
        ax2.hist(final_values, bins=50)

        st.pyplot(fig2)

    with tab3:

        st.subheader("Simulation Statistics")

        col1, col2, col3 = st.columns(3)

        col1.metric("Expected Portfolio Value", f"${expected_value:,.2f}")
        col2.metric("Probability of Loss", f"{probability_of_loss*100:.2f}%")
        col3.metric("Portfolio Survival Probability", f"{survival_probability*100:.2f}%")

        st.divider()

        st.subheader("Risk Metrics")

        var_95 = np.percentile(final_values, 5)

        p5 = np.percentile(final_values, 5)
        p50 = np.percentile(final_values, 50)
        p95 = np.percentile(final_values, 95)

        col4, col5, col6 = st.columns(3)

        col4.metric("Value at Risk (95%)", f"${var_95:,.2f}")
        col5.metric("Median Outcome", f"${p50:,.2f}")
        col6.metric("Best Likely Outcome (95%)", f"${p95:,.2f}")


st.markdown(
    """
    <hr style="margin-top:40px;"/>
    <div style="text-align:center; color:gray; font-size:13px;">
    © 2026 FrontierX · Developed by Amul Shinde. All rights reserved.
    </div>
    """,
    unsafe_allow_html=True
)