import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# --- Simulated hospital dataset ---
np.random.seed(42)
n = 500

data = pd.DataFrame({
    "Hospital_ID": np.random.randint(1, 11, n),
    "Region": np.random.choice(["Urban", "Rural"], n),
    "Mother_Age": np.random.randint(16, 45, n),
    "Prenatal_Visits": np.random.randint(0, 10, n),
    "Birth_Weight": np.random.normal(3100, 500, n).astype(int),
    "Gestation_Weeks": np.random.randint(28, 41, n),
    "Delivery_Type": np.random.choice(["Normal", "C-section"], n),
    "Infant_Survived": np.random.choice([0, 1], n, p=[0.05, 0.95]),
    "Year": np.random.choice([2021, 2022, 2023, 2024], n)
})

# --- Calculate Infant Mortality Rate per Hospital ---
imr = (
    data.groupby("Hospital_ID")["Infant_Survived"]
    .apply(lambda x: (1 - x.mean()) * 1000)
    .reset_index(name="Infant_Mortality_Rate")
)

print("\nInfant Mortality Rate per Hospital (per 1,000 births):\n")
print(imr.head())

# --- Visualize regional differences ---
sns.barplot(x="Region", y="Infant_Survived", data=data, estimator=lambda x: 1 - np.mean(x))
plt.title("Infant Mortality Proportion by Region")
plt.ylabel("Mortality Rate")
plt.show()

# --- Correlation heatmap for numerical features ---
plt.figure(figsize=(8,6))
sns.heatmap(data.corr(numeric_only=True), annot=True, cmap="coolwarm")
plt.title("Correlation Heatmap of Hospital Data")
plt.show()

# --- Insights ---
print("\nAverage Infant Mortality Rate (per 1,000 births):", round((1 - data['Infant_Survived'].mean()) * 1000, 2))
