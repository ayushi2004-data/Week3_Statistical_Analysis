import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns

# Load the cleaned Titanic dataset
df = pd.read_csv("titanic_cleaned.csv")

print("Dataset loaded successfully!")
print("Dataset shape:", df.shape)

print("\nFirst 5 rows:")
print(df.head())

# Chi-Square Test of Independence
contingency_table = pd.crosstab(df["sex"], df["survived"])

print("\nContingency Table:")
print(contingency_table)

chi2, p_value, degrees_of_freedom, expected = stats.chi2_contingency(contingency_table)

print("\nChi-Square Test Results:")
print("Chi-Square Statistic:", chi2)
print("P-value:", p_value)
print("Degrees of Freedom:", degrees_of_freedom)

# Welch's Independent Two-Sample T-Test
survivor_fare = df[df["survived"] == 1]["fare"]
non_survivor_fare = df[df["survived"] == 0]["fare"]

t_statistic, t_p_value = stats.ttest_ind(
    survivor_fare,
    non_survivor_fare,
    equal_var=False
)

print("\nWelch's Independent T-Test Results:")
print("Survivors' Mean Fare:", survivor_fare.mean())
print("Non-Survivors' Mean Fare:", non_survivor_fare.mean())
print("T-Statistic:", t_statistic)
print("P-value:", t_p_value)

# 95% Confidence Interval for Difference in Mean Fares

mean_difference = survivor_fare.mean() - non_survivor_fare.mean()

se_difference = (
    (survivor_fare.var(ddof=1) / len(survivor_fare)) +
    (non_survivor_fare.var(ddof=1) / len(non_survivor_fare))
) ** 0.5

degrees_of_freedom_ci = (
    (survivor_fare.var(ddof=1) / len(survivor_fare) +
     non_survivor_fare.var(ddof=1) / len(non_survivor_fare)) ** 2
    /
    (
        (survivor_fare.var(ddof=1) / len(survivor_fare)) ** 2
        / (len(survivor_fare) - 1)
        +
        (non_survivor_fare.var(ddof=1) / len(non_survivor_fare)) ** 2
        / (len(non_survivor_fare) - 1)
    )
)

critical_value = stats.t.ppf(0.975, degrees_of_freedom_ci)

margin_of_error = critical_value * se_difference

ci_lower = mean_difference - margin_of_error
ci_upper = mean_difference + margin_of_error

print("\n95% Confidence Interval for Difference in Mean Fares:")
print("Mean Difference:", mean_difference)
print("Lower Bound:", ci_lower)
print("Upper Bound:", ci_upper)

# Visualization 1: Survival by Gender

plt.figure(figsize=(8, 6))
sns.countplot(data=df, x="sex", hue="survived")

plt.title("Survival Distribution by Gender")
plt.xlabel("Gender")
plt.ylabel("Number of Passengers")
plt.legend(title="Survived", labels=["No", "Yes"])

plt.tight_layout()
plt.savefig("gender_survival.png", dpi=300)
plt.show()

# One-Way ANOVA Test: Fare across Passenger Classes

first_class_fare = df[df["pclass"] == 1]["fare"]
second_class_fare = df[df["pclass"] == 2]["fare"]
third_class_fare = df[df["pclass"] == 3]["fare"]

f_statistic, anova_p_value = stats.f_oneway(
    first_class_fare,
    second_class_fare,
    third_class_fare
)

print("\nOne-Way ANOVA Results:")
print("1st Class Mean Fare:", first_class_fare.mean())
print("2nd Class Mean Fare:", second_class_fare.mean())
print("3rd Class Mean Fare:", third_class_fare.mean())
print("F-Statistic:", f_statistic)
print("P-value:", anova_p_value)

# Visualization 2: Fare Distribution by Survival Status

plt.figure(figsize=(8, 6))
sns.boxplot(data=df, x="survived", y="fare")

plt.title("Fare Distribution by Survival Status")
plt.xlabel("Survival Status")
plt.ylabel("Fare")
plt.xticks([0, 1], ["Did Not Survive", "Survived"])

plt.tight_layout()
plt.savefig("fare_survival_boxplot.png", dpi=300)
plt.show()

# Visualization 3: Fare Distribution by Passenger Class

plt.figure(figsize=(8, 6))
sns.boxplot(data=df, x="pclass", y="fare")

plt.title("Fare Distribution by Passenger Class")
plt.xlabel("Passenger Class")
plt.ylabel("Fare")
plt.xticks([0, 1, 2], ["1st Class", "2nd Class", "3rd Class"])

plt.tight_layout()
plt.savefig("fare_class_boxplot.png", dpi=300)
plt.show()

# Tukey HSD Post-Hoc Test

from statsmodels.stats.multicomp import pairwise_tukeyhsd

tukey_result = pairwise_tukeyhsd(
    endog=df["fare"],
    groups=df["pclass"],
    alpha=0.05
)

print("\nTukey HSD Post-Hoc Test Results:")
print(tukey_result)

# Visualization 4: Survival Rate by Gender

gender_survival_rate = df.groupby("sex")["survived"].mean() * 100

plt.figure(figsize=(8, 6))
sns.barplot(
    x=gender_survival_rate.index,
    y=gender_survival_rate.values
)

plt.title("Survival Rate by Gender")
plt.xlabel("Gender")
plt.ylabel("Survival Rate (%)")

for i, value in enumerate(gender_survival_rate.values):
    plt.text(i, value + 2, f"{value:.2f}%", ha="center")

plt.ylim(0, 100)
plt.tight_layout()
plt.savefig("gender_survival_rate.png", dpi=300)
plt.show()

# Visualization 5: Fare Distribution by Survival Status

plt.figure(figsize=(8, 6))

sns.histplot(
    data=df,
    x="fare",
    hue="survived",
    kde=True,
    bins=30,
    element="step"
)

plt.title("Fare Distribution by Survival Status")
plt.xlabel("Fare")
plt.ylabel("Number of Passengers")
plt.legend(title="Survived", labels=["Yes", "No"])

plt.tight_layout()
plt.savefig("fare_distribution_survival.png", dpi=300)
plt.show()