import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.stats import shapiro
from scipy.stats import mannwhitneyu

# Load the data
df = pd.read_csv('/Users/jamesknaus/Development/AB_Analysis/AB_Test_Results.csv')

# Display basic information about the dataset
print(df.info())
print("\nFirst few rows of the dataset:")
print(df.head())

# Check for missing values
print("\nMissing values:")
print(df.isnull().sum())

# Display summary statistics
print("\nSummary statistics:")
print(df.describe())

# Check unique values in VARIANT_NAME column
print("\nUnique values in VARIANT_NAME column:")
print(df['VARIANT_NAME'].unique())

# Check if any users are in both control and variant groups
users_in_both = df.groupby('USER_ID')['VARIANT_NAME'].nunique()
users_in_both = users_in_both[users_in_both > 1]
print(f"\nNumber of users in both groups: {len(users_in_both)}")

# Remove users who are in both groups
df = df[~df['USER_ID'].isin(users_in_both.index)]

# Aggregate data by user
df_agg = df.groupby(['USER_ID', 'VARIANT_NAME'])['REVENUE'].sum().reset_index()

print("\nAfter cleaning and aggregation:")
print(df_agg.head())
print(df_agg.info())

control_users = df_agg[df_agg['VARIANT_NAME'] == 'control']['USER_ID'].nunique()
variant_users = df_agg[df_agg['VARIANT_NAME'] == 'variant']['USER_ID'].nunique()

# Calculate and print the percentage split
control_percentage = (control_users / total_users) * 100
variant_percentage = (variant_users / total_users) * 100
print(f"\nPercentage split:")
print(f"Control group: {control_percentage:.2f}%")
print(f"Variant group: {variant_percentage:.2f}%")

# Set the style for the plots
sns.set_style("whitegrid")

# Create a box plot
plt.figure(figsize=(10, 6))
sns.boxplot(x='VARIANT_NAME', y='REVENUE', data=df_agg)
plt.title('Distribution of Revenue by Variant')
plt.ylabel('Revenue')
plt.show()
plt.close()

# Create histograms
plt.figure(figsize=(12, 6))
sns.histplot(data=df_agg, x='REVENUE', hue='VARIANT_NAME', element='step', stat='density', common_norm=False)
plt.title('Distribution of Revenue by Variant')
plt.xlabel('Revenue')
plt.ylabel('Density')
plt.show()
plt.close()

# Create a violin plot
plt.figure(figsize=(10, 6))
sns.violinplot(x='VARIANT_NAME', y='REVENUE', data=df_agg)
plt.title('Distribution of Revenue by Variant')
plt.ylabel('Revenue')
plt.show()
plt.close()

# Calculate and print some statistics
control_mean = df_agg[df_agg['VARIANT_NAME'] == 'control']['REVENUE'].mean()
variant_mean = df_agg[df_agg['VARIANT_NAME'] == 'variant']['REVENUE'].mean()
print(f"Mean revenue for control group: {control_mean:.2f}")
print(f"Mean revenue for variant group: {variant_mean:.2f}")

control_median = df_agg[df_agg['VARIANT_NAME'] == 'control']['REVENUE'].median()
variant_median = df_agg[df_agg['VARIANT_NAME'] == 'variant']['REVENUE'].median()
print(f"Median revenue for control group: {control_median:.2f}")
print(f"Median revenue for variant group: {variant_median:.2f}")

# Sort the data by revenue in descending order and display the top 10 rows
top_revenue = df_agg.sort_values('REVENUE', ascending=False).head(10)
print("\nTop 10 revenue entries:")
print(top_revenue)

# Identify the user with the highest revenue
highest_revenue_user = top_revenue.iloc[0]
print(f"\nUser with highest revenue:")
print(f"User ID: {highest_revenue_user['USER_ID']}")
print(f"Variant: {highest_revenue_user['VARIANT_NAME']}")
print(f"Revenue: {highest_revenue_user['REVENUE']}")

# Calculate the percentage difference between the highest and second-highest revenue
if len(top_revenue) > 1:
    second_highest = top_revenue.iloc[1]['REVENUE']
    percent_diff = (highest_revenue_user['REVENUE'] - second_highest) / second_highest * 100
    print(f"\nPercentage difference between highest and second-highest revenue: {percent_diff:.2f}%")

# Remove the outlier
df_agg = df_agg[df_agg['USER_ID'] != 3342]

# Recalculate statistics
control_mean = df_agg[df_agg['VARIANT_NAME'] == 'control']['REVENUE'].mean()
variant_mean = df_agg[df_agg['VARIANT_NAME'] == 'variant']['REVENUE'].mean()
print(f"\nAfter removing outlier:")
print(f"Mean revenue for control group: {control_mean:.2f}")
print(f"Mean revenue for variant group: {variant_mean:.2f}")

control_median = df_agg[df_agg['VARIANT_NAME'] == 'control']['REVENUE'].median()
variant_median = df_agg[df_agg['VARIANT_NAME'] == 'variant']['REVENUE'].median()
print(f"Median revenue for control group: {control_median:.2f}")
# print(f"Median revenue for variant group: {variant_median:.2f}")

# Update visualizations with cleaned data
plt.figure(figsize=(10, 6))
sns.boxplot(x='VARIANT_NAME', y='REVENUE', data=df_agg)
plt.title('Distribution of Revenue by Variant (Outlier Removed)')
plt.ylabel('Revenue')
plt.savefig('revenue_boxplot_cleaned.png')
plt.close()

plt.figure(figsize=(12, 6))
sns.histplot(data=df_agg, x='REVENUE', hue='VARIANT_NAME', element='step', stat='density', common_norm=False)
plt.title('Distribution of Revenue by Variant (Outlier Removed)')
plt.xlabel('Revenue')
plt.ylabel('Density')
plt.savefig('revenue_histogram_cleaned.png')
plt.close()

# Visualize non-zero revenue
df_non_zero = df_agg[df_agg['REVENUE'] > 0]
plt.figure(figsize=(10, 6))
sns.boxplot(x='VARIANT_NAME', y='REVENUE', data=df_non_zero)
plt.title('Distribution of Non-Zero Revenue by Variant')
plt.ylabel('Revenue')
plt.savefig('non_zero_revenue_boxplot.png')
plt.close()


# Statistics for all users
all_stat = df_agg.groupby(by='VARIANT_NAME').agg({
    'USER_ID': 'nunique',
    'REVENUE': ['sum', 'mean', 'median', 'count']
})

# Calculate additional metrics
all_stat[('per_user', 'orders')] = all_stat[('REVENUE', 'count')] / all_stat[('USER_ID', 'nunique')]
all_stat[('per_user', 'revenue')] = all_stat[('REVENUE', 'sum')] / all_stat[('USER_ID', 'nunique')]

# Calculate conversion rate
paying_users = df_agg[df_agg.REVENUE > 0].groupby('VARIANT_NAME')['USER_ID'].nunique()
all_stat[('per_user', 'conversion_rate')] = paying_users / all_stat[('USER_ID', 'nunique')] * 100

print("Statistics for all users:")
print(all_stat)

# Statistics for paying users only
paid_stat = df_agg[df_agg.REVENUE > 0].groupby(by='VARIANT_NAME').agg({
    'USER_ID': 'nunique',
    'REVENUE': ['sum', 'mean', 'median', 'count']
})

# Calculate additional metrics for paying users
paid_stat[('per_user', 'orders')] = paid_stat[('REVENUE', 'count')] / paid_stat[('USER_ID', 'nunique')]
paid_stat[('per_user', 'revenue')] = paid_stat[('REVENUE', 'sum')] / paid_stat[('USER_ID', 'nunique')]

print("\nStatistics for paying users only:")
print(paid_stat)

# Create a figure with two subplots
f, axes = plt.subplots(2, figsize=(10,8))

# Distribution of revenue for all users
sns.distplot(df_agg.loc[df_agg['VARIANT_NAME'] == 'control', 'REVENUE'], ax=axes[0], label='control', kde_kws={'clip': (-10, 30)})
sns.distplot(df_agg.loc[df_agg['VARIANT_NAME'] == 'variant', 'REVENUE'], ax=axes[0], label='variant', kde_kws={'clip': (-10, 30)})
axes[0].set_title('Distribution of revenue of all users')
axes[0].set_xlim(-10, 30)

# Distribution of revenue for paying users only
sns.distplot(df_agg.loc[(df_agg['VARIANT_NAME'] == 'control') & (df_agg['REVENUE'] > 0), 'REVENUE'], ax=axes[1], label='control', kde_kws={'clip': (-10, 30)})
sns.distplot(df_agg.loc[(df_agg['VARIANT_NAME'] == 'variant') & (df_agg['REVENUE'] > 0), 'REVENUE'], ax=axes[1], label='variant', kde_kws={'clip': (-10, 30)})
axes[1].set_title('Paying user revenue distribution')
axes[1].set_xlim(-10, 30)

plt.legend()
plt.subplots_adjust(hspace=0.3)

plt.savefig('revenue_distributions_final.png')
plt.show()
plt.close()

print("Plot has been saved as 'revenue_distributions_final.png'")

# Test for normality in the variant group
stat, p_value = shapiro(df_agg.loc[df_agg.VARIANT_NAME == 'variant', 'REVENUE'])
print(f"Shapiro-Wilk test for variant group: statistic={stat:.4f}, p-value={p_value:.4f}")

# Test for normality in the control group
stat, p_value = shapiro(df_agg.loc[df_agg.VARIANT_NAME == 'control', 'REVENUE'])
print(f"Shapiro-Wilk test for control group: statistic={stat:.4f}, p-value={p_value:.4f}")

# Calculate the proportion of zero revenue values
zero_proportion = (df_agg['REVENUE'] == 0).mean()
print(f"Proportion of zero values: {zero_proportion:.2%}")

# Perform Mann-Whitney U test for all users
statistic, p_value = mannwhitneyu(
    df_agg.loc[df_agg.VARIANT_NAME == 'variant', 'REVENUE'],
    df_agg.loc[df_agg.VARIANT_NAME == 'control', 'REVENUE']
)
print(f"Mann-Whitney U test for all users: statistic={statistic:.4f}, p-value={p_value:.4f}")

# Perform Mann-Whitney U test for paying users only
statistic, p_value = mannwhitneyu(
    df_agg.loc[(df_agg.VARIANT_NAME == 'variant') & (df_agg.REVENUE > 0), 'REVENUE'],
    df_agg.loc[(df_agg.VARIANT_NAME == 'control') & (df_agg.REVENUE > 0), 'REVENUE']
)
print(f"Mann-Whitney U test for paying users: statistic={statistic:.4f}, p-value={p_value:.4f}")

# Define function to generate bootstrap samples
def get_bootstrap_samples(data, n_samples=1000):
    indices = np.random.randint(0, len(data), (n_samples, len(data)))
    samples = data[indices]
    return samples

# Define function to calculate confidence intervals
def stat_intervals(stat, alpha=0.05):
    boundaries = np.percentile(stat, [100 * alpha / 2., 100 * (1 - alpha / 2.)])
    return boundaries

# Generate bootstrap samples for control and variant groups (all users)
control = get_bootstrap_samples(df_agg.loc[df_agg.VARIANT_NAME == 'control', 'REVENUE'].values, 10000)
variant = get_bootstrap_samples(df_agg.loc[df_agg.VARIANT_NAME == 'variant', 'REVENUE'].values, 10000)

# Generate bootstrap samples for control and variant groups (paying users only)
control_paid = get_bootstrap_samples(df_agg.loc[(df_agg.VARIANT_NAME == 'control') & (df_agg.REVENUE > 0), 'REVENUE'].values, 10000)
variant_paid = get_bootstrap_samples(df_agg.loc[(df_agg.VARIANT_NAME == 'variant') & (df_agg.REVENUE > 0), 'REVENUE'].values, 10000)

# Plot sample mean distribution for all users
plt.figure(figsize=(10, 5))
sns.kdeplot(np.mean(control, axis=1), shade=True, label='control')
sns.kdeplot(np.mean(variant, axis=1), shade=True, label='variant')
plt.title('Sample mean distribution for all users')
plt.legend()
plt.show()

# Plot sample mean distribution for paying users
plt.figure(figsize=(10, 5))
sns.kdeplot(np.mean(control_paid, axis=1), shade=True, label='control')
sns.kdeplot(np.mean(variant_paid, axis=1), shade=True, label='variant')
plt.title('Sample mean distribution for paying users')
plt.legend()
plt.show()

# Define function to plot distribution of differences and confidence intervals
def plot_distribution_and_stat_intervals(variant, control, title, alpha=0.05):
    plt.figure(figsize=(10, 5))
    points = sns.kdeplot(variant - control, shade=False).get_lines()[0].get_data()
    x, y = points
    ymin, ymax = plt.ylim()
    plt.vlines(0, 0, ymax, label='0', color='gray')
    ci = stat_intervals(variant - control, alpha)
    plt.vlines(ci[0], 0, ymax, linestyles="dashed")
    plt.vlines(ci[1], 0, ymax, linestyles="dashed")
    plt.fill_between(x, y, where=(x >= ci[1]) | (x <= ci[0]), color='gainsboro')
    plt.fill_between(x, y, where=(x >= ci[0]) & (x <= ci[1]), color='red', alpha=0.5, label='95% confidence interval')
    plt.title(f'Distribution of difference between means (variant - control)\n{title}\n{100*(1-alpha)}% Confidence interval: [{ci[0]:.4f}, {ci[1]:.4f}]')
    plt.legend()
    plt.show()
    return ci

# Plot distribution of differences and confidence intervals for all users
ci_all = plot_distribution_and_stat_intervals(np.mean(variant, axis=1), np.mean(control, axis=1), 'All Users')

# Plot distribution of differences and confidence intervals for paying users
ci_paid = plot_distribution_and_stat_intervals(np.mean(variant_paid, axis=1), np.mean(control_paid, axis=1), 'Paying Users')