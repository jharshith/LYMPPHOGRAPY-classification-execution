import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Load the data
data_file_path = 'lymphography.data'
data = pd.read_csv(data_file_path, header=None)

# Assign column names to the DataFrame
column_names = [
    "class", "lymphatics", "block_of_affere", "bl_of_lymph_c", "bl_of_lymph_s", "by_pass", "extravasates",
    "regeneration_of", "early_uptake_in", "lym_nodes_dimin", "lym_nodes_enlar", "changes_in_lym",
    "defect_in_node", "changes_in_node", "changes_in_stru", "special_forms", "dislocation_of",
    "exclusion_of_no", "no_of_nodes_in"
]
data.columns = column_names

# Univariate Analysis
plt.figure(figsize=(12, 5))

# Histogram
plt.subplot(121)
sns.histplot(data["lymphatics"], color='r')
plt.title('Univariate: Histogram of Lymphatics')

#Kernel Density Estimate
plt.subplot(122)
sns.kdeplot(data["lymphatics"], color='b', fill=True)
plt.title('Univariate: KDE of Lymphatics')

plt.tight_layout()
plt.show()

# Bivariate Analysis
plt.figure(figsize=(12, 5))

# Multivariate Analysis
# Pair Plot
sns.pairplot(data, hue="class", palette="viridis")
plt.suptitle('Multivariate: Pair Plot', y=1.02)
plt.show()
