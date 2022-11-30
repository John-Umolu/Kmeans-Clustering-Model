# import the python libraries
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# read dataset csv file
df = pd.read_csv('country_data.csv')

# remove any null values from data rows
df = df.dropna()

# plot heatmap
fig = plt.figure(figsize=(15, 7))

# set the figure title
fig.canvas.manager.set_window_title('Coursework Task 2: Clustering By Umolu John Chukwuemeka')

# title by setting initial sizes
fig.suptitle('Heatmap Correlation', fontsize=14, fontweight='bold')

# plot the heatmap
sns.heatmap(df.corr(), annot=True)

# add a space at the bottom of the plot
fig.subplots_adjust(bottom=0.2)

# display the plot
plt.show()