# import the python libraries
import pandas as pd  # use perform dataset analysis
import matplotlib.pyplot as plt  # use for creating plots
from matplotlib.widgets import Slider  # use to create a slider to help user choose any input variable
import numpy as np  # use to perform mathematical functions
from sklearn.cluster import KMeans  # use to build and train the unsupervised model
from sklearn.preprocessing import StandardScaler  # for data standardisation
from sklearn.metrics import silhouette_samples, silhouette_score  # use to evaluate kmeans clustering

# declared variables to use
plotCount: int = 0  # use for track slider movement to clear previous plot
line_1 = (0, 0, 0)  # stores the last scatter plot points
legend_1 = (0, 0)  # stores the last legend value
centers_plot = (0, 0, 0)  # stores the last centroids text plot
cluster_txt = (0, 0)  # stores the last cluster text plot
value_1: int = 0  # store the first prediction value
value_2: int = 0  # store the second prediction value
value_3: int = 0  # store the third prediction value

# read dataset csv file
df = pd.read_csv('country_data.csv')
# remove any null values from data rows
df = df.dropna()
# display the dataframe
print(df)
# Standardize the data
# X = StandardScaler().fit_transform(df.iloc[:, [5, 7, 9]].values)
# get the X variables from the dataframe
X = df.iloc[:, [5, 7, 9]].values
# print the first five rows of the dataset
print(X[0:5, :])
# create the new figure with width = 15 and height = 7
fig = plt.figure(figsize=(15, 7))
# display figure suptitle
fig.suptitle('Kmeans Clustering By Umolu John Chukwuemeka (2065655)', color='blue', fontsize=16, fontweight='bold')
# create box to contain the model performance text using the matplotlib.patch.Patch properties
props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
# set the figure title
fig.canvas.manager.set_window_title('Coursework Task 2: Kmeans Clustering By Umolu John Chukwuemeka (2065655)')
# create a subplot with a single figure
axa = fig.add_subplot(121, projection='3d')
# create a subplot for the elbow evaluation
axb = fig.add_subplot(122)
# Tell algorithm the number of clusters it should look for:
kmeans = KMeans(n_clusters=2)
# run the Kmeans algorithm for the data X:
kmeans.fit(X)
# predict which cluster each data point X belongs to:
scatter1 = axa.scatter(X[:, 0], X[:, 1], X[:, 2], c=kmeans.predict(X), s=30, cmap='plasma')
# get the array of centroids locations
centers = kmeans.cluster_centers_
# loop through all the locations of the clusters
for i in range(centers.shape[0]):
    # set the text value using the index value of i
    centers_plot = axa.text(centers[i, 0], centers[i, 1], centers[i, 2], str(i), c='black',
                            bbox=dict(boxstyle='round', facecolor='white', edgecolor='black'))
# produce a legend with the unique colors from the scatter
legend1 = axa.legend(*scatter1.legend_elements(), loc="upper left", title="Clusters")
axa.add_artist(legend1)
# set the x-axis label
axa.set_xlabel('income')
# set the y-axis label
axa.set_ylabel('life_expec')
# set the z-axis label
axa.set_zlabel('gdpp', rotation=90)


# Elbow method evaluation
# https://towardsdatascience.com/k-means-clustering-algorithm-applications-evaluation-methods-and-drawbacks
# https://stackoverflow.com/questions/43784903/scikit-k-means-clustering-performance-measure
# get sum of squared distance as a list
sse = {}
for k_value in range(1, 11):
    km = KMeans(n_clusters=k_value, max_iter=1000)
    km.fit(X)
    sse[k_value] = km.inertia_ # Inertia: Sum of distances of samples to their closest cluster center
# plot elbow graph
axb.set_title("Elbow criterion evaluation")
axb.plot(list(sse.keys()), list(sse.values()), '-o')
axb.set_xlim(0, 11)
axb.set_ylim(0)
axb.set_xlabel("Number of cluster k")
axb.set_ylabel("Sum of squared distance (SSE)")

# Silhouette evaluation
labels = kmeans.fit_predict(X)
# get silhouette values
silhouette_vals = silhouette_samples(X, labels)
# print('Silhouette values:', silhouette_vals)
# The silhouette_score gives the average value for all the samples.
# This gives a perspective into the density and separation of the formed
# clusters
silhouette_avg = silhouette_score(X, labels)
# print silhouette values score
print('Silhouette score:', round(silhouette_avg, 4))
# create box to contain the model performance text using the matplotlib.patch.Patch properties
props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
# save result text
result_text = 'Evaluation using silhouette analysis: ' + '\n' + 'Number of cluster k: ' + str(2) + '\n' + 'Silhouette score: ' \
              + str(round(silhouette_avg, 4))
# display model performance at the bottom of the figure
plot_text = axb.text(0.5, -0.3, result_text, horizontalalignment='center',
                     verticalalignment='center', transform=axb.transAxes, wrap=True, bbox=props)

# move the figure up and sideways so that it's not on top of the slider
fig.subplots_adjust(left=0.2, bottom=0.3)


# function use to save figure image
def save_plot():
    # save the figure image
    fig.savefig('kmeans_clustering.png')
    # return to called location
    return


# function to call prediction function when the square foot slider is changed
def set_cluster(value):
    # make variables accessible from the outside
    global kmeans, centers, scatter1, legend_1, legend1, plot_text, result_text, cluster_txt, centers_plot
    # check if the slider value is greater or equal to the least x-input variable in the dataset
    if value > 0:
        # remove cluster text on plot
        cluster_txt.remove()
        # remove evaluation text on plot
        plot_text.remove()
        centers_plot.remove()
        # clear previous plots
        axa.clear()
        # Tell algorithm the number of clusters it should look for:
        kmeans = KMeans(n_clusters=value)
        # run the Kmeans algorithm for the data X:
        kmeans.fit(X)
        # predict which cluster each data point X belongs to:
        scatter1 = axa.scatter(X[:, 0], X[:, 1], X[:, 2], c=kmeans.predict(X), s=50, cmap='plasma')
        # get the array of centroids locations
        centers = kmeans.cluster_centers_
        # loop through all the locations of the clusters
        for i in range(centers.shape[0]):
            # set the text value using the index value of i
            centers_plot = axa.text(centers[i, 0], centers[i, 1], centers[i, 2], str(i), c='black',
                                    bbox=dict(boxstyle='round', facecolor='white', edgecolor='black'))
        # produce a legend with the unique colors from the scatter
        legend1 = axa.legend(*scatter1.legend_elements(), loc="upper left", title="Clusters")
        axa.add_artist(legend1)
        # set the x-axis label
        axa.set_xlabel('income')
        # set the y-axis label
        axa.set_ylabel('life_expec')
        # set the z-axis label
        axa.set_zlabel('gdpp', rotation=90)

        # Silhouette evaluation
        labels = kmeans.fit_predict(X)
        silhouette_vals = silhouette_samples(X, labels)
        # print('Silhouette values:', silhouette_vals)
        # The silhouette_score gives the average value for all the samples.
        # This gives a perspective into the density and separation of the formed
        # clusters
        silhouette_avg = silhouette_score(X, labels)
        # print Silhouette evaluation score
        print('Silhouette score:', round(silhouette_avg, 4))
        # save Silhouette evaluation text
        result_text = 'Evaluation using silhouette analysis: ' + '\n' + 'Number of cluster k: ' + str(value) + '\n' \
                      + 'Silhouette score: ' + str(round(silhouette_avg, 4))
        # display model performance at the bottom of the figure
        plot_text = axb.text(0.5, -0.3, result_text,
                             horizontalalignment='center', verticalalignment='center',
                             transform=axb.transAxes, wrap=True, bbox=props)

        # Get the clusters values and store as dataframe
        clusters_dataframe = pd.DataFrame(kmeans.labels_, columns=['Clusters'])
        # Get the total number of clusters
        cluster_values = '\n'.join(
            str(i) + ' : ' + str(e) for i, e in enumerate(clusters_dataframe['Clusters'].value_counts()))
        # display result on console
        print('Clusters values: \n', cluster_values)
        # display result on the figure
        cluster_txt = fig.text(x=-0.35, y=0.5, s='Data points: \n' + cluster_values, fontsize=10,
                               transform=axa.transAxes, bbox=props)

        # return to called location
        return


def predict_x1(value):
    # make variables accessible from the outside
    global value_1, value_2, value_3, plotCount, line_1, legend_1, centers_plot
    # store the new slider value
    value_1 = value
    # Plot remove routine that removes the previous scatter plot point in every 2 counts
    plotCount = plotCount + 1
    # check if plot count value is equal to 2, which represents the second time plot
    if plotCount == 2:
        # replace the plotCount value from 2 to 1
        plotCount = 1
        # remove the previous scatter plot points form the figure
        centers_plot.remove()
        line_1.remove()
        # remove the legend from the figure
        legend_1.remove()
    # Predicting the cluster of a data point
    sample_test = np.array([[value_1, value_2, value_3]])
    # plot the scatter plot
    line_1 = axa.scatter(sample_test[0, 0], sample_test[0, 1], sample_test[0, 2], c='red', marker='*', s=100,
                         label='Test data point')
    # loop through all the locations of the clusters
    for i in range(centers.shape[0]):
        # set the text value using the index value of i
        centers_plot = axa.text(centers[i, 0], centers[i, 1], centers[i, 2], str(i), c='black',
                                bbox=dict(boxstyle='round', facecolor='white', edgecolor='black'))
    # place the legend at the upper right position on the plot
    legend_1 = axa.legend(loc='upper center')
    # display the prediction as figure title
    axa.set_title('Test data point belongs to cluster {}'.format(kmeans.predict(sample_test)[0]))
    # display the prediction on the console
    print('test data point belongs to cluster {}'.format(kmeans.predict(sample_test)[0]))
    # save plot image
    save_plot()
    # return to called location
    return


def predict_x2(value):
    # make variables accessible from the outside
    global value_1, value_2, value_3, plotCount, line_1, legend_1, centers_plot
    # store the new slider value
    value_2 = value
    # Plot remove routine that removes the previous scatter plot point in every 2 counts
    plotCount = plotCount + 1
    # check if plot count value is equal to 2, which represents the second time plot
    if plotCount == 2:
        # replace the plotCount value from 2 to 1
        plotCount = 1
        # remove the previous scatter plot points form the figure
        centers_plot.remove()
        line_1.remove()
        # remove the legend from the figure
        legend_1.remove()
    # Predicting the cluster of a data point
    sample_test = np.array([[value_1, value_2, value_3]])
    # plot the scatter plot
    line_1 = axa.scatter(sample_test[0, 0], sample_test[0, 1], sample_test[0, 2], c='red', marker='*', s=100,
                         label='Test data point')
    # loop through all the locations of the clusters
    for i in range(centers.shape[0]):
        # set the text value using the index value of i
        centers_plot = axa.text(centers[i, 0], centers[i, 1], centers[i, 2], str(i), c='black',
                                bbox=dict(boxstyle='round', facecolor='white', edgecolor='black'))
    # place the legend at the upper right position on the plot
    legend_1 = axa.legend(loc='upper center')
    # display the prediction as figure title
    axa.set_title('Test data point belongs to cluster {}'.format(kmeans.predict(sample_test)[0]))
    # display the prediction on the console
    print('test data point belongs to cluster {}'.format(kmeans.predict(sample_test)[0]))
    # save plot image
    save_plot()
    # return to called location
    return


def predict_x3(value):
    # make variables accessible from the outside
    global value_1, value_2, value_3, plotCount, line_1, legend_1, centers_plot
    # store the new slider value
    value_3 = value
    # Plot remove routine that removes the previous scatter plot point in every 2 counts
    plotCount = plotCount + 1
    # check if plot count value is equal to 2, which represents the second time plot
    if plotCount == 2:
        # replace the plotCount value from 2 to 1
        plotCount = 1
        # remove the previous scatter plot points form the figure
        centers_plot.remove()
        line_1.remove()
        # remove the legend from the figure
        legend_1.remove()
    # Predicting the cluster of a data point
    sample_test = np.array([[value_1, value_2, value_3]])
    # plot the scatter plot
    line_1 = axa.scatter(sample_test[0, 0], sample_test[0, 1], sample_test[0, 2], c='red', marker='*', s=100,
                         label='Test data point')
    # loop through all the locations of the clusters
    for i in range(centers.shape[0]):
        # set the text value using the index value of i
        centers_plot = axa.text(centers[i, 0], centers[i, 1], centers[i, 2], str(i), c='black',
                                bbox=dict(boxstyle='round', facecolor='white', edgecolor='black'))
    # place the legend at the upper right position on the plot
    legend_1 = axa.legend(loc='upper center')
    # display the prediction as figure title
    axa.set_title('Test data point belongs to cluster {}'.format(kmeans.predict(sample_test)[0]))
    # display the prediction on the console
    print('test data point belongs to cluster {}'.format(kmeans.predict(sample_test)[0]))
    # save plot image
    save_plot()
    # return to called location
    return


# Set the sliders axes on the plot
x1SliderDim = plt.axes([axa.get_position().x0, axa.get_position().y0-0.1, axa.get_position().width, 0.03],
                       facecolor='lightgoldenrodyellow')
x2SliderDim = plt.axes([axa.get_position().x0, axa.get_position().y0-0.15, axa.get_position().width, 0.03],
                       facecolor='lightgoldenrodyellow')
x3SliderDim = plt.axes([axa.get_position().x0, axa.get_position().y0-0.2, axa.get_position().width, 0.03],
                       facecolor='lightgoldenrodyellow')
x4SliderDim = plt.axes([axa.get_position().x0, axa.get_position().y0-0.25, axa.get_position().width, 0.03],
                       facecolor='lightgoldenrodyellow')

# Make a horizontal slider to select square footage of home value
x1Slider = Slider(
    ax=x1SliderDim,
    label='Set number of clusters:',  # set the slider title
    valinit=2,  # set the initial slider position using the least value of X
    valmin=2,  # set the minimum slider value
    valstep=1,  # set the step value for each slider movement
    valmax=10  # set the maximum slider value using the maximum value of X variable
)
# Make a horizontal slider to select square footage of home value
x2Slider = Slider(
    ax=x2SliderDim,
    label='Set income value:',  # set the slider title
    valinit=round(np.min(np.array(X[:, 0]))),  # set the initial slider position using the least value of X
    valmin=round(np.min(np.array(X[:, 0]))),  # set the minimum slider value
    valstep=1,  # set the step value for each slider movement
    valmax=round(np.max(np.array(X[:, 0])))  # set the maximum slider value using the maximum value of X variable
)
# Make a horizontal slider to select square footage of home value
x3Slider = Slider(
    ax=x3SliderDim,
    label='Set life expectancy value:',  # set the slider title
    valinit=round(np.min(np.array(X[:, 1]))),  # set the initial slider position using the least value of X
    valmin=round(np.min(np.array(X[:, 1]))),  # set the minimum slider value
    valstep=1,  # set the step value for each slider movement
    valmax=round(np.max(np.array(X[:, 1])))  # set the maximum slider value using the maximum value of X variable
)
# Make a horizontal slider to select square footage of home value
x4Slider = Slider(
    ax=x4SliderDim,
    label='Set gdpp value:',  # set the slider title
    valinit=round(np.min(np.array(X[:, 2]))),  # set the initial slider position using the least value of X
    valmin=round(np.min(np.array(X[:, 2]))),  # set the minimum slider value
    valstep=1,  # set the step value for each slider movement
    valmax=round(np.max(np.array(X[:, 2])))  # set the maximum slider value using the maximum value of X variable
)
# Register the update function with each slider
x1Slider.on_changed(set_cluster)
x2Slider.on_changed(predict_x1)
x3Slider.on_changed(predict_x2)
x4Slider.on_changed(predict_x3)
# get the minimum values of each column as the default values
value_1 = round(np.min(np.array(X[:, 0])))
value_2 = round(np.min(np.array(X[:, 1])))
value_3 = round(np.min(np.array(X[:, 2])))

# Get the clusters values and store as dataframe
clusters_dataframe = pd.DataFrame(kmeans.labels_, columns=['Clusters'])
# Get the total number of clusters
cluster_values = '\n'.join(str(i) + ' : ' + str(e) for i, e in enumerate(clusters_dataframe['Clusters'].value_counts()))
# display result on console
print('Clusters values: \n', cluster_values)
# display result on the figure
cluster_txt = fig.text(x=-0.35, y=0.5, s='Data points: \n' + cluster_values, fontsize=10,
                       transform=axa.transAxes, bbox=props)

# show the plot
plt.show()

