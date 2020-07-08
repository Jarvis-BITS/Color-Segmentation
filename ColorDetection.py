from collections import Counter
import numpy as np
import cv2
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt


def RGB2HEX(colour):
    return "#{:02x}{:02x}{:02x}".format(int(colour[0]), int(colour[1]), int(colour[2]))


imge = cv2.imread(
    'C:\\Users\\user\\Desktop\\KratosLD-W3\\Color_Detection_Pic.jpg')
imge = cv2.cvtColor(imge, cv2.COLOR_BGR2RGB)


#  Reshaping image from 3D to 2D matrix by multiplying rows*columns:
img = imge.reshape((imge.shape[0] * imge.shape[1], 3))

clt = KMeans(n_clusters=5)
clt.fit(img)

labels = clt.labels_
centers = clt.cluster_centers_
centers = np.uint8(centers)  # Converting float back to 8 bit color values

segmented_img = centers[labels]
segmented_img = segmented_img.reshape(imge.shape)
segmented_img = cv2.cvtColor(segmented_img, cv2.COLOR_RGB2BGR)
cv2.imshow("Segmented Image", segmented_img)

#  Counting number of pixels belonging to each cluster:
numLabels = np.arange(0, 6)
hist, _ = np.histogram(clt.labels_, bins=numLabels)
hist = hist.astype("float32")
hist /= hist.sum()

# Creating bar graph to repsent colors with dimensions 100x600:
bar = np.zeros((100, 600, 3), dtype="uint8")
strt_pt = 0

# Zip is used to pair together Color along with its %age of occurance:
for (perct, color) in zip(hist, clt.cluster_centers_):
    # Plotting starting and endpoints of each color box
    end_pt = strt_pt + (perct * 600)
    cv2.rectangle(bar, (int(strt_pt), 0), (int(end_pt), 100),
                  (color.astype("uint8").tolist()), -1)
    strt_pt = end_pt  # Starting point of next color is end point of previous color

bar = cv2.cvtColor(bar, cv2.COLOR_RGB2BGR)
cv2.imshow("bar", bar)  # Showing color distribution


counts = Counter(clt.labels_)
# sort to ensure correct color percentage
counts = dict(sorted(counts.items()))

# We get ordered colors by iterating through the keys
ordered_colors = [clt.cluster_centers_[i] for i in counts.keys()]
hex_colors = [RGB2HEX(ordered_colors[i]) for i in counts.keys()]

# Plotting pie chart
plt.figure(figsize=(8, 6))
plt.pie(counts.values(), colors=hex_colors)
plt.show()

cv2.waitKey(0)
cv2.destroyAllWindows()
