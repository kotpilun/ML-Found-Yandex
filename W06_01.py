from skimage.io import imread
from skimage import img_as_float
import numpy as np
import pandas
from sklearn.cluster import KMeans
import math

image = imread('./DATA/W_06_01.jpg')
image = img_as_float(image)

rgb = pandas.DataFrame(np.reshape(image, (image.shape[0]*image.shape[1],image.shape[2])))

kls = KMeans(init='k-means++', random_state=241).fit(rgb)

#Идеи взяты отсюда:
#https://www.coursera.org/learn/vvedenie-mashinnoe-obuchenie/discussions/weeks/6/threads/5gdnafAhEeWlJhJpbKSYeQ
#https://github.com/lxdv/ml-hse-yandex/blob/master/14.%20statement-clustering.ipynb

#записываем номер кластера для каждого пикселя
rgb['Clusters'] = kls.predict(rgb)

#группируем средние значения по кластерам, берем только значения без меток, тогда rgb_means[x] - строка, а не столбец
rgb_groups = rgb.groupby('Clusters').mean().values

#делаем маску для исходной матрицы цветов, получаем точную копию цветов исходной матрицы цветам средних значений
#(можно еще циклом заменить в исх матрице значения на средние)
rgb_means_mask = [rgb_groups[x] for x in rgb['Clusters']]

#восстанавливаем картинку
mean_image = np.reshape(rgb_means_mask, (image.shape[0],image.shape[1],image.shape[2]))

#аналогично для медины
rgb_groups = rgb.groupby('Clusters').median().values
rgb_medians_mask = [rgb_groups[x] for x in rgb['Clusters']]
median_image = np.reshape(rgb_medians_mask, (image.shape[0],image.shape[1],image.shape[2]))


psnr_median = 10 * math.log10(1. / np.mean((image - median_image) ** 2))
psnr_mean = 10 * math.log10(1. / np.mean((image - mean_image) ** 2))

print(psnr_median,psnr_mean)

count = 0
rgb = pandas.DataFrame(np.reshape(image, (image.shape[0] * image.shape[1], image.shape[2])))
for i in range(1, 21):
    kls = KMeans(init='k-means++', random_state=241, n_clusters=i).fit(rgb)
    rgb['Clusters'] = kls.predict(rgb)

    rgb_groups = rgb.groupby('Clusters').mean().values
    rgb_means_mask = [rgb_groups[x] for x in rgb['Clusters']]
    mean_image = np.reshape(rgb_means_mask, (image.shape[0], image.shape[1], image.shape[2]))

    rgb_groups = rgb.groupby('Clusters').median().values
    rgb_medians_mask = [rgb_groups[x] for x in rgb['Clusters']]
    median_image = np.reshape(rgb_medians_mask, (image.shape[0], image.shape[1], image.shape[2]))

    psnr_median = 10 * math.log10(1. / np.mean((image - median_image) ** 2))
    psnr_mean = 10 * math.log10(1. / np.mean((image - mean_image) ** 2))
    print(i, end=' - ')
    print('median: ', psnr_median, end=', ')
    print('mean: ', psnr_mean)
    if psnr_median > 20 or psnr_mean > 20:
        count = i
        break