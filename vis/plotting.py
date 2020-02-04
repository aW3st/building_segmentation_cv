# Plotting functions for 

from IPython.display import clear_output
import matplotlib.pyplot as plt


def plot_tile(tile=None, mask=None, custom_title=None, ax):
  '''
  Plot a tile, no window transform, and optionally the mask as well.
  '''
  title = 'Tile'
  if ax is not None:
    fig, ax = plt.subplots(figsize=(10,10))

  rasterio.plot.show(tile, ax=ax)

  if mask is not None:
    ax.imshow(np.ma.masked_where(mask==0, mask), cmap='spring', alpha=0.4)
    title += ' & Buildings'

  if custom_title is not None:
    title = custom_title
  plt.title(title)


  def plot_dataset_sample(tf_dataset, sample_size=5):
  '''
  Shuffle a dataset, and ploit 
  '''

  tf_dataset.shuffle(buffer_size=sample_size)
  dataset_iter = iter(tf_dataset)

  fig = plt.figure(figsize=(30, 3*sample_size))

  for plot_num in range(1, sample_size + 1):
    d = next(dataset_iter)
    scene_id = d.get('scene_id').numpy()
    y = d.get('y').numpy()
    x = d.get('x').numpy()
    tile = d.get('tile').numpy()
    mask = d.get('mask').numpy()

    ax = fig.add_subplot(sample_size//3+1, 3, plot_num)
    title = f'{scene_id} at ({x}, {y})')
    plot_tile(tile, mask, custom_title=title)