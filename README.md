# galaxy-classification
Query data from Sloan Digital Sky Survey (SDSS) database. Train/Test a simple CNN to classify galaxies as spiral/elliptical.

## Usage
**Important Notes**:
1. **Must have account to query SDSS** (quick and free email registration) with CasJobs to download SDSS galaxy and image data.
Can be created [here](https://skyserver.sdss.org/CasJobs/).
2. CasJobs is very harsh towards incorrect login attempts. One incorrect attempt **may get you locked out** for some time. You can see how long by trying to log in on their website (link above).
3. If you have < 8GB you may run into memory issues when training the CNN. If this happens try:
  * Decreasing batch size
  * Making output size of convolution layers smaller
  * Increasing kernel size and/or stride in convolution layer

### Dependencies
* Typical Python packages (numpy, pandas, scikit, etc.)
* Tensorflow
* Keras
* SciServer

### Getting the Data:[sdss_query.py](/sdss_query.py)
To create **galaxy_data.csv** and **galaxy_images.npy** you must call create_datafiles(). Files will be created in your current directory.
Example of creating data files with default parameters:
```python
import sdss_query as sdss
sdss.create_datafiles()
```
You will be prompted to enter your username and password for CasJobs. If **login failure**, re-read import notes above. create_datafiles has many optional arguments allowing you to customize your query. 
* n_galaxies: int. Number of galaxies to query
* galaxy_type: str. Allows to exclusively query one type galaxy. Valid args are 'elliptical', 'spiral', or 'both'
* lower_z_limit: float. lower constraint on redshift
* upper_z_limit: float. upper constraint on redshift
* lower_flux_limit: float. lower constraint on g-band Petrosian flux
* upper_flux_limit: float. upper constrain on g-band Petrosian flux
* data_release: str. SDSS data release version. 
* image_data. bool. True to get image data and save to galaxy_images.npy. False to only save galaxy_data.
* image_scale_factor: float. Factor to multiply by the radius of 90% flux, producing the image scale in arcsec/pixel. 

Defaults arguments:
```python
create_datafiles(n_galaxies=150, galaxy_type='both', lower_z_limit=0.1, upper_z_limit=0.3, lower_flux_limit=50,
                      upper_flux_limit=500, data_release='DR15', image_data=True, image_scale_factor=0.01)
                                         
```
Image data is accessed via SkyServer.getJpegImgCutout(). Full documentation on this module can be found [here](https://www.sciserver.org/docs/sciscript-python/SciServer.html#module-SciServer.SkyServer).

**MORE ADVANCED QUERIES:**
The SDSS database has hundreds of tables, views, and variables not included in the create_datafiles function. You can create more complex queries by changing the SQL_Query variable (lines 51-65) to the specifics that you need. Full documentation and schema browser for the SDSS database can be accessed [here](https://skyserver.sdss.org/CasJobs/SchemaBrowser.aspx)

### Data Files
**galaxy_labels.npy**  
   NumPy array of labels classifying the galaxies. 0: likely spiral, 1: likely elliptical
      
**galaxy_images.npy**  
   NumPy array of n_galaxies images with RGB color channels. array.shape = (n_galaxies, 512, 512, 3). It is straightforward to view an individual galaxy image.
```python
import matplotlib.pyplot as plt
import numpy as np

image_data = np.load('galaxy_images.npy')
plt.imshow(image_data[0])
plt.show()
```

### Quick CNN[quick_cnn_trainer.py](/quick_cnn_trainer.py)

**get_data()** grabs the image and labels from galaxy_images.npy, galaxy_labels.npy 

**clean_data()** one-hot encodes the galaxy classifications and normalizes the images

**create_cnn()** instantiates a Keras Sequential model with structure:  
  [Conv2d->BatchNormalization->MaxPooling->Dropout->FullyConnected]
  If you have a powerful computer with enough RAM you can create a deeper network by uncommenting lines of additional conv2D/pooling layers. 
  
This module is meant to be a script to get a quick and dirty galaxy classification model trained. The terminal command:
```python
python quick_cnn_trainer.py
```
will run a scipt that calls get_data(), clean_data(), create_cnn(), performs a train/test split (test_size=0.3) and compile/fits the model with:
```python
    model.compile(loss="categorical_crossentropy", optimizer='adam', metrics=["accuracy"])
    performance = model.fit(X_train, y_train, batch_size=5, epochs = 15, validation_split=0.2, verbose=1)
```
Predictions will be made for X_test and a confusion matrix is printed to terminal. Loss and Accuracy plots across the 5 epochs are plotted and shown. Model is saved to your computer as 'my-galaxy-model.h5'.
  
