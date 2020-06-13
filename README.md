In order to generate the training/test datasets, run
<code>
python3 dataset_generator.py
</code>
this will generate a file called dataset.npz inside the directory dataset_generator.py is located in.

Afterwards, run the following command
<code>
python3 train.py
</code>
This will train the model using the dataset and produce a loss plot and a figure illustrating the predicted robot location on one of the maps in the testing set.
NOTE: Make sure to run train.py inside the same directory as dataset.npz and also have the maps folder with the 100 maps in the same directory as dataset_generator.py