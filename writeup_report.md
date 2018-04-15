## The Solution

Nvidia model, plain and simple.

![Nvidia Convnet model](nvidia-model.png )

#### Specifics

Dropouts were added to prevent overfitting, and a blurring effect was added in
the preprocessing pipeline. This was also where we had a

Added normalization after pipeline already did it.
The result was a model that made very weak updates.

#### Data Set

Data set collected with
* 1 Lap in counter clockwise direction
* 1 Lap in clockwise direction
* 1 Lap with lots of recovery poses
* 1 Lap recording only the turns

Data collection was very rote. Previous attempts had many laps collect4ed. the duplicate normalization really destroyed the model. It took a while before I
found out that my data was fine, it was the pipeline I had.
