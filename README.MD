This repository contains the occupancy data from the paper 
"Occupancy Prediction for Building Energy Systems with Latent Force Models". They were
collected at the Bosch Research Campus in Renningen, Germany. Additionally, an 
implementation of the Latent Force Model (LFM) predictor is provided

# Running the example
The LFM is provided by the `utilities` package inside this repository. Install the 
package with
```
pip install .
```
and then run `compare_prediction.py`. This reproduces the prediction results from the 
paper in section 4.5.

# Dataset
The occupancy dataset is found in the `data` folder inside the file `Occupancy.mat`.
The basic metadata for the whole dataset is found in the table below.

| Specification   | Field       |
|:----------------|:------------|
| Start time      | 01-Jan-2019 |
| End time        | 15-Oct-2022 |
| Sample time     | 15 minutes  |
| Number of Zones | 9           |

## Zone Descripton

![Zone layout of the building at Bosch](/data/Rng111zoning.png)

The dataset consists out of every zone in the layout above. Their index and 
description is found in the table below

| Zone | Index | Description       |
|:-----|:------|-------------------|
| 9    | 0     | Meeting Room      |
| 10   | 1     | Meeting Room      |
| 11   | 2     | Meeting Room      |
| 12   | 3     | Kitchen           |
| 13   | 4     | Copy Room         |
| 21   | 5     | Open Space Office |
| 22   | 6     | Open Space Office |
| 23   | 7     | Open Space Office |
| 24   | 8     | Open Space Office |

# License

Only the occupancy dataset is distributed under the CC-BY-NC-SA License. If using or 
distributing the data, please cite the following paper (currently under review):

```
@article{Wietzke2024,
  author   = {Wietzke, Thore and Gall, Jan and Graichen, Knut},
  title    = {Occupancy Prediction for Building Energy Systems with Latent Force Models},
  journal  = {Energy and Buildings},
  year     = {2024},
}
```
