# SAMTool - Semantic Segmentation Dataset Creation powered by Segment Anything Model from Meta

<p align="center">
    <img src="demo.gif" width="800"/>
</p>

## Installation

`pip3 install git+https://github.com/jjshoots/SAMtool`

## Usage

1. `samtool --imagedir <images directory> --labeldir <labels directory> --annotations <annotations.yaml file>`
2. Go to `127.0.0.1:7860`

#### Defining labels

The labels must be defined as a `yaml` file. Example contents of the file:
```yaml
__ignore__: 0
cat: 1
dog: 2
other: 3
```

## Labels Storage

In the name of performance, all label arrays are stored as uncompressed numpy boolean arrays on disk.
This results in the labels file being potentially very large.
If you would like to optimize the storage of the data, one way of doing it is to use numpy's `packbits` function.

#### To compress labels

```python
import os
import numpy as np

labels_dir = "YOUR_LABELS_DIRECTORY"
packed_labels_dir = "YOUR_PACKED_LABELS_DIRECTORY"

for filename in os.listdir(labels_dir):
    np.save(f"{packed_labels_dir}/{filename}", np.packbits(np.load(f"{labels_dir}/{filename}"), axis=0))
```

#### To read compressed labels

```python
import os
import numpy as np

packed_labels_dir = "YOUR_PACKED_LABELS_DIRECTORY"
for filename in os.listdir(packed_labels_dir):
    label = np.unpackbits(np.load(f"{packed_labels_dir}/{filename}"))
```
