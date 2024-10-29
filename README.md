# Cluster Cutting Tool

## Merging Units

Once artifact units have been sorted, run the following command:

```bash
python main.py score <path> <session> --persistent
```

Where:
- `<path>` is the folder path containing the `.clu`, `.spk`, and `.res` files.
- `<session>` is the session number corresponding to the number at the end of the `.clu` file.
- The `--persistent` flag keeps the program running and retains computations in memory, allowing it to avoid re-computing non-merged units.

The program will output its top recommendations for unit merges. After completing a merge, save the `.clu` file and press Enter to compute the next recommendations. The recommended unit merges are composed of the `n` best pairs found.

## Parameters

There are 7 optional parameters for the `score` command:
- `binSize` (int): Size of the bins for the correlogram.
- `binNumber` (int): Number of bins for the correlogram.
- `n` (int): Number of best pairs selected.
- `persistent` (bool): If set, the program waits after computation.
- `plot` (bool): Plots the score matrix.
- `max_workers` (int): Number of workers used for correlogram computation.
- `metric` (list of int): Specifies which metrics to use for score computation.

### Metric Parameter

The `metric` argument specifies which of the four metrics to use, in the following order:
- **Refractory period metric**: Higher for a larger refractory period in the cross-correlogram.
- **Symmetry metric**: Higher for symmetric cross-correlograms.
- **Similarity metric**: Higher when the two autocorrelograms are similar.
- **Waveforms metric**: Higher when the two waveforms are similar.

For example:

```bash
python main.py score <path> <session> --metric 1 0 0 0
```

would only use the **refractory period metric** to compute the score.

### Max Workers Parameter

Computing correlograms is a highly computationally intensive task. Threading is used to speed up computation, which can raise issues if CPU usage is already very high (though this is unlikely). In such cases, resource allocation and threading overhead can take longer than the computation itself. You can reduce the number of threads with:

```bash
python main.py score <path> <session> --max_workers <max_workers>
```

### Changing Parameters in Persistent Mode

Every parameter can be changed in persistent mode with the command:

```bash
:<arg>=<value>
```