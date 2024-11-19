# Cluster Cutting Tool

## Merging Units

Once artifact units have been sorted, run the following command:

```bash
python main.py score <path> <session> --persistent --trust
```

Where:
- `<path>` is the folder path containing the `.clu`, `.spk`, and `.res` files.
- `<session>` is the session number matching the number at the end of the `.clu` file.
- The `--persistent` flag keeps the program running and retains computations in memory, avoiding re-computation of non-merged units.
- The `--trust` flag indicates that the program will remember rejected merge recommendations, meaning if two units are recommended for merging and you reject the suggestion, they will not appear together in future recommendations.

The program will output its top recommendations for unit merges. After completing a merge, save the `.clu` file and press Enter to compute the next recommendations. The recommended unit merges consist of the `n` best pairs found.

## Parameters

The `score` command accepts 7 optional parameters:
- `binSize` (int): Size of the bins for the correlogram.
- `binNumber` (int): Number of bins for the correlogram.
- `n` (int): Number of top pairs selected.
- `persistent` (bool): If set, the program waits after computation.
- `trust` (bool): Remembers rejected recommendations.
- `plot` (bool): Plots the score matrix.
- `max_workers` (int): Number of workers used for correlogram computation.
- `metric` (list of int): Specifies which metrics to use for score computation.

### Metric Parameter

The `metric` argument specifies which of the five metrics to use, in the following order:
- **Refractory period metric**: Higher for a larger refractory period in the cross-correlogram.
- **Symmetry metric**: Higher for symmetric cross-correlograms.
- **Similarity metric**: Higher when the two autocorrelograms are similar.
- **Waveforms metric**: Higher when the two waveforms are similar.
- **Spike channel distance score**: Higher when the channels with the strongest activations coincide.

For example:

```bash
python main.py score <path> <session> --metric 0 0 0 1 0 
```

This would use only the **Waveforms metric** to compute the score.

### Max Workers Parameter

Computing correlograms is highly computationally intensive. Threading is used to speed up computation, but if CPU usage is already very high, threading may cause additional overhead. In such cases, you can reduce the number of threads with:

```bash
python main.py score <path> <session> --max_workers <max_workers>
```

### Persistent Mode

#### Commands

There are `4` commands in persistent mode:
- `<Enter>`: Does only necessary computation. This requires that clusters are not renumbered.
- `<r>`: Re-compute everything.
- `<q>`: Terminate the program.
- `<forget>`: Forgets all user decisions, resetting the memory associated with the `--trust` option.

#### Changing Parameters

Every parameter can be changed in persistent mode using the command:

```bash
:<arg>=<value>
```

For example:

```bash
:n=20
:metric=1 1 1 0 0
```

This command would update the `n` and `metric` parameters.