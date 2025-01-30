# Cluster Cutting Tool

## Merging Units

Once artifact units have been sorted, run the following command:

```bash
python main.py score <path> <session> --persistent --trust
```

Where:
- `<path>`: The folder path containing the `.clu`, `.spk`, and `.res` files.
- `<session>`: The session number matching the number at the end of the `.clu` file.
- `--persistent`: Keeps the program running and retains computations in memory, avoiding re-computation of non-merged units.
- `--trust`: Ensures the program remembers rejected merge recommendations. If two units are recommended for merging and you reject the suggestion, they will not appear together in future recommendations.

The program will output its top recommendations for unit merges. After completing a merge, save the `.clu` file and press Enter to compute the next recommendations. The recommended unit merges consist of the `n` best pairs identified.

---

## Parameters

The `score` command accepts seven optional parameters:
- **`binSize` (int):** Size of the bins for the correlogram.
- **`binNumber` (int):** Number of bins for the correlogram.
- **`n` (int):** Number of top pairs selected.
- **`persistent` (bool):** Keeps the program running after computation.
- **`trust` (bool):** Remembers rejected recommendations.
- **`plot` (bool):** Plots the score matrix.
- **`max_workers` (int):** Number of workers used for correlogram computation.
- **`metric` (list of int):** Specifies which metrics to use for score computation.

---

### Metric Parameter

The `metric` argument specifies which of the following metrics to use, in this order:
1. **Refractory Period Metric:** Higher for a larger refractory period in the cross-correlogram.
2. **Cross Correlogram Similarity Metric:** Higher when cross-corelogram are similar
3. **Waveforms Metric:** Higher when the two waveforms are similar.

For example:

```bash
python main.py score <path> <session> --metric 0 0 1
```

This command uses only the **Waveforms Metric** to compute the score.

---

### Max Workers Parameter

Computing correlograms is computationally intensive. Threading speeds up computation, but if CPU usage is high, threading may add additional overhead. In such cases, reduce the number of threads:

```bash
python main.py score <path> <session> --max_workers <max_workers>
```

---

## Persistent Mode

### Commands

In persistent mode, the following commands are available:
- `<Enter>`: Performs only necessary computations (requires that clusters are not renumbered).
- `<r>`: Recomputes everything.
- `<q>`: Terminates the program.
- `<forget>`: Forgets all user decisions, resetting the memory associated with the `--trust` option.

---

### Changing Parameters

Parameters can be updated during persistent mode using the following syntax:

```bash
:<arg>=<value>
```

Example:

```bash
:n=20
:metric=0 0 1
```

This updates the `n` and `metric` parameters.

---

## Manual Merging

When the algorithm cannot merge additional units, you can manually suggest candidates using the `notes` mode. To start this mode:

```bash
python main.py notes
```

Enter the candidate units to merge, two at a time, separated by a space:

```bash
candidate merge:45 5
```

Once all candidates are provided, press Enter. The program will group the candidate merges so that each cluster appears only once, making the merge process faster. You can then press `q` to quit or resume the merge process. 

Example:

```bash
candidate merge:6 97
candidate merge:9 98
candidate merge:41 96
candidate merge:51 106
candidate merge:45 100
candidate merge:51 90
candidate merge:67 101
candidate merge:90 106
candidate merge:
[6, 97]
[9, 98]
[41, 96]
[51, 106, 90]
[45, 100]
[67, 101]
candidate merge:q
```