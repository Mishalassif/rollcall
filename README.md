# Rollcall

PCA analysis of US congress rollcall data.

## Interactive Analyzer

The interactive analyzer plots the bills/members in the first two dominant eigenspaces, from where
you can select four regions to take a closer look at the bills/members. To run the interactive analyzer:

```
python interactive_analyzer.py H115
```

In the console, a prompt appears where entering b analyzes bills and m analyzes members. A further prompt
then appears to give a filename to save the selected data. If the filename is 'bill5', the details of the bills
in the four corner regions will be saved to the file 'output/raw/H115/interactive_output/bill5.csv'. After entering
the filename, the plot of bills/members appears from which you can select four rectangular regions by
clicking on their diagonal corners. The plot also shows the bill details upon hovering on a bill.


