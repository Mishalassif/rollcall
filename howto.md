* To select bills contained in a rectangle in the eigenspace, run "python interactive_analyser.py H(congressnumber)". For example, to run the analyser on congress number 113, run "python interactive_analyser H113", or for congress number 096 run "python interactive_analyser H096".

* You can analyze the eigenspace in members or bills. You are first asked to choose which of these you want, enter m for memberspace, b for billspace and any other key to quit the program.

* Once you enter this info, you are shown the plot of the corresponding eigenspace. You can now select two diagonally opposite corners of the rectangle you would like to analyse by clicking on the points inside the plot. The selected rectangle will be visible to you after two clicks, and then you are asked if you are happy with your choice to which you can reply either y or n. 

* If you say y, you are asked for a filename to save the details of the members/bills in the rectangle you chose. The file can then be found in output/raw/H113/interactive_output/filename.csv. If you enter the same filename for the same congress twice, the contents are overwritten automatically.

* If you say n, you start over the whole process and prompted to choose between m and b. 

* To quit the program, just press any key other than m or b when you are prompted to analyze members or bills.


