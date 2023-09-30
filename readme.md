## Data Science Projects

Some projects I have undertaken to learn and grow in data science. 
These projects mostly showcase how I would handle solving problems for particular datasets, 
but may include self-written statistical software and plotting algorithms.

Current projects:
1) House Price Dataset (Regression)
2) San Francisco's Incident Report Dataset (Forecasting)
3) Adult Salary Dataset (Classification)
4) Airport dataset (Forecasting)
5) Chest x-ray dataset (Computer vision)

The projects are currently found in the **regression_analysis**, **time_series_analysis**,
**classification_analysis** or **computer_vision** folder. 

In each project, there may be two types of iPython notebooks: 1) Exploratory data analysis (EDA), 2)
Confirmatory data analysis (CDA), which are both self-explanatory. 
Typically CDA is performed after EDA, and so should be read in the correct order. However, 
for complicated problems like computer vision I may start with modelling (CDA) first. Earlier notebooks are translations of the python files found in the project which I had written earlier,
and they can be run if wanted. The names of these python files are suffixed with
eda_jup.py or _cda_jup.py.

My approach to data science is mostly a graphical one. 
Hence, I will generally use plots to study the problem at hand, for example to find 
relationships/trends/seasonality/outliers. 
I prefer an approach not too rigorous (e.g. relying strongly on robust statistical testing for significance) 
and I like to validate my empirical results visually, i.e. by plotting residuals or CNN outputs.

My approach to M.L. is to leave it for work that requires automation, quick results, or for problems overly complicated such as pattern 
recognition with images - for where I would prefer CNNs. I like however to give a custom solution if possible, for example
by studying outliers, or for example using something like the IDENT procedure of Box-Jenkins for forecasting. 
