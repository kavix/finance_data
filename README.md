# Group_XX_Financial_Data_Science

Rename `XX` to your group number in the folder name and in these files:
`notebook/Group_XX_Market_Analysis.ipynb`, `report/Group_XX_Committee_Report.pdf`,
`presentation/Group_XX_Briefing.pdf`, `contribution/Group_XX_Contribution_Statement.pdf`.

## Run
```bash
pip install -r requirements.txt
cd notebook
jupyter notebook Group_XX_Market_Analysis.ipynb   # Kernel -> Restart & Run All
```

## Steps after you get the dataset
1. Put the file at `data/raw/raw_market_data.csv` (or set `DATA_SOURCE = "yfinance"`).
2. Search the whole project for `TODO` and complete every item (each has a ready prompt to give Claude).
3. Restart & Run All -> no errors, all outputs visible.
4. Write the report (`report/REPORT_TEMPLATE.md`), slides (`presentation/SLIDES_OUTLINE.md`), contribution statement; export to PDF with the exact filenames.
5. Delete the template/outline/note files, remove `out`-of-date files, then zip the folder.

## Structure
```
data/raw/raw_market_data.csv          unchanged source data
data/processed/clean_financial_data.csv
src/finance_tools.py                  reusable functions
notebook/Group_XX_Market_Analysis.ipynb
outputs/  data_audit.csv data_quality_log.csv summary_statistics.csv covariance_matrix.csv correlation_matrix.csv figures/
report/ presentation/ contribution/
```
