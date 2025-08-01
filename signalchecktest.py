KeyError: This app has encountered an error. The original error message is redacted to prevent data leaks. Full error details have been recorded in the logs (if you're on Streamlit Cloud, click on 'Manage app' in the lower right of your app).
Traceback:
File "/mount/src/signalchecktest/signalchecktest.py", line 1558, in <module>
    extreme_rsi_count = (results_df['RSI_Threshold'] <= 15).sum()
                         ~~~~~~~~~~^^^^^^^^^^^^^^^^^
File "/home/adminuser/venv/lib/python3.13/site-packages/pandas/core/frame.py", line 4102, in __getitem__
    indexer = self.columns.get_loc(key)
File "/home/adminuser/venv/lib/python3.13/site-packages/pandas/core/indexes/base.py", line 3812, in get_loc
    raise KeyError(key) from err
