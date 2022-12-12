import pandas as pd

def save_log(df:pd.DataFrame, csv_path, epoch, log_interval, **kwds):
    line = {
        k:v for k,v in kwds.items()
    }
    line['epoch'] = epoch

    df = pd.concat([df, pd.DataFrame([line])])
    if (epoch+1) % log_interval == 0:
        df.to_csv(csv_path, index=None)
    return df